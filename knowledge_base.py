"""
knowledge_base.py — verse-indexed, multi-view RAG over the enriched corpus.

The shift from the old design
-----------------------------
The old knowledge_base.py chunked source text into 380-token windows with
overlap. The new one indexes each verse as a single record but with three
*views* — three separate embeddings of three different framings of the same
verse — so that queries phrased in different registers can all find it.

The three views per verse, and what each one is good for:

    literal_view   — the English translation (and Sanskrit fragment if
                     available). Best for queries that share lexical features
                     with the text itself: "what does it mean to act without
                     attachment?" maps cleanly to BG 2.47's literal text.

    bhashya_view   — Śaṅkara's commentary on the verse. Best for queries that
                     ask about the Vedāntic explanation rather than the verse
                     itself: "how does adhyāsa relate to suffering?" finds
                     the bhāṣya passages where Śaṅkara unfolds adhyāsa.

    advisor_view   — the LLM-enriched composite (paraphrase + life situations
                     + emotions addressed + hypothetical questions). Best for
                     real-world questions in real-world language. This is
                     where the language gap closes.

At retrieval time we query all three indices, merge by verse_id (so each
verse appears at most once), and combine scores with a weighted sum that
gives the advisor_view the lion's share of credit while letting the
literal and bhāṣya views catch cases the LLM enrichment missed.

Why three indices and not one with concatenated views
-----------------------------------------------------
Concatenating literal + bhāṣya + advisor into one big text and embedding
that gives you the average direction across the three. Real semantic search
benefits from being able to match any one of the three angles strongly. The
extra storage (three vectors per verse instead of one) is trivial; the
retrieval-quality difference is large.

Storage layout
--------------
We keep three Chroma collections in artifacts/chroma/:
    advaita_literal
    advaita_bhashya
    advaita_advisor

Each holds the same set of verse_ids. We resolve a hit's full record by
reading data/corpus_enriched.jsonl (kept small enough to live in memory).
"""

from __future__ import annotations
import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import threading

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

import config
from corpus import EnrichedVerse, read_jsonl_enriched


# ──────────────────────────── Constants ────────────────────────────
COLLECTION_LITERAL = "advaita_literal"
COLLECTION_BHASHYA = "advaita_bhashya"
COLLECTION_ADVISOR = "advaita_advisor"

# Tier weights — multiplied into the cosine similarity at retrieval time.
# Same logic as before: primary scripture and Śaṅkara's pen outrank later
# voices when the cosine score is otherwise comparable.
TIER_WEIGHTS = {"primary": 1.10, "shankara": 1.10, "supporting": 1.00}

# View weights — how much each view's score contributes to the combined
# score per verse. The advisor view dominates because it is the one
# designed to bridge the language gap; literal and bhāṣya are insurance
# against the enrichment pipeline missing a topic.
VIEW_WEIGHTS = {"advisor": 0.55, "literal": 0.25, "bhashya": 0.20}


# ──────────────────────────── Hit dataclass ────────────────────────────
@dataclass
class Hit:
    """One retrieval result, post-merge across the three views."""
    verse: EnrichedVerse
    combined_score: float                                       # used for ranking
    view_scores: dict[str, float] = field(default_factory=dict) # diagnostics

    def __repr__(self) -> str:
        v = self.verse
        return (f"Hit({v.verse_ref}, tier={v.tier}, "
                f"score={self.combined_score:.3f}, views={self.view_scores})")

    def to_dict(self) -> dict:
        """Flatten a Hit to a JSON-serializable dict so the advisor can carry
        it in dspy.Prediction (which is pickled during GEPA optimization), and
        so the metric can read its fields without importing this module."""
        v = self.verse
        return {
            "verse_id": v.verse_id,
            "verse_ref": v.verse_ref,
            "work": v.work,
            "work_display": v.work_display,
            "section": v.section,
            "tier": v.tier,
            "translation": v.translation,
            "translator": v.translator,
            "bhashya": v.bhashya,
            "bhashya_translator": v.bhashya_translator,
            "paraphrase": v.paraphrase,
            "themes": list(v.themes),
            "life_situations": list(v.life_situations),
            "emotions_addressed": list(v.emotions_addressed),
            "hypothetical_questions": list(v.hypothetical_questions),
            "score": self.combined_score,
            "view_scores": dict(self.view_scores),
            # Legacy alias the old metric used:
            "meta": {
                "verse_ref": v.verse_ref,
                "work": v.work,
                "section": v.section,
                "tier": v.tier,
            },
        }


# ──────────────────────────── Internals ────────────────────────────
_chroma_client: chromadb.api.ClientAPI | None = None
_chroma_lock = threading.Lock()


def _client() -> chromadb.api.ClientAPI:
    """Return a process-wide singleton Chroma client.

    Creating a new PersistentClient per call causes a SQLite race condition
    when multiple GEPA threads call the retriever concurrently.
    """
    global _chroma_client
    if _chroma_client is None:
        with _chroma_lock:
            if _chroma_client is None:
                _chroma_client = chromadb.PersistentClient(
                    path=str(config.CHROMA_DIR),
                    settings=Settings(anonymized_telemetry=False),
                )
    return _chroma_client


def _embedder() -> SentenceTransformer:
    return SentenceTransformer(config.EMBED_MODEL, device=config.EMBED_DEVICE)


def _record_metadata(v: EnrichedVerse) -> dict:
    """Metadata stored alongside each chroma record so the retriever can
    filter and report without re-loading the JSONL on every call.

    chromadb requires scalar metadata values, so list-valued fields (themes,
    emotions) are joined with semicolons. The choice of ';' is safe because
    neither chroma nor our snake_case theme keys contain that character.
    """
    return {
        "verse_id": v.verse_id,
        "verse_ref": v.verse_ref,
        "work": v.work,
        "tier": v.tier,
        "section": v.section,
        "themes_csv": ";".join(v.themes),
        "emotions_csv": ";".join(v.emotions_addressed),
    }


# ──────────────────────────── Index build ────────────────────────────
def build_index(corpus_path: Path | None = None) -> dict[str, int]:
    """(Re)build all three view-indices from the enriched corpus.

    Returns a dict {view_name: n_records} for confirmation. The function is
    safe to re-run; it deletes existing collections first so partial state
    from a prior crash doesn't pollute results.
    """
    corpus_path = corpus_path or (config.DATA_DIR / "corpus_enriched.jsonl")
    if not corpus_path.exists():
        raise SystemExit(
            f"No enriched corpus at {corpus_path}.\n"
            f"Pipeline: download_sources.py → ingest_corpus.py → "
            f"enrich_corpus.py → knowledge_base.py --build"
        )

    print(f"Loading embedding model: {config.EMBED_MODEL} on {config.EMBED_DEVICE}")
    embedder = _embedder()

    client = _client()
    # Drop existing collections; build_index is "rebuild from scratch"
    for name in (COLLECTION_LITERAL, COLLECTION_BHASHYA, COLLECTION_ADVISOR):
        try:
            client.delete_collection(name)
        except Exception:
            pass

    coll_literal = client.create_collection(
        COLLECTION_LITERAL, metadata={"hnsw:space": "cosine"})
    coll_bhashya = client.create_collection(
        COLLECTION_BHASHYA, metadata={"hnsw:space": "cosine"})
    coll_advisor = client.create_collection(
        COLLECTION_ADVISOR, metadata={"hnsw:space": "cosine"})

    verses = list(read_jsonl_enriched(corpus_path))
    print(f"Indexing {len(verses)} verses across 3 views ...")

    counts = {"literal": 0, "bhashya": 0, "advisor": 0}

    # We batch by view so each call to encode() is efficient. For 3000 verses
    # at small-batch BGE this is a few seconds total per view, much faster
    # than one-at-a-time embedding.
    BATCH = 64
    for view_name, view_fn, coll in (
        ("literal", lambda v: v.literal_view(), coll_literal),
        ("bhashya", lambda v: v.bhashya_view(), coll_bhashya),
        ("advisor", lambda v: v.advisor_view(), coll_advisor),
    ):
        # Skip verses whose view is empty. A verse without a bhāṣya simply
        # doesn't appear in the bhāṣya index — the merger handles partial
        # coverage cleanly.
        records = [(v, view_fn(v)) for v in verses]
        records = [(v, t) for v, t in records if t.strip()]

        for i in tqdm(range(0, len(records), BATCH), desc=f"  view: {view_name}"):
            chunk = records[i:i + BATCH]
            ids = [v.verse_id for v, _ in chunk]
            texts = [t for _, t in chunk]
            metas = [_record_metadata(v) for v, _ in chunk]

            vectors = embedder.encode(
                texts,
                normalize_embeddings=True,
                show_progress_bar=False,
                batch_size=BATCH,
            ).tolist()
            coll.add(ids=ids, embeddings=vectors,
                     documents=texts, metadatas=metas)

        counts[view_name] = len(records)

    print(f"Index built: {counts}")
    return counts


# ──────────────────────────── Retriever ────────────────────────────
class AdvaitaRetriever:
    """Multi-view retriever returning Hit objects backed by EnrichedVerse.

    Construction loads the enriched corpus into memory (≈3000 records,
    ≈10 MB) so we can resolve hits to full records without per-call disk
    reads. This matters during GEPA optimization, which calls retrieve()
    hundreds of times per evaluation pass.

    The retriever is intentionally light: it doesn't filter by metadata,
    by tier, or by emotion at query time. Filtering happens at scoring
    (TIER_WEIGHTS) and at the SelectPassages stage downstream. Keeping
    retrieval permissive and selection picky is more robust than the
    reverse — when retrieval over-filters, you can never recover the
    missed verse later in the pipeline.
    """

    def __init__(self, top_k: int = config.TOP_K_RETRIEVE,
                 corpus_path: Path | None = None):
        self.top_k = top_k
        self._embedder: SentenceTransformer | None = None
        self._coll_literal = None
        self._coll_bhashya = None
        self._coll_advisor = None

        cp = corpus_path or (config.DATA_DIR / "corpus_enriched.jsonl")
        self._verses_by_id: dict[str, EnrichedVerse] = {
            v.verse_id: v for v in read_jsonl_enriched(cp)
        }

    def _ensure(self):
        """Lazy-load embedder and collections. We avoid loading at __init__
        so a process that only needs the corpus mapping (e.g. the metric)
        doesn't pay the SentenceTransformer load time."""
        if self._embedder is None:
            self._embedder = _embedder()
        if self._coll_advisor is None:
            client = _client()
            self._coll_literal = client.get_collection(COLLECTION_LITERAL)
            self._coll_bhashya = client.get_collection(COLLECTION_BHASHYA)
            self._coll_advisor = client.get_collection(COLLECTION_ADVISOR)

    def search(self, query: str, k: int | None = None) -> list[Hit]:
        """Run the query against all three views, merge by verse_id, and
        return the top-k Hits sorted by combined score."""
        self._ensure()
        k = k or self.top_k

        q_emb = self._embedder.encode(
            [query], normalize_embeddings=True, show_progress_bar=False
        ).tolist()

        # Over-fetch from each view; we want enough overlap that the merge
        # has something to work with. 3*k per view is a reasonable upper
        # bound: large enough to catch verses one view ranked low and another
        # ranked high, small enough that Chroma's HNSW stays fast.
        per_view_k = max(8, k * 3)

        view_results: dict[str, list[tuple[str, float, dict]]] = {}
        for name, coll in (("literal", self._coll_literal),
                           ("bhashya", self._coll_bhashya),
                           ("advisor", self._coll_advisor)):
            r = coll.query(query_embeddings=q_emb, n_results=per_view_k)
            ids = r["ids"][0]
            dists = r["distances"][0]   # cosine distance, in [0, 2]
            metas = r["metadatas"][0]
            view_results[name] = list(zip(ids, dists, metas))

        # Merge: for each verse_id seen in any view, compute its combined
        # score as Σ_v VIEW_WEIGHTS[v] * cos_sim(v) * tier_weight, where any
        # view that didn't return that verse contributes 0. This is a soft
        # voting scheme: a verse that appears strongly in one view but not
        # others can still rank highly if that one view's weight is enough.
        per_verse: dict[str, dict[str, float]] = {}
        per_verse_meta: dict[str, dict] = {}
        for view_name, results in view_results.items():
            for vid, dist, meta in results:
                cos_sim = 1.0 - dist
                per_verse.setdefault(vid, {})[view_name] = cos_sim
                per_verse_meta[vid] = meta

        hits: list[Hit] = []
        for vid, view_scores in per_verse.items():
            tier = per_verse_meta[vid].get("tier", "supporting")
            tw = TIER_WEIGHTS.get(tier, 1.0)
            combined = sum(
                VIEW_WEIGHTS[v] * view_scores.get(v, 0.0)
                for v in VIEW_WEIGHTS
            ) * tw
            verse = self._verses_by_id.get(vid)
            if verse is None:
                # Index has it but corpus file doesn't — corpus and index
                # have drifted. Skip rather than fabricate a record.
                continue
            hits.append(Hit(verse=verse,
                            combined_score=combined,
                            view_scores=view_scores))

        hits.sort(key=lambda h: h.combined_score, reverse=True)
        return hits[:k]

    def search_many(self, queries: Iterable[str],
                    k_per: int | None = None) -> list[Hit]:
        """Run multiple queries (e.g. from PlanRetrieval) and dedupe by
        verse_id, keeping the highest combined score across queries."""
        seen: dict[str, Hit] = {}
        for q in queries:
            for h in self.search(q, k=k_per):
                cur = seen.get(h.verse.verse_id)
                if cur is None or h.combined_score > cur.combined_score:
                    seen[h.verse.verse_id] = h
        out = list(seen.values())
        out.sort(key=lambda h: h.combined_score, reverse=True)
        return out


# ──────────────────────────── Formatter for the LLM ────────────────────────────
def format_hits_for_llm(hits: list[Hit]) -> str:
    """Render hits for the SelectPassages and SynthesizeAdvice prompts.

    We expose the verse_ref (so the synthesizer can cite it), the literal
    translation (so the synthesizer can quote it lightly), the bhāṣya
    snippet (so the synthesizer can ground its claims), and the advisor-view
    fields (so the synthesizer knows *why* this verse is being suggested for
    this user).

    Each hit is bounded in length so the prompt stays tractable on a 26B
    local model with an 8k context window.
    """
    blocks = []
    for i, h in enumerate(hits, start=1):
        v = h.verse
        # Use "Passage N:" prefix (not "[N]") so the LM cannot confuse the
        # integer position with a verse chapter.verse reference like "16.5".
        block = [f"Passage {i}: {v.verse_ref} — {v.work_display}, {v.section_display}"]
        block.append(f"    tier: {v.tier}   score: {h.combined_score:.3f}")
        if v.translation:
            block.append(f"    Translation: {v.translation.strip()[:600]}")
        if v.bhashya:
            block.append(f"    Bhāṣya (Śaṅkara): {v.bhashya.strip()[:800]}")
        if v.paraphrase:
            block.append(f"    Teaching: {v.paraphrase}")
        if v.life_situations:
            block.append(f"    Speaks to: {'; '.join(v.life_situations)}")
        if v.emotions_addressed:
            block.append(f"    Addresses: {', '.join(v.emotions_addressed)}")
        if v.themes:
            block.append(f"    Themes: {', '.join(v.themes)}")
        blocks.append("\n".join(block))
    return "\n\n".join(blocks)


# Alias kept so that advisor.py — and any prior code that imported the old
# name — works without modification. Both names refer to the same function
# because the new "passages" the advisor sees ARE Hit objects backed by
# EnrichedVerse records; the rendering is identical.
format_passages_for_llm = format_hits_for_llm


# ──────────────────────────── CLI ────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true",
                    help="(Re)build the multi-view index from corpus_enriched.jsonl")
    ap.add_argument("--query", type=str, default=None,
                    help="Run a test query against the index")
    args = ap.parse_args()

    if args.build:
        build_index()
        return

    if args.query:
        retr = AdvaitaRetriever()
        hits = retr.search(args.query)
        print(format_hits_for_llm(hits))
        return

    ap.print_help()


if __name__ == "__main__":
    main()
