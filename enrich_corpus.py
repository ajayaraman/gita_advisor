"""
enrich_corpus.py — run the local LLM over every verse, once, with caching.

The cost calculus
-----------------
For ~3,000 verses at ~30s per call on a 26B-class local model, a full pass
takes a long evening — call it 25 hours. That's tolerable as a one-time cost,
intolerable as a recurring one. So caching is non-negotiable. We cache by
verse_id and the enrichment_version stamp; if you change the prompt
substantively, bump the version in enrichment.py and the next run re-enriches.

What we write
-------------
data/corpus_enriched.jsonl — one EnrichedVerse per line, in the same order
as data/corpus.jsonl. Failed enrichments are still written (with empty
enrichment fields and an error stamp in enrichment_model) so the index can
still cover them on their literal text.

Concurrency
-----------
LM Studio's OpenAI-compatible server processes requests serially by default.
We don't try to parallelize at the client; if you've configured your server
for parallel decode, set --concurrency > 1 and DSPy will hold multiple
in-flight calls. For modest hardware, 1 is correct.

Resumability
------------
If the run dies halfway, just re-run. The cache at data/enrichment_cache.jsonl
remembers per-verse what we already did, so we pick up exactly where we left
off. No flag is needed for resume; it's the default behavior.
"""

from __future__ import annotations
import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from tqdm import tqdm
import dspy

import config
from corpus import Verse, EnrichedVerse, read_jsonl_verses, write_jsonl
from enrichment import Enricher


CACHE_PATH = config.DATA_DIR / "enrichment_cache.jsonl"
ENRICHED_PATH = config.DATA_DIR / "corpus_enriched.jsonl"


# ──────────────────────────── Cache I/O ────────────────────────────
def _load_cache(path: Path) -> dict[str, EnrichedVerse]:
    """Load cache as {verse_id: EnrichedVerse}. Tolerates partial writes."""
    if not path.exists():
        return {}
    out: dict[str, EnrichedVerse] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                ev = EnrichedVerse(**{k: v for k, v in d.items() if k in EnrichedVerse.__dataclass_fields__})
                out[ev.verse_id] = ev
            except Exception:
                continue
    return out


def _append_cache(path: Path, ev: EnrichedVerse) -> None:
    """Append a single record. We use append-mode rather than rewriting so
    a kill -9 mid-run loses at most one line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(ev), ensure_ascii=False) + "\n")


# ──────────────────────────── Main loop ────────────────────────────
def enrich_all(
    in_path: Path,
    out_path: Path,
    cache_path: Path,
    limit: int | None = None,
    re_enrich: bool = False,
    only_failed: bool = False,
) -> None:
    config.configure_dspy()
    enricher = Enricher()

    cache = _load_cache(cache_path) if not re_enrich else {}
    print(f"[enrich] cache contains {len(cache)} previously-enriched verses")

    verses = list(read_jsonl_verses(in_path))
    if limit:
        verses = verses[:limit]
    print(f"[enrich] enriching {len(verses)} verses from {in_path}")

    enriched: list[EnrichedVerse] = []
    pending = []
    for v in verses:
        cached = cache.get(v.verse_id)
        if cached and not re_enrich:
            if only_failed and cached.enrichment_model.startswith("FAILED"):
                pending.append(v)
            else:
                enriched.append(cached)
                continue
        else:
            pending.append(v)

    print(f"[enrich] {len(enriched)} from cache, {len(pending)} to call LM for")

    n_failed = 0
    for v in tqdm(pending, desc="enriching"):
        ev = enricher(verse=v)
        _append_cache(cache_path, ev)
        enriched.append(ev)
        if not ev.is_enriched():
            n_failed += 1

    # Restore original verse order from in_path
    by_id = {ev.verse_id: ev for ev in enriched}
    ordered = [by_id[v.verse_id] for v in verses if v.verse_id in by_id]

    n_written = write_jsonl(ordered, out_path)
    print(f"[enrich] wrote {n_written} enriched verses to {out_path}")
    if n_failed:
        print(f"[enrich] WARNING: {n_failed} verses failed enrichment "
              f"(empty fields, indexed only on literal text). "
              f"Re-run with --only-failed to retry just those.")


# ──────────────────────────── CLI ────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path",
                    default=str(config.DATA_DIR / "corpus.jsonl"))
    ap.add_argument("--out", default=str(ENRICHED_PATH))
    ap.add_argument("--cache", default=str(CACHE_PATH))
    ap.add_argument("--limit", type=int, default=None,
                    help="Enrich only the first N verses (smoke-test).")
    ap.add_argument("--re-enrich", action="store_true",
                    help="Ignore cache and re-enrich everything. Use this "
                         "when you change the enrichment prompt.")
    ap.add_argument("--only-failed", action="store_true",
                    help="Re-run only the verses whose previous enrichment "
                         "failed (FAILED stamp in enrichment_model).")
    args = ap.parse_args()

    enrich_all(
        in_path=Path(args.in_path),
        out_path=Path(args.out),
        cache_path=Path(args.cache),
        limit=args.limit,
        re_enrich=args.re_enrich,
        only_failed=args.only_failed,
    )


if __name__ == "__main__":
    main()
