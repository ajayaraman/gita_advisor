"""
corpus.py — the data model and on-disk storage for the verse corpus.

A note on dataclasses vs. plain dicts
-------------------------------------
We could have used dicts everywhere and saved keystrokes. We don't, because
the Verse type is the contract between five different modules — parsers,
enrichment, indexing, retrieval, and the metric — and a typed contract
catches mistakes that "I thought 'sources_cited' was a list" wouldn't.

The pipeline lifecycle of a verse
---------------------------------
    parsers.*        →  Verse           (no LLM-derived fields)
    enrichment.py    →  EnrichedVerse   (with LLM-derived fields)
    knowledge_base   →  reads EnrichedVerse, writes 3 embeddings per verse
    advisor.py       →  receives EnrichedVerse via retriever hits
    metrics.py       →  uses verse_id for exact citation grounding

Storage choice
--------------
JSONL on disk. Each line is a verse. Why not Parquet, sqlite, etc.?
- Easy to grep
- Easy to diff in PRs
- Easy for a human to spot-check enrichment quality (the whole point)
- We never need to scan more than a few thousand lines, so format doesn't matter
"""

from __future__ import annotations
import json
from dataclasses import dataclass, field, asdict, fields
from pathlib import Path
from typing import Iterable, Iterator


# ──────────────────────────── Verse: the raw record ────────────────────────────
@dataclass
class Verse:
    """A natural unit of scripture: one verse, one mantra, one sūtra.

    The required fields are minimal — every parser must produce at least these.
    Optional fields (sanskrit, transliteration, bhashya, ...) are filled when
    the source provides them.

    `verse_id` is the global unique key. Convention:
        '<work_slug>_<section_slug>_<verse_number>'
    e.g. 'bhagavad_gita_02_47', 'mundaka_upanishad_2_1_3'.

    `verse_ref` is the human-readable citation form:
        e.g. 'BG 2.47', 'Muṇḍaka Up. 2.1.3', 'Vivekacūḍāmaṇi 11'.
    The advisor's response uses this exact string in citations.
    """
    # Identity — required for every record
    verse_id: str
    work: str
    work_display: str
    verse_ref: str
    tier: str                         # primary | shankara | supporting

    # Section/chapter info — required when the work has chapters
    section: str = ""                 # 'chapter_02'
    section_display: str = ""         # 'Chapter 2: Sāṅkhya Yoga'

    # Content — at least one of {translation, bhashya} must be non-empty
    translation: str = ""             # English translation of the verse itself
    translator: str = ""              # who translated it (for attribution)

    sanskrit: str = ""                # original Devanāgarī
    transliteration: str = ""         # IAST roman transliteration
    word_meanings: str = ""           # word-by-word gloss when present

    bhashya: str = ""                 # Śaṅkara's commentary on this verse, if any
    bhashya_translator: str = ""      # who translated the bhāṣya

    # Provenance for accountability and license display
    source_key: str = ""              # the registry key this came from
    license: str = ""                 # license tag from registry

    def has_content(self) -> bool:
        """Used by parsers/loaders to drop empty records before they pollute
        the index. A 'verse' with only a verse_id and no actual text is junk."""
        return bool(self.translation.strip() or self.bhashya.strip())


# ──────────────────────────── EnrichedVerse: with LLM extractions ────────────────
@dataclass
class EnrichedVerse(Verse):
    """A Verse + the structured fields produced by the offline LLM pass.

    Every list defaults to empty so a verse that fails enrichment can still
    be stored (without enrichment, indexed only on its literal text/bhāṣya).
    """
    # The plain-English statement of what the verse teaches. Ideally 1–2
    # sentences. This is what the synthesizer reads downstream.
    paraphrase: str = ""

    # Vedānta concepts engaged by the verse. Tradition-native vocabulary.
    # Examples: 'karma_yoga', 'vairagya', 'sakshi', 'two_truths', 'adhyasa'.
    themes: list[str] = field(default_factory=list)

    # Mundane life situations where this verse would help. User-language.
    # Examples: 'facing failure after sustained effort', 'watching a parent decline'.
    life_situations: list[str] = field(default_factory=list)

    # Emotions addressed, from a small consistent vocabulary.
    # See enrichment.py EMOTION_VOCAB for the closed set.
    emotions_addressed: list[str] = field(default_factory=list)

    # What does this verse ask the seeker to do or shift?
    practical_teaching: str = ""

    # Hypothetical questions a real person might bring to this verse.
    # These are gold for retrieval; they bridge the language gap.
    hypothetical_questions: list[str] = field(default_factory=list)

    # Quality / debugging
    enrichment_model: str = ""        # which LM produced these fields
    enrichment_version: int = 1       # bump when the prompt changes substantively

    # ---- Derived "views" used at indexing time ----
    def literal_view(self) -> str:
        """The literal English translation, lightly enriched with the Sanskrit
        if available. Best for queries that share lexical features with the text."""
        parts = []
        if self.translation:
            parts.append(self.translation.strip())
        if self.transliteration:
            parts.append(f"({self.transliteration.strip()})")
        return "\n".join(parts)

    def bhashya_view(self) -> str:
        """Śaṅkara's commentary on this verse. Best for queries about the
        Vedāntic explanation rather than the verse text itself."""
        return self.bhashya.strip()

    def advisor_view(self) -> str:
        """The composed view that bridges the language gap.

        This is what makes the user-question-→-verse mapping work. A user who
        types 'I feel hollow even though I got everything I wanted' will not
        find anything in the Sanskrit. They will find a near-neighbor in this
        view if the enrichment did its job.
        """
        bits = []
        if self.paraphrase:
            bits.append(f"Teaching: {self.paraphrase}")
        if self.life_situations:
            bits.append(
                "Speaks to: " + "; ".join(self.life_situations)
            )
        if self.emotions_addressed:
            bits.append(
                "Addresses: " + ", ".join(self.emotions_addressed)
            )
        if self.themes:
            bits.append(
                "Themes: " + ", ".join(self.themes)
            )
        if self.hypothetical_questions:
            bits.append(
                "Questions this answers:\n  - "
                + "\n  - ".join(self.hypothetical_questions)
            )
        if self.practical_teaching:
            bits.append(f"Practical shift: {self.practical_teaching}")
        return "\n".join(bits)

    def is_enriched(self) -> bool:
        """Did enrichment populate at least the minimum-viable fields?"""
        return bool(self.paraphrase) and bool(self.life_situations) and bool(self.hypothetical_questions)


# ──────────────────────────── On-disk JSONL ────────────────────────────
def write_jsonl(records: Iterable[Verse], path: Path) -> int:
    """Write a stream of records as JSONL. Returns count written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl_verses(path: Path) -> Iterator[Verse]:
    """Read a JSONL file as Verse records. Skips lines we can't parse."""
    if not path.exists():
        return
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                yield _verse_from_dict(d, Verse)
            except Exception as e:
                print(f"[corpus] skipping malformed line {line_no} in {path}: {e}")


def read_jsonl_enriched(path: Path) -> Iterator[EnrichedVerse]:
    """Read a JSONL file as EnrichedVerse records."""
    if not path.exists():
        return
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                yield _verse_from_dict(d, EnrichedVerse)
            except Exception as e:
                print(f"[corpus] skipping malformed line {line_no} in {path}: {e}")


def _verse_from_dict(d: dict, cls):
    """Construct a Verse/EnrichedVerse, ignoring keys the dataclass doesn't know.

    This forward-compatibility matters: if a future version adds a field, old
    JSONL files should still load. And if enrichment adds extra debug fields,
    we don't want the dataclass to choke on them.
    """
    valid = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in d.items() if k in valid})
