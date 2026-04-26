"""
ingest_corpus.py — run the parsers and produce data/corpus.jsonl.

This script lives between download_sources.py (which gets bytes onto disk)
and enrich_corpus.py (which adds LLM-derived fields). Its specific job:

    1. Walk each enabled source in the registry.
    2. Dispatch to its parser, which yields Verse records.
    3. Merge records across sources by verse_ref.
       - The Gītā parser yields verses with translation but no bhāṣya.
       - The Sastry parser yields verses with bhāṣya but spotty translation.
       - We want one record per verse, with both populated when possible.
    4. Write the merged stream as JSONL to data/corpus.jsonl.

Why merge by verse_ref rather than verse_id
-------------------------------------------
The Gītā parser uses work='bhagavad_gita' and the Sastry parser uses
work='bhagavad_gita_bhashya'. Their verse_ids therefore differ (different
work prefix), but their verse_refs match — both render as 'BG 2.47'. We
key the merge on verse_ref since that's the reader-facing canonical citation.

Conflict policy when merging
----------------------------
- Translation: keep whichever record has it; if both, prefer the one whose
  source_key is in the GITA_TEXT_PRIORITY list. (We want the modern, clean
  Sivananda over Sastry's archaic English-of-Śaṅkara-paraphrasing-the-verse.)
- Bhāṣya: only one source produces this; conflicts shouldn't happen.
- Sanskrit / transliteration / word_meanings: prefer gita_json; richer.
"""

from __future__ import annotations
import argparse
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from tqdm import tqdm

import config
from corpus import Verse, write_jsonl
from sources_registry import enabled_sources, by_key, Source

# Parsers
from parsers import gita_json as parser_gita_json
from parsers import sastry_archive as parser_sastry


# When two sources both have a translation, this list decides which wins
GITA_TEXT_PRIORITY = ("gita_json_core", "sastry_gita_bhashya")


def _parse_source(src: Source, raw_dir: Path) -> Iterable[Verse]:
    """Dispatch to the right parser for a registry entry.

    Each parser is documented to take a directory and return an iterable of
    Verses; this function is just a switch table.
    """
    if src.parser == "gita_json":
        # The gita_json parser can take both the core dir and (optionally) a
        # translations dir. We pass the same dir for both since the downloader
        # puts all gita_json* files into per-source folders.
        if src.key == "gita_json_core":
            translations_dir = raw_dir.parent / "gita_json_translations"
            return parser_gita_json.parse(
                raw_dir,
                translations_dir if translations_dir.exists() else None,
            )
        # The translations source is "consumed" alongside core, not parsed alone
        return iter(())

    if src.parser == "sastry_archive":
        return parser_sastry.parse(raw_dir)

    if src.parser == "wisdomlib_html":
        # Stub for now — see parsers/wisdomlib_html.py to implement.
        # We don't fail the whole ingest just because one parser is unimplemented.
        print(f"[ingest] wisdomlib_html parser not implemented yet — skipping {src.key}")
        return iter(())

    if src.parser == "thibaut_sbe":
        print(f"[ingest] thibaut_sbe parser not implemented yet — skipping {src.key}")
        return iter(())

    if src.parser == "plain_text":
        # Reserved for user-dropped texts; future work
        return iter(())

    raise ValueError(f"Unknown parser type: {src.parser}")


def _merge(records: list[Verse]) -> list[Verse]:
    """Merge multiple parser outputs into one record per verse_ref.

    The output preserves the order of first appearance, so the corpus.jsonl
    file is naturally chapter-then-verse ordered.
    """
    by_ref: dict[str, Verse] = {}
    order: list[str] = []

    for r in records:
        if r.verse_ref not in by_ref:
            by_ref[r.verse_ref] = r
            order.append(r.verse_ref)
            continue

        existing = by_ref[r.verse_ref]

        # Translation: pick higher-priority source if both have one
        new_translation = existing.translation
        new_translator = existing.translator
        if r.translation and (
            not existing.translation
            or _priority(r.source_key) < _priority(existing.source_key)
        ):
            new_translation = r.translation
            new_translator = r.translator

        # Bhashya: only one source typically has it, take whichever isn't blank
        new_bhashya = existing.bhashya or r.bhashya
        new_bhashya_tr = existing.bhashya_translator or r.bhashya_translator

        # Sanskrit family of fields: prefer the existing record if it has them,
        # else take from the new record
        merged = Verse(
            verse_id=existing.verse_id,
            work=existing.work,           # keep the work_display of whichever came first
            work_display=existing.work_display,
            verse_ref=existing.verse_ref,
            tier=_choose_tier(existing.tier, r.tier),
            section=existing.section or r.section,
            section_display=existing.section_display or r.section_display,
            translation=new_translation,
            translator=new_translator,
            sanskrit=existing.sanskrit or r.sanskrit,
            transliteration=existing.transliteration or r.transliteration,
            word_meanings=existing.word_meanings or r.word_meanings,
            bhashya=new_bhashya,
            bhashya_translator=new_bhashya_tr,
            source_key=existing.source_key + "+" + r.source_key,
            license=existing.license or r.license,
        )
        by_ref[r.verse_ref] = merged

    return [by_ref[k] for k in order]


def _priority(source_key: str) -> int:
    """Lower is higher-priority. Sources not in the priority list rank last."""
    for i, key in enumerate(GITA_TEXT_PRIORITY):
        if source_key == key or source_key.startswith(key + "+") or source_key.endswith("+" + key):
            return i
    return 99


def _choose_tier(a: str, b: str) -> str:
    """When two records merge, the tier of the merged verse is the most
    'authoritative' of the two: primary > shankara > supporting.

    Why primary > shankara: when we have both the verse text (primary) and
    Śaṅkara's bhāṣya on it (shankara) folded into one record, the verse
    itself is what the citation refers to — so primary wins."""
    rank = {"primary": 0, "shankara": 1, "supporting": 2}
    return a if rank.get(a, 9) <= rank.get(b, 9) else b


# ──────────────────────────── CLI ────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(config.DATA_DIR / "corpus.jsonl"))
    args = ap.parse_args()

    raw_root = config.DATA_DIR / "raw"
    if not raw_root.exists():
        raise SystemExit("data/raw/ doesn't exist. Run download_sources.py first.")

    all_records: list[Verse] = []
    for src in enabled_sources():
        raw_dir = raw_root / src.key
        if not raw_dir.exists():
            print(f"[ingest] {src.key}: no files at {raw_dir}; skipping")
            continue
        print(f"[ingest] parsing {src.key} via {src.parser}")
        try:
            n_before = len(all_records)
            for v in _parse_source(src, raw_dir):
                if v.has_content():
                    all_records.append(v)
            print(f"[ingest]   yielded {len(all_records) - n_before} records")
        except Exception as e:
            print(f"[ingest] {src.key} failed: {e}")

    print(f"[ingest] merging {len(all_records)} records by verse_ref ...")
    merged = _merge(all_records)
    print(f"[ingest] {len(merged)} unique verses after merge")

    out_path = Path(args.out)
    n = write_jsonl(merged, out_path)
    print(f"[ingest] wrote {n} verses to {out_path}")
    print(f"[ingest] next: python enrich_corpus.py")


if __name__ == "__main__":
    main()
