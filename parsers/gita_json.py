"""
parsers/gita_json.py — turn the gita/gita verse-indexed JSON into Verse records.

The gita/gita repo (Unlicense, public-domain dedication) gives us four files
on the static mirror:

    chapters.json     — chapter metadata (number, name, summary)
    verse.json        — per-verse Sanskrit + transliteration + word_meanings
    translation.json  — per-verse English translations keyed by author_id
    authors.json      — author metadata for the translations

Why split parsing across multiple sources_registry entries
----------------------------------------------------------
We register `gita_json_core` (the verse text) and `gita_json_translations`
(the English translations) as separate sources. Both happen to feed this one
parser. The reason for the split is that translations come and go from the
upstream repo whereas the core verse data is essentially fixed; isolating
them lets us pin only what we need.

Translator allowlist
--------------------
Not every translator in the gita/gita translations.json is public-domain.
We hard-allowlist the ones we know are safe to redistribute. Anyone not on
the list is silently skipped — adding more is a one-line change.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Iterable

from corpus import Verse


# ──────────────────────────── Translator allowlist ────────────────────────────
# The keys are the author_id values used inside translation.json. The values
# are display strings + the year we want to use for attribution.
#
# Why this list and not just "all translations":
# - Some translators in the upstream repo (e.g. ISKCON Prabhupada) have
#   active publisher rights that we shouldn't rely on regardless of how the
#   upstream chose to license its compilation.
# - Reducing translation count keeps the index lean. Three voices are plenty.
#
# If you want to add a translator, verify their public-domain status (death
# year + 70 in most jurisdictions, or pre-1929 publication for US PD), then
# add a row.
ALLOWED_TRANSLATORS: dict[str, tuple[str, int | None]] = {
    # Swami Sivananda — d. 1963 — works are widely shared by The Divine Life
    # Society in keeping with their founder's non-commercial stance.
    "sivananda":   ("Swami Sivananda", 1969),

    # Swami Tejomayananda — modern; included only because some mirrors
    # release these under permissive terms; double-check before relying on it.
    # Disabled by default to be conservative.
    # "tejomayananda": ("Swami Tejomayananda", 1995),

    # Dr. S. Sankaranarayan — translation of Śaṅkara's Gītā Bhāṣya included
    # in some forks of gita/gita; verify the specific edition. Off by default.
    # "shankara":    ("Śaṅkara (tr. Sankaranarayan)", 1990),

    # The verse text itself is not a "translation" per se but a copy of the
    # critical text plus transliteration. We include it under the synthetic
    # author key 'sanskrit'.
    "sanskrit":    ("Sanskrit text + IAST", None),
}


# ──────────────────────────── Helpers ────────────────────────────
def _verse_id(chapter: int, verse_no: int) -> str:
    """Stable global key. Format: bhagavad_gita_<chap>_<verse>, zero-padded
    to two digits so 1.10 sorts after 1.9 and lexical ordering matches numeric."""
    return f"bhagavad_gita_{chapter:02d}_{verse_no:02d}"


def _verse_ref(chapter: int, verse_no: int) -> str:
    """Citation form used by the advisor in its replies."""
    return f"BG {chapter}.{verse_no}"


def _section_display(chapter_meta: dict) -> str:
    name = chapter_meta.get("name_translation") or chapter_meta.get("name", "")
    return f"Chapter {chapter_meta.get('chapter_number', '?')}: {name}"


# ──────────────────────────── Parser entry point ────────────────────────────
def parse(raw_dir_for_core: Path, raw_dir_for_translations: Path | None = None) -> Iterable[Verse]:
    """Walk the gita/gita JSON files and yield Verse records.

    Layout expected (after download_sources.py has run):
        raw_dir_for_core/chapters.json
        raw_dir_for_core/verse.json
        [optionally]
        raw_dir_for_translations/translation.json
        raw_dir_for_translations/authors.json

    If translations are not present, we still emit Verses with sanskrit +
    transliteration + word_meanings; the `translation` field falls back to
    the transliteration so the verse isn't content-empty. (Better: enable
    the gita_json_translations source.)
    """
    chapters = _load(raw_dir_for_core / "chapters.json")
    verses_raw = _load(raw_dir_for_core / "verse.json")

    chapters_by_id = {c["chapter_number"]: c for c in chapters}

    translations_by_verse: dict[int, dict[str, str]] = {}
    authors_by_id: dict[str, str] = {}
    if raw_dir_for_translations is not None:
        translations_by_verse = _load_translations(raw_dir_for_translations / "translation.json")
        authors_by_id = _load_authors(raw_dir_for_translations / "authors.json")

    # Pick the best available translator from the allowlist, in priority order.
    # First match wins. This keeps the index from carrying redundant English
    # translations of the same verse.
    translator_priority = ["sivananda", "sanskrit"]

    for v in verses_raw:
        chap_no = v["chapter_number"]
        verse_no = v["verse_number"]
        chap_meta = chapters_by_id.get(chap_no, {})
        verse_id = _verse_id(chap_no, verse_no)

        # Sanskrit text comes from the core file. The 'text' field has it
        # in Devanāgarī, often with a trailing newline and verse number.
        sanskrit = (v.get("text") or "").strip()
        translit = (v.get("transliteration") or "").strip()
        word_mean = (v.get("word_meanings") or "").strip()

        # Try to attach an English translation
        english = ""
        translator_label = ""
        v_translations = translations_by_verse.get(v.get("id") or v.get("externalId") or -1, {})
        for key in translator_priority:
            text = v_translations.get(key) or _translation_for(v_translations, key)
            if text:
                english = text.strip()
                meta = ALLOWED_TRANSLATORS.get(key)
                if meta:
                    translator_label = meta[0]
                break

        # Fallback: if no English translation, use word-meanings as a substitute
        # so the verse isn't content-empty. Better than nothing for retrieval,
        # though enrichment will be poorer.
        if not english:
            english = word_mean or translit

        yield Verse(
            verse_id=verse_id,
            work="bhagavad_gita",
            work_display="Bhagavad Gītā",
            verse_ref=_verse_ref(chap_no, verse_no),
            tier="primary",
            section=f"chapter_{chap_no:02d}",
            section_display=_section_display(chap_meta),
            translation=english,
            translator=translator_label,
            sanskrit=sanskrit,
            transliteration=translit,
            word_meanings=word_mean,
            bhashya="",                  # Gītā Bhāṣya is brought in by the Sastry parser
            bhashya_translator="",
            source_key="gita_json_core",
            license="unlicense",
        )


# ──────────────────────────── Internals ────────────────────────────
def _load(path: Path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _load_translations(path: Path) -> dict[int, dict[str, str]]:
    """The translations file has one entry per (verse, author). Group them
    by verse_id into a {verse_id: {author_id: text}} map.

    Schema seen in the wild varies slightly between forks of gita/gita; we
    cope by trying a few key names. If parsing fails entirely we return {}
    and proceed without translations rather than blowing up the whole ingest.
    """
    if not path.exists():
        return {}
    try:
        raw = _load(path)
    except Exception as e:
        print(f"[gita_json] failed to load translations: {e}")
        return {}

    out: dict[int, dict[str, str]] = {}
    for row in raw:
        vid = row.get("verse_id") or row.get("verseNumber") or row.get("verse_number_id") or row.get("id")
        text = row.get("description") or row.get("text") or row.get("translation")
        if vid is None or not text:
            continue

        # Skip non-English rows (Ramsukhdas Hindi etc.)
        lang = (row.get("lang") or "").lower()
        if lang and lang not in ("english", "en"):
            continue

        # Map the authorName (e.g. "Swami Sivananda") to an allowlist key
        # ("sivananda") via case-insensitive substring matching. The numeric
        # author_id field alone can't match the allowlist, which is why we
        # prefer authorName here.
        name_str = str(row.get("authorName") or row.get("author_id") or row.get("author") or "").strip()
        matched_key = next(
            (k for k in ALLOWED_TRANSLATORS if k.lower() in name_str.lower()),
            None,
        )
        if matched_key is None:
            continue
        out.setdefault(int(vid), {})[matched_key] = text
    return out


def _load_authors(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    try:
        raw = _load(path)
    except Exception:
        return {}
    return {row.get("id"): row.get("name", "") for row in raw if row.get("id")}


def _translation_for(v_translations: dict, author_key: str) -> str | None:
    """Tolerant lookup: some files use 'sivananda', some 'Sivananda', etc."""
    if author_key in v_translations:
        return v_translations[author_key]
    lk = author_key.lower()
    for k, val in v_translations.items():
        if str(k).lower() == lk:
            return val
    return None
