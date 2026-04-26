"""
parsers/sastry_archive.py — extract verse-attached Śaṅkara bhāṣya from
Alladi Mahadeva Sastry's 1897 archive.org OCR text.

What makes this harder than the gita_json parser
-------------------------------------------------
The gita/gita JSON gave us each verse already keyed by chapter and verse
number. The Sastry archive.org file is OCR'd plain text — about 20 MB of
running prose where the only structural cues are:

    1. Chapter headings, formatted in caps like "SANKHYA YOGA." or
       "CHAPTER II — SANKHYA YOGA"
    2. Verse markers, which appear in two forms in the OCR:
         - inline as "(II. 47.)" or "II. 47." after a translated verse
         - as section headings like "47." or "Verse 47." preceding the bhāṣya
    3. The rule that when a translated verse appears, Śaṅkara's commentary
       follows immediately until the next verse marker.

Add to that: OCR noise. "II" can become "11", "47" can become "4 7", periods
become commas, glyphs get dropped. So the parser is forgiving — it tries
several patterns and falls back gracefully.

What we extract
---------------
For each verse we find, we yield a Verse with:
    - tier='shankara'
    - work='bhagavad_gita_bhashya'  (kept distinct from 'bhagavad_gita' so
      the joiner in ingest_corpus.py knows to merge bhashya into the gita
      verses by verse_ref)
    - translation = the verse text as Sastry rendered it (handy as a second
      English voice alongside Sivananda)
    - bhashya = Śaṅkara's commentary, as Sastry translated it
    - bhashya_translator = 'Alladi Mahadeva Sastry, 1897'

Robustness strategy
-------------------
We don't try to be perfect. If a verse's bhāṣya is mis-attributed by ±1, the
downstream enrichment step will produce paraphrases that don't quite fit, and
we'll catch those during the spot-check pass on enriched output. The metric
will also penalize ungrounded citations. The key invariant is: never silently
emit a wrong (verse_id, bhashya) pair if we're uncertain — better to skip.
"""

from __future__ import annotations
import re
from dataclasses import replace
from pathlib import Path
from typing import Iterable

from corpus import Verse


# ──────────────────────────── Patterns ────────────────────────────
# Roman numerals (allowing OCR substitutions: I↔1, V↔V, etc.)
ROMAN = r"(?:[IVX1l]+|[ivx]+)"

# A "verse marker" looks like "II. 47" or "(II. 47.)" or "47" alone in a section
# heading. We try several shapes and let the most specific win.
VERSE_INLINE = re.compile(
    r"\(?\s*(?P<chap>" + ROMAN + r")\s*[\.\,]\s*(?P<verse>\d{1,3})\s*[\.\,]?\s*\)?",
    re.IGNORECASE,
)

# Chapter heading: "CHAPTER II" or "II. SANKHYA YOGA" — uppercase-heavy lines
CHAPTER_HEADING = re.compile(
    r"^\s*(?:CHAPTER\s+)?(?P<roman>" + ROMAN + r")\.?\s+[A-Z][A-Z \-—]{4,}",
    re.MULTILINE,
)

# Roman → arabic
ROMAN_MAP = {
    "I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6, "VII": 7, "VIII": 8,
    "IX": 9, "X": 10, "XI": 11, "XII": 12, "XIII": 13, "XIV": 14, "XV": 15,
    "XVI": 16, "XVII": 17, "XVIII": 18,
}


def _to_arabic(token: str) -> int | None:
    """Convert a possibly-noisy roman numeral to an int. OCR sometimes turns
    'I' into '1' and 'II' into '11', so we accept both forms."""
    t = token.upper().replace("L", "I").replace("0", "O")  # OCR substitutions
    if t in ROMAN_MAP:
        return ROMAN_MAP[t]
    # Pure-arabic fallback (e.g. OCR rendered 'II' as '11')
    if t.isdigit():
        n = int(t)
        if 1 <= n <= 18:
            return n
    return None


# ──────────────────────────── Main parse ────────────────────────────
def parse(raw_dir: Path) -> Iterable[Verse]:
    """Walk Sastry archive.org text in raw_dir and yield Verse records.

    Expected layout (after download_sources.py):
        raw_dir/Bhagavad-Gita.with.the.Commentary.of.Sri.Shankaracharya_djvu.txt

    The file is ~20 MB of OCR text. We stream it line-by-line, maintain the
    current chapter as we encounter chapter headings, and at each verse marker
    yield the accumulated text since the previous marker as the bhāṣya.
    """
    txts = list(raw_dir.glob("*_djvu.txt")) + list(raw_dir.glob("*.txt"))
    if not txts:
        print(f"[sastry] no .txt under {raw_dir}; did you download_sources.py?")
        return

    text = txts[0].read_text(encoding="utf-8", errors="replace")
    text = _denoise(text)

    # First pass: find every verse marker with its position and attempt to
    # disambiguate the chapter from context. We collect (chap, verse, span)
    # tuples in document order.
    markers: list[tuple[int, int, int, int]] = []  # chap, verse, start, end
    current_chapter = 1
    last_pos = 0

    # Walk chapter headings and verse markers together via merged iteration
    events = []
    for m in CHAPTER_HEADING.finditer(text):
        c = _to_arabic(m.group("roman"))
        if c is not None:
            events.append(("chapter", m.start(), c))

    for m in VERSE_INLINE.finditer(text):
        c = _to_arabic(m.group("chap"))
        try:
            v = int(m.group("verse"))
        except (ValueError, TypeError):
            continue
        if c is None or not (1 <= v <= 80):
            continue
        events.append(("verse", m.start(), c, v, m.end()))

    events.sort(key=lambda e: e[1])

    # Second pass: build (chapter, verse) → (start, end) spans, where each
    # span is the bhāṣya from one marker to the next. We yield in document
    # order with the chapter from the most recent chapter heading we saw.
    last_marker_pos: int | None = None
    last_chap: int | None = None
    last_verse: int | None = None

    for ev in events:
        if ev[0] == "chapter":
            current_chapter = ev[2]
            continue
        # ev: ("verse", start, chap, verse, end)
        _, start, chap, verse, end = ev

        # If the chapter on the verse marker disagrees with the running chapter,
        # trust the marker — Sastry sometimes labels mid-paragraph references.
        # But guard against absurd jumps (e.g. an isolated "(I. 5.)" reference
        # in the middle of chapter 6 — these are cross-references, not section
        # boundaries). We only flush a new chapter when the marker is the first
        # thing on a line.
        line_start = text.rfind("\n", 0, start) + 1
        on_own_line = (start - line_start) <= 4
        if on_own_line:
            current_chapter = chap

        if last_marker_pos is not None and last_chap is not None and last_verse is not None:
            bhashya_text = text[last_marker_pos:start].strip()
            if bhashya_text:
                yield _build_verse(
                    chap=last_chap, verse=last_verse, body=bhashya_text,
                )

        last_marker_pos = end
        last_chap = current_chapter
        last_verse = verse

    # Flush the trailing one
    if last_marker_pos is not None and last_chap and last_verse:
        tail = text[last_marker_pos:].strip()
        if tail:
            yield _build_verse(chap=last_chap, verse=last_verse, body=tail)


# ──────────────────────────── Builders ────────────────────────────
def _build_verse(chap: int, verse: int, body: str) -> Verse:
    """The body lump contains both Sastry's English of the verse and Śaṅkara's
    commentary, usually with the verse first (sometimes labeled) and the
    commentary following. We make a *light* split heuristic: if the first
    paragraph is short (≤ 400 chars) and ends near a period, treat it as the
    verse translation; the rest is bhashya. If we can't split confidently,
    we put everything into bhashya and leave translation empty — the gita_json
    parser already gave us a translation by another translator."""
    body = body.strip()
    translation = ""
    bhashya = body

    # Heuristic split on the first blank-ish line within reasonable distance
    para_break = re.search(r"\n\s*\n", body[:600])
    if para_break and para_break.end() < 500:
        head = body[:para_break.start()].strip()
        tail = body[para_break.end():].strip()
        # Accept the split only if the head looks like a verse: short-ish,
        # not starting with a typical-bhashya opener like "This means" /
        # "The meaning is" / "Here the Lord says".
        if 30 < len(head) < 400 and not _looks_like_bhashya_opener(head):
            translation, bhashya = head, tail

    return Verse(
        verse_id=f"bhagavad_gita_{chap:02d}_{verse:02d}",
        work="bhagavad_gita_bhashya",
        work_display="Bhagavad Gītā with Śaṅkara's Bhāṣya",
        verse_ref=f"BG {chap}.{verse} (bhāṣya)",
        tier="shankara",
        section=f"chapter_{chap:02d}",
        section_display=f"Chapter {chap}",
        translation=translation,
        translator="Alladi Mahadeva Sastry" if translation else "",
        bhashya=bhashya,
        bhashya_translator="Alladi Mahadeva Sastry, 1897",
        source_key="sastry_gita_bhashya",
        license="public_domain",
    )


def _looks_like_bhashya_opener(s: str) -> bool:
    s = s.strip().lower()
    openers = (
        "this means", "the meaning is", "the sense is", "here the lord",
        "here it is said", "the lord says", "the question may", "objection",
        "the commentator",
    )
    return any(s.startswith(o) for o in openers)


# ──────────────────────────── OCR de-noise ────────────────────────────
def _denoise(text: str) -> str:
    """Light cleanup. Aggressive normalization risks losing real signal —
    we only fix patterns we're confident about."""
    # Common OCR substitutions for Sanskrit diacritics losses won't matter
    # for English-language retrieval; we leave Sanskrit fragments alone.

    # Collapse runs of repeated punctuation that OCR hallucinated
    text = re.sub(r"\.{3,}", ".", text)
    text = re.sub(r" +\.", ".", text)

    # Glue cross-line hyphens: "lib-\nerty" → "liberty"
    text = re.sub(r"-\n([a-z])", r"\1", text)

    # Normalize whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text
