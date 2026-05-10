"""
parsers/swarupananda_archive.py — extract verse translations from Swami
Swarupananda's 1909 Srimad Bhagavad Gita (Advaita Ashrama, Calcutta).

Why this source
---------------
Swami Swarupananda was a direct disciple of Sri Ramakrishna Paramahamsa and a
founding member of the Ramakrishna Math. His 1909 translation was published by
Advaita Ashrama — the same institution that publishes the Complete Works of
Swami Vivekananda. It is the closest verse-indexed text in the Ramakrishna
lineage that is unambiguously in the public domain (published 1909, well before
the 1929 US cutoff). The archive.org OCR of this edition is the source we parse.

OCR structure (1909 edition)
-----------------------------
Unlike Sastry's OCR which uses Roman numerals, Swarupananda's original uses
clear chapter headings and arabic verse numbers. The typical OCR layout:

    CHAPTER I.
    THE DEPRESSION OF ARJUNA.

    1. Dhritarashtra said: ...

    2. Sanjaya said: ...

    CHAPTER II.
    ...

The verse number is an arabic digit at the start of a line, followed by a
period or space, then the verse text (possibly over multiple lines). There is
no Sankara bhashya here — just the translation — so we extract only the
verse text and store it in `translation`.

What we produce
---------------
For each verse we find, we yield a Verse with:
    - tier = 'supporting'       (a translation, not Sankara's own bhashya)
    - work = 'bhagavad_gita'    (so ingest_corpus merges it with gita_json verses
                                 by verse_ref, filling the translator field when
                                 gita_json leaves it empty, or adding Swarupananda
                                 as a second voice depending on priority)
    - translation = Swarupananda's English
    - translator  = 'Swami Swarupananda'
    - bhashya     = ''          (no commentary in this edition)
"""

from __future__ import annotations
import re
from pathlib import Path
from typing import Iterable

from corpus import Verse


# ──────────────────────────── Chapter / verse patterns ────────────────────────

# "CHAPTER I", "CHAPTER II", "CHAPTER 2", etc.
CHAPTER_HEADING = re.compile(
    r"^\s*CHAPTER\s+(?P<id>[IVXLC\d]+)[\s\.\-—]*",
    re.IGNORECASE | re.MULTILINE,
)

ROMAN_MAP = {
    "I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6, "VII": 7, "VIII": 8,
    "IX": 9, "X": 10, "XI": 11, "XII": 12, "XIII": 13, "XIV": 14, "XV": 15,
    "XVI": 16, "XVII": 17, "XVIII": 18,
}

# Arabic verse number at start of a line: "1. ", "47. ", "1 " — sometimes the
# period is lost to OCR. We also accept "Verse 47." as a fallback.
VERSE_NUM = re.compile(
    r"^\s*(?:Verse\s+)?(?P<verse>\d{1,3})[\.\s]\s*(?=[A-Z\"\'\(])",
    re.MULTILINE,
)

# Lines that are likely OCR noise or page headers (all caps, short, no real words)
NOISE_LINE = re.compile(r"^\s*[A-Z\s\.\-—]{1,40}\s*$", re.MULTILINE)


# ──────────────────────────── Main parse ──────────────────────────────────────

def parse(raw_dir: Path) -> Iterable[Verse]:
    """Walk Swarupananda 1909 OCR text in raw_dir and yield Verse records.

    Expected layout (after download_sources.py):
        raw_dir/...swarupananda..._djvu.txt   (or any .txt)
    """
    txts = list(raw_dir.glob("*_djvu.txt")) + list(raw_dir.glob("*.txt"))
    if not txts:
        print(f"[swarupananda] no .txt under {raw_dir}; did you run download_sources.py?")
        return

    text = txts[0].read_text(encoding="utf-8", errors="replace")
    text = _denoise(text)

    # Collect (event_type, position, data) for chapters and verse markers
    events: list[tuple] = []

    for m in CHAPTER_HEADING.finditer(text):
        chap = _to_arabic(m.group("id"))
        if chap is not None:
            events.append(("chapter", m.start(), chap))

    for m in VERSE_NUM.finditer(text):
        v = int(m.group("verse"))
        if 1 <= v <= 80:
            events.append(("verse", m.start(), v, m.end()))

    events.sort(key=lambda e: e[1])

    # Walk events: track chapter, accumulate text between verse markers
    current_chapter = 1
    last_verse: int | None = None
    last_verse_end: int | None = None

    for ev in events:
        if ev[0] == "chapter":
            # Flush pending verse before switching chapter
            if last_verse is not None and last_verse_end is not None:
                body = text[last_verse_end: ev[1]].strip()
                v = _build_verse(current_chapter, last_verse, body)
                if v is not None:
                    yield v
                last_verse = None
                last_verse_end = None
            current_chapter = ev[2]

        else:  # "verse"
            _, start, verse_num, end = ev
            if last_verse is not None and last_verse_end is not None:
                body = text[last_verse_end:start].strip()
                v = _build_verse(current_chapter, last_verse, body)
                if v is not None:
                    yield v
            last_verse = verse_num
            last_verse_end = end

    # Flush trailing verse
    if last_verse is not None and last_verse_end is not None:
        body = text[last_verse_end:].strip()
        v = _build_verse(current_chapter, last_verse, body)
        if v is not None:
            yield v


# ──────────────────────────── Builders ────────────────────────────────────────

_CHAPTER_NAMES = {
    1: "Chapter 1: The Depression of Arjuna",
    2: "Chapter 2: Sankhya Yoga",
    3: "Chapter 3: Karma Yoga",
    4: "Chapter 4: Jnana Yoga",
    5: "Chapter 5: Karma Sannyasa Yoga",
    6: "Chapter 6: Dhyana Yoga",
    7: "Chapter 7: Jnana Vijnana Yoga",
    8: "Chapter 8: Akshara Brahma Yoga",
    9: "Chapter 9: Raja Vidya Yoga",
    10: "Chapter 10: Vibhuti Yoga",
    11: "Chapter 11: Vishwarupa Darshana Yoga",
    12: "Chapter 12: Bhakti Yoga",
    13: "Chapter 13: Kshetra Kshetrajna Vibhaga Yoga",
    14: "Chapter 14: Gunatraya Vibhaga Yoga",
    15: "Chapter 15: Purushottama Yoga",
    16: "Chapter 16: Daivasura Sampad Vibhaga Yoga",
    17: "Chapter 17: Shraddhatraya Vibhaga Yoga",
    18: "Chapter 18: Moksha Sannyasa Yoga",
}


def _build_verse(chap: int, verse: int, body: str) -> Verse | None:
    """Clean body text and return a Verse. Returns None if body is too short."""
    body = _clean_body(body)
    if len(body) < 20:
        return None

    return Verse(
        verse_id=f"bhagavad_gita_{chap:02d}_{verse:02d}",
        work="bhagavad_gita",
        work_display="Bhagavad Gītā",
        verse_ref=f"BG {chap}.{verse}",
        tier="supporting",
        section=f"chapter_{chap:02d}",
        section_display=_CHAPTER_NAMES.get(chap, f"Chapter {chap}"),
        translation=body,
        translator="Swami Swarupananda",
        bhashya="",
        bhashya_translator="",
        source_key="swarupananda_gita",
        license="public_domain",
    )


def _clean_body(body: str) -> str:
    """Strip transliteration lines and all-caps OCR noise from verse body.

    Transliteration lines in the 1909 edition appear as runs of lowercase +
    diacritics with occasional | (verse boundary marker). We detect them by
    the pattern: mostly lowercase, contains common IAST diacritics or |.
    """
    lines = body.split("\n")
    kept = []
    for line in lines:
        s = line.strip()
        if not s:
            kept.append("")
            continue
        # Skip likely transliteration lines (many lowercase diacritics / bars)
        if re.search(r"[āīūṛṝḷṃḥśṣṭḍṇ|ṁ]", s):
            continue
        # Skip very short all-caps OCR noise lines
        if len(s) < 25 and s == s.upper() and not re.search(r"\d", s):
            continue
        kept.append(line)

    result = "\n".join(kept).strip()
    # Collapse multiple blank lines
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result


# ──────────────────────────── Helpers ─────────────────────────────────────────

def _to_arabic(token: str) -> int | None:
    t = token.strip().upper()
    if t in ROMAN_MAP:
        return ROMAN_MAP[t]
    if t.isdigit():
        n = int(t)
        if 1 <= n <= 18:
            return n
    return None


def _denoise(text: str) -> str:
    text = re.sub(r"\.{3,}", ".", text)
    text = re.sub(r" +\.", ".", text)
    text = re.sub(r"-\n([a-z])", r"\1", text)   # cross-line hyphens
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text
