"""
parsers/wikisource_text.py — extract verse-keyed commentary from Swami
Vivekananda's "Thoughts on the Gita" (1897), sourced from Wikisource.

What this text is
-----------------
"Thoughts on the Gita" is a single lecture Vivekananda delivered in 1897,
recorded and published in The Complete Works of Swami Vivekananda, Vol. 4.
It is public domain (Vivekananda died 1902; Wikisource edition is CC BY-SA).
It is NOT a verse-by-verse commentary — it is a philosophical discourse that
quotes and discusses specific Gita verses as anchor points.

What we extract
---------------
We scan the lecture for explicit verse references (e.g. "2.20", "chapter II",
"shloka 47") and for each reference extract the surrounding paragraph(s) as
contextual commentary. This commentary is stored in the `bhashya` field with
`bhashya_translator = "Swami Vivekananda, 1897"`.

We do NOT invent verse_refs. If a reference can't be resolved to a known BG
verse (chapter 1–18, verse 1–80), the passage is silently skipped.

Why `bhashya` and not `translation`
-------------------------------------
Vivekananda's passages around each verse are interpretive commentary, not a
translation of the Sanskrit. Using `bhashya` preserves the meaning while
keeping `translation` available for the actual verse texts from gita_json.
The ingest merger picks "whichever bhashya isn't blank", so this fills the
bhashya field for any verse that isn't already covered by Sastry.

Coverage note
-------------
Vivekananda focuses heavily on Ch. 2 (esp. 2.19-2.20, 2.47) and touches a
handful of verses in other chapters. This is partial-coverage by design —
the remaining verses are fine without it.
"""

from __future__ import annotations
import re
from pathlib import Path
from typing import Iterable

from corpus import Verse


# ──────────────────────────── Verse reference patterns ────────────────────────

# Match patterns like: 2.47, II.47, chapter 2 verse 47, shloka 47 of chapter 2
_REF_PATTERNS = [
    # Numeric: "2.47" or "2, 47" or "ii.47"
    re.compile(r"\b(?P<chap>1[0-8]|[1-9])[\.,]\s*(?P<verse>[1-9]\d?)\b"),
    # "chapter II, verse 47" or "Ch. 2, v. 47"
    re.compile(
        r"(?:chapter|ch\.?)\s*(?P<chap>[IVX]+|\d+)[,\s]+(?:verse|v\.?|shloka)\s*(?P<verse>\d+)",
        re.IGNORECASE,
    ),
    # "verse 47 of chapter II"
    re.compile(
        r"(?:verse|shloka)\s*(?P<verse>\d+)\s+of\s+(?:chapter|ch\.?)\s*(?P<chap>[IVX]+|\d+)",
        re.IGNORECASE,
    ),
]

ROMAN_MAP = {
    "I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6, "VII": 7, "VIII": 8,
    "IX": 9, "X": 10, "XI": 11, "XII": 12, "XIII": 13, "XIV": 14, "XV": 15,
    "XVI": 16, "XVII": 17, "XVIII": 18,
}


def _to_int(token: str) -> int | None:
    t = token.strip().upper()
    if t in ROMAN_MAP:
        return ROMAN_MAP[t]
    if t.isdigit():
        return int(t)
    return None


# ──────────────────────────── Wikitext cleanup ────────────────────────────────

def _strip_wikitext(text: str) -> str:
    """Strip Wikisource wikitext markup, leaving plain prose."""
    # Remove templates: {{...}}
    text = re.sub(r"\{\{[^}]*\}\}", "", text)
    # Remove links: [[File:...]], [[link|display]] → display, [[link]]
    text = re.sub(r"\[\[(?:File|Image|Category):[^\]]*\]\]", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\[\[(?:[^\]|]*\|)?([^\]]*)\]\]", r"\1", text)
    # Remove external links: [http://... text] → text
    text = re.sub(r"\[https?://\S+\s+([^\]]+)\]", r"\1", text)
    text = re.sub(r"\[https?://\S+\]", "", text)
    # Remove bold/italic markup
    text = re.sub(r"'{2,3}", "", text)
    # Remove headers: == Heading ==
    text = re.sub(r"={2,6}[^=\n]+={2,6}", "", text)
    # Remove ref tags
    text = re.sub(r"<ref[^>]*>.*?</ref>", "", text, flags=re.DOTALL)
    text = re.sub(r"<ref[^/]*/?>", "", text)
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", "", text)
    # Collapse whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ──────────────────────────── Main parse ──────────────────────────────────────

def parse(raw_dir: Path) -> Iterable[Verse]:
    """Walk Vivekananda lecture text in raw_dir and yield partial Verse records.

    Expected layout (after download_sources.py):
        raw_dir/Thoughts_on_the_Gita   (no extension — Wikisource raw wikitext)
        OR raw_dir/*.txt / *.wiki
    """
    candidates = (
        list(raw_dir.glob("Thoughts_on_the_Gita*"))
        + list(raw_dir.glob("*.wiki"))
        + list(raw_dir.glob("*.txt"))
        + list(raw_dir.glob("index.php"))   # Wikisource downloads as index.php
        + list(raw_dir.glob("*"))           # any single file as last resort
    )
    # Filter to files only (no dirs)
    candidates = [c for c in candidates if c.is_file()]
    if not candidates:
        print(f"[wikisource] no text file under {raw_dir}; did you run download_sources.py?")
        return

    raw = candidates[0].read_text(encoding="utf-8", errors="replace")
    text = _strip_wikitext(raw)

    # Split into paragraphs
    paragraphs = [p.strip() for p in re.split(r"\n\n+", text) if p.strip()]

    # For each paragraph, find verse references and yield one Verse per unique ref
    seen: set[str] = set()
    for para in paragraphs:
        if len(para) < 40:
            continue
        refs = _extract_refs(para)
        for chap, verse in refs:
            ref = f"BG {chap}.{verse}"
            if ref in seen:
                continue
            seen.add(ref)
            commentary = _clean_para(para)
            if len(commentary) < 30:
                continue
            yield Verse(
                verse_id=f"bhagavad_gita_{chap:02d}_{verse:02d}",
                work="bhagavad_gita",
                work_display="Bhagavad Gītā",
                verse_ref=ref,
                tier="supporting",
                section=f"chapter_{chap:02d}",
                section_display=f"Chapter {chap}",
                translation="",
                translator="",
                bhashya=commentary,
                bhashya_translator="Swami Vivekananda, 1897",
                source_key="vivekananda_gita",
                license="public_domain",
            )


# ──────────────────────────── Helpers ─────────────────────────────────────────

def _extract_refs(text: str) -> list[tuple[int, int]]:
    """Return list of (chapter, verse) pairs found in the text."""
    found: list[tuple[int, int]] = []
    for pattern in _REF_PATTERNS:
        for m in pattern.finditer(text):
            chap = _to_int(m.group("chap"))
            verse = _to_int(m.group("verse"))
            if chap and verse and 1 <= chap <= 18 and 1 <= verse <= 80:
                found.append((chap, verse))
    return found


def _clean_para(para: str) -> str:
    """Light cleanup: remove leading/trailing citation markers and excess space."""
    para = re.sub(r"^\[[\d\w]+\]\s*", "", para)   # [1] footnote markers
    para = re.sub(r"\[\d+\]", "", para)            # inline footnote refs
    return para.strip()
