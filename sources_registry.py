"""
sources_registry.py — the one place every open source lives.

Why a registry rather than scattered URLs?
------------------------------------------
Adding a new text to the corpus shouldn't mean editing five files. It should
mean adding one entry here. Downloads, parsing, re-indexing, and enrichment
all read from this registry, so the registry *is* the corpus definition.

How sources are categorized
---------------------------
Every source belongs to a "tier", which the retriever uses to break ties when
two passages score equally on cosine similarity:

    primary    — the śruti and the Gītā itself (the thing being commented on)
    shankara   — Śaṅkarācārya's bhāṣyas and prakaraṇa-granthas (his own pen)
    supporting — texts in his lineage but not by him (Aṣṭāvakra, Yoga Vāsiṣṭha,
                 Vidyāraṇya's Pañcadaśī, modern Ramaṇa & Nisargadatta where
                 explicitly placed in the Advaita stream)

The tier weights live in knowledge_base.py; this file just labels.

License classes
---------------
We track licensing because the project is meant to be shareable. We refuse to
register any source that is not unambiguously open. The classes are:

    public_domain — pre-1929 works in US PD; covers most 19th-c. translations
    unlicense     — Unlicense / CC0 / equivalent dedications
    cc_by         — Creative Commons Attribution (must preserve credit)
    cc_by_sa      — Creative Commons ShareAlike
    open_database — ODbL (the dataset license used by some github corpora)

Anything we'd label "publisher_copyright" simply doesn't get an entry. If you
want the modern Advaita Ashrama translations, you must obtain a license and
add the texts yourself in the user-supplied directory.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal


# ──────────────────────────── Type aliases ────────────────────────────
Tier = Literal["primary", "shankara", "supporting"]
License = Literal[
    "public_domain", "unlicense", "cc_by", "cc_by_sa", "open_database",
]
Parser = Literal[
    "gita_json",         # the gita/gita repo JSON layout (verse-indexed)
    "wisdomlib_html",    # one chapter per HTML page on wisdomlib
    "sastry_archive",    # Alladi Mahadeva Sastry OCR text from archive.org
    "thibaut_sbe",       # Thibaut's SBE Brahma Sutra translation HTML
    "plain_text",        # already-cleaned plain text the user dropped in
]


# ──────────────────────────── Source entry ────────────────────────────
@dataclass(frozen=True)
class Source:
    """One downloadable source. The registry is a list of these.

    The download_sources script understands two kinds of `urls`:
      - HTTPS URLs to direct files (json, html, txt) — fetched with `requests`
      - "git+https://..." URLs — cloned with `git clone --depth=1`

    The parser receives the local path(s) and is responsible for emitting
    Verse records into the corpus.
    """
    # Identity
    key: str                       # short slug used as folder name; must be unique
    name: str                      # human-readable name
    work: str                      # the work; matches Verse.work for grouping
    tier: Tier

    # Provenance
    license: License
    license_url: str = ""          # canonical license URL or attribution page
    translator: str = ""           # who did the English translation
    year: int | None = None        # year of the edition we're using

    # Download
    urls: tuple[str, ...] = ()     # one or more files / git repos
    parser: Parser = "plain_text"

    # Operational
    enabled: bool = True           # set False to skip without deleting the entry
    notes: str = ""                # anything a future reader should know


# ──────────────────────────── The registry ────────────────────────────
#
# This list is the source of truth. Everything else reads it.
#
# Conventions:
#   - One entry per *publication*, not per chapter file. The parser knows how
#     to walk its own files.
#   - URLs that work as of the writing of this comment are noted. If a URL
#     drifts, fix it here and re-run `download_sources.py`.
#   - When in doubt about license, leave the source disabled and add a note.
#
SOURCES: list[Source] = [

    # ─── Bhagavad Gītā: Sanskrit + transliteration + word meanings ───
    # The gita/gita repo gives us the cleanest verse-indexed data on the web.
    # Released under the Unlicense, which is a public-domain dedication. We
    # use the static GitHub Pages mirror because it's directly fetchable as
    # JSON files; cloning the repo is also fine.
    Source(
        key="gita_json_core",
        name="Bhagavad Gītā — verse-indexed JSON (core)",
        work="bhagavad_gita",
        tier="primary",
        license="unlicense",
        license_url="https://github.com/gita/gita/blob/main/LICENSE",
        translator="Sanskrit + IAST transliteration + word-by-word gloss",
        year=None,
        urls=(
            "https://ravisiyer.github.io/gita-data/v1/chapters.json",
            "https://ravisiyer.github.io/gita-data/v1/verse.json",
        ),
        parser="gita_json",
        notes=(
            "This is the spine of the Gītā corpus. Sanskrit + transliteration + "
            "word-meanings. Translations come from translator-specific files."
        ),
    ),

    # ─── Bhagavad Gītā: English translations (one or more) ───
    # The translations.json file is large (~2 MB) and contains multiple
    # translators keyed by author_id. Our parser will pick public-domain ones.
    Source(
        key="gita_json_translations",
        name="Bhagavad Gītā — English translations (multiple authors)",
        work="bhagavad_gita",
        tier="primary",
        license="unlicense",
        license_url="https://github.com/gita/gita/blob/main/LICENSE",
        translator="multiple — see per-verse author_id",
        year=None,
        urls=(
            "https://ravisiyer.github.io/gita-data/v1/translation.json",
            "https://ravisiyer.github.io/gita-data/v1/authors.json",
        ),
        parser="gita_json",
        notes=(
            "Parser keeps only translators whose works are public-domain or "
            "explicitly free; e.g. Swami Sivananda is OK, ISKCON Prabhupada "
            "is excluded. See parsers/gita_json.py for the allowlist."
        ),
    ),

    # ─── Śaṅkara's Gītā Bhāṣya, Sastry 1897 translation ───
    # The only full English translation of Śaṅkara's Gītā commentary that's
    # unambiguously in the public domain (Sastry died ~1926; first published
    # 1897). Lives on archive.org as OCR text. Parser handles OCR noise.
    Source(
        key="sastry_gita_bhashya",
        name="Śaṅkara's Bhagavad Gītā Bhāṣya — Sastry translation (1897)",
        work="bhagavad_gita_bhashya",
        tier="shankara",
        license="public_domain",
        license_url="https://archive.org/details/Bhagavad-Gita.with.the.Commentary.of.Sri.Shankaracharya",
        translator="Alladi Mahadeva Sastry",
        year=1897,
        urls=(
            # Direct OCR text. The /download/ path is reliably the raw file;
            # /stream/ is the HTML viewer and not what we want.
            "https://archive.org/download/Bhagavad-Gita.with.the.Commentary.of.Sri.Shankaracharya/Bhagavad-Gita.with.the.Commentary.of.Sri.Shankaracharya_djvu.txt",
        ),
        parser="sastry_archive",
        notes=(
            "OCR will have noise — broken hyphens, occasional 'rn' → 'm'. "
            "Parser uses verse-marker regex to chunk by verse and tries to "
            "associate Śaṅkara's commentary with the verse it follows."
        ),
    ),

    # ─── Telang's Gītā translation, SBE Vol. 8 (1882) ───
    # An alternative to Sastry for the Gītā translation itself. Useful when
    # we want a second voice for the verse text, since Sastry was sometimes
    # paraphrasing Śaṅkara's gloss into the translation.
    Source(
        key="telang_gita",
        name="Bhagavad Gītā — Telang translation, SBE Vol. 8 (1882)",
        work="bhagavad_gita",
        tier="primary",
        license="public_domain",
        license_url="https://en.wikipedia.org/wiki/Sacred_Books_of_the_East",
        translator="Kāshināth Trimbak Telang",
        year=1882,
        urls=tuple(
            f"https://www.wisdomlib.org/hinduism/book/the-bhagavadgita/d/doc{n}.html"
            for n in range(81668, 81686)  # chapters 1–18
        ),
        parser="wisdomlib_html",
        enabled=False,  # off by default — gita_json_translations gives us enough
        notes=(
            "Wisdomlib mirrors Telang's SBE 8 translation as one chapter per "
            "page. Enable if you want a second translation alongside gita_json."
        ),
    ),

    # ─── Mundaka Upaniṣad with Śaṅkara's Bhāṣya ───
    # Wisdomlib hosts a complete English edition of Mundaka with Śaṅkara's
    # commentary. Likely older Sitarama Sastri translation, public domain.
    Source(
        key="mundaka_shankara",
        name="Muṇḍaka Upaniṣad with Śaṅkara's Bhāṣya",
        work="mundaka_upanishad",
        tier="shankara",
        license="public_domain",
        license_url="https://www.wisdomlib.org/hinduism/book/mundaka-upanishad-shankara-bhashya",
        translator="Sitarama Sastri (1898)",
        year=1898,
        urls=(
            "https://www.wisdomlib.org/hinduism/book/mundaka-upanishad-shankara-bhashya",
        ),
        parser="wisdomlib_html",
        notes=(
            "The wisdomlib parser will follow the table-of-contents links from "
            "this index page to fetch each section."
        ),
    ),

    # ─── Brahma Sūtras with Śaṅkara's Bhāṣya, Thibaut translation ───
    # SBE volumes 34 (1890) and 38 (1896). The most-cited English translation
    # of the Brahma Sūtra Bhāṣya, used by every academic working in Vedānta.
    # Squarely public domain.
    Source(
        key="thibaut_brahma_sutra",
        name="Brahma Sūtras with Śaṅkara Bhāṣya — Thibaut translation",
        work="brahma_sutra_bhashya",
        tier="shankara",
        license="public_domain",
        license_url="https://archive.org/details/SacredBooksOfTheEastVol34",
        translator="George Thibaut (SBE 34 & 38)",
        year=1890,
        urls=(
            # archive.org full-text URLs for SBE 34 and 38
            "https://archive.org/download/SacredBooksOfTheEastVol34/sbe34_djvu.txt",
            "https://archive.org/download/SacredBooksOfTheEastVol38/sbe38_djvu.txt",
        ),
        parser="thibaut_sbe",
        enabled=False,  # parser not implemented in v1 — see parsers/README.md
        notes=(
            "Disabled by default until thibaut_sbe parser is written. The text "
            "is structured by adhikaraṇa (topic groups of sūtras), not by "
            "single sūtras, so the parser needs more care than the others."
        ),
    ),

    # ─── Vivekacūḍāmaṇi (Mohini Chatterji translation) ───
    # The most famous prakaraṇa attributed to Śaṅkara. 581 verses.
    # Mohini Chatterji's translation is early-20th-c., public domain.
    Source(
        key="vivekachudamani_chatterji",
        name="Vivekacūḍāmaṇi — Mohini Chatterji translation",
        work="vivekachudamani",
        tier="shankara",
        license="public_domain",
        translator="Mohini M. Chatterji",
        year=1932,
        urls=(
            # The user should fill this in; placeholder for the registry shape
            # so the downloader logs a clear "URL missing" message rather than
            # silently skipping.
            "",
        ),
        parser="plain_text",
        enabled=False,
        notes=(
            "Drop a clean copy at sources_local/vivekachudamani.txt and the "
            "plain_text parser will pick it up. Several archive.org editions "
            "exist; verse markers vary by edition."
        ),
    ),

    # ─── User-provided plain-text drop-in slot ───
    # If you already have a clean text file (a translation you typed up, a
    # lecture transcript you cleaned, anything), drop it in sources_local/
    # named tier__work__section.txt and the plain_text parser will fold it in.
    Source(
        key="user_local",
        name="User-provided plain-text sources",
        work="user_local",
        tier="supporting",
        license="public_domain",  # under your responsibility
        urls=(),
        parser="plain_text",
        notes=(
            "Anything in sources_local/. Convention: tier__work__section.txt "
            "(see parsers/plain_text.py)."
        ),
    ),
]


# ──────────────────────────── Helpers ────────────────────────────
def by_key(key: str) -> Source:
    """Look up a source by its registry key. Raises KeyError on miss."""
    for s in SOURCES:
        if s.key == key:
            return s
    raise KeyError(f"No source with key={key!r}")


def enabled_sources() -> list[Source]:
    return [s for s in SOURCES if s.enabled]


def by_parser(parser: Parser) -> list[Source]:
    """Group enabled sources by their parser, useful for the ingest loop."""
    return [s for s in SOURCES if s.enabled and s.parser == parser]


def attribution_for(work: str) -> list[str]:
    """Returns the attribution lines for any work, for citation footers.

    Even though all our sources are PD, citing translators is right. The
    advisor's response footer can call this and append the translators to
    the bibliography lines.
    """
    out = []
    for s in SOURCES:
        if s.work == work and s.translator:
            year = f", {s.year}" if s.year else ""
            out.append(f"{s.translator}{year} ({s.license})")
    return out
