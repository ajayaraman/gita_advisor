# HANDOFF.md — current state and next work

This document is the explicit handoff between the design session that built
this project and the local-development session that will continue it. It
catalogs what is done and tested, what is done but untested, and what is
still open. Claude Code: read this top-to-bottom before making changes.

## What is solid

The data pipeline up to retrieval is fully written and syntactically valid.
Every Python file in this project parses cleanly. The pipeline stages, in
order, are:

1. `download_sources.py` reads `sources_registry.py` and pulls every enabled
   source into `data/raw/<source_key>/`. The registry currently enables three
   sources by default — the gita/gita verse-indexed JSON (Sanskrit + word
   meanings), the gita/gita translations JSON (Swami Sivananda's English),
   and Sastry's 1897 archive.org OCR text of Śaṅkara's Gītā Bhāṣya. Five
   other sources are present in the registry but disabled, with notes in the
   entry explaining what's needed to turn them on.

2. `ingest_corpus.py` runs each enabled source's parser and merges Verse
   records by `verse_ref` so that the Gītā translation and Śaṅkara's bhāṣya
   end up on a single record per verse. The output is `data/corpus.jsonl`.

3. `enrich_corpus.py` runs the local LLM over each Verse to produce an
   EnrichedVerse with paraphrase, themes, life situations, emotions
   addressed, practical teaching, and hypothetical questions. Caching is
   append-only in `data/enrichment_cache.jsonl` so the (long) enrichment run
   is fully resumable. Output is `data/corpus_enriched.jsonl`.

4. `knowledge_base.py --build` reads the enriched corpus and writes three
   Chroma collections — one per view (literal, bhāṣya, advisor) — under
   `artifacts/chroma/`. The `AdvaitaRetriever` class queries all three at
   inference time and merges by verse_id with weighted scoring.

The new data model lives in `corpus.py`. The `Verse` and `EnrichedVerse`
dataclasses are the contract every downstream module reads from. JSONL I/O
is forward-compatible: adding a new field to the dataclass won't break old
files.

The enrichment prompt in `enrichment.py` is well-documented and uses a closed
emotion vocabulary (twenty entries) so retrieval can do faceted filtering
on emotions later. The prompt is wrapped in a `dspy.Module` so it could
itself be a target for GEPA optimization in a future loop.

## What is written but not yet exercised against real data

The pipeline above has been syntax-checked but never run end-to-end. Claude
Code's first useful job is to run the pipeline on the user's Mac and surface
whatever bugs only show up against real bytes. The likely friction points,
in order of probability:

The Sastry archive parser uses regex heuristics on noisy OCR. The verse-marker
patterns assume formats like "(II. 47.)" inline and chapter headings of a
certain shape. The OCR may have variants the regexes miss. The fix when this
happens is in `parsers/sastry_archive.py` — add patterns to `VERSE_INLINE`
and `CHAPTER_HEADING`. Spot-check on data/corpus.jsonl after running ingest
to catch bad splits.

The gita/gita translations.json schema can vary between forks of the upstream
repo. The parser tolerates a few key-name variations but might still need
adjustment if the mirror changes. If translations come through empty, look at
`parsers/gita_json.py` `_load_translations` — print the first row, see what
keys it actually has.

The enrichment validator (`_validate` in `enrichment.py`) requires at least
three hypothetical questions, two life situations, and a non-empty
paraphrase. A 26B local model will sometimes return four questions instead
of five and trip this. If the failure rate is high, loosen the threshold or
strengthen the prompt's "EXACTLY 5" instruction.

## What is still on the OLD interface and must be updated

These three files were written in turn 1 of the design session, before the
shift to verse-as-unit retrieval. They use a passages-as-strings shape that
the new retriever doesn't return. They must be updated to consume `Hit`
objects from `knowledge_base.AdvaitaRetriever`, but the changes are
mechanical, not architectural.

### 1. `signatures.py`

Open `signatures.py` and look at the `SelectPassages` and `SynthesizeAdvice`
signatures. Their `passages` field currently expects `list[str]`. It should
expect `list[str]` of *formatted hit blocks* — the synthesizer will continue
to receive strings, but they'll be the rich blocks produced by
`knowledge_base.format_hits_for_llm`. The signature docstrings need a small
update to mention that each formatted block contains a verse_ref the model
must cite verbatim.

The `SelectPassages` signature also needs to output `selected_verse_refs`
(list of citation strings) rather than `selected_indices`, so the metric can
do exact citation grounding without re-resolving indices.

### 2. `advisor.py`

In `advisor.py`, the `forward` method calls something like
`retriever.retrieve(query)` returning strings. Change it to call
`retriever.search_many(queries)` (returns `list[Hit]`), then pass
`format_hits_for_llm(hits)` into `SelectPassages`. After selection, look up
the selected verses by `verse_ref` from the Hits list and pass their
formatted blocks to `SynthesizeAdvice`. The advisor should keep the original
list of Hits in its return value as `retrieved_hits` so the metric can
verify citation grounding.

### 3. `metrics.py`

This is the highest-leverage change. The metric currently does fuzzy string
matching to verify that the advisor's citations came from retrieved
passages. Replace that with exact set-membership: extract verse_refs from
the synthesized response with a regex (e.g. `BG \d+\.\d+` and similar
patterns), then check that each cited ref is in `{h.verse.verse_ref for h
in retrieved_hits}`. Hallucinated citations become trivially detectable and
GEPA's reflection LM gets sharper feedback. The `tier_preference_score`
function can also become exact: it now reads the tier from `Hit.verse.tier`
rather than guessing from prefix matches.

## What's stubbed and waiting for an implementer

`parsers/wisdomlib_html.py` and `parsers/thibaut_sbe.py` don't exist yet.
The `ingest_corpus.py` orchestrator routes their source keys to a "skipping"
log line rather than crashing. If you want to bring in additional Upaniṣadic
material — Muṇḍaka with Śaṅkara from wisdomlib, the Brahma Sūtras with
Śaṅkara from Thibaut's archive.org SBE volumes — these are the parsers to
write. The `sources_registry.py` entries for these sources are present and
documented; just disabled.

A `parsers/plain_text.py` is also referenced but not implemented. Its job
is to pick up files the user drops into a `sources_local/` directory using
the convention `<tier>__<work>__<section>.txt`. This lets the user fold in
material from their own bookshelf or notes without touching the registry.

## Suggested first steps when you sit down with Claude Code

A reasonable opening sequence, before touching any code, is to verify the
project actually runs end-to-end on a small slice of the corpus. The
`enrich_corpus.py --limit 20` flag is there exactly for this — it enriches
the first twenty verses, takes about ten minutes on a 26B local model, and
produces enough data to validate the full pipeline including the index build
and a smoke-test query against the retriever. Doing this before the long
overnight enrichment run will save you from discovering bugs after thirty
hours of compute.

Once a slice works end-to-end, the three OLD-interface files above are the
right next target, in the order they're listed (signatures → advisor →
metrics), each followed by a smoke-test against the partial index.

After that, the long enrichment run is mostly waiting; while it runs you
can write the wisdomlib parser, regenerate the synthetic dataset against the
real verses (so the dataset's "correct citations" match what's actually in
the index), and review GEPA's reflection prompt.

## Things to leave alone unless you have measurement

The TIER_WEIGHTS and VIEW_WEIGHTS in `knowledge_base.py` were chosen by
reasoning rather than tuning. If you change them, do so with a held-out
evaluation set that measures retrieval quality before and after. Eyeballing
"this query now returns the verse I expected" is a trap — the change might
help that one query and silently hurt twenty others.

The closed EMOTION_VOCAB in `enrichment.py` also needs care. Adding entries
is fine and cheap. Removing entries orphans previously-enriched verses
whose emotions_addressed referenced the removed term, leaving them with
empty emotion lists. If you must remove, do a targeted re-enrichment of
affected verses afterward.

The verse_id format (`<work>_<section>_<verse>` zero-padded) is load-bearing.
Many things key off it. Don't refactor it without coordinating across
parsers, knowledge_base, and the metric.
