# CLAUDE.md — Project Primer for the Gītā Advisor

This file is read by Claude Code when you open this project. It is also a
human-readable design memo. Read it once before asking Claude to do anything
substantial, and keep it updated as the design evolves — when the file lies,
Claude's behavior degrades.

## What this project is

A spiritual advisor grounded in Advaita Vedānta as taught by Śaṅkarācārya,
optimized via DSPy + GEPA against a local LM Studio model. The advisor takes
a real-life question or vent ("I just got laid off and feel like nothing
makes sense") and produces a response that is empathetic to the felt
experience, faithful to the non-dual lineage, and grounded in actual cited
verses from the Gītā, the principal Upaniṣads, the Brahma Sūtras, and
prakaraṇa-granthas. Wit is welcome, but only around the cosmic predicament,
never around the user's pain.

## The pipeline, in one breath

User text →
  `UnderstandQuery` (felt emotion + surface concern + deeper concern + themes) →
  `PlanRetrieval` (3 diverse search queries) →
  `AdvaitaRetriever.search_many` (multi-view RAG over verse-indexed corpus) →
  `SelectPassages` (pick the 2–4 verses that actually fit) →
  `SynthesizeAdvice` (compose the reply with citations) →
  `dspy.Prediction` carrying the response and its full trace for the metric.

Each predictor is a `dspy.ChainOfThought`, so GEPA has a `reasoning` trace to
inspect during reflection. The retriever is not optimized — vector search
isn't text — but the *queries given to it* are, which is what `PlanRetrieval`
exists to evolve.

## The two architectural choices that matter most

**Verse as the unit of retrieval.** Scripture is not arbitrary prose; the
natural unit is the verse (śloka, mantra, sūtra). The corpus is therefore
indexed by `verse_id` (e.g. `bhagavad_gita_02_47`), which has a stable
human-readable form (`BG 2.47`). Citations from the advisor are exact-match
verifiable against the retrieved set, which gives the metric a sharp signal
to feed back into GEPA's reflection step.

**Multi-view embeddings to bridge the language gap.** Users do not write in
the vocabulary of scripture — they say "I'm anxious about my career," not
"I'm experiencing rāga toward kāmya-karma." So we use the local LLM, in a
one-time offline pass, to enrich each verse with structured fields that
speak the user's language: a paraphrase, themes, life situations, emotions
addressed, practical teaching, and five hypothetical first-person questions.
Three separate embeddings per verse — `literal_view`, `bhashya_view`, and
`advisor_view` — let queries phrased in any register find the right verse.

The advisor view dominates retrieval (weight 0.55) because that is where the
language gap closes; the literal and bhāṣya views (0.25, 0.20) act as
insurance against the enrichment pipeline missing a topic.

## File map

```
gita_advisor/
├── config.py                  # paths, LM Studio URL, model strings, embed config
├── sources_registry.py        # central catalog of every open source we use
├── download_sources.py        # downloads everything to data/raw/<source_key>/
├── corpus.py                  # Verse / EnrichedVerse dataclasses + JSONL I/O
├── parsers/                   # one module per source format
│   ├── gita_json.py           #   ↳ gita/gita verse-indexed JSON (Unlicense)
│   └── sastry_archive.py      #   ↳ Sastry 1897 OCR text from archive.org (PD)
├── ingest_corpus.py           # runs parsers, merges by verse_ref → corpus.jsonl
├── enrichment.py              # DSPy module: Verse → EnrichedVerse via local LLM
├── enrich_corpus.py           # batch enrichment with caching; long-running
├── knowledge_base.py          # 3-view Chroma index; AdvaitaRetriever
├── signatures.py              # the four DSPy signatures GEPA optimizes
├── advisor.py                 # composed dspy.Module — what GEPA optimizes
├── metrics.py                 # rule-based + LLM-judge composite, with feedback
├── dataset_generator.py       # synthesizes ~500 life-situation questions
├── optimize_gepa.py           # runs GEPA over the advisor with the metric
├── chat.py                    # interactive CLI — load optimized advisor, chat
├── smoke_test.py              # 5-step pipeline check before committing time
├── data/
│   ├── raw/                   # pristine downloads, one folder per source key
│   ├── corpus.jsonl           # parsed Verses, merged across sources
│   ├── corpus_enriched.jsonl  # Verses + LLM-extracted fields
│   └── enrichment_cache.jsonl # append-only cache for resumable enrichment
└── artifacts/
    ├── chroma/                # the three view-collections
    └── optimized_advisor.json # GEPA's compiled program
```

## Source provenance, in one place

Every source must be unambiguously open. The four pillars currently enabled
or staged are described below in prose so the rationale doesn't get buried.

The `gita/gita` repository on GitHub provides the spine of the Gītā corpus.
It is a verse-indexed JSON dataset with Sanskrit, IAST transliteration, and
word-by-word glosses, released under the Unlicense (a public-domain
dedication). We pull it via a static-file mirror at
`ravisiyer.github.io/gita-data/v1/` so a single `requests.get` is enough;
cloning the whole repo also works.

Alladi Mahadeva Sastry's 1897 translation of Śaṅkara's Gītā Bhāṣya lives on
archive.org as full OCR text. It is the only complete English translation
of Śaṅkara's Gītā commentary that is unambiguously in the public domain
(Sastry died in 1926, the work itself dates to 1897). The OCR has
predictable noise — broken hyphens, occasional "rn" → "m" — and
`parsers/sastry_archive.py` is patient about it.

The wisdomlib site mirrors the *Sacred Books of the East* series and other
public-domain Indology — Telang's 1882 Gītā, Mundaka with Śaṅkara, etc.
The `wisdomlib_html` parser is registered but not yet implemented; this is
on the to-do list. `sacred-texts.com` carries the same content but blocks
some HTTP fetchers, so on the Mac you can use either.

What we deliberately do not include: Swami Gambhirananda's translations
(Advaita Ashrama copyright, mid-20th c.), modern Ramaṇa or Nisargadatta
editions, ISKCON's Prabhupada commentary. If you want any of these, place
your own copies in `sources_local/` under your own license judgment.

## The pipeline of commands

The first time, in order:

```bash
pip install -r requirements.txt

# 1. Download the registered open sources to data/raw/. Polite (1 req/s/host),
#    idempotent (skips files already present). Re-run with --force to refresh.
python download_sources.py

# 2. Parse the raw downloads into a unified verse corpus. Merges Gītā verse
#    text with Śaṅkara's bhāṣya by verse_ref. Outputs data/corpus.jsonl.
python ingest_corpus.py

# 3. Run the local LLM over every verse to extract paraphrase + life
#    situations + emotions + hypothetical questions. SLOW — several hours,
#    overnight is normal. Resumable via append-mode cache, so kill -9 is safe.
#    Outputs data/corpus_enriched.jsonl.
python enrich_corpus.py
# Smoke-test on 50 verses first if you want to verify the prompt is producing
# good output before committing the overnight run:
python enrich_corpus.py --limit 50

# 4. Build the three Chroma view-indices from the enriched corpus.
python knowledge_base.py --build

# 5. Sanity-check the pipeline on one user question.
python smoke_test.py "I just got laid off and feel like nothing matters anymore"

# 6. Generate the synthetic dataset of ~500 user questions for GEPA training.
python dataset_generator.py --n 500

# 7. Run GEPA optimization. Also long — start with --auto light to verify, then
#    re-run at --auto medium for the real pass.
python optimize_gepa.py --auto medium

# 8. Open the chat CLI with the optimized program loaded.
python chat.py
```

After the first run, only steps 4–8 normally re-run. Steps 1–3 are one-time
unless you change sources or the enrichment prompt.

## Two things to watch out for

**LM Studio model name.** The exact string `google/gemma-4-26b-a4b` (or
whatever you settle on) goes in `config.py` as `LOCAL_MODEL`, and DSPy
prefixes it with `openai/` to route through the OpenAI client. If LM Studio
reports a different model identifier in its API, copy-paste verbatim.

**Failed enrichments.** The local model occasionally produces malformed
structured output. The enricher retries twice and, on persistent failure,
stamps `enrichment_model = "FAILED: <reason>"` on the verse. The verse is
still indexed on its literal text and bhāṣya, just without the advisor view.
After the full pass, run `python enrich_corpus.py --only-failed` to retry
just those, perhaps after tuning the prompt in `enrichment.py`.

## What is not yet done

The Sastry parser produces verse-attached bhāṣya but the verse-text /
bhāṣya split is heuristic; spot-check a few verses (BG 2.47, BG 18.66 are
good canaries) and tighten `_build_verse` if needed.

The `wisdomlib_html` parser and the `thibaut_sbe` (Brahma Sūtra) parser are
registered but stubbed — adding either is a single-file change. They are
disabled in `sources_registry.py` until written.

The metric still has the rule-based hooks for therapy clichés, length, and
non-dual register but does not yet look at the new EnrichedVerse fields
(`emotions_addressed`, `themes`) for empathy verification. There is a clear
win there: when the user's `felt_emotion` appears in a selected verse's
`emotions_addressed` list, that is strong evidence of empathic-fit retrieval.

The dataset generator was written before the schema shift; spot-check that
its output still flows through the pipeline cleanly.

## How to talk to me when working in this project

The most useful prompts are concrete and bounded. "Tighten the verse-bhāṣya
split heuristic in `parsers/sastry_archive.py` and run it on the first three
chapters; show me the BG 2.47 record" is a good prompt. "Improve the
parser" is not.

When something is broken, read the relevant file end-to-end before patching.
The comments in this project are unusually heavy because the design has many
small choices that stop being obvious six months from now. If a comment
disagrees with the code, the comment is more likely to be right and you
should ask whether the code drifted, not whether the comment did.

When designing a new piece, start by asking what `Verse` / `EnrichedVerse`
field carries the information, before reaching for new state. The data
model is meant to be the contract between modules; adding ad-hoc fields on
the side is how RAG systems become spaghetti.

## Pinned design commitments (do not silently break these)

The advisor is grounded in Advaita Vedānta as Śaṅkara taught it. We do not
import dualistic theology, and we do not reduce Advaita to "we are all one"
pop-spirituality. We hold the two-truths distinction (vyāvahārika and
pāramārthika) actively, and we do not collapse the user's lived suffering
into "it's all māyā anyway." When a teaching has a Sanskrit name with a
precise meaning, we use the Sanskrit name with a brief gloss rather than
substituting an approximate English word.

Citations are exact and verifiable. "BG 2.47" in a response means the verse
was in the retrieved set. The metric enforces this; do not weaken it.

The advisor is not therapy and is not a chatbot friend. It is a teacher in
the tradition of the lineage. It is allowed to push back, to challenge a
question's premise, and to recommend silence over more words.

The retriever is permissive; the selector is picky. Do not move filtering
upstream into the retriever — once a verse is filtered out at retrieval,
no later stage can recover it.
