# Gītā Advisor

A spiritual advisor grounded in Advaita Vedānta as taught by Śaṅkarācārya,
optimized via DSPy + GEPA against a local LM Studio model. The advisor takes
real-life questions or vents and produces responses that are empathetic to
the felt experience, faithful to the non-dual lineage, and grounded in
exact-cited verses from the Gītā with Śaṅkara's bhāṣya, the principal
Upaniṣads, the Brahma Sūtras, and the prakaraṇa-granthas.

## What makes this design unusual

The first unusual choice is that the unit of retrieval is the verse, not
the chunk. Scripture is not arbitrary prose: each Gītā śloka, each
Upaniṣadic mantra, each sūtra is a sealed teaching unit with a stable
citation reference. We index by `verse_id` (e.g. `bhagavad_gita_02_47`,
which renders as `BG 2.47` in citations) so the advisor's references can be
exact-match-verified against the retrieved set.

The second unusual choice is that we use the local LLM, in a one-time
offline pass, to enrich each verse with structured fields a real person's
question can match against. A user does not write "I am experiencing
rāga toward kāmya-karma"; they write "I worked on this for three years and
it just failed." So we ask the local model, for each verse, to produce a
plain-English paraphrase, the Vedāntic themes engaged, the life situations
addressed, the emotions met, the practical teaching offered, and five
hypothetical first-person questions the verse would speak to. We then
embed three views of each verse — the literal translation, Śaṅkara's
bhāṣya, and the LLM-enriched advisor view — and at retrieval time query
all three and merge by verse ID.

The advisor view dominates retrieval because that is where the language
gap closes. The literal and bhāṣya views act as insurance against the
enrichment pipeline missing a topic.

## Where the texts come from

Every source is unambiguously open. The verse-indexed JSON at
`github.com/gita/gita`, released under the Unlicense, gives us Sanskrit
plus IAST transliteration plus word-by-word glosses for the Gītā. Alladi
Mahadeva Sastry's 1897 translation of Śaṅkara's Gītā Bhāṣya, in the public
domain and full-text on archive.org, gives us Śaṅkara's commentary
attached to each verse. The wisdomlib mirror of the *Sacred Books of the
East* is staged for the Upaniṣad-with-Śaṅkara texts and the Brahma Sūtra
bhāṣya; those parsers are registered but not yet implemented. See
`sources_registry.py` for the complete catalog and `CLAUDE.md` for the
licensing rationale.

We deliberately exclude the modern Advaita Ashrama translations (active
copyright), modern Ramaṇa and Nisargadatta editions, and Prabhupada's
commentary. If you have your own license-cleared copies, drop them in
`sources_local/` and the `plain_text` parser will fold them in.

## Pipeline of commands

```bash
pip install -r requirements.txt

# 1. Download the registered open sources to data/raw/<source_key>/
python download_sources.py

# 2. Parse + merge into data/corpus.jsonl (one verse per line)
python ingest_corpus.py

# 3. Enrich every verse via the local LLM. SLOW — overnight.
#    Resumable; kill -9 is safe (append-mode cache).
python enrich_corpus.py --limit 50    # smoke-test the prompt first
python enrich_corpus.py               # then the real run

# 4. Build the three-view Chroma index
python knowledge_base.py --build

# 5. Try a query against the index
python knowledge_base.py --query "I just got laid off and feel hollow"

# 6. Smoke-test the full advisor pipeline
python smoke_test.py "I just got laid off and feel hollow"

# 7. Generate the synthetic question dataset and run GEPA
python dataset_generator.py --n 500
python optimize_gepa.py --auto medium

# 8. Open the chat CLI
python chat.py
```

## Project structure

The project is laid out so the data flow is left-to-right through the
pipeline: each script reads what the previous one wrote, with all
intermediate state on disk so any stage can be re-run independently. The
data model lives in `corpus.py` (`Verse` and `EnrichedVerse` dataclasses)
and is the contract between modules. The advisor itself is a `dspy.Module`
that GEPA optimizes; the metric in `metrics.py` is the specification GEPA
optimizes against, combining rule-based hygiene checks with an LLM-judge
rubric and producing structured feedback for GEPA's reflection step. See
`CLAUDE.md` for the full file map and the design commitments that should
not be silently broken.

## Configuration

`config.py` reads a small number of environment variables. The two that
matter most are `LM_STUDIO_BASE` (defaults to `http://localhost:1234/v1`)
and `LOCAL_MODEL` (defaults to `google/gemma-4-26b-a4b`, but copy whatever
LM Studio reports verbatim). The embedding model defaults to BGE-small on
Apple Silicon's MPS device; switch `EMBED_DEVICE` to `cpu` if you are not
on Apple Silicon.

## License

The code in this repository is yours to use. The texts in `data/raw/` come
with their own licenses, all unambiguously open and tracked in
`sources_registry.py`. Attributions for translators are preserved through
the pipeline and surfaced in citation footers.
# gita_advisor
