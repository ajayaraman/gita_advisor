"""
config.py — central configuration for the Gītā Advisor.

Three LMs are configured:

  - TASK_LM:       the local model running in LM Studio. Used at inference
                   time (understanding, retrieval planning, advice synthesis).

  - ENRICH_LM:     Claude Sonnet (API) for the offline enrichment pass.
                   The local 26B model truncates structured output at 1500
                   tokens and drops fields. Claude handles all six fields
                   cleanly in one call and costs ~$12-15 for the full 701-
                   verse corpus (one-time). Set ANTHROPIC_API_KEY in env.

  - REFLECTION_LM: Claude Opus 4.7 (API) for GEPA's reflection step.
                   GEPA asks the reflection LM to read metric feedback and
                   propose rewritten prompts — this scales strongly with
                   model quality. Opus 4.7 with extended thinking is the
                   right choice here. Same ANTHROPIC_API_KEY.
"""

from __future__ import annotations
import os
from pathlib import Path
import dspy
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")  # explicit path; works from any cwd

# ──────────────────────────── Paths ────────────────────────────
ROOT = Path(__file__).parent.resolve()
SOURCES_DIR = ROOT / "sources"
DATA_DIR = ROOT / "data"
ARTIFACTS_DIR = ROOT / "artifacts"
CHROMA_DIR = ARTIFACTS_DIR / "chroma"

for d in (SOURCES_DIR, DATA_DIR, ARTIFACTS_DIR, CHROMA_DIR):
    d.mkdir(parents=True, exist_ok=True)

DATASET_PATH = DATA_DIR / "synthetic_questions.jsonl"
OPTIMIZED_PROGRAM_PATH = ARTIFACTS_DIR / "optimized_advisor.json"

# ──────────────────────────── Task LM (local, inference-time) ────────────────────────────
LM_STUDIO_BASE = os.getenv("LM_STUDIO_BASE", "http://localhost:1234/v1")
LOCAL_MODEL = os.getenv("LOCAL_MODEL", "google/gemma-4-26b-a4b")

# DSPy uses LiteLLM-style model strings. "openai/" prefix routes through the
# OpenAI-compatible client, which LM Studio speaks.
TASK_MODEL_STRING = f"openai/{LOCAL_MODEL}"

TASK_LM_KWARGS = dict(
    api_base=LM_STUDIO_BASE,
    api_key=os.getenv("LM_STUDIO_KEY", "lm-studio"),  # any non-empty string
    temperature=0.6,
    max_tokens=2000,
    cache=True,
)


# ──────────────────────────── Enrichment LM (OpenAI gpt-4o-mini, offline batch) ─────────
# gpt-4o-mini is reliable at structured JSON output and cheap enough that the
# full 701-verse corpus costs under $1 (one-time).
#
# Cost estimate (full 701-verse corpus):
#   ~1800 input tokens/verse × 701 × $0.15/M ≈ $0.19 input
#   ~900  output tokens/verse × 701 × $0.60/M ≈ $0.38 output
#   Total ≈ $0.57 — effectively free at this scale.
#
# Key is read from .env (OPENAI_API_KEY). Override ENRICH_MODEL env var to
# swap in a different OpenAI model (e.g. "openai/gpt-4o" for harder cases).
ENRICH_MODEL = os.getenv("ENRICH_MODEL", "openai/gpt-4o-mini")

ENRICH_LM_KWARGS = dict(
    api_key=os.getenv("OPENAI_API_KEY", ""),
    temperature=0.3,   # lower than task LM — we want consistent structured output
    max_tokens=3000,   # enough headroom for all six fields + CoT reasoning
    cache=True,        # DSPy disk cache deduplicates identical calls on re-runs
    response_format={"type": "text"},  # DSPy 3.x sends json_object by default;
                                       # OpenAI now requires json_schema or text
)


# ──────────────────────────── Reflection LM (Claude Opus 4.7, GEPA) ─────────────────────
# GEPA's reflection step reads metric feedback and proposes rewritten prompts.
# This scales strongly with model quality — Opus 4.7 with extended thinking
# reasons through the failure patterns before writing the mutation, which
# produces meaningfully better prompt edits than a smaller model does.
#
# Extended thinking ("adaptive" mode) lets Opus decide how much reasoning to
# spend per reflection step. temperature=1.0 is required by the API when
# thinking is enabled — it matches GEPA's diversity requirement anyway.
#
# Same ANTHROPIC_API_KEY as the enrichment LM.
REFLECTION_MODEL = os.getenv("REFLECTION_MODEL", "anthropic/claude-opus-4-7")

REFLECTION_LM_KWARGS = dict(
    api_key=os.getenv("ANTHROPIC_API_KEY", ""),
    temperature=1.0,   # required for extended thinking; GEPA wants diversity here
    max_tokens=16000,  # reflection needs headroom for long critiques + prompt text
    thinking={"type": "enabled", "budget_tokens": 10000},  # adaptive extended thinking
    cache=False,       # reflection calls are intentionally diverse; caching defeats that
)


# ──────────────────────────── Configure helpers ───────────────────────────────────────
def configure_dspy() -> tuple[dspy.LM, dspy.LM]:
    """Configure DSPy for inference (task LM = local) and return (task_lm, reflection_lm).

    The reflection_lm returned here is Claude Opus 4.7 — pass it directly to
    GEPA's `reflection_lm` argument in optimize_gepa.py.
    """
    task_lm = dspy.LM(model=TASK_MODEL_STRING, **TASK_LM_KWARGS)
    reflection_lm = dspy.LM(model=REFLECTION_MODEL, **REFLECTION_LM_KWARGS)
    dspy.configure(lm=task_lm)
    return task_lm, reflection_lm


def configure_enrich_lm() -> dspy.LM:
    """Configure DSPy globally with the Claude Sonnet enrichment LM and return it.

    Call this instead of configure_dspy() when running enrich_corpus.py.
    Raises if ANTHROPIC_API_KEY is not set.
    """
    key = os.getenv("OPENAI_API_KEY", "")
    if not key:
        raise SystemExit(
            "OPENAI_API_KEY is not set. Add it to your .env file:\n"
            "  OPENAI_API_KEY=sk-proj-..."
        )
    lm = dspy.LM(model=ENRICH_MODEL, **ENRICH_LM_KWARGS)
    dspy.configure(lm=lm)
    return lm


# ──────────────────────────── Embeddings ─────────────────────────────────────────────
# Local sentence-transformer for retrieval. BGE-small is a sweet spot for
# semantic philosophy text on a Mac without burning RAM.
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
EMBED_DEVICE = os.getenv("EMBED_DEVICE", "mps")  # "mps" on Apple Silicon, "cpu" otherwise

TOP_K_RETRIEVE = 8       # passages to fetch per query
N_RETRIEVAL_QUERIES = 3  # the planner generates this many per user question
