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

  - REFLECTION_LM: gpt-4o (OpenAI) for GEPA's reflection step.
                   GEPA asks the reflection LM to read metric feedback and
                   propose rewritten prompts — this scales strongly with
                   model quality. gpt-4o reasons well enough to handle
                   nuanced Advaita feedback without breaking the budget.
                   Same OPENAI_API_KEY as enrichment.
"""

from __future__ import annotations
import os
import re
from pathlib import Path
import dspy
import dspy.adapters.chat_adapter as _chat_adapter_module
from dotenv import load_dotenv

# Gemma (and some other local models) output `[[ ## field ]]` without the closing `##`
# that DSPy's ChatAdapter expects (`[[ ## field ## ]]`). Patch the module-level regex
# to accept both forms before any adapter is instantiated.
_chat_adapter_module.field_header_pattern = re.compile(r"\[\[ ## (\w+)(?:\s*##)? \]\]")

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

# ──────────────────────────── Task LM — Gemini API (preferred) ───────────────────────────
# When GEMINI_API_KEY is set, route the task LM through Google AI Studio.
# Same Gemma 4 26B weights, but no local GPU required and the free tier is
# sufficient for inference + GEPA optimization runs.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_TASK_MODEL = os.getenv("GEMINI_TASK_MODEL", "gemini/gemma-4-26b-a4b-it")

GEMINI_TASK_LM_KWARGS = dict(
    api_key=GEMINI_API_KEY,
    temperature=0.6,
    # Gemma 4 thinking tokens count against max_tokens in the Gemini API.
    # Each pipeline call burns ~3-4k reasoning tokens before writing output,
    # so 4096 gets truncated. 16384 gives comfortable headroom for both.
    max_tokens=16384,
    cache=True,
)

# ──────────────────────────── Task LM — LM Studio fallback ───────────────────────────────
LM_STUDIO_BASE = os.getenv("LM_STUDIO_BASE", "http://localhost:1234/v1")
LOCAL_MODEL = os.getenv("LOCAL_MODEL", "google/gemma-4-26b-a4b")

# DSPy uses LiteLLM-style model strings. "openai/" prefix routes through the
# OpenAI-compatible client, which LM Studio speaks.
TASK_MODEL_STRING = f"openai/{LOCAL_MODEL}"

TASK_LM_KWARGS = dict(
    api_base=LM_STUDIO_BASE,
    api_key=os.getenv("LM_STUDIO_KEY", "lm-studio"),  # any non-empty string
    temperature=0.6,
    max_tokens=4096,  # ChainOfThought reasoning + all output fields easily exceeds 2k
    cache=True,
)

# Which backend to use: "gemini" if the API key is present, else "lm_studio".
# Override with TASK_LM_BACKEND=lm_studio to force local even when the key is set.
TASK_LM_BACKEND: str = os.getenv("TASK_LM_BACKEND", "gemini" if GEMINI_API_KEY else "lm_studio")


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


# ──────────────────────────── Proxy Task LM (gpt-4o-mini, GEPA optimization only) ────────
# When running GEPA with --proxy-task-lm, this model replaces Gemma 4 as the task LM
# during optimization. Prompts are model-agnostic text; they transfer back to Gemma 4
# at inference time. gpt-4o-mini runs ~20x faster than Gemma 4 thinking mode, bringing
# --auto light from ~260 hours to ~2-3 hours.
PROXY_TASK_MODEL = os.getenv("PROXY_TASK_MODEL", "openai/gpt-4o-mini")

PROXY_TASK_LM_KWARGS = dict(
    api_key=os.getenv("OPENAI_API_KEY", ""),
    temperature=0.6,
    max_tokens=4096,
    cache=True,
    response_format={"type": "text"},
)

# ──────────────────────────── Reflection LM (gpt-4o, GEPA) ──────────────────────────────
# GEPA's reflection step reads metric feedback and proposes rewritten prompts.
# This scales strongly with model quality. gpt-4o is the right balance here:
# it reasons well enough to write meaningful prompt mutations from nuanced
# Advaita feedback, and is affordable on a small OpenAI credit balance.
#
# Cost estimate per GEPA run (reflection calls only):
#   --auto light:  ~50 calls × 6k tokens ≈ $1.50
#   --auto medium: ~250 calls × 6k tokens ≈ $7.50
#
# gpt-4o-mini is too shallow for this task — it produces generic rewrites
# that ignore the tradition-specific feedback the metric provides.
# Same OPENAI_API_KEY as the enrichment LM.
REFLECTION_MODEL = os.getenv("REFLECTION_MODEL", "openai/gpt-4o")

REFLECTION_LM_KWARGS = dict(
    api_key=os.getenv("OPENAI_API_KEY", ""),
    temperature=1.0,   # GEPA wants diversity across reflection proposals
    max_tokens=6000,   # headroom for detailed critique + full rewritten prompt text
    response_format={"type": "text"},  # same fix as enrichment LM — avoid json_object
    cache=False,       # reflection calls are intentionally diverse; caching defeats that
)


# ──────────────────────────── Configure helpers ───────────────────────────────────────
def configure_dspy() -> tuple[dspy.LM, dspy.LM]:
    """Configure DSPy for inference and return (task_lm, reflection_lm).

    Prefers Gemini API when GEMINI_API_KEY is set (same Gemma 4 26B weights,
    hosted by Google, free tier). Falls back to LM Studio otherwise.
    Override with TASK_LM_BACKEND=lm_studio env var to force local.

    ChatAdapter fallback to JSONAdapter is disabled in both paths because:
    - LM Studio rejects json_object.
    - Gemma outputs `[[ ## field ]]` (no closing ##); the field_header_pattern
      patch at module load time makes ChatAdapter parse these correctly.
    """
    if TASK_LM_BACKEND == "gemini":
        task_lm = dspy.LM(model=GEMINI_TASK_MODEL, **GEMINI_TASK_LM_KWARGS)
        print(f"Task LM backend: Gemini API ({GEMINI_TASK_MODEL})")
    else:
        task_lm = dspy.LM(model=TASK_MODEL_STRING, **TASK_LM_KWARGS)
        print(f"Task LM backend: LM Studio ({TASK_MODEL_STRING} @ {LM_STUDIO_BASE})")

    reflection_lm = dspy.LM(model=REFLECTION_MODEL, **REFLECTION_LM_KWARGS)
    # use_json_adapter_fallback=False: LM Studio rejects json_object, so we must never fall back
    dspy.configure(lm=task_lm, adapter=dspy.ChatAdapter(use_json_adapter_fallback=False))
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
