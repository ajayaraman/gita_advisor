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
#
# Model selection guide (set GEMINI_TASK_MODEL env var to override):
#   gemini/gemini-2.5-flash   — default; ~15-25s/call; good Advaita quality; free tier
#   gemini/gemma-4-26b-a4b-it — highest quality (~80s/call); use for offline GEPA runs
#   gemini/gemini-2.0-flash   — fastest (~5s/call); lower Advaita fidelity
#
# Timing benchmark (2-call merged pipeline, April 2026):
#   gemma-4-26b-a4b-it:  ~80s total (thinking tokens dominate)
#   gemini-2.5-flash:    ~20-25s total (estimated)
#   gpt-4o-mini:         ~20s total (lower quality — therapy clichés, no Sanskrit)
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

# ──────────────────────────── Task LM — HuggingFace Router ──────────────────────────────
# router.huggingface.co/v1 is OpenAI-compatible; use the "openai/" LiteLLM prefix
# with api_base pointing at HF's router endpoint.
# Set HF_MODEL env var to use a different model slug (must be deployed on HF).
HF_TOKEN = os.getenv("HF_TOKEN", "")
HF_ROUTER_BASE = os.getenv("HF_ROUTER_BASE", "https://router.huggingface.co/v1")
HF_MODEL = os.getenv("HF_MODEL", "google/gemma-4-26B-A4B-it")
HF_MODEL_STRING = f"openai/{HF_MODEL}"

HF_LM_KWARGS = dict(
    api_base=HF_ROUTER_BASE,
    api_key=HF_TOKEN,
    temperature=0.6,
    max_tokens=4096,
    cache=True,
)

# ──────────────────────────── Task LM — OpenRouter ───────────────────────────────────────
# LiteLLM recognises the "openrouter/" prefix natively and routes through
# https://openrouter.ai/api/v1.  Pick any model slug from openrouter.ai/models.
#
# Speed vs quality guide (set OPENROUTER_MODEL to override):
#   openrouter/google/gemini-2.0-flash-001       — fastest (~3-5s); good quality
#   openrouter/google/gemini-2.5-flash-preview   — balanced (~8-12s)
#   openrouter/anthropic/claude-3-5-haiku        — reliable structured output
#   openrouter/google/gemma-3-27b-it             — closest to the local Gemma 4 weights
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
_openrouter_model_raw = os.getenv("OPENROUTER_MODEL", "google/gemini-2.0-flash-001")
# LiteLLM requires the "openrouter/" prefix; add it if the env var omits it.
OPENROUTER_MODEL = (
    _openrouter_model_raw
    if _openrouter_model_raw.startswith("openrouter/")
    else f"openrouter/{_openrouter_model_raw}"
)

OPENROUTER_LM_KWARGS = dict(
    api_key=OPENROUTER_API_KEY,
    temperature=0.6,
    max_tokens=4096,
    cache=True,
)

# Which backend to use: "openrouter" if that key is set (and Gemini is not),
# "gemini" if GEMINI_API_KEY is set, else "lm_studio".
# Force a specific one with TASK_LM_BACKEND=openrouter|gemini|lm_studio.
def _default_task_lm_backend() -> str:
    if "TASK_LM_BACKEND" in os.environ:
        return os.environ["TASK_LM_BACKEND"]
    if GEMINI_API_KEY:
        return "gemini"
    if OPENROUTER_API_KEY:
        return "openrouter"
    if HF_TOKEN:
        return "hf"
    return "lm_studio"

TASK_LM_BACKEND: str = _default_task_lm_backend()


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
def configure_dspy(backend: str | None = None) -> tuple[dspy.LM, dspy.LM]:
    """Configure DSPy for inference and return (task_lm, reflection_lm).

    backend overrides TASK_LM_BACKEND when provided explicitly (used by chat.py
    --backend flag). Accepted values: "gemini", "openrouter", "lm_studio".

    ChatAdapter fallback to JSONAdapter is disabled in all paths because:
    - LM Studio rejects json_object.
    - Gemma outputs `[[ ## field ]]` (no closing ##); the field_header_pattern
      patch at module load time makes ChatAdapter parse these correctly.
    """
    effective_backend = backend or TASK_LM_BACKEND
    if effective_backend == "gemini":
        task_lm = dspy.LM(model=GEMINI_TASK_MODEL, **GEMINI_TASK_LM_KWARGS)
        print(f"Task LM backend: Gemini API ({GEMINI_TASK_MODEL})")
    elif effective_backend == "openrouter":
        if not OPENROUTER_API_KEY:
            raise SystemExit("OPENROUTER_API_KEY is not set. Add it to your .env file.")
        task_lm = dspy.LM(model=OPENROUTER_MODEL, **OPENROUTER_LM_KWARGS)
        print(f"Task LM backend: OpenRouter ({OPENROUTER_MODEL})")
    elif effective_backend == "hf":
        if not HF_TOKEN:
            raise SystemExit("HF_TOKEN is not set. Add it to your .env file.")
        task_lm = dspy.LM(model=HF_MODEL_STRING, **HF_LM_KWARGS)
        print(f"Task LM backend: HuggingFace Router ({HF_MODEL} @ {HF_ROUTER_BASE})")
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
def _default_embed_device() -> str:
    if "EMBED_DEVICE" in os.environ:
        return os.environ["EMBED_DEVICE"]
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"

EMBED_DEVICE = _default_embed_device()

TOP_K_RETRIEVE = 8       # passages to fetch per query
N_RETRIEVAL_QUERIES = 3  # the planner generates this many per user question
