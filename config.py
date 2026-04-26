"""
config.py — central configuration for the Gītā Advisor.

Two LMs are configured:
  - TASK_LM:       the local model running in LM Studio. Used for the actual
                   pipeline (understanding, retrieval planning, advice synthesis).
  - REFLECTION_LM: optionally a stronger model for GEPA's reflection step.
                   Defaults to the same local model so everything stays offline,
                   but if you have API access to a larger model, point it there
                   — GEPA's quality scales meaningfully with reflection-LM quality.
"""

from __future__ import annotations
import os
from pathlib import Path
import dspy

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

# ──────────────────────────── LM ────────────────────────────
LM_STUDIO_BASE = os.getenv("LM_STUDIO_BASE", "http://localhost:1234/v1")
LOCAL_MODEL = os.getenv("LOCAL_MODEL", "google/gemma-4-26b-a4b")

# DSPy uses a LiteLLM-style model string. For an OpenAI-compatible local server
# we prefix with "openai/" so it routes through the OpenAI client.
TASK_MODEL_STRING = f"openai/{LOCAL_MODEL}"

# Generation params: spiritual-advice prose benefits from a touch of warmth,
# not zero-temperature recitation. Capped reasonably for non-rambling answers.
TASK_LM_KWARGS = dict(
    api_base=LM_STUDIO_BASE,
    api_key=os.getenv("LM_STUDIO_KEY", "lm-studio"),  # any non-empty string works
    model_type="chat",
    temperature=0.6,
    max_tokens=1500,
    cache=True,
)

REFLECTION_LM_KWARGS = dict(
    api_base=LM_STUDIO_BASE,
    api_key=os.getenv("LM_STUDIO_KEY", "lm-studio"),
    model_type="chat",
    # Reflection wants to be more deterministic and have headroom to write
    # detailed critiques; bump tokens up.
    temperature=1.0,  # GEPA recommends temp=1 for reflection diversity
    max_tokens=6000,
    cache=False,
)


def configure_dspy() -> tuple[dspy.LM, dspy.LM]:
    """Configure DSPy globally and return (task_lm, reflection_lm)."""
    task_lm = dspy.LM(model=TASK_MODEL_STRING, **TASK_LM_KWARGS)
    reflection_lm = dspy.LM(model=TASK_MODEL_STRING, **REFLECTION_LM_KWARGS)
    dspy.configure(lm=task_lm)
    return task_lm, reflection_lm


# ──────────────────────────── Embeddings ────────────────────────────
# Local sentence-transformer for retrieval. BGE-small is a sweet spot for
# semantic philosophy text on a Mac without burning RAM.
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
EMBED_DEVICE = os.getenv("EMBED_DEVICE", "mps")  # "mps" on Apple Silicon, "cpu" otherwise

# Chunking for the corpus
CHUNK_TOKENS = 380       # roughly one bhāṣya paragraph
CHUNK_OVERLAP = 60
TOP_K_RETRIEVE = 8       # passages to fetch per query
N_RETRIEVAL_QUERIES = 3  # the planner generates this many per user question
