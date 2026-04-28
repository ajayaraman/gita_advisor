"""
advisor.py — the composed DSPy module.

This is what GEPA optimizes. It chains four predictors:

    UnderstandQuery  →  PlanRetrieval  →  [retrieve]  →  SelectPassages  →  SynthesizeAdvice

Each predictor uses ChainOfThought so GEPA has a `reasoning` field to inspect
in its reflection step. The retriever itself is not optimized (it's vector
search), but the *queries given to it* are — that's where PlanRetrieval lives.
"""

from __future__ import annotations
import time
from dataclasses import dataclass
import dspy

from signatures import (
    UnderstandQuery,
    PlanRetrieval,
    SelectPassages,
    SynthesizeAdvice,
)
from knowledge_base import AdvaitaRetriever, format_passages_for_llm
import config


@dataclass
class AdviceTrace:
    """Everything the pipeline produced — useful for the metric to reason over."""
    user_question: str
    felt_emotion: str
    surface_concern: str
    deeper_concern: str
    vedantic_themes: list[str]
    queries: list[str]
    retrieved_passages: list[dict]
    selected_indices: list[int]
    selection_rationale: str
    response: str
    sources_cited: list[str]


class GitaAdvisor(dspy.Module):
    def __init__(self, retriever: AdvaitaRetriever | None = None):
        super().__init__()
        self.understand = dspy.ChainOfThought(UnderstandQuery)
        self.plan = dspy.ChainOfThought(PlanRetrieval)
        self.select = dspy.ChainOfThought(SelectPassages)
        self.synthesize = dspy.ChainOfThought(SynthesizeAdvice)
        self._retriever = retriever or AdvaitaRetriever()

    def forward(
        self,
        user_question: str,
        history: dspy.History | None = None,
        _stage_cb=None,
    ) -> dspy.Prediction:
        if history is None:
            history = dspy.History(messages=[])

        t0 = time.perf_counter()

        # 1. Understand
        if _stage_cb:
            _stage_cb("understanding your question…")
        u = self.understand(history=history, user_question=user_question)

        # 2. Plan retrieval queries
        if _stage_cb:
            _stage_cb("planning search queries…")
        p = self.plan(
            surface_concern=u.surface_concern,
            deeper_concern=u.deeper_concern,
            vedantic_themes=u.vedantic_themes,
        )
        queries = p.queries[: config.N_RETRIEVAL_QUERIES] if p.queries else [u.deeper_concern]

        # 3. Retrieve
        if _stage_cb:
            _stage_cb("searching scriptures…")
        hits = self._retriever.search_many(queries, k_per=config.TOP_K_RETRIEVE)
        candidates = hits[: max(8, config.TOP_K_RETRIEVE)]
        candidates_text = format_passages_for_llm(candidates)
        candidates_as_dicts = [h.to_dict() for h in candidates]

        previously_cited = [
            src
            for msg in history.messages
            for src in msg.get("sources_cited", [])
        ]

        # 4. Select
        if _stage_cb:
            _stage_cb("selecting passages…")
        s = self.select(
            deeper_concern=u.deeper_concern,
            candidate_passages=candidates_text,
            previously_cited=previously_cited,
        )
        valid_idx = [
            i for i in (s.selected_indices or [])
            if isinstance(i, int) and 1 <= i <= len(candidates)
        ]
        if not valid_idx:
            valid_idx = list(range(1, min(4, len(candidates) + 1)))

        selected = [candidates[i - 1] for i in valid_idx]
        selected_text = format_passages_for_llm(selected)

        # 5. Synthesize
        if _stage_cb:
            _stage_cb("composing response…")
        a = self.synthesize(
            history=history,
            user_question=user_question,
            felt_emotion=u.felt_emotion,
            deeper_concern=u.deeper_concern,
            selected_passages=selected_text,
        )

        t1 = time.perf_counter()
        print(f"[timing] total={t1-t0:.1f}s")

        return dspy.Prediction(
            response=a.response,
            sources_cited=a.sources_cited or [],
            synthesis_reasoning=getattr(a, "reasoning", ""),
            felt_emotion=u.felt_emotion,
            surface_concern=u.surface_concern,
            deeper_concern=u.deeper_concern,
            vedantic_themes=u.vedantic_themes,
            queries=queries,
            retrieved_passages=candidates_as_dicts,
            selected_indices=valid_idx,
            selection_rationale=s.selection_rationale,
        )


def load_optimized(path: str | None = None) -> GitaAdvisor:
    """Load an advisor with GEPA-optimized prompts if available, else fresh."""
    advisor = GitaAdvisor()
    p = path or str(config.OPTIMIZED_PROGRAM_PATH)
    try:
        advisor.load(p)
        print(f"Loaded optimized advisor from {p}")
    except FileNotFoundError:
        print(f"No optimized program at {p} — using base prompts.")
    except Exception as exc:
        print(f"Could not load optimized program ({exc}) — using base prompts.")
    return advisor
