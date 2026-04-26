"""
smoke_test.py — verify the full pipeline before spending hours on GEPA.

Runs:
  1. LM connectivity check
  2. Retriever connectivity check
  3. One end-to-end advisor call
  4. One metric call against the result

If any step fails, the error message tells you which knob to turn.

    python smoke_test.py "I just got laid off and feel like nothing makes sense anymore."
"""

from __future__ import annotations
import sys
import json
import dspy

import config
from advisor import GitaAdvisor
from knowledge_base import AdvaitaRetriever
from metrics import gita_metric


def step(label: str):
    print(f"\n── {label} " + "─" * (60 - len(label)))


def main():
    user_q = sys.argv[1] if len(sys.argv) > 1 else (
        "I just got laid off and feel like nothing makes sense anymore."
    )

    step("1. Configure LMs")
    task_lm, reflection_lm = config.configure_dspy()
    print(f"  task_lm:       {task_lm.model}")
    print(f"  reflection_lm: {reflection_lm.model}")

    step("2. LM round-trip")
    try:
        out = task_lm("Reply with the single word: ready.")
        print(f"  reply: {out!r}")
    except Exception as e:
        print(f"  FAILED — is LM Studio running at {config.LM_STUDIO_BASE}?\n  {e}")
        sys.exit(1)

    step("3. Retriever sanity")
    try:
        retr = AdvaitaRetriever()
        hits = retr.search("non-attachment to results of action", k=3)
        if not hits:
            print("  WARNING: no retrieval results. Did you build the index?")
            print("  Run:   python knowledge_base.py --build")
        else:
            for h in hits:
                v = h.verse
                section = f" — {v.section}" if v.section else ""
                print(f"  [{v.tier}] {v.work}{section}  score={h.combined_score:.3f}")
    except Exception as e:
        print(f"  FAILED — index probably not built. Run "
              f"`python knowledge_base.py --build` after dropping texts in sources/.")
        print(f"  {e}")
        sys.exit(1)

    step("4. End-to-end advisor call")
    advisor = GitaAdvisor()
    try:
        pred = advisor(user_question=user_q, history=dspy.History(messages=[]))
    except Exception as e:
        print(f"  FAILED — pipeline error: {e}")
        sys.exit(1)

    print(f"\n  user: {user_q}")
    print(f"\n  felt:    {pred.felt_emotion}")
    print(f"  surface: {pred.surface_concern}")
    print(f"  deeper:  {pred.deeper_concern}")
    print(f"  themes:  {pred.vedantic_themes}")
    print(f"  queries: {pred.queries}")
    print(f"  selected indices: {pred.selected_indices}")
    print(f"\n  --- response ---")
    print(pred.response)
    print(f"\n  sources cited: {pred.sources_cited}")

    step("5. Metric round-trip")
    gold = dspy.Example(user_question=user_q, history=dspy.History(messages=[])).with_inputs("user_question", "history")
    m = gita_metric(gold, pred)
    print(f"  composite score: {m.score:.3f}")
    print(f"\n  --- feedback (this is what GEPA's reflection LM sees) ---")
    print(m.feedback)

    step("Done")
    print("If you got here, you're ready to run:")
    print("  python dataset_generator.py --n 500")
    print("  python optimize_gepa.py --auto medium")


if __name__ == "__main__":
    main()
