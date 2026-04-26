"""
optimize_gepa.py — run GEPA reflective prompt evolution.

GEPA (Genetic-Pareto) treats the program's prompts as an evolving population.
At each step it:
  1. Runs the current candidate(s) on a minibatch of training examples
  2. Collects the (score, feedback) pairs from our metric
  3. Asks a *reflection LM* to read the failures + feedback and propose a
     mutated prompt
  4. Evaluates the mutant; keeps it if it Pareto-dominates the parent on the
     validation set
  5. Repeats

Because we wrote `gita_metric` to return rich textual feedback, the reflection
LM has something substantive to chew on instead of just gradient signal.

The dataset has no gold labels — that's deliberate. Our metric judges the
prediction directly. This is the regime GEPA is designed for.

Usage:
    python optimize_gepa.py --auto medium
    python optimize_gepa.py --max-metric-calls 800
    python optimize_gepa.py --auto light --val-frac 0.2
"""

from __future__ import annotations
import argparse
import json
import random
from pathlib import Path

import dspy
from dspy import GEPA

import config
from advisor import GitaAdvisor
from dataset_generator import load_jsonl, to_dspy_examples
import metrics as metrics_module
from metrics import gita_metric, quick_eval_score


def split(examples, val_frac: float, seed: int = 42):
    rng = random.Random(seed)
    shuffled = examples[:]
    rng.shuffle(shuffled)
    n_val = max(20, int(len(shuffled) * val_frac))
    return shuffled[n_val:], shuffled[:n_val]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(config.DATASET_PATH))
    ap.add_argument("--out", default=str(config.OPTIMIZED_PROGRAM_PATH))
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument(
        "--auto",
        choices=["light", "medium", "heavy"],
        default="medium",
        help="GEPA's auto-budget mode. 'light' for smoke-tests, 'medium' for "
             "a real run, 'heavy' for an overnight run on a meaty box.",
    )
    ap.add_argument(
        "--max-metric-calls",
        type=int,
        default=None,
        help="Override --auto with an explicit metric-call budget.",
    )
    ap.add_argument("--track-stats", action="store_true", default=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # Configure DSPy globally and grab the reflection LM
    task_lm, reflection_lm = config.configure_dspy()
    print(f"Task LM:       {task_lm.model}  ({config.LM_STUDIO_BASE})")
    print(f"Reflection LM: {reflection_lm.model}")

    # Use the reflection LM (gpt-4o) for judging instead of the task LM (Gemma).
    # Gemma judging its own responses produces noisy, self-congratulatory scores;
    # gpt-4o gives the reflection step the crisp, tradition-aware feedback it needs.
    metrics_module.configure_judge(reflection_lm)
    print(f"Judge LM:      {reflection_lm.model} (overriding task LM for judging)")

    # Dataset
    raw = load_jsonl(Path(args.dataset))
    examples = to_dspy_examples(raw)
    if len(examples) < 40:
        print(f"[warn] Only {len(examples)} examples — generate more with "
              f"`python dataset_generator.py --n 500`.")
    train, val = split(examples, args.val_frac, seed=args.seed)
    print(f"Train: {len(train)}   Val: {len(val)}")

    # Student program
    student = GitaAdvisor()

    # Optional: get a baseline number for context
    print("\nEvaluating baseline (un-optimized) on validation set ...")
    evaluator = dspy.Evaluate(
        devset=val,
        metric=quick_eval_score,
        num_threads=4,
        display_progress=True,
        display_table=0,
    )
    try:
        baseline_score = evaluator(student)
        print(f"Baseline score: {baseline_score:.3f}")
    except Exception as e:
        print(f"Baseline eval failed (continuing to optimization): {e}")

    # GEPA
    log_dir = str(config.ARTIFACTS_DIR / "gepa_logs")
    gepa_kwargs = dict(
        metric=gita_metric,
        reflection_lm=reflection_lm,
        track_stats=args.track_stats,
        seed=args.seed,
        # Show 6 training examples to the reflection LM per proposal step instead of
        # the default 3 — our 12 domains need diversity to avoid domain-specific over-fit.
        reflection_minibatch_size=6,
        # Parallel metric calls — LM Studio handles concurrent requests; without this
        # --auto medium can take 6+ hours on a single thread.
        num_threads=4,
        # When Gemma mangles a list field the reflection LM should know the format broke,
        # not just see a low score with no explanation.
        add_format_failure_as_feedback=True,
        # Persist per-step scores and prompts for post-run inspection.
        log_dir=log_dir,
    )
    if args.max_metric_calls is not None:
        gepa_kwargs["max_metric_calls"] = args.max_metric_calls
    else:
        gepa_kwargs["auto"] = args.auto

    print(f"\nStarting GEPA with {gepa_kwargs} ...")
    optimizer = GEPA(**gepa_kwargs)

    optimized = optimizer.compile(
        student=student,
        trainset=train,
        valset=val,
    )

    # Save
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    optimized.save(str(out_path))
    print(f"\nSaved optimized program to {out_path}")

    # Side-by-side eval
    print("\nFinal eval on validation set ...")
    final_score = evaluator(optimized)
    print(f"Optimized score: {final_score:.3f}")

    # Dump the optimized prompts for human inspection
    inspect_path = out_path.with_suffix(".prompts.txt")
    with inspect_path.open("w", encoding="utf-8") as f:
        f.write("# Optimized prompts after GEPA\n\n")
        for name, predictor in optimized.named_predictors():
            sig = predictor.signature
            f.write(f"## {name}\n")
            f.write(f"### instructions\n{sig.instructions}\n\n")
            f.write("### fields\n")
            for fname, field in sig.fields.items():
                desc = getattr(field.json_schema_extra, "get", lambda *_: "")("desc", "") \
                    if hasattr(field, "json_schema_extra") else ""
                f.write(f"- {fname}: {desc}\n")
            f.write("\n---\n\n")
    print(f"Wrote prompt inspection file to {inspect_path}")


if __name__ == "__main__":
    main()
