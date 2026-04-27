"""
run_overnight.py — orchestrates full GEPA optimization through light → medium,
then saves prompts and runs a multi-question test suite.

Usage:
    python run_overnight.py [--skip-light] [--skip-medium]

Writes a timestamped log to artifacts/overnight_run.log.
"""
from __future__ import annotations
import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
import json

ROOT = Path(__file__).parent.resolve()
LOG_PATH = ROOT / "artifacts" / "overnight_run.log"
OPTIMIZED_PATH = ROOT / "artifacts" / "optimized_advisor.json"
PROMPTS_PATH = ROOT / "artifacts" / "optimized_advisor.prompts.txt"
RESULTS_PATH = ROOT / "artifacts" / "test_results.json"

TEST_QUESTIONS = [
    "I just got laid off and feel like nothing matters anymore.",
    "I keep procrastinating on important work and feel guilty about it. How do I stop?",
    "My relationship ended and I feel like I've lost my identity. Who am I without this person?",
    "I'm terrified of death and can't stop thinking about it at night.",
    "I have achieved everything I wanted — career, family, money — and still feel empty.",
    "I feel angry at everyone around me but don't know why. How should I deal with this?",
    "I can't stop comparing myself to others and feeling like I'm always falling short.",
]


def ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str, f=None):
    line = f"[{ts()}] {msg}"
    print(line, flush=True)
    if f:
        f.write(line + "\n")
        f.flush()


def run_phase(cmd: list[str], phase: str, logfile) -> bool:
    log(f"=== STARTING {phase} ===", logfile)
    log(f"Command: {' '.join(cmd)}", logfile)
    start = time.time()
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(ROOT),
        )
        for line in proc.stdout:
            logfile.write(line)
            logfile.flush()
            # Echo key lines to terminal
            if any(k in line for k in ["score", "GEPA", "Step", "ERROR", "Saved", "Train:", "Val:", "Baseline"]):
                print(line, end="", flush=True)
        proc.wait()
        elapsed = time.time() - start
        if proc.returncode == 0:
            log(f"=== {phase} COMPLETED in {elapsed/60:.1f} min ===", logfile)
            return True
        else:
            log(f"=== {phase} FAILED (exit {proc.returncode}) after {elapsed/60:.1f} min ===", logfile)
            return False
    except Exception as e:
        log(f"=== {phase} ERROR: {e} ===", logfile)
        return False


def run_test_suite(logfile) -> dict:
    log("=== STARTING TEST SUITE ===", logfile)
    sys.path.insert(0, str(ROOT))

    import config
    from advisor import load_optimized
    from metrics import gita_metric
    import dspy
    from concurrent.futures import ThreadPoolExecutor, as_completed

    config.configure_dspy()

    advisor = load_optimized()
    n = len(TEST_QUESTIONS)

    def run_one(i_q):
        i, q = i_q
        try:
            pred = advisor(user_question=q, history=dspy.History(messages=[]))
            gold = dspy.Example(user_question=q).with_inputs("user_question")
            m = gita_metric(gold, pred)
            return i, q, {
                "question": q,
                "score": round(float(m.score), 3),
                "word_count": len(pred.response.split()),
                "sources_cited": pred.sources_cited,
                "response_excerpt": pred.response[:200],
                "feedback_excerpt": m.feedback[:500],
            }
        except Exception as e:
            return i, q, {"question": q, "error": str(e), "score": 0.0}

    indexed = list(enumerate(TEST_QUESTIONS, 1))
    results_map = {}
    with ThreadPoolExecutor(max_workers=n) as pool:
        futures = {pool.submit(run_one, iq): iq for iq in indexed}
        for fut in as_completed(futures):
            i, q, result = fut.result()
            results_map[i] = result
            if "error" in result:
                log(f"  [{i}/{n}] ERROR: {result['error']}", logfile)
            else:
                log(f"  [{i}/{n}] score={result['score']:.3f}  wc={result['word_count']}  sources={result['sources_cited']}", logfile)

    results = [results_map[i] for i in range(1, n + 1)]
    avg = sum(r.get("score", 0) for r in results) / n
    log(f"=== TEST SUITE DONE — avg score: {avg:.3f} ===", logfile)
    return {"questions": results, "avg_score": round(avg, 3), "timestamp": ts()}


def dump_prompts(logfile):
    """Re-extract and log optimized prompts to a human-readable file."""
    if not OPTIMIZED_PATH.exists():
        log("  No optimized program found — skipping prompt dump.", logfile)
        return

    sys.path.insert(0, str(ROOT))
    import config
    from advisor import GitaAdvisor
    config.configure_dspy()

    advisor = GitaAdvisor()
    try:
        advisor.load(str(OPTIMIZED_PATH))
    except Exception as e:
        log(f"  Could not load optimized program: {e}", logfile)
        return

    lines = ["# Optimized Prompts after GEPA overnight run", f"# Extracted at {ts()}", ""]
    for name, predictor in advisor.named_predictors():
        sig = predictor.signature
        lines.append(f"## {name}")
        lines.append(f"### Instructions")
        lines.append(sig.instructions or "(none)")
        lines.append("")
        lines.append("### Field descriptions")
        for fname, field in sig.fields.items():
            extras = field.json_schema_extra or {}
            desc = extras.get("desc", "") if isinstance(extras, dict) else ""
            lines.append(f"  {fname}: {desc}")
        lines.append("")
        lines.append("---")
        lines.append("")

    PROMPTS_PATH.write_text("\n".join(lines), encoding="utf-8")
    log(f"  Prompts written to {PROMPTS_PATH}", logfile)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-light", action="store_true")
    ap.add_argument("--skip-medium", action="store_true")
    ap.add_argument("--skip-tests", action="store_true")
    args = ap.parse_args()

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    with LOG_PATH.open("w", encoding="utf-8") as logfile:
        log("=== OVERNIGHT GEPA RUN STARTED ===", logfile)
        log(f"Dataset: {ROOT / 'data' / 'synthetic_questions.jsonl'}", logfile)
        log(f"Output:  {OPTIMIZED_PATH}", logfile)

        python = sys.executable

        # ── Phase 1: Light ──
        if not args.skip_light:
            ok = run_phase(
                [python, "optimize_gepa.py", "--auto", "light"],
                "GEPA LIGHT",
                logfile,
            )
            if not ok:
                log("Light phase failed — stopping overnight run.", logfile)
                sys.exit(1)
            # Back up light result
            if OPTIMIZED_PATH.exists():
                import shutil
                shutil.copy(OPTIMIZED_PATH, OPTIMIZED_PATH.with_suffix(".light.json"))
                log(f"  Backed up light result to {OPTIMIZED_PATH.with_suffix('.light.json')}", logfile)
        else:
            log("Skipping light phase (--skip-light).", logfile)

        # ── Phase 2: Medium ──
        if not args.skip_medium:
            ok = run_phase(
                [python, "optimize_gepa.py", "--auto", "medium"],
                "GEPA MEDIUM",
                logfile,
            )
            if not ok:
                log("Medium phase failed.", logfile)
                # Don't exit — still dump whatever we have
        else:
            log("Skipping medium phase (--skip-medium).", logfile)

        # ── Dump prompts ──
        log("Extracting optimized prompts ...", logfile)
        dump_prompts(logfile)

        # ── Test suite ──
        if not args.skip_tests:
            test_results = run_test_suite(logfile)
            RESULTS_PATH.write_text(json.dumps(test_results, indent=2, ensure_ascii=False), encoding="utf-8")
            log(f"Test results written to {RESULTS_PATH}", logfile)
        else:
            log("Skipping test suite (--skip-tests).", logfile)

        log("=== OVERNIGHT RUN COMPLETE ===", logfile)


if __name__ == "__main__":
    main()
