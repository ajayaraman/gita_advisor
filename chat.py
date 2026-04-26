"""
chat.py — interactive conversation with the advisor.

By default it loads the GEPA-optimized program from artifacts/. If that file
doesn't exist yet, it falls back to the un-optimized base prompts so you can
sanity-check the pipeline before running optimization.

Use --debug to print the intermediate state (felt emotion, retrieved sources,
selection rationale) — useful when iterating on the metric.
"""

from __future__ import annotations
import argparse
from pathlib import Path

import dspy
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.rule import Rule

import config
from advisor import load_optimized


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--program", default=str(config.OPTIMIZED_PROGRAM_PATH))
    ap.add_argument("--debug", action="store_true",
                    help="Show intermediate pipeline state for each turn.")
    args = ap.parse_args()

    config.configure_dspy()
    advisor = load_optimized(args.program)
    console = Console()

    console.print(Panel.fit(
        "[bold]Gītā Advisor[/bold]\n\n"
        "Speak from where you actually are. Type [italic]exit[/italic] or Ctrl-D to leave.",
        border_style="cyan",
    ))

    history = dspy.History(messages=[])

    while True:
        try:
            console.print()
            console.print("[bold cyan]you:[/bold cyan] ", end="")
            line = input().strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]नमस्ते।[/dim]")
            return

        if not line:
            continue
        if line.lower() in {"exit", "quit", ":q"}:
            console.print("[dim]नमस्ते।[/dim]")
            return

        try:
            with console.status("[dim]reflecting ...[/dim]", spinner="dots"):
                pred = advisor(user_question=line, history=history)
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            continue

        # Append this exchange to history so the next turn has context
        history.messages.append({"user_question": line, "response": pred.response})

        if args.debug:
            console.print(Rule("[dim]debug[/dim]", style="dim"))
            console.print(f"[dim]felt:[/dim]    {pred.felt_emotion}")
            console.print(f"[dim]surface:[/dim] {pred.surface_concern}")
            console.print(f"[dim]deeper:[/dim]  {pred.deeper_concern}")
            console.print(f"[dim]themes:[/dim]  {', '.join(pred.vedantic_themes)}")
            console.print(f"[dim]queries:[/dim] {pred.queries}")
            console.print(f"[dim]selected:[/dim] {pred.selected_indices}")
            for i in pred.selected_indices:
                if 1 <= i <= len(pred.retrieved_passages):
                    h = pred.retrieved_passages[i - 1]
                    m = h["meta"]
                    console.print(
                        f"  [dim]→ [{m['tier']}] {m['work']}"
                        f"{' — ' + m['section'] if m.get('section') else ''}"
                        f"  (score {h['score']:.3f})[/dim]"
                    )
            console.print(Rule(style="dim"))

        console.print()
        console.print(Panel(
            Markdown(pred.response),
            title="[bold]advisor[/bold]",
            border_style="yellow",
            padding=(1, 2),
        ))
        if pred.sources_cited:
            console.print(f"[dim]sources: {', '.join(pred.sources_cited)}[/dim]")


if __name__ == "__main__":
    main()
