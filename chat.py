"""
chat.py — interactive conversation with the advisor.

By default it loads the GEPA-optimized program from artifacts/. If that file
doesn't exist yet, it falls back to the un-optimized base prompts so you can
sanity-check the pipeline before running optimization.

Flags:
  --debug       Show intermediate pipeline state (felt emotion, queries, etc.)
  --thinking    Show the full synthesis reasoning trace (default: first 6 lines)
  --no-thinking Hide the reasoning trace entirely

After each response, source references are printed with numbers.
  show <N|ref>    Display the verse text, translation, and Śaṅkara's bhāṣya.
  explain <N|ref> Show the verse then stream a contextual explanation of how
                  it applies to the current conversation.
"""

from __future__ import annotations
import argparse
import time
import threading
from typing import Optional

import dspy
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text

import config
from advisor import load_optimized
from corpus import EnrichedVerse, Verse, read_jsonl_enriched, read_jsonl_verses


# ── speed constants ────────────────────────────────────────────────────────────
_THINKING_CPS = 800   # chars/sec for reasoning stream (secondary content, fast)
_RESPONSE_CPS = 300   # chars/sec for advisor response (primary content)
_THINKING_PREVIEW = 6 # lines shown in collapsed thinking mode


# ── verse corpus lookup ────────────────────────────────────────────────────────
def _load_verse_lookup() -> dict[str, Verse]:
    """Build a case-insensitive verse_ref → Verse dict from the corpus."""
    lookup: dict[str, Verse] = {}
    enriched = config.DATA_DIR / "corpus_enriched.jsonl"
    plain = config.DATA_DIR / "corpus.jsonl"

    if enriched.exists():
        loader, path = read_jsonl_enriched, enriched
    elif plain.exists():
        loader, path = read_jsonl_verses, plain
    else:
        return lookup

    for verse in loader(path):
        lookup[verse.verse_ref.lower().strip()] = verse
    return lookup


def _find_verse(lookup: dict, ref: str) -> Optional[Verse]:
    return lookup.get(ref.lower().strip())


def _resolve_ref(arg: str, sources_cited: list[str]) -> str:
    """Turn '1' → sources_cited[0], or return arg unchanged for direct ref lookup."""
    try:
        n = int(arg.strip())
        if 1 <= n <= len(sources_cited):
            return sources_cited[n - 1]
    except ValueError:
        pass
    return arg.strip()


# ── DSPy signature for contextual explanation ─────────────────────────────────
class _ExplainInContext(dspy.Signature):
    """You are the Gītā Advisor continuing a conversation. The user has asked
    you to unpack a specific verse or passage you cited. Explain what it means
    and why it speaks precisely to their situation — go deeper than the initial
    response did. Reference the user's words. Close with one concrete way to
    hold or work with this text this week."""

    verse_ref: str = dspy.InputField()
    verse_content: str = dspy.InputField(
        desc="Translation, original text (if available), and Śaṅkara's commentary."
    )
    conversation_context: str = dspy.InputField(
        desc="The user's question and the advisor's response where this verse was cited."
    )

    explanation: str = dspy.OutputField(
        desc="150-250 words. Grounded in Advaita. Do not merely restate the translation. "
             "End with a practical suggestion for this week."
    )


# ── streaming helpers ─────────────────────────────────────────────────────────
def _stream_chars(console: Console, text: str, cps: int):
    """Write text to the terminal character by character."""
    if not text:
        return
    delay = 1.0 / cps
    for ch in text:
        console.file.write(ch)
        console.file.flush()
        time.sleep(delay)
    console.file.write("\n")
    console.file.flush()


def _stream_response(console: Console, text: str, cps: int = _RESPONSE_CPS):
    """Stream the advisor response into a growing Markdown Panel via Rich Live."""
    if not text:
        return
    displayed = ""
    delay = 1.0 / cps
    with Live(console=console, refresh_per_second=min(cps, 30)) as live:
        for ch in text:
            displayed += ch
            live.update(Panel(
                Markdown(displayed),
                title="[bold]advisor[/bold]",
                border_style="yellow",
                padding=(1, 2),
            ))
            time.sleep(delay)


def _show_thinking(console: Console, reasoning: str, full: bool):
    """Stream the synthesis reasoning below a dim rule, collapsed to _THINKING_PREVIEW lines."""
    if not reasoning:
        return

    lines = reasoning.strip().splitlines()
    if not full and len(lines) > _THINKING_PREVIEW:
        display = "\n".join(lines[:_THINKING_PREVIEW])
        n_hidden = len(lines) - _THINKING_PREVIEW
    else:
        display = "\n".join(lines)
        n_hidden = 0

    console.print(Rule("[dim]thinking[/dim]", style="dim blue"))
    # Write dim italic via ANSI since we're streaming to file directly
    # (Rich markup can't be applied char-by-char; dim is cosmetic here)
    _stream_chars(console, display, cps=_THINKING_CPS)

    if n_hidden:
        console.print(f"[dim]  ↳ {n_hidden} more lines — use --thinking to expand[/dim]")
    console.print()


# ── verse display helpers ─────────────────────────────────────────────────────
def _show_verse(console: Console, verse: Verse):
    """Render a verse with its translation, original text, and commentary."""
    body = Text()

    if verse.sanskrit:
        body.append(verse.sanskrit + "\n", style="bold")
    if verse.transliteration:
        body.append(verse.transliteration + "\n", style="italic dim")

    if verse.translation:
        label = f"Translation ({verse.translator})" if verse.translator else "Translation"
        body.append(f"\n{label}:\n", style="dim")
        body.append(verse.translation + "\n")

    if verse.bhashya:
        translator_note = f" ({verse.bhashya_translator})" if verse.bhashya_translator else ""
        body.append(f"\nŚaṅkara's Bhāṣya{translator_note}:\n", style="dim")
        preview = verse.bhashya[:800] + ("…" if len(verse.bhashya) > 800 else "")
        body.append(preview + "\n", style="dim")

    ev = verse if isinstance(verse, EnrichedVerse) else None
    if ev and ev.paraphrase:
        body.append("\nTeaching: ", style="bold dim")
        body.append(ev.paraphrase + "\n", style="dim")
    if ev and ev.themes:
        body.append("Themes: ", style="bold dim")
        body.append(", ".join(ev.themes) + "\n", style="dim")
    if ev and ev.practical_teaching:
        body.append("Practical shift: ", style="bold dim")
        body.append(ev.practical_teaching + "\n", style="dim")

    section = verse.section_display or verse.section
    subtitle = verse.work_display + (f" — {section}" if section else "")
    console.print(Panel(
        body,
        title=f"[bold]{verse.verse_ref}[/bold]",
        subtitle=f"[dim]{subtitle}[/dim]",
        border_style="cyan",
        padding=(1, 2),
    ))


def _explain_in_context(
    console: Console,
    verse: Verse,
    history_messages: list[dict],
    cps: int = _RESPONSE_CPS,
):
    """Call the LM to explain the verse in context of the last conversation turn."""
    if history_messages:
        last = history_messages[-1]
        context = (
            f"User: {last.get('user_question', '')}\n\n"
            f"Advisor: {last.get('response', '')}"
        )
    else:
        context = "No prior conversation."

    bits = []
    if verse.translation:
        bits.append(f"Translation: {verse.translation}")
    if verse.sanskrit:
        bits.append(f"Sanskrit: {verse.sanskrit}")
    if verse.bhashya:
        bits.append(f"Śaṅkara's commentary: {verse.bhashya[:600]}")
    ev = verse if isinstance(verse, EnrichedVerse) else None
    if ev and ev.paraphrase:
        bits.append(f"Teaching: {ev.paraphrase}")
    verse_content = "\n\n".join(bits)

    explainer = dspy.ChainOfThought(_ExplainInContext)
    with console.status("[dim]expanding...[/dim]", spinner="dots"):
        try:
            result = explainer(
                verse_ref=verse.verse_ref,
                verse_content=verse_content,
                conversation_context=context,
            )
            explanation = result.explanation
        except Exception as exc:
            console.print(f"[red]Could not generate explanation: {exc}[/red]")
            return

    console.print()
    _stream_response(console, explanation, cps=cps)


# ── main loop ─────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--program", default=str(config.OPTIMIZED_PROGRAM_PATH))
    ap.add_argument("--debug", action="store_true",
                    help="Show intermediate pipeline state for each turn.")
    ap.add_argument("--thinking", action="store_true",
                    help="Show full synthesis reasoning trace (default: first 6 lines).")
    ap.add_argument("--no-thinking", action="store_true", dest="no_thinking",
                    help="Hide the reasoning trace entirely.")
    ap.add_argument("--backend", default=None,
                    choices=["gemini", "openrouter", "hf", "lm_studio"],
                    help="Override TASK_LM_BACKEND for this session.")
    args = ap.parse_args()

    config.configure_dspy(backend=args.backend)
    advisor = load_optimized(args.program)
    console = Console()

    with console.status("[dim]loading corpus...[/dim]", spinner="dots"):
        verse_lookup = _load_verse_lookup()

    console.print(Panel.fit(
        "[bold]Gītā Advisor[/bold]\n\n"
        "Speak from where you actually are.\n"
        "After a response: [italic]show <N>[/italic] to read a cited verse · "
        "[italic]explain <N>[/italic] for contextual breakdown.\n"
        "Type [italic]exit[/italic] or Ctrl-D to leave.",
        border_style="cyan",
    ))

    history = dspy.History(messages=[])
    last_pred = None

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

        # ── source exploration commands ───────────────────────────────────────
        cmd_lower = line.lower()
        if cmd_lower.startswith(("show ", "explain ")):
            if last_pred is None:
                console.print("[dim]No sources yet — ask a question first.[/dim]")
                continue
            cmd, _, arg = line.partition(" ")
            ref = _resolve_ref(arg, last_pred.sources_cited)
            verse = _find_verse(verse_lookup, ref)
            if verse is None:
                console.print(f"[dim]'{ref}' not found in corpus.[/dim]")
                if last_pred.sources_cited:
                    hint = "  ".join(
                        f"[{i+1}] {r}" for i, r in enumerate(last_pred.sources_cited)
                    )
                    console.print(f"[dim]Available: {hint}[/dim]")
                continue
            _show_verse(console, verse)
            if cmd.lower() == "explain":
                _explain_in_context(console, verse, history.messages)
            continue

        # ── normal question — run pipeline in background with live stage progress ──
        pred = None
        error = None
        stage = ["initializing..."]
        done = threading.Event()

        def run_advisor():
            nonlocal pred, error
            try:
                pred = advisor(
                    user_question=line,
                    history=history,
                    _stage_cb=lambda msg: stage.__setitem__(0, msg),
                )
            except Exception as exc:
                error = exc
            finally:
                done.set()

        threading.Thread(target=run_advisor, daemon=True).start()

        with Live(console=console, refresh_per_second=8) as live:
            while not done.wait(timeout=0.12):
                live.update(Text(f"  ◌  {stage[0]}", style="dim"))
            live.update(Text(""))

        if error:
            console.print(f"[red]Error: {error}[/red]")
            continue

        last_pred = pred
        history.messages.append({
            "user_question": line,
            "response": pred.response,
            "sources_cited": pred.sources_cited,
        })

        # debug trace
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

        # thinking section
        if not args.no_thinking:
            _show_thinking(
                console,
                getattr(pred, "synthesis_reasoning", ""),
                full=args.thinking,
            )

        # stream the response
        console.print()
        _stream_response(console, pred.response)

        # source footer with hints
        if pred.sources_cited:
            numbered = "  ".join(
                f"[{i+1}] {r}" for i, r in enumerate(pred.sources_cited)
            )
            console.print(f"\n[dim]sources: {numbered}[/dim]")
            console.print(
                "[dim]  → show <N> to read the verse  ·  explain <N> for contextual breakdown[/dim]"
            )


if __name__ == "__main__":
    main()
