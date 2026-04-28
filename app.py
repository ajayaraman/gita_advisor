"""
app.py — Enhanced Gradio web interface for the Gītā Advisor.

Features:
  - Real-time stage progress during inference (◌ understanding → searching → composing)
  - Character-by-character response streaming
  - Verse explorer: select any cited source to read Sanskrit, translation, Śaṅkara's bhāṣya
  - Warm spiritual aesthetic
"""

from __future__ import annotations
import threading
import time

import gradio as gr
import dspy

import config
from advisor import load_optimized
from knowledge_base import AdvaitaRetriever, format_passages_for_llm
from corpus import EnrichedVerse, Verse, read_jsonl_enriched, read_jsonl_verses


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

# ── startup — runs once when the Space boots ──────────────────────────────────
config.configure_dspy(backend="hf")
_advisor = load_optimized()
_retriever = AdvaitaRetriever()
_retriever._ensure()


def _load_verse_lookup() -> dict[str, Verse]:
    lookup: dict[str, Verse] = {}
    enriched = config.DATA_DIR / "corpus_enriched.jsonl"
    plain = config.DATA_DIR / "corpus.jsonl"
    if enriched.exists():
        for v in read_jsonl_enriched(enriched):
            lookup[v.verse_ref.lower().strip()] = v
    elif plain.exists():
        for v in read_jsonl_verses(plain):
            lookup[v.verse_ref.lower().strip()] = v
    return lookup


_verse_lookup = _load_verse_lookup()


# ── helpers ────────────────────────────────────────────────────────────────────

def _to_dspy_history(gradio_history: list) -> dspy.History:
    """Convert Gradio messages list to dspy.History, stripping source footers."""
    msgs = []
    i = 0
    while i + 1 < len(gradio_history):
        u, a = gradio_history[i], gradio_history[i + 1]
        if u.get("role") == "user" and a.get("role") == "assistant":
            content = a["content"]
            if "\n\n---\n" in content:
                content = content.split("\n\n---\n")[0]
            msgs.append({
                "user_question": u["content"],
                "response": content,
                "sources_cited": [],
            })
        i += 2
    return dspy.History(messages=msgs)


def _render_verse_html(verse: Verse) -> str:
    ev = verse if isinstance(verse, EnrichedVerse) else None
    parts: list[str] = []

    ref = verse.verse_ref
    work = getattr(verse, "work_display", None) or getattr(verse, "work", "")
    section = getattr(verse, "section_display", None) or getattr(verse, "section", "") or ""
    subtitle = f"{work} — {section}" if section else work

    parts.append(
        f'<div class="vp-header">'
        f'  <span class="vp-ref">{ref}</span>'
        f'  <span class="vp-subtitle">{subtitle}</span>'
        f'</div>'
    )

    if getattr(verse, "sanskrit", None):
        parts.append(f'<div class="vp-sanskrit">{verse.sanskrit}</div>')
    if getattr(verse, "transliteration", None):
        parts.append(f'<div class="vp-iast">{verse.transliteration}</div>')

    if getattr(verse, "translation", None):
        tr = getattr(verse, "translator", None)
        label = f"Translation ({tr})" if tr else "Translation"
        parts.append(f'<div class="vp-label">{label}</div>')
        parts.append(f'<div class="vp-body">{verse.translation}</div>')

    if getattr(verse, "bhashya", None):
        btr = getattr(verse, "bhashya_translator", None)
        note = f" ({btr})" if btr else ""
        preview = verse.bhashya[:900] + ("…" if len(verse.bhashya) > 900 else "")
        parts.append(f'<div class="vp-label">Śaṅkara\'s Bhāṣya{note}</div>')
        parts.append(f'<div class="vp-body vp-dim">{preview}</div>')

    if ev:
        if getattr(ev, "paraphrase", None):
            parts.append('<div class="vp-label">Teaching</div>')
            parts.append(f'<div class="vp-body">{ev.paraphrase}</div>')
        if getattr(ev, "themes", None):
            tags = "".join(f'<span class="vp-tag">{t}</span>' for t in ev.themes)
            parts.append(f'<div class="vp-tags">{tags}</div>')
        if getattr(ev, "practical_teaching", None):
            parts.append('<div class="vp-label">Practical Shift</div>')
            parts.append(f'<div class="vp-body vp-gold">{ev.practical_teaching}</div>')

    return '<div class="verse-panel">' + "\n".join(parts) + "</div>"


# ── CSS ────────────────────────────────────────────────────────────────────────

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,600;1,400&family=EB+Garamond:ital,wght@0,400;0,500;1,400;1,500&family=Lato:ital,wght@0,300;0,400;0,700;1,300;1,400&display=swap');

/* ── palette ──────────────────────────────────────────────────────────────── */
:root {
  --gold:        #C9A84C;
  --gold-dim:    #9A7830;
  --gold-glow:   rgba(201,168,76,0.18);
  --bg:          #100C07;
  --bg-mid:      #1C1208;
  --bg-card:     #251808;
  --bg-user:     #3D240C;
  --bg-bot:      #180F05;
  --border:      #5A3C18;
  --border-dim:  #3A2408;
  --text:        #ECD8B4;
  --text-dim:    #A08860;
  --text-muted:  #6A5030;
  --radius:      10px;
  --font-serif:  'EB Garamond', Georgia, 'Times New Roman', serif;
  --font-sans:   'Lato', system-ui, sans-serif;
  --font-display: 'Playfair Display', Georgia, serif;
}

/* ── base ─────────────────────────────────────────────────────────────────── */
body,
.gradio-container,
.main,
footer {
  background: var(--bg) !important;
  color: var(--text) !important;
  font-family: var(--font-sans) !important;
}

.gradio-container { max-width: 880px !important; margin: 0 auto !important; }

footer { display: none !important; }

/* ── header ───────────────────────────────────────────────────────────────── */
.app-header {
  text-align: center;
  padding: 2.4rem 1rem 1.6rem;
  border-bottom: 1px solid var(--border);
  margin-bottom: 0.5rem;
}
.app-title {
  font-family: var(--font-display);
  font-size: 2.6rem;
  color: var(--gold);
  letter-spacing: 0.05em;
  line-height: 1.15;
  margin: 0 0 0.5rem;
  font-weight: 400;
}
.app-subtitle {
  color: var(--text-muted);
  font-size: 0.95rem;
  font-weight: 300;
  font-style: italic;
  font-family: var(--font-serif);
  letter-spacing: 0.03em;
}
.app-ornament {
  margin-top: 1rem;
  color: var(--gold-dim);
  font-size: 0.85rem;
  letter-spacing: 0.7em;
}

/* ── chatbot container ────────────────────────────────────────────────────── */
#chatbot {
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
  background: var(--bg) !important;
}
#chatbot .wrap { background: var(--bg) !important; }

/* user bubble */
#chatbot .user.message {
  background: var(--bg-user) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) var(--radius) 3px var(--radius) !important;
  padding: 0.8rem 1.1rem !important;
  box-shadow: inset 0 1px 0 rgba(255,220,140,0.07) !important;
}
/* assistant bubble */
#chatbot .bot.message {
  background: var(--bg-bot) !important;
  border: 1px solid var(--gold-dim) !important;
  border-left: 3px solid var(--gold-dim) !important;
  border-radius: 3px var(--radius) var(--radius) var(--radius) !important;
  padding: 1.1rem 1.4rem 1.1rem 1.6rem !important;
  line-height: 1.9 !important;
  font-family: var(--font-serif) !important;
  font-size: 1.08rem !important;
  color: var(--text) !important;
}
/* user bubble */
#chatbot .user.message {
  background: var(--bg-user) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) var(--radius) 3px var(--radius) !important;
  padding: 0.8rem 1.1rem !important;
  box-shadow: inset 0 1px 0 rgba(255,220,140,0.07) !important;
  font-family: var(--font-sans) !important;
  font-size: 0.96rem !important;
}
/* inner content divs — transparent so bubble bg shows through */
#chatbot .message.panel-full-width,
#chatbot [data-testid="user"],
#chatbot [data-testid="bot"] {
  background: transparent !important;
  color: var(--text) !important;
  padding: 0 !important;
}

/* markdown inside bubbles */
#chatbot .bot.message p   { margin: 0.55em 0 !important; }
#chatbot .user.message p  { margin: 0.3em 0 !important; }
#chatbot .message hr  { border-color: var(--border) !important; margin: 0.6em 0 !important; }
#chatbot .message code {
  background: var(--bg-card) !important;
  color: var(--gold) !important;
  padding: 0.1em 0.4em !important;
  border-radius: 4px !important;
  font-size: 0.88em !important;
  font-family: var(--font-sans) !important;
}
#chatbot .message em { color: var(--text-dim) !important; }
#chatbot .message strong { color: var(--text) !important; font-weight: 500 !important; }

/* placeholder */
#chatbot .placeholder {
  color: var(--text-muted) !important;
  font-style: italic !important;
}

/* ── stage status ─────────────────────────────────────────────────────────── */
#stage-status {
  min-height: 1.8rem;
  text-align: center;
  padding: 0.4rem 0.5rem;
  font-family: 'Lato', sans-serif;
  line-height: 1.55;
}
#stage-status .stage-spinner {
  color: var(--gold);
  font-style: italic;
  font-size: 0.88rem;
  opacity: 0.9;
}
#stage-status .stage-card {
  display: inline-block;
  text-align: left;
  max-width: 90%;
  font-size: 0.84rem;
}
#stage-status .stage-row {
  display: flex;
  align-items: baseline;
  flex-wrap: wrap;
  gap: 0.3rem 0.6rem;
  margin-bottom: 0.2rem;
}
#stage-status .stage-icon { color: var(--gold-dim); }
#stage-status .stage-label {
  color: var(--text-muted);
  font-size: 0.73rem;
  text-transform: uppercase;
  letter-spacing: 0.09em;
  min-width: 4.5rem;
}
#stage-status .stage-val { color: var(--text); font-style: italic; }
#stage-status .stage-chip {
  display: inline-block;
  border: 1px solid var(--border);
  border-radius: 3px;
  padding: 0 0.35rem;
  font-size: 0.78rem;
  color: var(--text-dim);
  font-style: normal;
  margin: 0.1rem 0.1rem 0 0;
}
#stage-status .stage-source {
  display: inline-block;
  background: var(--bg-card);
  border: 1px solid var(--gold-dim);
  border-radius: 3px;
  padding: 0 0.35rem;
  font-size: 0.78rem;
  color: var(--gold);
  margin: 0.1rem 0.1rem 0 0;
}

/* ── input textbox ────────────────────────────────────────────────────────── */
#msg-input {
  background: var(--bg-card) !important;
  border-radius: var(--radius) !important;
}
#msg-input label.show_textbox_border {
  background: var(--bg-card) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
  transition: border-color 0.15s, box-shadow 0.15s !important;
}
#msg-input label.show_textbox_border:focus-within {
  border-color: var(--gold-dim) !important;
  box-shadow: 0 0 0 3px var(--gold-glow) !important;
}
#msg-input span.svelte-1hguek3 { display: none !important; } /* hide label text */
#msg-input textarea {
  background: var(--bg-card) !important;
  color: var(--text) !important;
  font-family: var(--font-serif) !important;
  font-size: 1.02rem !important;
  line-height: 1.55 !important;
  caret-color: var(--gold) !important;
  resize: none !important;
  border: none !important;
  outline: none !important;
}
#msg-input textarea::placeholder { color: var(--text-muted) !important; }

/* ── buttons ──────────────────────────────────────────────────────────────── */
#submit-btn {
  background: var(--gold-dim) !important;
  color: #0D0A07 !important;
  border: none !important;
  border-radius: var(--radius) !important;
  font-family: var(--font-sans) !important;
  font-weight: 700 !important;
  letter-spacing: 0.05em !important;
  transition: background 0.18s !important;
  height: 100% !important;
}
#submit-btn:hover { background: var(--gold) !important; cursor: pointer !important; }

#clear-btn {
  background: transparent !important;
  color: var(--text-muted) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
  font-family: var(--font-sans) !important;
  transition: color 0.15s, border-color 0.15s !important;
  width: 46px !important;
  min-width: 46px !important;
  max-width: 46px !important;
  flex-shrink: 0 !important;
  padding: 0 !important;
  font-size: 1.1rem !important;
}
#clear-btn:hover { color: var(--text-dim) !important; border-color: var(--text-muted) !important; cursor: pointer !important; }

/* ── examples ─────────────────────────────────────────────────────────────── */
.examples-holder .examples-inner-text { color: var(--text-muted) !important; font-size: 0.8rem !important; }
.examples-holder table { border: none !important; }
.examples-holder table td {
  background: var(--bg-card) !important;
  border: 1px solid var(--border) !important;
  color: var(--text-dim) !important;
  border-radius: 6px !important;
  font-size: 0.86rem !important;
  transition: background 0.15s, color 0.15s !important;
  cursor: pointer !important;
}
.examples-holder table td:hover {
  background: var(--bg-mid) !important;
  color: var(--text) !important;
  border-color: var(--gold-dim) !important;
}

/* ── explorer section ─────────────────────────────────────────────────────── */
.explorer-wrap {
  border-top: 1px solid var(--border);
  margin-top: 1.5rem;
  padding-top: 1.2rem;
}
.explorer-label {
  color: var(--text-muted);
  font-size: 0.75rem;
  text-transform: uppercase;
  letter-spacing: 0.12em;
  margin-bottom: 0.6rem;
  font-family: 'Lato', sans-serif;
}

#source-dd label { color: var(--text-dim) !important; font-size: 0.82rem !important; }
#source-dd select {
  background: var(--bg-card) !important;
  border: 1px solid var(--border) !important;
  color: var(--text) !important;
  border-radius: var(--radius) !important;
}

/* ── verse panel ──────────────────────────────────────────────────────────── */
.verse-panel {
  background: var(--bg-mid);
  border: 1px solid var(--gold-dim);
  border-radius: var(--radius);
  padding: 1.6rem 2rem 1.8rem;
  margin-top: 0.8rem;
  line-height: 1.85;
  font-family: var(--font-serif);
}
.vp-header {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  flex-wrap: wrap;
  gap: 0.4rem;
  border-bottom: 1px solid var(--border);
  padding-bottom: 0.8rem;
  margin-bottom: 1.1rem;
}
.vp-ref {
  font-family: var(--font-display);
  font-size: 1.2rem;
  color: var(--gold);
  font-weight: 400;
  letter-spacing: 0.04em;
}
.vp-subtitle {
  color: var(--text-muted);
  font-size: 0.82rem;
  font-style: italic;
  font-family: var(--font-sans);
}
.vp-sanskrit {
  font-size: 1.05rem;
  color: var(--text);
  font-style: italic;
  margin-bottom: 0.2rem;
  font-family: var(--font-serif);
}
.vp-iast {
  color: var(--text-dim);
  font-size: 0.9rem;
  font-style: italic;
  margin-bottom: 1rem;
  font-family: var(--font-serif);
}
.vp-label {
  color: var(--gold-dim);
  font-size: 0.70rem;
  text-transform: uppercase;
  letter-spacing: 0.13em;
  margin-top: 1.1rem;
  margin-bottom: 0.35rem;
  font-family: var(--font-sans);
  font-weight: 700;
}
.vp-body { color: var(--text); font-size: 1rem; font-family: var(--font-serif); line-height: 1.85; }
.vp-dim  { color: var(--text-dim) !important; font-style: italic; font-size: 0.93rem !important; }
.vp-gold { color: var(--gold) !important; font-style: italic; }

.vp-tags { display: flex; flex-wrap: wrap; gap: 0.35rem; margin-top: 0.8rem; }
.vp-tag {
  background: var(--bg-card);
  border: 1px solid var(--border);
  color: var(--text-muted);
  font-size: 0.73rem;
  padding: 0.12rem 0.6rem;
  border-radius: 20px;
  font-family: var(--font-sans);
}

/* ── explain button & output ──────────────────────────────────────────────── */
#explain-btn {
  background: transparent !important;
  color: var(--text-muted) !important;
  border: 1px solid var(--border-dim) !important;
  border-radius: 6px !important;
  font-family: 'Lato', sans-serif !important;
  font-size: 0.82rem !important;
  letter-spacing: 0.05em !important;
  margin-top: 0.6rem !important;
  transition: color 0.15s, border-color 0.15s, opacity 0.15s !important;
  opacity: 0.4 !important;
}
#explain-btn:not([disabled]):not(.disabled) {
  color: var(--gold-dim) !important;
  opacity: 1 !important;
}
#explain-btn:not([disabled]):not(.disabled):hover {
  color: var(--gold) !important;
  border-color: var(--gold-dim) !important;
}

.explain-panel {
  background: var(--bg-mid);
  border-left: 3px solid var(--gold-dim);
  border-radius: 0 var(--radius) var(--radius) 0;
  padding: 1.3rem 1.7rem;
  margin-top: 0.8rem;
  color: var(--text);
  font-size: 1rem;
  line-height: 1.9;
  font-style: italic;
  font-family: var(--font-serif);
}

/* ── reasoning panel ─────────────────────────────────────────── */
.reasoning-panel {
  font-family: var(--font-sans);
  font-size: 0.82rem;
  line-height: 1.65;
  color: var(--text-dim);
}
.reasoning-panel .r-section {
  margin-bottom: 1rem;
}
.reasoning-panel .r-label {
  color: var(--gold-dim);
  font-size: 0.70rem;
  text-transform: uppercase;
  letter-spacing: 0.12em;
  margin-bottom: 0.3rem;
}
.reasoning-panel .r-value {
  color: var(--text);
  font-size: 0.85rem;
}
.reasoning-panel .r-trace {
  white-space: pre-wrap;
  word-break: break-word;
  color: var(--text-dim);
  font-size: 0.80rem;
  border-left: 2px solid var(--border);
  padding-left: 0.8rem;
  margin-top: 0.3rem;
}
/* accordion styling */
#thinking-accordion > .label-wrap { color: var(--text-dim) !important; font-size: 0.82rem; }
#thinking-accordion { background: transparent !important; border: 1px solid var(--border) !important; border-radius: var(--radius) !important; margin-top: 0.5rem; }

/* ── scrollbar ────────────────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--gold-dim); }
"""


def _spin(text: str) -> str:
    return f'<div class="stage-spinner">◌  {text}</div>'


def _stage_understand(u) -> str:
    emotion = getattr(u, "felt_emotion", "") or ""
    concern = getattr(u, "deeper_concern", "") or ""
    themes = getattr(u, "vedantic_themes", []) or []
    themes_html = "".join(f'<span class="stage-chip">{t.split("(")[0].strip()}</span>' for t in themes[:4])
    rows = []
    if emotion:
        rows.append(
            f'<div class="stage-row">'
            f'<span class="stage-label">felt</span>'
            f'<span class="stage-val">{emotion}</span>'
            f'</div>'
        )
    if concern:
        rows.append(
            f'<div class="stage-row">'
            f'<span class="stage-label">concern</span>'
            f'<span class="stage-val">{concern}</span>'
            f'</div>'
        )
    if themes_html:
        rows.append(
            f'<div class="stage-row">'
            f'<span class="stage-label">themes</span>'
            f'<span>{themes_html}</span>'
            f'</div>'
        )
    return f'<div class="stage-card">{"".join(rows)}</div>'


def _stage_plan(queries: list[str]) -> str:
    chips = "".join(f'<span class="stage-chip">"{q}"</span>' for q in queries)
    return (
        f'<div class="stage-card">'
        f'<div class="stage-row">'
        f'<span class="stage-label">searching</span>'
        f'<span>{chips}</span>'
        f'</div></div>'
    )


def _stage_retrieve(n: int) -> str:
    return (
        f'<div class="stage-card">'
        f'<div class="stage-row">'
        f'<span class="stage-label">passages</span>'
        f'<span class="stage-val">{n} found &nbsp;—&nbsp; selecting…</span>'
        f'</div></div>'
    )


def _stage_select(sources: list[str]) -> str:
    chips = "".join(f'<span class="stage-source">{s}</span>' for s in sources)
    return (
        f'<div class="stage-card">'
        f'<div class="stage-row">'
        f'<span class="stage-label">selected</span>'
        f'<span>{chips}</span>'
        f'</div>'
        f'<div class="stage-row" style="margin-top:0.15rem">'
        f'<span class="stage-label"></span>'
        f'<span class="stage-spinner" style="font-size:0.82rem">◌  composing response…</span>'
        f'</div></div>'
    )


def _build_reasoning_html(pred) -> str:
    """Render the pipeline's reasoning trace as an HTML block for the accordion."""
    emotion = getattr(pred, "felt_emotion", "") or ""
    concern = getattr(pred, "deeper_concern", "") or ""
    themes = getattr(pred, "vedantic_themes", []) or []
    queries = getattr(pred, "queries", []) or []
    reasoning = getattr(pred, "synthesis_reasoning", "") or ""
    rationale = getattr(pred, "selection_rationale", "") or ""

    def section(label: str, content: str) -> str:
        return (
            f'<div class="r-section">'
            f'<div class="r-label">{label}</div>'
            f'<div class="r-value">{content}</div>'
            f'</div>'
        )

    parts = ['<div class="reasoning-panel">']
    if emotion:
        parts.append(section("Felt emotion", emotion))
    if concern:
        parts.append(section("Deeper concern", concern))
    if themes:
        parts.append(section("Vedāntic themes", " &nbsp;·&nbsp; ".join(themes)))
    if queries:
        qs = "".join(f"<li>{q}</li>" for q in queries)
        parts.append(section("Search queries", f"<ol style='margin:0;padding-left:1.2em'>{qs}</ol>"))
    if rationale:
        parts.append(section("Passage selection", rationale))
    if reasoning:
        escaped = reasoning.replace("<", "&lt;").replace(">", "&gt;")
        parts.append(
            '<div class="r-section">'
            '<div class="r-label">Model reasoning trace</div>'
            f'<div class="r-trace">{escaped}</div>'
            '</div>'
        )
    parts.append("</div>")
    return "\n".join(parts)


# ── respond (streaming generator) ─────────────────────────────────────────────

def respond(message: str, history: list):
    """Drive the 4-step pipeline manually so each step's output is shown live."""
    _no_src = gr.update(choices=[], value=None, visible=False)
    _noop   = gr.update()

    def _emit(hist, stage_content, thinking_content=_noop):
        return hist, stage_content, None, _no_src, "", thinking_content

    if not message.strip():
        yield *_emit(history, ""), _noop
        return

    history = history + [{"role": "user", "content": message}]
    dspy_hist = _to_dspy_history(history[:-1])

    # ── Step 1: understand ────────────────────────────────────────────────────
    yield *_emit(history, _spin("understanding your question…")),
    try:
        u = _advisor.understand(history=dspy_hist, user_question=message)
    except Exception as exc:
        history = history + [{"role": "assistant", "content": f"*Error — {exc}*"}]
        yield *_emit(history, ""),
        return

    # Show what was understood; plan is next
    yield *_emit(history, _stage_understand(u)),

    # ── Step 2: plan retrieval queries ────────────────────────────────────────
    try:
        p = _advisor.plan(
            surface_concern=u.surface_concern,
            deeper_concern=u.deeper_concern,
            vedantic_themes=u.vedantic_themes,
        )
    except Exception as exc:
        history = history + [{"role": "assistant", "content": f"*Error — {exc}*"}]
        yield *_emit(history, ""),
        return

    queries = p.queries[: config.N_RETRIEVAL_QUERIES] if p.queries else [u.deeper_concern]
    yield *_emit(history, _stage_plan(queries)),

    # ── Step 3: retrieve (fast, local Chroma) ────────────────────────────────
    hits = _advisor._retriever.search_many(queries, k_per=config.TOP_K_RETRIEVE)
    candidates = hits[: max(8, config.TOP_K_RETRIEVE)]
    candidates_text = format_passages_for_llm(candidates)
    candidates_as_dicts = [h.to_dict() for h in candidates]
    previously_cited = [
        src for msg in dspy_hist.messages for src in msg.get("sources_cited", [])
    ]
    yield *_emit(history, _stage_retrieve(len(candidates))),

    # ── Step 4: select passages ───────────────────────────────────────────────
    try:
        s = _advisor.select(
            deeper_concern=u.deeper_concern,
            candidate_passages=candidates_text,
            previously_cited=previously_cited,
        )
    except Exception as exc:
        history = history + [{"role": "assistant", "content": f"*Error — {exc}*"}]
        yield *_emit(history, ""),
        return

    valid_idx = [
        i for i in (s.selected_indices or [])
        if isinstance(i, int) and 1 <= i <= len(candidates)
    ]
    if not valid_idx:
        valid_idx = list(range(1, min(4, len(candidates) + 1)))
    selected = [candidates[i - 1] for i in valid_idx]
    selected_text = format_passages_for_llm(selected)

    # Show selected sources; synthesize is next
    selected_refs = [
        candidates_as_dicts[i - 1].get("verse_ref", f"#{i}").upper().replace("_", " ")
        for i in valid_idx
        if i - 1 < len(candidates_as_dicts)
    ]
    yield *_emit(history, _stage_select(selected_refs)),

    # ── Step 5: synthesize ────────────────────────────────────────────────────
    try:
        a = _advisor.synthesize(
            history=dspy_hist,
            user_question=message,
            felt_emotion=u.felt_emotion,
            deeper_concern=u.deeper_concern,
            selected_passages=selected_text,
        )
    except Exception as exc:
        history = history + [{"role": "assistant", "content": f"*Error — {exc}*"}]
        yield *_emit(history, ""),
        return

    pred = dspy.Prediction(
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
    thinking = _build_reasoning_html(pred)

    # ── Stream response word by word ─────────────────────────────────────────
    history = history + [{"role": "assistant", "content": ""}]
    streamed = ""
    words = pred.response.split(" ")
    for i, word in enumerate(words):
        streamed += word + (" " if i < len(words) - 1 else "")
        history[-1]["content"] = streamed
        yield history, "", None, _no_src, "", thinking
        time.sleep(0.018)

    if pred.sources_cited:
        footer = "\n\n---\n**Sources:** " + "  ·  ".join(f"`{s}`" for s in pred.sources_cited)
        history[-1]["content"] += footer

    sources = pred.sources_cited or []
    yield history, "", pred, gr.update(choices=sources, value=None, visible=bool(sources)), "", thinking


def show_verse(ref: str) -> tuple[str, str]:
    """Return (verse_html, explain_html) — clears any prior explanation."""
    if not ref:
        return "", ""
    verse = _verse_lookup.get(ref.lower().strip())
    if verse is None:
        return '<div class="verse-panel"><p style="color:var(--text-muted)">Verse not found in corpus.</p></div>', ""
    return _render_verse_html(verse), ""


def explain_verse(source_ref: str, history: list):
    """Generator: stream a contextual explanation of the selected verse."""
    if not source_ref:
        yield '<div class="explain-panel" style="color:var(--text-muted)">Select a verse first.</div>'
        return

    verse = _verse_lookup.get(source_ref.lower().strip())
    if verse is None:
        yield '<div class="explain-panel" style="color:var(--text-muted)">Verse not found in corpus.</div>'
        return

    # Build verse content string
    bits = []
    if getattr(verse, "translation", None):
        bits.append(f"Translation: {verse.translation}")
    if getattr(verse, "sanskrit", None):
        bits.append(f"Sanskrit: {verse.sanskrit}")
    if getattr(verse, "bhashya", None):
        bits.append(f"Śaṅkara's commentary: {verse.bhashya[:600]}")
    ev = verse if isinstance(verse, EnrichedVerse) else None
    if ev and getattr(ev, "paraphrase", None):
        bits.append(f"Teaching: {ev.paraphrase}")
    verse_content = "\n\n".join(bits)

    # Build conversation context from the last turn
    context = "No prior conversation."
    i = len(history) - 1
    while i >= 0:
        if history[i].get("role") == "assistant" and i > 0:
            user_msg = history[i - 1].get("content", "")
            bot_msg = history[i].get("content", "")
            if "\n\n---\n" in bot_msg:
                bot_msg = bot_msg.split("\n\n---\n")[0]
            context = f"User: {user_msg}\n\nAdvisor: {bot_msg}"
            break
        i -= 1

    # Run in thread so we can stream
    result_box: list = [None]
    err_box: list = [None]
    done = threading.Event()

    def _run():
        try:
            explainer = dspy.ChainOfThought(_ExplainInContext)
            result_box[0] = explainer(
                verse_ref=source_ref,
                verse_content=verse_content,
                conversation_context=context,
            )
        except Exception as exc:
            err_box[0] = exc
        finally:
            done.set()

    threading.Thread(target=_run, daemon=True).start()

    yield '<div class="explain-panel" style="color:var(--text-muted);font-style:italic;">◌  drawing the thread…</div>'
    while not done.wait(timeout=0.2):
        yield '<div class="explain-panel" style="color:var(--text-muted);font-style:italic;">◌  drawing the thread…</div>'

    if err_box[0]:
        yield f'<div class="explain-panel" style="color:var(--text-muted)">Could not generate explanation: {err_box[0]}</div>'
        return

    explanation = result_box[0].explanation
    # Stream character by character
    streamed = ""
    for char in explanation:
        streamed += char
        yield f'<div class="explain-panel">{streamed}</div>'


# ── layout ─────────────────────────────────────────────────────────────────────

EXAMPLES = [
    "I just got laid off and feel like nothing makes sense.",
    "I'm terrified of dying. Is that irrational?",
    "I keep hurting the people I love without meaning to.",
    "I've been meditating for years but still feel empty.",
    "My ambition feels hollow but I can't stop chasing it.",
]

with gr.Blocks(title="Gītā Advisor") as demo:

    pred_state = gr.State(None)

    gr.HTML("""
    <div class="app-header">
      <div class="app-title">Gītā Advisor</div>
      <div class="app-subtitle">Grounded in Advaita Vedānta as taught by Śaṅkarācārya</div>
      <div class="app-ornament">✦ &nbsp; ✦ &nbsp; ✦</div>
    </div>
    """)

    chatbot = gr.Chatbot(
        height=480,
        show_label=False,
        elem_id="chatbot",
        render_markdown=True,
        placeholder=(
            '<div style="text-align:center;padding:3rem 1rem;">'
            '<span style="color:#5A3F1E;font-style:italic;font-size:0.95rem;">'
            "Speak from where you actually are.<br>"
            '<span style="font-size:0.82rem">The teacher will meet you there.</span>'
            "</span></div>"
        ),
    )

    stage_html = gr.HTML("", elem_id="stage-status")

    with gr.Row(equal_height=True):
        msg_box = gr.Textbox(
            placeholder="Speak from where you actually are…",
            show_label=False,
            lines=2,
            max_lines=6,
            elem_id="msg-input",
            scale=7,
            container=False,
        )
        submit_btn = gr.Button("Ask →", variant="primary",   elem_id="submit-btn", size="lg", scale=1, min_width=110)
        clear_btn  = gr.Button("✕",    variant="secondary", elem_id="clear-btn",  size="lg", scale=0, min_width=46)

    gr.Examples(examples=EXAMPLES, inputs=msg_box, label="Opening moves")

    with gr.Column(elem_classes=["explorer-wrap"]):
        gr.HTML('<div class="explorer-label">Explore a cited verse</div>')
        source_dd = gr.Dropdown(
            choices=[],
            value=None,
            label="Select a cited source…",
            show_label=False,
            elem_id="source-dd",
            visible=False,
            interactive=True,
        )
        verse_html  = gr.HTML("")
        explain_btn = gr.Button("Explain in context →", elem_id="explain-btn", visible=True, interactive=False, size="sm")
        explain_out = gr.HTML("")

    with gr.Accordion("🧠  Model reasoning", open=False, elem_id="thinking-accordion"):
        thinking_html = gr.HTML("")

    # ── event wiring ──────────────────────────────────────────────────────────
    outputs = [chatbot, stage_html, pred_state, source_dd, msg_box, thinking_html]

    msg_box.submit(respond, [msg_box, chatbot], outputs)
    submit_btn.click(respond, [msg_box, chatbot], outputs)

    clear_btn.click(
        fn=lambda: ([], "", None, gr.update(choices=[], value=None, visible=False), "", "", "", ""),
        outputs=[chatbot, stage_html, pred_state, source_dd, msg_box, verse_html, explain_out, thinking_html],
    )

    source_dd.change(
        fn=lambda ref: (*show_verse(ref), gr.update(interactive=bool(ref))),
        inputs=source_dd,
        outputs=[verse_html, explain_out, explain_btn],
    )

    explain_btn.click(
        fn=explain_verse,
        inputs=[source_dd, chatbot],
        outputs=explain_out,
    )

demo.queue()

if __name__ == "__main__":
    demo.launch(server_port=7860, css=CSS)
