"""
app.py — Gradio web interface for the Gītā Advisor.

Wraps the same advisor pipeline as chat.py but exposes it as a Gradio
ChatInterface suitable for Hugging Face Spaces (free CPU tier).

Deploy:
  - Set GEMINI_API_KEY as a Space Secret (Space Settings → Secrets)
  - Push this file + all project files + artifacts/chroma/ + data/corpus_enriched.jsonl
"""

import gradio as gr
import dspy

import config
from advisor import load_optimized
from knowledge_base import AdvaitaRetriever

# ── startup — runs once when the Space boots ───────────────────────────────────
config.configure_dspy()
_advisor = load_optimized()

# Pre-warm retriever so the first user request isn't slow
_retriever = AdvaitaRetriever()
_retriever._ensure()


# ── chat handler ───────────────────────────────────────────────────────────────
def chat(message: str, history: list) -> str:
    # Gradio type="messages" passes history as list of {"role": ..., "content": ...}
    dspy_msgs = []
    i = 0
    while i + 1 < len(history):
        user_msg = history[i]
        bot_msg = history[i + 1]
        if user_msg.get("role") == "user" and bot_msg.get("role") == "assistant":
            dspy_msgs.append({
                "user_question": user_msg["content"],
                "response": bot_msg["content"],
                "sources_cited": [],
            })
        i += 2
    dspy_history = dspy.History(messages=dspy_msgs)

    pred = _advisor(user_question=message, history=dspy_history)

    reply = pred.response
    if pred.sources_cited:
        reply += "\n\n---\n**Sources:** " + " · ".join(pred.sources_cited)
    return reply


# ── Gradio app ─────────────────────────────────────────────────────────────────
demo = gr.ChatInterface(
    fn=chat,
    title="Gītā Advisor",
    description=(
        "A spiritual advisor grounded in Advaita Vedānta as taught by Śaṅkarācārya. "
        "Speak from where you actually are. The advisor cites exact verses from the "
        "Gītā with Śaṅkara's commentary."
    ),
    type="messages",
    examples=[
        "I just got laid off and feel like nothing makes sense.",
        "I'm terrified of dying. Is that irrational?",
        "I keep hurting the people I love without meaning to.",
        "I've been meditating for years but still feel empty. What am I missing?",
    ],
)

if __name__ == "__main__":
    demo.launch()
