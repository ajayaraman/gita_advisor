"""
enrichment.py — turn a Verse into an EnrichedVerse using the local LLM.

This module is the heart of the redesign. Instead of hoping that vector
similarity between a user's English question and a Sanskrit verse will find
the right teaching, we run a one-time offline pass that asks the local LLM
to translate each verse into the language a real person would use to seek
help. The output gets stored alongside the verse and embedded for retrieval.

What the prompt asks for, and why each field
--------------------------------------------
We extract six fields. Each one earns its place by closing a different gap
between scripture and a user's question:

  paraphrase             — what the verse teaches, in plain modern English.
                           This is what the synthesizer reads when writing
                           the advisor's reply, so paraphrase quality matters
                           more than embedding quality.

  themes                 — Vedānta concepts engaged. Tradition-native names
                           (karma_yoga, vairagya, sakshi, two_truths). Used
                           for filtering and for ensuring the metric can
                           verify Advaita-coherence.

  life_situations        — the predicaments where this verse helps. User-
                           language. This is the field that does the actual
                           bridging: a query about "facing failure" finds
                           BG 2.47 even though those words aren't in the verse.

  emotions_addressed     — drawn from a fixed vocabulary so we get faceted
                           filtering rather than free-text drift. The metric
                           uses this to verify that retrieved verses actually
                           address the user's felt emotion.

  practical_teaching     — what the verse asks the seeker to do or shift.
                           The synthesizer uses this as the seed for its
                           "concrete practice you can try this week" close.

  hypothetical_questions — five questions a real person might bring to the
                           verse. Highest-leverage field for retrieval recall.

A closed vocabulary for emotions
--------------------------------
We constrain `emotions_addressed` to the EMOTION_VOCAB list below. If we let
the LLM generate freely, we get drift: "sadness" / "sorrow" / "melancholy" /
"grief-tinged blue" all become separate buckets, and faceted filtering
becomes useless. Closed vocab keeps the index sharp.

We don't constrain themes the same way because the Sanskrit conceptual
vocabulary is open-ended and forcing the LLM into a small list would lose
information. We just normalize for casing/spacing in post-processing.

Working with a flaky local LLM
------------------------------
Local 26B-class models occasionally produce malformed structured output.
This module assumes that. The enrich() function:
  - validates output against minimum-quality checks
  - retries up to 2 times with temperature=0
  - on persistent failure, returns an EnrichedVerse with empty enrichment
    fields rather than raising — so the corpus can still index on the
    literal text + bhāṣya and the verse isn't lost
"""

from __future__ import annotations
import re
from dataclasses import asdict
import dspy

from corpus import Verse, EnrichedVerse


# ──────────────────────────── Closed emotion vocabulary ────────────────────────────
# Twenty buckets, ordered roughly from acute to diffuse. Adding entries is
# easy; removing them risks orphaning previously-enriched records.
EMOTION_VOCAB: tuple[str, ...] = (
    "grief",                # acute loss
    "anticipatory_grief",   # loss in advance
    "fear",                 # discrete fear
    "anxiety",              # chronic, diffuse
    "despair",              # loss of hope
    "shame",                # self-as-bad
    "guilt",                # action-as-bad
    "anger",
    "resentment",
    "envy",
    "jealousy",
    "longing",
    "loneliness",
    "doubt",                # epistemic; not knowing
    "disillusionment",      # the hollowness of attained goals
    "boredom",              # the inertness of repetition
    "restlessness",         # the inability to settle
    "frustration",
    "confusion",
    "numbness",             # affect-blunted
)


# ──────────────────────────── DSPy signature ────────────────────────────
class EnrichVerse(dspy.Signature):
    """You are an Advaita-Vedānta-trained reader producing structured metadata
    for a verse from the Bhagavad Gītā or a related scripture, so that a
    spiritual advisor can later find this verse when a real person describes
    a life situation in everyday language. Stay strictly within the framework
    of Śaṅkarācārya's non-dual interpretation. Do not import dualistic notions
    (separate creator/creature, soul-merging-into-God-as-other, etc.) and do
    not bypass the verse's plain meaning by always retreating to the absolute.

    The verse may include the Sanskrit, the English translation, and (when
    available) Śaṅkara's commentary. Read all three. Your output is structured
    fields, not prose. Be specific, lived, concrete. Avoid generic spiritual
    language ('find peace', 'be in the moment'). Avoid tradition-foreign
    therapy language ('honor your feelings'). When in doubt about a field,
    leave it shorter rather than padded."""

    # Inputs — the verse in its richest available form
    verse_ref: str = dspy.InputField(desc="Citation form, e.g. 'BG 2.47'.")
    sanskrit: str = dspy.InputField(desc="Devanāgarī text, may be empty.")
    translation: str = dspy.InputField(desc="English translation of the verse.")
    bhashya: str = dspy.InputField(desc="Śaṅkara's commentary on this verse, may be empty.")

    # Outputs
    paraphrase: str = dspy.OutputField(
        desc="One or two sentences in plain modern English stating what the "
             "verse teaches. Not a translation; a teaching summary. No jargon."
    )
    themes: list[str] = dspy.OutputField(
        desc="2–5 Vedānta concepts the verse engages, in tradition-native "
             "vocabulary with snake_case_keys, e.g. ['karma_yoga', 'non_attachment', "
             "'two_truths']. Use Sanskrit terms where they're the right name."
    )
    life_situations: list[str] = dspy.OutputField(
        desc="3–6 specific human predicaments this verse would help with, "
             "in everyday English. e.g. 'facing public failure after years of "
             "effort'. NOT 'finding peace' or 'spiritual growth'."
    )
    emotions_addressed: list[str] = dspy.OutputField(
        desc="The emotions this verse meets, drawn ONLY from this fixed list: "
             + ", ".join(EMOTION_VOCAB) + ". 1–4 entries."
    )
    practical_teaching: str = dspy.OutputField(
        desc="One sentence: what the verse asks the seeker to actually do or "
             "shift. If the verse is purely ontological, write 'pure ontology — "
             "no direct prescription' and the field will be ignored downstream."
    )
    hypothetical_questions: list[str] = dspy.OutputField(
        desc="EXACTLY 5 first-person questions a real person might write to a "
             "spiritual advisor that this verse would speak to. Specific, "
             "ungeneric, in the user's voice. NOT in scripture's voice. e.g. "
             "'I worked on this for three years and it just failed publicly — "
             "how do I keep going?'"
    )


# ──────────────────────────── Validators ────────────────────────────
THEME_KEY_RX = re.compile(r"^[a-z][a-z0-9_]{2,40}$")


def _normalize_theme(t: str) -> str:
    t = t.strip().lower()
    t = re.sub(r"[\s\-]+", "_", t)
    t = re.sub(r"[^a-z0-9_]", "", t)
    return t


def _validate(pred) -> tuple[bool, str]:
    """Light schema check. Returns (ok, reason_if_not_ok). Used to decide
    whether to retry the LM call with a stricter prompt."""
    paraphrase = (pred.paraphrase or "").strip()
    if len(paraphrase) < 20:
        return False, "paraphrase too short"

    qs = pred.hypothetical_questions or []
    if not isinstance(qs, list) or len(qs) < 3:
        return False, f"need ≥3 hypothetical_questions, got {len(qs)}"

    sits = pred.life_situations or []
    if not isinstance(sits, list) or len(sits) < 2:
        return False, f"need ≥2 life_situations, got {len(sits)}"

    emos = pred.emotions_addressed or []
    if not isinstance(emos, list) or not emos:
        return False, "emotions_addressed empty"
    bad = [e for e in emos if _normalize_theme(e) not in EMOTION_VOCAB]
    if bad:
        return False, f"emotions outside vocabulary: {bad}"

    themes = pred.themes or []
    if not isinstance(themes, list) or not themes:
        return False, "themes empty"

    return True, ""


# ──────────────────────────── Module ────────────────────────────
class Enricher(dspy.Module):
    """Wraps the EnrichVerse signature with retries and post-processing.

    Why ChainOfThought over Predict
    -------------------------------
    GEPA may eventually optimize this prompt too, and ChainOfThought gives it
    a `reasoning` trace to inspect during reflection. The cost is one extra
    paragraph of LM output per call, which is negligible at our scale.
    """

    def __init__(self, max_retries: int = 2):
        super().__init__()
        self.predict = dspy.ChainOfThought(EnrichVerse)
        self.max_retries = max_retries

    def forward(self, verse: Verse) -> EnrichedVerse:
        attempt = 0
        last_err = ""
        pred = None

        while attempt <= self.max_retries:
            try:
                pred = self.predict(
                    verse_ref=verse.verse_ref,
                    sanskrit=verse.sanskrit or "",
                    translation=verse.translation or "",
                    bhashya=verse.bhashya or "",
                )
                ok, reason = _validate(pred)
                if ok:
                    break
                last_err = reason
            except Exception as e:
                last_err = f"LM error: {e}"
            attempt += 1

        # Build the EnrichedVerse from the Verse + whatever we got
        base = asdict(verse)
        ev = EnrichedVerse(**base)

        if pred and not last_err:
            ev.paraphrase = (pred.paraphrase or "").strip()
            ev.practical_teaching = (pred.practical_teaching or "").strip()
            ev.themes = [
                _normalize_theme(t) for t in (pred.themes or [])
                if THEME_KEY_RX.match(_normalize_theme(t))
            ]
            ev.life_situations = [
                s.strip() for s in (pred.life_situations or [])
                if s and len(s.strip()) >= 5
            ]
            ev.emotions_addressed = [
                _normalize_theme(e) for e in (pred.emotions_addressed or [])
                if _normalize_theme(e) in EMOTION_VOCAB
            ]
            ev.hypothetical_questions = [
                q.strip() for q in (pred.hypothetical_questions or [])
                if q and len(q.strip()) >= 10
            ][:5]  # cap at 5

            # Stamp the model so re-runs after a model swap can be detected
            try:
                lm = dspy.settings.lm
                ev.enrichment_model = getattr(lm, "model", "") or ""
            except Exception:
                pass
        else:
            # Enrichment failed; keep the verse but mark it
            ev.enrichment_model = f"FAILED: {last_err}"

        return ev
