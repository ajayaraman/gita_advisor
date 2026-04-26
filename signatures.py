"""
signatures.py — DSPy signatures for the advisor pipeline.

Each signature is a small, focused contract. GEPA will rewrite the docstrings
("instructions") and the field descriptions during optimization, which is why
the *initial* prompts here are deliberately minimal — they're starting points,
not finished prompts. We give just enough scaffolding to make the very first
forward pass coherent. GEPA does the rest.
"""

from __future__ import annotations
from typing import Literal
import dspy


# ──────────────────────────── Stage 1: Understanding ────────────────────────────
class UnderstandQuery(dspy.Signature):
    """Read the user's life situation carefully. Identify the felt emotion,
    the underlying spiritual concern (not just the surface complaint), and the
    Vedāntic themes that are most relevant — drawing only from concepts native
    to Advaita Vedānta."""

    user_question: str = dspy.InputField(desc="The user's message; may be a question, a vent, or a description of a situation.")

    felt_emotion: str = dspy.OutputField(
        desc="The dominant emotion the user is experiencing, named precisely (e.g. 'anticipatory grief', not just 'sad')."
    )
    surface_concern: str = dspy.OutputField(
        desc="What the user is literally asking about, in one sentence."
    )
    deeper_concern: str = dspy.OutputField(
        desc="The underlying existential/spiritual concern — usually about identity, attachment, "
             "fear, dharma, or meaning — that the surface concern is a symptom of. One sentence."
    )
    vedantic_themes: list[str] = dspy.OutputField(
        desc="2-4 Advaita-Vedānta concepts most relevant to this situation. "
             "Use Sanskrit terms with brief gloss, e.g. 'adhyāsa (superimposition of self onto roles)', "
             "'vairāgya (dispassion)', 'sākṣī (witness consciousness)'."
    )


# ──────────────────────────── Stage 2: Retrieval Planning ────────────────────────────
class PlanRetrieval(dspy.Signature):
    """Given the user's situation and identified themes, generate diverse search
    queries to find relevant passages from the Advaita corpus (Bhagavad Gītā with
    Śaṅkara bhāṣya, Upaniṣads, Brahma Sūtras, prakaraṇa texts). Each query should
    target a different angle — one query about the philosophical principle,
    one about a parallel situation in the texts, one about the practical
    teaching offered by the lineage."""

    surface_concern: str = dspy.InputField()
    deeper_concern: str = dspy.InputField()
    vedantic_themes: list[str] = dspy.InputField()

    queries: list[str] = dspy.OutputField(
        desc="3 distinct search queries (each 5-15 words). Vary in angle: principle, parallel, practice."
    )


# ──────────────────────────── Stage 3: Source Selection ────────────────────────────
class SelectPassages(dspy.Signature):
    """From the retrieved candidate passages, select the ones that genuinely
    speak to this user's situation. Prefer primary sources (Gītā verses,
    Upaniṣadic mantras, Śaṅkara's bhāṣya) over secondary or modern commentary
    when both are available. Reject passages that are merely topically adjacent
    but don't address the actual spiritual concern."""

    deeper_concern: str = dspy.InputField()
    candidate_passages: str = dspy.InputField(
        desc="Numbered candidate passages with source attribution."
    )

    selected_indices: list[int] = dspy.OutputField(
        desc="Indices (1-based) of the 2-4 most relevant passages."
    )
    selection_rationale: str = dspy.OutputField(
        desc="One sentence per selection explaining why that passage speaks to this concern."
    )


# ──────────────────────────── Stage 4: Advice Synthesis ────────────────────────────
class SynthesizeAdvice(dspy.Signature):
    """Compose a response that is grounded in Advaita Vedānta as taught by
    Śaṅkarācārya, empathetic to the user's felt experience, and practically
    useful for their situation. Honor the two-truths distinction: meet the user
    in vyāvahārika (transactional reality) without ever denying the
    pāramārthika (absolute) view. Cite specific verses/passages by reference,
    integrate them into prose rather than dumping quotes, and keep wit gentle —
    light around the cosmic predicament, never light about the user's pain."""

    user_question: str = dspy.InputField()
    felt_emotion: str = dspy.InputField()
    deeper_concern: str = dspy.InputField()
    selected_passages: str = dspy.InputField(
        desc="The selected passages with full source attribution."
    )

    response: str = dspy.OutputField(
        desc="The advisor's reply to the user. 250-450 words. "
             "Open by acknowledging the felt experience. Move into the Vedāntic perspective. "
             "Cite at least one primary source (Gītā chapter:verse, Upaniṣad name + section, etc.). "
             "Close with a concrete practice or shift in perspective they can try this week. "
             "Address the user as 'you' throughout. Avoid Western therapy clichés."
    )
    sources_cited: list[str] = dspy.OutputField(
        desc="Source references actually cited in the response, e.g. 'BG 2.47', 'Bṛhadāraṇyaka Up. 4.4.5', 'Vivekacūḍāmaṇi 11'."
    )
