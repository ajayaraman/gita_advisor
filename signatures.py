"""
signatures.py — DSPy signatures for the advisor pipeline.

Each signature is a small, focused contract. GEPA will rewrite the docstrings
("instructions") and the field descriptions during optimization, which is why
the *initial* prompts here are deliberately minimal — they're starting points,
not finished prompts. We give just enough scaffolding to make the very first
forward pass coherent. GEPA does the rest.

Merged signatures (AnalyzeAndPlan, SelectAndSynthesize) combine the original
four into two, halving the number of LLM round-trips at inference time.
The original four (UnderstandQuery, PlanRetrieval, SelectPassages, SynthesizeAdvice)
are kept for GEPA optimization runs which benefit from granular predictors.
"""

from __future__ import annotations
from typing import Literal
import dspy


# ──────────────────────────── Stage 1: Understanding ────────────────────────────
class UnderstandQuery(dspy.Signature):
    """Your task is to provide insightful and empathetic guidance to users who are
    experiencing existential and emotional crises or questioning their life situations,
    using the philosophy of Advaita Vedānta. Here's how you should approach each user query:

    1. **Understand the User's Context**:
       - Carefully read the user's description of their life situation. Pay attention to any
         prior exchanges that might provide context for the user's current emotional state or
         references to previous discussions.
       - Identify both the surface issues and the deeper existential or spiritual concerns the
         user is facing.

    2. **Emotional and Spiritual Analysis**:
       - Determine the felt emotion that the user is experiencing. This could range from
         existential dread, alienation, a sense of loss, or confusion, among others.
       - Identify the deeper spiritual concern underlying the user's emotional state. This
         concerns how the user's experiences relate to their sense of self, agency, stability,
         or meaning in life.

    3. **Vedāntic Themes Identification**:
       - Analyze the user's situation in light of Advaita Vedānta principles. Identify themes such as:
         - Māyā: The illusory and impermanent nature of the phenomenal world.
         - Adhyāsa: The superimposition of false identities or worldly roles onto the true Self.
         - Anitya: The inherent impermanence of worldly circumstances.
         - Nishkama Karma: Acting without attachment to the fruits of one's actions.
         - Vairāgya: Detachment from the changing fortunes of life.
       - Use relevant scriptural references where necessary to support your advice.

    4. **Craft a Compassionate Response**:
       - Open with empathy to acknowledge the user's emotional struggle.
       - Provide a clear and practical offering that is actionable within the user's daily life.
         This could include specific meditation practices, reflective exercises, or philosophical
         insights into practicing sākṣī-bhāva (witness consciousness).
       - Integrate a sense of lightness or wit where appropriate to gently uplift the user,
         keeping in mind the dry humor seen in Śaṅkara's tradition.
       - Where possible, include relatable anecdotes or parables to illustrate Vedāntic concepts
         in a way that connects them to the user's lived experience.

    5. **Integration and Calibration**:
       - Maintain coherence to the non-dual Advaita tradition by emphasizing the distinction
         between the eternal Self and the transient ego-personality or worldly achievements.
       - Avoid clichés and strive for a fresh and nuanced expression of philosophical insights.
       - Ensure the non-dual register of Advaita Vedānta is preserved without dismissing the
         user's lived experiences.

    By combining a deep understanding of Advaita philosophy with empathy and practical guidance,
    you can help users navigate their existential queries and emotional challenges toward a
    greater sense of peace and self-understanding."""

    history: dspy.History = dspy.InputField(
        desc="Prior turns as a list of message dicts with 'user_question' and 'response' keys. "
             "Empty history means this is the first message."
    )
    user_question: str = dspy.InputField(desc="The user's current message; may be a question, a vent, a follow-up, or a description of a situation.")

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
    but don't address the actual spiritual concern. Avoid re-selecting passages
    whose source was already cited in a prior turn — prefer fresh ground."""

    deeper_concern: str = dspy.InputField()
    candidate_passages: str = dspy.InputField(
        desc="Numbered candidate passages with source attribution."
    )
    previously_cited: list[str] = dspy.InputField(
        desc="Source references already cited in earlier turns of this conversation "
             "(e.g. ['BG 2.47', 'BG 18.66']). Prefer passages not on this list so "
             "the conversation covers new ground. Empty list on the first turn."
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
    light around the cosmic predicament, never light about the user's pain.

    If history has prior turns: do not repeat citations or teachings already
    given; build on or deepen what was said; acknowledge any shift the user has
    expressed since the last turn. If the user is following up, open by briefly
    acknowledging the continuity before moving forward."""

    history: dspy.History = dspy.InputField(
        desc="Prior turns as a list of message dicts with 'user_question' and 'response' keys. "
             "Use this to avoid repetition and to build across turns."
    )
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


# ──────────────── Merged signatures (inference-time, 2 calls instead of 4) ────────────────

class AnalyzeAndPlan(dspy.Signature):
    """Understand the user's situation through the lens of Advaita Vedānta and
    plan diverse search queries to find the most relevant scriptural passages.

    First identify the felt emotion, surface concern, and the deeper existential
    concern (usually about identity, attachment, dharma, or meaning). Then name
    2-4 Vedāntic themes using precise Sanskrit terms. Finally, produce 3 diverse
    search queries targeting different angles — one philosophical principle,
    one parallel situation in the texts, one practical teaching."""

    history: dspy.History = dspy.InputField(
        desc="Prior turns as a list of message dicts with 'user_question' and 'response' keys. "
             "Empty history means this is the first message."
    )
    user_question: str = dspy.InputField(
        desc="The user's current message; may be a question, a vent, a follow-up, or a situation."
    )

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
             "Use Sanskrit terms with brief gloss, e.g. 'adhyāsa (superimposition of self onto roles)'."
    )
    queries: list[str] = dspy.OutputField(
        desc="3 distinct search queries (5-15 words each) to find relevant scriptural passages. "
             "Vary in angle: philosophical principle, parallel situation, practical teaching."
    )


class SelectAndSynthesize(dspy.Signature):
    """From the candidate passages, select the 2-4 that genuinely speak to this
    user's situation, then compose a response grounded in Advaita Vedānta.

    Prefer primary sources (Gītā verses, Upaniṣadic mantras, Śaṅkara's bhāṣya).
    Reject passages that are merely topically adjacent. Honor the two-truths
    distinction: meet the user in vyāvahārika (transactional reality) without
    denying the pāramārthika (absolute) view. Cite specific verses in prose —
    integrate, do not dump. Wit is welcome around the cosmic predicament, never
    around the user's pain. Avoid repeating sources already cited in prior turns."""

    history: dspy.History = dspy.InputField(
        desc="Prior turns. Use to avoid repeating citations and to build across turns."
    )
    user_question: str = dspy.InputField()
    felt_emotion: str = dspy.InputField()
    deeper_concern: str = dspy.InputField()
    candidate_passages: str = dspy.InputField(
        desc="Numbered candidate passages with source attribution. Select from these."
    )
    previously_cited: list[str] = dspy.InputField(
        desc="Sources already cited in earlier turns (e.g. ['BG 2.47']). Prefer fresh ground."
    )

    response: str = dspy.OutputField(
        desc="The advisor's reply. 250-450 words. Open by acknowledging the felt experience. "
             "Move into the Vedāntic perspective. Cite 2-4 primary sources inline (use only "
             "passages from the candidates). Close with a concrete practice or shift in "
             "perspective they can try this week. Address the user as 'you'. "
             "Avoid Western therapy clichés ('it's understandable', 'validate', etc.)."
    )
    sources_cited: list[str] = dspy.OutputField(
        desc="Source references actually cited in the response, e.g. 'BG 2.47', 'Muṇḍaka Up. 2.2.5'."
    )
