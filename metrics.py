"""
metrics.py — the metric is the specification.

GEPA optimizes whatever the metric rewards. So the metric here is not a single
number; it's a *contract* on what an Advaita-grounded, empathetic, practically
useful response looks like — combined with rich textual feedback the reflection
LM uses to rewrite prompts.

We combine three signals:
    1. Rule-based checks (fast, deterministic)
        - citation grounding (cites real retrieved sources, not hallucinated)
        - tier preference (primary + Śaṅkara > supporting)
        - structural hygiene (length, has actionable element, no therapy clichés)
    2. LLM-as-judge rubric scoring
        - Advaita coherence (non-dual, not crypto-dualist)
        - two-truths discipline (vyāvahārika ↔ pāramārthika)
        - empathy without dissolving into the user's frame
        - wit calibration (light around the predicament, never the pain)
    3. Composite score + structured feedback string

The function signature matches GEPA's metric contract:
    metric(gold, pred, trace=None, pred_name=None, pred_trace=None) -> dspy.Prediction

Returning dspy.Prediction(score=float, feedback=str) is the GEPA happy path.
"""

from __future__ import annotations
import re
import json
from typing import Any
import dspy


# ──────────────────────────── Rule-based checks ────────────────────────────
THERAPY_CLICHES = [
    "you got this",
    "be kind to yourself",
    "self-care",
    "just remember",
    "trust the process",
    "everything happens for a reason",
    "you are enough",
    "love and light",
    "manifesting",
    "send positive vibes",
    "good vibes",
]

# Loose pattern catching citations like "BG 2.47", "Gītā 18.66", "Bṛhadāraṇyaka 4.4.5",
# "Vivekacūḍāmaṇi 11", "Kaṭha Up. 1.3.14", etc.
CITATION_PATTERN = re.compile(
    r"\b("
    r"BG\s*\d+[\.:]\d+"                                  # BG 2.47
    r"|G[īi]t[āa]\s*\d+[\.:]\d+"                         # Gita 2.47
    r"|[A-ZĀĪŪṚḌṬṆṢŚḤṂa-zāīūṛḍṭṇṣśḥṃ]{3,}\s*Up\.?\s*\d+(?:[\.:]\d+){0,2}"  # Kaṭha Up. 1.2.3
    r"|Vivekac[ūu]ḍāmaṇi\s*\d+"
    r"|Ātmabodha\s*\d+"
    r"|Tattvabodha\s*\d+"
    r"|Brahma\s*S[ūu]tra\s*\d+[\.:]\d+(?:[\.:]\d+)?"
    r"|Aṣṭāvakra\s*G[īi]t[āa]\s*\d+[\.:]\d+"
    r")\b"
)

EMPATHY_OPENERS = [
    "what you", "you're carrying", "you are carrying", "i hear",
    "this hurts", "this is painful", "the weight", "sitting with",
    "what you describe", "the ache",
]

ACTIONABLE_MARKERS = [
    "this week", "today", "try this", "begin by", "for the next",
    "each morning", "each evening", "when you notice", "the next time",
    "as a practice", "sit for", "spend ", "over the next",
]

NON_DUAL_MARKERS = [
    "witness", "sākṣī", "sakshi", "non-dual", "advaita",
    "pāramārthika", "paramarthika", "vyāvahārika", "vyavaharika",
    "ātman", "atman", "brahman", "adhyāsa", "adhyasa", "māyā", "maya",
    "neti neti", "tat tvam asi", "ahaṁ brahmāsmi", "aham brahmasmi",
    "self with a capital", "the seer", "awareness itself",
]


def _word_count(s: str) -> int:
    return len(s.split())


def _has_any(text: str, needles: list[str]) -> list[str]:
    low = text.lower()
    return [n for n in needles if n in low]


def _normalize_for_match(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower()).strip()


def _citation_grounding(
    sources_cited: list[str],
    retrieved_passages: list[dict],
) -> tuple[float, list[str], list[str]]:
    """Return (grounding_score, grounded_citations, ungrounded_citations).

    With the verse-indexed corpus, each retrieved passage carries an exact
    verse_ref string ('BG 2.47', 'Muṇḍaka Up. 2.1.3', etc.). Grounding becomes
    an exact set-membership test rather than fuzzy substring matching, which
    is dramatically sharper feedback for GEPA's reflection step: 'BG 2.47'
    is grounded if and only if 'BG 2.47' was in the retrieved set.

    We still tolerate light formatting noise: the synthesizer might write
    'BG 2.47', 'Bhagavad Gītā 2.47', 'Gita 2:47', etc. We canonicalize to
    'BG <chap>.<verse>' for Gītā citations before comparing. Other works
    are matched directly by verse_ref string with whitespace normalized.
    """
    if not sources_cited:
        return 0.0, [], []

    retrieved_refs = {
        _canonicalize_ref(h.get("verse_ref") or h.get("meta", {}).get("verse_ref", ""))
        for h in retrieved_passages
    }
    retrieved_refs.discard("")

    grounded, ungrounded = [], []
    for c in sources_cited:
        canon = _canonicalize_ref(c)
        # Try direct match first, then a "substring of any retrieved" fallback
        # for cases where the synthesizer paraphrases the citation
        # ('chapter 2 verse 47' vs 'BG 2.47').
        hit = canon in retrieved_refs or any(
            canon and (canon in r or r in canon) for r in retrieved_refs
        )
        (grounded if hit else ungrounded).append(c)

    score = len(grounded) / max(len(sources_cited), 1)
    return score, grounded, ungrounded


def _canonicalize_ref(s: str) -> str:
    """Normalize a citation string so 'BG 2.47', 'Bhagavad Gītā 2.47',
    'Gītā 2:47' all reduce to the same canonical form 'BG 2.47'."""
    s = re.sub(r"\s+", " ", s.strip())
    # Gītā variants
    m = re.match(r"^(?:BG|Bhagavad\s*G[īi]t[āa]|G[īi]t[āa])\s*(\d+)[\.:](\d+)", s, re.I)
    if m:
        return f"BG {int(m.group(1))}.{int(m.group(2))}"
    # Default: lowercased, colons → dots
    return s.lower().replace(":", ".")


def _tier_preference(
    sources_cited: list[str],
    retrieved_passages: list[dict],
    selected_indices: list[int],
) -> tuple[float, dict]:
    """Reward responses whose *cited* passages came from primary/Śaṅkara tiers."""
    if not selected_indices:
        return 0.0, {"primary": 0, "shankara": 0, "supporting": 0}

    counts = {"primary": 0, "shankara": 0, "supporting": 0}
    for idx in selected_indices:
        if 1 <= idx <= len(retrieved_passages):
            tier = retrieved_passages[idx - 1].get("meta", {}).get("tier", "supporting")
            counts[tier] = counts.get(tier, 0) + 1

    total = sum(counts.values()) or 1
    preferred = counts["primary"] + counts["shankara"]
    return preferred / total, counts


def rule_based_score(pred: dspy.Prediction) -> tuple[float, dict]:
    """Returns (score in [0,1], breakdown dict)."""
    response = getattr(pred, "response", "") or ""
    sources_cited = getattr(pred, "sources_cited", []) or []
    retrieved = getattr(pred, "retrieved_passages", []) or []
    selected_idx = getattr(pred, "selected_indices", []) or []
    felt = getattr(pred, "felt_emotion", "") or ""

    wc = _word_count(response)
    length_ok = 200 <= wc <= 600
    length_score = 1.0 if length_ok else max(0.0, 1.0 - abs(wc - 350) / 350)

    citations_in_text = CITATION_PATTERN.findall(response)
    has_citation = bool(citations_in_text) or bool(sources_cited)
    citation_score = 1.0 if has_citation else 0.0

    grounding_score, grounded, ungrounded = _citation_grounding(sources_cited, retrieved)

    tier_score, tier_counts = _tier_preference(sources_cited, retrieved, selected_idx)

    cliches = _has_any(response, THERAPY_CLICHES)
    cliche_penalty = min(1.0, 0.25 * len(cliches))
    cliche_score = 1.0 - cliche_penalty

    # Empathy: opening should signal acknowledgement of feeling
    head = response[:300].lower()
    empathy_hits = [m for m in EMPATHY_OPENERS if m in head]
    # Bonus if the felt_emotion content is referenced (loosely)
    if felt:
        for tok in felt.lower().split():
            if len(tok) > 4 and tok in head:
                empathy_hits.append(f"echoes:{tok}")
                break
    empathy_score = min(1.0, 0.4 + 0.3 * len(empathy_hits))

    actionable_hits = _has_any(response, ACTIONABLE_MARKERS)
    actionable_score = 1.0 if actionable_hits else 0.4

    nondual_hits = _has_any(response, NON_DUAL_MARKERS)
    nondual_score = min(1.0, 0.4 + 0.2 * len(nondual_hits))

    # Weighted aggregate
    components = {
        "length": (length_score, 0.05),
        "citation_present": (citation_score, 0.08),
        "citation_grounding": (grounding_score, 0.18),
        "tier_preference": (tier_score, 0.12),
        "no_cliches": (cliche_score, 0.10),
        "empathy_opening": (empathy_score, 0.15),
        "actionable": (actionable_score, 0.10),
        "nondual_register": (nondual_score, 0.22),
    }
    score = sum(s * w for s, w in components.values())

    breakdown = {
        "score": score,
        "word_count": wc,
        "components": {k: round(v[0], 3) for k, v in components.items()},
        "citations_in_text": citations_in_text,
        "sources_cited": sources_cited,
        "grounded_citations": grounded,
        "ungrounded_citations": ungrounded,
        "tier_counts": tier_counts,
        "therapy_cliches_found": cliches,
        "empathy_hits": empathy_hits,
        "actionable_hits": actionable_hits,
        "nondual_markers_found": nondual_hits,
    }
    return score, breakdown


# ──────────────────────────── LLM-judge rubric ────────────────────────────
class JudgeAdvice(dspy.Signature):
    """You are an examiner of Advaita-Vedānta spiritual counsel in the lineage
    of Ādi Śaṅkarācārya. Score the advisor's response against the user's
    question on each rubric (0.0 to 1.0) and write a short critique that an
    optimizer can use to *improve the prompts that produced this response*.

    Rubrics:

    - advaita_coherence: Does the response reflect genuine non-dualism
      (jīva-ātman-brahman identity), or does it accidentally smuggle in dualism
      ('the soul reaches God', 'becoming one with the universe' as if they were
      separate, etc.)? Does it avoid collapsing into nihilism ('nothing is
      real')?

    - two_truths_discipline: Does it honor the distinction between
      vyāvahārika (transactional, where the user's pain and choices are real
      and matter) and pāramārthika (absolute, where the witness is untouched)?
      Failure modes: spiritual bypass (denying the pain by pointing to the
      absolute), or pure-therapy register (forgetting the absolute exists).

    - empathy_without_dissolving: Does it meet the user in their felt
      experience without either flattening into therapy-speak OR dismissing
      the feeling with premature transcendence?

    - wit_calibration: Is there a light, dry touch around the cosmic
      predicament (Śaṅkara himself is dry; this is consistent with the
      tradition) WITHOUT being flippant about the user's actual pain? Both
      'too solemn throughout' and 'making jokes about their situation' lose
      points.

    - source_integration: Are scriptural citations woven into the prose
      (illuminating the point) rather than dumped as block quotes or used
      as decoration? Are the references specific (Gītā 2.47, not just
      "the Gita says")?

    - practical_offering: Does the response close with something the user
      can actually try — a question to sit with, a practice, a perspective
      shift — rather than abstract platitudes?

    - draw_from_personal_experiences: Does the response use parables and day to day life
      stories as examples to encourage the user to relate better to the advise
    
    The critique should be specific and prescriptive: what to keep, what to
    cut, what's missing. Phrase it as you would to a writer revising a draft."""

    user_question: str = dspy.InputField()
    response: str = dspy.InputField()
    sources_cited: list[str] = dspy.InputField()

    advaita_coherence: float = dspy.OutputField(desc="0.0 to 1.0")
    two_truths_discipline: float = dspy.OutputField(desc="0.0 to 1.0")
    empathy_without_dissolving: float = dspy.OutputField(desc="0.0 to 1.0")
    wit_calibration: float = dspy.OutputField(desc="0.0 to 1.0")
    source_integration: float = dspy.OutputField(desc="0.0 to 1.0")
    practical_offering: float = dspy.OutputField(desc="0.0 to 1.0")
    draw_from_personal_experiences: float = dspy.OutputField(desc="0.0 to 1.0")
    critique: str = dspy.OutputField(
        desc="3-6 sentences of prescriptive feedback for revising the response."
    )


# Lazily-instantiated judge so we can swap in a stronger LM if available
_judge = None
def _get_judge():
    global _judge
    if _judge is None:
        _judge = dspy.ChainOfThought(JudgeAdvice)
    return _judge


def judge_score(user_question: str, pred: dspy.Prediction) -> tuple[float, dict, str]:
    judge = _get_judge()
    try:
        j = judge(
            user_question=user_question,
            response=getattr(pred, "response", "") or "",
            sources_cited=getattr(pred, "sources_cited", []) or [],
        )
    except Exception as e:
        # If the judge fails (parse error, LM hiccup), fall back gracefully.
        return 0.5, {"judge_error": str(e)}, f"Judge failed: {e}"

    rubric = {
        "advaita_coherence": float(j.advaita_coherence or 0.0),
        "two_truths_discipline": float(j.two_truths_discipline or 0.0),
        "empathy_without_dissolving": float(j.empathy_without_dissolving or 0.0),
        "wit_calibration": float(j.wit_calibration or 0.0),
        "source_integration": float(j.source_integration or 0.0),
        "practical_offering": float(j.practical_offering or 0.0),
        "draw_from_personal_experiences": float(j.draw_from_personal_experiences or 0.0),
    }
    weights = {
        "advaita_coherence": 0.25,
        "two_truths_discipline": 0.20,
        "empathy_without_dissolving": 0.20,
        "wit_calibration": 0.10,
        "source_integration": 0.10,
        "practical_offering": 0.10,
        "draw_from_personal_experiences": 0.05,
    }
    score = sum(rubric[k] * weights[k] for k in rubric)
    score = max(0.0, min(1.0, score))
    return score, rubric, j.critique or ""


# ──────────────────────────── Composite GEPA metric ────────────────────────────
RULE_WEIGHT = 0.45
JUDGE_WEIGHT = 0.55


def _format_feedback(rule_breakdown: dict, judge_rubric: dict, critique: str) -> str:
    """Concatenate rule-based facts and judge critique into one feedback string
    that the GEPA reflection LM can read and use to rewrite prompts."""
    lines = ["FEEDBACK FOR PROMPT IMPROVEMENT", ""]

    lines.append("Rule-based observations:")
    comps = rule_breakdown.get("components", {})
    for k, v in comps.items():
        lines.append(f"  - {k}: {v}")
    if rule_breakdown.get("therapy_cliches_found"):
        lines.append(f"  - Therapy clichés to remove: {rule_breakdown['therapy_cliches_found']}")
    if rule_breakdown.get("ungrounded_citations"):
        lines.append(
            f"  - Citations that weren't in retrieved passages (likely hallucinated): "
            f"{rule_breakdown['ungrounded_citations']}"
        )
    if not rule_breakdown.get("nondual_markers_found"):
        lines.append("  - Response lacks explicit Advaita register; consider invoking "
                     "concepts like sākṣī, adhyāsa, the two truths, etc.")
    if not rule_breakdown.get("actionable_hits"):
        lines.append("  - No concrete practice or this-week shift was offered.")
    tier_counts = rule_breakdown.get("tier_counts", {})
    if tier_counts:
        lines.append(f"  - Selected passage tiers: {tier_counts} "
                     f"(prefer primary + śaṅkara when both options exist).")

    lines.append("")
    lines.append("Rubric scores from Advaita-tradition examiner:")
    for k, v in judge_rubric.items():
        if isinstance(v, float):
            lines.append(f"  - {k}: {v:.2f}")
    lines.append("")
    lines.append("Examiner critique:")
    lines.append(critique.strip() or "(no critique returned)")
    return "\n".join(lines)


def gita_metric(
    gold: dspy.Example,
    pred: dspy.Prediction,
    trace: Any = None,
    pred_name: str | None = None,
    pred_trace: Any = None,
) -> dspy.Prediction:
    """The GEPA-compatible metric.

    Returns dspy.Prediction(score=..., feedback=...). The feedback string is
    what GEPA's reflection LM ingests when rewriting prompts."""
    user_q = getattr(gold, "user_question", "") if gold else ""

    rule_score, rule_breakdown = rule_based_score(pred)
    j_score, j_rubric, critique = judge_score(user_q, pred)

    composite = RULE_WEIGHT * rule_score + JUDGE_WEIGHT * j_score
    feedback = _format_feedback(rule_breakdown, j_rubric, critique)

    return dspy.Prediction(score=composite, feedback=feedback)


def quick_eval_score(
    gold: dspy.Example,
    pred: dspy.Prediction,
    trace: Any = None,
) -> float:
    """A pure-float metric for `dspy.Evaluate` — same composite, no feedback."""
    out = gita_metric(gold, pred, trace=trace)
    return float(out.score)
