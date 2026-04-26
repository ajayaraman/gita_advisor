"""
dataset_generator.py — produce ~500 unique, life-grounded questions.

The dataset is the GEPA training/validation pool. We want:
  - Coverage across life domains (career, grief, identity, dharma, practice, ...)
  - Variety in voice (anguished / intellectual / sarcastic / exhausted / hopeful)
  - Variety in form (direct question / vent / philosophical doubt / dilemma)
  - Variety in age & life-stage cues
  - Some cleanly Advaita-relevant, some that *force* the advisor to find the
    Advaita angle in something mundane (this is where over-fitting to "spiritual"
    questions usually shows up)

Strategy: structured combinatorics × LM rewriting × similarity dedupe.

We construct (domain, scenario, voice, form) tuples, send them to the local LM
to write each as a real human message, then dedupe by embedding similarity.
"""

from __future__ import annotations
import argparse
import json
import random
import re
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import dspy

import config


# ──────────────────────────── Taxonomy ────────────────────────────
DOMAINS: dict[str, list[str]] = {
    "career_and_purpose": [
        "got laid off after years of dedication",
        "achieved the big career goal and feels empty",
        "stuck in a job that pays well but feels meaningless",
        "wants to leave stable career to pursue art / spiritual path",
        "watching peers succeed while their own work plateaus",
        "facing retirement and loss of identity tied to work",
        "imposter syndrome after a major promotion",
        "publicly failed in front of colleagues",
    ],
    "romantic_relationships": [
        "going through a painful breakup after long relationship",
        "marriage has gone cold and considering divorce",
        "in love with someone who doesn't love them back",
        "obsessive jealousy about a partner's past",
        "tempted to have an affair",
        "partner died and grief is overwhelming",
        "afraid of commitment despite loving partner",
        "single in their 40s and despairing about it",
    ],
    "family": [
        "parent is dying and they have unresolved conflict",
        "estranged from a sibling for years",
        "parents pressuring them about marriage / career",
        "child making destructive life choices",
        "caring for an aging parent and exhausted",
        "had a falling out with adult child",
        "mother-in-law conflict ruining marriage",
        "feels they failed as a parent",
    ],
    "friendship_and_social": [
        "best friend betrayed their trust",
        "feels invisible and lonely in their 30s",
        "friend group has drifted apart with age",
        "social anxiety preventing them from connecting",
        "outgrown their old friends spiritually",
        "discovered close friend was talking behind their back",
    ],
    "mortality_and_loss": [
        "received a serious medical diagnosis",
        "watching a loved one die slowly",
        "afraid of death after a near-miss",
        "grieving a sudden, unexpected loss",
        "watching parents age and decline",
        "lost a child",
        "lost a pet who was their closest companion",
        "approaching old age with regret about unlived life",
    ],
    "identity_and_ego": [
        "tying self-worth entirely to external validation",
        "endlessly comparing themselves to others on social media",
        "going through midlife crisis questioning everything",
        "famous and feels everyone wants something from them",
        "lost sense of who they are after big life change",
        "racial / cultural identity feels splintered between worlds",
        "transitioning gender and family rejecting them",
    ],
    "material_life": [
        "drowning in debt and shame about it",
        "wealthy and feels guilty / disconnected because of it",
        "consumed by FOMO scrolling through richer friends' lives",
        "lost their home / financial security",
        "struggling to give up consumerist habits despite knowing better",
        "tempted by a get-rich-quick scheme",
    ],
    "existential": [
        "feels life has no meaning at all",
        "deeply depressed and going through the motions",
        "constant existential dread about the world's state",
        "doubting whether God / Brahman exists",
        "sees through everything and now nothing feels real",
        "feels they were 'born wrong' for this world",
    ],
    "spiritual_practice": [
        "meditation has gone dry after years of practice",
        "got addicted to spiritual highs and now they've stopped",
        "spiritual ego — feels superior to non-practitioners",
        "had a powerful experience and can't get back to it",
        "doubts whether their guru / lineage is right for them",
        "intellectually understands non-duality but doesn't feel it",
        "afraid that liberation means losing love for family",
        "can't reconcile traditional teachings with modern life",
    ],
    "ethics_and_dharma": [
        "told a serious lie and considering whether to confess",
        "harmed someone in the past and can't forgive themselves",
        "facing a moral dilemma at work involving dishonesty",
        "tempted to retaliate against someone who wronged them",
        "torn between duty to family and personal calling",
        "did something they're deeply ashamed of",
    ],
    "health_and_body": [
        "chronic illness reshaping their entire life",
        "struggling with addiction and relapse",
        "eating disorder they can't seem to escape",
        "chronic pain making spiritual practice feel impossible",
        "hates their aging body",
        "cancer diagnosis reframing everything",
    ],
    "modernity_specific": [
        "doomscrolling and feeling worse every day",
        "AI / automation making them feel obsolete",
        "climate dread paralyzing their life decisions",
        "political division has destroyed family relationships",
        "addicted to phone / can't focus / can't read books anymore",
        "online persona feels disconnected from real self",
    ],
}

VOICES = [
    "anguished",
    "exhausted",
    "intellectual and analytical",
    "darkly sarcastic",
    "quietly hopeful",
    "numb and dissociated",
    "frustrated and angry",
    "softly resigned",
]

FORMS = [
    "direct question",
    "venting paragraph",
    "philosophical doubt",
    "practical dilemma asking what to do",
    "stream-of-consciousness",
]

AGE_CUES = [
    "early 20s",
    "late 20s",
    "early 30s",
    "late 30s",
    "40s",
    "50s",
    "60s",
    "70s",
    "(no age cue)",
]


@dataclass
class QuestionRecord:
    id: str
    question: str
    domain: str
    scenario: str
    voice: str
    form: str
    age_cue: str


# ──────────────────────────── LM-driven phrasing ────────────────────────────
class WriteUserMessage(dspy.Signature):
    """Write a single, realistic message that a person might send to a spiritual
    advisor. The message must reflect the given scenario, voice, form, and age
    cue. Do NOT include scripture references, do NOT name Vedānta concepts —
    write as a real person speaking from their actual life. Avoid generic phrases
    like 'help me find peace' or 'I want to grow spiritually'. Be specific, lived,
    grounded in detail. 2-6 sentences."""

    scenario: str = dspy.InputField()
    voice: str = dspy.InputField()
    form: str = dspy.InputField()
    age_cue: str = dspy.InputField()

    message: str = dspy.OutputField(desc="The user's message, in first person.")


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")[:60]


def generate_questions(target_n: int = 500, seed: int = 7, use_local: bool = False) -> list[QuestionRecord]:
    """Generate ~target_n unique questions via combinatorics + LM rewriting."""
    rng = random.Random(seed)
    if use_local:
        config.configure_dspy()
    else:
        config.configure_enrich_lm()  # gpt-4o-mini: faster and more stylistically diverse
    writer = dspy.Predict(WriteUserMessage)

    # Build the (domain, scenario, voice, form, age) plan first
    combos: list[tuple[str, str, str, str, str]] = []
    for domain, scenarios in DOMAINS.items():
        for scenario in scenarios:
            # 5 variants per scenario varying voice/form/age
            voices = rng.sample(VOICES, k=5)
            forms = [rng.choice(FORMS) for _ in range(5)]
            ages = rng.sample(AGE_CUES, k=5)
            for v, f, a in zip(voices, forms, ages):
                combos.append((domain, scenario, v, f, a))

    rng.shuffle(combos)

    # Cap to a generous over-target; we'll dedupe down to target_n
    over_target = int(target_n * 1.25)
    combos = combos[:over_target]

    records: list[QuestionRecord] = []
    for i, (domain, scenario, voice, form, age) in enumerate(tqdm(combos, desc="Generating")):
        try:
            out = writer(scenario=scenario, voice=voice, form=form, age_cue=age)
            msg = (out.message or "").strip()
            if len(msg) < 30:
                continue
            records.append(QuestionRecord(
                id=f"q_{i:04d}_{_slug(domain)}",
                question=msg,
                domain=domain,
                scenario=scenario,
                voice=voice,
                form=form,
                age_cue=age,
            ))
        except Exception as e:
            # Local LMs occasionally hiccup. Log and continue.
            print(f"[warn] generation failure on combo {i}: {e}")
            continue

    return _dedupe_by_similarity(records, target_n=target_n)


def _dedupe_by_similarity(records: list[QuestionRecord], target_n: int, threshold: float = 0.92) -> list[QuestionRecord]:
    """Embed and remove near-duplicates greedily."""
    if not records:
        return records
    print(f"Deduping {len(records)} candidates ...")
    embedder = SentenceTransformer(config.EMBED_MODEL, device=config.EMBED_DEVICE)
    embs = embedder.encode(
        [r.question for r in records],
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=32,
    )
    keep_idx: list[int] = []
    kept_embs = []
    for i, e in enumerate(embs):
        if not kept_embs:
            keep_idx.append(i)
            kept_embs.append(e)
            continue
        sims = np.dot(np.stack(kept_embs), e)
        if float(sims.max()) < threshold:
            keep_idx.append(i)
            kept_embs.append(e)
        if len(keep_idx) >= target_n:
            break
    print(f"Kept {len(keep_idx)} after dedupe (target {target_n}).")
    return [records[i] for i in keep_idx]


def save_jsonl(records: list[QuestionRecord], path: Path):
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
    print(f"Wrote {len(records)} questions to {path}")


def load_jsonl(path: Path = config.DATASET_PATH) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def to_dspy_examples(records: list[dict]) -> list[dspy.Example]:
    """The dataset has no gold labels — that's fine. GEPA's metric uses LLM
    judgment + retrieval grounding rather than reference answers.
    We carry the metadata as inputs-of-record so the metric can use them."""
    out = []
    for r in records:
        ex = dspy.Example(
            user_question=r["question"],
            history=dspy.History(messages=[]),
            domain=r["domain"],
            scenario=r["scenario"],
        ).with_inputs("user_question", "history")
        out.append(ex)
    return out


# ──────────────────────────── CLI ────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=str, default=str(config.DATASET_PATH))
    ap.add_argument("--lm", choices=["openai", "local"], default="openai",
                    help="openai = gpt-4o-mini (default, faster); local = LM Studio task LM")
    args = ap.parse_args()

    records = generate_questions(target_n=args.n, seed=args.seed, use_local=(args.lm == "local"))
    save_jsonl(records, Path(args.out))


if __name__ == "__main__":
    main()
