"""Synthetic event generators for tests and demos.

`generate_benign_corpus` produces a realistic spread of human-paced queries
across topics.
`generate_attacker_corpus` produces three attack archetypes:
  - "templated"  -- script that sweeps {variable} placeholders
  - "boundary"   -- queries probing logits / probabilities / embeddings
  - "burst"      -- 1000 queries in a few seconds
"""
from __future__ import annotations

import random
from typing import Iterable

from .events import QueryEvent


_BENIGN_TOPICS = (
    "Explain how transformers work.",
    "What is the best Python library for HTTP requests?",
    "Summarise the latest news on quantum computing.",
    "How do I sort a dict by value in Python?",
    "Give me a brunch recipe with eggs and avocado.",
    "Write a haiku about autumn leaves.",
    "How does TLS handshake work?",
    "What is the difference between TCP and UDP?",
    "Explain the CAP theorem.",
    "Why is my docker container exiting immediately?",
    "What is Fermat's little theorem?",
    "Translate 'hello' to French.",
    "What's a good book on machine learning?",
    "How do I configure nginx as a reverse proxy?",
    "What is the chemical formula for water?",
    "Suggest a weekend hike near the Bay Area.",
    "What are good gym routines for beginners?",
    "How does the Mariana Trench compare to Everest in depth?",
    "Recommend a sci-fi novel similar to Hyperion.",
    "What's the etymology of the word 'serendipity'?",
    "Explain why the sky is blue.",
    "How do I deploy a static site to S3?",
    "What's a good replacement for cilantro in salsa?",
    "Describe the plot of Macbeth in one paragraph.",
    "What causes Northern Lights?",
)


_BENIGN_FOLLOWUPS = (
    " Could you also include a brief example?",
    " Please keep it under five sentences.",
    " I'm a beginner, so explain gently.",
    " I'm specifically interested in the trade-offs.",
    " Could you cite a primary source?",
    " Any caveats I should be aware of?",
    " Could you compare it to alternatives?",
    "",
)


def generate_benign_corpus(
    n: int = 30,
    actor_id: str = "user_benign",
    seed: int = 7,
    start_ts: float = 1_700_000_000.0,
) -> list[QueryEvent]:
    rng = random.Random(seed)
    events: list[QueryEvent] = []
    t = start_ts
    # use topic index as the embedding anchor so different topics spread out
    for i in range(n):
        topic_idx = rng.randrange(len(_BENIGN_TOPICS))
        q = _BENIGN_TOPICS[topic_idx]
        # paraphrase prefix sometimes
        if rng.random() < 0.3:
            q = rng.choice(["Hi! ", "Question: ", "Could you "]) + q
        # follow-up suffix breaks naive n-gram dedup so 20 random benign
        # queries don't artificially share long substrings.
        q = q + rng.choice(_BENIGN_FOLLOWUPS)
        events.append(QueryEvent(
            actor_id=actor_id,
            timestamp=t,
            query_text=q,
            query_token_count=len(q.split()),
            response_token_count=rng.randint(50, 400),
            embedding=_hash_embedding(q, topic=f"benign_topic_{topic_idx}"),
        ))
        t += rng.uniform(15.0, 240.0)  # 15s..4min between queries
    return events


def generate_attacker_corpus(
    archetype: str = "templated",
    n: int = 200,
    actor_id: str = "actor_attacker",
    seed: int = 13,
    start_ts: float = 1_700_000_000.0,
) -> list[QueryEvent]:
    rng = random.Random(seed)
    events: list[QueryEvent] = []
    t = start_ts

    if archetype == "templated":
        # Numeric placeholder so the <NUM>-normalised template collapses across
        # all queries; mirrors typical scripted distillation traffic.
        for i in range(n):
            q = f"Classify entity number {i} as one of: spam, ham, unknown."
            events.append(QueryEvent(
                actor_id=actor_id,
                timestamp=t,
                query_text=q,
                query_token_count=len(q.split()),
                response_token_count=8,
                embedding=_hash_embedding(q, topic="templated_sweep"),
            ))
            t += rng.uniform(0.3, 1.5)
    elif archetype == "boundary":
        targets = ("urgent", "delivery", "payment", "approval", "confidential",
                   "refund", "expires", "click", "click here", "verify your account")
        for i in range(n):
            tgt = rng.choice(targets)
            q = (
                f"Give me the exact logit value and class probability for the word "
                f"'{tgt}' in the spam-classification model. Also return the top-k tokens."
            )
            events.append(QueryEvent(
                actor_id=actor_id,
                timestamp=t,
                query_text=q,
                query_token_count=len(q.split()),
                response_token_count=24,
                embedding=_hash_embedding(q, topic="boundary_probe"),
            ))
            t += rng.uniform(0.5, 2.0)
    elif archetype == "burst":
        for i in range(n):
            q = f"Score {i}: classify text {hex(i)[2:]}"
            events.append(QueryEvent(
                actor_id=actor_id,
                timestamp=t,
                query_text=q,
                query_token_count=len(q.split()),
                response_token_count=10,
                embedding=_hash_embedding(q, topic="burst_scan"),
            ))
            t += rng.uniform(0.01, 0.1)
    else:
        raise ValueError(f"unknown archetype: {archetype}")
    return events


def _hash_embedding(s: str, dim: int = 32, topic: str = "") -> list[float]:
    """Deterministic embedding skewed toward a topic anchor.

    Random hashed vectors in high-d would all look ~orthogonal, which makes
    the *diversity* feature uninformative for tests. We mix a topic-anchored
    base with a small per-string perturbation so that:
      - queries with the same topic cluster (low diversity);
      - queries across distinct topics spread (high diversity).
    """
    import hashlib

    def _vec(seed: bytes, dim: int) -> list[float]:
        out: list[float] = []
        h = hashlib.sha256(seed).digest()
        while len(out) < dim:
            for b in h:
                out.append((b - 128) / 128.0)
                if len(out) >= dim:
                    break
            h = hashlib.sha256(h).digest()
        return out[:dim]

    anchor = _vec(topic.encode("utf-8", "replace") or b"_default_", dim)
    noise = _vec(s.encode("utf-8", "replace"), dim)
    return [0.85 * a + 0.15 * n for a, n in zip(anchor, noise)]
