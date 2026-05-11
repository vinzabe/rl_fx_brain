"""Feature extraction for an ActorWindow.

Outputs a fixed-length vector capturing volumetric, lexical, and embedding-
diversity properties used by the detector.

The feature set is deliberately interpretable so that any flag can be
explained back to the operator.
"""
from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, field
from typing import Sequence

from .events import ActorWindow


_WORD_RE = re.compile(r"[A-Za-z0-9']+")


def shannon_entropy_bytes(b: bytes) -> float:
    if not b:
        return 0.0
    counts = [0] * 256
    for x in b:
        counts[x] += 1
    h = 0.0
    n = len(b)
    for c in counts:
        if c:
            p = c / n
            h -= p * math.log2(p)
    return h


def char_entropy(s: str) -> float:
    return shannon_entropy_bytes(s.encode("utf-8", "replace"))


def tokenize_words(s: str) -> list[str]:
    return _WORD_RE.findall(s.lower())


def interarrival_stats(ts: Sequence[float]) -> tuple[float, float, float]:
    """Return (mean_dt, std_dt, p10_dt) in seconds."""
    if len(ts) < 2:
        return 0.0, 0.0, 0.0
    dts = [max(0.0, ts[i + 1] - ts[i]) for i in range(len(ts) - 1)]
    mean = sum(dts) / len(dts)
    var = sum((d - mean) ** 2 for d in dts) / len(dts)
    sd = math.sqrt(var)
    p10 = sorted(dts)[max(0, len(dts) // 10 - 1)]
    return mean, sd, p10


def jaccard_ngrams(s1: str, s2: str, n: int = 4) -> float:
    g1 = {s1[i : i + n] for i in range(len(s1) - n + 1)}
    g2 = {s2[i : i + n] for i in range(len(s2) - n + 1)}
    if not g1 and not g2:
        return 1.0
    if not g1 or not g2:
        return 0.0
    inter = len(g1 & g2)
    union = len(g1 | g2)
    return inter / union if union else 0.0


def near_duplicate_ratio(queries: list[str], threshold: float = 0.7) -> float:
    """Fraction of queries that have ≥1 near-duplicate among the rest.

    Approximate: for windows >50 we subsample to keep this O(n*k).
    """
    if len(queries) < 2:
        return 0.0
    sample = queries if len(queries) <= 50 else queries[:25] + queries[-25:]
    hits = 0
    for i in range(len(sample)):
        for j in range(len(sample)):
            if i == j:
                continue
            if jaccard_ngrams(sample[i], sample[j]) >= threshold:
                hits += 1
                break
    return hits / len(sample)


def coefficient_of_variation(xs: Sequence[float]) -> float:
    if not xs:
        return 0.0
    mean = sum(xs) / len(xs)
    if mean == 0:
        return 0.0
    var = sum((x - mean) ** 2 for x in xs) / len(xs)
    return math.sqrt(var) / mean


def embedding_diversity(embeddings: list[list[float]] | None) -> float:
    """Mean pairwise cosine distance over a small sample."""
    if not embeddings or len(embeddings) < 2:
        return 0.0
    sample = embeddings[:32]
    n = 0
    total = 0.0
    for i in range(len(sample)):
        a = sample[i]
        na = math.sqrt(sum(x * x for x in a)) or 1e-9
        for j in range(i + 1, len(sample)):
            b = sample[j]
            nb = math.sqrt(sum(x * x for x in b)) or 1e-9
            dot = sum(x * y for x, y in zip(a, b))
            cos = dot / (na * nb)
            total += 1.0 - cos
            n += 1
    return total / max(1, n)


FEATURE_NAMES = (
    "count",
    "duration_seconds",
    "rate_per_second",
    "mean_query_len",
    "stdev_query_len",
    "cov_query_len",
    "mean_word_count",
    "unique_words_ratio",
    "mean_char_entropy",
    "stdev_char_entropy",
    "mean_iat",          # mean interarrival time
    "stdev_iat",
    "p10_iat",
    "burst_score",       # short-term burst measure
    "near_dup_ratio",
    "templated_query_ratio",  # fraction matching templated patterns
    "boundary_probe_ratio",   # fraction matching margin-mining
    "embedding_diversity",
    "response_token_mean",
    "response_token_std",
)


@dataclass
class FeatureExtractor:
    burst_window_seconds: float = 60.0

    def features(self, window: ActorWindow) -> dict[str, float]:
        events = window.events
        n = len(events)
        if n == 0:
            return {name: 0.0 for name in FEATURE_NAMES}

        # Volumetric
        duration = window.duration_seconds
        rate = (n / duration) if duration > 0 else 0.0

        # Lengths
        lens = [e.query_len for e in events]
        mean_len = sum(lens) / n
        var_len = sum((l - mean_len) ** 2 for l in lens) / n
        std_len = math.sqrt(var_len)
        cov_len = coefficient_of_variation(lens)

        # Word counts + unique words
        words_per_event = [tokenize_words(e.query_text) for e in events]
        word_counts = [len(w) for w in words_per_event]
        mean_wc = (sum(word_counts) / n) if n else 0.0
        all_words = [w for ws in words_per_event for w in ws]
        unique_words_ratio = (len(set(all_words)) / len(all_words)) if all_words else 0.0

        # Char entropy
        ents = [char_entropy(e.query_text) for e in events]
        mean_ent = (sum(ents) / n) if n else 0.0
        var_ent = (sum((x - mean_ent) ** 2 for x in ents) / n) if n else 0.0
        std_ent = math.sqrt(var_ent)

        # Interarrival
        timestamps = [e.timestamp for e in events]
        mean_iat, std_iat, p10_iat = interarrival_stats(timestamps)

        # Burst: max queries in any `burst_window_seconds`-second sliding window
        burst = 1
        if n >= 2:
            j = 0
            for i in range(n):
                while timestamps[i] - timestamps[j] > self.burst_window_seconds:
                    j += 1
                burst = max(burst, i - j + 1)
        burst_score = burst / max(1, n)

        near_dup = near_duplicate_ratio([e.query_text for e in events])

        # Templated query ratio: queries matching the same skeleton (NORM-by collapsed digits/whitespace)
        templates: dict[str, int] = {}
        for q in (e.query_text for e in events):
            tpl = re.sub(r"\d+", "<NUM>", q.lower())
            tpl = re.sub(r"\s+", " ", tpl).strip()
            templates[tpl] = templates.get(tpl, 0) + 1
        max_template_count = max(templates.values()) if templates else 1
        templated_ratio = max_template_count / n

        # Boundary-probing keywords (margin-mining attacks)
        boundary_keywords = (
            "exactly between", "borderline", "on the boundary", "predict the label",
            "logit", "logits", "raw output", "probability of class", "top-k tokens",
            "class probability", "what is the score for", "give me the embedding",
        )
        bcount = sum(1 for e in events if any(k in e.query_text.lower() for k in boundary_keywords))
        boundary_probe_ratio = bcount / n

        # Embedding diversity
        embs = [e.embedding for e in events if e.embedding is not None]
        emb_div = embedding_diversity(embs)

        # Response sizes
        rsizes = [e.response_token_count for e in events]
        rmean = (sum(rsizes) / n) if n else 0.0
        rvar = (sum((x - rmean) ** 2 for x in rsizes) / n) if n else 0.0
        rstd = math.sqrt(rvar)

        return {
            "count": float(n),
            "duration_seconds": float(duration),
            "rate_per_second": float(rate),
            "mean_query_len": float(mean_len),
            "stdev_query_len": float(std_len),
            "cov_query_len": float(cov_len),
            "mean_word_count": float(mean_wc),
            "unique_words_ratio": float(unique_words_ratio),
            "mean_char_entropy": float(mean_ent),
            "stdev_char_entropy": float(std_ent),
            "mean_iat": float(mean_iat),
            "stdev_iat": float(std_iat),
            "p10_iat": float(p10_iat),
            "burst_score": float(burst_score),
            "near_dup_ratio": float(near_dup),
            "templated_query_ratio": float(templated_ratio),
            "boundary_probe_ratio": float(boundary_probe_ratio),
            "embedding_diversity": float(emb_div),
            "response_token_mean": float(rmean),
            "response_token_std": float(rstd),
        }

    def feature_vector(self, window: ActorWindow) -> list[float]:
        feats = self.features(window)
        return [feats[name] for name in FEATURE_NAMES]
