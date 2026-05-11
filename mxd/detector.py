"""Detector with two parallel signals:

1. **Rule-based** thresholds with explainable triggers. No training data
   required. Fixed, conservative thresholds chosen from the literature.
2. **IsolationForest** anomaly score (optional) when historic baselines are
   available; we expose `fit()` and `score()` for callers to plug into.

The decision logic combines both into a single Verdict.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Any

try:
    import joblib
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    _HAS_SKLEARN = True
except ImportError:  # pragma: no cover
    joblib = None  # type: ignore
    IsolationForest = None  # type: ignore
    StandardScaler = None  # type: ignore
    _HAS_SKLEARN = False

from .events import ActorWindow
from .features import FEATURE_NAMES, FeatureExtractor


_SEVERITIES = ("informational", "low", "medium", "high", "critical")
_SEVERITY_RANK = {s: i for i, s in enumerate(_SEVERITIES)}


@dataclass
class Verdict:
    actor_id: str
    is_suspicious: bool
    severity: str
    score: float
    triggers: list[dict[str, Any]] = field(default_factory=list)
    features: dict[str, float] = field(default_factory=dict)
    anomaly_score: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "actor_id": self.actor_id,
            "is_suspicious": self.is_suspicious,
            "severity": self.severity,
            "score": self.score,
            "triggers": self.triggers,
            "features": self.features,
            "anomaly_score": self.anomaly_score,
        }


@dataclass
class _Threshold:
    feature: str
    operator: str   # ">" or "<"
    value: float
    weight: float
    severity: str
    description: str


_DEFAULT_THRESHOLDS: tuple[_Threshold, ...] = (
    # Volume / rate
    _Threshold("count", ">", 500.0, 1.0, "high",
               "More than 500 queries in the rolling window indicates aggressive collection."),
    _Threshold("rate_per_second", ">", 5.0, 1.5, "high",
               "Sustained rate > 5 q/s well exceeds humanly-driven usage."),
    _Threshold("burst_score", ">", 0.4, 1.0, "medium",
               "Large fraction of queries clustered into a short burst."),
    # Lexical / templated
    _Threshold("templated_query_ratio", ">", 0.4, 1.5, "high",
               "More than 40% of queries share an identical skeleton; indicates scripted extraction."),
    _Threshold("cov_query_len", "<", 0.05, 1.0, "medium",
               "Query lengths are unusually uniform; suggests templated input."),
    _Threshold("unique_words_ratio", "<", 0.1, 1.0, "medium",
               "Very low lexical diversity for >10 queries."),
    _Threshold("near_dup_ratio", ">", 0.5, 1.0, "medium",
               "Many near-duplicate queries -- likely synonym sweeping or probing."),
    # Probing
    _Threshold("boundary_probe_ratio", ">", 0.1, 2.0, "critical",
               "Queries explicitly request logits/probabilities/embeddings."),
    # Timing
    _Threshold("p10_iat", "<", 0.3, 0.5, "low",
               "10th-percentile interarrival < 0.3s; consistent with automation."),
    # Diversity (only meaningful for substantial windows -- random small
    # samples of unrelated chat naturally span the embedding space)
    _Threshold("embedding_diversity", ">", 0.9, 1.0, "medium",
               "High embedding-space coverage; consistent with distillation-style sampling."),
)


# Rules that require a minimum count to be meaningful; below this they are
# suppressed to avoid false positives on trivially short windows.
_MIN_COUNT_FOR_RULE: dict[str, int] = {
    "rate_per_second": 10,
    "burst_score": 10,
    "templated_query_ratio": 10,
    "cov_query_len": 10,
    "unique_words_ratio": 10,
    "near_dup_ratio": 10,
    "boundary_probe_ratio": 5,
    "p10_iat": 10,
    "embedding_diversity": 50,
}


@dataclass
class ExtractionDetector:
    feature_extractor: FeatureExtractor = field(default_factory=FeatureExtractor)
    thresholds: tuple[_Threshold, ...] = _DEFAULT_THRESHOLDS
    suspicious_score: float = 1.5  # rule-score threshold above which we flag
    model_dir: str = ""
    _isoforest: Any = None
    _scaler: Any = None

    # ---------------- rule-only scoring ----------------
    def _rule_scores(self, feats: dict[str, float]) -> tuple[list[dict[str, Any]], float, str]:
        triggers: list[dict[str, Any]] = []
        total_weight = 0.0
        worst_rank = 0
        count = feats.get("count", 0.0)
        for thr in self.thresholds:
            min_count = _MIN_COUNT_FOR_RULE.get(thr.feature, 1)
            if count < min_count:
                continue
            v = feats.get(thr.feature, 0.0)
            hit = (thr.operator == ">" and v > thr.value) or (thr.operator == "<" and v < thr.value)
            if hit:
                triggers.append({
                    "feature": thr.feature,
                    "operator": thr.operator,
                    "value_observed": v,
                    "threshold": thr.value,
                    "severity": thr.severity,
                    "weight": thr.weight,
                    "description": thr.description,
                })
                total_weight += thr.weight
                worst_rank = max(worst_rank, _SEVERITY_RANK[thr.severity])
        severity = _SEVERITIES[worst_rank] if triggers else "informational"
        return triggers, total_weight, severity

    # ---------------- IsolationForest (optional) ----------------
    def fit_baseline(self, windows: list[ActorWindow]) -> None:
        if not _HAS_SKLEARN:
            raise RuntimeError("scikit-learn is required for the baseline model")
        if not windows:
            raise ValueError("need at least one baseline window")
        X = [self.feature_extractor.feature_vector(w) for w in windows]
        self._scaler = StandardScaler()
        Xs = self._scaler.fit_transform(X)
        # Contamination low -- assume baselines are mostly benign
        self._isoforest = IsolationForest(contamination=0.05, random_state=42, n_estimators=128)
        self._isoforest.fit(Xs)

    def save(self, path: str) -> None:
        if not _HAS_SKLEARN or self._isoforest is None:
            raise RuntimeError("no fitted model to save")
        joblib.dump({
            "schema_version": 1,
            "feature_names": list(FEATURE_NAMES),
            "isoforest": self._isoforest,
            "scaler": self._scaler,
        }, path)

    def load(self, path: str) -> None:
        if not _HAS_SKLEARN:
            raise RuntimeError("scikit-learn is required to load a model")
        data = joblib.load(path)
        if data.get("schema_version") != 1:
            raise ValueError("model schema mismatch")
        if tuple(data.get("feature_names", ())) != FEATURE_NAMES:
            raise ValueError("feature schema mismatch -- retrain")
        self._isoforest = data["isoforest"]
        self._scaler = data["scaler"]

    def _anomaly_score(self, feature_vec: list[float]) -> float | None:
        if self._isoforest is None or self._scaler is None:
            return None
        Xs = self._scaler.transform([feature_vec])
        # IsolationForest score_samples: higher = less anomalous; flip to risk in [0,1]
        raw = float(self._isoforest.score_samples(Xs)[0])
        return 1.0 / (1.0 + math.exp(8.0 * (raw + 0.55)))

    # ---------------- top-level ----------------
    def evaluate(self, window: ActorWindow) -> Verdict:
        feats = self.feature_extractor.features(window)
        feature_vec = [feats[n] for n in FEATURE_NAMES]
        triggers, rule_score, rule_severity = self._rule_scores(feats)
        anomaly = self._anomaly_score(feature_vec)
        score = rule_score + (anomaly or 0.0) * 2.0

        is_suspicious = (rule_score >= self.suspicious_score) or (anomaly is not None and anomaly >= 0.7)
        severity = rule_severity
        if anomaly is not None:
            if anomaly >= 0.85 and _SEVERITY_RANK[severity] < _SEVERITY_RANK["high"]:
                severity = "high"
            elif anomaly >= 0.7 and _SEVERITY_RANK[severity] < _SEVERITY_RANK["medium"]:
                severity = "medium"

        return Verdict(
            actor_id=window.actor_id,
            is_suspicious=is_suspicious,
            severity=severity,
            score=score,
            triggers=triggers,
            features=feats,
            anomaly_score=anomaly,
        )
