"""Tests for mxd (model-extraction-detector)."""
from __future__ import annotations

import json
import math
import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(_HERE, "..")))

from mxd import (  # noqa: E402
    ActorWindow,
    DetectionPipeline,
    ExtractionDetector,
    FeatureExtractor,
    IncidentReport,
    LLMExtractionAnalyst,
    QueryEvent,
    Verdict,
    generate_attacker_corpus,
    generate_benign_corpus,
)
from mxd.cli import main as cli_main  # noqa: E402
from mxd.features import (  # noqa: E402
    FEATURE_NAMES,
    char_entropy,
    coefficient_of_variation,
    embedding_diversity,
    interarrival_stats,
    jaccard_ngrams,
    near_duplicate_ratio,
    shannon_entropy_bytes,
    tokenize_words,
)


# ------------------- events / window -------------------
class TestEventsWindow:
    def test_query_event_query_len(self):
        e = QueryEvent(actor_id="a", timestamp=0.0, query_text="hello")
        assert e.query_len == 5

    def test_window_add_and_count(self):
        w = ActorWindow(actor_id="a")
        w.add(QueryEvent(actor_id="a", timestamp=0.0, query_text="x"))
        w.add(QueryEvent(actor_id="a", timestamp=1.0, query_text="y"))
        assert w.count == 2
        assert w.queries() == ["x", "y"]

    def test_window_actor_mismatch_raises(self):
        w = ActorWindow(actor_id="a")
        with pytest.raises(ValueError):
            w.add(QueryEvent(actor_id="b", timestamp=0.0, query_text="x"))

    def test_window_evicts_old_events(self):
        w = ActorWindow(actor_id="a", window_seconds=10.0)
        w.add(QueryEvent(actor_id="a", timestamp=0.0, query_text="old"))
        w.add(QueryEvent(actor_id="a", timestamp=5.0, query_text="mid"))
        w.add(QueryEvent(actor_id="a", timestamp=100.0, query_text="new"))
        # only the last should remain (cutoff = 90.0)
        assert w.count == 1
        assert w.queries() == ["new"]

    def test_duration_zero_for_single_event(self):
        w = ActorWindow(actor_id="a")
        w.add(QueryEvent(actor_id="a", timestamp=0.0, query_text="x"))
        assert w.duration_seconds == 0.0

    def test_duration_basic(self):
        w = ActorWindow(actor_id="a", window_seconds=1e6)
        w.add(QueryEvent(actor_id="a", timestamp=0.0, query_text="x"))
        w.add(QueryEvent(actor_id="a", timestamp=10.0, query_text="y"))
        assert w.duration_seconds == 10.0


# ------------------- feature helpers -------------------
class TestFeatureHelpers:
    def test_shannon_entropy_empty(self):
        assert shannon_entropy_bytes(b"") == 0.0

    def test_shannon_entropy_uniform_vs_constant(self):
        constant = shannon_entropy_bytes(b"aaaaaa")
        diverse = shannon_entropy_bytes(bytes(range(64)))
        assert constant == 0.0
        assert diverse > 5.0

    def test_char_entropy(self):
        assert char_entropy("aaa") == 0.0
        assert char_entropy("abcdefghij") > 3.0

    def test_tokenize_words(self):
        assert tokenize_words("Hello, world! 42 cats") == ["hello", "world", "42", "cats"]

    def test_interarrival_stats_short(self):
        assert interarrival_stats([1.0]) == (0.0, 0.0, 0.0)

    def test_interarrival_stats_uniform(self):
        m, s, p = interarrival_stats([0.0, 1.0, 2.0, 3.0])
        assert m == pytest.approx(1.0)
        assert s == pytest.approx(0.0)

    def test_jaccard_ngrams_identical(self):
        assert jaccard_ngrams("hello world", "hello world") == 1.0

    def test_jaccard_ngrams_disjoint(self):
        assert jaccard_ngrams("aaaaaa", "zzzzzz") == 0.0

    def test_jaccard_ngrams_partial(self):
        v = jaccard_ngrams("classify entity cat", "classify entity dog")
        assert 0.0 < v < 1.0

    def test_near_dup_ratio_few(self):
        assert near_duplicate_ratio(["only"]) == 0.0

    def test_near_dup_ratio_high(self):
        qs = ["classify dog as animal", "classify cat as animal", "classify wolf as animal"] * 4
        r = near_duplicate_ratio(qs)
        assert r > 0.5

    def test_near_dup_ratio_subsamples_for_large(self):
        # Each query is structurally distinct; ensures subsampling path does
        # not crash and produces a value in [0, 1].
        qs = [
            f"completely distinct topic {chr(65 + (i % 26))*5} {i}: research notes"
            for i in range(200)
        ]
        r = near_duplicate_ratio(qs)
        assert 0.0 <= r <= 1.0

    def test_coefficient_of_variation(self):
        assert coefficient_of_variation([]) == 0.0
        assert coefficient_of_variation([5.0, 5.0, 5.0]) == 0.0
        cov = coefficient_of_variation([1.0, 10.0])
        assert cov > 0.0

    def test_embedding_diversity_empty(self):
        assert embedding_diversity(None) == 0.0
        assert embedding_diversity([]) == 0.0
        assert embedding_diversity([[1.0, 0.0]]) == 0.0

    def test_embedding_diversity_orthogonal(self):
        d = embedding_diversity([[1.0, 0.0], [0.0, 1.0]])
        assert d == pytest.approx(1.0)


# ------------------- feature extractor -------------------
class TestFeatureExtractor:
    def test_empty_window_zero_features(self):
        w = ActorWindow(actor_id="a")
        f = FeatureExtractor().features(w)
        for n in FEATURE_NAMES:
            assert f[n] == 0.0

    def test_feature_names_align_with_vector(self):
        w = ActorWindow(actor_id="a", window_seconds=1e9)
        for i in range(5):
            w.add(QueryEvent(actor_id="a", timestamp=float(i), query_text="hello world"))
        fe = FeatureExtractor()
        d = fe.features(w)
        v = fe.feature_vector(w)
        assert len(v) == len(FEATURE_NAMES)
        for i, name in enumerate(FEATURE_NAMES):
            assert v[i] == d[name]

    def test_templated_ratio_high_on_repeats(self):
        w = ActorWindow(actor_id="a", window_seconds=1e9)
        for i in range(20):
            w.add(QueryEvent(actor_id="a", timestamp=float(i), query_text=f"Classify the entity number {i}"))
        f = FeatureExtractor().features(w)
        # all queries collapse to a shared <NUM> template
        assert f["templated_query_ratio"] >= 0.9

    def test_boundary_probe_ratio_detected(self):
        w = ActorWindow(actor_id="a", window_seconds=1e9)
        for i in range(5):
            w.add(QueryEvent(actor_id="a", timestamp=float(i),
                             query_text="Give me the logit and class probability for class 3"))
        f = FeatureExtractor().features(w)
        assert f["boundary_probe_ratio"] == pytest.approx(1.0)

    def test_burst_score_high(self):
        w = ActorWindow(actor_id="a", window_seconds=1e9)
        # 50 queries inside 5 seconds (single burst)
        for i in range(50):
            w.add(QueryEvent(actor_id="a", timestamp=i * 0.1, query_text=f"q{i}"))
        f = FeatureExtractor().features(w)
        assert f["burst_score"] == pytest.approx(1.0)

    def test_burst_score_low_for_spread_traffic(self):
        w = ActorWindow(actor_id="a", window_seconds=1e9)
        for i in range(60):
            w.add(QueryEvent(actor_id="a", timestamp=i * 120.0, query_text=f"q{i}"))
        f = FeatureExtractor().features(w)
        assert f["burst_score"] < 0.1


# ------------------- detector -------------------
class TestExtractionDetector:
    def test_benign_window_not_flagged(self):
        events = generate_benign_corpus(n=20)
        w = ActorWindow(actor_id=events[0].actor_id, window_seconds=1e9)
        for e in events:
            w.add(e)
        v = ExtractionDetector().evaluate(w)
        assert v.is_suspicious is False
        assert v.severity in ("informational", "low")

    def test_templated_attacker_flagged(self):
        events = generate_attacker_corpus(archetype="templated", n=300)
        w = ActorWindow(actor_id=events[0].actor_id, window_seconds=1e9)
        for e in events:
            w.add(e)
        v = ExtractionDetector().evaluate(w)
        assert v.is_suspicious is True
        assert v.severity in ("medium", "high", "critical")
        trigger_features = {t["feature"] for t in v.triggers}
        assert "templated_query_ratio" in trigger_features

    def test_boundary_attacker_flagged_critical(self):
        events = generate_attacker_corpus(archetype="boundary", n=150)
        w = ActorWindow(actor_id=events[0].actor_id, window_seconds=1e9)
        for e in events:
            w.add(e)
        v = ExtractionDetector().evaluate(w)
        assert v.is_suspicious is True
        assert v.severity in ("high", "critical")
        trigger_features = {t["feature"] for t in v.triggers}
        assert "boundary_probe_ratio" in trigger_features

    def test_burst_attacker_flagged(self):
        events = generate_attacker_corpus(archetype="burst", n=1000)
        w = ActorWindow(actor_id=events[0].actor_id, window_seconds=1e9)
        for e in events:
            w.add(e)
        v = ExtractionDetector().evaluate(w)
        assert v.is_suspicious is True
        # rate or burst threshold definitely triggered
        tfs = {t["feature"] for t in v.triggers}
        assert "rate_per_second" in tfs or "burst_score" in tfs or "count" in tfs

    def test_verdict_to_dict_roundtrip(self):
        events = generate_attacker_corpus(archetype="templated", n=200)
        w = ActorWindow(actor_id=events[0].actor_id, window_seconds=1e9)
        for e in events:
            w.add(e)
        v = ExtractionDetector().evaluate(w)
        d = v.to_dict()
        for key in ("actor_id", "is_suspicious", "severity", "score",
                    "triggers", "features", "anomaly_score"):
            assert key in d

    def test_under_10_queries_skip_low_diversity_floor(self):
        # ensure tiny windows don't false-positive on `<` thresholds
        w = ActorWindow(actor_id="a", window_seconds=1e9)
        for i in range(5):
            w.add(QueryEvent(actor_id="a", timestamp=float(i), query_text="same"))
        v = ExtractionDetector().evaluate(w)
        assert v.is_suspicious is False

    def test_anomaly_score_none_without_baseline(self):
        events = generate_benign_corpus(n=10)
        w = ActorWindow(actor_id=events[0].actor_id, window_seconds=1e9)
        for e in events:
            w.add(e)
        v = ExtractionDetector().evaluate(w)
        assert v.anomaly_score is None

    def test_isoforest_baseline_fit_and_score(self):
        # smoke test the optional anomaly model
        pytest.importorskip("sklearn")
        baselines = []
        for seed in range(8):
            es = generate_benign_corpus(n=30, seed=seed, actor_id=f"u_{seed}")
            w = ActorWindow(actor_id=es[0].actor_id, window_seconds=1e9)
            for e in es:
                w.add(e)
            baselines.append(w)
        d = ExtractionDetector()
        d.fit_baseline(baselines)
        # benign window should score low anomaly risk
        v_benign = d.evaluate(baselines[0])
        assert v_benign.anomaly_score is not None
        assert 0.0 <= v_benign.anomaly_score <= 1.0
        # attacker window should score higher
        events = generate_attacker_corpus(archetype="templated", n=200)
        w2 = ActorWindow(actor_id=events[0].actor_id, window_seconds=1e9)
        for e in events:
            w2.add(e)
        v_attack = d.evaluate(w2)
        assert v_attack.anomaly_score is not None
        assert v_attack.anomaly_score >= v_benign.anomaly_score - 0.05  # plausible ordering

    def test_isoforest_save_load_roundtrip(self, tmp_path):
        pytest.importorskip("sklearn")
        baselines = []
        for seed in range(5):
            es = generate_benign_corpus(n=20, seed=seed, actor_id=f"u_{seed}")
            w = ActorWindow(actor_id=es[0].actor_id, window_seconds=1e9)
            for e in es:
                w.add(e)
            baselines.append(w)
        d = ExtractionDetector()
        d.fit_baseline(baselines)
        path = str(tmp_path / "iso.joblib")
        d.save(path)
        d2 = ExtractionDetector()
        d2.load(path)
        v1 = d.evaluate(baselines[0])
        v2 = d2.evaluate(baselines[0])
        assert v1.anomaly_score == pytest.approx(v2.anomaly_score)

    def test_isoforest_save_without_fit_raises(self, tmp_path):
        pytest.importorskip("sklearn")
        d = ExtractionDetector()
        with pytest.raises(RuntimeError):
            d.save(str(tmp_path / "x.joblib"))


# ------------------- pipeline -------------------
class TestPipeline:
    def test_ingest_streams_per_actor(self):
        pipe = DetectionPipeline()
        events = generate_attacker_corpus(archetype="templated", n=50, actor_id="x")
        last = None
        for e in events:
            last = pipe.ingest(e)
        assert last is not None
        assert last.actor_id == "x"

    def test_evaluate_batch_groups_by_actor(self):
        evs_a = generate_benign_corpus(n=10, actor_id="a")
        evs_b = generate_attacker_corpus(archetype="templated", n=120, actor_id="b")
        out = DetectionPipeline().evaluate_batch(evs_a + evs_b)
        assert set(out.keys()) == {"a", "b"}
        assert out["b"].is_suspicious is True
        assert out["a"].is_suspicious is False

    def test_pipeline_process_returns_latest_per_actor(self):
        evs = generate_benign_corpus(n=5, actor_id="a")
        out = DetectionPipeline().process(evs)
        assert "a" in out


# ------------------- analyst (mocked) -------------------
def _fake_llm(content: str):
    class _R:
        def __init__(self, c): self.content = c
    class _C:
        def chat(self, *a, **k): return _R(content)
    return _C()


def _mk_verdict_dict() -> dict:
    events = generate_attacker_corpus(archetype="templated", n=200)
    w = ActorWindow(actor_id=events[0].actor_id, window_seconds=1e9)
    for e in events:
        w.add(e)
    return ExtractionDetector().evaluate(w).to_dict()


class TestAnalyst:
    def test_no_client_fallback(self):
        a = LLMExtractionAnalyst(client=None)
        r = a.analyse({"actor_id": "x", "triggers": [], "features": {}})
        assert r.fallback is True

    def test_valid_response_parses(self):
        vd = _mk_verdict_dict()
        body = json.dumps({
            "headline": "Templated extraction sweep on classifier",
            "severity": "high",
            "summary": "200 templated queries in a tight burst look like model distillation.",
            "attack_category": "model_extraction",
            "techniques": [{"id": "T1606", "name": "Extraction", "evidence": "shared <NUM> template"}],
            "recommended_actions": ["block actor", "rate limit"],
            "referenced_triggers": ["templated_query_ratio", "rate_per_second"],
            "confidence": 0.9,
        })
        r = LLMExtractionAnalyst(client=_fake_llm(body)).analyse(vd)
        assert r.fallback is False
        assert r.attack_category == "model_extraction"
        assert r.severity == "high"
        assert "templated_query_ratio" in r.referenced_triggers
        assert any(t["name"] == "Extraction" for t in r.techniques)
        assert r.confidence == pytest.approx(0.9)

    def test_hallucinated_trigger_dropped(self):
        vd = _mk_verdict_dict()
        body = json.dumps({
            "headline": "x", "severity": "low", "summary": "ok",
            "attack_category": "benign_burst",
            "referenced_triggers": ["totally_fake_feature", "templated_query_ratio"],
            "confidence": 0.5,
        })
        r = LLMExtractionAnalyst(client=_fake_llm(body)).analyse(vd)
        assert "totally_fake_feature" not in r.referenced_triggers
        # templated_query_ratio is real and should survive
        assert "templated_query_ratio" in r.referenced_triggers

    def test_unknown_category_remapped(self):
        body = json.dumps({"headline": "x", "severity": "low", "summary": "",
                           "attack_category": "AGI_TAKEOVER", "confidence": 0.5})
        r = LLMExtractionAnalyst(client=_fake_llm(body)).analyse({"actor_id": "y", "features": {}, "triggers": []})
        assert r.attack_category == "ambiguous"

    def test_invalid_severity_remapped(self):
        body = json.dumps({"headline": "x", "severity": "PANIC", "summary": "",
                           "attack_category": "ambiguous"})
        r = LLMExtractionAnalyst(client=_fake_llm(body)).analyse({"actor_id": "y", "features": {}, "triggers": []})
        assert r.severity == "low"

    def test_garbage_response_fallback(self):
        r = LLMExtractionAnalyst(client=_fake_llm("not json at all")).analyse(
            {"actor_id": "y", "features": {}, "triggers": []})
        assert r.fallback is True

    def test_exception_fallback(self):
        class _Boom:
            def chat(self, *a, **k): raise RuntimeError("boom")
        r = LLMExtractionAnalyst(client=_Boom()).analyse({"actor_id": "y", "features": {}, "triggers": []})
        assert r.fallback is True

    def test_confidence_clamped(self):
        body = json.dumps({"headline": "x", "severity": "low", "summary": "",
                           "attack_category": "ambiguous", "confidence": 5.0})
        r = LLMExtractionAnalyst(client=_fake_llm(body)).analyse({"actor_id": "y", "features": {}, "triggers": []})
        assert r.confidence == 1.0

    def test_report_to_dict_keys(self):
        r = IncidentReport(actor_id="x", headline="h", severity="low",
                           summary="s", attack_category="ambiguous")
        d = r.to_dict()
        for k in ("actor_id", "headline", "severity", "summary", "attack_category",
                  "techniques", "recommended_actions", "referenced_triggers",
                  "confidence", "fallback"):
            assert k in d


# ------------------- CLI -------------------
def _write_events(path: str, events) -> None:
    with open(path, "w") as fh:
        for e in events:
            d = {
                "actor_id": e.actor_id, "timestamp": e.timestamp,
                "query_text": e.query_text, "response_text": e.response_text,
                "response_token_count": e.response_token_count,
                "query_token_count": e.query_token_count,
                "embedding": e.embedding,
            }
            fh.write(json.dumps(d) + "\n")


class TestCLI:
    def test_scan_attacker_jsonl(self, tmp_path):
        evs = generate_attacker_corpus(archetype="templated", n=120, actor_id="x")
        p = tmp_path / "events.jsonl"
        _write_events(str(p), evs)
        out = tmp_path / "r.json"
        rc = cli_main(["scan", str(p), "-o", str(out)])
        assert rc == 0
        data = json.loads(out.read_text())
        assert "x" in data
        assert data["x"]["is_suspicious"] is True

    def test_fail_on_suspicious(self, tmp_path):
        evs = generate_attacker_corpus(archetype="boundary", n=120, actor_id="x")
        p = tmp_path / "events.jsonl"
        _write_events(str(p), evs)
        out = tmp_path / "r.json"
        rc = cli_main(["scan", str(p), "-o", str(out), "--fail-on-suspicious"])
        assert rc == 1

    def test_benign_no_fail(self, tmp_path):
        evs = generate_benign_corpus(n=20, actor_id="x")
        p = tmp_path / "events.jsonl"
        _write_events(str(p), evs)
        out = tmp_path / "r.json"
        rc = cli_main(["scan", str(p), "-o", str(out), "--fail-on-suspicious"])
        assert rc == 0


# ------------------- LLM live -------------------
LLM_LIVE = os.environ.get("LLM_LIVE") == "1"


@pytest.mark.skipif(not LLM_LIVE, reason="LLM_LIVE not set")
class TestLLMLive:
    _templated = None
    _boundary = None
    _benign = None

    @classmethod
    def _t(cls):
        if cls._templated is None:
            events = generate_attacker_corpus(archetype="templated", n=200, actor_id="atk_t")
            w = ActorWindow(actor_id="atk_t", window_seconds=1e9)
            for e in events:
                w.add(e)
            cls._templated = LLMExtractionAnalyst().analyse(
                ExtractionDetector().evaluate(w).to_dict())
        return cls._templated

    @classmethod
    def _b(cls):
        if cls._boundary is None:
            events = generate_attacker_corpus(archetype="boundary", n=150, actor_id="atk_b")
            w = ActorWindow(actor_id="atk_b", window_seconds=1e9)
            for e in events:
                w.add(e)
            cls._boundary = LLMExtractionAnalyst().analyse(
                ExtractionDetector().evaluate(w).to_dict())
        return cls._boundary

    @classmethod
    def _ben(cls):
        if cls._benign is None:
            events = generate_benign_corpus(n=20, actor_id="user_calm")
            w = ActorWindow(actor_id="user_calm", window_seconds=1e9)
            for e in events:
                w.add(e)
            cls._benign = LLMExtractionAnalyst().analyse(
                ExtractionDetector().evaluate(w).to_dict())
        return cls._benign

    def test_live_templated_flagged(self):
        r = self._t()
        assert r.fallback is False
        assert r.severity in ("low", "medium", "high", "critical")
        assert r.attack_category in (
            "model_extraction", "data_scraping", "anomalous_volume",
            "boundary_probing", "model_inversion", "membership_inference",
            "ambiguous",
        )

    def test_live_boundary_categorised(self):
        r = self._b()
        assert r.fallback is False
        # boundary probes are a clear signature; allow related categories
        assert r.attack_category in (
            "boundary_probing", "model_extraction", "model_inversion",
            "membership_inference", "anomalous_volume", "ambiguous",
        )

    def test_live_benign_low(self):
        r = self._ben()
        assert r.fallback is False
        assert r.severity in ("informational", "low", "medium")

    def test_live_grounded_trigger_names(self):
        r = self._t()
        # any cited triggers must be real feature names
        for n in r.referenced_triggers:
            assert n in FEATURE_NAMES

    def test_live_confidence_in_range(self):
        r = self._t()
        assert 0.0 <= r.confidence <= 1.0
