"""LLM-driven incident analyst: turn rule triggers into a forensic narrative."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

try:
    from .llm_client import LLMClient
except ImportError:
    from llm_client import LLMClient  # type: ignore


_AUTO = object()
JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)
ALLOWED_CATEGORIES = {
    "model_extraction",
    "model_inversion",
    "membership_inference",
    "boundary_probing",
    "data_scraping",
    "credential_stuffing",
    "anomalous_volume",
    "benign_burst",
    "ambiguous",
}


@dataclass
class IncidentReport:
    actor_id: str
    headline: str
    severity: str
    summary: str
    attack_category: str
    techniques: list[dict[str, str]] = field(default_factory=list)
    recommended_actions: list[str] = field(default_factory=list)
    referenced_triggers: list[str] = field(default_factory=list)
    confidence: float = 0.0
    fallback: bool = False
    raw_response: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "actor_id": self.actor_id,
            "headline": self.headline,
            "severity": self.severity,
            "summary": self.summary,
            "attack_category": self.attack_category,
            "techniques": self.techniques,
            "recommended_actions": self.recommended_actions,
            "referenced_triggers": self.referenced_triggers,
            "confidence": self.confidence,
            "fallback": self.fallback,
        }


class LLMExtractionAnalyst:
    SYSTEM = (
        "You are an analyst reviewing query patterns for evidence of model "
        "extraction / model stealing / inversion / membership inference / "
        "boundary probing. You receive: an actor_id, a list of static rule "
        "triggers, and a numeric feature dictionary. "
        "Respond ONLY with JSON: { headline (<=120 chars), severity (one of "
        "informational, low, medium, high, critical), summary, attack_category "
        "(one of model_extraction, model_inversion, membership_inference, "
        "boundary_probing, data_scraping, credential_stuffing, anomalous_volume, "
        "benign_burst, ambiguous), techniques (array of {id,name,evidence}; "
        "id may be empty), recommended_actions (array <=8), "
        "referenced_triggers (array of trigger feature names that you cited; "
        "feature names must be present in the input -- others will be discarded), "
        "confidence (float 0..1) }."
    )

    def __init__(self, client: Any = _AUTO, timeout: float = 180.0) -> None:
        if client is _AUTO:
            self.client = LLMClient(timeout=timeout)
        else:
            self.client = client

    def analyse(self, verdict_dict: dict[str, Any]) -> IncidentReport:
        actor_id = str(verdict_dict.get("actor_id", "unknown"))
        if self.client is None:
            return self._fallback(actor_id, "no_llm_client")
        try:
            payload = {
                "actor_id": actor_id,
                "severity": verdict_dict.get("severity"),
                "score": verdict_dict.get("score"),
                "anomaly_score": verdict_dict.get("anomaly_score"),
                "triggers": verdict_dict.get("triggers", []),
                "features": verdict_dict.get("features", {}),
            }
            resp = self.client.chat(
                [
                    {"role": "system", "content": self.SYSTEM},
                    {"role": "user", "content": json.dumps(payload, indent=2)},
                ],
                temperature=0.0,
                max_tokens=900,
            )
            return self._parse(resp.content, verdict_dict, actor_id)
        except Exception as exc:
            return self._fallback(actor_id, f"llm_error:{type(exc).__name__}")

    def _parse(self, text: str, verdict_dict: dict[str, Any], actor_id: str) -> IncidentReport:
        m = JSON_OBJ_RE.search(text)
        if not m:
            return self._fallback(actor_id, "no_json")
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            return self._fallback(actor_id, "invalid_json")

        headline = str(obj.get("headline", "Model-extraction review"))[:120]
        severity = str(obj.get("severity", "low")).lower()
        if severity not in {"informational", "low", "medium", "high", "critical"}:
            severity = "low"
        summary = str(obj.get("summary", ""))[:4000]
        attack_category = str(obj.get("attack_category", "ambiguous")).lower()
        if attack_category not in ALLOWED_CATEGORIES:
            attack_category = "ambiguous"

        valid_features = set(verdict_dict.get("features", {}).keys()) | {t.get("feature") for t in verdict_dict.get("triggers", [])}

        ref_raw = obj.get("referenced_triggers") or []
        referenced: list[str] = []
        if isinstance(ref_raw, list):
            for x in ref_raw[:16]:
                if isinstance(x, str) and x in valid_features:
                    referenced.append(x)

        techniques_raw = obj.get("techniques") or []
        techniques: list[dict[str, str]] = []
        if isinstance(techniques_raw, list):
            for t in techniques_raw[:6]:
                if not isinstance(t, dict):
                    continue
                techniques.append({
                    "id": str(t.get("id", ""))[:32],
                    "name": str(t.get("name", ""))[:120],
                    "evidence": str(t.get("evidence", ""))[:512],
                })

        actions = [str(x)[:240] for x in (obj.get("recommended_actions") or []) if isinstance(x, str)][:8]

        try:
            confidence = float(obj.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))

        return IncidentReport(
            actor_id=actor_id,
            headline=headline,
            severity=severity,
            summary=summary,
            attack_category=attack_category,
            techniques=techniques,
            recommended_actions=actions,
            referenced_triggers=referenced,
            confidence=confidence,
            fallback=False,
            raw_response=text,
        )

    def _fallback(self, actor_id: str, reason: str) -> IncidentReport:
        return IncidentReport(
            actor_id=actor_id,
            headline=f"Extraction analyst unavailable ({reason})",
            severity="low",
            summary=f"LLM analyst unavailable ({reason}); rule triggers stand alone.",
            attack_category="ambiguous",
            techniques=[],
            recommended_actions=[
                "Apply per-actor rate limits.",
                "Manually inspect query log for templated patterns.",
            ],
            referenced_triggers=[],
            confidence=0.2,
            fallback=True,
        )
