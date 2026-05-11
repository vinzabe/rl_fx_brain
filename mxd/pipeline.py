"""Detection pipeline: stream of QueryEvents -> Verdicts per actor."""
from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable

from .detector import ExtractionDetector, Verdict
from .events import ActorWindow, QueryEvent


@dataclass
class DetectionPipeline:
    detector: ExtractionDetector = field(default_factory=ExtractionDetector)
    window_seconds: float = 3600.0
    _windows: dict[str, ActorWindow] = field(default_factory=dict)

    def ingest(self, event: QueryEvent) -> Verdict:
        w = self._windows.get(event.actor_id)
        if w is None:
            w = ActorWindow(actor_id=event.actor_id, window_seconds=self.window_seconds)
            self._windows[event.actor_id] = w
        w.add(event)
        return self.detector.evaluate(w)

    def process(self, events: Iterable[QueryEvent]) -> dict[str, Verdict]:
        verdicts: dict[str, Verdict] = {}
        for e in events:
            verdicts[e.actor_id] = self.ingest(e)
        return verdicts

    def evaluate_batch(self, events: list[QueryEvent]) -> dict[str, Verdict]:
        """Group events by actor and evaluate once per actor (faster than streaming)."""
        per_actor: dict[str, list[QueryEvent]] = defaultdict(list)
        for e in events:
            per_actor[e.actor_id].append(e)
        out: dict[str, Verdict] = {}
        for actor_id, evs in per_actor.items():
            w = ActorWindow(actor_id=actor_id, window_seconds=self.window_seconds)
            for e in evs:
                w.events.append(e)  # bypass single-event eviction during batch build
            w._evict()
            out[actor_id] = self.detector.evaluate(w)
        return out
