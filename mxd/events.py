"""Event + actor-window dataclasses."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class QueryEvent:
    actor_id: str
    timestamp: float  # epoch seconds
    query_text: str
    response_text: str = ""
    response_token_count: int = 0
    query_token_count: int = 0
    embedding: list[float] | None = None  # optional pre-computed embedding (low-d)
    endpoint: str = "chat"
    ip: str = ""
    user_agent: str = ""
    request_id: str = ""

    @property
    def query_len(self) -> int:
        return len(self.query_text)


@dataclass
class ActorWindow:
    """A rolling window of events for a single actor."""
    actor_id: str
    events: list[QueryEvent] = field(default_factory=list)
    window_seconds: float = 3600.0  # default 1 hour

    def add(self, e: QueryEvent) -> None:
        if e.actor_id != self.actor_id:
            raise ValueError(f"actor mismatch: {e.actor_id} vs {self.actor_id}")
        self.events.append(e)
        self._evict()

    def _evict(self) -> None:
        if not self.events:
            return
        cutoff = self.events[-1].timestamp - self.window_seconds
        self.events = [e for e in self.events if e.timestamp >= cutoff]

    @property
    def count(self) -> int:
        return len(self.events)

    @property
    def duration_seconds(self) -> float:
        if len(self.events) < 2:
            return 0.0
        return self.events[-1].timestamp - self.events[0].timestamp

    def queries(self) -> list[str]:
        return [e.query_text for e in self.events]
