"""Reasoning trace logger — structured, append-only audit logging for the pipeline."""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any

logger = logging.getLogger("l90.trace")


@dataclass
class TraceEntry:
    """A single entry in the reasoning trace."""

    timestamp: float
    agent_name: str
    phase: str
    action: str
    input_summary: str = ""
    output_summary: str = ""
    decision: str = ""
    confidence: float | None = None
    details: dict[str, Any] = field(default_factory=dict)


class ReasoningTraceLogger:
    """Thread-safe, structured reasoning trace logger.

    - Writes every entry to the Blackboard's ``reasoning_trace`` list
    - Simultaneously emits to Python ``logging`` (structured JSON for audit)
    - Supports full JSON export
    """

    def __init__(self) -> None:
        self._entries: list[TraceEntry] = []
        self._lock = threading.Lock()

    def log(
        self,
        agent_name: str,
        phase: str,
        action: str,
        *,
        input_summary: str = "",
        output_summary: str = "",
        decision: str = "",
        confidence: float | None = None,
        details: dict[str, Any] | None = None,
    ) -> TraceEntry:
        """Record a trace entry. Thread-safe for parallel agent execution."""
        entry = TraceEntry(
            timestamp=time.time(),
            agent_name=agent_name,
            phase=phase,
            action=action,
            input_summary=input_summary,
            output_summary=output_summary,
            decision=decision,
            confidence=confidence,
            details=details or {},
        )

        with self._lock:
            self._entries.append(entry)

        # Also emit to standard logging for external audit collection
        logger.info(
            "TRACE | %s | %s | %s | decision=%s | confidence=%s",
            agent_name,
            phase,
            action,
            decision,
            confidence,
        )
        return entry

    def get_entries(self) -> list[dict[str, Any]]:
        """Return all entries as dicts (for Blackboard serialization)."""
        with self._lock:
            return [asdict(e) for e in self._entries]

    def export_json(self) -> str:
        """Export the full trace as a JSON string."""
        return json.dumps(self.get_entries(), indent=2, default=str)

    def clear(self) -> None:
        """Clear all entries."""
        with self._lock:
            self._entries.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)
