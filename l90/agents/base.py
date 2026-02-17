"""Base agent interface — all L90 agents inherit from this."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from l90.blackboard.blackboard import Blackboard
    from l90.tracing.logger import ReasoningTraceLogger


class BaseAgent(ABC):
    """Abstract base class for all L90 pipeline agents.

    Every agent:
    1. Reads from the Blackboard
    2. Performs its task
    3. Writes results back to the Blackboard
    4. Records its reasoning in the trace logger
    """

    def __init__(self, name: str, trace_logger: ReasoningTraceLogger | None = None) -> None:
        self._name = name
        self._trace = trace_logger

    @property
    def name(self) -> str:
        return self._name

    @abstractmethod
    async def execute(self, blackboard: Blackboard) -> Blackboard:
        """Execute this agent's task, mutating the Blackboard in place.

        Args:
            blackboard: The shared state object.

        Returns:
            The (mutated) Blackboard.
        """
        ...

    def _log_trace(
        self,
        phase: str,
        action: str,
        *,
        input_summary: str = "",
        output_summary: str = "",
        decision: str = "",
        confidence: float | None = None,
    ) -> None:
        """Convenience: log a trace entry if a logger is attached."""
        if self._trace:
            self._trace.log(
                agent_name=self._name,
                phase=phase,
                action=action,
                input_summary=input_summary,
                output_summary=output_summary,
                decision=decision,
                confidence=confidence,
            )
