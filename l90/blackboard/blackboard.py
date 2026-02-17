"""Blackboard — central shared reasoning state for the L90 pipeline."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Blackboard:
    """Shared state read and written by all agents throughout execution.

    Every field is documented because this structure IS the system's
    reasoning memory — auditability requires full transparency.
    """

    # ── Identity ───────────────────────────────────────────────
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # ── Query ──────────────────────────────────────────────────
    query: str = ""
    mode: str = ""                           # STRICT | PARTIAL | GENERAL | INCOGNITO | WORKSPACE
    user_id: str = ""
    workspace_id: str = ""

    # ── Access control ─────────────────────────────────────────
    allowed_sources: list[str] = field(default_factory=list)

    # ── Retrieval ──────────────────────────────────────────────
    retrieved_chunks: list[dict[str, Any]] = field(default_factory=list)

    # ── Analysis ───────────────────────────────────────────────
    analysis_results: list[dict[str, Any]] = field(default_factory=list)

    # ── Verification ───────────────────────────────────────────
    verification_results: list[dict[str, Any]] = field(default_factory=list)
    verification_passed: bool = False

    # ── Correction ─────────────────────────────────────────────
    correction_results: list[dict[str, Any]] = field(default_factory=list)
    correction_loop_count: int = 0

    # ── Output ─────────────────────────────────────────────────
    confidence_score: float = 0.0
    final_answer: str = ""

    # ── Audit & Trace ──────────────────────────────────────────
    reasoning_trace: list[dict[str, Any]] = field(default_factory=list)

    # ── Planning ───────────────────────────────────────────────
    execution_plan: dict[str, Any] = field(default_factory=dict)

    # ── Grounding ──────────────────────────────────────────────
    grounding_report: dict[str, Any] = field(default_factory=dict)

    # ── Code Verification (math/physics proofs) ────────────────
    code_verification: dict[str, Any] = field(default_factory=dict)

    # ── Deep Reasoning (Partial mode synthesis) ────────────────
    deep_reasoning: list[dict[str, Any]] = field(default_factory=list)

    # ── LaTeX Equations ────────────────────────────────────────
    latex_equations: list[str] = field(default_factory=list)

    # ── Metadata ───────────────────────────────────────────────
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    # ── Methods ────────────────────────────────────────────────

    def add_trace(
        self,
        agent_name: str,
        phase: str,
        action: str,
        *,
        decision: str = "",
        confidence: float | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Append an entry to the reasoning trace (append-only audit log)."""
        entry: dict[str, Any] = {
            "timestamp": time.time(),
            "agent": agent_name,
            "phase": phase,
            "action": action,
            "decision": decision,
        }
        if confidence is not None:
            entry["confidence"] = confidence
        if details:
            entry["details"] = details
        self.reasoning_trace.append(entry)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the entire Blackboard to a plain dict (for persistence / audit)."""
        return {
            "session_id": self.session_id,
            "query": self.query,
            "mode": self.mode,
            "user_id": self.user_id,
            "workspace_id": self.workspace_id,
            "allowed_sources": self.allowed_sources,
            "retrieved_chunks": self.retrieved_chunks,
            "analysis_results": self.analysis_results,
            "verification_results": self.verification_results,
            "verification_passed": self.verification_passed,
            "correction_results": self.correction_results,
            "correction_loop_count": self.correction_loop_count,
            "confidence_score": self.confidence_score,
            "final_answer": self.final_answer,
            "reasoning_trace": self.reasoning_trace,
            "execution_plan": self.execution_plan,
            "grounding_report": self.grounding_report,
            "code_verification": self.code_verification,
            "deep_reasoning": self.deep_reasoning,
            "latex_equations": self.latex_equations,
            "metadata": self.metadata,
            "created_at": self.created_at,
        }

    def reset(self) -> None:
        """Clear all mutable fields but keep session_id and created_at."""
        self.query = ""
        self.mode = ""
        self.user_id = ""
        self.workspace_id = ""
        self.allowed_sources = []
        self.retrieved_chunks = []
        self.analysis_results = []
        self.verification_results = []
        self.verification_passed = False
        self.correction_results = []
        self.correction_loop_count = 0
        self.confidence_score = 0.0
        self.final_answer = ""
        self.reasoning_trace = []
        self.execution_plan = {}
        self.grounding_report = {}
        self.code_verification = {}
        self.deep_reasoning = []
        self.latex_equations = []
        self.metadata = {}
