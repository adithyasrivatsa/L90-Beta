"""Grounding enforcer — independent layer that validates the generator's output.

This layer sits between the generator and the final output. It CANNOT be bypassed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from l90 import config
from l90.agents.base import BaseAgent
from l90.blackboard.blackboard import Blackboard
from l90.models.provider import ModelProvider
from l90.tracing.logger import ReasoningTraceLogger

logger = logging.getLogger(__name__)

INSUFFICIENT_ANSWER = "Insufficient verified information."

GROUNDING_CHECK_PROMPT = """\
You are the Grounding Enforcer of L90, a zero-hallucination scientific RAG system.

Your job is to verify that EVERY claim in the generated answer is grounded in the source chunks.

Given:
- The generated answer
- The source chunks used

For EACH claim in the answer, determine:
1. Is it directly supported by a source chunk? (cite which chunk)
2. Is it a reasonable inference from the chunks? (still acceptable)
3. Is it fabricated/hallucinated? (REJECT)

Respond with JSON:
{
  "all_grounded": true/false,
  "claims": [
    {"claim": "...", "grounded": true/false, "source_chunk_id": "...", "reason": "..."},
    ...
  ],
  "ungrounded_count": 0,
  "grounding_score": 0.0-1.0
}

Be STRICT. Any ungrounded claim means all_grounded = false.
"""


@dataclass
class GroundingReport:
    """Report produced by the grounding enforcer."""

    all_grounded: bool = False
    claims: list[dict[str, Any]] = field(default_factory=list)
    ungrounded_count: int = 0
    grounding_score: float = 0.0
    confidence_check_passed: bool = False
    verification_check_passed: bool = False
    final_verdict: str = "REJECTED"

    def to_dict(self) -> dict[str, Any]:
        return {
            "all_grounded": self.all_grounded,
            "claims": self.claims,
            "ungrounded_count": self.ungrounded_count,
            "grounding_score": self.grounding_score,
            "confidence_check_passed": self.confidence_check_passed,
            "verification_check_passed": self.verification_check_passed,
            "final_verdict": self.final_verdict,
        }


class GroundingEnforcer(BaseAgent):
    """Independent grounding enforcement layer.

    Checks:
    1. Citation check — every claim must map to a retrieved chunk
    2. Verification status — only verified chunks are allowed
    3. Confidence threshold — answer rejected below threshold
    4. Fallback — if any check fails, answer replaced with insufficient message

    This is NOT part of the generator. It is an independent, mandatory layer.
    """

    def __init__(
        self,
        confidence_threshold: float | None = None,
        trace_logger: ReasoningTraceLogger | None = None,
    ) -> None:
        super().__init__(name="GroundingEnforcer", trace_logger=trace_logger)
        self._threshold = confidence_threshold or config.GROUNDING_CONFIDENCE_THRESHOLD

    async def execute(self, blackboard: Blackboard) -> Blackboard:
        """Enforce grounding on the generated answer."""

        self._log_trace(
            phase="grounding",
            action="grounding_check_start",
            input_summary=f"Answer length: {len(blackboard.final_answer)}, Confidence: {blackboard.confidence_score}",
        )

        report = GroundingReport()

        # ── Check 1: Verification status ───────────────────────
        report.verification_check_passed = blackboard.verification_passed

        # ── Check 2: Confidence threshold ──────────────────────
        report.confidence_check_passed = (
            blackboard.confidence_score >= self._threshold
        )

        # ── Check 3: Citation / grounding check via LLM ───────
        if (
            blackboard.final_answer
            and blackboard.final_answer != INSUFFICIENT_ANSWER
            and blackboard.retrieved_chunks
        ):
            await self._check_grounding(blackboard, report)
        else:
            report.all_grounded = False
            report.grounding_score = 0.0

        # ── Final verdict ──────────────────────────────────────
        all_passed = (
            report.all_grounded
            and report.verification_check_passed
            and report.confidence_check_passed
        )

        if all_passed:
            report.final_verdict = "APPROVED"
        else:
            report.final_verdict = "REJECTED"
            blackboard.final_answer = INSUFFICIENT_ANSWER
            logger.warning(
                "Grounding REJECTED: grounded=%s, verified=%s, confidence=%s (threshold=%s)",
                report.all_grounded,
                report.verification_check_passed,
                report.confidence_check_passed,
                self._threshold,
            )

        # Write report to Blackboard
        blackboard.grounding_report = report.to_dict()

        self._log_trace(
            phase="grounding",
            action="grounding_check_complete",
            output_summary=f"Verdict: {report.final_verdict}",
            confidence=report.grounding_score,
            decision=(
                f"Grounded={report.all_grounded}, "
                f"Verified={report.verification_check_passed}, "
                f"Confidence={report.confidence_check_passed}"
            ),
        )

        blackboard.add_trace(
            agent_name=self._name,
            phase="grounding",
            action="grounding_enforced",
            confidence=report.grounding_score,
            details=report.to_dict(),
        )

        return blackboard

    async def _check_grounding(
        self,
        blackboard: Blackboard,
        report: GroundingReport,
    ) -> None:
        """Use LLM to verify claim-by-claim grounding."""
        model = ModelProvider.get_worker_model()

        chunk_context = "\n\n".join(
            f"[Chunk ID: {c.get('id', 'unknown')}]\n{c.get('document', '')}"
            for c in blackboard.retrieved_chunks
        )

        prompt = (
            f"Generated Answer:\n{blackboard.final_answer}\n\n"
            f"Source Chunks:\n{chunk_context}\n\n"
            "Check grounding of every claim."
        )

        try:
            result = await model.generate_json(
                prompt=prompt,
                system_instruction=GROUNDING_CHECK_PROMPT,
                temperature=0.0,
            )

            report.all_grounded = result.get("all_grounded", False)
            report.claims = result.get("claims", [])
            report.ungrounded_count = result.get("ungrounded_count", 0)
            report.grounding_score = result.get("grounding_score", 0.0)

        except Exception as exc:
            logger.error("Grounding check failed: %s", exc)
            report.all_grounded = False
            report.grounding_score = 0.0
