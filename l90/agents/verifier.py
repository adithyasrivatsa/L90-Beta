"""Verification agent — checks mathematical, physics, and logical correctness.

Enhanced: dimensional analysis, numerical bounds check, and code execution trigger validation.
"""

from __future__ import annotations

import logging
from typing import Any

from l90.agents.base import BaseAgent
from l90.blackboard.blackboard import Blackboard
from l90.models.provider import ModelProvider
from l90.tracing.logger import ReasoningTraceLogger

logger = logging.getLogger(__name__)

VERIFICATION_SYSTEM_PROMPT = """\
You are a Verification Agent in L90, a NASA/DARPA-grade scientific RAG system.

Your job is to verify the analysis results against the original retrieved chunks.

Check for:
1. **Mathematical correctness**: Are formulas and calculations accurate?
2. **Physics correctness**:
   - Do physical claims obey known laws?
   - **Dimensional Analysis**: Do the units matches? (e.g., $E=mc^2$ -> Joules = kg * (m/s)^2)
3. **Numerical Bounds**: Are the values physically reasonable? (e.g., velocity <= c)
4. **Logical consistency**: Are there contradictions between extracted facts?
5. **Document grounding**: Is every claim traceable to a specific chunk?

You MUST respond with a JSON object:
{
  "verification_passed": true/false,
  "math_check": {"passed": true/false, "issues": ["issue1", ...]},
  "physics_check": {"passed": true/false, "issues": ["issue1", ...], "dimensional_analysis": "valid/invalid"},
  "logic_check": {"passed": true/false, "issues": ["issue1", ...]},
  "grounding_check": {"passed": true/false, "ungrounded_claims": ["claim1", ...]},
  "confidence_score": 0.0-1.0,
  "summary": "brief verification summary"
}

Be STRICT. If ANY check fails, set verification_passed to false.
"""


class VerificationAgent(BaseAgent):
    """Verifies analysis results for correctness, grounding, and physical validity."""

    def __init__(self, trace_logger: ReasoningTraceLogger | None = None) -> None:
        super().__init__(name="VerificationAgent", trace_logger=trace_logger)

    async def execute(self, blackboard: Blackboard) -> Blackboard:
        """Run verification checks on analysis results."""

        self._log_trace(
            phase="verification",
            action="verification_start",
            input_summary=f"Verifying {len(blackboard.analysis_results)} analysis results",
        )

        if not blackboard.analysis_results:
            blackboard.verification_passed = False
            self._log_trace(
                phase="verification",
                action="verification_skip",
                decision="No analysis results to verify",
            )
            return blackboard

        model = ModelProvider.get_worker_model()

        # Build context: analysis results + original chunks for grounding check
        chunk_summary = "\n".join(
            f"[Chunk {i}] {c.get('document', '')[:500]}"  # More context for verification
            for i, c in enumerate(blackboard.retrieved_chunks)
        )

        analysis_summary = ""
        for i, result in enumerate(blackboard.analysis_results):
            analysis_summary += f"\n[Analysis {i}]\n"
            for key in ("formulas", "facts", "constraints", "relationships", "summary", "proof_steps"):
                if key in result:
                    analysis_summary += f"  {key}: {result[key]}\n"

        # Check if code verification was performed
        code_context = ""
        if blackboard.code_verification:
            code_context = f"\nPython Code Verification Result: {blackboard.code_verification}\n"

        prompt = (
            f"User Query: {blackboard.query}\n\n"
            f"Original Retrieved Chunks:\n{chunk_summary}\n\n"
            f"Analysis Results:\n{analysis_summary}\n\n"
            f"{code_context}\n"
            "Verify these results for correctness, grounding, and physical validity."
        )

        try:
            result = await model.generate_json(
                prompt=prompt,
                system_instruction=VERIFICATION_SYSTEM_PROMPT,
                temperature=0.0,
            )

            # If code verification failed, force verification failure
            if blackboard.code_verification and not blackboard.code_verification.get("success", True):
                result["verification_passed"] = False
                result["math_check"] = {"passed": False, "issues": ["Python code verification failed"]}

            verification_entry = {
                "verification_passed": result.get("verification_passed", False),
                "math_check": result.get("math_check", {}),
                "physics_check": result.get("physics_check", {}),
                "logic_check": result.get("logic_check", {}),
                "grounding_check": result.get("grounding_check", {}),
                "confidence_score": result.get("confidence_score", 0.0),
                "summary": result.get("summary", ""),
            }

            blackboard.verification_results.append(verification_entry)
            blackboard.verification_passed = verification_entry["verification_passed"]
            blackboard.confidence_score = verification_entry["confidence_score"]

            self._log_trace(
                phase="verification",
                action="verification_complete",
                output_summary=f"Passed={blackboard.verification_passed}, Confidence={blackboard.confidence_score}",
                confidence=blackboard.confidence_score,
                decision=verification_entry["summary"],
            )

        except Exception as exc:
            logger.error("Verification failed: %s", exc)
            blackboard.verification_passed = False
            blackboard.confidence_score = 0.0
            blackboard.verification_results.append({"error": str(exc)})
            self._log_trace(
                phase="verification",
                action="verification_error",
                decision=f"Error: {exc}",
            )

        blackboard.add_trace(
            agent_name=self._name,
            phase="verification",
            action="verification_recorded",
            confidence=blackboard.confidence_score,
            details={"passed": blackboard.verification_passed},
        )

        return blackboard
