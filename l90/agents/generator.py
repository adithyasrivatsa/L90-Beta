"""Generator agent — produces the final grounded answer from verified Blackboard content."""

from __future__ import annotations

import logging
from typing import Any

from l90.agents.base import BaseAgent
from l90.blackboard.blackboard import Blackboard
from l90.models.provider import ModelProvider
from l90.tracing.logger import ReasoningTraceLogger

logger = logging.getLogger(__name__)

GENERATOR_SYSTEM_PROMPT = """\
You are the Generator Agent of L90, a zero-hallucination scientific RAG system.

CRITICAL RULES:
1. Answer ONLY using the verified information provided below.
2. Every claim MUST be traceable to a specific source chunk.
3. If the information is insufficient, respond EXACTLY with:
   "Insufficient verified information."
4. Do NOT use your prior knowledge. Do NOT hallucinate.
5. Cite sources using [Source: filename] notation.
6. For mathematical/physics content, reproduce formulas exactly as found.

Format your answer clearly with proper structure.
"""

INSUFFICIENT_ANSWER = "Insufficient verified information."


class GeneratorAgent(BaseAgent):
    """Produces the final grounded answer.

    The generator:
    1. Reads ONLY verified Blackboard content
    2. Generates an answer grounded in the chunks
    3. Falls back to the insufficient-data message if ungrounded
    4. Passes output to the GroundingEnforcer (via the graph)
    """

    def __init__(self, trace_logger: ReasoningTraceLogger | None = None) -> None:
        super().__init__(name="GeneratorAgent", trace_logger=trace_logger)

    async def execute(self, blackboard: Blackboard) -> Blackboard:
        """Generate the final answer from verified Blackboard content."""

        self._log_trace(
            phase="generation",
            action="generation_start",
            input_summary=(
                f"Verified={blackboard.verification_passed}, "
                f"Confidence={blackboard.confidence_score}, "
                f"Chunks={len(blackboard.retrieved_chunks)}"
            ),
        )

        # Guard: if verification never passed and we have no data
        if not blackboard.retrieved_chunks:
            blackboard.final_answer = INSUFFICIENT_ANSWER
            self._log_trace(
                phase="generation",
                action="generation_insufficient",
                decision="No retrieved chunks available",
            )
            blackboard.add_trace(
                agent_name=self._name,
                phase="generation",
                action="insufficient_data",
            )
            return blackboard

        model = ModelProvider.get_worker_model()

        # Build context from verified chunks and analysis
        chunk_context = "\n\n".join(
            f"[Source: {c.get('metadata', {}).get('source', 'unknown')}]\n{c.get('document', '')}"
            for c in blackboard.retrieved_chunks
        )

        analysis_context = ""
        for result in blackboard.analysis_results:
            if "summary" in result:
                analysis_context += f"\nAnalysis: {result['summary']}\n"
            for key in ("formulas", "facts", "constraints", "relationships"):
                items = result.get(key, [])
                if items:
                    analysis_context += f"  {key}: {items}\n"

        prompt = (
            f"User Query: {blackboard.query}\n\n"
            f"Verified Source Material:\n{chunk_context}\n\n"
            f"Extracted Analysis:\n{analysis_context}\n\n"
            "Generate a comprehensive, fully grounded answer."
        )

        try:
            answer = await model.generate(
                prompt=prompt,
                system_instruction=GENERATOR_SYSTEM_PROMPT,
                temperature=0.0,
            )

            blackboard.final_answer = answer.strip()

            self._log_trace(
                phase="generation",
                action="generation_complete",
                output_summary=f"Answer length: {len(blackboard.final_answer)} chars",
            )

        except Exception as exc:
            logger.error("Generation failed: %s", exc)
            blackboard.final_answer = INSUFFICIENT_ANSWER
            self._log_trace(
                phase="generation",
                action="generation_error",
                decision=f"Error: {exc}",
            )

        blackboard.add_trace(
            agent_name=self._name,
            phase="generation",
            action="answer_generated",
            details={"answer_length": len(blackboard.final_answer)},
        )

        return blackboard
