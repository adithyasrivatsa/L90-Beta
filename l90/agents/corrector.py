"""Correction agent — additional retrieval, reasoning correction, conflict resolution."""

from __future__ import annotations

import logging
from typing import Any

from l90.agents.base import BaseAgent
from l90.blackboard.blackboard import Blackboard
from l90.models.provider import ModelProvider
from l90.tracing.logger import ReasoningTraceLogger
from l90.vectordb.chroma_store import ChromaStore

logger = logging.getLogger(__name__)

CORRECTION_SYSTEM_PROMPT = """\
You are a Correction Agent in L90, a zero-hallucination scientific RAG system.

Verification has FAILED. Your job is to:
1. Identify what went wrong (look at verification issues)
2. Suggest additional retrieval queries to fill knowledge gaps
3. Correct any reasoning errors found in the analysis
4. Resolve conflicts between sources

You MUST respond with a JSON object:
{
  "additional_queries": ["query1", "query2", ...],
  "corrected_facts": ["corrected_fact1", ...],
  "resolved_conflicts": ["resolution1", ...],
  "corrections_summary": "what was corrected and why"
}

Only correct what is necessary. Do not fabricate new information.
"""


class CorrectionAgent(BaseAgent):
    """Performs corrections when verification fails — retrieves more data, fixes reasoning."""

    def __init__(
        self,
        store: ChromaStore | None = None,
        trace_logger: ReasoningTraceLogger | None = None,
    ) -> None:
        super().__init__(name="CorrectionAgent", trace_logger=trace_logger)
        self._store = store or ChromaStore()

    async def execute(self, blackboard: Blackboard) -> Blackboard:
        """Run correction: analyze failures, retrieve more, correct reasoning."""

        blackboard.correction_loop_count += 1

        self._log_trace(
            phase="correction",
            action="correction_start",
            input_summary=f"Loop {blackboard.correction_loop_count}, Verification issues present",
        )

        model = ModelProvider.get_worker_model()

        # Build context from verification failures
        verification_summary = ""
        for v in blackboard.verification_results:
            for key in ("math_check", "physics_check", "logic_check", "grounding_check", "summary"):
                if key in v:
                    verification_summary += f"  {key}: {v[key]}\n"

        prompt = (
            f"User Query: {blackboard.query}\n\n"
            f"Verification Failures:\n{verification_summary}\n\n"
            f"Current Analysis:\n{blackboard.analysis_results}\n\n"
            "Perform corrections."
        )

        try:
            result = await model.generate_json(
                prompt=prompt,
                system_instruction=CORRECTION_SYSTEM_PROMPT,
                temperature=0.0,
            )

            correction_entry: dict[str, Any] = {
                "loop": blackboard.correction_loop_count,
                "additional_queries": result.get("additional_queries", []),
                "corrected_facts": result.get("corrected_facts", []),
                "resolved_conflicts": result.get("resolved_conflicts", []),
                "corrections_summary": result.get("corrections_summary", ""),
            }

            # Perform additional retrieval if suggested
            additional_queries = result.get("additional_queries", [])
            new_chunks = []
            for query in additional_queries[:3]:  # Cap at 3 additional queries
                chunks = self._store.query_multiple_collections(
                    collection_names=blackboard.allowed_sources,
                    query_text=query,
                    n_results=5,
                )
                new_chunks.extend(chunks)

            if new_chunks:
                blackboard.retrieved_chunks.extend(new_chunks)
                correction_entry["new_chunks_retrieved"] = len(new_chunks)

            blackboard.correction_results.append(correction_entry)

            # Reset verification flag so it gets re-checked
            blackboard.verification_passed = False

            self._log_trace(
                phase="correction",
                action="correction_complete",
                output_summary=(
                    f"Additional queries: {len(additional_queries)}, "
                    f"New chunks: {len(new_chunks)}, "
                    f"Corrections: {correction_entry['corrections_summary'][:100]}"
                ),
            )

        except Exception as exc:
            logger.error("Correction failed: %s", exc)
            blackboard.correction_results.append({
                "loop": blackboard.correction_loop_count,
                "error": str(exc),
            })
            self._log_trace(
                phase="correction",
                action="correction_error",
                decision=f"Error: {exc}",
            )

        blackboard.add_trace(
            agent_name=self._name,
            phase="correction",
            action="correction_recorded",
            details={"loop": blackboard.correction_loop_count},
        )

        return blackboard
