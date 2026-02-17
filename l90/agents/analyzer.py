"""Analysis agent — extracts structured information from retrieved chunks."""

from __future__ import annotations

import logging
from typing import Any

from l90.agents.base import BaseAgent
from l90.blackboard.blackboard import Blackboard
from l90.models.provider import ModelProvider
from l90.tracing.logger import ReasoningTraceLogger

logger = logging.getLogger(__name__)

ANALYSIS_SYSTEM_PROMPT = """\
You are an Analysis Agent in L90, a zero-hallucination scientific RAG system.

Given a set of retrieved document chunks, extract ALL of the following:
- formulas (mathematical/physics equations found in the text)
- facts (key factual statements)
- constraints (limitations, conditions, boundaries mentioned)
- relationships (connections between concepts, cause-effect)

You MUST respond with a JSON object:
{
  "formulas": ["formula1", "formula2", ...],
  "facts": ["fact1", "fact2", ...],
  "constraints": ["constraint1", ...],
  "relationships": ["relationship1", ...],
  "summary": "brief synthesis of findings"
}

Extract ONLY what is explicitly stated in the chunks. Do NOT infer or hallucinate.
"""


class AnalysisAgent(BaseAgent):
    """Extracts formulas, facts, constraints, and relationships from retrieved chunks."""

    def __init__(self, trace_logger: ReasoningTraceLogger | None = None) -> None:
        super().__init__(name="AnalysisAgent", trace_logger=trace_logger)

    async def execute(self, blackboard: Blackboard) -> Blackboard:
        """Analyze retrieved chunks and write structured results to the Blackboard."""

        self._log_trace(
            phase="analysis",
            action="analysis_start",
            input_summary=f"Analyzing {len(blackboard.retrieved_chunks)} chunks",
        )

        if not blackboard.retrieved_chunks:
            self._log_trace(
                phase="analysis",
                action="analysis_skip",
                decision="No chunks to analyze",
            )
            return blackboard

        model = ModelProvider.get_worker_model()

        # Build context from retrieved chunks
        chunk_texts = []
        for i, chunk in enumerate(blackboard.retrieved_chunks):
            source = chunk.get("metadata", {}).get("source", "unknown")
            text = chunk.get("document", "")
            chunk_texts.append(f"[Chunk {i + 1} | Source: {source}]\n{text}")

        context = "\n\n---\n\n".join(chunk_texts)

        prompt = (
            f"User Query: {blackboard.query}\n\n"
            f"Retrieved Chunks:\n{context}\n\n"
            "Analyze these chunks and extract structured information."
        )

        try:
            result = await model.generate_json(
                prompt=prompt,
                system_instruction=ANALYSIS_SYSTEM_PROMPT,
                temperature=0.0,
            )

            analysis_entry = {
                "formulas": result.get("formulas", []),
                "facts": result.get("facts", []),
                "constraints": result.get("constraints", []),
                "relationships": result.get("relationships", []),
                "summary": result.get("summary", ""),
                "chunks_analyzed": len(blackboard.retrieved_chunks),
            }

            blackboard.analysis_results.append(analysis_entry)

            self._log_trace(
                phase="analysis",
                action="analysis_complete",
                output_summary=(
                    f"Extracted {len(analysis_entry['formulas'])} formulas, "
                    f"{len(analysis_entry['facts'])} facts, "
                    f"{len(analysis_entry['constraints'])} constraints, "
                    f"{len(analysis_entry['relationships'])} relationships"
                ),
            )

        except Exception as exc:
            logger.error("Analysis failed: %s", exc)
            blackboard.analysis_results.append({
                "error": str(exc),
                "chunks_analyzed": len(blackboard.retrieved_chunks),
            })
            self._log_trace(
                phase="analysis",
                action="analysis_error",
                decision=f"Error: {exc}",
            )

        blackboard.add_trace(
            agent_name=self._name,
            phase="analysis",
            action="analysis_recorded",
            details={"results_count": len(blackboard.analysis_results)},
        )

        return blackboard
