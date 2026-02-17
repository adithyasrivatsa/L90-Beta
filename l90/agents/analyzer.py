"""Analysis agent — extracts structured information from retrieved chunks.

Enhanced: extracts LaTeX equations, dimensional analysis, and proof steps.
"""

from __future__ import annotations

import logging
from typing import Any

from l90.agents.base import BaseAgent
from l90.blackboard.blackboard import Blackboard
from l90.models.provider import ModelProvider
from l90.tracing.logger import ReasoningTraceLogger

logger = logging.getLogger(__name__)

ANALYSIS_SYSTEM_PROMPT = """\
You are an Analysis Agent in L90, a NASA/DARPA-grade scientific RAG system.

Given a set of retrieved document chunks, extract ALL of the following:

1. **Formulas**: Extract ALL mathematical and physics equations.
   - MUST be in LaTeX format (e.g., $E = mc^2$, $\\frac{d}{dx}f(x)$).
   - Preserve variable definitions.

2. **Facts**: Key factual statements, empirical data, and constants.
   - Include units for all physical quantities.

3. **Constraints**: Limitations, boundary conditions, validity ranges.

4. **Relationships**: Causal links, correlations, and hierarchical connections.

5. **Proof Steps**: If the text describes a derivation or proof, extract the logical steps.

You MUST respond with a JSON object:
{
  "formulas": ["$latex_eq1$", "$latex_eq2$", ...],
  "facts": ["fact1", "fact2", ...],
  "constraints": ["constraint1", ...],
  "relationships": ["relationship1", ...],
  "proof_steps": ["step1", "step2", ...],
  "summary": "brief synthesis of findings"
}

Extract ONLY what is explicitly stated in the chunks. Do NOT infer or hallucinate.
"""


class AnalysisAgent(BaseAgent):
    """Extracts formulas (LaTeX), facts, constraints, and relationships."""

    def __init__(self, trace_logger: ReasoningTraceLogger | None = None) -> None:
        super().__init__(name="AnalysisAgent", trace_logger=trace_logger)

    async def execute(self, blackboard: Blackboard) -> Blackboard:
        """Analyze retrieved chunks and write structured results."""

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
            "Analyze these chunks and extract structured information with LaTeX formulas."
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
                "proof_steps": result.get("proof_steps", []),
                "summary": result.get("summary", ""),
                "chunks_analyzed": len(blackboard.retrieved_chunks),
            }

            blackboard.analysis_results.append(analysis_entry)

            # Also push extracted equations to the specialized list
            if analysis_entry["formulas"]:
                blackboard.latex_equations.extend(analysis_entry["formulas"])

            self._log_trace(
                phase="analysis",
                action="analysis_complete",
                output_summary=(
                    f"Extracted {len(analysis_entry['formulas'])} formulas, "
                    f"{len(analysis_entry['facts'])} facts, "
                    f"{len(analysis_entry['proof_steps'])} proof steps"
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
