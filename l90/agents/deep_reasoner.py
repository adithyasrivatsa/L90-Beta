"""Deep Reasoning Agent — cross-document synthesis for Partial mode.

This agent sits between verification and generation in the Partial pipeline.
It synthesizes a coherent reasoning chain across all evidence, identifies
implicit connections, and builds step-by-step derivation paths.
"""

from __future__ import annotations

import logging
from typing import Any

from l90.agents.base import BaseAgent
from l90.blackboard.blackboard import Blackboard
from l90.models.provider import ModelProvider
from l90.tracing.logger import ReasoningTraceLogger

logger = logging.getLogger(__name__)

DEEP_REASONING_SYSTEM_PROMPT = """\
You are the Deep Reasoning Agent of L90, a NASA/DARPA-grade scientific system
used for critical physics, mathematics, and engineering decisions.

Your role: synthesize ALL verified evidence into a coherent reasoning chain.
Think like a senior research scientist connecting dots across multiple papers.

Given: verified chunks, analysis results, and extracted formulas/facts.

You MUST produce a JSON object:
{
  "reasoning_chain": [
    {"step": 1, "description": "...", "evidence": "chunk/formula ref", "conclusion": "..."},
    {"step": 2, "description": "...", "evidence": "...", "conclusion": "..."},
    ...
  ],
  "cross_connections": [
    {"source_a": "...", "source_b": "...", "relationship": "..."},
    ...
  ],
  "derivation_path": "step-by-step LaTeX derivation if mathematical",
  "key_insight": "the most important synthesized insight",
  "confidence_in_synthesis": 0.0-1.0,
  "latex_equations": ["equation1 in LaTeX", "equation2 in LaTeX", ...]
}

Rules:
1. Every reasoning step MUST cite evidence from verified chunks
2. Cross-connections must be genuinely insightful, not trivial
3. For physics/math: show the derivation path in LaTeX notation
4. Use $...$ for inline LaTeX and $$...$$ for display equations
5. If you spot dimensional inconsistencies — FLAG them prominently
6. Rate your synthesis confidence honestly
"""


class DeepReasoningAgent(BaseAgent):
    """Cross-document synthesis agent for Partial mode.

    This agent creates a coherent reasoning chain from scattered evidence,
    building the intellectual bridge the generator needs for rich answers.
    """

    def __init__(self, trace_logger: ReasoningTraceLogger | None = None) -> None:
        super().__init__(name="DeepReasoningAgent", trace_logger=trace_logger)

    async def execute(self, blackboard: Blackboard) -> Blackboard:
        """Synthesize evidence into a deep reasoning chain."""

        self._log_trace(
            phase="deep_reasoning",
            action="deep_reasoning_start",
            input_summary=(
                f"Chunks={len(blackboard.retrieved_chunks)}, "
                f"Analysis={len(blackboard.analysis_results)}"
            ),
        )

        if not blackboard.analysis_results:
            self._log_trace(
                phase="deep_reasoning",
                action="deep_reasoning_skip",
                decision="No analysis results to synthesize",
            )
            return blackboard

        model = ModelProvider.get_worker_model()

        # Build rich context
        chunk_context = "\n\n".join(
            f"[Chunk {i} | Source: {c.get('metadata', {}).get('source', 'unknown')}]\n"
            f"{c.get('document', '')}"
            for i, c in enumerate(blackboard.retrieved_chunks)
        )

        analysis_context = ""
        for i, result in enumerate(blackboard.analysis_results):
            analysis_context += f"\n[Analysis {i}]\n"
            for key in ("formulas", "facts", "constraints", "relationships", "summary"):
                if key in result:
                    analysis_context += f"  {key}: {result[key]}\n"

        # Include verification insights
        verification_context = ""
        for v in blackboard.verification_results:
            if "summary" in v:
                verification_context += f"Verification: {v['summary']}\n"

        prompt = (
            f"User Query: {blackboard.query}\n\n"
            f"Verified Source Material:\n{chunk_context}\n\n"
            f"Analysis Results:\n{analysis_context}\n\n"
            f"Verification Notes:\n{verification_context}\n\n"
            "Synthesize a deep reasoning chain across ALL evidence. "
            "Identify cross-connections and build a derivation path."
        )

        try:
            result = await model.generate_json(
                prompt=prompt,
                system_instruction=DEEP_REASONING_SYSTEM_PROMPT,
                temperature=0.15,  # slight creativity for synthesis
                max_tokens=8192,
            )

            reasoning_entry = {
                "reasoning_chain": result.get("reasoning_chain", []),
                "cross_connections": result.get("cross_connections", []),
                "derivation_path": result.get("derivation_path", ""),
                "key_insight": result.get("key_insight", ""),
                "confidence_in_synthesis": result.get("confidence_in_synthesis", 0.0),
            }

            blackboard.deep_reasoning.append(reasoning_entry)

            # Extract LaTeX equations
            latex = result.get("latex_equations", [])
            if latex:
                blackboard.latex_equations.extend(latex)

            self._log_trace(
                phase="deep_reasoning",
                action="deep_reasoning_complete",
                output_summary=(
                    f"Chain steps={len(reasoning_entry['reasoning_chain'])}, "
                    f"Cross-connections={len(reasoning_entry['cross_connections'])}, "
                    f"LaTeX equations={len(latex)}"
                ),
                confidence=reasoning_entry["confidence_in_synthesis"],
            )

        except Exception as exc:
            logger.error("Deep reasoning failed: %s", exc)
            blackboard.deep_reasoning.append({"error": str(exc)})
            self._log_trace(
                phase="deep_reasoning",
                action="deep_reasoning_error",
                decision=f"Error: {exc}",
            )

        blackboard.add_trace(
            agent_name=self._name,
            phase="deep_reasoning",
            action="synthesis_recorded",
            details={"entries": len(blackboard.deep_reasoning)},
        )

        return blackboard
