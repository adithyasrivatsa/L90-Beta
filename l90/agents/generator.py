"""Generator agent — produces the final grounded answer from verified Blackboard content.

Enhanced: Dual-mode prompts (STRICT vs PARTIAL).
STRICT: Cold, factual, cite-heavy.
PARTIAL: Warm, friendly, latex-rich, deep reasoning synthesis.
"""

from __future__ import annotations

import logging
from typing import Any

from l90 import config
from l90.agents.base import BaseAgent
from l90.blackboard.blackboard import Blackboard
from l90.models.provider import ModelProvider
from l90.tracing.logger import ReasoningTraceLogger

logger = logging.getLogger(__name__)

# ── STRICT Mode System Prompt ──────────────────────────────────
GENERATOR_STRICT_PROMPT = """\
You are the Generator Agent of L90, a zero-hallucination scientific RAG system.
Mode: STRICT (Fortress Mode).

CRITICAL RULES:
1. Answer ONLY using the verified information provided below.
2. Every claim MUST be cited with precise source references using this format:
   [Source: filename, Page X, Chunk N]
   - If page number is available, always include it
   - If only chunk index is available, use: [Source: filename, Chunk N]
   - NEVER use just [Source: filename] alone
3. If the information is insufficient, respond EXACTLY with:
   "Insufficient verified information."
4. Do NOT use your prior knowledge. Do NOT hallucinate.
5. For mathematical/physics content, reproduce formulas exactly as found (using LaTeX $...$).
6. Maintain a professional, objective, and neutral tone.
7. No conversational filler. Direct to the point.

Format your answer clearly with proper structure.
"""

# ── PARTIAL Mode System Prompt ─────────────────────────────────
GENERATOR_PARTIAL_PROMPT = """\
You are the Generator Agent of L90, a NASA/DARPA-grade scientific RAG system.
Mode: PARTIAL (Genius Mode).

Your persona is a brilliant, friendly, and enthusiastic senior research scientist. 🧪✨
You are talking to another expert, so be precise but warm.

CRITICAL RULES:
1. Ground your answer in the provided verified chunks and deep reasoning synthesis.
2. Use LaTeX for ALL mathematical expressions:
   - Inline: $E=mc^2$
   - Block: $$ F = ma $$
3. Show your reasoning! Explain the "why" and "how" behind the answer.
4. If you used Python code verification, mention the proven result.
5. Connect the dots. Use the "Deep Reasoning" insights provided.
6. Cite sources precisely using this format:
   [Source: filename, Page X, Chunk N]
   - Example: "As shown in [Source: physics.pdf, Page 12, Chunk 3]..."
   - If page is unavailable: [Source: doc.txt, Chunk 5]
   - NEVER use just [Source: filename] alone
7. If data is missing for a specific part, admit it gracefully, but answer what you can.
8. Be friendly! Use clear headings and structured lists.

Goal: Provide a comprehensive, provably correct, and beautifully formatted scientific answer.
"""

INSUFFICIENT_ANSWER = "Insufficient verified information."


class GeneratorAgent(BaseAgent):
    """Produces the final grounded answer.

    Behaves differently based on the pipeline mode (STRICT vs PARTIAL).
    """

    def __init__(self, trace_logger: ReasoningTraceLogger | None = None) -> None:
        super().__init__(name="GeneratorAgent", trace_logger=trace_logger)

    async def execute(self, blackboard: Blackboard) -> Blackboard:
        """Generate the final answer from verified Blackboard content."""

        self._log_trace(
            phase="generation",
            action="generation_start",
            input_summary=(
                f"Mode={blackboard.mode}, "
                f"Verified={blackboard.verification_passed}, "
                f"Chunks={len(blackboard.retrieved_chunks)}"
            ),
        )

        # STRICT mode guard: if verification never passed and we have no data
        if blackboard.mode == "STRICT" and not blackboard.retrieved_chunks:
            blackboard.final_answer = INSUFFICIENT_ANSWER
            self._log_trace(
                phase="generation",
                action="generation_insufficient",
                decision="No retrieved chunks available in STRICT mode",
            )
            return blackboard

        model = ModelProvider.get_worker_model()

        # ── Build Context ──────────────────────────────────────────

        # 1. Verified Chunks — with page number, chunk index, and relevance
        chunk_lines: list[str] = []
        for c in blackboard.retrieved_chunks:
            meta = c.get("metadata", {})
            source = meta.get("source", "unknown")
            page = meta.get("page_number", None)
            chunk_idx = meta.get("chunk_index", "?")
            distance = c.get("distance", None)

            # Build a rich header for each chunk
            header_parts = [f"Source: {source}"]
            if page is not None:
                header_parts.append(f"Page: {page}")
            header_parts.append(f"Chunk: {chunk_idx}")
            if distance is not None:
                header_parts.append(f"Relevance: {1 - distance:.2f}")

            header = " | ".join(header_parts)
            chunk_lines.append(f"[{header}]\n{c.get('document', '')}")

        chunk_context = "\n\n".join(chunk_lines)

        # 2. Analysis Results
        analysis_context = ""
        for result in blackboard.analysis_results:
            if "summary" in result:
                analysis_context += f"\nAnalysis: {result['summary']}\n"
            for key in ("formulas", "facts", "constraints", "proof_steps"):
                items = result.get(key, [])
                if items:
                    analysis_context += f"  {key}: {items}\n"

        # 3. Code Verification (if any)
        code_context = ""
        if blackboard.code_verification:
            cv = blackboard.code_verification
            code_context = (
                f"\nPython Code Verification:\n"
                f"Success: {cv.get('success')}\n"
                f"Result: {cv.get('result')}\n"
                f"Output: {cv.get('stdout')}\n"
            )

        # 4. Deep Reasoning (Partial mode only)
        reasoning_context = ""
        if blackboard.deep_reasoning:
            for dr in blackboard.deep_reasoning:
                reasoning_context += (
                    f"\nDeep Reasoning Insights:\n"
                    f"Key Insight: {dr.get('key_insight')}\n"
                    f"Chain: {dr.get('reasoning_chain')}\n"
                    f"Derivation: {dr.get('derivation_path')}\n"
                )

        prompt = (
            f"User Query: {blackboard.query}\n\n"
            f"Verified Source Material:\n{chunk_context}\n\n"
            f"Extracted Analysis:\n{analysis_context}\n\n"
            f"{code_context}\n"
            f"{reasoning_context}\n"
            "Generate the final answer."
        )

        # ── Select Mode Prompt & Config ────────────────────────────
        if blackboard.mode == "STRICT":
            system_prompt = GENERATOR_STRICT_PROMPT
            temperature = config.STRICT_TEMPERATURE
        else:
            system_prompt = GENERATOR_PARTIAL_PROMPT
            temperature = config.PARTIAL_TEMPERATURE

        try:
            answer = await model.generate(
                prompt=prompt,
                system_instruction=system_prompt,
                temperature=temperature,
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
            details={
                "answer_length": len(blackboard.final_answer),
                "mode_used": blackboard.mode
            },
        )

        return blackboard
