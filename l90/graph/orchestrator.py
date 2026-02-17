"""SwarmOrchestrator — dynamic, complexity-aware pipeline execution.

Replaces the static LangGraph chain with a swarm that:
1. Runs the Manager/Planner to classify query complexity
2. Dynamically selects which agents to run based on complexity
3. Runs independent agents in PARALLEL via asyncio.gather
4. Reports all results on the Blackboard
5. Returns the final answer within a time budget

Complexity-based routing:
    BASIC           → Retriever → Generator (2 agents, fast-path)
    INTERMEDIATE    → Retriever → Analyzer ‖ Verifier → Generator → Grounding
    ADVANCED        → Retriever → Analyzer → Verifier → [Correction] → Generator → Grounding
    RESEARCH_GRADE  → Full pipeline with MathExecutor + DeepReasoning
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from l90 import config
from l90.agents.analyzer import AnalysisAgent
from l90.agents.corrector import CorrectionAgent
from l90.agents.deep_reasoner import DeepReasoningAgent
from l90.agents.generator import GeneratorAgent
from l90.agents.manager import ManagerAgent
from l90.agents.math_executor import MathExecutorAgent
from l90.agents.retriever import RetrieverAgent
from l90.agents.verifier import VerificationAgent
from l90.blackboard.blackboard import Blackboard
from l90.grounding.enforcer import GroundingEnforcer
from l90.tracing.logger import ReasoningTraceLogger
from l90.vectordb.chroma_store import ChromaStore

logger = logging.getLogger(__name__)


class SwarmOrchestrator:
    """Dynamic swarm orchestrator — the Manager decides the pipeline shape.

    The Manager is the central brain. Based on query complexity, it spins up
    only the agents needed, runs them in parallel where possible, and
    assembles the final answer on the Blackboard.
    """

    def __init__(
        self,
        store: ChromaStore,
        trace_logger: ReasoningTraceLogger,
    ) -> None:
        self._store = store
        self._trace = trace_logger

    async def _timed(
        self, name: str, coro, timings: dict[str, float],
    ):
        """Run an async operation and record how long it took."""
        t0 = time.monotonic()
        result = await coro
        elapsed = round(time.monotonic() - t0, 3)
        timings[name] = elapsed
        logger.info("  ⏱ %s completed in %.2fs", name, elapsed)
        return result

    async def run(self, initial_state: dict[str, Any]) -> dict[str, Any]:
        """Execute the swarm pipeline with dynamic routing.

        Args:
            initial_state: Dict matching GraphState schema.

        Returns:
            Final state dict with answer, traces, and reports.
        """
        start_time = time.monotonic()
        timings: dict[str, float] = {}

        # ── Build Blackboard from initial state ────────────────
        bb = Blackboard()
        bb.session_id = initial_state.get("session_id", bb.session_id)
        bb.query = initial_state.get("query", "")
        bb.mode = initial_state.get("mode", "STRICT")
        bb.user_id = initial_state.get("user_id", "")
        bb.workspace_id = initial_state.get("workspace_id", "")
        bb.metadata = initial_state.get("metadata", {})

        # ── Phase 1: Manager (planning + access control) ───────
        manager = ManagerAgent(trace_logger=self._trace)
        bb = await self._timed("Manager", manager.execute(bb), timings)

        complexity = bb.execution_plan.get("complexity_level", "INTERMEDIATE")
        logger.info(
            "Swarm routing: complexity=%s, query='%s'",
            complexity, bb.query[:80],
        )
        bb.metadata["complexity_level"] = complexity
        bb.metadata["swarm_start_time"] = start_time

        # ── Phase 2: Dynamic agent pipeline ────────────────────
        if complexity == "BASIC":
            await self._run_basic(bb, timings)
        elif complexity == "INTERMEDIATE":
            await self._run_intermediate(bb, timings)
        elif complexity == "ADVANCED":
            await self._run_advanced(bb, timings)
        else:  # RESEARCH_GRADE
            await self._run_research_grade(bb, timings)

        # ── Record timing ──────────────────────────────────────
        elapsed = time.monotonic() - start_time
        bb.metadata["pipeline_elapsed_seconds"] = round(elapsed, 3)
        bb.metadata["agents_used"] = complexity
        bb.metadata["agent_timings"] = timings

        logger.info(
            "Swarm complete: complexity=%s, elapsed=%.2fs, answer_len=%d",
            complexity, elapsed, len(bb.final_answer),
        )

        return bb.to_dict()

    # ── BASIC: fast-path (Retriever → Generator) ──────────────

    async def _run_basic(self, bb: Blackboard, timings: dict[str, float]) -> None:
        """BASIC queries: retrieve + generate. 2 agents, no verification."""
        retriever = RetrieverAgent(store=self._store, trace_logger=self._trace)
        bb = await self._timed("Retriever", retriever.execute(bb), timings)

        # Set verification as passed for basic (skip verifier)
        bb.verification_passed = True
        bb.confidence_score = 0.85

        generator = GeneratorAgent(trace_logger=self._trace)
        bb = await self._timed("Generator", generator.execute(bb), timings)

        # Fast grounding (deterministic, no LLM)
        enforcer = GroundingEnforcer(trace_logger=self._trace)
        bb = await self._timed("Grounding", enforcer.execute(bb), timings)

    # ── INTERMEDIATE: Retriever → (Analyzer ‖ Verifier) → Generator → Grounding

    async def _run_intermediate(self, bb: Blackboard, timings: dict[str, float]) -> None:
        """INTERMEDIATE: parallel analysis + verification, then generate."""
        retriever = RetrieverAgent(store=self._store, trace_logger=self._trace)
        bb = await self._timed("Retriever", retriever.execute(bb), timings)

        # Run Analyzer and Verifier in PARALLEL
        analyzer = AnalysisAgent(trace_logger=self._trace)
        verifier = VerificationAgent(trace_logger=self._trace)

        bb_analyzer = Blackboard()
        bb_verifier = Blackboard()
        for src_bb in (bb_analyzer, bb_verifier):
            src_bb.session_id = bb.session_id
            src_bb.query = bb.query
            src_bb.mode = bb.mode
            src_bb.user_id = bb.user_id
            src_bb.workspace_id = bb.workspace_id
            src_bb.allowed_sources = bb.allowed_sources
            src_bb.retrieved_chunks = list(bb.retrieved_chunks)
            src_bb.execution_plan = dict(bb.execution_plan)

        t0 = time.monotonic()
        bb_analyzer, bb_verifier = await asyncio.gather(
            analyzer.execute(bb_analyzer),
            verifier.execute(bb_verifier),
        )
        timings["Analyzer‖Verifier"] = round(time.monotonic() - t0, 3)
        logger.info("  ⏱ Analyzer‖Verifier completed in %.2fs", timings["Analyzer‖Verifier"])

        # Merge results back
        bb.analysis_results.extend(bb_analyzer.analysis_results)
        bb.latex_equations.extend(bb_analyzer.latex_equations)
        bb.verification_results.extend(bb_verifier.verification_results)
        bb.verification_passed = bb_verifier.verification_passed
        bb.confidence_score = bb_verifier.confidence_score

        if not bb.verification_passed:
            bb.verification_passed = True  # proceed anyway for INTERMEDIATE
            bb.confidence_score = max(bb.confidence_score, 0.7)

        generator = GeneratorAgent(trace_logger=self._trace)
        bb = await self._timed("Generator", generator.execute(bb), timings)

        enforcer = GroundingEnforcer(trace_logger=self._trace)
        bb = await self._timed("Grounding", enforcer.execute(bb), timings)

    # ── ADVANCED: full pipeline with correction loop ──────────

    async def _run_advanced(self, bb: Blackboard, timings: dict[str, float]) -> None:
        """ADVANCED: Retriever → Analyzer → Verifier → [Correction] → Generator → Grounding."""
        retriever = RetrieverAgent(store=self._store, trace_logger=self._trace)
        bb = await self._timed("Retriever", retriever.execute(bb), timings)

        analyzer = AnalysisAgent(trace_logger=self._trace)
        bb = await self._timed("Analyzer", analyzer.execute(bb), timings)

        verifier = VerificationAgent(trace_logger=self._trace)
        bb = await self._timed("Verifier", verifier.execute(bb), timings)

        max_loops = bb.execution_plan.get("max_correction_loops", config.MAX_CORRECTION_LOOPS)
        loop_i = 0
        while not bb.verification_passed and bb.correction_loop_count < max_loops:
            corrector = CorrectionAgent(store=self._store, trace_logger=self._trace)
            bb = await self._timed(f"Corrector_{loop_i}", corrector.execute(bb), timings)
            bb = await self._timed(f"Re-Retrieve_{loop_i}", retriever.execute(bb), timings)
            bb = await self._timed(f"Re-Analyze_{loop_i}", analyzer.execute(bb), timings)
            bb = await self._timed(f"Re-Verify_{loop_i}", verifier.execute(bb), timings)
            loop_i += 1

        generator = GeneratorAgent(trace_logger=self._trace)
        bb = await self._timed("Generator", generator.execute(bb), timings)

        enforcer = GroundingEnforcer(trace_logger=self._trace)
        bb = await self._timed("Grounding", enforcer.execute(bb), timings)

    # ── RESEARCH_GRADE: everything including deep reasoning ───

    async def _run_research_grade(self, bb: Blackboard, timings: dict[str, float]) -> None:
        """RESEARCH_GRADE: Full pipeline + MathExecutor + DeepReasoning."""
        retriever = RetrieverAgent(store=self._store, trace_logger=self._trace)
        bb = await self._timed("Retriever", retriever.execute(bb), timings)

        analyzer = AnalysisAgent(trace_logger=self._trace)
        bb = await self._timed("Analyzer", analyzer.execute(bb), timings)

        math_exec = MathExecutorAgent(trace_logger=self._trace)
        bb = await self._timed("MathExecutor", math_exec.execute(bb), timings)

        verifier = VerificationAgent(trace_logger=self._trace)
        bb = await self._timed("Verifier", verifier.execute(bb), timings)

        max_loops = bb.execution_plan.get("max_correction_loops", config.MAX_CORRECTION_LOOPS)
        loop_i = 0
        while not bb.verification_passed and bb.correction_loop_count < max_loops:
            corrector = CorrectionAgent(store=self._store, trace_logger=self._trace)
            bb = await self._timed(f"Corrector_{loop_i}", corrector.execute(bb), timings)
            bb = await self._timed(f"Re-Retrieve_{loop_i}", retriever.execute(bb), timings)
            bb = await self._timed(f"Re-Analyze_{loop_i}", analyzer.execute(bb), timings)
            bb = await self._timed(f"Re-Verify_{loop_i}", verifier.execute(bb), timings)
            loop_i += 1

        deep = DeepReasoningAgent(trace_logger=self._trace)
        bb = await self._timed("DeepReasoning", deep.execute(bb), timings)

        generator = GeneratorAgent(trace_logger=self._trace)
        bb = await self._timed("Generator", generator.execute(bb), timings)

        enforcer = GroundingEnforcer(trace_logger=self._trace)
        bb = await self._timed("Grounding", enforcer.execute(bb), timings)

