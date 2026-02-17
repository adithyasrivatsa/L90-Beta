"""LangGraph node functions — each wraps an agent's execute() method."""

from __future__ import annotations

import logging
from typing import Any

from l90.agents.analyzer import AnalysisAgent
from l90.agents.corrector import CorrectionAgent
from l90.agents.deep_reasoner import DeepReasoningAgent
from l90.agents.generator import GeneratorAgent
from l90.agents.manager import ManagerAgent
from l90.agents.math_executor import MathExecutorAgent
from l90.agents.retriever import RetrieverAgent
from l90.agents.verifier import VerificationAgent
from l90.blackboard.blackboard import Blackboard
from l90.graph.state import GraphState
from l90.grounding.enforcer import GroundingEnforcer
from l90.tracing.logger import ReasoningTraceLogger
from l90.vectordb.chroma_store import ChromaStore

logger = logging.getLogger(__name__)

# ── Shared instances (created once per graph run) ──────────────
_trace_logger: ReasoningTraceLogger | None = None
_store: ChromaStore | None = None


def set_shared_resources(
    trace_logger: ReasoningTraceLogger,
    store: ChromaStore,
) -> None:
    """Set shared resources for the node functions."""
    global _trace_logger, _store
    _trace_logger = trace_logger
    _store = store


def _state_to_blackboard(state: GraphState) -> Blackboard:
    """Convert LangGraph state dict to a Blackboard object."""
    bb = Blackboard()
    bb.session_id = state.get("session_id", bb.session_id)
    bb.query = state.get("query", "")
    bb.mode = state.get("mode", "")
    bb.user_id = state.get("user_id", "")
    bb.workspace_id = state.get("workspace_id", "")
    bb.allowed_sources = state.get("allowed_sources", [])
    bb.retrieved_chunks = state.get("retrieved_chunks", [])
    bb.analysis_results = state.get("analysis_results", [])
    bb.verification_results = state.get("verification_results", [])
    bb.verification_passed = state.get("verification_passed", False)
    bb.correction_results = state.get("correction_results", [])
    bb.correction_loop_count = state.get("correction_loop_count", 0)
    bb.confidence_score = state.get("confidence_score", 0.0)
    bb.final_answer = state.get("final_answer", "")
    bb.reasoning_trace = state.get("reasoning_trace", [])
    bb.execution_plan = state.get("execution_plan", {})
    bb.grounding_report = state.get("grounding_report", {})
    bb.code_verification = state.get("code_verification", {})
    bb.deep_reasoning = state.get("deep_reasoning", [])
    bb.latex_equations = state.get("latex_equations", [])
    bb.metadata = state.get("metadata", {})
    return bb


def _blackboard_to_state(bb: Blackboard) -> dict[str, Any]:
    """Convert a Blackboard back to a state-update dict for LangGraph."""
    return bb.to_dict()


# ── Node functions ─────────────────────────────────────────────

async def manager_node(state: GraphState) -> dict[str, Any]:
    """Manager node — analyzes query, sets mode, creates execution plan."""
    bb = _state_to_blackboard(state)
    agent = ManagerAgent(trace_logger=_trace_logger)
    bb = await agent.execute(bb)
    return _blackboard_to_state(bb)


async def retrieval_node(state: GraphState) -> dict[str, Any]:
    """Retrieval node — searches allowed collections."""
    bb = _state_to_blackboard(state)
    agent = RetrieverAgent(store=_store, trace_logger=_trace_logger)
    bb = await agent.execute(bb)
    return _blackboard_to_state(bb)


async def analysis_node(state: GraphState) -> dict[str, Any]:
    """Analysis node — extracts structured information from chunks."""
    bb = _state_to_blackboard(state)
    agent = AnalysisAgent(trace_logger=_trace_logger)
    bb = await agent.execute(bb)
    return _blackboard_to_state(bb)


async def verification_node(state: GraphState) -> dict[str, Any]:
    """Verification node — checks correctness and grounding."""
    bb = _state_to_blackboard(state)
    agent = VerificationAgent(trace_logger=_trace_logger)
    bb = await agent.execute(bb)
    return _blackboard_to_state(bb)


async def correction_node(state: GraphState) -> dict[str, Any]:
    """Correction node — fixes issues when verification fails."""
    bb = _state_to_blackboard(state)
    agent = CorrectionAgent(store=_store, trace_logger=_trace_logger)
    bb = await agent.execute(bb)
    return _blackboard_to_state(bb)


async def generator_node(state: GraphState) -> dict[str, Any]:
    """Generator node — produces the final grounded answer."""
    bb = _state_to_blackboard(state)
    agent = GeneratorAgent(trace_logger=_trace_logger)
    bb = await agent.execute(bb)
    return _blackboard_to_state(bb)


async def grounding_node(state: GraphState) -> dict[str, Any]:
    """Grounding node — enforces strict grounding on the generated answer."""
    bb = _state_to_blackboard(state)
    enforcer = GroundingEnforcer(trace_logger=_trace_logger)
    bb = await enforcer.execute(bb)
    return _blackboard_to_state(bb)


# ── New Nodes ──────────────────────────────────────────────────

async def math_executor_node(state: GraphState) -> dict[str, Any]:
    """Math Executor node — runs Python to prove math/physics results."""
    bb = _state_to_blackboard(state)
    agent = MathExecutorAgent(trace_logger=_trace_logger)
    bb = await agent.execute(bb)
    return _blackboard_to_state(bb)


async def deep_reasoning_node(state: GraphState) -> dict[str, Any]:
    """Deep Reasoning node — synthesizes reasoning chain via LLM."""
    bb = _state_to_blackboard(state)
    agent = DeepReasoningAgent(trace_logger=_trace_logger)
    bb = await agent.execute(bb)
    return _blackboard_to_state(bb)


# ── Conditional routing functions ──────────────────────────────

def should_correct_or_continue(state: GraphState) -> str:
    """Route after verification: 'corrector' if failed, 'continue' if passed."""
    from l90 import config

    # If verification passed, move forward
    if state.get("verification_passed", False):
        return "continue"

    loop_count = state.get("correction_loop_count", 0)
    max_loops = config.MAX_CORRECTION_LOOPS

    if loop_count >= max_loops:
        logger.warning(
            "Max correction loops (%d) reached. Proceeding force-forward.",
            max_loops,
        )
        return "continue"

    return "corrector"
