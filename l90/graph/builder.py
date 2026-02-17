"""Graph builder — constructs pipelines for L90.

Provides two entry points:
  - build_swarm_orchestrator(): Recommended — dynamic, complexity-aware routing
  - build_graph(): Legacy — static LangGraph pipeline (kept for compatibility)
"""

from __future__ import annotations

import logging
from typing import Any

from l90.graph.orchestrator import SwarmOrchestrator
from l90.tracing.logger import ReasoningTraceLogger
from l90.vectordb.chroma_store import ChromaStore

logger = logging.getLogger(__name__)


def build_swarm_orchestrator(
    store: ChromaStore,
    trace_logger: ReasoningTraceLogger,
) -> SwarmOrchestrator:
    """Build the dynamic SwarmOrchestrator (recommended).

    The orchestrator dynamically routes based on query complexity:
      BASIC          → Retriever → Generator  (fast path)
      INTERMEDIATE   → Retriever → Analyzer ‖ Verifier → Generator → Grounding
      ADVANCED       → Full pipeline with correction loops
      RESEARCH_GRADE → Full pipeline + MathExecutor + DeepReasoning

    Args:
        store: Vector store instance.
        trace_logger: Logger for reasoning traces.

    Returns:
        SwarmOrchestrator instance.
    """
    return SwarmOrchestrator(store=store, trace_logger=trace_logger)


def build_graph(
    store: ChromaStore,
    trace_logger: ReasoningTraceLogger,
    mode: str = "STRICT",
) -> Any:
    """Build the legacy LangGraph pipeline (kept for backwards compatibility).

    DEPRECATED: Use build_swarm_orchestrator() instead.
    """
    from langgraph.graph import StateGraph, END

    from l90.graph.nodes import (
        manager_node,
        retrieval_node,
        analysis_node,
        verification_node,
        correction_node,
        generator_node,
        grounding_node,
        math_executor_node,
        deep_reasoning_node,
        set_shared_resources,
        should_correct_or_continue,
    )
    from l90.graph.state import GraphState

    set_shared_resources(trace_logger, store)

    workflow = StateGraph(GraphState)

    workflow.add_node("manager", manager_node)
    workflow.add_node("retriever", retrieval_node)
    workflow.add_node("analyzer", analysis_node)
    workflow.add_node("verifier", verification_node)
    workflow.add_node("corrector", correction_node)
    workflow.add_node("generator", generator_node)
    workflow.add_node("grounding", grounding_node)

    if mode == "STRICT":
        workflow.set_entry_point("manager")
        workflow.add_edge("manager", "retriever")
        workflow.add_edge("retriever", "analyzer")
        workflow.add_edge("analyzer", "verifier")
        workflow.add_conditional_edges(
            "verifier",
            should_correct_or_continue,
            {"corrector": "corrector", "continue": "generator"},
        )
        workflow.add_edge("corrector", "retriever")
        workflow.add_edge("generator", "grounding")
        workflow.add_edge("grounding", END)
    else:
        workflow.add_node("math_executor", math_executor_node)
        workflow.add_node("deep_reasoning", deep_reasoning_node)
        workflow.set_entry_point("manager")
        workflow.add_edge("manager", "retriever")
        workflow.add_edge("retriever", "analyzer")
        workflow.add_edge("analyzer", "verifier")
        workflow.add_conditional_edges(
            "verifier",
            should_correct_or_continue,
            {"corrector": "corrector", "continue": "math_executor"},
        )
        workflow.add_edge("corrector", "retriever")
        workflow.add_edge("math_executor", "deep_reasoning")
        workflow.add_edge("deep_reasoning", "generator")
        workflow.add_edge("generator", "grounding")
        workflow.add_edge("grounding", END)

    return workflow.compile()

