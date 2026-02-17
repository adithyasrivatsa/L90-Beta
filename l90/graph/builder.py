"""LangGraph graph builder — constructs and compiles the L90 pipeline graph."""

from __future__ import annotations

import logging

from langgraph.graph import END, START, StateGraph

from l90.graph.nodes import (
    analysis_node,
    correction_node,
    generator_node,
    grounding_node,
    manager_node,
    retrieval_node,
    set_shared_resources,
    should_correct_or_generate,
    verification_node,
)
from l90.graph.state import GraphState
from l90.tracing.logger import ReasoningTraceLogger
from l90.vectordb.chroma_store import ChromaStore

logger = logging.getLogger(__name__)


def build_graph(
    store: ChromaStore | None = None,
    trace_logger: ReasoningTraceLogger | None = None,
):
    """Build and compile the L90 LangGraph pipeline.

    Pipeline:
        START → manager → retrieval → analysis → verification
                                                     ↓
                                            [conditional]
                                           ↙            ↘
                                     corrector       generator → grounding → END
                                        ↓
                                   verification (loop)

    Returns:
        Compiled LangGraph runnable.
    """
    # Set up shared resources for node functions
    _store = store or ChromaStore()
    _trace = trace_logger or ReasoningTraceLogger()
    set_shared_resources(trace_logger=_trace, store=_store)

    # ── Build the graph ────────────────────────────────────────
    graph = StateGraph(GraphState)

    # Add nodes
    graph.add_node("manager", manager_node)
    graph.add_node("retrieval", retrieval_node)
    graph.add_node("analysis", analysis_node)
    graph.add_node("verification", verification_node)
    graph.add_node("corrector", correction_node)
    graph.add_node("generator", generator_node)
    graph.add_node("grounding", grounding_node)

    # Add edges — linear pipeline until verification
    graph.add_edge(START, "manager")
    graph.add_edge("manager", "retrieval")
    graph.add_edge("retrieval", "analysis")
    graph.add_edge("analysis", "verification")

    # Conditional: verification → corrector (loop) or generator
    graph.add_conditional_edges(
        "verification",
        should_correct_or_generate,
        {
            "corrector": "corrector",
            "generator": "generator",
        },
    )

    # Correction loops back to verification
    graph.add_edge("corrector", "verification")

    # Generator → grounding → END
    graph.add_edge("generator", "grounding")
    graph.add_edge("grounding", END)

    # ── Compile ────────────────────────────────────────────────
    compiled = graph.compile()
    logger.info("L90 pipeline graph compiled successfully")

    return compiled
