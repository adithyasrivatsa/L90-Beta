"""LangGraph state schema — the structured state that flows through the graph."""

from __future__ import annotations

from typing import Any, Annotated
from typing_extensions import TypedDict

import operator


def _merge_lists(a: list, b: list) -> list:
    """Merge two lists by extending (for reducer annotation)."""
    return a + b


class GraphState(TypedDict, total=False):
    """LangGraph state schema, mirroring the Blackboard structure.

    Uses LangGraph reducers for list fields so parallel nodes can
    append without overwriting each other.
    """

    # Identity
    session_id: str
    query: str
    mode: str
    user_id: str
    workspace_id: str

    # Access control
    allowed_sources: list[str]

    # Retrieval (reducer: merge lists from parallel retrieval agents)
    retrieved_chunks: Annotated[list[dict[str, Any]], _merge_lists]

    # Analysis
    analysis_results: Annotated[list[dict[str, Any]], _merge_lists]

    # Verification
    verification_results: Annotated[list[dict[str, Any]], _merge_lists]
    verification_passed: bool

    # Correction
    correction_results: Annotated[list[dict[str, Any]], _merge_lists]
    correction_loop_count: int

    # Output
    confidence_score: float
    final_answer: str

    # Audit
    reasoning_trace: Annotated[list[dict[str, Any]], _merge_lists]
    execution_plan: dict[str, Any]
    grounding_report: dict[str, Any]

    # Code verification (math/physics proofs)
    code_verification: dict[str, Any]

    # Deep reasoning (Partial mode synthesis)
    deep_reasoning: Annotated[list[dict[str, Any]], _merge_lists]

    # LaTeX equations extracted
    latex_equations: Annotated[list[str], _merge_lists]

    # Metadata
    metadata: dict[str, Any]
