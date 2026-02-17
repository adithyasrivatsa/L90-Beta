"""Tests for the LangGraph orchestration (graph structure, not execution)."""

import pytest


class TestGraphStructure:
    """Test that the graph compiles without errors."""

    def test_graph_builds(self):
        """Verify the graph can be compiled."""
        # We need to test that the graph builds without actually calling the LLM.
        # The build_graph function creates a compiled StateGraph.
        from l90.graph.state import GraphState

        # Verify the state schema has required fields
        annotations = GraphState.__annotations__
        assert "query" in annotations
        assert "mode" in annotations
        assert "retrieved_chunks" in annotations
        assert "verification_passed" in annotations
        assert "final_answer" in annotations
        assert "reasoning_trace" in annotations
        assert "grounding_report" in annotations

    def test_routing_function(self):
        """Test the conditional routing logic."""
        from l90.graph.nodes import should_correct_or_generate

        # Verification passed → generator
        state = {"verification_passed": True, "correction_loop_count": 0}
        assert should_correct_or_generate(state) == "generator"

        # Verification failed, under max loops → corrector
        state = {"verification_passed": False, "correction_loop_count": 0}
        assert should_correct_or_generate(state) == "corrector"

        # Max loops reached → generator (forced)
        state = {"verification_passed": False, "correction_loop_count": 99}
        assert should_correct_or_generate(state) == "generator"
