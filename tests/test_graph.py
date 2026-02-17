"""Tests for the LangGraph orchestration and SwarmOrchestrator."""

import pytest


class TestGraphStructure:
    """Test that the graph schema is correct."""

    def test_graph_state_has_required_fields(self):
        """Verify the state schema has required fields."""
        from l90.graph.state import GraphState

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
        from l90.graph.nodes import should_correct_or_continue

        # Verification passed → continue
        state = {"verification_passed": True, "correction_loop_count": 0}
        assert should_correct_or_continue(state) == "continue"

        # Verification failed, under max loops → corrector
        state = {"verification_passed": False, "correction_loop_count": 0}
        assert should_correct_or_continue(state) == "corrector"

        # Max loops reached → continue (forced)
        state = {"verification_passed": False, "correction_loop_count": 99}
        assert should_correct_or_continue(state) == "continue"


class TestFastClassifier:
    """Test the planner's fast local classifier."""

    def test_basic_what_is_query(self):
        from l90.agents.planner import PlannerLayer
        planner = PlannerLayer()
        plan = planner._fast_classify("What is gravity?")
        assert plan is not None
        assert plan.complexity_level == "BASIC"

    def test_basic_greeting(self):
        from l90.agents.planner import PlannerLayer
        planner = PlannerLayer()
        plan = planner._fast_classify("Hello!")
        assert plan is not None
        assert plan.complexity_level == "BASIC"

    def test_advanced_query_returns_none(self):
        from l90.agents.planner import PlannerLayer
        planner = PlannerLayer()
        plan = planner._fast_classify("Derive the Lagrangian for a relativistic particle")
        assert plan is None

    def test_math_query_returns_none(self):
        from l90.agents.planner import PlannerLayer
        planner = PlannerLayer()
        plan = planner._fast_classify("Calculate the integral of sin(x)")
        assert plan is None

    def test_short_question_classified_basic(self):
        from l90.agents.planner import PlannerLayer
        planner = PlannerLayer()
        plan = planner._fast_classify("How old is the Earth?")
        assert plan is not None
        assert plan.complexity_level == "BASIC"


class TestSwarmOrchestrator:
    """Test SwarmOrchestrator construction."""

    def test_orchestrator_builds(self):
        from l90.graph.builder import build_swarm_orchestrator
        from l90.graph.orchestrator import SwarmOrchestrator
        from l90.tracing.logger import ReasoningTraceLogger
        from l90.vectordb.chroma_store import ChromaStore

        store = ChromaStore()
        trace = ReasoningTraceLogger()
        orch = build_swarm_orchestrator(store=store, trace_logger=trace)
        assert isinstance(orch, SwarmOrchestrator)

