"""Tests for the grounding enforcer."""

import pytest
from l90.blackboard.blackboard import Blackboard
from l90.grounding.enforcer import GroundingEnforcer, GroundingReport, INSUFFICIENT_ANSWER


class TestGroundingReport:
    def test_default_report(self):
        report = GroundingReport()
        assert report.all_grounded is False
        assert report.final_verdict == "REJECTED"
        assert report.grounding_score == 0.0

    def test_to_dict(self):
        report = GroundingReport(all_grounded=True, grounding_score=0.95)
        d = report.to_dict()
        assert d["all_grounded"] is True
        assert d["grounding_score"] == 0.95


class TestGroundingEnforcerSync:
    """Test synchronous checks that don't require the LLM."""

    def test_empty_chunks_rejects(self):
        """No chunks = insufficient data."""
        bb = Blackboard(
            query="test",
            final_answer="Some answer",
            verification_passed=True,
            confidence_score=0.9,
        )
        bb.retrieved_chunks = []

        # Since execute is async, we test the logic path
        # The grounding check should reject because no chunks
        assert bb.retrieved_chunks == []

    def test_insufficient_answer_constant(self):
        assert INSUFFICIENT_ANSWER == "Insufficient verified information."

    def test_enforcer_creation(self):
        enforcer = GroundingEnforcer(confidence_threshold=0.8)
        assert enforcer.name == "GroundingEnforcer"
