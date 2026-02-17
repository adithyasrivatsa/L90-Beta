"""Tests for the Blackboard system."""

import pytest
from l90.blackboard.blackboard import Blackboard


class TestBlackboard:
    def test_creation_defaults(self):
        bb = Blackboard()
        assert bb.query == ""
        assert bb.mode == ""
        assert bb.retrieved_chunks == []
        assert bb.confidence_score == 0.0
        assert bb.session_id  # UUID auto-generated

    def test_add_trace(self):
        bb = Blackboard()
        bb.add_trace("TestAgent", "test_phase", "test_action", decision="test")
        assert len(bb.reasoning_trace) == 1
        entry = bb.reasoning_trace[0]
        assert entry["agent"] == "TestAgent"
        assert entry["phase"] == "test_phase"
        assert entry["action"] == "test_action"
        assert entry["decision"] == "test"
        assert "timestamp" in entry

    def test_add_trace_with_confidence(self):
        bb = Blackboard()
        bb.add_trace("Agent", "phase", "action", confidence=0.95)
        assert bb.reasoning_trace[0]["confidence"] == 0.95

    def test_to_dict(self):
        bb = Blackboard(query="test query", mode="STRICT")
        d = bb.to_dict()
        assert d["query"] == "test query"
        assert d["mode"] == "STRICT"
        assert "session_id" in d
        assert "created_at" in d
        assert isinstance(d["retrieved_chunks"], list)

    def test_reset(self):
        bb = Blackboard(query="test", mode="STRICT")
        bb.retrieved_chunks.append({"doc": "chunk1"})
        bb.confidence_score = 0.9
        original_session = bb.session_id

        bb.reset()

        assert bb.query == ""
        assert bb.mode == ""
        assert bb.retrieved_chunks == []
        assert bb.confidence_score == 0.0
        assert bb.session_id == original_session  # Session preserved

    def test_multiple_traces(self):
        bb = Blackboard()
        for i in range(5):
            bb.add_trace(f"Agent{i}", "phase", f"action_{i}")
        assert len(bb.reasoning_trace) == 5
