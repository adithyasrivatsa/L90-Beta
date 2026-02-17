"""Tests for the reasoning trace logger."""

import json
import pytest
from l90.tracing.logger import ReasoningTraceLogger, TraceEntry


class TestReasoningTraceLogger:
    def test_log_entry(self):
        logger = ReasoningTraceLogger()
        entry = logger.log(
            agent_name="TestAgent",
            phase="test",
            action="test_action",
            decision="approved",
            confidence=0.9,
        )
        assert isinstance(entry, TraceEntry)
        assert entry.agent_name == "TestAgent"
        assert entry.confidence == 0.9
        assert len(logger) == 1

    def test_multiple_entries(self):
        logger = ReasoningTraceLogger()
        for i in range(5):
            logger.log(f"Agent{i}", "phase", f"action_{i}")
        assert len(logger) == 5

    def test_get_entries(self):
        logger = ReasoningTraceLogger()
        logger.log("Agent1", "phase1", "action1")
        logger.log("Agent2", "phase2", "action2")
        entries = logger.get_entries()
        assert len(entries) == 2
        assert entries[0]["agent_name"] == "Agent1"
        assert entries[1]["agent_name"] == "Agent2"

    def test_export_json(self):
        logger = ReasoningTraceLogger()
        logger.log("Agent", "phase", "action")
        exported = logger.export_json()
        parsed = json.loads(exported)
        assert len(parsed) == 1
        assert parsed[0]["agent_name"] == "Agent"

    def test_clear(self):
        logger = ReasoningTraceLogger()
        logger.log("Agent", "phase", "action")
        assert len(logger) == 1
        logger.clear()
        assert len(logger) == 0

    def test_thread_safety(self):
        """Test that concurrent logging doesn't crash."""
        import threading
        logger = ReasoningTraceLogger()

        def log_entries():
            for i in range(100):
                logger.log(f"Agent_{threading.current_thread().name}", "phase", f"action_{i}")

        threads = [threading.Thread(target=log_entries) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(logger) == 400


class TestBlackboardPersistence:
    """Test the persistence abstraction."""

    @pytest.mark.asyncio
    async def test_in_memory_save_load(self):
        from l90.blackboard.persistence import InMemoryPersistence
        p = InMemoryPersistence()
        await p.save("session1", {"query": "test"})
        data = await p.load("session1")
        assert data == {"query": "test"}

    @pytest.mark.asyncio
    async def test_in_memory_delete(self):
        from l90.blackboard.persistence import InMemoryPersistence
        p = InMemoryPersistence()
        await p.save("session1", {"query": "test"})
        await p.delete("session1")
        data = await p.load("session1")
        assert data is None

    @pytest.mark.asyncio
    async def test_in_memory_exists(self):
        from l90.blackboard.persistence import InMemoryPersistence
        p = InMemoryPersistence()
        assert not await p.exists("session1")
        await p.save("session1", {"query": "test"})
        assert await p.exists("session1")

    @pytest.mark.asyncio
    async def test_factory_default_memory(self):
        from l90.blackboard.persistence import PersistenceFactory
        p = PersistenceFactory.get("memory")
        assert p.__class__.__name__ == "InMemoryPersistence"
