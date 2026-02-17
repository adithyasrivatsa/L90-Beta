"""Retrieval agent — searches allowed collections and writes chunks to the Blackboard."""

from __future__ import annotations

import logging
from typing import Any

from l90.agents.base import BaseAgent
from l90.blackboard.blackboard import Blackboard
from l90.tracing.logger import ReasoningTraceLogger
from l90.vectordb.chroma_store import ChromaStore

logger = logging.getLogger(__name__)


class RetrieverAgent(BaseAgent):
    """Retrieves relevant chunks ONLY from allowed collections.

    The agent respects the ``allowed_sources`` set by the Manager
    and filters by user/workspace metadata when applicable.
    """

    def __init__(
        self,
        store: ChromaStore | None = None,
        n_results: int = 10,
        trace_logger: ReasoningTraceLogger | None = None,
    ) -> None:
        super().__init__(name="RetrieverAgent", trace_logger=trace_logger)
        self._store = store or ChromaStore()
        self._n_results = n_results

    async def execute(self, blackboard: Blackboard) -> Blackboard:
        """Execute retrieval from allowed collections."""

        self._log_trace(
            phase="retrieval",
            action="retrieval_start",
            input_summary=f"Query: {blackboard.query[:100]}, Collections: {blackboard.allowed_sources}",
        )

        if not blackboard.allowed_sources:
            self._log_trace(
                phase="retrieval",
                action="retrieval_skip",
                decision="No allowed sources — skipping retrieval",
            )
            return blackboard

        # Build metadata filter for user/workspace isolation
        where_filter: dict[str, Any] | None = None
        if blackboard.user_id:
            where_filter = {"owner": blackboard.user_id}

        # Query across all allowed collections
        chunks = self._store.query_multiple_collections(
            collection_names=blackboard.allowed_sources,
            query_text=blackboard.query,
            n_results=self._n_results,
            where=where_filter,
        )

        # Append to Blackboard (don't overwrite — supports multi-pass)
        blackboard.retrieved_chunks.extend(chunks)

        self._log_trace(
            phase="retrieval",
            action="retrieval_complete",
            output_summary=f"Retrieved {len(chunks)} chunks from {len(blackboard.allowed_sources)} collections",
            decision=f"Total chunks on Blackboard: {len(blackboard.retrieved_chunks)}",
        )

        blackboard.add_trace(
            agent_name=self._name,
            phase="retrieval",
            action="chunks_retrieved",
            details={
                "num_chunks": len(chunks),
                "collections_queried": blackboard.allowed_sources,
            },
        )

        return blackboard
