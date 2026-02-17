"""Gemini embedding provider — concrete implementation of EmbeddingProvider.

Also implements ChromaDB's EmbeddingFunction protocol for direct collection usage.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, cast

from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from google import genai

from l90 import config
from l90.vectordb.embedding_base import EmbeddingProvider

logger = logging.getLogger(__name__)


class GeminiEmbeddingProvider(EmbeddingProvider, EmbeddingFunction):  # type: ignore[misc]
    """Gemini embedding provider using ``gemini-embedding-001``.

    Features:
    - Batched calls (configurable batch size)
    - Exponential backoff retry (3 attempts)
    - Rate limiting (RPM throttle)
    - Dual interface: EmbeddingProvider (async) + ChromaDB EmbeddingFunction (sync)
    """

    def __init__(
        self,
        model_name: str | None = None,
        api_key: str | None = None,
        batch_size: int | None = None,
        rpm_limit: int | None = None,
    ) -> None:
        self._model_name = model_name or config.EMBEDDING_MODEL_NAME
        self._client = genai.Client(api_key=api_key or config.GOOGLE_API_KEY)
        self._batch_size = batch_size or config.EMBEDDING_BATCH_SIZE
        self._rpm_limit = rpm_limit or config.EMBEDDING_RPM_LIMIT
        self._dimension_val = config.EMBEDDING_DIMENSION

        # Rate limiting state
        self._request_timestamps: list[float] = []

    # ── EmbeddingProvider interface (async) ─────────────────────

    @property
    def dimension(self) -> int:
        return self._dimension_val

    @property
    def provider_name(self) -> str:
        return f"gemini:{self._model_name}"

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed documents in batches with retry and rate limiting."""
        all_embeddings: list[list[float]] = []

        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            embeddings = await self._embed_with_retry(
                batch, task_type="RETRIEVAL_DOCUMENT"
            )
            all_embeddings.extend(embeddings)

        return all_embeddings

    async def embed_query(self, text: str) -> list[float]:
        """Embed a single query with retry."""
        results = await self._embed_with_retry([text], task_type="RETRIEVAL_QUERY")
        return results[0]

    # ── ChromaDB EmbeddingFunction protocol (sync) ─────────────

    def __call__(self, input: Documents) -> Embeddings:
        """Synchronous embedding for ChromaDB collection usage."""
        loop = asyncio.new_event_loop()
        try:
            embeddings = loop.run_until_complete(self.embed_documents(list(input)))
            return cast(Embeddings, embeddings)
        finally:
            loop.close()

    # ── Internal helpers ───────────────────────────────────────

    async def _embed_with_retry(
        self,
        texts: list[str],
        task_type: str = "RETRIEVAL_DOCUMENT",
        max_retries: int = 3,
    ) -> list[list[float]]:
        """Call the Gemini embedding API with exponential backoff retry."""
        for attempt in range(max_retries):
            try:
                await self._enforce_rate_limit()

                response = self._client.models.embed_content(
                    model=self._model_name,
                    contents=texts,
                    config={
                        "task_type": task_type,
                        "output_dimensionality": self._dimension_val,
                    },
                )

                self._record_request()

                embeddings = [
                    list(e.values) for e in response.embeddings  # type: ignore[union-attr]
                ]
                return embeddings

            except Exception as exc:
                wait = 2 ** attempt
                logger.warning(
                    "Embedding attempt %d/%d failed: %s. Retrying in %ds...",
                    attempt + 1,
                    max_retries,
                    exc,
                    wait,
                )
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(wait)

        # Should never reach here
        raise RuntimeError("Embedding failed after all retries")

    async def _enforce_rate_limit(self) -> None:
        """Simple RPM-based rate limiter."""
        now = time.time()
        # Clean timestamps older than 60s
        self._request_timestamps = [
            t for t in self._request_timestamps if now - t < 60
        ]

        if len(self._request_timestamps) >= self._rpm_limit:
            oldest = self._request_timestamps[0]
            wait_time = 60 - (now - oldest) + 0.1
            if wait_time > 0:
                logger.debug("Rate limit reached. Waiting %.1fs...", wait_time)
                await asyncio.sleep(wait_time)

    def _record_request(self) -> None:
        """Record a request timestamp for rate limiting."""
        self._request_timestamps.append(time.time())
