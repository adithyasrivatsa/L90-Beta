"""Local Nomic embedding provider — runs entirely on CPU, no API calls needed.

Uses `nomic-embed-text-v1.5` via sentence-transformers for high-quality
embeddings without any API quota or internet dependency.
"""

from __future__ import annotations

import logging
from typing import Any, cast

from chromadb.api.types import Documents, EmbeddingFunction, Embeddings

from l90.vectordb.embedding_base import EmbeddingProvider

logger = logging.getLogger(__name__)

# Nomic requires a task-type prefix for best results
_DOC_PREFIX = "search_document: "
_QUERY_PREFIX = "search_query: "


class NomicEmbeddingProvider(EmbeddingProvider, EmbeddingFunction):  # type: ignore[misc]
    """Local embedding provider using ``nomic-embed-text-v1.5``.

    Features:
    - Runs entirely on CPU — no API calls, no quota limits
    - ~137M parameters, ~550MB download (cached after first run)
    - 768-dimensional embeddings
    - Dual interface: EmbeddingProvider (async) + ChromaDB EmbeddingFunction (sync)
    """

    _DIMENSION = 768

    def __init__(self, model_name: str = "nomic-ai/nomic-embed-text-v1.5") -> None:
        self._model_name = model_name
        self._model = None  # Lazy load to avoid slow startup

    def _get_model(self):
        """Lazy-load the model on first use."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading local embedding model: %s (first load may take a minute)...", self._model_name)
            self._model = SentenceTransformer(self._model_name, trust_remote_code=True)
            logger.info("Embedding model loaded successfully.")
        return self._model

    # ── EmbeddingProvider interface (async) ─────────────────────

    @property
    def dimension(self) -> int:
        return self._DIMENSION

    @property
    def provider_name(self) -> str:
        return f"nomic-local:{self._model_name}"

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed documents locally with the Nomic prefix for retrieval."""
        model = self._get_model()
        prefixed = [_DOC_PREFIX + t for t in texts]
        embeddings = model.encode(prefixed, convert_to_numpy=True)
        return [e.tolist() for e in embeddings]

    async def embed_query(self, text: str) -> list[float]:
        """Embed a single query with the Nomic query prefix."""
        model = self._get_model()
        embedding = model.encode(_QUERY_PREFIX + text, convert_to_numpy=True)
        return embedding.tolist()

    # ── ChromaDB EmbeddingFunction protocol (sync) ─────────────

    def __call__(self, input: Documents) -> Embeddings:
        """Synchronous embedding for ChromaDB collection usage."""
        model = self._get_model()
        prefixed = [_DOC_PREFIX + t for t in input]
        embeddings = model.encode(prefixed, convert_to_numpy=True)
        return cast(Embeddings, [e.tolist() for e in embeddings])
