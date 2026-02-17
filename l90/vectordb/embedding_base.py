"""Embedding abstraction — swap embedding providers without changing downstream code."""

from __future__ import annotations

from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    """Abstract interface for embedding providers.

    Implementations: GeminiEmbeddingProvider, (future) OpenAI, Sentence Transformers, etc.
    """

    @abstractmethod
    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of document texts.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors (one per input text).
        """
        ...

    @abstractmethod
    async def embed_query(self, text: str) -> list[float]:
        """Embed a single query text.

        May apply different task-type optimizations than document embedding.

        Args:
            text: Query string to embed.

        Returns:
            Embedding vector.
        """
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the dimensionality of the embedding vectors."""
        ...

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return a human-readable name for this provider."""
        ...
