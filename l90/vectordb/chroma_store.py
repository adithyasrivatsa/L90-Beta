"""ChromaDB vector store — collection management, CRUD, and source-isolated queries."""

from __future__ import annotations

import logging
from typing import Any

import chromadb
from chromadb.config import Settings

from l90 import config
from l90.vectordb.embedding import GeminiEmbeddingProvider

logger = logging.getLogger(__name__)


class ChromaStore:
    """Manages all ChromaDB collections for L90.

    Enforces:
    - Multiple collections (user private, workspace, internal, incognito, approved library)
    - Metadata filtering (source, owner, workspace_id, security_level, etc.)
    - Source isolation at query time
    - User-level and workspace-level isolation
    """

    def __init__(
        self,
        persist_directory: str | None = None,
        embedding_provider: GeminiEmbeddingProvider | None = None,
    ) -> None:
        self._persist_dir = persist_directory or config.CHROMA_PERSIST_DIR
        self._embedding_fn = embedding_provider or GeminiEmbeddingProvider()

        self._client = chromadb.PersistentClient(
            path=self._persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )

        logger.info("ChromaDB initialized at %s", self._persist_dir)

    # ── Collection management ──────────────────────────────────

    def get_or_create_collection(self, name: str) -> chromadb.Collection:
        """Get or create a collection with the shared embedding function."""
        return self._client.get_or_create_collection(
            name=name,
            embedding_function=self._embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

    def delete_collection(self, name: str) -> None:
        """Delete a collection entirely (used for incognito cleanup)."""
        try:
            self._client.delete_collection(name=name)
            logger.info("Deleted collection: %s", name)
        except ValueError:
            logger.warning("Collection %s does not exist, skip delete", name)

    def list_collections(self) -> list[str]:
        """Return names of all existing collections."""
        return [c.name for c in self._client.list_collections()]

    # ── Document operations ────────────────────────────────────

    def add_documents(
        self,
        collection_name: str,
        documents: list[str],
        metadatas: list[dict[str, Any]],
        ids: list[str],
    ) -> None:
        """Add documents with metadata to a collection.

        Args:
            collection_name: Target collection.
            documents: Text chunks.
            metadatas: Per-chunk metadata (source, owner, workspace_id, etc.).
            ids: Unique IDs for each chunk.
        """
        collection = self.get_or_create_collection(collection_name)
        collection.add(
            documents=documents,
            metadatas=metadatas,  # type: ignore[arg-type]
            ids=ids,
        )
        logger.info(
            "Added %d documents to collection '%s'", len(documents), collection_name
        )

    def query(
        self,
        collection_name: str,
        query_text: str,
        n_results: int = 10,
        where: dict[str, Any] | None = None,
        where_document: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Query a collection with optional metadata filtering.

        Args:
            collection_name: Collection to query.
            query_text: The search query.
            n_results: Maximum number of results.
            where: Metadata filter (e.g., {"owner": "user123"}).
            where_document: Document content filter.

        Returns:
            List of result dicts with keys: id, document, metadata, distance.
        """
        collection = self.get_or_create_collection(collection_name)

        kwargs: dict[str, Any] = {
            "query_texts": [query_text],
            "n_results": n_results,
        }
        if where:
            kwargs["where"] = where
        if where_document:
            kwargs["where_document"] = where_document

        results = collection.query(**kwargs)

        # Flatten ChromaDB's nested result format into a clean list
        chunks: list[dict[str, Any]] = []
        if results and results["ids"]:
            for i, doc_id in enumerate(results["ids"][0]):
                chunk: dict[str, Any] = {
                    "id": doc_id,
                    "document": (results["documents"] or [[]])[0][i],
                    "metadata": (results["metadatas"] or [[]])[0][i],
                    "distance": (results["distances"] or [[]])[0][i],
                    "collection": collection_name,
                }
                chunks.append(chunk)

        return chunks

    def query_multiple_collections(
        self,
        collection_names: list[str],
        query_text: str,
        n_results: int = 10,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Query multiple collections in one call, merging and sorting by distance.

        Used by the retrieval agents to search across allowed sources.
        """
        all_chunks: list[dict[str, Any]] = []

        for name in collection_names:
            try:
                chunks = self.query(
                    collection_name=name,
                    query_text=query_text,
                    n_results=n_results,
                    where=where,
                )
                all_chunks.extend(chunks)
            except Exception as exc:
                logger.warning("Failed to query collection '%s': %s", name, exc)

        # Sort by distance (ascending = most relevant first)
        all_chunks.sort(key=lambda c: c.get("distance", float("inf")))
        return all_chunks[:n_results]

    def get_collection_count(self, collection_name: str) -> int:
        """Return the number of documents in a collection."""
        collection = self.get_or_create_collection(collection_name)
        return collection.count()
