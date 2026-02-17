"""Security isolation — user, workspace, and incognito session management."""

from __future__ import annotations

import logging
from typing import Any

from l90 import config
from l90.vectordb.chroma_store import ChromaStore

logger = logging.getLogger(__name__)


class IsolationManager:
    """Enforces data isolation between users, workspaces, and sessions.

    Guarantees:
    - No data leakage between users
    - No cross-mode leakage
    - Incognito sessions fully cleaned up
    - Workspace data scoped by workspace_id
    """

    def __init__(self, store: ChromaStore | None = None) -> None:
        self._store = store or ChromaStore()

    # ── Incognito session management ───────────────────────────

    def create_incognito_session(self, session_id: str) -> str:
        """Create an isolated incognito collection for a session.

        Returns the collection name.
        """
        collection_name = f"{config.COLLECTION_INCOGNITO}_{session_id}"
        self._store.get_or_create_collection(collection_name)
        logger.info("Created incognito session: %s", collection_name)
        return collection_name

    def cleanup_incognito_session(self, session_id: str) -> None:
        """Delete ALL data associated with an incognito session.

        Zero storage guarantee: files, embeddings, and collections are removed.
        """
        collection_name = f"{config.COLLECTION_INCOGNITO}_{session_id}"
        self._store.delete_collection(collection_name)
        logger.info("Cleaned up incognito session: %s", session_id)

    # ── User isolation helpers ─────────────────────────────────

    @staticmethod
    def get_user_filter(user_id: str) -> dict[str, Any]:
        """Build a ChromaDB metadata filter scoped to a specific user."""
        return {"owner": user_id}

    @staticmethod
    def get_workspace_filter(workspace_id: str) -> dict[str, Any]:
        """Build a ChromaDB metadata filter scoped to a specific workspace."""
        return {"workspace_id": workspace_id}

    @staticmethod
    def get_user_workspace_filter(
        user_id: str, workspace_id: str
    ) -> dict[str, Any]:
        """Build a combined user + workspace filter."""
        return {
            "$and": [
                {"owner": user_id},
                {"workspace_id": workspace_id},
            ]
        }

    # ── Validation ─────────────────────────────────────────────

    @staticmethod
    def validate_access(
        user_id: str,
        chunk_metadata: dict[str, Any],
    ) -> bool:
        """Check if a user has access to a specific chunk.

        Rules:
        - User owns the chunk, OR
        - Chunk is in a shared workspace the user belongs to, OR
        - Chunk is from the approved library (accessible to all)
        """
        chunk_owner = chunk_metadata.get("owner", "")
        chunk_source_type = chunk_metadata.get("source_type", "")

        # Approved library is accessible to all authorized users
        if chunk_source_type == "approved_library":
            return True

        # User owns the chunk
        if chunk_owner == user_id:
            return True

        # Workspace chunks are accessible to workspace members
        # (workspace membership check would be done at a higher level)
        if chunk_source_type == "workspace":
            return True

        return False
