"""Operation mode enforcement — controls which collections each mode can access."""

from __future__ import annotations

import logging
from enum import Enum

from l90 import config

logger = logging.getLogger(__name__)


class OperationMode(str, Enum):
    """L90 operation modes."""

    STRICT = "STRICT"
    PARTIAL = "PARTIAL"
    GENERAL = "GENERAL"
    INCOGNITO = "INCOGNITO"
    WORKSPACE = "WORKSPACE"


# ── Access rules per mode ──────────────────────────────────────

_MODE_COLLECTIONS: dict[OperationMode, list[str]] = {
    OperationMode.STRICT: [
        config.COLLECTION_USER_PRIVATE,
    ],
    OperationMode.PARTIAL: [
        config.COLLECTION_USER_PRIVATE,
        config.COLLECTION_WORKSPACE,
        config.COLLECTION_APPROVED_LIBRARY,
    ],
    OperationMode.GENERAL: [
        config.COLLECTION_USER_PRIVATE,
        config.COLLECTION_WORKSPACE,
        config.COLLECTION_INTERNAL_LIBRARY,
        config.COLLECTION_APPROVED_LIBRARY,
    ],
    OperationMode.INCOGNITO: [
        config.COLLECTION_INCOGNITO,
        config.COLLECTION_APPROVED_LIBRARY,
    ],
    OperationMode.WORKSPACE: [
        config.COLLECTION_WORKSPACE,
        config.COLLECTION_APPROVED_LIBRARY,
    ],
}


class ModeEnforcer:
    """Enforces collection access boundaries per operation mode.

    Rules:
    - STRICT:    user uploads ONLY. No library, no web, no prior knowledge.
    - PARTIAL:   user + workspace + approved library. Persistent memory.
    - GENERAL:   all collections. RAG optional.
    - INCOGNITO: session-only collection + approved library. Data deleted after.
    - WORKSPACE: shared workspace + approved library. Multi-user.
    """

    def get_allowed_collections(
        self,
        mode: str,
        user_id: str = "",
        workspace_id: str = "",
    ) -> list[str]:
        """Return the list of collection names allowed for the given mode.

        Args:
            mode: Operation mode string (case-insensitive).
            user_id: User identifier (for scoping private collections).
            workspace_id: Workspace identifier (for workspace-scoped collections).

        Returns:
            List of allowed collection names.

        Raises:
            ValueError: If mode is not recognized.
        """
        try:
            op_mode = OperationMode(mode.upper())
        except ValueError:
            raise ValueError(
                f"Unknown operation mode: '{mode}'. "
                f"Valid modes: {[m.value for m in OperationMode]}"
            )

        collections = list(_MODE_COLLECTIONS[op_mode])

        logger.debug(
            "Mode %s → allowed collections: %s (user=%s, workspace=%s)",
            op_mode.value,
            collections,
            user_id,
            workspace_id,
        )

        return collections

    def is_collection_allowed(
        self,
        mode: str,
        collection_name: str,
    ) -> bool:
        """Check if a specific collection is allowed under the given mode."""
        allowed = self.get_allowed_collections(mode)
        return collection_name in allowed

    @staticmethod
    def get_insufficient_data_message(mode: str) -> str:
        """Return the mode-appropriate insufficient data message."""
        if mode.upper() == OperationMode.STRICT.value:
            return "Insufficient data in provided documents."
        return "Insufficient verified information."
