"""Blackboard persistence — pluggable storage backends."""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from typing import Any

from l90 import config

logger = logging.getLogger(__name__)


class BlackboardPersistence(ABC):
    """Abstract interface for Blackboard storage backends."""

    @abstractmethod
    async def save(self, session_id: str, data: dict[str, Any]) -> None:
        """Persist a serialized Blackboard."""
        ...

    @abstractmethod
    async def load(self, session_id: str) -> dict[str, Any] | None:
        """Load a serialized Blackboard. Returns None if not found."""
        ...

    @abstractmethod
    async def delete(self, session_id: str) -> None:
        """Delete a persisted Blackboard (e.g., incognito cleanup)."""
        ...

    @abstractmethod
    async def exists(self, session_id: str) -> bool:
        """Check if a session exists."""
        ...


# ── Phase 1: In-Memory ─────────────────────────────────────────

class InMemoryPersistence(BlackboardPersistence):
    """Dict-based in-memory store. Data lost on process restart.

    Suitable for development and single-process deployments.
    """

    def __init__(self) -> None:
        self._store: dict[str, dict[str, Any]] = {}

    async def save(self, session_id: str, data: dict[str, Any]) -> None:
        self._store[session_id] = data
        logger.debug("Blackboard saved in-memory: %s", session_id)

    async def load(self, session_id: str) -> dict[str, Any] | None:
        return self._store.get(session_id)

    async def delete(self, session_id: str) -> None:
        self._store.pop(session_id, None)
        logger.debug("Blackboard deleted from memory: %s", session_id)

    async def exists(self, session_id: str) -> bool:
        return session_id in self._store


# ── Phase 2: Redis (placeholder) ───────────────────────────────

class RedisPersistence(BlackboardPersistence):
    """Redis-backed persistence with TTL support.

    Requires ``pip install l90[redis]`` and a running Redis instance.
    """

    def __init__(self, redis_url: str | None = None, ttl_seconds: int = 86400) -> None:
        self._ttl = ttl_seconds
        self._url = redis_url or config.REDIS_URL
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                import redis
                self._client = redis.from_url(self._url)
            except ImportError:
                raise ImportError(
                    "Redis support requires the 'redis' package. "
                    "Install with: pip install l90[redis]"
                )
        return self._client

    async def save(self, session_id: str, data: dict[str, Any]) -> None:
        r = self._get_client()
        key = f"l90:blackboard:{session_id}"
        r.setex(key, self._ttl, json.dumps(data, default=str))
        logger.debug("Blackboard saved to Redis: %s", session_id)

    async def load(self, session_id: str) -> dict[str, Any] | None:
        r = self._get_client()
        key = f"l90:blackboard:{session_id}"
        raw = r.get(key)
        if raw is None:
            return None
        return json.loads(raw)

    async def delete(self, session_id: str) -> None:
        r = self._get_client()
        key = f"l90:blackboard:{session_id}"
        r.delete(key)
        logger.debug("Blackboard deleted from Redis: %s", session_id)

    async def exists(self, session_id: str) -> bool:
        r = self._get_client()
        key = f"l90:blackboard:{session_id}"
        return bool(r.exists(key))


# ── Factory ─────────────────────────────────────────────────────

class PersistenceFactory:
    """Return the correct persistence backend based on config."""

    @staticmethod
    def get(backend: str | None = None) -> BlackboardPersistence:
        backend = backend or config.BLACKBOARD_BACKEND
        if backend == "redis":
            return RedisPersistence()
        return InMemoryPersistence()
