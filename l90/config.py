"""L90 Configuration — central settings loaded from environment variables."""

from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

# ── Load .env ──────────────────────────────────────────────────
load_dotenv()

# ── API Keys ───────────────────────────────────────────────────
GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")

# ── Model Names ────────────────────────────────────────────────
WORKER_MODEL_NAME: str = os.getenv("WORKER_MODEL_NAME", "gemini-2.5-flash")
MANAGER_MODEL_NAME: str = os.getenv("MANAGER_MODEL_NAME", "gemini-2.5-flash")
EMBEDDING_MODEL_NAME: str = os.getenv("EMBEDDING_MODEL_NAME", "gemini-embedding-001")

# ── ChromaDB ───────────────────────────────────────────────────
CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_data")

# ── Blackboard Persistence ─────────────────────────────────────
BLACKBOARD_BACKEND: str = os.getenv("BLACKBOARD_BACKEND", "memory")  # "memory" | "redis"
REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# ── Embedding ──────────────────────────────────────────────────
EMBEDDING_DIMENSION: int = int(os.getenv("EMBEDDING_DIMENSION", "3072"))
EMBEDDING_BATCH_SIZE: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "100"))
EMBEDDING_RPM_LIMIT: int = int(os.getenv("EMBEDDING_RPM_LIMIT", "300"))

# ── Grounding ──────────────────────────────────────────────────
GROUNDING_CONFIDENCE_THRESHOLD: float = float(
    os.getenv("GROUNDING_CONFIDENCE_THRESHOLD", "0.7")
)

# ── Pipeline ───────────────────────────────────────────────────
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))
MAX_CORRECTION_LOOPS: int = int(os.getenv("MAX_CORRECTION_LOOPS", "3"))

# ── Collection Names ──────────────────────────────────────────
COLLECTION_USER_PRIVATE = "user_private_collection"
COLLECTION_WORKSPACE = "workspace_collection"
COLLECTION_INTERNAL_LIBRARY = "internal_library_collection"
COLLECTION_INCOGNITO = "incognito_session_collection"
COLLECTION_APPROVED_LIBRARY = "approved_library_collection"

ALL_COLLECTIONS: list[str] = [
    COLLECTION_USER_PRIVATE,
    COLLECTION_WORKSPACE,
    COLLECTION_INTERNAL_LIBRARY,
    COLLECTION_INCOGNITO,
    COLLECTION_APPROVED_LIBRARY,
]

# ── Paths ──────────────────────────────────────────────────────
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
