"""Tests for vector database operations (ChromaStore structure, not embedding calls)."""

import pytest
from l90 import config


class TestCollectionNames:
    """Verify collection name constants are correct."""

    def test_collection_constants(self):
        assert config.COLLECTION_USER_PRIVATE == "user_private_collection"
        assert config.COLLECTION_WORKSPACE == "workspace_collection"
        assert config.COLLECTION_INTERNAL_LIBRARY == "internal_library_collection"
        assert config.COLLECTION_INCOGNITO == "incognito_session_collection"
        assert config.COLLECTION_APPROVED_LIBRARY == "approved_library_collection"

    def test_all_collections_list(self):
        assert len(config.ALL_COLLECTIONS) == 5
        assert config.COLLECTION_USER_PRIVATE in config.ALL_COLLECTIONS
        assert config.COLLECTION_APPROVED_LIBRARY in config.ALL_COLLECTIONS


class TestConfig:
    """Verify configuration defaults."""

    def test_defaults(self):
        assert config.CHUNK_SIZE == 1000
        assert config.CHUNK_OVERLAP == 200
        assert config.MAX_CORRECTION_LOOPS == 3
        assert config.EMBEDDING_DIMENSION == 3072
        assert config.GROUNDING_CONFIDENCE_THRESHOLD == 0.7
        assert config.BLACKBOARD_BACKEND == "memory"

    def test_model_names(self):
        assert "gemini" in config.WORKER_MODEL_NAME.lower() or config.WORKER_MODEL_NAME
        assert "gemini" in config.MANAGER_MODEL_NAME.lower() or config.MANAGER_MODEL_NAME
        assert "embedding" in config.EMBEDDING_MODEL_NAME.lower()
