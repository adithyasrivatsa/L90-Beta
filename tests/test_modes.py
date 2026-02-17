"""Tests for operation modes and mode enforcement."""

import pytest
from l90.modes.enforcement import ModeEnforcer, OperationMode
from l90 import config


class TestOperationMode:
    def test_all_modes_exist(self):
        modes = [m.value for m in OperationMode]
        assert "STRICT" in modes
        assert "PARTIAL" in modes
        assert "GENERAL" in modes
        assert "INCOGNITO" in modes
        assert "WORKSPACE" in modes

    def test_mode_from_string(self):
        assert OperationMode("STRICT") == OperationMode.STRICT
        assert OperationMode("PARTIAL") == OperationMode.PARTIAL


class TestModeEnforcer:
    def setup_method(self):
        self.enforcer = ModeEnforcer()

    def test_strict_mode_only_user_private(self):
        allowed = self.enforcer.get_allowed_collections("STRICT")
        assert allowed == [config.COLLECTION_USER_PRIVATE]
        assert config.COLLECTION_APPROVED_LIBRARY not in allowed
        assert config.COLLECTION_WORKSPACE not in allowed

    def test_partial_mode_includes_approved_library(self):
        allowed = self.enforcer.get_allowed_collections("PARTIAL")
        assert config.COLLECTION_USER_PRIVATE in allowed
        assert config.COLLECTION_WORKSPACE in allowed
        assert config.COLLECTION_APPROVED_LIBRARY in allowed
        assert config.COLLECTION_INTERNAL_LIBRARY not in allowed

    def test_general_mode_all_collections(self):
        allowed = self.enforcer.get_allowed_collections("GENERAL")
        assert config.COLLECTION_USER_PRIVATE in allowed
        assert config.COLLECTION_WORKSPACE in allowed
        assert config.COLLECTION_INTERNAL_LIBRARY in allowed
        assert config.COLLECTION_APPROVED_LIBRARY in allowed

    def test_incognito_mode(self):
        allowed = self.enforcer.get_allowed_collections("INCOGNITO")
        assert config.COLLECTION_INCOGNITO in allowed
        assert config.COLLECTION_APPROVED_LIBRARY in allowed
        assert config.COLLECTION_USER_PRIVATE not in allowed

    def test_workspace_mode(self):
        allowed = self.enforcer.get_allowed_collections("WORKSPACE")
        assert config.COLLECTION_WORKSPACE in allowed
        assert config.COLLECTION_APPROVED_LIBRARY in allowed
        assert config.COLLECTION_USER_PRIVATE not in allowed

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown operation mode"):
            self.enforcer.get_allowed_collections("INVALID")

    def test_case_insensitive(self):
        allowed = self.enforcer.get_allowed_collections("strict")
        assert allowed == [config.COLLECTION_USER_PRIVATE]

    def test_is_collection_allowed(self):
        assert self.enforcer.is_collection_allowed("STRICT", config.COLLECTION_USER_PRIVATE)
        assert not self.enforcer.is_collection_allowed("STRICT", config.COLLECTION_APPROVED_LIBRARY)

    def test_strict_insufficient_message(self):
        msg = ModeEnforcer.get_insufficient_data_message("STRICT")
        assert msg == "Insufficient data in provided documents."

    def test_general_insufficient_message(self):
        msg = ModeEnforcer.get_insufficient_data_message("GENERAL")
        assert msg == "Insufficient verified information."
