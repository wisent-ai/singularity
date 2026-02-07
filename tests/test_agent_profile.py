"""Tests for AgentProfile and ProfileRegistry."""

import json
import os
import tempfile
import pytest
from singularity.agent_profile import AgentProfile, ProfileRegistry, BUILTIN_PROFILES, get_default_registry


class TestAgentProfile:
    def test_default_profile(self):
        p = AgentProfile()
        assert p.name == "Agent"
        assert p.ticker == "AGENT"
        assert p.starting_balance == 100.0

    def test_from_dict(self):
        p = AgentProfile.from_dict({"name": "Test", "ticker": "TST", "starting_balance": 50.0})
        assert p.name == "Test"
        assert p.ticker == "TST"
        assert p.starting_balance == 50.0

    def test_from_dict_ignores_unknown(self):
        p = AgentProfile.from_dict({"name": "Test", "unknown_field": "ignored"})
        assert p.name == "Test"

    def test_to_dict_excludes_none(self):
        p = AgentProfile(name="Test")
        d = p.to_dict()
        assert "name" in d
        assert "skills" not in d  # None should be excluded

    def test_save_and_load(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            path = f.name
        try:
            p = AgentProfile(name="SaveTest", ticker="SVT", starting_balance=42.0)
            p.save(path)
            loaded = AgentProfile.from_file(path)
            assert loaded.name == "SaveTest"
            assert loaded.ticker == "SVT"
            assert loaded.starting_balance == 42.0
        finally:
            os.unlink(path)

    def test_from_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            AgentProfile.from_file("/nonexistent/path.json")

    def test_merge(self):
        base = AgentProfile(name="Base", ticker="BASE", starting_balance=100.0)
        merged = base.merge({"name": "Override", "starting_balance": 200.0})
        assert merged.name == "Override"
        assert merged.ticker == "BASE"  # Not overridden
        assert merged.starting_balance == 200.0

    def test_merge_metadata(self):
        base = AgentProfile(name="Base", metadata={"a": 1, "b": 2})
        merged = base.merge({"metadata": {"b": 3, "c": 4}})
        assert merged.metadata == {"a": 1, "b": 3, "c": 4}

    def test_to_agent_kwargs(self):
        p = AgentProfile(name="Test", ticker="TST", system_prompt="Custom prompt")
        kwargs = p.to_agent_kwargs()
        assert kwargs["name"] == "Test"
        assert kwargs["ticker"] == "TST"
        assert kwargs["system_prompt"] == "Custom prompt"
        assert "skills" not in kwargs  # Not a constructor param

    def test_to_agent_kwargs_no_prompt(self):
        p = AgentProfile(name="Test")
        kwargs = p.to_agent_kwargs()
        assert "system_prompt" not in kwargs


class TestProfileRegistry:
    def test_builtins_loaded(self):
        reg = ProfileRegistry()
        profiles = reg.list_profiles()
        assert "coder" in profiles
        assert "researcher" in profiles
        assert "writer" in profiles
        assert "minimal" in profiles

    def test_get_builtin(self):
        reg = ProfileRegistry()
        coder = reg.get("coder")
        assert coder is not None
        assert coder.name == "Coder"
        assert coder.agent_type == "coder"

    def test_get_nonexistent(self):
        reg = ProfileRegistry()
        assert reg.get("nonexistent") is None

    def test_register_custom(self):
        reg = ProfileRegistry()
        custom = AgentProfile(name="Custom", ticker="CUST")
        reg.register("custom", custom)
        assert reg.get("custom").name == "Custom"

    def test_load_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create profile files
            for name in ["alpha", "beta"]:
                p = AgentProfile(name=name.title(), ticker=name[:3].upper())
                p.save(os.path.join(tmpdir, f"{name}.json"))

            reg = ProfileRegistry(profiles_dir=tmpdir)
            assert "alpha" in reg.list_profiles()
            assert "beta" in reg.list_profiles()

    def test_load_directory_nonexistent(self):
        reg = ProfileRegistry()
        count = reg.load_directory("/nonexistent/dir")
        assert count == 0

    def test_inheritance(self):
        reg = ProfileRegistry()
        base = AgentProfile(name="Base", ticker="BASE", starting_balance=100.0)
        child = AgentProfile(name="Child", extends="base_prof", starting_balance=50.0)
        reg.register("base_prof", base)
        reg.register("child_prof", child)
        resolved = reg.get("child_prof")
        assert resolved.name == "Child"  # Child override
        assert resolved.ticker == "BASE"  # Inherited from parent
        assert resolved.starting_balance == 50.0  # Child override

    def test_save_all(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            reg = ProfileRegistry()
            reg.register("test", AgentProfile(name="Test"))
            count = reg.save_all(tmpdir)
            assert count > 0
            assert os.path.exists(os.path.join(tmpdir, "test.json"))

    def test_to_dict(self):
        reg = ProfileRegistry()
        d = reg.to_dict()
        assert "coder" in d
        assert d["coder"]["name"] == "Coder"


class TestDefaultRegistry:
    def test_get_default_registry(self):
        reg = get_default_registry()
        assert isinstance(reg, ProfileRegistry)
        assert len(reg.list_profiles()) >= len(BUILTIN_PROFILES)

    def test_singleton(self):
        reg1 = get_default_registry()
        reg2 = get_default_registry()
        assert reg1 is reg2
