"""Tests for SkillLoader - dynamic skill discovery and loading."""

import os
import pytest
import tempfile
from pathlib import Path

from singularity.skill_loader import (
    SkillLoader,
    DiscoveredSkill,
    SkillLoadError,
    discover_builtin_skills,
)
from singularity.skills.base import Skill, SkillRegistry, SkillManifest, SkillAction, SkillResult


# --- Fixtures ---

SAMPLE_SKILL_CODE = '''
from singularity.skills.base import Skill, SkillResult, SkillManifest, SkillAction
from typing import Dict

class SamplePluginSkill(Skill):
    @property
    def manifest(self):
        return SkillManifest(
            skill_id="sample_plugin",
            name="Sample Plugin",
            version="1.0",
            category="test",
            description="A test plugin skill",
            actions=[
                SkillAction(name="hello", description="Say hello",
                           parameters={"name": {"type": "string", "required": True}})
            ],
            required_credentials=[],
        )

    async def execute(self, action, params):
        if action == "hello":
            return SkillResult(success=True, message=f"Hello {params.get('name', 'world')}")
        return SkillResult(success=False, message="Unknown action")
'''

SAMPLE_SKILL_WITH_CREDS = '''
from singularity.skills.base import Skill, SkillResult, SkillManifest, SkillAction
from typing import Dict

class CredentialSkill(Skill):
    @property
    def manifest(self):
        return SkillManifest(
            skill_id="cred_skill",
            name="Credential Skill",
            version="1.0",
            category="test",
            description="Needs credentials",
            actions=[],
            required_credentials=["SECRET_KEY"],
        )

    async def execute(self, action, params):
        return SkillResult(success=False, message="Not implemented")
'''

BAD_SKILL_CODE = "this is not valid python !!!"

NON_SKILL_CODE = '''
class NotASkill:
    def do_thing(self):
        return "I'm not a skill"
'''


@pytest.fixture
def plugin_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def loader():
    return SkillLoader()


# --- Tests ---

class TestSkillLoader:

    def test_discover_empty_directory(self, loader, plugin_dir):
        results = loader.discover_directory(plugin_dir)
        assert results == []
        assert len(loader.errors) == 0

    def test_discover_nonexistent_directory(self, loader):
        results = loader.discover_directory("/nonexistent/path")
        assert results == []
        assert len(loader.errors) == 1
        assert "not found" in loader.errors[0].error.lower()

    def test_discover_plugin_skill(self, loader, plugin_dir):
        Path(plugin_dir, "sample.py").write_text(SAMPLE_SKILL_CODE)
        results = loader.discover_directory(plugin_dir, source="plugin")
        assert len(results) == 1
        assert results[0].skill_id == "sample_plugin"
        assert results[0].name == "Sample Plugin"
        assert results[0].source == "plugin"
        assert results[0].category == "test"

    def test_discover_skips_init_and_base(self, loader, plugin_dir):
        Path(plugin_dir, "__init__.py").write_text("# init")
        Path(plugin_dir, "base.py").write_text(SAMPLE_SKILL_CODE)
        Path(plugin_dir, "_private.py").write_text(SAMPLE_SKILL_CODE)
        results = loader.discover_directory(plugin_dir)
        assert len(results) == 0

    def test_discover_bad_python(self, loader, plugin_dir):
        Path(plugin_dir, "bad.py").write_text(BAD_SKILL_CODE)
        results = loader.discover_directory(plugin_dir)
        assert len(results) == 0
        assert len(loader.errors) == 1
        assert "import error" in loader.errors[0].error.lower()

    def test_discover_non_skill_classes(self, loader, plugin_dir):
        Path(plugin_dir, "notskill.py").write_text(NON_SKILL_CODE)
        results = loader.discover_directory(plugin_dir)
        assert len(results) == 0
        assert len(loader.errors) == 0

    def test_discover_module_nonexistent(self, loader):
        results = loader.discover_module("/nonexistent/file.py")
        assert len(results) == 0
        assert len(loader.errors) == 1

    def test_discover_module_non_python(self, loader, plugin_dir):
        txt = Path(plugin_dir, "readme.txt")
        txt.write_text("not python")
        results = loader.discover_module(str(txt))
        assert len(results) == 0

    def test_no_duplicates(self, loader, plugin_dir):
        Path(plugin_dir, "sample.py").write_text(SAMPLE_SKILL_CODE)
        loader.discover_directory(plugin_dir)
        loader.discover_directory(plugin_dir)  # Second scan
        assert len(loader.discovered) == 1

    def test_load_into_registry(self, loader, plugin_dir):
        Path(plugin_dir, "sample.py").write_text(SAMPLE_SKILL_CODE)
        loader.discover_directory(plugin_dir)
        
        registry = SkillRegistry()
        loaded, errors = loader.load_into_registry(registry, {}, require_credentials=False)
        assert "sample_plugin" in loaded
        assert len(errors) == 0
        assert registry.get("sample_plugin") is not None

    def test_load_filters_by_credentials(self, loader, plugin_dir):
        Path(plugin_dir, "cred.py").write_text(SAMPLE_SKILL_WITH_CREDS)
        loader.discover_directory(plugin_dir)
        
        registry = SkillRegistry()
        loaded, _ = loader.load_into_registry(registry, {}, require_credentials=True)
        assert "cred_skill" not in loaded

    def test_load_with_credentials_provided(self, loader, plugin_dir):
        Path(plugin_dir, "cred.py").write_text(SAMPLE_SKILL_WITH_CREDS)
        loader.discover_directory(plugin_dir)
        
        registry = SkillRegistry()
        loaded, _ = loader.load_into_registry(
            registry, {"SECRET_KEY": "abc123"}, require_credentials=True
        )
        assert "cred_skill" in loaded

    def test_load_specific_skill_ids(self, loader, plugin_dir):
        Path(plugin_dir, "sample.py").write_text(SAMPLE_SKILL_CODE)
        Path(plugin_dir, "cred.py").write_text(SAMPLE_SKILL_WITH_CREDS)
        loader.discover_directory(plugin_dir)
        
        registry = SkillRegistry()
        loaded, _ = loader.load_into_registry(
            registry, {}, skill_ids={"sample_plugin"}, require_credentials=False
        )
        assert "sample_plugin" in loaded
        assert "cred_skill" not in loaded

    def test_wiring_hooks(self, loader, plugin_dir):
        Path(plugin_dir, "sample.py").write_text(SAMPLE_SKILL_CODE)
        loader.discover_directory(plugin_dir)
        
        wired = []
        def hook(skill):
            wired.append(skill.manifest.skill_id)
        
        registry = SkillRegistry()
        loader.load_into_registry(
            registry, {}, require_credentials=False,
            wiring_hooks={"sample_plugin": hook}
        )
        assert "sample_plugin" in wired

    def test_get_available_with_credentials(self, loader, plugin_dir):
        Path(plugin_dir, "cred.py").write_text(SAMPLE_SKILL_WITH_CREDS)
        loader.discover_directory(plugin_dir)
        
        available = loader.get_available(credentials={"SECRET_KEY": "yes"})
        assert len(available) == 1
        assert available[0]["credentials_available"] is True
        
        available = loader.get_available(credentials={})
        assert available[0]["credentials_available"] is False
        assert "SECRET_KEY" in available[0]["missing_credentials"]

    def test_get_by_category(self, loader, plugin_dir):
        Path(plugin_dir, "sample.py").write_text(SAMPLE_SKILL_CODE)
        loader.discover_directory(plugin_dir)
        
        cats = loader.get_by_category()
        assert "test" in cats
        assert len(cats["test"]) == 1

    def test_summary(self, loader, plugin_dir):
        Path(plugin_dir, "sample.py").write_text(SAMPLE_SKILL_CODE)
        loader.discover_directory(plugin_dir, source="plugin")
        
        summary = loader.summary()
        assert summary["total_discovered"] == 1
        assert summary["by_source"]["plugin"] == 1
        assert "sample_plugin" in summary["skill_ids"]

    def test_clear(self, loader, plugin_dir):
        Path(plugin_dir, "sample.py").write_text(SAMPLE_SKILL_CODE)
        loader.discover_directory(plugin_dir)
        assert len(loader.discovered) == 1
        
        loader.clear()
        assert len(loader.discovered) == 0
        assert len(loader.errors) == 0

    def test_discovered_skill_to_dict(self, plugin_dir):
        Path(plugin_dir, "sample.py").write_text(SAMPLE_SKILL_CODE)
        loader = SkillLoader()
        results = loader.discover_directory(plugin_dir)
        
        d = results[0].to_dict()
        assert d["skill_id"] == "sample_plugin"
        assert d["name"] == "Sample Plugin"
        assert d["source"] == "builtin"
        assert d["category"] == "test"

    def test_error_to_dict(self):
        err = SkillLoadError("/some/path.py", "Import failed", "MySkill")
        d = err.to_dict()
        assert d["path"] == "/some/path.py"
        assert d["error"] == "Import failed"
        assert d["skill_name"] == "MySkill"

    def test_discover_builtin_skills(self):
        loader = discover_builtin_skills()
        summary = loader.summary()
        # Should discover at least the core skills (filesystem, shell, etc.)
        assert summary["total_discovered"] >= 5
        assert "filesystem" in summary["skill_ids"]
        assert "shell" in summary["skill_ids"]
        assert summary["errors"] == 0

    def test_get_errors(self, loader, plugin_dir):
        Path(plugin_dir, "bad.py").write_text(BAD_SKILL_CODE)
        loader.discover_directory(plugin_dir)
        errors = loader.get_errors()
        assert len(errors) >= 1
        assert "path" in errors[0]
        assert "error" in errors[0]
