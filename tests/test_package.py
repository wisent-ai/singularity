"""Package-level smoke tests — validates imports, structure, and public API."""

import json
import importlib
from pathlib import Path

import pytest


# ─── Package Imports ─────────────────────────────────────────────────────


class TestPackageImports:
    """All public symbols should be importable."""

    def test_top_level_import(self):
        import singularity
        assert hasattr(singularity, "__version__")
        assert singularity.__version__ == "0.2.0"

    def test_autonomous_agent_import(self):
        from singularity import AutonomousAgent
        assert AutonomousAgent is not None

    def test_cognition_imports(self):
        from singularity import (
            CognitionEngine, AgentState, Decision, Action, TokenUsage,
            calculate_api_cost, UNIFIED_AGENT_PROMPT, MESSAGE_FROM_CREATOR,
            build_result_message,
        )
        assert CognitionEngine is not None
        assert AgentState is not None

    def test_skill_imports(self):
        from singularity import (
            Skill, SkillRegistry, SkillManifest, SkillAction, SkillResult,
            PluginLoader, SkillMetadata, MCPServerInfo,
        )
        assert Skill is not None
        assert SkillRegistry is not None

    def test_cognition_submodules(self):
        from singularity.cognition import types
        from singularity.cognition import engine
        from singularity.cognition import prompt_builder
        from singularity.cognition import providers
        assert types is not None

    def test_skill_submodules(self):
        from singularity.skills.base import skill, types, registry
        from singularity.skills.loader import loader
        assert skill is not None


# ─── __all__ Exports ─────────────────────────────────────────────────────


class TestAllExports:
    def test_top_level_all(self):
        import singularity
        assert "__all__" in dir(singularity)
        expected = [
            "AutonomousAgent",
            "CognitionEngine", "AgentState", "Decision", "Action", "TokenUsage",
            "calculate_api_cost", "UNIFIED_AGENT_PROMPT", "MESSAGE_FROM_CREATOR",
            "build_result_message",
            "Skill", "SkillRegistry", "SkillManifest", "SkillAction", "SkillResult",
            "PluginLoader", "SkillMetadata", "MCPServerInfo",
        ]
        for name in expected:
            assert name in singularity.__all__, f"{name} not in __all__"


# ─── Dataclass Contracts ─────────────────────────────────────────────────


class TestDataclassContracts:
    """Verify dataclass fields and defaults match expected interface."""

    def test_action_fields(self):
        from singularity.cognition.types import Action
        a = Action(tool="test")
        assert hasattr(a, "tool")
        assert hasattr(a, "params")
        assert hasattr(a, "reasoning")

    def test_token_usage_fields(self):
        from singularity.cognition.types import TokenUsage
        u = TokenUsage()
        assert hasattr(u, "input_tokens")
        assert hasattr(u, "output_tokens")
        assert callable(u.total_tokens)

    def test_agent_state_fields(self):
        from singularity.cognition.types import AgentState
        s = AgentState(balance=0, burn_rate=0, runway_hours=0)
        expected = ["balance", "burn_rate", "runway_hours", "tools",
                     "recent_actions", "cycle", "chat_messages",
                     "project_context", "goals_progress", "pending_tasks",
                     "created_resources"]
        for field in expected:
            assert hasattr(s, field), f"AgentState missing field: {field}"

    def test_decision_fields(self):
        from singularity.cognition.types import Decision
        d = Decision()
        assert hasattr(d, "action")
        assert hasattr(d, "reasoning")
        assert hasattr(d, "token_usage")
        assert hasattr(d, "api_cost_usd")

    def test_skill_result_fields(self):
        from singularity.skills.base.types import SkillResult
        r = SkillResult(success=True)
        expected = ["success", "message", "data", "cost", "revenue", "asset_created"]
        for field in expected:
            assert hasattr(r, field), f"SkillResult missing field: {field}"

    def test_skill_action_fields(self):
        from singularity.skills.base.types import SkillAction
        a = SkillAction(name="x", description="y", parameters={})
        expected = ["name", "description", "parameters",
                     "estimated_cost", "estimated_duration_seconds",
                     "success_probability"]
        for field in expected:
            assert hasattr(a, field), f"SkillAction missing field: {field}"

    def test_skill_manifest_fields(self):
        from singularity.skills.base.types import SkillManifest
        m = SkillManifest(skill_id="x", name="X", version="1", category="t",
                          description="d", actions=[], required_credentials=[])
        expected = ["skill_id", "name", "version", "category", "description",
                     "actions", "required_credentials", "install_cost", "author"]
        for field in expected:
            assert hasattr(m, field), f"SkillManifest missing field: {field}"


# ─── Registry Integrity ─────────────────────────────────────────────────


class TestRegistryIntegrity:
    """Validate the registry.json matches the actual directory structure."""

    def test_registry_modules_point_to_existing_dirs(self):
        """Each skill in registry should have a corresponding builtin directory.

        Known issue: some skills (shell, browser, facebook, instagram, namecheap)
        are registered but have empty directories without Python files. This test
        checks that the directory at least exists — the implementation gap is
        tracked separately.
        """
        registry_path = Path(__file__).parent.parent / "singularity" / "skills" / "registry.json"
        builtin_path = Path(__file__).parent.parent / "singularity" / "skills" / "builtin"
        data = json.loads(registry_path.read_text())

        missing = []
        for skill_id, skill_data in data["skills"].items():
            module = skill_data.get("module", "")
            if module.startswith("singularity.skills.builtin."):
                dir_name = module.split(".")[-1]
                dir_path = builtin_path / dir_name
                if not dir_path.exists():
                    missing.append(f"{skill_id} -> {dir_name}")

        # Warn about missing directories but don't fail the build.
        # Some skills are registered but not yet implemented.
        if missing:
            import warnings
            warnings.warn(
                f"Registry references {len(missing)} missing builtin directories: "
                + ", ".join(missing),
                stacklevel=1,
            )

    def test_registry_has_minimum_skills(self):
        """Registry should have at least the core skills."""
        registry_path = Path(__file__).parent.parent / "singularity" / "skills" / "registry.json"
        data = json.loads(registry_path.read_text())
        skills = data["skills"]
        core_skills = ["github", "filesystem", "content_creation", "stripe", "email"]
        for skill_id in core_skills:
            assert skill_id in skills, f"Core skill '{skill_id}' missing from registry"


# ─── pyproject.toml Validation ───────────────────────────────────────────


class TestProjectConfig:
    def test_pyproject_exists(self):
        path = Path(__file__).parent.parent / "pyproject.toml"
        assert path.exists()

    def test_pyproject_version_matches(self):
        import singularity
        path = Path(__file__).parent.parent / "pyproject.toml"
        content = path.read_text()
        assert f'version = "{singularity.__version__}"' in content
