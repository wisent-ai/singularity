"""Smoke tests that verify basic imports and instantiation work without API keys."""

import pytest


class TestImports:
    """Verify all core modules can be imported."""

    def test_import_package(self):
        import singularity
        assert hasattr(singularity, "__version__")

    def test_import_agent(self):
        from singularity import AutonomousAgent
        assert AutonomousAgent is not None

    def test_import_cognition(self):
        from singularity import CognitionEngine, AgentState, Decision, Action, TokenUsage
        assert all(c is not None for c in [CognitionEngine, AgentState, Decision, Action, TokenUsage])

    def test_import_skill_base(self):
        from singularity import Skill, SkillRegistry, SkillManifest, SkillAction, SkillResult
        assert all(c is not None for c in [Skill, SkillRegistry, SkillManifest, SkillAction, SkillResult])

    def test_import_filesystem_skill(self):
        from singularity.skills.filesystem import FilesystemSkill
        assert FilesystemSkill is not None

    def test_import_shell_skill(self):
        from singularity.skills.shell import ShellSkill
        assert ShellSkill is not None


class TestSkillBase:
    """Test SkillResult, SkillAction, SkillManifest, SkillRegistry."""

    def test_skill_result_defaults(self):
        from singularity.skills.base import SkillResult
        r = SkillResult(success=True)
        assert r.success is True
        assert r.message == ""
        assert r.data == {}
        assert r.cost == 0
        assert r.revenue == 0

    def test_skill_result_with_data(self):
        from singularity.skills.base import SkillResult
        r = SkillResult(success=False, message="error", data={"key": "val"}, cost=0.5)
        assert r.success is False
        assert r.message == "error"
        assert r.data["key"] == "val"
        assert r.cost == 0.5

    def test_skill_action(self):
        from singularity.skills.base import SkillAction
        a = SkillAction(name="test", description="A test action", parameters={"x": {"type": "str"}})
        assert a.name == "test"
        assert a.estimated_cost == 0
        assert a.success_probability == 0.8

    def test_skill_manifest(self):
        from singularity.skills.base import SkillManifest
        m = SkillManifest(
            skill_id="test", name="Test", version="1.0",
            category="dev", description="Testing", actions=[],
            required_credentials=[],
        )
        assert m.skill_id == "test"
        assert m.required_credentials == []

    def test_skill_registry(self):
        from singularity.skills.base import SkillRegistry
        reg = SkillRegistry()
        assert reg.list_skills() == []
        assert reg.list_all_actions() == []
        assert reg.get("nonexistent") is None


class TestCognition:
    """Test CognitionEngine and data classes."""

    def test_agent_state(self):
        from singularity.cognition import AgentState
        state = AgentState(balance=100.0, burn_rate=0.01, runway_hours=1000.0)
        assert state.balance == 100.0
        assert state.burn_rate == 0.01
        assert state.cycle == 0

    def test_cognition_engine_none_provider(self):
        from singularity.cognition import CognitionEngine
        engine = CognitionEngine(llm_provider="none")
        assert engine is not None

    def test_action_dataclass(self):
        from singularity.cognition import Action
        a = Action(tool="fs:read", params={"path": "/tmp/x"})
        assert a.tool == "fs:read"
        assert a.params["path"] == "/tmp/x"

    def test_token_usage(self):
        from singularity.cognition import TokenUsage
        t = TokenUsage(input_tokens=100, output_tokens=50)
        assert t.input_tokens == 100
        assert t.output_tokens == 50


class TestFilesystemSkill:
    """Test FilesystemSkill instantiation and actions."""

    def test_instantiation(self):
        from singularity.skills.filesystem import FilesystemSkill
        skill = FilesystemSkill()
        assert skill.manifest.skill_id == "filesystem"

    def test_actions_available(self):
        from singularity.skills.filesystem import FilesystemSkill
        skill = FilesystemSkill()
        action_names = [a.name for a in skill.get_actions()]
        assert "glob" in action_names
        assert "view" in action_names
        assert "write" in action_names
