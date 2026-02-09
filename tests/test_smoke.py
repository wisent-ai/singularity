"""Smoke tests for singularity package.

These tests verify the package can be imported and core classes
can be instantiated without any API keys or external services.
"""

import pytest


class TestImports:
    """Verify all public API imports work."""

    def test_import_package(self):
        import singularity
        assert hasattr(singularity, '__version__')

    def test_import_autonomous_agent(self):
        from singularity import AutonomousAgent
        assert AutonomousAgent is not None

    def test_import_cognition(self):
        from singularity import CognitionEngine, AgentState, Decision, Action, TokenUsage
        assert all([CognitionEngine, AgentState, Decision, Action, TokenUsage])

    def test_import_skill_base(self):
        from singularity import Skill, SkillRegistry, SkillManifest, SkillAction, SkillResult
        assert all([Skill, SkillRegistry, SkillManifest, SkillAction, SkillResult])


class TestSkillBase:
    """Test skill framework basics."""

    def test_skill_registry_create(self):
        from singularity.skills.base import SkillRegistry
        registry = SkillRegistry()
        assert len(registry.skills) == 0

    def test_skill_result_success(self):
        from singularity.skills.base import SkillResult
        result = SkillResult(success=True, message="ok", data={"key": "val"})
        assert result.success is True
        assert result.message == "ok"
        assert result.data["key"] == "val"

    def test_skill_result_failure(self):
        from singularity.skills.base import SkillResult
        result = SkillResult(success=False, message="failed")
        assert result.success is False

    def test_skill_manifest_create(self):
        from singularity.skills.base import SkillManifest, SkillAction
        manifest = SkillManifest(
            skill_id="test",
            name="Test Skill",
            version="1.0.0",
            category="test",
            description="A test skill",
            actions=[
                SkillAction(
                    name="do_thing",
                    description="Does a thing",
                    parameters={"input": "string"},
                    estimated_cost=0.0,
                )
            ],
            required_credentials=[],
        )
        assert manifest.skill_id == "test"
        assert len(manifest.actions) == 1

    def test_action_dataclass(self):
        from singularity.cognition import Action
        action = Action(tool="shell:bash", params={"command": "ls"}, reasoning="list files")
        assert action.tool == "shell:bash"
        assert action.params["command"] == "ls"

    def test_agent_state_dataclass(self):
        from singularity.cognition import AgentState
        state = AgentState(
            balance=50.0,
            burn_rate=0.01,
            runway_hours=500.0,
            tools=[],
            recent_actions=[],
            cycle=5,
        )
        assert state.balance == 50.0
        assert state.cycle == 5


class TestSkillInstantiation:
    """Test that individual skills can be created without crashing."""

    def test_shell_skill(self):
        from singularity.skills.shell import ShellSkill
        skill = ShellSkill()
        assert skill.manifest.skill_id == "shell"
        assert skill.check_credentials() is True

    def test_filesystem_skill(self):
        from singularity.skills.filesystem import FilesystemSkill
        skill = FilesystemSkill()
        assert skill.manifest.skill_id == "filesystem"

    def test_request_skill(self):
        from singularity.skills.request import RequestSkill
        skill = RequestSkill()
        assert skill.manifest.skill_id == "request"

    def test_orchestrator_skill(self):
        from singularity.skills.orchestrator import OrchestratorSkill
        skill = OrchestratorSkill()
        assert skill.manifest.skill_id == "orchestrator"

    def test_memory_skill_without_cognee(self):
        from singularity.skills.memory import MemorySkill, HAS_COGNEE
        skill = MemorySkill()
        assert skill.manifest.skill_id == "memory"
        # Without cognee installed, check_credentials should be False
        if not HAS_COGNEE:
            assert skill.check_credentials() is False

    def test_content_skill(self):
        from singularity.skills.content import ContentCreationSkill
        skill = ContentCreationSkill()
        assert skill.manifest.skill_id == "content_creation"

    def test_self_modify_skill(self):
        from singularity.skills.self_modify import SelfModifySkill
        skill = SelfModifySkill()
        assert skill.manifest.skill_id == "self"
