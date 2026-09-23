"""Tests for Skill auto-wiring via configure() method."""
import pytest
from singularity.skills.base import Skill, SkillResult, SkillManifest, SkillAction, SkillRegistry
from typing import Dict, Any


class SimpleSkill(Skill):
    """Test skill that doesn't need wiring."""
    @property
    def manifest(self):
        return SkillManifest(
            skill_id="simple", name="Simple", version="1.0",
            category="test", description="A simple test skill",
            actions=[], required_credentials=[],
        )

    async def execute(self, action, params):
        return SkillResult(success=True)


class WirableSkill(Skill):
    """Test skill that uses configure() for auto-wiring."""
    def __init__(self, credentials=None):
        super().__init__(credentials)
        self.agent_name = None
        self.cognition_ref = None
        self.configured = False

    @property
    def manifest(self):
        return SkillManifest(
            skill_id="wirable", name="Wirable", version="1.0",
            category="test", description="A wirable test skill",
            actions=[], required_credentials=[],
        )

    def configure(self, context: Dict[str, Any]) -> None:
        self.agent_name = context.get("agent_name")
        self.cognition_ref = context.get("cognition")
        self.configured = True

    async def execute(self, action, params):
        return SkillResult(success=True)


class FailingConfigSkill(Skill):
    """Skill whose configure() raises an error."""
    @property
    def manifest(self):
        return SkillManifest(
            skill_id="failing_config", name="FailConfig", version="1.0",
            category="test", description="Fails during configure",
            actions=[], required_credentials=[],
        )

    def configure(self, context):
        raise ValueError("Configuration failed!")

    async def execute(self, action, params):
        return SkillResult(success=True)


def test_base_skill_configure_is_noop():
    """Default configure() should do nothing."""
    skill = SimpleSkill()
    skill.configure({"agent_name": "test"})  # Should not raise


def test_wirable_skill_configure():
    """Skills can override configure() to receive context."""
    skill = WirableSkill()
    assert skill.configured is False
    skill.configure({"agent_name": "TestAgent", "cognition": "mock_cognition"})
    assert skill.configured is True
    assert skill.agent_name == "TestAgent"
    assert skill.cognition_ref == "mock_cognition"


def test_registry_configure_all():
    """SkillRegistry.configure_all() calls configure() on all skills."""
    registry = SkillRegistry()
    registry.install(SimpleSkill)
    registry.install(WirableSkill)

    context = {"agent_name": "TestBot", "cognition": "mock"}
    errors = registry.configure_all(context)

    assert errors == {}
    wirable = registry.get("wirable")
    assert wirable.configured is True
    assert wirable.agent_name == "TestBot"


def test_registry_configure_all_handles_errors():
    """configure_all() captures errors without stopping."""
    registry = SkillRegistry()
    registry.install(WirableSkill)
    registry.install(FailingConfigSkill)

    errors = registry.configure_all({"agent_name": "Test"})

    assert "failing_config" in errors
    assert "Configuration failed!" in errors["failing_config"]
    # Other skills still configured
    wirable = registry.get("wirable")
    assert wirable.configured is True


def test_configure_with_empty_context():
    """configure() handles empty context gracefully."""
    skill = WirableSkill()
    skill.configure({})
    assert skill.configured is True
    assert skill.agent_name is None
    assert skill.cognition_ref is None


def test_configure_all_returns_empty_on_no_skills():
    """configure_all() on empty registry returns empty dict."""
    registry = SkillRegistry()
    errors = registry.configure_all({"agent_name": "Test"})
    assert errors == {}
