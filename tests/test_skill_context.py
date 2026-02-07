"""Tests for SkillContext inter-skill communication system."""

import pytest
import asyncio
from singularity.skills.base import (
    Skill, SkillResult, SkillManifest, SkillAction, SkillRegistry, SkillContext,
)


class MockSkillA(Skill):
    """A mock skill that can call other skills via context."""

    @property
    def manifest(self):
        return SkillManifest(
            skill_id="mock_a", name="Mock A", version="1.0",
            category="test", description="Test skill A",
            actions=[SkillAction(name="greet", description="Say hello",
                                 parameters={"name": {"type": "string", "required": True}})],
            required_credentials=[],
        )

    async def execute(self, action, params):
        if action == "greet":
            name = params.get("name", "world")
            # Use context to call skill B if available
            if self.context:
                result = await self.context.call_skill("mock_b", "uppercase", {"text": name})
                if result.success:
                    name = result.data.get("text", name)
            return SkillResult(success=True, message=f"Hello, {name}!", data={"greeting": f"Hello, {name}!"})
        return SkillResult(success=False, message=f"Unknown action: {action}")


class MockSkillB(Skill):
    """A mock skill providing text utilities."""

    @property
    def manifest(self):
        return SkillManifest(
            skill_id="mock_b", name="Mock B", version="1.0",
            category="test", description="Text utilities",
            actions=[SkillAction(name="uppercase", description="Uppercase text",
                                 parameters={"text": {"type": "string", "required": True}})],
            required_credentials=[],
        )

    async def execute(self, action, params):
        if action == "uppercase":
            text = params.get("text", "")
            return SkillResult(success=True, data={"text": text.upper()})
        return SkillResult(success=False, message=f"Unknown action: {action}")


@pytest.fixture
def registry_with_context():
    """Create a registry with two skills and a context."""
    reg = SkillRegistry()
    reg.install(MockSkillA)
    reg.install(MockSkillB)
    ctx = reg.create_context(agent_name="TestAgent", agent_ticker="TEST")
    return reg, ctx


class TestSkillContext:
    def test_context_creation(self, registry_with_context):
        reg, ctx = registry_with_context
        assert ctx.agent_name == "TestAgent"
        assert ctx.agent_ticker == "TEST"

    def test_skills_receive_context(self, registry_with_context):
        reg, ctx = registry_with_context
        skill_a = reg.get("mock_a")
        skill_b = reg.get("mock_b")
        assert skill_a.context is ctx
        assert skill_b.context is ctx

    def test_list_skills(self, registry_with_context):
        _, ctx = registry_with_context
        skills = ctx.list_skills()
        assert "mock_a" in skills
        assert "mock_b" in skills

    def test_list_actions(self, registry_with_context):
        _, ctx = registry_with_context
        actions = ctx.list_actions("mock_a")
        assert "greet" in actions
        actions_b = ctx.list_actions("mock_b")
        assert "uppercase" in actions_b
        assert ctx.list_actions("nonexistent") == []

    def test_get_skill(self, registry_with_context):
        reg, ctx = registry_with_context
        skill = ctx.get_skill("mock_a")
        assert skill is reg.get("mock_a")
        assert ctx.get_skill("nonexistent") is None

    @pytest.mark.asyncio
    async def test_call_skill(self, registry_with_context):
        _, ctx = registry_with_context
        result = await ctx.call_skill("mock_b", "uppercase", {"text": "hello"})
        assert result.success
        assert result.data["text"] == "HELLO"

    @pytest.mark.asyncio
    async def test_call_nonexistent_skill(self, registry_with_context):
        _, ctx = registry_with_context
        result = await ctx.call_skill("nonexistent", "action")
        assert not result.success
        assert "not found" in result.message

    @pytest.mark.asyncio
    async def test_cross_skill_call(self, registry_with_context):
        """Skill A calls Skill B through the context."""
        reg, ctx = registry_with_context
        skill_a = reg.get("mock_a")
        result = await skill_a.execute("greet", {"name": "world"})
        assert result.success
        assert result.data["greeting"] == "Hello, WORLD!"

    @pytest.mark.asyncio
    async def test_call_history(self, registry_with_context):
        reg, ctx = registry_with_context
        # Make a cross-skill call
        skill_a = reg.get("mock_a")
        await skill_a.execute("greet", {"name": "test"})
        history = ctx.call_history
        assert len(history) == 1
        assert history[0]["skill_id"] == "mock_b"
        assert history[0]["action"] == "uppercase"
        assert history[0]["result"] == "success"

    def test_clear_call_history(self, registry_with_context):
        _, ctx = registry_with_context
        ctx._call_log.append({"test": True})
        ctx.clear_call_history()
        assert len(ctx.call_history) == 0

    def test_agent_state_default(self):
        reg = SkillRegistry()
        ctx = SkillContext(registry=reg, agent_name="Bot", agent_ticker="BOT")
        state = ctx.agent_state
        assert state["agent_name"] == "Bot"
        assert state["agent_ticker"] == "BOT"

    def test_agent_state_custom(self):
        reg = SkillRegistry()
        custom_state = {"balance": 99.0, "cycle": 5}
        ctx = SkillContext(registry=reg, get_state_fn=lambda: custom_state)
        assert ctx.agent_state["balance"] == 99.0
        assert ctx.agent_state["cycle"] == 5

    def test_log_function(self):
        reg = SkillRegistry()
        logged = []
        ctx = SkillContext(registry=reg, log_fn=lambda tag, msg: logged.append((tag, msg)))
        ctx.log("TEST", "hello")
        assert logged == [("TEST", "hello")]

    def test_skill_without_context(self):
        """Skills work fine without a context (backward compatible)."""
        skill = MockSkillA()
        assert skill.context is None

    @pytest.mark.asyncio
    async def test_skill_without_context_executes(self):
        """Skills execute normally when context is None."""
        skill = MockSkillA()
        result = await skill.execute("greet", {"name": "world"})
        assert result.success
        # Without context, no cross-skill call, so name stays lowercase
        assert result.data["greeting"] == "Hello, world!"
