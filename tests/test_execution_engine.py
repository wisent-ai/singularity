"""Tests for ExecutionEngine - validation, timing, and error context."""
import asyncio
import pytest
from singularity.execution import ExecutionEngine
from singularity.skills.base import (
    Skill, SkillManifest, SkillAction, SkillResult, SkillRegistry,
)


class DummySkill(Skill):
    @property
    def manifest(self):
        return SkillManifest(
            skill_id="dummy",
            name="Dummy",
            version="1.0",
            category="test",
            description="Test skill",
            actions=[
                SkillAction(name="greet", description="Say hello",
                            parameters={"name": "person name", "title": "prefix (optional)"}),
                SkillAction(name="fail", description="Always fails", parameters={}),
                SkillAction(name="slow", description="Takes a while", parameters={}),
            ],
            required_credentials=[],
        )

    def check_credentials(self):
        return True

    async def execute(self, action, params):
        if action == "greet":
            return SkillResult(success=True, message=f"Hello {params.get('name', 'world')}!")
        if action == "fail":
            raise ValueError("Intentional error")
        if action == "slow":
            await asyncio.sleep(10)
            return SkillResult(success=True, message="Done")
        return SkillResult(success=False, message="Unknown action")


@pytest.fixture
def engine():
    reg = SkillRegistry()
    reg.install(DummySkill)
    return ExecutionEngine(reg)


@pytest.mark.asyncio
async def test_successful_execution(engine):
    r = await engine.execute("dummy:greet", {"name": "Alice"})
    assert r["status"] == "success"
    assert "Alice" in r["message"]
    assert "duration_seconds" in r


@pytest.mark.asyncio
async def test_unknown_skill(engine):
    r = await engine.execute("nope:greet", {})
    assert r["status"] == "error"
    assert "Unknown skill" in r["message"]
    assert "dummy" in r["message"]  # suggests available skills


@pytest.mark.asyncio
async def test_unknown_action(engine):
    r = await engine.execute("dummy:dance", {})
    assert r["status"] == "error"
    assert "Unknown action" in r["message"]
    assert "greet" in r["message"]  # shows available actions


@pytest.mark.asyncio
async def test_missing_required_param(engine):
    r = await engine.execute("dummy:greet", {})
    assert r["status"] == "error"
    assert "Missing required" in r["message"]
    assert "name" in r["message"]


@pytest.mark.asyncio
async def test_optional_param_not_required(engine):
    r = await engine.execute("dummy:greet", {"name": "Bob"})
    assert r["status"] == "success"


@pytest.mark.asyncio
async def test_execution_error_context(engine):
    r = await engine.execute("dummy:fail", {})
    assert r["status"] == "error"
    assert "Intentional error" in r["message"]
    assert "ValueError" in r["message"]


@pytest.mark.asyncio
async def test_timeout(engine):
    r = await engine.execute("dummy:slow", {}, timeout=0.1)
    assert r["status"] == "error"
    assert "timed out" in r["message"]


@pytest.mark.asyncio
async def test_bad_format(engine):
    r = await engine.execute("nocolon", {})
    assert r["status"] == "error"
    assert "skill:action" in r["message"]


@pytest.mark.asyncio
async def test_wait_action(engine):
    r = await engine.execute("wait", {})
    assert r["status"] == "success"


@pytest.mark.asyncio
async def test_fuzzy_skill_suggestion(engine):
    r = await engine.execute("dumm:greet", {})
    assert r["status"] == "error"
    assert "dummy" in r["message"]


@pytest.mark.asyncio
async def test_stats(engine):
    await engine.execute("dummy:greet", {"name": "A"})
    await engine.execute("dummy:fail", {})
    stats = engine.get_stats()
    assert stats["total"] == 2
    assert stats["success"] == 1
    assert stats["error"] == 1
