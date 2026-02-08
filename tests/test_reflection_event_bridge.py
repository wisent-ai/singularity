"""Tests for ReflectionEventBridgeSkill."""
import pytest, json
from unittest.mock import AsyncMock, MagicMock
from pathlib import Path
from singularity.skills.reflection_event_bridge import (
    ReflectionEventBridgeSkill, BRIDGE_STATE_FILE,
)


@pytest.fixture(autouse=True)
def clean_data():
    if BRIDGE_STATE_FILE.exists():
        BRIDGE_STATE_FILE.unlink()
    yield
    if BRIDGE_STATE_FILE.exists():
        BRIDGE_STATE_FILE.unlink()


@pytest.fixture
def skill():
    return ReflectionEventBridgeSkill()


@pytest.fixture
def skill_with_context():
    s = ReflectionEventBridgeSkill()
    ctx = MagicMock()
    ctx.call_skill = AsyncMock(return_value=MagicMock(
        success=True, message="ok",
        data={"reflection_id": "ref_123", "matches": [
            {"playbook_id": "pb_1", "name": "Test Playbook", "relevance_score": 0.9, "effectiveness": 0.8}
        ]},
    ))
    s.context = ctx
    return s


def test_manifest(skill):
    m = skill.manifest
    assert m.skill_id == "reflection_event_bridge"
    actions = [a.name for a in m.actions]
    assert "wire" in actions
    assert "unwire" in actions
    assert "configure" in actions
    assert "emit" in actions
    assert "auto_reflect" in actions
    assert "status" in actions
    assert "history" in actions
    assert "suggest_playbook" in actions


@pytest.mark.asyncio
async def test_wire_default(skill):
    result = await skill.execute("wire", {})
    assert result.success
    assert result.data["total_active"] >= 2  # failure + cycle subs
    assert result.data["config"]["reflect_on_failures"] is True
    assert result.data["config"]["reflect_on_successes"] is False


@pytest.mark.asyncio
async def test_wire_with_success_reflection(skill):
    result = await skill.execute("wire", {"reflect_on_successes": True})
    assert result.success
    assert result.data["total_active"] >= 3  # failure + success + cycle


@pytest.mark.asyncio
async def test_unwire(skill):
    await skill.execute("wire", {})
    result = await skill.execute("unwire", {})
    assert result.success
    assert result.data["removed_count"] >= 2
    assert result.data["total_active"] == 0


@pytest.mark.asyncio
async def test_configure(skill):
    result = await skill.execute("configure", {
        "reflect_on_successes": True,
        "extract_patterns_every_n": 5,
    })
    assert result.success
    assert len(result.data["updated"]) == 2
    assert result.data["config"]["reflect_on_successes"] is True
    assert result.data["config"]["extract_patterns_every_n"] == 5


@pytest.mark.asyncio
async def test_auto_reflect_failure(skill_with_context):
    result = await skill_with_context.execute("auto_reflect", {
        "event_type": "action.failed",
        "skill_id": "github",
        "action": "create_pr",
        "error": "Authentication failed",
    })
    assert result.success
    assert result.data["reflection"]["is_failure"] is True
    assert result.data["reflection"]["skill_id"] == "github"
    assert result.data["stats"]["failures"] == 1


@pytest.mark.asyncio
async def test_auto_reflect_success(skill_with_context):
    result = await skill_with_context.execute("auto_reflect", {
        "event_type": "action.success",
        "skill_id": "content",
        "action": "generate",
        "outcome": "Generated blog post",
    })
    assert result.success
    assert result.data["reflection"]["is_failure"] is False
    assert result.data["stats"]["successes"] == 1


@pytest.mark.asyncio
async def test_emit_valid_event(skill):
    result = await skill.execute("emit", {
        "event_type": "reflection.created",
        "data": {"test": True},
    })
    assert result.success
    assert "reflection.created" in result.message


@pytest.mark.asyncio
async def test_emit_invalid_event(skill):
    result = await skill.execute("emit", {"event_type": "invalid.type"})
    assert not result.success


@pytest.mark.asyncio
async def test_status_inactive(skill):
    result = await skill.execute("status", {})
    assert result.success
    assert result.data["active"] is False


@pytest.mark.asyncio
async def test_status_active(skill):
    await skill.execute("wire", {})
    result = await skill.execute("status", {})
    assert result.success
    assert result.data["active"] is True


@pytest.mark.asyncio
async def test_history(skill_with_context):
    await skill_with_context.execute("auto_reflect", {
        "event_type": "action.failed", "skill_id": "test", "action": "run",
    })
    result = await skill_with_context.execute("history", {"type": "all", "limit": 10})
    assert result.success
    assert result.data["total_reflections"] >= 1
    assert result.data["total_events"] >= 1  # reflection.created emitted


@pytest.mark.asyncio
async def test_suggest_playbook(skill_with_context):
    result = await skill_with_context.execute("suggest_playbook", {
        "task": "Deploy a new service",
        "tags": ["deployment"],
    })
    assert result.success
    assert result.data["best_match"]["playbook_id"] == "pb_1"
    assert result.data["event_emitted"] == "playbook.suggested"


@pytest.mark.asyncio
async def test_suggest_playbook_no_task(skill):
    result = await skill.execute("suggest_playbook", {"task": ""})
    assert not result.success


@pytest.mark.asyncio
async def test_pattern_extraction_trigger(skill_with_context):
    """After N reflections, pattern extraction should trigger."""
    await skill_with_context.execute("configure", {"extract_patterns_every_n": 2})
    await skill_with_context.execute("auto_reflect", {
        "event_type": "action.failed", "skill_id": "s1", "action": "a1",
    })
    await skill_with_context.execute("auto_reflect", {
        "event_type": "action.failed", "skill_id": "s2", "action": "a2",
    })
    status = await skill_with_context.execute("status", {})
    assert status.data["stats"]["patterns_extracted"] >= 1


@pytest.mark.asyncio
async def test_persistence(skill):
    await skill.execute("wire", {})
    await skill.execute("configure", {"reflect_on_successes": True})
    # Reload
    skill2 = ReflectionEventBridgeSkill()
    status = await skill2.execute("status", {})
    assert status.data["active"] is True
    assert status.data["config"]["reflect_on_successes"] is True


@pytest.mark.asyncio
async def test_unknown_action(skill):
    result = await skill.execute("nonexistent", {})
    assert not result.success


def test_estimate_cost(skill):
    assert skill.estimate_cost("wire", {}) == 0.0
