"""Tests for SkillEventBridgeSkill."""
import pytest, json, asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path
from singularity.skills.skill_event_bridge import (
    SkillEventBridgeSkill, BRIDGE_DEFINITIONS, BRIDGE_FILE,
)

@pytest.fixture(autouse=True)
def clean_data():
    if BRIDGE_FILE.exists():
        BRIDGE_FILE.unlink()
    yield
    if BRIDGE_FILE.exists():
        BRIDGE_FILE.unlink()

@pytest.fixture
def skill():
    s = SkillEventBridgeSkill()
    return s

@pytest.fixture
def skill_with_context():
    s = SkillEventBridgeSkill()
    ctx = MagicMock()
    ctx.list_skills.return_value = [
        "incident_response", "self_healing", "agent_reputation", "event",
    ]
    ctx.call_skill = AsyncMock(return_value=MagicMock(success=True, message="ok", data={}))
    s.context = ctx
    return s

def test_manifest(skill):
    m = skill.manifest
    assert m.skill_id == "skill_event_bridge"
    actions = [a.name for a in m.actions]
    assert "wire" in actions
    assert "unwire" in actions
    assert "trigger" in actions
    assert "status" in actions
    assert "bridges" in actions
    assert "history" in actions

@pytest.mark.asyncio
async def test_bridges_lists_definitions(skill):
    result = await skill.execute("bridges", {})
    assert result.success
    assert result.data["total_count"] == len(BRIDGE_DEFINITIONS)
    assert all(not d["active"] for d in result.data["definitions"])

@pytest.mark.asyncio
async def test_wire_all_bridges(skill_with_context):
    result = await skill_with_context.execute("wire", {})
    assert result.success
    assert len(result.data["wired"]) > 0
    assert result.data["total_active"] > 0

@pytest.mark.asyncio
async def test_wire_specific_bridge(skill_with_context):
    result = await skill_with_context.execute("wire", {"bridge_ids": ["incident_lifecycle"]})
    assert result.success
    assert "incident_lifecycle" in result.data["wired"]

@pytest.mark.asyncio
async def test_wire_invalid_bridge(skill_with_context):
    result = await skill_with_context.execute("wire", {"bridge_ids": ["nonexistent"]})
    assert result.success
    assert len(result.data["skipped"]) == 1

@pytest.mark.asyncio
async def test_unwire_bridge(skill_with_context):
    await skill_with_context.execute("wire", {"bridge_ids": ["incident_lifecycle"]})
    result = await skill_with_context.execute("unwire", {"bridge_ids": ["incident_lifecycle"]})
    assert result.success
    assert "incident_lifecycle" in result.data["unwired"]
    assert result.data["total_active"] == 0

@pytest.mark.asyncio
async def test_trigger_event(skill_with_context):
    result = await skill_with_context.execute("trigger", {
        "topic": "test.event",
        "data": {"key": "value"},
        "source": "test",
    })
    assert result.success
    assert result.data["topic"] == "test.event"

@pytest.mark.asyncio
async def test_trigger_with_reaction(skill_with_context):
    await skill_with_context.execute("wire", {})
    result = await skill_with_context.execute("trigger", {
        "topic": "health.scan_complete",
        "data": {"issues_found": 3, "scan_summary": "degraded skills"},
    })
    assert result.success
    # health_to_incident bridge should have fired
    assert result.data["reactions_executed"] is not None

@pytest.mark.asyncio
async def test_condition_evaluation(skill):
    assert skill._evaluate_condition("issues_found > 0", {"issues_found": 3})
    assert not skill._evaluate_condition("issues_found > 0", {"issues_found": 0})
    assert skill._evaluate_condition("status == ok", {"status": "ok"})
    assert not skill._evaluate_condition("status == ok", {"status": "fail"})

@pytest.mark.asyncio
async def test_emit_bridge_events(skill_with_context):
    await skill_with_context.execute("wire", {"bridge_ids": ["incident_lifecycle"]})
    emitted = await skill_with_context.emit_bridge_events(
        "incident_response", "detect",
        {"incident_id": "INC-123", "severity": "sev2", "status": "detected"},
    )
    assert len(emitted) == 1
    assert emitted[0]["topic"] == "incident.detected"
    assert emitted[0]["data"]["incident_id"] == "INC-123"

@pytest.mark.asyncio
async def test_emit_no_match(skill_with_context):
    await skill_with_context.execute("wire", {"bridge_ids": ["incident_lifecycle"]})
    emitted = await skill_with_context.emit_bridge_events(
        "some_other_skill", "detect", {"id": "123"},
    )
    assert len(emitted) == 0

@pytest.mark.asyncio
async def test_status_shows_stats(skill_with_context):
    await skill_with_context.execute("wire", {})
    await skill_with_context.execute("trigger", {"topic": "test.event", "data": {}})
    result = await skill_with_context.execute("status", {})
    assert result.success
    assert result.data["stats"]["total_events_emitted"] >= 1

@pytest.mark.asyncio
async def test_history(skill_with_context):
    await skill_with_context.execute("trigger", {"topic": "a.b", "data": {}})
    await skill_with_context.execute("trigger", {"topic": "c.d", "data": {}})
    result = await skill_with_context.execute("history", {"limit": 10})
    assert result.success
    assert len(result.data["events"]) == 2
