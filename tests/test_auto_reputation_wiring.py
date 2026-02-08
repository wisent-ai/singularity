"""Tests for auto-reputation wiring: TaskDelegation -> AgentReputation."""
import pytest
from unittest.mock import AsyncMock, MagicMock
from singularity.skills.task_delegation import TaskDelegationSkill, DELEGATION_FILE
from singularity.skills.base import SkillResult


@pytest.fixture(autouse=True)
def clean_data(tmp_path, monkeypatch):
    test_file = tmp_path / "task_delegations.json"
    monkeypatch.setattr("singularity.skills.task_delegation.DELEGATION_FILE", test_file)
    yield test_file


@pytest.fixture
def mock_context():
    ctx = MagicMock()
    ctx.call_skill = AsyncMock(return_value=SkillResult(
        success=True, message="Reputation updated"
    ))
    return ctx


@pytest.fixture
def skill():
    s = TaskDelegationSkill()
    s.initialized = True
    return s


@pytest.fixture
def skill_with_context(skill, mock_context):
    skill.context = mock_context
    return skill


async def _create_delegation(skill, agent_id="agent-alpha", budget=10.0):
    r = await skill.execute("delegate", {
        "task_name": "Test task",
        "task_description": "A delegated task",
        "budget": budget,
        "agent_id": agent_id,
    })
    assert r.success
    return r.data["delegation_id"]


@pytest.mark.asyncio
async def test_completion_triggers_reputation_update(skill_with_context, mock_context):
    dlg_id = await _create_delegation(skill_with_context)
    r = await skill_with_context.execute("report_completion", {
        "delegation_id": dlg_id,
        "status": "completed",
        "budget_spent": 4.0,
    })
    assert r.success
    assert r.data["reputation_updated"] is True
    # Verify call_skill was called with agent_reputation
    calls = [c for c in mock_context.call_skill.call_args_list
             if c[0][0] == "agent_reputation"]
    assert len(calls) == 1
    args = calls[0][0]
    assert args[1] == "record_task_outcome"
    params = args[2]
    assert params["agent_id"] == "agent-alpha"
    assert params["success"] is True
    assert params["on_time"] is True
    assert 0.0 <= params["budget_efficiency"] <= 1.0


@pytest.mark.asyncio
async def test_failure_triggers_negative_reputation(skill_with_context, mock_context):
    dlg_id = await _create_delegation(skill_with_context)
    r = await skill_with_context.execute("report_completion", {
        "delegation_id": dlg_id,
        "status": "failed",
        "error": "Task crashed",
        "budget_spent": 2.0,
    })
    assert r.success
    assert r.data["reputation_updated"] is True
    calls = [c for c in mock_context.call_skill.call_args_list
             if c[0][0] == "agent_reputation"]
    params = calls[0][0][2]
    assert params["success"] is False


@pytest.mark.asyncio
async def test_budget_efficiency_computed_correctly(skill_with_context, mock_context):
    dlg_id = await _create_delegation(skill_with_context, budget=10.0)
    await skill_with_context.execute("report_completion", {
        "delegation_id": dlg_id,
        "status": "completed",
        "budget_spent": 3.0,
    })
    calls = [c for c in mock_context.call_skill.call_args_list
             if c[0][0] == "agent_reputation"]
    params = calls[0][0][2]
    # 1.0 - (3.0 / 10.0) = 0.7
    assert params["budget_efficiency"] == 0.7


@pytest.mark.asyncio
async def test_no_context_skips_reputation(skill):
    """Without context, reputation update is skipped gracefully."""
    dlg_id = await _create_delegation(skill)
    r = await skill.execute("report_completion", {
        "delegation_id": dlg_id,
        "status": "completed",
    })
    assert r.success
    assert r.data["reputation_updated"] is False


@pytest.mark.asyncio
async def test_no_agent_id_skips_reputation(skill_with_context, mock_context):
    """Without agent_id, reputation update is skipped."""
    r = await skill_with_context.execute("delegate", {
        "task_name": "No agent task",
        "task_description": "No agent assigned",
        "budget": 5.0,
    })
    dlg_id = r.data["delegation_id"]
    r2 = await skill_with_context.execute("report_completion", {
        "delegation_id": dlg_id,
        "status": "completed",
    })
    assert r2.success
    assert r2.data["reputation_updated"] is False


@pytest.mark.asyncio
async def test_reputation_error_doesnt_break_delegation(skill_with_context, mock_context):
    """If reputation call fails, delegation still succeeds."""
    dlg_id = await _create_delegation(skill_with_context)
    # Now make call_skill raise for the reputation call
    mock_context.call_skill = AsyncMock(side_effect=Exception("Reputation broken"))
    r = await skill_with_context.execute("report_completion", {
        "delegation_id": dlg_id,
        "status": "completed",
    })
    assert r.success
    assert r.data["reputation_updated"] is False
