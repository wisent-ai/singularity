"""Tests for TaskReputationBridgeSkill."""

import pytest
import json
from unittest.mock import MagicMock
from singularity.skills.task_reputation_bridge import (
    TaskReputationBridgeSkill, BRIDGE_FILE,
)


@pytest.fixture
def skill(tmp_path):
    """Create skill with temp data path."""
    s = TaskReputationBridgeSkill()
    test_file = tmp_path / "task_reputation_bridge.json"
    import singularity.skills.task_reputation_bridge as mod
    mod.BRIDGE_FILE = test_file
    s._store = None
    return s


def _mock_context(delegations=None, rep_result=None):
    """Create mock SkillContext with delegation and reputation data."""
    ctx = MagicMock()

    async def mock_invoke(skill_id, action, params):
        result = MagicMock()
        if skill_id == "task_delegation" and action == "history":
            result.success = True
            result.data = {"delegations": delegations or []}
        elif skill_id == "agent_reputation" and action == "record_task_outcome":
            result.success = True
            result.data = rep_result or {
                "agent_id": params.get("agent_id"),
                "competence": 55.0,
                "reliability": 52.0,
                "overall": 53.5,
            }
        elif skill_id == "agent_reputation" and action == "get_reputation":
            result.success = True
            result.data = {"agent_id": params.get("agent_id"), "competence": 55.0, "overall": 53.0}
        else:
            result.success = False
            result.data = {}
        return result

    ctx.invoke_skill = mock_invoke
    return ctx


def _make_delegation(d_id, agent_id, status, budget=10.0, spent=5.0, task_name="test_task"):
    """Helper to create delegation dicts."""
    return {
        "delegation_id": d_id,
        "assigned_to": agent_id,
        "status": status,
        "task_name": task_name,
        "budget": budget,
        "budget_spent": spent,
        "created_at": "2026-01-01T00:00:00Z",
        "completed_at": "2026-01-01T01:00:00Z",
        "timeout_minutes": 120,
    }


@pytest.mark.asyncio
async def test_manifest(skill):
    m = skill.manifest
    assert m.skill_id == "task_reputation_bridge"
    assert len(m.actions) == 6
    names = [a.name for a in m.actions]
    assert "sync" in names
    assert "configure" in names
    assert "agent_report" in names


@pytest.mark.asyncio
async def test_sync_no_delegations(skill):
    skill.context = _mock_context(delegations=[])
    r = await skill.execute("sync", {})
    assert r.success
    assert "no new delegations" in r.message


@pytest.mark.asyncio
async def test_sync_completed_delegation(skill):
    delegations = [_make_delegation("DEL-1", "agent-1", "completed")]
    skill.context = _mock_context(delegations=delegations)
    r = await skill.execute("sync", {})
    assert r.success
    assert len(r.data["updates"]) == 1
    assert r.data["updates"][0]["agent_id"] == "agent-1"
    assert r.data["updates"][0]["success"] is True


@pytest.mark.asyncio
async def test_sync_failed_delegation(skill):
    delegations = [_make_delegation("DEL-2", "agent-2", "failed")]
    skill.context = _mock_context(delegations=delegations)
    r = await skill.execute("sync", {})
    assert r.success
    assert r.data["updates"][0]["success"] is False


@pytest.mark.asyncio
async def test_sync_skips_pending(skill):
    delegations = [
        _make_delegation("DEL-3", "agent-3", "pending"),
        _make_delegation("DEL-4", "agent-4", "in_progress"),
    ]
    skill.context = _mock_context(delegations=delegations)
    r = await skill.execute("sync", {})
    assert r.success
    assert len(r.data["updates"]) == 0


@pytest.mark.asyncio
async def test_sync_dedup(skill):
    """Syncing twice doesn't double-count."""
    delegations = [_make_delegation("DEL-5", "agent-5", "completed")]
    skill.context = _mock_context(delegations=delegations)

    r1 = await skill.execute("sync", {})
    assert len(r1.data["updates"]) == 1

    r2 = await skill.execute("sync", {})
    assert len(r2.data["updates"]) == 0
    assert r2.data["skipped_count"] == 1


@pytest.mark.asyncio
async def test_sync_dry_run(skill):
    delegations = [_make_delegation("DEL-6", "agent-6", "completed")]
    skill.context = _mock_context(delegations=delegations)
    r = await skill.execute("sync", {"dry_run": True})
    assert r.success
    assert r.data["dry_run"] is True
    assert r.data["updates"][0]["dry_run"] is True
    # Dry run shouldn't mark as synced
    r2 = await skill.execute("sync", {"dry_run": False})
    assert len(r2.data["updates"]) == 1


@pytest.mark.asyncio
async def test_sync_budget_efficiency(skill):
    delegations = [_make_delegation("DEL-7", "agent-7", "completed", budget=10.0, spent=3.0)]
    skill.context = _mock_context(delegations=delegations)
    r = await skill.execute("sync", {})
    assert r.data["updates"][0]["budget_efficiency"] == 0.3


@pytest.mark.asyncio
async def test_configure(skill):
    r = await skill.execute("configure", {
        "success_competence_base": 3.0,
        "timeliness_threshold_minutes": 60,
    })
    assert r.success
    assert r.data["config"]["success_competence_base"] == 3.0
    assert r.data["config"]["timeliness_threshold_minutes"] == 60


@pytest.mark.asyncio
async def test_configure_no_changes(skill):
    r = await skill.execute("configure", {})
    assert r.success
    assert "No changes" in r.message


@pytest.mark.asyncio
async def test_stats(skill):
    r = await skill.execute("stats", {})
    assert r.success
    assert "stats" in r.data
    assert "agent_summaries" in r.data


@pytest.mark.asyncio
async def test_agent_report_not_found(skill):
    r = await skill.execute("agent_report", {"agent_id": "nonexistent"})
    assert r.success
    assert r.data["found"] is False


@pytest.mark.asyncio
async def test_agent_report_with_data(skill):
    delegations = [_make_delegation("DEL-8", "agent-8", "completed")]
    skill.context = _mock_context(delegations=delegations)
    await skill.execute("sync", {})

    r = await skill.execute("agent_report", {"agent_id": "agent-8"})
    assert r.success
    assert r.data["found"] is True
    assert r.data["total_tasks"] == 1
    assert r.data["success_rate"] == 1.0


@pytest.mark.asyncio
async def test_history(skill):
    r = await skill.execute("history", {"limit": 5})
    assert r.success
    assert r.data["total"] == 0


@pytest.mark.asyncio
async def test_reset_sync_requires_confirm(skill):
    r = await skill.execute("reset_sync", {"confirm": False})
    assert not r.success


@pytest.mark.asyncio
async def test_reset_sync(skill):
    # Sync first
    delegations = [_make_delegation("DEL-9", "agent-9", "completed")]
    skill.context = _mock_context(delegations=delegations)
    await skill.execute("sync", {})

    # Reset
    r = await skill.execute("reset_sync", {"confirm": True})
    assert r.success
    assert r.data["cleared_count"] == 1

    # Sync again should re-process
    r2 = await skill.execute("sync", {})
    assert len(r2.data["updates"]) == 1
