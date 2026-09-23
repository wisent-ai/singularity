#!/usr/bin/env python3
"""Tests for scheduler tick + event bridge integration in AutonomousLoopSkill."""

import pytest
import json
from pathlib import Path
from unittest.mock import patch, AsyncMock, MagicMock
from singularity.skills.autonomous_loop import AutonomousLoopSkill, LOOP_STATE_FILE
from singularity.skills.base import SkillResult


@pytest.fixture
def skill(tmp_path):
    test_file = tmp_path / "autonomous_loop.json"
    with patch("singularity.skills.autonomous_loop.LOOP_STATE_FILE", test_file):
        s = AutonomousLoopSkill()
        yield s


def _make_mock_context(scheduler_due=0, scheduler_results=None):
    """Create a mock context with scheduler and event bridge skills."""
    ctx = MagicMock()

    # Mock scheduler skill with tick() and get_due_count()
    mock_scheduler = MagicMock()
    mock_scheduler.get_due_count = MagicMock(return_value=scheduler_due)
    mock_scheduler.tick = AsyncMock(return_value=scheduler_results or [])

    def mock_get_skill(skill_id):
        if skill_id == "scheduler":
            return mock_scheduler
        return None

    ctx.get_skill = MagicMock(side_effect=mock_get_skill)

    async def mock_call_skill(skill_id, action, params=None):
        # Strategy
        if skill_id == "strategy" and action == "assess":
            return SkillResult(success=True, message="OK", data={
                "pillars": {
                    "self_improvement": {"score": 70, "capabilities": [], "gaps": []},
                    "revenue": {"score": 30, "capabilities": [], "gaps": ["billing"]},
                    "replication": {"score": 50, "capabilities": [], "gaps": []},
                    "goal_setting": {"score": 60, "capabilities": [], "gaps": []},
                },
                "weakest_pillar": "revenue", "strongest_pillar": "self_improvement",
                "summary": "Revenue weak",
            })
        # Goal manager
        if skill_id == "goal_manager" and action == "next":
            return SkillResult(success=True, data={"goal_id": "g1", "title": "Test", "pillar": "revenue"})
        if skill_id == "goal_manager" and action == "get":
            return SkillResult(success=True, data={"milestones": []})
        # Event bridges - all succeed
        if skill_id in ("goal_progress_events", "fleet_health_events",
                        "circuit_breaker_event_bridge", "circuit_sharing_events"):
            return SkillResult(success=True, message=f"{skill_id} synced")
        # Auto reputation bridge
        if skill_id == "auto_reputation_bridge" and action == "poll":
            return SkillResult(success=True, message="Polled", data={"processed": 2})
        # Outcome tracker, feedback
        if skill_id in ("outcome_tracker", "feedback_loop"):
            return SkillResult(success=True, message="OK", data={"adaptations": [], "patterns": []})
        # Scheduler presets
        if skill_id == "scheduler_presets":
            return SkillResult(success=True, message="Applied")
        return SkillResult(success=False, message=f"Unknown: {skill_id}:{action}")

    ctx.call_skill = AsyncMock(side_effect=mock_call_skill)
    return ctx, mock_scheduler


@pytest.mark.asyncio
async def test_tick_scheduler_no_due_tasks(skill):
    """Tick returns empty when no tasks are due."""
    ctx, mock_sched = _make_mock_context(scheduler_due=0)
    skill.context = ctx
    state = skill._load()
    result = await skill._tick_scheduler(state)
    assert result["tasks_executed"] == 0
    mock_sched.tick.assert_not_called()


@pytest.mark.asyncio
async def test_tick_scheduler_with_due_tasks(skill):
    """Tick executes due tasks and records stats."""
    results = [
        SkillResult(success=True, message="Task 1 done"),
        SkillResult(success=False, message="Task 2 failed"),
        SkillResult(success=True, message="Task 3 done"),
    ]
    ctx, mock_sched = _make_mock_context(scheduler_due=3, scheduler_results=results)
    skill.context = ctx
    state = skill._load()
    summary = await skill._tick_scheduler(state)
    assert summary["tasks_executed"] == 3
    assert summary["tasks_succeeded"] == 2
    assert summary["tasks_failed"] == 1
    assert state["stats"]["scheduler_ticks"] == 1
    assert state["stats"]["scheduler_tasks_executed"] == 3
    mock_sched.tick.assert_called_once()


@pytest.mark.asyncio
async def test_tick_scheduler_no_context(skill):
    """Tick gracefully handles missing context."""
    skill.context = None
    state = skill._load()
    result = await skill._tick_scheduler(state)
    assert result["tasks_executed"] == 0


@pytest.mark.asyncio
async def test_tick_scheduler_no_scheduler_skill(skill):
    """Tick gracefully handles missing scheduler skill."""
    ctx = MagicMock()
    ctx.get_skill = MagicMock(return_value=None)
    skill.context = ctx
    state = skill._load()
    result = await skill._tick_scheduler(state)
    assert result["tasks_executed"] == 0


@pytest.mark.asyncio
async def test_sync_goal_progress_events(skill):
    """Goal progress events bridge is called and stats updated."""
    ctx, _ = _make_mock_context()
    skill.context = ctx
    state = skill._load()
    await skill._sync_goal_progress_events(state)
    ctx.call_skill.assert_any_call("goal_progress_events", "monitor", {})
    assert state["stats"]["goal_progress_syncs"] == 1


@pytest.mark.asyncio
async def test_sync_fleet_health_events(skill):
    """Fleet health events bridge is called and stats updated."""
    ctx, _ = _make_mock_context()
    skill.context = ctx
    state = skill._load()
    await skill._sync_fleet_health_events(state)
    ctx.call_skill.assert_any_call("fleet_health_events", "monitor", {})
    assert state["stats"]["fleet_health_syncs"] == 1


@pytest.mark.asyncio
async def test_sync_auto_reputation(skill):
    """Auto reputation bridge is polled and stats updated."""
    ctx, _ = _make_mock_context()
    skill.context = ctx
    state = skill._load()
    await skill._sync_auto_reputation(state)
    ctx.call_skill.assert_any_call("auto_reputation_bridge", "poll", {})
    assert state["stats"]["reputation_polls"] == 1
    assert state["stats"]["reputation_updates"] == 2


@pytest.mark.asyncio
async def test_sync_methods_fail_silently(skill):
    """All sync methods handle errors without raising."""
    ctx = MagicMock()
    ctx.call_skill = AsyncMock(side_effect=RuntimeError("boom"))
    skill.context = ctx
    state = skill._load()
    # None of these should raise
    await skill._sync_goal_progress_events(state)
    await skill._sync_fleet_health_events(state)
    await skill._sync_auto_reputation(state)


@pytest.mark.asyncio
async def test_step_calls_scheduler_tick(skill):
    """Full _step() calls scheduler tick before starting phases."""
    ctx, mock_sched = _make_mock_context(scheduler_due=1, scheduler_results=[
        SkillResult(success=True, message="Ran task"),
    ])
    skill.context = ctx
    result = await skill.execute("step", {})
    assert result.success
    mock_sched.get_due_count.assert_called()
    mock_sched.tick.assert_called_once()


@pytest.mark.asyncio
async def test_step_calls_all_event_bridges(skill):
    """Full _step() calls all event bridge sync methods."""
    ctx, _ = _make_mock_context()
    skill.context = ctx
    result = await skill.execute("step", {})
    assert result.success
    called_skills = [call.args[0] for call in ctx.call_skill.call_args_list]
    assert "goal_progress_events" in called_skills
    assert "fleet_health_events" in called_skills
    assert "auto_reputation_bridge" in called_skills


@pytest.mark.asyncio
async def test_scheduler_tick_recorded_in_journal(skill):
    """When scheduler executes tasks, results appear in journal."""
    ctx, _ = _make_mock_context(scheduler_due=2, scheduler_results=[
        SkillResult(success=True, message="OK"),
        SkillResult(success=True, message="OK"),
    ])
    skill.context = ctx
    result = await skill.execute("step", {})
    assert result.success
    state = skill._load()
    journal = state.get("journal", [])
    assert len(journal) > 0
    last_entry = journal[-1]
    assert "scheduler_tick" in last_entry.get("phases", {})
    assert last_entry["phases"]["scheduler_tick"]["tasks_executed"] == 2
