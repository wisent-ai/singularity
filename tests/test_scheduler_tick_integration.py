#!/usr/bin/env python3
"""Tests for scheduler tick integration in AutonomousLoopSkill."""

import pytest
import json
from pathlib import Path
from unittest.mock import patch, AsyncMock, MagicMock
from singularity.skills.autonomous_loop import AutonomousLoopSkill, LOOP_STATE_FILE
from singularity.skills.base import SkillResult


@pytest.fixture
def skill(tmp_path):
    """Create an AutonomousLoopSkill with a temporary data path."""
    test_file = tmp_path / "autonomous_loop.json"
    with patch("singularity.skills.autonomous_loop.LOOP_STATE_FILE", test_file):
        s = AutonomousLoopSkill()
        yield s


@pytest.fixture
def mock_context():
    """Create a mock SkillContext."""
    ctx = MagicMock()
    async def mock_call_skill(skill_id, action, params=None):
        return SkillResult(success=True, message="ok", data={})
    ctx.call_skill = AsyncMock(side_effect=mock_call_skill)
    return ctx


# ── Scheduler Tick Tests ──────────────────────────────────


@pytest.mark.asyncio
async def test_tick_scheduler_calls_tick(skill):
    """Verify _tick_scheduler calls scheduler.tick() when available."""
    mock_scheduler = MagicMock()
    mock_scheduler.tick = AsyncMock(return_value=[
        SkillResult(success=True, message="task1 executed"),
        SkillResult(success=True, message="task2 executed"),
    ])
    
    mock_registry = {"scheduler": mock_scheduler}
    ctx = MagicMock()
    ctx._registry = mock_registry
    skill.context = ctx
    
    state = {"stats": {}}
    await skill._tick_scheduler(state)
    
    mock_scheduler.tick.assert_called_once()
    assert state["stats"]["scheduler_ticks"] == 1
    assert state["stats"]["scheduler_tasks_executed"] == 2


@pytest.mark.asyncio
async def test_tick_scheduler_no_scheduler(skill):
    """Verify _tick_scheduler is fail-silent when scheduler not registered."""
    ctx = MagicMock()
    ctx._registry = {}
    skill.context = ctx
    
    state = {"stats": {}}
    await skill._tick_scheduler(state)
    # Should not crash, no stats updated
    assert "scheduler_ticks" not in state["stats"]


@pytest.mark.asyncio
async def test_tick_scheduler_no_context(skill):
    """Verify _tick_scheduler handles missing context gracefully."""
    skill.context = None
    state = {"stats": {}}
    await skill._tick_scheduler(state)
    assert "scheduler_ticks" not in state["stats"]


@pytest.mark.asyncio
async def test_tick_scheduler_exception_handled(skill):
    """Verify _tick_scheduler swallows exceptions."""
    mock_scheduler = MagicMock()
    mock_scheduler.tick = AsyncMock(side_effect=RuntimeError("boom"))
    
    ctx = MagicMock()
    ctx._registry = {"scheduler": mock_scheduler}
    skill.context = ctx
    
    state = {"stats": {}}
    await skill._tick_scheduler(state)
    # Should not raise


@pytest.mark.asyncio
async def test_tick_scheduler_zero_tasks(skill):
    """Verify stats update correctly when no tasks are due."""
    mock_scheduler = MagicMock()
    mock_scheduler.tick = AsyncMock(return_value=[])
    
    ctx = MagicMock()
    ctx._registry = {"scheduler": mock_scheduler}
    skill.context = ctx
    
    state = {"stats": {}}
    await skill._tick_scheduler(state)
    
    assert state["stats"]["scheduler_ticks"] == 1
    assert state["stats"]["scheduler_tasks_executed"] == 0


@pytest.mark.asyncio
async def test_tick_scheduler_accumulates_stats(skill):
    """Verify scheduler stats accumulate across multiple ticks."""
    mock_scheduler = MagicMock()
    mock_scheduler.tick = AsyncMock(return_value=[
        SkillResult(success=True, message="done"),
    ])
    
    ctx = MagicMock()
    ctx._registry = {"scheduler": mock_scheduler}
    skill.context = ctx
    
    state = {"stats": {"scheduler_ticks": 5, "scheduler_tasks_executed": 10}}
    await skill._tick_scheduler(state)
    
    assert state["stats"]["scheduler_ticks"] == 6
    assert state["stats"]["scheduler_tasks_executed"] == 11


# ── Auto-Reputation Poll Tests ──────────────────────────────


@pytest.mark.asyncio
async def test_poll_auto_reputation(skill, mock_context):
    """Verify _poll_auto_reputation calls the bridge skill."""
    skill.context = mock_context
    state = {"stats": {}}
    
    await skill._poll_auto_reputation(state)
    
    mock_context.call_skill.assert_called_with(
        "auto_reputation_bridge", "poll", {}
    )
    assert state["stats"]["reputation_polls"] == 1


@pytest.mark.asyncio
async def test_poll_auto_reputation_no_context(skill):
    """Verify _poll_auto_reputation handles missing context."""
    skill.context = None
    state = {"stats": {}}
    await skill._poll_auto_reputation(state)
    assert "reputation_polls" not in state["stats"]


@pytest.mark.asyncio
async def test_poll_auto_reputation_exception(skill):
    """Verify _poll_auto_reputation swallows exceptions."""
    ctx = MagicMock()
    ctx.call_skill = AsyncMock(side_effect=RuntimeError("fail"))
    skill.context = ctx
    
    state = {"stats": {}}
    await skill._poll_auto_reputation(state)
    # Should not raise


# ── Goal Progress Monitor Tests ──────────────────────────────


@pytest.mark.asyncio
async def test_monitor_goal_progress(skill, mock_context):
    """Verify _monitor_goal_progress calls the bridge skill."""
    skill.context = mock_context
    state = {"stats": {}}
    
    await skill._monitor_goal_progress(state)
    
    mock_context.call_skill.assert_called_with(
        "goal_progress_events", "monitor", {}
    )
    assert state["stats"]["goal_progress_monitors"] == 1


@pytest.mark.asyncio
async def test_monitor_goal_progress_no_context(skill):
    """Verify _monitor_goal_progress handles missing context."""
    skill.context = None
    state = {"stats": {}}
    await skill._monitor_goal_progress(state)
    assert "goal_progress_monitors" not in state["stats"]


@pytest.mark.asyncio
async def test_monitor_goal_progress_exception(skill):
    """Verify _monitor_goal_progress swallows exceptions."""
    ctx = MagicMock()
    ctx.call_skill = AsyncMock(side_effect=RuntimeError("fail"))
    skill.context = ctx
    
    state = {"stats": {}}
    await skill._monitor_goal_progress(state)
    # Should not raise


# ── Integration: _step calls new methods ────────────────────


@pytest.mark.asyncio
async def test_step_calls_tick_scheduler(skill):
    """Verify _step calls _tick_scheduler during execution."""
    tick_called = []
    original_tick = skill._tick_scheduler
    
    async def track_tick(state):
        tick_called.append(True)
        return await original_tick(state)
    
    skill._tick_scheduler = track_tick
    
    # Mock context with all required skills
    ctx = MagicMock()
    async def mock_call(skill_id, action, params=None):
        if skill_id == "strategy" and action == "assess":
            return SkillResult(success=True, data={
                "pillars": {
                    "self_improvement": {"score": 70},
                    "revenue": {"score": 30},
                    "replication": {"score": 50},
                    "goal_setting": {"score": 60},
                },
                "weakest_pillar": "revenue",
                "summary": "Revenue is weak",
            })
        if skill_id == "goal_manager" and action == "next":
            return SkillResult(success=False, message="No goals")
        return SkillResult(success=True, data={})
    
    ctx.call_skill = AsyncMock(side_effect=mock_call)
    ctx._registry = {}
    skill.context = ctx
    
    await skill._step({})
    assert len(tick_called) == 1
