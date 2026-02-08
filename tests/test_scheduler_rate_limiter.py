"""Tests for SchedulerSkill tick rate limiting."""

import asyncio
import time
import pytest
from unittest.mock import AsyncMock, MagicMock
from singularity.skills.scheduler import SchedulerSkill
from singularity.skills.base import SkillResult, SkillContext, SkillRegistry


@pytest.fixture
def scheduler():
    sched = SchedulerSkill()
    registry = SkillRegistry()
    ctx = SkillContext(registry=registry, agent_name="TestAgent")
    ctx.call_skill = AsyncMock(return_value=SkillResult(success=True, message="ok"))
    ctx.list_skills = MagicMock(return_value=["filesystem", "shell"])
    sched.set_context(ctx)
    # Disable rate limiting by default for isolated tests
    sched._rate_limit_config["enabled"] = False
    return sched


@pytest.fixture
def rate_limited_scheduler():
    sched = SchedulerSkill()
    registry = SkillRegistry()
    ctx = SkillContext(registry=registry, agent_name="TestAgent")
    ctx.call_skill = AsyncMock(return_value=SkillResult(success=True, message="ok"))
    ctx.list_skills = MagicMock(return_value=["filesystem", "shell"])
    sched.set_context(ctx)
    sched._rate_limit_config["min_tick_interval"] = 1.0
    sched._rate_limit_config["max_tasks_per_tick"] = 2
    sched._rate_limit_config["per_skill_cooldown"] = 1.0
    sched._rate_limit_config["enabled"] = True
    return sched


def _add_due_task(sched, name, skill_id="filesystem", action="ls"):
    """Helper to add a task that's immediately due."""
    import uuid
    from datetime import datetime
    from singularity.skills.scheduler import ScheduledTask
    task = ScheduledTask(
        id=f"sched_{uuid.uuid4().hex[:8]}",
        name=name, skill_id=skill_id, action=action, params={},
        schedule_type="recurring", interval_seconds=60,
        created_at=datetime.now().isoformat(),
        next_run_at=time.time() - 1,  # Already due
        status="pending", run_count=0, max_runs=0,
        enabled=True, cron_expression=None,
    )
    sched._tasks[task.id] = task
    return task


@pytest.mark.asyncio
async def test_tick_min_interval_throttles(rate_limited_scheduler):
    """Tick called too soon should be throttled."""
    _add_due_task(rate_limited_scheduler, "task1")
    results1 = await rate_limited_scheduler.tick()
    assert len(results1) == 1
    # Immediately tick again - should be throttled
    _add_due_task(rate_limited_scheduler, "task2")
    results2 = await rate_limited_scheduler.tick()
    assert len(results2) == 0
    assert rate_limited_scheduler._tick_stats["throttled_ticks"] >= 1


@pytest.mark.asyncio
async def test_max_tasks_per_tick(rate_limited_scheduler):
    """Only max_tasks_per_tick tasks should execute per tick."""
    rate_limited_scheduler._last_tick_at = 0  # Reset so tick isn't throttled
    rate_limited_scheduler._rate_limit_config["per_skill_cooldown"] = 0  # Disable cooldown for this test
    for i in range(5):
        _add_due_task(rate_limited_scheduler, f"task{i}", skill_id=f"skill{i}")
    results = await rate_limited_scheduler.tick()
    assert len(results) == 2  # max_tasks_per_tick = 2
    assert rate_limited_scheduler._tick_stats["last_tick_tasks_deferred"] == 3


@pytest.mark.asyncio
async def test_per_skill_cooldown(rate_limited_scheduler):
    """Same skill shouldn't run twice within cooldown period."""
    rate_limited_scheduler._last_tick_at = 0
    rate_limited_scheduler._rate_limit_config["max_tasks_per_tick"] = 10
    _add_due_task(rate_limited_scheduler, "task1", skill_id="filesystem")
    results1 = await rate_limited_scheduler.tick()
    assert len(results1) == 1
    # Add another task for same skill, tick again after resetting interval
    rate_limited_scheduler._last_tick_at = 0
    _add_due_task(rate_limited_scheduler, "task2", skill_id="filesystem")
    results2 = await rate_limited_scheduler.tick()
    assert len(results2) == 0  # Blocked by per-skill cooldown
    assert rate_limited_scheduler._tick_stats["skill_cooldown_hits"] >= 1


@pytest.mark.asyncio
async def test_priority_skills_bypass_limits(rate_limited_scheduler):
    """Priority skills should bypass both per-tick and cooldown limits."""
    rate_limited_scheduler._last_tick_at = 0
    rate_limited_scheduler._rate_limit_config["max_tasks_per_tick"] = 1
    rate_limited_scheduler._rate_limit_config["priority_skills"] = ["critical_skill"]
    rate_limited_scheduler._rate_limit_config["per_skill_cooldown"] = 0
    _add_due_task(rate_limited_scheduler, "normal", skill_id="normal_skill")
    _add_due_task(rate_limited_scheduler, "critical", skill_id="critical_skill")
    results = await rate_limited_scheduler.tick()
    # Both should execute: normal fills the 1 slot, critical bypasses limit
    assert len(results) == 2


@pytest.mark.asyncio
async def test_rate_limiting_disabled(scheduler):
    """With rate limiting disabled, all tasks execute."""
    for i in range(10):
        _add_due_task(scheduler, f"task{i}", skill_id=f"skill{i}")
    results = await scheduler.tick()
    assert len(results) == 10


@pytest.mark.asyncio
async def test_configure_rate_limit_action(scheduler):
    """configure_rate_limit action should update config."""
    result = await scheduler.execute("configure_rate_limit", {
        "min_tick_interval": 15.0,
        "max_tasks_per_tick": 3,
        "per_skill_cooldown": 20.0,
        "priority_skills": ["important"],
        "enabled": True,
    })
    assert result.success
    assert scheduler._rate_limit_config["min_tick_interval"] == 15.0
    assert scheduler._rate_limit_config["max_tasks_per_tick"] == 3
    assert scheduler._rate_limit_config["per_skill_cooldown"] == 20.0
    assert scheduler._rate_limit_config["priority_skills"] == ["important"]
    assert scheduler._rate_limit_config["enabled"] is True


@pytest.mark.asyncio
async def test_configure_rate_limit_validation(scheduler):
    """Invalid values should be rejected."""
    result = await scheduler.execute("configure_rate_limit", {"min_tick_interval": -1})
    assert not result.success
    result = await scheduler.execute("configure_rate_limit", {"max_tasks_per_tick": 0})
    assert not result.success
    result = await scheduler.execute("configure_rate_limit", {})
    assert not result.success


@pytest.mark.asyncio
async def test_rate_limit_status_action(rate_limited_scheduler):
    """rate_limit_status should return config and stats."""
    result = await rate_limited_scheduler.execute("rate_limit_status", {})
    assert result.success
    assert "config" in result.data
    assert "stats" in result.data
    assert result.data["config"]["min_tick_interval"] == 1.0
    assert result.data["config"]["max_tasks_per_tick"] == 2


@pytest.mark.asyncio
async def test_tick_stats_tracking(rate_limited_scheduler):
    """Tick stats should be tracked across multiple ticks."""
    rate_limited_scheduler._rate_limit_config["per_skill_cooldown"] = 0
    _add_due_task(rate_limited_scheduler, "t1", skill_id="s1")
    await rate_limited_scheduler.tick()
    assert rate_limited_scheduler._tick_stats["total_ticks"] == 1
    assert rate_limited_scheduler._tick_stats["last_tick_tasks_executed"] == 1
    # Immediate second tick should be throttled
    await rate_limited_scheduler.tick()
    assert rate_limited_scheduler._tick_stats["total_ticks"] == 2
    assert rate_limited_scheduler._tick_stats["throttled_ticks"] == 1
