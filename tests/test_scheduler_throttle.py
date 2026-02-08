"""Tests for SchedulerSkill tick rate limiting / throttle feature."""

import asyncio
import time
import pytest
from unittest.mock import AsyncMock, MagicMock
from singularity.skills.scheduler import SchedulerSkill
from singularity.skills.base import SkillResult, SkillContext, SkillRegistry


@pytest.fixture
def sched():
    """Scheduler with mocked context and throttling enabled."""
    s = SchedulerSkill()
    registry = SkillRegistry()
    ctx = SkillContext(registry=registry, agent_name="TestAgent")
    ctx.call_skill = AsyncMock(return_value=SkillResult(success=True, message="ok"))
    ctx.list_skills = MagicMock(return_value=["filesystem", "shell"])
    s.set_context(ctx)
    return s


def _add_due_tasks(sched, n=10):
    """Helper: add n tasks that are immediately due."""
    for i in range(n):
        task_id = f"sched_test{i:03d}"
        from singularity.skills.scheduler import ScheduledTask
        sched._tasks[task_id] = ScheduledTask(
            id=task_id, name=f"task_{i}", skill_id="filesystem",
            action="ls", params={}, schedule_type="recurring",
            interval_seconds=300, created_at="2025-01-01T00:00:00",
            next_run_at=time.time() - 10,  # already overdue
        )


@pytest.mark.asyncio
async def test_throttle_defaults(sched):
    """Throttle config has sensible defaults."""
    cfg = sched._throttle_config
    assert cfg["enabled"] is True
    assert cfg["min_tick_interval"] == 5.0
    assert cfg["max_tasks_per_tick"] == 5
    assert cfg["max_tick_duration"] == 30.0
    assert cfg["burst_max_tasks"] == 20


@pytest.mark.asyncio
async def test_max_tasks_per_tick(sched):
    """Tick executes at most max_tasks_per_tick tasks."""
    _add_due_tasks(sched, 10)
    sched._throttle_config["max_tasks_per_tick"] = 3
    sched._throttle_config["min_tick_interval"] = 0  # disable interval for test
    results = await sched.tick()
    assert len(results) == 3


@pytest.mark.asyncio
async def test_min_tick_interval_skip(sched):
    """Tick is skipped if called too soon after previous tick."""
    _add_due_tasks(sched, 5)
    sched._throttle_config["min_tick_interval"] = 10.0
    sched._last_tick_at = time.time() - 2  # last tick 2s ago (< 10s)
    results = await sched.tick()
    assert len(results) == 0
    assert sched._throttle_stats["skipped_ticks"] >= 1


@pytest.mark.asyncio
async def test_min_tick_interval_allow(sched):
    """Tick proceeds when enough time has elapsed."""
    _add_due_tasks(sched, 2)
    sched._throttle_config["min_tick_interval"] = 1.0
    sched._last_tick_at = time.time() - 5  # last tick 5s ago (> 1s)
    results = await sched.tick()
    assert len(results) == 2


@pytest.mark.asyncio
async def test_burst_protection(sched):
    """Burst protection limits tasks within the burst window."""
    sched._throttle_config["burst_window"] = 60.0
    sched._throttle_config["burst_max_tasks"] = 5
    sched._throttle_config["max_tasks_per_tick"] = 10
    sched._throttle_config["min_tick_interval"] = 0
    # Simulate 4 tasks already run in the window
    sched._tick_history = [
        {"timestamp": time.time() - 10, "tasks_run": 4, "duration": 0.1, "throttled": False}
    ]
    _add_due_tasks(sched, 8)
    results = await sched.tick()
    assert len(results) <= 1  # Only 1 remaining in burst budget (5 - 4)


@pytest.mark.asyncio
async def test_throttle_disabled(sched):
    """When throttle is disabled, all due tasks execute."""
    sched._throttle_config["enabled"] = False
    _add_due_tasks(sched, 8)
    results = await sched.tick()
    assert len(results) == 8


@pytest.mark.asyncio
async def test_priority_ordering(sched):
    """Most overdue tasks execute first when throttled."""
    sched._throttle_config["max_tasks_per_tick"] = 2
    sched._throttle_config["min_tick_interval"] = 0
    sched._throttle_config["priority_on_throttle"] = True
    from singularity.skills.scheduler import ScheduledTask
    now = time.time()
    # Task A is 100s overdue, Task B is 10s overdue
    sched._tasks["a"] = ScheduledTask(
        id="a", name="old_task", skill_id="filesystem", action="ls",
        params={}, schedule_type="recurring", interval_seconds=300,
        created_at="2025-01-01", next_run_at=now - 100,
    )
    sched._tasks["b"] = ScheduledTask(
        id="b", name="new_task", skill_id="filesystem", action="ls",
        params={}, schedule_type="recurring", interval_seconds=300,
        created_at="2025-01-01", next_run_at=now - 10,
    )
    sched._tasks["c"] = ScheduledTask(
        id="c", name="newest_task", skill_id="filesystem", action="ls",
        params={}, schedule_type="recurring", interval_seconds=300,
        created_at="2025-01-01", next_run_at=now - 1,
    )
    results = await sched.tick()
    assert len(results) == 2
    # The two most overdue (a, b) should have been picked
    executed_names = {r.data["task"]["name"] for r in results}
    assert "old_task" in executed_names


@pytest.mark.asyncio
async def test_configure_throttle(sched):
    """configure_throttle action updates settings."""
    result = await sched.execute("configure_throttle", {
        "min_tick_interval": 10.0,
        "max_tasks_per_tick": 8,
    })
    assert result.success
    assert sched._throttle_config["min_tick_interval"] == 10.0
    assert sched._throttle_config["max_tasks_per_tick"] == 8


@pytest.mark.asyncio
async def test_configure_throttle_validation(sched):
    """configure_throttle rejects invalid values."""
    result = await sched.execute("configure_throttle", {
        "min_tick_interval": -5,
    })
    assert not result.success
    assert "non-negative" in result.message


@pytest.mark.asyncio
async def test_configure_throttle_no_params(sched):
    """configure_throttle fails with no valid params."""
    result = await sched.execute("configure_throttle", {"invalid_key": 5})
    assert not result.success


@pytest.mark.asyncio
async def test_throttle_status(sched):
    """throttle_status returns config and stats."""
    _add_due_tasks(sched, 3)
    sched._throttle_config["min_tick_interval"] = 0
    await sched.tick()
    result = await sched.execute("throttle_status", {})
    assert result.success
    assert "config" in result.data
    assert "stats" in result.data
    assert result.data["stats"]["total_ticks"] >= 1


@pytest.mark.asyncio
async def test_throttle_status_with_history(sched):
    """throttle_status includes tick history when requested."""
    _add_due_tasks(sched, 2)
    sched._throttle_config["min_tick_interval"] = 0
    await sched.tick()
    result = await sched.execute("throttle_status", {
        "include_history": True, "history_limit": 5,
    })
    assert result.success
    assert "tick_history" in result.data
    assert "history_summary" in result.data
    assert len(result.data["tick_history"]) >= 1


@pytest.mark.asyncio
async def test_tick_records_history(sched):
    """Each tick is recorded in tick_history."""
    _add_due_tasks(sched, 2)
    sched._throttle_config["min_tick_interval"] = 0
    await sched.tick()
    assert len(sched._tick_history) >= 1
    entry = sched._tick_history[-1]
    assert "timestamp" in entry
    assert "tasks_run" in entry
    assert entry["tasks_run"] == 2


@pytest.mark.asyncio
async def test_throttle_stats_tracking(sched):
    """Stats are updated across multiple ticks."""
    _add_due_tasks(sched, 10)
    sched._throttle_config["max_tasks_per_tick"] = 3
    sched._throttle_config["min_tick_interval"] = 0
    await sched.tick()
    stats = sched._throttle_stats
    assert stats["total_ticks"] >= 1
    assert stats["throttled_ticks"] >= 1
    assert stats["tasks_deferred"] >= 1
