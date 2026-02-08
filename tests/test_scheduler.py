"""Tests for SchedulerSkill - time-based task scheduling."""

import asyncio
import time
import pytest
from unittest.mock import AsyncMock, MagicMock
from singularity.skills.scheduler import SchedulerSkill, ScheduledTask
from singularity.skills.base import SkillResult, SkillContext, SkillRegistry


@pytest.fixture
def scheduler():
    return SchedulerSkill()


@pytest.fixture
def scheduler_with_context():
    """Scheduler with a mocked skill context for cross-skill calls."""
    sched = SchedulerSkill()
    registry = SkillRegistry()
    ctx = SkillContext(registry=registry, agent_name="TestAgent")
    ctx.call_skill = AsyncMock(return_value=SkillResult(success=True, message="mock result"))
    ctx.list_skills = MagicMock(return_value=["filesystem", "shell", "memory"])
    sched.set_context(ctx)
    return sched


@pytest.mark.asyncio
async def test_schedule_one_shot(scheduler_with_context):
    result = await scheduler_with_context.execute("schedule", {
        "name": "test task",
        "skill_id": "filesystem",
        "action": "ls",
        "delay_seconds": 10,
    })
    assert result.success
    assert "test task" in result.message
    assert result.data["schedule_type"] == "once"
    assert result.data["id"].startswith("sched_")


@pytest.mark.asyncio
async def test_schedule_recurring(scheduler_with_context):
    result = await scheduler_with_context.execute("schedule", {
        "name": "health check",
        "skill_id": "shell",
        "action": "bash",
        "params": {"command": "echo ok"},
        "recurring": True,
        "interval_seconds": 60,
    })
    assert result.success
    assert result.data["schedule_type"] == "recurring"
    assert result.data["interval_seconds"] == 60


@pytest.mark.asyncio
async def test_schedule_recurring_requires_interval(scheduler_with_context):
    result = await scheduler_with_context.execute("schedule", {
        "name": "bad task",
        "skill_id": "shell",
        "action": "bash",
        "recurring": True,
    })
    assert not result.success
    assert "interval" in result.message.lower()


@pytest.mark.asyncio
async def test_schedule_validates_skill(scheduler_with_context):
    result = await scheduler_with_context.execute("schedule", {
        "name": "bad skill",
        "skill_id": "nonexistent",
        "action": "do",
    })
    assert not result.success
    assert "not found" in result.message.lower()


@pytest.mark.asyncio
async def test_cancel_task(scheduler_with_context):
    sched_result = await scheduler_with_context.execute("schedule", {
        "name": "to cancel",
        "skill_id": "filesystem",
        "action": "ls",
        "delay_seconds": 999,
    })
    task_id = sched_result.data["id"]
    cancel_result = await scheduler_with_context.execute("cancel", {"task_id": task_id})
    assert cancel_result.success
    assert cancel_result.data["status"] == "cancelled"


@pytest.mark.asyncio
async def test_cancel_nonexistent(scheduler_with_context):
    result = await scheduler_with_context.execute("cancel", {"task_id": "nope"})
    assert not result.success


@pytest.mark.asyncio
async def test_list_tasks(scheduler_with_context):
    await scheduler_with_context.execute("schedule", {
        "name": "task 1", "skill_id": "shell", "action": "bash", "delay_seconds": 100
    })
    await scheduler_with_context.execute("schedule", {
        "name": "task 2", "skill_id": "memory", "action": "recall", "delay_seconds": 200
    })
    result = await scheduler_with_context.execute("list", {})
    assert result.success
    assert result.data["total"] == 2


@pytest.mark.asyncio
async def test_list_excludes_completed(scheduler_with_context):
    sched = await scheduler_with_context.execute("schedule", {
        "name": "quick task", "skill_id": "shell", "action": "bash", "delay_seconds": 0
    })
    task_id = sched.data["id"]
    # Run it to complete it
    await scheduler_with_context.execute("run_now", {"task_id": task_id})
    result = await scheduler_with_context.execute("list", {})
    assert result.data["total"] == 0  # Completed task excluded
    result2 = await scheduler_with_context.execute("list", {"include_completed": True})
    assert result2.data["total"] == 1


@pytest.mark.asyncio
async def test_run_now(scheduler_with_context):
    sched = await scheduler_with_context.execute("schedule", {
        "name": "run me", "skill_id": "shell", "action": "bash",
        "params": {"command": "echo hello"}, "delay_seconds": 9999
    })
    task_id = sched.data["id"]
    result = await scheduler_with_context.execute("run_now", {"task_id": task_id})
    assert result.success
    assert "mock result" in result.message
    scheduler_with_context.context.call_skill.assert_called_once_with(
        "shell", "bash", {"command": "echo hello"}
    )


@pytest.mark.asyncio
async def test_pause_resume(scheduler_with_context):
    sched = await scheduler_with_context.execute("schedule", {
        "name": "recurring job", "skill_id": "shell", "action": "bash",
        "recurring": True, "interval_seconds": 30
    })
    task_id = sched.data["id"]
    pause_result = await scheduler_with_context.execute("pause", {"task_id": task_id})
    assert pause_result.success
    assert not pause_result.data["enabled"]
    resume_result = await scheduler_with_context.execute("resume", {"task_id": task_id})
    assert resume_result.success
    assert resume_result.data["enabled"]


@pytest.mark.asyncio
async def test_pause_only_recurring(scheduler_with_context):
    sched = await scheduler_with_context.execute("schedule", {
        "name": "one shot", "skill_id": "shell", "action": "bash", "delay_seconds": 10
    })
    result = await scheduler_with_context.execute("pause", {"task_id": sched.data["id"]})
    assert not result.success
    assert "recurring" in result.message.lower()


@pytest.mark.asyncio
async def test_history(scheduler_with_context):
    sched = await scheduler_with_context.execute("schedule", {
        "name": "history test", "skill_id": "shell", "action": "bash", "delay_seconds": 0
    })
    await scheduler_with_context.execute("run_now", {"task_id": sched.data["id"]})
    result = await scheduler_with_context.execute("history", {})
    assert result.success
    assert result.data["total"] == 1
    assert result.data["history"][0]["task_name"] == "history test"


@pytest.mark.asyncio
async def test_pending_tasks(scheduler_with_context):
    # Schedule one due now and one far in future
    await scheduler_with_context.execute("schedule", {
        "name": "due now", "skill_id": "shell", "action": "bash", "delay_seconds": 0
    })
    await scheduler_with_context.execute("schedule", {
        "name": "due later", "skill_id": "shell", "action": "bash", "delay_seconds": 9999
    })
    result = await scheduler_with_context.execute("pending", {"within_seconds": 5})
    assert result.success
    assert result.data["count"] == 1
    assert result.data["pending"][0]["name"] == "due now"


@pytest.mark.asyncio
async def test_tick_executes_due_tasks(scheduler_with_context):
    await scheduler_with_context.execute("schedule", {
        "name": "due task", "skill_id": "shell", "action": "bash", "delay_seconds": 0
    })
    results = await scheduler_with_context.tick()
    assert len(results) == 1
    assert results[0].success


@pytest.mark.asyncio
async def test_tick_skips_future_tasks(scheduler_with_context):
    await scheduler_with_context.execute("schedule", {
        "name": "future task", "skill_id": "shell", "action": "bash", "delay_seconds": 9999
    })
    results = await scheduler_with_context.tick()
    assert len(results) == 0


@pytest.mark.asyncio
async def test_recurring_reschedules_after_run(scheduler_with_context):
    sched = await scheduler_with_context.execute("schedule", {
        "name": "repeater", "skill_id": "shell", "action": "bash",
        "recurring": True, "interval_seconds": 30, "delay_seconds": 0
    })
    task_id = sched.data["id"]
    await scheduler_with_context.tick()
    # Task should still be pending (rescheduled)
    task = scheduler_with_context._tasks[task_id]
    assert task.status == "pending"
    assert task.run_count == 1
    assert task.next_run_at > time.time()


@pytest.mark.asyncio
async def test_max_runs_stops_recurring(scheduler_with_context):
    sched = await scheduler_with_context.execute("schedule", {
        "name": "limited", "skill_id": "shell", "action": "bash",
        "recurring": True, "interval_seconds": 1, "delay_seconds": 0, "max_runs": 1
    })
    task_id = sched.data["id"]
    await scheduler_with_context.tick()
    task = scheduler_with_context._tasks[task_id]
    assert task.status == "completed"
    assert task.run_count == 1


@pytest.mark.asyncio
async def test_get_due_count(scheduler_with_context):
    assert scheduler_with_context.get_due_count() == 0
    await scheduler_with_context.execute("schedule", {
        "name": "due", "skill_id": "shell", "action": "bash", "delay_seconds": 0
    })
    assert scheduler_with_context.get_due_count() == 1


@pytest.mark.asyncio
async def test_no_context_returns_error(scheduler):
    sched = await scheduler.execute("schedule", {
        "name": "test", "skill_id": "shell", "action": "bash"
    })
    assert sched.success  # Scheduling works without context
    result = await scheduler.execute("run_now", {"task_id": sched.data["id"]})
    assert not result.success
    assert "context" in result.message.lower()


@pytest.mark.asyncio
async def test_manifest(scheduler):
    m = scheduler.manifest
    assert m.skill_id == "scheduler"
    assert m.category == "autonomy"
    assert len(m.actions) == 10


@pytest.mark.asyncio
async def test_unknown_action(scheduler):
    result = await scheduler.execute("nonexistent", {})
    assert not result.success
