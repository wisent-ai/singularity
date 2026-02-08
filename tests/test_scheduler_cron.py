#!/usr/bin/env python3
"""Tests for SchedulerSkill cron integration."""

import pytest
import asyncio
from singularity.skills.scheduler import SchedulerSkill


@pytest.fixture
def scheduler():
    s = SchedulerSkill()
    return s


@pytest.mark.asyncio
async def test_schedule_cron_valid(scheduler):
    result = await scheduler.execute("schedule_cron", {
        "name": "Hourly check",
        "cron_expression": "0 * * * *",
        "skill_id": "strategy",
        "action": "assess",
    })
    assert result.success
    assert "cron" in result.message.lower() or "Hourly" in result.message
    assert result.data.get("schedule_type") == "cron"
    assert result.data.get("cron_expression") == "0 * * * *"


@pytest.mark.asyncio
async def test_schedule_cron_invalid_expression(scheduler):
    result = await scheduler.execute("schedule_cron", {
        "name": "Bad cron",
        "cron_expression": "invalid",
        "skill_id": "strategy",
        "action": "assess",
    })
    assert not result.success
    assert "invalid" in result.message.lower() or "Invalid" in result.message


@pytest.mark.asyncio
async def test_schedule_cron_alias(scheduler):
    result = await scheduler.execute("schedule_cron", {
        "name": "Daily report",
        "cron_expression": "@daily",
        "skill_id": "strategy",
        "action": "assess",
    })
    assert result.success
    assert result.data.get("schedule_type") == "cron"


@pytest.mark.asyncio
async def test_schedule_cron_missing_fields(scheduler):
    result = await scheduler.execute("schedule_cron", {
        "name": "",
        "cron_expression": "0 * * * *",
        "skill_id": "strategy",
        "action": "assess",
    })
    assert not result.success


@pytest.mark.asyncio
async def test_parse_cron_valid(scheduler):
    result = await scheduler.execute("parse_cron", {
        "cron_expression": "*/15 * * * *",
    })
    assert result.success
    assert "description" in result.data
    assert "upcoming_runs" in result.data
    assert len(result.data["upcoming_runs"]) == 5


@pytest.mark.asyncio
async def test_parse_cron_invalid(scheduler):
    result = await scheduler.execute("parse_cron", {
        "cron_expression": "bad cron expr here now",
    })
    assert not result.success


@pytest.mark.asyncio
async def test_cron_pause_resume(scheduler):
    r = await scheduler.execute("schedule_cron", {
        "name": "Pausable",
        "cron_expression": "*/5 * * * *",
        "skill_id": "strategy",
        "action": "assess",
    })
    task_id = r.data["id"]

    # Pause
    r2 = await scheduler.execute("pause", {"task_id": task_id})
    assert r2.success

    # Resume
    r3 = await scheduler.execute("resume", {"task_id": task_id})
    assert r3.success
    assert r3.data.get("status") == "pending"


@pytest.mark.asyncio
async def test_cron_cancel(scheduler):
    r = await scheduler.execute("schedule_cron", {
        "name": "Cancellable",
        "cron_expression": "@hourly",
        "skill_id": "strategy",
        "action": "assess",
    })
    task_id = r.data["id"]

    r2 = await scheduler.execute("cancel", {"task_id": task_id})
    assert r2.success
    assert r2.data.get("status") == "cancelled"


@pytest.mark.asyncio
async def test_cron_list_shows_description(scheduler):
    await scheduler.execute("schedule_cron", {
        "name": "Listed cron",
        "cron_expression": "*/10 * * * *",
        "skill_id": "strategy",
        "action": "assess",
    })
    r = await scheduler.execute("list", {})
    assert r.success
    tasks = r.data["tasks"]
    cron_tasks = [t for t in tasks if t.get("schedule_type") == "cron"]
    assert len(cron_tasks) >= 1
    assert "cron_description" in cron_tasks[0]


@pytest.mark.asyncio
async def test_cron_with_max_runs(scheduler):
    r = await scheduler.execute("schedule_cron", {
        "name": "Limited cron",
        "cron_expression": "* * * * *",
        "skill_id": "strategy",
        "action": "assess",
        "max_runs": 3,
    })
    assert r.success
    assert r.data.get("max_runs") == 3


@pytest.mark.asyncio
async def test_existing_schedule_action_still_works(scheduler):
    """Verify backward compatibility - old schedule action still works."""
    result = await scheduler.execute("schedule", {
        "name": "Old style",
        "skill_id": "strategy",
        "action": "assess",
        "delay_seconds": 60,
    })
    assert result.success
    assert result.data.get("schedule_type") == "once"
