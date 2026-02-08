"""Tests for SchedulerPresetsSkill dashboard action."""

import json
import asyncio
import time
import pytest
from pathlib import Path
from singularity.skills.scheduler_presets import (
    SchedulerPresetsSkill, PRESETS_FILE, DATA_DIR,
)


@pytest.fixture(autouse=True)
def clean_data():
    if PRESETS_FILE.exists():
        PRESETS_FILE.unlink()
    sched = DATA_DIR / "scheduler.json"
    if sched.exists():
        sched.unlink()
    hist = DATA_DIR / "scheduler_history.json"
    if hist.exists():
        hist.unlink()
    yield
    if PRESETS_FILE.exists():
        PRESETS_FILE.unlink()
    if sched.exists():
        sched.unlink()
    if hist.exists():
        hist.unlink()


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def make_skill():
    return SchedulerPresetsSkill()


def test_dashboard_in_manifest():
    skill = make_skill()
    actions = [a.name for a in skill.manifest.actions]
    assert "dashboard" in actions


def test_dashboard_no_presets():
    skill = make_skill()
    result = run(skill.execute("dashboard", {}))
    assert result.success
    assert result.data["summary"]["overall_health"] == "no_presets"
    assert result.data["summary"]["presets_applied"] == 0
    assert result.data["summary"]["total_tasks"] == 0


def test_dashboard_with_applied_preset():
    skill = make_skill()
    run(skill.execute("apply", {"preset_id": "health_monitoring"}))
    result = run(skill.execute("dashboard", {}))
    assert result.success
    summary = result.data["summary"]
    assert summary["presets_applied"] == 1
    assert summary["total_tasks"] > 0
    presets = result.data["presets"]
    assert len(presets) == 1
    assert presets[0]["preset_id"] == "health_monitoring"
    assert presets[0]["task_count"] > 0
    assert len(presets[0]["tasks"]) > 0


def test_dashboard_task_details():
    skill = make_skill()
    run(skill.execute("apply", {"preset_id": "self_tuning"}))
    result = run(skill.execute("dashboard", {}))
    assert result.success
    preset = result.data["presets"][0]
    for task in preset["tasks"]:
        assert "task_id" in task
        assert "name" in task
        assert "health" in task
        assert "next_run_in" in task
        assert "interval" in task
        assert "run_count" in task


def test_dashboard_filter_preset():
    skill = make_skill()
    run(skill.execute("apply", {"preset_id": "health_monitoring"}))
    run(skill.execute("apply", {"preset_id": "self_tuning"}))
    result = run(skill.execute("dashboard", {"preset_id": "self_tuning"}))
    assert result.success
    assert len(result.data["presets"]) == 1
    assert result.data["presets"][0]["preset_id"] == "self_tuning"


def test_dashboard_filter_invalid_preset():
    skill = make_skill()
    result = run(skill.execute("dashboard", {"preset_id": "nonexistent"}))
    assert not result.success
    assert "not applied" in result.message


def test_dashboard_health_status():
    skill = make_skill()
    run(skill.execute("apply", {"preset_id": "health_monitoring"}))
    result = run(skill.execute("dashboard", {}))
    summary = result.data["summary"]
    assert summary["overall_health"] in [
        "all_healthy", "mostly_healthy", "degraded", "unhealthy",
    ]


def test_dashboard_success_rate_format():
    skill = make_skill()
    run(skill.execute("apply", {"preset_id": "alert_polling"}))
    result = run(skill.execute("dashboard", {}))
    preset = result.data["presets"][0]
    # success_rate should be "n/a" when no executions
    assert preset["success_rate"] == "n/a"
    assert preset["total_executions"] == 0


def test_dashboard_message_format():
    skill = make_skill()
    run(skill.execute("apply", {"preset_id": "self_assessment"}))
    result = run(skill.execute("dashboard", {}))
    assert "[" in result.message  # has health icon
    assert "presets" in result.message
    assert "tasks" in result.message


def test_dashboard_multiple_presets():
    skill = make_skill()
    run(skill.execute("apply_all", {}))
    result = run(skill.execute("dashboard", {}))
    assert result.success
    summary = result.data["summary"]
    assert summary["presets_applied"] >= 3
    assert summary["total_tasks"] >= 5
    assert len(result.data["presets"]) >= 3
