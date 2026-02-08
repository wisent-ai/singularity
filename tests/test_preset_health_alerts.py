"""Tests for SchedulerPresetsSkill health alerts feature."""

import pytest
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch
from singularity.skills.scheduler_presets import (
    SchedulerPresetsSkill, BUILTIN_PRESETS,
)
from singularity.skills.base import SkillResult, SkillContext, SkillRegistry


def _mock_call_skill_factory():
    """Create a mock call_skill that returns unique task IDs."""
    counter = [0]
    async def mock_call_skill(*args, **kwargs):
        counter[0] += 1
        return SkillResult(
            success=True, message="scheduled",
            data={"id": f"sched_mock_{counter[0]}"}
        )
    return mock_call_skill


@pytest.fixture
def presets_skill(tmp_path):
    """Create a clean presets skill with isolated state."""
    with patch("singularity.skills.scheduler_presets.DATA_DIR", tmp_path), \
         patch("singularity.skills.scheduler_presets.PRESETS_FILE", tmp_path / "scheduler_presets.json"):
        skill = SchedulerPresetsSkill()
        registry = SkillRegistry()
        ctx = SkillContext(registry=registry, agent_name="TestAgent")
        ctx.call_skill = _mock_call_skill_factory()
        ctx.list_skills = MagicMock(return_value=list({
            s.skill_id for p in BUILTIN_PRESETS.values() for s in p.schedules
        }))
        skill.set_context(ctx)
        yield skill


def test_manifest_includes_health_alert_actions(presets_skill):
    """Manifest should include health_alerts, configure_alerts, alert_history."""
    actions = [a.name for a in presets_skill.manifest.actions]
    assert "health_alerts" in actions
    assert "configure_alerts" in actions
    assert "alert_history" in actions


def test_version_bumped(presets_skill):
    """Version should be 3.0.0 with health alerts."""
    assert presets_skill.manifest.version == "3.0.0"


def test_default_alert_config(presets_skill):
    """Default alert config should have sensible thresholds."""
    cfg = presets_skill._alert_config
    assert cfg["failure_streak_threshold"] == 3
    assert cfg["success_rate_threshold"] == 50.0
    assert cfg["recovery_streak_threshold"] == 2
    assert cfg["emit_on_task_fail"] is True
    assert cfg["emit_on_preset_unhealthy"] is True
    assert cfg["emit_on_preset_recovered"] is True


@pytest.mark.asyncio
async def test_health_alerts_no_applied_presets(presets_skill):
    """Health alerts on empty should return 0 presets scanned."""
    result = await presets_skill.execute("health_alerts", {})
    assert result.success
    assert result.data["total_healthy"] == 0
    assert result.data["total_degraded"] == 0
    assert result.data["total_unhealthy"] == 0


@pytest.mark.asyncio
async def test_health_alerts_with_healthy_tasks(presets_skill):
    """Health alerts should report healthy when tasks have no failures."""
    await presets_skill.execute("apply", {"preset_id": "health_monitoring"})
    task_ids = presets_skill._applied["health_monitoring"]["task_ids"]

    scheduler_tasks = {}
    for tid in task_ids:
        scheduler_tasks[tid] = {
            "id": tid, "name": f"Task {tid}", "last_success": True,
            "run_count": 5, "status": "pending",
        }
    execution_history = []
    for tid in task_ids:
        for i in range(5):
            execution_history.append({
                "task_id": tid, "success": True,
                "executed_at": f"2026-01-01T00:0{i}:00",
            })

    with patch.object(presets_skill, "_read_scheduler_data",
                       return_value=(scheduler_tasks, execution_history)):
        result = await presets_skill.execute("health_alerts", {})

    assert result.success
    assert result.data["total_healthy"] == 1
    assert result.data["total_unhealthy"] == 0
    assert len(result.data["alerts_emitted"]) == 0


@pytest.mark.asyncio
async def test_health_alerts_detects_failure_streak(presets_skill):
    """Health alerts should detect tasks with consecutive failures."""
    await presets_skill.execute("apply", {"preset_id": "self_tuning"})
    task_ids = presets_skill._applied["self_tuning"]["task_ids"]
    tid = task_ids[0]

    scheduler_tasks = {
        tid: {"id": tid, "name": "Auto-Tune", "last_success": False,
               "run_count": 5, "status": "pending"},
    }
    # 3 consecutive failures (meets threshold of 3)
    execution_history = [
        {"task_id": tid, "success": False, "executed_at": f"2026-01-01T00:0{i}:00"}
        for i in range(3)
    ]

    with patch.object(presets_skill, "_read_scheduler_data",
                       return_value=(scheduler_tasks, execution_history)):
        result = await presets_skill.execute("health_alerts", {})

    assert result.success
    # self_tuning has 1 task, all failing = unhealthy
    assert result.data["total_unhealthy"] == 1
    assert len(result.data["alerts_emitted"]) > 0
    alert = result.data["alerts_emitted"][0]
    assert alert["topic"] == "preset.task_failed"
    assert "failure_streak" in str(alert["reasons"])


@pytest.mark.asyncio
async def test_health_alerts_detects_low_success_rate(presets_skill):
    """Health alerts should detect tasks with low success rate."""
    await presets_skill.execute("apply", {"preset_id": "self_tuning"})
    task_ids = presets_skill._applied["self_tuning"]["task_ids"]
    tid = task_ids[0]

    scheduler_tasks = {
        tid: {"id": tid, "name": "Auto-Tune", "last_success": True,
               "run_count": 10, "status": "pending"},
    }
    # 1 success then 4 failures = 20% success rate (below 50% threshold)
    execution_history = [
        {"task_id": tid, "success": True, "executed_at": "2026-01-01T00:00:00"},
        {"task_id": tid, "success": False, "executed_at": "2026-01-01T00:01:00"},
        {"task_id": tid, "success": False, "executed_at": "2026-01-01T00:02:00"},
        {"task_id": tid, "success": False, "executed_at": "2026-01-01T00:03:00"},
        {"task_id": tid, "success": False, "executed_at": "2026-01-01T00:04:00"},
    ]

    with patch.object(presets_skill, "_read_scheduler_data",
                       return_value=(scheduler_tasks, execution_history)):
        result = await presets_skill.execute("health_alerts", {})

    assert result.success
    assert result.data["total_unhealthy"] >= 1


@pytest.mark.asyncio
async def test_health_alerts_recovery(presets_skill):
    """Health alerts should detect recovery after failures resolve."""
    await presets_skill.execute("apply", {"preset_id": "self_tuning"})
    task_ids = presets_skill._applied["self_tuning"]["task_ids"]
    tid = task_ids[0]

    # Put the task in alerting state
    presets_skill._alert_state[tid] = {
        "failure_streak": 3, "success_streak": 0, "status": "alerting",
        "preset_id": "self_tuning", "task_name": "Auto-Tune",
        "last_checked": None, "total_alerts_emitted": 1,
    }

    scheduler_tasks = {
        tid: {"id": tid, "name": "Auto-Tune", "last_success": True,
               "run_count": 10, "status": "pending"},
    }
    # 2 recent successes (meets recovery threshold of 2)
    execution_history = [
        {"task_id": tid, "success": True, "executed_at": "2026-01-01T00:05:00"},
        {"task_id": tid, "success": True, "executed_at": "2026-01-01T00:06:00"},
    ]

    with patch.object(presets_skill, "_read_scheduler_data",
                       return_value=(scheduler_tasks, execution_history)):
        result = await presets_skill.execute("health_alerts", {})

    assert result.success
    assert len(result.data["recoveries_emitted"]) > 0
    recovery = result.data["recoveries_emitted"][0]
    assert recovery["topic"] == "preset.task_recovered"
    assert presets_skill._alert_state[tid]["status"] == "healthy"


@pytest.mark.asyncio
async def test_health_alerts_degraded_preset(presets_skill):
    """Preset with some healthy and some unhealthy tasks = degraded."""
    await presets_skill.execute("apply", {"preset_id": "health_monitoring"})
    task_ids = presets_skill._applied["health_monitoring"]["task_ids"]
    assert len(task_ids) >= 2, "health_monitoring should have 2+ tasks"

    tid_healthy = task_ids[0]
    tid_failing = task_ids[1]

    scheduler_tasks = {
        tid_healthy: {"id": tid_healthy, "name": "Healthy Task", "last_success": True, "status": "pending"},
        tid_failing: {"id": tid_failing, "name": "Failing Task", "last_success": False, "status": "pending"},
    }
    execution_history = [
        {"task_id": tid_healthy, "success": True, "executed_at": f"2026-01-01T00:0{i}:00"}
        for i in range(5)
    ] + [
        {"task_id": tid_failing, "success": False, "executed_at": f"2026-01-01T00:0{i}:00"}
        for i in range(3)
    ]

    with patch.object(presets_skill, "_read_scheduler_data",
                       return_value=(scheduler_tasks, execution_history)):
        result = await presets_skill.execute("health_alerts", {})

    assert result.success
    assert result.data["total_degraded"] == 1


@pytest.mark.asyncio
async def test_configure_alerts(presets_skill):
    """Configure alerts should update thresholds."""
    result = await presets_skill.execute("configure_alerts", {
        "failure_streak_threshold": 5,
        "success_rate_threshold": 75.0,
    })
    assert result.success
    assert presets_skill._alert_config["failure_streak_threshold"] == 5
    assert presets_skill._alert_config["success_rate_threshold"] == 75.0


@pytest.mark.asyncio
async def test_configure_alerts_validation(presets_skill):
    """Configure alerts should validate threshold ranges."""
    result = await presets_skill.execute("configure_alerts", {
        "failure_streak_threshold": 0,  # below minimum of 1
    })
    assert not result.success


@pytest.mark.asyncio
async def test_configure_alerts_no_changes(presets_skill):
    """Configure alerts with no params should return current config."""
    result = await presets_skill.execute("configure_alerts", {})
    assert result.success
    assert "current_config" in result.data


@pytest.mark.asyncio
async def test_alert_history_empty(presets_skill):
    """Alert history should work when empty."""
    result = await presets_skill.execute("alert_history", {})
    assert result.success
    assert result.data["total_events"] == 0
    assert result.data["events"] == []


@pytest.mark.asyncio
async def test_alert_history_after_alerts(presets_skill):
    """Alert history should contain events after health_alerts scan."""
    await presets_skill.execute("apply", {"preset_id": "self_tuning"})
    task_ids = presets_skill._applied["self_tuning"]["task_ids"]
    tid = task_ids[0]

    scheduler_tasks = {
        tid: {"id": tid, "name": "Auto-Tune", "last_success": False,
               "run_count": 5, "status": "pending"},
    }
    execution_history = [
        {"task_id": tid, "success": False, "executed_at": f"2026-01-01T00:0{i}:00"}
        for i in range(4)
    ]

    with patch.object(presets_skill, "_read_scheduler_data",
                       return_value=(scheduler_tasks, execution_history)):
        await presets_skill.execute("health_alerts", {})

    result = await presets_skill.execute("alert_history", {})
    assert result.success
    assert result.data["total_events"] > 0


@pytest.mark.asyncio
async def test_health_alerts_filter_by_preset(presets_skill):
    """Health alerts should filter by preset_id."""
    result = await presets_skill.execute("health_alerts", {
        "preset_id": "nonexistent",
    })
    assert not result.success

    await presets_skill.execute("apply", {"preset_id": "self_tuning"})
    scheduler_tasks = {}
    with patch.object(presets_skill, "_read_scheduler_data",
                       return_value=(scheduler_tasks, [])):
        result = await presets_skill.execute("health_alerts", {
            "preset_id": "self_tuning",
        })
    assert result.success


@pytest.mark.asyncio
async def test_alert_state_persisted(presets_skill, tmp_path):
    """Alert state should be included in saved state."""
    with patch("singularity.skills.scheduler_presets.DATA_DIR", tmp_path), \
         patch("singularity.skills.scheduler_presets.PRESETS_FILE", tmp_path / "scheduler_presets.json"):
        presets_skill._alert_state["test_task"] = {
            "failure_streak": 2, "status": "alerting",
        }
        presets_skill._alert_history.append({
            "topic": "preset.task_failed", "timestamp": "2026-01-01",
        })
        presets_skill._save_state()
        # Reload into a fresh instance
        skill2 = SchedulerPresetsSkill()
        assert skill2._alert_state.get("test_task", {}).get("status") == "alerting"
        assert len(skill2._alert_history) >= 1
