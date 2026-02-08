"""Tests for SchedulerPresetsSkill - pre-built automation schedule bundles."""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from singularity.skills.scheduler_presets import SchedulerPresetsSkill, PRESETS
from singularity.skills.base import SkillResult, SkillContext, SkillRegistry


@pytest.fixture
def skill(tmp_path):
    s = SchedulerPresetsSkill()
    # Use temp path to avoid cross-test contamination
    s._data_dir = tmp_path
    s._presets_file = tmp_path / "scheduler_presets.json"
    s._installed_tasks = {}
    s._active_presets = {}
    return s


@pytest.fixture
def skill_with_context(tmp_path):
    s = SchedulerPresetsSkill()
    s._data_dir = tmp_path
    s._presets_file = tmp_path / "scheduler_presets.json"
    s._installed_tasks = {}
    s._active_presets = {}
    registry = SkillRegistry()
    ctx = SkillContext(registry=registry, agent_name="TestAgent")

    call_count = {"n": 0}

    async def mock_call_skill(skill_id, action, params):
        call_count["n"] += 1
        if action == "schedule":
            return SkillResult(
                success=True,
                message="Scheduled",
                data={"id": f"sched_{call_count['n']:04d}"},
            )
        elif action == "cancel":
            return SkillResult(success=True, message="Cancelled")
        elif action == "list":
            return SkillResult(success=True, message="Listed", data={"tasks": []})
        elif action in ("pause", "resume"):
            return SkillResult(success=True, message=f"{action}d")
        return SkillResult(success=True, message="ok")

    ctx.call_skill = mock_call_skill
    ctx.list_skills = MagicMock(return_value=["scheduler"])
    s.set_context(ctx)
    return s


@pytest.mark.asyncio
async def test_list_presets(skill):
    result = await skill.execute("list", {})
    assert result.success
    assert result.data["total"] >= 5  # 5 built-in presets
    names = [p["id"] for p in result.data["presets"]]
    assert "self_improvement" in names
    assert "operations" in names
    assert "revenue" in names
    assert "replication" in names
    assert "full_autonomy" in names


@pytest.mark.asyncio
async def test_activate_dry_run(skill):
    result = await skill.execute("activate", {
        "preset": "operations",
        "dry_run": True,
    })
    assert result.success
    assert result.data["dry_run"] is True
    assert len(result.data["tasks"]) == 3
    assert "operations" not in skill._installed_tasks


@pytest.mark.asyncio
async def test_activate_preset(skill_with_context):
    result = await skill_with_context.execute("activate", {"preset": "operations"})
    assert result.success
    assert result.data["installed"] == 3
    assert len(result.data["task_ids"]) == 3
    assert "operations" in skill_with_context._installed_tasks


@pytest.mark.asyncio
async def test_activate_unknown_preset(skill):
    result = await skill.execute("activate", {"preset": "nonexistent"})
    assert not result.success
    assert "Unknown preset" in result.message


@pytest.mark.asyncio
async def test_activate_already_active(skill_with_context):
    await skill_with_context.execute("activate", {"preset": "operations"})
    result = await skill_with_context.execute("activate", {"preset": "operations"})
    assert not result.success
    assert "already active" in result.message


@pytest.mark.asyncio
async def test_activate_with_multiplier(skill):
    result = await skill.execute("activate", {
        "preset": "operations",
        "dry_run": True,
        "interval_multiplier": 2.0,
    })
    assert result.success
    # Operations alert polling is 300s base, with 2x multiplier = 600s
    alert_task = next(t for t in result.data["tasks"] if "Alert" in t["name"])
    assert alert_task["interval_seconds"] == 600.0


@pytest.mark.asyncio
async def test_deactivate_preset(skill_with_context):
    await skill_with_context.execute("activate", {"preset": "operations"})
    result = await skill_with_context.execute("deactivate", {"preset": "operations"})
    assert result.success
    assert result.data["cancelled"] == 3
    assert "operations" not in skill_with_context._installed_tasks


@pytest.mark.asyncio
async def test_deactivate_inactive_preset(skill):
    result = await skill.execute("deactivate", {"preset": "operations"})
    assert not result.success
    assert "not active" in result.message


@pytest.mark.asyncio
async def test_full_autonomy_bundle(skill_with_context):
    result = await skill_with_context.execute("activate", {
        "preset": "full_autonomy",
        "dry_run": True,
    })
    assert result.success
    # Should include tasks from all 4 sub-presets
    total = sum(len(PRESETS[p]["tasks"]) for p in ["self_improvement", "operations", "revenue", "replication"])
    assert len(result.data["tasks"]) == total


@pytest.mark.asyncio
async def test_create_custom_preset(skill):
    result = await skill.execute("create_preset", {
        "name": "my_custom",
        "display_name": "My Custom Preset",
        "description": "Custom automation bundle",
        "tasks": [
            {"name": "Check", "skill_id": "health_monitor", "action": "check", "interval_seconds": 120},
            {"skill_id": "observability", "action": "query"},
        ],
    })
    assert result.success
    assert result.data["preset"]["name"] == "My Custom Preset"
    assert len(result.data["preset"]["tasks"]) == 2


@pytest.mark.asyncio
async def test_create_preset_cannot_override_builtin(skill):
    result = await skill.execute("create_preset", {
        "name": "operations",
        "display_name": "Override",
        "description": "nope",
        "tasks": [{"skill_id": "x", "action": "y"}],
    })
    assert not result.success
    assert "built-in" in result.message


@pytest.mark.asyncio
async def test_status_no_active(skill):
    result = await skill.execute("status", {})
    assert result.success
    assert "No active" in result.message


@pytest.mark.asyncio
async def test_preset_definitions_valid():
    """Validate all built-in presets have required fields."""
    for pid, preset in PRESETS.items():
        assert "name" in preset, f"Preset {pid} missing name"
        assert "description" in preset, f"Preset {pid} missing description"
        if not preset.get("is_bundle"):
            for task in preset.get("tasks", []):
                assert "skill_id" in task, f"Task in {pid} missing skill_id"
                assert "action" in task, f"Task in {pid} missing action"
                assert "interval_seconds" in task, f"Task in {pid} missing interval_seconds"
                assert task["interval_seconds"] > 0


@pytest.mark.asyncio
async def test_format_interval(skill):
    assert skill._format_interval(30) == "30s"
    assert skill._format_interval(300) == "5m"
    assert skill._format_interval(3600) == "1.0h"
    assert skill._format_interval(86400) == "1.0d"
