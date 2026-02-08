"""Tests for SchedulerPresetsSkill."""

import pytest
import json
from singularity.skills.scheduler_presets import (
    SchedulerPresetsSkill, PRESETS, DATA_FILE,
)
import singularity.skills.scheduler_presets as mod


@pytest.fixture
def skill(tmp_path):
    """Create skill with temp data path."""
    s = SchedulerPresetsSkill()
    mod.DATA_FILE = tmp_path / "scheduler_presets.json"
    return s


@pytest.mark.asyncio
async def test_manifest(skill):
    m = skill.manifest
    assert m.skill_id == "scheduler_presets"
    assert len(m.actions) == 5
    names = [a.name for a in m.actions]
    assert "apply" in names
    assert "remove" in names
    assert "list" in names
    assert "status" in names
    assert "customize" in names


@pytest.mark.asyncio
async def test_unknown_action(skill):
    r = await skill.execute("nonexistent", {})
    assert not r.success
    assert "Unknown action" in r.message


@pytest.mark.asyncio
async def test_list_presets(skill):
    r = await skill.execute("list", {})
    assert r.success
    assert len(r.data["presets"]) == len(PRESETS)
    names = [p["name"] for p in r.data["presets"]]
    assert "health_check" in names
    assert "full_autonomy" in names


@pytest.mark.asyncio
async def test_list_shows_active_status(skill):
    # Apply a preset
    await skill.execute("apply", {"preset": "health_check"})
    r = await skill.execute("list", {})
    hc = [p for p in r.data["presets"] if p["name"] == "health_check"][0]
    assert hc["active"] is True


@pytest.mark.asyncio
async def test_apply_preset(skill):
    r = await skill.execute("apply", {"preset": "health_check"})
    assert r.success
    assert r.data["preset"] == "health_check"
    assert len(r.data["scheduled"]) > 0
    assert r.data["dry_run"] is False


@pytest.mark.asyncio
async def test_apply_unknown_preset(skill):
    r = await skill.execute("apply", {"preset": "nonexistent"})
    assert not r.success
    assert "Unknown preset" in r.message


@pytest.mark.asyncio
async def test_apply_dry_run(skill):
    r = await skill.execute("apply", {"preset": "reputation_sync", "dry_run": True})
    assert r.success
    assert r.data["dry_run"] is True
    assert len(r.data["scheduled"]) > 0

    # Should not be active after dry run
    r2 = await skill.execute("list", {})
    rs = [p for p in r2.data["presets"] if p["name"] == "reputation_sync"][0]
    assert rs["active"] is False


@pytest.mark.asyncio
async def test_apply_already_active(skill):
    await skill.execute("apply", {"preset": "health_check"})
    r = await skill.execute("apply", {"preset": "health_check"})
    assert r.success
    assert r.data.get("already_active") is True


@pytest.mark.asyncio
async def test_remove_preset(skill):
    await skill.execute("apply", {"preset": "health_check"})
    r = await skill.execute("remove", {"preset": "health_check"})
    assert r.success

    # Should no longer be active
    r2 = await skill.execute("list", {})
    hc = [p for p in r2.data["presets"] if p["name"] == "health_check"][0]
    assert hc["active"] is False


@pytest.mark.asyncio
async def test_remove_inactive_preset(skill):
    r = await skill.execute("remove", {"preset": "health_check"})
    assert r.success
    assert r.data.get("was_active") is False


@pytest.mark.asyncio
async def test_remove_unknown_preset(skill):
    r = await skill.execute("remove", {"preset": "nonexistent"})
    assert not r.success


@pytest.mark.asyncio
async def test_status_empty(skill):
    r = await skill.execute("status", {})
    assert r.success
    assert "No presets" in r.message


@pytest.mark.asyncio
async def test_status_with_active(skill):
    await skill.execute("apply", {"preset": "self_improvement"})
    r = await skill.execute("status", {})
    assert r.success
    assert "self_improvement" in r.data["active_presets"]


@pytest.mark.asyncio
async def test_customize_interval(skill):
    r = await skill.execute("customize", {
        "preset": "health_check",
        "task_name": "poll_alerts",
        "interval_seconds": 120,
    })
    assert r.success
    assert r.data["interval_seconds"] == 120


@pytest.mark.asyncio
async def test_customize_unknown_task(skill):
    r = await skill.execute("customize", {
        "preset": "health_check",
        "task_name": "nonexistent_task",
        "interval_seconds": 120,
    })
    assert not r.success
    assert "not found" in r.message.lower()


@pytest.mark.asyncio
async def test_customize_too_small_interval(skill):
    r = await skill.execute("customize", {
        "preset": "health_check",
        "task_name": "poll_alerts",
        "interval_seconds": 5,
    })
    assert not r.success
    assert "at least 10" in r.message


@pytest.mark.asyncio
async def test_full_autonomy_resolves_tasks(skill):
    r = await skill.execute("apply", {"preset": "full_autonomy", "dry_run": True})
    assert r.success
    # Full autonomy includes all sub-presets
    assert len(r.data["scheduled"]) >= 4  # At least tasks from all included presets


@pytest.mark.asyncio
async def test_preset_definitions_valid():
    """Verify all preset definitions have required fields."""
    for name, preset in PRESETS.items():
        assert "name" in preset
        assert "description" in preset
        assert "category" in preset
        if not preset.get("includes"):
            assert len(preset["tasks"]) > 0, f"Preset {name} has no tasks"
        for task in preset.get("tasks", []):
            assert "name" in task
            assert "skill_id" in task
            assert "action" in task
            assert "interval_seconds" in task
            assert task["interval_seconds"] >= 10
