"""Tests for SchedulerPresetsSkill."""

import json
import asyncio
import pytest
from pathlib import Path
from singularity.skills.scheduler_presets import (
    SchedulerPresetsSkill, PRESETS_FILE, DATA_DIR, BUILTIN_PRESETS,
)


@pytest.fixture(autouse=True)
def clean_data():
    if PRESETS_FILE.exists():
        PRESETS_FILE.unlink()
    sched = DATA_DIR / "scheduler.json"
    if sched.exists():
        sched.unlink()
    yield
    if PRESETS_FILE.exists():
        PRESETS_FILE.unlink()
    if sched.exists():
        sched.unlink()


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def make_skill():
    return SchedulerPresetsSkill()


def test_manifest():
    skill = make_skill()
    m = skill.manifest
    assert m.skill_id == "scheduler_presets"
    actions = [a.name for a in m.actions]
    assert "list_presets" in actions
    assert "apply" in actions
    assert "apply_all" in actions
    assert "remove" in actions
    assert "status" in actions
    assert "create_custom" in actions
    assert "recommend" in actions


def test_list_presets():
    skill = make_skill()
    result = run(skill.execute("list_presets", {}))
    assert result.success
    presets = result.data["presets"]
    assert len(presets) >= 8  # all builtins
    ids = [p["preset_id"] for p in presets]
    assert "health_monitoring" in ids
    assert "self_tuning" in ids
    assert "alert_polling" in ids


def test_list_presets_filter_pillar():
    skill = make_skill()
    result = run(skill.execute("list_presets", {"pillar": "self_improvement"}))
    assert result.success
    for p in result.data["presets"]:
        assert p["pillar"] == "self_improvement"


def test_apply_preset_direct():
    """Apply a preset - should create scheduler entries via direct file."""
    skill = make_skill()
    result = run(skill.execute("apply", {"preset_id": "self_tuning"}))
    assert result.success
    assert "task_ids" in result.data
    assert len(result.data["task_ids"]) == 1
    # Verify scheduler.json was created
    sched_file = DATA_DIR / "scheduler.json"
    assert sched_file.exists()
    data = json.loads(sched_file.read_text())
    assert len(data["tasks"]) == 1


def test_apply_unknown_preset():
    skill = make_skill()
    result = run(skill.execute("apply", {"preset_id": "nonexistent"}))
    assert not result.success
    assert "Unknown preset" in result.message


def test_apply_duplicate():
    skill = make_skill()
    run(skill.execute("apply", {"preset_id": "self_tuning"}))
    result = run(skill.execute("apply", {"preset_id": "self_tuning"}))
    assert not result.success
    assert "already applied" in result.message


def test_apply_with_multiplier():
    skill = make_skill()
    result = run(skill.execute("apply", {"preset_id": "self_tuning", "interval_multiplier": 2.0}))
    assert result.success
    sched = json.loads((DATA_DIR / "scheduler.json").read_text())
    task = list(sched["tasks"].values())[0]
    assert task["interval_seconds"] == 1800  # 900 * 2


def test_remove_preset():
    skill = make_skill()
    run(skill.execute("apply", {"preset_id": "self_tuning"}))
    result = run(skill.execute("remove", {"preset_id": "self_tuning"}))
    assert result.success
    assert result.data["cancelled"] >= 0
    # Status should show nothing
    status = run(skill.execute("status", {}))
    assert status.data["total_presets"] == 0


def test_status():
    skill = make_skill()
    run(skill.execute("apply", {"preset_id": "self_tuning"}))
    run(skill.execute("apply", {"preset_id": "feedback_loop"}))
    result = run(skill.execute("status", {}))
    assert result.success
    assert result.data["total_presets"] == 2
    assert result.data["total_tasks"] == 2  # 1 + 1


def test_apply_all():
    skill = make_skill()
    result = run(skill.execute("apply_all", {}))
    assert result.success
    assert result.data["applied"] == len(BUILTIN_PRESETS)
    status = run(skill.execute("status", {}))
    assert status.data["total_presets"] == len(BUILTIN_PRESETS)


def test_remove_all():
    skill = make_skill()
    run(skill.execute("apply_all", {}))
    result = run(skill.execute("remove_all", {}))
    assert result.success
    assert result.data["removed"] == len(BUILTIN_PRESETS)


def test_create_custom_preset():
    skill = make_skill()
    result = run(skill.execute("create_custom", {
        "preset_id": "my_preset",
        "name": "My Custom Preset",
        "schedules": [
            {"skill_id": "diagnostics", "action": "scan", "interval_seconds": 600, "name": "Custom Scan"},
        ],
    }))
    assert result.success
    # Should appear in list
    listed = run(skill.execute("list_presets", {}))
    ids = [p["preset_id"] for p in listed.data["presets"]]
    assert "my_preset" in ids
    # Should be applyable
    apply_result = run(skill.execute("apply", {"preset_id": "my_preset"}))
    assert apply_result.success


def test_create_custom_validates():
    skill = make_skill()
    # Missing schedules
    r = run(skill.execute("create_custom", {"preset_id": "x", "name": "X", "schedules": []}))
    assert not r.success
    # Too short interval
    r = run(skill.execute("create_custom", {"preset_id": "x", "name": "X",
            "schedules": [{"skill_id": "a", "action": "b", "interval_seconds": 1}]}))
    assert not r.success


def test_recommend():
    skill = make_skill()
    result = run(skill.execute("recommend", {}))
    assert result.success
    recs = result.data["recommendations"]
    assert len(recs) >= 8
    # Should be sorted by priority
    assert recs[0]["preset_id"] == "health_monitoring"


def test_persistence():
    skill1 = make_skill()
    run(skill1.execute("apply", {"preset_id": "self_tuning"}))
    # New instance should load state
    skill2 = make_skill()
    status = run(skill2.execute("status", {}))
    assert status.data["total_presets"] == 1


def test_humanize_interval():
    assert SchedulerPresetsSkill._humanize_interval(30) == "30s"
    assert SchedulerPresetsSkill._humanize_interval(300) == "5m"
    assert SchedulerPresetsSkill._humanize_interval(3600) == "1h"
    assert SchedulerPresetsSkill._humanize_interval(7200) == "2h"
    assert SchedulerPresetsSkill._humanize_interval(86400) == "1.0d"
