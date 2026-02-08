"""Tests for Revenue Observability Sync preset in SchedulerPresetsSkill."""

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


def test_revenue_observability_sync_preset_exists():
    """New revenue_observability_sync preset should be in BUILTIN_PRESETS."""
    assert "revenue_observability_sync" in BUILTIN_PRESETS
    preset = BUILTIN_PRESETS["revenue_observability_sync"]
    assert preset.pillar == "revenue"
    assert preset.name == "Revenue Observability Sync"
    assert len(preset.schedules) == 3
    assert "revenue_reporting" in preset.depends_on


def test_revenue_observability_sync_schedules():
    """Preset should have metrics sync, dashboard snapshot, and alert check."""
    preset = BUILTIN_PRESETS["revenue_observability_sync"]
    names = [s.name for s in preset.schedules]
    assert "Revenue Metrics Sync" in names
    assert "Revenue Dashboard Snapshot" in names
    assert "Revenue Alert Check" in names


def test_revenue_metrics_sync_config():
    """Revenue Metrics Sync should call revenue_observability_bridge.sync every 15 min."""
    preset = BUILTIN_PRESETS["revenue_observability_sync"]
    sync_schedule = [s for s in preset.schedules if s.name == "Revenue Metrics Sync"][0]
    assert sync_schedule.skill_id == "revenue_observability_bridge"
    assert sync_schedule.action == "sync"
    assert sync_schedule.interval_seconds == 900
    assert sync_schedule.params.get("force") is True


def test_dashboard_snapshot_config():
    """Dashboard Snapshot should call revenue_analytics_dashboard.snapshot every hour."""
    preset = BUILTIN_PRESETS["revenue_observability_sync"]
    snap = [s for s in preset.schedules if s.name == "Revenue Dashboard Snapshot"][0]
    assert snap.skill_id == "revenue_analytics_dashboard"
    assert snap.action == "snapshot"
    assert snap.interval_seconds == 3600


def test_revenue_reporting_enhanced():
    """revenue_reporting preset should now include observability sync."""
    preset = BUILTIN_PRESETS["revenue_reporting"]
    names = [s.name for s in preset.schedules]
    assert "Revenue Observability Sync" in names
    assert "Revenue Dashboard Overview" in names
    assert len(preset.schedules) >= 3


def test_revenue_reporting_sync_interval():
    """Revenue Observability Sync in revenue_reporting runs every 30 min."""
    preset = BUILTIN_PRESETS["revenue_reporting"]
    sync = [s for s in preset.schedules if s.name == "Revenue Observability Sync"][0]
    assert sync.skill_id == "revenue_observability_bridge"
    assert sync.action == "sync"
    assert sync.interval_seconds == 1800


def test_list_presets_includes_new():
    """list_presets should include the new revenue_observability_sync preset."""
    skill = SchedulerPresetsSkill()
    result = run(skill.execute("list_presets", {}))
    assert result.success
    ids = [p["preset_id"] for p in result.data["presets"]]
    assert "revenue_observability_sync" in ids


def test_list_presets_filter_revenue():
    """Filtering by revenue pillar should include new preset."""
    skill = SchedulerPresetsSkill()
    result = run(skill.execute("list_presets", {"pillar": "revenue"}))
    assert result.success
    ids = [p["preset_id"] for p in result.data["presets"]]
    assert "revenue_observability_sync" in ids
    assert "revenue_reporting" in ids


def test_full_autonomy_includes_new():
    """Full autonomy preset list should include revenue_observability_sync."""
    from singularity.skills.scheduler_presets import FULL_AUTONOMY_PRESETS
    assert "revenue_observability_sync" in FULL_AUTONOMY_PRESETS
