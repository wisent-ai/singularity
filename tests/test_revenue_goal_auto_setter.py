"""Tests for RevenueGoalAutoSetterSkill."""

import json
import pytest
import asyncio
from pathlib import Path

from singularity.skills.revenue_goal_auto_setter import (
    RevenueGoalAutoSetterSkill,
    DATA_FILE,
    DASHBOARD_FILE,
    GOALS_FILE,
    DATA_DIR,
)


@pytest.fixture(autouse=True)
def clean_data(tmp_path, monkeypatch):
    monkeypatch.setattr("singularity.skills.revenue_goal_auto_setter.DATA_FILE", tmp_path / "revenue_goal_auto_setter.json")
    monkeypatch.setattr("singularity.skills.revenue_goal_auto_setter.DASHBOARD_FILE", tmp_path / "dashboard.json")
    monkeypatch.setattr("singularity.skills.revenue_goal_auto_setter.GOALS_FILE", tmp_path / "goals.json")
    monkeypatch.setattr("singularity.skills.revenue_goal_auto_setter.DATA_DIR", tmp_path)
    yield tmp_path


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def _write_dashboard(tmp_path, snapshots, config=None):
    data = {"snapshots": snapshots, "config": config or {"compute_cost_per_hour": 0.10, "revenue_target_daily": 1.00}}
    (tmp_path / "dashboard.json").write_text(json.dumps(data, default=str))


def _make_snapshots(count=5, base_revenue=0.5, growth=0.1):
    """Generate test snapshots with linear growth."""
    return [
        {
            "total_revenue": round(base_revenue + growth * i, 6),
            "total_cost": 0.2,
            "by_source": {
                "task_pricing": {"revenue": round((base_revenue + growth * i) * 0.6, 6), "transactions": 5},
                "marketplace": {"revenue": round((base_revenue + growth * i) * 0.4, 6), "transactions": 3},
                "usage_tracking": {"revenue": 0, "transactions": 0},
            },
        }
        for i in range(count)
    ]


def test_manifest():
    skill = RevenueGoalAutoSetterSkill()
    m = skill.manifest()
    assert m.skill_id == "revenue_goal_auto_setter"
    assert len(m.actions) == 8
    action_names = [a.name for a in m.actions]
    assert "assess" in action_names
    assert "create_goals" in action_names
    assert "track" in action_names
    assert "adjust" in action_names


def test_assess_no_data():
    skill = RevenueGoalAutoSetterSkill()
    result = run(skill.execute("assess", {}))
    assert result.success
    assert result.data["metrics"]["snapshot_count"] == 0


def test_assess_with_data(clean_data):
    snaps = _make_snapshots(5, base_revenue=0.5, growth=0.1)
    _write_dashboard(clean_data, snaps)
    skill = RevenueGoalAutoSetterSkill()
    result = run(skill.execute("assess", {}))
    assert result.success
    recs = result.data["recommendations"]
    assert len(recs) >= 1
    types = [r["type"] for r in recs]
    assert "breakeven" in types  # Revenue < compute cost


def test_assess_at_breakeven(clean_data):
    snaps = _make_snapshots(5, base_revenue=3.0, growth=0.1)
    _write_dashboard(clean_data, snaps)
    skill = RevenueGoalAutoSetterSkill()
    result = run(skill.execute("assess", {}))
    assert result.success
    types = [r["type"] for r in result.data["recommendations"]]
    assert "breakeven" not in types  # Revenue > compute cost
    assert "growth" in types


def test_assess_recommends_diversification(clean_data):
    snaps = [{"total_revenue": 1.0, "total_cost": 0.3,
              "by_source": {"a": {"revenue": 1.0, "transactions": 5},
                            "b": {"revenue": 0, "transactions": 0},
                            "c": {"revenue": 0, "transactions": 0},
                            "d": {"revenue": 0, "transactions": 0}}}]
    _write_dashboard(clean_data, snaps)
    skill = RevenueGoalAutoSetterSkill()
    result = run(skill.execute("assess", {}))
    assert result.success
    types = [r["type"] for r in result.data["recommendations"]]
    assert "diversification" in types


def test_create_goals(clean_data):
    snaps = _make_snapshots(5, base_revenue=0.5, growth=0.1)
    _write_dashboard(clean_data, snaps)
    skill = RevenueGoalAutoSetterSkill()
    run(skill.execute("assess", {}))
    result = run(skill.execute("create_goals", {}))
    assert result.success
    assert len(result.data["created"]) >= 1
    # Verify goals were written to goals file
    goals = json.loads((clean_data / "goals.json").read_text())
    assert len(goals["goals"]) >= 1
    assert goals["goals"][0]["pillar"] == "revenue"


def test_create_goals_auto_create(clean_data):
    snaps = _make_snapshots(5, base_revenue=0.5, growth=0.1)
    _write_dashboard(clean_data, snaps)
    skill = RevenueGoalAutoSetterSkill()
    result = run(skill.execute("assess", {"auto_create": True}))
    assert result.success
    assert len(result.data["goals_created"]) >= 1


def test_no_duplicate_goals(clean_data):
    snaps = _make_snapshots(5, base_revenue=0.5, growth=0.1)
    _write_dashboard(clean_data, snaps)
    skill = RevenueGoalAutoSetterSkill()
    run(skill.execute("assess", {"auto_create": True}))
    # Run again - should not create duplicates
    state = skill._load()
    state["last_assess_at"] = None  # Reset cooldown
    skill._save(state)
    run(skill.execute("assess", {"auto_create": True}))
    goals = json.loads((clean_data / "goals.json").read_text())
    breakeven_goals = [g for g in goals["goals"] if "break-even" in g["title"].lower()]
    assert len(breakeven_goals) <= 1


def test_track(clean_data):
    snaps = _make_snapshots(5, base_revenue=0.5, growth=0.1)
    _write_dashboard(clean_data, snaps)
    skill = RevenueGoalAutoSetterSkill()
    run(skill.execute("assess", {"auto_create": True}))
    result = run(skill.execute("track", {}))
    assert result.success
    assert "tracked_goals" in result.data


def test_adjust_no_change(clean_data):
    snaps = _make_snapshots(5, base_revenue=0.5, growth=0.1)
    _write_dashboard(clean_data, snaps)
    skill = RevenueGoalAutoSetterSkill()
    run(skill.execute("assess", {}))
    result = run(skill.execute("adjust", {}))
    assert result.success
    assert result.data["needs_action"] is False


def test_adjust_with_change(clean_data):
    snaps = _make_snapshots(5, base_revenue=0.5, growth=0.1)
    _write_dashboard(clean_data, snaps)
    skill = RevenueGoalAutoSetterSkill()
    run(skill.execute("assess", {}))
    # Change revenue significantly
    snaps2 = _make_snapshots(5, base_revenue=2.0, growth=0.3)
    _write_dashboard(clean_data, snaps2)
    result = run(skill.execute("adjust", {}))
    assert result.success
    assert result.data["needs_action"] is True
    assert len(result.data["adjustments"]) >= 1


def test_report(clean_data):
    snaps = _make_snapshots(5, base_revenue=0.5, growth=0.1)
    _write_dashboard(clean_data, snaps)
    skill = RevenueGoalAutoSetterSkill()
    run(skill.execute("assess", {"auto_create": True}))
    result = run(skill.execute("report", {}))
    assert result.success
    assert "summary" in result.data
    assert result.data["summary"]["total_goals_created"] >= 1


def test_configure(clean_data):
    skill = RevenueGoalAutoSetterSkill()
    result = run(skill.execute("configure", {"goal_stretch_factor": 1.5, "auto_create_goals": True}))
    assert result.success
    assert "goal_stretch_factor" in result.data["updated"]
    state = skill._load()
    assert state["config"]["goal_stretch_factor"] == 1.5


def test_status(clean_data):
    skill = RevenueGoalAutoSetterSkill()
    result = run(skill.execute("status", {}))
    assert result.success
    assert "config" in result.data
    assert "stats" in result.data


def test_history(clean_data):
    snaps = _make_snapshots(5, base_revenue=0.5, growth=0.1)
    _write_dashboard(clean_data, snaps)
    skill = RevenueGoalAutoSetterSkill()
    run(skill.execute("assess", {"auto_create": True}))
    result = run(skill.execute("history", {}))
    assert result.success
    assert result.data["total"] >= 1


def test_max_active_goals_limit(clean_data):
    snaps = _make_snapshots(5, base_revenue=0.5, growth=0.1)
    _write_dashboard(clean_data, snaps)
    skill = RevenueGoalAutoSetterSkill()
    run(skill.execute("configure", {"max_active_revenue_goals": 1}))
    run(skill.execute("assess", {"auto_create": True}))
    # Only 1 goal should be created
    goals = json.loads((clean_data / "goals.json").read_text())
    assert len(goals["goals"]) == 1
