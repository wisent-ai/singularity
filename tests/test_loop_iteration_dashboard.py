"""Tests for LoopIterationDashboardSkill."""
import json
import pytest
from pathlib import Path
from unittest.mock import patch
from singularity.skills.loop_iteration_dashboard import (
    LoopIterationDashboardSkill, DASHBOARD_STATE_FILE, LOOP_STATE_FILE,
    CIRCUIT_FILE, GOALS_FILE, SCHEDULER_FILE, FLEET_FILE, REPUTATION_FILE,
    _now_iso,
)


def _make_journal_entry(iteration_id="iter-1", outcome="success", duration=10.0, pillar="revenue", revenue=0.5):
    return {
        "iteration_id": iteration_id,
        "started_at": _now_iso(),
        "completed_at": _now_iso(),
        "outcome": outcome,
        "duration_seconds": duration,
        "pillar": pillar,
        "task_description": f"Task for {iteration_id}",
        "phases": {
            "assess": {"weakest_pillar": "revenue", "overall_score": 60},
            "decide": {"task": "do something", "pillar": pillar, "priority": "high"},
            "plan": {"steps": ["step1", "step2"]},
            "act": {"success": outcome == "success", "actions_taken": 2, "actions_succeeded": 2 if outcome == "success" else 0},
            "measure": {"revenue": revenue, "cost": 0.1},
            "learn": {"feedback_recorded": True},
        },
    }


def _make_loop_data(entries=None, stats=None):
    return {
        "journal": entries or [],
        "stats": stats or {"total_iterations": len(entries or [])},
    }


@pytest.fixture
def skill(tmp_path):
    """Create skill with temp data files."""
    patches = {
        "singularity.skills.loop_iteration_dashboard.DASHBOARD_STATE_FILE": tmp_path / "lid.json",
        "singularity.skills.loop_iteration_dashboard.LOOP_STATE_FILE": tmp_path / "loop.json",
        "singularity.skills.loop_iteration_dashboard.CIRCUIT_FILE": tmp_path / "cb.json",
        "singularity.skills.loop_iteration_dashboard.GOALS_FILE": tmp_path / "goals.json",
        "singularity.skills.loop_iteration_dashboard.SCHEDULER_FILE": tmp_path / "sched.json",
        "singularity.skills.loop_iteration_dashboard.FLEET_FILE": tmp_path / "fleet.json",
        "singularity.skills.loop_iteration_dashboard.REPUTATION_FILE": tmp_path / "rep.json",
    }
    with patch.multiple("singularity.skills.loop_iteration_dashboard", **{k.split(".")[-1]: v for k, v in patches.items()}):
        s = LoopIterationDashboardSkill()
        # Write empty data files
        for p in [tmp_path / "loop.json", tmp_path / "cb.json", tmp_path / "goals.json",
                   tmp_path / "sched.json", tmp_path / "fleet.json", tmp_path / "rep.json"]:
            p.write_text(json.dumps({}))
        yield s, tmp_path


@pytest.mark.asyncio
async def test_latest_no_data(skill):
    s, tmp = skill
    result = await s.execute("latest", {})
    assert result.success
    assert result.data["iteration"] is None


@pytest.mark.asyncio
async def test_latest_with_data(skill):
    s, tmp = skill
    entries = [_make_journal_entry("iter-1"), _make_journal_entry("iter-2", outcome="partial")]
    (tmp / "loop.json").write_text(json.dumps(_make_loop_data(entries)))
    result = await s.execute("latest", {})
    assert result.success
    assert result.data["iteration"]["iteration_id"] == "iter-2"
    assert "overall_health_score" in result.data
    assert "subsystem_health" in result.data


@pytest.mark.asyncio
async def test_history(skill):
    s, tmp = skill
    entries = [_make_journal_entry(f"iter-{i}") for i in range(5)]
    (tmp / "loop.json").write_text(json.dumps(_make_loop_data(entries)))
    result = await s.execute("history", {"limit": 3})
    assert result.success
    assert len(result.data["iterations"]) == 3
    assert result.data["total_iterations"] == 5


@pytest.mark.asyncio
async def test_trends_insufficient_data(skill):
    s, tmp = skill
    (tmp / "loop.json").write_text(json.dumps(_make_loop_data([_make_journal_entry()])))
    result = await s.execute("trends", {})
    assert result.success
    assert result.data["trends"] == {}


@pytest.mark.asyncio
async def test_trends_with_data(skill):
    s, tmp = skill
    entries = [_make_journal_entry(f"iter-{i}", revenue=0.1 * i) for i in range(10)]
    (tmp / "loop.json").write_text(json.dumps(_make_loop_data(entries)))
    result = await s.execute("trends", {"window": 10})
    assert result.success
    trends = result.data["trends"]
    assert "success_rate" in trends
    assert "avg_duration_seconds" in trends
    assert "revenue" in trends
    assert trends["revenue"]["direction"] == "improving"


@pytest.mark.asyncio
async def test_compare(skill):
    s, tmp = skill
    entries = [_make_journal_entry("A", duration=5), _make_journal_entry("B", duration=15)]
    (tmp / "loop.json").write_text(json.dumps(_make_loop_data(entries)))
    result = await s.execute("compare", {"iteration_a": "A", "iteration_b": "B"})
    assert result.success
    assert result.data["deltas"]["duration_seconds"] == 10.0


@pytest.mark.asyncio
async def test_compare_missing(skill):
    s, tmp = skill
    (tmp / "loop.json").write_text(json.dumps(_make_loop_data([])))
    result = await s.execute("compare", {"iteration_a": "X", "iteration_b": "Y"})
    assert not result.success


@pytest.mark.asyncio
async def test_subsystem_health(skill):
    s, tmp = skill
    entries = [_make_journal_entry(f"i-{i}") for i in range(5)]
    (tmp / "loop.json").write_text(json.dumps(_make_loop_data(entries)))
    result = await s.execute("subsystem_health", {})
    assert result.success
    health = result.data["subsystem_health"]
    assert "loop_execution" in health
    assert "circuit_breaker" in health
    assert result.data["overall_health_score"] >= 0


@pytest.mark.asyncio
async def test_alerts_failure_streak(skill):
    s, tmp = skill
    entries = [_make_journal_entry(f"i-{i}", outcome="failed") for i in range(5)]
    (tmp / "loop.json").write_text(json.dumps(_make_loop_data(entries)))
    result = await s.execute("alerts", {})
    assert result.success
    types = [a["type"] for a in result.data["current_alerts"]]
    assert "failure_streak" in types
    assert "low_success_rate" in types


@pytest.mark.asyncio
async def test_alerts_no_issues(skill):
    s, tmp = skill
    entries = [_make_journal_entry(f"i-{i}") for i in range(5)]
    (tmp / "loop.json").write_text(json.dumps(_make_loop_data(entries)))
    result = await s.execute("alerts", {})
    assert result.success
    assert result.data["critical_count"] == 0


@pytest.mark.asyncio
async def test_configure(skill):
    s, tmp = skill
    result = await s.execute("configure", {"config": {"success_rate_alert_threshold": 0.8}})
    assert result.success
    assert any("success_rate_alert_threshold" in u for u in result.data["updated"])


@pytest.mark.asyncio
async def test_unknown_action(skill):
    s, tmp = skill
    result = await s.execute("nonexistent", {})
    assert not result.success


@pytest.mark.asyncio
async def test_manifest(skill):
    s, tmp = skill
    m = s.manifest
    assert m.skill_id == "loop_iteration_dashboard"
    assert len(m.actions) == 7
