"""Tests for AlertSchedulerBridgeSkill."""

import asyncio
import json
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from singularity.skills.alert_scheduler_bridge import (
    AlertSchedulerBridgeSkill,
    STATE_FILE,
)


@pytest.fixture
def skill(tmp_path):
    """Create skill with temp state file."""
    state_file = tmp_path / "alert_scheduler_bridge.json"
    with patch.object(
        AlertSchedulerBridgeSkill, "_load_state",
        return_value=AlertSchedulerBridgeSkill(None)._default_state(),
    ):
        s = AlertSchedulerBridgeSkill()
    s._state = s._default_state()
    # Redirect state file to tmp
    import singularity.skills.alert_scheduler_bridge as mod
    mod.STATE_FILE = state_file
    mod.DATA_DIR = tmp_path
    return s


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


class TestSetup:
    def test_setup_default(self, skill):
        result = run(skill.execute("setup", {}))
        assert result.success
        assert skill._state["enabled"]
        assert "300s" in result.message
        assert "3600s" in result.message

    def test_setup_custom_intervals(self, skill):
        result = run(skill.execute("setup", {
            "alert_interval": 600,
            "forecast_interval": 7200,
            "forecast_periods": 14,
        }))
        assert result.success
        cfg = skill._state["config"]
        assert cfg["alert_check_interval_seconds"] == 600
        assert cfg["forecast_interval_seconds"] == 7200
        assert cfg["forecast_periods"] == 14

    def test_setup_minimum_intervals(self, skill):
        run(skill.execute("setup", {"alert_interval": 5, "forecast_interval": 10}))
        cfg = skill._state["config"]
        assert cfg["alert_check_interval_seconds"] == 30  # min 30
        assert cfg["forecast_interval_seconds"] == 60     # min 60

    def test_setup_returns_schedule_specs(self, skill):
        result = run(skill.execute("setup", {}))
        specs = result.data["schedules"]
        assert len(specs) == 2
        assert specs[0]["skill_id"] == "revenue_alert_escalation"
        assert specs[1]["skill_id"] == "revenue_forecast"


class TestStatus:
    def test_status_inactive(self, skill):
        result = run(skill.execute("status", {}))
        assert result.success
        assert "PAUSED" in result.message

    def test_status_after_setup(self, skill):
        run(skill.execute("setup", {}))
        result = run(skill.execute("status", {}))
        assert "ACTIVE" in result.message
        assert result.data["enabled"]


class TestRunNow:
    def test_run_now_without_context(self, skill):
        run(skill.execute("setup", {}))
        result = run(skill.execute("run_now", {}))
        assert result.success
        assert skill._state["stats"]["total_checks"] == 1

    def test_run_now_records_history(self, skill):
        run(skill.execute("setup", {}))
        run(skill.execute("run_now", {}))
        assert len(skill._state["history"]) == 1
        assert skill._state["history"][0]["trigger"] == "manual"

    def test_run_now_with_metrics(self, skill):
        run(skill.execute("setup", {}))
        result = run(skill.execute("run_now", {"metrics": {"revenue.total": 100}}))
        assert result.success


class TestConfigure:
    def test_configure_updates(self, skill):
        result = run(skill.execute("configure", {"alert_interval": 900}))
        assert result.success
        assert skill._state["config"]["alert_check_interval_seconds"] == 900

    def test_configure_empty(self, skill):
        result = run(skill.execute("configure", {}))
        assert not result.success

    def test_configure_multiple(self, skill):
        result = run(skill.execute("configure", {
            "forecast_periods": 30,
            "reactive_forecast": False,
        }))
        assert result.success
        assert skill._state["config"]["forecast_periods"] == 30
        assert skill._state["config"]["reactive_forecast_on_alert"] is False


class TestPauseResume:
    def test_pause(self, skill):
        run(skill.execute("setup", {}))
        result = run(skill.execute("pause", {}))
        assert result.success
        assert not skill._state["enabled"]

    def test_pause_already_paused(self, skill):
        result = run(skill.execute("pause", {}))
        assert not result.success

    def test_resume(self, skill):
        run(skill.execute("setup", {}))
        run(skill.execute("pause", {}))
        result = run(skill.execute("resume", {}))
        assert result.success
        assert skill._state["enabled"]

    def test_resume_already_active(self, skill):
        run(skill.execute("setup", {}))
        result = run(skill.execute("resume", {}))
        assert not result.success


class TestHistory:
    def test_history_empty(self, skill):
        result = run(skill.execute("history", {}))
        assert result.success
        assert result.data["total"] == 0

    def test_history_after_runs(self, skill):
        run(skill.execute("setup", {}))
        run(skill.execute("run_now", {}))
        run(skill.execute("run_now", {}))
        result = run(skill.execute("history", {"limit": 10}))
        assert result.data["total"] == 2


class TestDashboard:
    def test_dashboard(self, skill):
        run(skill.execute("setup", {}))
        run(skill.execute("run_now", {}))
        result = run(skill.execute("dashboard", {}))
        assert result.success
        assert "ACTIVE" in result.message
        assert result.data["monitoring"]["total_checks"] == 1


class TestManifest:
    def test_manifest(self, skill):
        m = skill.manifest()
        assert m.name == "alert_scheduler_bridge"
        actions = [a.name for a in m.actions]
        assert "setup" in actions
        assert "run_now" in actions
        assert "dashboard" in actions
        assert len(actions) == 8


class TestUnknownAction:
    def test_unknown(self, skill):
        result = run(skill.execute("nonexistent", {}))
        assert not result.success
