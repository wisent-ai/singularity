#!/usr/bin/env python3
"""Tests for FleetHealthEventBridgeSkill."""

import asyncio
import json
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from singularity.skills.fleet_health_events import (
    FleetHealthEventBridgeSkill,
    BRIDGE_STATE_FILE,
)
from singularity.skills.base import SkillResult


@pytest.fixture
def skill(tmp_path):
    """Create skill with patched state file."""
    test_file = tmp_path / "fleet_health_events.json"
    with patch.object(FleetHealthEventBridgeSkill, "_load_state"):
        s = FleetHealthEventBridgeSkill()
    s._last_snapshot = {}
    s._event_history = []
    s._config = s._default_config()
    s._stats = s._default_stats()

    def patched_save():
        test_file.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "last_snapshot": s._last_snapshot,
            "event_history": s._event_history,
            "config": s._config,
            "stats": s._stats,
        }
        with open(test_file, "w") as f:
            json.dump(data, f, indent=2, default=str)

    s._save_state = patched_save
    return s


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


class TestManifest:
    def test_skill_id(self, skill):
        assert skill.manifest.skill_id == "fleet_health_events"

    def test_version(self, skill):
        assert skill.manifest.version == "1.0.0"

    def test_category(self, skill):
        assert skill.manifest.category == "replication"

    def test_actions(self, skill):
        names = [a.name for a in skill.manifest.actions]
        for action in ["monitor", "configure", "status", "history", "emit_test", "fleet_check"]:
            assert action in names

    def test_unknown_action(self, skill):
        result = run(skill.execute("nonexistent", {}))
        assert not result.success
        assert "Unknown action" in result.message


class TestDefaultConfig:
    def test_emit_flags(self, skill):
        assert skill._config["emit_on_heal"] is True
        assert skill._config["emit_on_scale"] is True
        assert skill._config["emit_on_fleet_alert"] is True

    def test_threshold(self, skill):
        assert skill._config["unhealthy_threshold"] == 0.5

    def test_priorities(self, skill):
        assert skill._config["priority_heal"] == "high"
        assert skill._config["priority_fleet_alert"] == "critical"


class TestConfigure:
    def test_update_emit_flags(self, skill):
        result = run(skill.execute("configure", {"emit_on_heal": False}))
        assert result.success
        assert skill._config["emit_on_heal"] is False

    def test_update_threshold(self, skill):
        result = run(skill.execute("configure", {"unhealthy_threshold": 0.3}))
        assert result.success
        assert skill._config["unhealthy_threshold"] == 0.3

    def test_clamp_threshold(self, skill):
        run(skill.execute("configure", {"unhealthy_threshold": 5.0}))
        assert skill._config["unhealthy_threshold"] == 1.0
        run(skill.execute("configure", {"unhealthy_threshold": -1.0}))
        assert skill._config["unhealthy_threshold"] == 0.0

    def test_update_priority(self, skill):
        result = run(skill.execute("configure", {"priority_heal": "critical"}))
        assert result.success
        assert skill._config["priority_heal"] == "critical"

    def test_no_params(self, skill):
        result = run(skill.execute("configure", {}))
        assert not result.success


class TestMonitor:
    def test_no_fleet_manager(self, skill):
        """When fleet health manager is not available, return gracefully."""
        result = run(skill.execute("monitor", {}))
        assert result.success
        assert result.data["events_emitted"] == 0
        assert skill._stats["monitors_run"] == 1

    def test_detect_heal_incidents(self, skill):
        """Detect new heal incidents and emit events."""
        mock_context = MagicMock()
        incidents_result = SkillResult(
            success=True, message="ok",
            data={"incidents": [
                {"action": "heal_restart", "agent_id": "agent-1",
                 "timestamp": "2026-01-01T00:01:00Z", "success": True,
                 "reason": "unhealthy", "attempt": 1, "details": {}},
            ]},
        )
        status_result = SkillResult(
            success=True, message="ok",
            data={"agents": {"agent-1": {"status": "healthy"}}},
        )

        async def mock_call(skill_id, action, params):
            if action == "incidents":
                return incidents_result
            return status_result

        mock_context.call_skill = AsyncMock(side_effect=mock_call)
        skill.context = mock_context

        result = run(skill.execute("monitor", {}))
        assert result.success
        assert result.data["events_emitted"] >= 1
        assert skill._stats["heals_detected"] == 1

    def test_detect_scale_incidents(self, skill):
        mock_context = MagicMock()

        async def mock_call(skill_id, action, params):
            if action == "incidents":
                return SkillResult(success=True, message="ok", data={"incidents": [
                    {"action": "scale_up", "agent_id": "agent-2",
                     "timestamp": "2026-01-01T00:02:00Z", "reason": "high load",
                     "fleet_size_before": 2, "fleet_size_after": 3, "details": {}},
                ]})
            return SkillResult(success=True, message="ok", data={"agents": {}})

        mock_context.call_skill = AsyncMock(side_effect=mock_call)
        skill.context = mock_context

        result = run(skill.execute("monitor", {}))
        assert result.success
        assert skill._stats["scales_detected"] == 1

    def test_watermark_prevents_duplicates(self, skill):
        """Second monitor call doesn't re-emit same incidents."""
        mock_context = MagicMock()
        incidents = [
            {"action": "heal_restart", "agent_id": "a1",
             "timestamp": "2026-01-01T00:01:00Z", "success": True,
             "reason": "unhealthy", "attempt": 1, "details": {}},
        ]

        async def mock_call(skill_id, action, params):
            if action == "incidents":
                return SkillResult(success=True, message="ok", data={"incidents": incidents})
            return SkillResult(success=True, message="ok", data={"agents": {}})

        mock_context.call_skill = AsyncMock(side_effect=mock_call)
        skill.context = mock_context

        run(skill.execute("monitor", {}))
        first_heals = skill._stats["heals_detected"]

        result = run(skill.execute("monitor", {}))
        assert skill._stats["heals_detected"] == first_heals  # No new heals


class TestFleetCheck:
    def test_no_fleet_manager(self, skill):
        result = run(skill.execute("fleet_check", {}))
        assert result.success
        assert result.data["fleet_alert_emitted"] is False

    def test_no_agents(self, skill):
        mock_context = MagicMock()
        mock_context.call_skill = AsyncMock(return_value=SkillResult(
            success=True, message="ok", data={"agents": {}},
        ))
        skill.context = mock_context

        result = run(skill.execute("fleet_check", {}))
        assert result.success
        assert result.data["total_agents"] == 0

    def test_healthy_fleet_no_alert(self, skill):
        mock_context = MagicMock()
        mock_context.call_skill = AsyncMock(return_value=SkillResult(
            success=True, message="ok",
            data={"agents": {
                "a1": {"status": "healthy"},
                "a2": {"status": "healthy"},
            }},
        ))
        skill.context = mock_context

        result = run(skill.execute("fleet_check", {}))
        assert result.success
        assert result.data["fleet_alert_emitted"] is False
        assert result.data["healthy"] == 2

    def test_unhealthy_fleet_emits_alert(self, skill):
        mock_context = MagicMock()
        mock_context.call_skill = AsyncMock(return_value=SkillResult(
            success=True, message="ok",
            data={"agents": {
                "a1": {"status": "dead"},
                "a2": {"status": "unhealthy"},
                "a3": {"status": "healthy"},
            }},
        ))
        skill.context = mock_context

        result = run(skill.execute("fleet_check", {}))
        assert result.success
        assert result.data["unhealthy"] == 1
        assert result.data["dead"] == 1
        assert skill._stats["fleet_checks_run"] == 1


class TestStatus:
    def test_returns_stats(self, skill):
        result = run(skill.execute("status", {}))
        assert result.success
        assert "stats" in result.data
        assert "config" in result.data

    def test_includes_fleet_summary(self, skill):
        skill._last_snapshot["fleet_summary"] = {"healthy": 3, "total": 4}
        result = run(skill.execute("status", {}))
        assert result.success
        assert "3 healthy" in result.message


class TestHistory:
    def test_empty_history(self, skill):
        result = run(skill.execute("history", {}))
        assert result.success
        assert result.data["total"] == 0

    def test_with_events(self, skill):
        skill._event_history = [
            {"topic": "fleet_health.heal_completed", "timestamp": "2026-01-01T00:01:00Z", "emitted": True},
            {"topic": "fleet_health.scale_up", "timestamp": "2026-01-01T00:02:00Z", "emitted": False},
        ]
        result = run(skill.execute("history", {}))
        assert result.success
        assert result.data["total"] == 2

    def test_topic_filter(self, skill):
        skill._event_history = [
            {"topic": "fleet_health.heal_completed", "timestamp": "t1", "emitted": True},
            {"topic": "fleet_health.scale_up", "timestamp": "t2", "emitted": True},
        ]
        result = run(skill.execute("history", {"topic_filter": "fleet_health.heal"}))
        assert result.data["total"] == 1


class TestEmitTest:
    def test_without_eventbus(self, skill):
        result = run(skill.execute("emit_test", {}))
        assert result.success
        assert result.data["emitted"] is False

    def test_with_eventbus(self, skill):
        mock_context = MagicMock()
        mock_context.call_skill = AsyncMock(return_value=SkillResult(
            success=True, message="ok",
        ))
        skill.context = mock_context

        result = run(skill.execute("emit_test", {}))
        assert result.success
        assert result.data["emitted"] is True
        assert skill._stats["events_emitted"] == 1


class TestCredentials:
    def test_check_credentials(self, skill):
        assert skill.check_credentials() is True
