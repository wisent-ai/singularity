#!/usr/bin/env python3
"""Tests for CircuitSharingEventBridgeSkill."""

import asyncio
import json
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from singularity.skills.circuit_sharing_events import (
    CircuitSharingEventBridgeSkill,
    BRIDGE_STATE_FILE,
)
from singularity.skills.base import SkillResult


@pytest.fixture
def tmp_data(tmp_path):
    """Patch data file to use tmp dir."""
    test_file = tmp_path / "circuit_sharing_events.json"
    with patch.object(
        CircuitSharingEventBridgeSkill, "_load_state"
    ) as mock_load:
        mock_load.side_effect = lambda self=None: None
        skill = CircuitSharingEventBridgeSkill()
    # Re-init with empty state
    skill._known_peers = {}
    skill._last_sync_snapshot = {}
    skill._event_history = []
    skill._config = skill._default_config()
    skill._stats = skill._default_stats()
    # Patch save to use tmp
    original_save = skill._save_state
    def patched_save():
        import singularity.skills.circuit_sharing_events as mod
        old = mod.BRIDGE_STATE_FILE
        mod.BRIDGE_STATE_FILE = test_file  # won't actually be used since we override
        test_file.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "known_peers": skill._known_peers,
            "last_sync_snapshot": skill._last_sync_snapshot,
            "event_history": skill._event_history,
            "config": skill._config,
            "stats": skill._stats,
        }
        with open(test_file, "w") as f:
            json.dump(data, f, indent=2, default=str)
        mod.BRIDGE_STATE_FILE = old
    skill._save_state = patched_save
    return skill


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


class TestInstantiation:
    def test_manifest(self, tmp_data):
        skill = tmp_data
        assert skill.manifest.skill_id == "circuit_sharing_events"
        assert skill.manifest.version == "1.0.0"
        assert skill.manifest.category == "replication"

    def test_actions(self, tmp_data):
        skill = tmp_data
        names = [a.name for a in skill.manifest.actions]
        assert "monitor" in names
        assert "configure" in names
        assert "status" in names
        assert "history" in names
        assert "emit_test" in names
        assert "fleet_check" in names

    def test_default_config(self, tmp_data):
        skill = tmp_data
        assert skill._config["emit_on_state_adopted"] is True
        assert skill._config["emit_on_fleet_alert"] is True
        assert skill._config["fleet_open_threshold"] == 0.5

    def test_unknown_action(self, tmp_data):
        result = run(tmp_data.execute("nonexistent", {}))
        assert not result.success
        assert "Unknown action" in result.message


class TestConfigure:
    def test_configure_emission_flags(self, tmp_data):
        skill = tmp_data
        result = run(skill.execute("configure", {
            "emit_on_state_adopted": False,
            "emit_on_fleet_alert": False,
        }))
        assert result.success
        assert skill._config["emit_on_state_adopted"] is False
        assert skill._config["emit_on_fleet_alert"] is False

    def test_configure_threshold(self, tmp_data):
        skill = tmp_data
        result = run(skill.execute("configure", {"fleet_open_threshold": 0.3}))
        assert result.success
        assert skill._config["fleet_open_threshold"] == 0.3

    def test_configure_clamps_threshold(self, tmp_data):
        skill = tmp_data
        run(skill.execute("configure", {"fleet_open_threshold": 2.0}))
        assert skill._config["fleet_open_threshold"] == 1.0
        run(skill.execute("configure", {"fleet_open_threshold": -1.0}))
        assert skill._config["fleet_open_threshold"] == 0.0

    def test_configure_no_params(self, tmp_data):
        result = run(tmp_data.execute("configure", {}))
        assert not result.success


class TestStatus:
    def test_status_empty(self, tmp_data):
        result = run(tmp_data.execute("status", {}))
        assert result.success
        assert result.data["stats"]["events_emitted"] == 0
        assert len(result.data["known_peers"]) == 0


class TestHistory:
    def test_history_empty(self, tmp_data):
        result = run(tmp_data.execute("history", {}))
        assert result.success
        assert result.data["events"] == []

    def test_history_with_entries(self, tmp_data):
        skill = tmp_data
        skill._event_history = [
            {"topic": "circuit_sharing.test", "emitted": True, "timestamp": "2026-01-01"},
            {"topic": "circuit_sharing.peer_discovered", "emitted": True, "timestamp": "2026-01-02"},
        ]
        result = run(skill.execute("history", {"limit": 10}))
        assert result.success
        assert len(result.data["events"]) == 2

    def test_history_filter(self, tmp_data):
        skill = tmp_data
        skill._event_history = [
            {"topic": "circuit_sharing.test", "emitted": True, "timestamp": "t1"},
            {"topic": "circuit_sharing.peer_discovered", "emitted": True, "timestamp": "t2"},
        ]
        result = run(skill.execute("history", {"topic_filter": "circuit_sharing.peer"}))
        assert len(result.data["events"]) == 1


class TestEmitTest:
    def test_emit_test_no_eventbus(self, tmp_data):
        """Without EventBus, emit_test should report failure gracefully."""
        result = run(tmp_data.execute("emit_test", {}))
        assert result.success  # Action itself succeeds
        assert result.data["emitted"] is False


class TestMonitor:
    def test_monitor_no_sharing_skill(self, tmp_data):
        """Monitor without circuit sharing skill available."""
        result = run(tmp_data.execute("monitor", {}))
        assert result.success
        assert result.data["events_emitted"] == 0

    def test_monitor_detects_new_peer(self, tmp_data):
        skill = tmp_data
        mock_status = SkillResult(
            success=True,
            message="ok",
            data={
                "peers": [
                    {"peer_id": "agent-alpha", "circuits": 5, "open_circuits": 1},
                ],
                "config": {},
            },
        )
        mock_history = SkillResult(
            success=True,
            message="ok",
            data={"history": []},
        )
        skill.context = MagicMock()
        async def mock_call(skill_id, action, params):
            if skill_id == "circuit_sharing" and action == "status":
                return mock_status
            if skill_id == "circuit_sharing" and action == "history":
                return mock_history
            return SkillResult(success=False, message="unknown")
        skill.context.call_skill = mock_call

        result = run(skill.execute("monitor", {}))
        assert result.success
        assert "agent-alpha" in skill._known_peers
        assert skill._stats["peers_discovered"] == 1

    def test_monitor_detects_sync_adoption(self, tmp_data):
        skill = tmp_data
        mock_status = SkillResult(
            success=True, message="ok",
            data={"peers": [], "config": {}},
        )
        mock_history = SkillResult(
            success=True, message="ok",
            data={"history": [
                {
                    "timestamp": "2026-02-08T12:00:00",
                    "operation": "pull",
                    "agent_id": "agent-beta",
                    "states_adopted": 2,
                    "circuits_processed": 5,
                    "strategy": "pessimistic",
                },
            ]},
        )
        skill.context = MagicMock()
        async def mock_call(skill_id, action, params):
            if action == "status":
                return mock_status
            if action == "history":
                return mock_history
            return SkillResult(success=False, message="unknown")
        skill.context.call_skill = mock_call

        result = run(skill.execute("monitor", {}))
        assert result.success
        assert skill._stats["states_adopted_detected"] == 2
        assert skill._stats["conflicts_detected"] == 2


class TestFleetCheck:
    def test_fleet_check_no_sharing_skill(self, tmp_data):
        result = run(tmp_data.execute("fleet_check", {}))
        assert result.success
        assert result.data["fleet_alert_emitted"] is False

    def test_fleet_check_below_threshold(self, tmp_data):
        skill = tmp_data
        mock_status = SkillResult(
            success=True, message="ok",
            data={
                "peers": [
                    {"peer_id": "a1", "circuits": 10, "open_circuits": 1},
                    {"peer_id": "a2", "circuits": 10, "open_circuits": 2},
                ],
            },
        )
        skill.context = MagicMock()
        async def mock_call(sid, action, params):
            return mock_status
        skill.context.call_skill = mock_call

        result = run(skill.execute("fleet_check", {}))
        assert result.success
        assert result.data["fleet_alert_emitted"] is False
        assert result.data["fleet_open_fraction"] == 0.15

    def test_fleet_check_above_threshold(self, tmp_data):
        skill = tmp_data
        skill._config["fleet_open_threshold"] = 0.3
        mock_status = SkillResult(
            success=True, message="ok",
            data={
                "peers": [
                    {"peer_id": "a1", "circuits": 10, "open_circuits": 8},
                    {"peer_id": "a2", "circuits": 10, "open_circuits": 7},
                ],
            },
        )
        skill.context = MagicMock()
        async def mock_call(sid, action, params):
            return mock_status
        skill.context.call_skill = mock_call

        result = run(skill.execute("fleet_check", {}))
        assert result.success
        assert result.data["fleet_open_fraction"] == 0.75
        # Alert was attempted (but won't emit without EventBus)
        assert skill._stats["fleet_alerts_emitted"] == 1

    def test_fleet_check_no_peers(self, tmp_data):
        skill = tmp_data
        mock_status = SkillResult(
            success=True, message="ok",
            data={"peers": []},
        )
        skill.context = MagicMock()
        async def mock_call(sid, action, params):
            return mock_status
        skill.context.call_skill = mock_call

        result = run(skill.execute("fleet_check", {}))
        assert result.success
        assert result.data["peers_count"] == 0
