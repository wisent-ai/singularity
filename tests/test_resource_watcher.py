#!/usr/bin/env python3
"""Tests for ResourceWatcherSkill."""

import asyncio
import json
import pytest
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import patch

from singularity.skills.resource_watcher import ResourceWatcherSkill, DATA_FILE


@pytest.fixture(autouse=True)
def clean_data():
    """Remove data file before/after each test."""
    if DATA_FILE.exists():
        DATA_FILE.unlink()
    yield
    if DATA_FILE.exists():
        DATA_FILE.unlink()


@pytest.fixture
def skill():
    return ResourceWatcherSkill()


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


class TestRecord:
    def test_record_basic(self, skill):
        r = run(skill.execute("record", {"skill_id": "github", "action": "create_pr", "cost": 0.05, "tokens": 1500}))
        assert r.success
        assert "github:create_pr" in r.message
        assert skill._data["cumulative_cost"] == 0.05
        assert skill._data["cumulative_tokens"] == 1500

    def test_record_updates_action_stats(self, skill):
        run(skill.execute("record", {"skill_id": "llm", "action": "think", "cost": 0.02, "tokens": 800}))
        run(skill.execute("record", {"skill_id": "llm", "action": "think", "cost": 0.04, "tokens": 1200}))
        stats = skill._data["action_stats"]["llm:think"]
        assert stats["count"] == 2
        assert stats["total_cost"] == 0.06
        assert stats["avg_cost"] == 0.03

    def test_record_tracks_model_usage(self, skill):
        run(skill.execute("record", {"skill_id": "x", "action": "y", "cost": 0.01, "tokens": 500, "model": "gpt-4o"}))
        assert "gpt-4o" in skill._data["model_usage"]
        assert skill._data["model_usage"]["gpt-4o"]["calls"] == 1

    def test_record_persists_to_disk(self, skill):
        run(skill.execute("record", {"skill_id": "a", "action": "b", "cost": 0.1}))
        assert DATA_FILE.exists()
        data = json.loads(DATA_FILE.read_text())
        assert data["cumulative_cost"] == 0.1

    def test_log_capped_at_1000(self, skill):
        for i in range(1010):
            skill._data["consumption_log"].append({"i": i})
        run(skill.execute("record", {"skill_id": "x", "action": "y"}))
        assert len(skill._data["consumption_log"]) == 1000


class TestStatus:
    def test_status_healthy(self, skill):
        r = run(skill.execute("status"))
        assert r.success
        assert r.data["health"] == "healthy"
        assert r.data["pct_used"] == 0.0

    def test_status_warning(self, skill):
        skill._data["cumulative_cost"] = 80
        skill._data["total_budget"] = 100
        r = run(skill.execute("status"))
        assert r.data["health"] == "warning"

    def test_status_critical(self, skill):
        skill._data["cumulative_cost"] = 96
        skill._data["total_budget"] = 100
        r = run(skill.execute("status"))
        assert r.data["health"] == "critical"

    def test_status_with_live_balance(self, skill):
        skill.set_agent_hooks(get_balance=lambda: 42.0)
        r = run(skill.execute("status"))
        assert r.data["live_agent_balance"] == 42.0


class TestForecast:
    def test_forecast_no_data(self, skill):
        r = run(skill.execute("forecast"))
        assert r.success
        assert r.data["status"] == "no_data"

    def test_forecast_with_data(self, skill):
        now = datetime.utcnow()
        skill._data["consumption_log"] = [
            {"timestamp": (now - timedelta(minutes=10)).isoformat(), "cost": 0.5},
            {"timestamp": now.isoformat(), "cost": 0.5},
        ]
        skill._data["cumulative_cost"] = 1.0
        skill._data["total_budget"] = 100.0
        r = run(skill.execute("forecast"))
        assert r.success
        assert "exhaustion_time" in r.data
        assert r.data["burn_rate_per_min"] > 0
        assert "1h" in r.data["horizons"]


class TestTopCosts:
    def test_top_costs_empty(self, skill):
        r = run(skill.execute("top_costs"))
        assert r.success
        assert r.data["top_actions"] == []

    def test_top_costs_ranked(self, skill):
        run(skill.execute("record", {"skill_id": "cheap", "action": "a", "cost": 0.001}))
        run(skill.execute("record", {"skill_id": "expensive", "action": "b", "cost": 0.50}))
        r = run(skill.execute("top_costs", {"limit": 2}))
        assert r.data["top_actions"][0]["action"] == "expensive:b"


class TestBudgetAndAlerts:
    def test_set_budget(self, skill):
        r = run(skill.execute("set_budget", {"total_budget": 50.0, "thresholds": [0.5, 0.8]}))
        assert r.success
        assert skill._data["total_budget"] == 50.0
        assert skill._data["alert_thresholds"] == [0.5, 0.8]

    def test_set_budget_rejects_zero(self, skill):
        r = run(skill.execute("set_budget", {"total_budget": 0}))
        assert not r.success

    def test_alerts_fire_on_threshold(self, skill):
        run(skill.execute("set_budget", {"total_budget": 10.0, "thresholds": [0.5]}))
        r = run(skill.execute("record", {"skill_id": "x", "action": "y", "cost": 6.0}))
        assert len(r.data["alerts_fired"]) == 1
        assert r.data["alerts_fired"][0]["pct_label"] == "50%"

    def test_alerts_dont_repeat(self, skill):
        run(skill.execute("set_budget", {"total_budget": 10.0, "thresholds": [0.5]}))
        run(skill.execute("record", {"skill_id": "x", "action": "y", "cost": 6.0}))
        r = run(skill.execute("record", {"skill_id": "x", "action": "y", "cost": 0.1}))
        assert len(r.data["alerts_fired"]) == 0

    def test_list_alerts(self, skill):
        run(skill.execute("set_budget", {"total_budget": 10.0, "thresholds": [0.5]}))
        run(skill.execute("record", {"skill_id": "x", "action": "y", "cost": 6.0}))
        r = run(skill.execute("alerts"))
        assert len(r.data["alerts"]) == 1


class TestRecommendAndReset:
    def test_recommend_empty(self, skill):
        r = run(skill.execute("recommend"))
        assert r.success
        assert r.data["recommendations"] == []

    def test_recommend_expensive_action(self, skill):
        for _ in range(5):
            run(skill.execute("record", {"skill_id": "llm", "action": "generate", "cost": 0.05}))
        r = run(skill.execute("recommend"))
        types = [rec["type"] for rec in r.data["recommendations"]]
        assert "expensive_action" in types

    def test_reset_keeps_budget(self, skill):
        run(skill.execute("set_budget", {"total_budget": 42.0}))
        run(skill.execute("record", {"skill_id": "x", "action": "y", "cost": 5.0}))
        run(skill.execute("reset", {"keep_budget": True}))
        assert skill._data["total_budget"] == 42.0
        assert skill._data["cumulative_cost"] == 0

    def test_reset_clears_budget(self, skill):
        run(skill.execute("set_budget", {"total_budget": 42.0}))
        run(skill.execute("reset", {"keep_budget": False}))
        assert skill._data["total_budget"] == 100.0  # default


class TestHelpers:
    def test_budget_context_string(self, skill):
        skill._data["total_budget"] = 100.0
        skill._data["cumulative_cost"] = 25.0
        ctx = skill.get_budget_context()
        assert "$75.00" in ctx
        assert "25%" in ctx

    def test_find_cheaper_model(self, skill):
        cheaper = skill._find_cheaper_model("gpt-4o")
        assert cheaper is not None
        # Should be a model with lower output cost
        from singularity.skills.resource_watcher import MODEL_COSTS
        assert MODEL_COSTS[cheaper]["output"] < MODEL_COSTS["gpt-4o"]["output"]

    def test_manifest(self, skill):
        m = skill.manifest
        assert m.skill_id == "resource_watcher"
        assert len(m.actions) == 8
