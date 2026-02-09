#!/usr/bin/env python3
"""Tests for ForecastStrategyBridgeSkill."""

import pytest
import asyncio
from unittest.mock import patch

from singularity.skills.forecast_strategy_bridge import (
    ForecastStrategyBridgeSkill,
)


@pytest.fixture
def bridge():
    skill = ForecastStrategyBridgeSkill()
    # Reset state
    skill._state = skill._default_state()
    return skill


@pytest.fixture
def sample_forecast_data():
    return {
        "time_series": [
            {"value": 10.0, "timestamp": "2025-01-01"},
            {"value": 12.0, "timestamp": "2025-01-02"},
            {"value": 15.0, "timestamp": "2025-01-03"},
            {"value": 18.0, "timestamp": "2025-01-04"},
            {"value": 20.0, "timestamp": "2025-01-05"},
            {"value": 22.0, "timestamp": "2025-01-06"},
        ],
        "forecasts": [{"model": "linear", "periods": 3}],
        "backtest_results": {"best_model": "linear", "mae": 1.2},
        "stats": {"total_forecasts": 5, "total_trend_checks": 3},
        "config": {"cost_per_period": 25.0, "ema_alpha": 0.3},
    }


@pytest.fixture
def sample_strategy_data():
    return {
        "pillars": {
            "revenue": {"name": "Revenue", "score": 20.0, "capabilities": [], "gaps": [], "last_assessed": None},
            "self_improvement": {"name": "Self Improvement", "score": 50.0, "capabilities": [], "gaps": [], "last_assessed": None},
            "replication": {"name": "Replication", "score": 10.0, "capabilities": [], "gaps": [], "last_assessed": None},
            "goal_setting": {"name": "Goal Setting", "score": 30.0, "capabilities": [], "gaps": [], "last_assessed": None},
        },
        "journal": [],
        "work_log": [],
        "recommendations": [],
        "session_count": 0,
        "created_at": "2025-01-01T00:00:00Z",
        "last_updated": "2025-01-01T00:00:00Z",
    }


def test_manifest(bridge):
    m = bridge.manifest
    assert m.skill_id == "forecast_strategy_bridge"
    actions = [a.name for a in m.actions]
    assert "sync" in actions
    assert "analyze" in actions
    assert "auto_assess" in actions
    assert "threshold" in actions
    assert "history" in actions
    assert "status" in actions


def test_check_credentials(bridge):
    assert bridge.check_credentials() is True


def test_default_state(bridge):
    state = bridge._default_state()
    assert state["config"]["base_revenue_score"] == 30
    assert state["stats"]["total_syncs"] == 0
    assert state["last_sync"] is None


def test_calculate_score_no_data(bridge):
    result = bridge._calculate_revenue_score({"time_series": []})
    assert result["score"] == bridge._state["config"]["min_score"]
    assert len(result["gaps"]) > 0


def test_calculate_score_with_growth(bridge, sample_forecast_data):
    result = bridge._calculate_revenue_score(sample_forecast_data)
    assert result["score"] > 30  # Above base due to growth
    assert any("growth" in f["name"] or "has_revenue" in f["name"] for f in result["factors"])


def test_calculate_score_profitable(bridge):
    data = {
        "time_series": [{"value": 50.0}, {"value": 55.0}, {"value": 60.0}],
        "forecasts": [{"model": "linear"}],
        "backtest_results": {},
        "stats": {"total_forecasts": 1},
        "config": {"cost_per_period": 40.0},
    }
    result = bridge._calculate_revenue_score(data)
    assert any(f["name"] == "profitable" for f in result["factors"])
    assert any("Profitable" in c for c in result["capabilities"])


def test_sync_no_forecast(bridge):
    with patch.object(bridge, "_read_forecast_state", return_value=None):
        result = asyncio.get_event_loop().run_until_complete(bridge.execute("sync"))
    assert not result.success
    assert "No forecast data" in result.message


def test_sync_updates_strategy(bridge, sample_forecast_data, sample_strategy_data):
    with patch.object(bridge, "_read_forecast_state", return_value=sample_forecast_data), \
         patch.object(bridge, "_read_strategy_state", return_value=sample_strategy_data), \
         patch.object(bridge, "_write_strategy_state") as mock_write, \
         patch.object(bridge, "_save_state"):
        result = asyncio.get_event_loop().run_until_complete(bridge.execute("sync"))
    assert result.success
    assert "direction" in result.data
    mock_write.assert_called_once()
    written = mock_write.call_args[0][0]
    assert written["pillars"]["revenue"]["score"] != 20.0  # Changed from original


def test_sync_unchanged_skips(bridge, sample_forecast_data, sample_strategy_data):
    # Pre-calculate what score would be
    score = bridge._calculate_revenue_score(sample_forecast_data)["score"]
    sample_strategy_data["pillars"]["revenue"]["score"] = score
    bridge._state["last_sync"] = {"timestamp": "exists"}

    with patch.object(bridge, "_read_forecast_state", return_value=sample_forecast_data), \
         patch.object(bridge, "_read_strategy_state", return_value=sample_strategy_data):
        result = asyncio.get_event_loop().run_until_complete(bridge.execute("sync"))
    assert result.success
    assert result.data["changed"] is False


def test_analyze_no_data(bridge):
    with patch.object(bridge, "_read_forecast_state", return_value=None):
        result = asyncio.get_event_loop().run_until_complete(bridge.execute("analyze"))
    assert not result.success


def test_analyze_with_data(bridge, sample_forecast_data, sample_strategy_data):
    with patch.object(bridge, "_read_forecast_state", return_value=sample_forecast_data), \
         patch.object(bridge, "_read_strategy_state", return_value=sample_strategy_data):
        result = asyncio.get_event_loop().run_until_complete(bridge.execute("analyze"))
    assert result.success
    assert "action_plan" in result.data
    assert "revenue_assessment" in result.data
    assert "strategic_context" in result.data


def test_threshold_update(bridge):
    result = asyncio.get_event_loop().run_until_complete(
        bridge.execute("threshold", {"base_revenue_score": 40, "growth_trend_bonus": 15})
    )
    assert result.success
    assert bridge._state["config"]["base_revenue_score"] == 40
    assert bridge._state["config"]["growth_trend_bonus"] == 15


def test_threshold_show_config(bridge):
    result = asyncio.get_event_loop().run_until_complete(bridge.execute("threshold", {}))
    assert result.success
    assert "config" in result.data


def test_history_empty(bridge):
    result = asyncio.get_event_loop().run_until_complete(bridge.execute("history"))
    assert result.success
    assert result.data["total"] == 0


def test_status(bridge):
    result = asyncio.get_event_loop().run_until_complete(bridge.execute("status"))
    assert result.success
    assert "stats" in result.data


def test_auto_assess(bridge, sample_forecast_data, sample_strategy_data):
    with patch.object(bridge, "_read_forecast_state", return_value=sample_forecast_data), \
         patch.object(bridge, "_read_strategy_state", return_value=sample_strategy_data), \
         patch.object(bridge, "_write_strategy_state"), \
         patch.object(bridge, "_save_state"):
        result = asyncio.get_event_loop().run_until_complete(bridge.execute("auto_assess"))
    assert result.success
    assert "score" in result.data
    assert "trend" in result.data
    assert "breakeven" in result.data
    assert "recommendation" in result.data


def test_generate_recommendation_strong(bridge):
    score = {"score": 75}
    trend = {"direction": "growth"}
    breakeven = {"is_profitable": True}
    rec = bridge._generate_recommendation(score, trend, breakeven)
    assert "strong" in rec["summary"].lower() or "other pillars" in rec["summary"].lower()


def test_generate_recommendation_declining(bridge):
    score = {"score": 40}
    trend = {"direction": "decline"}
    breakeven = {"is_profitable": False}
    rec = bridge._generate_recommendation(score, trend, breakeven)
    assert "declining" in rec["summary"].lower() or "urgent" in rec["summary"].lower()


def test_generate_recommendation_weak(bridge):
    score = {"score": 20}
    trend = {"direction": "stable"}
    breakeven = {"is_profitable": False}
    rec = bridge._generate_recommendation(score, trend, breakeven)
    assert "weak" in rec["summary"].lower() or "prioritize" in rec["summary"].lower()


def test_unknown_action(bridge):
    result = asyncio.get_event_loop().run_until_complete(bridge.execute("nonexistent"))
    assert not result.success
    assert "Unknown action" in result.message
