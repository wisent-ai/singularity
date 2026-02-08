#!/usr/bin/env python3
"""Tests for DashboardObservabilityBridgeSkill."""

import pytest
import time
import json
from singularity.skills.dashboard_observability_bridge import (
    DashboardObservabilityBridgeSkill,
    PILLAR_METRIC_PREFIXES,
)


@pytest.fixture
def skill():
    s = DashboardObservabilityBridgeSkill()
    s._state = s._default_state()
    return s


@pytest.fixture
def sample_metrics():
    """Create sample ObservabilitySkill metrics data."""
    now = time.time()
    return {
        "series": {
            "skill.success|skill=code_review": {
                "name": "skill.success",
                "type": "counter",
                "labels": {"skill": "code_review"},
                "points": [
                    {"value": 1, "timestamp": now - 7000},
                    {"value": 1, "timestamp": now - 5000},
                    {"value": 1, "timestamp": now - 3000},
                    {"value": 1, "timestamp": now - 1000},
                ],
            },
            "revenue.earnings|service=api": {
                "name": "revenue.earnings",
                "type": "gauge",
                "labels": {"service": "api"},
                "points": [
                    {"value": 5.0, "timestamp": now - 6000},
                    {"value": 10.0, "timestamp": now - 3000},
                    {"value": 15.0, "timestamp": now - 500},
                ],
            },
            "skill.latency|skill=shell": {
                "name": "skill.latency",
                "type": "histogram",
                "labels": {"skill": "shell"},
                "points": [
                    {"value": 0.5, "timestamp": now - 4000},
                    {"value": 0.8, "timestamp": now - 2000},
                    {"value": 0.3, "timestamp": now - 100},
                ],
            },
        },
    }


@pytest.fixture
def sample_alerts():
    return {
        "rules": {
            "high_latency": {
                "name": "High Latency Alert",
                "metric": "skill.latency",
                "condition": "gt",
                "threshold": 1.0,
                "state": "inactive",
                "last_checked": time.time() - 60,
                "last_fired": None,
                "fire_count": 0,
                "labels": {},
            },
            "error_spike": {
                "name": "Error Spike",
                "metric": "skill.failure",
                "condition": "gt",
                "threshold": 5,
                "state": "firing",
                "last_checked": time.time(),
                "last_fired": time.time(),
                "fire_count": 3,
                "labels": {"severity": "critical"},
            },
        },
    }


def test_manifest(skill):
    m = skill.manifest
    assert m.skill_id == "dashboard_observability_bridge"
    assert m.category == "infrastructure"
    assert len(m.actions) == 10


def test_wire(skill):
    result = skill.execute("wire")
    assert result.success
    assert skill._state["wired"] is True


def test_unwire(skill):
    skill._state["wired"] = True
    result = skill.execute("unwire")
    assert result.success
    assert skill._state["wired"] is False


def test_build_metric_summaries(skill, sample_metrics):
    now = time.time()
    summaries = skill._build_metric_summaries(sample_metrics, now)
    assert len(summaries) == 3
    # Check summary structure
    s = summaries[0]
    assert "name" in s
    assert "latest_value" in s
    assert "avg_1h" in s
    assert "sparkline" in s
    assert "pillar" in s


def test_build_alert_snapshot(skill, sample_alerts):
    snapshot = skill._build_alert_snapshot(sample_alerts)
    assert len(snapshot) == 2
    # Firing alerts should be first
    assert snapshot[0]["state"] == "firing"
    assert snapshot[0]["name"] == "Error Spike"


def test_classify_pillar(skill):
    assert skill._classify_pillar("skill.success") == "self_improvement"
    assert skill._classify_pillar("revenue.earnings") == "revenue"
    assert skill._classify_pillar("replica.count") == "replication"
    assert skill._classify_pillar("goal.progress") == "goal_setting"
    assert skill._classify_pillar("unknown.metric") == "general"


def test_pillar_scores_from_metrics(skill, sample_metrics):
    now = time.time()
    scores = skill._calculate_pillar_scores(sample_metrics, now)
    assert "self_improvement" in scores
    assert "revenue" in scores
    assert scores["self_improvement"]["score"] >= 0
    assert scores["self_improvement"]["metric_count"] >= 1
    assert "grade" in scores["self_improvement"]


def test_detect_trends_with_change(skill):
    """Test trend detection when metrics change significantly."""
    now = time.time()
    metrics = {
        "series": {
            "revenue.earnings|svc=api": {
                "name": "revenue.earnings",
                "type": "gauge",
                "labels": {},
                "points": [
                    # Previous window: low values
                    {"value": 1.0, "timestamp": now - 5000},
                    {"value": 1.5, "timestamp": now - 4500},
                    # Recent window: high values (improving)
                    {"value": 5.0, "timestamp": now - 2000},
                    {"value": 6.0, "timestamp": now - 1000},
                ],
            },
        },
    }
    trends = skill._detect_trends(metrics, now, window=3600)
    assert len(trends) >= 1
    t = trends[0]
    assert t["metric"] == "revenue.earnings"
    assert t["direction"] == "improving"
    assert t["change_pct"] > 0


def test_detect_trends_error_metric(skill):
    """Error metrics improving means they go DOWN."""
    now = time.time()
    metrics = {
        "series": {
            "error.count|": {
                "name": "error.count",
                "type": "counter",
                "labels": {},
                "points": [
                    {"value": 10.0, "timestamp": now - 5000},
                    {"value": 8.0, "timestamp": now - 4500},
                    {"value": 2.0, "timestamp": now - 2000},
                    {"value": 1.0, "timestamp": now - 1000},
                ],
            },
        },
    }
    trends = skill._detect_trends(metrics, now, window=3600)
    assert len(trends) >= 1
    assert trends[0]["direction"] == "improving"  # Errors went down


def test_configure(skill):
    result = skill.execute("configure", {
        "max_metrics_in_summary": 50,
        "auto_refresh": False,
    })
    assert result.success
    assert skill._state["config"]["max_metrics_in_summary"] == 50
    assert skill._state["config"]["auto_refresh"] is False


def test_configure_empty(skill):
    result = skill.execute("configure", {})
    assert not result.success


def test_metric_summary_with_pillar_filter(skill, sample_metrics):
    now = time.time()
    skill._state["metric_summaries"] = skill._build_metric_summaries(sample_metrics, now)
    result = skill.execute("metric_summary", {"pillar": "revenue"})
    assert result.success
    for m in result.data["metrics"]:
        assert m["pillar"] == "revenue" or m["name"].startswith("revenue.")


def test_alert_status_action(skill, sample_alerts):
    skill._state["alert_snapshot"] = skill._build_alert_snapshot(sample_alerts)
    result = skill.execute("alert_status")
    assert result.success
    assert result.data["firing_count"] == 1
    assert result.data["total"] == 2


def test_history(skill):
    skill._log_event("test.event", {"key": "value"})
    skill._log_event("test.other", {"key": "val2"})
    result = skill.execute("history", {"limit": 10})
    assert result.success
    assert len(result.data["events"]) == 2
    # Filter by type
    result2 = skill.execute("history", {"event_type": "test.event"})
    assert len(result2.data["events"]) == 1


def test_status(skill):
    result = skill.execute("status")
    assert result.success
    assert "wired" in result.data
    assert "config" in result.data


def test_unknown_action(skill):
    result = skill.execute("nonexistent")
    assert not result.success


def test_grade(skill):
    assert skill._grade(95) == "A"
    assert skill._grade(85) == "B"
    assert skill._grade(75) == "C"
    assert skill._grade(65) == "D"
    assert skill._grade(50) == "F"
