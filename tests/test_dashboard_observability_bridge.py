"""Tests for DashboardObservabilityBridgeSkill."""
import json
import pytest
import time
from pathlib import Path
from singularity.skills.dashboard_observability_bridge import (
    DashboardObservabilityBridgeSkill, _sparkline, DATA_DIR, BRIDGE_CONFIG_FILE,
    METRICS_FILE, ALERTS_FILE, DEFAULT_METRIC_MAPPINGS,
)


@pytest.fixture(autouse=True)
def clean_data(tmp_path, monkeypatch):
    monkeypatch.setattr("singularity.skills.dashboard_observability_bridge.DATA_DIR", tmp_path)
    monkeypatch.setattr("singularity.skills.dashboard_observability_bridge.BRIDGE_CONFIG_FILE", tmp_path / "dashboard_obs_bridge.json")
    monkeypatch.setattr("singularity.skills.dashboard_observability_bridge.METRICS_FILE", tmp_path / "observability_metrics.json")
    monkeypatch.setattr("singularity.skills.dashboard_observability_bridge.ALERTS_FILE", tmp_path / "observability_alerts.json")
    yield tmp_path


def _seed_metrics(tmp_path, metric_name="skill.latency", values=None, labels=None):
    """Seed ObservabilitySkill metrics data."""
    if values is None:
        values = [0.1, 0.2, 0.3, 0.15, 0.25]
    labels = labels or {}
    series_key = f"{metric_name}{{}}"
    if labels:
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        series_key = f"{metric_name}{{{label_str}}}"

    now = time.time()
    points = [{"ts": now - (len(values) - i) * 60, "value": v} for i, v in enumerate(values)]
    data = {
        "series": {
            series_key: {"name": metric_name, "type": "gauge", "labels": labels, "points": points, "created": "2025-01-01T00:00:00Z"}
        },
        "metadata": {"created": "2025-01-01T00:00:00Z", "total_points": len(points)},
    }
    (tmp_path / "observability_metrics.json").write_text(json.dumps(data))
    return data


def _seed_alerts(tmp_path, firing=0, ok=1, cooldown=0):
    """Seed ObservabilitySkill alerts data."""
    rules = {}
    for i in range(firing):
        rules[f"alert_firing_{i}"] = {"metric_name": "test.metric", "condition": "above", "threshold": 90, "severity": "warning", "state": "firing"}
    for i in range(ok):
        rules[f"alert_ok_{i}"] = {"metric_name": "test.metric", "condition": "below", "threshold": 50, "severity": "info", "state": "ok"}
    for i in range(cooldown):
        rules[f"alert_cooldown_{i}"] = {"metric_name": "test.metric", "condition": "above", "threshold": 80, "severity": "warning", "state": "cooldown"}
    data = {"rules": rules, "history": []}
    (tmp_path / "observability_alerts.json").write_text(json.dumps(data))
    return data


@pytest.mark.asyncio
async def test_sync_pulls_metrics(clean_data):
    _seed_metrics(clean_data, "skill.latency", [0.1, 0.2, 0.3])
    skill = DashboardObservabilityBridgeSkill()
    result = await skill.execute("sync", {"section": "performance"})
    assert result.success
    assert result.data["total_metrics"] > 0
    perf = result.data["sections"]["performance"]
    assert len(perf["metrics"]) > 0


@pytest.mark.asyncio
async def test_sync_all_sections(clean_data):
    skill = DashboardObservabilityBridgeSkill()
    result = await skill.execute("sync")
    assert result.success
    assert "performance" in result.data["sections"]
    assert "revenue" in result.data["sections"]


@pytest.mark.asyncio
async def test_push_snapshot(clean_data):
    skill = DashboardObservabilityBridgeSkill()
    scores = {"self_improvement": {"score": 75, "grade": "C"}, "revenue": {"score": 40, "grade": "F"}}
    result = await skill.execute("push_snapshot", {"scores": scores})
    assert result.success
    assert result.data["pushed_count"] == 2
    metrics = json.loads((clean_data / "observability_metrics.json").read_text())
    assert len(metrics["series"]) == 2


@pytest.mark.asyncio
async def test_widget_sparkline(clean_data):
    _seed_metrics(clean_data, "skill.latency", [0.1, 0.2, 0.3, 0.4, 0.5])
    skill = DashboardObservabilityBridgeSkill()
    result = await skill.execute("widget", {"metric_name": "skill.latency", "widget_type": "sparkline"})
    assert result.success
    assert "sparkline" in result.data


@pytest.mark.asyncio
async def test_widget_gauge(clean_data):
    _seed_metrics(clean_data, "skill.latency", [50.0, 60.0, 70.0])
    skill = DashboardObservabilityBridgeSkill()
    result = await skill.execute("widget", {"metric_name": "skill.latency", "widget_type": "gauge"})
    assert result.success
    assert "bar" in result.data


@pytest.mark.asyncio
async def test_widget_counter(clean_data):
    _seed_metrics(clean_data, "skill.latency", [1.0, 2.0, 3.0])
    skill = DashboardObservabilityBridgeSkill()
    result = await skill.execute("widget", {"metric_name": "skill.latency", "widget_type": "counter"})
    assert result.success
    assert result.data["total"] == 6.0


@pytest.mark.asyncio
async def test_widget_trend(clean_data):
    _seed_metrics(clean_data, "skill.latency", [1.0, 2.0, 3.0, 4.0, 5.0])
    skill = DashboardObservabilityBridgeSkill()
    result = await skill.execute("widget", {"metric_name": "skill.latency", "widget_type": "trend"})
    assert result.success
    assert result.data["trend"] == "rising"


@pytest.mark.asyncio
async def test_alerts_summary_healthy(clean_data):
    _seed_alerts(clean_data, firing=0, ok=3)
    skill = DashboardObservabilityBridgeSkill()
    result = await skill.execute("alerts_summary")
    assert result.success
    assert result.data["health"] == "healthy"
    assert result.data["health_score"] == 100


@pytest.mark.asyncio
async def test_alerts_summary_firing(clean_data):
    _seed_alerts(clean_data, firing=2, ok=1)
    skill = DashboardObservabilityBridgeSkill()
    result = await skill.execute("alerts_summary")
    assert result.success
    assert result.data["health"] == "degraded"
    assert result.data["firing_count"] == 2


@pytest.mark.asyncio
async def test_configure_section(clean_data):
    skill = DashboardObservabilityBridgeSkill()
    result = await skill.execute("configure", {
        "section": "performance",
        "metrics": [{"name": "custom.metric", "aggregation": "sum", "label": "Custom"}],
        "window_hours": 6,
    })
    assert result.success
    config = json.loads((clean_data / "dashboard_obs_bridge.json").read_text())
    assert config["mappings"]["performance"]["window_hours"] == 6


@pytest.mark.asyncio
async def test_trend_action(clean_data):
    _seed_metrics(clean_data, "skill.latency", [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5])
    skill = DashboardObservabilityBridgeSkill()
    result = await skill.execute("trend", {"metric_name": "skill.latency"})
    assert result.success
    assert result.data["trend"] == "rising"
    assert result.data["sparkline"]


@pytest.mark.asyncio
async def test_auto_sync(clean_data):
    _seed_metrics(clean_data, "skill.latency", [0.1, 0.2])
    skill = DashboardObservabilityBridgeSkill()
    scores = {"overall": {"score": 60, "grade": "D"}}
    result = await skill.execute("auto_sync", {"scores": scores})
    assert result.success
    assert result.data["pull"]
    assert result.data["push"]["pushed_count"] == 1


@pytest.mark.asyncio
async def test_status(clean_data):
    skill = DashboardObservabilityBridgeSkill()
    result = await skill.execute("status")
    assert result.success
    assert "sections" in result.data
    assert result.data["total_configured_metrics"] > 0


def test_sparkline():
    assert _sparkline([1, 2, 3, 4, 5]) != ""
    assert len(_sparkline([1, 2, 3])) == 3
    assert _sparkline([]) == ""
    assert _sparkline([5, 5, 5]) == "▅▅▅"


@pytest.mark.asyncio
async def test_unknown_action(clean_data):
    skill = DashboardObservabilityBridgeSkill()
    result = await skill.execute("nonexistent")
    assert not result.success
