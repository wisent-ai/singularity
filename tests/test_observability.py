"""Tests for ObservabilitySkill."""

import pytest
import time
from singularity.skills.observability import (
    ObservabilitySkill, METRICS_FILE, ALERTS_FILE,
    COUNTER, GAUGE, HISTOGRAM,
    _percentile, _aggregate, _series_key, _match_labels,
)
import singularity.skills.observability as mod


@pytest.fixture
def skill(tmp_path):
    """Create skill with temp data paths."""
    s = ObservabilitySkill()
    mod.METRICS_FILE = tmp_path / "metrics.json"
    mod.ALERTS_FILE = tmp_path / "alerts.json"
    return s


@pytest.mark.asyncio
async def test_manifest(skill):
    m = skill.manifest
    assert m.skill_id == "observability"
    assert len(m.actions) == 8
    names = [a.name for a in m.actions]
    assert "emit" in names
    assert "query" in names
    assert "alert_create" in names
    assert "check_alerts" in names


@pytest.mark.asyncio
async def test_emit_gauge(skill):
    r = await skill.execute("emit", {"name": "cpu.usage", "value": 75.5})
    assert r.success
    assert r.data["value"] == 75.5


@pytest.mark.asyncio
async def test_emit_counter_accumulates(skill):
    await skill.execute("emit", {"name": "requests", "value": 10, "metric_type": "counter"})
    r = await skill.execute("emit", {"name": "requests", "value": 5, "metric_type": "counter"})
    assert r.success
    assert r.data["value"] == 15  # 10 + 5


@pytest.mark.asyncio
async def test_emit_with_labels(skill):
    r = await skill.execute("emit", {
        "name": "latency", "value": 120, "labels": {"skill": "code_review", "action": "review"},
    })
    assert r.success
    assert "code_review" in r.data["key"]


@pytest.mark.asyncio
async def test_emit_validation(skill):
    r = await skill.execute("emit", {"name": "", "value": 1})
    assert not r.success
    r = await skill.execute("emit", {"name": "x"})
    assert not r.success
    r = await skill.execute("emit", {"name": "x", "value": "not_a_number"})
    assert not r.success


@pytest.mark.asyncio
async def test_query_basic(skill):
    await skill.execute("emit", {"name": "temp", "value": 10})
    await skill.execute("emit", {"name": "temp", "value": 20})
    await skill.execute("emit", {"name": "temp", "value": 30})
    r = await skill.execute("query", {"name": "temp", "aggregation": "avg"})
    assert r.success
    assert r.data["value"] == 20.0
    assert r.data["point_count"] == 3


@pytest.mark.asyncio
async def test_query_sum(skill):
    await skill.execute("emit", {"name": "earnings", "value": 100})
    await skill.execute("emit", {"name": "earnings", "value": 200})
    r = await skill.execute("query", {"name": "earnings", "aggregation": "sum"})
    assert r.success
    assert r.data["value"] == 300.0


@pytest.mark.asyncio
async def test_query_with_labels_filter(skill):
    await skill.execute("emit", {"name": "lat", "value": 10, "labels": {"env": "prod"}})
    await skill.execute("emit", {"name": "lat", "value": 90, "labels": {"env": "staging"}})
    r = await skill.execute("query", {"name": "lat", "labels": {"env": "prod"}, "aggregation": "avg"})
    assert r.success
    assert r.data["value"] == 10.0


@pytest.mark.asyncio
async def test_query_group_by(skill):
    await skill.execute("emit", {"name": "req", "value": 5, "labels": {"service": "api"}})
    await skill.execute("emit", {"name": "req", "value": 3, "labels": {"service": "web"}})
    r = await skill.execute("query", {"name": "req", "aggregation": "sum", "group_by": "service"})
    assert r.success
    assert r.data["results"]["api"]["value"] == 5.0
    assert r.data["results"]["web"]["value"] == 3.0


@pytest.mark.asyncio
async def test_query_no_data(skill):
    r = await skill.execute("query", {"name": "nonexistent"})
    assert r.success
    assert r.data["result"] is None


@pytest.mark.asyncio
async def test_alert_create_and_list(skill):
    r = await skill.execute("alert_create", {
        "name": "high_error_rate",
        "metric_name": "errors",
        "condition": "above",
        "threshold": 10,
        "severity": "critical",
    })
    assert r.success
    r2 = await skill.execute("alert_list", {})
    assert r2.success
    assert r2.data["total"] == 1
    assert r2.data["rules"][0]["name"] == "high_error_rate"


@pytest.mark.asyncio
async def test_alert_delete(skill):
    await skill.execute("alert_create", {
        "name": "to_delete", "metric_name": "x", "condition": "above", "threshold": 1,
    })
    r = await skill.execute("alert_delete", {"name": "to_delete"})
    assert r.success
    r2 = await skill.execute("alert_list", {})
    assert r2.data["total"] == 0


@pytest.mark.asyncio
async def test_check_alerts_fires(skill):
    await skill.execute("emit", {"name": "error_count", "value": 50})
    await skill.execute("alert_create", {
        "name": "high_errors", "metric_name": "error_count",
        "condition": "above", "threshold": 10, "window_minutes": 60,
    })
    r = await skill.execute("check_alerts", {})
    assert r.success
    assert len(r.data["fired"]) == 1
    assert r.data["fired"][0]["name"] == "high_errors"


@pytest.mark.asyncio
async def test_check_alerts_below_condition(skill):
    await skill.execute("emit", {"name": "uptime", "value": 0.5})
    await skill.execute("alert_create", {
        "name": "low_uptime", "metric_name": "uptime",
        "condition": "below", "threshold": 0.99,
    })
    r = await skill.execute("check_alerts", {})
    assert len(r.data["fired"]) == 1


@pytest.mark.asyncio
async def test_export(skill):
    await skill.execute("emit", {"name": "metric1", "value": 42})
    await skill.execute("emit", {"name": "metric2", "value": 99})
    r = await skill.execute("export", {"name": "metric1"})
    assert r.success
    assert r.data["series_count"] == 1


@pytest.mark.asyncio
async def test_status(skill):
    await skill.execute("emit", {"name": "a", "value": 1})
    await skill.execute("emit", {"name": "b", "value": 2})
    r = await skill.execute("status", {})
    assert r.success
    assert r.data["series_count"] == 2
    assert r.data["total_points"] == 2


def test_percentile():
    assert _percentile([1, 2, 3, 4, 5], 50) == 3.0
    assert _percentile([1, 2, 3, 4, 5], 0) == 1.0
    assert _percentile([], 50) == 0.0


def test_series_key():
    assert _series_key("cpu", {}) == "cpu"
    assert _series_key("cpu", {"host": "a"}) == "cpu{host=a}"


def test_match_labels():
    assert _match_labels({"a": "1", "b": "2"}, {"a": "1"})
    assert not _match_labels({"a": "1"}, {"a": "2"})
    assert _match_labels({"a": "1"}, {})
