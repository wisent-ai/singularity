"""Tests for ServiceMonitoringDashboardSkill - unified operational dashboard."""

import pytest
import json
import time
from singularity.skills.service_monitoring_dashboard import (
    ServiceMonitoringDashboardSkill,
    _compute_uptime,
    _compute_trend,
    _determine_severity,
    _service_status,
    STATUS_HEALTHY,
    STATUS_DEGRADED,
    STATUS_DOWN,
    STATUS_UNKNOWN,
    SEVERITY_OK,
    SEVERITY_WARNING,
    SEVERITY_CRITICAL,
    TREND_UP,
    TREND_DOWN,
    TREND_STABLE,
)


@pytest.fixture
def skill(tmp_path):
    s = ServiceMonitoringDashboardSkill(data_path=tmp_path / "dashboard.json")
    return s


@pytest.fixture
def config():
    return {
        "degraded_latency_ms": 1000,
        "critical_latency_ms": 5000,
        "degraded_error_rate": 0.05,
        "critical_error_rate": 0.20,
    }


# --- Unit tests for helper functions ---

def test_compute_uptime_empty():
    assert _compute_uptime([], "svc1", 24) == 100.0


def test_compute_uptime_all_healthy():
    now = time.time()
    snaps = [{"service_id": "svc1", "timestamp": now - 100, "status": "healthy"},
             {"service_id": "svc1", "timestamp": now - 50, "status": "healthy"}]
    assert _compute_uptime(snaps, "svc1", 24) == 100.0


def test_compute_uptime_mixed():
    now = time.time()
    snaps = [{"service_id": "svc1", "timestamp": now - 100, "status": "healthy"},
             {"service_id": "svc1", "timestamp": now - 50, "status": "down"}]
    assert _compute_uptime(snaps, "svc1", 24) == 50.0


def test_compute_trend_stable():
    assert _compute_trend([1, 1, 1, 1]) == TREND_STABLE


def test_compute_trend_up():
    assert _compute_trend([1, 1, 5, 5]) == TREND_UP


def test_compute_trend_down():
    assert _compute_trend([5, 5, 1, 1]) == TREND_DOWN


def test_compute_trend_insufficient():
    assert _compute_trend([5]) == TREND_STABLE


def test_determine_severity_ok(config):
    assert _determine_severity(0.0, 100, config) == SEVERITY_OK


def test_determine_severity_warning_latency(config):
    assert _determine_severity(0.0, 1500, config) == SEVERITY_WARNING


def test_determine_severity_critical_error(config):
    assert _determine_severity(0.25, 100, config) == SEVERITY_CRITICAL


def test_service_status_healthy(config):
    assert _service_status(0.0, 50, config) == STATUS_HEALTHY


def test_service_status_degraded(config):
    assert _service_status(0.06, 50, config) == STATUS_DEGRADED


def test_service_status_down(config):
    assert _service_status(0.25, 50, config) == STATUS_DOWN


# --- Async skill action tests ---

def test_manifest(skill):
    m = skill.manifest
    assert m.skill_id == "service_monitoring_dashboard"
    assert m.category == "operations"
    action_names = [a.name for a in m.actions]
    assert "overview" in action_names
    assert "register_service" in action_names
    assert "record_check" in action_names
    assert "services" in action_names
    assert "revenue" in action_names
    assert "report" in action_names


@pytest.mark.asyncio
async def test_register_service(skill):
    r = await skill.execute("register_service", {"service_id": "api1", "name": "API Service"})
    assert r.success
    assert r.data["service_id"] == "api1"
    assert r.data["service"]["name"] == "API Service"


@pytest.mark.asyncio
async def test_register_missing_params(skill):
    r = await skill.execute("register_service", {"service_id": "", "name": ""})
    assert not r.success


@pytest.mark.asyncio
async def test_record_check(skill):
    await skill.execute("register_service", {"service_id": "api1", "name": "API"})
    r = await skill.execute("record_check", {
        "service_id": "api1", "latency_ms": 50, "error_rate": 0.01,
        "request_count": 100, "revenue": 5.0, "cost": 1.0,
    })
    assert r.success
    assert r.data["status"] == STATUS_HEALTHY


@pytest.mark.asyncio
async def test_record_check_not_found(skill):
    r = await skill.execute("record_check", {"service_id": "nonexistent"})
    assert not r.success


@pytest.mark.asyncio
async def test_overview_empty(skill):
    r = await skill.execute("overview", {})
    assert r.success
    assert r.data["overall_health"] == SEVERITY_OK
    assert r.data["services"]["total"] == 0


@pytest.mark.asyncio
async def test_overview_with_services(skill):
    await skill.execute("register_service", {"service_id": "s1", "name": "Svc1"})
    await skill.execute("record_check", {"service_id": "s1", "latency_ms": 50, "revenue": 10})
    r = await skill.execute("overview", {})
    assert r.success
    assert r.data["services"]["total"] == 1
    assert r.data["revenue"]["total_revenue"] == 10.0


@pytest.mark.asyncio
async def test_services_list(skill):
    await skill.execute("register_service", {"service_id": "s1", "name": "A"})
    await skill.execute("register_service", {"service_id": "s2", "name": "B"})
    await skill.execute("record_check", {"service_id": "s1", "latency_ms": 50})
    r = await skill.execute("services", {})
    assert r.success
    assert r.data["total"] == 2


@pytest.mark.asyncio
async def test_services_filter(skill):
    await skill.execute("register_service", {"service_id": "s1", "name": "A"})
    await skill.execute("record_check", {"service_id": "s1", "latency_ms": 50, "error_rate": 0.0})
    r = await skill.execute("services", {"status_filter": "down"})
    assert r.success
    assert r.data["total"] == 0


@pytest.mark.asyncio
async def test_revenue_dashboard(skill):
    await skill.execute("register_service", {"service_id": "s1", "name": "Svc"})
    await skill.execute("record_check", {"service_id": "s1", "revenue": 10.0, "cost": 3.0, "request_count": 50})
    r = await skill.execute("revenue", {"window_hours": 1})
    assert r.success
    assert r.data["total_revenue"] == 10.0
    assert r.data["total_cost"] == 3.0
    assert r.data["total_profit"] == 7.0


@pytest.mark.asyncio
async def test_uptime_report(skill):
    await skill.execute("register_service", {"service_id": "s1", "name": "Svc"})
    await skill.execute("record_check", {"service_id": "s1", "latency_ms": 50})
    r = await skill.execute("uptime", {"window_hours": 1})
    assert r.success
    assert r.data["average_uptime_pct"] == 100.0


@pytest.mark.asyncio
async def test_trends(skill):
    await skill.execute("register_service", {"service_id": "s1", "name": "Svc"})
    for i in range(4):
        await skill.execute("record_check", {"service_id": "s1", "latency_ms": 100 + i * 10, "revenue": i})
    r = await skill.execute("trends", {})
    assert r.success
    assert len(r.data["services"]) == 1


@pytest.mark.asyncio
async def test_report(skill):
    await skill.execute("register_service", {"service_id": "s1", "name": "MyAPI"})
    await skill.execute("record_check", {"service_id": "s1", "latency_ms": 50, "revenue": 5})
    r = await skill.execute("report", {})
    assert r.success
    assert "MyAPI" in r.data["report"]
    assert "OPERATIONAL STATUS REPORT" in r.data["report"]


@pytest.mark.asyncio
async def test_configure(skill):
    r = await skill.execute("configure", {"degraded_latency_ms": 2000})
    assert r.success
    assert r.data["config"]["degraded_latency_ms"] == 2000


@pytest.mark.asyncio
async def test_configure_no_changes(skill):
    r = await skill.execute("configure", {})
    assert r.success
    assert "No configuration changes" in r.message


@pytest.mark.asyncio
async def test_status(skill):
    r = await skill.execute("status", {})
    assert r.success
    assert r.data["service_count"] == 0
    assert r.data["data_freshness"] == "never"


@pytest.mark.asyncio
async def test_unknown_action(skill):
    r = await skill.execute("nonexistent_action", {})
    assert not r.success
