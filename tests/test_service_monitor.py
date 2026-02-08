"""Tests for ServiceMonitorSkill - service health monitoring, uptime, SLA compliance."""

import pytest
import json
from pathlib import Path
from unittest.mock import patch
from singularity.skills.service_monitor import ServiceMonitorSkill, MONITOR_FILE, _default_state, _calculate_uptime, _now_ts


@pytest.fixture(autouse=True)
def clean_data(tmp_path):
    """Use temp file for tests."""
    test_file = tmp_path / "service_monitor.json"
    with patch("singularity.skills.service_monitor.MONITOR_FILE", test_file):
        with patch("singularity.skills.service_monitor.DATA_DIR", tmp_path):
            yield test_file


@pytest.fixture
def skill(clean_data):
    with patch("singularity.skills.service_monitor.MONITOR_FILE", clean_data):
        with patch("singularity.skills.service_monitor.DATA_DIR", clean_data.parent):
            s = ServiceMonitorSkill()
            return s


@pytest.mark.asyncio
async def test_register_service(skill, clean_data):
    with patch("singularity.skills.service_monitor.MONITOR_FILE", clean_data):
        r = await skill.execute("register", {"service_id": "api-v1", "name": "API v1", "sla_target": 99.95, "tags": ["revenue"]})
    assert r.success
    assert "api-v1" in r.data["service"]["service_id"]
    assert r.data["service"]["sla_target"] == 99.95


@pytest.mark.asyncio
async def test_register_duplicate(skill, clean_data):
    with patch("singularity.skills.service_monitor.MONITOR_FILE", clean_data):
        await skill.execute("register", {"service_id": "svc1", "name": "Svc"})
        r = await skill.execute("register", {"service_id": "svc1", "name": "Svc"})
    assert not r.success
    assert "already registered" in r.message


@pytest.mark.asyncio
async def test_check_service(skill, clean_data):
    with patch("singularity.skills.service_monitor.MONITOR_FILE", clean_data):
        await skill.execute("register", {"service_id": "web", "name": "Web App"})
        r = await skill.execute("check", {"service_id": "web"})
    assert r.success
    assert r.data["results"][0]["status"] == "up"
    assert r.data["all_healthy"]


@pytest.mark.asyncio
async def test_check_simulated_down(skill, clean_data):
    with patch("singularity.skills.service_monitor.MONITOR_FILE", clean_data):
        await skill.execute("register", {"service_id": "db", "name": "Database"})
        r = await skill.execute("check", {"service_id": "db", "simulated_status": "down"})
    assert r.success
    assert r.data["results"][0]["status"] == "down"
    assert not r.data["all_healthy"]


@pytest.mark.asyncio
async def test_status_with_uptime(skill, clean_data):
    with patch("singularity.skills.service_monitor.MONITOR_FILE", clean_data):
        await skill.execute("register", {"service_id": "api", "name": "API"})
        for _ in range(5):
            await skill.execute("check", {"service_id": "api"})
        r = await skill.execute("status", {"service_id": "api"})
    assert r.success
    assert r.data["uptime"]["1h"] == 100.0
    assert r.data["total_checks"] == 5


@pytest.mark.asyncio
async def test_overview(skill, clean_data):
    with patch("singularity.skills.service_monitor.MONITOR_FILE", clean_data):
        await skill.execute("register", {"service_id": "a", "name": "A", "tags": ["prod"]})
        await skill.execute("register", {"service_id": "b", "name": "B", "tags": ["dev"]})
        await skill.execute("check", {})
        r = await skill.execute("overview", {})
    assert r.success
    assert r.data["summary"]["total_services"] == 2


@pytest.mark.asyncio
async def test_overview_tag_filter(skill, clean_data):
    with patch("singularity.skills.service_monitor.MONITOR_FILE", clean_data):
        await skill.execute("register", {"service_id": "a", "name": "A", "tags": ["prod"]})
        await skill.execute("register", {"service_id": "b", "name": "B", "tags": ["dev"]})
        r = await skill.execute("overview", {"tag": "prod"})
    assert r.success
    assert r.data["summary"]["total_services"] == 1


@pytest.mark.asyncio
async def test_incident_detection(skill, clean_data):
    with patch("singularity.skills.service_monitor.MONITOR_FILE", clean_data):
        await skill.execute("register", {"service_id": "svc", "name": "Service"})
        await skill.execute("check", {"service_id": "svc", "simulated_status": "down"})
        r = await skill.execute("incidents", {"service_id": "svc"})
    assert r.success
    assert r.data["ongoing"] >= 1


@pytest.mark.asyncio
async def test_sla_report(skill, clean_data):
    with patch("singularity.skills.service_monitor.MONITOR_FILE", clean_data):
        await skill.execute("register", {"service_id": "api", "name": "API", "sla_target": 99.0})
        for _ in range(10):
            await skill.execute("check", {"service_id": "api"})
        r = await skill.execute("sla_report", {"service_id": "api"})
    assert r.success
    assert r.data["reports"][0]["uptime_pct"] == 100.0
    assert r.data["reports"][0]["compliant"] is True


@pytest.mark.asyncio
async def test_revenue_report(skill, clean_data):
    with patch("singularity.skills.service_monitor.MONITOR_FILE", clean_data):
        await skill.execute("register", {"service_id": "paid", "name": "Paid API"})
        # Manually set revenue
        with patch("singularity.skills.service_monitor.MONITOR_FILE", clean_data):
            skill.record_service_revenue("paid", revenue=50.0, cost=10.0, requests=100)
        r = await skill.execute("revenue_report", {"service_id": "paid"})
    assert r.success
    assert r.data["reports"][0]["revenue"] == 50.0
    assert r.data["reports"][0]["profit"] == 40.0
    assert r.data["totals"]["margin_pct"] == 80.0


@pytest.mark.asyncio
async def test_status_page(skill, clean_data):
    with patch("singularity.skills.service_monitor.MONITOR_FILE", clean_data):
        await skill.execute("register", {"service_id": "web", "name": "Web"})
        await skill.execute("check", {"service_id": "web"})
        r = await skill.execute("status_page", {})
    assert r.success
    assert r.data["system_status"] == "operational"
    assert len(r.data["services"]) == 1


@pytest.mark.asyncio
async def test_configure(skill, clean_data):
    with patch("singularity.skills.service_monitor.MONITOR_FILE", clean_data):
        r = await skill.execute("configure", {"default_sla_target": 99.5, "auto_incident_detection": False})
    assert r.success
    assert r.data["config"]["default_sla_target"] == 99.5
    assert r.data["config"]["auto_incident_detection"] is False


@pytest.mark.asyncio
async def test_unregister(skill, clean_data):
    with patch("singularity.skills.service_monitor.MONITOR_FILE", clean_data):
        await skill.execute("register", {"service_id": "tmp", "name": "Temp"})
        r = await skill.execute("unregister", {"service_id": "tmp"})
    assert r.success
    assert "tmp" in r.data["removed_service_id"]


def test_calculate_uptime():
    now = _now_ts()
    checks = [
        {"timestamp": now - 100, "status": "up"},
        {"timestamp": now - 80, "status": "up"},
        {"timestamp": now - 60, "status": "down"},
        {"timestamp": now - 40, "status": "up"},
        {"timestamp": now - 20, "status": "up"},
    ]
    uptime = _calculate_uptime(checks, 200)
    assert uptime == 80.0  # 4/5 = 80%
