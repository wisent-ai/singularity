"""Tests for ServiceMonitorSkill."""
import pytest
from singularity.skills.service_monitor import ServiceMonitorSkill


@pytest.fixture
def monitor(tmp_path):
    path = tmp_path / "service_monitor.json"
    return ServiceMonitorSkill(data_path=path)


@pytest.mark.asyncio
async def test_register_service(monitor):
    r = await monitor.execute("register", {"service_id": "code_review", "name": "Code Review"})
    assert r.success
    assert r.data["service_id"] == "code_review"
    assert r.data["action"] == "registered"


@pytest.mark.asyncio
async def test_register_requires_fields(monitor):
    r = await monitor.execute("register", {"service_id": ""})
    assert not r.success


@pytest.mark.asyncio
async def test_register_with_custom_sla(monitor):
    r = await monitor.execute("register", {
        "service_id": "premium",
        "name": "Premium",
        "sla": {"availability": 99.99, "latency_p95_ms": 1000},
    })
    assert r.success
    assert r.data["sla"]["availability"] == 99.99
    assert r.data["sla"]["latency_p95_ms"] == 1000


@pytest.mark.asyncio
async def test_record_success(monitor):
    await monitor.execute("register", {"service_id": "s1", "name": "S1"})
    r = await monitor.execute("record", {"service_id": "s1", "success": True, "latency_ms": 200, "revenue": 0.50, "cost": 0.05})
    assert r.success
    assert r.data["total_requests"] == 1
    assert r.data["revenue_total"] == 0.50
    assert r.revenue == 0.50


@pytest.mark.asyncio
async def test_record_failure(monitor):
    await monitor.execute("register", {"service_id": "s1", "name": "S1"})
    r = await monitor.execute("record", {"service_id": "s1", "success": False})
    assert r.success
    assert r.data["health"] in ("healthy", "degraded", "down", "unknown")


@pytest.mark.asyncio
async def test_record_unregistered(monitor):
    r = await monitor.execute("record", {"service_id": "nope", "success": True})
    assert not r.success


@pytest.mark.asyncio
async def test_status_single(monitor):
    await monitor.execute("register", {"service_id": "s1", "name": "S1"})
    for _ in range(5):
        await monitor.execute("record", {"service_id": "s1", "success": True, "latency_ms": 100, "revenue": 1.0, "cost": 0.1})
    r = await monitor.execute("status", {"service_id": "s1"})
    assert r.success
    assert r.data["total_requests"] == 5
    assert r.data["total_revenue"] == 5.0
    assert r.data["success_rate"] == 100.0


@pytest.mark.asyncio
async def test_status_all(monitor):
    await monitor.execute("register", {"service_id": "s1", "name": "S1"})
    await monitor.execute("register", {"service_id": "s2", "name": "S2"})
    r = await monitor.execute("status", {})
    assert r.success
    assert len(r.data["services"]) == 2


@pytest.mark.asyncio
async def test_dashboard(monitor):
    await monitor.execute("register", {"service_id": "s1", "name": "S1"})
    for i in range(10):
        await monitor.execute("record", {"service_id": "s1", "success": i < 9, "latency_ms": 100 + i * 10, "revenue": 0.5, "cost": 0.05})
    r = await monitor.execute("dashboard", {})
    assert r.success
    overview = r.data["overview"]
    assert overview["total_services"] == 1
    assert overview["total_requests"] == 10
    assert overview["total_revenue"] == 5.0


@pytest.mark.asyncio
async def test_sla_check_compliant(monitor):
    await monitor.execute("register", {"service_id": "s1", "name": "S1"})
    for _ in range(20):
        await monitor.execute("record", {"service_id": "s1", "success": True, "latency_ms": 100})
    r = await monitor.execute("sla_check", {})
    assert r.success
    assert r.data["compliant_count"] == 1
    assert len(r.data["breaches"]) == 0


@pytest.mark.asyncio
async def test_sla_check_breach(monitor):
    await monitor.execute("register", {"service_id": "s1", "name": "S1", "sla": {"error_rate_max": 5.0}})
    # 4 out of 10 fail = 40% error rate
    for i in range(10):
        await monitor.execute("record", {"service_id": "s1", "success": i < 6})
    r = await monitor.execute("sla_check", {})
    assert r.success
    assert len(r.data["breaches"]) == 1
    assert "error_rate" in r.data["breaches"][0]["violated"]


@pytest.mark.asyncio
async def test_incidents(monitor):
    await monitor.execute("register", {"service_id": "s1", "name": "S1", "sla": {"error_rate_max": 1.0}})
    for _ in range(5):
        await monitor.execute("record", {"service_id": "s1", "success": False})
    await monitor.execute("sla_check", {})
    r = await monitor.execute("incidents", {"status": "active"})
    assert r.success
    assert len(r.data["incidents"]) >= 1


@pytest.mark.asyncio
async def test_top_services(monitor):
    await monitor.execute("register", {"service_id": "s1", "name": "S1"})
    await monitor.execute("register", {"service_id": "s2", "name": "S2"})
    for _ in range(5):
        await monitor.execute("record", {"service_id": "s1", "success": True, "revenue": 2.0})
    for _ in range(3):
        await monitor.execute("record", {"service_id": "s2", "success": True, "revenue": 5.0})
    r = await monitor.execute("top_services", {"sort_by": "revenue"})
    assert r.success
    assert r.data["ranking"][0]["service_id"] == "s2"  # Higher total revenue


@pytest.mark.asyncio
async def test_recommend(monitor):
    await monitor.execute("register", {"service_id": "s1", "name": "S1"})
    r = await monitor.execute("recommend", {})
    assert r.success
    # s1 has 0 requests, should get "attention" recommendation
    assert any(rec["type"] == "attention" for rec in r.data["recommendations"])


@pytest.mark.asyncio
async def test_recommend_high_error(monitor):
    await monitor.execute("register", {"service_id": "s1", "name": "S1", "sla": {"error_rate_max": 5.0}})
    for _ in range(10):
        await monitor.execute("record", {"service_id": "s1", "success": False})
    r = await monitor.execute("recommend", {})
    assert r.success
    assert any(rec["type"] == "fix" for rec in r.data["recommendations"])


@pytest.mark.asyncio
async def test_configure(monitor):
    r = await monitor.execute("configure", {"key": "snapshot_interval_seconds", "value": 60})
    assert r.success


@pytest.mark.asyncio
async def test_configure_nested(monitor):
    r = await monitor.execute("configure", {"key": "default_sla.availability", "value": 99.99})
    assert r.success


@pytest.mark.asyncio
async def test_configure_unknown_key(monitor):
    r = await monitor.execute("configure", {"key": "nonexistent", "value": 42})
    assert not r.success


@pytest.mark.asyncio
async def test_customer_tracking(monitor):
    await monitor.execute("register", {"service_id": "s1", "name": "S1"})
    await monitor.execute("record", {"service_id": "s1", "success": True, "customer_id": "c1"})
    await monitor.execute("record", {"service_id": "s1", "success": True, "customer_id": "c2"})
    await monitor.execute("record", {"service_id": "s1", "success": True, "customer_id": "c1"})
    r = await monitor.execute("status", {"service_id": "s1"})
    assert r.data["unique_customers"] == 2


@pytest.mark.asyncio
async def test_unknown_action(monitor):
    r = await monitor.execute("nonexistent_action", {})
    assert not r.success
