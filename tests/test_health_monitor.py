"""Tests for AgentHealthMonitor skill."""
import pytest
import json
from datetime import datetime, timedelta
from unittest.mock import patch
from singularity.skills.health_monitor import AgentHealthMonitor


@pytest.fixture
def monitor(tmp_path):
    path = tmp_path / "health_monitor.json"
    return AgentHealthMonitor(data_path=path)


@pytest.mark.asyncio
async def test_register_agent(monitor):
    r = await monitor.execute("register_agent", {"agent_id": "r1", "agent_name": "Replica-1"})
    assert r.success
    assert "r1" in r.data["agent_id"]


@pytest.mark.asyncio
async def test_register_duplicate(monitor):
    await monitor.execute("register_agent", {"agent_id": "r1"})
    r = await monitor.execute("register_agent", {"agent_id": "r1", "agent_name": "Updated"})
    assert r.success  # overwrite is allowed


@pytest.mark.asyncio
async def test_heartbeat(monitor):
    await monitor.execute("register_agent", {"agent_id": "r1"})
    r = await monitor.execute("heartbeat", {"agent_id": "r1", "status": "ok", "metrics": {"cpu_percent": 50}})
    assert r.success
    assert r.data["heartbeat_count"] == 1
    assert r.data["health_state"] == "healthy"


@pytest.mark.asyncio
async def test_heartbeat_unregistered(monitor):
    r = await monitor.execute("heartbeat", {"agent_id": "unknown"})
    assert not r.success


@pytest.mark.asyncio
async def test_check_health_single(monitor):
    await monitor.execute("register_agent", {"agent_id": "r1"})
    await monitor.execute("heartbeat", {"agent_id": "r1"})
    r = await monitor.execute("check_health", {"agent_id": "r1"})
    assert r.success
    assert r.data["agents"]["r1"]["health_state"] == "healthy"


@pytest.mark.asyncio
async def test_check_health_degraded(monitor):
    await monitor.execute("register_agent", {"agent_id": "r1", "expected_heartbeat_seconds": 10})
    await monitor.execute("heartbeat", {"agent_id": "r1"})
    # Simulate time passing beyond degraded threshold
    data = json.loads(monitor.data_path.read_text())
    old_time = (datetime.now() - timedelta(seconds=60)).isoformat()
    data["agents"]["r1"]["last_heartbeat"] = old_time
    monitor.data_path.write_text(json.dumps(data))
    r = await monitor.execute("check_health", {"agent_id": "r1"})
    assert r.success
    state = r.data["agents"]["r1"]["health_state"]
    assert state in ("degraded", "dead")


@pytest.mark.asyncio
async def test_fleet_status_empty(monitor):
    r = await monitor.execute("fleet_status", {})
    assert r.success
    assert r.data["agents_count"] == 0


@pytest.mark.asyncio
async def test_fleet_status_multiple(monitor):
    for i in range(3):
        await monitor.execute("register_agent", {"agent_id": f"r{i}"})
        await monitor.execute("heartbeat", {"agent_id": f"r{i}"})
    r = await monitor.execute("fleet_status", {})
    assert r.success
    assert r.data["agents_count"] == 3
    assert r.data["state_counts"]["healthy"] == 3


@pytest.mark.asyncio
async def test_deregister_agent(monitor):
    await monitor.execute("register_agent", {"agent_id": "r1"})
    r = await monitor.execute("deregister_agent", {"agent_id": "r1"})
    assert r.success
    r2 = await monitor.execute("fleet_status", {})
    assert r2.data["agents_count"] == 0


@pytest.mark.asyncio
async def test_metric_anomaly_alerts(monitor):
    await monitor.execute("register_agent", {"agent_id": "r1"})
    r = await monitor.execute("heartbeat", {"agent_id": "r1", "metrics": {"error_rate": 0.8, "memory_percent": 95}})
    assert r.data["alerts_generated"] >= 2


@pytest.mark.asyncio
async def test_get_and_acknowledge_alert(monitor):
    await monitor.execute("register_agent", {"agent_id": "r1"})
    await monitor.execute("heartbeat", {"agent_id": "r1", "metrics": {"error_rate": 0.9}})
    alerts = await monitor.execute("get_alerts", {"severity": "warning"})
    assert len(alerts.data["alerts"]) >= 1
    aid = alerts.data["alerts"][0]["alert_id"]
    ack = await monitor.execute("acknowledge_alert", {"alert_id": aid})
    assert ack.success


@pytest.mark.asyncio
async def test_update_config(monitor):
    r = await monitor.execute("update_config", {"heartbeat_interval_seconds": 30, "auto_restart_enabled": False})
    assert r.success
    assert "heartbeat_interval_seconds" in r.data["updated"]


@pytest.mark.asyncio
async def test_trigger_restart(monitor):
    await monitor.execute("register_agent", {"agent_id": "r1"})
    r = await monitor.execute("trigger_restart", {"agent_id": "r1", "reason": "test restart"})
    assert r.success
    assert r.data["restart_count"] == 1


@pytest.mark.asyncio
async def test_unknown_action(monitor):
    r = await monitor.execute("nonexistent", {})
    assert not r.success
