"""Tests for FleetHealthManagerSkill."""

import json
import asyncio
import pytest
from pathlib import Path
from singularity.skills.fleet_health_manager import (
    FleetHealthManagerSkill, FLEET_MANAGER_FILE,
)


@pytest.fixture(autouse=True)
def clean_data():
    if FLEET_MANAGER_FILE.exists():
        FLEET_MANAGER_FILE.unlink()
    yield
    if FLEET_MANAGER_FILE.exists():
        FLEET_MANAGER_FILE.unlink()


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def make_skill():
    return FleetHealthManagerSkill()


def test_manifest():
    s = make_skill()
    m = s.manifest
    assert m.skill_id == "fleet_health_manager"
    actions = [a.name for a in m.actions]
    assert "assess" in actions
    assert "heal" in actions
    assert "scale" in actions
    assert "rolling_update" in actions
    assert "set_policy" in actions
    assert "status" in actions
    assert "incidents" in actions
    assert "register_agent" in actions


def test_register_agent():
    s = make_skill()
    r = run(s.execute("register_agent", {"agent_id": "a1", "agent_type": "specialist"}))
    assert r.success
    assert r.data["fleet_size"] == 1
    # Duplicate
    r2 = run(s.execute("register_agent", {"agent_id": "a1"}))
    assert not r2.success


def test_assess_empty_fleet():
    s = make_skill()
    r = run(s.execute("assess", {}))
    assert r.success
    assert r.data["fleet_size"] == 0
    assert r.data["health_score"] == 100.0


def test_assess_with_unhealthy():
    s = make_skill()
    run(s.execute("register_agent", {"agent_id": "a1"}))
    run(s.execute("register_agent", {"agent_id": "a2"}))
    # Manually set a2 as unresponsive
    data = json.loads(FLEET_MANAGER_FILE.read_text())
    data["fleet_state"]["a2"]["health_status"] = "unresponsive"
    FLEET_MANAGER_FILE.write_text(json.dumps(data))
    r = run(s.execute("assess", {}))
    assert r.success
    assert r.data["healthy"] == 1
    assert r.data["unhealthy"] == 1
    assert len(r.data["recommendations"]) >= 1


def test_heal_restart():
    s = make_skill()
    run(s.execute("register_agent", {"agent_id": "a1"}))
    data = json.loads(FLEET_MANAGER_FILE.read_text())
    data["fleet_state"]["a1"]["health_status"] = "unresponsive"
    FLEET_MANAGER_FILE.write_text(json.dumps(data))
    r = run(s.execute("heal", {"agent_id": "a1"}))
    assert r.success
    assert r.data["action"] == "restart"
    assert r.data["heal_attempt"] == 1


def test_heal_replace_after_max_attempts():
    s = make_skill()
    run(s.execute("register_agent", {"agent_id": "a1"}))
    data = json.loads(FLEET_MANAGER_FILE.read_text())
    data["fleet_state"]["a1"]["health_status"] = "dead"
    data["fleet_state"]["a1"]["heal_attempts"] = 3
    FLEET_MANAGER_FILE.write_text(json.dumps(data))
    r = run(s.execute("heal", {"agent_id": "a1"}))
    assert r.success
    assert r.data["action"] == "replace"
    assert "new_agent_id" in r.data


def test_scale_up():
    s = make_skill()
    r = run(s.execute("scale", {"direction": "up", "count": 3}))
    assert r.success
    assert r.data["count"] == 3
    assert len(r.data["new_agents"]) == 3
    assert r.data["fleet_size"] == 3


def test_scale_down():
    s = make_skill()
    run(s.execute("register_agent", {"agent_id": "a1"}))
    run(s.execute("register_agent", {"agent_id": "a2"}))
    run(s.execute("register_agent", {"agent_id": "a3"}))
    r = run(s.execute("scale", {"direction": "down", "count": 2}))
    assert r.success
    assert r.data["fleet_size"] == 1


def test_scale_respects_limits():
    s = make_skill()
    run(s.execute("set_policy", {"policy_updates": {"max_fleet_size": 2}}))
    run(s.execute("scale", {"direction": "up", "count": 5}))
    r = run(s.execute("status", {}))
    assert r.data["fleet_size"] <= 2


def test_rolling_update():
    s = make_skill()
    run(s.execute("register_agent", {"agent_id": "a1"}))
    run(s.execute("register_agent", {"agent_id": "a2"}))
    r = run(s.execute("rolling_update", {"new_config": {"version": "2.0"}}))
    assert r.success
    assert r.data["total_agents"] == 2


def test_set_policy():
    s = make_skill()
    r = run(s.execute("set_policy", {"policy_updates": {"min_fleet_size": 3, "max_fleet_size": 20}}))
    assert r.success
    assert r.data["applied"]["min_fleet_size"]["new"] == 3
    # Invalid key
    r2 = run(s.execute("set_policy", {"policy_updates": {"invalid_key": 5}}))
    assert not r2.success


def test_status():
    s = make_skill()
    run(s.execute("register_agent", {"agent_id": "a1"}))
    r = run(s.execute("status", {}))
    assert r.success
    assert r.data["fleet_size"] == 1
    assert "health_counts" in r.data
    assert "policies" in r.data


def test_incidents():
    s = make_skill()
    run(s.execute("register_agent", {"agent_id": "a1"}))
    run(s.execute("scale", {"direction": "up", "count": 1}))
    r = run(s.execute("incidents", {"limit": 10}))
    assert r.success
    assert len(r.data["incidents"]) >= 1
