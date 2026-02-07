"""Tests for AgentNetworkSkill - service discovery, capability routing, and RPC."""

import json
import pytest
import tempfile
from pathlib import Path
from singularity.skills.agent_network import AgentNetworkSkill, _load_data, _save_data, _default_data, _generate_agent_id


@pytest.fixture
def tmp_data(tmp_path):
    """Provide a temp data file for isolation."""
    return tmp_path / "agent_network.json"


@pytest.fixture
def skill(tmp_data):
    s = AgentNetworkSkill()
    s._data_path = tmp_data
    _save_data(_default_data(), tmp_data)
    return s


@pytest.mark.asyncio
async def test_register_and_topology(skill):
    """Register self and verify topology."""
    r = await skill.execute("register", {
        "name": "Agent-Alpha",
        "endpoint": "http://localhost:8001",
        "capabilities": [
            {"name": "code_review", "skill_id": "revenue_services", "description": "Reviews code"},
            {"name": "summarize", "skill_id": "revenue_services", "description": "Summarizes text"},
        ],
    })
    assert r.success
    assert "Agent-Alpha" in r.message
    agent_id = r.data["agent_id"]

    # Check topology
    t = await skill.execute("get_topology", {})
    assert t.success
    assert t.data["self"]["name"] == "Agent-Alpha"
    assert t.data["self"]["agent_id"] == agent_id


@pytest.mark.asyncio
async def test_add_peer_and_discover(skill):
    """Add peers and discover by capability."""
    await skill.execute("add_peer", {
        "name": "Agent-Beta",
        "endpoint": "/tmp/beta_inbox",
        "capabilities": [{"name": "data_analysis", "skill_id": "revenue_services"}],
    })
    await skill.execute("add_peer", {
        "name": "Agent-Gamma",
        "endpoint": "/tmp/gamma_inbox",
        "capabilities": [{"name": "code_review", "skill_id": "revenue_services"}],
    })

    # Discover by capability
    r = await skill.execute("discover", {"capability": "data_analysis"})
    assert r.success
    assert len(r.data["agents"]) == 1
    assert r.data["agents"][0]["name"] == "Agent-Beta"

    # Discover by name
    r2 = await skill.execute("discover", {"name_pattern": "gamma"})
    assert r2.success
    assert len(r2.data["agents"]) == 1
    assert r2.data["agents"][0]["name"] == "Agent-Gamma"


@pytest.mark.asyncio
async def test_route_best_agent(skill):
    """Route to the best agent for a capability."""
    await skill.execute("add_peer", {
        "name": "Fast-Agent",
        "endpoint": "/tmp/fast",
        "capabilities": [{"name": "code_review", "confidence": 0.9}],
    })
    await skill.execute("add_peer", {
        "name": "Slow-Agent",
        "endpoint": "/tmp/slow",
        "capabilities": [{"name": "code_review", "confidence": 0.3}],
    })

    r = await skill.execute("route", {"capability": "code_review"})
    assert r.success
    assert r.data["total_candidates"] == 2
    assert r.data["best"]["name"] in ("Fast-Agent", "Slow-Agent")


@pytest.mark.asyncio
async def test_file_based_rpc(skill, tmp_path):
    """Test file-based RPC delivery."""
    inbox = tmp_path / "beta_inbox"
    inbox.mkdir()

    await skill.execute("register", {
        "name": "Agent-Alpha",
        "endpoint": str(tmp_path / "alpha_inbox"),
        "capabilities": [],
    })
    await skill.execute("add_peer", {
        "name": "Agent-Beta",
        "endpoint": str(inbox),
        "capabilities": [{"name": "summarize"}],
    })

    beta_id = _generate_agent_id("Agent-Beta", str(inbox))
    r = await skill.execute("rpc_call", {
        "agent_id": beta_id,
        "skill_id": "revenue_services",
        "action": "summarize",
        "params": {"text": "Hello world"},
    })
    assert r.success
    assert r.data["rpc"]["status"] == "sent"

    # Verify file was written to inbox
    rpc_files = list(inbox.glob("rpc_*.json"))
    assert len(rpc_files) == 1


@pytest.mark.asyncio
async def test_rpc_respond(skill, tmp_path):
    """Test responding to a pending RPC."""
    inbox = tmp_path / "alpha_inbox"
    inbox.mkdir()

    await skill.execute("register", {
        "name": "Agent-Alpha",
        "endpoint": str(inbox),
        "capabilities": [],
    })

    # Manually inject a pending RPC
    data = _load_data(skill._data_path)
    data["pending_rpcs"].append({
        "rpc_id": "rpc_test123",
        "from_agent": "agent_unknown",
        "to_agent": data["self"]["agent_id"],
        "skill_id": "test",
        "action": "ping",
        "params": {},
        "status": "pending",
    })
    _save_data(data, skill._data_path)

    r = await skill.execute("rpc_respond", {
        "rpc_id": "rpc_test123",
        "success": True,
        "result": {"pong": True},
    })
    assert r.success

    # Verify pending is cleared
    p = await skill.execute("get_pending_rpcs", {})
    assert len(p.data["pending"]) == 0


@pytest.mark.asyncio
async def test_remove_peer(skill):
    """Test removing a peer."""
    await skill.execute("add_peer", {
        "name": "Temp-Agent",
        "endpoint": "/tmp/temp",
        "capabilities": [],
    })
    agent_id = _generate_agent_id("Temp-Agent", "/tmp/temp")

    r = await skill.execute("remove_peer", {"agent_id": agent_id})
    assert r.success

    # Should not find it anymore
    d = await skill.execute("discover", {"name_pattern": "temp"})
    assert len(d.data["agents"]) == 0


@pytest.mark.asyncio
async def test_deregister(skill):
    """Test deregistration."""
    await skill.execute("register", {"name": "X", "endpoint": "/x", "capabilities": []})
    r = await skill.execute("deregister", {})
    assert r.success

    t = await skill.execute("get_topology", {})
    assert t.data["self"] is None


@pytest.mark.asyncio
async def test_refresh_peers_expires_stale(skill):
    """Test that refresh_peers marks old peers as stale."""
    await skill.execute("add_peer", {
        "name": "Old-Agent",
        "endpoint": "/tmp/old",
        "capabilities": [],
    })
    # Manually set last_seen to long ago
    data = _load_data(skill._data_path)
    agent_id = _generate_agent_id("Old-Agent", "/tmp/old")
    data["peers"][agent_id]["last_seen"] = "2020-01-01T00:00:00Z"
    _save_data(data, skill._data_path)

    r = await skill.execute("refresh_peers", {})
    assert r.success
    assert "Old-Agent" in r.data["expired"]


@pytest.mark.asyncio
async def test_discover_no_match(skill):
    """Discover returns empty when no match."""
    await skill.execute("add_peer", {
        "name": "Agent-X",
        "endpoint": "/tmp/x",
        "capabilities": [{"name": "unrelated"}],
    })
    r = await skill.execute("discover", {"capability": "quantum_computing"})
    assert r.success
    assert len(r.data["agents"]) == 0


@pytest.mark.asyncio
async def test_validation_errors(skill):
    """Test various validation errors."""
    r1 = await skill.execute("register", {"name": "", "endpoint": ""})
    assert not r1.success

    r2 = await skill.execute("rpc_call", {"agent_id": "nonexistent"})
    assert not r2.success

    r3 = await skill.execute("remove_peer", {"agent_id": "nonexistent"})
    assert not r3.success

    r4 = await skill.execute("unknown_action", {})
    assert not r4.success
