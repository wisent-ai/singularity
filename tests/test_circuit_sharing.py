"""Tests for CrossAgentCircuitSharingSkill."""
import pytest
import json
import time

from singularity.skills.circuit_breaker import CircuitBreakerSkill, CircuitState
from singularity.skills.circuit_sharing import CrossAgentCircuitSharingSkill
from singularity.skills.base import SkillRegistry, SkillResult


@pytest.fixture
def setup(tmp_path, monkeypatch):
    """Create a circuit sharing skill with circuit breaker, temp dirs."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setattr("singularity.skills.circuit_breaker.DATA_DIR", data_dir)
    monkeypatch.setattr("singularity.skills.circuit_breaker.CIRCUIT_FILE", data_dir / "cb.json")
    monkeypatch.setattr("singularity.skills.circuit_sharing.DATA_DIR", data_dir)
    monkeypatch.setattr("singularity.skills.circuit_sharing.SHARED_CIRCUIT_FILE", data_dir / "shared.json")
    monkeypatch.setattr("singularity.skills.circuit_sharing.SYNC_HISTORY_FILE", data_dir / "history.json")

    registry = SkillRegistry()
    cb = CircuitBreakerSkill()
    cb._config["min_window_size"] = 3
    cb._config["consecutive_failure_threshold"] = 3
    cb._config["cooldown_seconds"] = 0.1
    registry.skills["circuit_breaker"] = cb

    cs = CrossAgentCircuitSharingSkill()
    cs._config["agent_id"] = "agent_alpha"
    cs._config["shared_store_path"] = str(data_dir / "shared.json")
    cs._config["min_peer_window_size"] = 1
    registry.skills["circuit_sharing"] = cs

    ctx = registry.create_context(agent_name="agent_alpha")
    return cb, cs, data_dir


@pytest.mark.asyncio
async def test_export_empty(setup):
    cb, cs, _ = setup
    result = await cs.execute("export", {})
    assert result.success
    assert result.data["circuit_count"] == 0


@pytest.mark.asyncio
async def test_export_with_circuits(setup):
    cb, cs, _ = setup
    await cb.execute("record", {"skill_id": "email", "success": True, "cost": 0.01})
    await cb.execute("record", {"skill_id": "github", "success": False})
    result = await cs.execute("export", {})
    assert result.success
    assert result.data["circuit_count"] == 2
    assert "email" in result.data["circuits"]
    assert "github" in result.data["circuits"]


@pytest.mark.asyncio
async def test_import_adopts_open_circuit_pessimistic(setup):
    cb, cs, _ = setup
    # Create local closed circuit with some data
    for _ in range(3):
        await cb.execute("record", {"skill_id": "api_skill", "success": True})
    # Import open state from peer
    remote = {"api_skill": {"state": "open", "window_size": 5, "failure_rate": 0.8, "consecutive_successes": 0, "last_state_change": None}}
    result = await cs.execute("import_states", {"agent_id": "agent_beta", "circuits": remote})
    assert result.success
    assert result.data["adopted"] == 1
    # Verify local circuit is now open
    assert cb._circuits["api_skill"].state == CircuitState.OPEN


@pytest.mark.asyncio
async def test_import_skips_forced_circuits(setup):
    cb, cs, _ = setup
    await cb.execute("record", {"skill_id": "locked", "success": True})
    await cb.execute("force_open", {"skill_id": "locked"})
    remote = {"locked": {"state": "closed", "window_size": 10, "consecutive_successes": 5, "last_state_change": None}}
    result = await cs.execute("import_states", {"agent_id": "agent_beta", "circuits": remote})
    assert result.data["adopted"] == 0
    assert cb._circuits["locked"].state == CircuitState.FORCED_OPEN


@pytest.mark.asyncio
async def test_import_skips_insufficient_data(setup):
    cb, cs, _ = setup
    cs._config["min_peer_window_size"] = 5
    await cb.execute("record", {"skill_id": "s1", "success": True})
    remote = {"s1": {"state": "open", "window_size": 2}}
    result = await cs.execute("import_states", {"agent_id": "peer", "circuits": remote})
    assert result.data["adopted"] == 0


@pytest.mark.asyncio
async def test_publish_and_pull(setup):
    cb, cs, data_dir = setup
    # Agent alpha publishes
    for _ in range(3):
        await cb.execute("record", {"skill_id": "email", "success": False})
    await cs.execute("publish", {})
    # Simulate agent beta publishing to same store
    store_path = data_dir / "shared.json"
    store = json.loads(store_path.read_text())
    store["agents"]["agent_beta"] = {
        "circuits": {"payment": {"state": "open", "window_size": 10, "failure_rate": 0.9, "consecutive_successes": 0, "last_state_change": None}},
        "timestamp": "2026-01-01T00:00:00",
        "circuit_count": 1,
    }
    store_path.write_text(json.dumps(store))
    # Agent alpha pulls
    result = await cs.execute("pull", {})
    assert result.success
    assert result.data["peers_found"] == 1
    assert result.data["total_adopted"] == 1
    assert cb._circuits["payment"].state == CircuitState.OPEN


@pytest.mark.asyncio
async def test_sync_bidirectional(setup):
    cb, cs, data_dir = setup
    await cb.execute("record", {"skill_id": "s1", "success": True})
    result = await cs.execute("sync", {})
    assert result.success
    assert "Publish" in result.message
    store = json.loads((data_dir / "shared.json").read_text())
    assert "agent_alpha" in store["agents"]


@pytest.mark.asyncio
async def test_optimistic_strategy(setup):
    cb, cs, _ = setup
    cs._config["merge_strategy"] = "optimistic"
    # Local is closed with no failures - optimistic won't adopt
    await cb.execute("record", {"skill_id": "s1", "success": True})
    await cb.execute("record", {"skill_id": "s1", "success": True})
    await cb.execute("record", {"skill_id": "s1", "success": True})
    remote = {"s1": {"state": "open", "window_size": 5, "last_state_change": None}}
    result = await cs.execute("import_states", {"agent_id": "peer", "circuits": remote})
    assert result.data["adopted"] == 0  # Optimistic: local OK, don't adopt


@pytest.mark.asyncio
async def test_newest_strategy(setup):
    cb, cs, _ = setup
    cs._config["merge_strategy"] = "newest"
    await cb.execute("record", {"skill_id": "s1", "success": True})
    cb._circuits["s1"].last_state_change = 1000.0  # old
    remote = {"s1": {"state": "open", "window_size": 5, "last_state_change": "2026-06-01T00:00:00"}}
    result = await cs.execute("import_states", {"agent_id": "peer", "circuits": remote})
    assert result.data["adopted"] == 1


@pytest.mark.asyncio
async def test_configure(setup):
    _, cs, _ = setup
    result = await cs.execute("configure", {"merge_strategy": "optimistic", "agent_id": "new_id"})
    assert result.success
    assert cs._config["merge_strategy"] == "optimistic"
    assert cs._config["agent_id"] == "new_id"


@pytest.mark.asyncio
async def test_configure_invalid_strategy(setup):
    _, cs, _ = setup
    result = await cs.execute("configure", {"merge_strategy": "invalid"})
    assert not result.success


@pytest.mark.asyncio
async def test_status(setup):
    _, cs, _ = setup
    result = await cs.execute("status", {})
    assert result.success
    assert "agent_alpha" in result.message
    assert result.data["merge_strategy"] == "pessimistic"


@pytest.mark.asyncio
async def test_history(setup):
    cb, cs, _ = setup
    await cb.execute("record", {"skill_id": "s1", "success": True})
    await cs.execute("export", {})
    await cs.execute("publish", {})
    result = await cs.execute("history", {"limit": 5})
    assert result.success
    assert len(result.data["history"]) >= 2


@pytest.mark.asyncio
async def test_pull_empty_store(setup):
    _, cs, _ = setup
    result = await cs.execute("pull", {})
    assert result.success
    assert result.data["peers_found"] == 0


@pytest.mark.asyncio
async def test_pessimistic_recovery_adoption(setup):
    """Pessimistic strategy adopts CLOSED if peer shows strong recovery."""
    cb, cs, _ = setup
    # Local circuit is open
    for _ in range(5):
        await cb.execute("record", {"skill_id": "s1", "success": False})
    assert cb._circuits["s1"].state == CircuitState.OPEN
    # Peer reports closed with consecutive successes
    remote = {"s1": {"state": "closed", "window_size": 10, "consecutive_successes": 5, "last_state_change": None}}
    result = await cs.execute("import_states", {"agent_id": "peer", "circuits": remote})
    assert result.data["adopted"] == 1
    assert cb._circuits["s1"].state == CircuitState.CLOSED
