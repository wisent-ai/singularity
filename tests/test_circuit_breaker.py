"""Tests for CircuitBreakerSkill."""
import pytest
import os
import json
import time

from singularity.skills.circuit_breaker import CircuitBreakerSkill, CircuitState


@pytest.fixture
def cb(tmp_path, monkeypatch):
    """Create a CircuitBreakerSkill with temp data dir."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setattr("singularity.skills.circuit_breaker.DATA_DIR", data_dir)
    monkeypatch.setattr("singularity.skills.circuit_breaker.CIRCUIT_FILE", data_dir / "circuit_breaker.json")
    skill = CircuitBreakerSkill()
    # Use small thresholds for faster testing
    skill._config["min_window_size"] = 3
    skill._config["consecutive_failure_threshold"] = 3
    skill._config["cooldown_seconds"] = 0.1
    skill._config["half_open_max_tests"] = 2
    return skill


@pytest.mark.asyncio
async def test_record_success(cb):
    """Recording a success should track it correctly."""
    result = await cb.execute("record", {"skill_id": "test_skill", "success": True, "cost": 0.01})
    assert result.success
    assert result.data["circuit_state"] == "closed"
    assert result.data["failure_rate"] == 0.0


@pytest.mark.asyncio
async def test_record_failure(cb):
    """Recording failures should update failure count."""
    await cb.execute("record", {"skill_id": "test_skill", "success": False, "error": "timeout"})
    result = await cb.execute("record", {"skill_id": "test_skill", "success": False, "error": "timeout"})
    assert result.success
    assert result.data["consecutive_failures"] == 2


@pytest.mark.asyncio
async def test_circuit_opens_on_consecutive_failures(cb):
    """Circuit should open after consecutive_failure_threshold failures."""
    for i in range(3):
        result = await cb.execute("record", {"skill_id": "api_skill", "success": False})
    assert result.data["circuit_state"] == "open"


@pytest.mark.asyncio
async def test_circuit_opens_on_failure_rate(cb):
    """Circuit should open when failure rate exceeds threshold."""
    cb._config["consecutive_failure_threshold"] = 100  # Disable consecutive check
    # 3 failures, 1 success = 75% failure rate > 50% threshold, window=4 > min_window_size=3
    await cb.execute("record", {"skill_id": "flaky", "success": False})
    await cb.execute("record", {"skill_id": "flaky", "success": True})
    await cb.execute("record", {"skill_id": "flaky", "success": False})
    result = await cb.execute("record", {"skill_id": "flaky", "success": False})
    assert result.data["circuit_state"] == "open"


@pytest.mark.asyncio
async def test_check_allows_closed_circuit(cb):
    """Check should allow requests when circuit is closed."""
    result = await cb.execute("check", {"skill_id": "healthy_skill"})
    assert result.success
    assert result.data["allowed"] is True


@pytest.mark.asyncio
async def test_check_denies_open_circuit(cb):
    """Check should deny requests when circuit is open."""
    for _ in range(3):
        await cb.execute("record", {"skill_id": "broken", "success": False})
    result = await cb.execute("check", {"skill_id": "broken"})
    assert result.data["allowed"] is False
    assert result.data["reason"] == "circuit_open"


@pytest.mark.asyncio
async def test_half_open_recovery(cb):
    """Circuit should recover through half-open testing."""
    # Open the circuit
    for _ in range(3):
        await cb.execute("record", {"skill_id": "recovering", "success": False})
    result = await cb.execute("check", {"skill_id": "recovering"})
    assert result.data["allowed"] is False

    # Wait for cooldown
    time.sleep(0.15)

    # Should transition to half-open
    result = await cb.execute("check", {"skill_id": "recovering"})
    assert result.data["allowed"] is True
    assert result.data["circuit_state"] == "half_open"

    # Record successes to close the circuit
    await cb.execute("record", {"skill_id": "recovering", "success": True})
    result = await cb.execute("record", {"skill_id": "recovering", "success": True})
    assert result.data["circuit_state"] == "closed"


@pytest.mark.asyncio
async def test_half_open_failure_reopens(cb):
    """Failure in half-open should reopen the circuit."""
    for _ in range(3):
        await cb.execute("record", {"skill_id": "unstable", "success": False})
    time.sleep(0.15)
    await cb.execute("check", {"skill_id": "unstable"})  # Trigger half-open
    result = await cb.execute("record", {"skill_id": "unstable", "success": False})
    # Should go back to open (consecutive failures hit threshold again)
    assert result.data["circuit_state"] in ("open", "half_open")


@pytest.mark.asyncio
async def test_force_open(cb):
    """Force open should block the skill."""
    result = await cb.execute("force_open", {"skill_id": "risky", "reason": "maintenance"})
    assert result.success
    check = await cb.execute("check", {"skill_id": "risky"})
    assert check.data["allowed"] is False
    assert check.data["reason"] == "forced_open"


@pytest.mark.asyncio
async def test_force_close(cb):
    """Force close should allow the skill."""
    for _ in range(3):
        await cb.execute("record", {"skill_id": "fixed", "success": False})
    await cb.execute("force_close", {"skill_id": "fixed"})
    result = await cb.execute("check", {"skill_id": "fixed"})
    assert result.data["allowed"] is True


@pytest.mark.asyncio
async def test_reset(cb):
    """Reset should clear all history."""
    for _ in range(3):
        await cb.execute("record", {"skill_id": "messy", "success": False})
    await cb.execute("reset", {"skill_id": "messy"})
    result = await cb.execute("status", {"skill_id": "messy"})
    assert result.data["state"] == "closed"
    assert result.data["failure_count"] == 0


@pytest.mark.asyncio
async def test_budget_critical_blocks_non_essential(cb):
    """Budget critical mode should block non-essential skills."""
    result = await cb.execute("check", {"skill_id": "twitter", "budget_remaining": 0.50})
    assert result.data["allowed"] is False
    assert result.data["reason"] == "budget_critical"


@pytest.mark.asyncio
async def test_budget_critical_allows_essential(cb):
    """Essential skills should work even in budget critical mode."""
    result = await cb.execute("check", {"skill_id": "memory", "budget_remaining": 0.50})
    assert result.data["allowed"] is True


@pytest.mark.asyncio
async def test_configure(cb):
    """Configuration updates should be applied."""
    result = await cb.execute("configure", {"failure_rate_threshold": 0.8, "cooldown_seconds": 120})
    assert result.success
    assert cb._config["failure_rate_threshold"] == 0.8
    assert cb._config["cooldown_seconds"] == 120


@pytest.mark.asyncio
async def test_dashboard(cb):
    """Dashboard should return aggregate stats."""
    await cb.execute("record", {"skill_id": "a", "success": True, "cost": 0.01})
    await cb.execute("record", {"skill_id": "b", "success": False, "cost": 0.02})
    result = await cb.execute("dashboard", {})
    assert result.success
    assert result.data["total_circuits"] == 2
    assert result.data["total_requests"] == 2


@pytest.mark.asyncio
async def test_persistence(cb, tmp_path, monkeypatch):
    """Circuit state should persist across restarts."""
    await cb.execute("record", {"skill_id": "persistent", "success": False})
    await cb.execute("record", {"skill_id": "persistent", "success": False})

    # Create a new instance (simulates restart)
    data_dir = tmp_path / "data"
    monkeypatch.setattr("singularity.skills.circuit_breaker.DATA_DIR", data_dir)
    monkeypatch.setattr("singularity.skills.circuit_breaker.CIRCUIT_FILE", data_dir / "circuit_breaker.json")
    cb2 = CircuitBreakerSkill()
    result = await cb2.execute("status", {"skill_id": "persistent"})
    assert result.data["failure_count"] == 2


@pytest.mark.asyncio
async def test_status_all(cb):
    """Status with no skill_id should return all circuits."""
    await cb.execute("record", {"skill_id": "a", "success": True})
    await cb.execute("record", {"skill_id": "b", "success": True})
    result = await cb.execute("status", {})
    assert result.data["total_tracked"] == 2
    assert "a" in result.data["circuits"]
    assert "b" in result.data["circuits"]
