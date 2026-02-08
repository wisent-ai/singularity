"""Tests for CircuitBreakerSkill + AdaptiveCircuitThresholdsSkill integration."""

import pytest
import time
from unittest.mock import MagicMock
from singularity.skills.circuit_breaker import (
    CircuitBreakerSkill,
    CircuitRecord,
    CircuitState,
    wire_adaptive_thresholds,
)
from singularity.skills.adaptive_circuit_thresholds import AdaptiveCircuitThresholdsSkill


@pytest.fixture
def cb():
    skill = CircuitBreakerSkill()
    skill._config["cooldown_seconds"] = 0.01  # Fast tests
    return skill


@pytest.fixture
def adaptive():
    return AdaptiveCircuitThresholdsSkill()


def _record(success=True, cost=0.01):
    return CircuitRecord(timestamp=time.time(), success=success, cost=cost)


class TestSetAdaptiveSource:
    def test_set_adaptive_source(self, cb, adaptive):
        cb.set_adaptive_source(adaptive)
        assert cb._adaptive_source is adaptive

    def test_set_adaptive_source_none(self, cb, adaptive):
        cb.set_adaptive_source(adaptive)
        cb.set_adaptive_source(None)
        assert cb._adaptive_source is None


class TestGetEffectiveConfig:
    def test_no_adaptive_source_returns_global(self, cb):
        cfg = cb._get_effective_config("test_skill")
        assert cfg is cb._config

    def test_no_override_returns_global(self, cb, adaptive):
        cb.set_adaptive_source(adaptive)
        cfg = cb._get_effective_config("untuned_skill")
        assert cfg is cb._config

    def test_override_merges_with_global(self, cb, adaptive):
        cb.set_adaptive_source(adaptive)
        adaptive._overrides["llm_skill"] = {
            "failure_rate_threshold": 0.7,
            "consecutive_failure_threshold": 10,
        }
        cfg = cb._get_effective_config("llm_skill")
        # Override values
        assert cfg["failure_rate_threshold"] == 0.7
        assert cfg["consecutive_failure_threshold"] == 10
        # Global fallback values
        assert cfg["min_window_size"] == cb._config["min_window_size"]
        assert cfg["cooldown_seconds"] == cb._config["cooldown_seconds"]

    def test_override_doesnt_mutate_global(self, cb, adaptive):
        cb.set_adaptive_source(adaptive)
        adaptive._overrides["llm_skill"] = {"failure_rate_threshold": 0.9}
        cb._get_effective_config("llm_skill")
        assert cb._config["failure_rate_threshold"] == 0.5  # Unchanged


class TestAdaptiveEvaluation:
    def test_skill_with_high_threshold_stays_closed(self, cb, adaptive):
        """A skill with 70% adaptive threshold shouldn't open at 60% failure rate."""
        cb.set_adaptive_source(adaptive)
        adaptive._overrides["tolerant_skill"] = {
            "failure_rate_threshold": 0.7,
            "consecutive_failure_threshold": 20,
            "cost_per_success_threshold": 1.0,
        }
        circuit = cb._get_circuit("tolerant_skill")
        # Add 6 failures and 4 successes (60% failure rate)
        for _ in range(6):
            circuit.window.append(_record(success=False))
            circuit.failure_count += 1
        for _ in range(4):
            circuit.window.append(_record(success=True))
            circuit.success_count += 1
        cb._evaluate_circuit(circuit)
        assert circuit.state == CircuitState.CLOSED

    def test_skill_with_low_threshold_opens(self, cb, adaptive):
        """A skill with 10% adaptive threshold should open at 20% failure rate."""
        cb.set_adaptive_source(adaptive)
        adaptive._overrides["strict_skill"] = {
            "failure_rate_threshold": 0.1,
            "consecutive_failure_threshold": 2,
            "cost_per_success_threshold": 0.01,
        }
        circuit = cb._get_circuit("strict_skill")
        for _ in range(2):
            circuit.window.append(_record(success=False))
            circuit.failure_count += 1
            circuit.consecutive_failures += 1
        for _ in range(8):
            circuit.window.append(_record(success=True))
            circuit.success_count += 1
        cb._evaluate_circuit(circuit)
        assert circuit.state == CircuitState.OPEN

    def test_unadapted_skill_uses_global(self, cb, adaptive):
        """Skills without overrides use global thresholds (50%)."""
        cb.set_adaptive_source(adaptive)
        circuit = cb._get_circuit("normal_skill")
        # 40% failure rate - below global 50%
        for _ in range(4):
            circuit.window.append(_record(success=False))
            circuit.failure_count += 1
        for _ in range(6):
            circuit.window.append(_record(success=True))
            circuit.success_count += 1
        cb._evaluate_circuit(circuit)
        assert circuit.state == CircuitState.CLOSED

    def test_adaptive_cooldown_override(self, cb, adaptive):
        """Adaptive cooldown override affects half-open transition timing."""
        cb.set_adaptive_source(adaptive)
        adaptive._overrides["fast_recover"] = {
            "failure_rate_threshold": 0.5,
            "consecutive_failure_threshold": 5,
            "cooldown_seconds": 0.001,  # Very fast cooldown
            "cost_per_success_threshold": 0.1,
        }
        circuit = cb._get_circuit("fast_recover")
        circuit.state = CircuitState.OPEN
        circuit.last_state_change = time.time() - 0.01  # 10ms ago
        cb._evaluate_circuit(circuit)
        assert circuit.state == CircuitState.HALF_OPEN

    def test_adaptive_cost_threshold(self, cb, adaptive):
        """Adaptive cost threshold properly triggers circuit open."""
        cb.set_adaptive_source(adaptive)
        adaptive._overrides["expensive_skill"] = {
            "failure_rate_threshold": 0.9,
            "consecutive_failure_threshold": 50,
            "cost_per_success_threshold": 0.05,
        }
        circuit = cb._get_circuit("expensive_skill")
        # High cost per success but low failure rate
        circuit.window.append(_record(success=True, cost=0.50))
        for _ in range(4):
            circuit.window.append(_record(success=False, cost=0.50))
            circuit.failure_count += 1
        circuit.success_count += 1
        cb._evaluate_circuit(circuit)
        assert circuit.state == CircuitState.OPEN


class TestWireAdaptiveThresholds:
    def test_wire_success(self, cb, adaptive):
        registry = MagicMock()
        registry.get.side_effect = lambda sid: {
            "circuit_breaker": cb,
            "adaptive_circuit_thresholds": adaptive,
        }.get(sid)
        result = wire_adaptive_thresholds(registry)
        assert result is True
        assert cb._adaptive_source is adaptive

    def test_wire_missing_cb(self, adaptive):
        registry = MagicMock()
        registry.get.side_effect = lambda sid: {
            "adaptive_circuit_thresholds": adaptive,
        }.get(sid)
        result = wire_adaptive_thresholds(registry)
        assert result is False

    def test_wire_missing_adaptive(self, cb):
        registry = MagicMock()
        registry.get.side_effect = lambda sid: {
            "circuit_breaker": cb,
        }.get(sid)
        result = wire_adaptive_thresholds(registry)
        assert result is False

    def test_wire_both_missing(self):
        registry = MagicMock()
        registry.get.return_value = None
        result = wire_adaptive_thresholds(registry)
        assert result is False
