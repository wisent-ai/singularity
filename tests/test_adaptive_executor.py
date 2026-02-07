"""Tests for AdaptiveExecutor - the 'adapt' step in act→measure→adapt."""

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from pathlib import Path
from singularity.adaptive_executor import (
    AdaptiveExecutor, CircuitState, ExecutionAdvice, ADAPTIVE_FILE,
)


@pytest.fixture
def executor(tmp_path, monkeypatch):
    """Create an AdaptiveExecutor with temp storage."""
    test_file = tmp_path / "adaptive.json"
    monkeypatch.setattr("singularity.adaptive_executor.ADAPTIVE_FILE", test_file)
    return AdaptiveExecutor(balance=100.0)


@pytest.fixture
def executor_with_perf(executor):
    """Executor loaded with mock performance data."""
    executor.load_performance_data({
        "skill_stats": {
            "github": {
                "total": 100, "successes": 95,
                "avg_cost": 0.01, "avg_latency_ms": 200,
                "actions": {
                    "create_issue": {"total": 50, "successes": 48},
                    "star_repo": {"total": 30, "successes": 10},
                }
            },
            "shell": {
                "total": 50, "successes": 15,
                "avg_cost": 0.005, "avg_latency_ms": 500,
                "actions": {
                    "run": {"total": 50, "successes": 15},
                }
            },
            "content": {
                "total": 80, "successes": 72,
                "avg_cost": 0.05, "avg_latency_ms": 3000,
                "actions": {},
            },
        }
    })
    return executor


class TestCircuitBreaker:
    def test_circuit_starts_closed(self, executor):
        advice = executor.get_advice("github", "create_issue")
        assert advice.should_execute is True
        assert advice.circuit_state == "closed"

    def test_circuit_opens_after_failures(self, executor):
        for i in range(5):
            executor.record_outcome("failing_skill", "act", success=False, error="boom")
        advice = executor.get_advice("failing_skill", "act")
        assert advice.should_execute is False
        assert advice.circuit_state == "open"
        assert "Circuit OPEN" in advice.reason

    def test_circuit_half_open_after_cooldown(self, executor):
        for i in range(5):
            executor.record_outcome("test_skill", "act", success=False, error="err")
        circuit = executor._get_circuit("test_skill")
        circuit.cooldown_seconds = 0  # Immediate cooldown for test
        advice = executor.get_advice("test_skill", "act")
        assert advice.circuit_state == "half_open"
        assert advice.should_execute is True

    def test_circuit_closes_after_half_open_successes(self, executor):
        for i in range(5):
            executor.record_outcome("test_skill", "act", success=False, error="err")
        circuit = executor._get_circuit("test_skill")
        circuit.state = "half_open"
        executor.record_outcome("test_skill", "act", success=True)
        executor.record_outcome("test_skill", "act", success=True)
        assert circuit.state == "closed"

    def test_circuit_reopens_on_half_open_failure(self, executor):
        circuit = executor._get_circuit("test_skill")
        circuit.state = "half_open"
        circuit.cooldown_seconds = 300
        executor.record_outcome("test_skill", "act", success=False, error="still broken")
        assert circuit.state == "open"
        assert circuit.cooldown_seconds == 600  # Doubled

    def test_manual_reset(self, executor):
        for i in range(5):
            executor.record_outcome("bad_skill", "act", success=False, error="err")
        assert executor._get_circuit("bad_skill").state == "open"
        result = executor.reset_circuit("bad_skill")
        assert result is True
        assert executor._get_circuit("bad_skill").state == "closed"


class TestSmartRetry:
    def test_high_success_rate_gets_few_retries(self, executor_with_perf):
        advice = executor_with_perf.get_advice("github", "create_issue")
        assert advice.retry_config["max_retries"] <= 1

    def test_low_success_rate_gets_more_retries(self, executor_with_perf):
        advice = executor_with_perf.get_advice("shell", "run")
        assert advice.retry_config["max_retries"] >= 1

    @pytest.mark.asyncio
    async def test_execute_with_retry_success(self, executor):
        call_count = 0
        async def mock_execute(skill, action, params):
            nonlocal call_count
            call_count += 1
            return {"status": "success", "data": {"result": "ok"}}
        result = await executor.execute_with_retry(mock_execute, "github", "list", {})
        assert result["status"] == "success"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_execute_with_retry_eventual_success(self, executor):
        call_count = 0
        async def mock_execute(skill, action, params):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                return {"status": "failed", "message": "transient error"}
            return {"status": "success", "data": {}}

        advice = ExecutionAdvice(
            retry_config={"max_retries": 3, "base_delay_ms": 10, "max_delay_ms": 50}
        )
        result = await executor.execute_with_retry(
            mock_execute, "github", "list", {}, advice=advice
        )
        assert result["status"] == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_blocked_execution(self, executor):
        for i in range(5):
            executor.record_outcome("broken", "act", success=False, error="err")
        async def mock_execute(skill, action, params):
            return {"status": "success"}
        result = await executor.execute_with_retry(mock_execute, "broken", "act", {})
        assert result["status"] == "blocked"


class TestSkillRanking:
    def test_rank_by_effectiveness(self, executor_with_perf):
        rankings = executor_with_perf.rank_skills(["github", "shell", "content"])
        assert len(rankings) == 3
        skill_ids = [r[0] for r in rankings]
        assert skill_ids[0] == "github"  # Highest success rate
        assert rankings[0][1] > rankings[-1][1]

    def test_open_circuit_gets_zero_score(self, executor_with_perf):
        for i in range(5):
            executor_with_perf.record_outcome("shell", "run", success=False, error="err")
        rankings = executor_with_perf.rank_skills(["github", "shell"])
        shell_score = next(s for sid, s in rankings if sid == "shell")
        assert shell_score == 0.0


class TestCostGuard:
    def test_high_cost_ratio_warning(self, executor):
        executor.update_balance(1.0)
        advice = executor.get_advice("expensive", "op", estimated_cost=0.5)
        assert advice.cost_warning is not None
        assert "50.0%" in advice.cost_warning

    def test_low_success_cost_warning(self, executor_with_perf):
        # shell:run has 15/50 = 30% success rate which is at the threshold
        # We need a skill below 30%, so add one
        executor_with_perf.load_performance_data({
            "skill_stats": {
                "flaky": {
                    "total": 100, "successes": 20,
                    "avg_cost": 0.05, "avg_latency_ms": 1000,
                    "actions": {"call": {"total": 100, "successes": 20}},
                }
            }
        })
        advice = executor_with_perf.get_advice("flaky", "call", estimated_cost=0.01)
        assert advice.cost_warning is not None


class TestPersistence:
    def test_state_persists(self, tmp_path, monkeypatch):
        test_file = tmp_path / "adaptive.json"
        monkeypatch.setattr("singularity.adaptive_executor.ADAPTIVE_FILE", test_file)
        ex1 = AdaptiveExecutor(balance=50.0)
        for i in range(5):
            ex1.record_outcome("flaky", "op", success=False, error="err")
        assert test_file.exists()
        ex2 = AdaptiveExecutor(balance=50.0)
        circuit = ex2._get_circuit("flaky")
        assert circuit.state == "open"
        assert circuit.failure_count == 5

    def test_journal_persists(self, tmp_path, monkeypatch):
        test_file = tmp_path / "adaptive.json"
        monkeypatch.setattr("singularity.adaptive_executor.ADAPTIVE_FILE", test_file)
        ex1 = AdaptiveExecutor()
        for i in range(5):
            ex1.record_outcome("test", "op", success=False, error="err")
        ex2 = AdaptiveExecutor()
        journal = ex2.get_journal()
        assert len(journal) > 0


class TestStatus:
    def test_get_status(self, executor):
        executor.record_outcome("a", "x", success=False, error="e")
        status = executor.get_status()
        assert "circuits" in status
        assert "open_circuits" in status
        assert "journal_entries" in status
        assert status["balance"] == 100.0
