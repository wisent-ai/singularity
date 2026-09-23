"""Tests for execution statistics tracking in AutonomousAgent."""
import pytest
from singularity.autonomous_agent import AutonomousAgent
from singularity.cognition import AgentState


@pytest.fixture
def agent():
    a = AutonomousAgent(name="TestAgent", ticker="TEST", llm_provider="none", starting_balance=10.0)
    return a


class TestRecordExecution:
    def test_records_success(self, agent):
        agent._record_execution("filesystem", True, 50.0)
        stats = agent._exec_stats["filesystem"]
        assert stats["successes"] == 1
        assert stats["failures"] == 0
        assert stats["total_time_ms"] == 50.0
        assert stats["consecutive_failures"] == 0

    def test_records_failure(self, agent):
        agent._record_execution("shell", False, 100.0)
        stats = agent._exec_stats["shell"]
        assert stats["successes"] == 0
        assert stats["failures"] == 1
        assert stats["consecutive_failures"] == 1

    def test_consecutive_failures_reset_on_success(self, agent):
        agent._record_execution("shell", False, 10.0)
        agent._record_execution("shell", False, 10.0)
        assert agent._exec_stats["shell"]["consecutive_failures"] == 2
        agent._record_execution("shell", True, 10.0)
        assert agent._exec_stats["shell"]["consecutive_failures"] == 0

    def test_multiple_skills_tracked_independently(self, agent):
        agent._record_execution("fs", True, 10.0)
        agent._record_execution("shell", False, 20.0)
        assert agent._exec_stats["fs"]["successes"] == 1
        assert agent._exec_stats["shell"]["failures"] == 1


class TestGetExecutionStats:
    def test_empty_when_no_executions(self, agent):
        assert agent._get_execution_stats() == {}

    def test_returns_skill_stats(self, agent):
        agent._record_execution("fs", True, 50.0)
        agent._record_execution("fs", True, 30.0)
        agent._record_execution("fs", False, 20.0)
        result = agent._get_execution_stats()
        ss = result["skill_stats"]["fs"]
        assert ss["successes"] == 2
        assert ss["failures"] == 1
        assert ss["total"] == 3
        assert abs(ss["success_rate"] - 2/3) < 0.01

    def test_warns_on_consecutive_failures(self, agent):
        for _ in range(3):
            agent._record_execution("bad_skill", False, 10.0)
        result = agent._get_execution_stats()
        assert len(result["warnings"]) >= 1
        assert "bad_skill" in result["warnings"][0]

    def test_warns_on_low_success_rate(self, agent):
        agent._record_execution("flaky", True, 10.0)
        agent._record_execution("flaky", False, 10.0)
        agent._record_execution("flaky", False, 10.0)
        agent._exec_stats["flaky"]["consecutive_failures"] = 0
        result = agent._get_execution_stats()
        assert any("low success rate" in w for w in result["warnings"])


class TestAgentStateExecutionStats:
    def test_agent_state_has_execution_stats_field(self):
        state = AgentState(balance=10.0, burn_rate=0.01, runway_hours=100.0)
        assert state.execution_stats == {}

    def test_agent_state_with_stats(self):
        stats = {"skill_stats": {"fs": {"total": 5}}, "warnings": []}
        state = AgentState(balance=10.0, burn_rate=0.01, runway_hours=100.0, execution_stats=stats)
        assert state.execution_stats["skill_stats"]["fs"]["total"] == 5
