"""Tests for resilient run loop features."""
import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from singularity.autonomous_agent import AutonomousAgent
from singularity.cognition import Decision, Action, TokenUsage


@pytest.fixture
def agent():
    """Create a test agent with no LLM."""
    with patch.object(AutonomousAgent, '_init_skills'):
        a = AutonomousAgent(
            name="TestAgent", ticker="TEST", llm_provider="none",
            starting_balance=10.0, cycle_interval_seconds=0.01,
        )
        a.skills = MagicMock()
        a.skills.skills = {}
        return a


def test_max_cycles_param(agent):
    assert agent.max_cycles is None
    with patch.object(AutonomousAgent, '_init_skills'):
        a2 = AutonomousAgent(llm_provider="none", max_cycles=5)
        assert a2.max_cycles == 5


def test_error_tracking_init(agent):
    assert agent.consecutive_errors == 0
    assert agent.total_errors == 0
    assert agent.last_error is None


def test_stop_with_reason(agent):
    agent.stop("test_reason")
    assert agent.running is False
    assert agent._stop_reason == "test_reason"


def test_shutdown_callback_registration(agent):
    cb = MagicMock()
    agent.on_shutdown(cb)
    assert cb in agent._shutdown_callbacks


@pytest.mark.asyncio
async def test_run_max_cycles(agent):
    agent.max_cycles = 3
    mock_decision = Decision(
        action=Action(tool="wait", params={}),
        reasoning="test", token_usage=TokenUsage(), api_cost_usd=0.0,
    )
    agent.cognition = MagicMock()
    agent.cognition.think = AsyncMock(return_value=mock_decision)

    result = await agent.run()
    assert result["reason"] == "max_cycles_reached"
    assert result["cycles"] == 3


@pytest.mark.asyncio
async def test_run_error_recovery(agent):
    """Agent recovers from transient errors."""
    agent.max_cycles = 4
    agent.cycle_interval = 0.001
    call_count = 0

    async def flaky_think(state):
        nonlocal call_count
        call_count += 1
        if call_count <= 2:
            raise RuntimeError("API timeout")
        return Decision(
            action=Action(tool="wait", params={}),
            reasoning="ok", token_usage=TokenUsage(), api_cost_usd=0.0,
        )

    agent.cognition = MagicMock()
    agent.cognition.think = flaky_think

    result = await agent.run()
    assert agent.total_errors == 2
    assert agent.consecutive_errors == 0  # Reset after success
    assert result["cycles"] == 4


@pytest.mark.asyncio
async def test_run_max_consecutive_errors(agent):
    agent.max_consecutive_errors = 3
    agent.cycle_interval = 0.001

    agent.cognition = MagicMock()
    agent.cognition.think = AsyncMock(side_effect=RuntimeError("always fail"))

    result = await agent.run()
    assert result["reason"] == "max_errors_reached"
    assert agent.total_errors == 3


@pytest.mark.asyncio
async def test_shutdown_callbacks_called(agent):
    agent.max_cycles = 1
    mock_decision = Decision(
        action=Action(tool="wait", params={}),
        reasoning="test", token_usage=TokenUsage(), api_cost_usd=0.0,
    )
    agent.cognition = MagicMock()
    agent.cognition.think = AsyncMock(return_value=mock_decision)

    cb = MagicMock()
    agent.on_shutdown(cb)

    await agent.run()
    cb.assert_called_once()
    info = cb.call_args[0][0]
    assert "reason" in info
    assert "cycles" in info


@pytest.mark.asyncio
async def test_run_once(agent):
    mock_decision = Decision(
        action=Action(tool="wait", params={}),
        reasoning="test", token_usage=TokenUsage(), api_cost_usd=0.0,
    )
    agent.cognition = MagicMock()
    agent.cognition.think = AsyncMock(return_value=mock_decision)

    result = await agent.run_once()
    assert result["cycle"] == 1
    assert result["tool"] == "wait"
    assert "balance" in result


@pytest.mark.asyncio
async def test_run_returns_summary(agent):
    agent.max_cycles = 1
    mock_decision = Decision(
        action=Action(tool="wait", params={}),
        reasoning="test", token_usage=TokenUsage(), api_cost_usd=0.0,
    )
    agent.cognition = MagicMock()
    agent.cognition.think = AsyncMock(return_value=mock_decision)

    result = await agent.run()
    assert "reason" in result
    assert "cycles" in result
    assert "balance" in result
    assert "total_api_cost" in result
    assert "runtime_hours" in result
