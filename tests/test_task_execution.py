"""Tests for task-oriented agent execution."""
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from singularity.autonomous_agent import AutonomousAgent
from singularity.cognition import Decision, Action, TokenUsage


@pytest.fixture
def agent():
    """Create agent with no LLM for testing."""
    with patch.dict("os.environ", {}, clear=False):
        a = AutonomousAgent(
            name="TestAgent",
            ticker="TEST",
            llm_provider="none",
            starting_balance=10.0,
            cycle_interval_seconds=0,
        )
        return a


def test_objective_parameter():
    """Test that objective is stored on the agent."""
    with patch.dict("os.environ", {}, clear=False):
        a = AutonomousAgent(
            name="Test", ticker="T", llm_provider="none",
            objective="Write a poem", max_cycles=5,
        )
        assert a.objective == "Write a poem"
        assert a.max_cycles == 5


def test_build_objective_context(agent):
    """Test objective context string generation."""
    agent.objective = "Find the answer to life"
    agent.max_cycles = 10
    agent.cycle = 3
    ctx = agent._build_objective_context()
    assert "Find the answer to life" in ctx
    assert "CURRENT OBJECTIVE" in ctx
    assert "7/10" in ctx  # remaining/total
    assert "agent:done" in ctx


def test_no_objective_context(agent):
    """Test that no context is generated without objective."""
    assert agent._build_objective_context() == ""


@pytest.mark.asyncio
async def test_max_cycles_stops_agent(agent):
    """Test that max_cycles limits agent execution."""
    agent.max_cycles = 3
    # Mock cognition to return a simple action
    mock_decision = Decision(
        action=Action(tool="wait", params={}),
        reasoning="waiting",
        token_usage=TokenUsage(input_tokens=10, output_tokens=5),
        api_cost_usd=0.0001,
    )
    agent.cognition.think = AsyncMock(return_value=mock_decision)
    await agent.run()
    assert agent.cycle == 3


@pytest.mark.asyncio
async def test_agent_done_stops_execution(agent):
    """Test that agent:done tool stops the agent."""
    agent.objective = "Do something"
    agent.max_cycles = 100
    # First call returns a done action
    done_decision = Decision(
        action=Action(tool="agent:done", params={
            "summary": "Task complete",
            "result": "42",
        }),
        reasoning="Found the answer",
        token_usage=TokenUsage(input_tokens=10, output_tokens=5),
        api_cost_usd=0.0001,
    )
    agent.cognition.think = AsyncMock(return_value=done_decision)
    await agent.run()
    assert agent.cycle == 1
    assert agent._task_result is not None
    assert agent._task_result["summary"] == "Task complete"
    assert agent._task_result["result"] == "42"


@pytest.mark.asyncio
async def test_run_task_returns_result(agent):
    """Test run_task convenience method."""
    done_decision = Decision(
        action=Action(tool="agent:done", params={
            "summary": "Done", "result": "output",
        }),
        reasoning="complete",
        token_usage=TokenUsage(input_tokens=10, output_tokens=5),
        api_cost_usd=0.001,
    )
    agent.cognition.think = AsyncMock(return_value=done_decision)
    result = await agent.run_task("Test objective", max_cycles=10)
    assert result["status"] == "completed"
    assert result["objective"] == "Test objective"
    assert result["result"]["result"] == "output"
    assert result["cycles_used"] == 1


@pytest.mark.asyncio
async def test_run_task_max_cycles_reached(agent):
    """Test run_task when max cycles is reached."""
    wait_decision = Decision(
        action=Action(tool="wait", params={}),
        reasoning="waiting",
        token_usage=TokenUsage(input_tokens=10, output_tokens=5),
        api_cost_usd=0.0001,
    )
    agent.cognition.think = AsyncMock(return_value=wait_decision)
    result = await agent.run_task("Impossible task", max_cycles=2)
    assert result["status"] == "max_cycles_reached"
    assert result["cycles_used"] == 2
