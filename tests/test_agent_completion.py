"""Tests for agent completion signaling and objective-driven execution."""
import asyncio
import pytest
from singularity.autonomous_agent import AutonomousAgent, AgentResult
from singularity.cognition import Action


def test_agent_result_dataclass():
    r = AgentResult(status="completed", message="done", cycles_run=5, total_cost=0.01)
    assert r.success is True
    assert r.cycles_run == 5
    assert "completed" in r.summary()


def test_agent_result_failed():
    r = AgentResult(status="failed", message="could not do it")
    assert r.success is False


def test_agent_result_summary():
    r = AgentResult(status="max_cycles", cycles_run=100, total_cost=1.5, total_tokens=5000)
    s = r.summary()
    assert "max_cycles" in s
    assert "100" in s


@pytest.mark.asyncio
async def test_done_action():
    agent = AutonomousAgent(name="Test", llm_provider="none")
    agent._completion_result = None
    action = Action(tool="done", params={"message": "task finished"})
    result = await agent._execute(action)
    assert result["status"] == "success"
    assert agent._completion_result is not None
    assert agent._completion_result.status == "completed"
    assert agent._completion_result.message == "task finished"
    assert agent.running is False


@pytest.mark.asyncio
async def test_fail_action():
    agent = AutonomousAgent(name="Test", llm_provider="none")
    agent._completion_result = None
    action = Action(tool="fail", params={"message": "impossible"})
    result = await agent._execute(action)
    assert result["status"] == "failed"
    assert agent._completion_result is not None
    assert agent._completion_result.status == "failed"
    assert agent.running is False


@pytest.mark.asyncio
async def test_wait_action():
    agent = AutonomousAgent(name="Test", llm_provider="none")
    action = Action(tool="wait", params={})
    result = await agent._execute(action)
    assert result["status"] == "waited"


@pytest.mark.asyncio
async def test_done_with_outputs():
    agent = AutonomousAgent(name="Test", llm_provider="none")
    agent._completion_result = None
    action = Action(tool="done", params={"message": "built it", "file": "out.txt", "lines": 42})
    await agent._execute(action)
    assert agent._completion_result.outputs == {"file": "out.txt", "lines": 42}
