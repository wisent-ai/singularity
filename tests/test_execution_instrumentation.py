"""Tests for ExecutionInstrumentation - metrics, bridge events, and EventBus wiring."""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from singularity.execution_instrumentation import ExecutionInstrumentation


class FakeObservability:
    """Mock ObservabilitySkill that records emitted metrics."""
    def __init__(self):
        self.emitted = []

    def _emit(self, params):
        self.emitted.append(params)
        return MagicMock(success=True)


class FakeBridge:
    """Mock SkillEventBridgeSkill that records bridge calls."""
    def __init__(self):
        self.calls = []

    async def emit_bridge_events(self, skill_id, action, result_data):
        self.calls.append({"skill_id": skill_id, "action": action, "data": result_data})
        return []


class FakeAgent:
    """Minimal agent mock with skills registry and event emission."""
    def __init__(self, observability=None, bridge=None):
        self.skills = MagicMock()
        self.skills.skills = MagicMock()
        self._emitted_events = []

        # Set up skills.values() to return our fakes
        skill_list = []
        if observability:
            skill_list.append(observability)
        if bridge:
            skill_list.append(bridge)
        self.skills.skills.values = MagicMock(return_value=skill_list)

    async def _emit_event(self, topic, data=None, priority=None):
        self._emitted_events.append({"topic": topic, "data": data})


@pytest.fixture
def obs():
    return FakeObservability()

@pytest.fixture
def bridge():
    return FakeBridge()

@pytest.fixture
def agent(obs, bridge):
    return FakeAgent(observability=obs, bridge=bridge)


def test_lazy_init_discovers_skills(agent, obs, bridge):
    """Instrumentation discovers ObservabilitySkill and SkillEventBridgeSkill."""
    instr = ExecutionInstrumentation(agent)
    with patch("singularity.execution_instrumentation.ExecutionInstrumentation._lazy_init") as mock_init:
        # Direct test of lazy init
        pass

    instr2 = ExecutionInstrumentation(agent)
    # Patch isinstance checks
    with patch("singularity.skills.observability.ObservabilitySkill", FakeObservability):
        with patch("singularity.skills.skill_event_bridge.SkillEventBridgeSkill", FakeBridge):
            instr2._lazy_init()
    assert instr2._initialized


@pytest.mark.asyncio
async def test_successful_execution_emits_metrics(agent, obs, bridge):
    """Successful execution emits count, latency, and success metrics."""
    instr = ExecutionInstrumentation(agent)
    instr._initialized = True
    instr._observability = obs
    instr._bridge = bridge

    async def execute_fn():
        return {"status": "success", "data": {"result": 42}, "message": "ok"}

    result = await instr.instrumented_execute("code_review", "review", {}, execute_fn)

    assert result["status"] == "success"
    # Check metrics were emitted
    names = [m["name"] for m in obs.emitted]
    assert "skill.execution.count" in names
    assert "skill.execution.latency_ms" in names
    assert "skill.execution.success" in names
    assert "skill.execution.errors" not in names


@pytest.mark.asyncio
async def test_failed_execution_emits_error_metrics(agent, obs, bridge):
    """Failed execution emits error counter."""
    instr = ExecutionInstrumentation(agent)
    instr._initialized = True
    instr._observability = obs
    instr._bridge = bridge

    async def execute_fn():
        return {"status": "failed", "data": {}, "message": "something broke"}

    result = await instr.instrumented_execute("shell", "run", {}, execute_fn)

    assert result["status"] == "failed"
    names = [m["name"] for m in obs.emitted]
    assert "skill.execution.errors" in names
    assert "skill.execution.count" in names


@pytest.mark.asyncio
async def test_exception_emits_error_and_reraises(agent, obs, bridge):
    """Exceptions still emit metrics and are re-raised."""
    instr = ExecutionInstrumentation(agent)
    instr._initialized = True
    instr._observability = obs
    instr._bridge = bridge

    async def execute_fn():
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        await instr.instrumented_execute("github", "push", {}, execute_fn)

    names = [m["name"] for m in obs.emitted]
    assert "skill.execution.count" in names
    assert "skill.execution.errors" in names


@pytest.mark.asyncio
async def test_bridge_events_called(agent, obs, bridge):
    """Bridge events are emitted after execution."""
    instr = ExecutionInstrumentation(agent)
    instr._initialized = True
    instr._observability = obs
    instr._bridge = bridge

    async def execute_fn():
        return {"status": "success", "data": {"pr_url": "http://example.com"}, "message": "ok"}

    await instr.instrumented_execute("github", "create_pr", {"title": "test"}, execute_fn)

    assert len(bridge.calls) == 1
    assert bridge.calls[0]["skill_id"] == "github"
    assert bridge.calls[0]["action"] == "create_pr"
    assert bridge.calls[0]["data"] == {"pr_url": "http://example.com"}


@pytest.mark.asyncio
async def test_eventbus_event_emitted(agent, obs, bridge):
    """EventBus skill.executed event is published."""
    instr = ExecutionInstrumentation(agent)
    instr._initialized = True
    instr._observability = obs
    instr._bridge = bridge

    async def execute_fn():
        return {"status": "success", "data": {}, "message": "done"}

    await instr.instrumented_execute("memory", "save", {}, execute_fn)

    assert len(agent._emitted_events) == 1
    evt = agent._emitted_events[0]
    assert evt["topic"] == "skill.executed"
    assert evt["data"]["skill"] == "memory"
    assert evt["data"]["action"] == "save"
    assert evt["data"]["success"] is True
    assert "latency_ms" in evt["data"]


@pytest.mark.asyncio
async def test_stats_tracking(agent, obs, bridge):
    """Stats are tracked correctly across executions."""
    instr = ExecutionInstrumentation(agent)
    instr._initialized = True
    instr._observability = obs
    instr._bridge = bridge

    async def success_fn():
        return {"status": "success", "data": {}, "message": "ok"}

    async def fail_fn():
        return {"status": "failed", "data": {}, "message": "err"}

    await instr.instrumented_execute("a", "b", {}, success_fn)
    await instr.instrumented_execute("c", "d", {}, fail_fn)
    await instr.instrumented_execute("e", "f", {}, success_fn)

    stats = instr.get_stats()
    assert stats["total_instrumented"] == 3
    assert stats["total_errors"] == 1
    assert stats["observability_connected"] is True
    assert stats["bridge_connected"] is True
    assert abs(stats["error_rate"] - 1/3) < 0.01


@pytest.mark.asyncio
async def test_works_without_observability(agent, bridge):
    """Instrumentation works gracefully without ObservabilitySkill."""
    agent_no_obs = FakeAgent(bridge=bridge)
    instr = ExecutionInstrumentation(agent_no_obs)
    instr._initialized = True
    instr._observability = None
    instr._bridge = bridge

    async def execute_fn():
        return {"status": "success", "data": {}, "message": "ok"}

    result = await instr.instrumented_execute("test", "run", {}, execute_fn)
    assert result["status"] == "success"
    stats = instr.get_stats()
    assert stats["observability_connected"] is False


@pytest.mark.asyncio
async def test_works_without_bridge(agent, obs):
    """Instrumentation works gracefully without SkillEventBridgeSkill."""
    agent_no_bridge = FakeAgent(observability=obs)
    instr = ExecutionInstrumentation(agent_no_bridge)
    instr._initialized = True
    instr._observability = obs
    instr._bridge = None

    async def execute_fn():
        return {"status": "success", "data": {}, "message": "ok"}

    result = await instr.instrumented_execute("test", "run", {}, execute_fn)
    assert result["status"] == "success"
    stats = instr.get_stats()
    assert stats["bridge_connected"] is False


@pytest.mark.asyncio
async def test_metric_labels_correct(agent, obs, bridge):
    """Emitted metrics have correct skill/action labels."""
    instr = ExecutionInstrumentation(agent)
    instr._initialized = True
    instr._observability = obs
    instr._bridge = bridge

    async def execute_fn():
        return {"status": "success", "data": {}, "message": "ok"}

    await instr.instrumented_execute("code_review", "analyze", {"file": "x.py"}, execute_fn)

    for metric in obs.emitted:
        assert metric["labels"]["skill"] == "code_review"
        assert metric["labels"]["action"] == "analyze"
