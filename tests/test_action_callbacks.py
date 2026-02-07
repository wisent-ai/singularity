"""Tests for action callback/listener system in AutonomousAgent."""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from singularity.autonomous_agent import AutonomousAgent


@pytest.fixture
def agent():
    """Create a minimal agent for testing."""
    with patch.object(AutonomousAgent, '_init_skills'):
        a = AutonomousAgent(name="Test", ticker="TST", llm_provider="none")
    return a


class TestActionListeners:
    def test_add_listener(self, agent):
        cb = lambda info: None
        agent.add_action_listener(cb)
        assert cb in agent._action_listeners

    def test_add_multiple_listeners(self, agent):
        cb1 = lambda info: None
        cb2 = lambda info: None
        agent.add_action_listener(cb1)
        agent.add_action_listener(cb2)
        assert len(agent._action_listeners) == 2

    def test_remove_listener(self, agent):
        cb = lambda info: None
        agent.add_action_listener(cb)
        assert agent.remove_action_listener(cb) is True
        assert cb not in agent._action_listeners

    def test_remove_nonexistent_listener(self, agent):
        cb = lambda info: None
        assert agent.remove_action_listener(cb) is False

    @pytest.mark.asyncio
    async def test_sync_listener_called(self, agent):
        received = []
        agent.add_action_listener(lambda info: received.append(info))
        action_info = {"cycle": 1, "tool": "test:action", "result": {"status": "success"}}
        await agent._notify_listeners(action_info)
        assert len(received) == 1
        assert received[0]["tool"] == "test:action"

    @pytest.mark.asyncio
    async def test_async_listener_called(self, agent):
        received = []

        async def async_cb(info):
            received.append(info)

        agent.add_action_listener(async_cb)
        action_info = {"cycle": 1, "tool": "test:action"}
        await agent._notify_listeners(action_info)
        assert len(received) == 1

    @pytest.mark.asyncio
    async def test_listener_error_does_not_crash(self, agent):
        """Errors in listeners should be caught, not crash the agent."""
        def bad_listener(info):
            raise ValueError("kaboom")

        received = []
        agent.add_action_listener(bad_listener)
        agent.add_action_listener(lambda info: received.append(info))

        await agent._notify_listeners({"cycle": 1, "tool": "test"})
        # Second listener should still be called
        assert len(received) == 1

    @pytest.mark.asyncio
    async def test_multiple_listeners_all_called(self, agent):
        results = {"a": False, "b": False}

        def listener_a(info):
            results["a"] = True

        def listener_b(info):
            results["b"] = True

        agent.add_action_listener(listener_a)
        agent.add_action_listener(listener_b)
        await agent._notify_listeners({"cycle": 1})
        assert results["a"] is True
        assert results["b"] is True

    def test_listeners_initialized_empty(self, agent):
        assert agent._action_listeners == []

    @pytest.mark.asyncio
    async def test_action_info_fields(self, agent):
        """Verify the action info dict has expected fields."""
        captured = []
        agent.add_action_listener(lambda info: captured.append(info))

        info = {
            "cycle": 5,
            "tool": "shell:bash",
            "params": {"command": "ls"},
            "result": {"status": "success", "data": {"output": "file.txt"}},
            "reasoning": "list files",
            "api_cost_usd": 0.001,
            "tokens": 150,
            "balance": 99.5,
            "timestamp": "2025-01-01T00:00:00",
        }
        await agent._notify_listeners(info)

        assert captured[0]["cycle"] == 5
        assert captured[0]["tool"] == "shell:bash"
        assert captured[0]["balance"] == 99.5
        assert captured[0]["reasoning"] == "list files"
