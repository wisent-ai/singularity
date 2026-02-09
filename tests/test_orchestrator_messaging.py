"""Tests for OrchestratorSkill messaging — _message() and _broadcast() KeyError fix.

The bug: _message() and _broadcast() accessed _message_boxes[agent_id] without
checking if the key existed. When an agent's message box wasn't pre-initialized
(e.g., agent loaded from a previous session, or message box setup failed),
this caused a KeyError crash.

The fix: Guard both methods to create the message box on demand if missing.
"""

import asyncio
import pytest
from unittest.mock import MagicMock
from singularity.skills.orchestrator import (
    OrchestratorSkill,
    _all_living_agents,
    _message_boxes,
    LivingAgent,
    LifeStatus,
)
from datetime import datetime


@pytest.fixture(autouse=True)
def clean_global_state():
    """Clear global agent/message state before each test."""
    _all_living_agents.clear()
    _message_boxes.clear()
    yield
    _all_living_agents.clear()
    _message_boxes.clear()


@pytest.fixture
def skill():
    """Create an OrchestratorSkill with a mock agent."""
    s = OrchestratorSkill()
    mock_agent = MagicMock()
    mock_agent.name = "TestAgent"
    mock_agent.balance = 100.0
    s._my_agent = mock_agent
    s._my_id = "test_agent_001"
    s._agent_factory = lambda **kwargs: MagicMock()
    return s


def _register_agent(agent_id, name, with_message_box=True):
    """Helper to register a living agent in global state."""
    living = LivingAgent(
        id=agent_id,
        name=name,
        purpose="test",
        wallet=10.0,
        status=LifeStatus.ALIVE,
        born_at=datetime.now(),
        creator_id="someone",
    )
    _all_living_agents[agent_id] = living
    if with_message_box:
        _message_boxes[agent_id] = asyncio.Queue()
    return living


class TestMessageWithoutMessageBox:
    """Test _message() when recipient has no pre-initialized message box."""

    @pytest.mark.asyncio
    async def test_message_creates_missing_box(self, skill):
        """Messaging an agent without a message box should create one, not crash."""
        # Register agent WITHOUT a message box
        _register_agent("recipient_001", "Recipient", with_message_box=False)

        result = await skill.execute("message", {
            "to": "Recipient",
            "message": "Hello from test",
        })

        assert result.success is True
        assert "recipient_001" in _message_boxes
        # Message should be in the box
        msg = _message_boxes["recipient_001"].get_nowait()
        assert msg["message"] == "Hello from test"

    @pytest.mark.asyncio
    async def test_message_with_existing_box(self, skill):
        """Messaging an agent with an existing box should work as before."""
        _register_agent("recipient_002", "Bob", with_message_box=True)

        result = await skill.execute("message", {
            "to": "Bob",
            "message": "Hey Bob",
        })

        assert result.success is True
        msg = _message_boxes["recipient_002"].get_nowait()
        assert msg["message"] == "Hey Bob"

    @pytest.mark.asyncio
    async def test_message_to_nonexistent_agent(self, skill):
        """Messaging a non-existent agent should fail gracefully."""
        result = await skill.execute("message", {
            "to": "nobody",
            "message": "Hello?",
        })

        assert result.success is False
        assert "No agent found" in result.message


class TestBroadcastWithoutMessageBox:
    """Test _broadcast() when some agents lack message boxes."""

    @pytest.mark.asyncio
    async def test_broadcast_creates_missing_boxes(self, skill):
        """Broadcast should create missing message boxes, not crash."""
        # One agent with box, one without
        _register_agent("agent_a", "Alice", with_message_box=True)
        _register_agent("agent_b", "Bob", with_message_box=False)

        result = await skill.execute("broadcast", {
            "message": "Hello everyone",
        })

        assert result.success is True
        # Both should have the message
        assert "agent_b" in _message_boxes
        msg_a = _message_boxes["agent_a"].get_nowait()
        msg_b = _message_boxes["agent_b"].get_nowait()
        assert msg_a["message"] == "Hello everyone"
        assert msg_b["message"] == "Hello everyone"

    @pytest.mark.asyncio
    async def test_broadcast_skips_self(self, skill):
        """Broadcast should not send to the broadcasting agent itself."""
        _register_agent(skill._my_id, "TestAgent", with_message_box=True)
        _register_agent("other_001", "Other", with_message_box=True)

        result = await skill.execute("broadcast", {"message": "Hi all"})

        assert result.success is True
        # Self should NOT have the message
        assert _message_boxes[skill._my_id].empty()
        # Other should have it
        msg = _message_boxes["other_001"].get_nowait()
        assert msg["message"] == "Hi all"

    @pytest.mark.asyncio
    async def test_broadcast_skips_dead_agents(self, skill):
        """Broadcast should not send to dead agents."""
        dead = _register_agent("dead_001", "DeadAgent", with_message_box=True)
        dead.status = LifeStatus.DEAD
        _register_agent("alive_001", "AliveAgent", with_message_box=True)

        result = await skill.execute("broadcast", {"message": "Anyone alive?"})

        assert result.success is True
        assert _message_boxes["dead_001"].empty()
        msg = _message_boxes["alive_001"].get_nowait()
        assert msg["message"] == "Anyone alive?"


class TestCheckMessages:
    """Test _check_messages creates box on demand."""

    @pytest.mark.asyncio
    async def test_check_messages_creates_box(self, skill):
        """Checking messages when no box exists should create one."""
        # Don't pre-create a box for the skill's agent
        result = await skill.execute("check_messages", {})

        assert result.success is True
        assert skill._my_id in _message_boxes
        assert result.data["count"] == 0
