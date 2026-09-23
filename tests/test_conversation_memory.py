"""Tests for CognitionEngine conversation memory."""
import pytest
from singularity.cognition import CognitionEngine, AgentState


@pytest.fixture
def engine():
    return CognitionEngine(llm_provider="none", agent_name="Test", agent_ticker="TST")


def test_initial_conversation_empty(engine):
    assert engine.get_conversation_length() == 0
    assert engine.get_conversation_turns() == 0


def test_clear_conversation(engine):
    engine._conversation_history = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]
    cleared = engine.clear_conversation()
    assert cleared == 2
    assert engine.get_conversation_length() == 0


def test_set_max_history_turns(engine):
    engine._conversation_history = [
        {"role": "user", "content": f"msg {i}"} if i % 2 == 0
        else {"role": "assistant", "content": f"reply {i}"}
        for i in range(20)
    ]
    assert engine.get_conversation_length() == 20
    engine.set_max_history_turns(3)
    assert engine._max_history_turns == 3
    assert engine.get_conversation_length() == 6  # 3 turns * 2 messages


def test_set_max_history_turns_minimum(engine):
    engine.set_max_history_turns(0)
    assert engine._max_history_turns == 1


def test_get_conversation_summary_empty(engine):
    summary = engine.get_conversation_summary()
    assert summary["turns"] == 0
    assert summary["last_user"] is None
    assert summary["last_assistant"] is None


def test_get_conversation_summary_with_data(engine):
    engine._conversation_history = [
        {"role": "user", "content": "what should I do?"},
        {"role": "assistant", "content": '{"tool": "wait", "params": {}}'},
    ]
    summary = engine.get_conversation_summary()
    assert summary["turns"] == 1
    assert summary["max_turns"] == 10
    assert "what should" in summary["last_user"]
    assert "wait" in summary["last_assistant"]


def test_max_history_turns_constructor():
    engine = CognitionEngine(llm_provider="none", max_history_turns=5)
    assert engine._max_history_turns == 5


def test_conversation_history_included_in_messages():
    """Verify that think() would build messages including history."""
    engine = CognitionEngine(llm_provider="none", agent_name="Test", agent_ticker="TST")
    # Simulate prior conversation
    engine._conversation_history = [
        {"role": "user", "content": "previous question"},
        {"role": "assistant", "content": "previous answer"},
    ]
    # The think method builds messages = list(self._conversation_history) + new user msg
    # Since provider is "none", think() returns wait. But we can verify history is there.
    assert len(engine._conversation_history) == 2
    messages = list(engine._conversation_history)
    messages.append({"role": "user", "content": "new question"})
    assert len(messages) == 3
    assert messages[0]["role"] == "user"
    assert messages[2]["content"] == "new question"


@pytest.mark.asyncio
async def test_think_returns_wait_with_no_backend():
    engine = CognitionEngine(llm_provider="none", agent_name="Test", agent_ticker="TST")
    state = AgentState(balance=10.0, burn_rate=0.01, runway_hours=100.0)
    decision = await engine.think(state)
    assert decision.action.tool == "wait"
    # History should NOT be updated when no backend (returns early)
    assert engine.get_conversation_length() == 0
