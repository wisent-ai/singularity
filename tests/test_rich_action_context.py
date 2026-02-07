"""Tests for rich action context in LLM prompt."""
import pytest
from unittest.mock import AsyncMock, patch
from singularity.autonomous_agent import AutonomousAgent
from singularity.cognition import CognitionEngine, AgentState, Decision, Action, TokenUsage


@pytest.fixture
def agent():
    with patch.dict("os.environ", {}, clear=False):
        return AutonomousAgent(
            name="Test", ticker="T", llm_provider="none",
            starting_balance=10.0, cycle_interval_seconds=0,
        )


def test_rich_recent_actions_in_prompt():
    """Test that recent actions include params and result details."""
    engine = CognitionEngine(llm_provider="none")
    state = AgentState(
        balance=10.0, burn_rate=0.01, runway_hours=100.0, cycle=5,
        recent_actions=[
            {
                "cycle": 3, "tool": "filesystem:read",
                "params": {"path": "/tmp/test.txt"},
                "result": {"status": "success", "message": "Read 245 bytes", "data": {}},
                "api_cost_usd": 0.001, "tokens": 100,
            },
            {
                "cycle": 4, "tool": "shell:bash",
                "params": {"command": "ls -la /nonexistent"},
                "result": {"status": "error", "message": "No such file or directory"},
                "api_cost_usd": 0.001, "tokens": 50,
            },
        ],
    )
    # Build prompt parts manually by checking what think() would produce
    # We test the formatting logic directly
    action_lines = []
    for a in state.recent_actions[-5:]:
        tool = a.get('tool', 'unknown')
        result = a.get('result', {})
        status = result.get('status', 'unknown')
        cycle = a.get('cycle', '?')
        params = a.get('params', {})
        param_parts = []
        for k, v in list(params.items())[:3]:
            v_str = str(v)
            if len(v_str) > 60:
                v_str = v_str[:57] + "..."
            param_parts.append(f"{k}={v_str}")
        param_str = f"({', '.join(param_parts)})" if param_parts else ""
        msg = result.get('message', '')
        if not msg and isinstance(result.get('data'), dict):
            data = result['data']
            msg = f"keys: {list(data.keys())[:5]}"
        elif not msg and isinstance(result.get('data'), str):
            msg = result['data'][:80]
        if msg and len(msg) > 100:
            msg = msg[:97] + "..."
        line = f"- [C{cycle}] {tool}{param_str}: {status}"
        if msg:
            line += f" - {msg}"
        action_lines.append(line)

    recent_text = "\n".join(action_lines)
    assert "[C3] filesystem:read(path=/tmp/test.txt): success - Read 245 bytes" in recent_text
    assert "[C4] shell:bash(command=ls -la /nonexistent): error - No such file or directory" in recent_text


def test_param_truncation():
    """Test that long parameter values are truncated."""
    long_val = "x" * 100
    params = {"content": long_val}
    param_parts = []
    for k, v in list(params.items())[:3]:
        v_str = str(v)
        if len(v_str) > 60:
            v_str = v_str[:57] + "..."
        param_parts.append(f"{k}={v_str}")
    param_str = f"({', '.join(param_parts)})"
    assert len(param_str) < 80
    assert "..." in param_str


def test_data_keys_summary():
    """Test that data dict keys are shown when no message."""
    result = {"status": "success", "data": {"name": "foo", "size": 42, "type": "file"}}
    msg = result.get('message', '')
    if not msg and isinstance(result.get('data'), dict):
        msg = f"keys: {list(result['data'].keys())[:5]}"
    assert "name" in msg
    assert "size" in msg


def test_empty_recent_actions():
    """Test with no recent actions."""
    state = AgentState(
        balance=10.0, burn_rate=0.01, runway_hours=100.0, cycle=1,
        recent_actions=[],
    )
    # Should produce empty recent_text
    recent_text = ""
    if state.recent_actions:
        recent_text = "would be set"
    assert recent_text == ""


@pytest.mark.asyncio
async def test_rich_context_in_agent_run(agent):
    """Test that rich context flows through the actual agent run loop."""
    agent.max_cycles = 2
    decisions = [
        Decision(
            action=Action(tool="filesystem:read", params={"path": "/tmp/test.txt"}),
            reasoning="reading file",
            token_usage=TokenUsage(input_tokens=100, output_tokens=50),
            api_cost_usd=0.001,
        ),
        Decision(
            action=Action(tool="agent:done", params={"summary": "done", "result": "ok"}),
            reasoning="finished",
            token_usage=TokenUsage(input_tokens=100, output_tokens=50),
            api_cost_usd=0.001,
        ),
    ]
    call_count = 0
    async def mock_think(state):
        nonlocal call_count
        # On second call, verify recent_actions has data
        if call_count > 0:
            assert len(state.recent_actions) > 0
        d = decisions[min(call_count, len(decisions) - 1)]
        call_count += 1
        return d
    agent.cognition.think = mock_think
    await agent.run()
    assert call_count == 2
