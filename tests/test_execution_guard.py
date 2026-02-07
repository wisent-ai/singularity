"""Tests for ExecutionGuard."""
import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock
from singularity.execution_guard import ExecutionGuard, ExecutionResult, _classify_error_from_message


def _make_skill(skill_id="test_skill", success=True, data=None, message="ok", delay=0):
    """Create a mock skill."""
    skill = MagicMock()
    skill.manifest.skill_id = skill_id

    async def execute(action_name, params):
        if delay:
            await asyncio.sleep(delay)
        result = MagicMock()
        result.success = success
        result.data = data or {"result": "done"}
        result.message = message
        return result

    skill.execute = execute
    return skill


@pytest.mark.asyncio
async def test_successful_execution():
    guard = ExecutionGuard(default_timeout=5.0)
    skill = _make_skill(data={"key": "value"})
    result = await guard.execute(skill, "do_thing", {"x": 1})
    assert result.success is True
    assert result.data == {"key": "value"}
    assert result.execution_time_ms > 0


@pytest.mark.asyncio
async def test_timeout_enforcement():
    guard = ExecutionGuard(default_timeout=0.1)
    skill = _make_skill(delay=1.0)
    result = await guard.execute(skill, "slow_action", {})
    assert result.success is False
    assert result.error_type == "timeout"
    assert "timed out" in result.message


@pytest.mark.asyncio
async def test_error_handling():
    skill = MagicMock()
    skill.manifest.skill_id = "buggy"
    skill.execute = AsyncMock(side_effect=ValueError("bad input"))
    guard = ExecutionGuard()
    result = await guard.execute(skill, "crash", {})
    assert result.success is False
    assert result.error_type == "permanent"
    assert "ValueError" in result.message


@pytest.mark.asyncio
async def test_transient_error():
    skill = MagicMock()
    skill.manifest.skill_id = "net"
    skill.execute = AsyncMock(side_effect=ConnectionError("refused"))
    guard = ExecutionGuard()
    result = await guard.execute(skill, "fetch", {})
    assert result.success is False
    assert result.error_type == "transient"


@pytest.mark.asyncio
async def test_stats_tracking():
    guard = ExecutionGuard()
    skill = _make_skill(skill_id="tracker")
    await guard.execute(skill, "a", {})
    await guard.execute(skill, "b", {})
    stats = guard.get_stats("tracker")
    assert stats["total_calls"] == 2
    assert stats["successes"] == 2
    assert stats["success_rate"] == 1.0


@pytest.mark.asyncio
async def test_output_truncation():
    guard = ExecutionGuard(max_output_size=100)
    skill = _make_skill(data="x" * 500)
    result = await guard.execute(skill, "big", {})
    assert result.success is True
    assert result.truncated is True
    assert len(result.data) < 500


@pytest.mark.asyncio
async def test_per_skill_timeout():
    guard = ExecutionGuard(default_timeout=10.0, skill_timeouts={"fast": 0.05})
    skill = _make_skill(skill_id="fast", delay=1.0)
    result = await guard.execute(skill, "go", {})
    assert result.error_type == "timeout"


def test_classify_error():
    assert _classify_error_from_message("connection refused") == "transient"
    assert _classify_error_from_message("permission denied") == "permanent"
    assert _classify_error_from_message("timed out") == "timeout"
    assert _classify_error_from_message("something weird") == "unknown"


def test_summary_empty():
    guard = ExecutionGuard()
    assert "No executions" in guard.summary()


@pytest.mark.asyncio
async def test_to_dict():
    guard = ExecutionGuard()
    skill = _make_skill()
    result = await guard.execute(skill, "act", {})
    d = result.to_dict()
    assert d["status"] == "success"
    assert "execution_time_ms" in d
