"""Tests for execution safety features: timeout, duplicate detection, error tracking."""
import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from singularity.cognition import Action


@pytest.fixture
def agent():
    """Create agent with mocked LLM."""
    with patch.dict("os.environ", {}, clear=False):
        from singularity.autonomous_agent import AutonomousAgent
        a = AutonomousAgent(
            name="TestAgent",
            ticker="TEST",
            llm_provider="none",
            action_timeout=2.0,
            max_consecutive_errors=3,
        )
        return a


class TestActionTimeout:
    @pytest.mark.asyncio
    async def test_wait_action_no_timeout(self, agent):
        result = await agent._execute(Action(tool="wait"))
        assert result["status"] == "waited"

    @pytest.mark.asyncio
    async def test_timeout_returns_error(self, agent):
        """Skill that hangs should be timed out."""
        async def slow_execute(action_name, params):
            await asyncio.sleep(10)

        mock_skill = MagicMock()
        mock_skill.execute = slow_execute
        agent.skills.skills["slow"] = mock_skill

        result = await agent._execute(Action(tool="slow:do_thing"))
        assert result["status"] == "error"
        assert "timed out" in result["message"]
        assert result["elapsed_seconds"] >= 2.0

    @pytest.mark.asyncio
    async def test_successful_action_tracks_timing(self, agent):
        """Successful actions should track elapsed time."""
        mock_result = MagicMock(success=True, data={"ok": 1}, message="done")
        mock_skill = MagicMock()
        mock_skill.execute = AsyncMock(return_value=mock_result)
        agent.skills.skills["fast"] = mock_skill

        result = await agent._execute(Action(tool="fast:go"))
        assert result["status"] == "success"
        assert "elapsed_seconds" in result
        assert len(agent._action_timings) == 1


class TestDuplicateDetection:
    @pytest.mark.asyncio
    async def test_duplicate_failed_action_warns(self, agent):
        """Repeating a failed action should log a warning."""
        agent.recent_actions = [{
            "cycle": 1, "tool": "fs:write", "params": {"path": "/tmp/x"},
            "result": {"status": "error", "message": "Permission denied"},
        }]
        warning = agent._detect_duplicate_action("fs:write", {"path": "/tmp/x"})
        assert warning is not None
        assert "WARNING" in warning
        assert "Permission denied" in warning

    @pytest.mark.asyncio
    async def test_duplicate_success_notes(self, agent):
        agent.recent_actions = [{
            "cycle": 1, "tool": "fs:read", "params": {"path": "/tmp/y"},
            "result": {"status": "success"},
        }]
        warning = agent._detect_duplicate_action("fs:read", {"path": "/tmp/y"})
        assert warning is not None
        assert "NOTE" in warning

    @pytest.mark.asyncio
    async def test_no_duplicate_for_different_params(self, agent):
        agent.recent_actions = [{
            "cycle": 1, "tool": "fs:read", "params": {"path": "/tmp/a"},
            "result": {"status": "success"},
        }]
        warning = agent._detect_duplicate_action("fs:read", {"path": "/tmp/b"})
        assert warning is None


class TestErrorTracking:
    @pytest.mark.asyncio
    async def test_consecutive_errors_tracked(self, agent):
        agent._record_error("tool_a")
        agent._record_error("tool_b")
        assert agent._consecutive_errors == 2
        assert agent._error_streak_tools == ["tool_a", "tool_b"]

    @pytest.mark.asyncio
    async def test_success_resets_error_count(self, agent):
        agent._record_error("tool_a")
        agent._record_error("tool_b")
        mock_result = MagicMock(success=True, data={}, message="ok")
        mock_skill = MagicMock()
        mock_skill.execute = AsyncMock(return_value=mock_result)
        agent.skills.skills["good"] = mock_skill

        await agent._execute(Action(tool="good:go"))
        assert agent._consecutive_errors == 0

    @pytest.mark.asyncio
    async def test_unknown_skill_lists_available(self, agent):
        result = await agent._execute(Action(tool="nonexistent:action"))
        assert result["status"] == "error"
        assert "Unknown skill" in result["message"]

    @pytest.mark.asyncio
    async def test_invalid_tool_format(self, agent):
        result = await agent._execute(Action(tool="nocolon"))
        assert result["status"] == "error"
        assert "skill:action" in result["message"]

    @pytest.mark.asyncio
    async def test_exception_includes_type(self, agent):
        """Exceptions should include the error type name."""
        mock_skill = MagicMock()
        mock_skill.execute = AsyncMock(side_effect=ValueError("bad value"))
        agent.skills.skills["broken"] = mock_skill

        result = await agent._execute(Action(tool="broken:go"))
        assert result["status"] == "error"
        assert "ValueError" in result["message"]
        assert "bad value" in result["message"]
