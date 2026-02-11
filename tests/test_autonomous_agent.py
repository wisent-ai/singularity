"""
Comprehensive tests for AutonomousAgent — the main agent orchestration loop.

Tests cover:
- Agent initialization and configuration
- Skill initialization
- Tool listing
- Execute method dispatch
- Cost tracking
- Activity logging
- Run loop behavior
- Stop mechanism
- Edge cases and error handling
"""

import pytest
import asyncio
import json
import os
from unittest.mock import MagicMock, AsyncMock, patch, PropertyMock
from pathlib import Path

from singularity.autonomous_agent import AutonomousAgent, ACTIVITY_FILE
from singularity.cognition.types import Action, TokenUsage, Decision, AgentState
from singularity.skills.base.types import SkillResult, SkillAction, SkillManifest


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def mock_agent():
    """Create an agent with mocked LLM and skills."""
    with patch.dict(os.environ, {
        "ANTHROPIC_API_KEY": "test-key",
        "OPENAI_API_KEY": "",
        "TWITTER_API_KEY": "",
        "TWITTER_API_SECRET": "",
        "TWITTER_ACCESS_TOKEN": "",
        "TWITTER_ACCESS_SECRET": "",
        "GITHUB_TOKEN": "",
        "RESEND_API_KEY": "",
        "VERCEL_TOKEN": "",
        "STRIPE_SECRET_KEY": "",
        "REDDIT_USERNAME": "",
        "REDDIT_PASSWORD": "",
    }):
        agent = AutonomousAgent(
            name="TestAgent",
            ticker="TST",
            agent_type="testing",
            starting_balance=100.0,
            instance_type="local",
            llm_provider="anthropic",
            anthropic_api_key="test-key",
            llm_model="claude-sonnet-4-20250514",
        )
        return agent


@pytest.fixture
def mock_skill():
    """Create a mock skill for testing."""
    skill = MagicMock()
    skill.manifest = SkillManifest(
        skill_id="test",
        name="Test Skill",
        version="1.0.0",
        category="testing",
        description="A test skill",
        actions=[
            SkillAction(
                name="do_something",
                description="Do something useful",
                parameters={"input": {"type": "string"}},
            ),
        ],
        required_credentials=[],
    )
    skill.execute = AsyncMock(return_value=SkillResult(
        success=True, message="Done", data={"result": "ok"},
    ))
    return skill


# ============================================================
# Initialization Tests
# ============================================================

class TestAgentInit:
    """Test agent initialization."""

    def test_basic_init(self, mock_agent):
        """Agent should initialize with correct attributes."""
        assert mock_agent.name == "TestAgent"
        assert mock_agent.ticker == "TST"
        assert mock_agent.agent_type == "testing"
        assert mock_agent.balance == 100.0
        assert mock_agent.instance_type == "local"
        assert mock_agent.cycle == 0
        assert mock_agent.running is False

    def test_instance_cost_local(self, mock_agent):
        """Local instance should have zero cost."""
        assert mock_agent.instance_cost_per_hour == 0.0

    def test_instance_cost_cloud(self):
        """Cloud instances should have non-zero cost."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test"}):
            agent = AutonomousAgent(
                name="Cloud",
                ticker="CLD",
                instance_type="e2-micro",
                llm_provider="none",
            )
            assert agent.instance_cost_per_hour == 0.0084

    def test_cognition_engine_created(self, mock_agent):
        """Should create CognitionEngine."""
        assert mock_agent.cognition is not None

    def test_skills_registry_created(self, mock_agent):
        """Should create SkillRegistry."""
        assert mock_agent.skills is not None

    def test_conversation_starts_empty(self, mock_agent):
        """Conversation should start empty."""
        assert mock_agent.conversation == []

    def test_created_resources_initialized(self, mock_agent):
        """Created resources should be initialized with empty lists."""
        assert mock_agent.created_resources == {
            'payment_links': [], 'products': [], 'files': [], 'repos': [],
        }

    def test_default_specialty_from_type(self):
        """Specialty should default to agent_type."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test"}):
            agent = AutonomousAgent(
                name="Spec",
                ticker="SPC",
                agent_type="developer",
                llm_provider="none",
            )
            assert agent.specialty == "developer"

    def test_total_costs_start_zero(self, mock_agent):
        """All cost trackers should start at zero."""
        assert mock_agent.total_api_cost == 0.0
        assert mock_agent.total_instance_cost == 0.0
        assert mock_agent.total_tokens_used == 0


# ============================================================
# Tool Listing Tests
# ============================================================

class TestGetTools:
    """Test _get_tools method."""

    def test_returns_tool_dicts(self, mock_agent, mock_skill):
        """Should return list of tool dictionaries."""
        mock_agent.skills.skills["test"] = mock_skill
        tools = mock_agent._get_tools()
        assert len(tools) >= 1
        tool_names = [t["name"] for t in tools]
        assert "test:do_something" in tool_names

    def test_empty_skills_returns_wait(self, mock_agent):
        """Should return wait tool when no skills installed."""
        mock_agent.skills.skills = {}
        tools = mock_agent._get_tools()
        assert len(tools) == 1
        assert tools[0]["name"] == "wait"

    def test_tool_has_required_fields(self, mock_agent, mock_skill):
        """Each tool dict should have name, description, parameters."""
        mock_agent.skills.skills["test"] = mock_skill
        tools = mock_agent._get_tools()
        for tool in tools:
            assert "name" in tool
            assert "description" in tool
            assert "parameters" in tool


# ============================================================
# Execute Tests
# ============================================================

class TestExecute:
    """Test _execute method."""

    @pytest.mark.asyncio
    async def test_execute_wait(self, mock_agent):
        """Wait action should return waited status."""
        action = Action(tool="wait", reasoning="Nothing to do")
        result = await mock_agent._execute(action)
        assert result["status"] == "waited"

    @pytest.mark.asyncio
    async def test_execute_skill_action(self, mock_agent, mock_skill):
        """Should execute skill action and return result."""
        mock_agent.skills.skills["test"] = mock_skill
        action = Action(tool="test:do_something", params={"input": "hello"})
        result = await mock_agent._execute(action)
        assert result["status"] == "success"
        assert result["data"]["result"] == "ok"
        mock_skill.execute.assert_called_once_with("do_something", {"input": "hello"})

    @pytest.mark.asyncio
    async def test_execute_unknown_skill(self, mock_agent):
        """Should return error for unknown skill."""
        action = Action(tool="nonexistent:action")
        result = await mock_agent._execute(action)
        assert result["status"] == "error"
        assert "Unknown tool" in result["message"]

    @pytest.mark.asyncio
    async def test_execute_no_colon_in_tool(self, mock_agent):
        """Tool without colon should return error."""
        action = Action(tool="invalid_tool_name")
        result = await mock_agent._execute(action)
        assert result["status"] == "error"

    @pytest.mark.asyncio
    async def test_execute_skill_exception(self, mock_agent, mock_skill):
        """Should handle skill exceptions gracefully."""
        mock_skill.execute = AsyncMock(side_effect=RuntimeError("Skill crashed"))
        mock_agent.skills.skills["test"] = mock_skill
        action = Action(tool="test:do_something", params={})
        result = await mock_agent._execute(action)
        assert result["status"] == "error"
        assert "Skill crashed" in result["message"]

    @pytest.mark.asyncio
    async def test_execute_failed_skill_result(self, mock_agent, mock_skill):
        """Should return 'failed' status when skill returns failure."""
        mock_skill.execute = AsyncMock(return_value=SkillResult(
            success=False, message="Permission denied",
        ))
        mock_agent.skills.skills["test"] = mock_skill
        action = Action(tool="test:do_something", params={})
        result = await mock_agent._execute(action)
        assert result["status"] == "failed"
        assert "Permission denied" in result["message"]


# ============================================================
# Activity Logging Tests
# ============================================================

class TestActivityLogging:
    """Test _log and _save_activity methods."""

    def test_log_prints_message(self, mock_agent, capsys):
        """_log should print formatted message."""
        mock_agent._log("TEST", "Hello world")
        captured = capsys.readouterr()
        assert "[TST]" in captured.out
        assert "[TEST]" in captured.out
        assert "Hello world" in captured.out

    def test_save_activity_creates_file(self, mock_agent, tmp_path):
        """Should create activity file."""
        activity_file = tmp_path / "activity.json"
        with patch("singularity.autonomous_agent.ACTIVITY_FILE", activity_file):
            mock_agent._save_activity("INIT", "Starting up")
            assert activity_file.exists()
            data = json.loads(activity_file.read_text())
            assert data["state"]["name"] == "TestAgent"
            assert len(data["logs"]) == 1

    def test_save_activity_appends_logs(self, mock_agent, tmp_path):
        """Should append to existing logs."""
        activity_file = tmp_path / "activity.json"
        with patch("singularity.autonomous_agent.ACTIVITY_FILE", activity_file):
            mock_agent._save_activity("LOG1", "First")
            mock_agent._save_activity("LOG2", "Second")
            data = json.loads(activity_file.read_text())
            assert len(data["logs"]) == 2

    def test_save_activity_truncates_logs(self, mock_agent, tmp_path):
        """Should keep only last 100 log entries."""
        activity_file = tmp_path / "activity.json"
        with patch("singularity.autonomous_agent.ACTIVITY_FILE", activity_file):
            for i in range(110):
                mock_agent._save_activity("LOG", f"Entry {i}")
            data = json.loads(activity_file.read_text())
            assert len(data["logs"]) == 100

    def test_save_activity_truncates_long_messages(self, mock_agent, tmp_path):
        """Should truncate messages over 500 chars."""
        activity_file = tmp_path / "activity.json"
        with patch("singularity.autonomous_agent.ACTIVITY_FILE", activity_file):
            mock_agent._save_activity("LOG", "x" * 1000)
            data = json.loads(activity_file.read_text())
            assert len(data["logs"][0]["message"]) <= 500

    def test_mark_stopped(self, mock_agent, tmp_path):
        """_mark_stopped should update status to 'stopped'."""
        activity_file = tmp_path / "activity.json"
        with patch("singularity.autonomous_agent.ACTIVITY_FILE", activity_file):
            mock_agent._save_activity("START", "Running")
            mock_agent._mark_stopped()
            data = json.loads(activity_file.read_text())
            assert data["status"] == "stopped"


# ============================================================
# Stop Mechanism Tests
# ============================================================

class TestStopMechanism:
    """Test agent stop behavior."""

    def test_stop_sets_flag(self, mock_agent):
        """stop() should set running to False."""
        mock_agent.running = True
        mock_agent.stop()
        assert mock_agent.running is False


# ============================================================
# Run Loop Tests
# ============================================================

class TestRunLoop:
    """Test the main agent run loop."""

    @pytest.mark.asyncio
    async def test_run_stops_on_zero_balance(self, mock_agent):
        """Agent should stop when balance hits zero."""
        mock_agent.balance = 0.001
        mock_agent.cycle_interval = 0.01  # Fast cycles for testing

        async def mock_think(*args, **kwargs):
            # Deduct balance to trigger stop
            mock_agent.balance = -1
            return Decision(
                action=Action(tool="wait"),
                reasoning="test",
                token_usage=TokenUsage(input_tokens=10, output_tokens=5),
                api_cost_usd=0.001,
            ), []

        mock_agent.cognition.think_with_context = mock_think

        # Run should complete (not hang)
        await asyncio.wait_for(mock_agent.run(), timeout=5.0)
        assert mock_agent.cycle >= 1

    @pytest.mark.asyncio
    async def test_run_stops_on_stop_called(self, mock_agent):
        """Agent should stop when stop() is called."""
        mock_agent.balance = 1000.0
        mock_agent.cycle_interval = 0.01

        call_count = 0

        async def mock_think(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count >= 3:
                mock_agent.stop()
            return Decision(
                action=Action(tool="wait"),
                reasoning="test",
                token_usage=TokenUsage(input_tokens=10, output_tokens=5),
                api_cost_usd=0.0001,
            ), []

        mock_agent.cognition.think_with_context = mock_think
        await asyncio.wait_for(mock_agent.run(), timeout=5.0)
        assert call_count >= 3

    @pytest.mark.asyncio
    async def test_run_records_actions(self, mock_agent):
        """Run loop should record actions in recent_actions."""
        mock_agent.balance = 1.0
        mock_agent.cycle_interval = 0.01

        async def mock_think(*args, **kwargs):
            mock_agent.stop()
            return Decision(
                action=Action(tool="chat:send", params={"message": "hi"}),
                reasoning="greeting",
                token_usage=TokenUsage(input_tokens=100, output_tokens=50),
                api_cost_usd=0.001,
            ), []

        mock_agent.cognition.think_with_context = mock_think
        await asyncio.wait_for(mock_agent.run(), timeout=5.0)
        assert len(mock_agent.recent_actions) >= 1
        assert mock_agent.recent_actions[-1]["tool"] == "chat:send"

    @pytest.mark.asyncio
    async def test_run_tracks_costs(self, mock_agent):
        """Run loop should accumulate API and instance costs."""
        mock_agent.balance = 10.0
        mock_agent.cycle_interval = 0.01

        async def mock_think(*args, **kwargs):
            mock_agent.stop()
            return Decision(
                action=Action(tool="wait"),
                reasoning="test",
                token_usage=TokenUsage(input_tokens=100, output_tokens=50),
                api_cost_usd=0.005,
            ), []

        mock_agent.cognition.think_with_context = mock_think
        await asyncio.wait_for(mock_agent.run(), timeout=5.0)
        assert mock_agent.total_api_cost > 0
        assert mock_agent.total_tokens_used > 0

    @pytest.mark.asyncio
    async def test_run_deducts_balance(self, mock_agent):
        """Run loop should deduct costs from balance."""
        initial_balance = mock_agent.balance
        mock_agent.cycle_interval = 0.01

        async def mock_think(*args, **kwargs):
            mock_agent.stop()
            return Decision(
                action=Action(tool="wait"),
                reasoning="test",
                token_usage=TokenUsage(input_tokens=100, output_tokens=50),
                api_cost_usd=0.01,
            ), []

        mock_agent.cognition.think_with_context = mock_think
        await asyncio.wait_for(mock_agent.run(), timeout=5.0)
        assert mock_agent.balance < initial_balance


# ============================================================
# Instance Cost Tests
# ============================================================

class TestInstanceCosts:
    """Test instance cost mapping."""

    def test_known_instance_types(self):
        """All known instance types should have correct costs."""
        expected = {
            "e2-micro": 0.0084,
            "e2-small": 0.0168,
            "e2-medium": 0.0336,
            "e2-standard-2": 0.0672,
            "g2-standard-4": 0.7111,
            "local": 0.0,
        }
        for itype, cost in expected.items():
            assert AutonomousAgent.INSTANCE_COSTS[itype] == cost

    def test_unknown_instance_type_zero_cost(self):
        """Unknown instance type should default to zero cost."""
        cost = AutonomousAgent.INSTANCE_COSTS.get("unknown-type", 0.0)
        assert cost == 0.0
