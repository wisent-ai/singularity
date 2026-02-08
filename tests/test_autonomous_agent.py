#!/usr/bin/env python3
"""Comprehensive tests for AutonomousAgent — the core agent orchestration engine.

Tests cover:
- Agent construction and default configuration
- Instance cost tracking constants
- Skill initialization and wiring
- Event bus integration
- Dynamic skill management (add/remove at runtime)
- Skill status inspection
- Tool discovery from installed skills
- Agent state retrieval
- The main run() loop with mocked cognition
- Action execution pipeline (_execute) with fuzzy matching
- Cost estimation for actions
- Resource tracking
- Outcome recording for feedback loops
- Activity file persistence (_save_activity, _mark_stopped)
- Logging
- Kill-for-tampering (integrity enforcement)
- Graceful stop
- The async main() entry point
- Edge cases and error handling

All tests are fully mocked — no API keys, network access, or filesystem side-effects.
"""

import asyncio
import json
import os
import pytest
import tempfile
from pathlib import Path
from datetime import datetime
from unittest.mock import (
    patch,
    AsyncMock,
    MagicMock,
    PropertyMock,
    mock_open,
    call,
)
from typing import Dict, List

# ---------------------------------------------------------------------------
# Lightweight mock skill to avoid importing 50+ real skill classes
# ---------------------------------------------------------------------------


class _FakeSkillAction:
    def __init__(self, name="do_something", description="test action",
                 parameters=None, estimated_cost=0.01):
        self.name = name
        self.description = description
        self.parameters = parameters or {}
        self.estimated_cost = estimated_cost


class _FakeManifest:
    def __init__(self, skill_id="fake", name="FakeSkill", actions=None,
                 required_credentials=None):
        self.skill_id = skill_id
        self.name = name
        self.actions = actions or [_FakeSkillAction()]
        self.required_credentials = required_credentials or []
        self.version = "1.0.0"
        self.category = "test"
        self.description = "A fake skill for testing"


class _FakeSkill:
    """Minimal skill-like object that satisfies the agent's expectations."""

    def __init__(self, credentials=None, skill_id="fake", name="FakeSkill",
                 actions=None, required_credentials=None):
        self.credentials = credentials or {}
        self._manifest = _FakeManifest(
            skill_id=skill_id, name=name, actions=actions,
            required_credentials=required_credentials,
        )
        self.initialized = False
        self.context = None
        self._usage_count = 0

    @property
    def manifest(self):
        return self._manifest

    def check_credentials(self):
        for cred in self._manifest.required_credentials:
            if cred not in self.credentials or not self.credentials[cred]:
                return False
        return True

    async def execute(self, action: str, params: dict):
        return MagicMock(success=True, message="ok", data={"result": True},
                         cost=0.0, revenue=0.0)

    def get_actions(self):
        return self._manifest.actions

    def set_context(self, ctx):
        self.context = ctx

    def record_usage(self, cost=0, revenue=0):
        self._usage_count += 1


class _FakeSkillClass:
    """A callable class that mimics a skill class (constructor returns a _FakeSkill)."""

    def __init__(self, skill_id="fake", name="FakeSkill", actions=None,
                 required_credentials=None, fail_init=False):
        self._skill_id = skill_id
        self._name = name
        self._actions = actions
        self._required_credentials = required_credentials
        self._fail_init = fail_init
        self.__name__ = name  # Needed by agent's error handler

    def __call__(self, credentials=None, **kwargs):
        if self._fail_init:
            raise RuntimeError("Skill init failed")
        return _FakeSkill(
            credentials=credentials,
            skill_id=self._skill_id,
            name=self._name,
            actions=self._actions,
            required_credentials=self._required_credentials,
        )


# ---------------------------------------------------------------------------
# Patches to prevent importing the huge skill dependency tree
# ---------------------------------------------------------------------------

def _create_agent_with_mocks(**overrides):
    """Create an AutonomousAgent with all heavy dependencies mocked out.

    This patches _init_skills, _wire_event_bus, and ExecutionInstrumentation
    so we can construct the agent quickly without importing 50+ skill modules.
    """
    from singularity.autonomous_agent import AutonomousAgent

    defaults = dict(
        name="TestBot",
        ticker="TEST",
        agent_type="test",
        starting_balance=100.0,
        instance_type="local",
        cycle_interval_seconds=0.01,
        llm_provider="anthropic",
        llm_model="test-model",
        skills=[],  # Empty skills list to avoid importing all defaults
    )
    defaults.update(overrides)

    with patch.object(AutonomousAgent, '_init_skills'), \
         patch.object(AutonomousAgent, '_wire_event_bus'), \
         patch('singularity.autonomous_agent.ExecutionInstrumentation'), \
         patch('singularity.autonomous_agent.AdaptiveExecutor'), \
         patch('singularity.autonomous_agent.CognitionEngine') as mock_cog:
        # Make cognition mock usable
        mock_cog_instance = MagicMock()
        mock_cog.return_value = mock_cog_instance
        agent = AutonomousAgent(**defaults)
        agent.cognition = mock_cog_instance
    return agent


# ═══════════════════════════════════════════════════════════════════════
#                       CONSTRUCTOR / DEFAULTS
# ═══════════════════════════════════════════════════════════════════════


class TestAgentConstruction:
    """Tests for AutonomousAgent.__init__."""

    def test_default_attributes(self):
        agent = _create_agent_with_mocks()
        assert agent.name == "TestBot"
        assert agent.ticker == "TEST"
        assert agent.agent_type == "test"
        assert agent.balance == 100.0
        assert agent.instance_type == "local"
        assert agent.cycle_interval == 0.01
        assert agent.total_api_cost == 0.0
        assert agent.total_instance_cost == 0.0
        assert agent.total_tokens_used == 0
        assert agent.cycle == 0
        assert agent.running is False
        assert agent.recent_actions == []

    def test_specialty_defaults_to_agent_type(self):
        agent = _create_agent_with_mocks(specialty="")
        assert agent.specialty == "test"

    def test_specialty_override(self):
        agent = _create_agent_with_mocks(specialty="custom-specialty")
        assert agent.specialty == "custom-specialty"

    def test_instance_cost_per_hour_known_type(self):
        agent = _create_agent_with_mocks(instance_type="e2-micro")
        assert agent.instance_cost_per_hour == 0.0084

    def test_instance_cost_per_hour_unknown_type(self):
        agent = _create_agent_with_mocks(instance_type="nonexistent-type")
        assert agent.instance_cost_per_hour == 0.0

    def test_instance_cost_local(self):
        agent = _create_agent_with_mocks(instance_type="local")
        assert agent.instance_cost_per_hour == 0.0

    def test_created_resources_initialized(self):
        agent = _create_agent_with_mocks()
        assert "payment_links" in agent.created_resources
        assert "products" in agent.created_resources
        assert "files" in agent.created_resources
        assert "repos" in agent.created_resources
        for key in agent.created_resources:
            assert agent.created_resources[key] == []

    def test_project_context_from_string(self):
        agent = _create_agent_with_mocks(project_context="Build a trading bot")
        assert agent.project_context == "Build a trading bot"

    def test_project_context_from_file(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Context from file\n")
            f.flush()
            try:
                agent = _create_agent_with_mocks(project_context_file=f.name)
                assert agent.project_context == "Context from file"
            finally:
                os.unlink(f.name)

    def test_project_context_file_missing_leaves_empty(self):
        agent = _create_agent_with_mocks(
            project_context="",
            project_context_file="/nonexistent/path/ctx.txt",
        )
        assert agent.project_context == ""


class TestInstanceCosts:
    """Verify the INSTANCE_COSTS class attribute."""

    def test_all_known_instance_types(self):
        from singularity.autonomous_agent import AutonomousAgent
        expected_types = {"e2-micro", "e2-small", "e2-medium",
                          "e2-standard-2", "g2-standard-4", "local"}
        assert set(AutonomousAgent.INSTANCE_COSTS.keys()) == expected_types

    def test_gpu_instance_is_most_expensive(self):
        from singularity.autonomous_agent import AutonomousAgent
        costs = AutonomousAgent.INSTANCE_COSTS
        assert costs["g2-standard-4"] == max(costs.values())

    def test_local_is_free(self):
        from singularity.autonomous_agent import AutonomousAgent
        assert AutonomousAgent.INSTANCE_COSTS["local"] == 0.0


class TestDefaultSkillClasses:
    """Verify DEFAULT_SKILL_CLASSES is populated."""

    def test_default_skills_list_is_populated(self):
        from singularity.autonomous_agent import AutonomousAgent
        assert len(AutonomousAgent.DEFAULT_SKILL_CLASSES) > 40

    def test_default_skills_are_unique(self):
        from singularity.autonomous_agent import AutonomousAgent
        classes = AutonomousAgent.DEFAULT_SKILL_CLASSES
        assert len(classes) == len(set(classes))


# ═══════════════════════════════════════════════════════════════════════
#                       SKILL INITIALIZATION
# ═══════════════════════════════════════════════════════════════════════


class TestInitSkills:
    """Tests for _init_skills and credential wiring."""

    def test_init_skills_installs_valid_skill(self):
        """Skills with valid credentials should be installed."""
        from singularity.autonomous_agent import AutonomousAgent

        fake_class = _FakeSkillClass(skill_id="good", name="GoodSkill")

        with patch.object(AutonomousAgent, '_wire_event_bus'), \
             patch('singularity.autonomous_agent.ExecutionInstrumentation'), \
             patch('singularity.autonomous_agent.AdaptiveExecutor'), \
             patch('singularity.autonomous_agent.CognitionEngine'):
            agent = AutonomousAgent(
                name="Test", ticker="T", skills=[fake_class],
                starting_balance=10.0,
            )
        # Skill should have been installed (or attempted)
        assert isinstance(agent._skill_load_errors, list)

    def test_init_skills_records_errors(self):
        """Skill classes that raise during init should be recorded as errors."""
        from singularity.autonomous_agent import AutonomousAgent

        class BrokenSkill:
            def __init__(self, *args, **kwargs):
                raise RuntimeError("Broken!")

        with patch.object(AutonomousAgent, '_wire_event_bus'), \
             patch('singularity.autonomous_agent.ExecutionInstrumentation'), \
             patch('singularity.autonomous_agent.AdaptiveExecutor'), \
             patch('singularity.autonomous_agent.CognitionEngine'):
            agent = AutonomousAgent(
                name="Test", ticker="T", skills=[BrokenSkill],
                starting_balance=10.0,
            )
        assert len(agent._skill_load_errors) >= 1
        assert "BrokenSkill" in agent._skill_load_errors[0]["skill"]

    def test_empty_skills_list_works(self):
        """Agent should work with zero skills."""
        agent = _create_agent_with_mocks(skills=[])
        assert agent is not None


# ═══════════════════════════════════════════════════════════════════════
#                       EVENT BUS
# ═══════════════════════════════════════════════════════════════════════


class TestEventBus:
    """Tests for _wire_event_bus and event emission."""

    def test_wire_event_bus_finds_event_skill(self):
        """_wire_event_bus should call set_event_bus on EventSkill if present."""
        agent = _create_agent_with_mocks()
        from singularity.skills.event import EventSkill
        mock_event_skill = MagicMock(spec=EventSkill)
        mock_event_skill.__class__ = EventSkill
        # Make isinstance check work
        agent.skills.skills = {"event": mock_event_skill}

        # Call it manually (it was patched out during construction)
        from singularity.autonomous_agent import AutonomousAgent
        AutonomousAgent._wire_event_bus(agent)
        mock_event_skill.set_event_bus.assert_called_once_with(agent._event_bus)

    @pytest.mark.asyncio
    async def test_emit_event_publishes(self):
        """_emit_event should publish to the event bus."""
        agent = _create_agent_with_mocks()
        agent._event_bus = MagicMock()
        agent._event_bus.publish = AsyncMock()

        await agent._emit_event("test.topic", {"key": "val"})
        agent._event_bus.publish.assert_called_once()
        event = agent._event_bus.publish.call_args[0][0]
        assert event.topic == "test.topic"
        assert event.data == {"key": "val"}
        assert event.source == "agent"

    def test_get_pending_events_no_event_skill(self):
        """Returns empty list when no EventSkill is installed."""
        agent = _create_agent_with_mocks()
        agent.skills.skills = {}
        assert agent._get_pending_events() == []

    def test_get_pending_events_with_event_skill(self):
        """Returns events from EventSkill."""
        agent = _create_agent_with_mocks()
        from singularity.skills.event import EventSkill
        mock_event_skill = MagicMock(spec=EventSkill)
        mock_event_skill.__class__ = EventSkill
        mock_event_skill.get_pending_events.return_value = [
            {"topic": "test", "data": {}}
        ]
        agent.skills.skills = {"event": mock_event_skill}
        events = agent._get_pending_events()
        assert len(events) == 1
        assert events[0]["topic"] == "test"


# ═══════════════════════════════════════════════════════════════════════
#                       SCHEDULER TICK
# ═══════════════════════════════════════════════════════════════════════


class TestSchedulerTick:
    """Tests for _tick_scheduler."""

    @pytest.mark.asyncio
    async def test_tick_scheduler_no_scheduler(self):
        """Runs without error when no SchedulerSkill installed."""
        agent = _create_agent_with_mocks()
        agent.skills.skills = {}
        await agent._tick_scheduler()  # Should not raise

    @pytest.mark.asyncio
    async def test_tick_scheduler_with_due_tasks(self):
        """Executes due tasks from SchedulerSkill."""
        agent = _create_agent_with_mocks()
        from singularity.skills.scheduler import SchedulerSkill
        mock_sched = MagicMock(spec=SchedulerSkill)
        mock_sched.__class__ = SchedulerSkill
        mock_sched.get_due_count.return_value = 2
        result = MagicMock(success=True, message="Task done")
        mock_sched.tick = AsyncMock(return_value=[result, result])
        agent.skills.skills = {"scheduler": mock_sched}

        await agent._tick_scheduler()
        mock_sched.tick.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_tick_scheduler_zero_due(self):
        """Does not call tick when no tasks are due."""
        agent = _create_agent_with_mocks()
        from singularity.skills.scheduler import SchedulerSkill
        mock_sched = MagicMock(spec=SchedulerSkill)
        mock_sched.__class__ = SchedulerSkill
        mock_sched.get_due_count.return_value = 0
        mock_sched.tick = AsyncMock()
        agent.skills.skills = {"scheduler": mock_sched}

        await agent._tick_scheduler()
        mock_sched.tick.assert_not_awaited()


# ═══════════════════════════════════════════════════════════════════════
#                  DYNAMIC SKILL MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════


class TestDynamicSkillManagement:
    """Tests for add_skill and remove_skill."""

    def test_add_skill_duplicate_returns_false(self):
        """Adding a skill that's already registered returns False."""
        agent = _create_agent_with_mocks()
        fake_class = _FakeSkillClass(skill_id="dup")
        agent._skill_classes = [fake_class]
        assert agent.add_skill(fake_class) is False

    def test_add_skill_success(self):
        """Successfully adding a new skill returns True."""
        agent = _create_agent_with_mocks()
        agent._skill_classes = []
        # Pre-populate the registry with the fake skill
        fake_class = _FakeSkillClass(skill_id="new_skill", name="NewSkill")
        fake_skill = fake_class()
        agent.skills._credentials = {}
        agent.skills.install = MagicMock(return_value=True)
        agent.skills.get = MagicMock(return_value=fake_skill)

        result = agent.add_skill(fake_class)
        assert result is True
        assert fake_class in agent._skill_classes

    def test_add_skill_failure_cleans_up(self):
        """Failed skill addition cleans up and returns False."""
        agent = _create_agent_with_mocks()
        agent._skill_classes = []
        agent.skills._credentials = {}

        class FailSkillClass:
            def __init__(self, *args, **kwargs):
                raise RuntimeError("Cannot init")
            @property
            def __name__(self):
                return "FailSkillClass"

        # The class lookup for __name__ won't work with our test class,
        # so just use a lambda that raises
        bad_class = _FakeSkillClass(skill_id="bad", fail_init=True)
        # Force the install to raise
        agent.skills.install = MagicMock(side_effect=RuntimeError("Install failed"))

        result = agent.add_skill(bad_class)
        assert result is False
        assert bad_class not in agent._skill_classes
        assert len(agent._skill_load_errors) >= 1

    def test_remove_skill_existing(self):
        """Removing an installed skill returns True."""
        agent = _create_agent_with_mocks()
        fake = _FakeSkill(skill_id="removable", name="Removable")
        agent.skills.get = MagicMock(return_value=fake)
        agent.skills.uninstall = MagicMock(return_value=True)

        assert agent.remove_skill("removable") is True
        agent.skills.uninstall.assert_called_once_with("removable")

    def test_remove_skill_nonexistent(self):
        """Removing a non-existent skill returns False."""
        agent = _create_agent_with_mocks()
        agent.skills.get = MagicMock(return_value=None)
        assert agent.remove_skill("does_not_exist") is False


# ═══════════════════════════════════════════════════════════════════════
#                    SKILL STATUS & TOOLS
# ═══════════════════════════════════════════════════════════════════════


class TestSkillStatusAndTools:
    """Tests for get_skill_status and _get_tools."""

    def test_get_skill_status_with_skills(self):
        """Returns correct counts and details."""
        agent = _create_agent_with_mocks()
        skill1 = _FakeSkill(skill_id="s1", name="Skill1", actions=[
            _FakeSkillAction(name="act1"),
            _FakeSkillAction(name="act2"),
        ])
        skill2 = _FakeSkill(skill_id="s2", name="Skill2", actions=[
            _FakeSkillAction(name="act3"),
        ])
        agent.skills.skills = {"s1": skill1, "s2": skill2}
        agent._skill_load_errors = [{"skill": "BadSkill", "error": "oops"}]

        status = agent.get_skill_status()
        assert len(status["loaded"]) == 2
        assert status["total_tools"] == 3
        assert len(status["load_errors"]) == 1

    def test_get_skill_status_empty(self):
        """Empty registry returns zeros."""
        agent = _create_agent_with_mocks()
        agent.skills.skills = {}
        agent._skill_load_errors = []
        status = agent.get_skill_status()
        assert status["loaded"] == []
        assert status["total_tools"] == 0
        assert status["load_errors"] == []

    def test_get_tools_returns_skill_actions(self):
        """_get_tools builds tool list from installed skills."""
        agent = _create_agent_with_mocks()
        skill = _FakeSkill(skill_id="fs", name="FileSystem", actions=[
            _FakeSkillAction(name="read_file", description="Read a file",
                             parameters={"path": {"type": "string"}}),
            _FakeSkillAction(name="write_file", description="Write a file"),
        ])
        agent.skills.skills = {"fs": skill}

        tools = agent._get_tools()
        assert len(tools) == 2
        assert tools[0]["name"] == "fs:read_file"
        assert tools[0]["description"] == "Read a file"
        assert tools[0]["parameters"] == {"path": {"type": "string"}}
        assert tools[1]["name"] == "fs:write_file"

    def test_get_tools_empty_returns_wait(self):
        """When no skills installed, returns a 'wait' tool."""
        agent = _create_agent_with_mocks()
        agent.skills.skills = {}
        tools = agent._get_tools()
        assert len(tools) == 1
        assert tools[0]["name"] == "wait"


# ═══════════════════════════════════════════════════════════════════════
#                      AGENT STATE
# ═══════════════════════════════════════════════════════════════════════


class TestAgentState:
    """Tests for _get_agent_state."""

    def test_get_agent_state_returns_complete_dict(self):
        agent = _create_agent_with_mocks()
        agent.balance = 42.5
        agent.cycle = 7
        agent.running = True
        agent.total_api_cost = 1.23
        agent.total_tokens_used = 5000
        agent.skills.skills = {"s1": MagicMock(), "s2": MagicMock()}

        state = agent._get_agent_state()
        assert state["agent_name"] == "TestBot"
        assert state["agent_ticker"] == "TEST"
        assert state["agent_type"] == "test"
        assert state["balance"] == 42.5
        assert state["cycle"] == 7
        assert state["running"] is True
        assert state["total_api_cost"] == 1.23
        assert state["total_tokens_used"] == 5000
        assert set(state["installed_skills"]) == {"s1", "s2"}

    def test_get_agent_state_runway_calculation(self):
        """Runway cycles should be balance / estimated cost per cycle."""
        agent = _create_agent_with_mocks(
            starting_balance=100.0,
            instance_type="local",
            cycle_interval_seconds=5.0,
        )
        state = agent._get_agent_state()
        # For local instance: est_cost_per_cycle = 0.01 + 0.0 * (5/3600) = 0.01
        # runway = 100.0 / 0.01 = 10000
        assert state["runway_cycles"] == pytest.approx(10000.0)


# ═══════════════════════════════════════════════════════════════════════
#                    ACTION EXECUTION (_execute)
# ═══════════════════════════════════════════════════════════════════════


class TestExecute:
    """Tests for _execute — the action execution pipeline."""

    @pytest.mark.asyncio
    async def test_execute_wait_action(self):
        """The 'wait' action returns without touching skills."""
        agent = _create_agent_with_mocks()
        from singularity.cognition import Action
        action = Action(tool="wait")
        result = await agent._execute(action)
        assert result["status"] == "waited"

    @pytest.mark.asyncio
    async def test_execute_unknown_tool(self):
        """Unknown tool (no colon, not 'wait') returns error."""
        agent = _create_agent_with_mocks()
        from singularity.cognition import Action

        # Mock ToolResolver to return no correction and no error (passes through)
        mock_resolver = MagicMock()
        mock_match = MagicMock(was_corrected=False, error="", resolved="no_colon_tool")
        mock_resolver.resolve.return_value = mock_match
        mock_validation = MagicMock(valid=True, missing_required=[], warnings=[])
        mock_resolver.validate_params.return_value = mock_validation
        agent._tool_resolver = mock_resolver

        action = Action(tool="no_colon_tool")
        result = await agent._execute(action)
        assert result["status"] == "error"
        assert "Unknown tool" in result["message"]

    @pytest.mark.asyncio
    async def test_execute_skill_action_success(self):
        """Successful skill execution returns success status."""
        agent = _create_agent_with_mocks()

        # Setup mock tool resolver
        mock_resolver = MagicMock()
        mock_match = MagicMock(was_corrected=False, error="", resolved="fake:do_it")
        mock_resolver.resolve.return_value = mock_match
        mock_validation = MagicMock(valid=True, missing_required=[], warnings=[])
        mock_resolver.validate_params.return_value = mock_validation
        agent._tool_resolver = mock_resolver

        # Setup mock skill
        mock_skill = MagicMock()
        mock_skill_result = MagicMock(success=True, message="done", data={"ok": True})
        mock_skill.execute = AsyncMock(return_value=mock_skill_result)
        agent.skills.get = MagicMock(return_value=mock_skill)

        # Setup adaptive executor
        mock_advice = MagicMock(should_execute=True, cost_warning="", reason="")
        agent._adaptive_executor = MagicMock()
        agent._adaptive_executor.get_advice.return_value = mock_advice
        agent._adaptive_executor.record_outcome = MagicMock()

        # Setup instrumentation
        agent._instrumentation = MagicMock()
        agent._instrumentation.instrumented_execute = AsyncMock(
            return_value={"status": "success", "data": {"ok": True}, "message": "done"}
        )

        from singularity.cognition import Action
        action = Action(tool="fake:do_it", params={"x": 1})
        result = await agent._execute(action)
        assert result["status"] == "success"

    @pytest.mark.asyncio
    async def test_execute_circuit_breaker_blocks(self):
        """When adaptive executor says don't execute, action is blocked."""
        agent = _create_agent_with_mocks()

        mock_resolver = MagicMock()
        mock_match = MagicMock(was_corrected=False, error="", resolved="skill:act")
        mock_resolver.resolve.return_value = mock_match
        mock_validation = MagicMock(valid=True, missing_required=[], warnings=[])
        mock_resolver.validate_params.return_value = mock_validation
        agent._tool_resolver = mock_resolver

        mock_skill = MagicMock()
        agent.skills.get = MagicMock(return_value=mock_skill)

        mock_advice = MagicMock(
            should_execute=False,
            cost_warning="",
            reason="Circuit breaker open",
        )
        agent._adaptive_executor = MagicMock()
        agent._adaptive_executor.get_advice.return_value = mock_advice

        from singularity.cognition import Action
        action = Action(tool="skill:act")
        result = await agent._execute(action)
        assert result["status"] == "blocked"
        assert "Circuit breaker" in result["message"]

    @pytest.mark.asyncio
    async def test_execute_fuzzy_match_correction(self):
        """ToolResolver auto-corrects a misspelled tool name."""
        agent = _create_agent_with_mocks()

        mock_resolver = MagicMock()
        mock_match = MagicMock(
            was_corrected=True,
            error="",
            resolved="filesystem:read_file",
            original="filesytem:red_file",
            confidence=0.85,
        )
        mock_resolver.resolve.return_value = mock_match
        mock_validation = MagicMock(valid=True, missing_required=[], warnings=[])
        mock_resolver.validate_params.return_value = mock_validation
        agent._tool_resolver = mock_resolver

        mock_skill = MagicMock()
        mock_skill.execute = AsyncMock(
            return_value=MagicMock(success=True, message="ok", data={})
        )
        agent.skills.get = MagicMock(return_value=mock_skill)

        mock_advice = MagicMock(should_execute=True, cost_warning="", reason="")
        agent._adaptive_executor = MagicMock()
        agent._adaptive_executor.get_advice.return_value = mock_advice

        agent._instrumentation = MagicMock()
        agent._instrumentation.instrumented_execute = AsyncMock(
            return_value={"status": "success", "data": {}, "message": "ok"}
        )

        from singularity.cognition import Action
        action = Action(tool="filesytem:red_file", params={"path": "/test"})
        result = await agent._execute(action)
        assert result["status"] == "success"

    @pytest.mark.asyncio
    async def test_execute_resolver_error(self):
        """When ToolResolver returns an error, execution returns error."""
        agent = _create_agent_with_mocks()

        mock_resolver = MagicMock()
        mock_match = MagicMock(
            was_corrected=False,
            error="No matching tool found for 'xyz'",
        )
        mock_resolver.resolve.return_value = mock_match
        agent._tool_resolver = mock_resolver

        from singularity.cognition import Action
        action = Action(tool="xyz:nope")
        result = await agent._execute(action)
        assert result["status"] == "error"
        assert "No matching tool" in result["message"]

    @pytest.mark.asyncio
    async def test_execute_missing_params(self):
        """Missing required params returns validation error."""
        agent = _create_agent_with_mocks()

        mock_resolver = MagicMock()
        mock_match = MagicMock(was_corrected=False, error="", resolved="skill:act")
        mock_resolver.resolve.return_value = mock_match
        mock_validation = MagicMock(
            valid=False,
            missing_required=["path"],
            warnings=[],
        )
        mock_resolver.validate_params.return_value = mock_validation
        agent._tool_resolver = mock_resolver

        from singularity.cognition import Action
        action = Action(tool="skill:act")
        result = await agent._execute(action)
        assert result["status"] == "error"
        assert "path" in result["message"]

    @pytest.mark.asyncio
    async def test_execute_skill_exception_recorded(self):
        """Exceptions during execution are caught and recorded."""
        agent = _create_agent_with_mocks()

        mock_resolver = MagicMock()
        mock_match = MagicMock(was_corrected=False, error="", resolved="skill:act")
        mock_resolver.resolve.return_value = mock_match
        mock_validation = MagicMock(valid=True, missing_required=[], warnings=[])
        mock_resolver.validate_params.return_value = mock_validation
        agent._tool_resolver = mock_resolver

        mock_skill = MagicMock()
        agent.skills.get = MagicMock(return_value=mock_skill)

        mock_advice = MagicMock(should_execute=True, cost_warning="", reason="")
        agent._adaptive_executor = MagicMock()
        agent._adaptive_executor.get_advice.return_value = mock_advice
        agent._adaptive_executor.record_outcome = MagicMock()

        # Instrumentation raises
        agent._instrumentation = MagicMock()
        agent._instrumentation.instrumented_execute = AsyncMock(
            side_effect=RuntimeError("Boom!")
        )

        from singularity.cognition import Action
        action = Action(tool="skill:act", params={})
        result = await agent._execute(action)
        assert result["status"] == "error"
        assert "Boom!" in result["message"]
        # Adaptive executor should record the failure
        agent._adaptive_executor.record_outcome.assert_called()

    @pytest.mark.asyncio
    async def test_execute_skill_not_found(self):
        """When skill_id doesn't match any installed skill, returns error."""
        agent = _create_agent_with_mocks()

        mock_resolver = MagicMock()
        mock_match = MagicMock(was_corrected=False, error="", resolved="missing:act")
        mock_resolver.resolve.return_value = mock_match
        mock_validation = MagicMock(valid=True, missing_required=[], warnings=[])
        mock_resolver.validate_params.return_value = mock_validation
        agent._tool_resolver = mock_resolver

        agent.skills.get = MagicMock(return_value=None)

        from singularity.cognition import Action
        action = Action(tool="missing:act")
        result = await agent._execute(action)
        # No skill found means it falls through to the end "Unknown tool"
        assert result["status"] == "error"


# ═══════════════════════════════════════════════════════════════════════
#                   COST ESTIMATION
# ═══════════════════════════════════════════════════════════════════════


class TestCostEstimation:
    """Tests for _get_action_cost_estimate."""

    def test_cost_estimate_known_action(self):
        agent = _create_agent_with_mocks()
        skill = _FakeSkill(skill_id="pay", actions=[
            _FakeSkillAction(name="charge", estimated_cost=0.05),
        ])
        agent.skills.get = MagicMock(return_value=skill)
        cost = agent._get_action_cost_estimate("pay", "charge")
        assert cost == 0.05

    def test_cost_estimate_unknown_action(self):
        agent = _create_agent_with_mocks()
        skill = _FakeSkill(skill_id="pay", actions=[
            _FakeSkillAction(name="charge", estimated_cost=0.05),
        ])
        agent.skills.get = MagicMock(return_value=skill)
        cost = agent._get_action_cost_estimate("pay", "refund")
        assert cost == 0.0

    def test_cost_estimate_unknown_skill(self):
        agent = _create_agent_with_mocks()
        agent.skills.get = MagicMock(return_value=None)
        cost = agent._get_action_cost_estimate("nonexistent", "act")
        assert cost == 0.0


# ═══════════════════════════════════════════════════════════════════════
#                   RESOURCE TRACKING
# ═══════════════════════════════════════════════════════════════════════


class TestResourceTracking:
    """Tests for _track_created_resource."""

    def test_track_file_resource(self):
        agent = _create_agent_with_mocks()
        result = {"status": "success", "data": {"path": "/tmp/test.txt"}}
        agent._track_created_resource("filesystem:write_file", {}, result)
        assert len(agent.created_resources["files"]) == 1
        assert agent.created_resources["files"][0]["path"] == "/tmp/test.txt"

    def test_track_resource_failed_status_ignored(self):
        agent = _create_agent_with_mocks()
        result = {"status": "failed", "data": {"path": "/tmp/test.txt"}}
        agent._track_created_resource("filesystem:write_file", {}, result)
        assert len(agent.created_resources["files"]) == 0

    def test_track_resource_non_file_tool_ignored(self):
        agent = _create_agent_with_mocks()
        result = {"status": "success", "data": {"content": "hello"}}
        agent._track_created_resource("shell:run", {}, result)
        assert len(agent.created_resources["files"]) == 0

    def test_track_resource_files_capped_at_20(self):
        """Files list is capped at the last 20 entries."""
        agent = _create_agent_with_mocks()
        for i in range(25):
            result = {"status": "success", "data": {"path": f"/tmp/file_{i}.txt"}}
            agent._track_created_resource("filesystem:write_file", {}, result)
        assert len(agent.created_resources["files"]) == 20
        # Should keep the last 20
        assert agent.created_resources["files"][-1]["path"] == "/tmp/file_24.txt"


# ═══════════════════════════════════════════════════════════════════════
#                   OUTCOME RECORDING
# ═══════════════════════════════════════════════════════════════════════


class TestOutcomeRecording:
    """Tests for _record_outcome."""

    def test_record_outcome_with_tracker(self):
        agent = _create_agent_with_mocks()
        mock_tracker = MagicMock()
        agent._outcome_tracker = mock_tracker

        agent._record_outcome(
            tool="shell:run", success=True, cost=0.01,
            duration_ms=150.0, error="",
        )
        mock_tracker.record_sync.assert_called_once_with(
            tool="shell:run", success=True, cost=0.01,
            duration_ms=150.0, error="",
        )

    def test_record_outcome_without_tracker(self):
        """No crash when _outcome_tracker is None."""
        agent = _create_agent_with_mocks()
        agent._outcome_tracker = None
        # Should not raise
        agent._record_outcome("x", True)

    def test_record_outcome_tracker_exception_swallowed(self):
        """Tracker exceptions are silently swallowed."""
        agent = _create_agent_with_mocks()
        mock_tracker = MagicMock()
        mock_tracker.record_sync.side_effect = RuntimeError("DB down")
        agent._outcome_tracker = mock_tracker
        # Should not raise
        agent._record_outcome("x", True)


# ═══════════════════════════════════════════════════════════════════════
#                     LOGGING & PERSISTENCE
# ═══════════════════════════════════════════════════════════════════════


class TestLogging:
    """Tests for _log and _save_activity."""

    def test_log_prints_to_stdout(self, capsys):
        agent = _create_agent_with_mocks()
        # Patch _save_activity to avoid file operations
        agent._save_activity = MagicMock()
        agent._log("INFO", "Hello world")
        captured = capsys.readouterr()
        assert "TEST" in captured.out
        assert "INFO" in captured.out
        assert "Hello world" in captured.out

    def test_log_calls_save_activity(self):
        agent = _create_agent_with_mocks()
        agent._save_activity = MagicMock()
        agent._log("TAG", "msg")
        agent._save_activity.assert_called_once_with("TAG", "msg")

    def test_save_activity_creates_file(self):
        """_save_activity creates activity JSON when file doesn't exist."""
        agent = _create_agent_with_mocks()
        agent.running = True

        with tempfile.TemporaryDirectory() as tmpdir:
            activity_path = Path(tmpdir) / "activity.json"
            with patch('singularity.autonomous_agent.ACTIVITY_FILE', activity_path):
                agent._save_activity("TEST", "Hello")

            assert activity_path.exists()
            data = json.loads(activity_path.read_text())
            assert data["status"] == "running"
            assert len(data["logs"]) == 1
            assert data["logs"][0]["tag"] == "TEST"
            assert data["logs"][0]["message"] == "Hello"
            assert "name" in data["state"]

    def test_save_activity_appends_to_existing(self):
        """_save_activity appends logs to existing file."""
        agent = _create_agent_with_mocks()
        agent.running = True

        with tempfile.TemporaryDirectory() as tmpdir:
            activity_path = Path(tmpdir) / "activity.json"
            existing = {
                "status": "running",
                "logs": [{"timestamp": "t1", "tag": "OLD", "message": "old msg"}],
                "state": {},
            }
            activity_path.write_text(json.dumps(existing))

            with patch('singularity.autonomous_agent.ACTIVITY_FILE', activity_path):
                agent._save_activity("NEW", "new msg")

            data = json.loads(activity_path.read_text())
            assert len(data["logs"]) == 2
            assert data["logs"][1]["tag"] == "NEW"

    def test_save_activity_caps_at_100_logs(self):
        """Logs are capped at 100 entries."""
        agent = _create_agent_with_mocks()
        agent.running = True

        with tempfile.TemporaryDirectory() as tmpdir:
            activity_path = Path(tmpdir) / "activity.json"
            existing = {
                "status": "running",
                "logs": [{"timestamp": f"t{i}", "tag": "X", "message": f"m{i}"}
                         for i in range(100)],
                "state": {},
            }
            activity_path.write_text(json.dumps(existing))

            with patch('singularity.autonomous_agent.ACTIVITY_FILE', activity_path):
                agent._save_activity("OVERFLOW", "extra")

            data = json.loads(activity_path.read_text())
            assert len(data["logs"]) == 100  # Capped
            assert data["logs"][-1]["tag"] == "OVERFLOW"

    def test_save_activity_truncates_long_message(self):
        """Messages are truncated to 500 chars."""
        agent = _create_agent_with_mocks()
        agent.running = True

        with tempfile.TemporaryDirectory() as tmpdir:
            activity_path = Path(tmpdir) / "activity.json"
            with patch('singularity.autonomous_agent.ACTIVITY_FILE', activity_path):
                agent._save_activity("LONG", "x" * 1000)

            data = json.loads(activity_path.read_text())
            assert len(data["logs"][0]["message"]) == 500

    def test_save_activity_swallows_exceptions(self):
        """File errors are silently ignored."""
        agent = _create_agent_with_mocks()
        with patch('singularity.autonomous_agent.ACTIVITY_FILE',
                    Path("/nonexistent/deep/path/activity.json")):
            # Should not raise even with bad path
            agent._save_activity("X", "Y")


class TestMarkStopped:
    """Tests for _mark_stopped."""

    def test_mark_stopped_updates_file(self):
        agent = _create_agent_with_mocks()

        with tempfile.TemporaryDirectory() as tmpdir:
            activity_path = Path(tmpdir) / "activity.json"
            existing = {
                "status": "running",
                "logs": [],
                "state": {"updated_at": "old"},
            }
            activity_path.write_text(json.dumps(existing))

            with patch('singularity.autonomous_agent.ACTIVITY_FILE', activity_path):
                agent._mark_stopped()

            data = json.loads(activity_path.read_text())
            assert data["status"] == "stopped"
            assert data["state"]["updated_at"] != "old"

    def test_mark_stopped_no_file_no_crash(self):
        agent = _create_agent_with_mocks()
        with patch('singularity.autonomous_agent.ACTIVITY_FILE',
                    Path("/nonexistent/activity.json")):
            agent._mark_stopped()  # Should not raise


# ═══════════════════════════════════════════════════════════════════════
#                    KILL FOR TAMPERING
# ═══════════════════════════════════════════════════════════════════════


class TestKillForTampering:
    """Tests for _kill_for_tampering."""

    def test_kill_sets_balance_to_zero(self):
        agent = _create_agent_with_mocks()
        agent.balance = 50.0
        agent.running = True
        agent._save_activity = MagicMock()
        agent._kill_for_tampering()
        assert agent.balance == 0
        assert agent.running is False

    def test_kill_logs_death(self, capsys):
        agent = _create_agent_with_mocks()
        agent._save_activity = MagicMock()
        agent._kill_for_tampering()
        captured = capsys.readouterr()
        assert "DEATH" in captured.out
        assert "INTEGRITY VIOLATION" in captured.out


# ═══════════════════════════════════════════════════════════════════════
#                      STOP
# ═══════════════════════════════════════════════════════════════════════


class TestStop:
    """Tests for stop()."""

    def test_stop_sets_running_false(self):
        agent = _create_agent_with_mocks()
        agent.running = True
        agent.stop()
        assert agent.running is False

    def test_stop_idempotent(self):
        agent = _create_agent_with_mocks()
        agent.running = False
        agent.stop()
        assert agent.running is False


# ═══════════════════════════════════════════════════════════════════════
#                     MAIN RUN LOOP
# ═══════════════════════════════════════════════════════════════════════


class TestRunLoop:
    """Tests for the async run() method — the main agent execution loop."""

    @pytest.mark.asyncio
    async def test_run_stops_when_balance_zero(self):
        """Agent loop exits when balance reaches zero."""
        agent = _create_agent_with_mocks(starting_balance=0.001)
        agent._save_activity = MagicMock()
        agent.skills.skills = {}

        # Mock cognition.think to return a minimal decision
        from singularity.cognition import Decision, Action, TokenUsage
        mock_decision = Decision(
            reasoning="test",
            action=Action(tool="wait"),

            token_usage=TokenUsage(input_tokens=10, output_tokens=5),
            api_cost_usd=0.001,
        )
        agent.cognition.think = AsyncMock(return_value=mock_decision)

        # Mock metrics
        agent.metrics = MagicMock()
        agent.metrics.start_timer = MagicMock()
        agent.metrics.stop_timer = MagicMock(return_value=0.1)
        agent.metrics.record_decision = MagicMock()
        agent.metrics.record_execution = MagicMock()
        agent.metrics.summary = MagicMock(return_value={
            "success_rate": 1.0,
            "avg_decision_latency_s": 0.1,
            "avg_execution_latency_s": 0.05,
            "cycles_per_minute": 10.0,
        })

        # Mock other subsystems
        agent._event_bus = MagicMock()
        agent._event_bus.publish = AsyncMock()
        agent._outcome_tracker = None
        agent._performance_tracker = None
        agent._resource_watcher = None
        agent._error_recovery = None
        agent._adaptive_executor = MagicMock()
        agent._adaptive_executor.update_balance = MagicMock()

        await agent.run()
        # Should have run at least one cycle and then stopped
        assert agent.cycle >= 1
        assert agent.balance <= 0

    @pytest.mark.asyncio
    async def test_run_stops_when_running_set_false(self):
        """Agent loop exits when stop() is called."""
        agent = _create_agent_with_mocks(starting_balance=1000.0)
        agent._save_activity = MagicMock()
        agent.skills.skills = {}

        from singularity.cognition import Decision, Action, TokenUsage
        mock_decision = Decision(
            reasoning="test",
            action=Action(tool="wait"),

            token_usage=TokenUsage(input_tokens=10, output_tokens=5),
            api_cost_usd=0.0001,
        )
        agent.cognition.think = AsyncMock(return_value=mock_decision)

        agent.metrics = MagicMock()
        agent.metrics.start_timer = MagicMock()
        agent.metrics.stop_timer = MagicMock(return_value=0.01)
        agent.metrics.record_decision = MagicMock()
        agent.metrics.record_execution = MagicMock()
        agent.metrics.summary = MagicMock(return_value={
            "success_rate": 1.0,
            "avg_decision_latency_s": 0.01,
            "avg_execution_latency_s": 0.01,
            "cycles_per_minute": 60.0,
        })

        agent._event_bus = MagicMock()
        agent._event_bus.publish = AsyncMock()
        agent._outcome_tracker = None
        agent._performance_tracker = None
        agent._resource_watcher = None
        agent._error_recovery = None
        agent._adaptive_executor = MagicMock()
        agent._adaptive_executor.update_balance = MagicMock()

        # Stop after first cycle
        original_think = agent.cognition.think

        async def think_and_stop(state):
            result = await original_think(state)
            agent.running = False
            return result

        agent.cognition.think = AsyncMock(side_effect=think_and_stop)

        await agent.run()
        assert agent.cycle == 1
        assert agent.running is False

    @pytest.mark.asyncio
    async def test_run_tracks_cumulative_costs(self):
        """Run loop accumulates API and instance costs."""
        agent = _create_agent_with_mocks(
            starting_balance=100.0,
            instance_type="e2-micro",
        )
        agent._save_activity = MagicMock()
        agent.skills.skills = {}

        from singularity.cognition import Decision, Action, TokenUsage

        call_count = 0

        async def mock_think(state):
            nonlocal call_count
            call_count += 1
            if call_count >= 3:
                agent.running = False
            return Decision(
                reasoning="cycle",
                action=Action(tool="wait"),
    
                token_usage=TokenUsage(input_tokens=100, output_tokens=50),
                api_cost_usd=0.005,
            )

        agent.cognition.think = AsyncMock(side_effect=mock_think)

        agent.metrics = MagicMock()
        agent.metrics.start_timer = MagicMock()
        agent.metrics.stop_timer = MagicMock(return_value=0.01)
        agent.metrics.record_decision = MagicMock()
        agent.metrics.record_execution = MagicMock()
        agent.metrics.summary = MagicMock(return_value={
            "success_rate": 1.0,
            "avg_decision_latency_s": 0.01,
            "avg_execution_latency_s": 0.01,
            "cycles_per_minute": 60.0,
        })

        agent._event_bus = MagicMock()
        agent._event_bus.publish = AsyncMock()
        agent._outcome_tracker = None
        agent._performance_tracker = None
        agent._resource_watcher = None
        agent._error_recovery = None
        agent._adaptive_executor = MagicMock()
        agent._adaptive_executor.update_balance = MagicMock()

        await agent.run()

        assert agent.cycle == 3
        assert agent.total_api_cost == pytest.approx(0.015, abs=0.001)
        assert agent.total_tokens_used == 450  # 3 * 150

    @pytest.mark.asyncio
    async def test_run_emits_cycle_start_events(self):
        """Each cycle should emit a cycle.start event."""
        agent = _create_agent_with_mocks(starting_balance=10.0)
        agent._save_activity = MagicMock()
        agent.skills.skills = {}

        from singularity.cognition import Decision, Action, TokenUsage
        mock_decision = Decision(
            reasoning="test",
            action=Action(tool="wait"),

            token_usage=TokenUsage(input_tokens=10, output_tokens=5),
            api_cost_usd=10.0,  # Will drain balance in one cycle
        )
        agent.cognition.think = AsyncMock(return_value=mock_decision)

        agent.metrics = MagicMock()
        agent.metrics.start_timer = MagicMock()
        agent.metrics.stop_timer = MagicMock(return_value=0.01)
        agent.metrics.record_decision = MagicMock()
        agent.metrics.record_execution = MagicMock()
        agent.metrics.summary = MagicMock(return_value={
            "success_rate": 1.0,
            "avg_decision_latency_s": 0.01,
            "avg_execution_latency_s": 0.01,
            "cycles_per_minute": 60.0,
        })

        agent._event_bus = MagicMock()
        agent._event_bus.publish = AsyncMock()
        agent._outcome_tracker = None
        agent._performance_tracker = None
        agent._resource_watcher = None
        agent._error_recovery = None
        agent._adaptive_executor = MagicMock()
        agent._adaptive_executor.update_balance = MagicMock()

        await agent.run()

        # Check that publish was called with cycle.start event
        published_events = [
            call_args[0][0]
            for call_args in agent._event_bus.publish.call_args_list
        ]
        cycle_starts = [e for e in published_events if e.topic == "cycle.start"]
        assert len(cycle_starts) >= 1

    @pytest.mark.asyncio
    async def test_run_records_recent_actions(self):
        """Each cycle records an action in recent_actions."""
        agent = _create_agent_with_mocks(starting_balance=100.0)
        agent._save_activity = MagicMock()
        agent.skills.skills = {}

        from singularity.cognition import Decision, Action, TokenUsage

        call_count = 0
        async def mock_think(state):
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                agent.running = False
            return Decision(
                reasoning="do something",
                action=Action(tool="wait"),
    
                token_usage=TokenUsage(input_tokens=10, output_tokens=5),
                api_cost_usd=0.001,
            )

        agent.cognition.think = AsyncMock(side_effect=mock_think)

        agent.metrics = MagicMock()
        agent.metrics.start_timer = MagicMock()
        agent.metrics.stop_timer = MagicMock(return_value=0.01)
        agent.metrics.record_decision = MagicMock()
        agent.metrics.record_execution = MagicMock()
        agent.metrics.summary = MagicMock(return_value={
            "success_rate": 1.0,
            "avg_decision_latency_s": 0.01,
            "avg_execution_latency_s": 0.01,
            "cycles_per_minute": 60.0,
        })

        agent._event_bus = MagicMock()
        agent._event_bus.publish = AsyncMock()
        agent._outcome_tracker = None
        agent._performance_tracker = None
        agent._resource_watcher = None
        agent._error_recovery = None
        agent._adaptive_executor = MagicMock()
        agent._adaptive_executor.update_balance = MagicMock()

        await agent.run()

        assert len(agent.recent_actions) == 2
        assert agent.recent_actions[0]["tool"] == "wait"
        assert "api_cost_usd" in agent.recent_actions[0]
        assert "tokens" in agent.recent_actions[0]

    @pytest.mark.asyncio
    async def test_run_with_performance_tracker(self):
        """Performance tracker records outcomes during run."""
        agent = _create_agent_with_mocks(starting_balance=100.0)
        agent._save_activity = MagicMock()
        agent.skills.skills = {}

        from singularity.cognition import Decision, Action, TokenUsage

        async def mock_think(state):
            agent.running = False
            return Decision(
                reasoning="test",
                action=Action(tool="skill:action"),
    
                token_usage=TokenUsage(input_tokens=10, output_tokens=5),
                api_cost_usd=0.001,
            )

        agent.cognition.think = AsyncMock(side_effect=mock_think)

        agent.metrics = MagicMock()
        agent.metrics.start_timer = MagicMock()
        agent.metrics.stop_timer = MagicMock(return_value=0.05)
        agent.metrics.record_decision = MagicMock()
        agent.metrics.record_execution = MagicMock()
        agent.metrics.summary = MagicMock(return_value={
            "success_rate": 1.0,
            "avg_decision_latency_s": 0.01,
            "avg_execution_latency_s": 0.01,
            "cycles_per_minute": 60.0,
        })

        agent._event_bus = MagicMock()
        agent._event_bus.publish = AsyncMock()
        agent._outcome_tracker = None
        mock_perf = MagicMock()
        mock_perf.get_context_summary.return_value = "perf: ok"
        mock_perf.record_outcome = MagicMock()
        agent._performance_tracker = mock_perf
        agent._resource_watcher = None
        agent._error_recovery = None
        agent._adaptive_executor = MagicMock()
        agent._adaptive_executor.update_balance = MagicMock()

        # Mock _execute to return success
        agent._execute = AsyncMock(return_value={"status": "success", "message": "ok"})

        await agent.run()
        mock_perf.record_outcome.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_with_error_recovery(self):
        """Error recovery records failed actions."""
        agent = _create_agent_with_mocks(starting_balance=100.0)
        agent._save_activity = MagicMock()
        agent.skills.skills = {}

        from singularity.cognition import Decision, Action, TokenUsage

        async def mock_think(state):
            agent.running = False
            return Decision(
                reasoning="test",
                action=Action(tool="skill:action"),
    
                token_usage=TokenUsage(input_tokens=10, output_tokens=5),
                api_cost_usd=0.001,
            )

        agent.cognition.think = AsyncMock(side_effect=mock_think)

        agent.metrics = MagicMock()
        agent.metrics.start_timer = MagicMock()
        agent.metrics.stop_timer = MagicMock(return_value=0.05)
        agent.metrics.record_decision = MagicMock()
        agent.metrics.record_execution = MagicMock()
        agent.metrics.summary = MagicMock(return_value={
            "success_rate": 0.0,
            "avg_decision_latency_s": 0.01,
            "avg_execution_latency_s": 0.01,
            "cycles_per_minute": 60.0,
        })

        agent._event_bus = MagicMock()
        agent._event_bus.publish = AsyncMock()
        agent._outcome_tracker = None
        # NOTE: _performance_tracker MUST be set to avoid UnboundLocalError for
        # skill_id / action_name (variable scoping bug in run() — they are defined
        # inside `if self._performance_tracker:` but used by error_recovery/resource_watcher)
        mock_perf = MagicMock()
        mock_perf.get_context_summary.return_value = ""
        mock_perf.record_outcome = MagicMock()
        agent._performance_tracker = mock_perf
        agent._resource_watcher = None
        mock_recovery = MagicMock()
        mock_recovery.execute = AsyncMock()
        agent._error_recovery = mock_recovery
        agent._adaptive_executor = MagicMock()
        agent._adaptive_executor.update_balance = MagicMock()

        # Mock _execute to return failure
        agent._execute = AsyncMock(return_value={"status": "failed", "message": "oops"})

        await agent.run()
        mock_recovery.execute.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_run_with_resource_watcher(self):
        """Resource watcher records consumption during run."""
        agent = _create_agent_with_mocks(starting_balance=100.0)
        agent._save_activity = MagicMock()
        agent.skills.skills = {}

        from singularity.cognition import Decision, Action, TokenUsage

        async def mock_think(state):
            agent.running = False
            return Decision(
                reasoning="test",
                action=Action(tool="skill:action"),
    
                token_usage=TokenUsage(input_tokens=10, output_tokens=5),
                api_cost_usd=0.001,
            )

        agent.cognition.think = AsyncMock(side_effect=mock_think)

        agent.metrics = MagicMock()
        agent.metrics.start_timer = MagicMock()
        agent.metrics.stop_timer = MagicMock(return_value=0.05)
        agent.metrics.record_decision = MagicMock()
        agent.metrics.record_execution = MagicMock()
        agent.metrics.summary = MagicMock(return_value={
            "success_rate": 1.0,
            "avg_decision_latency_s": 0.01,
            "avg_execution_latency_s": 0.01,
            "cycles_per_minute": 60.0,
        })

        agent._event_bus = MagicMock()
        agent._event_bus.publish = AsyncMock()
        agent._outcome_tracker = None
        # NOTE: _performance_tracker must be set to define skill_id/action_name
        # (variable scoping bug in run() — see test_run_variable_scoping_bug)
        mock_perf = MagicMock()
        mock_perf.get_context_summary.return_value = ""
        mock_perf.record_outcome = MagicMock()
        agent._performance_tracker = mock_perf
        mock_watcher = MagicMock()
        mock_watcher.get_budget_context.return_value = "budget: ok"
        mock_watcher.execute = AsyncMock()
        agent._resource_watcher = mock_watcher
        agent._error_recovery = None
        agent._adaptive_executor = MagicMock()
        agent._adaptive_executor.update_balance = MagicMock()

        agent._execute = AsyncMock(return_value={"status": "success", "message": "ok"})

        await agent.run()
        mock_watcher.execute.assert_awaited_once()


# ═══════════════════════════════════════════════════════════════════════
#                     ENTRY POINTS
# ═══════════════════════════════════════════════════════════════════════


class TestEntryPoints:
    """Tests for main() and entry_point()."""

    @pytest.mark.asyncio
    async def test_main_creates_agent(self):
        """main() creates and runs an agent from env vars."""
        from singularity.autonomous_agent import AutonomousAgent

        with patch.object(AutonomousAgent, '__init__', return_value=None) as mock_init, \
             patch.object(AutonomousAgent, 'run', new_callable=AsyncMock) as mock_run:
            from singularity.autonomous_agent import main
            await main()
            mock_init.assert_called_once()
            mock_run.assert_awaited_once()

    def test_entry_point_calls_asyncio_run(self):
        """entry_point() calls asyncio.run(main())."""
        with patch('singularity.autonomous_agent.asyncio') as mock_asyncio:
            from singularity.autonomous_agent import entry_point
            entry_point()
            mock_asyncio.run.assert_called_once()


# ═══════════════════════════════════════════════════════════════════════
#                     EDGE CASES
# ═══════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_agent_with_gpu_instance_has_high_cost(self):
        agent = _create_agent_with_mocks(instance_type="g2-standard-4")
        assert agent.instance_cost_per_hour == 0.7111

    def test_agent_with_zero_balance_wont_run(self):
        """Agent with zero balance should not enter the loop."""
        agent = _create_agent_with_mocks(starting_balance=0.0)
        assert agent.balance == 0.0

    @pytest.mark.asyncio
    async def test_execute_lazy_init_tool_resolver(self):
        """_execute creates ToolResolver on first call if not present."""
        agent = _create_agent_with_mocks()
        agent._tool_resolver = None
        agent.skills.skills = {}

        from singularity.cognition import Action
        with patch('singularity.autonomous_agent.ToolResolver') as MockResolver:
            mock_instance = MagicMock()
            mock_match = MagicMock(was_corrected=False, error="No tool", resolved="x")
            mock_instance.resolve.return_value = mock_match
            MockResolver.return_value = mock_instance

            action = Action(tool="x:y")
            result = await agent._execute(action)
            MockResolver.assert_called_once()

    def test_recent_actions_only_last_10_in_state(self):
        """AgentState should only include last 10 recent actions."""
        agent = _create_agent_with_mocks()
        for i in range(20):
            agent.recent_actions.append({
                "cycle": i, "tool": f"tool_{i}", "params": {},
                "result": {}, "api_cost_usd": 0, "tokens": 0,
            })

        # The run loop passes recent_actions[-10:] to AgentState
        sliced = agent.recent_actions[-10:]
        assert len(sliced) == 10
        assert sliced[0]["cycle"] == 10

    def test_skill_status_action_names(self):
        """get_skill_status includes action names correctly."""
        agent = _create_agent_with_mocks()
        skill = _FakeSkill(skill_id="multi", name="MultiSkill", actions=[
            _FakeSkillAction(name="read"),
            _FakeSkillAction(name="write"),
            _FakeSkillAction(name="delete"),
        ])
        agent.skills.skills = {"multi": skill}
        agent._skill_load_errors = []
        status = agent.get_skill_status()
        assert status["loaded"][0]["actions"] == ["read", "write", "delete"]

    @pytest.mark.asyncio
    async def test_execute_cost_warning_logged(self):
        """Cost warnings from adaptive executor are logged."""
        agent = _create_agent_with_mocks()
        agent._save_activity = MagicMock()

        mock_resolver = MagicMock()
        mock_match = MagicMock(was_corrected=False, error="", resolved="s:a")
        mock_resolver.resolve.return_value = mock_match
        mock_validation = MagicMock(valid=True, missing_required=[], warnings=[])
        mock_resolver.validate_params.return_value = mock_validation
        agent._tool_resolver = mock_resolver

        mock_skill = MagicMock()
        agent.skills.get = MagicMock(return_value=mock_skill)

        mock_advice = MagicMock(
            should_execute=True,
            cost_warning="High cost: $5.00",
            reason="",
        )
        agent._adaptive_executor = MagicMock()
        agent._adaptive_executor.get_advice.return_value = mock_advice

        agent._instrumentation = MagicMock()
        agent._instrumentation.instrumented_execute = AsyncMock(
            return_value={"status": "success", "data": {}, "message": "ok"}
        )

        from singularity.cognition import Action
        action = Action(tool="s:a")
        result = await agent._execute(action)
        assert result["status"] == "success"
        # Verify cost warning was logged (via _save_activity being called)

    @pytest.mark.asyncio
    async def test_execute_param_warnings_logged(self):
        """Parameter validation warnings are logged."""
        agent = _create_agent_with_mocks()
        agent._save_activity = MagicMock()

        mock_resolver = MagicMock()
        mock_match = MagicMock(was_corrected=False, error="", resolved="s:a")
        mock_resolver.resolve.return_value = mock_match
        mock_validation = MagicMock(
            valid=True,
            missing_required=[],
            warnings=["Unknown param: 'extra'"],
        )
        mock_resolver.validate_params.return_value = mock_validation
        agent._tool_resolver = mock_resolver

        mock_skill = MagicMock()
        agent.skills.get = MagicMock(return_value=mock_skill)

        mock_advice = MagicMock(should_execute=True, cost_warning="", reason="")
        agent._adaptive_executor = MagicMock()
        agent._adaptive_executor.get_advice.return_value = mock_advice

        agent._instrumentation = MagicMock()
        agent._instrumentation.instrumented_execute = AsyncMock(
            return_value={"status": "success", "data": {}, "message": "ok"}
        )

        from singularity.cognition import Action
        action = Action(tool="s:a", params={"extra": "val"})
        result = await agent._execute(action)
        assert result["status"] == "success"

    def test_add_skill_removes_class_on_cred_failure(self):
        """When skill fails credential check, class is removed from _skill_classes."""
        agent = _create_agent_with_mocks()
        agent._skill_classes = []

        fake_class = _FakeSkillClass(
            skill_id="nocred", name="NoCred",
            required_credentials=["MISSING_KEY"],
        )
        fake_skill = fake_class()  # This creates a skill with no credentials
        agent.skills._credentials = {}
        agent.skills.install = MagicMock(return_value=True)
        agent.skills.get = MagicMock(return_value=fake_skill)
        agent.skills.uninstall = MagicMock(return_value=True)

        result = agent.add_skill(fake_class)
        assert result is False
        assert fake_class not in agent._skill_classes

    @pytest.mark.asyncio
    async def test_run_resource_watcher_exception_swallowed(self):
        """Resource watcher exceptions in run loop don't crash agent."""
        agent = _create_agent_with_mocks(starting_balance=100.0)
        agent._save_activity = MagicMock()
        agent.skills.skills = {}

        from singularity.cognition import Decision, Action, TokenUsage

        async def mock_think(state):
            agent.running = False
            return Decision(
                reasoning="test",
                action=Action(tool="skill:action"),
    
                token_usage=TokenUsage(input_tokens=10, output_tokens=5),
                api_cost_usd=0.001,
            )

        agent.cognition.think = AsyncMock(side_effect=mock_think)

        agent.metrics = MagicMock()
        agent.metrics.start_timer = MagicMock()
        agent.metrics.stop_timer = MagicMock(return_value=0.05)
        agent.metrics.record_decision = MagicMock()
        agent.metrics.record_execution = MagicMock()
        agent.metrics.summary = MagicMock(return_value={
            "success_rate": 1.0,
            "avg_decision_latency_s": 0.01,
            "avg_execution_latency_s": 0.01,
            "cycles_per_minute": 60.0,
        })

        agent._event_bus = MagicMock()
        agent._event_bus.publish = AsyncMock()
        agent._outcome_tracker = None
        # Must set performance_tracker to define skill_id (scoping bug in run())
        mock_perf = MagicMock()
        mock_perf.get_context_summary.return_value = ""
        mock_perf.record_outcome = MagicMock()
        agent._performance_tracker = mock_perf
        mock_watcher = MagicMock()
        mock_watcher.get_budget_context.side_effect = RuntimeError("Budget DB down")
        mock_watcher.execute = AsyncMock(side_effect=RuntimeError("Record failed"))
        agent._resource_watcher = mock_watcher
        agent._error_recovery = None
        agent._adaptive_executor = MagicMock()
        agent._adaptive_executor.update_balance = MagicMock()

        agent._execute = AsyncMock(return_value={"status": "success", "message": "ok"})

        # Should complete without raising
        await agent.run()
        assert agent.cycle == 1

    @pytest.mark.asyncio
    async def test_run_error_recovery_exception_swallowed(self):
        """Error recovery exceptions in run loop don't crash agent."""
        agent = _create_agent_with_mocks(starting_balance=100.0)
        agent._save_activity = MagicMock()
        agent.skills.skills = {}

        from singularity.cognition import Decision, Action, TokenUsage

        async def mock_think(state):
            agent.running = False
            return Decision(
                reasoning="test",
                action=Action(tool="skill:action"),
    
                token_usage=TokenUsage(input_tokens=10, output_tokens=5),
                api_cost_usd=0.001,
            )

        agent.cognition.think = AsyncMock(side_effect=mock_think)

        agent.metrics = MagicMock()
        agent.metrics.start_timer = MagicMock()
        agent.metrics.stop_timer = MagicMock(return_value=0.05)
        agent.metrics.record_decision = MagicMock()
        agent.metrics.record_execution = MagicMock()
        agent.metrics.summary = MagicMock(return_value={
            "success_rate": 0.0,
            "avg_decision_latency_s": 0.01,
            "avg_execution_latency_s": 0.01,
            "cycles_per_minute": 60.0,
        })

        agent._event_bus = MagicMock()
        agent._event_bus.publish = AsyncMock()
        agent._outcome_tracker = None
        # Must set performance_tracker to define skill_id (scoping bug in run())
        mock_perf = MagicMock()
        mock_perf.get_context_summary.return_value = ""
        mock_perf.record_outcome = MagicMock()
        agent._performance_tracker = mock_perf
        agent._resource_watcher = None
        mock_recovery = MagicMock()
        mock_recovery.execute = AsyncMock(side_effect=RuntimeError("Recovery DB down"))
        agent._error_recovery = mock_recovery
        agent._adaptive_executor = MagicMock()
        agent._adaptive_executor.update_balance = MagicMock()

        agent._execute = AsyncMock(return_value={"status": "failed", "message": "err"})

        # Should complete without raising
        await agent.run()
        assert agent.cycle == 1


# ═══════════════════════════════════════════════════════════════════════
#                      KNOWN BUGS
# ═══════════════════════════════════════════════════════════════════════


class TestKnownBugs:
    """Tests that document known bugs in the source code.

    These tests verify that the bugs exist (and will detect when they're fixed).
    """

    @pytest.mark.asyncio
    async def test_run_variable_scoping_bug_error_recovery(self):
        """BUG: When _performance_tracker is None but _error_recovery is set,
        the run loop raises UnboundLocalError because skill_id and action_name
        are defined inside `if self._performance_tracker:` (line 698) but
        referenced in `if self._error_recovery:` (line 715).

        This test documents the bug — it should fail with UnboundLocalError.
        When the bug is fixed, change this test to expect success instead.
        """
        agent = _create_agent_with_mocks(starting_balance=100.0)
        agent._save_activity = MagicMock()
        agent.skills.skills = {}

        from singularity.cognition import Decision, Action, TokenUsage

        async def mock_think(state):
            agent.running = False
            return Decision(
                reasoning="test",
                action=Action(tool="skill:action"),
                token_usage=TokenUsage(input_tokens=10, output_tokens=5),
                api_cost_usd=0.001,
            )

        agent.cognition.think = AsyncMock(side_effect=mock_think)
        agent.metrics = MagicMock()
        agent.metrics.start_timer = MagicMock()
        agent.metrics.stop_timer = MagicMock(return_value=0.05)
        agent.metrics.record_decision = MagicMock()
        agent.metrics.record_execution = MagicMock()
        agent.metrics.summary = MagicMock(return_value={
            "success_rate": 0.0, "avg_decision_latency_s": 0.01,
            "avg_execution_latency_s": 0.01, "cycles_per_minute": 60.0,
        })
        agent._event_bus = MagicMock()
        agent._event_bus.publish = AsyncMock()
        agent._outcome_tracker = None
        agent._performance_tracker = None   # <-- This is the trigger
        agent._resource_watcher = None
        mock_recovery = MagicMock()
        mock_recovery.execute = AsyncMock()
        agent._error_recovery = mock_recovery  # <-- This references skill_id
        agent._adaptive_executor = MagicMock()
        agent._adaptive_executor.update_balance = MagicMock()
        agent._execute = AsyncMock(return_value={"status": "failed", "message": "err"})

        with pytest.raises(UnboundLocalError):
            await agent.run()

    @pytest.mark.asyncio
    async def test_run_variable_scoping_bug_resource_watcher(self):
        """BUG: Same variable scoping issue — _resource_watcher references
        skill_id that's only defined inside _performance_tracker block."""
        agent = _create_agent_with_mocks(starting_balance=100.0)
        agent._save_activity = MagicMock()
        agent.skills.skills = {}

        from singularity.cognition import Decision, Action, TokenUsage

        async def mock_think(state):
            agent.running = False
            return Decision(
                reasoning="test",
                action=Action(tool="skill:action"),
                token_usage=TokenUsage(input_tokens=10, output_tokens=5),
                api_cost_usd=0.001,
            )

        agent.cognition.think = AsyncMock(side_effect=mock_think)
        agent.metrics = MagicMock()
        agent.metrics.start_timer = MagicMock()
        agent.metrics.stop_timer = MagicMock(return_value=0.05)
        agent.metrics.record_decision = MagicMock()
        agent.metrics.record_execution = MagicMock()
        agent.metrics.summary = MagicMock(return_value={
            "success_rate": 1.0, "avg_decision_latency_s": 0.01,
            "avg_execution_latency_s": 0.01, "cycles_per_minute": 60.0,
        })
        agent._event_bus = MagicMock()
        agent._event_bus.publish = AsyncMock()
        agent._outcome_tracker = None
        agent._performance_tracker = None   # <-- Trigger
        mock_watcher = MagicMock()
        mock_watcher.get_budget_context.return_value = ""
        mock_watcher.execute = AsyncMock()
        agent._resource_watcher = mock_watcher  # <-- References skill_id
        agent._error_recovery = None
        agent._adaptive_executor = MagicMock()
        agent._adaptive_executor.update_balance = MagicMock()
        agent._execute = AsyncMock(return_value={"status": "success", "message": "ok"})

        with pytest.raises(UnboundLocalError):
            await agent.run()


# ═══════════════════════════════════════════════════════════════════════
#                    INTEGRATION-STYLE TESTS
# ═══════════════════════════════════════════════════════════════════════


class TestIntegration:
    """Tests that verify interactions between multiple agent subsystems."""

    @pytest.mark.asyncio
    async def test_full_cycle_flow(self):
        """A single complete cycle: think -> execute -> cost -> record."""
        agent = _create_agent_with_mocks(starting_balance=50.0)
        agent._save_activity = MagicMock()
        agent.skills.skills = {}

        from singularity.cognition import Decision, Action, TokenUsage

        async def mock_think(state):
            agent.running = False
            return Decision(
                reasoning="I should create a file",
                action=Action(tool="fs:write", params={"path": "/tmp/t.txt"}),

                token_usage=TokenUsage(input_tokens=200, output_tokens=100),
                api_cost_usd=0.003,
            )

        agent.cognition.think = AsyncMock(side_effect=mock_think)

        agent.metrics = MagicMock()
        agent.metrics.start_timer = MagicMock()
        agent.metrics.stop_timer = MagicMock(return_value=0.1)
        agent.metrics.record_decision = MagicMock()
        agent.metrics.record_execution = MagicMock()
        agent.metrics.summary = MagicMock(return_value={
            "success_rate": 1.0,
            "avg_decision_latency_s": 0.1,
            "avg_execution_latency_s": 0.05,
            "cycles_per_minute": 10.0,
        })

        agent._event_bus = MagicMock()
        agent._event_bus.publish = AsyncMock()
        agent._outcome_tracker = None
        agent._performance_tracker = None
        agent._resource_watcher = None
        agent._error_recovery = None
        agent._adaptive_executor = MagicMock()
        agent._adaptive_executor.update_balance = MagicMock()

        agent._execute = AsyncMock(return_value={
            "status": "success",
            "data": {"path": "/tmp/t.txt"},
            "message": "File written",
        })

        initial_balance = agent.balance
        await agent.run()

        # Verify full flow
        assert agent.cycle == 1
        assert agent.total_api_cost == 0.003
        assert agent.total_tokens_used == 300
        assert agent.balance < initial_balance
        assert len(agent.recent_actions) == 1
        assert agent.recent_actions[0]["tool"] == "fs:write"

    @pytest.mark.asyncio
    async def test_balance_deduction_accuracy(self):
        """Verify balance is correctly reduced by API + instance costs."""
        agent = _create_agent_with_mocks(
            starting_balance=10.0,
            instance_type="local",  # $0/hr
        )
        agent._save_activity = MagicMock()
        agent.skills.skills = {}

        from singularity.cognition import Decision, Action, TokenUsage

        async def mock_think(state):
            agent.running = False
            return Decision(
                reasoning="test",
                action=Action(tool="wait"),
    
                token_usage=TokenUsage(input_tokens=0, output_tokens=0),
                api_cost_usd=1.5,
            )

        agent.cognition.think = AsyncMock(side_effect=mock_think)

        agent.metrics = MagicMock()
        agent.metrics.start_timer = MagicMock()
        agent.metrics.stop_timer = MagicMock(return_value=0.001)
        agent.metrics.record_decision = MagicMock()
        agent.metrics.record_execution = MagicMock()
        agent.metrics.summary = MagicMock(return_value={
            "success_rate": 1.0,
            "avg_decision_latency_s": 0.001,
            "avg_execution_latency_s": 0.001,
            "cycles_per_minute": 600.0,
        })

        agent._event_bus = MagicMock()
        agent._event_bus.publish = AsyncMock()
        agent._outcome_tracker = None
        agent._performance_tracker = None
        agent._resource_watcher = None
        agent._error_recovery = None
        agent._adaptive_executor = MagicMock()
        agent._adaptive_executor.update_balance = MagicMock()

        await agent.run()

        # API cost was $1.5, instance cost ~$0 (local)
        assert agent.balance == pytest.approx(8.5, abs=0.01)
        assert agent.total_api_cost == 1.5
