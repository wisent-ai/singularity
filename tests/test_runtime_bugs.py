"""
Tests for runtime bug fixes in the Singularity agent framework.

These tests verify fixes for three production bugs:
1. cognition.py: Vertex AI uses sync client in async context (blocks event loop)
2. orchestrator.py: broadcast/message crash with KeyError on missing message boxes
3. request.py: aiohttp session has no timeout (agents hang indefinitely)

All tests run WITHOUT API keys — they test code structure and logic, not live APIs.
"""

import asyncio
import inspect
import importlib
import sys
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

import pytest


# ---------------------------------------------------------------------------
# 1. Vertex AI async client fix — cognition.py
# ---------------------------------------------------------------------------

class TestVertexAsyncFix:
    """Verify cognition.py uses AsyncAnthropicVertex, not sync AnthropicVertex."""

    def test_imports_async_vertex_client(self):
        """The module should import AsyncAnthropicVertex, not AnthropicVertex."""
        import singularity.cognition as cog
        source = inspect.getsource(cog)

        # Should import AsyncAnthropicVertex
        assert "AsyncAnthropicVertex" in source, (
            "cognition.py must import AsyncAnthropicVertex for non-blocking Vertex AI calls"
        )
        # Should NOT import plain AnthropicVertex (only AsyncAnthropicVertex)
        # Check that 'AnthropicVertex' only appears as part of 'AsyncAnthropicVertex'
        lines = source.split('\n')
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('#'):
                continue
            if 'AnthropicVertex' in stripped and 'AsyncAnthropicVertex' not in stripped:
                pytest.fail(
                    f"Found sync AnthropicVertex reference: '{stripped}'. "
                    "Must use AsyncAnthropicVertex for async compatibility."
                )

    def test_vertex_init_uses_async_client(self):
        """The CognitionEngine should instantiate AsyncAnthropicVertex, not AnthropicVertex."""
        import singularity.cognition as cog
        source = inspect.getsource(cog.CognitionEngine.__init__)

        assert "AsyncAnthropicVertex" in source, (
            "CognitionEngine.__init__ must create AsyncAnthropicVertex instance"
        )

    def test_think_vertex_branch_uses_await(self):
        """The think() method's vertex branch must use 'await' for the API call."""
        import singularity.cognition as cog
        source = inspect.getsource(cog.CognitionEngine.think)

        # Find the vertex branch and check it uses await
        in_vertex_block = False
        found_await = False
        for line in source.split('\n'):
            stripped = line.strip()
            if 'self.llm_type == "vertex"' in stripped and 'vertex_gemini' not in stripped:
                in_vertex_block = True
                continue  # Skip the elif line itself
            if in_vertex_block:
                if 'await self.llm.messages.create' in stripped:
                    found_await = True
                    break
                # If we hit the next elif/else, we left the vertex block
                if stripped.startswith('elif ') or stripped.startswith('else:'):
                    break

        assert found_await, (
            "think() vertex branch must use 'await self.llm.messages.create()' "
            "to avoid blocking the event loop"
        )

    def test_think_is_async(self):
        """The think() method must be a coroutine function."""
        from singularity.cognition import CognitionEngine
        assert asyncio.iscoroutinefunction(CognitionEngine.think), (
            "CognitionEngine.think must be async"
        )


# ---------------------------------------------------------------------------
# 2. Orchestrator message delivery KeyError fix
# ---------------------------------------------------------------------------

class TestOrchestratorMessageFix:
    """Verify orchestrator.py guards against missing message boxes."""

    def _make_orchestrator(self):
        """Create an OrchestratorSkill with minimal setup."""
        from singularity.skills.orchestrator import (
            OrchestratorSkill,
            _all_living_agents,
            _message_boxes,
            LivingAgent,
            LifeStatus,
        )

        # Clear global state
        _all_living_agents.clear()
        _message_boxes.clear()

        skill = OrchestratorSkill(credentials={})
        mock_agent = MagicMock()
        mock_agent.name = "TestCreator"
        mock_agent.balance = 100.0
        skill.set_parent_agent(mock_agent, agent_factory=MagicMock())

        return skill, _all_living_agents, _message_boxes

    @pytest.mark.asyncio
    async def test_broadcast_no_keyerror_on_missing_box(self):
        """broadcast() must NOT crash if agent's message box doesn't exist."""
        from singularity.skills.orchestrator import (
            LivingAgent,
            LifeStatus,
        )

        skill, agents, boxes = self._make_orchestrator()

        # Register an agent WITHOUT a message box (simulates race condition)
        agent_id = "orphan_agent_001"
        agents[agent_id] = LivingAgent(
            id=agent_id,
            name="Orphan",
            purpose="test",
            wallet=10.0,
            status=LifeStatus.ALIVE,
            born_at=datetime.now(),
            creator_id="someone_else",
        )
        # Intentionally do NOT create boxes[agent_id]

        # This should NOT raise KeyError
        result = await skill._broadcast({"message": "Hello everyone!"})
        assert result.success is True
        assert "Orphan" in result.data["sent_to"]

        # Verify the message was actually delivered
        assert agent_id in boxes
        assert not boxes[agent_id].empty()
        msg = boxes[agent_id].get_nowait()
        assert msg["message"] == "Hello everyone!"

    @pytest.mark.asyncio
    async def test_message_no_keyerror_on_missing_box(self):
        """message() must NOT crash if recipient's message box doesn't exist."""
        from singularity.skills.orchestrator import (
            LivingAgent,
            LifeStatus,
        )

        skill, agents, boxes = self._make_orchestrator()

        # Register recipient WITHOUT a message box
        agent_id = "recipient_001"
        agents[agent_id] = LivingAgent(
            id=agent_id,
            name="Recipient",
            purpose="test",
            wallet=10.0,
            status=LifeStatus.ALIVE,
            born_at=datetime.now(),
            creator_id="someone",
        )
        # Intentionally do NOT create boxes[agent_id]

        # This should NOT raise KeyError
        result = await skill._message({"to": "Recipient", "message": "Hello!"})
        assert result.success is True

        # Verify message delivered
        assert agent_id in boxes
        msg = boxes[agent_id].get_nowait()
        assert msg["message"] == "Hello!"

    @pytest.mark.asyncio
    async def test_broadcast_skips_self(self):
        """broadcast() should not send message to the sender."""
        from singularity.skills.orchestrator import (
            LivingAgent,
            LifeStatus,
        )

        skill, agents, boxes = self._make_orchestrator()

        # Register the skill's own agent
        my_id = skill._my_id
        agents[my_id] = LivingAgent(
            id=my_id,
            name="Self",
            purpose="test",
            wallet=10.0,
            status=LifeStatus.ALIVE,
            born_at=datetime.now(),
            creator_id="nobody",
        )
        boxes[my_id] = asyncio.Queue()

        # Register another agent
        other_id = "other_001"
        agents[other_id] = LivingAgent(
            id=other_id,
            name="Other",
            purpose="test",
            wallet=10.0,
            status=LifeStatus.ALIVE,
            born_at=datetime.now(),
            creator_id="nobody",
        )

        result = await skill._broadcast({"message": "Test"})
        assert result.success is True
        assert "Other" in result.data["sent_to"]
        assert "Self" not in result.data["sent_to"]

    @pytest.mark.asyncio
    async def test_check_messages_creates_box_if_missing(self):
        """check_messages() should create message box if it doesn't exist."""
        skill, agents, boxes = self._make_orchestrator()

        # Ensure the skill's message box doesn't exist
        my_id = skill._my_id
        assert my_id not in boxes

        result = await skill._check_messages({})
        assert result.success is True
        assert result.data["count"] == 0
        # Box should now exist
        assert my_id in boxes


# ---------------------------------------------------------------------------
# 3. Request skill timeout fix
# ---------------------------------------------------------------------------

class TestRequestTimeoutFix:
    """Verify request.py uses a timeout for HTTP calls."""

    def test_create_linear_ticket_has_timeout(self):
        """_create_linear_ticket must set a timeout on the aiohttp session."""
        from singularity.skills.request import RequestSkill
        source = inspect.getsource(RequestSkill._create_linear_ticket)

        assert "ClientTimeout" in source or "timeout" in source.lower(), (
            "_create_linear_ticket must set a timeout to prevent indefinite hangs"
        )

    def test_request_imports_asyncio(self):
        """request.py must import asyncio for TimeoutError handling."""
        import singularity.skills.request as req_mod
        source = inspect.getsource(req_mod)
        assert "import asyncio" in source, (
            "request.py must import asyncio for proper timeout error handling"
        )

    def test_timeout_error_handled(self):
        """_create_linear_ticket must handle TimeoutError gracefully."""
        from singularity.skills.request import RequestSkill
        source = inspect.getsource(RequestSkill._create_linear_ticket)

        assert "TimeoutError" in source, (
            "_create_linear_ticket must catch TimeoutError to handle network timeouts"
        )

    @pytest.mark.asyncio
    async def test_create_linear_ticket_timeout_returns_none(self):
        """When Linear API times out, _create_linear_ticket should return None."""
        from singularity.skills.request import RequestSkill

        skill = RequestSkill(credentials={})

        # Mock aiohttp to simulate a timeout
        with patch('singularity.skills.request.aiohttp') as mock_aiohttp:
            mock_session = AsyncMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=False)
            mock_session.post = AsyncMock(side_effect=asyncio.TimeoutError())
            mock_aiohttp.ClientSession = MagicMock(return_value=mock_session)
            mock_aiohttp.ClientTimeout = MagicMock()

            result = await skill._create_linear_ticket({"title": "test", "description": "test"})
            # Should return None on timeout, not crash
            assert result is None


# ---------------------------------------------------------------------------
# 4. Smoke tests — basic imports and data structures
# ---------------------------------------------------------------------------

class TestSmokeTests:
    """Basic smoke tests to verify the package loads correctly."""

    def test_package_imports(self):
        """Core package should import without errors."""
        from singularity import (
            AutonomousAgent,
            CognitionEngine,
            AgentState,
            Decision,
            Action,
            TokenUsage,
            Skill,
            SkillRegistry,
            SkillManifest,
            SkillAction,
            SkillResult,
        )

    def test_action_dataclass(self):
        """Action dataclass should work correctly."""
        from singularity.cognition import Action
        a = Action(tool="test:action", params={"key": "val"}, reasoning="because")
        assert a.tool == "test:action"
        assert a.params == {"key": "val"}
        assert a.reasoning == "because"

    def test_token_usage(self):
        """TokenUsage should calculate totals."""
        from singularity.cognition import TokenUsage
        usage = TokenUsage(input_tokens=100, output_tokens=50)
        assert usage.total_tokens() == 150

    def test_api_cost_calculation(self):
        """Cost calculation should work for known providers."""
        from singularity.cognition import calculate_api_cost, TokenUsage
        usage = TokenUsage(input_tokens=1_000_000, output_tokens=1_000_000)
        cost = calculate_api_cost("anthropic", "claude-sonnet-4-20250514", usage)
        assert cost == 3.0 + 15.0  # $3/M input + $15/M output

    def test_skill_result_defaults(self):
        """SkillResult should have correct defaults."""
        from singularity.skills.base import SkillResult
        r = SkillResult(success=True, message="ok")
        assert r.success is True
        assert r.data == {}
        assert r.revenue == 0
        assert r.cost == 0

    def test_skill_registry_operations(self):
        """SkillRegistry should handle install/uninstall/get."""
        from singularity.skills.base import SkillRegistry, Skill, SkillManifest, SkillAction, SkillResult

        class DummySkill(Skill):
            @property
            def manifest(self):
                return SkillManifest(
                    skill_id="dummy",
                    name="Dummy",
                    version="1.0.0",
                    category="test",
                    description="A test skill",
                    actions=[
                        SkillAction(
                            name="noop",
                            description="Does nothing",
                            parameters={},
                        )
                    ],
                    required_credentials=[],
                )

            def check_credentials(self) -> bool:
                return True

            async def execute(self, action, params):
                return SkillResult(success=True, message="noop")

        registry = SkillRegistry()
        assert registry.install(DummySkill)
        assert registry.get("dummy") is not None
        assert registry.uninstall("dummy")
        assert registry.get("dummy") is None

    def test_cognition_engine_no_llm_fallback(self):
        """CognitionEngine with no valid provider should set llm_type to 'none'."""
        from singularity.cognition import CognitionEngine
        engine = CognitionEngine(
            llm_provider="nonexistent",
            agent_name="Test",
        )
        assert engine.llm_type == "none"
        assert engine.llm is None

    @pytest.mark.asyncio
    async def test_think_no_backend_returns_wait(self):
        """think() with no LLM backend should return a 'wait' action."""
        from singularity.cognition import CognitionEngine, AgentState
        engine = CognitionEngine(
            llm_provider="nonexistent",
            agent_name="Test",
        )
        state = AgentState(
            balance=10.0,
            burn_rate=0.01,
            runway_hours=100,
            tools=[{"name": "test", "description": "test"}],
            cycle=1,
        )
        decision = await engine.think(state)
        assert decision.action.tool == "wait"
        assert decision.api_cost_usd == 0.0

    def test_parse_action_valid_json(self):
        """_parse_action should parse valid JSON responses (no nested objects)."""
        from singularity.cognition import CognitionEngine
        engine = CognitionEngine(llm_provider="nonexistent")
        # The regex in _parse_action uses [^{}]* so it can't handle nested braces.
        # This test uses a response with no nested objects, which is a valid LLM output.
        action = engine._parse_action('{"tool": "shell:bash", "reasoning": "listing files"}')
        assert action.tool == "shell:bash"
        assert action.reasoning == "listing files"

    def test_parse_action_invalid_json(self):
        """_parse_action should fallback gracefully on invalid input."""
        from singularity.cognition import CognitionEngine
        engine = CognitionEngine(llm_provider="nonexistent")
        action = engine._parse_action("I don't know what to do")
        assert action.tool == "wait"

    def test_parse_action_skill_format(self):
        """_parse_action should detect skill:action format in text."""
        from singularity.cognition import CognitionEngine
        engine = CognitionEngine(llm_provider="nonexistent")
        action = engine._parse_action("I think we should use content:write to create a blog post")
        assert action.tool == "content:write"

    def test_agent_state_defaults(self):
        """AgentState should have sensible defaults."""
        from singularity.cognition import AgentState
        state = AgentState(balance=50.0, burn_rate=0.01, runway_hours=500)
        assert state.tools == []
        assert state.recent_actions == []
        assert state.cycle == 0

    def test_decision_defaults(self):
        """Decision should have sensible defaults."""
        from singularity.cognition import Decision, Action, TokenUsage
        d = Decision(action=Action(tool="wait"))
        assert d.reasoning == ""
        assert d.api_cost_usd == 0.0
        assert d.token_usage.total_tokens() == 0

    def test_system_prompt_format(self):
        """System prompt should include agent identity."""
        from singularity.cognition import CognitionEngine
        engine = CognitionEngine(
            llm_provider="nonexistent",
            agent_name="TestBot",
            agent_ticker="TB",
            agent_specialty="testing",
        )
        prompt = engine.get_system_prompt()
        assert "TestBot" in prompt
        assert "TB" in prompt
        assert "testing" in prompt

    def test_system_prompt_append(self):
        """Appending to system prompt should work."""
        from singularity.cognition import CognitionEngine
        engine = CognitionEngine(llm_provider="nonexistent", agent_name="Test")
        original = engine.get_system_prompt()
        engine.append_to_prompt("ADDITIONAL INSTRUCTION: Be helpful.")
        new_prompt = engine.get_system_prompt()
        assert "ADDITIONAL INSTRUCTION: Be helpful." in new_prompt
        assert len(new_prompt) > len(original)
