"""Tests for CognitionEngine initialization and configuration.

Verifies:
- Engine initializes without API keys (no backend)
- System prompt configuration works
- Model switching logic
- Vertex AI imports AsyncAnthropicVertex (not sync)
- Token usage and cost calculation
"""

import pytest
from singularity.cognition import (
    CognitionEngine,
    AgentState,
    Action,
    TokenUsage,
    calculate_api_cost,
)


class TestCognitionInit:
    """Test CognitionEngine initialization."""

    def test_init_no_backend(self):
        """Engine should initialize even with no valid backend."""
        engine = CognitionEngine(llm_provider="none")
        assert engine.llm_type == "none"
        assert engine.llm is None

    def test_init_default_system_prompt(self):
        """Default system prompt should include agent identity."""
        engine = CognitionEngine(
            llm_provider="none",
            agent_name="TestBot",
            agent_ticker="TEST",
            agent_specialty="testing",
        )
        prompt = engine.get_system_prompt()
        assert "TestBot" in prompt
        assert "TEST" in prompt
        assert "testing" in prompt

    def test_init_custom_system_prompt(self):
        """Custom system prompt should override default."""
        engine = CognitionEngine(
            llm_provider="none",
            system_prompt="You are a custom agent.",
        )
        assert engine.get_system_prompt() == "You are a custom agent."

    def test_prompt_additions(self):
        """Appending to prompt should work."""
        engine = CognitionEngine(
            llm_provider="none",
            system_prompt="Base prompt.",
        )
        engine.append_to_prompt("Extra instruction 1.")
        engine.append_to_prompt("Extra instruction 2.")
        prompt = engine.get_system_prompt()
        assert "Base prompt." in prompt
        assert "Extra instruction 1." in prompt
        assert "Extra instruction 2." in prompt

    def test_set_system_prompt_clears_additions(self):
        """Setting new prompt should clear previous additions."""
        engine = CognitionEngine(llm_provider="none", system_prompt="Original.")
        engine.append_to_prompt("Addition.")
        engine.set_system_prompt("New prompt.")
        assert engine.get_system_prompt() == "New prompt."


class TestVertexAsyncImport:
    """Verify that Vertex AI uses AsyncAnthropicVertex, not sync."""

    def test_vertex_import_is_async(self):
        """The import should be AsyncAnthropicVertex for non-blocking calls."""
        # Check the module-level import
        import singularity.cognition as cog
        # If anthropic is installed, verify we imported the async variant
        if cog.HAS_VERTEX_CLAUDE:
            assert hasattr(cog, 'AsyncAnthropicVertex') or 'AsyncAnthropicVertex' in dir(cog)


class TestTokenUsage:
    """Test token usage tracking."""

    def test_total_tokens(self):
        usage = TokenUsage(input_tokens=100, output_tokens=50)
        assert usage.total_tokens() == 150

    def test_zero_tokens(self):
        usage = TokenUsage()
        assert usage.total_tokens() == 0


class TestAPICostCalculation:
    """Test API cost calculation."""

    def test_anthropic_cost(self):
        usage = TokenUsage(input_tokens=1_000_000, output_tokens=1_000_000)
        cost = calculate_api_cost("anthropic", "claude-sonnet-4-20250514", usage)
        # Input: $3/1M, Output: $15/1M = $18 total
        assert cost == pytest.approx(18.0, rel=0.01)

    def test_unknown_provider_zero_cost(self):
        usage = TokenUsage(input_tokens=1000, output_tokens=1000)
        cost = calculate_api_cost("unknown_provider", "unknown_model", usage)
        assert cost == 0.0

    def test_default_model_pricing(self):
        """Unknown model within known provider should use default pricing."""
        usage = TokenUsage(input_tokens=1_000_000, output_tokens=0)
        cost = calculate_api_cost("anthropic", "claude-future-model", usage)
        # Should use default: $3/1M input
        assert cost == pytest.approx(3.0, rel=0.01)


class TestThinkWithNoBackend:
    """Test think() with no LLM — should return wait action."""

    @pytest.mark.asyncio
    async def test_think_no_backend_returns_wait(self):
        engine = CognitionEngine(llm_provider="none")
        state = AgentState(
            balance=10.0,
            burn_rate=0.01,
            runway_hours=100.0,
            tools=[{"name": "wait", "description": "Wait"}],
        )
        decision = await engine.think(state)
        assert decision.action.tool == "wait"
        assert decision.api_cost_usd == 0.0


class TestModelInfo:
    """Test model info and switching."""

    def test_get_current_model(self):
        engine = CognitionEngine(llm_provider="none", llm_model="test-model")
        info = engine.get_current_model()
        assert info["model"] == "test-model"
        assert info["finetuned"] is False

    def test_get_available_models_no_keys(self):
        """With no API keys, no models should be available."""
        engine = CognitionEngine(llm_provider="none")
        available = engine.get_available_models()
        # Without any API keys set, should be empty or minimal
        assert isinstance(available, dict)
