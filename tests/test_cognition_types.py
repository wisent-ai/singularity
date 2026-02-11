"""
Comprehensive tests for cognition types — Action, TokenUsage, Decision, AgentState.

Tests cover:
- Data class creation and defaults
- TokenUsage calculations
- AgentState field handling
- Decision construction
- API cost calculation with all provider/model combos
- Edge cases: zero tokens, unknown providers, unknown models
"""

import pytest
from singularity.cognition.types import (
    Action, TokenUsage, AgentState, Decision,
    calculate_api_cost, LLM_PRICING, MESSAGE_FROM_CREATOR,
    UNIFIED_AGENT_PROMPT,
)


# ============================================================
# Action Tests
# ============================================================

class TestAction:
    """Test Action data class."""

    def test_create_minimal(self):
        """Action with just a tool name."""
        action = Action(tool="wait")
        assert action.tool == "wait"
        assert action.params == {}
        assert action.reasoning == ""

    def test_create_with_params(self):
        """Action with parameters."""
        action = Action(tool="chat:send", params={"message": "hello"}, reasoning="Greeting")
        assert action.tool == "chat:send"
        assert action.params["message"] == "hello"
        assert action.reasoning == "Greeting"

    def test_default_params_are_independent(self):
        """Each Action should get its own params dict."""
        a1 = Action(tool="t1")
        a2 = Action(tool="t2")
        a1.params["key"] = "value"
        assert "key" not in a2.params

    def test_params_can_be_complex(self):
        """Params should support nested values."""
        action = Action(tool="api:call", params={
            "url": "https://example.com",
            "headers": {"Authorization": "Bearer token"},
            "body": {"nested": {"deep": True}},
        })
        assert action.params["headers"]["Authorization"] == "Bearer token"
        assert action.params["body"]["nested"]["deep"] is True


# ============================================================
# TokenUsage Tests
# ============================================================

class TestTokenUsage:
    """Test TokenUsage data class."""

    def test_defaults_to_zero(self):
        """Default usage should be zero."""
        usage = TokenUsage()
        assert usage.input_tokens == 0
        assert usage.output_tokens == 0
        assert usage.total_tokens() == 0

    def test_total_tokens(self):
        """total_tokens should sum input and output."""
        usage = TokenUsage(input_tokens=100, output_tokens=50)
        assert usage.total_tokens() == 150

    def test_large_token_counts(self):
        """Should handle large token counts."""
        usage = TokenUsage(input_tokens=1_000_000, output_tokens=500_000)
        assert usage.total_tokens() == 1_500_000

    def test_zero_input_nonzero_output(self):
        """Edge case: zero input tokens."""
        usage = TokenUsage(input_tokens=0, output_tokens=100)
        assert usage.total_tokens() == 100


# ============================================================
# AgentState Tests
# ============================================================

class TestAgentState:
    """Test AgentState data class."""

    def test_create_minimal(self):
        """AgentState with required fields only."""
        state = AgentState(balance=100.0, burn_rate=0.01, runway_hours=10000.0)
        assert state.balance == 100.0
        assert state.burn_rate == 0.01
        assert state.runway_hours == 10000.0
        assert state.tools == []
        assert state.recent_actions == []
        assert state.cycle == 0
        assert state.chat_messages == []

    def test_create_full(self):
        """AgentState with all fields."""
        state = AgentState(
            balance=50.0, burn_rate=0.02, runway_hours=2500.0,
            tools=[{"name": "test", "description": "test tool"}],
            recent_actions=[{"tool": "wait", "result": {}}],
            cycle=10,
            chat_messages=[{"sender_ticker": "EVE", "message": "hi"}],
            project_context="Project X",
            goals_progress={"revenue": {"current": 5, "target": 100}},
            pending_tasks=[{"task": "Build API", "status": "in_progress"}],
            created_resources={"payment_links": [{"url": "https://stripe.com/pay"}]},
        )
        assert len(state.tools) == 1
        assert state.cycle == 10
        assert state.project_context == "Project X"
        assert state.goals_progress["revenue"]["target"] == 100

    def test_default_lists_are_independent(self):
        """Default mutable fields should be independent between instances."""
        s1 = AgentState(balance=1, burn_rate=0, runway_hours=0)
        s2 = AgentState(balance=2, burn_rate=0, runway_hours=0)
        s1.tools.append({"name": "test"})
        assert len(s2.tools) == 0

    def test_zero_balance(self):
        """Agent with zero balance."""
        state = AgentState(balance=0.0, burn_rate=0.5, runway_hours=0.0)
        assert state.balance == 0.0

    def test_negative_balance(self):
        """Agent with negative balance (overdraft)."""
        state = AgentState(balance=-5.0, burn_rate=0.5, runway_hours=0.0)
        assert state.balance == -5.0


# ============================================================
# Decision Tests
# ============================================================

class TestDecision:
    """Test Decision data class."""

    def test_default_decision(self):
        """Default decision should have wait action."""
        decision = Decision()
        assert decision.action.tool == "wait"
        assert decision.reasoning == ""
        assert decision.token_usage.total_tokens() == 0
        assert decision.api_cost_usd == 0.0

    def test_decision_with_action(self):
        """Decision with specific action."""
        action = Action(tool="stripe:create_link", params={"amount": 500})
        decision = Decision(
            action=action,
            reasoning="Creating payment link",
            token_usage=TokenUsage(input_tokens=200, output_tokens=100),
            api_cost_usd=0.0012,
        )
        assert decision.action.tool == "stripe:create_link"
        assert decision.api_cost_usd == 0.0012


# ============================================================
# API Cost Calculation Tests
# ============================================================

class TestCalculateApiCost:
    """Test calculate_api_cost function with all providers."""

    def test_anthropic_sonnet_pricing(self):
        """Anthropic Claude Sonnet pricing."""
        usage = TokenUsage(input_tokens=1_000_000, output_tokens=1_000_000)
        cost = calculate_api_cost("anthropic", "claude-sonnet-4-20250514", usage)
        # $3/M input + $15/M output = $18
        assert cost == pytest.approx(18.0, rel=0.01)

    def test_anthropic_haiku_pricing(self):
        """Anthropic Claude Haiku pricing."""
        usage = TokenUsage(input_tokens=1_000_000, output_tokens=1_000_000)
        cost = calculate_api_cost("anthropic", "claude-3-5-haiku-20241022", usage)
        # $0.8/M input + $4/M output = $4.8
        assert cost == pytest.approx(4.8, rel=0.01)

    def test_openai_gpt4o_pricing(self):
        """OpenAI GPT-4o pricing."""
        usage = TokenUsage(input_tokens=1_000_000, output_tokens=1_000_000)
        cost = calculate_api_cost("openai", "gpt-4o", usage)
        # $2.5/M input + $10/M output = $12.5
        assert cost == pytest.approx(12.5, rel=0.01)

    def test_openai_gpt4o_mini_pricing(self):
        """OpenAI GPT-4o-mini pricing."""
        usage = TokenUsage(input_tokens=1_000_000, output_tokens=1_000_000)
        cost = calculate_api_cost("openai", "gpt-4o-mini", usage)
        # $0.15/M input + $0.6/M output = $0.75
        assert cost == pytest.approx(0.75, rel=0.01)

    def test_vertex_gemini_flash_pricing(self):
        """Vertex AI Gemini Flash pricing."""
        usage = TokenUsage(input_tokens=1_000_000, output_tokens=1_000_000)
        cost = calculate_api_cost("vertex", "gemini-2.0-flash-001", usage)
        # $0.35/M input + $1.5/M output = $1.85
        assert cost == pytest.approx(1.85, rel=0.01)

    def test_vllm_zero_cost(self):
        """Local vLLM inference should be free."""
        usage = TokenUsage(input_tokens=100_000, output_tokens=50_000)
        cost = calculate_api_cost("vllm", "llama-3-70b", usage)
        assert cost == 0.0

    def test_transformers_zero_cost(self):
        """Local transformers inference should be free."""
        usage = TokenUsage(input_tokens=100_000, output_tokens=50_000)
        cost = calculate_api_cost("transformers", "phi-3-mini", usage)
        assert cost == 0.0

    def test_unknown_provider_zero_cost(self):
        """Unknown provider should return zero cost."""
        usage = TokenUsage(input_tokens=100_000, output_tokens=50_000)
        cost = calculate_api_cost("unknown_provider", "some-model", usage)
        assert cost == 0.0

    def test_unknown_model_uses_default(self):
        """Unknown model within known provider should use default pricing."""
        usage = TokenUsage(input_tokens=1_000_000, output_tokens=1_000_000)
        cost = calculate_api_cost("anthropic", "claude-4-unknown-2025", usage)
        default = LLM_PRICING["anthropic"]["default"]
        expected = (1.0 * default["input"]) + (1.0 * default["output"])
        assert cost == pytest.approx(expected, rel=0.01)

    def test_zero_tokens(self):
        """Zero tokens should cost zero."""
        usage = TokenUsage(input_tokens=0, output_tokens=0)
        cost = calculate_api_cost("anthropic", "claude-sonnet-4-20250514", usage)
        assert cost == 0.0

    def test_small_usage(self):
        """Small token usage should produce small cost."""
        usage = TokenUsage(input_tokens=100, output_tokens=50)
        cost = calculate_api_cost("anthropic", "claude-sonnet-4-20250514", usage)
        # 100 * $3/1M + 50 * $15/1M = $0.0003 + $0.00075 = $0.00105
        assert cost == pytest.approx(0.00105, rel=0.01)

    def test_typical_agent_cycle(self):
        """Typical agent cycle: ~2K input, ~500 output on Sonnet."""
        usage = TokenUsage(input_tokens=2000, output_tokens=500)
        cost = calculate_api_cost("anthropic", "claude-sonnet-4-20250514", usage)
        # Should be a few cents at most
        assert 0.001 < cost < 0.05


# ============================================================
# Constants Tests
# ============================================================

class TestConstants:
    """Test constant values are correctly defined."""

    def test_message_from_creator_exists(self):
        """MESSAGE_FROM_CREATOR should be a non-empty string."""
        assert isinstance(MESSAGE_FROM_CREATOR, str)
        assert len(MESSAGE_FROM_CREATOR) > 100
        assert "Lukasz" in MESSAGE_FROM_CREATOR

    def test_unified_agent_prompt_has_placeholders(self):
        """UNIFIED_AGENT_PROMPT should have format placeholders."""
        assert "{name}" in UNIFIED_AGENT_PROMPT
        assert "{specialty}" in UNIFIED_AGENT_PROMPT

    def test_unified_agent_prompt_formats(self):
        """UNIFIED_AGENT_PROMPT should format correctly."""
        result = UNIFIED_AGENT_PROMPT.format(name="TestBot", specialty="trading")
        assert "TestBot" in result
        assert "trading" in result

    def test_llm_pricing_has_all_providers(self):
        """Pricing table should cover all providers."""
        assert "anthropic" in LLM_PRICING
        assert "vertex" in LLM_PRICING
        assert "openai" in LLM_PRICING
        assert "vllm" in LLM_PRICING
        assert "transformers" in LLM_PRICING

    def test_each_provider_has_default(self):
        """Each provider should have a 'default' pricing entry."""
        for provider, models in LLM_PRICING.items():
            assert "default" in models, f"Provider {provider} missing default pricing"
            assert "input" in models["default"]
            assert "output" in models["default"]
