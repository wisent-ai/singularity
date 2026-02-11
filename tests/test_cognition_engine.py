"""
Comprehensive tests for CognitionEngine — the core LLM decision-making engine.

Tests cover:
- Initialization with all provider types
- Auto-detection logic
- Model switching
- System prompt management
- Training data collection and export
- Decision finalization and cost tracking
- Multi-turn conversation context
- Fine-tuning workflow
- Edge cases and error handling
"""

import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch, PropertyMock
from dataclasses import dataclass

from singularity.cognition.engine import CognitionEngine
from singularity.cognition.types import (
    Action, TokenUsage, AgentState, Decision, calculate_api_cost,
)


# ============================================================
# Initialization Tests
# ============================================================

class TestCognitionEngineInit:
    """Test engine initialization with different providers."""

    def test_init_with_no_provider(self):
        """Engine should initialize with llm_type='none' when no provider available."""
        engine = CognitionEngine(
            llm_provider="none",
            agent_name="Test",
            agent_ticker="TST",
        )
        assert engine.llm_type == "none"
        assert engine.llm is None
        assert engine.agent_name == "Test"
        assert engine.agent_ticker == "TST"

    def test_init_with_anthropic(self):
        """Engine should initialize Anthropic backend with API key."""
        engine = CognitionEngine(
            llm_provider="anthropic",
            anthropic_api_key="test-key",
            agent_name="Test",
            agent_ticker="TST",
        )
        assert engine.llm_type == "anthropic"
        assert engine.llm is not None

    def test_init_with_openai(self):
        """Engine should initialize OpenAI backend."""
        engine = CognitionEngine(
            llm_provider="openai",
            openai_api_key="test-key",
            agent_name="Test",
            agent_ticker="TST",
        )
        assert engine.llm_type == "openai"
        assert engine.llm is not None

    def test_init_stores_agent_metadata(self):
        """Engine should store all agent metadata."""
        engine = CognitionEngine(
            llm_provider="none",
            agent_name="Alice",
            agent_ticker="ALC",
            agent_type="trader",
            agent_specialty="crypto",
            llm_model="claude-sonnet-4-20250514",
        )
        assert engine.agent_name == "Alice"
        assert engine.agent_ticker == "ALC"
        assert engine.agent_type == "trader"
        assert engine.agent_specialty == "crypto"
        assert engine.llm_model == "claude-sonnet-4-20250514"

    def test_init_default_specialty_from_type(self):
        """Specialty should default to agent_type if not provided."""
        engine = CognitionEngine(
            llm_provider="none",
            agent_name="Test",
            agent_ticker="TST",
            agent_type="developer",
        )
        assert engine.agent_specialty == "developer"

    def test_init_default_specialty_general(self):
        """Specialty should default to 'general' if neither specialty nor type provided."""
        engine = CognitionEngine(
            llm_provider="none",
            agent_name="Test",
            agent_ticker="TST",
            agent_type="",
            agent_specialty="",
        )
        assert engine.agent_specialty == "general"

    def test_init_custom_system_prompt(self):
        """Engine should accept a custom system prompt."""
        engine = CognitionEngine(
            llm_provider="none",
            agent_name="Test",
            agent_ticker="TST",
            system_prompt="You are a test agent.",
        )
        assert engine.system_prompt == "You are a test agent."

    def test_init_system_prompt_from_file(self, tmp_path):
        """Engine should load system prompt from file."""
        prompt_file = tmp_path / "prompt.txt"
        prompt_file.write_text("Custom prompt from file.")
        engine = CognitionEngine(
            llm_provider="none",
            agent_name="Test",
            agent_ticker="TST",
            system_prompt_file=str(prompt_file),
        )
        assert engine.system_prompt == "Custom prompt from file."

    def test_init_system_prompt_file_missing(self):
        """Engine should handle missing prompt file gracefully."""
        engine = CognitionEngine(
            llm_provider="none",
            agent_name="Test",
            agent_ticker="TST",
            system_prompt_file="/nonexistent/path.txt",
        )
        # Should not crash, system_prompt should be empty or default
        assert engine.system_prompt == ""

    def test_init_project_context_from_file(self, tmp_path):
        """Engine should load project context from file."""
        ctx_file = tmp_path / "context.txt"
        ctx_file.write_text("Project: Test Suite")
        engine = CognitionEngine(
            llm_provider="none",
            agent_name="Test",
            agent_ticker="TST",
            project_context_file=str(ctx_file),
        )
        assert engine.project_context == "Project: Test Suite"

    def test_init_cost_callback(self):
        """Engine should accept and store cost callback."""
        callback = MagicMock()
        engine = CognitionEngine(
            llm_provider="none",
            agent_name="Test",
            agent_ticker="TST",
            cost_callback=callback,
        )
        assert engine._cost_callback is callback

    def test_init_training_state(self):
        """Engine should start with empty training state."""
        engine = CognitionEngine(
            llm_provider="none",
            agent_name="Test",
            agent_ticker="TST",
        )
        assert engine._training_examples == []
        assert engine._finetuned_model_id is None
        assert engine._prompt_additions == []


# ============================================================
# Auto-Detection Tests
# ============================================================

class TestAutoDetection:
    """Test provider auto-detection logic."""

    def test_auto_detect_falls_to_none(self):
        """With no providers and no API keys, should return 'none'."""
        engine = CognitionEngine.__new__(CognitionEngine)
        engine.vertex_project = None
        # auto_detect tests the global flags; since we have mock modules,
        # the flags may be True, so we patch them
        with patch("singularity.cognition.engine.HAS_ANTHROPIC", False), \
             patch("singularity.cognition.engine.HAS_OPENAI", False), \
             patch("singularity.cognition.engine.HAS_VLLM", False), \
             patch("singularity.cognition.engine.HAS_TRANSFORMERS", False), \
             patch("singularity.cognition.engine.HAS_VERTEX_CLAUDE", False), \
             patch("singularity.cognition.engine.HAS_VERTEX_GEMINI", False), \
             patch("singularity.cognition.engine.DEVICE", "cpu"):
            result = engine._auto_detect("")
            assert result == "none"

    def test_auto_detect_prefers_vertex(self):
        """If vertex_project is set and vertex is available, prefer vertex."""
        engine = CognitionEngine.__new__(CognitionEngine)
        engine.vertex_project = "my-project"
        with patch("singularity.cognition.engine.HAS_VERTEX_CLAUDE", True), \
             patch("singularity.cognition.engine.HAS_ANTHROPIC", True), \
             patch("singularity.cognition.engine.DEVICE", "cpu"):
            result = engine._auto_detect("some-key")
            assert result == "vertex"

    def test_auto_detect_anthropic_with_key(self):
        """If Anthropic is available with API key, use it."""
        engine = CognitionEngine.__new__(CognitionEngine)
        engine.vertex_project = None
        with patch("singularity.cognition.engine.HAS_ANTHROPIC", True), \
             patch("singularity.cognition.engine.HAS_VERTEX_CLAUDE", False), \
             patch("singularity.cognition.engine.HAS_VERTEX_GEMINI", False), \
             patch("singularity.cognition.engine.HAS_VLLM", False), \
             patch("singularity.cognition.engine.HAS_TRANSFORMERS", False), \
             patch("singularity.cognition.engine.DEVICE", "cpu"):
            result = engine._auto_detect("test-key")
            assert result == "anthropic"

    def test_auto_detect_anthropic_without_key_falls_through(self):
        """Anthropic without API key should fall through to OpenAI."""
        engine = CognitionEngine.__new__(CognitionEngine)
        engine.vertex_project = None
        with patch("singularity.cognition.engine.HAS_ANTHROPIC", True), \
             patch("singularity.cognition.engine.HAS_OPENAI", True), \
             patch("singularity.cognition.engine.HAS_VERTEX_CLAUDE", False), \
             patch("singularity.cognition.engine.HAS_VERTEX_GEMINI", False), \
             patch("singularity.cognition.engine.HAS_VLLM", False), \
             patch("singularity.cognition.engine.HAS_TRANSFORMERS", False), \
             patch("singularity.cognition.engine.DEVICE", "cpu"):
            result = engine._auto_detect("")
            assert result == "openai"


# ============================================================
# System Prompt Tests
# ============================================================

class TestSystemPrompt:
    """Test system prompt management."""

    def test_get_system_prompt_basic(self, engine_no_llm):
        """Basic prompt retrieval."""
        engine_no_llm.system_prompt = "Base prompt"
        assert engine_no_llm.get_system_prompt() == "Base prompt"

    def test_get_system_prompt_with_additions(self, engine_no_llm):
        """Prompt with additions appended."""
        engine_no_llm.system_prompt = "Base"
        engine_no_llm.append_to_prompt("Addition 1")
        engine_no_llm.append_to_prompt("Addition 2")
        result = engine_no_llm.get_system_prompt()
        assert "Base" in result
        assert "Addition 1" in result
        assert "Addition 2" in result

    def test_set_system_prompt_clears_additions(self, engine_no_llm):
        """Setting system prompt should clear additions."""
        engine_no_llm.system_prompt = "Old"
        engine_no_llm.append_to_prompt("Extra")
        engine_no_llm.set_system_prompt("New prompt")
        assert engine_no_llm.get_system_prompt() == "New prompt"
        assert engine_no_llm._prompt_additions == []

    def test_append_to_prompt_accumulates(self, engine_no_llm):
        """Multiple append calls should accumulate."""
        engine_no_llm.append_to_prompt("A")
        engine_no_llm.append_to_prompt("B")
        engine_no_llm.append_to_prompt("C")
        assert len(engine_no_llm._prompt_additions) == 3

    def test_get_system_prompt_empty(self, engine_no_llm):
        """Empty prompt with no additions returns empty string."""
        engine_no_llm.system_prompt = ""
        engine_no_llm._prompt_additions = []
        assert engine_no_llm.get_system_prompt() == ""


# ============================================================
# Model Switching Tests
# ============================================================

class TestModelSwitching:
    """Test model switching functionality."""

    def test_get_current_model(self, engine_no_llm):
        """Should return current model info."""
        info = engine_no_llm.get_current_model()
        assert "model" in info
        assert "provider" in info
        assert "finetuned" in info
        assert info["finetuned"] is False

    def test_switch_to_claude_via_anthropic(self, engine_anthropic):
        """Should switch to a different Claude model."""
        result = engine_anthropic.switch_model("claude-3-5-haiku-20241022")
        assert result is True
        assert engine_anthropic.llm_model == "claude-3-5-haiku-20241022"
        assert engine_anthropic.llm_type == "anthropic"

    def test_switch_to_gpt_model(self, engine_anthropic):
        """Should switch to GPT model via OpenAI."""
        result = engine_anthropic.switch_model("gpt-4o")
        assert result is True
        assert engine_anthropic.llm_model == "gpt-4o"
        assert engine_anthropic.llm_type == "openai"

    def test_switch_to_unknown_model_fails(self, engine_no_llm):
        """Should fail when switching to unknown model family."""
        result = engine_no_llm.switch_model("llama-3-70b")
        assert result is False

    def test_switch_model_preserves_state_on_failure(self, engine_anthropic):
        """Failed switch should preserve previous model state."""
        old_model = engine_anthropic.llm_model
        old_type = engine_anthropic.llm_type
        # Try switching to a model that requires vertex
        with patch("singularity.cognition.engine.HAS_VERTEX_CLAUDE", False), \
             patch("singularity.cognition.engine.HAS_VERTEX_GEMINI", False):
            result = engine_anthropic.switch_model("gemini-2.0-flash-001")
            assert result is False
            assert engine_anthropic.llm_model == old_model
            assert engine_anthropic.llm_type == old_type

    def test_switch_to_finetuned_model(self, engine_anthropic):
        """Should switch to fine-tuned model (ft: prefix)."""
        result = engine_anthropic.switch_model("ft:gpt-4o-mini:my-org::my-id")
        assert result is True
        assert engine_anthropic.llm_model == "ft:gpt-4o-mini:my-org::my-id"
        assert engine_anthropic.llm_type == "openai"

    def test_get_available_models_with_anthropic(self, engine_anthropic):
        """Should list anthropic models when API key is set."""
        models = engine_anthropic.get_available_models()
        assert "anthropic" in models
        assert "claude-sonnet-4-20250514" in models["anthropic"]

    def test_is_local_model(self, engine_no_llm):
        """is_local_model should be True for vllm/transformers."""
        engine_no_llm.llm_type = "vllm"
        assert engine_no_llm.is_local_model() is True
        engine_no_llm.llm_type = "transformers"
        assert engine_no_llm.is_local_model() is True
        engine_no_llm.llm_type = "anthropic"
        assert engine_no_llm.is_local_model() is False

    def test_get_model_returns_llm(self, engine_anthropic):
        """get_model should return the LLM instance."""
        assert engine_anthropic.get_model() is engine_anthropic.llm

    def test_get_tokenizer_none_for_api_models(self, engine_anthropic):
        """get_tokenizer should return None for API-based models."""
        assert engine_anthropic.get_tokenizer() is None


# ============================================================
# Training Data Tests
# ============================================================

class TestTrainingData:
    """Test training data collection and export."""

    def test_record_training_example(self, engine_no_llm):
        """Should record training examples."""
        engine_no_llm.record_training_example("What?", "42", "success")
        assert len(engine_no_llm._training_examples) == 1
        ex = engine_no_llm._training_examples[0]
        assert ex["outcome"] == "success"
        assert ex["messages"][1]["content"] == "What?"
        assert ex["messages"][2]["content"] == "42"

    def test_get_training_examples_filtered(self, engine_no_llm):
        """Should filter examples by outcome."""
        engine_no_llm.record_training_example("Q1", "A1", "success")
        engine_no_llm.record_training_example("Q2", "A2", "failure")
        engine_no_llm.record_training_example("Q3", "A3", "success")
        successes = engine_no_llm.get_training_examples("success")
        assert len(successes) == 2
        failures = engine_no_llm.get_training_examples("failure")
        assert len(failures) == 1

    def test_get_training_examples_all(self, engine_no_llm):
        """Should return all examples when no filter."""
        engine_no_llm.record_training_example("Q1", "A1", "success")
        engine_no_llm.record_training_example("Q2", "A2", "failure")
        all_ex = engine_no_llm.get_training_examples()
        assert len(all_ex) == 2

    def test_get_training_examples_returns_copy(self, engine_no_llm):
        """Should return a copy, not the original list."""
        engine_no_llm.record_training_example("Q", "A", "success")
        examples = engine_no_llm.get_training_examples()
        examples.clear()
        assert len(engine_no_llm._training_examples) == 1

    def test_clear_training_examples(self, engine_no_llm):
        """Should clear all examples and return count."""
        engine_no_llm.record_training_example("Q1", "A1", "success")
        engine_no_llm.record_training_example("Q2", "A2", "success")
        count = engine_no_llm.clear_training_examples()
        assert count == 2
        assert len(engine_no_llm._training_examples) == 0

    def test_export_training_data_as_string(self, engine_no_llm):
        """Should export as JSONL string."""
        engine_no_llm.record_training_example("Q1", "A1", "success")
        engine_no_llm.record_training_example("Q2", "A2", "failure")  # Excluded
        engine_no_llm.record_training_example("Q3", "A3", "success")
        content = engine_no_llm.export_training_data()
        lines = content.strip().split("\n")
        assert len(lines) == 2  # Only success examples

    def test_export_training_data_to_file(self, engine_no_llm, tmp_path):
        """Should export to file when filepath given."""
        engine_no_llm.record_training_example("Q1", "A1", "success")
        filepath = tmp_path / "training.jsonl"
        result = engine_no_llm.export_training_data(str(filepath))
        assert result == str(filepath)
        assert filepath.exists()

    def test_training_example_truncates_system_prompt(self, engine_no_llm):
        """System prompt in training examples should be truncated to 1000 chars."""
        engine_no_llm.system_prompt = "X" * 2000
        engine_no_llm._prompt_additions = []
        engine_no_llm.record_training_example("Q", "A", "success")
        sys_content = engine_no_llm._training_examples[0]["messages"][0]["content"]
        assert len(sys_content) <= 1000


# ============================================================
# Decision Finalization Tests
# ============================================================

class TestDecisionFinalization:
    """Test _finalize_decision and cost calculation."""

    def test_finalize_decision_basic(self, engine_no_llm):
        """Should parse response text into Decision with costs."""
        text = "REASON: Testing\nTOOL: chat:send\nPARAM_message: Hello"
        usage = TokenUsage(input_tokens=100, output_tokens=50)
        decision = engine_no_llm._finalize_decision(text, usage)
        assert isinstance(decision, Decision)
        assert decision.token_usage == usage
        assert decision.api_cost_usd >= 0

    def test_finalize_decision_calls_cost_callback(self, engine_no_llm):
        """Should call cost callback if set."""
        callback = MagicMock()
        engine_no_llm._cost_callback = callback
        text = "REASON: Test\nTOOL: wait"
        usage = TokenUsage(input_tokens=100, output_tokens=50)
        engine_no_llm._finalize_decision(text, usage)
        callback.assert_called_once()

    def test_finalize_decision_no_callback_when_zero_tokens(self, engine_no_llm):
        """Should not call callback when input tokens are 0."""
        callback = MagicMock()
        engine_no_llm._cost_callback = callback
        text = "REASON: Test\nTOOL: wait"
        usage = TokenUsage(input_tokens=0, output_tokens=0)
        engine_no_llm._finalize_decision(text, usage)
        callback.assert_not_called()


# ============================================================
# Think (Single-Turn) Tests
# ============================================================

class TestThink:
    """Test single-turn think() method."""

    @pytest.mark.asyncio
    async def test_think_with_no_llm_returns_wait(self, engine_no_llm, sample_agent_state):
        """Should return wait decision when no LLM is available."""
        decision = await engine_no_llm.think(sample_agent_state)
        assert decision.action.tool == "wait"
        assert "No LLM" in decision.action.reasoning

    @pytest.mark.asyncio
    async def test_think_calls_backend(self, engine_anthropic, sample_agent_state):
        """Should call generate_with_backend and return Decision."""
        with patch("singularity.cognition.engine.generate_with_backend",
                   new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = (
                "REASON: Test action\nTOOL: chat:send\nPARAM_message: hello",
                TokenUsage(input_tokens=50, output_tokens=30)
            )
            decision = await engine_anthropic.think(sample_agent_state)
            assert decision.action.tool == "chat:send"
            mock_gen.assert_called_once()

    @pytest.mark.asyncio
    async def test_think_handles_backend_error(self, engine_anthropic, sample_agent_state):
        """Should handle backend errors gracefully."""
        with patch("singularity.cognition.engine.generate_with_backend",
                   new_callable=AsyncMock) as mock_gen:
            mock_gen.side_effect = Exception("API error")
            decision = await engine_anthropic.think(sample_agent_state)
            assert decision.action.tool == "wait"
            assert "Error" in decision.action.reasoning


# ============================================================
# Think With Context (Multi-Turn) Tests
# ============================================================

class TestThinkWithContext:
    """Test multi-turn think_with_context() method."""

    @pytest.mark.asyncio
    async def test_think_with_context_no_llm(self, engine_no_llm, sample_agent_state):
        """Should return wait and empty conversation when no LLM."""
        decision, conv = await engine_no_llm.think_with_context(sample_agent_state)
        assert decision.action.tool == "wait"
        assert conv == []

    @pytest.mark.asyncio
    async def test_think_with_context_creates_initial_conversation(
        self, engine_anthropic, sample_agent_state
    ):
        """Should create initial conversation from state when none provided."""
        with patch("singularity.cognition.engine.generate_with_messages",
                   new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = (
                "REASON: First action\nTOOL: wait",
                TokenUsage(input_tokens=100, output_tokens=30)
            )
            decision, conv = await engine_anthropic.think_with_context(sample_agent_state)
            # Should have created user message + appended assistant response
            assert len(conv) == 2
            assert conv[0]["role"] == "user"
            assert conv[1]["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_think_with_context_continues_conversation(
        self, engine_anthropic, sample_agent_state
    ):
        """Should append to existing conversation."""
        existing = [
            {"role": "user", "content": "Previous state"},
            {"role": "assistant", "content": "Previous response"},
            {"role": "user", "content": "New state"},
        ]
        with patch("singularity.cognition.engine.generate_with_messages",
                   new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = (
                "REASON: Continue\nTOOL: wait",
                TokenUsage(input_tokens=200, output_tokens=40)
            )
            decision, conv = await engine_anthropic.think_with_context(
                sample_agent_state, existing
            )
            assert len(conv) == 4  # 3 existing + 1 new assistant
            assert conv[-1]["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_think_with_context_handles_error(
        self, engine_anthropic, sample_agent_state
    ):
        """Should return wait decision and preserve conversation on error."""
        existing = [{"role": "user", "content": "test"}]
        with patch("singularity.cognition.engine.generate_with_messages",
                   new_callable=AsyncMock) as mock_gen:
            mock_gen.side_effect = RuntimeError("connection lost")
            decision, conv = await engine_anthropic.think_with_context(
                sample_agent_state, existing
            )
            assert decision.action.tool == "wait"
            assert "Error" in decision.action.reasoning
            # Conversation should be preserved (not corrupted)
            assert conv == existing


# ============================================================
# Fine-Tuning Tests
# ============================================================

class TestFineTuning:
    """Test fine-tuning workflow."""

    @pytest.mark.asyncio
    async def test_start_finetune_needs_min_examples(self, engine_no_llm):
        """Should require at least 10 success examples."""
        for i in range(5):
            engine_no_llm.record_training_example(f"Q{i}", f"A{i}", "success")
        result = await engine_no_llm.start_finetune()
        assert "error" in result
        assert "10" in result["error"]

    def test_use_finetuned_model_returns_false_when_none(self, engine_no_llm):
        """Should return False when no finetuned model exists."""
        assert engine_no_llm.use_finetuned_model() is False

    def test_use_finetuned_model_attempts_switch(self, engine_anthropic):
        """Should attempt to switch to finetuned model."""
        engine_anthropic._finetuned_model_id = "ft:gpt-4o-mini:test::abc123"
        result = engine_anthropic.use_finetuned_model()
        assert result is True
        assert engine_anthropic.llm_model == "ft:gpt-4o-mini:test::abc123"
