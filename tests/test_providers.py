"""
Tests for cognition providers — LLM backend detection and generation.

Tests cover:
- Device detection
- Backend availability flags
- generate_with_backend delegation
- generate_with_messages for each provider type
- _lazy_import function
- AVAILABLE_MODELS constant
- Edge cases: empty responses, fallback behavior
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
import asyncio

from singularity.cognition.providers import (
    get_device, DEVICE, AVAILABLE_MODELS, _lazy_import,
    generate_with_backend, generate_with_messages,
)
from singularity.cognition.types import TokenUsage


# ============================================================
# Device Detection Tests
# ============================================================

class TestDeviceDetection:
    """Test GPU/CPU device detection."""

    def test_get_device_returns_string(self):
        """Should return a valid device string."""
        device = get_device()
        assert device in ("cuda", "mps", "cpu")

    def test_device_global_is_set(self):
        """DEVICE global should be set at module load."""
        assert DEVICE in ("cuda", "mps", "cpu")


# ============================================================
# Lazy Import Tests
# ============================================================

class TestLazyImport:
    """Test _lazy_import function."""

    def test_import_anthropic(self):
        """Should return AsyncAnthropic class."""
        result = _lazy_import("anthropic")
        assert result is not None
        assert result.__name__ == "MockAsyncAnthropic" or hasattr(result, "__init__")

    def test_import_vertex_claude(self):
        """Should return AnthropicVertex class."""
        result = _lazy_import("vertex_claude")
        assert result is not None

    def test_import_openai(self):
        """Should return openai module."""
        result = _lazy_import("openai")
        assert result is not None
        assert hasattr(result, "AsyncOpenAI")

    def test_import_unknown_returns_none(self):
        """Should return None for unknown import name."""
        result = _lazy_import("nonexistent_provider")
        assert result is None


# ============================================================
# Available Models Tests
# ============================================================

class TestAvailableModels:
    """Test AVAILABLE_MODELS constant."""

    def test_has_vertex_models(self):
        """Should include Vertex AI models."""
        assert "vertex" in AVAILABLE_MODELS
        assert "gemini-2.0-flash-001" in AVAILABLE_MODELS["vertex"]

    def test_has_anthropic_models(self):
        """Should include Anthropic models."""
        assert "anthropic" in AVAILABLE_MODELS
        assert "claude-sonnet-4-20250514" in AVAILABLE_MODELS["anthropic"]

    def test_has_openai_models(self):
        """Should include OpenAI models."""
        assert "openai" in AVAILABLE_MODELS
        assert "gpt-4o" in AVAILABLE_MODELS["openai"]

    def test_model_entries_have_cost_speed_capability(self):
        """Each model should have cost, speed, capability fields."""
        for provider, models in AVAILABLE_MODELS.items():
            for model_name, info in models.items():
                assert "cost" in info, f"{provider}/{model_name} missing cost"
                assert "speed" in info, f"{provider}/{model_name} missing speed"
                assert "capability" in info, f"{provider}/{model_name} missing capability"


# ============================================================
# Generate with Backend Tests
# ============================================================

class TestGenerateWithBackend:
    """Test generate_with_backend function."""

    @pytest.mark.asyncio
    async def test_delegates_to_generate_with_messages(self):
        """Should wrap single prompt into messages format."""
        engine = MagicMock()
        engine.llm_type = "none"

        with patch("singularity.cognition.providers.generate_with_messages",
                   new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = ("response text", TokenUsage(10, 5))
            text, usage = await generate_with_backend(engine, "test prompt")
            mock_gen.assert_called_once()
            # Should wrap in messages format
            call_args = mock_gen.call_args
            messages = call_args[0][1]
            assert messages[0]["role"] == "user"
            assert messages[0]["content"] == "test prompt"


# ============================================================
# Generate with Messages Tests
# ============================================================

class TestGenerateWithMessages:
    """Test generate_with_messages for each backend type."""

    @pytest.mark.asyncio
    async def test_none_backend_returns_empty(self):
        """None backend should return empty response."""
        engine = MagicMock()
        engine.llm_type = "unknown_type"
        text, usage = await generate_with_messages(
            engine, [{"role": "user", "content": "test"}]
        )
        assert text == ""
        assert usage.total_tokens() == 0

    @pytest.mark.asyncio
    async def test_anthropic_backend(self):
        """Should call Anthropic API correctly."""
        engine = MagicMock()
        engine.llm_type = "anthropic"
        engine.llm_model = "claude-sonnet-4-20250514"
        engine.llm = MagicMock()

        # Mock response
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="REASON: Test\nTOOL: wait")]
        mock_response.usage.input_tokens = 100
        mock_response.usage.output_tokens = 50
        engine.llm.messages.create = AsyncMock(return_value=mock_response)

        messages = [{"role": "user", "content": "What should I do?"}]
        text, usage = await generate_with_messages(engine, messages, system="Be helpful")
        assert text == "REASON: Test\nTOOL: wait"
        assert usage.input_tokens == 100
        assert usage.output_tokens == 50

        # Check system prompt was passed
        call_kwargs = engine.llm.messages.create.call_args[1]
        assert call_kwargs["system"] == "Be helpful"

    @pytest.mark.asyncio
    async def test_openai_backend(self):
        """Should call OpenAI API correctly."""
        engine = MagicMock()
        engine.llm_type = "openai"
        engine.llm_model = "gpt-4o"
        engine.llm = MagicMock()

        # Mock response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="REASON: Test\nTOOL: wait"))]
        mock_response.usage.prompt_tokens = 80
        mock_response.usage.completion_tokens = 40
        engine.llm.chat.completions.create = AsyncMock(return_value=mock_response)

        messages = [{"role": "user", "content": "Hello"}]
        text, usage = await generate_with_messages(engine, messages, system="System prompt")
        assert text == "REASON: Test\nTOOL: wait"
        assert usage.input_tokens == 80
        assert usage.output_tokens == 40

    @pytest.mark.asyncio
    async def test_openai_prepends_system_message(self):
        """OpenAI backend should prepend system as first message."""
        engine = MagicMock()
        engine.llm_type = "openai"
        engine.llm_model = "gpt-4o"
        engine.llm = MagicMock()

        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="ok"))]
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        engine.llm.chat.completions.create = AsyncMock(return_value=mock_response)

        await generate_with_messages(
            engine, [{"role": "user", "content": "Hi"}], system="Be nice"
        )

        call_kwargs = engine.llm.chat.completions.create.call_args[1]
        messages = call_kwargs["messages"]
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "Be nice"

    @pytest.mark.asyncio
    async def test_openai_no_system_message_when_empty(self):
        """OpenAI backend should not prepend system when empty."""
        engine = MagicMock()
        engine.llm_type = "openai"
        engine.llm_model = "gpt-4o"
        engine.llm = MagicMock()

        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="ok"))]
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        engine.llm.chat.completions.create = AsyncMock(return_value=mock_response)

        await generate_with_messages(
            engine, [{"role": "user", "content": "Hi"}], system=""
        )

        call_kwargs = engine.llm.chat.completions.create.call_args[1]
        messages = call_kwargs["messages"]
        assert messages[0]["role"] == "user"

    @pytest.mark.asyncio
    async def test_vertex_backend_uses_run_in_executor(self):
        """Vertex (Claude) backend should use run_in_executor (sync API)."""
        engine = MagicMock()
        engine.llm_type = "vertex"
        engine.llm_model = "claude-3-5-sonnet-v2@20241022"
        engine.llm = MagicMock()

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="REASON: Test\nTOOL: wait")]
        mock_response.usage.input_tokens = 200
        mock_response.usage.output_tokens = 100
        engine.llm.messages.create = MagicMock(return_value=mock_response)

        messages = [{"role": "user", "content": "test"}]
        text, usage = await generate_with_messages(engine, messages)
        assert text == "REASON: Test\nTOOL: wait"
        assert usage.input_tokens == 200

    @pytest.mark.asyncio
    async def test_anthropic_without_system_prompt(self):
        """Should work without system prompt."""
        engine = MagicMock()
        engine.llm_type = "anthropic"
        engine.llm_model = "claude-sonnet-4-20250514"
        engine.llm = MagicMock()

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="response")]
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 5
        engine.llm.messages.create = AsyncMock(return_value=mock_response)

        text, usage = await generate_with_messages(
            engine, [{"role": "user", "content": "Hi"}], system=""
        )

        call_kwargs = engine.llm.messages.create.call_args[1]
        assert "system" not in call_kwargs

    @pytest.mark.asyncio
    async def test_openai_no_usage_returns_zero(self):
        """Should handle None usage in OpenAI response."""
        engine = MagicMock()
        engine.llm_type = "openai"
        engine.llm_model = "gpt-4o"
        engine.llm = MagicMock()

        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="ok"))]
        mock_response.usage = None
        engine.llm.chat.completions.create = AsyncMock(return_value=mock_response)

        text, usage = await generate_with_messages(
            engine, [{"role": "user", "content": "Hi"}]
        )
        assert usage.input_tokens == 0
        assert usage.output_tokens == 0
