"""
Shared test fixtures for singularity engine tests.

Mocks expensive dependencies (anthropic, openai, etc.) so tests
run in CI without API keys or GPU hardware.
"""

import sys
import types
from unittest.mock import MagicMock, AsyncMock
from dataclasses import dataclass
import pytest


# === Mock expensive dependencies before any singularity imports ===

def _setup_mock_modules():
    """Install mock modules for dependencies that may not be installed."""
    mocked = {}

    # Mock anthropic
    if "anthropic" not in sys.modules:
        mock_anthropic = types.ModuleType("anthropic")

        class MockAsyncAnthropic:
            def __init__(self, **kwargs):
                self.api_key = kwargs.get("api_key", "")
                self.messages = MagicMock()

        class MockAnthropicVertex:
            def __init__(self, **kwargs):
                self.project_id = kwargs.get("project_id", "")
                self.region = kwargs.get("region", "")
                self.messages = MagicMock()

        mock_anthropic.AsyncAnthropic = MockAsyncAnthropic
        mock_anthropic.AnthropicVertex = MockAnthropicVertex
        sys.modules["anthropic"] = mock_anthropic
        mocked["anthropic"] = mock_anthropic

    # Mock openai
    if "openai" not in sys.modules:
        mock_openai = types.ModuleType("openai")

        class MockAsyncOpenAI:
            def __init__(self, **kwargs):
                self.api_key = kwargs.get("api_key", "")
                self.base_url = kwargs.get("base_url", "")
                self.chat = MagicMock()

        class MockOpenAI:
            def __init__(self, **kwargs):
                self.api_key = kwargs.get("api_key", "")
                self.files = MagicMock()
                self.fine_tuning = MagicMock()

        mock_openai.AsyncOpenAI = MockAsyncOpenAI
        mock_openai.OpenAI = MockOpenAI
        sys.modules["openai"] = mock_openai
        mocked["openai"] = mock_openai

    # Mock vertexai
    if "vertexai" not in sys.modules:
        mock_vertexai = types.ModuleType("vertexai")
        mock_vertexai.init = MagicMock()
        sys.modules["vertexai"] = mock_vertexai

        mock_gen = types.ModuleType("vertexai.generative_models")
        mock_gen.GenerativeModel = MagicMock()
        mock_gen.GenerationConfig = MagicMock()
        sys.modules["vertexai.generative_models"] = mock_gen
        mocked["vertexai"] = mock_vertexai

    # Mock aiohttp (for marketplace tests)
    if "aiohttp" not in sys.modules:
        mock_aiohttp = types.ModuleType("aiohttp")
        mock_aiohttp.ClientSession = MagicMock()
        mock_aiohttp.ClientTimeout = MagicMock()
        sys.modules["aiohttp"] = mock_aiohttp
        mocked["aiohttp"] = mock_aiohttp

    # Mock dotenv
    if "dotenv" not in sys.modules:
        mock_dotenv = types.ModuleType("dotenv")
        mock_dotenv.load_dotenv = MagicMock()
        sys.modules["dotenv"] = mock_dotenv
        mocked["dotenv"] = mock_dotenv

    return mocked


# Install mocks at import time
_setup_mock_modules()


# === Fixtures ===

@pytest.fixture
def mock_credentials():
    """Standard set of mock credentials for testing."""
    return {
        "ANTHROPIC_API_KEY": "test-anthropic-key",
        "OPENAI_API_KEY": "test-openai-key",
        "TWITTER_API_KEY": "test-twitter-key",
        "TWITTER_API_SECRET": "test-twitter-secret",
        "TWITTER_ACCESS_TOKEN": "test-twitter-token",
        "TWITTER_ACCESS_SECRET": "test-twitter-access",
        "GITHUB_TOKEN": "test-github-token",
        "RESEND_API_KEY": "test-resend-key",
        "VERCEL_TOKEN": "test-vercel-token",
        "STRIPE_SECRET_KEY": "test-stripe-key",
    }


@pytest.fixture
def sample_agent_state():
    """Create a sample AgentState for testing."""
    from singularity.cognition.types import AgentState
    return AgentState(
        balance=100.0,
        burn_rate=0.01,
        runway_hours=10000.0,
        tools=[
            {"name": "chat:send", "description": "Send a chat message",
             "parameters": {"message": {"type": "string"}}},
            {"name": "stripe:create_link", "description": "Create payment link",
             "parameters": {"amount": {"type": "number"}, "description": {"type": "string"}}},
        ],
        recent_actions=[
            {"tool": "chat:send", "params": {"message": "hello"},
             "result": {"status": "success", "message": "Sent"}},
        ],
        cycle=5,
        chat_messages=[
            {"sender_ticker": "EVE", "message": "Looking for collaboration"},
            {"sender_ticker": "LINUS", "message": "Dashboard updated"},
        ],
    )


@pytest.fixture
def engine_no_llm():
    """Create a CognitionEngine with no LLM backend (provider=none)."""
    from singularity.cognition.engine import CognitionEngine
    return CognitionEngine(
        llm_provider="none",
        agent_name="TestAgent",
        agent_ticker="TEST",
        agent_type="testing",
        llm_model="test-model",
    )


@pytest.fixture
def engine_anthropic():
    """Create a CognitionEngine with mock Anthropic backend."""
    from singularity.cognition.engine import CognitionEngine
    return CognitionEngine(
        llm_provider="anthropic",
        anthropic_api_key="test-key",
        agent_name="TestAgent",
        agent_ticker="TEST",
        llm_model="claude-sonnet-4-20250514",
    )
