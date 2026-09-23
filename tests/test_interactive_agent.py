"""Tests for InteractiveAgent."""
import pytest
from unittest.mock import MagicMock
from singularity.interactive_agent import InteractiveAgent, ChatMessage, ChatResponse


class TestChatMessage:
    def test_creation(self):
        msg = ChatMessage(role="user", content="hello")
        assert msg.role == "user"
        assert msg.content == "hello"
        assert msg.timestamp
        assert msg.metadata == {}


class TestChatResponse:
    def test_creation(self):
        resp = ChatResponse(message="Hello!")
        assert resp.message == "Hello!"
        assert resp.tool_calls == []
        assert resp.total_cost_usd == 0.0


class TestInteractiveAgent:
    def test_init_without_llm(self):
        agent = InteractiveAgent(name="TestBot", llm_provider="none")
        assert agent.name == "TestBot"
        assert agent.message_count == 0
        assert agent.history == []

    def test_get_tools_has_respond(self):
        agent = InteractiveAgent(llm_provider="none")
        tools = agent._get_tools()
        tool_names = [t["name"] for t in tools]
        assert "respond" in tool_names

    def test_clear_history(self):
        agent = InteractiveAgent(llm_provider="none")
        agent.history.append(ChatMessage(role="user", content="hi"))
        agent.clear_history()
        assert len(agent.history) == 0

    def test_get_stats(self):
        agent = InteractiveAgent(llm_provider="none")
        stats = agent.get_stats()
        assert stats["messages"] == 0
        assert "skills_loaded" in stats

    def test_format_history(self):
        agent = InteractiveAgent(llm_provider="none")
        agent.history.append(ChatMessage(role="user", content="hello"))
        agent.history.append(ChatMessage(role="assistant", content="hi"))
        agent.history.append(ChatMessage(role="user", content="new"))
        result = agent._format_history()
        assert "[USER]: hello" in result
        assert "[ASSISTANT]: hi" in result

    def test_synthesize_no_calls(self):
        agent = InteractiveAgent(llm_provider="none")
        result = agent._synthesize_response("test", [])
        assert "rephrasing" in result.lower() or "process" in result.lower()

    def test_synthesize_with_results(self):
        agent = InteractiveAgent(llm_provider="none")
        tool_calls = [{"tool": "fs:read", "result": {"status": "success", "message": "data"}}]
        result = agent._synthesize_response("test", tool_calls)
        assert "found" in result.lower()

    def test_default_prompt(self):
        agent = InteractiveAgent(name="Bot", llm_provider="none")
        prompt = agent._default_system_prompt()
        assert "Bot" in prompt
        assert "respond" in prompt

    @pytest.mark.asyncio
    async def test_execute_unknown_tool(self):
        agent = InteractiveAgent(llm_provider="none")
        from singularity.cognition import Action
        result = await agent._execute(Action(tool="nonexistent:action", params={}))
        assert result["status"] == "error"

    @pytest.mark.asyncio
    async def test_execute_wait(self):
        agent = InteractiveAgent(llm_provider="none")
        from singularity.cognition import Action
        result = await agent._execute(Action(tool="wait", params={}))
        assert result["status"] == "skipped"
