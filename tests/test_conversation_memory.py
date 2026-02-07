"""Tests for conversation memory, configurable LLM params, and nested JSON parsing."""
import pytest
from singularity.cognition import CognitionEngine, Action


@pytest.fixture
def engine():
    return CognitionEngine(llm_provider="none")


class TestConversationMemory:
    def test_initial_history_empty(self, engine):
        assert engine.get_conversation_history() == []

    def test_record_exchange(self, engine):
        engine._record_exchange("hello", "world")
        h = engine.get_conversation_history()
        assert len(h) == 2
        assert h[0] == {"role": "user", "content": "hello"}
        assert h[1] == {"role": "assistant", "content": "world"}

    def test_history_trimming(self, engine):
        engine.set_max_history(2)
        engine._record_exchange("a", "b")
        engine._record_exchange("c", "d")
        engine._record_exchange("e", "f")
        h = engine.get_conversation_history()
        assert len(h) == 4  # 2 turns * 2 messages
        assert h[0]["content"] == "c"

    def test_clear_conversation(self, engine):
        engine._record_exchange("a", "b")
        count = engine.clear_conversation()
        assert count == 2
        assert engine.get_conversation_history() == []

    def test_build_messages(self, engine):
        engine._record_exchange("prev", "resp")
        msgs = engine._build_messages("current")
        assert len(msgs) == 3
        assert msgs[-1] == {"role": "user", "content": "current"}

    def test_format_history_as_text_empty(self, engine):
        assert engine._format_history_as_text() == ""

    def test_format_history_as_text(self, engine):
        engine._record_exchange("hello", "world")
        text = engine._format_history_as_text()
        assert "User: hello" in text
        assert "Assistant: world" in text


class TestConfigurableParams:
    def test_default_max_tokens(self, engine):
        assert engine._max_tokens == 1024

    def test_set_max_tokens(self, engine):
        engine.set_max_tokens(2048)
        assert engine._max_tokens == 2048

    def test_set_temperature(self, engine):
        engine.set_temperature(0.5)
        assert engine._temperature == 0.5

    def test_temperature_clamped(self, engine):
        engine.set_temperature(5.0)
        assert engine._temperature == 2.0
        engine.set_temperature(-1.0)
        assert engine._temperature == 0.0


class TestNestedJsonParsing:
    def test_simple_json(self, engine):
        r = '{"tool": "shell:bash", "params": {"command": "ls"}, "reasoning": "list"}'
        action = engine._parse_action(r)
        assert action.tool == "shell:bash"
        assert action.params == {"command": "ls"}

    def test_nested_params(self, engine):
        r = '{"tool": "fs:write", "params": {"path": "x.json", "content": {"key": "val"}}, "reasoning": "write"}'
        action = engine._parse_action(r)
        assert action.tool == "fs:write"
        assert action.params["content"] == {"key": "val"}

    def test_deeply_nested(self, engine):
        r = '{"tool": "test:action", "params": {"a": {"b": {"c": [1,2,3]}}}, "reasoning": "deep"}'
        action = engine._parse_action(r)
        assert action.tool == "test:action"
        assert action.params["a"]["b"]["c"] == [1, 2, 3]

    def test_json_in_text(self, engine):
        r = 'I think we should do this: {"tool": "shell:bash", "params": {"cmd": "echo hi"}, "reasoning": "test"} ok?'
        action = engine._parse_action(r)
        assert action.tool == "shell:bash"

    def test_fallback_tool_match(self, engine):
        r = "Let me use shell:bash to run a command"
        action = engine._parse_action(r)
        assert action.tool == "shell:bash"

    def test_unparseable(self, engine):
        action = engine._parse_action("I don't know what to do")
        assert action.tool == "wait"

    def test_json_with_escaped_quotes(self, engine):
        r = '{"tool": "fs:write", "params": {"content": "say \\"hello\\""}, "reasoning": "test"}'
        action = engine._parse_action(r)
        assert action.tool == "fs:write"
        assert "hello" in action.params["content"]
