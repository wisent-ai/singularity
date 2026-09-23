"""Tests for cognition improvements: nested JSON parsing and configurable params."""
import pytest
from singularity.cognition import CognitionEngine, Action


@pytest.fixture
def engine():
    return CognitionEngine(llm_provider="none")


class TestParseActionNestedJSON:
    """Test _parse_action handles nested JSON in params."""

    def test_flat_json(self, engine):
        resp = '{"tool": "wait", "params": {}, "reasoning": "idle"}'
        action = engine._parse_action(resp)
        assert action.tool == "wait"

    def test_nested_params(self, engine):
        resp = '{"tool": "fs:write", "params": {"path": "a.py", "content": "x = {1: 2}"}, "reasoning": "write"}'
        action = engine._parse_action(resp)
        assert action.tool == "fs:write"
        assert action.params["path"] == "a.py"
        assert "x = {1: 2}" in action.params["content"]

    def test_deeply_nested(self, engine):
        resp = '{"tool": "shell:bash", "params": {"command": "echo \'{\\"a\\": 1}\'"}, "reasoning": "test"}'
        action = engine._parse_action(resp)
        assert action.tool == "shell:bash"
        assert "command" in action.params

    def test_json_in_prose(self, engine):
        resp = 'I will write the file.\n{"tool": "fs:write", "params": {"path": "x.py", "data": {"nested": true}}, "reasoning": "ok"}\nDone.'
        action = engine._parse_action(resp)
        assert action.tool == "fs:write"
        assert action.params["data"]["nested"] is True

    def test_multiple_json_picks_tool(self, engine):
        resp = 'Config: {"key": "val"}\nAction: {"tool": "wait", "params": {}, "reasoning": "none"}'
        action = engine._parse_action(resp)
        assert action.tool == "wait"

    def test_fallback_tool_pattern(self, engine):
        resp = "I should use shell:bash to run it"
        action = engine._parse_action(resp)
        assert action.tool == "shell:bash"

    def test_no_json_returns_wait(self, engine):
        action = engine._parse_action("no valid json here")
        assert action.tool == "wait"

    def test_list_in_params(self, engine):
        resp = '{"tool": "fs:write", "params": {"files": ["a.py", "b.py"]}, "reasoning": "multi"}'
        action = engine._parse_action(resp)
        assert action.tool == "fs:write"
        assert action.params["files"] == ["a.py", "b.py"]


class TestConfigurableParams:
    """Test that max_tokens and temperature are configurable."""

    def test_default_max_tokens(self):
        engine = CognitionEngine(llm_provider="none")
        assert engine.max_tokens == 1024

    def test_custom_max_tokens(self):
        engine = CognitionEngine(llm_provider="none", max_tokens=2048)
        assert engine.max_tokens == 2048

    def test_default_temperature(self):
        engine = CognitionEngine(llm_provider="none")
        assert engine.temperature == 0.2

    def test_custom_temperature(self):
        engine = CognitionEngine(llm_provider="none", temperature=0.7)
        assert engine.temperature == 0.7
