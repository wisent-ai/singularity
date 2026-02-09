"""Tests for CognitionEngine._parse_action and _extract_tool_json.

The previous regex r'\\{[^{}]*"tool"[^{}]*\\}' rejected any JSON containing
nested braces. This meant LLM responses with nested params like:
    {"tool": "shell:bash", "params": {"command": "ls"}, "reasoning": "list files"}
would fail to match, silently dropping all params and degrading every agent cycle
that used structured parameters.

These tests verify the fix handles all real-world LLM response formats.
"""

import pytest
from singularity.cognition import CognitionEngine, Action


@pytest.fixture
def engine():
    """Create a CognitionEngine with no LLM backend (for unit testing parse logic)."""
    return CognitionEngine(llm_provider="none")


class TestParseActionNestedJSON:
    """Test _parse_action with nested JSON — the main bug fix."""

    def test_flat_json_still_works(self, engine):
        """Flat JSON without nested params should still parse."""
        response = '{"tool": "wait", "params": {}, "reasoning": "nothing to do"}'
        action = engine._parse_action(response)
        assert action.tool == "wait"
        assert action.params == {}
        assert action.reasoning == "nothing to do"

    def test_nested_params_object(self, engine):
        """Nested params object — the exact case the old regex failed on."""
        response = '{"tool": "shell:bash", "params": {"command": "echo hello"}, "reasoning": "test"}'
        action = engine._parse_action(response)
        assert action.tool == "shell:bash"
        assert action.params == {"command": "echo hello"}
        assert action.reasoning == "test"

    def test_deeply_nested_params(self, engine):
        """Params with multiple levels of nesting."""
        response = '{"tool": "content:create", "params": {"config": {"type": "blog", "settings": {"length": 500}}}, "reasoning": "create content"}'
        action = engine._parse_action(response)
        assert action.tool == "content:create"
        assert action.params["config"]["type"] == "blog"
        assert action.params["config"]["settings"]["length"] == 500

    def test_json_embedded_in_text(self, engine):
        """LLMs often wrap JSON in explanatory text."""
        response = """I think we should run a command. Here's my action:

{"tool": "shell:bash", "params": {"command": "git status", "timeout": 30}, "reasoning": "check repo state"}

This will help us understand the current state."""
        action = engine._parse_action(response)
        assert action.tool == "shell:bash"
        assert action.params["command"] == "git status"
        assert action.params["timeout"] == 30

    def test_json_in_markdown_code_block(self, engine):
        """LLMs sometimes wrap JSON in markdown code blocks."""
        response = """```json
{"tool": "filesystem:write", "params": {"path": "/tmp/test.txt", "content": "hello"}, "reasoning": "write file"}
```"""
        action = engine._parse_action(response)
        assert action.tool == "filesystem:write"
        assert action.params["path"] == "/tmp/test.txt"
        assert action.params["content"] == "hello"

    def test_params_with_array_values(self, engine):
        """Params containing arrays."""
        response = '{"tool": "email:send", "params": {"to": ["a@b.com", "c@d.com"], "subject": "test"}, "reasoning": "send email"}'
        action = engine._parse_action(response)
        assert action.tool == "email:send"
        assert action.params["to"] == ["a@b.com", "c@d.com"]

    def test_params_with_braces_in_strings(self, engine):
        """String values containing braces should not confuse the parser."""
        response = '{"tool": "shell:bash", "params": {"command": "echo {hello}"}, "reasoning": "test braces"}'
        action = engine._parse_action(response)
        assert action.tool == "shell:bash"
        assert action.params["command"] == "echo {hello}"

    def test_empty_params(self, engine):
        """Empty params should work."""
        response = '{"tool": "orchestrator:who_exists", "params": {}, "reasoning": "check agents"}'
        action = engine._parse_action(response)
        assert action.tool == "orchestrator:who_exists"
        assert action.params == {}

    def test_missing_params_key(self, engine):
        """Missing params should default to empty dict."""
        response = '{"tool": "wait", "reasoning": "no action needed"}'
        action = engine._parse_action(response)
        assert action.tool == "wait"
        assert action.params == {}

    def test_missing_reasoning_key(self, engine):
        """Missing reasoning should default to empty string."""
        response = '{"tool": "shell:bash", "params": {"command": "ls"}}'
        action = engine._parse_action(response)
        assert action.tool == "shell:bash"
        assert action.reasoning == ""


class TestParseActionFallbacks:
    """Test fallback parsing when JSON extraction fails."""

    def test_tool_colon_format_fallback(self, engine):
        """When no valid JSON, should fall back to tool:action pattern."""
        response = "I think we should use shell:bash to run a command"
        action = engine._parse_action(response)
        assert action.tool == "shell:bash"
        assert action.params == {}

    def test_no_parseable_content(self, engine):
        """When nothing parseable, should return wait action."""
        response = "I'm not sure what to do. Let me think about this more."
        action = engine._parse_action(response)
        assert action.tool == "wait"

    def test_invalid_json(self, engine):
        """Malformed JSON should fall back gracefully."""
        response = '{"tool": "shell:bash", "params": {"command": "ls"'  # missing closing braces
        action = engine._parse_action(response)
        # Should fall back to tool:action pattern
        assert action.tool == "shell:bash"

    def test_json_without_tool_key(self, engine):
        """JSON that doesn't contain 'tool' should be skipped."""
        response = 'Here is some data: {"name": "test", "value": 42}. I think we should use shell:bash.'
        action = engine._parse_action(response)
        assert action.tool == "shell:bash"

    def test_empty_response(self, engine):
        """Empty response should return wait."""
        action = engine._parse_action("")
        assert action.tool == "wait"


class TestExtractToolJSON:
    """Test _extract_tool_json directly."""

    def test_returns_none_for_no_json(self, engine):
        """Should return None when no JSON found."""
        result = engine._extract_tool_json("no json here")
        assert result is None

    def test_returns_none_for_json_without_tool(self, engine):
        """Should return None when JSON doesn't contain 'tool'."""
        result = engine._extract_tool_json('{"name": "test"}')
        assert result is None

    def test_extracts_first_tool_json(self, engine):
        """Should extract the first JSON object with 'tool' key."""
        text = '{"ignored": true} {"tool": "shell:bash", "params": {}}'
        result = engine._extract_tool_json(text)
        assert result is not None
        assert result.tool == "shell:bash"

    def test_handles_multiple_tool_jsons(self, engine):
        """Should return the first valid tool JSON."""
        text = '{"tool": "wait"} {"tool": "shell:bash"}'
        result = engine._extract_tool_json(text)
        assert result is not None
        assert result.tool == "wait"
