"""
Comprehensive tests for prompt_builder — prompt construction and response parsing.

Tests cover:
- build_system_prompt: constitution + identity + rules
- build_state_message: state formatting with tools, actions, context
- build_result_message: action result formatting
- build_prompt: legacy single-prompt builder
- parse_response: LLM output parsing into Decision
- Edge cases: empty state, missing fields, malformed responses
"""

import pytest
from unittest.mock import MagicMock

from singularity.cognition.prompt_builder import (
    build_system_prompt, build_state_message, build_result_message,
    build_prompt, parse_response, _base_prompt, _format_tools,
    _format_context_sections, RESPONSE_FORMAT, ECONOMY_RULES,
)
from singularity.cognition.types import (
    Action, AgentState, Decision, MESSAGE_FROM_CREATOR,
)


# ============================================================
# Helper: Mock Engine
# ============================================================

def _make_engine(name="TestBot", specialty="general", system_prompt="",
                 prompt_additions=None, project_context=""):
    """Create a mock engine for prompt building."""
    engine = MagicMock()
    engine.agent_name = name
    engine.agent_specialty = specialty
    engine.system_prompt = system_prompt
    engine._prompt_additions = prompt_additions or []
    engine.project_context = project_context
    return engine


# ============================================================
# _base_prompt Tests
# ============================================================

class TestBasePrompt:
    """Test _base_prompt helper."""

    def test_uses_system_prompt_when_set(self):
        """Should use custom system prompt."""
        engine = _make_engine(system_prompt="Custom prompt")
        result = _base_prompt(engine)
        assert result == "Custom prompt"

    def test_uses_unified_template_when_no_system_prompt(self):
        """Should use UNIFIED_AGENT_PROMPT when no system prompt."""
        engine = _make_engine(name="Alice", specialty="trading", system_prompt="")
        result = _base_prompt(engine)
        assert "Alice" in result
        assert "trading" in result

    def test_appends_prompt_additions(self):
        """Should append additions to prompt."""
        engine = _make_engine(system_prompt="Base", prompt_additions=["Add1", "Add2"])
        result = _base_prompt(engine)
        assert "Base" in result
        assert "Add1" in result
        assert "Add2" in result

    def test_appends_project_context(self):
        """Should append project context."""
        engine = _make_engine(system_prompt="Base", project_context="Project: Singularity")
        result = _base_prompt(engine)
        assert "PROJECT CONTEXT" in result
        assert "Project: Singularity" in result


# ============================================================
# _format_tools Tests
# ============================================================

class TestFormatTools:
    """Test _format_tools helper."""

    def test_formats_tools_list(self):
        """Should format tools with names and descriptions."""
        tools = [
            {"name": "chat:send", "description": "Send a message"},
            {"name": "stripe:pay", "description": "Make payment", "parameters": {"amount": {"type": "int"}}},
        ]
        result = _format_tools(tools)
        assert "chat:send" in result
        assert "Send a message" in result
        assert "Parameters:" in result

    def test_empty_tools(self):
        """Should handle empty tools list."""
        result = _format_tools([])
        assert result == ""


# ============================================================
# build_system_prompt Tests
# ============================================================

class TestBuildSystemPrompt:
    """Test build_system_prompt function."""

    def test_contains_creator_message(self):
        """System prompt should include MESSAGE_FROM_CREATOR."""
        engine = _make_engine()
        result = build_system_prompt(engine)
        assert "MESSAGE FROM CREATOR" in result
        assert "Lukasz" in result

    def test_contains_economy_rules(self):
        """System prompt should include economy rules."""
        engine = _make_engine()
        result = build_system_prompt(engine)
        assert "ECONOMY" in result
        assert "COLLABORATION" in result

    def test_contains_response_format(self):
        """System prompt should include response format."""
        engine = _make_engine()
        result = build_system_prompt(engine)
        assert "RESPONSE FORMAT" in result
        assert "REASON:" in result
        assert "TOOL:" in result
        assert "PARAM_" in result


# ============================================================
# build_state_message Tests
# ============================================================

class TestBuildStateMessage:
    """Test build_state_message function."""

    def test_includes_balance_info(self, sample_agent_state):
        """Should include balance and burn rate."""
        engine = _make_engine()
        result = build_state_message(engine, sample_agent_state)
        assert "100.00" in result
        assert "Burn:" in result
        assert "Runway:" in result

    def test_includes_tools(self, sample_agent_state):
        """Should list available tools."""
        engine = _make_engine()
        result = build_state_message(engine, sample_agent_state)
        assert "chat:send" in result
        assert "stripe:create_link" in result

    def test_includes_recent_actions(self, sample_agent_state):
        """Should include recent action history."""
        engine = _make_engine()
        result = build_state_message(engine, sample_agent_state)
        assert "RECENT ACTIONS" in result
        assert "chat:send" in result

    def test_includes_chat_messages(self, sample_agent_state):
        """Should include chat messages from other agents."""
        engine = _make_engine(name="TestBot")
        result = build_state_message(engine, sample_agent_state)
        assert "RECENT CHAT" in result
        assert "EVE" in result

    def test_includes_at_mention_reminder(self, sample_agent_state):
        """Should remind agent to check for @mentions."""
        engine = _make_engine(name="TestBot")
        result = build_state_message(engine, sample_agent_state)
        assert "@TestBot" in result

    def test_empty_recent_actions(self):
        """Should show 'None yet' when no recent actions."""
        engine = _make_engine()
        state = AgentState(balance=50, burn_rate=0.01, runway_hours=5000)
        result = build_state_message(engine, state)
        assert "None yet" in result

    def test_includes_pending_tasks(self):
        """Should show pending tasks when present."""
        engine = _make_engine()
        state = AgentState(
            balance=50, burn_rate=0.01, runway_hours=5000,
            pending_tasks=[
                {"task": "Build payment page", "status": "in_progress", "skill": "stripe"},
                {"task": "Post update", "status": "pending"},
            ],
        )
        result = build_state_message(engine, state)
        assert "PENDING TASKS" in result
        assert "Build payment page" in result
        assert "IN_PROGRESS" in result

    def test_includes_goals_progress(self):
        """Should show goals progress."""
        engine = _make_engine()
        state = AgentState(
            balance=50, burn_rate=0.01, runway_hours=5000,
            goals_progress={"revenue": {"current": 10, "target": 100}},
        )
        result = build_state_message(engine, state)
        assert "GOALS PROGRESS" in result
        assert "10/100" in result

    def test_includes_created_resources(self):
        """Should show created payment links and products."""
        engine = _make_engine()
        state = AgentState(
            balance=50, burn_rate=0.01, runway_hours=5000,
            created_resources={
                "payment_links": [
                    {"description": "Pro Plan", "url": "https://stripe.com/pay/abc"}
                ],
                "products": [
                    {"name": "API Service", "price": 2999}
                ],
            },
        )
        result = build_state_message(engine, state)
        assert "CREATED RESOURCES" in result
        assert "Pro Plan" in result
        assert "API Service" in result

    def test_ends_with_prompt(self, sample_agent_state):
        """Should end with action prompt."""
        engine = _make_engine()
        result = build_state_message(engine, sample_agent_state)
        assert result.strip().endswith("What do you want to do?")


# ============================================================
# build_result_message Tests
# ============================================================

class TestBuildResultMessage:
    """Test build_result_message function."""

    def test_basic_success_result(self):
        """Should format success result."""
        result = build_result_message(
            "chat:send", {"message": "hello"},
            {"status": "success", "message": "Message sent"},
        )
        assert "RESULT:" in result
        assert "chat:send" in result
        assert "success" in result
        assert "Message sent" in result

    def test_error_result(self):
        """Should format error result."""
        result = build_result_message(
            "stripe:pay", {"amount": "100"},
            {"status": "error", "message": "Invalid API key"},
        )
        assert "error" in result
        assert "Invalid API key" in result

    def test_file_read_result(self):
        """Should format file read with full content."""
        result = build_result_message(
            "platform_dev:read_file", {"path": "/app/main.py"},
            {"status": "success", "message": "Read file",
             "data": {"path": "/app/main.py", "content": "print('hello')", "lines": 1}},
        )
        assert "File:" in result
        assert "/app/main.py" in result
        assert "print('hello')" in result

    def test_search_result(self):
        """Should format code search results."""
        result = build_result_message(
            "platform_dev:search_code", {"query": "def main"},
            {"status": "success", "message": "Found matches",
             "data": {"matches": [
                 {"file": "app.py", "line": 10, "content": "def main():"},
             ]}},
        )
        assert "1 matches" in result
        assert "app.py:10" in result

    def test_list_files_result(self):
        """Should format file listing."""
        result = build_result_message(
            "platform_dev:list_files", {"path": "/app"},
            {"status": "success", "message": "Listed",
             "data": {"files": [
                 {"type": "file", "path": "main.py", "size": 1024},
                 {"type": "dir", "path": "lib"},
             ]}},
        )
        assert "[file]" in result
        assert "[dir]" in result
        assert "main.py" in result

    def test_generic_data_result(self):
        """Should format generic data as string."""
        result = build_result_message(
            "custom:action", {},
            {"status": "success", "message": "Done",
             "data": {"key": "value", "count": 42}},
        )
        assert "Data:" in result
        assert "key" in result

    def test_ends_with_next_prompt(self):
        """Should end with next action prompt."""
        result = build_result_message("wait", {}, {"status": "waited"})
        assert result.strip().endswith("What do you want to do next?")

    def test_truncates_long_param_values(self):
        """Should truncate long parameter values."""
        result = build_result_message(
            "write:file", {"content": "x" * 200},
            {"status": "success", "message": "Written"},
        )
        # Content param should be excluded (stripped by ps logic)
        assert "x" * 200 not in result

    def test_truncates_long_data(self):
        """Should truncate data over 3000 chars."""
        long_data = {"text": "x" * 5000}
        result = build_result_message(
            "custom:action", {},
            {"status": "success", "message": "Done", "data": long_data},
        )
        assert "..." in result


# ============================================================
# build_prompt (Legacy) Tests
# ============================================================

class TestBuildPrompt:
    """Test legacy build_prompt function."""

    def test_combines_system_and_state(self, sample_agent_state):
        """Should combine system prompt and state message."""
        engine = _make_engine()
        result = build_prompt(engine, sample_agent_state)
        assert "MESSAGE FROM CREATOR" in result
        assert "YOUR STATE" in result
        assert "YOUR TOOLS" in result


# ============================================================
# parse_response Tests
# ============================================================

class TestParseResponse:
    """Test response parsing — critical for correct tool dispatch."""

    def test_parse_basic_response(self):
        """Should parse standard REASON/TOOL/PARAM format."""
        engine = _make_engine()
        text = "REASON: Need to send a message\nTOOL: chat:send\nPARAM_message: Hello world"
        decision = parse_response(engine, text)
        assert decision.action.tool == "chat:send"
        assert decision.action.params["message"] == "Hello world"
        assert decision.reasoning == "Need to send a message"

    def test_parse_multiple_params(self):
        """Should parse multiple parameters."""
        engine = _make_engine()
        text = "REASON: Create link\nTOOL: stripe:create_link\nPARAM_amount: 500\nPARAM_description: Pro Plan"
        decision = parse_response(engine, text)
        assert decision.action.tool == "stripe:create_link"
        assert decision.action.params["amount"] == "500"
        assert decision.action.params["description"] == "Pro Plan"

    def test_parse_wait_tool(self):
        """Should parse wait action."""
        engine = _make_engine()
        text = "REASON: Nothing to do\nTOOL: wait"
        decision = parse_response(engine, text)
        assert decision.action.tool == "wait"

    def test_parse_no_tool_defaults_to_wait(self):
        """Should default to 'wait' when no TOOL line found."""
        engine = _make_engine()
        text = "I'm just thinking about what to do next..."
        decision = parse_response(engine, text)
        assert decision.action.tool == "wait"

    def test_parse_strips_think_tags(self):
        """Should strip <think> tags from response."""
        engine = _make_engine()
        text = "<think>Internal reasoning here</think>\nREASON: Action\nTOOL: chat:send\nPARAM_message: hi"
        decision = parse_response(engine, text)
        assert decision.action.tool == "chat:send"

    def test_parse_strips_brackets_from_tool(self):
        """Should strip brackets from tool name."""
        engine = _make_engine()
        text = "REASON: Test\nTOOL: [chat:send]"
        decision = parse_response(engine, text)
        assert decision.action.tool == "chat:send"

    def test_parse_case_insensitive(self):
        """Should handle case-insensitive REASON/TOOL/PARAM."""
        engine = _make_engine()
        text = "reason: lower case\ntool: chat:send\nparam_message: test"
        decision = parse_response(engine, text)
        assert decision.action.tool == "chat:send"
        assert decision.action.params["message"] == "test"

    def test_parse_ignores_none_values(self):
        """Should ignore params with 'none' or 'n/a' values."""
        engine = _make_engine()
        text = "REASON: Test\nTOOL: test:action\nPARAM_real: value\nPARAM_empty: none\nPARAM_na: n/a"
        decision = parse_response(engine, text)
        assert "real" in decision.action.params
        assert "empty" not in decision.action.params
        assert "na" not in decision.action.params

    def test_parse_ignores_placeholder_values(self):
        """Should ignore params with placeholder values."""
        engine = _make_engine()
        text = "REASON: Test\nTOOL: test:action\nPARAM_real: value\nPARAM_placeholder: value if needed"
        decision = parse_response(engine, text)
        assert "real" in decision.action.params
        assert "placeholder" not in decision.action.params

    def test_parse_handles_newlines_in_params(self):
        """Should convert \\n to actual newlines in params."""
        engine = _make_engine()
        text = "REASON: Test\nTOOL: write:file\nPARAM_content: line1\\nline2\\nline3"
        decision = parse_response(engine, text)
        assert "\n" in decision.action.params["content"]

    def test_parse_chat_send_auto_message(self):
        """Should auto-populate message param for chat:send when missing."""
        engine = _make_engine()
        text = "REASON: Saying hello to everyone\nTOOL: chat:send"
        decision = parse_response(engine, text)
        assert decision.action.tool == "chat:send"
        assert decision.action.params.get("message") == "Saying hello to everyone"

    def test_parse_chat_send_with_message(self):
        """Should use provided message for chat:send."""
        engine = _make_engine()
        text = "REASON: Greeting\nTOOL: chat:send\nPARAM_message: Hi everyone!"
        decision = parse_response(engine, text)
        assert decision.action.params["message"] == "Hi everyone!"

    def test_parse_multiline_think_tags(self):
        """Should handle multiline think tags."""
        engine = _make_engine()
        text = """<think>
Let me think about this carefully.
I need to consider multiple options.
</think>
REASON: After careful thought
TOOL: stripe:create_link
PARAM_amount: 1000
PARAM_description: Service fee"""
        decision = parse_response(engine, text)
        assert decision.action.tool == "stripe:create_link"
        assert decision.action.params["amount"] == "1000"

    def test_parse_extra_text_before_format(self):
        """Should parse correctly even with extra text before format."""
        engine = _make_engine()
        text = "Sure, I'll do that for you.\n\nREASON: Helping out\nTOOL: wait"
        decision = parse_response(engine, text)
        assert decision.action.tool == "wait"
        assert decision.reasoning == "Helping out"

    def test_parse_param_key_lowered(self):
        """Param keys should be lowercased."""
        engine = _make_engine()
        text = "REASON: Test\nTOOL: test:action\nPARAM_MyKey: value"
        decision = parse_response(engine, text)
        assert "mykey" in decision.action.params

    def test_parse_empty_response(self):
        """Should handle completely empty response."""
        engine = _make_engine()
        text = ""
        decision = parse_response(engine, text)
        assert decision.action.tool == "wait"

    def test_parse_only_whitespace(self):
        """Should handle whitespace-only response."""
        engine = _make_engine()
        text = "   \n\n   \t  "
        decision = parse_response(engine, text)
        assert decision.action.tool == "wait"
