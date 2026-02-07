"""Tests for ExecutionTracker."""
import pytest
from singularity.execution_tracker import ExecutionTracker, ToolStats


class TestToolStats:
    def test_initial_state(self):
        s = ToolStats()
        assert s.total == 0
        assert s.success_rate == 1.0

    def test_record_success(self):
        s = ToolStats()
        s.record("success")
        assert s.success == 1
        assert s.success_rate == 1.0

    def test_record_failure(self):
        s = ToolStats()
        s.record("failed", "bad input")
        assert s.failed == 1
        assert s.last_error == "bad input"
        assert s.success_rate == 0.0

    def test_record_error(self):
        s = ToolStats()
        s.record("error", "connection timeout")
        assert s.errors == 1
        assert s.last_error == "connection timeout"

    def test_mixed(self):
        s = ToolStats()
        s.record("success")
        s.record("success")
        s.record("failed", "x")
        s.record("error", "y")
        assert s.total == 4
        assert s.success_rate == 0.5


class TestExecutionTracker:
    def test_record_and_summary(self):
        t = ExecutionTracker()
        t.record("fs:read", "success")
        t.record("fs:read", "success")
        t.record("fs:write", "error", "permission denied")
        summary = t.get_summary()
        assert summary["total_executions"] == 3
        assert summary["tools_used"]["fs:read"]["calls"] == 2
        assert summary["tools_used"]["fs:write"]["success_rate"] == 0.0

    def test_fuzzy_match_exact(self):
        t = ExecutionTracker()
        tools = ["filesystem:read", "filesystem:write", "shell:bash"]
        assert t.find_closest_tool("filesystem:read", tools) == "filesystem:read"

    def test_fuzzy_match_typo(self):
        t = ExecutionTracker()
        tools = ["filesystem:read", "filesystem:write", "shell:bash"]
        result = t.find_closest_tool("filesystm:read", tools)
        assert result == "filesystem:read"

    def test_fuzzy_match_component(self):
        t = ExecutionTracker()
        tools = ["filesystem:read", "filesystem:write", "shell:bash"]
        result = t.find_closest_tool("filesys:reed", tools)
        assert result == "filesystem:read"

    def test_fuzzy_match_no_match(self):
        t = ExecutionTracker()
        tools = ["filesystem:read", "shell:bash"]
        result = t.find_closest_tool("zzzzz:xxxxx", tools)
        assert result is None

    def test_suggest_tools(self):
        t = ExecutionTracker()
        tools = ["filesystem:read", "filesystem:write", "shell:bash"]
        suggestions = t.suggest_tools("filesystem:reed", tools)
        assert len(suggestions) > 0
        assert "filesystem:read" in suggestions

    def test_prompt_context_empty(self):
        t = ExecutionTracker()
        assert t.get_prompt_context() == ""

    def test_prompt_context_with_failures(self):
        t = ExecutionTracker()
        t.record("shell:bash", "error", "command not found")
        t.record("shell:bash", "error", "timeout")
        ctx = t.get_prompt_context()
        assert "shell:bash" in ctx
        assert "low success rate" in ctx.lower() or "0/2" in ctx

    def test_prompt_context_with_corrections(self):
        t = ExecutionTracker()
        t.corrections_made = 3
        t.record("fs:read", "success")
        ctx = t.get_prompt_context()
        assert "3" in ctx
        assert "auto-corrected" in ctx
