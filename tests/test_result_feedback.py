"""Tests for result_feedback module."""

import pytest
from singularity.result_feedback import (
    _truncate,
    _format_data,
    format_action_result,
    format_recent_actions,
)


class TestTruncate:
    def test_short_text(self):
        assert _truncate("hello", 100) == "hello"

    def test_long_text(self):
        text = "x" * 1000
        result = _truncate(text, 100)
        assert len(result) < 200  # Should be roughly max_len
        assert "truncated" in result

    def test_exact_length(self):
        text = "x" * 100
        assert _truncate(text, 100) == text


class TestFormatData:
    def test_none(self):
        assert _format_data(None) == ""

    def test_string(self):
        assert _format_data("hello") == "hello"

    def test_long_string(self):
        result = _format_data("x" * 1000, max_len=100)
        assert len(result) < 200

    def test_dict_with_priority_keys(self):
        data = {"content": "file data", "other": "stuff", "path": "/tmp/f"}
        result = _format_data(data)
        assert "content:" in result
        assert "path:" in result

    def test_empty_list(self):
        assert "(empty list)" in _format_data([])

    def test_list(self):
        result = _format_data(["a", "b", "c"])
        assert "[0]: a" in result
        assert "[2]: c" in result

    def test_long_list(self):
        result = _format_data(list(range(20)))
        assert "15 more items" in result


class TestFormatActionResult:
    def test_success(self):
        action = {"tool": "fs:read", "result": {"status": "success", "data": {"content": "hello"}}, "cycle": 1}
        result = format_action_result(action)
        assert "✓" in result
        assert "fs:read" in result
        assert "hello" in result

    def test_error(self):
        action = {"tool": "shell:bash", "result": {"status": "error", "message": "not found"}, "cycle": 2}
        result = format_action_result(action)
        assert "✗" in result
        assert "not found" in result

    def test_no_data(self):
        action = {"tool": "wait", "result": {"status": "waited"}, "cycle": 3}
        result = format_action_result(action)
        assert "wait" in result


class TestFormatRecentActions:
    def test_empty(self):
        assert format_recent_actions([]) == ""

    def test_single_action(self):
        actions = [{"tool": "fs:read", "result": {"status": "success", "data": {"content": "hi"}}, "cycle": 1}]
        result = format_recent_actions(actions)
        assert "Recent actions" in result
        assert "fs:read" in result

    def test_multiple_actions(self):
        actions = [
            {"tool": "fs:read", "result": {"status": "success", "data": {"content": "a"}}, "cycle": 1},
            {"tool": "shell:bash", "result": {"status": "error", "message": "fail"}, "cycle": 2},
            {"tool": "fs:write", "result": {"status": "success", "data": {"path": "/f"}}, "cycle": 3},
        ]
        result = format_recent_actions(actions)
        assert "fs:read" in result
        assert "shell:bash" in result
        assert "fs:write" in result

    def test_max_actions_limit(self):
        actions = [{"tool": f"t:{i}", "result": {"status": "success"}, "cycle": i} for i in range(20)]
        result = format_recent_actions(actions, max_actions=3)
        # Should only show the last 3
        assert "t:17" in result
        assert "t:18" in result
        assert "t:19" in result
        assert "t:0" not in result

    def test_total_length_limit(self):
        actions = [
            {"tool": "fs:read", "result": {"status": "success", "data": {"content": "x" * 5000}}, "cycle": i}
            for i in range(5)
        ]
        result = format_recent_actions(actions, max_total_len=500)
        assert len(result) <= 600  # Allow some slack for truncation message
