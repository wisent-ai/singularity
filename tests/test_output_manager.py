"""Tests for output_manager module."""
import pytest
from singularity.output_manager import (
    truncate_result, format_action_history, _summarize_params, _summarize_data, _detect_patterns,
)

class TestTruncateResult:
    def test_small_result_unchanged(self):
        result = {"status": "success", "data": "hello", "message": "ok"}
        assert truncate_result(result) == result

    def test_large_string_data_truncated(self):
        result = {"status": "success", "data": "x" * 5000, "message": "ok"}
        truncated = truncate_result(result, max_chars=500)
        assert truncated["_truncated"] is True
        assert len(str(truncated["data"])) < 5000

    def test_large_dict_data_truncated(self):
        result = {"status": "success", "data": {f"key_{i}": "v" * 200 for i in range(20)}, "message": "ok"}
        truncated = truncate_result(result, max_chars=500)
        assert truncated["_truncated"] is True

    def test_preserves_status_and_message(self):
        result = {"status": "error", "data": "x" * 5000, "message": "something failed"}
        truncated = truncate_result(result, max_chars=500)
        assert truncated["status"] == "error"
        assert truncated["message"] == "something failed"

    def test_no_data_field(self):
        result = {"status": "success", "message": "ok"}
        assert truncate_result(result) == result

class TestFormatActionHistory:
    def test_empty_actions(self):
        assert format_action_history([]) == ""

    def test_single_action_minimal(self):
        actions = [{"cycle": 1, "tool": "fs:read", "result": {"status": "success"}, "params": {}}]
        result = format_action_history(actions, detail_level="minimal")
        assert "fs:read" in result

    def test_failed_action(self):
        actions = [{"cycle": 1, "tool": "fs:read", "result": {"status": "error"}, "params": {}}]
        result = format_action_history(actions, detail_level="minimal")
        assert "fs:read" in result

    def test_normal_detail_shows_params(self):
        actions = [{"cycle": 1, "tool": "fs:read", "result": {"status": "success", "message": "ok"}, "params": {"path": "/tmp/test.py"}}]
        result = format_action_history(actions, detail_level="normal")
        assert "path=" in result

    def test_max_actions_limit(self):
        actions = [{"cycle": i, "tool": "fs:read", "result": {"status": "success"}, "params": {}} for i in range(20)]
        result = format_action_history(actions, max_actions=5, detail_level="minimal")
        assert "20 total" in result

class TestDetectPatterns:
    def test_no_pattern_with_few_actions(self):
        assert _detect_patterns([{"tool": "a"}]) == ""

    def test_repeated_tool(self):
        actions = [{"tool": "fs:read", "result": {"status": "success"}} for _ in range(5)]
        result = _detect_patterns(actions)
        assert "fs:read" in result

    def test_repeated_failures(self):
        actions = [{"tool": f"t{i}", "result": {"status": "error"}} for i in range(5)]
        result = _detect_patterns(actions)
        assert "failed" in result

    def test_no_false_positive(self):
        actions = [
            {"tool": "a:x", "result": {"status": "success"}},
            {"tool": "b:y", "result": {"status": "success"}},
            {"tool": "c:z", "result": {"status": "success"}},
        ]
        assert _detect_patterns(actions) == ""

class TestHelpers:
    def test_summarize_params_empty(self):
        assert _summarize_params({}) == ""

    def test_summarize_data_none(self):
        assert _summarize_data(None) == ""

    def test_summarize_data_list(self):
        result = _summarize_data([1, 2, 3])
        assert "3 items" in result
