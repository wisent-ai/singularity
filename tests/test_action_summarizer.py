"""Tests for ActionSummarizer module."""

import pytest
from singularity.action_summarizer import (
    summarize_actions,
    _format_action,
    _failure_streak,
    _detect_loop,
    _detect_repeated_errors,
    _tool_stats,
    _detect_warnings,
    _summarize_params,
)


def _make_action(tool="fs:read", status="success", message="", params=None, cycle=1, cost=0.001):
    return {
        "tool": tool,
        "params": params or {},
        "result": {"status": status, "message": message, "data": {}},
        "api_cost_usd": cost,
        "tokens": 100,
        "cycle": cycle,
    }


class TestSummarizeActions:
    def test_empty_returns_empty(self):
        assert summarize_actions([]) == ""

    def test_single_action(self):
        actions = [_make_action(tool="fs:read", status="success", cycle=1)]
        result = summarize_actions(actions)
        assert "fs:read" in result
        assert "success" in result

    def test_includes_error_message(self):
        actions = [_make_action(tool="shell:bash", status="error", message="Permission denied", cycle=1)]
        result = summarize_actions(actions)
        assert "Permission denied" in result

    def test_failure_streak_warning(self):
        actions = [
            _make_action(status="error", message="fail", cycle=i)
            for i in range(1, 5)
        ]
        result = summarize_actions(actions)
        assert "FAILURE STREAK" in result

    def test_loop_warning(self):
        actions = [
            _make_action(tool="fs:read", cycle=i)
            for i in range(1, 5)
        ]
        result = summarize_actions(actions)
        assert "LOOP DETECTED" in result

    def test_tool_stats_shown(self):
        actions = [
            _make_action(tool="fs:read", status="success", cycle=1),
            _make_action(tool="fs:read", status="success", cycle=2),
            _make_action(tool="fs:read", status="error", cycle=3),
        ]
        result = summarize_actions(actions)
        assert "Tool stats" in result
        assert "fs:read" in result

    def test_max_display_limits_output(self):
        actions = [_make_action(cycle=i) for i in range(1, 20)]
        result = summarize_actions(actions, max_display=3)
        # Should only show last 3 actions in detail
        assert "[17]" in result
        assert "[18]" in result
        assert "[19]" in result


class TestFormatAction:
    def test_success_format(self):
        action = _make_action(tool="github:create_repo", status="success", cycle=5)
        result = _format_action(action)
        assert "[5]" in result
        assert "github:create_repo" in result
        assert "success" in result

    def test_error_shows_message(self):
        action = _make_action(tool="shell:bash", status="error", message="Command not found", cycle=3)
        result = _format_action(action)
        assert "Error: Command not found" in result

    def test_params_shown(self):
        action = _make_action(tool="fs:write", params={"path": "/tmp/test.py", "content": "x" * 100}, cycle=1)
        result = _format_action(action)
        assert "path=/tmp/test.py" in result
        assert "<100 chars>" in result  # Long content is summarized

    def test_long_error_truncated(self):
        long_msg = "A" * 200
        action = _make_action(status="error", message=long_msg, cycle=1)
        result = _format_action(action)
        assert "..." in result
        assert len(result) < 300


class TestFailureStreak:
    def test_no_failures(self):
        actions = [_make_action(status="success") for _ in range(3)]
        assert _failure_streak(actions) == 0

    def test_all_failures(self):
        actions = [_make_action(status="error") for _ in range(4)]
        assert _failure_streak(actions) == 4

    def test_mixed_ends_with_failure(self):
        actions = [
            _make_action(status="success"),
            _make_action(status="error"),
            _make_action(status="error"),
        ]
        assert _failure_streak(actions) == 2

    def test_mixed_ends_with_success(self):
        actions = [
            _make_action(status="error"),
            _make_action(status="error"),
            _make_action(status="success"),
        ]
        assert _failure_streak(actions) == 0


class TestDetectLoop:
    def test_no_loop(self):
        actions = [
            _make_action(tool="fs:read"),
            _make_action(tool="shell:bash"),
            _make_action(tool="github:push"),
        ]
        assert _detect_loop(actions) is None

    def test_loop_detected(self):
        actions = [_make_action(tool="fs:read") for _ in range(4)]
        result = _detect_loop(actions)
        assert result is not None
        assert result[0] == "fs:read"
        assert result[1] >= 3

    def test_wait_not_counted_as_loop(self):
        actions = [_make_action(tool="wait") for _ in range(5)]
        assert _detect_loop(actions) is None

    def test_too_few_actions(self):
        actions = [_make_action(tool="fs:read"), _make_action(tool="fs:read")]
        assert _detect_loop(actions) is None


class TestDetectRepeatedErrors:
    def test_no_errors(self):
        actions = [_make_action(status="success") for _ in range(3)]
        assert _detect_repeated_errors(actions) is None

    def test_repeated_error_detected(self):
        actions = [
            _make_action(tool="shell:bash", status="error", message="Permission denied"),
            _make_action(tool="shell:bash", status="error", message="Permission denied"),
        ]
        result = _detect_repeated_errors(actions)
        assert result is not None
        assert result[0] == "shell:bash"
        assert "Permission denied" in result[1]
        assert result[2] == 2

    def test_different_errors_not_flagged(self):
        actions = [
            _make_action(tool="shell:bash", status="error", message="Error A"),
            _make_action(tool="shell:bash", status="error", message="Error B"),
        ]
        assert _detect_repeated_errors(actions) is None


class TestToolStats:
    def test_empty(self):
        assert _tool_stats([]) == []

    def test_single_use_not_shown(self):
        actions = [_make_action(tool="fs:read")]
        assert _tool_stats(actions) == []

    def test_multiple_uses_shown(self):
        actions = [
            _make_action(tool="fs:read", status="success"),
            _make_action(tool="fs:read", status="error"),
        ]
        stats = _tool_stats(actions)
        assert len(stats) == 1
        assert "fs:read" in stats[0]
        assert "1/2" in stats[0]
        assert "50%" in stats[0]


class TestSummarizeParams:
    def test_empty(self):
        assert _summarize_params({}) == ""

    def test_short_values(self):
        result = _summarize_params({"path": "/tmp/test"})
        assert "path=/tmp/test" in result

    def test_long_strings_summarized(self):
        result = _summarize_params({"content": "x" * 100})
        assert "<100 chars>" in result


class TestHighCostWarning:
    def test_high_cost_warning(self):
        actions = [_make_action(cost=0.1, cycle=i) for i in range(1, 8)]
        warnings = _detect_warnings(actions)
        assert any("HIGH COST" in w for w in warnings)

    def test_low_cost_no_warning(self):
        actions = [_make_action(cost=0.001, cycle=i) for i in range(1, 4)]
        warnings = _detect_warnings(actions)
        assert not any("HIGH COST" in w for w in warnings)
