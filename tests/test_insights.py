"""Tests for ActionInsights module."""

import pytest
from singularity.insights import ActionInsights


def _action(tool, status, cost=0.001, message=""):
    """Helper to build a test action dict."""
    result = {"status": status}
    if message:
        result["message"] = message
    return {
        "cycle": 1,
        "tool": tool,
        "params": {},
        "result": result,
        "api_cost_usd": cost,
        "tokens": 100,
    }


class TestComputeBasics:
    def test_empty_actions(self):
        ins = ActionInsights([])
        r = ins.compute()
        assert r["total_actions"] == 0
        assert r["success_rate"] == 0.0
        assert r["recommendations"] == []

    def test_all_success(self):
        actions = [_action("fs:read", "success") for _ in range(5)]
        r = ActionInsights(actions).compute()
        assert r["total_actions"] == 5
        assert r["success_rate"] == 1.0

    def test_mixed_results(self):
        actions = [
            _action("fs:read", "success"),
            _action("fs:write", "failed", message="permission denied"),
            _action("shell:bash", "error", message="timeout"),
            _action("fs:read", "success"),
        ]
        r = ActionInsights(actions).compute()
        assert r["total_actions"] == 4
        assert r["success_rate"] == 0.5

    def test_cost_tracking(self):
        actions = [
            _action("fs:read", "success", cost=0.01),
            _action("fs:write", "success", cost=0.02),
        ]
        r = ActionInsights(actions).compute()
        assert r["total_cost"] == pytest.approx(0.03)
        assert r["avg_cost_per_action"] == pytest.approx(0.015)


class TestSkillStats:
    def test_per_skill_breakdown(self):
        actions = [
            _action("fs:read", "success"),
            _action("fs:write", "failed"),
            _action("shell:bash", "success"),
            _action("shell:bash", "success"),
        ]
        r = ActionInsights(actions).compute()
        assert r["skill_stats"]["fs"]["total"] == 2
        assert r["skill_stats"]["fs"]["success"] == 1
        assert r["skill_stats"]["fs"]["failed"] == 1
        assert r["skill_stats"]["shell"]["total"] == 2
        assert r["skill_stats"]["shell"]["success_rate"] == 1.0


class TestErrorPatterns:
    def test_repeated_errors(self):
        actions = [
            _action("fs:write", "error", message="permission denied"),
            _action("fs:write", "error", message="permission denied"),
            _action("fs:read", "success"),
        ]
        r = ActionInsights(actions).compute()
        assert len(r["error_patterns"]) == 1
        assert r["error_patterns"][0]["count"] == 2
        assert "permission denied" in r["error_patterns"][0]["message"]

    def test_no_errors(self):
        actions = [_action("fs:read", "success")]
        r = ActionInsights(actions).compute()
        assert r["error_patterns"] == []


class TestStreak:
    def test_success_streak(self):
        actions = [
            _action("fs:read", "failed"),
            _action("fs:read", "success"),
            _action("fs:read", "success"),
            _action("fs:read", "success"),
        ]
        r = ActionInsights(actions).compute()
        assert r["current_streak"]["type"] == "success"
        assert r["current_streak"]["count"] == 3

    def test_failure_streak(self):
        actions = [
            _action("fs:read", "success"),
            _action("fs:write", "error"),
            _action("fs:write", "failed"),
        ]
        r = ActionInsights(actions).compute()
        assert r["current_streak"]["type"] == "failure"
        assert r["current_streak"]["count"] == 2


class TestRecommendations:
    def test_low_success_rate_warning(self):
        actions = [
            _action("fs:read", "failed"),
            _action("fs:read", "error"),
            _action("fs:read", "success"),
        ]
        r = ActionInsights(actions).compute()
        assert any("Low success rate" in rec for rec in r["recommendations"])

    def test_failure_streak_warning(self):
        actions = [_action("fs:read", "failed") for _ in range(4)]
        r = ActionInsights(actions).compute()
        assert any("failure streak" in rec for rec in r["recommendations"])

    def test_no_warnings_when_all_good(self):
        actions = [_action("fs:read", "success") for _ in range(3)]
        r = ActionInsights(actions).compute()
        # Should have no negative recommendations
        assert not any("Low success" in rec for rec in r["recommendations"])


class TestFormatForPrompt:
    def test_empty_returns_empty(self):
        assert ActionInsights([]).format_for_prompt() == ""

    def test_has_header(self):
        actions = [_action("fs:read", "success")]
        text = ActionInsights(actions).format_for_prompt()
        assert "Performance Insights" in text
        assert "100%" in text

    def test_shows_skill_breakdown(self):
        actions = [
            _action("fs:read", "success"),
            _action("shell:bash", "success"),
        ]
        text = ActionInsights(actions).format_for_prompt()
        assert "fs:" in text or "shell:" in text

    def test_shows_error_patterns(self):
        actions = [
            _action("fs:write", "error", message="disk full"),
            _action("fs:write", "error", message="disk full"),
        ]
        text = ActionInsights(actions).format_for_prompt()
        assert "disk full" in text
