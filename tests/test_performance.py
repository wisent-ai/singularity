"""Tests for PerformanceTracker skill."""
import json
import pytest
import asyncio
from pathlib import Path
from unittest.mock import patch

from singularity.skills.performance import PerformanceTracker, PERF_FILE


@pytest.fixture(autouse=True)
def clean_perf_file():
    """Remove perf file before/after each test."""
    if PERF_FILE.exists():
        PERF_FILE.unlink()
    yield
    if PERF_FILE.exists():
        PERF_FILE.unlink()


@pytest.fixture
def tracker():
    return PerformanceTracker()


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


class TestRecording:
    def test_record_success(self, tracker):
        result = run(tracker.execute("record", {
            "skill_id": "github", "action": "create_pr", "success": True,
            "latency_ms": 250, "cost_usd": 0.01,
        }))
        assert result.success
        assert "OK" in result.message

    def test_record_failure(self, tracker):
        result = run(tracker.execute("record", {
            "skill_id": "shell", "action": "run", "success": False,
            "error": "command not found",
        }))
        assert result.success
        assert "FAIL" in result.message

    def test_record_persists(self, tracker):
        run(tracker.execute("record", {
            "skill_id": "github", "action": "list", "success": True,
        }))
        data = json.loads(PERF_FILE.read_text())
        assert len(data["records"]) == 1
        assert data["records"][0]["skill_id"] == "github"

    def test_record_requires_skill_id(self, tracker):
        result = run(tracker.execute("record", {"action": "x", "success": True}))
        assert not result.success

    def test_programmatic_api(self, tracker):
        tracker.record_outcome("shell", "run", True, latency_ms=100)
        data = json.loads(PERF_FILE.read_text())
        assert len(data["records"]) == 1


class TestSkillReport:
    def _seed(self, tracker, n_ok=5, n_fail=2):
        for _ in range(n_ok):
            tracker.record_outcome("github", "create_pr", True, latency_ms=200, cost_usd=0.01)
        for _ in range(n_fail):
            tracker.record_outcome("github", "create_pr", False, error="rate limit")

    def test_report_basic(self, tracker):
        self._seed(tracker)
        result = run(tracker.execute("skill_report", {"skill_id": "github"}))
        assert result.success
        assert result.data["overall"]["total"] == 7
        assert result.data["overall"]["successes"] == 5

    def test_report_by_action(self, tracker):
        self._seed(tracker)
        tracker.record_outcome("github", "list_repos", True)
        result = run(tracker.execute("skill_report", {"skill_id": "github"}))
        assert "create_pr" in result.data["by_action"]
        assert "list_repos" in result.data["by_action"]

    def test_report_empty(self, tracker):
        result = run(tracker.execute("skill_report", {"skill_id": "nonexistent"}))
        assert result.success
        assert result.data["total"] == 0


class TestRankings:
    def test_rankings_by_success_rate(self, tracker):
        for _ in range(5):
            tracker.record_outcome("github", "pr", True)
        for _ in range(5):
            tracker.record_outcome("shell", "run", False)
        result = run(tracker.execute("rankings", {"sort_by": "success_rate"}))
        assert result.success
        assert result.data["rankings"][0]["skill_id"] == "github"

    def test_rankings_by_usage(self, tracker):
        for _ in range(10):
            tracker.record_outcome("shell", "run", True)
        tracker.record_outcome("github", "pr", True)
        result = run(tracker.execute("rankings", {"sort_by": "usage"}))
        assert result.data["rankings"][0]["skill_id"] == "shell"


class TestTrends:
    def test_trends_insufficient_data(self, tracker):
        tracker.record_outcome("x", "y", True)
        result = run(tracker.execute("trends", {}))
        assert result.data.get("insufficient_data")

    def test_trends_detects_degradation(self, tracker):
        # First half: all successes
        for _ in range(10):
            tracker.record_outcome("shell", "run", True)
        # Second half: all failures
        for _ in range(10):
            tracker.record_outcome("shell", "run", False)
        result = run(tracker.execute("trends", {}))
        trends = result.data["trends"]
        assert any(t["direction"] == "degrading" for t in trends)


class TestInsights:
    def test_unreliable_skill_insight(self, tracker):
        for _ in range(5):
            tracker.record_outcome("bad_skill", "act", False, error="fail")
        result = run(tracker.execute("insights", {}))
        types = [i["type"] for i in result.data["insights"]]
        assert "unreliable_skill" in types

    def test_repeated_error_insight(self, tracker):
        for _ in range(4):
            tracker.record_outcome("x", "y", False, error="timeout")
        result = run(tracker.execute("insights", {}))
        types = [i["type"] for i in result.data["insights"]]
        assert "repeated_error" in types


class TestSessions:
    def test_start_session(self, tracker):
        result = run(tracker.execute("start_session", {"session_id": "s1"}))
        assert result.success
        assert result.data["session_id"] == "s1"

    def test_session_summary(self, tracker):
        run(tracker.execute("start_session", {}))
        tracker.record_outcome("github", "pr", True)
        tracker.record_outcome("github", "pr", False)
        result = run(tracker.execute("session_summary", {}))
        assert result.success
        assert result.data["current_session"]["total"] == 2


class TestContextSummary:
    def test_empty_context(self, tracker):
        assert tracker.get_context_summary() == ""

    def test_context_with_data(self, tracker):
        for _ in range(3):
            tracker.record_outcome("github", "pr", True)
        ctx = tracker.get_context_summary()
        assert "github" in ctx
        assert "Historical Performance" in ctx


class TestReset:
    def test_reset(self, tracker):
        tracker.record_outcome("x", "y", True)
        result = run(tracker.execute("reset", {}))
        assert result.success
        data = json.loads(PERF_FILE.read_text())
        assert len(data["records"]) == 0
