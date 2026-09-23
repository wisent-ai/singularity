"""Tests for ContextAggregator."""
import pytest
from singularity.context import ContextAggregator


def test_empty_context():
    agg = ContextAggregator()
    ctx = agg.get_context()
    assert ctx == {}


def test_record_action_and_session_summary():
    agg = ContextAggregator()
    agg.record_action("filesystem:write", {"path": "test.py"}, {"status": "success"}, cost=0.01, cycle=1)
    agg.record_action("shell:bash", {"command": "ls"}, {"status": "success"}, cost=0.005, cycle=2)
    ctx = agg.get_context()
    assert "session_summary" in ctx
    assert ctx["session_summary"]["total_actions"] == 2
    assert ctx["session_summary"]["successes"] == 2
    assert ctx["session_summary"]["success_rate"] == 100.0


def test_skill_health_tracking():
    agg = ContextAggregator()
    # Record mixed results for a skill
    for _ in range(3):
        agg.record_action("shell:bash", {}, {"status": "error"}, cycle=1)
    agg.record_action("shell:bash", {}, {"status": "success"}, cycle=2)
    ctx = agg.get_context()
    assert "skill_health" in ctx
    assert any("shell" in s for s in ctx["skill_health"].get("failing_skills", []))


def test_healthy_skills():
    agg = ContextAggregator()
    for i in range(5):
        agg.record_action("filesystem:read", {}, {"status": "success"}, cycle=i)
    ctx = agg.get_context()
    health = ctx.get("skill_health", {})
    assert "filesystem" in health.get("healthy_skills", [])


def test_loop_detection():
    agg = ContextAggregator()
    for i in range(6):
        tool = "filesystem:write" if i % 2 == 0 else "filesystem:read"
        agg.record_action(tool, {}, {"status": "success"}, cycle=i)
    ctx = agg.get_context()
    patterns = ctx.get("action_patterns", {})
    assert "possible_loop" in patterns


def test_consecutive_failure_detection():
    agg = ContextAggregator()
    agg.record_action("a:b", {}, {"status": "success"}, cycle=1)
    for i in range(4):
        agg.record_action("shell:bash", {}, {"status": "error"}, cycle=i+2)
    ctx = agg.get_context()
    patterns = ctx.get("action_patterns", {})
    assert "consecutive_failures" in patterns
    assert patterns["consecutive_failures"]["count"] >= 3


def test_cost_trajectory():
    agg = ContextAggregator()
    # First half cheap, second half expensive
    for i in range(4):
        agg.record_action("a:b", {}, {"status": "success"}, cost=0.001, cycle=i)
    for i in range(4):
        agg.record_action("a:b", {}, {"status": "success"}, cost=0.01, cycle=i+4)
    ctx = agg.get_context()
    cost = ctx.get("cost_trajectory", {})
    assert "avg_cost_per_action" in cost
    assert cost.get("cost_trend", "").startswith("increasing")


def test_most_used_tools():
    agg = ContextAggregator()
    for i in range(5):
        agg.record_action("filesystem:write", {}, {"status": "success"}, cycle=i)
    for i in range(2):
        agg.record_action("shell:bash", {}, {"status": "success"}, cycle=i+5)
    ctx = agg.get_context()
    tools = ctx["action_patterns"]["most_used_tools"]
    assert tools[0]["tool"] == "filesystem:write"
    assert tools[0]["count"] == 5


def test_max_history_trimming():
    agg = ContextAggregator(max_history=10)
    for i in range(20):
        agg.record_action("a:b", {}, {"status": "success"}, cycle=i)
    assert len(agg._action_log) == 10


def test_params_summary():
    agg = ContextAggregator()
    long_val = "x" * 100
    summary = agg._summarize_params({"path": long_val, "a": 1, "b": 2, "c": 3})
    assert "..." in summary
    assert "+1 more" in summary
