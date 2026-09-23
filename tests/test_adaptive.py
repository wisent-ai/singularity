"""Tests for AdaptiveIntelligence module."""

from singularity.adaptive import AdaptiveIntelligence, ToolStats


def test_tool_stats_defaults():
    s = ToolStats()
    assert s.total == 0
    assert s.success_rate == 1.0


def test_record_success():
    ai = AdaptiveIntelligence()
    ai.record_outcome("shell:bash", {}, success=True)
    assert ai.tool_stats["shell:bash"].successes == 1
    assert ai.tool_stats["shell:bash"].failures == 0
    assert ai.tool_stats["shell:bash"].consecutive_failures == 0


def test_record_failure():
    ai = AdaptiveIntelligence()
    ai.record_outcome("shell:bash", {}, success=False, error_message="timeout")
    assert ai.tool_stats["shell:bash"].failures == 1
    assert ai.tool_stats["shell:bash"].consecutive_failures == 1
    assert ai.tool_stats["shell:bash"].last_error == "timeout"


def test_consecutive_failures_reset_on_success():
    ai = AdaptiveIntelligence()
    ai.record_outcome("x:y", {}, False, "err1")
    ai.record_outcome("x:y", {}, False, "err2")
    assert ai.tool_stats["x:y"].consecutive_failures == 2
    ai.record_outcome("x:y", {}, True)
    assert ai.tool_stats["x:y"].consecutive_failures == 0


def test_loop_detection_same_tool():
    ai = AdaptiveIntelligence(history_window=4)
    for _ in range(4):
        ai.record_outcome("a:b", {}, True)
    assert ai.is_looping()


def test_loop_detection_pattern():
    ai = AdaptiveIntelligence(history_window=4)
    for _ in range(2):
        ai.record_outcome("a:b", {}, True)
        ai.record_outcome("c:d", {}, True)
    assert ai.is_looping()


def test_no_loop_varied():
    ai = AdaptiveIntelligence(history_window=4)
    ai.record_outcome("a:b", {}, True)
    ai.record_outcome("c:d", {}, True)
    ai.record_outcome("e:f", {}, True)
    ai.record_outcome("g:h", {}, True)
    assert not ai.is_looping()


def test_context_empty_initially():
    ai = AdaptiveIntelligence()
    assert ai.get_context() == ""


def test_context_shows_failing_tools():
    ai = AdaptiveIntelligence()
    ai.record_outcome("x:y", {}, False, "broken")
    ai.record_outcome("x:y", {}, False, "still broken")
    ctx = ai.get_context()
    assert "LOW SUCCESS RATE" in ctx or "FAILURE STREAKS" in ctx
    assert "x:y" in ctx


def test_context_shows_loop_warning():
    ai = AdaptiveIntelligence(history_window=4)
    for _ in range(4):
        ai.record_outcome("a:b", {}, False, "err")
    ctx = ai.get_context()
    assert "LOOP DETECTED" in ctx


def test_context_session_stats():
    ai = AdaptiveIntelligence()
    for i in range(5):
        ai.record_outcome(f"t{i}:a", {}, True)
    ctx = ai.get_context()
    assert "5/5" in ctx
    assert "100%" in ctx


def test_warnings_generated_at_threshold():
    ai = AdaptiveIntelligence(loop_threshold=3)
    for i in range(3):
        ai.record_outcome("bad:tool", {}, False, "fail")
    assert len(ai.warnings) > 0
    assert "bad:tool" in ai.warnings[0]


def test_failing_tools_list():
    ai = AdaptiveIntelligence()
    ai.record_outcome("a:b", {}, False)
    ai.record_outcome("a:b", {}, False)
    ai.record_outcome("c:d", {}, True)
    assert "a:b" in ai.get_failing_tools()
    assert "c:d" not in ai.get_failing_tools()
