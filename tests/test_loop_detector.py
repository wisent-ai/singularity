"""Tests for LoopDetector."""
import pytest
from singularity.loop_detector import LoopDetector, LoopAlert


def _action(tool, params=None, status="success", message=""):
    return {
        "tool": tool,
        "params": params or {},
        "result": {"status": status, "message": message},
        "cycle": 1,
    }


class TestExactRepeats:
    def test_detects_exact_repeats(self):
        d = LoopDetector(exact_repeat_threshold=3)
        actions = [_action("fs:read", {"path": "a.txt"})] * 3
        alerts = d.analyze(actions)
        assert any(a.pattern_type == "exact_repeat" for a in alerts)

    def test_no_alert_below_threshold(self):
        d = LoopDetector(exact_repeat_threshold=3)
        actions = [_action("fs:read", {"path": "a.txt"})] * 2
        alerts = d.analyze(actions)
        assert not any(a.pattern_type == "exact_repeat" for a in alerts)

    def test_different_params_no_exact_repeat(self):
        d = LoopDetector(exact_repeat_threshold=3)
        actions = [
            _action("fs:read", {"path": "a.txt"}),
            _action("fs:read", {"path": "b.txt"}),
            _action("fs:read", {"path": "c.txt"}),
        ]
        alerts = d.analyze(actions)
        assert not any(a.pattern_type == "exact_repeat" for a in alerts)


class TestToolRepeats:
    def test_detects_tool_repeats(self):
        d = LoopDetector(tool_repeat_threshold=4)
        actions = [
            _action("fs:read", {"path": f"{i}.txt"}) for i in range(4)
        ]
        alerts = d.analyze(actions)
        assert any(a.pattern_type == "tool_repeat" for a in alerts)

    def test_no_alert_mixed_tools(self):
        d = LoopDetector(tool_repeat_threshold=4)
        actions = [
            _action("fs:read"), _action("shell:bash"),
            _action("fs:read"), _action("shell:bash"),
        ]
        alerts = d.analyze(actions)
        assert not any(a.pattern_type == "tool_repeat" for a in alerts)


class TestErrorStreak:
    def test_detects_error_streak(self):
        d = LoopDetector(error_streak_threshold=3)
        actions = [
            _action("a", status="failed", message="err1"),
            _action("b", status="failed", message="err2"),
            _action("c", status="error", message="err3"),
        ]
        alerts = d.analyze(actions)
        assert any(a.pattern_type == "error_loop" for a in alerts)

    def test_success_breaks_streak(self):
        d = LoopDetector(error_streak_threshold=3)
        actions = [
            _action("a", status="failed"),
            _action("b", status="success"),
            _action("c", status="failed"),
            _action("d", status="failed"),
        ]
        alerts = d.analyze(actions)
        assert not any(a.pattern_type == "error_loop" for a in alerts)


class TestPingPong:
    def test_detects_ping_pong(self):
        d = LoopDetector()
        actions = [
            _action("fs:read"), _action("shell:bash"),
            _action("fs:read"), _action("shell:bash"),
        ]
        alerts = d.analyze(actions)
        assert any(a.pattern_type == "ping_pong" for a in alerts)

    def test_no_ping_pong_with_three_tools(self):
        d = LoopDetector()
        actions = [
            _action("a"), _action("b"), _action("c"), _action("a"),
        ]
        alerts = d.analyze(actions)
        assert not any(a.pattern_type == "ping_pong" for a in alerts)


class TestFormatWarnings:
    def test_empty_alerts(self):
        d = LoopDetector()
        assert d.format_warnings([]) == ""

    def test_formats_critical(self):
        d = LoopDetector()
        alert = LoopAlert("error_loop", "3 failures", "critical", "stop", 3)
        text = d.format_warnings([alert])
        assert "LOOP DETECTION" in text
        assert "🔴" in text
        assert "stop" in text

    def test_formats_warning(self):
        d = LoopDetector()
        alert = LoopAlert("tool_repeat", "repeat", "warning", "change", 5)
        text = d.format_warnings([alert])
        assert "🟡" in text


class TestShouldForceRethink:
    def test_critical_forces_rethink(self):
        d = LoopDetector()
        alerts = [LoopAlert("x", "d", "critical", "s")]
        assert d.should_force_rethink(alerts)

    def test_warning_no_rethink(self):
        d = LoopDetector()
        alerts = [LoopAlert("x", "d", "warning", "s")]
        assert not d.should_force_rethink(alerts)


class TestTotalAlerts:
    def test_counts_alerts(self):
        d = LoopDetector(exact_repeat_threshold=2)
        actions = [_action("a")] * 2
        d.analyze(actions)
        assert d.total_alerts > 0

    def test_empty_no_alerts(self):
        d = LoopDetector()
        d.analyze([])
        assert d.total_alerts == 0
