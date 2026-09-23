"""Tests for SkillHealthMonitor."""

from singularity.skill_health import SkillHealthMonitor


def _action(tool, status, message=""):
    return {"tool": tool, "result": {"status": status, "message": message}}


def test_empty_actions():
    m = SkillHealthMonitor()
    h = m.analyze([])
    assert h["skill_stats"] == {}
    assert h["failing_skills"] == []
    assert h["consecutive_failures"] == 0


def test_all_success():
    m = SkillHealthMonitor()
    actions = [_action("fs:read", "success") for _ in range(5)]
    h = m.analyze(actions)
    assert h["skill_stats"]["fs:read"]["rate"] == 1.0
    assert h["failing_skills"] == []
    assert h["consecutive_failures"] == 0


def test_mixed_results():
    m = SkillHealthMonitor()
    actions = [
        _action("fs:read", "success"),
        _action("fs:read", "error", "file not found"),
        _action("fs:read", "success"),
    ]
    h = m.analyze(actions)
    assert h["skill_stats"]["fs:read"]["success"] == 2
    assert h["skill_stats"]["fs:read"]["failure"] == 1
    assert h["skill_stats"]["fs:read"]["rate"] == 0.67


def test_consecutive_failures_detected():
    m = SkillHealthMonitor(failure_threshold=3)
    actions = [
        _action("shell:bash", "error", "timeout"),
        _action("shell:bash", "error", "timeout"),
        _action("shell:bash", "error", "timeout"),
    ]
    h = m.analyze(actions)
    assert "shell:bash" in h["failing_skills"]
    assert h["consecutive_failures"] == 3


def test_consecutive_failures_reset_on_success():
    m = SkillHealthMonitor(failure_threshold=3)
    actions = [
        _action("shell:bash", "error", "timeout"),
        _action("shell:bash", "error", "timeout"),
        _action("shell:bash", "success"),
        _action("shell:bash", "error", "timeout"),
    ]
    h = m.analyze(actions)
    assert "shell:bash" not in h["failing_skills"]


def test_overall_consecutive_failures():
    m = SkillHealthMonitor()
    actions = [
        _action("fs:read", "success"),
        _action("shell:bash", "error", "err1"),
        _action("fs:write", "failed", "err2"),
    ]
    h = m.analyze(actions)
    assert h["consecutive_failures"] == 2


def test_generate_context_healthy():
    m = SkillHealthMonitor()
    actions = [_action("fs:read", "success") for _ in range(5)]
    ctx = m.generate_context(actions)
    assert ctx == ""


def test_generate_context_warnings():
    m = SkillHealthMonitor(failure_threshold=2)
    actions = [
        _action("shell:bash", "error", "command not found"),
        _action("shell:bash", "error", "command not found"),
    ]
    ctx = m.generate_context(actions)
    assert "FAILING TOOLS" in ctx
    assert "shell:bash" in ctx
    assert "command not found" in ctx


def test_generate_context_streak_warning():
    m = SkillHealthMonitor(failure_threshold=10)
    actions = [
        _action("a:x", "error", "e1"),
        _action("b:y", "failed", "e2"),
        _action("c:z", "error", "e3"),
    ]
    ctx = m.generate_context(actions)
    assert "3 consecutive failures" in ctx
    assert "reconsider" in ctx


def test_lookback_limit():
    m = SkillHealthMonitor(lookback=3)
    actions = [_action("fs:read", "error", "old")] * 10 + [_action("fs:read", "success")] * 3
    h = m.analyze(actions)
    assert h["skill_stats"]["fs:read"]["rate"] == 1.0


def test_last_errors_tracked():
    m = SkillHealthMonitor()
    actions = [
        _action("fs:read", "error", "first error"),
        _action("fs:read", "error", "second error"),
    ]
    h = m.analyze(actions)
    assert h["last_errors"]["fs:read"] == "second error"


def test_multiple_skills():
    m = SkillHealthMonitor(failure_threshold=2)
    actions = [
        _action("fs:read", "success"),
        _action("shell:bash", "error", "e1"),
        _action("shell:bash", "error", "e2"),
        _action("fs:write", "success"),
    ]
    h = m.analyze(actions)
    assert "shell:bash" in h["failing_skills"]
    assert "fs:read" not in h["failing_skills"]
    assert h["skill_stats"]["fs:read"]["rate"] == 1.0
    assert h["skill_stats"]["shell:bash"]["rate"] == 0.0
