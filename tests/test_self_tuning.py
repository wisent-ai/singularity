"""Tests for SelfTuningSkill."""

import json
import asyncio
import pytest
from singularity.skills.self_tuning import (
    SelfTuningSkill, TUNING_FILE,
    STRATEGY_LINEAR, STRATEGY_STEP, STRATEGY_EXPONENTIAL,
)


@pytest.fixture(autouse=True)
def clean_data():
    if TUNING_FILE.exists():
        TUNING_FILE.unlink()
    yield
    if TUNING_FILE.exists():
        TUNING_FILE.unlink()


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def make_skill():
    return SelfTuningSkill()


def test_manifest():
    skill = make_skill()
    m = skill.manifest
    assert m.skill_id == "self_tuning"
    actions = [a.name for a in m.actions]
    assert "tune" in actions
    assert "add_rule" in actions
    assert "list_rules" in actions
    assert "delete_rule" in actions
    assert "history" in actions
    assert "rollback" in actions
    assert "status" in actions


def _add_test_rule(skill, name="test_rule", metric="latency_ms", condition="above",
                    threshold=1000, target_skill="api_gateway", target_param="timeout_ms",
                    strategy="step", value=100, min_val=100, max_val=10000):
    return run(skill.execute("add_rule", {
        "name": name,
        "metric_name": metric,
        "condition": condition,
        "threshold": threshold,
        "target_skill": target_skill,
        "target_param": target_param,
        "adjustment": {
            "strategy": strategy,
            "value": value,
            "default": 500,
            "min": min_val,
            "max": max_val,
        },
        "cooldown_minutes": 0,  # No cooldown for tests
    }))


def test_add_rule():
    skill = make_skill()
    result = _add_test_rule(skill)
    assert result.success
    assert "rule_id" in result.data
    assert result.data["rule"]["name"] == "test_rule"
    assert result.data["rule"]["condition"] == "above"


def test_add_rule_validation():
    skill = make_skill()
    # Missing name
    r = run(skill.execute("add_rule", {"metric_name": "x", "condition": "above", "threshold": 1,
                                         "target_skill": "s", "target_param": "p",
                                         "adjustment": {"strategy": "step"}}))
    assert not r.success

    # Invalid condition
    r = run(skill.execute("add_rule", {"name": "x", "metric_name": "x", "condition": "invalid",
                                         "target_skill": "s", "target_param": "p",
                                         "adjustment": {"strategy": "step"}}))
    assert not r.success

    # Missing threshold for above condition
    r = run(skill.execute("add_rule", {"name": "x", "metric_name": "x", "condition": "above",
                                         "target_skill": "s", "target_param": "p",
                                         "adjustment": {"strategy": "step"}}))
    assert not r.success


def test_list_rules():
    skill = make_skill()
    _add_test_rule(skill, name="rule1")
    _add_test_rule(skill, name="rule2")
    result = run(skill.execute("list_rules", {}))
    assert result.success
    assert len(result.data["rules"]) == 2


def test_delete_rule():
    skill = make_skill()
    add_result = _add_test_rule(skill)
    rule_id = add_result.data["rule_id"]
    result = run(skill.execute("delete_rule", {"rule_id": rule_id}))
    assert result.success
    assert "Deleted" in result.message

    # Verify deleted
    list_result = run(skill.execute("list_rules", {}))
    assert len(list_result.data["rules"]) == 0


def test_tune_no_metric_data():
    """Tune with no metric data should skip rules."""
    skill = make_skill()
    _add_test_rule(skill)
    result = run(skill.execute("tune", {}))
    assert result.success
    assert len(result.data["adjustments"]) == 0
    assert len(result.data["skipped"]) == 1
    assert "no metric data" in result.data["skipped"][0]["reason"]


def test_tune_dry_run():
    """Dry run should show adjustments without applying."""
    skill = make_skill()
    _add_test_rule(skill)
    result = run(skill.execute("tune", {"dry_run": True}))
    assert result.success
    assert "[DRY RUN]" in result.message


def test_evaluate_condition_above():
    skill = make_skill()
    assert skill._evaluate_condition(100, "above", 50, []) is True
    assert skill._evaluate_condition(30, "above", 50, []) is False


def test_evaluate_condition_below():
    skill = make_skill()
    assert skill._evaluate_condition(30, "below", 50, []) is True
    assert skill._evaluate_condition(100, "below", 50, []) is False


def test_evaluate_condition_rising():
    skill = make_skill()
    assert skill._evaluate_condition(100, "rising", None, [10, 20, 30, 40, 50]) is True
    assert skill._evaluate_condition(100, "rising", None, [50, 40, 30, 20, 10]) is False
    assert skill._evaluate_condition(100, "rising", None, [10]) is False


def test_evaluate_condition_volatile():
    skill = make_skill()
    # High variance
    assert skill._evaluate_condition(100, "volatile", None, [10, 100, 20, 90, 15]) is True
    # Low variance
    assert skill._evaluate_condition(100, "volatile", None, [99, 100, 101, 100, 99]) is False


def test_compute_adjustment_step():
    skill = make_skill()
    config = {"strategy": "step", "value": 10, "min": 0, "max": 100}
    # Above → decrease
    result = skill._compute_adjustment(50, config, "above", 80, 60)
    assert result == 40

    # Below → increase
    result = skill._compute_adjustment(50, config, "below", 20, 60)
    assert result == 60


def test_compute_adjustment_exponential():
    skill = make_skill()
    config = {"strategy": "exponential", "factor": 2.0, "value": 1, "min": 1, "max": 1000}
    # Above → decrease (divide)
    result = skill._compute_adjustment(100, config, "above", 200, 100)
    assert result == 50.0

    # Below → increase (multiply)
    result = skill._compute_adjustment(100, config, "below", 50, 100)
    assert result == 200.0


def test_compute_adjustment_linear():
    skill = make_skill()
    config = {"strategy": "linear", "value": 10, "min": 0, "max": 1000}
    # Above with threshold=100, metric=200 → ratio=1.0, delta=10
    result = skill._compute_adjustment(500, config, "above", 200, 100)
    assert result == 490.0


def test_compute_adjustment_clamping():
    skill = make_skill()
    config = {"strategy": "step", "value": 100, "min": 50, "max": 200}
    # Should clamp to min
    result = skill._compute_adjustment(60, config, "above", 100, 50)
    assert result == 50

    # Should clamp to max
    result = skill._compute_adjustment(190, config, "below", 10, 50)
    assert result == 200


def test_status():
    skill = make_skill()
    _add_test_rule(skill)
    result = run(skill.execute("status", {}))
    assert result.success
    assert result.data["active_rules"] == 1
    assert result.data["total_rules"] == 1


def test_history_empty():
    skill = make_skill()
    result = run(skill.execute("history", {}))
    assert result.success
    assert len(result.data["history"]) == 0


def test_unknown_action():
    skill = make_skill()
    result = run(skill.execute("nonexistent", {}))
    assert not result.success
