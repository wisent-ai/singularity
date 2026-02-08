"""Tests for RevenueAlertEscalationSkill."""
import pytest
from unittest.mock import patch
from singularity.skills.revenue_alert_escalation import (
    RevenueAlertEscalationSkill, DEFAULT_RULES,
)


@pytest.fixture
def skill(tmp_path):
    with patch("singularity.skills.revenue_alert_escalation.DATA_DIR", tmp_path), \
         patch("singularity.skills.revenue_alert_escalation.STATE_FILE", tmp_path / "state.json"):
        s = RevenueAlertEscalationSkill()
        yield s


@pytest.mark.asyncio
async def test_check_empty_metrics_fires_zero_alerts(skill):
    """Empty metrics should fire zero_revenue, low_success_rate, no_customers."""
    result = await skill.execute("check", {})
    assert result.success
    assert result.data["active_count"] >= 1
    fired_rules = [a["rule"] for a in result.data["fired"]]
    assert "zero_revenue" in fired_rules


@pytest.mark.asyncio
async def test_check_healthy_metrics(skill):
    """Healthy metrics should not fire any alerts."""
    metrics = {
        "revenue.total": 100.0,
        "revenue.requests.total": 50,
        "revenue.requests.success_rate": 95.0,
        "revenue.customers.active": 5,
    }
    result = await skill.execute("check", {"metrics": metrics})
    assert result.success
    assert result.data["active_count"] == 0


@pytest.mark.asyncio
async def test_check_zero_revenue_fires_alert(skill):
    metrics = {
        "revenue.total": 0.0,
        "revenue.requests.total": 0,
        "revenue.requests.success_rate": 0,
        "revenue.customers.active": 0,
    }
    result = await skill.execute("check", {"metrics": metrics})
    assert result.success
    fired_rules = [a["rule"] for a in result.data["fired"]]
    assert "zero_revenue" in fired_rules


@pytest.mark.asyncio
async def test_check_low_success_rate(skill):
    metrics = {
        "revenue.total": 50.0,
        "revenue.requests.total": 100,
        "revenue.requests.success_rate": 50.0,
        "revenue.customers.active": 3,
    }
    result = await skill.execute("check", {"metrics": metrics})
    assert result.success
    fired_rules = [a["rule"] for a in result.data["fired"]]
    assert "low_success_rate" in fired_rules


@pytest.mark.asyncio
async def test_alert_resolves_when_metrics_improve(skill):
    await skill.execute("check", {"metrics": {"revenue.total": 0.0, "revenue.requests.success_rate": 95.0, "revenue.customers.active": 2}})
    result = await skill.execute("check", {"metrics": {"revenue.total": 100.0, "revenue.requests.success_rate": 95.0, "revenue.customers.active": 2}})
    assert result.success
    resolved_rules = [a["rule"] for a in result.data["resolved"]]
    assert "zero_revenue" in resolved_rules


@pytest.mark.asyncio
async def test_rules_list(skill):
    result = await skill.execute("rules", {})
    assert result.success
    assert len(result.data["rules"]) == len(DEFAULT_RULES)


@pytest.mark.asyncio
async def test_rules_disable_enable(skill):
    result = await skill.execute("rules", {"action": "disable", "rule": "zero_revenue"})
    assert result.success
    result = await skill.execute("rules", {"action": "get", "rule": "zero_revenue"})
    assert not result.data["rule"]["enabled"]
    result = await skill.execute("rules", {"action": "enable", "rule": "zero_revenue"})
    assert result.success


@pytest.mark.asyncio
async def test_rules_set(skill):
    result = await skill.execute("rules", {
        "action": "set", "rule": "custom_rule",
        "config": {"metric": "revenue.total", "condition": "below", "threshold": 10, "severity": "high", "enabled": True},
    })
    assert result.success
    result = await skill.execute("rules", {"action": "get", "rule": "custom_rule"})
    assert result.data["rule"]["threshold"] == 10


@pytest.mark.asyncio
async def test_rules_reset(skill):
    await skill.execute("rules", {"action": "set", "rule": "custom_new_rule", "config": {"threshold": 999}})
    result = await skill.execute("rules", {"action": "reset"})
    assert result.success
    result = await skill.execute("rules", {})
    assert "custom_new_rule" not in result.data["rules"]


@pytest.mark.asyncio
async def test_status(skill):
    result = await skill.execute("status", {})
    assert result.success
    assert "active_count" in result.data
    assert "stats" in result.data


@pytest.mark.asyncio
async def test_history_after_alert(skill):
    """Zero revenue fires alerts, history should record them."""
    await skill.execute("check", {"metrics": {"revenue.total": 0.0, "revenue.requests.success_rate": 0, "revenue.customers.active": 0}})
    result = await skill.execute("history", {"limit": 10})
    assert result.success
    assert len(result.data["history"]) >= 1


@pytest.mark.asyncio
async def test_history_filter_by_rule(skill):
    await skill.execute("check", {"metrics": {"revenue.total": 0.0, "revenue.requests.success_rate": 50, "revenue.customers.active": 1}})
    result = await skill.execute("history", {"rule": "zero_revenue"})
    assert result.success
    assert all(h["rule"] == "zero_revenue" for h in result.data["history"])


@pytest.mark.asyncio
async def test_configure(skill):
    result = await skill.execute("configure", {"baseline_window": 10, "auto_create_incidents": False})
    assert result.success
    assert "baseline_window" in result.data["changed"]
    assert result.data["config"]["baseline_window"] == 10
    assert result.data["config"]["auto_create_incidents"] is False


@pytest.mark.asyncio
async def test_health_no_data(skill):
    result = await skill.execute("health", {})
    assert result.success
    assert "health_score" in result.data
    assert any("check" in r.lower() for r in result.data["recommendations"])


@pytest.mark.asyncio
async def test_health_with_active_alerts(skill):
    await skill.execute("check", {"metrics": {"revenue.total": 0.0, "revenue.requests.success_rate": 50, "revenue.customers.active": 0}})
    result = await skill.execute("health", {})
    assert result.success
    assert result.data["health_score"] < 100
    assert result.data["active_alerts"] > 0


@pytest.mark.asyncio
async def test_unknown_action(skill):
    result = await skill.execute("invalid_action", {})
    assert not result.success
