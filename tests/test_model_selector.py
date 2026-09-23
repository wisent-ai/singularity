"""Tests for CostAwareModelSelector."""

import pytest
from datetime import datetime, timedelta
from singularity.model_selector import (
    CostAwareModelSelector,
    ModelTier,
    ModelRecommendation,
    DEFAULT_MODEL_TIERS,
)


@pytest.fixture
def selector():
    return CostAwareModelSelector(cooldown_seconds=0)


def test_critical_balance_forces_economy(selector):
    rec = selector.recommend(
        balance=0.05, burn_rate=0.01, runway_hours=0.5,
        current_model="claude-sonnet-4-20250514",
        current_provider="anthropic",
        available_providers=["anthropic", "openai"],
    )
    assert rec.should_switch
    assert rec.financial_health == "critical"


def test_comfortable_allows_premium(selector):
    rec = selector.recommend(
        balance=50.0, burn_rate=0.01, runway_hours=20.0,
        current_model="claude-sonnet-4-20250514",
        current_provider="anthropic",
        available_providers=["anthropic"],
    )
    assert not rec.should_switch
    assert rec.financial_health == "healthy"


def test_cautious_prefers_standard(selector):
    rec = selector.recommend(
        balance=5.0, burn_rate=0.01, runway_hours=3.0,
        current_model="claude-sonnet-4-20250514",
        current_provider="anthropic",
        available_providers=["anthropic", "openai"],
    )
    assert rec.should_switch
    assert rec.financial_health == "cautious"


def test_no_switch_when_already_economy(selector):
    rec = selector.recommend(
        balance=0.05, burn_rate=0.01, runway_hours=0.5,
        current_model="gpt-4o-mini",
        current_provider="openai",
        available_providers=["openai"],
    )
    assert not rec.should_switch


def test_cooldown_prevents_switch():
    sel = CostAwareModelSelector(cooldown_seconds=300)
    sel.state.last_switch_time = datetime.now()
    rec = sel.recommend(
        balance=0.01, burn_rate=0.01, runway_hours=0.1,
        current_model="claude-sonnet-4-20250514",
        current_provider="anthropic",
        available_providers=["anthropic", "openai"],
    )
    assert not rec.should_switch
    assert "cooldown" in rec.reason.lower()


def test_apply_records_switch(selector):
    rec = ModelRecommendation(
        should_switch=True, model_id="gpt-4o-mini", provider="openai",
        reason="test", tier_name="gpt-4o-mini", financial_health="critical",
        estimated_savings=0.005,
    )
    selector.apply(rec)
    assert selector.state.switch_count == 1
    assert selector.state.current_tier == "gpt-4o-mini"
    assert len(selector.state.switches) == 1


def test_no_providers_available(selector):
    rec = selector.recommend(
        balance=0.01, burn_rate=0.01, runway_hours=0.1,
        current_model="claude-sonnet-4-20250514",
        current_provider="anthropic",
        available_providers=["some_nonexistent_provider"],
    )
    assert not rec.should_switch


def test_estimate_savings(selector):
    savings = selector._estimate_savings(
        "claude-sonnet-4-20250514", "gpt-4o-mini"
    )
    assert savings > 0


def test_get_status(selector):
    status = selector.get_status()
    assert "thresholds" in status
    assert "switch_count" in status
    assert status["switch_count"] == 0


def test_health_classification(selector):
    assert selector._classify_health(0.01, 0.5) == "critical"
    assert selector._classify_health(5.0, 3.0) == "cautious"
    assert selector._classify_health(50.0, 20.0) == "healthy"
    assert selector._classify_health(10.0, 6.0) == "moderate"


def test_default_tiers_exist():
    assert len(DEFAULT_MODEL_TIERS) >= 3
    caps = {t.capability for t in DEFAULT_MODEL_TIERS}
    assert "economy" in caps
    assert "standard" in caps
    assert "premium" in caps


def test_model_tier_avg_cost():
    tier = ModelTier("test", "test", "test-model", 3.0, 15.0, "premium")
    cost = tier.avg_cost_per_call
    assert cost > 0
    assert cost == (1500 * 3.0 + 400 * 15.0) / 1_000_000
