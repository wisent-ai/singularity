"""Tests for RevenueAnalyticsDashboardSkill."""

import json
import pytest
import asyncio
from pathlib import Path

from singularity.skills.revenue_analytics_dashboard import (
    RevenueAnalyticsDashboardSkill,
    DASHBOARD_FILE,
    DATA_DIR,
    SOURCE_FILES,
)


@pytest.fixture(autouse=True)
def clean_data(tmp_path, monkeypatch):
    monkeypatch.setattr("singularity.skills.revenue_analytics_dashboard.DASHBOARD_FILE", tmp_path / "revenue_analytics_dashboard.json")
    monkeypatch.setattr("singularity.skills.revenue_analytics_dashboard.DATA_DIR", tmp_path)
    # Point source files to tmp_path too
    new_sources = {k: tmp_path / v.name for k, v in SOURCE_FILES.items()}
    monkeypatch.setattr("singularity.skills.revenue_analytics_dashboard.SOURCE_FILES", new_sources)
    yield tmp_path


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def _write_source(tmp_path, filename, data):
    (tmp_path / filename).write_text(json.dumps(data, default=str))


def test_overview_empty():
    skill = RevenueAnalyticsDashboardSkill()
    result = run(skill.execute("overview", {}))
    assert result.success
    assert result.data["total_revenue"] == 0
    assert result.data["sources_active"] == 0


def test_overview_with_data(clean_data):
    _write_source(clean_data, "task_pricing.json", {
        "revenue_summary": {"total_revenue": 1.50, "total_actual_cost": 0.50, "quote_count": 10, "acceptance_rate": 80},
    })
    _write_source(clean_data, "pricing_service_bridge.json", {
        "stats": {"total_revenue_usd": 2.00, "total_actual_usd": 0.80, "tasks_executed": 15, "tasks_quoted": 20, "total_profit_usd": 1.20, "avg_margin_pct": 60},
        "task_quotes": {},
    })
    skill = RevenueAnalyticsDashboardSkill()
    result = run(skill.execute("overview", {}))
    assert result.success
    assert result.data["total_revenue"] == 3.50
    assert result.data["sources_active"] == 2


def test_by_source(clean_data):
    _write_source(clean_data, "task_pricing.json", {
        "revenue_summary": {"total_revenue": 5.0, "total_actual_cost": 2.0, "quote_count": 50},
    })
    _write_source(clean_data, "marketplace.json", {
        "orders": [{"id": 1}], "revenue_log": [{"price": 3.0, "actual_cost": 1.0}], "services": [],
    })
    skill = RevenueAnalyticsDashboardSkill()
    result = run(skill.execute("by_source", {}))
    assert result.success
    assert "task_pricing" in result.data["sources"]
    assert "marketplace" in result.data["sources"]
    assert result.data["sources"]["task_pricing"]["revenue_share_pct"] > 0


def test_profitability(clean_data):
    _write_source(clean_data, "task_pricing.json", {
        "revenue_summary": {"total_revenue": 10.0, "total_actual_cost": 3.0, "quote_count": 100},
    })
    skill = RevenueAnalyticsDashboardSkill()
    result = run(skill.execute("profitability", {}))
    assert result.success
    assert result.data["overall_margin_pct"] == 70.0
    assert result.data["source_profitability"]["task_pricing"]["profitable"]


def test_customers(clean_data):
    _write_source(clean_data, "usage_tracking.json", {
        "customers": {
            "cust-1": {"tier": "premium", "total_revenue": 5.0, "total_requests": 100, "registered_at": "2026-01-01"},
            "cust-2": {"tier": "free", "total_revenue": 1.0, "total_requests": 20, "registered_at": "2026-01-15"},
        },
        "invoices": [],
    })
    skill = RevenueAnalyticsDashboardSkill()
    result = run(skill.execute("customers", {}))
    assert result.success
    assert result.data["total_customers"] == 2
    assert result.data["top_customers"][0]["customer_id"] == "cust-1"


def test_snapshot():
    skill = RevenueAnalyticsDashboardSkill()
    result = run(skill.execute("snapshot", {}))
    assert result.success
    assert result.data["total_snapshots"] == 1
    # Take another
    result2 = run(skill.execute("snapshot", {}))
    assert result2.data["total_snapshots"] == 2


def test_trends_insufficient_data():
    skill = RevenueAnalyticsDashboardSkill()
    result = run(skill.execute("trends", {}))
    assert result.success
    assert result.data["trend"] == "insufficient_data"


def test_trends_with_snapshots():
    skill = RevenueAnalyticsDashboardSkill()
    # Take multiple snapshots
    for _ in range(3):
        run(skill.execute("snapshot", {}))
    result = run(skill.execute("trends", {"window_hours": 24}))
    assert result.success
    assert "direction" in result.data
    assert len(result.data["timeline"]) >= 2


def test_forecast_insufficient():
    skill = RevenueAnalyticsDashboardSkill()
    result = run(skill.execute("forecast", {}))
    assert result.success
    assert result.data["forecast"] == "insufficient_data"


def test_forecast_with_data():
    skill = RevenueAnalyticsDashboardSkill()
    for _ in range(4):
        run(skill.execute("snapshot", {}))
    result = run(skill.execute("forecast", {"days_ahead": 3}))
    assert result.success
    assert len(result.data["forecasted_days"]) == 3


def test_recommendations():
    skill = RevenueAnalyticsDashboardSkill()
    result = run(skill.execute("recommendations", {}))
    assert result.success
    assert len(result.data["recommendations"]) > 0


def test_configure():
    skill = RevenueAnalyticsDashboardSkill()
    result = run(skill.execute("configure", {
        "compute_cost_per_hour": 0.25,
        "revenue_target_daily": 5.00,
    }))
    assert result.success
    assert result.data["config"]["compute_cost_per_hour"] == 0.25
    assert result.data["config"]["revenue_target_daily"] == 5.00


def test_status():
    skill = RevenueAnalyticsDashboardSkill()
    result = run(skill.execute("status", {}))
    assert result.success
    assert "sources" in result.data


def test_unknown_action():
    skill = RevenueAnalyticsDashboardSkill()
    result = run(skill.execute("nonexistent", {}))
    assert not result.success
