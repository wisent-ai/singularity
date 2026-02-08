"""Tests for new revenue sources in RevenueAnalyticsDashboardSkill."""

import json
import pytest
import asyncio

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
    new_sources = {k: tmp_path / v.name for k, v in SOURCE_FILES.items()}
    monkeypatch.setattr("singularity.skills.revenue_analytics_dashboard.SOURCE_FILES", new_sources)
    yield tmp_path


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def _write(tmp_path, filename, data):
    (tmp_path / filename).write_text(json.dumps(data, default=str))


def test_database_revenue_bridge_collected(clean_data):
    """DatabaseRevenueBridge data appears in overview."""
    _write(clean_data, "database_revenue_bridge.json", {
        "jobs": [{"id": "j1", "customer_id": "c1", "service": "data_analysis"}],
        "revenue": {"total": 0.50, "by_service": {"data_analysis": 0.50}, "by_customer": {"c1": 0.50}},
        "stats": {"total_requests": 10, "successful_requests": 9, "failed_requests": 1},
        "reports": {"r1": {}},
        "schemas_created": {"s1": {}},
    })
    skill = RevenueAnalyticsDashboardSkill()
    result = run(skill.execute("overview", {}))
    assert result.success
    assert result.data["total_revenue"] == 0.50
    assert result.data["sources_active"] == 1


def test_http_revenue_bridge_collected(clean_data):
    """HTTPRevenueBridge data appears in overview."""
    _write(clean_data, "http_revenue_bridge.json", {
        "revenue": {"total": 1.20, "by_service": {"proxy_request": 1.0, "webhook_relay": 0.2}, "by_customer": {"c2": 1.20}},
        "stats": {"total_requests": 25, "successful_requests": 24, "failed_requests": 1},
        "history": [],
    })
    skill = RevenueAnalyticsDashboardSkill()
    result = run(skill.execute("overview", {}))
    assert result.success
    assert result.data["total_revenue"] == 1.20
    assert result.data["sources_active"] == 1


def test_api_marketplace_collected(clean_data):
    """APIMarketplace data appears in overview."""
    _write(clean_data, "usage.json", {
        "revenue": {"total": 2.00, "by_api": {"weather": 1.5, "geocode": 0.5}, "by_customer": {"c3": 2.0}},
        "total_calls": 50,
    })
    skill = RevenueAnalyticsDashboardSkill()
    result = run(skill.execute("overview", {}))
    assert result.success
    assert result.data["total_revenue"] == 2.00
    assert result.data["sources_active"] == 1


def test_all_three_new_sources_combined(clean_data):
    """All 3 new sources aggregate correctly."""
    _write(clean_data, "database_revenue_bridge.json", {
        "revenue": {"total": 1.0, "by_service": {}, "by_customer": {"c1": 1.0}},
        "stats": {"total_requests": 5, "successful_requests": 5, "failed_requests": 0},
        "reports": {}, "schemas_created": {}, "jobs": [],
    })
    _write(clean_data, "http_revenue_bridge.json", {
        "revenue": {"total": 2.0, "by_service": {}, "by_customer": {"c2": 2.0}},
        "stats": {"total_requests": 10, "successful_requests": 10, "failed_requests": 0},
    })
    _write(clean_data, "usage.json", {
        "revenue": {"total": 3.0, "by_api": {}, "by_customer": {"c3": 3.0}},
        "total_calls": 15,
    })
    skill = RevenueAnalyticsDashboardSkill()
    result = run(skill.execute("overview", {}))
    assert result.success
    assert result.data["total_revenue"] == 6.0
    assert result.data["sources_active"] == 3


def test_new_sources_in_by_source(clean_data):
    """New sources appear in by_source breakdown."""
    _write(clean_data, "database_revenue_bridge.json", {
        "revenue": {"total": 0.50, "by_service": {"data_analysis": 0.50}, "by_customer": {}},
        "stats": {"total_requests": 5, "successful_requests": 4, "failed_requests": 1},
        "reports": {"r1": {}}, "schemas_created": {}, "jobs": [],
    })
    _write(clean_data, "http_revenue_bridge.json", {
        "revenue": {"total": 0.30, "by_service": {"proxy_request": 0.30}, "by_customer": {}},
        "stats": {"total_requests": 3, "successful_requests": 3, "failed_requests": 0},
    })
    skill = RevenueAnalyticsDashboardSkill()
    result = run(skill.execute("by_source", {}))
    assert result.success
    assert "database_revenue_bridge" in result.data["sources"]
    assert "http_revenue_bridge" in result.data["sources"]
    src = result.data["sources"]["database_revenue_bridge"]
    assert src["revenue"] == 0.50
    assert src["reports_generated"] == 1


def test_new_sources_customers(clean_data):
    """Customers from new sources appear in customer analytics."""
    _write(clean_data, "database_revenue_bridge.json", {
        "revenue": {"total": 1.0, "by_service": {}, "by_customer": {"db-cust-1": 1.0}},
        "stats": {"total_requests": 3, "successful_requests": 3, "failed_requests": 0},
        "reports": {}, "schemas_created": {},
        "jobs": [
            {"customer_id": "db-cust-1", "service": "data_analysis"},
            {"customer_id": "db-cust-1", "service": "report_generation"},
        ],
    })
    _write(clean_data, "http_revenue_bridge.json", {
        "revenue": {"total": 0.5, "by_service": {}, "by_customer": {"http-cust-1": 0.5}},
        "stats": {"total_requests": 2, "successful_requests": 2, "failed_requests": 0},
    })
    skill = RevenueAnalyticsDashboardSkill()
    result = run(skill.execute("customers", {}))
    assert result.success
    assert result.data["total_customers"] == 2
    cust_ids = [c["customer_id"] for c in result.data["top_customers"]]
    assert "db-cust-1" in cust_ids
    assert "http-cust-1" in cust_ids


def test_mixed_old_and_new_sources(clean_data):
    """Old and new sources aggregate together correctly."""
    _write(clean_data, "task_pricing.json", {
        "revenue_summary": {"total_revenue": 5.0, "total_actual_cost": 2.0, "quote_count": 50},
    })
    _write(clean_data, "database_revenue_bridge.json", {
        "revenue": {"total": 1.0, "by_service": {}, "by_customer": {}},
        "stats": {"total_requests": 10, "successful_requests": 10, "failed_requests": 0},
        "reports": {}, "schemas_created": {}, "jobs": [],
    })
    skill = RevenueAnalyticsDashboardSkill()
    result = run(skill.execute("overview", {}))
    assert result.success
    assert result.data["total_revenue"] == 6.0
    assert result.data["sources_active"] == 2


def test_status_shows_all_sources():
    """Status action shows all 10 source files."""
    skill = RevenueAnalyticsDashboardSkill()
    result = run(skill.execute("status", {}))
    assert result.success
    source_names = list(result.data["sources"].keys())
    assert "database_revenue_bridge" in source_names
    assert "http_revenue_bridge" in source_names
    assert "api_marketplace" in source_names
