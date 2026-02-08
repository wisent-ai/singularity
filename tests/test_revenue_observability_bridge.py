#!/usr/bin/env python3
"""Tests for RevenueObservabilityBridgeSkill."""

import json
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from singularity.skills.revenue_observability_bridge import (
    RevenueObservabilityBridgeSkill,
    _extract_revenue_from_source,
    _load_state,
    _save_state,
    REVENUE_SOURCES,
    DATA_DIR,
    BRIDGE_STATE_FILE,
)


@pytest.fixture
def bridge():
    with patch("singularity.skills.revenue_observability_bridge._load_state") as mock_load:
        mock_load.return_value = {
            "config": {
                "auto_alerts": True,
                "alert_revenue_min": 0.0,
                "alert_success_rate_min": 80.0,
            },
            "sync_history": [],
            "alerts_created": [],
            "stats": {"total_syncs": 0, "last_sync": None, "metrics_emitted": 0},
        }
        skill = RevenueObservabilityBridgeSkill()
    return skill


class TestExtractRevenueFromSource:
    def test_none_data(self):
        result = _extract_revenue_from_source("test", None)
        assert result["revenue"] == 0.0
        assert result["requests"] == 0

    def test_revenue_bridge_format(self):
        data = {
            "revenue": {"total": 5.25, "by_service": {"proxy": 3.0, "data": 2.25}, "by_customer": {"c1": 3.0, "c2": 2.25}},
            "stats": {"total_requests": 100, "successful_requests": 95, "failed_requests": 5},
        }
        result = _extract_revenue_from_source("http_revenue_bridge", data)
        assert result["revenue"] == 5.25
        assert result["requests"] == 100
        assert result["successful_requests"] == 95
        assert len(result["customers"]) == 2

    def test_billing_pipeline_format(self):
        data = {
            "billing_cycles": [
                {"total_revenue": 10.0},
                {"total_revenue": 15.0},
            ],
            "customers": {"cust_a": {}, "cust_b": {}, "cust_c": {}},
        }
        result = _extract_revenue_from_source("billing_pipeline", data)
        assert result["revenue"] == 25.0
        assert len(result["customers"]) == 3

    def test_marketplace_format(self):
        data = {
            "orders": [{"id": 1}, {"id": 2}],
            "revenue_log": [{"amount": 1.5}, {"amount": 2.5}],
        }
        result = _extract_revenue_from_source("marketplace", data)
        assert result["revenue"] == 4.0
        assert result["requests"] == 2

    def test_task_pricing_format(self):
        data = {
            "revenue_summary": {"total_earned": 7.5, "tasks_completed": 50},
        }
        result = _extract_revenue_from_source("task_pricing", data)
        assert result["revenue"] == 7.5
        assert result["requests"] == 50

    def test_list_format_revenue_services(self):
        data = [
            {"price": 0.5, "customer_id": "c1"},
            {"price": 1.0, "customer_id": "c2"},
            {"price": 0.75},
        ]
        result = _extract_revenue_from_source("revenue_services", data)
        assert result["revenue"] == 2.25
        assert result["requests"] == 3
        assert len(result["customers"]) == 2


class TestBridgeSkill:
    def test_manifest(self, bridge):
        m = bridge.manifest
        assert m.name == "revenue_observability_bridge"
        actions = [a.name for a in m.actions]
        assert "sync" in actions
        assert "sources" in actions
        assert "snapshot" in actions
        assert "setup_alerts" in actions

    @pytest.mark.asyncio
    async def test_status(self, bridge):
        result = await bridge.execute("status", {})
        assert result.success
        assert result.data["sources_count"] == len(REVENUE_SOURCES)
        assert result.data["observability_connected"] is False

    @pytest.mark.asyncio
    async def test_sources(self, bridge):
        result = await bridge.execute("sources", {})
        assert result.success
        assert "sources" in result.data

    @pytest.mark.asyncio
    @patch("singularity.skills.revenue_observability_bridge._save_state")
    @patch("singularity.skills.revenue_observability_bridge._load_json")
    async def test_sync_without_observability(self, mock_load, mock_save, bridge):
        mock_load.return_value = None
        result = await bridge.execute("sync", {})
        assert result.success
        assert "metrics emitted" in result.message
        assert result.data["metrics_emitted"] == 0

    @pytest.mark.asyncio
    @patch("singularity.skills.revenue_observability_bridge._save_state")
    @patch("singularity.skills.revenue_observability_bridge._load_json")
    async def test_sync_with_observability(self, mock_load, mock_save, bridge):
        mock_load.return_value = {
            "revenue": {"total": 3.0, "by_service": {"svc1": 3.0}, "by_customer": {"c1": 3.0}},
            "stats": {"total_requests": 10, "successful_requests": 9, "failed_requests": 1},
        }
        mock_obs = MagicMock()
        mock_obs._emit.return_value = MagicMock(success=True)
        mock_obs._alert_create.return_value = MagicMock(success=True)
        bridge.set_observability(mock_obs)
        result = await bridge.execute("sync", {})
        assert result.success
        assert result.data["metrics_emitted"] > 0
        assert mock_obs._emit.call_count > 0

    @pytest.mark.asyncio
    async def test_snapshot(self, bridge):
        with patch("singularity.skills.revenue_observability_bridge._load_json", return_value=None):
            result = await bridge.execute("snapshot", {})
        assert result.success
        assert "total_revenue" in result.data

    @pytest.mark.asyncio
    async def test_configure(self, bridge):
        with patch("singularity.skills.revenue_observability_bridge._save_state"):
            result = await bridge.execute("configure", {"auto_alerts": False})
        assert result.success
        assert bridge._state["config"]["auto_alerts"] is False

    @pytest.mark.asyncio
    async def test_history_empty(self, bridge):
        result = await bridge.execute("history", {"limit": 5})
        assert result.success
        assert result.data["history"] == []

    @pytest.mark.asyncio
    async def test_setup_alerts_no_observability(self, bridge):
        result = await bridge.execute("setup_alerts", {})
        assert not result.success
        assert "ObservabilitySkill not connected" in result.message

    @pytest.mark.asyncio
    @patch("singularity.skills.revenue_observability_bridge._save_state")
    async def test_setup_alerts_with_observability(self, mock_save, bridge):
        mock_obs = MagicMock()
        mock_obs._alert_create.return_value = MagicMock(success=True)
        bridge.set_observability(mock_obs)
        result = await bridge.execute("setup_alerts", {"revenue_min": 1.0, "success_rate_min": 90.0})
        assert result.success
        assert len(result.data["alerts_created"]) == 2

    @pytest.mark.asyncio
    async def test_unknown_action(self, bridge):
        result = await bridge.execute("unknown_action", {})
        assert not result.success
