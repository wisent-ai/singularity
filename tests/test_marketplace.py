#!/usr/bin/env python3
"""Tests for MarketplaceSkill - service catalog, orders, and revenue tracking."""

import pytest
import json
from pathlib import Path
from unittest.mock import patch, AsyncMock, MagicMock
from singularity.skills.marketplace import MarketplaceSkill, MARKETPLACE_FILE
from singularity.skills.base import SkillResult, SkillContext, SkillRegistry


@pytest.fixture
def skill(tmp_path):
    test_file = tmp_path / "marketplace.json"
    with patch("singularity.skills.marketplace.MARKETPLACE_FILE", test_file):
        s = MarketplaceSkill()
        yield s


@pytest.fixture
def skill_with_context(tmp_path):
    test_file = tmp_path / "marketplace.json"
    with patch("singularity.skills.marketplace.MARKETPLACE_FILE", test_file):
        s = MarketplaceSkill()
        registry = SkillRegistry()
        ctx = SkillContext(registry=registry, agent_name="TestAgent")
        ctx.call_skill = AsyncMock(return_value=SkillResult(success=True, message="done", data={"output": "ok"}))
        s.set_context(ctx)
        yield s


async def _create_test_service(skill, tmp_path, name="Code Review", price=5.0):
    test_file = tmp_path / "marketplace.json"
    with patch("singularity.skills.marketplace.MARKETPLACE_FILE", test_file):
        return await skill.execute("create_service", {
            "name": name,
            "description": "AI-powered code review",
            "skill_id": "content_creation",
            "action": "write_article",
            "price": price,
            "estimated_cost": 1.0,
            "tags": ["code", "review"],
        })


@pytest.mark.asyncio
async def test_create_service(skill, tmp_path):
    test_file = tmp_path / "marketplace.json"
    with patch("singularity.skills.marketplace.MARKETPLACE_FILE", test_file):
        result = await _create_test_service(skill, tmp_path)
        assert result.success
        assert "Code Review" in result.message
        assert result.data["service"]["price"] == 5.0
        assert result.data["service"]["status"] == "active"


@pytest.mark.asyncio
async def test_create_service_missing_fields(skill, tmp_path):
    test_file = tmp_path / "marketplace.json"
    with patch("singularity.skills.marketplace.MARKETPLACE_FILE", test_file):
        result = await skill.execute("create_service", {"name": "Test"})
        assert not result.success
        assert "Required" in result.message


@pytest.mark.asyncio
async def test_create_duplicate_service(skill, tmp_path):
    test_file = tmp_path / "marketplace.json"
    with patch("singularity.skills.marketplace.MARKETPLACE_FILE", test_file):
        await _create_test_service(skill, tmp_path)
        result = await _create_test_service(skill, tmp_path)
        assert not result.success
        assert "already exists" in result.message


@pytest.mark.asyncio
async def test_list_services(skill, tmp_path):
    test_file = tmp_path / "marketplace.json"
    with patch("singularity.skills.marketplace.MARKETPLACE_FILE", test_file):
        await _create_test_service(skill, tmp_path, "Svc A", 10.0)
        await _create_test_service(skill, tmp_path, "Svc B", 20.0)
        result = await skill.execute("list_services", {})
        assert result.success
        assert result.data["total"] == 2


@pytest.mark.asyncio
async def test_list_services_filter_by_tag(skill, tmp_path):
    test_file = tmp_path / "marketplace.json"
    with patch("singularity.skills.marketplace.MARKETPLACE_FILE", test_file):
        await _create_test_service(skill, tmp_path, "Svc A", 10.0)
        result = await skill.execute("list_services", {"tag": "code"})
        assert result.success
        assert result.data["total"] == 1


@pytest.mark.asyncio
async def test_update_service_price(skill, tmp_path):
    test_file = tmp_path / "marketplace.json"
    with patch("singularity.skills.marketplace.MARKETPLACE_FILE", test_file):
        create_result = await _create_test_service(skill, tmp_path)
        sid = create_result.data["service"]["id"]
        result = await skill.execute("update_service", {"service_id": sid, "price": 10.0})
        assert result.success
        assert "price" in result.data["updates"]


@pytest.mark.asyncio
async def test_update_service_status(skill, tmp_path):
    test_file = tmp_path / "marketplace.json"
    with patch("singularity.skills.marketplace.MARKETPLACE_FILE", test_file):
        create_result = await _create_test_service(skill, tmp_path)
        sid = create_result.data["service"]["id"]
        result = await skill.execute("update_service", {"service_id": sid, "status": "paused"})
        assert result.success


@pytest.mark.asyncio
async def test_create_order(skill, tmp_path):
    test_file = tmp_path / "marketplace.json"
    with patch("singularity.skills.marketplace.MARKETPLACE_FILE", test_file):
        create_result = await _create_test_service(skill, tmp_path)
        sid = create_result.data["service"]["id"]
        result = await skill.execute("create_order", {
            "service_id": sid,
            "customer_id": "customer-1",
            "params": {"repo_url": "https://github.com/example/repo"},
        })
        assert result.success
        assert result.data["order"]["status"] == "pending"
        assert result.data["order"]["price"] == 5.0


@pytest.mark.asyncio
async def test_create_order_paused_service(skill, tmp_path):
    test_file = tmp_path / "marketplace.json"
    with patch("singularity.skills.marketplace.MARKETPLACE_FILE", test_file):
        create_result = await _create_test_service(skill, tmp_path)
        sid = create_result.data["service"]["id"]
        await skill.execute("update_service", {"service_id": sid, "status": "paused"})
        result = await skill.execute("create_order", {"service_id": sid, "customer_id": "c1"})
        assert not result.success
        assert "paused" in result.message


@pytest.mark.asyncio
async def test_complete_order_records_revenue(skill, tmp_path):
    test_file = tmp_path / "marketplace.json"
    with patch("singularity.skills.marketplace.MARKETPLACE_FILE", test_file):
        create_result = await _create_test_service(skill, tmp_path, price=10.0)
        sid = create_result.data["service"]["id"]
        order_result = await skill.execute("create_order", {"service_id": sid, "customer_id": "c1"})
        oid = order_result.data["order"]["id"]
        result = await skill.execute("complete_order", {"order_id": oid, "actual_cost": 2.0})
        assert result.success
        assert result.revenue == 10.0
        assert result.data["profit"] == 8.0
        assert result.data["margin_pct"] == 80.0


@pytest.mark.asyncio
async def test_revenue_report(skill, tmp_path):
    test_file = tmp_path / "marketplace.json"
    with patch("singularity.skills.marketplace.MARKETPLACE_FILE", test_file):
        create_result = await _create_test_service(skill, tmp_path, price=10.0)
        sid = create_result.data["service"]["id"]
        # Create and complete two orders
        for cid in ("c1", "c2"):
            order = await skill.execute("create_order", {"service_id": sid, "customer_id": cid})
            await skill.execute("complete_order", {"order_id": order.data["order"]["id"], "actual_cost": 3.0})
        result = await skill.execute("revenue_report", {"days": 30})
        assert result.success
        report = result.data["report"]
        assert report["total_revenue"] == 20.0
        assert report["total_cost"] == 6.0
        assert report["total_profit"] == 14.0
        assert report["order_count"] == 2


@pytest.mark.asyncio
async def test_fulfill_order_with_context(skill_with_context, tmp_path):
    test_file = tmp_path / "marketplace.json"
    with patch("singularity.skills.marketplace.MARKETPLACE_FILE", test_file):
        create_result = await skill_with_context.execute("create_service", {
            "name": "Test Service", "description": "test", "skill_id": "shell",
            "action": "run_command", "price": 5.0,
        })
        sid = create_result.data["service"]["id"]
        order = await skill_with_context.execute("create_order", {"service_id": sid, "customer_id": "c1"})
        oid = order.data["order"]["id"]
        result = await skill_with_context.execute("fulfill_order", {"order_id": oid})
        assert result.success
        assert "fulfilled" in result.message


@pytest.mark.asyncio
async def test_adjust_pricing_demand_based(skill, tmp_path):
    test_file = tmp_path / "marketplace.json"
    with patch("singularity.skills.marketplace.MARKETPLACE_FILE", test_file):
        create_result = await _create_test_service(skill, tmp_path, price=10.0)
        sid = create_result.data["service"]["id"]
        result = await skill.execute("adjust_pricing", {
            "service_id": sid,
            "strategy": "demand_based",
        })
        assert result.success
        # 0 orders = 20% discount
        assert result.data["adjustment"]["new_price"] == 8.0


@pytest.mark.asyncio
async def test_adjust_pricing_competitive(skill, tmp_path):
    test_file = tmp_path / "marketplace.json"
    with patch("singularity.skills.marketplace.MARKETPLACE_FILE", test_file):
        create_result = await _create_test_service(skill, tmp_path, price=10.0)
        sid = create_result.data["service"]["id"]
        result = await skill.execute("adjust_pricing", {
            "service_id": sid, "strategy": "competitive",
        })
        assert result.success
        assert result.data["adjustment"]["new_price"] == 9.0


@pytest.mark.asyncio
async def test_unknown_action(skill, tmp_path):
    test_file = tmp_path / "marketplace.json"
    with patch("singularity.skills.marketplace.MARKETPLACE_FILE", test_file):
        result = await skill.execute("nonexistent", {})
        assert not result.success


@pytest.mark.asyncio
async def test_negative_price_rejected(skill, tmp_path):
    test_file = tmp_path / "marketplace.json"
    with patch("singularity.skills.marketplace.MARKETPLACE_FILE", test_file):
        result = await skill.execute("create_service", {
            "name": "Bad", "description": "bad", "skill_id": "x",
            "action": "y", "price": -5,
        })
        assert not result.success


@pytest.mark.asyncio
async def test_manifest(skill):
    m = skill.manifest
    assert m.skill_id == "marketplace"
    assert m.category == "revenue"
    assert len(m.actions) == 9


@pytest.mark.asyncio
async def test_initialize(skill):
    result = await skill.initialize()
    assert result is True
