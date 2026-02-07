"""Tests for UsageTrackingSkill - per-customer API usage metering, rate limiting, and billing."""

import asyncio
import json
import pytest
from pathlib import Path
from unittest.mock import patch

from singularity.skills.usage_tracking import UsageTrackingSkill, USAGE_FILE


@pytest.fixture(autouse=True)
def clean_state(tmp_path):
    """Use temp file for each test."""
    test_file = tmp_path / "usage_tracking.json"
    with patch("singularity.skills.usage_tracking.USAGE_FILE", test_file):
        yield test_file


@pytest.fixture
def skill(clean_state):
    with patch("singularity.skills.usage_tracking.USAGE_FILE", clean_state):
        return UsageTrackingSkill()


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def test_manifest(skill):
    m = skill.manifest
    assert m.skill_id == "usage_tracking"
    assert m.category == "revenue"
    action_names = [a.name for a in m.actions]
    assert "register_customer" in action_names
    assert "record_usage" in action_names
    assert "check_rate_limit" in action_names
    assert "generate_invoice" in action_names


def test_register_customer(skill):
    result = run(skill.execute("register_customer", {"name": "Acme Corp", "tier": "basic"}))
    assert result.success
    assert "api_key" in result.data
    assert result.data["tier"] == "basic"
    assert result.data["customer_id"].startswith("cust_")


def test_register_customer_invalid_tier(skill):
    result = run(skill.execute("register_customer", {"name": "Bad", "tier": "nonexistent"}))
    assert not result.success
    assert "Invalid tier" in result.message


def test_record_usage(skill):
    # Register customer first
    reg = run(skill.execute("register_customer", {"name": "Test Co", "tier": "premium"}))
    api_key = reg.data["api_key"]

    # Record usage
    result = run(skill.execute("record_usage", {
        "api_key": api_key, "skill_id": "code_review", "action": "review",
        "cost": 0.01, "latency_ms": 150, "success": True,
    }))
    assert result.success
    assert result.data["revenue"] == 0.0005  # premium price per request


def test_record_usage_invalid_key(skill):
    result = run(skill.execute("record_usage", {"api_key": "invalid_key", "skill_id": "x", "action": "y"}))
    assert not result.success
    assert "Invalid API key" in result.message


def test_check_rate_limit_allowed(skill):
    reg = run(skill.execute("register_customer", {"name": "Rate Test", "tier": "premium"}))
    api_key = reg.data["api_key"]

    result = run(skill.execute("check_rate_limit", {"api_key": api_key}))
    assert result.success
    assert result.data["allowed"] is True


def test_check_rate_limit_exceeded(skill):
    reg = run(skill.execute("register_customer", {"name": "Limited", "tier": "free"}))
    api_key = reg.data["api_key"]

    # Free tier: 10 requests per minute - record 10 to exhaust
    for _ in range(10):
        run(skill.execute("record_usage", {"api_key": api_key, "skill_id": "x", "action": "y"}))

    result = run(skill.execute("check_rate_limit", {"api_key": api_key}))
    assert result.success
    assert result.data["allowed"] is False
    assert result.data["violated"] == "requests_per_minute"


def test_get_usage_report_single_customer(skill):
    reg = run(skill.execute("register_customer", {"name": "Report Co", "tier": "basic"}))
    api_key = reg.data["api_key"]
    cid = reg.data["customer_id"]

    # Record some usage
    for i in range(5):
        run(skill.execute("record_usage", {
            "api_key": api_key, "skill_id": "code_review", "action": "review",
            "cost": 0.01, "latency_ms": 100 + i * 10,
        }))

    result = run(skill.execute("get_usage_report", {"customer_id": cid, "period": "daily"}))
    assert result.success
    assert result.data["total_requests"] == 5
    assert result.data["name"] == "Report Co"


def test_get_usage_report_all_customers(skill):
    for name in ["Alice", "Bob"]:
        reg = run(skill.execute("register_customer", {"name": name}))
        run(skill.execute("record_usage", {
            "api_key": reg.data["api_key"], "skill_id": "seo", "action": "audit",
        }))

    result = run(skill.execute("get_usage_report", {"period": "daily"}))
    assert result.success
    assert result.data["customer_count"] == 2


def test_generate_invoice(skill):
    reg = run(skill.execute("register_customer", {"name": "Invoice Co", "tier": "basic"}))
    api_key = reg.data["api_key"]
    cid = reg.data["customer_id"]

    for _ in range(10):
        run(skill.execute("record_usage", {
            "api_key": api_key, "skill_id": "summarize", "action": "run", "cost": 0.005,
        }))

    result = run(skill.execute("generate_invoice", {"customer_id": cid}))
    assert result.success
    assert result.data["invoice_id"].startswith("inv_")
    assert result.data["total_requests"] == 10
    # Basic tier: 0.001 per request × 10 = 0.01
    assert result.data["total"] == 0.01


def test_get_analytics(skill):
    for name, tier in [("Whale", "premium"), ("Minnow", "free")]:
        reg = run(skill.execute("register_customer", {"name": name, "tier": tier}))
        for _ in range(3):
            run(skill.execute("record_usage", {
                "api_key": reg.data["api_key"], "skill_id": "review", "action": "run",
            }))

    result = run(skill.execute("get_analytics", {}))
    assert result.success
    assert result.data["summary"]["customer_count"] == 2
    assert result.data["summary"]["total_requests"] == 6
    assert len(result.data["popular_services"]) >= 1


def test_update_tier(skill):
    reg = run(skill.execute("register_customer", {"name": "Upgrader", "tier": "free"}))
    cid = reg.data["customer_id"]

    result = run(skill.execute("update_tier", {"customer_id": cid, "tier": "premium"}))
    assert result.success
    assert result.data["old_tier"] == "free"
    assert result.data["new_tier"] == "premium"


def test_unknown_action(skill):
    result = run(skill.execute("nonexistent", {}))
    assert not result.success
    assert "Unknown action" in result.message
