#!/usr/bin/env python3
"""Tests for PricingServiceBridgeSkill - auto-price API tasks and track revenue."""

import json
import pytest
import asyncio
from pathlib import Path
from unittest.mock import patch

from singularity.skills.pricing_service_bridge import PricingServiceBridgeSkill, BRIDGE_FILE


@pytest.fixture
def skill(tmp_path):
    test_file = tmp_path / "pricing_service_bridge.json"
    with patch("singularity.skills.pricing_service_bridge.BRIDGE_FILE", test_file):
        s = PricingServiceBridgeSkill()
        yield s


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


class TestQuoteTask:
    def test_quote_task(self, skill):
        result = run(skill.execute("quote_task", {
            "task_id": "task-001",
            "skill_id": "code_review",
            "action": "review",
            "description": "Review my Python code",
        }))
        assert result.success
        assert result.data["price"] > 0
        assert result.data["status"] == "pending"
        assert result.data["quote_id"].startswith("QT-")

    def test_quote_requires_task_id(self, skill):
        result = run(skill.execute("quote_task", {"skill_id": "code_review", "action": "review"}))
        assert not result.success

    def test_duplicate_quote_returns_existing(self, skill):
        run(skill.execute("quote_task", {
            "task_id": "task-dup", "skill_id": "shell", "action": "run",
        }))
        result = run(skill.execute("quote_task", {
            "task_id": "task-dup", "skill_id": "shell", "action": "run",
        }))
        assert result.success
        assert "already quoted" in result.message

    def test_urgency_affects_price(self, skill):
        normal = run(skill.execute("quote_task", {
            "task_id": "t-normal", "skill_id": "code_review", "action": "review",
            "urgency": "normal",
        }))
        critical = run(skill.execute("quote_task", {
            "task_id": "t-critical", "skill_id": "code_review", "action": "review",
            "urgency": "critical",
        }))
        assert critical.data["price"] > normal.data["price"]


class TestAcceptQuote:
    def test_accept_quote(self, skill):
        run(skill.execute("quote_task", {
            "task_id": "t-acc", "skill_id": "content", "action": "write",
        }))
        result = run(skill.execute("accept_task_quote", {"task_id": "t-acc"}))
        assert result.success
        assert result.data["status"] == "accepted"

    def test_accept_missing_quote(self, skill):
        result = run(skill.execute("accept_task_quote", {"task_id": "nonexistent"}))
        assert not result.success

    def test_double_accept(self, skill):
        run(skill.execute("quote_task", {
            "task_id": "t-dbl", "skill_id": "shell", "action": "run",
        }))
        run(skill.execute("accept_task_quote", {"task_id": "t-dbl"}))
        result = run(skill.execute("accept_task_quote", {"task_id": "t-dbl"}))
        assert result.success  # idempotent


class TestRecordCompletion:
    def test_record_completion(self, skill):
        run(skill.execute("quote_task", {
            "task_id": "t-comp", "skill_id": "code_review", "action": "review",
        }))
        result = run(skill.execute("record_completion", {
            "task_id": "t-comp", "actual_cost": 0.01, "execution_time_ms": 500,
        }))
        assert result.success
        assert result.data["status"] == "completed"
        assert result.data["revenue"] > 0
        assert result.data["profit"] is not None

    def test_record_estimates_cost_from_time(self, skill):
        run(skill.execute("quote_task", {
            "task_id": "t-est", "skill_id": "shell", "action": "run",
        }))
        result = run(skill.execute("record_completion", {
            "task_id": "t-est", "execution_time_ms": 2000,
        }))
        assert result.success
        assert result.data["actual_cost"] > 0


class TestRevenueDashboard:
    def test_empty_dashboard(self, skill):
        result = run(skill.execute("revenue_dashboard", {}))
        assert result.success
        assert result.data["all_time"]["tasks_quoted"] == 0

    def test_dashboard_after_tasks(self, skill):
        for i in range(3):
            run(skill.execute("quote_task", {
                "task_id": f"t-dash-{i}", "skill_id": "content", "action": "write",
            }))
            run(skill.execute("record_completion", {
                "task_id": f"t-dash-{i}", "actual_cost": 0.005,
            }))
        result = run(skill.execute("revenue_dashboard", {"time_range_hours": 1}))
        assert result.success
        assert result.data["all_time"]["tasks_executed"] == 3
        assert result.data["all_time"]["total_revenue_usd"] > 0


class TestHooks:
    def test_pre_execute_hook(self, skill):
        hook_result = skill.hook_pre_execute("t-hook", "code_review", "review", {})
        assert hook_result["allow"] is True
        assert hook_result["quote"] is not None
        assert hook_result["quote"]["price"] > 0

    def test_pre_execute_gated(self, skill):
        run(skill.execute("configure", {"quote_gated": True}))
        hook_result = skill.hook_pre_execute("t-gated", "shell", "run", {})
        assert hook_result["allow"] is False
        assert "pending" in hook_result["reason"].lower()

    def test_post_execute_hook(self, skill):
        skill.hook_pre_execute("t-post", "content", "write", {})
        result = skill.hook_post_execute("t-post", True, execution_time_ms=1000)
        assert result is not None


class TestConfig:
    def test_configure(self, skill):
        result = run(skill.execute("configure", {"quote_gated": True}))
        assert result.success
        assert result.data["quote_gated"] is True

    def test_status(self, skill):
        result = run(skill.execute("status", {}))
        assert result.success
        assert "config" in result.data


class TestPendingQuotes:
    def test_pending_quotes(self, skill):
        run(skill.execute("quote_task", {
            "task_id": "t-pend", "skill_id": "shell", "action": "run",
        }))
        result = run(skill.execute("pending_quotes", {}))
        assert result.success
        assert result.data["count"] == 1
