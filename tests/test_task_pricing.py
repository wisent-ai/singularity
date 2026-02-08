#!/usr/bin/env python3
"""Tests for TaskPricingSkill - dynamic pricing engine for agent services."""

import json
import pytest
import asyncio
from pathlib import Path
from unittest.mock import patch

from singularity.skills.task_pricing import TaskPricingSkill, PRICING_FILE


@pytest.fixture
def skill(tmp_path):
    """Create a TaskPricingSkill with temp data file."""
    test_file = tmp_path / "task_pricing.json"
    with patch("singularity.skills.task_pricing.PRICING_FILE", test_file):
        s = TaskPricingSkill()
        yield s


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


class TestEstimate:
    def test_basic_estimate(self, skill):
        result = run(skill.execute("estimate", {
            "description": "Review my Python code for bugs",
            "skills_needed": ["code_review"],
        }))
        assert result.success
        assert result.data["price"] > 0
        assert result.data["estimated_cost"] > 0
        assert result.data["breakdown"]["complexity"] in ("low", "medium", "high")

    def test_high_complexity_costs_more(self, skill):
        low = run(skill.execute("estimate", {
            "description": "Simple format check",
            "skills_needed": ["code_review"],
        }))
        high = run(skill.execute("estimate", {
            "description": "Comprehensive security audit of enterprise multi-step integration architecture",
            "skills_needed": ["code_review", "browser", "web_scraper"],
        }))
        assert high.data["price"] > low.data["price"]

    def test_urgency_multiplier(self, skill):
        normal = run(skill.execute("estimate", {
            "description": "Analyze data", "skills_needed": ["content"], "urgency": "normal",
        }))
        critical = run(skill.execute("estimate", {
            "description": "Analyze data", "skills_needed": ["content"], "urgency": "critical",
        }))
        assert critical.data["price"] > normal.data["price"]

    def test_empty_description_fails(self, skill):
        result = run(skill.execute("estimate", {"description": ""}))
        assert not result.success

    def test_string_skills_parsed(self, skill):
        result = run(skill.execute("estimate", {
            "description": "Review code",
            "skills_needed": "code_review, content",
        }))
        assert result.success
        assert len(result.data["breakdown"]["skill_costs"]) == 2


class TestQuote:
    def test_generate_quote(self, skill):
        result = run(skill.execute("quote", {
            "description": "Build a landing page",
            "skills_needed": ["content", "deployment"],
            "customer_id": "cust-123",
        }))
        assert result.success
        assert result.data["quote_id"].startswith("QT-")
        assert result.data["status"] == "pending"
        assert result.data["customer_id"] == "cust-123"

    def test_accept_quote(self, skill):
        q = run(skill.execute("quote", {
            "description": "Write docs", "skills_needed": ["content"],
        }))
        quote_id = q.data["quote_id"]
        result = run(skill.execute("accept_quote", {"quote_id": quote_id}))
        assert result.success
        assert result.data["status"] == "accepted"

    def test_accept_nonexistent_quote_fails(self, skill):
        result = run(skill.execute("accept_quote", {"quote_id": "QT-FAKE"}))
        assert not result.success

    def test_double_accept_fails(self, skill):
        q = run(skill.execute("quote", {"description": "Task", "skills_needed": ["content"]}))
        qid = q.data["quote_id"]
        run(skill.execute("accept_quote", {"quote_id": qid}))
        result = run(skill.execute("accept_quote", {"quote_id": qid}))
        assert not result.success


class TestRecordActual:
    def test_record_and_calibrate(self, skill):
        q = run(skill.execute("quote", {"description": "Analyze data", "skills_needed": ["content"]}))
        qid = q.data["quote_id"]
        run(skill.execute("accept_quote", {"quote_id": qid}))
        result = run(skill.execute("record_actual", {
            "quote_id": qid, "actual_cost": 0.05, "revenue_collected": 0.10,
        }))
        assert result.success
        assert result.data["profit"] > 0
        assert "error_pct" in result.data

    def test_record_nonexistent_fails(self, skill):
        result = run(skill.execute("record_actual", {"quote_id": "QT-NONE", "actual_cost": 0.01}))
        assert not result.success

    def test_calibration_adjusts_after_multiple_records(self, skill):
        # Record several underestimates to trigger calibration
        for i in range(6):
            q = run(skill.execute("quote", {"description": f"Task {i}", "skills_needed": ["content"]}))
            qid = q.data["quote_id"]
            estimated = q.data["estimated_cost"]
            # Actual cost is 2x the estimate (consistent underestimate)
            run(skill.execute("record_actual", {
                "quote_id": qid, "actual_cost": estimated * 2.0, "revenue_collected": q.data["price"],
            }))
        report = run(skill.execute("pricing_report", {}))
        assert report.success
        # Calibration should detect positive bias (underestimating)
        assert report.data["calibration"]["avg_error_pct"] > 0


class TestPricingReport:
    def test_empty_report(self, skill):
        result = run(skill.execute("pricing_report", {}))
        assert result.success
        assert result.data["entries"] == 0

    def test_report_with_data(self, skill):
        q = run(skill.execute("quote", {"description": "Review code", "skills_needed": ["code_review"]}))
        run(skill.execute("record_actual", {
            "quote_id": q.data["quote_id"], "actual_cost": 0.02, "revenue_collected": q.data["price"],
        }))
        result = run(skill.execute("pricing_report", {}))
        assert result.success
        assert result.data["entries_analyzed"] == 1
        assert "financial" in result.data


class TestConfig:
    def test_adjust_margin(self, skill):
        result = run(skill.execute("adjust_config", {"margin_pct": 50.0}))
        assert result.success
        assert result.data["default_margin_pct"] == 50.0

    def test_set_skill_cost(self, skill):
        result = run(skill.execute("set_skill_cost", {"skill_id": "custom_skill", "cost": 0.05}))
        assert result.success
        # Now estimate should use the override
        est = run(skill.execute("estimate", {"description": "Use custom", "skills_needed": ["custom_skill"]}))
        assert est.success


class TestBulkEstimate:
    def test_bulk_estimate(self, skill):
        result = run(skill.execute("bulk_estimate", {
            "tasks": [
                {"description": "Task A", "skills_needed": ["content"]},
                {"description": "Task B", "skills_needed": ["code_review"]},
                {"description": "Task C", "skills_needed": ["browser"]},
            ],
        }))
        assert result.success
        assert result.data["task_count"] == 3
        assert result.data["batch_discount_pct"] == 5.0
        assert result.data["discounted_total"] < result.data["total_price"]

    def test_bulk_string_tasks(self, skill):
        result = run(skill.execute("bulk_estimate", {
            "tasks": ["Simple task 1", "Simple task 2", "Simple task 3"],
        }))
        assert result.success
        assert result.data["task_count"] == 3


class TestEdgeCases:
    def test_unknown_action(self, skill):
        result = run(skill.execute("nonexistent", {}))
        assert not result.success

    def test_manifest(self, skill):
        m = skill.manifest
        assert m.skill_id == "task_pricing"
        assert len(m.actions) == 8
