"""Tests for CostOptimizerSkill."""
import json
import pytest
import asyncio
from pathlib import Path
from unittest.mock import patch

from singularity.skills.cost_optimizer import CostOptimizerSkill, COST_FILE


@pytest.fixture(autouse=True)
def clean_data(tmp_path, monkeypatch):
    """Use a temp file for cost data."""
    test_file = tmp_path / "cost_optimizer.json"
    monkeypatch.setattr("singularity.skills.cost_optimizer.COST_FILE", test_file)
    yield test_file


@pytest.fixture
def skill():
    return CostOptimizerSkill()


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


class TestRecord:
    def test_record_basic(self, skill):
        r = run(skill.execute("record", {"skill_id": "llm", "action": "generate", "cost": 0.05, "revenue": 0.10}))
        assert r.success
        assert r.data["cost"] == 0.05
        assert r.data["revenue"] == 0.10
        assert r.data["profit"] == 0.05

    def test_record_updates_totals(self, skill):
        run(skill.execute("record", {"skill_id": "llm", "action": "gen", "cost": 0.01}))
        run(skill.execute("record", {"skill_id": "llm", "action": "gen", "cost": 0.02}))
        r = run(skill.execute("summary", {}))
        assert r.success
        assert r.data["total_cost"] == 0.03
        assert r.data["total_actions"] == 2

    def test_record_with_tokens(self, skill):
        r = run(skill.execute("record", {"skill_id": "llm", "action": "gen", "cost": 0.01, "tokens_used": 1000}))
        assert r.success
        assert r.data["tokens_used"] == 1000


class TestAnalyze:
    def test_analyze_empty(self, skill):
        r = run(skill.execute("analyze", {}))
        assert r.success
        assert r.data["entries_count"] == 0

    def test_analyze_groups_by_skill(self, skill):
        run(skill.execute("record", {"skill_id": "llm", "action": "gen", "cost": 0.05}))
        run(skill.execute("record", {"skill_id": "browser", "action": "fetch", "cost": 0.01}))
        r = run(skill.execute("analyze", {"group_by": "skill"}))
        assert r.success
        assert "llm" in r.data["groups"]
        assert "browser" in r.data["groups"]
        assert r.data["profitability"]["total_cost"] == 0.06

    def test_analyze_hotspots(self, skill):
        for i in range(3):
            run(skill.execute("record", {"skill_id": "expensive", "action": "op", "cost": 1.0}))
        run(skill.execute("record", {"skill_id": "cheap", "action": "op", "cost": 0.001}))
        r = run(skill.execute("analyze", {}))
        assert r.data["hotspots"][0]["name"] == "expensive"


class TestBudget:
    def test_set_and_check_budget(self, skill):
        run(skill.execute("set_budget", {"category": "llm", "limit_usd": 1.0, "period": "monthly"}))
        run(skill.execute("record", {"skill_id": "llm", "action": "gen", "cost": 0.5}))
        r = run(skill.execute("check_budget", {"skill_id": "llm", "estimated_cost": 0.1}))
        assert r.success
        assert r.data["allowed"]

    def test_budget_warning(self, skill):
        run(skill.execute("set_budget", {"category": "llm", "limit_usd": 1.0, "period": "monthly"}))
        run(skill.execute("record", {"skill_id": "llm", "action": "gen", "cost": 0.85}))
        r = run(skill.execute("check_budget", {"skill_id": "llm", "estimated_cost": 0.01}))
        assert len(r.data["warnings"]) > 0

    def test_budget_block(self, skill):
        run(skill.execute("set_budget", {"category": "total", "limit_usd": 1.0, "period": "total"}))
        run(skill.execute("record", {"skill_id": "llm", "action": "gen", "cost": 1.5}))
        r = run(skill.execute("check_budget", {"skill_id": "llm", "estimated_cost": 0.01}))
        assert r.data["blocked"]


class TestOptimize:
    def test_optimize_empty(self, skill):
        r = run(skill.execute("optimize", {}))
        assert r.success
        assert r.data["suggestions"] == []

    def test_optimize_detects_cost_centers(self, skill):
        for _ in range(3):
            run(skill.execute("record", {"skill_id": "waste", "action": "burn", "cost": 0.5}))
        r = run(skill.execute("optimize", {}))
        assert any(s["type"] == "cost_center" for s in r.data["suggestions"])

    def test_optimize_detects_low_margin(self, skill):
        for _ in range(3):
            run(skill.execute("record", {"skill_id": "svc", "action": "serve", "cost": 0.8, "revenue": 1.0}))
        r = run(skill.execute("optimize", {}))
        assert any(s["type"] == "low_margin" for s in r.data["suggestions"])


class TestProject:
    def test_project_insufficient_data(self, skill):
        r = run(skill.execute("project", {}))
        assert r.success
        assert r.data["reason"] == "insufficient_data"

    def test_project_with_data(self, skill):
        for i in range(5):
            run(skill.execute("record", {"skill_id": "llm", "action": "gen", "cost": 0.1, "revenue": 0.2}))
        r = run(skill.execute("project", {"days_ahead": 30}))
        assert r.success
        assert r.data["projected_cost"] > 0
        assert r.data["projected_revenue"] > 0
        assert r.data["projected_profit"] > 0


class TestSummary:
    def test_summary_empty(self, skill):
        r = run(skill.execute("summary", {}))
        assert r.success
        assert r.data["total_cost"] == 0

    def test_summary_with_data(self, skill):
        run(skill.execute("record", {"skill_id": "a", "action": "x", "cost": 0.1, "revenue": 0.3}))
        run(skill.execute("record", {"skill_id": "b", "action": "y", "cost": 0.2, "tokens_used": 500}))
        r = run(skill.execute("summary", {}))
        assert r.data["total_cost"] == 0.3
        assert r.data["total_revenue"] == 0.3
        assert r.data["unique_skills"] == 2
        assert r.data["total_tokens"] == 500
