#!/usr/bin/env python3
"""Tests for DashboardSkill - Self-monitoring dashboard."""

import json
import pytest
import asyncio
from pathlib import Path
from unittest.mock import patch

from singularity.skills.dashboard import DashboardSkill, DATA_DIR, DASHBOARD_FILE


@pytest.fixture
def skill(tmp_path, monkeypatch):
    """Create DashboardSkill with temp data directory."""
    monkeypatch.setattr("singularity.skills.dashboard.DATA_DIR", tmp_path)
    monkeypatch.setattr("singularity.skills.dashboard.PERF_FILE", tmp_path / "performance.json")
    monkeypatch.setattr("singularity.skills.dashboard.RESOURCE_FILE", tmp_path / "resource_watcher.json")
    monkeypatch.setattr("singularity.skills.dashboard.GOALS_FILE", tmp_path / "goals.json")
    monkeypatch.setattr("singularity.skills.dashboard.SERVICE_FILE", tmp_path / "service_hosting.json")
    monkeypatch.setattr("singularity.skills.dashboard.USAGE_FILE", tmp_path / "usage_tracking.json")
    monkeypatch.setattr("singularity.skills.dashboard.HEALTH_FILE", tmp_path / "health_monitor.json")
    monkeypatch.setattr("singularity.skills.dashboard.DASHBOARD_FILE", tmp_path / "dashboard_history.json")
    return DashboardSkill()


@pytest.fixture
def populated_skill(skill, tmp_path, monkeypatch):
    """Skill with sample data in all subsystems."""
    # Performance data
    perf_data = {
        "records": [
            {"skill_id": "code_review", "action": "review", "success": True, "cost": 0.01, "revenue": 0.05, "latency": 0.5, "timestamp": "2025-01-01T00:00:00"},
            {"skill_id": "code_review", "action": "review", "success": True, "cost": 0.01, "revenue": 0.05, "latency": 0.3, "timestamp": "2025-01-02T00:00:00"},
            {"skill_id": "web_scraper", "action": "scrape", "success": False, "cost": 0.005, "revenue": 0, "latency": 2.0, "timestamp": "2025-01-03T00:00:00"},
        ]
    }
    (tmp_path / "performance.json").write_text(json.dumps(perf_data))

    # Goals data
    goals_data = {
        "goals": [
            {"title": "Revenue MVP", "status": "active", "pillar": "revenue", "priority": "high", "milestones": [{"name": "m1", "completed": True}, {"name": "m2", "completed": False}]},
            {"title": "Deploy v2", "status": "completed", "pillar": "replication", "priority": "medium", "milestones": []},
            {"title": "Self-eval", "status": "active", "pillar": "self_improvement", "priority": "medium", "milestones": [{"name": "m1", "completed": False}]},
        ]
    }
    (tmp_path / "goals.json").write_text(json.dumps(goals_data))

    # Resource data
    resource_data = {
        "budgets": {"api": {"limit": 10.0, "spent": 3.5}, "compute": {"limit": 5.0, "spent": 1.0}},
        "alerts": [{"type": "threshold", "resolved": False}],
        "consumption": [],
    }
    (tmp_path / "resource_watcher.json").write_text(json.dumps(resource_data))

    # Usage/revenue data
    usage_data = {
        "customers": {"cust1": {"status": "active"}, "cust2": {"status": "inactive"}},
        "usage": [
            {"skill_id": "code_review", "cost": 0.50},
            {"skill_id": "code_review", "cost": 0.30},
            {"skill_id": "content", "cost": 0.20},
        ],
    }
    (tmp_path / "usage_tracking.json").write_text(json.dumps(usage_data))

    # Service hosting data
    service_data = {
        "services": [
            {"name": "code-review-api", "status": "active", "request_count": 42, "endpoint": "/api/review"},
            {"name": "text-summarizer", "status": "inactive", "request_count": 5, "endpoint": "/api/summarize"},
        ]
    }
    (tmp_path / "service_hosting.json").write_text(json.dumps(service_data))

    # Health monitor data
    health_data = {
        "agents": {"agent-1": {"status": "healthy"}, "agent-2": {"status": "healthy"}, "agent-3": {"status": "unhealthy"}}
    }
    (tmp_path / "health_monitor.json").write_text(json.dumps(health_data))

    return skill


def test_manifest(skill):
    m = skill.manifest
    assert m.skill_id == "dashboard"
    assert m.category == "monitor"
    assert len(m.actions) == 5


def test_actions_list(skill):
    actions = skill.get_actions()
    names = [a.name for a in actions]
    assert "full_report" in names
    assert "summary" in names
    assert "pillar_scorecard" in names
    assert "export_html" in names
    assert "section_report" in names


def test_full_report_empty(skill):
    result = asyncio.get_event_loop().run_until_complete(skill.execute("full_report"))
    assert result.success
    assert "sections" in result.data
    assert "pillar_scores" in result.data


def test_full_report_with_data(populated_skill):
    result = asyncio.get_event_loop().run_until_complete(populated_skill.execute("full_report"))
    assert result.success
    perf = result.data["sections"]["performance"]
    assert perf["available"]
    assert perf["total_actions"] == 3
    assert perf["success_rate"] == pytest.approx(66.7, abs=0.1)


def test_summary(populated_skill):
    result = asyncio.get_event_loop().run_until_complete(populated_skill.execute("summary"))
    assert result.success
    assert "success rate" in result.message.lower() or "Success Rate" in result.message or "%" in result.message
    assert "pillar_scores" in result.data


def test_pillar_scorecard(populated_skill):
    result = asyncio.get_event_loop().run_until_complete(populated_skill.execute("pillar_scorecard"))
    assert result.success
    scores = result.data["scores"]
    assert "self_improvement" in scores
    assert "revenue" in scores
    assert "replication" in scores
    assert "goal_setting" in scores
    assert "overall" in scores
    for k, v in scores.items():
        assert 0 <= v["score"] <= 100
        assert v["grade"] in ("A", "B", "C", "D", "F")


def test_export_html(populated_skill, tmp_path):
    out = str(tmp_path / "test_dash.html")
    result = asyncio.get_event_loop().run_until_complete(populated_skill.execute("export_html", {"output_path": out}))
    assert result.success
    html = Path(out).read_text()
    assert "Singularity Agent Dashboard" in html
    assert "Self Improvement" in html or "self_improvement" in html.lower()


def test_section_report_performance(populated_skill):
    result = asyncio.get_event_loop().run_until_complete(populated_skill.execute("section_report", {"section": "performance"}))
    assert result.success
    assert result.data["performance"]["total_actions"] == 3


def test_section_report_goals(populated_skill):
    result = asyncio.get_event_loop().run_until_complete(populated_skill.execute("section_report", {"section": "goals"}))
    assert result.success
    g = result.data["goals"]
    assert g["total_goals"] == 3
    assert g["by_status"]["active"] == 2


def test_section_report_invalid(skill):
    result = asyncio.get_event_loop().run_until_complete(skill.execute("section_report", {"section": "bogus"}))
    assert not result.success


def test_unknown_action(skill):
    result = asyncio.get_event_loop().run_until_complete(skill.execute("nonexistent"))
    assert not result.success


def test_revenue_collection(populated_skill):
    result = asyncio.get_event_loop().run_until_complete(populated_skill.execute("section_report", {"section": "revenue"}))
    assert result.success
    rev = result.data["revenue"]
    assert rev["total_revenue"] == 1.0
    assert rev["total_customers"] == 2
    assert rev["active_customers"] == 1


def test_fleet_collection(populated_skill):
    result = asyncio.get_event_loop().run_until_complete(populated_skill.execute("section_report", {"section": "fleet"}))
    assert result.success
    fleet = result.data["fleet"]
    assert fleet["total_agents"] == 3
    assert fleet["healthy"] == 2
    assert fleet["fleet_health_pct"] == pytest.approx(66.7, abs=0.1)
