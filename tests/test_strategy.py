#!/usr/bin/env python3
"""Tests for StrategySkill - autonomous strategic reasoning."""

import pytest
import json
from pathlib import Path
from unittest.mock import patch
from singularity.skills.strategy import StrategySkill, STRATEGY_FILE, Pillar


@pytest.fixture
def skill(tmp_path):
    """Create a StrategySkill with a temporary data path."""
    test_file = tmp_path / "strategy.json"
    with patch("singularity.skills.strategy.STRATEGY_FILE", test_file):
        s = StrategySkill()
        yield s


@pytest.mark.asyncio
async def test_assess_pillar(skill, tmp_path):
    test_file = tmp_path / "strategy.json"
    with patch("singularity.skills.strategy.STRATEGY_FILE", test_file):
        result = await skill.execute("assess", {
            "pillar": "self_improvement",
            "score": 45,
            "capabilities": ["ExperimentSkill", "SelfModifySkill", "MemorySkill"],
            "gaps": ["Performance evaluation", "Strategy selection"],
        })
        assert result.success
        assert result.data["new_score"] == 45
        assert result.data["capabilities_count"] == 3
        assert result.data["gaps_count"] == 2


@pytest.mark.asyncio
async def test_assess_invalid_pillar(skill, tmp_path):
    test_file = tmp_path / "strategy.json"
    with patch("singularity.skills.strategy.STRATEGY_FILE", test_file):
        result = await skill.execute("assess", {
            "pillar": "invalid_pillar",
            "score": 50,
            "capabilities": [],
            "gaps": [],
        })
        assert not result.success
        assert "Invalid pillar" in result.message


@pytest.mark.asyncio
async def test_assess_clamps_score(skill, tmp_path):
    test_file = tmp_path / "strategy.json"
    with patch("singularity.skills.strategy.STRATEGY_FILE", test_file):
        result = await skill.execute("assess", {
            "pillar": "revenue",
            "score": 150,
            "capabilities": [],
            "gaps": [],
        })
        assert result.success
        assert result.data["new_score"] == 100


@pytest.mark.asyncio
async def test_diagnose_no_assessments(skill, tmp_path):
    test_file = tmp_path / "strategy.json"
    with patch("singularity.skills.strategy.STRATEGY_FILE", test_file):
        result = await skill.execute("diagnose", {})
        assert result.success
        assert result.data["assessed"] is False


@pytest.mark.asyncio
async def test_diagnose_finds_weakest(skill, tmp_path):
    test_file = tmp_path / "strategy.json"
    with patch("singularity.skills.strategy.STRATEGY_FILE", test_file):
        # Assess multiple pillars
        await skill.execute("assess", {"pillar": "self_improvement", "score": 60, "capabilities": ["a"], "gaps": []})
        await skill.execute("assess", {"pillar": "revenue", "score": 10, "capabilities": [], "gaps": ["API", "payments"]})
        await skill.execute("assess", {"pillar": "replication", "score": 30, "capabilities": [], "gaps": ["spawning"]})
        await skill.execute("assess", {"pillar": "goal_setting", "score": 50, "capabilities": ["planner"], "gaps": []})

        result = await skill.execute("diagnose", {})
        assert result.success
        assert result.data["weakest_pillar"] == "revenue"
        assert result.data["weakest_score"] == 10
        assert result.data["strongest_pillar"] == "self_improvement"


@pytest.mark.asyncio
async def test_recommend_prioritizes_by_score(skill, tmp_path):
    test_file = tmp_path / "strategy.json"
    with patch("singularity.skills.strategy.STRATEGY_FILE", test_file):
        await skill.execute("assess", {"pillar": "revenue", "score": 10, "capabilities": [], "gaps": ["API endpoints", "payment processing"]})
        await skill.execute("assess", {"pillar": "self_improvement", "score": 60, "capabilities": [], "gaps": ["feedback loops"]})

        result = await skill.execute("recommend", {"count": 3})
        assert result.success
        recs = result.data["recommendations"]
        assert len(recs) >= 2
        # Revenue gaps should be higher priority
        assert recs[0]["pillar"] == "revenue"
        assert recs[0]["priority"] == "critical"


@pytest.mark.asyncio
async def test_log_work(skill, tmp_path):
    test_file = tmp_path / "strategy.json"
    with patch("singularity.skills.strategy.STRATEGY_FILE", test_file):
        result = await skill.execute("log_work", {
            "pillar": "self_improvement",
            "description": "Built StrategySkill for meta-cognitive reasoning",
            "impact": "high",
            "score_delta": 10,
        })
        assert result.success
        assert result.data["impact"] == "high"
        assert result.data["score_delta"] == 10


@pytest.mark.asyncio
async def test_log_work_adjusts_score(skill, tmp_path):
    test_file = tmp_path / "strategy.json"
    with patch("singularity.skills.strategy.STRATEGY_FILE", test_file):
        await skill.execute("assess", {"pillar": "revenue", "score": 20, "capabilities": [], "gaps": []})
        await skill.execute("log_work", {"pillar": "revenue", "description": "Added ServiceAPI", "impact": "high", "score_delta": 15})

        result = await skill.execute("status", {})
        assert result.data["pillars"]["revenue"]["score"] == 35


@pytest.mark.asyncio
async def test_journal(skill, tmp_path):
    test_file = tmp_path / "strategy.json"
    with patch("singularity.skills.strategy.STRATEGY_FILE", test_file):
        result = await skill.execute("journal", {
            "entry": "Revenue pillar is weakest. Focus next 3 sessions on API and payments.",
            "category": "decision",
        })
        assert result.success
        assert result.data["category"] == "decision"


@pytest.mark.asyncio
async def test_review(skill, tmp_path):
    test_file = tmp_path / "strategy.json"
    with patch("singularity.skills.strategy.STRATEGY_FILE", test_file):
        await skill.execute("journal", {"entry": "Insight 1", "category": "insight"})
        await skill.execute("log_work", {"pillar": "replication", "description": "Built orchestrator", "impact": "high"})

        result = await skill.execute("review", {"limit": 5})
        assert result.success
        assert len(result.data["journal"]) == 1
        assert len(result.data["work_log"]) == 1


@pytest.mark.asyncio
async def test_status_overview(skill, tmp_path):
    test_file = tmp_path / "strategy.json"
    with patch("singularity.skills.strategy.STRATEGY_FILE", test_file):
        await skill.execute("assess", {"pillar": "self_improvement", "score": 50, "capabilities": ["a", "b"], "gaps": ["c"]})
        await skill.execute("assess", {"pillar": "revenue", "score": 20, "capabilities": [], "gaps": ["x", "y"]})

        result = await skill.execute("status", {})
        assert result.success
        assert result.data["pillars"]["self_improvement"]["score"] == 50
        assert result.data["pillars"]["revenue"]["score"] == 20
        assert result.data["average_score"] > 0


@pytest.mark.asyncio
async def test_start_session(skill, tmp_path):
    test_file = tmp_path / "strategy.json"
    with patch("singularity.skills.strategy.STRATEGY_FILE", test_file):
        result = await skill.execute("start_session", {"focus": "Build revenue generation"})
        assert result.success
        assert result.data["session_number"] == 1
        assert result.data["focus"] == "Build revenue generation"

        # Second session increments
        result2 = await skill.execute("start_session", {})
        assert result2.data["session_number"] == 2


@pytest.mark.asyncio
async def test_unknown_action(skill, tmp_path):
    test_file = tmp_path / "strategy.json"
    with patch("singularity.skills.strategy.STRATEGY_FILE", test_file):
        result = await skill.execute("nonexistent", {})
        assert not result.success


@pytest.mark.asyncio
async def test_manifest(skill):
    m = skill.manifest
    assert m.skill_id == "strategy"
    assert m.category == "meta-cognition"
    assert len(m.actions) == 8
    action_names = [a.name for a in m.actions]
    assert "assess" in action_names
    assert "diagnose" in action_names
    assert "recommend" in action_names
    assert "status" in action_names


@pytest.mark.asyncio
async def test_full_workflow(skill, tmp_path):
    """End-to-end: assess all pillars, diagnose, get recommendations, log work."""
    test_file = tmp_path / "strategy.json"
    with patch("singularity.skills.strategy.STRATEGY_FILE", test_file):
        # Start session
        await skill.execute("start_session", {"focus": "Bootstrap strategy"})

        # Assess all pillars
        await skill.execute("assess", {"pillar": "self_improvement", "score": 55, "capabilities": ["experiment", "self_modify", "memory"], "gaps": ["performance eval", "strategy engine"]})
        await skill.execute("assess", {"pillar": "revenue", "score": 15, "capabilities": ["service_api"], "gaps": ["payment processing", "customer discovery", "pricing"]})
        await skill.execute("assess", {"pillar": "replication", "score": 25, "capabilities": ["orchestrator"], "gaps": ["resource management", "coordination"]})
        await skill.execute("assess", {"pillar": "goal_setting", "score": 40, "capabilities": ["planner"], "gaps": ["milestone tracking", "long-term strategy"]})

        # Diagnose
        diag = await skill.execute("diagnose", {})
        assert diag.data["weakest_pillar"] == "revenue"

        # Get recommendations
        recs = await skill.execute("recommend", {"count": 5})
        assert recs.data["count"] >= 3

        # Log work
        await skill.execute("log_work", {"pillar": "goal_setting", "description": "Built strategy engine", "impact": "high", "score_delta": 15})

        # Journal insight
        await skill.execute("journal", {"entry": "Strategy engine enables meta-reasoning. Next: integrate with planner for actionable goals.", "category": "insight"})

        # Final status
        status = await skill.execute("status", {})
        assert status.data["pillars"]["goal_setting"]["score"] == 55  # 40 + 15 delta
        assert status.data["session_count"] == 1
