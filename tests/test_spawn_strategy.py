"""Tests for SpawnStrategySkill."""
import pytest
import asyncio
import json
import tempfile
from unittest.mock import MagicMock, patch
from singularity.skills.spawn_strategy import SpawnStrategySkill, SPAWN_DATA_DIR


@pytest.fixture
def skill():
    s = SpawnStrategySkill({})
    # Use a mock parent agent
    mock_agent = MagicMock()
    mock_agent.balance = 100.0
    s.set_parent_agent(mock_agent)
    return s


def test_manifest(skill):
    m = skill.manifest
    assert m.skill_id == "spawn_strategy"
    assert len(m.actions) == 8
    action_names = {a.name for a in m.actions}
    assert "evaluate_readiness" in action_names
    assert "recommend" in action_names
    assert "record_spawn" in action_names
    assert "offspring_report" in action_names
    assert "calculate_budget" in action_names


def test_check_credentials(skill):
    assert skill.check_credentials() is True


@pytest.mark.asyncio
async def test_evaluate_readiness_high_balance(skill):
    result = await skill.execute("evaluate_readiness", {"current_balance": 100.0, "burn_rate": 0.01})
    assert result.success
    assert result.data["financial_ready"] is True
    assert result.data["readiness_score"] >= 50


@pytest.mark.asyncio
async def test_evaluate_readiness_low_balance(skill):
    result = await skill.execute("evaluate_readiness", {"current_balance": 2.0, "burn_rate": 0.5})
    assert result.success
    assert result.data["readiness_score"] < 50


@pytest.mark.asyncio
async def test_evaluate_readiness_auto_detect_balance(skill):
    result = await skill.execute("evaluate_readiness", {})
    assert result.success
    assert result.data["balance"] == 100.0


@pytest.mark.asyncio
async def test_list_strategies(skill):
    result = await skill.execute("list_strategies", {})
    assert result.success
    assert len(result.data["strategies"]) >= 5


@pytest.mark.asyncio
async def test_recommend_revenue_goal(skill):
    result = await skill.execute("recommend", {"goal": "earn money and generate revenue"})
    assert result.success
    recs = result.data["recommendations"]
    assert len(recs) > 0
    # Revenue hunter should be ranked high
    assert any(r["strategy_id"] == "revenue_hunter" for r in recs)


@pytest.mark.asyncio
async def test_recommend_code_goal(skill):
    result = await skill.execute("recommend", {"goal": "build software and code"})
    assert result.success
    recs = result.data["recommendations"]
    assert any(r["strategy_id"] == "code_specialist" for r in recs)


@pytest.mark.asyncio
async def test_record_spawn(skill):
    result = await skill.execute("record_spawn", {
        "agent_name": "TestAgent",
        "purpose": "Test spawning",
        "budget_allocated": 10.0,
    })
    assert result.success
    assert result.data["agent_name"] == "TestAgent"
    assert result.data["budget_allocated"] == 10.0


@pytest.mark.asyncio
async def test_update_offspring(skill):
    # First record a spawn
    await skill.execute("record_spawn", {
        "agent_name": "TrackedAgent",
        "purpose": "Track performance",
        "budget_allocated": 15.0,
    })
    # Then update it
    result = await skill.execute("update_offspring", {
        "agent_name": "TrackedAgent",
        "status": "alive",
        "revenue_generated": 25.0,
        "note": "Doing well",
    })
    assert result.success
    assert result.data["revenue_generated"] == 25.0
    assert result.data["roi"] > 0  # 25/15 - 1 = 0.67


@pytest.mark.asyncio
async def test_offspring_report_empty(skill):
    result = await skill.execute("offspring_report", {})
    assert result.success
    # When empty, it returns total_spawns at top level
    assert "total_spawns" in result.data or "summary" in result.data


@pytest.mark.asyncio
async def test_offspring_report_with_data():
    s = SpawnStrategySkill({})
    mock_agent = MagicMock()
    mock_agent.balance = 100.0
    s.set_parent_agent(mock_agent)
    await s.execute("record_spawn", {
        "agent_name": "Agent1", "purpose": "Test", "budget_allocated": 10.0
    })
    await s.execute("record_spawn", {
        "agent_name": "Agent2", "purpose": "Test2", "budget_allocated": 5.0
    })
    result = await s.execute("offspring_report", {})
    assert result.success
    assert "agents" in result.data
    agents = result.data["agents"]
    assert len(agents) == 2


@pytest.mark.asyncio
async def test_calculate_budget_moderate(skill):
    result = await skill.execute("calculate_budget", {
        "current_balance": 100.0, "risk_tolerance": "moderate"
    })
    assert result.success
    assert result.data["can_spawn"] is True
    budget = result.data["recommended_budget"]
    assert 1.0 <= budget <= 30.0


@pytest.mark.asyncio
async def test_calculate_budget_conservative(skill):
    r1 = await skill.execute("calculate_budget", {
        "current_balance": 100.0, "risk_tolerance": "conservative"
    })
    r2 = await skill.execute("calculate_budget", {
        "current_balance": 100.0, "risk_tolerance": "aggressive"
    })
    assert r1.data["recommended_budget"] < r2.data["recommended_budget"]


@pytest.mark.asyncio
async def test_calculate_budget_too_low(skill):
    result = await skill.execute("calculate_budget", {"current_balance": 3.0})
    assert result.success
    # With $3, can't safely spawn (need $5 reserve + $1 min)
    assert result.data["can_spawn"] is False


@pytest.mark.asyncio
async def test_create_strategy(skill):
    result = await skill.execute("create_strategy", {
        "name": "Content Creator",
        "description": "Spawn agents that create content",
        "purpose_template": "Create valuable content to attract users",
        "recommended_budget_pct": 0.12,
        "agent_type": "creator",
    })
    assert result.success
    assert "strategy_id" in result.data
    # Verify it shows up in list
    list_result = await skill.execute("list_strategies", {})
    ids = [s["strategy_id"] for s in list_result.data["strategies"]]
    assert result.data["strategy_id"] in ids
