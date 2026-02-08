"""Tests for AgentSpawnerSkill - autonomous replication decision-making."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from singularity.skills.agent_spawner import AgentSpawnerSkill, DEFAULT_POLICIES
from singularity.skills.base import SkillResult, SkillContext, SkillRegistry


@pytest.fixture
def skill(tmp_path):
    s = AgentSpawnerSkill()
    s._data_dir = tmp_path
    s._state_file = tmp_path / "agent_spawner.json"
    s._state = {"policies": dict(DEFAULT_POLICIES), "replicas": {}, "config": {}, "history": []}
    return s


@pytest.fixture
def skill_with_context(tmp_path):
    s = AgentSpawnerSkill()
    s._data_dir = tmp_path
    s._state_file = tmp_path / "agent_spawner.json"
    s._state = {"policies": dict(DEFAULT_POLICIES), "replicas": {}, "config": {}, "history": []}

    registry = SkillRegistry()
    ctx = SkillContext(registry=registry, agent_name="TestAgent")

    async def mock_call(skill_id, action, params):
        if skill_id == "replication" and action == "spawn":
            return SkillResult(success=True, message="Spawned", data={"replica_id": "rep_test123"})
        if skill_id == "replication" and action == "terminate":
            return SkillResult(success=True, message="Terminated")
        if skill_id == "task_queue" and action == "stats":
            return SkillResult(success=True, message="Stats", data={"pending": 15})
        if skill_id == "self_assessment" and action == "gaps":
            return SkillResult(success=True, message="Gaps", data={
                "gaps": [{"skill": "dns_automation", "impact_score": 0.9}]
            })
        return SkillResult(success=True, message="ok", data={})

    ctx.call_skill = mock_call
    ctx.list_skills = MagicMock(return_value=["replication", "task_queue", "self_assessment"])
    s.set_context(ctx)
    return s


@pytest.mark.asyncio
async def test_fleet_empty(skill):
    result = await skill.execute("fleet", {})
    assert result.success
    assert result.data["active_count"] == 0


@pytest.mark.asyncio
async def test_spawn_manual(skill):
    result = await skill.execute("spawn", {
        "name": "test-replica",
        "type": "generalist",
        "budget": 1.0,
        "reason": "testing",
    })
    assert result.success
    assert "test-replica" in result.message
    # Check fleet
    fleet = await skill.execute("fleet", {})
    assert fleet.data["active_count"] == 1


@pytest.mark.asyncio
async def test_spawn_with_context(skill_with_context):
    result = await skill_with_context.execute("spawn", {
        "name": "ctx-replica",
        "type": "specialist",
        "skills": ["dns_automation"],
        "budget": 2.0,
    })
    assert result.success
    assert "Spawned" in result.message


@pytest.mark.asyncio
async def test_spawn_exceeds_cap(skill):
    skill._state["config"]["max_replicas"] = 1
    await skill.execute("spawn", {"name": "r1", "budget": 0.5})
    result = await skill.execute("spawn", {"name": "r2", "budget": 0.5})
    assert not result.success
    assert "cap" in result.message.lower()


@pytest.mark.asyncio
async def test_spawn_exceeds_daily_budget(skill):
    skill._state["config"]["daily_budget"] = 1.0
    await skill.execute("spawn", {"name": "r1", "budget": 0.8})
    result = await skill.execute("spawn", {"name": "r2", "budget": 0.5})
    assert not result.success
    assert "budget" in result.message.lower()


@pytest.mark.asyncio
async def test_retire_replica(skill):
    await skill.execute("spawn", {"name": "to-retire", "budget": 1.0})
    fleet = await skill.execute("fleet", {})
    rid = fleet.data["fleet"][0]["id"]
    result = await skill.execute("retire", {"replica_id": rid, "reason": "no longer needed"})
    assert result.success
    fleet2 = await skill.execute("fleet", {})
    assert fleet2.data["active_count"] == 0


@pytest.mark.asyncio
async def test_retire_not_found(skill):
    result = await skill.execute("retire", {"replica_id": "nonexistent"})
    assert not result.success


@pytest.mark.asyncio
async def test_policies_list(skill):
    result = await skill.execute("policies", {})
    assert result.success
    assert len(result.data["policies"]) == 4


@pytest.mark.asyncio
async def test_policies_update(skill):
    result = await skill.execute("policies", {"policy_id": "workload", "enabled": False})
    assert result.success
    assert skill._state["policies"]["workload"]["enabled"] is False


@pytest.mark.asyncio
async def test_configure(skill):
    result = await skill.execute("configure", {"max_replicas": 5, "daily_budget": 10.0})
    assert result.success
    assert skill._state["config"]["max_replicas"] == 5
    assert skill._state["config"]["daily_budget"] == 10.0


@pytest.mark.asyncio
async def test_evaluate_dry_run(skill_with_context):
    result = await skill_with_context.execute("evaluate", {"dry_run": True})
    assert result.success
    recs = result.data["recommendations"]
    # Should have at least resilience trigger (only 1 agent running < min 2)
    policy_names = [r["policy"] for r in recs]
    assert len(recs) >= 2
    assert "resilience" in policy_names


@pytest.mark.asyncio
async def test_evaluate_resilience_trigger(skill):
    """Resilience trigger fires when fewer than min_agents are running."""
    result = await skill.execute("evaluate", {"dry_run": True})
    assert result.success
    policy_names = [r["policy"] for r in result.data["recommendations"]]
    assert "resilience" in policy_names


@pytest.mark.asyncio
async def test_history(skill):
    await skill.execute("spawn", {"name": "h1", "budget": 0.5})
    await skill.execute("spawn", {"name": "h2", "budget": 0.5})
    result = await skill.execute("history", {"limit": 10})
    assert result.success
    assert len(result.data["history"]) == 2


@pytest.mark.asyncio
async def test_fleet_include_retired(skill):
    await skill.execute("spawn", {"name": "r1", "budget": 1.0})
    fleet = await skill.execute("fleet", {})
    rid = fleet.data["fleet"][0]["id"]
    await skill.execute("retire", {"replica_id": rid})
    # Without include_retired
    fleet2 = await skill.execute("fleet", {})
    assert fleet2.data["active_count"] == 0
    assert fleet2.data["total_count"] == 0
    # With include_retired
    fleet3 = await skill.execute("fleet", {"include_retired": True})
    assert fleet3.data["total_count"] == 1
