"""Tests for ReputationWeightedAssigner skill."""
import pytest
import json
from pathlib import Path
from unittest.mock import patch, AsyncMock, MagicMock
from singularity.skills.reputation_assigner import ReputationWeightedAssigner, ASSIGNER_FILE
from singularity.skills.base import SkillResult


@pytest.fixture
def assigner(tmp_path):
    test_file = tmp_path / "assignments.json"
    with patch("singularity.skills.reputation_assigner.ASSIGNER_FILE", test_file):
        a = ReputationWeightedAssigner()
        a._assignments = []
        yield a


def _mock_context(route_matches=None, reputation_data=None, tally_winner=None):
    """Build a mock context that responds to skill calls."""
    ctx = MagicMock()

    async def mock_call(skill_id, action, params=None):
        params = params or {}
        if skill_id == "agent_network" and action == "route":
            return SkillResult(
                success=True,
                data={"matches": route_matches or []},
            )
        if skill_id == "agent_reputation" and action == "get_reputation":
            agent_id = params.get("agent_id", "")
            rep = (reputation_data or {}).get(agent_id, {"overall": 50.0, "dimensions": {}})
            return SkillResult(success=True, data=rep)
        if skill_id == "agent_reputation" and action == "record_task_outcome":
            return SkillResult(success=True)
        if skill_id == "agent_reputation" and action == "record_event":
            return SkillResult(success=True)
        if skill_id == "consensus_protocol":
            if action == "tally":
                return SkillResult(success=True, data={"winner": tally_winner or ""})
            return SkillResult(success=True)
        if skill_id == "task_delegation":
            return SkillResult(success=True)
        return SkillResult(success=False, message="Unknown skill")

    ctx.call_skill = AsyncMock(side_effect=mock_call)
    return ctx


@pytest.mark.asyncio
async def test_manifest(assigner):
    m = assigner.manifest
    assert m.skill_id == "reputation_assigner"
    assert len(m.actions) == 7


@pytest.mark.asyncio
async def test_find_candidates_no_context(assigner):
    result = await assigner.execute("find_candidates", {"capability": "coding"})
    assert result.success
    assert result.data["candidates"] == []


@pytest.mark.asyncio
async def test_find_candidates_with_context(assigner):
    ctx = _mock_context(
        route_matches=[
            {"agent_id": "agent_a"},
            {"agent_id": "agent_b"},
        ],
        reputation_data={
            "agent_a": {"overall": 75.0, "dimensions": {"competence": 80}},
            "agent_b": {"overall": 25.0, "dimensions": {"competence": 30}},
        },
    )
    assigner.context = ctx
    result = await assigner.execute("find_candidates", {"capability": "coding", "min_reputation": 30})
    assert result.success
    # Only agent_a should pass min_reputation=30
    assert len(result.data["candidates"]) == 1
    assert result.data["candidates"][0]["agent_id"] == "agent_a"


@pytest.mark.asyncio
async def test_score_candidates(assigner):
    ctx = _mock_context(
        reputation_data={
            "a1": {"overall": 80, "dimensions": {"competence": 90, "reliability": 70, "trustworthiness": 60, "cooperation": 50, "leadership": 40}},
            "a2": {"overall": 60, "dimensions": {"competence": 50, "reliability": 80, "trustworthiness": 70, "cooperation": 60, "leadership": 50}},
        }
    )
    assigner.context = ctx
    result = await assigner.execute("score_candidates", {"agent_ids": ["a1", "a2"]})
    assert result.success
    scored = result.data["scored"]
    assert len(scored) == 2
    assert scored[0]["agent_id"] == "a1"  # Higher competence+reliability


@pytest.mark.asyncio
async def test_assign_direct(assigner):
    result = await assigner.execute("assign", {
        "task_name": "Fix bug",
        "task_description": "Fix the login bug",
        "agent_id": "agent_x",
        "budget": 5.0,
    })
    assert result.success
    assert result.data["agent_id"] == "agent_x"
    assert len(assigner._assignments) == 1
    assert assigner._assignments[0]["method"] == "direct"


@pytest.mark.asyncio
async def test_assign_requires_fields(assigner):
    r1 = await assigner.execute("assign", {"agent_id": "a", "budget": 1})
    assert not r1.success
    r2 = await assigner.execute("assign", {"task_name": "t", "budget": 1})
    assert not r2.success


@pytest.mark.asyncio
async def test_assign_auto(assigner):
    ctx = _mock_context(
        route_matches=[{"agent_id": "best_agent"}, {"agent_id": "ok_agent"}],
        reputation_data={
            "best_agent": {"overall": 85, "dimensions": {"competence": 90, "reliability": 80, "trustworthiness": 70, "cooperation": 60, "leadership": 50}},
            "ok_agent": {"overall": 55, "dimensions": {"competence": 50, "reliability": 50, "trustworthiness": 50, "cooperation": 50, "leadership": 50}},
        },
    )
    assigner.context = ctx
    result = await assigner.execute("assign_auto", {
        "task_name": "Deploy",
        "capability": "devops",
        "task_description": "Deploy to prod",
        "budget": 10.0,
    })
    assert result.success
    assert result.data["agent_id"] == "best_agent"
    assert result.data["method"] == "reputation"


@pytest.mark.asyncio
async def test_assign_auto_with_consensus(assigner):
    ctx = _mock_context(
        route_matches=[{"agent_id": "a1"}, {"agent_id": "a2"}],
        reputation_data={
            "a1": {"overall": 80, "dimensions": {"competence": 80, "reliability": 80, "trustworthiness": 80, "cooperation": 80, "leadership": 80}},
            "a2": {"overall": 70, "dimensions": {"competence": 70, "reliability": 70, "trustworthiness": 70, "cooperation": 70, "leadership": 70}},
        },
        tally_winner="a2",
    )
    assigner.context = ctx
    result = await assigner.execute("assign_auto", {
        "task_name": "Important",
        "capability": "coding",
        "task_description": "Critical fix",
        "budget": 20.0,
        "use_consensus": True,
    })
    assert result.success
    assert result.data["agent_id"] == "a2"
    assert result.data["method"] == "consensus"


@pytest.mark.asyncio
async def test_complete_updates_reputation(assigner):
    # Create an assignment
    await assigner.execute("assign", {
        "task_name": "Test", "task_description": "Desc",
        "agent_id": "agent_q", "budget": 5.0,
    })
    aid = assigner._assignments[0]["id"]

    ctx = _mock_context()
    assigner.context = ctx
    result = await assigner.execute("complete", {
        "assignment_id": aid, "success": True, "quality_score": 90, "budget_used": 2.0,
    })
    assert result.success
    assert result.data["reputation_updated"]
    assert assigner._assignments[0]["status"] == "completed"

    # Verify reputation was updated
    calls = [c for c in ctx.call_skill.call_args_list if c[0][0] == "agent_reputation"]
    assert len(calls) >= 1


@pytest.mark.asyncio
async def test_complete_not_found(assigner):
    result = await assigner.execute("complete", {"assignment_id": "nope", "success": True})
    assert not result.success


@pytest.mark.asyncio
async def test_history(assigner):
    for i in range(3):
        await assigner.execute("assign", {
            "task_name": f"T{i}", "task_description": "D",
            "agent_id": f"agent_{i}", "budget": 1.0,
        })
    result = await assigner.execute("history", {"last_n": 2})
    assert result.success
    assert len(result.data["assignments"]) == 2


@pytest.mark.asyncio
async def test_history_filter_by_agent(assigner):
    await assigner.execute("assign", {"task_name": "A", "task_description": "D", "agent_id": "x", "budget": 1})
    await assigner.execute("assign", {"task_name": "B", "task_description": "D", "agent_id": "y", "budget": 1})
    result = await assigner.execute("history", {"agent_id": "x"})
    assert len(result.data["assignments"]) == 1


@pytest.mark.asyncio
async def test_leaderboard(assigner):
    # Two agents, one succeeds, one fails
    await assigner.execute("assign", {"task_name": "A", "task_description": "D", "agent_id": "winner", "budget": 5})
    assigner._assignments[0]["status"] = "completed"
    assigner._assignments[0]["success"] = True
    assigner._assignments[0]["quality_score"] = 85

    await assigner.execute("assign", {"task_name": "B", "task_description": "D", "agent_id": "loser", "budget": 5})
    assigner._assignments[1]["status"] = "completed"
    assigner._assignments[1]["success"] = False
    assigner._assignments[1]["quality_score"] = 20

    result = await assigner.execute("leaderboard", {})
    assert result.success
    lb = result.data["leaderboard"]
    assert len(lb) == 2
    assert lb[0]["agent_id"] == "winner"
    assert lb[0]["success_rate"] == 100.0


@pytest.mark.asyncio
async def test_unknown_action(assigner):
    result = await assigner.execute("nonexistent", {})
    assert not result.success
