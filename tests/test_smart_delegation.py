"""Tests for SmartDelegationSkill."""

import pytest
import json
from unittest.mock import AsyncMock, MagicMock, patch
from singularity.skills.smart_delegation import SmartDelegationSkill, SMART_DELEGATION_FILE
from singularity.skills.base import SkillResult


@pytest.fixture
def skill(tmp_path):
    """Create skill with temp data path."""
    s = SmartDelegationSkill()
    test_file = tmp_path / "smart_delegations.json"
    import singularity.skills.smart_delegation as mod
    mod.SMART_DELEGATION_FILE = test_file
    s._ensure_data()
    return s


@pytest.fixture
def mock_context():
    """Create a mock SkillContext."""
    ctx = MagicMock()
    ctx.call_skill = AsyncMock()
    return ctx


def rep_result(agent_id, overall=50.0, comp=50.0, rel=50.0, trust=50.0, lead=50.0, coop=50.0, tasks=0, completed=0, failed=0):
    """Helper to build a reputation result."""
    return SkillResult(success=True, message="ok", data={
        "agent_id": agent_id, "competence": comp, "reliability": rel,
        "trustworthiness": trust, "leadership": lead, "cooperation": coop,
        "overall": overall, "total_tasks": tasks, "tasks_completed": completed,
        "tasks_failed": failed, "votes_cast": 0, "endorsements_received": 0,
        "penalties_received": 0, "first_seen": "", "last_updated": "",
    })


class TestSmartDelegate:
    @pytest.mark.asyncio
    async def test_requires_task_name(self, skill):
        r = await skill.execute("smart_delegate", {"task_description": "x", "budget": 10})
        assert not r.success
        assert "task_name" in r.message

    @pytest.mark.asyncio
    async def test_requires_candidates(self, skill):
        r = await skill.execute("smart_delegate", {
            "task_name": "test", "task_description": "desc", "budget": 10,
        })
        assert not r.success
        assert "candidate" in r.message.lower()

    @pytest.mark.asyncio
    async def test_smart_delegate_picks_best_reputation(self, skill, mock_context):
        skill.context = mock_context
        # Agent A has higher reputation than Agent B
        async def mock_call(skill_id, action, params=None):
            params = params or {}
            if skill_id == "agent_reputation" and action == "get_reputation":
                aid = params.get("agent_id")
                if aid == "agent_a":
                    return rep_result("agent_a", overall=80, comp=85, rel=75)
                return rep_result("agent_b", overall=40, comp=35, rel=45)
            if skill_id == "task_delegation" and action == "delegate":
                return SkillResult(success=True, message="delegated", data={"delegation_id": "dlg_123"})
            return SkillResult(success=False, message="unknown")

        mock_context.call_skill = AsyncMock(side_effect=mock_call)

        r = await skill.execute("smart_delegate", {
            "task_name": "important_task", "task_description": "do something",
            "budget": 10, "candidate_agents": ["agent_a", "agent_b"],
        })
        assert r.success
        assert r.data["selected_agent"] == "agent_a"
        assert r.data["delegation_id"] == "dlg_123"

    @pytest.mark.asyncio
    async def test_min_reputation_filters(self, skill, mock_context):
        skill.context = mock_context
        async def mock_call(skill_id, action, params=None):
            params = params or {}
            if skill_id == "agent_reputation":
                return rep_result(params.get("agent_id", ""), overall=30,
                                  comp=30, rel=30, trust=30, lead=30, coop=30)
            return SkillResult(success=False, message="x")

        mock_context.call_skill = AsyncMock(side_effect=mock_call)
        r = await skill.execute("smart_delegate", {
            "task_name": "t", "task_description": "d", "budget": 5,
            "candidate_agents": ["low_rep"], "min_reputation": 50,
        })
        assert not r.success
        assert "threshold" in r.message.lower()


class TestReputationRoute:
    @pytest.mark.asyncio
    async def test_route_ranks_agents(self, skill, mock_context):
        skill.context = mock_context
        async def mock_call(skill_id, action, params=None):
            params = params or {}
            aid = params.get("agent_id", "")
            scores = {"fast": 90, "medium": 60, "slow": 30}
            s = scores.get(aid, 50)
            return rep_result(aid, overall=s, comp=s, rel=s, trust=s, lead=s, coop=s)

        mock_context.call_skill = AsyncMock(side_effect=mock_call)
        r = await skill.execute("reputation_route", {
            "candidate_agents": ["slow", "fast", "medium"],
        })
        assert r.success
        agents = r.data["ranked_agents"]
        assert agents[0]["agent_id"] == "fast"
        assert agents[1]["agent_id"] == "medium"
        assert agents[2]["agent_id"] == "slow"


class TestConsensusAssign:
    @pytest.mark.asyncio
    async def test_needs_at_least_2_candidates(self, skill):
        r = await skill.execute("consensus_assign", {
            "task_name": "t", "task_description": "d", "budget": 5,
            "candidate_agents": ["only_one"],
        })
        assert not r.success
        assert "2 candidate" in r.message

    @pytest.mark.asyncio
    async def test_consensus_elect_winner(self, skill, mock_context):
        skill.context = mock_context
        async def mock_call(skill_id, action, params=None):
            params = params or {}
            if skill_id == "agent_reputation" and action == "get_reputation":
                aid = params.get("agent_id", "")
                if aid == "expert":
                    return rep_result("expert", overall=90)
                return rep_result(aid, overall=50)
            if skill_id == "consensus_protocol" and action == "elect":
                return SkillResult(success=True, message="elected", data={
                    "election_id": "elect_123", "winner": "expert",
                    "result": {"winner": "expert"},
                })
            if skill_id == "agent_reputation" and action == "record_vote":
                return SkillResult(success=True, message="ok")
            if skill_id == "task_delegation" and action == "delegate":
                return SkillResult(success=True, message="ok", data={"delegation_id": "dlg_456"})
            return SkillResult(success=False, message="unknown")

        mock_context.call_skill = AsyncMock(side_effect=mock_call)
        r = await skill.execute("consensus_assign", {
            "task_name": "critical_task", "task_description": "important",
            "budget": 50, "candidate_agents": ["expert", "novice"],
        })
        assert r.success
        assert r.data["winner"] == "expert"
        assert r.data["delegation_id"] == "dlg_456"


class TestAutoReport:
    @pytest.mark.asyncio
    async def test_auto_report_success(self, skill, mock_context):
        skill.context = mock_context
        async def mock_call(skill_id, action, params=None):
            if skill_id == "task_delegation" and action == "check":
                return SkillResult(success=True, message="ok", data={
                    "delegation_id": "dlg_1", "agent_id": "worker",
                    "task_name": "job", "budget": 10, "status": "in_progress",
                })
            if skill_id == "agent_reputation" and action == "record_task_outcome":
                return SkillResult(success=True, message="reputation updated", data={
                    "agent_id": "worker", "competence": 55, "overall": 53,
                })
            if skill_id == "task_delegation" and action == "report_completion":
                return SkillResult(success=True, message="ok")
            return SkillResult(success=False, message="x")

        mock_context.call_skill = AsyncMock(side_effect=mock_call)
        r = await skill.execute("auto_report", {
            "delegation_id": "dlg_1", "success": True, "budget_spent": 5,
        })
        assert r.success
        assert r.data["agent_id"] == "worker"
        assert r.data["reputation_updated"]
        assert r.data["budget_efficiency"] == 0.5


class TestRecommend:
    @pytest.mark.asyncio
    async def test_recommend_ranks_agents(self, skill, mock_context):
        skill.context = mock_context
        async def mock_call(skill_id, action, params=None):
            params = params or {}
            aid = params.get("agent_id", "")
            scores = {"pro": 85, "mid": 55, "noob": 30}
            return rep_result(aid, overall=scores.get(aid, 50), tasks=10, completed=8, failed=2)

        mock_context.call_skill = AsyncMock(side_effect=mock_call)
        r = await skill.execute("recommend", {
            "task_description": "complex data analysis",
            "candidate_agents": ["noob", "pro", "mid"], "top_n": 2,
        })
        assert r.success
        recs = r.data["recommendations"]
        assert len(recs) == 2
        assert recs[0]["agent_id"] == "pro"
        assert recs[0]["rank"] == 1
        assert recs[0]["success_rate"] == 80.0


class TestUnknownAction:
    @pytest.mark.asyncio
    async def test_unknown_action(self, skill):
        r = await skill.execute("nonexistent", {})
        assert not r.success
        assert "Unknown action" in r.message
