"""Tests for AgentReputationSkill."""

import pytest
from singularity.skills.agent_reputation import AgentReputationSkill, AgentReputation, _clamp


@pytest.fixture
def skill():
    s = AgentReputationSkill()
    s._persist_path = "/dev/null"
    return s


class TestBasicReputation:
    @pytest.mark.asyncio
    async def test_get_new_agent(self, skill):
        result = await skill.execute("get_reputation", {"agent_id": "agent1"})
        assert result.success
        assert result.data["overall"] == 50.0
        assert result.data["competence"] == 50.0

    @pytest.mark.asyncio
    async def test_record_event(self, skill):
        result = await skill.execute("record_event", {
            "agent_id": "a1", "event_type": "task_completed",
            "dimension": "competence", "delta": 5.0, "source": "test",
        })
        assert result.success
        assert result.data["new_value"] == 55.0

    @pytest.mark.asyncio
    async def test_clamp_values(self, skill):
        await skill.execute("record_event", {
            "agent_id": "a1", "event_type": "boost",
            "dimension": "competence", "delta": 200.0,
        })
        rep = await skill.execute("get_reputation", {"agent_id": "a1"})
        assert rep.data["competence"] == 100.0

    @pytest.mark.asyncio
    async def test_negative_clamp(self, skill):
        await skill.execute("record_event", {
            "agent_id": "a1", "event_type": "penalty",
            "dimension": "reliability", "delta": -200.0,
        })
        rep = await skill.execute("get_reputation", {"agent_id": "a1"})
        assert rep.data["reliability"] == 0.0


class TestTaskOutcome:
    @pytest.mark.asyncio
    async def test_task_success(self, skill):
        result = await skill.execute("record_task_outcome", {
            "agent_id": "worker1", "success": True,
            "budget_efficiency": 0.5, "on_time": True,
        })
        assert result.success
        assert result.data["competence"] > 50.0
        assert result.data["reliability"] > 50.0
        assert result.data["tasks_completed"] == 1

    @pytest.mark.asyncio
    async def test_task_failure(self, skill):
        result = await skill.execute("record_task_outcome", {
            "agent_id": "worker1", "success": False,
        })
        assert result.success
        assert result.data["competence"] < 50.0
        assert result.data["reliability"] < 50.0
        assert result.data["tasks_failed"] == 1

    @pytest.mark.asyncio
    async def test_budget_efficiency_bonus(self, skill):
        # Very efficient: budget_efficiency = 0.1 (used only 10% of budget)
        await skill.execute("record_task_outcome", {
            "agent_id": "efficient", "success": True, "budget_efficiency": 0.1,
        })
        # Wasteful: budget_efficiency = 1.0 (used 100% of budget)
        await skill.execute("record_task_outcome", {
            "agent_id": "wasteful", "success": True, "budget_efficiency": 1.0,
        })
        eff = await skill.execute("get_reputation", {"agent_id": "efficient"})
        wst = await skill.execute("get_reputation", {"agent_id": "wasteful"})
        assert eff.data["competence"] > wst.data["competence"]


class TestVoting:
    @pytest.mark.asyncio
    async def test_vote_correct(self, skill):
        result = await skill.execute("record_vote", {
            "agent_id": "voter1", "outcome_correct": True,
        })
        assert result.success
        assert result.data["trustworthiness"] > 50.0
        assert result.data["cooperation"] > 50.0

    @pytest.mark.asyncio
    async def test_vote_incorrect(self, skill):
        result = await skill.execute("record_vote", {
            "agent_id": "voter1", "outcome_correct": False,
        })
        assert result.data["trustworthiness"] < 50.0


class TestEndorseAndPenalize:
    @pytest.mark.asyncio
    async def test_endorse(self, skill):
        result = await skill.execute("endorse", {
            "from_agent": "a1", "to_agent": "a2",
            "dimension": "competence", "reason": "great work",
        })
        assert result.success
        assert result.data["delta"] > 0
        # Endorser gets cooperation boost
        endorser = await skill.execute("get_reputation", {"agent_id": "a1"})
        assert endorser.data["cooperation"] > 50.0

    @pytest.mark.asyncio
    async def test_self_endorse_rejected(self, skill):
        result = await skill.execute("endorse", {
            "from_agent": "a1", "to_agent": "a1",
        })
        assert not result.success

    @pytest.mark.asyncio
    async def test_penalize(self, skill):
        result = await skill.execute("penalize", {
            "agent_id": "bad_agent", "dimension": "trustworthiness",
            "amount": 10.0, "reason": "cheating",
        })
        assert result.success
        assert result.data["new_value"] == 40.0


class TestLeaderboardAndCompare:
    @pytest.mark.asyncio
    async def test_leaderboard(self, skill):
        # Create agents with different scores
        await skill.execute("record_event", {
            "agent_id": "top", "event_type": "boost",
            "dimension": "competence", "delta": 30.0,
        })
        await skill.execute("record_event", {
            "agent_id": "mid", "event_type": "boost",
            "dimension": "competence", "delta": 10.0,
        })
        result = await skill.execute("get_leaderboard", {"dimension": "competence"})
        assert result.success
        lb = result.data["leaderboard"]
        assert lb[0]["agent_id"] == "top"
        assert lb[1]["agent_id"] == "mid"

    @pytest.mark.asyncio
    async def test_compare(self, skill):
        await skill.execute("record_event", {
            "agent_id": "a", "event_type": "boost",
            "dimension": "competence", "delta": 20.0,
        })
        result = await skill.execute("compare", {"agent_a": "a", "agent_b": "b"})
        assert result.success
        comp = result.data["comparison"]["competence"]
        assert comp["advantage"] == "a"


class TestHistoryAndReset:
    @pytest.mark.asyncio
    async def test_history(self, skill):
        await skill.execute("record_event", {
            "agent_id": "a1", "event_type": "test",
            "dimension": "competence", "delta": 1.0,
        })
        result = await skill.execute("get_history", {"agent_id": "a1"})
        assert result.success
        assert len(result.data["events"]) == 1

    @pytest.mark.asyncio
    async def test_reset(self, skill):
        await skill.execute("record_event", {
            "agent_id": "a1", "event_type": "boost",
            "dimension": "competence", "delta": 30.0,
        })
        await skill.execute("reset", {"agent_id": "a1"})
        rep = await skill.execute("get_reputation", {"agent_id": "a1"})
        assert rep.data["overall"] == 50.0


def test_clamp_function():
    assert _clamp(50) == 50
    assert _clamp(-10) == 0.0
    assert _clamp(200) == 100.0


def test_manifest(skill):
    m = skill.manifest
    assert m.skill_id == "agent_reputation"
    assert len(m.actions) == 10
