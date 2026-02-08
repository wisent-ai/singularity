"""Tests for ReputationWeightedVotingSkill."""
import pytest
import os
import json

from singularity.skills.reputation_weighted_voting import ReputationWeightedVotingSkill
from singularity.skills.agent_reputation import AgentReputationSkill
from singularity.skills.consensus import ConsensusProtocolSkill


@pytest.fixture
def skills(tmp_path, monkeypatch):
    """Create all three skills with temp data dirs."""
    data_dir = str(tmp_path / "data")
    os.makedirs(data_dir, exist_ok=True)
    monkeypatch.setattr("singularity.skills.reputation_weighted_voting.DATA_DIR", data_dir)
    monkeypatch.setattr("singularity.skills.reputation_weighted_voting.CONFIG_FILE", os.path.join(data_dir, "rwv_config.json"))
    monkeypatch.setattr("singularity.skills.reputation_weighted_voting.AUDIT_FILE", os.path.join(data_dir, "rwv_audit.json"))
    monkeypatch.setattr("singularity.skills.agent_reputation.DATA_DIR", data_dir)
    monkeypatch.setattr("singularity.skills.consensus.CONSENSUS_FILE", tmp_path / "data" / "consensus.json")

    rep = AgentReputationSkill()
    rep._persist_path = os.path.join(data_dir, "rep.json")
    cons = ConsensusProtocolSkill()
    rwv = ReputationWeightedVotingSkill()

    # Wire skills together via monkey-patching _get methods
    rwv._get_reputation_skill = lambda: rep
    rwv._get_consensus_skill = lambda: cons

    return {"rwv": rwv, "rep": rep, "cons": cons}


@pytest.mark.asyncio
async def test_vote_with_reputation(skills):
    """Vote weight should reflect agent reputation."""
    rwv, rep, cons = skills["rwv"], skills["rep"], skills["cons"]

    # Create a proposal
    result = await cons.execute("propose", {
        "title": "Test proposal", "description": "Testing", "proposer": "agent-1"
    })
    proposal_id = result.data["proposal_id"]

    # Boost agent-1's reputation
    await rep.execute("record_event", {"agent_id": "agent-1", "event_type": "test", "dimension": "trustworthiness", "delta": 30.0})

    # Cast reputation-weighted vote
    result = await rwv.execute("vote", {"proposal_id": proposal_id, "voter": "agent-1", "choice": "approve"})
    assert result.success
    assert result.data["weight"] > 1.0  # High-rep agent gets > 1.0 weight
    assert result.data["weight_source"] == "reputation"


@pytest.mark.asyncio
async def test_vote_low_reputation(skills):
    """Low-reputation agents should get lower vote weight."""
    rwv, rep, cons = skills["rwv"], skills["rep"], skills["cons"]

    result = await cons.execute("propose", {
        "title": "Test proposal", "description": "Testing", "proposer": "agent-2"
    })
    proposal_id = result.data["proposal_id"]

    # Penalize agent-2
    await rep.execute("penalize", {"agent_id": "agent-2", "dimension": "trustworthiness", "amount": 30.0, "reason": "test"})

    result = await rwv.execute("vote", {"proposal_id": proposal_id, "voter": "agent-2", "choice": "reject"})
    assert result.success
    assert result.data["weight"] < 1.0


@pytest.mark.asyncio
async def test_vote_override_weight(skills):
    """Manual weight override should skip reputation lookup."""
    rwv, cons = skills["rwv"], skills["cons"]

    result = await cons.execute("propose", {
        "title": "Override test", "description": "Testing", "proposer": "agent-3"
    })
    proposal_id = result.data["proposal_id"]

    result = await rwv.execute("vote", {
        "proposal_id": proposal_id, "voter": "agent-3", "choice": "approve", "override_weight": 2.5
    })
    assert result.success
    assert result.data["weight"] == 2.5
    assert result.data["weight_source"] == "manual_override"


@pytest.mark.asyncio
async def test_tally_with_feedback(skills):
    """Tally should record reputation feedback for voters."""
    rwv, rep, cons = skills["rwv"], skills["rep"], skills["cons"]

    result = await cons.execute("propose", {
        "title": "Feedback test", "description": "Testing", "proposer": "agent-a", "min_voters": 1
    })
    pid = result.data["proposal_id"]

    # Cast votes directly via consensus (simulating prior votes)
    await cons.execute("vote", {"proposal_id": pid, "voter": "agent-a", "choice": "approve"})
    await cons.execute("vote", {"proposal_id": pid, "voter": "agent-b", "choice": "reject"})

    # Tally via reputation-weighted skill
    result = await rwv.execute("tally", {"proposal_id": pid, "force_close": True})
    assert result.success
    assert "reputation_feedback" in result.data
    assert len(result.data["reputation_feedback"]) == 2


@pytest.mark.asyncio
async def test_simulate(skills):
    """Simulate should return weights for all agents."""
    rwv, rep = skills["rwv"], skills["rep"]

    # Set up agents with different reputations
    await rep.execute("record_event", {"agent_id": "high-rep", "event_type": "test", "dimension": "trustworthiness", "delta": 40.0})
    await rep.execute("record_event", {"agent_id": "low-rep", "event_type": "test", "dimension": "trustworthiness", "delta": -20.0})

    result = await rwv.execute("simulate", {"agent_ids": ["high-rep", "low-rep", "neutral-agent"]})
    assert result.success
    sims = result.data["simulations"]
    assert len(sims) == 3
    # High-rep agent should have highest weight
    assert sims[0]["agent_id"] == "high-rep"
    assert sims[0]["vote_weight"] > sims[-1]["vote_weight"]


@pytest.mark.asyncio
async def test_configure(skills):
    """Configure should update settings."""
    rwv = skills["rwv"]
    result = await rwv.execute("configure", {"sensitivity": 3.0, "min_weight": 0.5, "max_weight": 2.0})
    assert result.success
    assert result.data["config"]["sensitivity"] == 3.0
    assert result.data["config"]["min_weight"] == 0.5


@pytest.mark.asyncio
async def test_audit(skills):
    """Audit should track votes and tallies."""
    rwv, cons = skills["rwv"], skills["cons"]

    result = await cons.execute("propose", {
        "title": "Audit test", "description": "Testing", "proposer": "agent-1"
    })
    pid = result.data["proposal_id"]
    await rwv.execute("vote", {"proposal_id": pid, "voter": "agent-1", "choice": "approve"})

    result = await rwv.execute("audit", {"proposal_id": pid})
    assert result.success
    assert len(result.data["entries"]) >= 1
    assert result.data["entries"][0]["type"] == "vote"


@pytest.mark.asyncio
async def test_elect_with_reputation(skills):
    """Election should factor in candidate reputation scores."""
    rwv, rep = skills["rwv"], skills["rep"]

    await rep.execute("record_event", {"agent_id": "star", "event_type": "test", "dimension": "competence", "delta": 40.0})
    await rep.execute("record_event", {"agent_id": "star", "event_type": "test", "dimension": "trustworthiness", "delta": 30.0})

    result = await rwv.execute("elect", {
        "role": "team_lead",
        "candidates": ["star", "average", "newbie"],
    })
    assert result.success
    assert result.data["winner"] == "star"


@pytest.mark.asyncio
async def test_elect_plurality(skills):
    """Plurality election with reputation-weighted voter influence."""
    rwv, rep = skills["rwv"], skills["rep"]

    # Give voter-a high rep, voter-b low rep
    await rep.execute("record_event", {"agent_id": "voter-a", "event_type": "test", "dimension": "trustworthiness", "delta": 40.0})
    await rep.execute("penalize", {"agent_id": "voter-b", "dimension": "trustworthiness", "amount": 30.0, "reason": "test"})

    # voter-a votes for A, voter-b votes for B
    result = await rwv.execute("elect", {
        "role": "lead",
        "candidates": ["candidate-A", "candidate-B"],
        "voters": {"voter-a": "candidate-A", "voter-b": "candidate-B"},
        "method": "plurality",
    })
    assert result.success
    # voter-a has higher rep so candidate-A should win
    assert result.data["winner"] == "candidate-A"


@pytest.mark.asyncio
async def test_category_dimension_override(skills):
    """Strategy proposals should weight leadership more heavily."""
    rwv, rep, cons = skills["rwv"], skills["rep"], skills["cons"]

    # Agent with high leadership but low competence
    await rep.execute("record_event", {"agent_id": "leader", "event_type": "test", "dimension": "leadership", "delta": 40.0})
    await rep.execute("penalize", {"agent_id": "leader", "dimension": "competence", "amount": 20.0, "reason": "test"})

    # Create strategy proposal
    result = await cons.execute("propose", {
        "title": "Strategy change", "description": "New strategy", "proposer": "other", "category": "strategy"
    })
    pid = result.data["proposal_id"]

    # Vote on it - should get higher weight due to leadership
    result = await rwv.execute("vote", {"proposal_id": pid, "voter": "leader", "choice": "approve"})
    assert result.success
    strategy_weight = result.data["weight"]

    # Compare: simulate with default category
    sim = await rwv.execute("simulate", {"agent_ids": ["leader"]})
    default_weight = sim.data["simulations"][0]["vote_weight"]

    # Strategy weight should be different from default (leadership weighted more)
    assert strategy_weight != default_weight


@pytest.mark.asyncio
async def test_neutral_agent_weight(skills):
    """Agent with default reputation should get ~1.0 weight."""
    rwv, cons = skills["rwv"], skills["cons"]

    result = await cons.execute("propose", {
        "title": "Neutral test", "description": "Testing", "proposer": "neutral-agent"
    })
    pid = result.data["proposal_id"]

    result = await rwv.execute("vote", {"proposal_id": pid, "voter": "neutral-agent", "choice": "approve"})
    assert result.success
    # Neutral agent (all dims at 50.0) should get weight ~1.0
    assert 0.9 <= result.data["weight"] <= 1.1
