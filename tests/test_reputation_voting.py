"""Tests for ReputationWeightedVotingSkill."""

import json
import pytest
from pathlib import Path
from unittest.mock import patch

from singularity.skills.reputation_voting import (
    ReputationWeightedVotingSkill,
    compute_vote_weight,
    DATA_FILE,
    DEFAULT_VOTE_DIMENSION_WEIGHTS,
    CATEGORY_DIMENSION_OVERRIDES,
)


@pytest.fixture(autouse=True)
def clean_data(tmp_path, monkeypatch):
    """Use temp directory for data files."""
    data_file = tmp_path / "reputation_voting.json"
    rep_file = tmp_path / "agent_reputation.json"
    monkeypatch.setattr(
        "singularity.skills.reputation_voting.DATA_FILE", data_file
    )
    # Patch the reputation data file path used in _get_agent_reputation
    original_get_rep = ReputationWeightedVotingSkill._get_agent_reputation

    def patched_get_rep(self, agent_id):
        if rep_file.exists():
            try:
                data = json.loads(rep_file.read_text())
                agents = data.get("agents", {})
                if agent_id in agents:
                    agent = agents[agent_id]
                    return {
                        "competence": agent.get("competence", 50.0),
                        "reliability": agent.get("reliability", 50.0),
                        "trustworthiness": agent.get("trustworthiness", 50.0),
                        "leadership": agent.get("leadership", 50.0),
                        "cooperation": agent.get("cooperation", 50.0),
                        "overall": agent.get("overall", 50.0),
                    }
            except Exception:
                pass
        return {d: 50.0 for d in ["competence", "reliability", "trustworthiness", "leadership", "cooperation", "overall"]}

    monkeypatch.setattr(ReputationWeightedVotingSkill, "_get_agent_reputation", patched_get_rep)
    yield {"data_file": data_file, "rep_file": rep_file}


def write_reputation(rep_file, agents_data):
    """Helper to write agent reputation data."""
    rep_file.parent.mkdir(parents=True, exist_ok=True)
    rep_file.write_text(json.dumps({"agents": agents_data}, indent=2))


# ─── Unit Tests: compute_vote_weight ───────────────────────────────

def test_neutral_reputation_gives_weight_1():
    rep = {d: 50.0 for d in DEFAULT_VOTE_DIMENSION_WEIGHTS}
    assert compute_vote_weight(rep) == 1.0


def test_max_reputation_gives_max_weight():
    rep = {d: 100.0 for d in DEFAULT_VOTE_DIMENSION_WEIGHTS}
    assert compute_vote_weight(rep) == 3.0


def test_zero_reputation_gives_min_weight():
    rep = {d: 0.0 for d in DEFAULT_VOTE_DIMENSION_WEIGHTS}
    assert compute_vote_weight(rep) == 0.1


def test_high_reputation_above_1():
    rep = {d: 75.0 for d in DEFAULT_VOTE_DIMENSION_WEIGHTS}
    w = compute_vote_weight(rep)
    assert 1.0 < w < 3.0


def test_low_reputation_below_1():
    rep = {d: 25.0 for d in DEFAULT_VOTE_DIMENSION_WEIGHTS}
    w = compute_vote_weight(rep)
    assert 0.1 < w < 1.0


def test_custom_weight_range():
    rep = {d: 100.0 for d in DEFAULT_VOTE_DIMENSION_WEIGHTS}
    w = compute_vote_weight(rep, min_weight=0.5, max_weight=5.0)
    assert w == 5.0


# ─── Integration Tests: Skill Actions ─────────────────────────────

@pytest.mark.asyncio
async def test_create_proposal(clean_data):
    skill = ReputationWeightedVotingSkill()
    result = await skill.execute("create_proposal", {
        "title": "Adopt new strategy",
        "description": "Switch to aggressive revenue pursuit",
        "proposer": "agent-1",
        "category": "strategy",
    })
    assert result.success
    assert "rwv-" in result.data["proposal_id"]
    assert result.data["proposal"]["reputation_weighted"] is True


@pytest.mark.asyncio
async def test_cast_vote_with_reputation(clean_data):
    skill = ReputationWeightedVotingSkill()
    # Create proposal
    r = await skill.execute("create_proposal", {
        "title": "Test proposal",
        "description": "Testing",
        "proposer": "agent-1",
    })
    pid = r.data["proposal_id"]

    # Cast vote (neutral reputation = weight 1.0)
    r2 = await skill.execute("cast_vote", {
        "proposal_id": pid, "voter": "agent-2", "choice": "approve",
    })
    assert r2.success
    assert r2.data["weight"] == 1.0


@pytest.mark.asyncio
async def test_tally_weighted_votes(clean_data):
    skill = ReputationWeightedVotingSkill()
    r = await skill.execute("create_proposal", {
        "title": "Tally test", "description": "Testing tally",
        "proposer": "agent-1", "min_voters": 1,
    })
    pid = r.data["proposal_id"]

    await skill.execute("cast_vote", {"proposal_id": pid, "voter": "a1", "choice": "approve"})
    await skill.execute("cast_vote", {"proposal_id": pid, "voter": "a2", "choice": "reject"})

    r3 = await skill.execute("tally", {"proposal_id": pid, "force_close": True})
    assert r3.success
    # Equal reputation = equal weights, 1 approve vs 1 reject, should be rejected (not > 50%)
    assert r3.data["result"]["status"] == "rejected"


@pytest.mark.asyncio
async def test_tally_passes_with_majority(clean_data):
    skill = ReputationWeightedVotingSkill()
    r = await skill.execute("create_proposal", {
        "title": "Pass test", "description": "Should pass",
        "proposer": "agent-1", "min_voters": 1,
    })
    pid = r.data["proposal_id"]

    await skill.execute("cast_vote", {"proposal_id": pid, "voter": "a1", "choice": "approve"})
    await skill.execute("cast_vote", {"proposal_id": pid, "voter": "a2", "choice": "approve"})
    await skill.execute("cast_vote", {"proposal_id": pid, "voter": "a3", "choice": "reject"})

    r2 = await skill.execute("tally", {"proposal_id": pid, "force_close": True})
    assert r2.data["result"]["status"] == "passed"
    assert r2.data["result"]["approve_pct"] > 50.0


@pytest.mark.asyncio
async def test_run_election(clean_data):
    skill = ReputationWeightedVotingSkill()
    r = await skill.execute("run_election", {
        "role": "team_lead",
        "candidates": ["agent-A", "agent-B"],
        "voters": {"v1": "agent-A", "v2": "agent-A", "v3": "agent-B"},
    })
    assert r.success
    assert r.data["winner"] == "agent-A"


@pytest.mark.asyncio
async def test_get_voter_weight(clean_data):
    skill = ReputationWeightedVotingSkill()
    r = await skill.execute("get_voter_weight", {"agent_id": "agent-1"})
    assert r.success
    assert r.data["weight"] == 1.0  # Neutral reputation


@pytest.mark.asyncio
async def test_configure(clean_data):
    skill = ReputationWeightedVotingSkill()
    r = await skill.execute("configure", {
        "min_weight": 0.5, "max_weight": 5.0,
    })
    assert r.success
    assert "min_weight=0.5" in r.message


@pytest.mark.asyncio
async def test_invalid_vote_choice(clean_data):
    skill = ReputationWeightedVotingSkill()
    r = await skill.execute("create_proposal", {
        "title": "Test", "description": "Test", "proposer": "a1",
    })
    pid = r.data["proposal_id"]
    r2 = await skill.execute("cast_vote", {
        "proposal_id": pid, "voter": "a2", "choice": "maybe",
    })
    assert not r2.success


@pytest.mark.asyncio
async def test_vote_on_nonexistent_proposal(clean_data):
    skill = ReputationWeightedVotingSkill()
    r = await skill.execute("cast_vote", {
        "proposal_id": "fake", "voter": "a1", "choice": "approve",
    })
    assert not r.success
