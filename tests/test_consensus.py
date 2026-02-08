"""Tests for ConsensusProtocolSkill."""
import pytest
import json
from pathlib import Path
from singularity.skills.consensus import ConsensusProtocolSkill, CONSENSUS_FILE


@pytest.fixture(autouse=True)
def clean_store():
    if CONSENSUS_FILE.exists():
        CONSENSUS_FILE.unlink()
    yield
    if CONSENSUS_FILE.exists():
        CONSENSUS_FILE.unlink()


@pytest.fixture
def skill():
    s = ConsensusProtocolSkill()
    s._store = None
    return s


@pytest.mark.asyncio
async def test_propose_and_vote(skill):
    r = await skill.execute("propose", {"title": "Scale up", "description": "Add 2 agents", "proposer": "agent-1"})
    assert r.success
    pid = r.data["proposal_id"]

    r = await skill.execute("vote", {"proposal_id": pid, "voter": "agent-1", "choice": "approve"})
    assert r.success
    assert r.data["total_votes"] == 1

    r = await skill.execute("vote", {"proposal_id": pid, "voter": "agent-2", "choice": "reject"})
    assert r.success
    assert r.data["total_votes"] == 2


@pytest.mark.asyncio
async def test_tally_simple_majority_pass(skill):
    r = await skill.execute("propose", {"title": "T", "description": "D", "proposer": "a1"})
    pid = r.data["proposal_id"]
    await skill.execute("vote", {"proposal_id": pid, "voter": "a1", "choice": "approve"})
    await skill.execute("vote", {"proposal_id": pid, "voter": "a2", "choice": "approve"})
    await skill.execute("vote", {"proposal_id": pid, "voter": "a3", "choice": "reject"})

    r = await skill.execute("tally", {"proposal_id": pid})
    assert r.success
    assert r.data["result"]["status"] == "passed"
    assert r.data["result"]["approve_pct"] == pytest.approx(66.67, rel=0.1)


@pytest.mark.asyncio
async def test_tally_simple_majority_reject(skill):
    r = await skill.execute("propose", {"title": "T", "description": "D", "proposer": "a1"})
    pid = r.data["proposal_id"]
    await skill.execute("vote", {"proposal_id": pid, "voter": "a1", "choice": "approve"})
    await skill.execute("vote", {"proposal_id": pid, "voter": "a2", "choice": "reject"})
    await skill.execute("vote", {"proposal_id": pid, "voter": "a3", "choice": "reject"})

    r = await skill.execute("tally", {"proposal_id": pid})
    assert r.success
    assert r.data["result"]["status"] == "rejected"


@pytest.mark.asyncio
async def test_tally_supermajority(skill):
    r = await skill.execute("propose", {"title": "T", "description": "D", "proposer": "a1", "quorum_rule": "supermajority"})
    pid = r.data["proposal_id"]
    await skill.execute("vote", {"proposal_id": pid, "voter": "a1", "choice": "approve"})
    await skill.execute("vote", {"proposal_id": pid, "voter": "a2", "choice": "approve"})
    await skill.execute("vote", {"proposal_id": pid, "voter": "a3", "choice": "reject"})
    r = await skill.execute("tally", {"proposal_id": pid})
    # 66.67% approve, threshold is 66.67 - not strictly greater
    assert r.data["result"]["status"] == "rejected"


@pytest.mark.asyncio
async def test_weighted_voting(skill):
    r = await skill.execute("propose", {"title": "T", "description": "D", "proposer": "a1", "quorum_rule": "weighted_majority"})
    pid = r.data["proposal_id"]
    await skill.execute("vote", {"proposal_id": pid, "voter": "expert", "choice": "approve", "weight": 5.0})
    await skill.execute("vote", {"proposal_id": pid, "voter": "novice", "choice": "reject", "weight": 1.0})
    r = await skill.execute("tally", {"proposal_id": pid})
    assert r.data["result"]["status"] == "passed"


@pytest.mark.asyncio
async def test_plurality_election(skill):
    r = await skill.execute("elect", {
        "role": "task-handler",
        "candidates": ["a1", "a2", "a3"],
        "votes": {"v1": "a2", "v2": "a2", "v3": "a1"},
    })
    assert r.success
    assert r.data["winner"] == "a2"


@pytest.mark.asyncio
async def test_ranked_choice_election(skill):
    r = await skill.execute("elect", {
        "role": "leader",
        "candidates": ["a1", "a2", "a3"],
        "method": "ranked_choice",
        "rankings": {"v1": ["a3", "a2", "a1"], "v2": ["a2", "a3", "a1"], "v3": ["a3", "a1", "a2"]},
    })
    assert r.success
    assert r.data["winner"] == "a3"


@pytest.mark.asyncio
async def test_score_election(skill):
    r = await skill.execute("elect", {
        "role": "specialist",
        "candidates": ["a1", "a2"],
        "method": "score",
        "scores": {"v1": {"a1": 8, "a2": 6}, "v2": {"a1": 7, "a2": 9}},
    })
    assert r.success
    assert r.data["winner"] in ["a1", "a2"]


@pytest.mark.asyncio
async def test_proportional_allocation(skill):
    r = await skill.execute("allocate", {
        "resource": "compute_budget",
        "total_amount": 100,
        "requests": [
            {"agent_id": "a1", "requested_amount": 60, "priority": 1},
            {"agent_id": "a2", "requested_amount": 40, "priority": 2},
        ],
    })
    assert r.success
    alloc = r.data["allocation"]
    assert alloc["a1"] == 60
    assert alloc["a2"] == 40


@pytest.mark.asyncio
async def test_priority_allocation(skill):
    r = await skill.execute("allocate", {
        "resource": "tokens",
        "total_amount": 50,
        "requests": [
            {"agent_id": "a1", "requested_amount": 40, "priority": 10},
            {"agent_id": "a2", "requested_amount": 40, "priority": 5},
        ],
        "method": "priority_weighted",
    })
    assert r.success
    alloc = r.data["allocation"]
    assert alloc["a1"] == 40
    assert alloc["a2"] == 10


@pytest.mark.asyncio
async def test_conflict_resolution(skill):
    r = await skill.execute("resolve", {
        "parties": ["a1", "a2"],
        "issue": "Both want the same customer",
        "positions": {"a1": "I saw them first", "a2": "I'm more qualified"},
    })
    assert r.success
    cid = r.data["conflict_id"]

    r = await skill.execute("resolve", {
        "conflict_id": cid,
        "resolution": "a1 handles onboarding, a2 handles technical work",
    })
    assert r.success
    assert r.data["conflict"]["status"] == "resolved"


@pytest.mark.asyncio
async def test_status_and_history(skill):
    await skill.execute("propose", {"title": "Test", "description": "D", "proposer": "a1"})
    r = await skill.execute("status", {})
    assert r.success
    assert len(r.data["proposals"]) == 1

    r = await skill.execute("history", {"limit": 10})
    assert r.success


@pytest.mark.asyncio
async def test_vote_update(skill):
    r = await skill.execute("propose", {"title": "T", "description": "D", "proposer": "a1"})
    pid = r.data["proposal_id"]
    await skill.execute("vote", {"proposal_id": pid, "voter": "a1", "choice": "reject"})
    r = await skill.execute("vote", {"proposal_id": pid, "voter": "a1", "choice": "approve"})
    assert r.success
    assert "updated" in r.message.lower()
