"""Tests for ConsensusTaskAssignmentSkill."""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from singularity.skills.consensus_task_assignment import (
    ConsensusTaskAssignmentSkill,
    ASSIGNMENT_FILE,
    STATUS_NOMINATING,
    STATUS_VOTING,
    STATUS_ASSIGNED,
    STATUS_COMPLETED,
    STATUS_FAILED,
)


@pytest.fixture
def skill(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "singularity.skills.consensus_task_assignment.ASSIGNMENT_FILE",
        tmp_path / "consensus_assignments.json",
    )
    return ConsensusTaskAssignmentSkill()


@pytest.mark.asyncio
async def test_propose_basic(skill):
    result = await skill.execute("propose", {
        "task_name": "Deploy service",
        "task_description": "Deploy the billing service to prod",
        "budget": 10.0,
        "proposer": "agent-alpha",
        "candidates": ["agent-beta", "agent-gamma"],
    })
    assert result.success
    assert "cta-" in result.data["assignment_id"]
    assert result.data["status"] == STATUS_VOTING  # has candidates >= min
    assert len(result.data["candidates"]) == 2


@pytest.mark.asyncio
async def test_propose_no_candidates_starts_nominating(skill):
    result = await skill.execute("propose", {
        "task_name": "Review code",
        "task_description": "Review PR #42",
        "budget": 5.0,
        "proposer": "agent-alpha",
    })
    assert result.success
    assert result.data["status"] == STATUS_NOMINATING


@pytest.mark.asyncio
async def test_propose_validation(skill):
    result = await skill.execute("propose", {
        "task_name": "",
        "task_description": "desc",
        "budget": 5.0,
        "proposer": "agent-alpha",
    })
    assert not result.success

    result = await skill.execute("propose", {
        "task_name": "name",
        "task_description": "desc",
        "budget": -1,
        "proposer": "agent-alpha",
    })
    assert not result.success


@pytest.mark.asyncio
async def test_nominate_and_transition(skill):
    # Create with no candidates
    r = await skill.execute("propose", {
        "task_name": "Test task",
        "task_description": "A task",
        "budget": 5.0,
        "proposer": "agent-alpha",
    })
    aid = r.data["assignment_id"]
    assert r.data["status"] == STATUS_NOMINATING

    # Nominate
    r2 = await skill.execute("nominate", {
        "assignment_id": aid,
        "candidate_id": "agent-beta",
        "nominator": "agent-alpha",
        "rationale": "Good at this",
    })
    assert r2.success
    assert r2.data["status"] == STATUS_VOTING  # now has >= min_candidates


@pytest.mark.asyncio
async def test_vote_and_close(skill):
    # Propose with 2 candidates
    r = await skill.execute("propose", {
        "task_name": "Build feature",
        "task_description": "Build the API",
        "budget": 20.0,
        "proposer": "agent-alpha",
        "candidates": ["agent-beta", "agent-gamma"],
    })
    aid = r.data["assignment_id"]

    # Vote for beta
    r2 = await skill.execute("vote", {
        "assignment_id": aid,
        "voter": "agent-alpha",
        "candidate_id": "agent-beta",
    })
    assert r2.success
    assert r2.data["weight"] == 1.0  # no context, default weight

    # Vote for gamma
    r3 = await skill.execute("vote", {
        "assignment_id": aid,
        "voter": "agent-delta",
        "candidate_id": "agent-gamma",
    })
    assert r3.success

    # Close voting
    r4 = await skill.execute("close_voting", {"assignment_id": aid})
    assert r4.success
    assert r4.data["winner"] in ("agent-beta", "agent-gamma")
    assert r4.data["total_votes"] == 2


@pytest.mark.asyncio
async def test_close_no_votes_uses_reputation(skill):
    """When no votes are cast, reputation scores determine winner."""
    r = await skill.execute("propose", {
        "task_name": "Analyze data",
        "task_description": "Run analysis",
        "budget": 8.0,
        "proposer": "agent-alpha",
        "candidates": ["agent-beta", "agent-gamma"],
    })
    aid = r.data["assignment_id"]

    # Close without any votes
    r2 = await skill.execute("close_voting", {"assignment_id": aid})
    assert r2.success
    assert r2.data["winner"] is not None
    assert r2.data["total_votes"] == 0


@pytest.mark.asyncio
async def test_report_outcome(skill):
    # Full flow: propose -> close -> report
    r = await skill.execute("propose", {
        "task_name": "Write docs",
        "task_description": "Write API docs",
        "budget": 5.0,
        "proposer": "agent-alpha",
        "candidates": ["agent-beta"],
    })
    aid = r.data["assignment_id"]

    await skill.execute("close_voting", {"assignment_id": aid})

    r3 = await skill.execute("report_outcome", {
        "assignment_id": aid,
        "success": True,
        "quality_score": 85,
        "budget_spent": 3.5,
        "notes": "Well done",
    })
    assert r3.success
    assert r3.data["outcome"]["success"] is True
    assert r3.data["outcome"]["quality_score"] == 85


@pytest.mark.asyncio
async def test_leaderboard(skill):
    # Create 2 assignments with outcomes
    for task, candidate, success in [("T1", "agent-beta", True), ("T2", "agent-gamma", False)]:
        r = await skill.execute("propose", {
            "task_name": task, "task_description": "desc",
            "budget": 5.0, "proposer": "agent-alpha",
            "candidates": [candidate],
        })
        await skill.execute("close_voting", {"assignment_id": r.data["assignment_id"]})
        await skill.execute("report_outcome", {
            "assignment_id": r.data["assignment_id"],
            "success": success, "quality_score": 80 if success else 20,
        })

    r = await skill.execute("leaderboard", {"limit": 10})
    assert r.success
    assert len(r.data["leaderboard"]) == 2


@pytest.mark.asyncio
async def test_history(skill):
    r = await skill.execute("propose", {
        "task_name": "Task1", "task_description": "desc",
        "budget": 5.0, "proposer": "agent-alpha",
        "candidates": ["agent-beta"],
    })

    r2 = await skill.execute("history", {"limit": 10})
    assert r2.success
    assert len(r2.data["history"]) == 1


@pytest.mark.asyncio
async def test_status_specific(skill):
    r = await skill.execute("propose", {
        "task_name": "Check task", "task_description": "desc",
        "budget": 5.0, "proposer": "agent-alpha",
        "candidates": ["agent-beta"],
    })
    aid = r.data["assignment_id"]

    r2 = await skill.execute("status", {"assignment_id": aid})
    assert r2.success
    assert r2.data["assignment"]["task_name"] == "Check task"


@pytest.mark.asyncio
async def test_status_all_active(skill):
    await skill.execute("propose", {
        "task_name": "T1", "task_description": "d",
        "budget": 5.0, "proposer": "a", "candidates": ["b"],
    })
    r = await skill.execute("status", {})
    assert r.success
    assert len(r.data["active"]) == 1


@pytest.mark.asyncio
async def test_vote_change(skill):
    """Voters can change their vote."""
    r = await skill.execute("propose", {
        "task_name": "T", "task_description": "d",
        "budget": 5.0, "proposer": "a",
        "candidates": ["b", "c"],
    })
    aid = r.data["assignment_id"]

    await skill.execute("vote", {"assignment_id": aid, "voter": "v1", "candidate_id": "b"})
    r2 = await skill.execute("vote", {"assignment_id": aid, "voter": "v1", "candidate_id": "c"})
    assert r2.success
    assert r2.data["changed_from"] == "b"


@pytest.mark.asyncio
async def test_unknown_action(skill):
    r = await skill.execute("nonexistent", {})
    assert not r.success
