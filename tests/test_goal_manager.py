#!/usr/bin/env python3
"""Tests for GoalManagerSkill."""
import json
import pytest
from unittest.mock import patch
from pathlib import Path
from singularity.skills.goal_manager import GoalManagerSkill, GOALS_FILE


@pytest.fixture
def skill(tmp_path):
    """Create a GoalManagerSkill with temp storage."""
    test_file = tmp_path / "goals.json"
    with patch.object(
        GoalManagerSkill, '_ensure_data',
        lambda self: None
    ):
        s = GoalManagerSkill()
    # Monkey-patch the file path
    import singularity.skills.goal_manager as mod
    original = mod.GOALS_FILE
    mod.GOALS_FILE = test_file
    s._ensure_data()
    yield s
    mod.GOALS_FILE = original


@pytest.mark.asyncio
async def test_create_goal(skill):
    r = await skill.execute("create", {
        "title": "Build API endpoint",
        "pillar": "revenue",
        "priority": "high",
        "milestones": ["Design schema", "Implement handler", "Write tests"],
    })
    assert r.success
    assert "goal_id" in r.data
    assert r.data["pillar"] == "revenue"
    assert r.data["milestones"] == 3


@pytest.mark.asyncio
async def test_create_requires_title(skill):
    r = await skill.execute("create", {"pillar": "revenue"})
    assert not r.success


@pytest.mark.asyncio
async def test_create_invalid_pillar(skill):
    r = await skill.execute("create", {"title": "Test", "pillar": "invalid"})
    assert not r.success


@pytest.mark.asyncio
async def test_next_empty(skill):
    r = await skill.execute("next", {})
    assert r.success
    assert r.data["next_goal"] is None


@pytest.mark.asyncio
async def test_next_returns_highest_priority(skill):
    await skill.execute("create", {"title": "Low", "pillar": "other", "priority": "low"})
    await skill.execute("create", {"title": "Critical", "pillar": "revenue", "priority": "critical"})
    await skill.execute("create", {"title": "Medium", "pillar": "other", "priority": "medium"})

    r = await skill.execute("next", {})
    assert r.success
    assert r.data["title"] == "Critical"
    assert r.data["priority"] == "critical"


@pytest.mark.asyncio
async def test_next_filters_by_pillar(skill):
    await skill.execute("create", {"title": "Revenue goal", "pillar": "revenue", "priority": "high"})
    await skill.execute("create", {"title": "Self goal", "pillar": "self_improvement", "priority": "critical"})

    r = await skill.execute("next", {"pillar": "revenue"})
    assert r.success
    assert r.data["pillar"] == "revenue"


@pytest.mark.asyncio
async def test_complete_milestone(skill):
    cr = await skill.execute("create", {
        "title": "Test goal",
        "pillar": "other",
        "milestones": ["Step 1", "Step 2"],
    })
    goal_id = cr.data["goal_id"]

    r = await skill.execute("progress", {"goal_id": goal_id, "milestone_index": 0})
    assert r.success
    assert "1/2" in r.data["progress"]


@pytest.mark.asyncio
async def test_auto_complete_on_all_milestones(skill):
    cr = await skill.execute("create", {
        "title": "Auto complete test",
        "pillar": "other",
        "milestones": ["Only step"],
    })
    goal_id = cr.data["goal_id"]

    r = await skill.execute("progress", {"goal_id": goal_id, "milestone_index": 0})
    assert r.success
    assert "completed" in r.message.lower()


@pytest.mark.asyncio
async def test_complete_goal(skill):
    cr = await skill.execute("create", {"title": "To complete", "pillar": "revenue"})
    goal_id = cr.data["goal_id"]

    r = await skill.execute("complete", {"goal_id": goal_id, "outcome": "Shipped it"})
    assert r.success

    # Should not appear in active list
    lr = await skill.execute("list", {"status": "active"})
    assert lr.data["count"] == 0


@pytest.mark.asyncio
async def test_abandon_goal(skill):
    cr = await skill.execute("create", {"title": "To abandon", "pillar": "other"})
    goal_id = cr.data["goal_id"]

    r = await skill.execute("abandon", {"goal_id": goal_id, "reason": "Not needed"})
    assert r.success
    assert r.data["reason"] == "Not needed"


@pytest.mark.asyncio
async def test_add_milestone(skill):
    cr = await skill.execute("create", {"title": "Extensible", "pillar": "other"})
    goal_id = cr.data["goal_id"]

    r = await skill.execute("add_milestone", {"goal_id": goal_id, "title": "New step"})
    assert r.success
    assert r.data["milestone_index"] == 0


@pytest.mark.asyncio
async def test_list_goals(skill):
    await skill.execute("create", {"title": "A", "pillar": "revenue"})
    await skill.execute("create", {"title": "B", "pillar": "self_improvement"})

    r = await skill.execute("list", {})
    assert r.success
    assert r.data["count"] == 2


@pytest.mark.asyncio
async def test_focus_plan(skill):
    await skill.execute("create", {"title": "Rev goal", "pillar": "revenue", "priority": "high", "milestones": ["Do it"]})
    await skill.execute("create", {"title": "Self goal", "pillar": "self_improvement", "milestones": ["Learn"]})

    r = await skill.execute("focus", {})
    assert r.success
    actionable = [f for f in r.data["focus_plan"] if f["status"] == "actionable"]
    assert len(actionable) == 2


@pytest.mark.asyncio
async def test_analyze(skill):
    cr = await skill.execute("create", {"title": "Done", "pillar": "revenue"})
    await skill.execute("complete", {"goal_id": cr.data["goal_id"]})
    await skill.execute("create", {"title": "Active", "pillar": "revenue"})

    r = await skill.execute("analyze", {})
    assert r.success
    assert r.data["completed_count"] == 1
    assert r.data["active_count"] == 1


@pytest.mark.asyncio
async def test_dependency_blocking(skill):
    cr1 = await skill.execute("create", {"title": "First", "pillar": "other"})
    gid1 = cr1.data["goal_id"]
    await skill.execute("create", {"title": "Depends on first", "pillar": "other", "priority": "critical", "depends_on": [gid1]})

    # The dependent goal should be blocked; "First" should be next
    r = await skill.execute("next", {})
    assert r.success
    assert r.data["title"] == "First"


@pytest.mark.asyncio
async def test_manifest(skill):
    m = skill.manifest
    assert m.skill_id == "goals"
    assert len(m.actions) == 10
