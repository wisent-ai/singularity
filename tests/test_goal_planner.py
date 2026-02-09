"""Tests for the GoalPlannerSkill."""

import pytest
import json
from singularity.skills.goal_planner import GoalPlannerSkill


@pytest.fixture
def skill():
    s = GoalPlannerSkill()
    s.initialized = True
    return s


@pytest.mark.asyncio
async def test_set_goal(skill):
    result = await skill.execute("set_goal", {"title": "Test", "description": "A test goal"})
    assert result.success
    assert result.data["goal_id"] == "G1"
    assert result.data["priority"] == "medium"


@pytest.mark.asyncio
async def test_set_goal_with_priority(skill):
    result = await skill.execute("set_goal", {"title": "Urgent", "description": "Fix bug", "priority": "critical"})
    assert result.success
    assert result.data["priority"] == "critical"


@pytest.mark.asyncio
async def test_set_goal_validation(skill):
    result = await skill.execute("set_goal", {"title": "", "description": "no title"})
    assert not result.success


@pytest.mark.asyncio
async def test_add_and_complete_task(skill):
    await skill.execute("set_goal", {"title": "Build", "description": "Build feature"})
    result = await skill.execute("add_task", {"goal_id": "G1", "task": "Write code"})
    assert result.success
    assert result.data["task_index"] == 0

    result = await skill.execute("complete_task", {"goal_id": "G1", "task_index": 0, "outcome": "done"})
    assert result.success
    assert result.data["progress_pct"] == 100


@pytest.mark.asyncio
async def test_complete_goal(skill):
    await skill.execute("set_goal", {"title": "Ship", "description": "Ship feature"})
    result = await skill.execute("complete_goal", {"goal_id": "G1", "summary": "Shipped!"})
    assert result.success
    assert "duration_seconds" in result.data


@pytest.mark.asyncio
async def test_fail_goal(skill):
    await skill.execute("set_goal", {"title": "Try", "description": "Try something"})
    result = await skill.execute("fail_goal", {"goal_id": "G1", "reason": "blocked"})
    assert result.success
    assert "blocked" in result.data["reason"]


@pytest.mark.asyncio
async def test_reprioritize(skill):
    await skill.execute("set_goal", {"title": "Task", "description": "A task", "priority": "low"})
    result = await skill.execute("reprioritize", {"goal_id": "G1", "new_priority": "high"})
    assert result.success
    assert result.data["new_priority"] == "high"


@pytest.mark.asyncio
async def test_list_goals(skill):
    await skill.execute("set_goal", {"title": "A", "description": "First", "priority": "low"})
    await skill.execute("set_goal", {"title": "B", "description": "Second", "priority": "critical"})
    result = await skill.execute("list_goals", {})
    assert result.success
    assert result.data["count"] == 2
    assert "critical" in result.data["goals"][0]  # Critical first


@pytest.mark.asyncio
async def test_get_plan(skill):
    await skill.execute("set_goal", {"title": "Plan", "description": "Test planning"})
    await skill.execute("add_task", {"goal_id": "G1", "task": "Step 1"})
    result = await skill.execute("get_plan", {})
    assert result.success
    assert result.data["active_count"] == 1
    assert "Step 1" in result.data["suggestion"]


@pytest.mark.asyncio
async def test_evaluate(skill):
    await skill.execute("set_goal", {"title": "Done", "description": "Will complete"})
    await skill.execute("set_goal", {"title": "Active", "description": "Still going"})
    await skill.execute("complete_goal", {"goal_id": "G1", "summary": "Done"})
    result = await skill.execute("evaluate", {})
    assert result.success
    assert result.data["completed"] == 1
    assert result.data["active"] == 1
    assert result.data["completion_rate_pct"] == 50


@pytest.mark.asyncio
async def test_export_import(skill):
    await skill.execute("set_goal", {"title": "Export", "description": "Test export"})
    export_result = await skill.execute("export_goals", {})
    assert export_result.success
    goals_json = export_result.data["goals_json"]

    new_skill = GoalPlannerSkill()
    import_result = await new_skill.execute("import_goals", {"goals_json": goals_json})
    assert import_result.success
    assert import_result.data["imported_count"] == 1


@pytest.mark.asyncio
async def test_sub_goals(skill):
    await skill.execute("set_goal", {"title": "Parent", "description": "Parent goal"})
    result = await skill.execute("set_goal", {"title": "Child", "description": "Sub-goal", "parent_id": "G1"})
    assert result.success
    parent = await skill.execute("get_goal", {"goal_id": "G1"})
    assert "G2" in parent.data["children"]
