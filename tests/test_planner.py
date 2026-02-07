"""Tests for PlannerSkill - goal decomposition and planning."""
import pytest
import json
import asyncio
from pathlib import Path
from unittest.mock import patch

from singularity.skills.planner import PlannerSkill, PLANS_FILE


@pytest.fixture
def planner(tmp_path):
    """Create a PlannerSkill with isolated data dir."""
    test_file = tmp_path / "plans.json"
    with patch("singularity.skills.planner.PLANS_FILE", test_file):
        skill = PlannerSkill()
        yield skill


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


class TestGoalCreation:
    def test_create_goal(self, planner):
        result = run(planner.execute("create_goal", {
            "title": "Build revenue system",
            "description": "Create services that generate income",
            "priority": "high",
            "pillar": "revenue",
            "success_criteria": "Monthly revenue > $100",
        }))
        assert result.success
        assert result.data["goal_id"]
        assert result.data["priority"] == "high"

    def test_create_goal_requires_title(self, planner):
        result = run(planner.execute("create_goal", {"title": ""}))
        assert not result.success

    def test_create_goal_default_priority(self, planner):
        result = run(planner.execute("create_goal", {"title": "Test goal", "priority": "invalid"}))
        assert result.success
        # Invalid priority defaults to medium

    def test_create_goal_semicolon_criteria(self, planner):
        result = run(planner.execute("create_goal", {
            "title": "Multi-criteria",
            "success_criteria": "Criterion A; Criterion B; Criterion C",
        }))
        assert result.success
        goal_result = run(planner.execute("get_goal", {"goal_id": result.data["goal_id"]}))
        assert len(goal_result.data["goal"]["success_criteria"]) == 3


class TestTaskManagement:
    def test_add_task(self, planner):
        g = run(planner.execute("create_goal", {"title": "Goal"}))
        gid = g.data["goal_id"]
        result = run(planner.execute("add_task", {
            "goal_id": gid,
            "title": "First task",
            "effort": "small",
        }))
        assert result.success
        assert result.data["status"] == "pending"

    def test_add_task_with_deps(self, planner):
        g = run(planner.execute("create_goal", {"title": "Goal"}))
        gid = g.data["goal_id"]
        t1 = run(planner.execute("add_task", {"goal_id": gid, "title": "Task 1"}))
        t2 = run(planner.execute("add_task", {
            "goal_id": gid,
            "title": "Task 2",
            "depends_on": t1.data["task_id"],
        }))
        assert t2.success
        assert t2.data["status"] == "blocked"

    def test_add_task_invalid_dep(self, planner):
        g = run(planner.execute("create_goal", {"title": "Goal"}))
        result = run(planner.execute("add_task", {
            "goal_id": g.data["goal_id"],
            "title": "Bad dep",
            "depends_on": "nonexistent",
        }))
        assert not result.success

    def test_update_task_status(self, planner):
        g = run(planner.execute("create_goal", {"title": "Goal"}))
        gid = g.data["goal_id"]
        t = run(planner.execute("add_task", {"goal_id": gid, "title": "Task"}))
        result = run(planner.execute("update_task", {
            "goal_id": gid,
            "task_id": t.data["task_id"],
            "status": "completed",
            "note": "Done!",
        }))
        assert result.success
        assert result.data["new_status"] == "completed"

    def test_completing_task_unblocks_dependents(self, planner):
        g = run(planner.execute("create_goal", {"title": "Goal"}))
        gid = g.data["goal_id"]
        t1 = run(planner.execute("add_task", {"goal_id": gid, "title": "T1"}))
        t2 = run(planner.execute("add_task", {
            "goal_id": gid, "title": "T2",
            "depends_on": t1.data["task_id"],
        }))
        assert t2.data["status"] == "blocked"
        run(planner.execute("update_task", {
            "goal_id": gid, "task_id": t1.data["task_id"], "status": "completed",
        }))
        goal = run(planner.execute("get_goal", {"goal_id": gid}))
        t2_updated = [t for t in goal.data["goal"]["tasks"] if t["id"] == t2.data["task_id"]][0]
        assert t2_updated["status"] == "pending"

    def test_remove_task(self, planner):
        g = run(planner.execute("create_goal", {"title": "Goal"}))
        gid = g.data["goal_id"]
        t = run(planner.execute("add_task", {"goal_id": gid, "title": "Remove me"}))
        result = run(planner.execute("remove_task", {"goal_id": gid, "task_id": t.data["task_id"]}))
        assert result.success
        goal = run(planner.execute("get_goal", {"goal_id": gid}))
        assert len(goal.data["goal"]["tasks"]) == 0


class TestNextTask:
    def test_next_task_picks_highest_priority(self, planner):
        g1 = run(planner.execute("create_goal", {"title": "Low goal", "priority": "low"}))
        g2 = run(planner.execute("create_goal", {"title": "Critical goal", "priority": "critical"}))
        run(planner.execute("add_task", {"goal_id": g1.data["goal_id"], "title": "Low task"}))
        run(planner.execute("add_task", {"goal_id": g2.data["goal_id"], "title": "Critical task"}))
        result = run(planner.execute("next_task", {}))
        assert result.success
        assert result.data["task"]["goal_priority"] == "critical"

    def test_next_task_empty(self, planner):
        result = run(planner.execute("next_task", {}))
        assert result.success
        assert result.data["task"] is None


class TestGoalCompletion:
    def test_auto_complete_goal(self, planner):
        g = run(planner.execute("create_goal", {"title": "Auto-complete me"}))
        gid = g.data["goal_id"]
        t = run(planner.execute("add_task", {"goal_id": gid, "title": "Only task"}))
        run(planner.execute("update_task", {
            "goal_id": gid, "task_id": t.data["task_id"], "status": "completed",
        }))
        goal = run(planner.execute("get_goal", {"goal_id": gid}))
        assert goal.data["goal"]["status"] == "completed"

    def test_replan_resets_failed(self, planner):
        g = run(planner.execute("create_goal", {"title": "Replan me"}))
        gid = g.data["goal_id"]
        t = run(planner.execute("add_task", {"goal_id": gid, "title": "Will fail"}))
        run(planner.execute("update_task", {
            "goal_id": gid, "task_id": t.data["task_id"], "status": "failed",
        }))
        result = run(planner.execute("replan", {"goal_id": gid, "reason": "Retry"}))
        assert result.success
        assert result.data["summary"]["failed_reset"] == 1


class TestProgress:
    def test_progress_report(self, planner):
        g = run(planner.execute("create_goal", {"title": "G", "pillar": "revenue"}))
        gid = g.data["goal_id"]
        t = run(planner.execute("add_task", {"goal_id": gid, "title": "T"}))
        run(planner.execute("update_task", {
            "goal_id": gid, "task_id": t.data["task_id"], "status": "completed",
        }))
        result = run(planner.execute("progress", {}))
        assert result.success
        assert result.data["overall"]["completed_tasks"] == 1
        assert "revenue" in result.data["by_pillar"]
