"""Tests for GoalDependencyGraphSkill - dependency analysis, critical path, cycles."""

import asyncio
import json
import pytest
from pathlib import Path
from unittest.mock import patch

from singularity.skills.goal_dependency_graph import GoalDependencyGraphSkill, GOALS_FILE


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def make_goals_data(goals, completed=None):
    return {
        "goals": goals,
        "completed_goals": completed or [],
        "session_log": [],
    }


def make_goal(gid, title, pillar="self_improvement", priority="medium", deps=None, status="active"):
    pri_map = {"critical": 4, "high": 3, "medium": 2, "low": 1}
    return {
        "id": gid,
        "title": title,
        "pillar": pillar,
        "priority": priority,
        "priority_score": pri_map.get(priority, 2),
        "status": status,
        "depends_on": deps or [],
        "milestones": [],
    }


@pytest.fixture
def skill():
    return GoalDependencyGraphSkill()


# ── Visualize ──────────────────────────────────────────────

def test_visualize_empty(skill):
    data = make_goals_data([])
    with patch.object(skill, "_load_goals", return_value=data):
        result = run(skill.execute("visualize", {}))
    assert result.success
    assert result.data["node_count"] == 0


def test_visualize_with_deps(skill):
    data = make_goals_data([
        make_goal("a", "Goal A"),
        make_goal("b", "Goal B", deps=["a"]),
    ])
    with patch.object(skill, "_load_goals", return_value=data):
        result = run(skill.execute("visualize", {}))
    assert result.success
    assert result.data["node_count"] == 2
    assert result.data["edge_count"] == 1


# ── Critical Path ──────────────────────────────────────────

def test_critical_path_linear_chain(skill):
    data = make_goals_data([
        make_goal("a", "Step 1"),
        make_goal("b", "Step 2", deps=["a"]),
        make_goal("c", "Step 3", deps=["b"]),
    ])
    with patch.object(skill, "_load_goals", return_value=data):
        result = run(skill.execute("critical_path", {}))
    assert result.success
    assert result.data["length"] == 3


def test_critical_path_empty(skill):
    data = make_goals_data([])
    with patch.object(skill, "_load_goals", return_value=data):
        result = run(skill.execute("critical_path", {}))
    assert result.success
    assert result.data["length"] == 0


# ── Execution Order ────────────────────────────────────────

def test_execution_order_topo_sort(skill):
    data = make_goals_data([
        make_goal("a", "Foundation"),
        make_goal("b", "Feature", deps=["a"]),
        make_goal("c", "Polish", deps=["b"]),
    ])
    with patch.object(skill, "_load_goals", return_value=data):
        result = run(skill.execute("execution_order", {}))
    assert result.success
    assert result.data["total_waves"] == 3
    assert result.data["has_cycles"] is False
    # Wave 1 should be "a", wave 2 "b", wave 3 "c"
    groups = result.data["parallel_groups"]
    assert groups[0]["goals"][0]["goal_id"] == "a"
    assert groups[1]["goals"][0]["goal_id"] == "b"


def test_execution_order_parallel(skill):
    data = make_goals_data([
        make_goal("a", "Task A"),
        make_goal("b", "Task B"),
        make_goal("c", "Final", deps=["a", "b"]),
    ])
    with patch.object(skill, "_load_goals", return_value=data):
        result = run(skill.execute("execution_order", {}))
    assert result.success
    assert result.data["total_waves"] == 2
    # Wave 1 has a and b in parallel
    wave1 = result.data["parallel_groups"][0]
    wave1_ids = {g["goal_id"] for g in wave1["goals"]}
    assert wave1_ids == {"a", "b"}


# ── Cycle Detection ────────────────────────────────────────

def test_no_cycles(skill):
    data = make_goals_data([
        make_goal("a", "Goal A"),
        make_goal("b", "Goal B", deps=["a"]),
    ])
    with patch.object(skill, "_load_goals", return_value=data):
        result = run(skill.execute("detect_cycles", {}))
    assert result.success
    assert result.data["count"] == 0


def test_detect_cycle(skill):
    data = make_goals_data([
        make_goal("a", "Goal A", deps=["b"]),
        make_goal("b", "Goal B", deps=["a"]),
    ])
    with patch.object(skill, "_load_goals", return_value=data):
        result = run(skill.execute("detect_cycles", {}))
    assert not result.success  # Cycles are a problem
    assert result.data["count"] > 0


# ── Impact Analysis ────────────────────────────────────────

def test_impact_unblocks_dependents(skill):
    data = make_goals_data([
        make_goal("a", "Blocker"),
        make_goal("b", "Blocked 1", deps=["a"]),
        make_goal("c", "Blocked 2", deps=["a"]),
    ])
    with patch.object(skill, "_load_goals", return_value=data):
        result = run(skill.execute("impact", {"goal_id": "a"}))
    assert result.success
    assert result.data["total_unblocked"] == 2


def test_impact_cascade(skill):
    data = make_goals_data([
        make_goal("a", "Root"),
        make_goal("b", "Mid", deps=["a"]),
        make_goal("c", "Leaf", deps=["b"]),
    ])
    with patch.object(skill, "_load_goals", return_value=data):
        result = run(skill.execute("impact", {"goal_id": "a"}))
    assert result.success
    assert len(result.data["directly_unblocked"]) == 1
    assert len(result.data["cascade_unblocked"]) == 1
    assert result.data["total_unblocked"] == 2


# ── Bottlenecks ────────────────────────────────────────────

def test_bottlenecks(skill):
    data = make_goals_data([
        make_goal("a", "Core", priority="critical"),
        make_goal("b", "Feature 1", deps=["a"]),
        make_goal("c", "Feature 2", deps=["a"]),
        make_goal("d", "Feature 3", deps=["a"]),
    ])
    with patch.object(skill, "_load_goals", return_value=data):
        result = run(skill.execute("bottlenecks", {}))
    assert result.success
    assert result.data["bottlenecks"][0]["goal_id"] == "a"
    assert result.data["bottlenecks"][0]["direct_blockers"] == 3


# ── Health ─────────────────────────────────────────────────

def test_health_healthy(skill):
    data = make_goals_data([
        make_goal("a", "Goal A"),
        make_goal("b", "Goal B", deps=["a"]),
    ])
    with patch.object(skill, "_load_goals", return_value=data):
        result = run(skill.execute("health", {}))
    assert result.success
    assert result.data["health"] in ("healthy", "warning")
    assert result.data["metrics"]["total_active_goals"] == 2
    assert result.data["metrics"]["has_cycles"] is False


def test_health_empty(skill):
    data = make_goals_data([])
    with patch.object(skill, "_load_goals", return_value=data):
        result = run(skill.execute("health", {}))
    assert result.success
    assert result.data["health"] == "empty"
