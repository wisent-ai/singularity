"""Tests for GoalDependencyGraphSkill."""

import json
import pytest
from pathlib import Path
from unittest.mock import patch

from singularity.skills.goal_graph import GoalDependencyGraphSkill, GOALS_FILE


def _make_goals(goals_dict):
    """Helper: create goals data structure."""
    return {"goals": goals_dict}


def _write_goals(tmp_path, goals_dict):
    """Write goals to a temp file and patch GOALS_FILE."""
    f = tmp_path / "goals.json"
    f.write_text(json.dumps(_make_goals(goals_dict), indent=2))
    return f


@pytest.fixture
def skill():
    return GoalDependencyGraphSkill()


@pytest.fixture
def sample_goals():
    """A diamond dependency graph: A -> B, A -> C, B -> D, C -> D"""
    return {
        "a": {"title": "Goal A", "status": "active", "priority": "high", "pillar": "revenue", "depends_on": []},
        "b": {"title": "Goal B", "status": "active", "priority": "medium", "pillar": "revenue", "depends_on": ["a"]},
        "c": {"title": "Goal C", "status": "active", "priority": "low", "pillar": "self_improvement", "depends_on": ["a"]},
        "d": {"title": "Goal D", "status": "active", "priority": "critical", "pillar": "revenue", "depends_on": ["b", "c"]},
    }


@pytest.fixture
def parallel_goals():
    """Two independent chains: A->B and C->D"""
    return {
        "a": {"title": "Chain1-Start", "status": "active", "priority": "high", "pillar": "revenue", "depends_on": []},
        "b": {"title": "Chain1-End", "status": "active", "priority": "medium", "pillar": "revenue", "depends_on": ["a"]},
        "c": {"title": "Chain2-Start", "status": "active", "priority": "high", "pillar": "replication", "depends_on": []},
        "d": {"title": "Chain2-End", "status": "active", "priority": "medium", "pillar": "replication", "depends_on": ["c"]},
    }


class TestGraphBuilding:
    def test_build_graph_forward_edges(self, skill, sample_goals):
        fwd, rev = skill._build_graph(sample_goals)
        assert "b" in fwd.get("a", [])
        assert "c" in fwd.get("a", [])
        assert "d" in fwd.get("b", [])

    def test_build_graph_reverse_edges(self, skill, sample_goals):
        fwd, rev = skill._build_graph(sample_goals)
        assert "a" in rev.get("b", [])
        assert "b" in rev.get("d", [])
        assert "c" in rev.get("d", [])

    def test_empty_goals(self, skill):
        fwd, rev = skill._build_graph({})
        assert fwd == {}
        assert rev == {}


class TestCycleDetection:
    def test_no_cycles_in_dag(self, skill, sample_goals):
        cycles = skill._detect_cycles(sample_goals)
        assert len(cycles) == 0

    def test_detect_simple_cycle(self, skill):
        goals = {
            "x": {"title": "X", "status": "active", "depends_on": ["y"]},
            "y": {"title": "Y", "status": "active", "depends_on": ["x"]},
        }
        cycles = skill._detect_cycles(goals)
        assert len(cycles) > 0


class TestTopologicalSort:
    def test_order_respects_dependencies(self, skill, sample_goals):
        order = skill._topological_sort(sample_goals)
        pos = {gid: i for i, gid in enumerate(order)}
        assert pos["a"] < pos["b"]
        assert pos["a"] < pos["c"]
        assert pos["b"] < pos["d"]
        assert pos["c"] < pos["d"]

    def test_filter_by_pillar(self, skill, sample_goals):
        order = skill._topological_sort(sample_goals, pillar="revenue")
        assert "c" not in order  # c is self_improvement
        assert "a" in order

    def test_empty_goals(self, skill):
        order = skill._topological_sort({})
        assert order == []


class TestCriticalPath:
    def test_finds_longest_chain(self, skill, sample_goals):
        path = skill._find_critical_path(sample_goals)
        assert len(path) >= 3  # At least A -> B/C -> D

    def test_empty_goals(self, skill):
        path = skill._find_critical_path({})
        assert path == []


class TestParallelPaths:
    def test_finds_independent_chains(self, skill, parallel_goals):
        paths = skill._find_parallel_paths(parallel_goals)
        assert len(paths) == 2

    def test_single_connected_component(self, skill, sample_goals):
        paths = skill._find_parallel_paths(sample_goals)
        assert len(paths) == 1
        assert len(paths[0]) == 4


class TestImpactAnalysis:
    def test_root_has_most_impact(self, skill, sample_goals):
        impact = skill._compute_impact("a", sample_goals)
        assert impact["total_downstream"] == 3  # b, c, d

    def test_leaf_has_no_impact(self, skill, sample_goals):
        impact = skill._compute_impact("d", sample_goals)
        assert impact["total_downstream"] == 0


class TestSuggestNext:
    def test_suggests_ready_goals(self, skill, sample_goals):
        suggestions = skill._suggest_next(sample_goals)
        # Only "a" is ready (no deps)
        assert len(suggestions) == 1
        assert suggestions[0]["goal_id"] == "a"

    def test_all_ready(self, skill):
        goals = {
            "x": {"title": "X", "status": "active", "priority": "low", "depends_on": []},
            "y": {"title": "Y", "status": "active", "priority": "critical", "depends_on": []},
        }
        suggestions = skill._suggest_next(goals)
        assert suggestions[0]["goal_id"] == "y"  # Higher priority


class TestActions:
    @pytest.mark.asyncio
    async def test_analyze_no_goals(self, skill):
        with patch.object(skill, "_load_goals", return_value={}):
            result = await skill.execute("analyze")
            assert result.success
            assert result.data["total_goals"] == 0

    @pytest.mark.asyncio
    async def test_analyze_with_goals(self, skill, sample_goals):
        with patch.object(skill, "_load_goals", return_value=sample_goals):
            with patch.object(skill, "_save_graph_data"):
                result = await skill.execute("analyze")
                assert result.success
                assert result.data["stats"]["total_goals"] == 4

    @pytest.mark.asyncio
    async def test_topological_order(self, skill, sample_goals):
        with patch.object(skill, "_load_goals", return_value=sample_goals):
            result = await skill.execute("topological_order")
            assert result.success
            order = result.data["order"]
            assert order[0]["goal_id"] == "a"

    @pytest.mark.asyncio
    async def test_critical_path_action(self, skill, sample_goals):
        with patch.object(skill, "_load_goals", return_value=sample_goals):
            result = await skill.execute("critical_path")
            assert result.success
            assert result.data["length"] >= 3

    @pytest.mark.asyncio
    async def test_detect_cycles_clean(self, skill, sample_goals):
        with patch.object(skill, "_load_goals", return_value=sample_goals):
            result = await skill.execute("detect_cycles")
            assert result.success
            assert result.data["is_dag"]

    @pytest.mark.asyncio
    async def test_impact_analysis_action(self, skill, sample_goals):
        with patch.object(skill, "_load_goals", return_value=sample_goals):
            result = await skill.execute("impact_analysis", {"goal_id": "a"})
            assert result.success
            assert result.data["total_downstream"] == 3

    @pytest.mark.asyncio
    async def test_impact_analysis_missing_goal(self, skill):
        with patch.object(skill, "_load_goals", return_value={}):
            result = await skill.execute("impact_analysis", {"goal_id": "missing"})
            assert not result.success

    @pytest.mark.asyncio
    async def test_parallel_paths_action(self, skill, parallel_goals):
        with patch.object(skill, "_load_goals", return_value=parallel_goals):
            result = await skill.execute("parallel_paths")
            assert result.success
            assert result.data["total_chains"] == 2

    @pytest.mark.asyncio
    async def test_suggest_next_action(self, skill, sample_goals):
        with patch.object(skill, "_load_goals", return_value=sample_goals):
            result = await skill.execute("suggest_next")
            assert result.success
            assert len(result.data["suggestions"]) >= 1

    @pytest.mark.asyncio
    async def test_unknown_action(self, skill):
        result = await skill.execute("nonexistent")
        assert not result.success

    @pytest.mark.asyncio
    async def test_cascade_complete(self, skill, tmp_path, sample_goals):
        goals_file = _write_goals(tmp_path, sample_goals)
        with patch("singularity.skills.goal_graph.GOALS_FILE", goals_file):
            result = await skill.execute("cascade_complete", {"goal_id": "a"})
            assert result.success
            assert result.data["completed"]["goal_id"] == "a"

    @pytest.mark.asyncio
    async def test_manifest(self, skill):
        m = skill.manifest
        assert m.skill_id == "goal_dependency_graph"
        assert len(m.actions) == 8
