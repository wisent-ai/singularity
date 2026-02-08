"""Tests for CheckpointComparisonAnalyticsSkill."""
import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from singularity.skills.checkpoint_comparison import (
    CheckpointComparisonAnalyticsSkill,
    _classify_file,
    _compute_file_diff,
    _compute_progress_score,
    _detect_regressions,
    ANALYTICS_FILE,
    CHECKPOINT_DIR,
    CHECKPOINT_INDEX,
)


@pytest.fixture
def skill():
    return CheckpointComparisonAnalyticsSkill()


# --- Unit tests for helper functions ---

def test_classify_file():
    assert _classify_file("feedback_loop.json") == "self_improvement"
    assert _classify_file("revenue_catalog.json") == "revenue"
    assert _classify_file("agent_spawner.json") == "replication"
    assert _classify_file("goal_manager.json") == "goal_setting"
    assert _classify_file("random_file.json") == "general"


def test_compute_file_diff_added():
    a = {}
    b = {"new.json": {"size": 100, "hash": "abc"}}
    diff = _compute_file_diff(a, b)
    assert len(diff["added"]) == 1
    assert diff["added"][0]["file"] == "new.json"
    assert diff["size_delta"] == 100
    assert diff["files_delta"] == 1


def test_compute_file_diff_removed():
    a = {"old.json": {"size": 200, "hash": "xyz"}}
    b = {}
    diff = _compute_file_diff(a, b)
    assert len(diff["removed"]) == 1
    assert diff["size_delta"] == -200
    assert diff["files_delta"] == -1


def test_compute_file_diff_modified():
    a = {"f.json": {"size": 100, "hash": "aaa"}}
    b = {"f.json": {"size": 150, "hash": "bbb"}}
    diff = _compute_file_diff(a, b)
    assert len(diff["modified"]) == 1
    assert diff["modified"][0]["size_delta"] == 50


def test_compute_file_diff_unchanged():
    a = {"f.json": {"size": 100, "hash": "same"}}
    b = {"f.json": {"size": 100, "hash": "same"}}
    diff = _compute_file_diff(a, b)
    assert diff["unchanged_count"] == 1
    assert len(diff["added"]) == 0


def test_progress_score():
    diff = {"added": [{"file": "a.json"}], "removed": [], "modified": [{"file": "b.json"}, {"file": "c.json"}]}
    meta_a = {"total_size_bytes": 1000, "files_count": 5}
    meta_b = {"total_size_bytes": 1200, "files_count": 6}
    score = _compute_progress_score(diff, meta_a, meta_b)
    assert 0 <= score["total"] <= 100
    assert score["grade"] in ("A", "B", "C", "D", "F")
    assert score["data_growth"] >= 0
    assert score["stability"] == 25  # No removals


def test_progress_score_with_removals():
    diff = {"added": [], "removed": [{"file": "a.json"}, {"file": "b.json"}], "modified": []}
    meta_a = {"total_size_bytes": 1000, "files_count": 5}
    meta_b = {"total_size_bytes": 800, "files_count": 3}
    score = _compute_progress_score(diff, meta_a, meta_b)
    assert score["stability"] < 25  # Penalty for removals


def test_detect_regressions_file_removed():
    diff = {"added": [], "removed": [{"file": "important.json", "size": 5000}], "modified": []}
    regs = _detect_regressions(diff)
    assert len(regs) == 1
    assert regs[0]["type"] == "file_removed"
    assert regs[0]["severity"] == "high"


def test_detect_regressions_shrinkage():
    diff = {"added": [], "removed": [], "modified": [{"file": "data.json", "size_before": 1000, "size_after": 200, "size_delta": -800}]}
    regs = _detect_regressions(diff)
    assert len(regs) == 1
    assert regs[0]["type"] == "data_shrinkage"
    assert regs[0]["shrink_percent"] == 80.0


# --- Integration tests ---

@pytest.mark.asyncio
async def test_manifest(skill):
    m = skill.manifest
    assert m.skill_id == "checkpoint_comparison"
    assert len(m.actions) == 8
    action_names = {a.name for a in m.actions}
    assert "compare" in action_names
    assert "timeline" in action_names
    assert "trends" in action_names
    assert "report" in action_names


@pytest.mark.asyncio
async def test_unknown_action(skill):
    result = await skill.execute("nonexistent")
    assert not result.success


@pytest.mark.asyncio
async def test_compare_missing_params(skill):
    result = await skill.execute("compare", {})
    assert not result.success


@pytest.mark.asyncio
async def test_status(skill):
    result = await skill.execute("status")
    assert result.success
    assert "checkpoint_count" in result.data
