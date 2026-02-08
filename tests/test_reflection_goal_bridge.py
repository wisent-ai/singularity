#!/usr/bin/env python3
"""Tests for ReflectionGoalBridgeSkill."""

import json
import pytest
import asyncio
import time
from pathlib import Path
from unittest.mock import patch

from singularity.skills.reflection_goal_bridge import (
    ReflectionGoalBridgeSkill,
    DATA_FILE,
    DEFAULT_CONFIG,
)


@pytest.fixture
def tmp_data(tmp_path):
    data_file = tmp_path / "reflection_goal_bridge.json"
    reflections_file = tmp_path / "reflections.json"
    goals_file = tmp_path / "goals.json"
    return tmp_path, data_file, reflections_file, goals_file


@pytest.fixture
def skill(tmp_data):
    tmp_path, data_file, reflections_file, goals_file = tmp_data
    with patch("singularity.skills.reflection_goal_bridge.DATA_FILE", data_file):
        s = ReflectionGoalBridgeSkill()
        yield s


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def make_reflections(tags_success_pairs):
    """Create reflection data from (tags, success) pairs."""
    reflections = []
    for i, (tags, success) in enumerate(tags_success_pairs):
        reflections.append({
            "id": f"ref_{i}",
            "task": f"Task {i}",
            "tags": tags,
            "success": success,
            "actions_taken": [f"action_{i}"],
            "improvements": [f"should improve {tags[0]} handling"] if not success else [],
            "analysis": f"Analysis for task {i}",
            "timestamp": time.time() - (len(tags_success_pairs) - i) * 60,
        })
    return reflections


def write_reflections(tmp_path, reflections):
    """Write reflections to the expected file location."""
    ref_file = tmp_path / "reflections.json"
    ref_file.parent.mkdir(parents=True, exist_ok=True)
    with open(ref_file, "w") as f:
        json.dump({"reflections": reflections, "playbooks": {}, "insights": [], "stats": {}}, f)


class TestScan:
    def test_scan_with_no_reflections(self, skill, tmp_data):
        async def mock_get(lookback):
            return []
        with patch.object(skill, '_get_reflections', side_effect=mock_get):
            result = run(skill.execute("scan", {"force": True}))
        assert result.success
        assert "not enough" in result.message.lower() or result.data.get("reflections_found", 0) == 0

    def test_scan_finds_weak_tags(self, skill, tmp_data):
        # Create reflections where 'deployment' tag has low success rate
        reflections = make_reflections([
            (["deployment"], False), (["deployment"], False), (["deployment"], False),
            (["deployment"], True), (["coding"], True), (["coding"], True),
            (["coding"], True), (["coding"], True), (["coding"], True),
        ])
        async def mock_get(lookback):
            return reflections
        with patch.object(skill, '_get_reflections', side_effect=mock_get):
            result = run(skill.execute("scan", {"force": True}))
        assert result.success
        weaknesses = result.data.get("weaknesses", [])
        weak_tags = [w for w in weaknesses if w["type"] == "weak_tag"]
        assert len(weak_tags) >= 1
        assert weak_tags[0]["tag"] == "deployment"
        assert weak_tags[0]["success_rate"] <= 0.4

    def test_scan_detects_declining_performance(self, skill):
        # Early: 80% success, Recent: 30% success
        early = [(["task"], True)] * 8 + [(["task"], False)] * 2
        recent = [(["task"], True)] * 3 + [(["task"], False)] * 7
        reflections = make_reflections(early + recent)
        async def mock_get(lookback):
            return reflections
        with patch.object(skill, '_get_reflections', side_effect=mock_get):
            result = run(skill.execute("scan", {"force": True}))
        assert result.success
        weaknesses = result.data.get("weaknesses", [])
        declining = [w for w in weaknesses if w["type"] == "declining_performance"]
        assert len(declining) == 1
        assert declining[0]["decline"] > 0.15

    def test_scan_cooldown(self, skill):
        skill._scans.append({"timestamp": time.time()})
        async def mock_get(lookback):
            return []
        with patch.object(skill, '_get_reflections', side_effect=mock_get):
            result = run(skill.execute("scan", {}))
        assert not result.success
        assert "cooldown" in result.message.lower()

    def test_scan_force_bypasses_cooldown(self, skill):
        skill._scans.append({"timestamp": time.time()})
        async def mock_get(lookback):
            return []
        with patch.object(skill, '_get_reflections', side_effect=mock_get):
            result = run(skill.execute("scan", {"force": True}))
        assert result.success

    def test_scan_detects_pillar_zero_success(self, skill):
        reflections = make_reflections([
            (["revenue"], False), (["revenue"], False), (["revenue"], False),
            (["coding"], True), (["coding"], True),
        ])
        async def mock_get(lookback):
            return reflections
        with patch.object(skill, '_get_reflections', side_effect=mock_get):
            result = run(skill.execute("scan", {"force": True}))
        assert result.success
        weaknesses = result.data.get("weaknesses", [])
        zero_success = [w for w in weaknesses if w["type"] == "pillar_zero_success"]
        assert len(zero_success) >= 1


class TestCreateGoals:
    def test_create_goals_from_recommendations(self, skill, tmp_data):
        tmp_path = tmp_data[0]
        skill._recommendations = [{
            "id": "rec_test1",
            "hash": "rec_test1",
            "status": "pending",
            "weakness": {"type": "weak_tag", "tag": "deploy"},
            "goal_title": "Improve deploy success",
            "goal_description": "Deploy tasks fail too often",
            "goal_pillar": "revenue",
            "goal_priority": "high",
            "goal_milestones": ["Analyze failures", "Fix root cause"],
            "created_at": "2025-01-01",
        }]
        goals_file = Path(__file__).parent.parent / "singularity" / "data" / "goals.json"
        with patch("singularity.skills.reflection_goal_bridge.Path") as mock_path_cls:
            # Use real Path but redirect data dir
            mock_path_cls.side_effect = Path
            mock_path_cls.__truediv__ = Path.__truediv__
            result = run(skill.execute("create_goals", {"dry_run": True}))
        assert result.success
        assert len(result.data["created"]) == 1

    def test_create_goals_no_pending(self, skill):
        result = run(skill.execute("create_goals", {}))
        assert result.success
        assert "no pending" in result.message.lower()

    def test_dry_run_does_not_change_status(self, skill):
        skill._recommendations = [{
            "id": "rec_dry",
            "hash": "rec_dry",
            "status": "pending",
            "weakness": {},
            "goal_title": "Test goal",
            "goal_description": "Test",
            "goal_pillar": "other",
            "goal_priority": "medium",
            "goal_milestones": ["Step 1"],
            "created_at": "2025-01-01",
        }]
        run(skill.execute("create_goals", {"dry_run": True}))
        assert skill._recommendations[0]["status"] == "pending"


class TestRecommendations:
    def test_list_recommendations(self, skill):
        skill._recommendations = [
            {"id": "r1", "status": "pending", "goal_title": "A"},
            {"id": "r2", "status": "created", "goal_title": "B"},
            {"id": "r3", "status": "dismissed", "goal_title": "C"},
        ]
        result = run(skill.execute("recommendations", {}))
        assert result.success
        assert len(result.data["recommendations"]) == 3
        assert result.data["counts"]["pending"] == 1
        assert result.data["counts"]["created"] == 1

    def test_filter_recommendations(self, skill):
        skill._recommendations = [
            {"id": "r1", "status": "pending", "goal_title": "A"},
            {"id": "r2", "status": "created", "goal_title": "B"},
        ]
        result = run(skill.execute("recommendations", {"status": "pending"}))
        assert len(result.data["recommendations"]) == 1


class TestDismiss:
    def test_dismiss_recommendation(self, skill, tmp_data):
        skill._recommendations = [{"id": "rec_x", "status": "pending", "goal_title": "X"}]
        result = run(skill.execute("dismiss", {"recommendation_id": "rec_x", "reason": "Not relevant"}))
        assert result.success
        assert skill._recommendations[0]["status"] == "dismissed"

    def test_dismiss_nonexistent(self, skill):
        result = run(skill.execute("dismiss", {"recommendation_id": "nonexistent"}))
        assert not result.success

    def test_dismiss_already_created(self, skill, tmp_data):
        skill._recommendations = [{"id": "rec_y", "status": "created", "goal_title": "Y"}]
        result = run(skill.execute("dismiss", {"recommendation_id": "rec_y"}))
        assert not result.success


class TestTrack:
    def test_track_no_goals(self, skill):
        result = run(skill.execute("track", {}))
        assert result.success
        assert "no goals" in result.message.lower()

    def test_track_with_goals(self, skill):
        skill._created_goals = [{
            "recommendation_id": "rec_1",
            "goal_id": "goal_abc",
            "title": "Test goal",
            "pillar": "self_improvement",
            "created_at": "2025-01-01",
            "status": "active",
        }]
        result = run(skill.execute("track", {}))
        assert result.success
        assert len(result.data["tracked_goals"]) == 1


class TestConfigure:
    def test_configure_valid_key(self, skill, tmp_data):
        result = run(skill.execute("configure", {"key": "weak_tag_threshold", "value": 0.5}))
        assert result.success
        assert skill._config["weak_tag_threshold"] == 0.5

    def test_configure_invalid_key(self, skill):
        result = run(skill.execute("configure", {"key": "nonexistent_key", "value": 42}))
        assert not result.success

    def test_configure_bool_conversion(self, skill, tmp_data):
        result = run(skill.execute("configure", {"key": "auto_create_goals", "value": "true"}))
        assert result.success
        assert skill._config["auto_create_goals"] is True


class TestStatus:
    def test_status_empty(self, skill):
        result = run(skill.execute("status", {}))
        assert result.success
        assert "config" in result.data
        assert "stats" in result.data

    def test_status_with_data(self, skill):
        skill._recommendations = [{"status": "pending"}, {"status": "created"}]
        skill._scans = [{"timestamp": time.time()}]
        result = run(skill.execute("status", {}))
        assert result.data["recommendations"]["pending"] == 1
        assert result.data["recommendations"]["created"] == 1
        assert result.data["last_scan"] is not None


class TestHistory:
    def test_history_empty(self, skill):
        result = run(skill.execute("history", {}))
        assert result.success
        assert len(result.data["scans"]) == 0

    def test_history_with_scans(self, skill):
        skill._scans = [{"timestamp": time.time(), "weaknesses_found": 3}]
        result = run(skill.execute("history", {"limit": 5}))
        assert len(result.data["scans"]) == 1


class TestTagToPillar:
    def test_known_tags(self, skill):
        assert skill._tag_to_pillar("revenue") == "revenue"
        assert skill._tag_to_pillar("replication") == "replication"
        assert skill._tag_to_pillar("testing") == "self_improvement"
        assert skill._tag_to_pillar("planning") == "goal_setting"

    def test_fuzzy_matching(self, skill):
        assert skill._tag_to_pillar("revenue_tracking") == "revenue"
        assert skill._tag_to_pillar("code_review") == "self_improvement"

    def test_unknown_tag(self, skill):
        assert skill._tag_to_pillar("xyzzy_unknown") == "other"


class TestWeaknessAnalysis:
    def test_analyze_empty(self, skill):
        result = skill._analyze_weaknesses([])
        assert result == []

    def test_analyze_recurring_improvements(self, skill):
        reflections = []
        for i in range(5):
            reflections.append({
                "tags": ["coding"],
                "success": False,
                "improvements": ["should validate input parameters before processing"],
            })
        weaknesses = skill._analyze_weaknesses(reflections)
        recurring = [w for w in weaknesses if w["type"] == "recurring_improvement"]
        assert len(recurring) >= 1
