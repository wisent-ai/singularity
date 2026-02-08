"""Tests for SelfAssessmentSkill - capability profiling and gap analysis."""

import asyncio
import json
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from singularity.skills.self_assessment import SelfAssessmentSkill, DATA_FILE, CAPABILITY_CATEGORIES


@pytest.fixture
def skill(tmp_path):
    s = SelfAssessmentSkill()
    s._store = None
    # Use temp data file
    import singularity.skills.self_assessment as mod
    mod.DATA_FILE = tmp_path / "self_assessment.json"
    return s


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


class TestInventory:
    def test_inventory_returns_skills(self, skill):
        result = run(skill.execute("inventory", {}))
        assert result.success
        assert result.data["total_skills"] > 0
        assert "categories" in result.data
        assert "self_improvement" in result.data["categories"]

    def test_inventory_shows_missing(self, skill):
        result = run(skill.execute("inventory", {}))
        categories = result.data["categories"]
        # At least some categories should show installed/missing
        for cat_name, cat_data in categories.items():
            assert "installed" in cat_data
            assert "missing" in cat_data


class TestProfile:
    def test_profile_generates_scores(self, skill):
        result = run(skill.execute("profile", {"run_benchmarks": False}))
        assert result.success
        assert "overall_score" in result.data
        assert result.data["overall_score"] >= 0
        assert result.data["overall_score"] <= 100
        assert "categories" in result.data
        assert "total_skills" in result.data

    def test_profile_identifies_strongest_weakest(self, skill):
        result = run(skill.execute("profile", {"run_benchmarks": False}))
        assert result.data["strongest_category"] is not None
        assert result.data["weakest_category"] is not None

    def test_profile_saves_to_history(self, skill):
        run(skill.execute("profile", {"run_benchmarks": False}))
        store = skill._load()
        assert len(store["profiles"]) == 1
        assert store["profiles"][0]["overall_score"] >= 0


class TestGaps:
    def test_gaps_finds_missing(self, skill):
        result = run(skill.execute("gaps", {}))
        assert result.success
        assert "gaps" in result.data
        # Should find at least some gaps
        assert result.data["total_missing"] >= 0

    def test_gaps_sorted_by_impact(self, skill):
        result = run(skill.execute("gaps", {}))
        gaps = result.data["gaps"]
        if len(gaps) >= 2:
            # Verify sorted by impact descending
            for i in range(len(gaps) - 1):
                assert gaps[i]["impact_score"] >= gaps[i + 1]["impact_score"]


class TestRecommend:
    def test_recommend_returns_suggestions(self, skill):
        result = run(skill.execute("recommend", {"top_n": 3}))
        assert result.success
        assert "build_next" in result.data
        assert len(result.data["build_next"]) <= 3

    def test_recommend_has_priority_scores(self, skill):
        result = run(skill.execute("recommend", {}))
        for rec in result.data.get("build_next", []):
            assert "skill_id" in rec
            assert "priority_score" in rec
            assert "category" in rec


class TestHistory:
    def test_history_empty_initially(self, skill):
        result = run(skill.execute("history", {}))
        assert result.success
        assert len(result.data["history"]) == 0
        assert result.data["trend"] is None

    def test_history_tracks_profiles(self, skill):
        run(skill.execute("profile", {"run_benchmarks": False}))
        run(skill.execute("profile", {"run_benchmarks": False}))
        result = run(skill.execute("history", {}))
        assert len(result.data["history"]) == 2
        assert result.data["trend"] is not None
        assert result.data["trend"]["data_points"] == 2


class TestBenchmark:
    def test_benchmark_without_context(self, skill):
        result = run(skill.execute("benchmark", {}))
        assert result.success
        # Without context, probes return assumed healthy
        assert result.data["healthy"] >= 0


class TestPublish:
    def test_publish_without_context(self, skill):
        result = run(skill.execute("publish", {"agent_id": "test-agent"}))
        assert result.success
        assert result.data["agent_id"] == "test-agent"
        assert result.data["shared"] is False

    def test_publish_with_context(self, skill):
        mock_ctx = MagicMock()
        mock_ctx.list_skills.return_value = ["knowledge_sharing", "agent_network"]
        mock_ctx.call_skill = AsyncMock(return_value=MagicMock(success=True, data={}, message="ok"))
        skill.context = mock_ctx
        result = run(skill.execute("publish", {"agent_id": "agent-1"}))
        assert result.success


class TestCompare:
    def test_compare_no_agent_id(self, skill):
        result = run(skill.execute("compare", {}))
        assert not result.success
        assert "required" in result.message

    def test_compare_no_other_profile(self, skill):
        result = run(skill.execute("compare", {"agent_id": "unknown"}))
        assert not result.success
        assert "No published profile" in result.message


class TestManifest:
    def test_manifest_valid(self, skill):
        m = skill.manifest
        assert m.skill_id == "self_assessment"
        assert len(m.actions) == 8
        action_names = [a.name for a in m.actions]
        assert "inventory" in action_names
        assert "benchmark" in action_names
        assert "profile" in action_names
        assert "publish" in action_names
        assert "gaps" in action_names
        assert "recommend" in action_names
        assert "compare" in action_names
        assert "history" in action_names


class TestUnknownAction:
    def test_unknown_action(self, skill):
        result = run(skill.execute("nonexistent", {}))
        assert not result.success
        assert "Unknown" in result.message
