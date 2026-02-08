#!/usr/bin/env python3
"""Tests for AgentReflectionSkill."""

import json
import pytest
import asyncio
from pathlib import Path
from unittest.mock import patch

from singularity.skills.agent_reflection import AgentReflectionSkill, REFLECTION_FILE


@pytest.fixture
def skill(tmp_path):
    """Create a skill with temp data file."""
    test_file = tmp_path / "reflections.json"
    with patch("singularity.skills.agent_reflection.REFLECTION_FILE", test_file):
        s = AgentReflectionSkill()
        yield s


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


class TestReflect:
    def test_basic_reflection(self, skill):
        result = run(skill.execute("reflect", {
            "task": "Deploy service to production",
            "actions_taken": ["build_image", "push_registry", "deploy_k8s"],
            "outcome": "Service deployed successfully",
            "success": True,
            "analysis": "Docker build was slow but deployment went smooth",
            "tags": ["deployment", "docker"],
        }))
        assert result.success
        assert "reflection_id" in result.data
        assert result.data["total_reflections"] == 1

    def test_missing_required_fields(self, skill):
        result = run(skill.execute("reflect", {"task": "test"}))
        assert not result.success

    def test_multiple_reflections(self, skill):
        for i in range(5):
            run(skill.execute("reflect", {
                "task": f"Task {i}",
                "actions_taken": [f"action_{i}"],
                "outcome": f"Outcome {i}",
                "success": i % 2 == 0,
                "analysis": f"Analysis {i}",
                "tags": ["test"],
            }))
        result = run(skill.execute("review", {"what": "reflections"}))
        assert result.success
        assert len(result.data["reflections"]) == 5


class TestPlaybooks:
    def test_create_playbook(self, skill):
        result = run(skill.execute("create_playbook", {
            "name": "deploy_service",
            "task_pattern": "deploy a service to production",
            "steps": ["Build Docker image", "Push to registry", "Update k8s manifest", "Apply and verify"],
            "pitfalls": ["Forgetting to update env vars", "Not checking health endpoint"],
            "tags": ["deployment"],
        }))
        assert result.success
        assert result.data["playbook"]["name"] == "deploy_service"
        assert len(result.data["playbook"]["steps"]) == 4

    def test_duplicate_playbook_rejected(self, skill):
        run(skill.execute("create_playbook", {
            "name": "pb1", "task_pattern": "test", "steps": ["step1"],
        }))
        result = run(skill.execute("create_playbook", {
            "name": "pb1", "task_pattern": "test2", "steps": ["step2"],
        }))
        assert not result.success
        assert "already exists" in result.message

    def test_find_playbook(self, skill):
        run(skill.execute("create_playbook", {
            "name": "deploy_service",
            "task_pattern": "deploy a service to production",
            "steps": ["Build", "Push", "Deploy"],
            "tags": ["deployment"],
        }))
        run(skill.execute("create_playbook", {
            "name": "fix_bug",
            "task_pattern": "debug and fix a code bug",
            "steps": ["Reproduce", "Diagnose", "Fix", "Test"],
            "tags": ["debugging"],
        }))
        result = run(skill.execute("find_playbook", {
            "task_description": "deploy a new service to production environment",
        }))
        assert result.success
        assert len(result.data["matches"]) >= 1
        assert result.data["matches"][0]["name"] == "deploy_service"

    def test_find_playbook_with_tags(self, skill):
        run(skill.execute("create_playbook", {
            "name": "pb1", "task_pattern": "general task",
            "steps": ["s1"], "tags": ["alpha"],
        }))
        run(skill.execute("create_playbook", {
            "name": "pb2", "task_pattern": "general task",
            "steps": ["s1"], "tags": ["beta"],
        }))
        result = run(skill.execute("find_playbook", {
            "task_description": "general task", "tags": ["beta"],
        }))
        assert result.success
        matches = result.data["matches"]
        beta_match = [m for m in matches if m["name"] == "pb2"]
        assert len(beta_match) >= 1

    def test_record_playbook_use(self, skill):
        run(skill.execute("create_playbook", {
            "name": "pb1", "task_pattern": "test",
            "steps": ["s1"],
        }))
        run(skill.execute("record_playbook_use", {
            "playbook_name": "pb1", "success": True, "notes": "Worked great",
        }))
        run(skill.execute("record_playbook_use", {
            "playbook_name": "pb1", "success": True,
        }))
        run(skill.execute("record_playbook_use", {
            "playbook_name": "pb1", "success": False,
        }))
        result = run(skill.execute("record_playbook_use", {
            "playbook_name": "pb1", "success": True,
        }))
        assert result.success
        assert result.data["uses"] == 4
        assert result.data["successes"] == 3
        assert abs(result.data["effectiveness"] - 0.75) < 0.01

    def test_evolve_playbook(self, skill):
        run(skill.execute("create_playbook", {
            "name": "pb1", "task_pattern": "test",
            "steps": ["step1", "step2", "step3"],
            "pitfalls": ["pitfall1"],
        }))
        result = run(skill.execute("evolve_playbook", {
            "playbook_name": "pb1",
            "add_steps": ["step4"],
            "remove_steps": [1],
            "add_pitfalls": ["pitfall2"],
            "update_pattern": "evolved test pattern",
        }))
        assert result.success
        pb = result.data["playbook"]
        assert pb["version"] == 2
        assert "step4" in pb["steps"]
        assert "step2" not in pb["steps"]
        assert len(pb["pitfalls"]) == 2


class TestPatterns:
    def test_extract_patterns(self, skill):
        # Create mix of successes and failures
        for i in range(10):
            run(skill.execute("reflect", {
                "task": f"Task {i}",
                "actions_taken": ["common_action", f"specific_{i}"],
                "outcome": f"Outcome {i}",
                "success": i >= 3,
                "analysis": f"Analysis {i}",
                "improvements": ["try harder"] if i < 3 else [],
                "tags": ["test_tag"],
            }))
        result = run(skill.execute("extract_patterns", {"lookback": 20}))
        assert result.success
        patterns = result.data["patterns"]
        assert patterns["summary"]["total_reflections"] == 10
        assert patterns["summary"]["successes"] == 7
        assert len(patterns["tag_patterns"]) >= 1

    def test_extract_patterns_with_filter(self, skill):
        run(skill.execute("reflect", {
            "task": "t1", "actions_taken": [], "outcome": "ok",
            "success": True, "analysis": "good", "tags": ["alpha"],
        }))
        run(skill.execute("reflect", {
            "task": "t1b", "actions_taken": [], "outcome": "ok2",
            "success": True, "analysis": "good2", "tags": ["alpha"],
        }))
        run(skill.execute("reflect", {
            "task": "t2", "actions_taken": [], "outcome": "fail",
            "success": False, "analysis": "bad", "tags": ["beta"],
        }))
        result = run(skill.execute("extract_patterns", {"filter_tag": "alpha"}))
        assert result.success
        assert result.data["patterns"]["summary"]["total_reflections"] == 2


class TestInsights:
    def test_add_insight(self, skill):
        result = run(skill.execute("add_insight", {
            "insight": "Batch deployments are more reliable than individual ones",
            "category": "reliability",
            "confidence": 0.8,
        }))
        assert result.success
        assert result.data["insight"]["confidence"] == 0.8

    def test_review_insights(self, skill):
        run(skill.execute("add_insight", {
            "insight": "Insight 1", "category": "perf",
        }))
        run(skill.execute("add_insight", {
            "insight": "Insight 2", "category": "cost",
        }))
        result = run(skill.execute("review", {"what": "insights"}))
        assert result.success
        assert len(result.data["insights"]) == 2


class TestReview:
    def test_review_all(self, skill):
        run(skill.execute("reflect", {
            "task": "t1", "actions_taken": [], "outcome": "ok",
            "success": True, "analysis": "good",
        }))
        run(skill.execute("create_playbook", {
            "name": "pb1", "task_pattern": "test", "steps": ["s1"],
        }))
        run(skill.execute("add_insight", {"insight": "test insight"}))
        result = run(skill.execute("review", {"what": "all"}))
        assert result.success
        assert "reflections" in result.data
        assert "playbooks" in result.data
        assert "insights" in result.data
        assert "stats" in result.data


class TestEdgeCases:
    def test_unknown_action(self, skill):
        result = run(skill.execute("nonexistent", {}))
        assert not result.success

    def test_record_use_nonexistent_playbook(self, skill):
        result = run(skill.execute("record_playbook_use", {
            "playbook_name": "ghost", "success": True,
        }))
        assert not result.success

    def test_evolve_nonexistent_playbook(self, skill):
        result = run(skill.execute("evolve_playbook", {
            "playbook_name": "ghost",
        }))
        assert not result.success

    def test_find_no_playbooks(self, skill):
        result = run(skill.execute("find_playbook", {
            "task_description": "something",
        }))
        assert result.success
        assert result.data["matches"] == []
