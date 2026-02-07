#!/usr/bin/env python3
"""Tests for AutonomousLoopSkill - central executive for autonomous operation."""

import pytest
import json
from pathlib import Path
from unittest.mock import patch, AsyncMock, MagicMock
from singularity.skills.autonomous_loop import AutonomousLoopSkill, LOOP_STATE_FILE, LoopPhase
from singularity.skills.base import SkillResult


@pytest.fixture
def skill(tmp_path):
    """Create an AutonomousLoopSkill with a temporary data path."""
    test_file = tmp_path / "autonomous_loop.json"
    with patch("singularity.skills.autonomous_loop.LOOP_STATE_FILE", test_file):
        s = AutonomousLoopSkill()
        yield s


@pytest.fixture
def mock_context():
    """Create a mock SkillContext with strategy and goal_manager responses."""
    ctx = MagicMock()

    async def mock_call_skill(skill_id, action, params=None):
        if skill_id == "strategy" and action == "assess":
            return SkillResult(
                success=True,
                message="Assessment complete",
                data={
                    "pillars": {
                        "self_improvement": {"score": 70, "capabilities": ["A"], "gaps": []},
                        "revenue": {"score": 30, "capabilities": [], "gaps": ["No billing"]},
                        "replication": {"score": 50, "capabilities": ["B"], "gaps": ["Coordination"]},
                        "goal_setting": {"score": 60, "capabilities": ["C"], "gaps": []},
                    },
                    "weakest_pillar": "revenue",
                    "strongest_pillar": "self_improvement",
                    "summary": "Revenue is weakest",
                }
            )
        elif skill_id == "goal_manager" and action == "next":
            return SkillResult(success=True, message="Next goal", data={
                "goal_id": "g1", "title": "Build billing", "pillar": "revenue", "priority": "high",
            })
        elif skill_id == "goal_manager" and action == "get":
            return SkillResult(success=True, message="Goal details", data={
                "milestones": [{"id": "m1", "title": "Add Stripe", "status": "pending"}],
            })
        elif skill_id == "outcome_tracker" and action == "log":
            return SkillResult(success=True, message="Logged")
        elif skill_id == "feedback_loop" and action == "analyze":
            return SkillResult(success=True, message="Analyzed", data={
                "adaptations": [{"type": "test"}], "patterns": [],
            })
        return SkillResult(success=False, message=f"Unknown: {skill_id}:{action}")

    ctx.call_skill = AsyncMock(side_effect=mock_call_skill)
    return ctx


@pytest.mark.asyncio
async def test_status_initial(skill, tmp_path):
    test_file = tmp_path / "autonomous_loop.json"
    with patch("singularity.skills.autonomous_loop.LOOP_STATE_FILE", test_file):
        result = await skill.execute("status", {})
        assert result.success
        assert result.data["phase"] == "idle"
        assert result.data["iteration_count"] == 0


@pytest.mark.asyncio
async def test_assess_no_context(skill, tmp_path):
    test_file = tmp_path / "autonomous_loop.json"
    with patch("singularity.skills.autonomous_loop.LOOP_STATE_FILE", test_file):
        result = await skill.execute("assess", {})
        assert result.success
        assert "weakest" in result.message.lower() or "assessment" in result.message.lower()


@pytest.mark.asyncio
async def test_assess_with_context(skill, mock_context, tmp_path):
    test_file = tmp_path / "autonomous_loop.json"
    with patch("singularity.skills.autonomous_loop.LOOP_STATE_FILE", test_file):
        skill.context = mock_context
        result = await skill.execute("assess", {})
        assert result.success
        assert result.data.get("weakest_pillar") == "revenue"


@pytest.mark.asyncio
async def test_decide_from_goal(skill, mock_context, tmp_path):
    test_file = tmp_path / "autonomous_loop.json"
    with patch("singularity.skills.autonomous_loop.LOOP_STATE_FILE", test_file):
        skill.context = mock_context
        # First assess
        await skill.execute("assess", {})
        # Then decide
        result = await skill.execute("decide", {})
        assert result.success
        assert "Build billing" in result.data.get("task_description", "")


@pytest.mark.asyncio
async def test_full_step(skill, mock_context, tmp_path):
    test_file = tmp_path / "autonomous_loop.json"
    with patch("singularity.skills.autonomous_loop.LOOP_STATE_FILE", test_file):
        skill.context = mock_context
        result = await skill.execute("step", {})
        assert result.success
        assert result.data.get("outcome") in ("success", "partial")
        assert result.data.get("pillar") == "revenue"
        # Verify stats updated
        status = await skill.execute("status", {})
        assert status.data["stats"]["total_iterations"] == 1


@pytest.mark.asyncio
async def test_journal_after_step(skill, mock_context, tmp_path):
    test_file = tmp_path / "autonomous_loop.json"
    with patch("singularity.skills.autonomous_loop.LOOP_STATE_FILE", test_file):
        skill.context = mock_context
        await skill.execute("step", {})
        result = await skill.execute("journal", {"limit": 5})
        assert result.success
        assert result.data["total_entries"] == 1
        assert len(result.data["entries"]) == 1


@pytest.mark.asyncio
async def test_run_multiple_iterations(skill, mock_context, tmp_path):
    test_file = tmp_path / "autonomous_loop.json"
    with patch("singularity.skills.autonomous_loop.LOOP_STATE_FILE", test_file):
        skill.context = mock_context
        result = await skill.execute("run", {"iterations": 2})
        assert result.success
        assert result.data["iterations_completed"] == 2


@pytest.mark.asyncio
async def test_configure(skill, tmp_path):
    test_file = tmp_path / "autonomous_loop.json"
    with patch("singularity.skills.autonomous_loop.LOOP_STATE_FILE", test_file):
        result = await skill.execute("configure", {
            "auto_learn": False, "max_iterations": 5,
        })
        assert result.success
        status = await skill.execute("status", {})
        assert status.data["config"]["auto_learn"] is False
        assert status.data["config"]["max_iterations"] == 5


@pytest.mark.asyncio
async def test_reset(skill, mock_context, tmp_path):
    test_file = tmp_path / "autonomous_loop.json"
    with patch("singularity.skills.autonomous_loop.LOOP_STATE_FILE", test_file):
        skill.context = mock_context
        await skill.execute("step", {})
        result = await skill.execute("reset", {})
        assert result.success
        status = await skill.execute("status", {})
        assert status.data["phase"] == "idle"
        assert status.data["iteration_count"] == 0


@pytest.mark.asyncio
async def test_decide_without_assessment(skill, tmp_path):
    test_file = tmp_path / "autonomous_loop.json"
    with patch("singularity.skills.autonomous_loop.LOOP_STATE_FILE", test_file):
        result = await skill.execute("decide", {})
        assert not result.success
        assert "assess" in result.message.lower()


@pytest.mark.asyncio
async def test_unknown_action(skill, tmp_path):
    test_file = tmp_path / "autonomous_loop.json"
    with patch("singularity.skills.autonomous_loop.LOOP_STATE_FILE", test_file):
        result = await skill.execute("nonexistent", {})
        assert not result.success
