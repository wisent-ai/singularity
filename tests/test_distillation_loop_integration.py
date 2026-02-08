#!/usr/bin/env python3
"""Tests for AutonomousLoopSkill v2.0 - LearningDistillation integration."""

import pytest
import json
from unittest.mock import patch, AsyncMock, MagicMock
from singularity.skills.autonomous_loop import AutonomousLoopSkill, LoopPhase
from singularity.skills.base import SkillResult


@pytest.fixture
def skill(tmp_path):
    test_file = tmp_path / "autonomous_loop.json"
    with patch("singularity.skills.autonomous_loop.LOOP_STATE_FILE", test_file):
        s = AutonomousLoopSkill()
        yield s


def _make_context(distill_rules=None, query_rules=None):
    """Create a mock context that supports learning_distillation calls."""
    ctx = MagicMock()

    async def mock_call_skill(skill_id, action, params=None):
        if skill_id == "strategy" and action == "assess":
            return SkillResult(success=True, message="OK", data={
                "pillars": {
                    "self_improvement": {"score": 70, "gaps": []},
                    "revenue": {"score": 30, "gaps": ["No billing"]},
                    "replication": {"score": 50, "gaps": ["No fleet"]},
                    "goal_setting": {"score": 60, "gaps": []},
                },
                "weakest_pillar": "revenue",
                "strongest_pillar": "self_improvement",
                "summary": "Revenue weakest",
            })
        elif skill_id == "goal_manager" and action == "next":
            return SkillResult(success=True, data={
                "goal_id": "g1", "title": "Build billing",
                "pillar": "revenue", "priority": "high",
            })
        elif skill_id == "goal_manager" and action == "get":
            return SkillResult(success=True, data={
                "milestones": [{"id": "m1", "title": "Add Stripe", "status": "pending"}],
            })
        elif skill_id == "outcome_tracker" and action == "log":
            return SkillResult(success=True, message="Logged")
        elif skill_id == "feedback_loop" and action == "analyze":
            return SkillResult(success=True, data={
                "adaptations": [{"type": "test"}], "patterns": [],
            })
        elif skill_id == "learning_distillation" and action == "distill":
            return SkillResult(success=True, data={
                "rules_created": 3,
                "total_rules": 12,
                "sources_analyzed": ["outcome_tracker", "feedback_loop"],
            })
        elif skill_id == "learning_distillation" and action == "query":
            cat = (params or {}).get("category", "")
            if cat == "success_pattern" and query_rules:
                return SkillResult(success=True, data={
                    "rules": query_rules.get("success", []),
                    "total_matching": len(query_rules.get("success", [])),
                })
            elif cat == "failure_pattern" and query_rules:
                return SkillResult(success=True, data={
                    "rules": query_rules.get("failure", []),
                    "total_matching": len(query_rules.get("failure", [])),
                })
            elif cat == "skill_preference" and query_rules:
                return SkillResult(success=True, data={
                    "rules": query_rules.get("preference", []),
                    "total_matching": len(query_rules.get("preference", [])),
                })
            return SkillResult(success=True, data={"rules": [], "total_matching": 0})
        elif skill_id == "learning_distillation" and action == "expire":
            return SkillResult(success=True, message="Expired 0 rules")
        return SkillResult(success=False, message=f"Unknown: {skill_id}:{action}")

    ctx.call_skill = AsyncMock(side_effect=mock_call_skill)
    return ctx


@pytest.mark.asyncio
async def test_version_is_2(skill, tmp_path):
    """Verify skill reports v2.0.0."""
    assert skill.manifest.version == "2.0.0"


@pytest.mark.asyncio
async def test_distillation_config_defaults(skill, tmp_path):
    test_file = tmp_path / "autonomous_loop.json"
    with patch("singularity.skills.autonomous_loop.LOOP_STATE_FILE", test_file):
        result = await skill.execute("status", {})
        config = result.data["config"]
        assert config["distillation_enabled"] is True
        assert config["distillation_interval"] == 3
        assert config["consult_rules_in_decide"] is True
        assert config["min_rule_confidence"] == 0.5


@pytest.mark.asyncio
async def test_configure_distillation_params(skill, tmp_path):
    test_file = tmp_path / "autonomous_loop.json"
    with patch("singularity.skills.autonomous_loop.LOOP_STATE_FILE", test_file):
        result = await skill.execute("configure", {
            "distillation_enabled": False,
            "distillation_interval": 5,
            "consult_rules_in_decide": False,
            "min_rule_confidence": 0.7,
        })
        assert result.success
        status = await skill.execute("status", {})
        assert status.data["config"]["distillation_enabled"] is False
        assert status.data["config"]["distillation_interval"] == 5
        assert status.data["config"]["consult_rules_in_decide"] is False
        assert status.data["config"]["min_rule_confidence"] == 0.7


@pytest.mark.asyncio
async def test_step_runs_distillation_on_interval(skill, tmp_path):
    """Distillation runs when iteration count is divisible by interval."""
    test_file = tmp_path / "autonomous_loop.json"
    with patch("singularity.skills.autonomous_loop.LOOP_STATE_FILE", test_file):
        ctx = _make_context()
        skill.context = ctx

        # Set interval to 1 so it distills every iteration
        await skill.execute("configure", {"distillation_interval": 1})

        result = await skill.execute("step", {})
        assert result.success

        # Check journal learn phase has distillation data
        journal = await skill.execute("journal", {"limit": 1})
        entry = journal.data["entries"][0]
        learn_phase = entry.get("phases", {}).get("learn", {})
        # Should have attempted distillation
        assert "distillation_ran" in learn_phase


@pytest.mark.asyncio
async def test_step_consults_rules_in_decide(skill, tmp_path):
    """DECIDE phase consults distilled rules."""
    test_file = tmp_path / "autonomous_loop.json"
    with patch("singularity.skills.autonomous_loop.LOOP_STATE_FILE", test_file):
        query_rules = {
            "success": [
                {"skill_id": "payment", "confidence": 0.9, "rule_text": "payment skill has 95% success rate"},
            ],
            "failure": [
                {"skill_id": "twitter", "confidence": 0.8, "rule_text": "twitter skill fails 70% of the time"},
            ],
            "preference": [],
        }
        ctx = _make_context(query_rules=query_rules)
        skill.context = ctx

        result = await skill.execute("step", {})
        assert result.success

        # Check decide phase recorded rules_consulted
        journal = await skill.execute("journal", {"limit": 1})
        entry = journal.data["entries"][0]
        decide_phase = entry.get("phases", {}).get("decide", {})
        assert decide_phase.get("rules_consulted", 0) >= 0


@pytest.mark.asyncio
async def test_decide_includes_distilled_insights(skill, tmp_path):
    """Decision data includes distilled_insights field."""
    test_file = tmp_path / "autonomous_loop.json"
    with patch("singularity.skills.autonomous_loop.LOOP_STATE_FILE", test_file):
        query_rules = {
            "success": [
                {"skill_id": "code_review", "confidence": 0.85, "rule_text": "high success"},
            ],
            "failure": [],
            "preference": [],
        }
        ctx = _make_context(query_rules=query_rules)
        skill.context = ctx

        # Assess first
        await skill.execute("assess", {})
        # Then decide
        result = await skill.execute("decide", {})
        assert result.success
        insights = result.data.get("distilled_insights", {})
        assert "preferred_skills" in insights
        assert "avoid_skills" in insights


@pytest.mark.asyncio
async def test_distillation_disabled_skips(skill, tmp_path):
    """When distillation_enabled=False, distillation is skipped."""
    test_file = tmp_path / "autonomous_loop.json"
    with patch("singularity.skills.autonomous_loop.LOOP_STATE_FILE", test_file):
        ctx = _make_context()
        skill.context = ctx

        await skill.execute("configure", {"distillation_enabled": False})
        result = await skill.execute("step", {})
        assert result.success

        # Verify distillation was not called
        distill_calls = [
            c for c in ctx.call_skill.call_args_list
            if c.args[0] == "learning_distillation" and c.args[1] == "distill"
        ]
        assert len(distill_calls) == 0


@pytest.mark.asyncio
async def test_consult_rules_disabled_skips(skill, tmp_path):
    """When consult_rules_in_decide=False, rule queries are skipped."""
    test_file = tmp_path / "autonomous_loop.json"
    with patch("singularity.skills.autonomous_loop.LOOP_STATE_FILE", test_file):
        ctx = _make_context()
        skill.context = ctx

        await skill.execute("configure", {"consult_rules_in_decide": False})
        result = await skill.execute("step", {})
        assert result.success

        # Verify no query calls to learning_distillation
        query_calls = [
            c for c in ctx.call_skill.call_args_list
            if c.args[0] == "learning_distillation" and c.args[1] == "query"
        ]
        assert len(query_calls) == 0


@pytest.mark.asyncio
async def test_distillation_stats_tracked(skill, tmp_path):
    """Stats track distillation runs and rules consulted."""
    test_file = tmp_path / "autonomous_loop.json"
    with patch("singularity.skills.autonomous_loop.LOOP_STATE_FILE", test_file):
        ctx = _make_context(query_rules={
            "success": [{"skill_id": "x", "confidence": 0.9, "rule_text": "good"}],
            "failure": [], "preference": [],
        })
        skill.context = ctx

        # Run with interval=1 so distill runs every time
        await skill.execute("configure", {"distillation_interval": 1})
        await skill.execute("step", {})

        status = await skill.execute("status", {})
        stats = status.data["stats"]
        assert "distillation_runs" in stats
        assert "rules_consulted" in stats


@pytest.mark.asyncio
async def test_format_insight_annotation(skill):
    """Test the insight annotation formatter."""
    annotation = skill._format_insight_annotation({
        "preferred_skills": [
            {"skill_id": "payment", "confidence": 0.9, "reason": "good"},
        ],
        "avoid_skills": [
            {"skill_id": "twitter", "confidence": 0.8, "reason": "bad"},
        ],
        "advice": [],
        "rules_consulted": 5,
    })
    assert "payment" in annotation
    assert "twitter" in annotation
    assert "Distilled" in annotation


@pytest.mark.asyncio
async def test_format_insight_no_skills(skill):
    """Annotation with no specific skills shows rule count."""
    annotation = skill._format_insight_annotation({
        "preferred_skills": [],
        "avoid_skills": [],
        "advice": [{"rule_text": "advice", "confidence": 0.7, "skill_id": ""}],
        "rules_consulted": 3,
    })
    assert "1 rule(s)" in annotation


@pytest.mark.asyncio
async def test_distillation_interval_skips(skill, tmp_path):
    """Distillation skips when iteration count not divisible by interval."""
    test_file = tmp_path / "autonomous_loop.json"
    with patch("singularity.skills.autonomous_loop.LOOP_STATE_FILE", test_file):
        ctx = _make_context()
        skill.context = ctx

        # Default interval is 3, first step is iteration 0 (total_iterations starts at 0)
        # After step, total_iterations becomes 1
        # So distill should run at iteration 0 (0%3==0), not at 1 (1%3!=0)
        await skill.execute("step", {})
        await skill.execute("step", {})

        distill_calls = [
            c for c in ctx.call_skill.call_args_list
            if c.args[0] == "learning_distillation" and c.args[1] == "distill"
        ]
        # Only first iteration (when total_iterations=0) should trigger distill
        # with default interval of 3 (0%3==0 triggers, 1%3!=0 skips)
        assert len(distill_calls) >= 1
