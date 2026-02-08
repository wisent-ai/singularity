"""Tests for LearningDistillationSkill."""
import json
import pytest
from pathlib import Path
from unittest.mock import patch
from singularity.skills.learning_distillation import (
    LearningDistillationSkill,
    RuleCategory,
    RULES_FILE,
    VALID_CATEGORIES,
)


@pytest.fixture
def skill(tmp_path):
    with patch("singularity.skills.learning_distillation.DATA_DIR", tmp_path):
        with patch("singularity.skills.learning_distillation.RULES_FILE", tmp_path / "learning_rules.json"):
            s = LearningDistillationSkill()
            yield s


@pytest.fixture
def outcomes_file(tmp_path):
    return tmp_path / "outcomes.json"


@pytest.fixture
def feedback_file(tmp_path):
    return tmp_path / "feedback_loop.json"


@pytest.fixture
def experiments_file(tmp_path):
    return tmp_path / "experiments.json"


@pytest.fixture
def profiler_file(tmp_path):
    return tmp_path / "skill_profiler.json"


def write_json(path: Path, data):
    with open(path, "w") as f:
        json.dump(data, f)


@pytest.mark.asyncio
async def test_manifest(skill):
    m = skill.manifest
    assert m.skill_id == "learning_distillation"
    assert len(m.actions) == 8
    action_names = [a.name for a in m.actions]
    assert "distill" in action_names
    assert "query" in action_names
    assert "add_rule" in action_names
    assert "reinforce" in action_names
    assert "weaken" in action_names


@pytest.mark.asyncio
async def test_add_rule(skill):
    result = await skill.execute("add_rule", {
        "rule_text": "Always prefer skill_A for fast tasks",
        "category": "skill_preference",
        "confidence": 0.8,
        "skill_id": "skill_A",
    })
    assert result.success
    assert "rule_id" in result.data


@pytest.mark.asyncio
async def test_add_rule_missing_text(skill):
    result = await skill.execute("add_rule", {"rule_text": ""})
    assert not result.success


@pytest.mark.asyncio
async def test_add_rule_invalid_category(skill):
    result = await skill.execute("add_rule", {
        "rule_text": "some rule",
        "category": "nonexistent_category",
    })
    assert not result.success


@pytest.mark.asyncio
async def test_query_empty(skill):
    result = await skill.execute("query", {})
    assert result.success
    assert result.data["total_matching"] == 0


@pytest.mark.asyncio
async def test_query_after_add(skill):
    await skill.execute("add_rule", {
        "rule_text": "Rule one",
        "category": "general",
        "confidence": 0.9,
        "skill_id": "s1",
    })
    await skill.execute("add_rule", {
        "rule_text": "Rule two",
        "category": "failure_pattern",
        "confidence": 0.3,
        "skill_id": "s2",
    })
    # Query all with min confidence 0.5
    result = await skill.execute("query", {"min_confidence": 0.5})
    assert result.success
    assert result.data["total_matching"] == 1

    # Query by skill_id
    result2 = await skill.execute("query", {"skill_id": "s2", "min_confidence": 0.0})
    assert result2.data["total_matching"] == 1

    # Query by category
    result3 = await skill.execute("query", {"category": "failure_pattern", "min_confidence": 0.0})
    assert result3.data["total_matching"] == 1


@pytest.mark.asyncio
async def test_reinforce(skill):
    r = await skill.execute("add_rule", {
        "rule_text": "Test rule",
        "confidence": 0.5,
    })
    rule_id = r.data["rule_id"]
    result = await skill.execute("reinforce", {"rule_id": rule_id})
    assert result.success
    assert result.data["new_confidence"] > 0.5


@pytest.mark.asyncio
async def test_reinforce_not_found(skill):
    result = await skill.execute("reinforce", {"rule_id": "nonexistent"})
    assert not result.success


@pytest.mark.asyncio
async def test_weaken(skill):
    r = await skill.execute("add_rule", {
        "rule_text": "Test rule",
        "confidence": 0.8,
    })
    rule_id = r.data["rule_id"]
    result = await skill.execute("weaken", {"rule_id": rule_id})
    assert result.success
    assert result.data["new_confidence"] < 0.8


@pytest.mark.asyncio
async def test_status(skill):
    await skill.execute("add_rule", {"rule_text": "R1", "category": "general", "confidence": 0.9})
    await skill.execute("add_rule", {"rule_text": "R2", "category": "failure_pattern", "confidence": 0.3})
    result = await skill.execute("status", {})
    assert result.success
    assert result.data["total_rules"] == 2
    assert "general" in result.data["by_category"]


@pytest.mark.asyncio
async def test_configure(skill):
    result = await skill.execute("configure", {"min_samples": 10, "high_success_threshold": 0.9})
    assert result.success
    assert result.data["changes"]["min_samples"]["new"] == 10


@pytest.mark.asyncio
async def test_configure_no_params(skill):
    result = await skill.execute("configure", {})
    assert not result.success


@pytest.mark.asyncio
async def test_distill_from_outcomes(skill, tmp_path):
    # Create outcome data with clear patterns
    outcomes = {
        "outcomes": [
            {"tool": "skill_a:action1", "success": True, "cost": 0.01}
            for _ in range(10)
        ] + [
            {"tool": "skill_b:action2", "success": False, "cost": 0.05, "error": "timeout"}
            for _ in range(10)
        ],
        "metadata": {},
    }
    write_json(tmp_path / "outcomes.json", outcomes)
    result = await skill.execute("distill", {"sources": "outcomes"})
    assert result.success
    assert result.data["new_rules"] > [] if isinstance(result.data["new_rules"], list) else True
    # Should have created rules for high-success skill_a and low-success skill_b
    rules = result.data["new_rules"]
    categories = [r["category"] for r in rules]
    assert "success_pattern" in categories or "failure_pattern" in categories


@pytest.mark.asyncio
async def test_distill_from_feedback(skill, tmp_path):
    feedback = {
        "adaptations": [
            {"description": "Use caching for API calls", "status": "applied",
             "outcome": "positive", "type": "performance"},
            {"description": "Increase retry count", "status": "applied",
             "outcome": "negative", "type": "reliability"},
        ],
        "reviews": [],
    }
    write_json(tmp_path / "feedback_loop.json", feedback)
    result = await skill.execute("distill", {"sources": "feedback"})
    assert result.success
    assert len(result.data["new_rules"]) == 2


@pytest.mark.asyncio
async def test_distill_from_experiments(skill, tmp_path):
    experiments = {
        "experiments": [{
            "name": "prompt_style_test",
            "hypothesis": "Concise prompts work better than verbose",
            "status": "concluded",
            "conclusion": {"winner": "concise"},
            "variants": {
                "concise": {"trials": 50, "successes": 40},
                "verbose": {"trials": 50, "successes": 25},
            },
        }],
    }
    write_json(tmp_path / "experiments.json", experiments)
    result = await skill.execute("distill", {"sources": "experiments"})
    assert result.success
    assert len(result.data["new_rules"]) == 1
    assert "concise" in result.data["new_rules"][0]["text"]


@pytest.mark.asyncio
async def test_distill_no_data(skill):
    result = await skill.execute("distill", {})
    assert result.success
    assert result.data["new_rules"] == []


@pytest.mark.asyncio
async def test_expire_old_rules(skill):
    # Add a low-confidence rule with old timestamp
    r = await skill.execute("add_rule", {"rule_text": "Old rule", "confidence": 0.2})
    rule_id = r.data["rule_id"]

    # Manually set the created_at and last_reinforced to 60 days ago
    data = skill._load()
    for rule in data["rules"]:
        if rule["id"] == rule_id:
            from datetime import datetime, timedelta
            old_date = (datetime.now() - timedelta(days=60)).isoformat()
            rule["created_at"] = old_date
            rule["last_reinforced"] = old_date
    skill._save(data)

    result = await skill.execute("expire", {})
    assert result.success
    assert result.data["expired"] == 1
    assert result.data["remaining"] == 0


@pytest.mark.asyncio
async def test_unknown_action(skill):
    result = await skill.execute("nonexistent", {})
    assert not result.success
