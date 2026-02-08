#!/usr/bin/env python3
"""Tests for DecisionLogSkill."""
import json
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

from singularity.skills.decision_log import DecisionLogSkill, DECISION_LOG_FILE


@pytest.fixture
def skill(tmp_path):
    """Create a DecisionLogSkill with isolated storage."""
    test_file = tmp_path / "decision_log.json"
    with patch("singularity.skills.decision_log.DECISION_LOG_FILE", test_file):
        s = DecisionLogSkill()
        yield s


@pytest.mark.asyncio
async def test_log_decision(skill):
    result = await skill.execute("log_decision", {
        "category": "skill_selection",
        "context": "Need to process CSV data",
        "choice": "Use DataTransformSkill",
        "alternatives": ["Write custom parser", "Use pandas"],
        "reasoning": "DataTransformSkill has built-in CSV support with zero deps",
        "confidence": "high",
        "tags": ["data", "csv"],
    })
    assert result.success
    assert result.data["decision_id"]
    assert result.data["category"] == "skill_selection"
    assert result.data["confidence"] == 0.7


@pytest.mark.asyncio
async def test_log_decision_validation(skill):
    # Missing required fields
    result = await skill.execute("log_decision", {"category": "strategy"})
    assert not result.success

    # Invalid category
    result = await skill.execute("log_decision", {
        "category": "invalid_cat",
        "context": "test", "choice": "test", "reasoning": "test",
    })
    assert not result.success


@pytest.mark.asyncio
async def test_record_outcome(skill):
    # Log a decision first
    log_result = await skill.execute("log_decision", {
        "category": "prioritization",
        "context": "Choose next task",
        "choice": "Work on revenue",
        "reasoning": "Revenue pillar is weakest",
        "confidence": "medium",
    })
    decision_id = log_result.data["decision_id"]

    # Record outcome
    result = await skill.execute("record_outcome", {
        "decision_id": decision_id,
        "success": True,
        "details": "Revenue skill shipped successfully",
        "impact_score": 0.8,
    })
    assert result.success
    assert result.data["success"] is True

    # Can't record twice
    result2 = await skill.execute("record_outcome", {
        "decision_id": decision_id, "success": False,
    })
    assert not result2.success


@pytest.mark.asyncio
async def test_query_decisions(skill):
    # Log several decisions
    for i in range(5):
        cat = "strategy" if i % 2 == 0 else "skill_selection"
        await skill.execute("log_decision", {
            "category": cat,
            "context": f"Context {i}",
            "choice": f"Choice {i}",
            "reasoning": f"Reasoning {i}",
            "tags": ["test"],
        })

    # Query all
    result = await skill.execute("query", {"limit": 10})
    assert result.success
    assert result.data["total_matching"] == 5

    # Query by category
    result = await skill.execute("query", {"category": "strategy"})
    assert result.data["total_matching"] == 3

    # Query by tag
    result = await skill.execute("query", {"tag": "test"})
    assert result.data["total_matching"] == 5


@pytest.mark.asyncio
async def test_analyze(skill):
    # Log decisions with outcomes
    ids = []
    for i in range(6):
        r = await skill.execute("log_decision", {
            "category": "strategy",
            "context": f"Situation {i}",
            "choice": f"Action {i}",
            "reasoning": f"Because {i}",
            "confidence": "high" if i < 4 else "low",
        })
        ids.append(r.data["decision_id"])

    # Record outcomes
    for i, did in enumerate(ids):
        await skill.execute("record_outcome", {
            "decision_id": did,
            "success": i < 4,  # 4 successes, 2 failures
            "impact_score": 0.5,
        })

    result = await skill.execute("analyze", {})
    assert result.success
    assert result.data["total_decisions"] == 6
    assert result.data["with_outcomes"] == 6
    assert result.data["accuracy"] == pytest.approx(0.667, abs=0.01)
    assert isinstance(result.data["insights"], list)


@pytest.mark.asyncio
async def test_recommend(skill):
    # Build up decision history
    r1 = await skill.execute("log_decision", {
        "category": "skill_selection",
        "context": "Need to parse CSV data file",
        "choice": "Use DataTransformSkill",
        "reasoning": "Built-in CSV support",
    })
    await skill.execute("record_outcome", {
        "decision_id": r1.data["decision_id"],
        "success": True,
    })

    r2 = await skill.execute("log_decision", {
        "category": "skill_selection",
        "context": "Need to transform JSON data",
        "choice": "Use DataTransformSkill",
        "reasoning": "Handles JSON natively",
    })
    await skill.execute("record_outcome", {
        "decision_id": r2.data["decision_id"],
        "success": True,
    })

    # Now ask for recommendation
    result = await skill.execute("recommend", {
        "context": "I need to process a CSV data file",
    })
    assert result.success
    assert len(result.data["recommendations"]) > 0
    assert result.data["recommendations"][0]["choice"] == "Use DataTransformSkill"


@pytest.mark.asyncio
async def test_playbook_generate(skill):
    # Build history
    for i in range(4):
        r = await skill.execute("log_decision", {
            "category": "strategy",
            "context": f"Low revenue situation {i}",
            "choice": "Focus on revenue pillar",
            "reasoning": "Revenue is weakest",
        })
        await skill.execute("record_outcome", {
            "decision_id": r.data["decision_id"],
            "success": True,
        })

    result = await skill.execute("playbook", {"action": "generate"})
    assert result.success
    assert len(result.data["playbook"]) >= 1
    assert result.data["playbook"][0]["recommended_choice"] == "Focus on revenue pillar"


@pytest.mark.asyncio
async def test_stats(skill):
    await skill.execute("log_decision", {
        "category": "error_handling",
        "context": "API timeout",
        "choice": "Retry with backoff",
        "reasoning": "Transient error likely",
    })
    result = await skill.execute("stats", {})
    assert result.success
    assert result.data["total_decisions"] == 1
    assert result.data["pending_outcomes"] == 1


@pytest.mark.asyncio
async def test_unknown_action(skill):
    result = await skill.execute("nonexistent", {})
    assert not result.success
