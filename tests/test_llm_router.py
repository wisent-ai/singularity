#!/usr/bin/env python3
"""Tests for CostAwareLLMRouter skill."""

import json
import pytest
import time
from pathlib import Path
from unittest.mock import patch

from singularity.skills.llm_router import (
    CostAwareLLMRouter,
    MODEL_TIERS,
    COMPLEXITY_SIGNALS,
    ROUTER_FILE,
)


@pytest.fixture
def router(tmp_path):
    """Create a router with temporary storage."""
    with patch("singularity.skills.llm_router.ROUTER_FILE", tmp_path / "router.json"):
        with patch("singularity.skills.llm_router.DATA_DIR", tmp_path):
            r = CostAwareLLMRouter()
            yield r


@pytest.mark.asyncio
async def test_route_simple_task(router):
    result = await router.execute("route", {"task": "List all files in the directory"})
    assert result.success
    assert result.data["complexity"]["classification"] == "simple"
    assert result.data["recommendation"]["tier"] == "budget"


@pytest.mark.asyncio
async def test_route_complex_task(router):
    result = await router.execute("route", {
        "task": "Design and architect a multi-step deployment strategy with security audit"
    })
    assert result.success
    assert result.data["complexity"]["classification"] == "complex"
    assert result.data["recommendation"]["tier"] == "premium"


@pytest.mark.asyncio
async def test_route_medium_task(router):
    result = await router.execute("route", {
        "task": "Analyze the performance of this function and suggest improvements"
    })
    assert result.success
    assert result.data["complexity"]["classification"] == "medium"


@pytest.mark.asyncio
async def test_route_returns_savings_estimate(router):
    result = await router.execute("route", {"task": "Format this JSON data"})
    assert result.success
    assert "cost_estimate" in result.data
    assert result.data["cost_estimate"]["savings_percent"] >= 0


@pytest.mark.asyncio
async def test_route_best_quality_forces_premium(router):
    result = await router.execute("route", {
        "task": "Summarize this text",
        "required_quality": "best",
    })
    assert result.success
    assert result.data["recommendation"]["tier"] == "premium"


@pytest.mark.asyncio
async def test_record_outcome(router):
    # First route a task
    route_result = await router.execute("route", {"task": "Extract names from text"})
    routing_id = route_result.data["routing_id"]

    # Record outcome
    result = await router.execute("record_outcome", {
        "routing_id": routing_id,
        "success": True,
        "quality_score": 0.9,
    })
    assert result.success
    assert result.data["outcome_success"] is True
    assert result.data["model_stats"]["total"] == 1


@pytest.mark.asyncio
async def test_record_outcome_updates_performance(router):
    # Route and record multiple outcomes
    for i in range(3):
        route = await router.execute("route", {"task": f"Convert format #{i}"})
        rid = route.data["routing_id"]
        await router.execute("record_outcome", {
            "routing_id": rid,
            "success": i < 2,  # 2 success, 1 failure
        })

    # Check leaderboard
    result = await router.execute("model_leaderboard", {})
    assert result.success
    assert len(result.data["leaderboard"]) > 0


@pytest.mark.asyncio
async def test_savings_report_empty(router):
    result = await router.execute("savings_report", {})
    assert result.success
    assert result.data["total_routed"] == 0


@pytest.mark.asyncio
async def test_savings_report_with_data(router):
    route = await router.execute("route", {"task": "Translate this text"})
    rid = route.data["routing_id"]
    await router.execute("record_outcome", {"routing_id": rid, "success": True})

    result = await router.execute("savings_report", {})
    assert result.success
    assert result.data["total_routed"] == 1
    assert result.data["total_completed"] == 1


@pytest.mark.asyncio
async def test_set_budget(router):
    result = await router.execute("set_budget", {"limit_usd": 0.50, "reset_period": True})
    assert result.success
    assert result.data["budget_limit"] == 0.50
    assert result.data["budget_mode"] is True


@pytest.mark.asyncio
async def test_status(router):
    result = await router.execute("status", {})
    assert result.success
    assert "tiers_available" in result.data
    assert set(result.data["tiers_available"]) == {"budget", "standard", "premium"}


@pytest.mark.asyncio
async def test_override(router):
    result = await router.execute("override", {"provider": "anthropic", "model": "claude-3-opus"})
    assert result.success
    assert result.data["provider"] == "anthropic"


@pytest.mark.asyncio
async def test_unknown_action(router):
    result = await router.execute("nonexistent", {})
    assert not result.success


@pytest.mark.asyncio
async def test_route_empty_task(router):
    result = await router.execute("route", {"task": ""})
    assert not result.success


@pytest.mark.asyncio
async def test_record_outcome_invalid_id(router):
    result = await router.execute("record_outcome", {
        "routing_id": "nonexistent",
        "success": True,
    })
    assert not result.success


def test_classify_complexity(router):
    # Simple
    c = router.classify_complexity("List all items")
    assert c["classification"] == "simple"

    # Complex
    c = router.classify_complexity(
        "Architect a multi-step strategy to refactor and optimize the deployment pipeline"
    )
    assert c["classification"] == "complex"


def test_model_tiers_structure():
    """Verify MODEL_TIERS has valid structure."""
    for tier, config in MODEL_TIERS.items():
        assert "models" in config
        assert len(config["models"]) > 0
        for m in config["models"]:
            assert "provider" in m
            assert "model" in m
            assert m["cost_per_1m_input"] >= 0
