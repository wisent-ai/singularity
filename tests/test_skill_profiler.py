"""Tests for SkillPerformanceProfiler."""

import pytest
import json
from pathlib import Path
from unittest.mock import patch
from singularity.skills.skill_profiler import SkillPerformanceProfiler, PROFILER_FILE


@pytest.fixture
def profiler(tmp_path):
    """Create a profiler with tmp storage."""
    test_file = tmp_path / "skill_profiler.json"
    with patch("singularity.skills.skill_profiler.PROFILER_FILE", test_file):
        p = SkillPerformanceProfiler()
        p.set_installed_skills(["memory", "shell", "github", "content", "payment"])
        yield p


@pytest.mark.asyncio
async def test_record_event(profiler):
    result = await profiler.execute("record", {
        "skill_id": "memory", "action": "save", "success": True,
        "latency_ms": 50, "cost_usd": 0.001, "revenue_usd": 0,
    })
    assert result.success
    assert result.data["skill_id"] == "memory"


@pytest.mark.asyncio
async def test_record_requires_params(profiler):
    result = await profiler.execute("record", {"skill_id": ""})
    assert not result.success


@pytest.mark.asyncio
async def test_profile_empty(profiler):
    result = await profiler.execute("profile", {})
    assert result.success
    assert result.data["summary"]["skills_installed"] == 5
    assert result.data["summary"]["skills_used"] == 0


@pytest.mark.asyncio
async def test_profile_with_events(profiler):
    for i in range(5):
        await profiler.execute("record", {
            "skill_id": "memory", "action": "save", "success": True,
            "cost_usd": 0.01,
        })
    await profiler.execute("record", {
        "skill_id": "shell", "action": "run", "success": False,
    })
    result = await profiler.execute("profile", {})
    assert result.success
    assert result.data["summary"]["skills_used"] == 2
    assert result.data["summary"]["total_calls"] == 6
    skills = result.data["skills"]
    mem = next(s for s in skills if s["skill_id"] == "memory")
    assert mem["total_calls"] == 5
    assert mem["success_rate"] == 1.0


@pytest.mark.asyncio
async def test_unused_skills(profiler):
    await profiler.execute("record", {
        "skill_id": "memory", "action": "save", "success": True,
    })
    result = await profiler.execute("unused", {})
    assert result.success
    assert "memory" not in result.data["unused_skills"]
    assert "shell" in result.data["unused_skills"]
    assert len(result.data["unused_skills"]) == 4


@pytest.mark.asyncio
async def test_roi_ranking(profiler):
    await profiler.execute("record", {
        "skill_id": "payment", "action": "charge", "success": True,
        "cost_usd": 0.01, "revenue_usd": 5.0,
    })
    await profiler.execute("record", {
        "skill_id": "content", "action": "write", "success": True,
        "cost_usd": 0.05, "revenue_usd": 0.0,
    })
    result = await profiler.execute("roi", {"top_n": 5})
    assert result.success
    rankings = result.data["rankings"]
    assert rankings[0]["skill_id"] == "payment"
    assert rankings[0]["net_usd"] > 0


@pytest.mark.asyncio
async def test_recommend(profiler):
    # Create some mixed data
    for _ in range(5):
        await profiler.execute("record", {
            "skill_id": "shell", "action": "run", "success": False,
        })
    await profiler.execute("record", {
        "skill_id": "memory", "action": "save", "success": True,
    })
    result = await profiler.execute("recommend", {})
    assert result.success
    recs = result.data["recommendations"]
    # Should recommend fixing shell (>50% failure)
    fix_recs = [r for r in recs if r["type"] == "fix_or_disable"]
    assert any(r["skill"] == "shell" for r in fix_recs)
    # Should recommend pruning unused
    prune_recs = [r for r in recs if r["type"] == "prune"]
    assert len(prune_recs) > 0
    assert result.data["health_score"] >= 0


@pytest.mark.asyncio
async def test_trends_empty(profiler):
    result = await profiler.execute("trends", {})
    assert result.success
    assert result.data["total_events"] == 0


@pytest.mark.asyncio
async def test_trends_with_data(profiler):
    for _ in range(3):
        await profiler.execute("record", {
            "skill_id": "memory", "action": "save", "success": True,
        })
    result = await profiler.execute("trends", {"sessions": 5})
    assert result.success
    assert len(result.data["trends"]) >= 1


@pytest.mark.asyncio
async def test_budget(profiler):
    await profiler.execute("record", {
        "skill_id": "content", "action": "write", "success": True,
        "cost_usd": 0.10, "revenue_usd": 1.00,
    })
    await profiler.execute("record", {
        "skill_id": "shell", "action": "run", "success": True,
        "cost_usd": 0.01,
    })
    result = await profiler.execute("budget", {})
    assert result.success
    assert result.data["total_cost_usd"] == 0.11
    assert result.data["total_revenue_usd"] == 1.0
    allocs = result.data["allocations"]
    assert len(allocs) == 2


@pytest.mark.asyncio
async def test_reset(profiler):
    await profiler.execute("record", {
        "skill_id": "memory", "action": "save", "success": True,
    })
    result = await profiler.execute("reset", {})
    assert result.success
    profile = await profiler.execute("profile", {})
    assert profile.data["summary"]["total_calls"] == 0


@pytest.mark.asyncio
async def test_unknown_action(profiler):
    result = await profiler.execute("nonexistent", {})
    assert not result.success


@pytest.mark.asyncio
async def test_manifest(profiler):
    m = profiler.manifest
    assert m.skill_id == "skill_profiler"
    assert len(m.actions) == 8
