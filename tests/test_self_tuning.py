"""Tests for SelfTuningSkill."""

import pytest
import json
from singularity.skills.self_tuning import SelfTuningSkill, DATA_FILE
import singularity.skills.self_tuning as mod


@pytest.fixture
def skill(tmp_path):
    """Create skill with temp data paths and nulled external data."""
    s = SelfTuningSkill()
    mod.DATA_FILE = tmp_path / "self_tuning.json"

    # Null out external data by default - tests must explicitly provide mocks
    async def no_router():
        return None
    async def no_metrics(_):
        return None
    async def no_budget():
        return None

    s._get_router_data = no_router
    s._get_skill_metrics = no_metrics
    s._get_budget_data = no_budget
    return s


MOCK_ROUTER_DATA = {
    "model_performance": {
        "openai:gpt-4o-mini": {"total": 10, "successes": 9, "failures": 1, "total_quality": 7.5, "quality_count": 8, "total_tokens": 5000},
        "anthropic:claude-sonnet": {"total": 8, "successes": 4, "failures": 4, "total_quality": 3.0, "quality_count": 4, "total_tokens": 3000},
        "openai:gpt-4o": {"total": 3, "successes": 3, "failures": 0, "total_quality": 3.0, "quality_count": 3, "total_tokens": 2000},
    },
    "budget_mode": True,
    "budget_limit_usd": 10.0,
    "spent_this_period": 7.5,
}


def _with_router(skill):
    """Inject mock router data."""
    async def mock_router():
        return dict(MOCK_ROUTER_DATA)
    skill._get_router_data = mock_router

    async def mock_budget():
        return {
            "budget_mode": True,
            "budget_limit_usd": 10.0,
            "spent_this_period": 7.5,
        }
    skill._get_budget_data = mock_budget
    return skill


@pytest.mark.asyncio
async def test_manifest(skill):
    m = skill.manifest
    assert m.skill_id == "self_tuning"
    assert len(m.actions) == 7
    names = [a.name for a in m.actions]
    assert "analyze" in names
    assert "tune_router" in names
    assert "tune_budget" in names
    assert "recommend" in names
    assert "status" in names


@pytest.mark.asyncio
async def test_unknown_action(skill):
    r = await skill.execute("nonexistent", {})
    assert not r.success
    assert "Unknown action" in r.message


@pytest.mark.asyncio
async def test_configure(skill):
    r = await skill.execute("configure", {"min_samples": 10, "success_rate_threshold": 0.8})
    assert r.success
    assert r.data["config"]["min_samples"] == 10
    assert r.data["config"]["success_rate_threshold"] == 0.8


@pytest.mark.asyncio
async def test_configure_no_changes(skill):
    r = await skill.execute("configure", {})
    assert r.success
    assert "No configuration changes" in r.message


@pytest.mark.asyncio
async def test_status_initial(skill):
    r = await skill.execute("status", {})
    assert r.success
    assert r.data["stats"]["total_tune_cycles"] == 0


@pytest.mark.asyncio
async def test_history_empty(skill):
    r = await skill.execute("history", {})
    assert r.success
    assert r.data["history"] == []


@pytest.mark.asyncio
async def test_analyze_no_data(skill):
    r = await skill.execute("analyze", {})
    assert r.success
    assert "no data" in r.message


@pytest.mark.asyncio
async def test_analyze_with_router(skill):
    _with_router(skill)
    r = await skill.execute("analyze", {})
    assert r.success
    assert r.data["router_analysis"] is not None
    assert r.data["router_analysis"]["total_tasks"] == 21


@pytest.mark.asyncio
async def test_tune_router_demotes_bad_model(skill):
    _with_router(skill)
    await skill.execute("configure", {"min_samples": 3})
    r = await skill.execute("tune_router", {})
    assert r.success
    adjustments = r.data["adjustments"]
    demoted = [a for a in adjustments if a["direction"] == "demote"]
    assert len(demoted) >= 1
    assert any("claude-sonnet" in a["model"] for a in demoted)


@pytest.mark.asyncio
async def test_tune_router_promotes_good_model(skill):
    _with_router(skill)
    await skill.execute("configure", {"min_samples": 3})
    r = await skill.execute("tune_router", {})
    assert r.success
    adjustments = r.data["adjustments"]
    promoted = [a for a in adjustments if a["direction"] == "promote"]
    assert len(promoted) >= 1
    assert any("gpt-4o-mini" in a["model"] or "gpt-4o" in a["model"] for a in promoted)


@pytest.mark.asyncio
async def test_tune_router_dry_run(skill):
    _with_router(skill)
    await skill.execute("configure", {"min_samples": 3})
    r = await skill.execute("tune_router", {"dry_run": True})
    assert r.success
    assert "[DRY RUN]" in r.message
    s = await skill.execute("status", {})
    assert s.data["stats"]["total_tune_cycles"] == 0


@pytest.mark.asyncio
async def test_tune_router_updates_stats(skill):
    _with_router(skill)
    await skill.execute("configure", {"min_samples": 3})
    await skill.execute("tune_router", {})
    s = await skill.execute("status", {})
    assert s.data["stats"]["total_tune_cycles"] == 1
    assert s.data["stats"]["total_adjustments"] > 0


@pytest.mark.asyncio
async def test_tune_router_no_data(skill):
    r = await skill.execute("tune_router", {})
    assert not r.success
    assert "No router performance data" in r.message


@pytest.mark.asyncio
async def test_tune_budget_no_data(skill):
    r = await skill.execute("tune_budget", {})
    assert not r.success


@pytest.mark.asyncio
async def test_tune_budget_with_data(skill):
    _with_router(skill)
    r = await skill.execute("tune_budget", {"dry_run": True})
    assert r.success
    assert r.data["current_utilization"] == 0.75


@pytest.mark.asyncio
async def test_recommend_with_data(skill):
    _with_router(skill)
    await skill.execute("configure", {"min_samples": 3})
    r = await skill.execute("recommend", {})
    assert r.success
    recs = r.data["recommendations"]
    assert len(recs) >= 1


@pytest.mark.asyncio
async def test_recommend_no_data(skill):
    r = await skill.execute("recommend", {})
    assert r.success
    assert "No tuning recommendations" in r.message


@pytest.mark.asyncio
async def test_min_samples_respected(skill):
    _with_router(skill)
    await skill.execute("configure", {"min_samples": 5})
    r = await skill.execute("tune_router", {})
    adjustments = r.data.get("adjustments", [])
    models_adjusted = [a["model"] for a in adjustments]
    assert "openai:gpt-4o" not in models_adjusted


@pytest.mark.asyncio
async def test_history_records_tuning(skill):
    _with_router(skill)
    await skill.execute("configure", {"min_samples": 3})
    await skill.execute("tune_router", {})
    r = await skill.execute("history", {})
    assert len(r.data["history"]) == 1
    assert r.data["history"][0]["type"] == "router_tune"
