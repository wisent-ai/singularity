"""Tests for FleetOrchestrationPoliciesSkill."""
import json
import pytest
from pathlib import Path
from singularity.skills.fleet_orchestration_policies import (
    FleetOrchestrationPoliciesSkill,
    BUILTIN_POLICIES,
    BUILTIN_BUNDLES,
    _compare_configs,
    _score_fleet_state,
)


@pytest.fixture
def tmp_data(tmp_path):
    return tmp_path / "fleet_orch_policies.json"


@pytest.fixture
def skill(tmp_data):
    return FleetOrchestrationPoliciesSkill(data_path=tmp_data)


@pytest.mark.asyncio
async def test_list_policies(skill):
    r = await skill.execute("list_policies", {})
    assert r.success
    assert len(r.data["policies"]) == len(BUILTIN_POLICIES)
    ids = {p["id"] for p in r.data["policies"]}
    assert "cost_aware" in ids
    assert "resilience" in ids
    assert "revenue_optimized" in ids


@pytest.mark.asyncio
async def test_list_policies_filter_category(skill):
    r = await skill.execute("list_policies", {"category": "cost"})
    assert r.success
    assert all(p["category"] == "cost" for p in r.data["policies"])


@pytest.mark.asyncio
async def test_preview(skill):
    r = await skill.execute("preview", {"policy_id": "resilience"})
    assert r.success
    assert r.data["config"]["min_fleet_size"] == 2
    assert r.data["traits"]["availability_priority"] == "high"


@pytest.mark.asyncio
async def test_preview_not_found(skill):
    r = await skill.execute("preview", {"policy_id": "nonexistent"})
    assert not r.success


@pytest.mark.asyncio
async def test_deploy(skill):
    r = await skill.execute("deploy", {"policy_id": "cost_aware"})
    assert r.success
    assert r.data["policy_id"] == "cost_aware"
    # Verify status reflects it
    s = await skill.execute("status", {})
    assert s.data["active_policy"]["id"] == "cost_aware"


@pytest.mark.asyncio
async def test_deploy_dry_run(skill):
    r = await skill.execute("deploy", {"policy_id": "resilience", "dry_run": True})
    assert r.success
    assert r.data["dry_run"] is True
    # Should NOT be active
    s = await skill.execute("status", {})
    assert s.data["active_policy"] is None


@pytest.mark.asyncio
async def test_deploy_switch_tracking(skill):
    await skill.execute("deploy", {"policy_id": "balanced"})
    await skill.execute("deploy", {"policy_id": "resilience"})
    s = await skill.execute("status", {})
    assert s.data["stats"]["total_deploys"] == 2
    assert s.data["stats"]["total_switches"] == 1


@pytest.mark.asyncio
async def test_compare(skill):
    r = await skill.execute("compare", {"policy_a": "cost_aware", "policy_b": "resilience"})
    assert r.success
    assert r.data["summary"]["different"] > 0
    diffs = r.data["diffs"]
    min_fleet = next(d for d in diffs if d["key"] == "min_fleet_size")
    assert min_fleet["value_a"] == 1
    assert min_fleet["value_b"] == 2


@pytest.mark.asyncio
async def test_recommend_budget_tight(skill):
    r = await skill.execute("recommend", {"budget_remaining_pct": 10})
    assert r.success
    assert r.data["recommendation"]["policy_id"] == "cost_aware"


@pytest.mark.asyncio
async def test_recommend_production_sla(skill):
    r = await skill.execute("recommend", {"is_production": True, "has_sla": True, "health_pct": 70})
    assert r.success
    assert r.data["recommendation"]["policy_id"] == "resilience"


@pytest.mark.asyncio
async def test_recommend_high_revenue(skill):
    r = await skill.execute("recommend", {"revenue_per_hour": 5.0, "has_sla": True})
    assert r.success
    assert r.data["recommendation"]["policy_id"] == "revenue_optimized"


@pytest.mark.asyncio
async def test_customize(skill):
    r = await skill.execute("customize", {
        "base_policy_id": "balanced",
        "custom_id": "my_policy",
        "custom_name": "My Custom Policy",
        "overrides": {"max_fleet_size": 20, "scale_up_threshold": 0.5},
    })
    assert r.success
    assert r.data["config"]["max_fleet_size"] == 20
    assert r.data["config"]["scale_up_threshold"] == 0.5
    # Should appear in list
    l = await skill.execute("list_policies", {})
    ids = {p["id"] for p in l.data["policies"]}
    assert "my_policy" in ids


@pytest.mark.asyncio
async def test_customize_invalid_field(skill):
    r = await skill.execute("customize", {
        "base_policy_id": "balanced",
        "custom_id": "bad",
        "overrides": {"nonexistent_field": 42},
    })
    assert not r.success


@pytest.mark.asyncio
async def test_schedule_bundle(skill):
    r = await skill.execute("schedule", {"bundle_id": "production_standard"})
    assert r.success
    assert len(r.data["schedules"]) == 2


@pytest.mark.asyncio
async def test_schedule_custom(skill):
    r = await skill.execute("schedule", {"schedules": [
        {"policy_id": "cost_aware", "hours": "22:00-06:00"},
        {"policy_id": "resilience", "hours": "06:00-22:00"},
    ]})
    assert r.success
    assert len(r.data["schedules"]) == 2


@pytest.mark.asyncio
async def test_unknown_action(skill):
    r = await skill.execute("bogus", {})
    assert not r.success


def test_compare_configs_helper():
    a = BUILTIN_POLICIES["cost_aware"]["config"]
    b = BUILTIN_POLICIES["resilience"]["config"]
    diffs = _compare_configs(a, b)
    assert len(diffs) > 0
    assert any(not d["same"] for d in diffs)


def test_score_fleet_state():
    scores = _score_fleet_state({"budget_remaining_pct": 5})
    assert scores["cost_aware"] == 90
