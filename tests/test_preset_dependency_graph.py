"""Tests for SchedulerPresetsSkill dependency graph feature."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from singularity.skills.scheduler_presets import (
    SchedulerPresetsSkill, BUILTIN_PRESETS, FULL_AUTONOMY_PRESETS,
    PresetDefinition, PresetSchedule,
)
from singularity.skills.base import SkillResult, SkillContext, SkillRegistry


@pytest.fixture
def presets_skill():
    skill = SchedulerPresetsSkill()
    registry = SkillRegistry()
    ctx = SkillContext(registry=registry, agent_name="TestAgent")
    ctx.call_skill = AsyncMock(return_value=SkillResult(
        success=True, message="scheduled", data={"id": "sched_mock123"}
    ))
    ctx.list_skills = MagicMock(return_value=list({
        s.skill_id for p in BUILTIN_PRESETS.values() for s in p.schedules
    }))
    skill.set_context(ctx)
    return skill


def test_preset_definitions_have_depends_on():
    """All PresetDefinitions should have a depends_on field."""
    for pid, preset in BUILTIN_PRESETS.items():
        assert hasattr(preset, "depends_on"), f"{pid} missing depends_on"
        assert isinstance(preset.depends_on, list), f"{pid}.depends_on not a list"


def test_dependencies_reference_valid_presets():
    """All depends_on entries should reference existing preset IDs."""
    for pid, preset in BUILTIN_PRESETS.items():
        for dep in preset.depends_on:
            assert dep in BUILTIN_PRESETS, f"{pid} depends on unknown preset: {dep}"


def test_no_self_dependencies():
    """No preset should depend on itself."""
    for pid, preset in BUILTIN_PRESETS.items():
        assert pid not in preset.depends_on, f"{pid} depends on itself"


def test_expected_dependencies():
    """Verify specific expected dependency relationships."""
    assert "health_monitoring" in BUILTIN_PRESETS["alert_polling"].depends_on
    assert "self_assessment" in BUILTIN_PRESETS["self_tuning"].depends_on
    assert "health_monitoring" in BUILTIN_PRESETS["adaptive_thresholds"].depends_on
    assert "revenue_goals" in BUILTIN_PRESETS["revenue_goal_evaluation"].depends_on
    assert "feedback_loop" in BUILTIN_PRESETS["experiment_management"].depends_on
    assert "health_monitoring" in BUILTIN_PRESETS["dashboard_auto_check"].depends_on
    assert "circuit_sharing_monitor" in BUILTIN_PRESETS["fleet_health_auto_heal"].depends_on


def test_root_presets_have_no_deps():
    """Root presets (no dependencies) should exist."""
    roots = [pid for pid, p in BUILTIN_PRESETS.items() if not p.depends_on]
    assert len(roots) > 0
    assert "health_monitoring" in roots


@pytest.mark.asyncio
async def test_dependency_graph_full(presets_skill):
    """dependency_graph action returns full graph info."""
    result = await presets_skill.execute("dependency_graph", {})
    assert result.success
    assert "graph" in result.data
    assert "topological_order" in result.data
    assert "roots" in result.data
    assert "leaves" in result.data
    assert "cycles" in result.data
    assert result.data["cycles"] == []
    assert len(result.data["topological_order"]) == len(BUILTIN_PRESETS)


@pytest.mark.asyncio
async def test_dependency_graph_single_preset(presets_skill):
    """dependency_graph for a single preset shows its deps."""
    result = await presets_skill.execute("dependency_graph", {"preset_id": "alert_polling"})
    assert result.success
    assert "health_monitoring" in result.data["transitive_deps"]
    assert "health_monitoring" in result.data["direct_deps"]


@pytest.mark.asyncio
async def test_topological_order_deps_before_dependents(presets_skill):
    """In topological order, dependencies appear before their dependents."""
    result = await presets_skill.execute("dependency_graph", {})
    order = result.data["topological_order"]
    idx = {pid: i for i, pid in enumerate(order)}
    for pid, preset in BUILTIN_PRESETS.items():
        for dep in preset.depends_on:
            assert idx[dep] < idx[pid], f"{dep} should come before {pid} in topo order"


@pytest.mark.asyncio
async def test_apply_with_deps(presets_skill):
    """apply_with_deps should apply dependencies before the target preset."""
    result = await presets_skill.execute("apply_with_deps", {"preset_id": "alert_polling"})
    assert result.success
    order = result.data["apply_order"]
    assert order.index("health_monitoring") < order.index("alert_polling")
    assert result.data["applied"] >= 2  # at least health_monitoring + alert_polling


@pytest.mark.asyncio
async def test_apply_with_deps_skips_already_applied(presets_skill):
    """apply_with_deps skips presets that are already applied."""
    # First apply the dependency
    await presets_skill.execute("apply", {"preset_id": "health_monitoring"})
    # Now apply with deps - health_monitoring should be skipped
    result = await presets_skill.execute("apply_with_deps", {"preset_id": "alert_polling"})
    assert result.success
    assert "health_monitoring" in result.data["skipped"]


@pytest.mark.asyncio
async def test_apply_all_uses_topological_order(presets_skill):
    """apply_all should return apply_order showing topological sorting."""
    result = await presets_skill.execute("apply_all", {})
    assert result.success
    assert "apply_order" in result.data
    order = result.data["apply_order"]
    # Verify deps before dependents
    idx = {pid: i for i, pid in enumerate(order)}
    for pid, preset in BUILTIN_PRESETS.items():
        for dep in preset.depends_on:
            if dep in idx and pid in idx:
                assert idx[dep] < idx[pid], f"{dep} should come before {pid}"


@pytest.mark.asyncio
async def test_dependency_graph_unknown_preset(presets_skill):
    """dependency_graph with unknown preset_id should fail."""
    result = await presets_skill.execute("dependency_graph", {"preset_id": "nonexistent"})
    assert not result.success


@pytest.mark.asyncio
async def test_apply_with_deps_unknown_preset(presets_skill):
    """apply_with_deps with unknown preset_id should fail."""
    result = await presets_skill.execute("apply_with_deps", {"preset_id": "nonexistent"})
    assert not result.success
