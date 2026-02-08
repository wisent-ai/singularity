"""Tests for maintenance scheduler presets and autonomous loop integration."""

import json
import asyncio
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from singularity.skills.scheduler_presets import (
    SchedulerPresetsSkill, PRESETS_FILE, DATA_DIR, BUILTIN_PRESETS,
    FULL_AUTONOMY_PRESETS,
)
from singularity.skills.autonomous_loop import AutonomousLoopSkill, LOOP_STATE_FILE
from singularity.skills.base import SkillResult


@pytest.fixture(autouse=True)
def clean_data():
    if PRESETS_FILE.exists():
        PRESETS_FILE.unlink()
    sched = DATA_DIR / "scheduler.json"
    if sched.exists():
        sched.unlink()
    if LOOP_STATE_FILE.exists():
        LOOP_STATE_FILE.unlink()
    yield
    if PRESETS_FILE.exists():
        PRESETS_FILE.unlink()
    if sched.exists():
        sched.unlink()
    if LOOP_STATE_FILE.exists():
        LOOP_STATE_FILE.unlink()


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def make_preset_skill():
    return SchedulerPresetsSkill()


# ── New Preset Definitions ──


class TestAdaptiveThresholdsPreset:
    def test_preset_exists(self):
        assert "adaptive_thresholds" in BUILTIN_PRESETS

    def test_preset_config(self):
        p = BUILTIN_PRESETS["adaptive_thresholds"]
        assert p.pillar == "self_improvement"
        assert len(p.schedules) == 2
        actions = [s.action for s in p.schedules]
        assert "tune_all" in actions
        assert "profiles" in actions

    def test_tune_all_interval(self):
        p = BUILTIN_PRESETS["adaptive_thresholds"]
        tune = [s for s in p.schedules if s.action == "tune_all"][0]
        assert tune.interval_seconds == 1800
        assert tune.skill_id == "adaptive_circuit_thresholds"

    def test_apply_preset(self):
        skill = make_preset_skill()
        result = run(skill.execute("apply", {"preset_id": "adaptive_thresholds"}))
        assert result.success
        assert len(result.data["task_ids"]) == 2


class TestRevenueGoalsPreset:
    def test_preset_exists(self):
        assert "revenue_goals" in BUILTIN_PRESETS

    def test_preset_config(self):
        p = BUILTIN_PRESETS["revenue_goals"]
        assert p.pillar == "revenue"
        assert len(p.schedules) == 3
        actions = [s.action for s in p.schedules]
        assert "assess" in actions
        assert "track" in actions
        assert "adjust" in actions

    def test_all_target_revenue_goal_setter(self):
        p = BUILTIN_PRESETS["revenue_goals"]
        for s in p.schedules:
            assert s.skill_id == "revenue_goal_auto_setter"

    def test_apply_preset(self):
        skill = make_preset_skill()
        result = run(skill.execute("apply", {"preset_id": "revenue_goals"}))
        assert result.success
        assert len(result.data["task_ids"]) == 3


class TestExperimentManagementPreset:
    def test_preset_exists(self):
        assert "experiment_management" in BUILTIN_PRESETS

    def test_preset_config(self):
        p = BUILTIN_PRESETS["experiment_management"]
        assert p.pillar == "self_improvement"
        assert len(p.schedules) == 2
        actions = [s.action for s in p.schedules]
        assert "conclude_all" in actions
        assert "learnings" in actions

    def test_apply_preset(self):
        skill = make_preset_skill()
        result = run(skill.execute("apply", {"preset_id": "experiment_management"}))
        assert result.success
        assert len(result.data["task_ids"]) == 2


class TestCircuitSharingMonitorPreset:
    def test_preset_exists(self):
        assert "circuit_sharing_monitor" in BUILTIN_PRESETS

    def test_preset_config(self):
        p = BUILTIN_PRESETS["circuit_sharing_monitor"]
        assert p.pillar == "replication"
        assert len(p.schedules) == 2
        actions = [s.action for s in p.schedules]
        assert "monitor" in actions
        assert "fleet_check" in actions

    def test_monitor_interval(self):
        p = BUILTIN_PRESETS["circuit_sharing_monitor"]
        mon = [s for s in p.schedules if s.action == "monitor"][0]
        assert mon.interval_seconds == 300  # every 5 min

    def test_apply_preset(self):
        skill = make_preset_skill()
        result = run(skill.execute("apply", {"preset_id": "circuit_sharing_monitor"}))
        assert result.success
        assert len(result.data["task_ids"]) == 2


class TestFullAutonomyInclusion:
    def test_new_presets_in_full_autonomy(self):
        """All new presets should be included in FULL_AUTONOMY_PRESETS."""
        assert "adaptive_thresholds" in FULL_AUTONOMY_PRESETS
        assert "revenue_goals" in FULL_AUTONOMY_PRESETS
        assert "experiment_management" in FULL_AUTONOMY_PRESETS
        assert "circuit_sharing_monitor" in FULL_AUTONOMY_PRESETS

    def test_full_autonomy_count(self):
        """Full autonomy should have all presets."""
        assert len(FULL_AUTONOMY_PRESETS) == len(BUILTIN_PRESETS)


# ── Autonomous Loop Integration ──


class TestLoopMaintenancePresets:
    def test_ensure_maintenance_presets_calls_scheduler(self):
        """Loop should call scheduler_presets.apply for each maintenance preset."""
        loop_skill = AutonomousLoopSkill()
        mock_context = MagicMock()
        mock_context.call_skill = AsyncMock(return_value=SkillResult(
            success=True, message="Applied", data={"task_ids": ["t1"]}
        ))
        loop_skill.context = mock_context
        state = loop_skill._default_state()

        run(loop_skill._ensure_maintenance_presets(state))

        calls = mock_context.call_skill.call_args_list
        preset_ids = [c[0][2]["preset_id"] for c in calls]
        assert "adaptive_thresholds" in preset_ids
        assert "revenue_goals" in preset_ids
        assert "experiment_management" in preset_ids
        assert "circuit_sharing_monitor" in preset_ids

    def test_presets_applied_once(self):
        """Should only apply presets on first call, not subsequent calls."""
        loop_skill = AutonomousLoopSkill()
        mock_context = MagicMock()
        mock_context.call_skill = AsyncMock(return_value=SkillResult(
            success=True, message="Applied", data={"task_ids": ["t1"]}
        ))
        loop_skill.context = mock_context
        state = loop_skill._default_state()

        run(loop_skill._ensure_maintenance_presets(state))
        assert state.get("maintenance_presets_applied") is not None

        mock_context.call_skill.reset_mock()
        run(loop_skill._ensure_maintenance_presets(state))
        mock_context.call_skill.assert_not_called()

    def test_presets_fail_silent(self):
        """Should not raise if scheduler_presets skill is unavailable."""
        loop_skill = AutonomousLoopSkill()
        mock_context = MagicMock()
        mock_context.call_skill = AsyncMock(side_effect=Exception("Skill not found"))
        loop_skill.context = mock_context
        state = loop_skill._default_state()

        # Should not raise
        run(loop_skill._ensure_maintenance_presets(state))

    def test_presets_no_context(self):
        """Should handle missing context gracefully."""
        loop_skill = AutonomousLoopSkill()
        loop_skill.context = None
        state = loop_skill._default_state()
        run(loop_skill._ensure_maintenance_presets(state))
        assert "maintenance_presets_applied" not in state

    def test_partial_preset_application(self):
        """Should track which presets were successfully applied."""
        loop_skill = AutonomousLoopSkill()
        mock_context = MagicMock()

        call_count = 0
        async def side_effect(skill_id, action, params):
            nonlocal call_count
            call_count += 1
            if params.get("preset_id") == "revenue_goals":
                raise Exception("Skill unavailable")
            return SkillResult(success=True, message="OK", data={"task_ids": ["t1"]})

        mock_context.call_skill = AsyncMock(side_effect=side_effect)
        loop_skill.context = mock_context
        state = loop_skill._default_state()

        run(loop_skill._ensure_maintenance_presets(state))

        applied = state["maintenance_presets_applied"]["presets"]
        assert "adaptive_thresholds" in applied
        assert "revenue_goals" not in applied
        assert "experiment_management" in applied
        assert "circuit_sharing_monitor" in applied
