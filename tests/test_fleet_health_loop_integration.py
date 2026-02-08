#!/usr/bin/env python3
"""Tests for fleet health event bridge integration in AutonomousLoopSkill."""

import pytest
import json
from pathlib import Path
from unittest.mock import patch, AsyncMock, MagicMock
from singularity.skills.autonomous_loop import AutonomousLoopSkill, LOOP_STATE_FILE
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
    """Create a mock SkillContext."""
    ctx = MagicMock()
    async def mock_call_skill(skill_id, action, params=None):
        return SkillResult(success=True, message="ok", data={})
    ctx.call_skill = AsyncMock(side_effect=mock_call_skill)
    return ctx


# ── Fleet Health Monitor Tests ──────────────────────────────


@pytest.mark.asyncio
async def test_monitor_fleet_health_calls_bridge(skill, mock_context):
    """Verify _monitor_fleet_health calls fleet_health_events.monitor()."""
    skill.context = mock_context
    state = {"stats": {}}
    await skill._monitor_fleet_health(state)

    mock_context.call_skill.assert_called_once_with(
        "fleet_health_events", "monitor", {}
    )
    assert state["stats"]["fleet_health_monitors"] == 1


@pytest.mark.asyncio
async def test_monitor_fleet_health_increments_stats(skill, mock_context):
    """Verify stats increment on repeated calls."""
    skill.context = mock_context
    state = {"stats": {"fleet_health_monitors": 3}}
    await skill._monitor_fleet_health(state)
    assert state["stats"]["fleet_health_monitors"] == 4


@pytest.mark.asyncio
async def test_monitor_fleet_health_no_context(skill):
    """Verify _monitor_fleet_health handles missing context gracefully."""
    skill.context = None
    state = {"stats": {}}
    await skill._monitor_fleet_health(state)
    assert "fleet_health_monitors" not in state["stats"]


@pytest.mark.asyncio
async def test_monitor_fleet_health_exception_handled(skill):
    """Verify _monitor_fleet_health swallows exceptions."""
    ctx = MagicMock()
    ctx.call_skill = AsyncMock(side_effect=RuntimeError("connection failed"))
    skill.context = ctx
    state = {"stats": {}}
    await skill._monitor_fleet_health(state)
    assert "fleet_health_monitors" not in state["stats"]


# ── Fleet Health Check Tests ──────────────────────────────


@pytest.mark.asyncio
async def test_check_fleet_health_calls_bridge(skill, mock_context):
    """Verify _check_fleet_health calls fleet_health_events.fleet_check()."""
    skill.context = mock_context
    state = {"stats": {}, "iteration_count": 0, "config": {}}
    await skill._check_fleet_health(state)

    mock_context.call_skill.assert_called_once_with(
        "fleet_health_events", "fleet_check", {}
    )
    assert state["stats"]["fleet_health_checks"] == 1


@pytest.mark.asyncio
async def test_check_fleet_health_respects_interval(skill, mock_context):
    """Verify _check_fleet_health only runs every N iterations."""
    skill.context = mock_context

    # iteration_count=3 with default interval=5 -> should NOT run
    state = {"stats": {}, "iteration_count": 3, "config": {}}
    await skill._check_fleet_health(state)
    mock_context.call_skill.assert_not_called()
    assert "fleet_health_checks" not in state["stats"]


@pytest.mark.asyncio
async def test_check_fleet_health_runs_at_interval(skill, mock_context):
    """Verify _check_fleet_health runs at exact interval multiples."""
    skill.context = mock_context

    # iteration_count=5 with default interval=5 -> should run
    state = {"stats": {}, "iteration_count": 5, "config": {}}
    await skill._check_fleet_health(state)
    mock_context.call_skill.assert_called_once()
    assert state["stats"]["fleet_health_checks"] == 1


@pytest.mark.asyncio
async def test_check_fleet_health_custom_interval(skill, mock_context):
    """Verify custom fleet_check_interval is respected."""
    skill.context = mock_context

    # Custom interval=3, iteration=6 -> should run
    state = {"stats": {}, "iteration_count": 6, "config": {"fleet_check_interval": 3}}
    await skill._check_fleet_health(state)
    mock_context.call_skill.assert_called_once()

    mock_context.call_skill.reset_mock()
    # Custom interval=3, iteration=7 -> should NOT run
    state2 = {"stats": {}, "iteration_count": 7, "config": {"fleet_check_interval": 3}}
    await skill._check_fleet_health(state2)
    mock_context.call_skill.assert_not_called()


@pytest.mark.asyncio
async def test_check_fleet_health_no_context(skill):
    """Verify _check_fleet_health handles missing context gracefully."""
    skill.context = None
    state = {"stats": {}, "iteration_count": 0, "config": {}}
    await skill._check_fleet_health(state)
    assert "fleet_health_checks" not in state["stats"]


@pytest.mark.asyncio
async def test_check_fleet_health_exception_handled(skill):
    """Verify _check_fleet_health swallows exceptions."""
    ctx = MagicMock()
    ctx.call_skill = AsyncMock(side_effect=RuntimeError("fleet unavailable"))
    skill.context = ctx
    state = {"stats": {}, "iteration_count": 0, "config": {}}
    await skill._check_fleet_health(state)
    assert "fleet_health_checks" not in state["stats"]
