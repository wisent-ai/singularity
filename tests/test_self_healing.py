"""Tests for SelfHealingSkill."""

import json
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from singularity.skills.self_healing import (
    SelfHealingSkill, HealthStatus, DiagnosisType, RepairStrategy, HEALING_FILE,
)


@pytest.fixture
def tmp_healing(tmp_path):
    """Use a temp file for healing data."""
    import singularity.skills.self_healing as mod
    original = mod.HEALING_FILE
    mod.HEALING_FILE = tmp_path / "self_healing.json"
    yield tmp_path
    mod.HEALING_FILE = original


@pytest.fixture
def skill(tmp_healing):
    s = SelfHealingSkill()
    return s


@pytest.mark.asyncio
async def test_status_empty(skill):
    result = await skill.execute("status", {})
    assert result.success
    assert result.data["subsystem_count"] == 0
    assert result.data["quarantine_count"] == 0


@pytest.mark.asyncio
async def test_quarantine_and_release(skill):
    r = await skill.execute("quarantine", {"skill_id": "test_skill", "reason": "broken", "duration_hours": 1})
    assert r.success
    assert "test_skill" in r.data["skill_id"]

    status = await skill.execute("status", {})
    assert status.data["quarantine_count"] == 1

    rel = await skill.execute("release", {"skill_id": "test_skill"})
    assert rel.success

    status2 = await skill.execute("status", {})
    assert status2.data["quarantine_count"] == 0


@pytest.mark.asyncio
async def test_quarantine_missing(skill):
    r = await skill.execute("release", {"skill_id": "nonexistent"})
    assert not r.success


@pytest.mark.asyncio
async def test_heal_noop(skill):
    r = await skill.execute("heal", {"skill_id": "test_skill", "strategy": "noop", "reason": "test"})
    assert r.success
    assert r.data["strategy"] == "noop"


@pytest.mark.asyncio
async def test_heal_invalid_strategy(skill):
    r = await skill.execute("heal", {"skill_id": "test_skill", "strategy": "invalid"})
    assert not r.success


@pytest.mark.asyncio
async def test_heal_missing_params(skill):
    r = await skill.execute("heal", {"skill_id": "test_skill"})
    assert not r.success


@pytest.mark.asyncio
async def test_diagnose_healthy(skill):
    r = await skill.execute("diagnose", {"skill_id": "some_skill"})
    assert r.success
    assert r.data["diagnosis"]["type"] == DiagnosisType.NONE


@pytest.mark.asyncio
async def test_diagnose_with_symptoms(skill):
    r = await skill.execute("diagnose", {
        "skill_id": "bad_skill",
        "symptoms": {"error_rate": 0.8, "failures": 10},
    })
    assert r.success
    assert r.data["diagnosis"]["type"] != DiagnosisType.NONE
    assert len(r.data["recommended_repairs"]) > 0


@pytest.mark.asyncio
async def test_scan_no_context(skill):
    """Scan without context should work (no skills found)."""
    r = await skill.execute("scan", {})
    assert r.success
    assert r.data["total_scanned"] == 0


@pytest.mark.asyncio
async def test_auto_heal_dry_run(skill):
    r = await skill.execute("auto_heal", {"dry_run": True})
    assert r.success
    assert r.data["dry_run"] is True


@pytest.mark.asyncio
async def test_healing_report_empty(skill):
    r = await skill.execute("healing_report", {"timeframe_hours": 24})
    assert r.success
    assert r.data["total_repairs"] == 0


@pytest.mark.asyncio
async def test_healing_report_with_data(skill):
    # Do some heals first
    await skill.execute("heal", {"skill_id": "s1", "strategy": "noop", "reason": "test"})
    await skill.execute("heal", {"skill_id": "s2", "strategy": "noop", "reason": "test"})
    r = await skill.execute("healing_report", {"timeframe_hours": 1})
    assert r.success
    assert r.data["total_repairs"] == 2
    assert r.data["overall_success_rate"] == 1.0


@pytest.mark.asyncio
async def test_unknown_action(skill):
    r = await skill.execute("bogus", {})
    assert not r.success


@pytest.mark.asyncio
async def test_quarantine_requires_skill_id(skill):
    r = await skill.execute("quarantine", {})
    assert not r.success


@pytest.mark.asyncio
async def test_diagnose_requires_skill_id(skill):
    r = await skill.execute("diagnose", {})
    assert not r.success
