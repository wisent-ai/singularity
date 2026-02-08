"""Tests for AdaptiveCircuitThresholdsSkill."""
import pytest
import time

from singularity.skills.adaptive_circuit_thresholds import AdaptiveCircuitThresholdsSkill


@pytest.fixture
def skill(tmp_path, monkeypatch):
    """Create skill with temp data dir."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setattr("singularity.skills.adaptive_circuit_thresholds.DATA_DIR", data_dir)
    monkeypatch.setattr("singularity.skills.adaptive_circuit_thresholds.THRESHOLDS_FILE", data_dir / "adaptive_thresholds.json")
    s = AdaptiveCircuitThresholdsSkill()
    s._config["min_samples"] = 5  # Lower for testing
    return s


def _make_records(total, failure_rate, cost=0.01):
    """Generate synthetic circuit records."""
    now = time.time()
    failures = int(total * failure_rate)
    records = []
    for i in range(total):
        records.append({
            "timestamp": now - (total - i) * 60,
            "success": i >= failures,
            "cost": cost,
            "duration_ms": 100.0,
        })
    return records


@pytest.mark.asyncio
async def test_analyze_computes_profile(skill):
    """Analyze should compute a statistical profile and thresholds."""
    records = _make_records(20, 0.1)
    result = await skill.execute("analyze", {
        "skill_id": "test_api",
        "circuit_data": {"records": records},
    })
    assert result.success
    assert "test_api" in result.data["skill_id"]
    assert "stats" in result.data
    assert "thresholds" in result.data
    assert result.data["thresholds"]["failure_rate_threshold"] > 0.1  # Above baseline


@pytest.mark.asyncio
async def test_analyze_insufficient_data(skill):
    """Analyze with too few records should fail."""
    records = _make_records(3, 0.1)
    result = await skill.execute("analyze", {
        "skill_id": "tiny_skill",
        "circuit_data": {"records": records},
    })
    assert not result.success
    assert "Insufficient" in result.message


@pytest.mark.asyncio
async def test_analyze_high_failure_rate(skill):
    """High failure rate skills get higher thresholds."""
    records = _make_records(20, 0.4)
    result = await skill.execute("analyze", {
        "skill_id": "flaky_api",
        "circuit_data": {"records": records},
    })
    assert result.success
    threshold = result.data["thresholds"]["failure_rate_threshold"]
    assert threshold > 0.4  # Threshold above baseline


@pytest.mark.asyncio
async def test_analyze_low_failure_rate(skill):
    """Low failure rate skills get lower thresholds."""
    records = _make_records(20, 0.0)  # Perfect reliability
    result = await skill.execute("analyze", {
        "skill_id": "reliable_skill",
        "circuit_data": {"records": records},
    })
    assert result.success
    threshold = result.data["thresholds"]["failure_rate_threshold"]
    assert threshold < 0.5  # Tighter than global default


@pytest.mark.asyncio
async def test_tune_applies_override(skill):
    """Tune should store override after analysis."""
    records = _make_records(20, 0.15)
    await skill.execute("analyze", {"skill_id": "svc", "circuit_data": {"records": records}})
    result = await skill.execute("tune", {"skill_id": "svc"})
    assert result.success
    assert "svc" in skill._overrides
    assert skill._overrides["svc"]["failure_rate_threshold"] > 0


@pytest.mark.asyncio
async def test_tune_without_analysis(skill):
    """Tune without prior analysis should fail."""
    result = await skill.execute("tune", {"skill_id": "unknown"})
    assert not result.success


@pytest.mark.asyncio
async def test_tune_all(skill):
    """tune_all should analyze and tune multiple skills."""
    circuits = {
        "api_a": {"records": _make_records(15, 0.1)},
        "api_b": {"records": _make_records(15, 0.3)},
        "api_c": {"records": _make_records(3, 0.5)},  # Too few
    }
    result = await skill.execute("tune_all", {"circuits_data": circuits})
    assert result.success
    assert len(result.data["analyzed"]) == 2
    assert len(result.data["skipped"]) == 1


@pytest.mark.asyncio
async def test_profiles_action(skill):
    """Profiles should list analyzed skills."""
    records = _make_records(20, 0.2)
    await skill.execute("analyze", {"skill_id": "profiled_skill", "circuit_data": {"records": records}})
    result = await skill.execute("profiles", {})
    assert result.success
    assert len(result.data["profiles"]) == 1
    assert result.data["profiles"][0]["skill_id"] == "profiled_skill"


@pytest.mark.asyncio
async def test_profiles_single_skill(skill):
    """Profiles with skill_id should return just that one."""
    records = _make_records(20, 0.2)
    await skill.execute("analyze", {"skill_id": "specific", "circuit_data": {"records": records}})
    result = await skill.execute("profiles", {"skill_id": "specific"})
    assert result.success
    assert "profile" in result.data


@pytest.mark.asyncio
async def test_reset(skill):
    """Reset should remove profile and override."""
    records = _make_records(20, 0.2)
    await skill.execute("analyze", {"skill_id": "resettable", "circuit_data": {"records": records}})
    await skill.execute("tune", {"skill_id": "resettable"})
    assert "resettable" in skill._overrides
    result = await skill.execute("reset", {"skill_id": "resettable"})
    assert result.success
    assert "resettable" not in skill._profiles
    assert "resettable" not in skill._overrides


@pytest.mark.asyncio
async def test_configure(skill):
    """Configure should update tuning parameters."""
    result = await skill.execute("configure", {"sensitivity": 3.0, "min_samples": 8})
    assert result.success
    assert skill._config["sensitivity"] == 3.0
    assert skill._config["min_samples"] == 8


@pytest.mark.asyncio
async def test_status(skill):
    """Status should return summary."""
    result = await skill.execute("status", {})
    assert result.success
    assert "total_profiles" in result.data


@pytest.mark.asyncio
async def test_history(skill):
    """History should track tuning events."""
    records = _make_records(20, 0.1)
    await skill.execute("analyze", {"skill_id": "hist_skill", "circuit_data": {"records": records}})
    result = await skill.execute("history", {"limit": 10})
    assert result.success
    assert len(result.data["history"]) >= 1


@pytest.mark.asyncio
async def test_get_override_for_skill(skill):
    """get_override_for_skill API for circuit breaker integration."""
    records = _make_records(20, 0.2)
    await skill.execute("analyze", {"skill_id": "api_x", "circuit_data": {"records": records}})
    await skill.execute("tune", {"skill_id": "api_x"})
    override = skill.get_override_for_skill("api_x")
    assert override is not None
    assert "failure_rate_threshold" in override
    assert skill.get_override_for_skill("nonexistent") is None


@pytest.mark.asyncio
async def test_auto_apply(skill):
    """With enable_auto_apply, analyze should also apply override."""
    skill._config["enable_auto_apply"] = True
    records = _make_records(20, 0.15)
    result = await skill.execute("analyze", {"skill_id": "auto_svc", "circuit_data": {"records": records}})
    assert result.success
    assert result.data["auto_applied"]
    assert "auto_svc" in skill._overrides


@pytest.mark.asyncio
async def test_synthesize_from_summary(skill):
    """Analyze with summary data (no raw records) should synthesize."""
    result = await skill.execute("analyze", {
        "skill_id": "summary_skill",
        "circuit_data": {
            "window_size": 20,
            "failure_rate": 0.25,
            "total_cost": 0.5,
            "consecutive_failures": 2,
        },
    })
    assert result.success
    assert result.data["stats"]["total_records"] == 20


@pytest.mark.asyncio
async def test_persistence(skill, tmp_path, monkeypatch):
    """State should persist across instances."""
    records = _make_records(20, 0.2)
    await skill.execute("analyze", {"skill_id": "persist_me", "circuit_data": {"records": records}})
    await skill.execute("tune", {"skill_id": "persist_me"})
    # Create new instance
    skill2 = AdaptiveCircuitThresholdsSkill()
    assert "persist_me" in skill2._profiles
    assert "persist_me" in skill2._overrides
