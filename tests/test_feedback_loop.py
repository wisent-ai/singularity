"""Tests for FeedbackLoopSkill - the adapt component of act->measure->adapt."""
import json
import pytest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

from singularity.skills.feedback_loop import FeedbackLoopSkill, FEEDBACK_FILE, PERF_FILE


@pytest.fixture(autouse=True)
def clean_data(tmp_path, monkeypatch):
    """Use temp files for all tests."""
    fb_file = tmp_path / "feedback_loop.json"
    pf_file = tmp_path / "performance.json"
    monkeypatch.setattr("singularity.skills.feedback_loop.FEEDBACK_FILE", fb_file)
    monkeypatch.setattr("singularity.skills.feedback_loop.PERF_FILE", pf_file)
    return fb_file, pf_file


def _write_perf_data(pf_file, records):
    """Helper to write performance records."""
    pf_file.parent.mkdir(parents=True, exist_ok=True)
    with open(pf_file, "w") as f:
        json.dump({"records": records, "sessions": [], "insights": []}, f)


def _make_records(skill_id, action, count, success_rate, cost=0.01, latency=100):
    """Generate performance records."""
    now = datetime.now()
    records = []
    for i in range(count):
        records.append({
            "skill_id": skill_id,
            "action": action,
            "success": i < int(count * success_rate),
            "cost_usd": cost,
            "latency_ms": latency,
            "timestamp": (now - timedelta(hours=i)).isoformat(),
            "error": "" if i < int(count * success_rate) else "test error",
        })
    return records


@pytest.mark.asyncio
async def test_review_no_data(clean_data):
    fb_file, pf_file = clean_data
    skill = FeedbackLoopSkill()
    result = await skill.execute("review", {})
    assert result.success
    assert result.data["record_count"] == 0


@pytest.mark.asyncio
async def test_review_detects_failing_skill(clean_data):
    fb_file, pf_file = clean_data
    records = _make_records("shell", "run", 10, 0.3)
    _write_perf_data(pf_file, records)

    skill = FeedbackLoopSkill()
    result = await skill.execute("review", {"window_hours": 48})
    assert result.success
    suggestions = result.data["suggestions"]
    types = [s["type"] for s in suggestions]
    assert "avoid_failing_skill" in types


@pytest.mark.asyncio
async def test_review_detects_successful_skill(clean_data):
    fb_file, pf_file = clean_data
    records = _make_records("content", "generate", 10, 1.0)
    _write_perf_data(pf_file, records)

    skill = FeedbackLoopSkill()
    result = await skill.execute("review", {"window_hours": 48})
    assert result.success
    types = [s["type"] for s in result.data["suggestions"]]
    assert "prefer_successful_skill" in types


@pytest.mark.asyncio
async def test_apply_adaptation(clean_data):
    fb_file, pf_file = clean_data
    records = _make_records("shell", "run", 10, 0.2)
    _write_perf_data(pf_file, records)

    prompt_additions = []
    skill = FeedbackLoopSkill()
    skill.set_cognition_hooks(
        append_prompt=lambda text: prompt_additions.append(text),
        get_prompt=lambda: "base prompt",
    )

    # First review
    await skill.execute("review", {"window_hours": 48})

    # Get suggestions
    fb_data = skill._load()
    pending = [a for a in fb_data["adaptations"] if not a.get("applied")]
    assert len(pending) > 0

    # Apply one
    result = await skill.execute("apply", {"adaptation_id": pending[0]["id"]})
    assert result.success
    assert len(prompt_additions) > 0
    assert "LEARNED ADAPTATION" in prompt_additions[0]


@pytest.mark.asyncio
async def test_apply_all(clean_data):
    fb_file, pf_file = clean_data
    records = _make_records("shell", "run", 10, 0.2)
    _write_perf_data(pf_file, records)

    skill = FeedbackLoopSkill()
    skill.set_cognition_hooks(
        append_prompt=lambda text: None,
        get_prompt=lambda: "base",
    )

    await skill.execute("review", {"window_hours": 48})
    result = await skill.execute("apply_all", {})
    assert result.success
    assert len(result.data.get("applied", [])) > 0 or result.data.get("total_pending", 0) > 0


@pytest.mark.asyncio
async def test_status(clean_data):
    skill = FeedbackLoopSkill()
    result = await skill.execute("status", {})
    assert result.success
    assert "total_adaptations" in result.data


@pytest.mark.asyncio
async def test_revert_unapplied(clean_data):
    fb_file, pf_file = clean_data
    records = _make_records("shell", "run", 10, 0.2)
    _write_perf_data(pf_file, records)

    skill = FeedbackLoopSkill()
    await skill.execute("review", {"window_hours": 48})

    fb_data = skill._load()
    if fb_data["adaptations"]:
        adapt_id = fb_data["adaptations"][0]["id"]
        result = await skill.execute("revert", {"adaptation_id": adapt_id})
        assert result.success


@pytest.mark.asyncio
async def test_evaluate_no_pending(clean_data):
    skill = FeedbackLoopSkill()
    result = await skill.execute("evaluate", {})
    assert result.success
    assert result.data["evaluated"] == 0


@pytest.mark.asyncio
async def test_manifest_and_credentials():
    skill = FeedbackLoopSkill()
    assert skill.manifest.skill_id == "feedback"
    assert skill.check_credentials()
    assert len(skill.manifest.actions) == 6
