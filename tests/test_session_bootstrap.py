"""Tests for SessionBootstrapSkill - unified session startup orchestration."""
import json
import pytest
from datetime import datetime, timedelta
from pathlib import Path

from singularity.skills.session_bootstrap import (
    SessionBootstrapSkill, SESSION_HISTORY_FILE,
    PERF_FILE, FEEDBACK_FILE, STRATEGY_FILE, GOALS_FILE,
)


@pytest.fixture(autouse=True)
def clean_data(tmp_path, monkeypatch):
    """Use temp files for all tests."""
    mod = "singularity.skills.session_bootstrap"
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setattr(f"{mod}.DATA_DIR", data_dir)
    monkeypatch.setattr(f"{mod}.PERF_FILE", data_dir / "performance.json")
    monkeypatch.setattr(f"{mod}.FEEDBACK_FILE", data_dir / "feedback_loop.json")
    monkeypatch.setattr(f"{mod}.STRATEGY_FILE", data_dir / "strategy.json")
    monkeypatch.setattr(f"{mod}.GOALS_FILE", data_dir / "goals.json")
    monkeypatch.setattr(f"{mod}.SESSION_HISTORY_FILE", data_dir / "session_history.json")
    return data_dir


def _write_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f)


@pytest.mark.asyncio
async def test_boot_cold_start(clean_data):
    """Boot with no data should give cold start recommendation."""
    skill = SessionBootstrapSkill()
    result = await skill.execute("boot", {})
    assert result.success
    assert "Session #1" in result.message
    assert result.data["recommendation"]["source"] == "cold_start"


@pytest.mark.asyncio
async def test_boot_with_user_focus(clean_data):
    """User-specified focus should take priority."""
    skill = SessionBootstrapSkill()
    result = await skill.execute("boot", {"session_focus": "revenue"})
    assert result.success
    assert result.data["recommendation"]["source"] == "user_focus"
    assert result.data["recommendation"]["focus_pillar"] == "revenue"


@pytest.mark.asyncio
async def test_boot_with_strategy_data(clean_data):
    """Boot should detect weakest pillar from strategy data."""
    _write_json(clean_data / "strategy.json", {
        "pillars": {
            "self_improvement": {"score": 60, "capabilities": [], "gaps": []},
            "revenue": {"score": 10, "capabilities": [], "gaps": ["Need payment system"]},
            "replication": {"score": 40, "capabilities": [], "gaps": []},
            "goal_setting": {"score": 50, "capabilities": [], "gaps": []},
        },
        "recommendations": [],
        "journal": [], "work_log": [],
    })
    skill = SessionBootstrapSkill()
    result = await skill.execute("boot", {})
    assert result.success
    assert result.data["strategy"]["weakest_pillar"] == "revenue"
    assert result.data["strategy"]["weakest_score"] == 10


@pytest.mark.asyncio
async def test_boot_with_goals(clean_data):
    """Boot should find next actionable goal."""
    _write_json(clean_data / "goals.json", {
        "goals": [{
            "id": "goal_abc123",
            "title": "Build payment API",
            "pillar": "revenue",
            "priority": "high",
            "priority_score": 3,
            "status": "active",
            "depends_on": [],
            "milestones": [
                {"index": 0, "title": "Design API", "completed": True},
                {"index": 1, "title": "Implement endpoints", "completed": False},
            ],
        }],
        "completed_goals": [],
    })
    skill = SessionBootstrapSkill()
    result = await skill.execute("boot", {})
    assert result.success
    assert result.data["goals"]["next_goal_title"] == "Build payment API"
    assert result.data["goals"]["next_milestone"] == "Implement endpoints"
    assert result.data["recommendation"]["source"] == "goal_manager"


@pytest.mark.asyncio
async def test_boot_with_performance(clean_data):
    """Boot should analyze performance records."""
    now = datetime.now()
    records = [
        {"skill_id": "shell", "success": True, "cost": 0.01, "revenue": 0,
         "timestamp": (now - timedelta(hours=i)).isoformat()}
        for i in range(5)
    ] + [
        {"skill_id": "browser", "success": False, "cost": 0.05, "revenue": 0,
         "timestamp": (now - timedelta(hours=i)).isoformat()}
        for i in range(3)
    ]
    _write_json(clean_data / "performance.json", {
        "records": records, "sessions": [], "insights": [],
    })
    skill = SessionBootstrapSkill()
    result = await skill.execute("boot", {})
    assert result.success
    assert result.data["performance"]["available"]
    assert result.data["performance"]["recent_action_count"] == 8


@pytest.mark.asyncio
async def test_quick_boot(clean_data):
    """Quick boot should skip performance and feedback."""
    skill = SessionBootstrapSkill()
    result = await skill.execute("quick", {})
    assert result.success
    assert "Quick Boot" in result.message
    assert "performance" not in result.data


@pytest.mark.asyncio
async def test_log_outcome_and_recap(clean_data):
    """Log outcome and then recap should show it."""
    skill = SessionBootstrapSkill()
    # First boot to create a session
    await skill.execute("boot", {})
    # Log outcome
    result = await skill.execute("log_outcome", {
        "summary": "Built the payment API",
        "pillar": "revenue",
        "next_steps": ["Add tests", "Deploy to prod"],
    })
    assert result.success
    assert "payment API" in result.message
    # Recap
    result = await skill.execute("recap", {"count": 5})
    assert result.success
    assert len(result.data["sessions"]) == 1
    assert result.data["sessions"][0]["completed"]


@pytest.mark.asyncio
async def test_last_session_continuity(clean_data):
    """Boot should continue from last session's next_steps."""
    skill = SessionBootstrapSkill()
    # Boot and log first session
    await skill.execute("boot", {})
    await skill.execute("log_outcome", {
        "summary": "Started payment API",
        "pillar": "revenue",
        "next_steps": ["Finish payment endpoints"],
    })
    # Boot second session - should pick up next_steps
    result = await skill.execute("boot", {})
    assert result.success
    assert result.data["recommendation"]["source"] == "last_session_continuity"
    assert "Finish payment endpoints" in result.data["recommendation"]["action"]


@pytest.mark.asyncio
async def test_session_numbering(clean_data):
    """Sessions should be numbered incrementally."""
    skill = SessionBootstrapSkill()
    r1 = await skill.execute("boot", {})
    assert r1.data["session_number"] == 1
    r2 = await skill.execute("boot", {})
    assert r2.data["session_number"] == 2


@pytest.mark.asyncio
async def test_log_without_boot_fails(clean_data):
    """Logging outcome without booting should fail."""
    skill = SessionBootstrapSkill()
    result = await skill.execute("log_outcome", {"summary": "test"})
    assert not result.success


@pytest.mark.asyncio
async def test_boot_with_feedback_data(clean_data):
    """Boot should read feedback adaptations."""
    _write_json(clean_data / "feedback_loop.json", {
        "adaptations": [
            {"status": "active", "type": "prefer_skill", "description": "Use shell over browser", "applied_at": "2024-01-01"},
            {"status": "successful", "type": "avoid_action", "description": "Skip flaky test", "applied_at": "2024-01-01"},
        ],
        "reviews": [{"patterns": ["shell_preferred", "browser_unreliable"]}],
        "adaptation_outcomes": [],
    })
    skill = SessionBootstrapSkill()
    result = await skill.execute("boot", {})
    assert result.success
    assert result.data["feedback"]["active_adaptations"] == 1
    assert result.data["feedback"]["successful_adaptations"] == 1
