"""Tests for TaskDelegationSkill."""
import pytest
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from singularity.skills.task_delegation import TaskDelegationSkill, DELEGATION_FILE


@pytest.fixture(autouse=True)
def clean_data(tmp_path, monkeypatch):
    test_file = tmp_path / "task_delegations.json"
    monkeypatch.setattr("singularity.skills.task_delegation.DELEGATION_FILE", test_file)
    yield test_file


@pytest.fixture
def skill():
    s = TaskDelegationSkill()
    s.initialized = True
    return s


@pytest.mark.asyncio
async def test_manifest(skill):
    m = skill.manifest
    assert m.skill_id == "task_delegation"
    action_names = [a.name for a in m.actions]
    assert "delegate" in action_names
    assert "spawn_for" in action_names
    assert "check" in action_names
    assert "recall" in action_names
    assert "batch" in action_names
    assert "ledger" in action_names


@pytest.mark.asyncio
async def test_delegate_basic(skill):
    r = await skill.execute("delegate", {
        "task_name": "Analyze data",
        "task_description": "Run analysis on Q4 sales data",
        "budget": 5.0,
    })
    assert r.success
    assert r.data["delegation_id"].startswith("dlg_")
    assert r.data["budget"] == 5.0
    assert r.data["status"] == "pending"


@pytest.mark.asyncio
async def test_delegate_validation(skill):
    r = await skill.execute("delegate", {"task_description": "x", "budget": 1})
    assert not r.success
    r = await skill.execute("delegate", {"task_name": "x", "budget": 1})
    assert not r.success
    r = await skill.execute("delegate", {"task_name": "x", "task_description": "y", "budget": 0})
    assert not r.success
    r = await skill.execute("delegate", {"task_name": "x", "task_description": "y", "budget": 999})
    assert not r.success


@pytest.mark.asyncio
async def test_delegate_with_agent_id(skill):
    r = await skill.execute("delegate", {
        "task_name": "Test task",
        "task_description": "Do something",
        "budget": 2.0,
        "agent_id": "agent-42",
        "priority": "high",
    })
    assert r.success
    assert r.data["agent_id"] == "agent-42"  # Agent ID stored even without context
    assert r.data["priority"] == "high"


@pytest.mark.asyncio
async def test_check_status(skill):
    r = await skill.execute("delegate", {
        "task_name": "Check me",
        "task_description": "A task to check",
        "budget": 3.0,
    })
    dlg_id = r.data["delegation_id"]
    r2 = await skill.execute("check", {"delegation_id": dlg_id})
    assert r2.success
    assert r2.data["status"] == "pending"
    assert r2.data["task_name"] == "Check me"


@pytest.mark.asyncio
async def test_check_not_found(skill):
    r = await skill.execute("check", {"delegation_id": "nonexistent"})
    assert not r.success


@pytest.mark.asyncio
async def test_recall(skill):
    r = await skill.execute("delegate", {
        "task_name": "Recall me",
        "task_description": "To be recalled",
        "budget": 10.0,
    })
    dlg_id = r.data["delegation_id"]
    r2 = await skill.execute("recall", {"delegation_id": dlg_id, "reason": "No longer needed"})
    assert r2.success
    assert r2.data["budget_reclaimed"] == 10.0
    # Check it's actually recalled
    r3 = await skill.execute("check", {"delegation_id": dlg_id})
    assert r3.data["status"] == "recalled"


@pytest.mark.asyncio
async def test_recall_completed_fails(skill):
    r = await skill.execute("delegate", {
        "task_name": "Done task",
        "task_description": "Already done",
        "budget": 5.0,
    })
    dlg_id = r.data["delegation_id"]
    await skill.execute("report_completion", {
        "delegation_id": dlg_id, "status": "completed", "result": {"ok": True},
    })
    r2 = await skill.execute("recall", {"delegation_id": dlg_id})
    assert not r2.success


@pytest.mark.asyncio
async def test_report_completion(skill):
    r = await skill.execute("delegate", {
        "task_name": "Complete me",
        "task_description": "Task to complete",
        "budget": 8.0,
    })
    dlg_id = r.data["delegation_id"]
    r2 = await skill.execute("report_completion", {
        "delegation_id": dlg_id,
        "status": "completed",
        "result": {"output": "success"},
        "budget_spent": 3.0,
    })
    assert r2.success
    assert r2.data["budget_spent"] == 3.0
    assert r2.data["budget_reclaimed"] == 5.0


@pytest.mark.asyncio
async def test_report_failure(skill):
    r = await skill.execute("delegate", {
        "task_name": "Fail task",
        "task_description": "Will fail",
        "budget": 4.0,
    })
    dlg_id = r.data["delegation_id"]
    r2 = await skill.execute("report_completion", {
        "delegation_id": dlg_id,
        "status": "failed",
        "error": "Something broke",
        "budget_spent": 1.5,
    })
    assert r2.success
    assert r2.data["status"] == "failed"


@pytest.mark.asyncio
async def test_results(skill):
    # Create and complete two tasks
    for name in ["Task A", "Task B"]:
        r = await skill.execute("delegate", {
            "task_name": name, "task_description": f"Do {name}", "budget": 2.0,
        })
        await skill.execute("report_completion", {
            "delegation_id": r.data["delegation_id"],
            "status": "completed",
            "result": {"name": name},
        })
    r = await skill.execute("results", {})
    assert r.success
    assert r.data["count"] == 2


@pytest.mark.asyncio
async def test_ledger(skill):
    await skill.execute("delegate", {
        "task_name": "T1", "task_description": "D1", "budget": 10.0,
    })
    await skill.execute("delegate", {
        "task_name": "T2", "task_description": "D2", "budget": 5.0,
    })
    r = await skill.execute("ledger", {})
    assert r.success
    assert r.data["ledger"]["total_budget_allocated"] == 15.0
    assert len(r.data["active_delegations"]) == 2


@pytest.mark.asyncio
async def test_batch_equal(skill):
    tasks = [
        {"task_name": f"Batch {i}", "task_description": f"Desc {i}", "budget": 1.0}
        for i in range(3)
    ]
    r = await skill.execute("batch", {
        "tasks": tasks, "total_budget": 9.0, "strategy": "equal",
    })
    assert r.success
    assert r.data["successes"] == 3


@pytest.mark.asyncio
async def test_batch_priority_based(skill):
    tasks = [
        {"task_name": "Critical", "task_description": "Urgent", "budget": 1, "priority": "critical"},
        {"task_name": "Low", "task_description": "Not urgent", "budget": 1, "priority": "low"},
    ]
    r = await skill.execute("batch", {
        "tasks": tasks, "total_budget": 10.0, "strategy": "priority_based",
    })
    assert r.success
    assert r.data["successes"] == 2


@pytest.mark.asyncio
async def test_history(skill):
    for i in range(5):
        await skill.execute("delegate", {
            "task_name": f"H{i}", "task_description": f"History {i}", "budget": 1.0,
        })
    r = await skill.execute("history", {"limit": 3})
    assert r.success
    assert r.data["count"] == 3


@pytest.mark.asyncio
async def test_history_filter(skill):
    r = await skill.execute("delegate", {
        "task_name": "Pending", "task_description": "stays pending", "budget": 1.0,
    })
    dlg_id = r.data["delegation_id"]
    await skill.execute("report_completion", {
        "delegation_id": dlg_id, "status": "completed",
    })
    r = await skill.execute("history", {"status": "completed"})
    assert r.success
    assert all(h["status"] == "completed" for h in r.data["history"])


@pytest.mark.asyncio
async def test_spawn_for_no_context(skill):
    r = await skill.execute("spawn_for", {
        "task_name": "Spawn task",
        "task_description": "Need a worker",
        "budget": 5.0,
    })
    assert r.success
    assert r.data["spawned"] is False
    assert r.data["status"] == "pending"


@pytest.mark.asyncio
async def test_unknown_action(skill):
    r = await skill.execute("nonexistent", {})
    assert not r.success


@pytest.mark.asyncio
async def test_initialize(skill):
    s = TaskDelegationSkill()
    assert await s.initialize() is True
