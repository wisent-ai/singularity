#!/usr/bin/env python3
"""Tests for TaskDelegator skill."""

import pytest
import json
import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from singularity.skills.task_delegator import TaskDelegator, DELEGATOR_FILE
from singularity.skills.base import SkillResult


@pytest.fixture(autouse=True)
def clean_data(tmp_path, monkeypatch):
    """Use temp file for each test."""
    test_file = tmp_path / "task_delegator.json"
    monkeypatch.setattr("singularity.skills.task_delegator.DELEGATOR_FILE", test_file)
    yield test_file


@pytest.fixture
def delegator():
    return TaskDelegator()


@pytest.mark.asyncio
async def test_create_pool_basic(delegator):
    result = await delegator.execute("create_pool", {
        "name": "Test Pool",
        "subtasks": [
            {"name": "task_a", "skill_id": "content", "action": "generate"},
            {"name": "task_b", "skill_id": "content", "action": "summarize"},
        ],
    })
    assert result.success
    assert result.data["subtask_count"] == 2
    assert result.data["pattern"] == "fan_out"
    assert "pool_id" in result.data


@pytest.mark.asyncio
async def test_create_pool_pipeline_sets_deps(delegator):
    result = await delegator.execute("create_pool", {
        "name": "Pipeline",
        "pattern": "pipeline",
        "subtasks": [
            {"name": "step1"},
            {"name": "step2"},
            {"name": "step3"},
        ],
    })
    assert result.success
    pool_id = result.data["pool_id"]

    status = await delegator.execute("pool_status", {"pool_id": pool_id})
    subtasks = status.data["subtasks"]
    # step2 should depend on step1, step3 on step2
    assert len(subtasks) == 3


@pytest.mark.asyncio
async def test_create_pool_requires_name(delegator):
    result = await delegator.execute("create_pool", {"subtasks": [{"name": "a"}]})
    assert not result.success


@pytest.mark.asyncio
async def test_create_pool_requires_subtasks(delegator):
    result = await delegator.execute("create_pool", {"name": "Empty", "subtasks": []})
    assert not result.success


@pytest.mark.asyncio
async def test_run_pool(delegator):
    create = await delegator.execute("create_pool", {
        "name": "Run Test",
        "subtasks": [{"name": "simple_task"}],
    })
    pool_id = create.data["pool_id"]

    run = await delegator.execute("run_pool", {"pool_id": pool_id})
    assert run.success
    assert run.data["status"] == "running"


@pytest.mark.asyncio
async def test_report_subtask_completion(delegator):
    create = await delegator.execute("create_pool", {
        "name": "Report Test",
        "subtasks": [{"name": "task_a"}],
    })
    pool_id = create.data["pool_id"]
    subtask_id = create.data["subtask_ids"][0]

    await delegator.execute("run_pool", {"pool_id": pool_id})

    result = await delegator.execute("report_subtask", {
        "pool_id": pool_id,
        "subtask_id": subtask_id,
        "status": "completed",
        "result": {"output": "done"},
    })
    assert result.success
    assert result.data["pool_status"] == "completed"


@pytest.mark.asyncio
async def test_report_subtask_failure(delegator):
    create = await delegator.execute("create_pool", {
        "name": "Fail Test",
        "subtasks": [{"name": "failing_task"}],
    })
    pool_id = create.data["pool_id"]
    subtask_id = create.data["subtask_ids"][0]

    await delegator.execute("run_pool", {"pool_id": pool_id})

    result = await delegator.execute("report_subtask", {
        "pool_id": pool_id,
        "subtask_id": subtask_id,
        "status": "failed",
        "error": "Something broke",
    })
    assert result.success
    assert result.data["pool_status"] == "failed"


@pytest.mark.asyncio
async def test_cancel_pool(delegator):
    create = await delegator.execute("create_pool", {
        "name": "Cancel Test",
        "subtasks": [{"name": "a"}, {"name": "b"}],
    })
    pool_id = create.data["pool_id"]
    await delegator.execute("run_pool", {"pool_id": pool_id})

    cancel = await delegator.execute("cancel_pool", {
        "pool_id": pool_id,
        "reason": "No longer needed",
    })
    assert cancel.success
    assert cancel.data["cancelled_subtasks"] == 2


@pytest.mark.asyncio
async def test_retry_failed_subtasks(delegator):
    create = await delegator.execute("create_pool", {
        "name": "Retry Test",
        "subtasks": [{"name": "flaky_task"}],
    })
    pool_id = create.data["pool_id"]
    subtask_id = create.data["subtask_ids"][0]

    await delegator.execute("run_pool", {"pool_id": pool_id})
    await delegator.execute("report_subtask", {
        "pool_id": pool_id,
        "subtask_id": subtask_id,
        "status": "failed",
        "error": "Transient error",
    })

    retry = await delegator.execute("retry_failed", {"pool_id": pool_id})
    assert retry.success
    assert len(retry.data["retried"]) == 1

    # Verify subtask is pending again
    status = await delegator.execute("pool_status", {"pool_id": pool_id})
    assert status.data["summary"].get("pending", 0) == 1


@pytest.mark.asyncio
async def test_get_results_fan_out(delegator):
    create = await delegator.execute("create_pool", {
        "name": "Results Test",
        "pattern": "fan_out",
        "subtasks": [{"name": "a"}, {"name": "b"}],
    })
    pool_id = create.data["pool_id"]
    ids = create.data["subtask_ids"]

    await delegator.execute("run_pool", {"pool_id": pool_id})
    await delegator.execute("report_subtask", {
        "pool_id": pool_id, "subtask_id": ids[0],
        "status": "completed", "result": {"val": 1},
    })
    await delegator.execute("report_subtask", {
        "pool_id": pool_id, "subtask_id": ids[1],
        "status": "completed", "result": {"val": 2},
    })

    results = await delegator.execute("get_results", {"pool_id": pool_id})
    assert results.success
    assert len(results.data["results"]) == 2
    assert results.data["aggregated"]["a"]["val"] == 1


@pytest.mark.asyncio
async def test_get_results_map_reduce(delegator):
    create = await delegator.execute("create_pool", {
        "name": "MapReduce",
        "pattern": "map_reduce",
        "subtasks": [{"name": "map1"}, {"name": "map2"}],
    })
    pool_id = create.data["pool_id"]
    ids = create.data["subtask_ids"]

    await delegator.execute("run_pool", {"pool_id": pool_id})
    for i, sid in enumerate(ids):
        await delegator.execute("report_subtask", {
            "pool_id": pool_id, "subtask_id": sid,
            "status": "completed", "result": {"n": i},
        })

    results = await delegator.execute("get_results", {"pool_id": pool_id})
    assert results.success
    assert isinstance(results.data["aggregated"], list)
    assert len(results.data["aggregated"]) == 2


@pytest.mark.asyncio
async def test_list_pools(delegator):
    await delegator.execute("create_pool", {
        "name": "Pool A", "subtasks": [{"name": "x"}],
    })
    await delegator.execute("create_pool", {
        "name": "Pool B", "subtasks": [{"name": "y"}],
    })

    listing = await delegator.execute("list_pools", {})
    assert listing.success
    assert listing.data["total"] == 2


@pytest.mark.asyncio
async def test_list_pools_filter_status(delegator):
    create = await delegator.execute("create_pool", {
        "name": "Active", "subtasks": [{"name": "x"}],
    })
    await delegator.execute("run_pool", {"pool_id": create.data["pool_id"]})
    await delegator.execute("create_pool", {
        "name": "Idle", "subtasks": [{"name": "y"}],
    })

    listing = await delegator.execute("list_pools", {"status": "running"})
    assert listing.data["total"] == 1


@pytest.mark.asyncio
async def test_stats(delegator):
    create = await delegator.execute("create_pool", {
        "name": "Stats Test", "subtasks": [{"name": "a"}],
    })
    pool_id = create.data["pool_id"]
    subtask_id = create.data["subtask_ids"][0]

    await delegator.execute("run_pool", {"pool_id": pool_id})
    await delegator.execute("report_subtask", {
        "pool_id": pool_id, "subtask_id": subtask_id,
        "status": "completed", "result": {},
    })

    stats = await delegator.execute("stats", {})
    assert stats.success
    assert stats.data["stats"]["total_pools_created"] == 1
    assert stats.data["stats"]["total_subtasks_completed"] == 1


@pytest.mark.asyncio
async def test_invalid_pattern(delegator):
    result = await delegator.execute("create_pool", {
        "name": "Bad", "pattern": "invalid", "subtasks": [{"name": "a"}],
    })
    assert not result.success


@pytest.mark.asyncio
async def test_unknown_action(delegator):
    result = await delegator.execute("nonexistent", {})
    assert not result.success


@pytest.mark.asyncio
async def test_manifest(delegator):
    m = delegator.manifest
    assert m.skill_id == "task_delegator"
    assert len(m.actions) == 9


@pytest.mark.asyncio
async def test_max_retries_enforced(delegator):
    create = await delegator.execute("create_pool", {
        "name": "Max Retry", "subtasks": [{"name": "flaky"}],
    })
    pool_id = create.data["pool_id"]
    subtask_id = create.data["subtask_ids"][0]

    # Fail and retry 3 times
    for _ in range(3):
        await delegator.execute("run_pool", {"pool_id": pool_id})
        await delegator.execute("report_subtask", {
            "pool_id": pool_id, "subtask_id": subtask_id,
            "status": "failed", "error": "still broken",
        })
        await delegator.execute("retry_failed", {"pool_id": pool_id})

    # 4th retry should be skipped
    await delegator.execute("report_subtask", {
        "pool_id": pool_id, "subtask_id": subtask_id,
        "status": "failed", "error": "still broken",
    })
    retry = await delegator.execute("retry_failed", {"pool_id": pool_id})
    assert len(retry.data["skipped"]) == 1
