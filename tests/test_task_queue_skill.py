"""Tests for TaskQueueSkill."""
import pytest
import tempfile
import os
from singularity.skills.task_queue import TaskQueueSkill


@pytest.fixture
def skill(tmp_path):
    return TaskQueueSkill(queue_dir=str(tmp_path / "queue"))


@pytest.mark.asyncio
async def test_enqueue_basic(skill):
    r = await skill.execute("enqueue", {"title": "Test task", "priority": "high"})
    assert r.success
    assert "task_id" in r.data
    assert r.data["priority"] == "high"


@pytest.mark.asyncio
async def test_enqueue_empty_title(skill):
    r = await skill.execute("enqueue", {"title": ""})
    assert not r.success


@pytest.mark.asyncio
async def test_dequeue_priority_order(skill):
    await skill.execute("enqueue", {"title": "Low task", "priority": "low"})
    await skill.execute("enqueue", {"title": "Critical task", "priority": "critical"})
    r = await skill.execute("dequeue", {})
    assert r.success
    assert r.data["title"] == "Critical task"


@pytest.mark.asyncio
async def test_complete_task(skill):
    r1 = await skill.execute("enqueue", {"title": "Do something"})
    tid = r1.data["task_id"]
    await skill.execute("dequeue", {})
    r = await skill.execute("complete", {"task_id": tid, "result": "Done!"})
    assert r.success
    get_r = await skill.execute("get", {"task_id": tid})
    assert get_r.data["task"]["status"] == "completed"


@pytest.mark.asyncio
async def test_fail_with_retry(skill):
    r1 = await skill.execute("enqueue", {"title": "Flaky task", "max_retries": 2})
    tid = r1.data["task_id"]
    await skill.execute("dequeue", {})
    r = await skill.execute("fail", {"task_id": tid, "reason": "timeout"})
    assert r.success
    assert r.data["will_retry"] is True
    get_r = await skill.execute("get", {"task_id": tid})
    assert get_r.data["task"]["status"] == "pending"


@pytest.mark.asyncio
async def test_fail_exhausted_retries(skill):
    r1 = await skill.execute("enqueue", {"title": "Bad task", "max_retries": 1})
    tid = r1.data["task_id"]
    await skill.execute("dequeue", {})
    r = await skill.execute("fail", {"task_id": tid, "reason": "error"})
    assert r.data["will_retry"] is False
    get_r = await skill.execute("get", {"task_id": tid})
    assert get_r.data["task"]["status"] == "failed"


@pytest.mark.asyncio
async def test_cancel_task(skill):
    r1 = await skill.execute("enqueue", {"title": "Cancel me"})
    tid = r1.data["task_id"]
    r = await skill.execute("cancel", {"task_id": tid, "reason": "No longer needed"})
    assert r.success
    get_r = await skill.execute("get", {"task_id": tid})
    assert get_r.data["task"]["status"] == "cancelled"


@pytest.mark.asyncio
async def test_dependencies_blocking(skill):
    r1 = await skill.execute("enqueue", {"title": "Step 1"})
    tid1 = r1.data["task_id"]
    r2 = await skill.execute("enqueue", {"title": "Step 2", "depends_on": tid1})
    assert r2.data["status"] == "blocked"
    # Can't dequeue blocked task
    await skill.execute("dequeue", {})  # Gets Step 1
    r = await skill.execute("dequeue", {})
    assert not r.success  # Step 2 still blocked


@pytest.mark.asyncio
async def test_dependencies_unblock(skill):
    r1 = await skill.execute("enqueue", {"title": "Step 1"})
    tid1 = r1.data["task_id"]
    r2 = await skill.execute("enqueue", {"title": "Step 2", "depends_on": tid1})
    tid2 = r2.data["task_id"]
    await skill.execute("dequeue", {})  # Gets Step 1
    await skill.execute("complete", {"task_id": tid1})
    r = await skill.execute("dequeue", {})  # Should get Step 2 now
    assert r.success
    assert r.data["task_id"] == tid2


@pytest.mark.asyncio
async def test_list_filter_by_status(skill):
    await skill.execute("enqueue", {"title": "T1"})
    r1 = await skill.execute("enqueue", {"title": "T2"})
    await skill.execute("dequeue", {})
    r = await skill.execute("list", {"status": "in_progress"})
    assert r.data["total"] == 1


@pytest.mark.asyncio
async def test_stats(skill):
    await skill.execute("enqueue", {"title": "T1", "category": "revenue"})
    await skill.execute("enqueue", {"title": "T2", "category": "improvement"})
    r = await skill.execute("stats", {})
    assert r.success
    assert r.data["total_tasks"] == 2
    assert "revenue" in r.data["by_category"]


@pytest.mark.asyncio
async def test_prioritize(skill):
    r1 = await skill.execute("enqueue", {"title": "Task", "priority": "low"})
    tid = r1.data["task_id"]
    r = await skill.execute("prioritize", {"task_id": tid, "priority": "critical"})
    assert r.success
    assert r.data["new_priority"] == "critical"


@pytest.mark.asyncio
async def test_persistence(tmp_path):
    """Tasks survive skill restart."""
    qdir = str(tmp_path / "queue")
    s1 = TaskQueueSkill(queue_dir=qdir)
    await s1.execute("enqueue", {"title": "Persistent task"})
    s2 = TaskQueueSkill(queue_dir=qdir)
    r = await s2.execute("list", {})
    assert r.data["total"] == 1
    assert r.data["tasks"][0]["title"] == "Persistent task"


@pytest.mark.asyncio
async def test_dequeue_empty(skill):
    r = await skill.execute("dequeue", {})
    assert not r.success


@pytest.mark.asyncio
async def test_unknown_action(skill):
    r = await skill.execute("nonexistent", {})
    assert not r.success


@pytest.mark.asyncio
async def test_category_filter_dequeue(skill):
    await skill.execute("enqueue", {"title": "Revenue task", "category": "revenue"})
    await skill.execute("enqueue", {"title": "Dev task", "category": "dev"})
    r = await skill.execute("dequeue", {"category": "dev"})
    assert r.success
    assert r.data["category"] == "dev"
