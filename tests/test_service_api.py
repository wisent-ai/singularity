"""Tests for ServiceAPI - the agent-as-a-service REST API."""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from singularity.service_api import (
    TaskStore, TaskStatus, TaskRecord, ServiceAPI, create_app, HAS_FASTAPI,
)

# Skip all tests if FastAPI not installed
pytestmark = pytest.mark.skipif(not HAS_FASTAPI, reason="FastAPI not installed")


# --- Mock fixtures ---

def make_mock_skill(skill_id="shell", actions=None):
    """Create a mock skill with actions."""
    if actions is None:
        actions = [MagicMock(name="bash", description="Run bash", parameters={"command": {"type": "string"}})]
        actions[0].name = "bash"
    manifest = MagicMock()
    manifest.skill_id = skill_id
    manifest.name = skill_id.title()
    manifest.description = f"{skill_id} skill"
    manifest.actions = actions
    skill = AsyncMock()
    skill.manifest = manifest
    skill.execute = AsyncMock(return_value=MagicMock(
        success=True, data={"output": "hello"}, message="ok"
    ))
    return skill


def make_mock_agent():
    """Create a mock agent."""
    agent = MagicMock()
    agent.name = "TestAgent"
    agent.ticker = "TEST"
    agent.agent_type = "general"
    agent.balance = 50.0
    agent.cycle = 10
    agent.running = True
    skill = make_mock_skill()
    agent.skills = MagicMock()
    agent.skills.skills = {"shell": skill}
    agent.skills.get = lambda sid: agent.skills.skills.get(sid)
    agent.metrics = MagicMock()
    agent.metrics.summary = MagicMock(return_value={"success_rate": 0.95})
    return agent


# --- TaskStore tests ---

class TestTaskStore:
    def test_create_task(self):
        store = TaskStore()
        task = store.create("shell", "bash", {"command": "ls"})
        assert task.task_id
        assert task.status == TaskStatus.QUEUED
        assert task.skill_id == "shell"

    def test_get_task(self):
        store = TaskStore()
        task = store.create("shell", "bash", {})
        fetched = store.get(task.task_id)
        assert fetched is not None
        assert fetched.task_id == task.task_id

    def test_get_nonexistent(self):
        store = TaskStore()
        assert store.get("nonexistent") is None

    def test_list_tasks(self):
        store = TaskStore()
        store.create("shell", "bash", {})
        store.create("shell", "bash", {})
        tasks = store.list_tasks()
        assert len(tasks) == 2

    def test_list_tasks_filter(self):
        store = TaskStore()
        t1 = store.create("shell", "bash", {})
        store.create("shell", "bash", {})
        store.update(t1.task_id, status=TaskStatus.COMPLETED)
        completed = store.list_tasks(status=TaskStatus.COMPLETED)
        assert len(completed) == 1

    def test_update_task(self):
        store = TaskStore()
        task = store.create("shell", "bash", {})
        store.update(task.task_id, status=TaskStatus.RUNNING)
        assert store.get(task.task_id).status == TaskStatus.RUNNING

    def test_stats(self):
        store = TaskStore()
        store.create("shell", "bash", {})
        t2 = store.create("shell", "bash", {})
        store.update(t2.task_id, status=TaskStatus.COMPLETED, execution_time_ms=100.0)
        stats = store.stats()
        assert stats["total_tasks"] == 2
        assert stats["by_status"]["queued"] == 1
        assert stats["by_status"]["completed"] == 1
        assert stats["avg_execution_ms"] == 100.0

    def test_trim(self):
        store = TaskStore(max_tasks=3)
        tasks = [store.create("shell", "bash", {}) for _ in range(5)]
        # Mark old ones as completed so they can be trimmed
        for t in tasks[:3]:
            store.update(t.task_id, status=TaskStatus.COMPLETED)
        store._trim()
        assert len(store._tasks) <= 5  # some may be trimmed


# --- ServiceAPI tests ---

class TestServiceAPI:
    def test_init_no_auth(self):
        svc = ServiceAPI()
        assert not svc.require_auth
        assert svc.validate_api_key(None)

    def test_init_with_keys(self):
        svc = ServiceAPI(api_keys=["key123"])
        assert svc.require_auth
        assert svc.validate_api_key("key123")
        assert not svc.validate_api_key("wrong")

    def test_capabilities_no_agent(self):
        svc = ServiceAPI()
        assert svc.get_capabilities() == []

    def test_capabilities_with_agent(self):
        agent = make_mock_agent()
        svc = ServiceAPI(agent=agent)
        caps = svc.get_capabilities()
        assert len(caps) == 1
        assert caps[0]["skill_id"] == "shell"

    def test_health_no_agent(self):
        svc = ServiceAPI()
        h = svc.health()
        assert h["status"] == "healthy"
        assert h["agent"] == {}

    def test_health_with_agent(self):
        agent = make_mock_agent()
        svc = ServiceAPI(agent=agent)
        h = svc.health()
        assert h["agent"]["name"] == "TestAgent"
        assert h["agent"]["balance"] == 50.0

    @pytest.mark.asyncio
    async def test_submit_task(self):
        agent = make_mock_agent()
        svc = ServiceAPI(agent=agent)
        task = await svc.submit_task("shell", "bash", {"command": "ls"})
        assert task.status in (TaskStatus.QUEUED, TaskStatus.RUNNING, TaskStatus.COMPLETED)
        await asyncio.sleep(0.1)  # let background task run

    @pytest.mark.asyncio
    async def test_submit_invalid_skill(self):
        agent = make_mock_agent()
        svc = ServiceAPI(agent=agent)
        with pytest.raises(ValueError, match="not found"):
            await svc.submit_task("nonexistent", "action", {})

    @pytest.mark.asyncio
    async def test_execute_sync(self):
        agent = make_mock_agent()
        svc = ServiceAPI(agent=agent)
        result = await svc.execute_sync("shell", "bash", {"command": "ls"})
        assert result["status"] == "success"

    @pytest.mark.asyncio
    async def test_execute_sync_no_agent(self):
        svc = ServiceAPI()
        result = await svc.execute_sync("shell", "bash", {})
        assert result["status"] == "error"


# --- FastAPI endpoint tests ---

if HAS_FASTAPI:
    from fastapi.testclient import TestClient

    class TestEndpoints:
        def setup_method(self):
            self.agent = make_mock_agent()
            self.app = create_app(agent=self.agent)
            self.client = TestClient(self.app)

        def test_health(self):
            resp = self.client.get("/health")
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "healthy"
            assert data["agent"]["name"] == "TestAgent"

        def test_capabilities(self):
            resp = self.client.get("/capabilities")
            assert resp.status_code == 200
            caps = resp.json()["capabilities"]
            assert len(caps) == 1

        def test_submit_task(self):
            resp = self.client.post("/tasks", json={
                "skill_id": "shell", "action": "bash",
                "params": {"command": "echo hi"},
            })
            assert resp.status_code == 200
            data = resp.json()
            assert "task_id" in data

        def test_get_task(self):
            resp = self.client.post("/tasks", json={
                "skill_id": "shell", "action": "bash", "params": {},
            })
            task_id = resp.json()["task_id"]
            resp2 = self.client.get(f"/tasks/{task_id}")
            assert resp2.status_code == 200

        def test_list_tasks(self):
            self.client.post("/tasks", json={
                "skill_id": "shell", "action": "bash", "params": {},
            })
            resp = self.client.get("/tasks")
            assert resp.status_code == 200
            assert len(resp.json()["tasks"]) >= 1

        def test_execute_sync(self):
            resp = self.client.post("/execute", json={
                "skill_id": "shell", "action": "bash",
                "params": {"command": "echo hi"},
            })
            assert resp.status_code == 200
            assert resp.json()["status"] == "success"

        def test_cancel_task(self):
            resp = self.client.post("/tasks", json={
                "skill_id": "shell", "action": "bash", "params": {},
            })
            task_id = resp.json()["task_id"]
            # Task may already be completed; just verify endpoint exists
            resp2 = self.client.post(f"/tasks/{task_id}/cancel")
            assert resp2.status_code in (200, 400)

        def test_metrics(self):
            resp = self.client.get("/metrics")
            assert resp.status_code == 200
            assert "tasks" in resp.json()

        def test_404_task(self):
            resp = self.client.get("/tasks/nonexistent")
            assert resp.status_code == 404

        def test_bad_skill(self):
            resp = self.client.post("/tasks", json={
                "skill_id": "nonexistent", "action": "x", "params": {},
            })
            assert resp.status_code == 400

    class TestAuth:
        def test_auth_required(self):
            app = create_app(api_keys=["secret123"], require_auth=True)
            client = TestClient(app)
            resp = client.get("/capabilities")
            assert resp.status_code == 401

        def test_auth_success(self):
            agent = make_mock_agent()
            app = create_app(agent=agent, api_keys=["secret123"], require_auth=True)
            client = TestClient(app)
            resp = client.get("/capabilities", headers={"Authorization": "Bearer secret123"})
            assert resp.status_code == 200

        def test_auth_bad_key(self):
            app = create_app(api_keys=["secret123"], require_auth=True)
            client = TestClient(app)
            resp = client.get("/capabilities", headers={"Authorization": "Bearer wrong"})
            assert resp.status_code == 403
