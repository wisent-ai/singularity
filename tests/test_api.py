"""Tests for the Singularity Service API."""

import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
from singularity.api.models import TaskStatus, TaskSubmission
from singularity.api.server import TaskQueue, create_app


# ---- TaskQueue Tests ----

class TestTaskQueue:
    def setup_method(self):
        self.queue = TaskQueue()

    def test_submit_returns_task_id(self):
        task = TaskSubmission(skill_id="filesystem", action="ls", params={"path": "."})
        task_id = self.queue.submit(task)
        assert task_id.startswith("task_")
        assert len(task_id) == 17  # "task_" + 12 hex chars

    def test_submit_creates_pending_task(self):
        task = TaskSubmission(skill_id="shell", action="bash", params={"command": "echo hi"})
        task_id = self.queue.submit(task)
        data = self.queue.get(task_id)
        assert data["status"] == TaskStatus.pending
        assert data["skill_id"] == "shell"
        assert data["action"] == "bash"

    def test_get_nonexistent_returns_none(self):
        assert self.queue.get("task_nonexistent") is None

    def test_complete_updates_status(self):
        task = TaskSubmission(skill_id="fs", action="ls", params={})
        task_id = self.queue.submit(task)
        self.queue.complete(task_id, {"files": ["a.txt"]}, 42.5)
        data = self.queue.get(task_id)
        assert data["status"] == TaskStatus.completed
        assert data["result"] == {"files": ["a.txt"]}
        assert data["duration_ms"] == 42.5
        assert data["completed_at"] is not None

    def test_fail_updates_status(self):
        task = TaskSubmission(skill_id="fs", action="ls", params={})
        task_id = self.queue.submit(task)
        self.queue.fail(task_id, "Permission denied", 10.0)
        data = self.queue.get(task_id)
        assert data["status"] == TaskStatus.failed
        assert data["error"] == "Permission denied"

    def test_metrics_tracked(self):
        task = TaskSubmission(skill_id="shell", action="bash", params={})
        t1 = self.queue.submit(task)
        t2 = self.queue.submit(task)
        self.queue.complete(t1, {}, 50.0)
        self.queue.fail(t2, "err", 30.0)
        assert self.queue.metrics["total_requests"] == 2
        assert self.queue.metrics["successful_requests"] == 1
        assert self.queue.metrics["failed_requests"] == 1
        assert self.queue.avg_response_time() == 40.0

    def test_list_tasks_with_filter(self):
        task = TaskSubmission(skill_id="fs", action="ls", params={})
        t1 = self.queue.submit(task)
        t2 = self.queue.submit(task)
        self.queue.complete(t1, {}, 10.0)
        pending = self.queue.list_tasks(status=TaskStatus.pending)
        assert len(pending) == 1
        assert pending[0]["task_id"] == t2

    def test_counts(self):
        task = TaskSubmission(skill_id="fs", action="ls", params={})
        t1 = self.queue.submit(task)
        t2 = self.queue.submit(task)
        t3 = self.queue.submit(task)
        self.queue.complete(t1, {}, 10.0)
        self.queue.fail(t2, "err", 10.0)
        assert self.queue.pending_count() == 1
        assert self.queue.completed_count() == 1
        assert self.queue.failed_count() == 1


# ---- API Endpoint Tests ----

def _make_mock_agent():
    """Create a mock agent with filesystem and shell skills."""
    agent = MagicMock()
    agent.name = "TestAgent"
    agent.ticker = "TEST"
    agent.agent_type = "test"
    agent.running = False
    agent.balance = 50.0
    agent.total_api_cost = 0.5
    agent.total_tokens_used = 1000
    agent.cycle = 5

    # Mock skills
    fs_action = MagicMock()
    fs_action.name = "ls"
    fs_action.description = "List directory"
    fs_action.parameters = {"path": {"type": "string"}}

    fs_manifest = MagicMock()
    fs_manifest.skill_id = "filesystem"
    fs_manifest.name = "Filesystem"
    fs_manifest.description = "File operations"
    fs_manifest.actions = [fs_action]

    fs_skill = MagicMock()
    fs_skill.manifest = fs_manifest
    fs_skill.check_credentials.return_value = True

    result = MagicMock()
    result.success = True
    result.data = {"files": ["test.txt"]}
    result.message = "Listed files"
    fs_skill.execute = AsyncMock(return_value=result)

    agent.skills = MagicMock()
    agent.skills.skills = {"filesystem": fs_skill}
    agent.skills.get = lambda sid: {"filesystem": fs_skill}.get(sid)

    return agent


@pytest.fixture
def mock_agent():
    return _make_mock_agent()


@pytest.fixture
def client(mock_agent):
    from fastapi.testclient import TestClient
    app = create_app(mock_agent)
    return TestClient(app)


class TestHealthEndpoint:
    def test_health_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["agent_attached"] is True

    def test_health_no_agent(self):
        from fastapi.testclient import TestClient
        app = create_app(None)
        c = TestClient(app)
        resp = c.get("/health")
        assert resp.status_code == 200
        assert resp.json()["agent_attached"] is False


class TestStatusEndpoint:
    def test_status_returns_agent_info(self, client):
        resp = client.get("/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "TestAgent"
        assert data["ticker"] == "TEST"
        assert data["balance"] == 50.0

    def test_status_no_agent_returns_503(self):
        from fastapi.testclient import TestClient
        app = create_app(None)
        c = TestClient(app)
        resp = c.get("/status")
        assert resp.status_code == 503


class TestSkillsEndpoint:
    def test_list_skills(self, client):
        resp = client.get("/skills")
        assert resp.status_code == 200
        skills = resp.json()
        assert len(skills) == 1
        assert skills[0]["skill_id"] == "filesystem"
        assert len(skills[0]["actions"]) == 1

    def test_get_skill(self, client):
        resp = client.get("/skills/filesystem")
        assert resp.status_code == 200
        assert resp.json()["skill_id"] == "filesystem"

    def test_get_missing_skill(self, client):
        resp = client.get("/skills/nonexistent")
        assert resp.status_code == 404


class TestDirectExecution:
    def test_execute_skill_action(self, client):
        resp = client.post("/skills/filesystem/ls", json={"params": {"path": "."}})
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        assert data["data"] == {"files": ["test.txt"]}
        assert "duration_ms" in data

    def test_execute_missing_skill(self, client):
        resp = client.post("/skills/nonexistent/action", json={"params": {}})
        assert resp.status_code == 404

    def test_execute_missing_action(self, client):
        resp = client.post("/skills/filesystem/nonexistent", json={"params": {}})
        assert resp.status_code == 404


class TestTaskEndpoints:
    def test_submit_task(self, client):
        resp = client.post("/tasks", json={
            "skill_id": "filesystem",
            "action": "ls",
            "params": {"path": "."},
        })
        assert resp.status_code == 201
        data = resp.json()
        assert data["task_id"].startswith("task_")
        assert data["skill_id"] == "filesystem"

    def test_submit_invalid_skill(self, client):
        resp = client.post("/tasks", json={
            "skill_id": "nonexistent",
            "action": "do",
            "params": {},
        })
        assert resp.status_code == 404

    def test_list_tasks(self, client):
        client.post("/tasks", json={
            "skill_id": "filesystem", "action": "ls", "params": {},
        })
        resp = client.get("/tasks")
        assert resp.status_code == 200
        assert len(resp.json()) >= 1

    def test_get_task(self, client):
        resp = client.post("/tasks", json={
            "skill_id": "filesystem", "action": "ls", "params": {},
        })
        task_id = resp.json()["task_id"]
        resp2 = client.get(f"/tasks/{task_id}")
        assert resp2.status_code == 200
        assert resp2.json()["task_id"] == task_id

    def test_get_missing_task(self, client):
        resp = client.get("/tasks/task_nonexistent")
        assert resp.status_code == 404


class TestMetrics:
    def test_metrics_endpoint(self, client):
        # Execute a skill action to generate metrics
        client.post("/skills/filesystem/ls", json={"params": {"path": "."}})
        resp = client.get("/metrics")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_requests"] >= 1
        assert "uptime_seconds" in data
