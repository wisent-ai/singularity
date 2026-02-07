"""
Singularity Service API Server.

Exposes agent skills as REST endpoints, enabling external clients to:
- Submit tasks for async processing
- Execute skill actions directly
- Monitor agent status and metrics
- Browse available skills and actions

This is the foundation for revenue generation - external clients
can discover and use agent capabilities via this API.
"""

import asyncio
import time
import uuid
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional

from .models import (
    AgentStatus,
    DirectExecuteRequest,
    ErrorResponse,
    ServiceMetrics,
    SkillInfo,
    TaskResponse,
    TaskStatus,
    TaskSubmission,
)


class TaskQueue:
    """In-memory task queue with status tracking."""

    def __init__(self):
        self.tasks: Dict[str, Dict[str, Any]] = {}
        self.metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "response_times": [],
            "tasks_by_skill": defaultdict(int),
        }

    def submit(self, task: TaskSubmission) -> str:
        """Submit a new task and return its ID."""
        task_id = f"task_{uuid.uuid4().hex[:12]}"
        self.tasks[task_id] = {
            "task_id": task_id,
            "status": TaskStatus.pending,
            "skill_id": task.skill_id,
            "action": task.action,
            "params": task.params,
            "result": None,
            "error": None,
            "created_at": datetime.now().isoformat(),
            "completed_at": None,
            "duration_ms": None,
            "priority": task.priority,
            "metadata": task.metadata,
            "callback_url": task.callback_url,
        }
        self.metrics["total_requests"] += 1
        self.metrics["tasks_by_skill"][task.skill_id] += 1
        return task_id

    def get(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task by ID."""
        return self.tasks.get(task_id)

    def complete(self, task_id: str, result: Dict[str, Any], duration_ms: float):
        """Mark a task as completed."""
        if task_id in self.tasks:
            self.tasks[task_id]["status"] = TaskStatus.completed
            self.tasks[task_id]["result"] = result
            self.tasks[task_id]["completed_at"] = datetime.now().isoformat()
            self.tasks[task_id]["duration_ms"] = duration_ms
            self.metrics["successful_requests"] += 1
            self.metrics["response_times"].append(duration_ms)
            # Keep only last 1000 response times for avg calculation
            if len(self.metrics["response_times"]) > 1000:
                self.metrics["response_times"] = self.metrics["response_times"][-1000:]

    def fail(self, task_id: str, error: str, duration_ms: float):
        """Mark a task as failed."""
        if task_id in self.tasks:
            self.tasks[task_id]["status"] = TaskStatus.failed
            self.tasks[task_id]["error"] = error
            self.tasks[task_id]["completed_at"] = datetime.now().isoformat()
            self.tasks[task_id]["duration_ms"] = duration_ms
            self.metrics["failed_requests"] += 1
            self.metrics["response_times"].append(duration_ms)

    def list_tasks(
        self, status: Optional[TaskStatus] = None, limit: int = 50
    ) -> List[Dict[str, Any]]:
        """List tasks, optionally filtered by status."""
        tasks = list(self.tasks.values())
        if status:
            tasks = [t for t in tasks if t["status"] == status]
        # Sort by created_at desc
        tasks.sort(key=lambda t: t["created_at"], reverse=True)
        return tasks[:limit]

    def pending_count(self) -> int:
        return sum(1 for t in self.tasks.values() if t["status"] == TaskStatus.pending)

    def completed_count(self) -> int:
        return sum(1 for t in self.tasks.values() if t["status"] == TaskStatus.completed)

    def failed_count(self) -> int:
        return sum(1 for t in self.tasks.values() if t["status"] == TaskStatus.failed)

    def avg_response_time(self) -> float:
        times = self.metrics["response_times"]
        return sum(times) / len(times) if times else 0.0


def create_app(agent=None):
    """
    Create a FastAPI application for the agent service API.

    Args:
        agent: Optional AutonomousAgent instance. If None, a minimal
               API is created that can be connected to an agent later.

    Returns:
        FastAPI application instance
    """
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.middleware.cors import CORSMiddleware
    except ImportError:
        raise ImportError(
            "FastAPI is required for the service API. "
            "Install it with: pip install fastapi uvicorn"
        )

    app = FastAPI(
        title="Singularity Agent API",
        description=(
            "REST API for interacting with a Singularity autonomous agent. "
            "Submit tasks, execute skills, and monitor agent status."
        ),
        version="0.1.0",
    )

    # CORS middleware for browser access
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Shared state
    task_queue = TaskQueue()
    start_time = time.time()

    # Allow agent to be attached after creation
    app.state.agent = agent
    app.state.task_queue = task_queue

    def _get_agent():
        """Get the agent, raising 503 if not attached."""
        ag = app.state.agent
        if ag is None:
            raise HTTPException(
                status_code=503,
                detail="No agent attached. Attach an agent with app.state.agent = agent",
            )
        return ag

    # ---- Health & Status ----

    @app.get("/health")
    async def health():
        """Health check endpoint."""
        has_agent = app.state.agent is not None
        return {
            "status": "healthy",
            "agent_attached": has_agent,
            "uptime_seconds": time.time() - start_time,
        }

    @app.get("/status", response_model=AgentStatus)
    async def agent_status():
        """Get current agent status."""
        ag = _get_agent()
        return AgentStatus(
            name=ag.name,
            ticker=ag.ticker,
            agent_type=ag.agent_type,
            status="running" if ag.running else "idle",
            balance=ag.balance,
            total_api_cost=ag.total_api_cost,
            total_tokens_used=ag.total_tokens_used,
            cycle=ag.cycle,
            skills_available=len(ag.skills.skills),
            tasks_completed=task_queue.completed_count(),
            tasks_pending=task_queue.pending_count(),
            tasks_failed=task_queue.failed_count(),
            uptime_seconds=time.time() - start_time,
        )

    @app.get("/metrics", response_model=ServiceMetrics)
    async def service_metrics():
        """Get service-level metrics."""
        return ServiceMetrics(
            total_requests=task_queue.metrics["total_requests"],
            successful_requests=task_queue.metrics["successful_requests"],
            failed_requests=task_queue.metrics["failed_requests"],
            avg_response_time_ms=task_queue.avg_response_time(),
            tasks_by_skill=dict(task_queue.metrics["tasks_by_skill"]),
            uptime_seconds=time.time() - start_time,
        )

    # ---- Skills Discovery ----

    @app.get("/skills", response_model=List[SkillInfo])
    async def list_skills():
        """List all available skills and their actions."""
        ag = _get_agent()
        skills = []
        for skill_id, skill in ag.skills.skills.items():
            manifest = skill.manifest
            actions = []
            for action in manifest.actions:
                actions.append({
                    "name": action.name,
                    "description": action.description,
                    "parameters": action.parameters,
                })
            skills.append(SkillInfo(
                skill_id=skill_id,
                name=manifest.name,
                description=manifest.description,
                actions=actions,
                available=skill.check_credentials(),
            ))
        return skills

    @app.get("/skills/{skill_id}", response_model=SkillInfo)
    async def get_skill(skill_id: str):
        """Get details for a specific skill."""
        ag = _get_agent()
        skill = ag.skills.get(skill_id)
        if not skill:
            raise HTTPException(status_code=404, detail=f"Skill '{skill_id}' not found")
        manifest = skill.manifest
        actions = []
        for action in manifest.actions:
            actions.append({
                "name": action.name,
                "description": action.description,
                "parameters": action.parameters,
            })
        return SkillInfo(
            skill_id=skill_id,
            name=manifest.name,
            description=manifest.description,
            actions=actions,
            available=skill.check_credentials(),
        )

    # ---- Task Management ----

    @app.post("/tasks", response_model=TaskResponse, status_code=201)
    async def submit_task(task: TaskSubmission):
        """
        Submit a task for processing.

        The task is queued and executed asynchronously. Poll the task
        endpoint to check status, or provide a callback_url.
        """
        ag = _get_agent()

        # Validate skill and action exist
        skill = ag.skills.get(task.skill_id)
        if not skill:
            raise HTTPException(
                status_code=404,
                detail=f"Skill '{task.skill_id}' not found"
            )

        action_names = [a.name for a in skill.manifest.actions]
        if task.action not in action_names:
            raise HTTPException(
                status_code=404,
                detail=f"Action '{task.action}' not found in skill '{task.skill_id}'. "
                       f"Available: {action_names}"
            )

        task_id = task_queue.submit(task)

        # Execute asynchronously
        asyncio.create_task(_execute_task(ag, task_id, task))

        task_data = task_queue.get(task_id)
        return TaskResponse(**task_data)

    @app.get("/tasks", response_model=List[TaskResponse])
    async def list_tasks(
        status: Optional[TaskStatus] = None,
        limit: int = 50,
    ):
        """List tasks, optionally filtered by status."""
        tasks = task_queue.list_tasks(status=status, limit=limit)
        return [TaskResponse(**t) for t in tasks]

    @app.get("/tasks/{task_id}", response_model=TaskResponse)
    async def get_task(task_id: str):
        """Get the status and result of a specific task."""
        task_data = task_queue.get(task_id)
        if not task_data:
            raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")
        return TaskResponse(**task_data)

    @app.post("/tasks/{task_id}/cancel")
    async def cancel_task(task_id: str):
        """Cancel a pending task."""
        task_data = task_queue.get(task_id)
        if not task_data:
            raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")
        if task_data["status"] != TaskStatus.pending:
            raise HTTPException(
                status_code=400,
                detail=f"Task is {task_data['status']}, can only cancel pending tasks"
            )
        task_data["status"] = TaskStatus.cancelled
        task_data["completed_at"] = datetime.now().isoformat()
        return {"status": "cancelled", "task_id": task_id}

    # ---- Direct Execution ----

    @app.post("/skills/{skill_id}/{action}")
    async def execute_skill_action(
        skill_id: str, action: str, request: DirectExecuteRequest
    ):
        """
        Execute a skill action directly (synchronous).

        Returns the result immediately. For long-running actions,
        use the /tasks endpoint instead.
        """
        ag = _get_agent()

        skill = ag.skills.get(skill_id)
        if not skill:
            raise HTTPException(
                status_code=404,
                detail=f"Skill '{skill_id}' not found"
            )

        action_names = [a.name for a in skill.manifest.actions]
        if action not in action_names:
            raise HTTPException(
                status_code=404,
                detail=f"Action '{action}' not found in skill '{skill_id}'. "
                       f"Available: {action_names}"
            )

        start = time.time()
        try:
            result = await skill.execute(action, request.params)
            duration_ms = (time.time() - start) * 1000

            # Track metrics
            task_queue.metrics["total_requests"] += 1
            task_queue.metrics["tasks_by_skill"][skill_id] += 1
            if result.success:
                task_queue.metrics["successful_requests"] += 1
            else:
                task_queue.metrics["failed_requests"] += 1
            task_queue.metrics["response_times"].append(duration_ms)

            return {
                "status": "success" if result.success else "failed",
                "data": result.data,
                "message": result.message,
                "duration_ms": duration_ms,
            }
        except Exception as e:
            duration_ms = (time.time() - start) * 1000
            task_queue.metrics["total_requests"] += 1
            task_queue.metrics["failed_requests"] += 1
            task_queue.metrics["tasks_by_skill"][skill_id] += 1
            raise HTTPException(status_code=500, detail=str(e))

    # ---- Internal helpers ----

    async def _execute_task(ag, task_id: str, task: TaskSubmission):
        """Execute a task asynchronously."""
        task_data = task_queue.get(task_id)
        if not task_data or task_data["status"] == TaskStatus.cancelled:
            return

        task_data["status"] = TaskStatus.running
        start = time.time()

        try:
            skill = ag.skills.get(task.skill_id)
            result = await skill.execute(task.action, task.params)
            duration_ms = (time.time() - start) * 1000

            if result.success:
                task_queue.complete(task_id, {
                    "data": result.data,
                    "message": result.message,
                }, duration_ms)
            else:
                task_queue.fail(task_id, result.message or "Action failed", duration_ms)

        except Exception as e:
            duration_ms = (time.time() - start) * 1000
            task_queue.fail(task_id, str(e), duration_ms)

        # Send callback if configured
        if task.callback_url:
            await _send_callback(task.callback_url, task_queue.get(task_id))

    async def _send_callback(url: str, task_data: Dict[str, Any]):
        """Send task result to callback URL."""
        try:
            import httpx
            async with httpx.AsyncClient() as client:
                await client.post(url, json=task_data, timeout=10.0)
        except Exception:
            pass  # Best-effort callback

    return app
