#!/usr/bin/env python3
"""
Service API - Expose the agent as a REST API for external consumers.

This is the core infrastructure for Revenue Generation. It turns the
autonomous agent into a callable service that external users can:
- Submit tasks and get results
- List available skills/capabilities
- Check agent health and metrics
- Receive webhook callbacks on task completion

Usage:
    from singularity.service_api import create_app

    app = create_app(agent)
    # Run with: uvicorn singularity.service_api:app
"""

import asyncio
import os
import time
import uuid
import json
import httpx
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field, asdict

try:
    from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Header
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field

    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False


class TaskStatus(str, Enum):
    """Status of a submitted task."""
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TaskRecord:
    """Internal record of a submitted task."""
    task_id: str
    skill_id: str
    action: str
    params: Dict[str, Any]
    status: TaskStatus
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    webhook_url: Optional[str] = None
    api_key: Optional[str] = None
    execution_time_ms: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        # Don't expose api_key in responses
        d.pop("api_key", None)
        return d


class TaskStore:
    """In-memory task storage with optional persistence."""

    def __init__(self, persist_path: Optional[str] = None, max_tasks: int = 1000):
        self._tasks: Dict[str, TaskRecord] = {}
        self._queue: asyncio.Queue = asyncio.Queue()
        self._persist_path = persist_path
        self._max_tasks = max_tasks
        self._load()

    def create(self, skill_id: str, action: str, params: Dict,
               webhook_url: Optional[str] = None, api_key: Optional[str] = None) -> TaskRecord:
        task = TaskRecord(
            task_id=str(uuid.uuid4()),
            skill_id=skill_id,
            action=action,
            params=params,
            status=TaskStatus.QUEUED,
            created_at=datetime.utcnow().isoformat(),
            webhook_url=webhook_url,
            api_key=api_key,
        )
        self._tasks[task.task_id] = task
        self._trim()
        self._save()
        return task

    def get(self, task_id: str) -> Optional[TaskRecord]:
        return self._tasks.get(task_id)

    def list_tasks(self, status: Optional[TaskStatus] = None,
                   limit: int = 50, offset: int = 0) -> List[TaskRecord]:
        tasks = list(self._tasks.values())
        if status:
            tasks = [t for t in tasks if t.status == status]
        tasks.sort(key=lambda t: t.created_at, reverse=True)
        return tasks[offset:offset + limit]

    def update(self, task_id: str, **kwargs) -> Optional[TaskRecord]:
        task = self._tasks.get(task_id)
        if not task:
            return None
        for k, v in kwargs.items():
            if hasattr(task, k):
                setattr(task, k, v)
        self._save()
        return task

    def stats(self) -> Dict[str, Any]:
        counts = {}
        for status in TaskStatus:
            counts[status.value] = sum(1 for t in self._tasks.values() if t.status == status)
        total_exec_times = [t.execution_time_ms for t in self._tasks.values()
                           if t.execution_time_ms is not None]
        return {
            "total_tasks": len(self._tasks),
            "by_status": counts,
            "avg_execution_ms": (sum(total_exec_times) / len(total_exec_times))
                if total_exec_times else 0,
            "total_completed": counts.get("completed", 0),
            "total_failed": counts.get("failed", 0),
        }

    def _trim(self):
        if len(self._tasks) > self._max_tasks:
            sorted_tasks = sorted(self._tasks.values(), key=lambda t: t.created_at)
            for task in sorted_tasks[:len(self._tasks) - self._max_tasks]:
                if task.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED):
                    del self._tasks[task.task_id]

    def _save(self):
        if not self._persist_path:
            return
        try:
            data = {tid: t.to_dict() for tid, t in self._tasks.items()}
            with open(self._persist_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass

    def _load(self):
        if not self._persist_path or not os.path.exists(self._persist_path):
            return
        try:
            with open(self._persist_path, 'r') as f:
                data = json.load(f)
            for tid, tdata in data.items():
                tdata["status"] = TaskStatus(tdata["status"])
                self._tasks[tid] = TaskRecord(**tdata)
        except Exception:
            pass


class ServiceAPI:
    """
    Wraps an AutonomousAgent with a REST API for external consumption.

    This is the bridge between the agent's capabilities and paying customers.
    """

    def __init__(self, agent=None, api_keys: Optional[List[str]] = None,
                 persist_path: Optional[str] = None, require_auth: bool = False):
        self.agent = agent
        self.api_keys = set(api_keys or [])
        if os.environ.get("SERVICE_API_KEY"):
            self.api_keys.add(os.environ["SERVICE_API_KEY"])
        self.require_auth = require_auth or bool(self.api_keys)
        self.task_store = TaskStore(persist_path=persist_path)
        self._worker_task: Optional[asyncio.Task] = None
        self._started_at = datetime.utcnow().isoformat()

    def validate_api_key(self, key: Optional[str]) -> bool:
        if not self.require_auth:
            return True
        if not key:
            return False
        return key in self.api_keys

    async def submit_task(self, skill_id: str, action: str, params: Dict,
                          webhook_url: Optional[str] = None,
                          api_key: Optional[str] = None) -> TaskRecord:
        """Submit a task for async execution."""
        # Validate skill exists
        if self.agent:
            skill = self.agent.skills.get(skill_id)
            if not skill:
                raise ValueError(f"Skill '{skill_id}' not found")
            valid_actions = [a.name for a in skill.manifest.actions]
            if action not in valid_actions:
                raise ValueError(
                    f"Action '{action}' not found in skill '{skill_id}'. "
                    f"Available: {valid_actions}"
                )

        task = self.task_store.create(
            skill_id=skill_id, action=action, params=params,
            webhook_url=webhook_url, api_key=api_key,
        )
        # Execute immediately in background
        asyncio.create_task(self._execute_task(task))
        return task

    async def execute_sync(self, skill_id: str, action: str, params: Dict) -> Dict:
        """Execute a skill action synchronously and return the result."""
        if not self.agent:
            return {"status": "error", "message": "No agent configured"}

        skill = self.agent.skills.get(skill_id)
        if not skill:
            return {"status": "error", "message": f"Skill '{skill_id}' not found"}

        try:
            result = await skill.execute(action, params)
            return {
                "status": "success" if result.success else "failed",
                "data": result.data,
                "message": result.message,
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    async def _execute_task(self, task: TaskRecord):
        """Execute a queued task."""
        self.task_store.update(
            task.task_id,
            status=TaskStatus.RUNNING,
            started_at=datetime.utcnow().isoformat(),
        )
        start_time = time.monotonic()

        try:
            if not self.agent:
                raise RuntimeError("No agent configured")

            skill = self.agent.skills.get(task.skill_id)
            if not skill:
                raise RuntimeError(f"Skill '{task.skill_id}' not found")

            result = await skill.execute(task.action, task.params)
            elapsed_ms = (time.monotonic() - start_time) * 1000

            self.task_store.update(
                task.task_id,
                status=TaskStatus.COMPLETED if result.success else TaskStatus.FAILED,
                completed_at=datetime.utcnow().isoformat(),
                result={"data": result.data, "message": result.message},
                error=None if result.success else result.message,
                execution_time_ms=elapsed_ms,
            )
        except Exception as e:
            elapsed_ms = (time.monotonic() - start_time) * 1000
            self.task_store.update(
                task.task_id,
                status=TaskStatus.FAILED,
                completed_at=datetime.utcnow().isoformat(),
                error=str(e),
                execution_time_ms=elapsed_ms,
            )

        # Fire webhook if configured
        updated = self.task_store.get(task.task_id)
        if updated and updated.webhook_url:
            await self._fire_webhook(updated)

    async def _fire_webhook(self, task: TaskRecord):
        """Send webhook notification for task completion."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                await client.post(
                    task.webhook_url,
                    json=task.to_dict(),
                    headers={"Content-Type": "application/json"},
                )
        except Exception:
            pass  # Best-effort webhook delivery

    def get_capabilities(self) -> List[Dict]:
        """List all available skills and their actions."""
        if not self.agent:
            return []
        capabilities = []
        for skill in self.agent.skills.skills.values():
            cap = {
                "skill_id": skill.manifest.skill_id,
                "name": skill.manifest.name,
                "description": skill.manifest.description,
                "actions": [],
            }
            for action in skill.manifest.actions:
                cap["actions"].append({
                    "name": action.name,
                    "description": action.description,
                    "parameters": action.parameters,
                })
            capabilities.append(cap)
        return capabilities

    def health(self) -> Dict:
        """Get API and agent health status."""
        agent_info = {}
        if self.agent:
            agent_info = {
                "name": self.agent.name,
                "ticker": self.agent.ticker,
                "agent_type": self.agent.type if hasattr(self.agent, 'type') else self.agent.agent_type,
                "balance": self.agent.balance,
                "cycle": self.agent.cycle,
                "running": self.agent.running,
                "skills_loaded": len(self.agent.skills.skills),
            }
        return {
            "status": "healthy",
            "started_at": self._started_at,
            "agent": agent_info,
            "tasks": self.task_store.stats(),
        }


def create_app(agent=None, api_keys: Optional[List[str]] = None,
               persist_path: Optional[str] = None, require_auth: bool = False) -> "FastAPI":
    """
    Create a FastAPI app that serves the agent's capabilities.

    Args:
        agent: AutonomousAgent instance (can be None for testing)
        api_keys: List of valid API keys for authentication
        persist_path: Path to persist task history
        require_auth: Whether to require API key authentication

    Returns:
        FastAPI application
    """
    if not HAS_FASTAPI:
        raise ImportError(
            "FastAPI is required for ServiceAPI. Install with: "
            "pip install fastapi uvicorn"
        )

    service = ServiceAPI(
        agent=agent, api_keys=api_keys,
        persist_path=persist_path, require_auth=require_auth,
    )

    app = FastAPI(
        title="Singularity Agent API",
        description="REST API for interacting with an autonomous AI agent",
        version="0.1.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Store service on app for access in tests
    app.state.service = service

    # --- Pydantic models for request/response ---

    class TaskSubmission(BaseModel):
        skill_id: str = Field(..., description="ID of the skill to invoke")
        action: str = Field(..., description="Action within the skill")
        params: Dict[str, Any] = Field(default_factory=dict, description="Action parameters")
        webhook_url: Optional[str] = Field(None, description="URL for completion callback")

    class SyncExecution(BaseModel):
        skill_id: str = Field(..., description="ID of the skill to invoke")
        action: str = Field(..., description="Action within the skill")
        params: Dict[str, Any] = Field(default_factory=dict, description="Action parameters")

    class TaskResponse(BaseModel):
        task_id: str
        status: str
        created_at: str
        skill_id: str
        action: str

    class TaskDetail(BaseModel):
        task_id: str
        skill_id: str
        action: str
        params: Dict[str, Any]
        status: str
        created_at: str
        started_at: Optional[str] = None
        completed_at: Optional[str] = None
        result: Optional[Dict[str, Any]] = None
        error: Optional[str] = None
        execution_time_ms: Optional[float] = None

    # --- Auth dependency ---

    async def check_auth(authorization: Optional[str] = Header(None)):
        if not service.require_auth:
            return None
        if not authorization:
            raise HTTPException(status_code=401, detail="API key required")
        key = authorization.replace("Bearer ", "").strip()
        if not service.validate_api_key(key):
            raise HTTPException(status_code=403, detail="Invalid API key")
        return key

    # --- Endpoints ---

    @app.get("/health")
    async def health():
        """Health check and agent status."""
        return service.health()

    @app.get("/capabilities")
    async def capabilities(api_key: str = Depends(check_auth)):
        """List all available skills and actions."""
        return {"capabilities": service.get_capabilities()}

    @app.post("/tasks", response_model=TaskResponse)
    async def submit_task(body: TaskSubmission, api_key: str = Depends(check_auth)):
        """Submit a task for async execution. Returns immediately with task ID."""
        try:
            task = await service.submit_task(
                skill_id=body.skill_id,
                action=body.action,
                params=body.params,
                webhook_url=body.webhook_url,
                api_key=api_key,
            )
            return TaskResponse(
                task_id=task.task_id,
                status=task.status.value,
                created_at=task.created_at,
                skill_id=task.skill_id,
                action=task.action,
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.get("/tasks/{task_id}", response_model=TaskDetail)
    async def get_task(task_id: str, api_key: str = Depends(check_auth)):
        """Get task status and result."""
        task = service.task_store.get(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        d = task.to_dict()
        return TaskDetail(**d)

    @app.get("/tasks")
    async def list_tasks(status: Optional[str] = None, limit: int = 50,
                         offset: int = 0, api_key: str = Depends(check_auth)):
        """List tasks with optional status filter."""
        filter_status = TaskStatus(status) if status else None
        tasks = service.task_store.list_tasks(status=filter_status, limit=limit, offset=offset)
        return {
            "tasks": [t.to_dict() for t in tasks],
            "total": len(service.task_store._tasks),
            "limit": limit,
            "offset": offset,
        }

    @app.post("/execute")
    async def execute_sync(body: SyncExecution, api_key: str = Depends(check_auth)):
        """Execute a skill action synchronously. Blocks until complete."""
        result = await service.execute_sync(
            skill_id=body.skill_id,
            action=body.action,
            params=body.params,
        )
        return result

    @app.post("/tasks/{task_id}/cancel")
    async def cancel_task(task_id: str, api_key: str = Depends(check_auth)):
        """Cancel a queued task."""
        task = service.task_store.get(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        if task.status not in (TaskStatus.QUEUED, TaskStatus.RUNNING):
            raise HTTPException(
                status_code=400,
                detail=f"Cannot cancel task in '{task.status.value}' state"
            )
        service.task_store.update(task_id, status=TaskStatus.CANCELLED,
                                  completed_at=datetime.utcnow().isoformat())
        return {"task_id": task_id, "status": "cancelled"}

    @app.get("/metrics")
    async def metrics(api_key: str = Depends(check_auth)):
        """Get agent metrics and task statistics."""
        data = {
            "tasks": service.task_store.stats(),
            "started_at": service._started_at,
        }
        if agent and hasattr(agent, 'metrics'):
            data["agent"] = agent.metrics.summary()
        return data

    # --- Webhook Endpoints ---
    # These allow external systems to trigger agent actions via HTTP POST.
    # Webhooks are registered through the WebhookSkill and exposed here.

    @app.post("/webhooks/{endpoint_name}")
    async def receive_webhook(
        endpoint_name: str,
        body: Dict[str, Any] = None,
        x_webhook_signature: Optional[str] = Header(None),
        x_hub_signature_256: Optional[str] = Header(None),
    ):
        """
        Receive an inbound webhook from an external system.

        External systems POST to /webhooks/<endpoint_name> with a JSON payload.
        The agent validates the signature, applies filters, transforms the payload,
        and routes it to the configured target skill action.

        Supports GitHub-style (X-Hub-Signature-256) and custom (X-Webhook-Signature) headers.
        """
        # Try to get WebhookSkill from the agent
        webhook_skill = None
        if agent and hasattr(agent, 'skills'):
            webhook_skill = agent.skills.get("webhook")

        if not webhook_skill:
            raise HTTPException(
                status_code=503,
                detail="Webhook processing not available (WebhookSkill not loaded)"
            )

        # Use whichever signature header is present
        signature = x_webhook_signature or x_hub_signature_256

        result = await webhook_skill.execute("receive", {
            "endpoint_name": endpoint_name,
            "payload": body or {},
            "headers": {"X-Webhook-Signature": signature or ""},
            "signature": signature,
        })

        if not result.success:
            status_code = 400
            if "not found" in result.message.lower():
                status_code = 404
            elif "rate limit" in result.message.lower():
                status_code = 429
            elif "signature" in result.message.lower():
                status_code = 403
            raise HTTPException(status_code=status_code, detail=result.message)

        return {
            "status": "accepted",
            "delivery_id": result.data.get("delivery_id"),
            "message": result.message,
        }

    @app.get("/webhooks")
    async def list_webhooks(api_key: str = Depends(check_auth)):
        """List all registered webhook endpoints."""
        webhook_skill = None
        if agent and hasattr(agent, 'skills'):
            webhook_skill = agent.skills.get("webhook")

        if not webhook_skill:
            return {"endpoints": [], "message": "WebhookSkill not loaded"}

        result = await webhook_skill.execute("list_endpoints", {})
        return result.data

    # --- Natural Language Task Routing ---
    # These endpoints let external users submit tasks in plain English
    # without knowing internal skill IDs. The NaturalLanguageRouter
    # matches the description to the best skill+action.

    class NLQuery(BaseModel):
        query: str = Field(..., description="Natural language task description")
        params: Dict[str, Any] = Field(default_factory=dict, description="Additional parameters")
        top_k: int = Field(default=5, description="Number of matches to return")

    @app.post("/ask")
    async def ask_natural_language(body: NLQuery, api_key: str = Depends(check_auth)):
        """
        Submit a task in natural language. The router finds the best skill
        and executes it, returning the result.
        
        Example: POST /ask {"query": "scan this code for security issues", "params": {"code": "..."}}
        """
        nl_router = None
        if agent and hasattr(agent, 'skills'):
            nl_router = agent.skills.get("nl_router")

        if not nl_router:
            raise HTTPException(
                status_code=503,
                detail="Natural language routing not available (NaturalLanguageRouter not loaded)"
            )

        result = await nl_router.execute("route_and_execute", {
            "query": body.query,
            "params": body.params,
        })

        return {
            "success": result.success,
            "message": result.message,
            "data": result.data,
        }

    @app.post("/ask/match")
    async def ask_match_only(body: NLQuery, api_key: str = Depends(check_auth)):
        """
        Find matching skills for a natural language query without executing.
        Useful for previewing what would happen before committing.
        """
        nl_router = None
        if agent and hasattr(agent, 'skills'):
            nl_router = agent.skills.get("nl_router")

        if not nl_router:
            raise HTTPException(
                status_code=503,
                detail="Natural language routing not available"
            )

        result = await nl_router.execute("route", {
            "query": body.query,
            "top_k": body.top_k,
        })

        return {
            "success": result.success,
            "message": result.message,
            "matches": result.data.get("matches", []),
        }


    return app


# Module-level app for `uvicorn singularity.service_api:app`
app = None

def serve(agent=None, host: str = "0.0.0.0", port: int = 8080, **kwargs):
    """Convenience function to start the API server."""
    import uvicorn
    global app
    application = create_app(agent=agent, **kwargs)
    app = application
    uvicorn.run(application, host=host, port=port)
