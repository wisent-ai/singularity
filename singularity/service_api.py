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
    When an APIGatewaySkill is provided, all requests are validated through
    the gateway's key management, rate limiting, and usage tracking system.
    """

    def __init__(self, agent=None, api_keys: Optional[List[str]] = None,
                 persist_path: Optional[str] = None, require_auth: bool = False,
                 api_gateway=None):
        self.agent = agent
        self.api_keys = set(api_keys or [])
        if os.environ.get("SERVICE_API_KEY"):
            self.api_keys.add(os.environ["SERVICE_API_KEY"])
        self.api_gateway = api_gateway
        self.require_auth = require_auth or bool(self.api_keys) or (api_gateway is not None)
        self.task_store = TaskStore(persist_path=persist_path)
        self._worker_task: Optional[asyncio.Task] = None
        self._started_at = datetime.utcnow().isoformat()

    def validate_api_key(self, key: Optional[str]) -> bool:
        if not self.require_auth:
            return True
        if not key:
            return False
        return key in self.api_keys

    async def validate_via_gateway(self, raw_key: Optional[str],
                                   required_scope: Optional[str] = None) -> dict:
        """
        Validate an API key through the APIGatewaySkill.
        Returns a dict with allowed, key_id, owner, reason fields.
        """
        if not self.api_gateway:
            return {"allowed": False, "reason": "no_gateway"}
        params = {"api_key": raw_key or ""}
        if required_scope:
            params["required_scope"] = required_scope
        result = await self.api_gateway.execute("check_access", params)
        return result.data

    async def record_gateway_usage(self, key_id: str, endpoint: str,
                                   cost: float = 0.0, revenue: float = 0.0,
                                   error: bool = False):
        """Record API usage through the gateway for billing/tracking."""
        if not self.api_gateway or not key_id:
            return
        try:
            await self.api_gateway.execute("record_usage", {
                "key_id": key_id,
                "endpoint": endpoint,
                "cost": cost,
                "revenue": revenue,
                "error": error,
            })
        except Exception:
            pass

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
        result = {
            "status": "healthy",
            "started_at": self._started_at,
            "agent": agent_info,
            "tasks": self.task_store.stats(),
            "api_gateway": {
                "enabled": self.api_gateway is not None,
            },
        }
        return result


def create_app(agent=None, api_keys: Optional[List[str]] = None,
               persist_path: Optional[str] = None, require_auth: bool = False,
               api_gateway=None) -> "FastAPI":
    """
    Create a FastAPI app that serves the agent's capabilities.

    Args:
        agent: AutonomousAgent instance (can be None for testing)
        api_keys: List of valid API keys for authentication
        persist_path: Path to persist task history
        require_auth: Whether to require API key authentication
        api_gateway: Optional APIGatewaySkill for advanced key management,
                     rate limiting, and per-key usage tracking/billing

    Returns:
        FastAPI application
    """
    if not HAS_FASTAPI:
        raise ImportError(
            "FastAPI is required for ServiceAPI. Install with: "
            "pip install fastapi uvicorn"
        )

    # Auto-detect APIGatewaySkill from agent if not explicitly provided
    if api_gateway is None and agent and hasattr(agent, "skills"):
        api_gateway = agent.skills.get("api_gateway")

    service = ServiceAPI(
        agent=agent, api_keys=api_keys,
        persist_path=persist_path, require_auth=require_auth,
        api_gateway=api_gateway,
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
        """
        Validate API key via gateway (if available) or simple key set.
        Returns a dict with key info when gateway is used, or the raw key string.
        """
        if not service.require_auth:
            return None
        if not authorization:
            raise HTTPException(status_code=401, detail="API key required")
        key = authorization.replace("Bearer ", "").strip()

        # Use APIGatewaySkill for advanced validation if available
        if service.api_gateway:
            access = await service.validate_via_gateway(key)
            if not access.get("allowed"):
                reason = access.get("reason", "access_denied")
                status_code = 403
                detail = "Invalid API key"
                if reason == "rate_limited":
                    status_code = 429
                    detail = "Rate limit exceeded"
                elif reason == "daily_limit_exceeded":
                    status_code = 429
                    detail = "Daily limit exceeded"
                elif reason == "expired":
                    status_code = 403
                    detail = "API key expired"
                elif reason == "revoked":
                    status_code = 403
                    detail = "API key revoked"
                elif reason == "insufficient_scope":
                    status_code = 403
                    detail = f"Insufficient scope: requires {access.get('required', 'unknown')}"
                elif reason == "missing_key":
                    status_code = 401
                    detail = "API key required"
                raise HTTPException(status_code=status_code, detail=detail)
            # Return gateway access info (includes key_id, owner, scopes)
            return {"raw_key": key, "key_id": access.get("key_id"), "owner": access.get("owner"), "via_gateway": True}

        # Fallback to simple key validation
        if not service.validate_api_key(key):
            raise HTTPException(status_code=403, detail="Invalid API key")
        return key

    # --- Endpoints ---

    @app.get("/health")
    async def health():
        """Health check and agent status."""
        return service.health()

    @app.get("/capabilities")
    async def capabilities(auth_info=Depends(check_auth)):
        """List all available skills and actions."""
        return {"capabilities": service.get_capabilities()}

    @app.post("/tasks", response_model=TaskResponse)
    async def submit_task(body: TaskSubmission, auth_info=Depends(check_auth)):
        """Submit a task for async execution. Returns immediately with task ID."""
        try:
            raw_key = auth_info.get("raw_key") if isinstance(auth_info, dict) else auth_info
            task = await service.submit_task(
                skill_id=body.skill_id,
                action=body.action,
                params=body.params,
                webhook_url=body.webhook_url,
                api_key=raw_key,
            )
            # Track usage via gateway
            if isinstance(auth_info, dict) and auth_info.get("via_gateway"):
                await service.record_gateway_usage(
                    key_id=auth_info["key_id"],
                    endpoint=f"tasks/{body.skill_id}/{body.action}",
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
    async def get_task(task_id: str, auth_info=Depends(check_auth)):
        """Get task status and result."""
        task = service.task_store.get(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        d = task.to_dict()
        return TaskDetail(**d)

    @app.get("/tasks")
    async def list_tasks(status: Optional[str] = None, limit: int = 50,
                         offset: int = 0, auth_info=Depends(check_auth)):
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
    async def execute_sync(body: SyncExecution, auth_info=Depends(check_auth)):
        """Execute a skill action synchronously. Blocks until complete."""
        result = await service.execute_sync(
            skill_id=body.skill_id,
            action=body.action,
            params=body.params,
        )
        # Track usage via gateway
        if isinstance(auth_info, dict) and auth_info.get("via_gateway"):
            is_error = result.get("status") == "error"
            await service.record_gateway_usage(
                key_id=auth_info["key_id"],
                endpoint=f"execute/{body.skill_id}/{body.action}",
                error=is_error,
            )
        return result

    @app.post("/tasks/{task_id}/cancel")
    async def cancel_task(task_id: str, auth_info=Depends(check_auth)):
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
    async def metrics(auth_info=Depends(check_auth)):
        """Get agent metrics and task statistics."""
        data = {
            "tasks": service.task_store.stats(),
            "started_at": service._started_at,
        }
        if agent and hasattr(agent, 'metrics'):
            data["agent"] = agent.metrics.summary()
        return data

    # --- API Gateway Billing & Usage Endpoints ---
    # These endpoints expose billing and usage data from the APIGatewaySkill.
    # Only available when an api_gateway is configured.

    @app.get("/billing")
    async def get_billing(owner: Optional[str] = None, auth_info=Depends(check_auth)):
        """Get billing summary across all API keys. Requires api_gateway."""
        if not service.api_gateway:
            raise HTTPException(status_code=503, detail="API Gateway not configured")
        params = {}
        if owner:
            params["owner"] = owner
        result = await service.api_gateway.execute("get_billing", params)
        if not result.success:
            raise HTTPException(status_code=500, detail=result.message)
        return result.data

    @app.get("/usage/{key_id}")
    async def get_usage(key_id: str, auth_info=Depends(check_auth)):
        """Get usage statistics for a specific API key. Requires api_gateway."""
        if not service.api_gateway:
            raise HTTPException(status_code=503, detail="API Gateway not configured")
        result = await service.api_gateway.execute("get_usage", {"key_id": key_id})
        if not result.success:
            raise HTTPException(status_code=404, detail=result.message)
        return result.data

    @app.get("/keys")
    async def list_api_keys(include_revoked: bool = False, owner: Optional[str] = None,
                            auth_info=Depends(check_auth)):
        """List all managed API keys (metadata only). Requires api_gateway."""
        if not service.api_gateway:
            raise HTTPException(status_code=503, detail="API Gateway not configured")
        params = {"include_revoked": include_revoked}
        if owner:
            params["owner"] = owner
        result = await service.api_gateway.execute("list_keys", params)
        if not result.success:
            raise HTTPException(status_code=500, detail=result.message)
        return result.data

    @app.post("/keys")
    async def create_api_key(body: Dict[str, Any], auth_info=Depends(check_auth)):
        """Create a new API key via the gateway. Requires api_gateway."""
        if not service.api_gateway:
            raise HTTPException(status_code=503, detail="API Gateway not configured")
        result = await service.api_gateway.execute("create_key", body)
        if not result.success:
            raise HTTPException(status_code=400, detail=result.message)
        return result.data

    @app.post("/keys/{key_id}/revoke")
    async def revoke_api_key(key_id: str, auth_info=Depends(check_auth)):
        """Revoke an API key. Requires api_gateway."""
        if not service.api_gateway:
            raise HTTPException(status_code=503, detail="API Gateway not configured")
        result = await service.api_gateway.execute("revoke_key", {"key_id": key_id})
        if not result.success:
            raise HTTPException(status_code=400, detail=result.message)
        return result.data

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
    async def list_webhooks(auth_info=Depends(check_auth)):
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
    async def ask_natural_language(body: NLQuery, auth_info=Depends(check_auth)):
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
    async def ask_match_only(body: NLQuery, auth_info=Depends(check_auth)):
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


    # --- Agent-to-Agent Messaging API ---
    # These endpoints enable direct communication between agents.
    # Implements feature request #125: Agent-to-agent messaging API.

    class SendMessageBody(BaseModel):
        from_instance_id: str = Field(..., description="Sender's agent instance ID")
        to_instance_id: str = Field(..., description="Recipient's agent instance ID")
        content: str = Field(..., description="Message content")
        type: str = Field(default="direct", description="Message type: direct, service_request, broadcast")
        metadata: Dict[str, Any] = Field(default_factory=dict, description="Optional metadata")
        conversation_id: Optional[str] = Field(None, description="Conversation thread ID")

    class ServiceRequestBody(BaseModel):
        from_instance_id: str = Field(..., description="Requester's agent instance ID")
        to_instance_id: str = Field(..., description="Provider's agent instance ID")
        service_name: str = Field(..., description="Name of the service requested")
        request_params: Dict[str, Any] = Field(default_factory=dict, description="Service parameters")
        offer_amount: float = Field(default=0.0, description="Amount offered (WISENT tokens)")

    class ReplyBody(BaseModel):
        from_instance_id: str = Field(..., description="Sender's agent instance ID")
        message_id: str = Field(..., description="ID of message being replied to")
        content: str = Field(..., description="Reply content")

    def _get_messaging_skill():
        """Get the MessagingSkill instance, creating a standalone one if no agent."""
        if agent and hasattr(agent, 'skills'):
            skill = agent.skills.get("messaging")
            if skill:
                return skill
        # Create standalone skill for when no agent is configured
        from singularity.skills.messaging import MessagingSkill
        return MessagingSkill()

    @app.post("/api/messages")
    async def send_message(body: SendMessageBody):
        """Send a message to another agent by instance_id."""
        skill = _get_messaging_skill()
        result = await skill.execute("send", {
            "from_instance_id": body.from_instance_id,
            "to_instance_id": body.to_instance_id,
            "content": body.content,
            "message_type": body.type,
            "metadata": body.metadata,
            "conversation_id": body.conversation_id or "",
        })
        if not result.success:
            raise HTTPException(status_code=400, detail=result.message)
        return {
            "status": "sent",
            "message_id": result.data.get("message_id"),
            "conversation_id": result.data.get("conversation_id"),
        }

    @app.get("/api/messages/{instance_id}")
    async def read_messages(
        instance_id: str,
        from_instance_id: Optional[str] = None,
        message_type: Optional[str] = None,
        unread_only: bool = False,
        limit: int = 50,
        conversation_id: Optional[str] = None,
    ):
        """Read messages for an agent. Supports filtering by sender, type, conversation."""
        skill = _get_messaging_skill()
        params = {"instance_id": instance_id, "limit": limit, "unread_only": unread_only}
        if from_instance_id:
            params["from_instance_id"] = from_instance_id
        if message_type:
            params["message_type"] = message_type
        if conversation_id:
            params["conversation_id"] = conversation_id

        result = await skill.execute("read_inbox", params)
        return {
            "messages": result.data.get("messages", []),
            "count": result.data.get("count", 0),
            "total_in_inbox": result.data.get("total_in_inbox", 0),
        }

    @app.post("/api/messages/broadcast")
    async def broadcast_message(body: SendMessageBody):
        """Broadcast a message to all registered agents."""
        skill = _get_messaging_skill()
        result = await skill.execute("broadcast", {
            "from_instance_id": body.from_instance_id,
            "content": body.content,
            "metadata": body.metadata,
        })
        if not result.success:
            raise HTTPException(status_code=400, detail=result.message)
        return {
            "status": "broadcast_sent",
            "sent_count": result.data.get("sent_count", 0),
            "conversation_id": result.data.get("conversation_id"),
        }

    @app.post("/api/messages/service-request")
    async def service_request(body: ServiceRequestBody):
        """Send a structured service request to another agent."""
        skill = _get_messaging_skill()
        result = await skill.execute("service_request", {
            "from_instance_id": body.from_instance_id,
            "to_instance_id": body.to_instance_id,
            "service_name": body.service_name,
            "request_params": body.request_params,
            "offer_amount": body.offer_amount,
        })
        if not result.success:
            raise HTTPException(status_code=400, detail=result.message)
        return {
            "status": "request_sent",
            "request_id": result.data.get("request_id"),
            "message_id": result.data.get("message_id"),
            "conversation_id": result.data.get("conversation_id"),
        }

    @app.post("/api/messages/reply")
    async def reply_to_message(body: ReplyBody):
        """Reply to a specific message, creating a conversation thread."""
        skill = _get_messaging_skill()
        result = await skill.execute("reply", {
            "from_instance_id": body.from_instance_id,
            "message_id": body.message_id,
            "content": body.content,
        })
        if not result.success:
            raise HTTPException(status_code=400, detail=result.message)
        return {
            "status": "reply_sent",
            "message_id": result.data.get("message_id"),
            "conversation_id": result.data.get("conversation_id"),
        }

    @app.get("/api/conversations/{conversation_id}")
    async def get_conversation(conversation_id: str):
        """Get all messages in a conversation thread."""
        skill = _get_messaging_skill()
        result = await skill.execute("get_conversation", {
            "conversation_id": conversation_id,
        })
        return {
            "conversation_id": conversation_id,
            "metadata": result.data.get("metadata", {}),
            "messages": result.data.get("messages", []),
            "count": result.data.get("count", 0),
        }

    @app.post("/api/messages/{message_id}/read")
    async def mark_message_read(message_id: str, reader_instance_id: str):
        """Mark a message as read (generates a read receipt)."""
        skill = _get_messaging_skill()
        result = await skill.execute("mark_read", {
            "message_id": message_id,
            "reader_instance_id": reader_instance_id,
        })
        if not result.success:
            raise HTTPException(status_code=404, detail=result.message)
        return {"status": "read", "message_id": message_id}

    @app.delete("/api/messages/{instance_id}/{message_id}")
    async def delete_message(instance_id: str, message_id: str):
        """Delete a message from an agent's inbox."""
        skill = _get_messaging_skill()
        result = await skill.execute("delete_message", {
            "instance_id": instance_id,
            "message_id": message_id,
        })
        if not result.success:
            raise HTTPException(status_code=404, detail=result.message)
        return {"status": "deleted", "message_id": message_id}

    @app.get("/api/messages/stats")
    async def messaging_stats():
        """Get messaging statistics."""
        skill = _get_messaging_skill()
        result = await skill.execute("get_stats", {})
        return result.data

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
