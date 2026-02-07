"""Pydantic models for the Service API."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class TaskStatus(str, Enum):
    """Task lifecycle status."""
    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"
    cancelled = "cancelled"


class TaskSubmission(BaseModel):
    """Request to submit a new task."""
    skill_id: str = Field(..., description="Skill to use (e.g., 'filesystem', 'shell')")
    action: str = Field(..., description="Action to execute (e.g., 'glob', 'bash')")
    params: Dict[str, Any] = Field(default_factory=dict, description="Action parameters")
    callback_url: Optional[str] = Field(None, description="URL to POST results to when complete")
    priority: int = Field(default=0, description="Priority (higher = more urgent)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Custom metadata")


class TaskResponse(BaseModel):
    """Response for a submitted or retrieved task."""
    task_id: str
    status: TaskStatus
    skill_id: str
    action: str
    params: Dict[str, Any]
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: str
    completed_at: Optional[str] = None
    duration_ms: Optional[float] = None
    priority: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SkillInfo(BaseModel):
    """Information about an available skill."""
    skill_id: str
    name: str
    description: str
    actions: List[Dict[str, Any]]
    available: bool


class AgentStatus(BaseModel):
    """Current agent status."""
    name: str
    ticker: str
    agent_type: str
    status: str
    balance: float
    total_api_cost: float
    total_tokens_used: int
    cycle: int
    skills_available: int
    tasks_completed: int
    tasks_pending: int
    tasks_failed: int
    uptime_seconds: float


class ServiceMetrics(BaseModel):
    """Service-level metrics."""
    total_requests: int
    successful_requests: int
    failed_requests: int
    avg_response_time_ms: float
    tasks_by_skill: Dict[str, int]
    uptime_seconds: float


class DirectExecuteRequest(BaseModel):
    """Request to execute a skill action directly (synchronous)."""
    params: Dict[str, Any] = Field(default_factory=dict, description="Action parameters")


class ErrorResponse(BaseModel):
    """Error response."""
    error: str
    detail: Optional[str] = None
