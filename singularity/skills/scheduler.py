#!/usr/bin/env python3
"""
SchedulerSkill - Time-based autonomous task scheduling.

Enables the agent to schedule future work, set timers, and run recurring tasks.
This is foundational for true autonomy: without time-awareness, an agent can
only react to prompts. With a scheduler, it can:

- Run periodic self-evaluations (Self-Improvement)
- Schedule recurring service delivery (Revenue)
- Monitor child agent health on intervals (Replication)
- Review and reprioritize goals periodically (Goal Setting)

Supports:
- One-shot delayed tasks (run action X in Y seconds)
- Recurring tasks with configurable intervals
- Cron expression scheduling (e.g., "*/5 * * * *", "@daily")
- Task cancellation and listing
- Execution history with success/failure tracking
- Persistent schedule that survives restarts (via JSON)
"""

import asyncio
import json
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from enum import Enum

from .base import Skill, SkillManifest, SkillAction, SkillResult
from ..cron_parser import CronExpression, CronParseError


class ScheduleType(Enum):
    ONCE = "once"           # Run once after delay
    RECURRING = "recurring"  # Run repeatedly at interval
    CRON = "cron"           # Cron expression


class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ScheduledTask:
    """A task scheduled for future execution."""
    id: str
    name: str
    skill_id: str          # Skill to call
    action: str            # Action to execute
    params: Dict           # Parameters for the action
    schedule_type: str     # once / recurring / cron
    interval_seconds: float  # Interval for recurring, delay for once
    created_at: str
    next_run_at: float     # Unix timestamp of next scheduled run
    status: str = "pending"
    run_count: int = 0
    max_runs: int = 0      # 0 = unlimited (for recurring/cron)
    last_run_at: Optional[str] = None
    last_result: Optional[str] = None
    last_success: Optional[bool] = None
    enabled: bool = True
    cron_expression: Optional[str] = None  # Cron expression string (for cron type)

    def to_dict(self) -> Dict:
        d = {
            "id": self.id,
            "name": self.name,
            "skill_id": self.skill_id,
            "action": self.action,
            "params": self.params,
            "schedule_type": self.schedule_type,
            "interval_seconds": self.interval_seconds,
            "created_at": self.created_at,
            "next_run_at": self.next_run_at,
            "status": self.status,
            "run_count": self.run_count,
            "max_runs": self.max_runs,
            "last_run_at": self.last_run_at,
            "last_result": self.last_result,
            "last_success": self.last_success,
            "enabled": self.enabled,
        }
        if self.cron_expression:
            d["cron_expression"] = self.cron_expression
        return d


@dataclass
class ExecutionRecord:
    """Record of a task execution."""
    task_id: str
    task_name: str
    executed_at: str
    success: bool
    message: str
    duration_seconds: float


class SchedulerSkill(Skill):
    """
    Time-based task scheduling for autonomous agents.

    Enables agents to schedule future actions, run recurring tasks,
    and maintain time-awareness for truly autonomous operation.
    Now with full cron expression support.
    """

    def __init__(self, credentials: Dict = None):
        super().__init__(credentials)
        self._tasks: Dict[str, ScheduledTask] = {}
        self._execution_history: List[ExecutionRecord] = []
        self._async_tasks: Dict[str, asyncio.Task] = {}
        self._cron_cache: Dict[str, CronExpression] = {}
        self._running = False
        self._scheduler_task: Optional[asyncio.Task] = None
        self._data_dir = Path(__file__).parent.parent / "data"
        self._schedule_file = self._data_dir / "scheduler.json"
        self._max_history = 100

    def _get_cron(self, task: ScheduledTask) -> Optional[CronExpression]:
        """Get parsed cron expression for a task, with caching."""
        if task.schedule_type != "cron" or not task.cron_expression:
            return None
        if task.id not in self._cron_cache:
            try:
                self._cron_cache[task.id] = CronExpression(task.cron_expression)
            except CronParseError:
                return None
        return self._cron_cache[task.id]

    def _compute_next_cron_run(self, task: ScheduledTask) -> Optional[float]:
        """Compute the next run timestamp for a cron task."""
        cron = self._get_cron(task)
        if not cron:
            return None
        next_dt = cron.next_run(after=datetime.now())
        if next_dt is None:
            return None
        return next_dt.timestamp()

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="scheduler",
            name="Scheduler",
            version="2.0.0",
            category="autonomy",
            description="Schedule future tasks, recurring jobs, cron-based schedules, and time-based actions for autonomous operation",
            actions=[
                SkillAction(
                    name="schedule",
                    description="Schedule a skill action to run after a delay or on a recurring basis",
                    parameters={
                        "name": {
                            "type": "string",
                            "required": True,
                            "description": "Human-readable name for this scheduled task"
                        },
                        "skill_id": {
                            "type": "string",
                            "required": True,
                            "description": "ID of the skill to execute"
                        },
                        "action": {
                            "type": "string",
                            "required": True,
                            "description": "Action name within the skill"
                        },
                        "params": {
                            "type": "object",
                            "required": False,
                            "description": "Parameters to pass to the action"
                        },
                        "delay_seconds": {
                            "type": "number",
                            "required": False,
                            "description": "Seconds to wait before first execution (default 0)"
                        },
                        "interval_seconds": {
                            "type": "number",
                            "required": False,
                            "description": "For recurring: seconds between executions"
                        },
                        "recurring": {
                            "type": "boolean",
                            "required": False,
                            "description": "Whether to repeat (default false = one-shot)"
                        },
                        "max_runs": {
                            "type": "integer",
                            "required": False,
                            "description": "Maximum number of runs (0 = unlimited, default 0)"
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="schedule_cron",
                    description="Schedule a skill action using a cron expression (e.g. '*/5 * * * *', '0 9 * * mon-fri', '@daily')",
                    parameters={
                        "name": {
                            "type": "string",
                            "required": True,
                            "description": "Human-readable name for this scheduled task"
                        },
                        "cron_expression": {
                            "type": "string",
                            "required": True,
                            "description": "Cron expression: 'min hour dom month dow' or alias (@daily, @hourly, @weekly, @monthly, @yearly)"
                        },
                        "skill_id": {
                            "type": "string",
                            "required": True,
                            "description": "ID of the skill to execute"
                        },
                        "action": {
                            "type": "string",
                            "required": True,
                            "description": "Action name within the skill"
                        },
                        "params": {
                            "type": "object",
                            "required": False,
                            "description": "Parameters to pass to the action"
                        },
                        "max_runs": {
                            "type": "integer",
                            "required": False,
                            "description": "Maximum number of runs (0 = unlimited, default 0)"
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="parse_cron",
                    description="Parse and validate a cron expression, showing the next N run times and human-readable description",
                    parameters={
                        "cron_expression": {
                            "type": "string",
                            "required": True,
                            "description": "Cron expression to parse and validate"
                        },
                        "show_next": {
                            "type": "integer",
                            "required": False,
                            "description": "Number of upcoming run times to show (default 5)"
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="cancel",
                    description="Cancel a scheduled task",
                    parameters={
                        "task_id": {
                            "type": "string",
                            "required": True,
                            "description": "ID of the task to cancel"
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="list",
                    description="List all scheduled tasks and their status",
                    parameters={
                        "include_completed": {
                            "type": "boolean",
                            "required": False,
                            "description": "Include completed/cancelled tasks (default false)"
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="history",
                    description="View execution history of scheduled tasks",
                    parameters={
                        "task_id": {
                            "type": "string",
                            "required": False,
                            "description": "Filter history by task ID"
                        },
                        "limit": {
                            "type": "integer",
                            "required": False,
                            "description": "Max records to return (default 20)"
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="pause",
                    description="Pause a recurring or cron task (can be resumed later)",
                    parameters={
                        "task_id": {
                            "type": "string",
                            "required": True,
                            "description": "ID of the task to pause"
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="resume",
                    description="Resume a paused task",
                    parameters={
                        "task_id": {
                            "type": "string",
                            "required": True,
                            "description": "ID of the task to resume"
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="run_now",
                    description="Immediately execute a scheduled task (does not affect its schedule)",
                    parameters={
                        "task_id": {
                            "type": "string",
                            "required": True,
                            "description": "ID of the task to run immediately"
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="pending",
                    description="Get tasks that are due to run soon (within next N seconds)",
                    parameters={
                        "within_seconds": {
                            "type": "number",
                            "required": False,
                            "description": "Look-ahead window in seconds (default 60)"
                        },
                    },
                    estimated_cost=0,
                ),
            ],
            required_credentials=[],
        )

    def check_credentials(self) -> bool:
        return True

    async def execute(self, action: str, params: Dict) -> SkillResult:
        handlers = {
            "schedule": self._schedule,
            "schedule_cron": self._schedule_cron,
            "parse_cron": self._parse_cron,
            "cancel": self._cancel,
            "list": self._list,
            "history": self._history,
            "pause": self._pause,
            "resume": self._resume,
            "run_now": self._run_now,
            "pending": self._pending,
        }

        handler = handlers.get(action)
        if not handler:
            return SkillResult(success=False, message=f"Unknown action: {action}")
        return await handler(params)

    async def _schedule(self, params: Dict) -> SkillResult:
        """Schedule a new task."""
        name = params.get("name", "").strip()
        skill_id = params.get("skill_id", "").strip()
        action = params.get("action", "").strip()
        task_params = params.get("params", {})
        delay = params.get("delay_seconds", 0)
        interval = params.get("interval_seconds", 0)
        recurring = params.get("recurring", False)
        max_runs = params.get("max_runs", 0)

        if not name:
            return SkillResult(success=False, message="Task name is required")
        if not skill_id:
            return SkillResult(success=False, message="skill_id is required")
        if not action:
            return SkillResult(success=False, message="action is required")

        # Validate that the target skill exists (if we have context)
        if self.context:
            available_skills = self.context.list_skills()
            if skill_id not in available_skills:
                return SkillResult(
                    success=False,
                    message=f"Skill '{skill_id}' not found. Available: {available_skills}"
                )

        if recurring and interval <= 0:
            return SkillResult(
                success=False,
                message="Recurring tasks require interval_seconds > 0"
            )

        schedule_type = "recurring" if recurring else "once"
        now = time.time()
        next_run = now + max(delay, 0)

        task = ScheduledTask(
            id=f"sched_{uuid.uuid4().hex[:8]}",
            name=name,
            skill_id=skill_id,
            action=action,
            params=task_params,
            schedule_type=schedule_type,
            interval_seconds=interval if recurring else delay,
            created_at=datetime.now().isoformat(),
            next_run_at=next_run,
            max_runs=max_runs,
        )

        self._tasks[task.id] = task
        self._save_schedule()

        return SkillResult(
            success=True,
            message=f"Scheduled '{name}' ({schedule_type}) - next run in {max(delay, 0):.0f}s",
            data=task.to_dict()
        )

    async def _schedule_cron(self, params: Dict) -> SkillResult:
        """Schedule a task using a cron expression."""
        name = params.get("name", "").strip()
        cron_expr = params.get("cron_expression", "").strip()
        skill_id = params.get("skill_id", "").strip()
        action = params.get("action", "").strip()
        task_params = params.get("params", {})
        max_runs = params.get("max_runs", 0)

        if not name:
            return SkillResult(success=False, message="Task name is required")
        if not cron_expr:
            return SkillResult(success=False, message="cron_expression is required")
        if not skill_id:
            return SkillResult(success=False, message="skill_id is required")
        if not action:
            return SkillResult(success=False, message="action is required")

        # Validate cron expression
        try:
            cron = CronExpression(cron_expr)
        except CronParseError as e:
            return SkillResult(
                success=False,
                message=f"Invalid cron expression '{cron_expr}': {e}"
            )

        # Compute next run time
        next_dt = cron.next_run(after=datetime.now())
        if next_dt is None:
            return SkillResult(
                success=False,
                message=f"Cron expression '{cron_expr}' has no upcoming run times"
            )

        # Validate that the target skill exists (if we have context)
        if self.context:
            available_skills = self.context.list_skills()
            if skill_id not in available_skills:
                return SkillResult(
                    success=False,
                    message=f"Skill '{skill_id}' not found. Available: {available_skills}"
                )

        task_id = f"sched_{uuid.uuid4().hex[:8]}"
        task = ScheduledTask(
            id=task_id,
            name=name,
            skill_id=skill_id,
            action=action,
            params=task_params,
            schedule_type="cron",
            interval_seconds=0,
            created_at=datetime.now().isoformat(),
            next_run_at=next_dt.timestamp(),
            max_runs=max_runs,
            cron_expression=cron_expr,
        )

        self._tasks[task.id] = task
        self._cron_cache[task.id] = cron
        self._save_schedule()

        remaining = next_dt.timestamp() - time.time()
        time_desc = f"{remaining:.0f}s" if remaining < 3600 else f"{remaining/3600:.1f}h"

        return SkillResult(
            success=True,
            message=f"Scheduled cron '{name}' ({cron.describe()}) - next run: {next_dt.strftime('%Y-%m-%d %H:%M')} (in {time_desc})",
            data={
                **task.to_dict(),
                "cron_description": cron.describe(),
                "next_run_iso": next_dt.isoformat(),
            }
        )

    async def _parse_cron(self, params: Dict) -> SkillResult:
        """Parse and validate a cron expression, showing upcoming runs."""
        cron_expr = params.get("cron_expression", "").strip()
        show_next = params.get("show_next", 5)

        if not cron_expr:
            return SkillResult(success=False, message="cron_expression is required")

        try:
            cron = CronExpression(cron_expr)
        except CronParseError as e:
            return SkillResult(
                success=False,
                message=f"Invalid cron expression '{cron_expr}': {e}"
            )

        upcoming = cron.next_n_runs(show_next, after=datetime.now())
        upcoming_strs = [dt.strftime("%Y-%m-%d %H:%M (%a)") for dt in upcoming]

        return SkillResult(
            success=True,
            message=f"Valid cron: {cron.describe()}",
            data={
                "expression": cron_expr,
                "description": cron.describe(),
                "upcoming_runs": upcoming_strs,
                "fields": {
                    "minutes": sorted(cron.minutes),
                    "hours": sorted(cron.hours),
                    "days_of_month": sorted(cron.days_of_month),
                    "months": sorted(cron.months),
                    "days_of_week": sorted(cron.days_of_week),
                },
            }
        )

    async def _cancel(self, params: Dict) -> SkillResult:
        """Cancel a scheduled task."""
        task_id = params.get("task_id", "").strip()
        if not task_id:
            return SkillResult(success=False, message="task_id required")

        task = self._tasks.get(task_id)
        if not task:
            return SkillResult(success=False, message=f"Task not found: {task_id}")

        task.status = "cancelled"
        task.enabled = False

        if task_id in self._async_tasks:
            self._async_tasks[task_id].cancel()
            del self._async_tasks[task_id]

        self._cron_cache.pop(task_id, None)
        self._save_schedule()

        return SkillResult(
            success=True,
            message=f"Cancelled task '{task.name}' ({task_id})",
            data=task.to_dict()
        )

    async def _list(self, params: Dict) -> SkillResult:
        """List all scheduled tasks."""
        include_completed = params.get("include_completed", False)

        tasks = []
        for task in self._tasks.values():
            if not include_completed and task.status in ("completed", "cancelled"):
                continue
            task_info = task.to_dict()
            if task.enabled and task.status == "pending":
                remaining = task.next_run_at - time.time()
                if remaining > 0:
                    task_info["next_run_in"] = f"{remaining:.0f}s"
                else:
                    task_info["next_run_in"] = "overdue"
                if task.schedule_type == "cron" and task.cron_expression:
                    cron = self._get_cron(task)
                    if cron:
                        task_info["cron_description"] = cron.describe()
            tasks.append(task_info)

        active_count = sum(1 for t in tasks if t.get("status") == "pending" and t.get("enabled"))
        return SkillResult(
            success=True,
            message=f"{len(tasks)} scheduled tasks ({active_count} active)",
            data={"tasks": tasks, "total": len(tasks), "active": active_count}
        )

    async def _history(self, params: Dict) -> SkillResult:
        """View execution history."""
        task_id = params.get("task_id")
        limit = params.get("limit", 20)

        records = self._execution_history
        if task_id:
            records = [r for r in records if r.task_id == task_id]

        records = records[-limit:][::-1]

        history = []
        for r in records:
            history.append({
                "task_id": r.task_id,
                "task_name": r.task_name,
                "executed_at": r.executed_at,
                "success": r.success,
                "message": r.message,
                "duration_seconds": r.duration_seconds,
            })

        return SkillResult(
            success=True,
            message=f"{len(history)} execution records",
            data={"history": history, "total": len(history)}
        )

    async def _pause(self, params: Dict) -> SkillResult:
        """Pause a recurring or cron task."""
        task_id = params.get("task_id", "").strip()
        if not task_id:
            return SkillResult(success=False, message="task_id required")

        task = self._tasks.get(task_id)
        if not task:
            return SkillResult(success=False, message=f"Task not found: {task_id}")

        if task.schedule_type not in ("recurring", "cron"):
            return SkillResult(success=False, message="Only recurring or cron tasks can be paused")

        task.enabled = False
        self._save_schedule()

        return SkillResult(
            success=True,
            message=f"Paused task '{task.name}'",
            data=task.to_dict()
        )

    async def _resume(self, params: Dict) -> SkillResult:
        """Resume a paused task."""
        task_id = params.get("task_id", "").strip()
        if not task_id:
            return SkillResult(success=False, message="task_id required")

        task = self._tasks.get(task_id)
        if not task:
            return SkillResult(success=False, message=f"Task not found: {task_id}")

        if task.status == "cancelled":
            return SkillResult(success=False, message="Cannot resume a cancelled task")

        task.enabled = True
        task.status = "pending"

        if task.schedule_type == "cron":
            next_ts = self._compute_next_cron_run(task)
            if next_ts:
                task.next_run_at = next_ts
                next_dt = datetime.fromtimestamp(next_ts)
                self._save_schedule()
                return SkillResult(
                    success=True,
                    message=f"Resumed cron task '{task.name}' - next run: {next_dt.strftime('%Y-%m-%d %H:%M')}",
                    data=task.to_dict()
                )
            else:
                return SkillResult(
                    success=False,
                    message=f"Could not compute next run for cron expression '{task.cron_expression}'"
                )
        else:
            task.next_run_at = time.time() + task.interval_seconds
            self._save_schedule()
            return SkillResult(
                success=True,
                message=f"Resumed task '{task.name}' - next run in {task.interval_seconds:.0f}s",
                data=task.to_dict()
            )

    async def _run_now(self, params: Dict) -> SkillResult:
        """Execute a scheduled task immediately."""
        task_id = params.get("task_id", "").strip()
        if not task_id:
            return SkillResult(success=False, message="task_id required")

        task = self._tasks.get(task_id)
        if not task:
            return SkillResult(success=False, message=f"Task not found: {task_id}")

        result = await self._execute_task(task)
        return result

    async def _pending(self, params: Dict) -> SkillResult:
        """Get tasks due to run within a time window."""
        within = params.get("within_seconds", 60)
        now = time.time()
        cutoff = now + within

        pending_tasks = []
        for task in self._tasks.values():
            if (task.enabled and
                task.status == "pending" and
                task.next_run_at <= cutoff):
                task_info = task.to_dict()
                remaining = task.next_run_at - now
                task_info["due_in_seconds"] = max(0, remaining)
                task_info["overdue"] = remaining < 0
                pending_tasks.append(task_info)

        pending_tasks.sort(key=lambda t: t.get("next_run_at", 0))

        return SkillResult(
            success=True,
            message=f"{len(pending_tasks)} tasks due within {within:.0f}s",
            data={"pending": pending_tasks, "count": len(pending_tasks)}
        )

    async def _execute_task(self, task: ScheduledTask) -> SkillResult:
        """Execute a single scheduled task via skill context."""
        if not self.context:
            return SkillResult(
                success=False,
                message="No skill context available - cannot execute tasks"
            )

        start_time = time.time()
        task.status = "running"

        try:
            result = await self.context.call_skill(
                task.skill_id, task.action, task.params
            )

            duration = time.time() - start_time
            task.run_count += 1
            task.last_run_at = datetime.now().isoformat()
            task.last_result = result.message[:200]
            task.last_success = result.success

            record = ExecutionRecord(
                task_id=task.id,
                task_name=task.name,
                executed_at=datetime.now().isoformat(),
                success=result.success,
                message=result.message[:200],
                duration_seconds=round(duration, 3),
            )
            self._execution_history.append(record)
            if len(self._execution_history) > self._max_history:
                self._execution_history = self._execution_history[-self._max_history:]

            # Update task status based on type
            if task.schedule_type == "once":
                task.status = "completed"
            elif task.schedule_type == "recurring":
                if task.max_runs > 0 and task.run_count >= task.max_runs:
                    task.status = "completed"
                else:
                    task.status = "pending"
                    task.next_run_at = time.time() + task.interval_seconds
            elif task.schedule_type == "cron":
                if task.max_runs > 0 and task.run_count >= task.max_runs:
                    task.status = "completed"
                else:
                    task.status = "pending"
                    next_ts = self._compute_next_cron_run(task)
                    if next_ts:
                        task.next_run_at = next_ts
                    else:
                        task.status = "completed"

            self._save_schedule()

            return SkillResult(
                success=True,
                message=f"Executed '{task.name}': {result.message[:100]}",
                data={
                    "task": task.to_dict(),
                    "execution": {
                        "success": result.success,
                        "message": result.message,
                        "duration": round(duration, 3),
                    }
                }
            )

        except Exception as e:
            duration = time.time() - start_time
            task.last_run_at = datetime.now().isoformat()
            task.last_result = str(e)[:200]
            task.last_success = False

            record = ExecutionRecord(
                task_id=task.id,
                task_name=task.name,
                executed_at=datetime.now().isoformat(),
                success=False,
                message=str(e)[:200],
                duration_seconds=round(duration, 3),
            )
            self._execution_history.append(record)

            if task.schedule_type == "once":
                task.status = "failed"
            elif task.schedule_type == "recurring":
                task.status = "pending"
                task.next_run_at = time.time() + task.interval_seconds
            elif task.schedule_type == "cron":
                task.status = "pending"
                next_ts = self._compute_next_cron_run(task)
                if next_ts:
                    task.next_run_at = next_ts
                else:
                    task.status = "failed"

            self._save_schedule()

            return SkillResult(
                success=False,
                message=f"Task '{task.name}' failed: {e}",
                data={"task": task.to_dict()}
            )

    async def tick(self) -> List[SkillResult]:
        """
        Check and execute any due tasks. Should be called periodically
        by the agent's main loop.

        Returns list of results from executed tasks.
        """
        now = time.time()
        results = []

        for task in list(self._tasks.values()):
            if (task.enabled and
                task.status == "pending" and
                task.next_run_at <= now):
                result = await self._execute_task(task)
                results.append(result)

        return results

    def get_due_count(self) -> int:
        """Get count of tasks that are currently due (for agent loop integration)."""
        now = time.time()
        return sum(
            1 for task in self._tasks.values()
            if task.enabled and task.status == "pending" and task.next_run_at <= now
        )

    def _save_schedule(self):
        """Persist schedule to disk."""
        try:
            self._data_dir.mkdir(parents=True, exist_ok=True)
            data = {
                "tasks": {tid: t.to_dict() for tid, t in self._tasks.items()},
                "saved_at": datetime.now().isoformat(),
            }
            self._schedule_file.write_text(json.dumps(data, indent=2))
        except Exception:
            pass

    def load_schedule(self):
        """Load schedule from disk (call on startup)."""
        try:
            if self._schedule_file.exists():
                data = json.loads(self._schedule_file.read_text())
                for tid, tdata in data.get("tasks", {}).items():
                    if tdata.get("status") in ("completed", "cancelled"):
                        continue
                    task = ScheduledTask(
                        id=tdata["id"],
                        name=tdata["name"],
                        skill_id=tdata["skill_id"],
                        action=tdata["action"],
                        params=tdata.get("params", {}),
                        schedule_type=tdata["schedule_type"],
                        interval_seconds=tdata["interval_seconds"],
                        created_at=tdata["created_at"],
                        next_run_at=tdata["next_run_at"],
                        status=tdata.get("status", "pending"),
                        run_count=tdata.get("run_count", 0),
                        max_runs=tdata.get("max_runs", 0),
                        last_run_at=tdata.get("last_run_at"),
                        last_result=tdata.get("last_result"),
                        last_success=tdata.get("last_success"),
                        enabled=tdata.get("enabled", True),
                        cron_expression=tdata.get("cron_expression"),
                    )
                    self._tasks[tid] = task
                    # Re-compute next_run for cron tasks on load
                    if task.schedule_type == "cron" and task.cron_expression:
                        next_ts = self._compute_next_cron_run(task)
                        if next_ts:
                            task.next_run_at = next_ts
        except Exception:
            pass
