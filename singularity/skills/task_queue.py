#!/usr/bin/env python3
"""
TaskQueueSkill - Persistent task queue for autonomous agent work management.

Provides a file-backed priority queue that enables the agent to:
- Accept work from external clients, other agents, or itself
- Manage tasks with priorities, deadlines, and dependencies
- Track task outcomes for performance analysis
- Retry failed tasks with exponential backoff
- Generate work reports and statistics

Serves all four pillars:
- Revenue: Accept and track client work requests
- Replication: Coordinate work between agent instances
- Goal Setting: Break goals into executable tasks
- Self-Improvement: Analyze task success/failure patterns

Zero external dependencies - uses only Python stdlib.
"""

import json
import os
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from .base import Skill, SkillManifest, SkillAction, SkillResult


# Default storage directory
DEFAULT_QUEUE_DIR = Path(__file__).parent.parent / "data" / "task_queue"


class TaskQueueSkill(Skill):
    """Persistent task queue with priority, dependencies, and retry support."""

    # Task statuses
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    BLOCKED = "blocked"  # waiting on dependencies

    # Priority levels
    CRITICAL = 0
    HIGH = 1
    MEDIUM = 2
    LOW = 3

    PRIORITY_NAMES = {0: "critical", 1: "high", 2: "medium", 3: "low"}
    PRIORITY_FROM_NAME = {"critical": 0, "high": 1, "medium": 2, "low": 3}

    def __init__(self, credentials: Dict = None, queue_dir: str = None):
        super().__init__(credentials)
        self._queue_dir = Path(queue_dir) if queue_dir else DEFAULT_QUEUE_DIR
        self._tasks: Dict[str, Dict] = {}
        self._loaded = False

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="taskqueue",
            name="Task Queue",
            version="1.0.0",
            category="workflow",
            description="Persistent task queue with priorities, dependencies, retry, and analytics",
            actions=[
                SkillAction(
                    name="enqueue",
                    description="Add a new task to the queue",
                    parameters={
                        "title": {
                            "type": "string",
                            "required": True,
                            "description": "Short task title",
                        },
                        "description": {
                            "type": "string",
                            "required": False,
                            "description": "Detailed task description",
                        },
                        "priority": {
                            "type": "string",
                            "required": False,
                            "description": "Priority: critical, high, medium (default), low",
                        },
                        "category": {
                            "type": "string",
                            "required": False,
                            "description": "Task category (e.g., revenue, improvement, client_work)",
                        },
                        "deadline": {
                            "type": "string",
                            "required": False,
                            "description": "ISO format deadline (e.g., 2025-01-15T12:00:00)",
                        },
                        "depends_on": {
                            "type": "string",
                            "required": False,
                            "description": "Comma-separated task IDs this depends on",
                        },
                        "max_retries": {
                            "type": "integer",
                            "required": False,
                            "description": "Max retry attempts on failure (default: 3)",
                        },
                        "metadata": {
                            "type": "object",
                            "required": False,
                            "description": "Additional metadata (client_id, source, etc.)",
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="dequeue",
                    description="Get the next highest-priority task ready for execution",
                    parameters={
                        "category": {
                            "type": "string",
                            "required": False,
                            "description": "Only dequeue from this category",
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="complete",
                    description="Mark a task as completed with results",
                    parameters={
                        "task_id": {
                            "type": "string",
                            "required": True,
                            "description": "ID of the task to complete",
                        },
                        "result": {
                            "type": "string",
                            "required": False,
                            "description": "Result summary",
                        },
                        "output": {
                            "type": "object",
                            "required": False,
                            "description": "Structured output data",
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="fail",
                    description="Mark a task as failed (will auto-retry if retries remain)",
                    parameters={
                        "task_id": {
                            "type": "string",
                            "required": True,
                            "description": "ID of the task that failed",
                        },
                        "reason": {
                            "type": "string",
                            "required": True,
                            "description": "Why the task failed",
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="cancel",
                    description="Cancel a pending or in-progress task",
                    parameters={
                        "task_id": {
                            "type": "string",
                            "required": True,
                            "description": "ID of the task to cancel",
                        },
                        "reason": {
                            "type": "string",
                            "required": False,
                            "description": "Cancellation reason",
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="list",
                    description="List tasks filtered by status, category, or priority",
                    parameters={
                        "status": {
                            "type": "string",
                            "required": False,
                            "description": "Filter by status: pending, in_progress, completed, failed, cancelled, blocked",
                        },
                        "category": {
                            "type": "string",
                            "required": False,
                            "description": "Filter by category",
                        },
                        "priority": {
                            "type": "string",
                            "required": False,
                            "description": "Filter by priority: critical, high, medium, low",
                        },
                        "limit": {
                            "type": "integer",
                            "required": False,
                            "description": "Max results (default: 20)",
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="get",
                    description="Get full details of a specific task",
                    parameters={
                        "task_id": {
                            "type": "string",
                            "required": True,
                            "description": "Task ID to retrieve",
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="stats",
                    description="Get queue statistics and performance metrics",
                    parameters={},
                    estimated_cost=0,
                ),
                SkillAction(
                    name="retry_failed",
                    description="Re-enqueue all failed tasks that have retries remaining",
                    parameters={
                        "category": {
                            "type": "string",
                            "required": False,
                            "description": "Only retry failed tasks in this category",
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="prioritize",
                    description="Change the priority of an existing task",
                    parameters={
                        "task_id": {
                            "type": "string",
                            "required": True,
                            "description": "Task ID",
                        },
                        "priority": {
                            "type": "string",
                            "required": True,
                            "description": "New priority: critical, high, medium, low",
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="clear_completed",
                    description="Archive and remove completed tasks older than N hours",
                    parameters={
                        "hours": {
                            "type": "integer",
                            "required": False,
                            "description": "Remove tasks completed more than N hours ago (default: 24)",
                        },
                    },
                    estimated_cost=0,
                ),
            ],
            required_credentials=[],
        )

    def check_credentials(self) -> bool:
        return True  # No credentials needed

    def _ensure_loaded(self):
        """Load tasks from disk if not already loaded."""
        if self._loaded:
            return
        self._queue_dir.mkdir(parents=True, exist_ok=True)
        tasks_file = self._queue_dir / "tasks.json"
        if tasks_file.exists():
            try:
                with open(tasks_file, "r") as f:
                    self._tasks = json.load(f)
            except (json.JSONDecodeError, IOError):
                self._tasks = {}
        self._loaded = True

    def _save(self):
        """Persist tasks to disk."""
        self._queue_dir.mkdir(parents=True, exist_ok=True)
        tasks_file = self._queue_dir / "tasks.json"
        with open(tasks_file, "w") as f:
            json.dump(self._tasks, f, indent=2, default=str)

    def _resolve_priority(self, priority_str: str) -> int:
        """Convert priority string to int."""
        if not priority_str:
            return self.MEDIUM
        p = priority_str.strip().lower()
        return self.PRIORITY_FROM_NAME.get(p, self.MEDIUM)

    def _check_dependencies(self, task: Dict) -> bool:
        """Check if all dependencies are completed."""
        deps = task.get("depends_on", [])
        if not deps:
            return True
        for dep_id in deps:
            dep_task = self._tasks.get(dep_id)
            if not dep_task or dep_task.get("status") != self.COMPLETED:
                return False
        return True

    def _update_blocked_status(self):
        """Update blocked/pending status based on dependencies."""
        for task_id, task in self._tasks.items():
            if task["status"] in (self.COMPLETED, self.FAILED, self.CANCELLED, self.IN_PROGRESS):
                continue
            deps = task.get("depends_on", [])
            if deps:
                # Check if any dependency failed or was cancelled
                any_failed = any(
                    self._tasks.get(d, {}).get("status") in (self.FAILED, self.CANCELLED)
                    for d in deps
                )
                if any_failed:
                    task["status"] = self.BLOCKED
                elif self._check_dependencies(task):
                    if task["status"] == self.BLOCKED:
                        task["status"] = self.PENDING
                else:
                    task["status"] = self.BLOCKED

    async def execute(self, action: str, params: Dict) -> SkillResult:
        self._ensure_loaded()

        actions_map = {
            "enqueue": self._enqueue,
            "dequeue": self._dequeue,
            "complete": self._complete,
            "fail": self._fail,
            "cancel": self._cancel,
            "list": self._list,
            "get": self._get,
            "stats": self._stats,
            "retry_failed": self._retry_failed,
            "prioritize": self._prioritize,
            "clear_completed": self._clear_completed,
        }

        handler = actions_map.get(action)
        if not handler:
            return SkillResult(success=False, message=f"Unknown action: {action}")

        return handler(params)

    def _enqueue(self, params: Dict) -> SkillResult:
        title = params.get("title", "").strip()
        if not title:
            return SkillResult(success=False, message="Task title is required")

        task_id = str(uuid.uuid4())[:8]
        now = datetime.now().isoformat()

        # Parse dependencies
        depends_on = []
        deps_str = params.get("depends_on", "")
        if deps_str:
            depends_on = [d.strip() for d in deps_str.split(",") if d.strip()]
            # Validate dependencies exist
            for dep_id in depends_on:
                if dep_id not in self._tasks:
                    return SkillResult(
                        success=False,
                        message=f"Dependency task not found: {dep_id}",
                    )

        priority = self._resolve_priority(params.get("priority", "medium"))

        task = {
            "id": task_id,
            "title": title,
            "description": params.get("description", ""),
            "status": self.PENDING,
            "priority": priority,
            "priority_name": self.PRIORITY_NAMES.get(priority, "medium"),
            "category": params.get("category", "general"),
            "created_at": now,
            "updated_at": now,
            "started_at": None,
            "completed_at": None,
            "deadline": params.get("deadline"),
            "depends_on": depends_on,
            "max_retries": int(params.get("max_retries", 3)),
            "retry_count": 0,
            "result": None,
            "output": None,
            "failure_reasons": [],
            "metadata": params.get("metadata", {}),
        }

        # Check if blocked by dependencies
        if depends_on and not self._check_dependencies(task):
            task["status"] = self.BLOCKED

        self._tasks[task_id] = task
        self._save()

        return SkillResult(
            success=True,
            message=f"Task enqueued: {title} [{task_id}] (priority: {task['priority_name']})",
            data={"task_id": task_id, "status": task["status"], "priority": task["priority_name"]},
        )

    def _dequeue(self, params: Dict) -> SkillResult:
        self._update_blocked_status()

        category_filter = params.get("category")

        # Find eligible tasks: pending, not blocked, sorted by priority then creation time
        eligible = []
        for task in self._tasks.values():
            if task["status"] != self.PENDING:
                continue
            if category_filter and task.get("category") != category_filter:
                continue
            if not self._check_dependencies(task):
                continue

            # Check if past retry backoff period
            if task.get("retry_count", 0) > 0:
                last_failure = task.get("failure_reasons", [])
                if last_failure:
                    last_time = last_failure[-1].get("timestamp", "")
                    if last_time:
                        try:
                            last_dt = datetime.fromisoformat(last_time)
                            backoff_seconds = min(300, 2 ** task["retry_count"] * 10)
                            if datetime.now() < last_dt + timedelta(seconds=backoff_seconds):
                                continue  # Still in backoff period
                        except (ValueError, TypeError):
                            pass

            eligible.append(task)

        if not eligible:
            return SkillResult(
                success=False,
                message="No tasks available in queue",
                data={"queue_size": len(self._tasks)},
            )

        # Sort: lower priority number = higher priority, then by creation time
        eligible.sort(key=lambda t: (t["priority"], t["created_at"]))

        # Also prioritize tasks with deadlines approaching
        def urgency_key(t):
            base = t["priority"]
            if t.get("deadline"):
                try:
                    deadline = datetime.fromisoformat(t["deadline"])
                    hours_left = (deadline - datetime.now()).total_seconds() / 3600
                    if hours_left < 1:
                        base = -1  # Urgent override
                    elif hours_left < 24:
                        base = min(base, 0)  # Bump to critical
                except (ValueError, TypeError):
                    pass
            return (base, t["created_at"])

        eligible.sort(key=urgency_key)

        task = eligible[0]
        task["status"] = self.IN_PROGRESS
        task["started_at"] = datetime.now().isoformat()
        task["updated_at"] = datetime.now().isoformat()
        self._save()

        return SkillResult(
            success=True,
            message=f"Dequeued: {task['title']} [{task['id']}]",
            data={
                "task_id": task["id"],
                "title": task["title"],
                "description": task["description"],
                "priority": task["priority_name"],
                "category": task.get("category", "general"),
                "deadline": task.get("deadline"),
                "metadata": task.get("metadata", {}),
                "retry_count": task.get("retry_count", 0),
            },
        )

    def _complete(self, params: Dict) -> SkillResult:
        task_id = params.get("task_id", "").strip()
        if not task_id:
            return SkillResult(success=False, message="task_id is required")

        task = self._tasks.get(task_id)
        if not task:
            return SkillResult(success=False, message=f"Task not found: {task_id}")

        if task["status"] not in (self.IN_PROGRESS, self.PENDING):
            return SkillResult(
                success=False,
                message=f"Cannot complete task with status '{task['status']}'",
            )

        now = datetime.now().isoformat()
        task["status"] = self.COMPLETED
        task["completed_at"] = now
        task["updated_at"] = now
        task["result"] = params.get("result", "Completed successfully")
        task["output"] = params.get("output", {})

        # Calculate duration
        if task.get("started_at"):
            try:
                started = datetime.fromisoformat(task["started_at"])
                completed = datetime.fromisoformat(now)
                task["duration_seconds"] = (completed - started).total_seconds()
            except (ValueError, TypeError):
                pass

        # Update any blocked tasks that depend on this one
        self._update_blocked_status()
        self._save()

        return SkillResult(
            success=True,
            message=f"Task completed: {task['title']} [{task_id}]",
            data={
                "task_id": task_id,
                "duration_seconds": task.get("duration_seconds"),
                "unblocked": self._count_newly_unblocked(task_id),
            },
        )

    def _count_newly_unblocked(self, completed_id: str) -> int:
        """Count tasks that were unblocked by completing this task."""
        count = 0
        for task in self._tasks.values():
            if completed_id in task.get("depends_on", []) and task["status"] == self.PENDING:
                count += 1
        return count

    def _fail(self, params: Dict) -> SkillResult:
        task_id = params.get("task_id", "").strip()
        if not task_id:
            return SkillResult(success=False, message="task_id is required")

        reason = params.get("reason", "Unknown failure")

        task = self._tasks.get(task_id)
        if not task:
            return SkillResult(success=False, message=f"Task not found: {task_id}")

        if task["status"] not in (self.IN_PROGRESS, self.PENDING):
            return SkillResult(
                success=False,
                message=f"Cannot fail task with status '{task['status']}'",
            )

        now = datetime.now().isoformat()
        task["retry_count"] = task.get("retry_count", 0) + 1
        task["failure_reasons"].append({
            "reason": reason,
            "timestamp": now,
            "attempt": task["retry_count"],
        })
        task["updated_at"] = now

        max_retries = task.get("max_retries", 3)
        if task["retry_count"] < max_retries:
            # Re-enqueue for retry with backoff
            task["status"] = self.PENDING
            backoff = min(300, 2 ** task["retry_count"] * 10)
            self._save()
            return SkillResult(
                success=True,
                message=f"Task failed (attempt {task['retry_count']}/{max_retries}), will retry after {backoff}s backoff: {task['title']}",
                data={
                    "task_id": task_id,
                    "retry_count": task["retry_count"],
                    "max_retries": max_retries,
                    "backoff_seconds": backoff,
                    "will_retry": True,
                },
            )
        else:
            # No retries left - permanently failed
            task["status"] = self.FAILED
            task["completed_at"] = now
            self._update_blocked_status()
            self._save()
            return SkillResult(
                success=True,
                message=f"Task permanently failed after {max_retries} attempts: {task['title']}",
                data={
                    "task_id": task_id,
                    "retry_count": task["retry_count"],
                    "max_retries": max_retries,
                    "will_retry": False,
                    "all_reasons": [f["reason"] for f in task["failure_reasons"]],
                },
            )

    def _cancel(self, params: Dict) -> SkillResult:
        task_id = params.get("task_id", "").strip()
        if not task_id:
            return SkillResult(success=False, message="task_id is required")

        task = self._tasks.get(task_id)
        if not task:
            return SkillResult(success=False, message=f"Task not found: {task_id}")

        if task["status"] in (self.COMPLETED, self.CANCELLED):
            return SkillResult(
                success=False,
                message=f"Cannot cancel task with status '{task['status']}'",
            )

        now = datetime.now().isoformat()
        task["status"] = self.CANCELLED
        task["updated_at"] = now
        task["completed_at"] = now
        task["result"] = params.get("reason", "Cancelled")
        self._update_blocked_status()
        self._save()

        return SkillResult(
            success=True,
            message=f"Task cancelled: {task['title']} [{task_id}]",
            data={"task_id": task_id},
        )

    def _list(self, params: Dict) -> SkillResult:
        self._update_blocked_status()

        status_filter = params.get("status")
        category_filter = params.get("category")
        priority_filter = params.get("priority")
        limit = int(params.get("limit", 20))

        filtered = list(self._tasks.values())

        if status_filter:
            filtered = [t for t in filtered if t["status"] == status_filter]
        if category_filter:
            filtered = [t for t in filtered if t.get("category") == category_filter]
        if priority_filter:
            pval = self._resolve_priority(priority_filter)
            filtered = [t for t in filtered if t["priority"] == pval]

        # Sort by priority, then creation time
        filtered.sort(key=lambda t: (t["priority"], t["created_at"]))

        tasks_summary = []
        for t in filtered[:limit]:
            summary = {
                "id": t["id"],
                "title": t["title"],
                "status": t["status"],
                "priority": t["priority_name"],
                "category": t.get("category", "general"),
                "created_at": t["created_at"],
                "retry_count": t.get("retry_count", 0),
            }
            if t.get("deadline"):
                summary["deadline"] = t["deadline"]
            tasks_summary.append(summary)

        return SkillResult(
            success=True,
            message=f"Found {len(filtered)} tasks (showing {len(tasks_summary)})",
            data={"tasks": tasks_summary, "total": len(filtered)},
        )

    def _get(self, params: Dict) -> SkillResult:
        task_id = params.get("task_id", "").strip()
        if not task_id:
            return SkillResult(success=False, message="task_id is required")

        task = self._tasks.get(task_id)
        if not task:
            return SkillResult(success=False, message=f"Task not found: {task_id}")

        return SkillResult(
            success=True,
            message=f"Task: {task['title']} [{task_id}]",
            data={"task": task},
        )

    def _stats(self, params: Dict) -> SkillResult:
        self._update_blocked_status()

        status_counts = {}
        category_counts = {}
        priority_counts = {}
        total_duration = 0
        completed_count = 0
        total_retries = 0

        for task in self._tasks.values():
            status = task["status"]
            status_counts[status] = status_counts.get(status, 0) + 1

            cat = task.get("category", "general")
            category_counts[cat] = category_counts.get(cat, 0) + 1

            pname = task.get("priority_name", "medium")
            priority_counts[pname] = priority_counts.get(pname, 0) + 1

            if task["status"] == self.COMPLETED and task.get("duration_seconds"):
                total_duration += task["duration_seconds"]
                completed_count += 1

            total_retries += task.get("retry_count", 0)

        avg_duration = total_duration / completed_count if completed_count else 0

        total_tasks = len(self._tasks)
        success_rate = (
            status_counts.get(self.COMPLETED, 0) / total_tasks * 100
            if total_tasks > 0
            else 0
        )
        failure_rate = (
            status_counts.get(self.FAILED, 0) / total_tasks * 100
            if total_tasks > 0
            else 0
        )

        # Find overdue tasks
        overdue = 0
        now = datetime.now()
        for task in self._tasks.values():
            if task["status"] in (self.PENDING, self.IN_PROGRESS, self.BLOCKED) and task.get("deadline"):
                try:
                    deadline = datetime.fromisoformat(task["deadline"])
                    if now > deadline:
                        overdue += 1
                except (ValueError, TypeError):
                    pass

        return SkillResult(
            success=True,
            message=f"Queue stats: {total_tasks} total, {status_counts.get(self.PENDING, 0)} pending, {status_counts.get(self.IN_PROGRESS, 0)} active",
            data={
                "total_tasks": total_tasks,
                "by_status": status_counts,
                "by_category": category_counts,
                "by_priority": priority_counts,
                "success_rate_pct": round(success_rate, 1),
                "failure_rate_pct": round(failure_rate, 1),
                "avg_duration_seconds": round(avg_duration, 1),
                "total_retries": total_retries,
                "overdue_tasks": overdue,
            },
        )

    def _retry_failed(self, params: Dict) -> SkillResult:
        category_filter = params.get("category")
        retried = 0

        for task in self._tasks.values():
            if task["status"] != self.FAILED:
                continue
            if category_filter and task.get("category") != category_filter:
                continue
            if task.get("retry_count", 0) < task.get("max_retries", 3):
                task["status"] = self.PENDING
                task["updated_at"] = datetime.now().isoformat()
                retried += 1

        if retried > 0:
            self._save()

        return SkillResult(
            success=True,
            message=f"Re-enqueued {retried} failed tasks for retry",
            data={"retried": retried},
        )

    def _prioritize(self, params: Dict) -> SkillResult:
        task_id = params.get("task_id", "").strip()
        if not task_id:
            return SkillResult(success=False, message="task_id is required")

        priority_str = params.get("priority", "").strip()
        if not priority_str:
            return SkillResult(success=False, message="priority is required")

        task = self._tasks.get(task_id)
        if not task:
            return SkillResult(success=False, message=f"Task not found: {task_id}")

        if task["status"] in (self.COMPLETED, self.CANCELLED):
            return SkillResult(
                success=False,
                message=f"Cannot reprioritize task with status '{task['status']}'",
            )

        new_priority = self._resolve_priority(priority_str)
        old_priority_name = task["priority_name"]
        task["priority"] = new_priority
        task["priority_name"] = self.PRIORITY_NAMES.get(new_priority, "medium")
        task["updated_at"] = datetime.now().isoformat()
        self._save()

        return SkillResult(
            success=True,
            message=f"Task [{task_id}] priority changed: {old_priority_name} → {task['priority_name']}",
            data={"task_id": task_id, "old_priority": old_priority_name, "new_priority": task["priority_name"]},
        )

    def _clear_completed(self, params: Dict) -> SkillResult:
        hours = int(params.get("hours", 24))
        cutoff = datetime.now() - timedelta(hours=hours)
        archived = []

        to_remove = []
        for task_id, task in self._tasks.items():
            if task["status"] != self.COMPLETED:
                continue
            completed_at = task.get("completed_at")
            if completed_at:
                try:
                    completed_dt = datetime.fromisoformat(completed_at)
                    if completed_dt < cutoff:
                        to_remove.append(task_id)
                        archived.append({
                            "id": task_id,
                            "title": task["title"],
                            "completed_at": completed_at,
                        })
                except (ValueError, TypeError):
                    pass

        # Save to archive file before removing
        if archived:
            archive_file = self._queue_dir / "archive.jsonl"
            with open(archive_file, "a") as f:
                for task_id in to_remove:
                    f.write(json.dumps(self._tasks[task_id], default=str) + "\n")
                    del self._tasks[task_id]
            self._save()

        return SkillResult(
            success=True,
            message=f"Archived {len(archived)} completed tasks older than {hours} hours",
            data={"archived_count": len(archived), "archived_tasks": archived},
        )
