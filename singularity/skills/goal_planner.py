#!/usr/bin/env python3
"""
Goal Planner Skill - Structured goal setting, planning, and progress tracking.

Gives agents the ability to:
- Set goals with priorities, deadlines, and success criteria
- Break goals into actionable sub-tasks
- Track progress and update status
- Evaluate and reprioritize based on outcomes
- Maintain a persistent goal ledger across cycles

This addresses the Goal-Setting pillar: agents need structured planning
beyond just appending text to their prompt.
"""

import json
import time
from enum import Enum
from typing import Dict, List, Optional, Any
from .base import Skill, SkillManifest, SkillAction, SkillResult


class GoalStatus(str, Enum):
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"
    CANCELLED = "cancelled"


class GoalPriority(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class GoalPlannerSkill(Skill):
    """Skill for structured goal setting, planning, and progress tracking."""

    def __init__(self, credentials: Dict = None):
        super().__init__(credentials)
        self._goals: Dict[str, Dict] = {}
        self._next_id: int = 1
        self._history: List[Dict] = []

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="goals",
            name="Goal Planner",
            version="1.0.0",
            category="meta",
            description="Set goals, break them into tasks, track progress, and reprioritize",
            actions=[
                SkillAction(
                    name="set_goal",
                    description="Create a new goal with priority and success criteria",
                    parameters={
                        "title": {
                            "type": "string",
                            "required": True,
                            "description": "Short title for the goal",
                        },
                        "description": {
                            "type": "string",
                            "required": True,
                            "description": "Detailed description of what to achieve",
                        },
                        "priority": {
                            "type": "string",
                            "required": False,
                            "description": "Priority: critical, high, medium, low (default: medium)",
                        },
                        "success_criteria": {
                            "type": "string",
                            "required": False,
                            "description": "How to measure success (comma-separated criteria)",
                        },
                        "parent_id": {
                            "type": "string",
                            "required": False,
                            "description": "Parent goal ID if this is a sub-goal",
                        },
                        "deadline_minutes": {
                            "type": "number",
                            "required": False,
                            "description": "Deadline in minutes from now",
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="add_task",
                    description="Add a task (sub-step) to an existing goal",
                    parameters={
                        "goal_id": {
                            "type": "string",
                            "required": True,
                            "description": "ID of the goal to add a task to",
                        },
                        "task": {
                            "type": "string",
                            "required": True,
                            "description": "Description of the task",
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="complete_task",
                    description="Mark a task as completed within a goal",
                    parameters={
                        "goal_id": {
                            "type": "string",
                            "required": True,
                            "description": "ID of the goal",
                        },
                        "task_index": {
                            "type": "number",
                            "required": True,
                            "description": "Index of the task to complete (0-based)",
                        },
                        "outcome": {
                            "type": "string",
                            "required": False,
                            "description": "Notes on the outcome of this task",
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="update_progress",
                    description="Update progress notes on a goal",
                    parameters={
                        "goal_id": {
                            "type": "string",
                            "required": True,
                            "description": "ID of the goal",
                        },
                        "note": {
                            "type": "string",
                            "required": True,
                            "description": "Progress note to add",
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="complete_goal",
                    description="Mark a goal as completed with a summary",
                    parameters={
                        "goal_id": {
                            "type": "string",
                            "required": True,
                            "description": "ID of the goal to complete",
                        },
                        "summary": {
                            "type": "string",
                            "required": False,
                            "description": "Summary of what was achieved",
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="fail_goal",
                    description="Mark a goal as failed with a reason",
                    parameters={
                        "goal_id": {
                            "type": "string",
                            "required": True,
                            "description": "ID of the goal to mark as failed",
                        },
                        "reason": {
                            "type": "string",
                            "required": False,
                            "description": "Why the goal failed",
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="reprioritize",
                    description="Change the priority of a goal",
                    parameters={
                        "goal_id": {
                            "type": "string",
                            "required": True,
                            "description": "ID of the goal to reprioritize",
                        },
                        "new_priority": {
                            "type": "string",
                            "required": True,
                            "description": "New priority: critical, high, medium, low",
                        },
                        "reason": {
                            "type": "string",
                            "required": False,
                            "description": "Why the priority changed",
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="list_goals",
                    description="List all goals, optionally filtered by status",
                    parameters={
                        "status": {
                            "type": "string",
                            "required": False,
                            "description": "Filter by status: active, completed, failed, paused, cancelled",
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="get_goal",
                    description="Get detailed info about a specific goal including tasks and progress",
                    parameters={
                        "goal_id": {
                            "type": "string",
                            "required": True,
                            "description": "ID of the goal",
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="get_plan",
                    description="Get a prioritized action plan: what to work on next based on active goals",
                    parameters={},
                    estimated_cost=0,
                ),
                SkillAction(
                    name="evaluate",
                    description="Get a summary of goal performance: completion rate, active goals, overdue items",
                    parameters={},
                    estimated_cost=0,
                ),
                SkillAction(
                    name="export_goals",
                    description="Export all goals as JSON for persistence or transfer",
                    parameters={},
                    estimated_cost=0,
                ),
                SkillAction(
                    name="import_goals",
                    description="Import goals from a JSON string (e.g., from a previous session)",
                    parameters={
                        "goals_json": {
                            "type": "string",
                            "required": True,
                            "description": "JSON string of goals to import",
                        },
                    },
                    estimated_cost=0,
                ),
            ],
            required_credentials=[],
        )

    async def initialize(self) -> bool:
        self.initialized = True
        return True

    async def execute(self, action: str, params: Dict) -> SkillResult:
        handlers = {
            "set_goal": self._set_goal,
            "add_task": self._add_task,
            "complete_task": self._complete_task,
            "update_progress": self._update_progress,
            "complete_goal": self._complete_goal,
            "fail_goal": self._fail_goal,
            "reprioritize": self._reprioritize,
            "list_goals": self._list_goals,
            "get_goal": self._get_goal,
            "get_plan": self._get_plan,
            "evaluate": self._evaluate,
            "export_goals": self._export_goals,
            "import_goals": self._import_goals,
        }

        handler = handlers.get(action)
        if not handler:
            return SkillResult(success=False, message=f"Unknown action: {action}")

        return handler(params)

    def _generate_id(self) -> str:
        goal_id = f"G{self._next_id}"
        self._next_id += 1
        return goal_id

    def _get_goal_or_error(self, goal_id: str) -> Optional[Dict]:
        return self._goals.get(goal_id)

    def _record_event(self, goal_id: str, event_type: str, details: str):
        self._history.append({
            "goal_id": goal_id,
            "event": event_type,
            "details": details,
            "timestamp": time.time(),
        })

    # === Goal CRUD ===

    def _set_goal(self, params: Dict) -> SkillResult:
        title = params.get("title", "").strip()
        description = params.get("description", "").strip()
        if not title or not description:
            return SkillResult(success=False, message="Title and description are required")

        priority_str = params.get("priority", "medium").strip().lower()
        try:
            priority = GoalPriority(priority_str)
        except ValueError:
            return SkillResult(
                success=False,
                message=f"Invalid priority '{priority_str}'. Use: critical, high, medium, low",
            )

        criteria_str = params.get("success_criteria", "")
        criteria = [c.strip() for c in criteria_str.split(",") if c.strip()] if criteria_str else []

        parent_id = params.get("parent_id", "")
        if parent_id and parent_id not in self._goals:
            return SkillResult(success=False, message=f"Parent goal not found: {parent_id}")

        deadline = None
        deadline_minutes = params.get("deadline_minutes")
        if deadline_minutes is not None:
            try:
                deadline = time.time() + float(deadline_minutes) * 60
            except (ValueError, TypeError):
                pass

        goal_id = self._generate_id()
        now = time.time()

        self._goals[goal_id] = {
            "id": goal_id,
            "title": title,
            "description": description,
            "priority": priority.value,
            "status": GoalStatus.ACTIVE.value,
            "success_criteria": criteria,
            "tasks": [],
            "progress_notes": [],
            "parent_id": parent_id if parent_id else None,
            "children": [],
            "created_at": now,
            "updated_at": now,
            "deadline": deadline,
            "completed_at": None,
            "summary": None,
        }

        if parent_id and parent_id in self._goals:
            self._goals[parent_id]["children"].append(goal_id)

        self._record_event(goal_id, "created", f"Goal created: {title}")

        return SkillResult(
            success=True,
            message=f"Goal created: [{goal_id}] {title} (priority: {priority.value})",
            data={"goal_id": goal_id, "title": title, "priority": priority.value},
        )

    def _add_task(self, params: Dict) -> SkillResult:
        goal_id = params.get("goal_id", "").strip()
        task_desc = params.get("task", "").strip()
        if not goal_id or not task_desc:
            return SkillResult(success=False, message="goal_id and task are required")

        goal = self._get_goal_or_error(goal_id)
        if not goal:
            return SkillResult(success=False, message=f"Goal not found: {goal_id}")
        if goal["status"] != GoalStatus.ACTIVE.value:
            return SkillResult(success=False, message=f"Goal {goal_id} is not active")

        task = {
            "description": task_desc,
            "completed": False,
            "outcome": None,
            "added_at": time.time(),
            "completed_at": None,
        }
        goal["tasks"].append(task)
        goal["updated_at"] = time.time()
        task_index = len(goal["tasks"]) - 1

        self._record_event(goal_id, "task_added", task_desc)

        return SkillResult(
            success=True,
            message=f"Task added to {goal_id} at index {task_index}: {task_desc}",
            data={"goal_id": goal_id, "task_index": task_index},
        )

    def _complete_task(self, params: Dict) -> SkillResult:
        goal_id = params.get("goal_id", "").strip()
        if not goal_id:
            return SkillResult(success=False, message="goal_id is required")

        goal = self._get_goal_or_error(goal_id)
        if not goal:
            return SkillResult(success=False, message=f"Goal not found: {goal_id}")

        try:
            task_index = int(params.get("task_index", -1))
        except (ValueError, TypeError):
            return SkillResult(success=False, message="Valid task_index is required")

        if task_index < 0 or task_index >= len(goal["tasks"]):
            return SkillResult(
                success=False,
                message=f"Invalid task_index {task_index}. Goal has {len(goal['tasks'])} tasks.",
            )

        task = goal["tasks"][task_index]
        if task["completed"]:
            return SkillResult(success=False, message=f"Task {task_index} is already completed")

        task["completed"] = True
        task["outcome"] = params.get("outcome", "done")
        task["completed_at"] = time.time()
        goal["updated_at"] = time.time()

        completed = sum(1 for t in goal["tasks"] if t["completed"])
        total = len(goal["tasks"])

        self._record_event(goal_id, "task_completed", f"Task {task_index}: {task['description']}")

        return SkillResult(
            success=True,
            message=f"Task {task_index} completed ({completed}/{total} tasks done)",
            data={
                "goal_id": goal_id,
                "task_index": task_index,
                "tasks_completed": completed,
                "tasks_total": total,
                "progress_pct": round(completed / total * 100) if total > 0 else 0,
            },
        )

    def _update_progress(self, params: Dict) -> SkillResult:
        goal_id = params.get("goal_id", "").strip()
        note = params.get("note", "").strip()
        if not goal_id or not note:
            return SkillResult(success=False, message="goal_id and note are required")

        goal = self._get_goal_or_error(goal_id)
        if not goal:
            return SkillResult(success=False, message=f"Goal not found: {goal_id}")

        goal["progress_notes"].append({"note": note, "timestamp": time.time()})
        goal["updated_at"] = time.time()

        self._record_event(goal_id, "progress", note)

        return SkillResult(
            success=True,
            message=f"Progress noted for {goal_id}: {note[:80]}",
            data={"goal_id": goal_id, "notes_count": len(goal["progress_notes"])},
        )

    def _complete_goal(self, params: Dict) -> SkillResult:
        goal_id = params.get("goal_id", "").strip()
        if not goal_id:
            return SkillResult(success=False, message="goal_id is required")

        goal = self._get_goal_or_error(goal_id)
        if not goal:
            return SkillResult(success=False, message=f"Goal not found: {goal_id}")
        if goal["status"] != GoalStatus.ACTIVE.value:
            return SkillResult(success=False, message=f"Goal {goal_id} is not active (status: {goal['status']})")

        goal["status"] = GoalStatus.COMPLETED.value
        goal["completed_at"] = time.time()
        goal["summary"] = params.get("summary", "Goal completed")
        goal["updated_at"] = time.time()

        self._record_event(goal_id, "completed", goal["summary"])

        duration = goal["completed_at"] - goal["created_at"]

        return SkillResult(
            success=True,
            message=f"Goal completed: [{goal_id}] {goal['title']} (took {duration:.0f}s)",
            data={
                "goal_id": goal_id,
                "title": goal["title"],
                "duration_seconds": round(duration),
            },
        )

    def _fail_goal(self, params: Dict) -> SkillResult:
        goal_id = params.get("goal_id", "").strip()
        if not goal_id:
            return SkillResult(success=False, message="goal_id is required")

        goal = self._get_goal_or_error(goal_id)
        if not goal:
            return SkillResult(success=False, message=f"Goal not found: {goal_id}")
        if goal["status"] != GoalStatus.ACTIVE.value:
            return SkillResult(success=False, message=f"Goal {goal_id} is not active")

        reason = params.get("reason", "No reason provided")
        goal["status"] = GoalStatus.FAILED.value
        goal["completed_at"] = time.time()
        goal["summary"] = f"FAILED: {reason}"
        goal["updated_at"] = time.time()

        self._record_event(goal_id, "failed", reason)

        return SkillResult(
            success=True,
            message=f"Goal failed: [{goal_id}] {goal['title']} - {reason}",
            data={"goal_id": goal_id, "reason": reason},
        )

    def _reprioritize(self, params: Dict) -> SkillResult:
        goal_id = params.get("goal_id", "").strip()
        new_priority_str = params.get("new_priority", "").strip().lower()
        if not goal_id or not new_priority_str:
            return SkillResult(success=False, message="goal_id and new_priority are required")

        goal = self._get_goal_or_error(goal_id)
        if not goal:
            return SkillResult(success=False, message=f"Goal not found: {goal_id}")

        try:
            new_priority = GoalPriority(new_priority_str)
        except ValueError:
            return SkillResult(
                success=False,
                message=f"Invalid priority '{new_priority_str}'. Use: critical, high, medium, low",
            )

        old_priority = goal["priority"]
        goal["priority"] = new_priority.value
        goal["updated_at"] = time.time()

        reason = params.get("reason", "")
        self._record_event(
            goal_id, "reprioritized", f"{old_priority} -> {new_priority.value}: {reason}"
        )

        return SkillResult(
            success=True,
            message=f"Goal {goal_id} reprioritized: {old_priority} -> {new_priority.value}",
            data={"goal_id": goal_id, "old_priority": old_priority, "new_priority": new_priority.value},
        )

    # === Query / analysis ===

    def _list_goals(self, params: Dict) -> SkillResult:
        status_filter = params.get("status", "").strip().lower()

        goals = list(self._goals.values())
        if status_filter:
            goals = [g for g in goals if g["status"] == status_filter]

        priority_order = {
            GoalPriority.CRITICAL.value: 0,
            GoalPriority.HIGH.value: 1,
            GoalPriority.MEDIUM.value: 2,
            GoalPriority.LOW.value: 3,
        }
        goals.sort(key=lambda g: (priority_order.get(g["priority"], 99), -g["created_at"]))

        summary = []
        for g in goals:
            completed_tasks = sum(1 for t in g["tasks"] if t["completed"])
            total_tasks = len(g["tasks"])
            progress = f"{completed_tasks}/{total_tasks}" if total_tasks > 0 else "no tasks"

            overdue = ""
            if g["deadline"] and g["status"] == GoalStatus.ACTIVE.value and time.time() > g["deadline"]:
                overdue = " [OVERDUE]"

            summary.append(
                f"[{g['id']}] ({g['priority']}) {g['title']} - {g['status']} ({progress}){overdue}"
            )

        return SkillResult(
            success=True,
            message=f"Found {len(goals)} goals" + (f" (status={status_filter})" if status_filter else ""),
            data={
                "goals": summary,
                "count": len(goals),
            },
        )

    def _get_goal(self, params: Dict) -> SkillResult:
        goal_id = params.get("goal_id", "").strip()
        if not goal_id:
            return SkillResult(success=False, message="goal_id is required")

        goal = self._get_goal_or_error(goal_id)
        if not goal:
            return SkillResult(success=False, message=f"Goal not found: {goal_id}")

        tasks_info = []
        for i, t in enumerate(goal["tasks"]):
            status = "done" if t["completed"] else "pending"
            tasks_info.append(f"  [{i}] [{status}] {t['description']}")

        notes_info = []
        for n in goal["progress_notes"][-5:]:
            notes_info.append(f"  - {n['note']}")

        detail = {
            **goal,
            "tasks_display": tasks_info,
            "recent_notes": notes_info,
        }

        return SkillResult(
            success=True,
            message=f"Goal {goal_id}: {goal['title']}",
            data=detail,
        )

    def _get_plan(self, params: Dict) -> SkillResult:
        active_goals = [g for g in self._goals.values() if g["status"] == GoalStatus.ACTIVE.value]

        if not active_goals:
            return SkillResult(
                success=True,
                message="No active goals. Use goals:set_goal to create one.",
                data={"plan": [], "active_count": 0},
            )

        priority_order = {
            GoalPriority.CRITICAL.value: 0,
            GoalPriority.HIGH.value: 1,
            GoalPriority.MEDIUM.value: 2,
            GoalPriority.LOW.value: 3,
        }
        active_goals.sort(key=lambda g: (priority_order.get(g["priority"], 99), g["created_at"]))

        plan = []
        for g in active_goals:
            pending_tasks = [
                t["description"] for t in g["tasks"] if not t["completed"]
            ]
            overdue = g["deadline"] and time.time() > g["deadline"]

            entry = {
                "goal_id": g["id"],
                "title": g["title"],
                "priority": g["priority"],
                "overdue": overdue,
                "next_tasks": pending_tasks[:3],
                "tasks_remaining": len(pending_tasks),
            }
            plan.append(entry)

        top = plan[0]
        if top["next_tasks"]:
            suggestion = f"Work on [{top['goal_id']}] {top['title']}: next task is '{top['next_tasks'][0]}'"
        else:
            suggestion = f"Work on [{top['goal_id']}] {top['title']}: add tasks to break it down"

        return SkillResult(
            success=True,
            message=f"Plan: {suggestion}",
            data={"plan": plan, "suggestion": suggestion, "active_count": len(active_goals)},
        )

    def _evaluate(self, params: Dict) -> SkillResult:
        all_goals = list(self._goals.values())
        if not all_goals:
            return SkillResult(
                success=True,
                message="No goals to evaluate.",
                data={"total": 0},
            )

        total = len(all_goals)
        active = sum(1 for g in all_goals if g["status"] == GoalStatus.ACTIVE.value)
        completed = sum(1 for g in all_goals if g["status"] == GoalStatus.COMPLETED.value)
        failed = sum(1 for g in all_goals if g["status"] == GoalStatus.FAILED.value)

        completion_rate = round(completed / total * 100) if total > 0 else 0

        overdue = []
        now = time.time()
        for g in all_goals:
            if g["status"] == GoalStatus.ACTIVE.value and g["deadline"] and now > g["deadline"]:
                overdue.append(f"[{g['id']}] {g['title']}")

        total_tasks = 0
        completed_tasks = 0
        for g in all_goals:
            total_tasks += len(g["tasks"])
            completed_tasks += sum(1 for t in g["tasks"] if t["completed"])

        durations = []
        for g in all_goals:
            if g["completed_at"]:
                durations.append(g["completed_at"] - g["created_at"])

        avg_duration = round(sum(durations) / len(durations)) if durations else 0

        return SkillResult(
            success=True,
            message=f"Evaluation: {completed}/{total} goals completed ({completion_rate}%), {active} active, {failed} failed",
            data={
                "total_goals": total,
                "active": active,
                "completed": completed,
                "failed": failed,
                "completion_rate_pct": completion_rate,
                "overdue": overdue,
                "total_tasks": total_tasks,
                "completed_tasks": completed_tasks,
                "avg_duration_seconds": avg_duration,
                "history_events": len(self._history),
            },
        )

    # === Import/Export for persistence ===

    def _export_goals(self, params: Dict) -> SkillResult:
        export_data = {
            "goals": self._goals,
            "next_id": self._next_id,
            "history": self._history[-50:],
            "exported_at": time.time(),
        }

        return SkillResult(
            success=True,
            message=f"Exported {len(self._goals)} goals",
            data={"goals_json": json.dumps(export_data), "count": len(self._goals)},
        )

    def _import_goals(self, params: Dict) -> SkillResult:
        goals_json = params.get("goals_json", "").strip()
        if not goals_json:
            return SkillResult(success=False, message="goals_json is required")

        try:
            data = json.loads(goals_json)
        except json.JSONDecodeError as e:
            return SkillResult(success=False, message=f"Invalid JSON: {e}")

        imported_goals = data.get("goals", {})
        if not isinstance(imported_goals, dict):
            return SkillResult(success=False, message="Invalid goals format: expected dict")

        self._goals.update(imported_goals)

        if "next_id" in data:
            self._next_id = max(self._next_id, data["next_id"])

        if "history" in data and isinstance(data["history"], list):
            self._history.extend(data["history"])

        return SkillResult(
            success=True,
            message=f"Imported {len(imported_goals)} goals",
            data={"imported_count": len(imported_goals), "total_goals": len(self._goals)},
        )
