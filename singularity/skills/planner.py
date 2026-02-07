#!/usr/bin/env python3
"""
Planner Skill - Autonomous goal decomposition and planning.

Provides agents with structured goal management:
- Create high-level goals with success criteria
- Decompose goals into executable tasks with dependencies
- Track progress and completion across sessions
- Auto-prioritize based on impact, urgency, and dependencies
- Replan when tasks fail or conditions change
- Persist plans to disk for cross-session continuity

This is the foundation of the Goal Setting pillar - enabling
agents to reason about what to do, not just how to do it.
"""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from enum import Enum
from .base import Skill, SkillResult, SkillManifest, SkillAction


PLANS_FILE = Path(__file__).parent.parent / "data" / "plans.json"


class Priority(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class TaskStatus(str, Enum):
    PENDING = "pending"
    BLOCKED = "blocked"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class GoalStatus(str, Enum):
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"
    ARCHIVED = "archived"


class PlannerSkill(Skill):
    """
    Autonomous goal decomposition and planning skill.

    Enables agents to:
    - Define high-level goals with success criteria
    - Break goals into ordered tasks with dependencies
    - Track progress toward goal completion
    - Reprioritize based on changing conditions
    - Get next recommended action based on priority + dependencies
    - Persist everything across sessions
    """

    def __init__(self, credentials: Dict[str, str] = None):
        super().__init__(credentials)
        self._ensure_data()

    def _ensure_data(self):
        PLANS_FILE.parent.mkdir(parents=True, exist_ok=True)
        if not PLANS_FILE.exists():
            self._save({"goals": [], "metadata": {"created_at": datetime.now().isoformat()}})

    def _load(self) -> Dict:
        try:
            with open(PLANS_FILE, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {"goals": [], "metadata": {}}

    def _save(self, data: Dict):
        with open(PLANS_FILE, "w") as f:
            json.dump(data, f, indent=2, default=str)

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="planner",
            name="Goal Planner",
            version="1.0.0",
            category="planning",
            description="Autonomous goal decomposition, task planning, and progress tracking",
            required_credentials=[],
            install_cost=0,
            actions=[
                SkillAction(
                    name="create_goal",
                    description="Create a new high-level goal with success criteria",
                    parameters={
                        "title": "Short goal title",
                        "description": "Detailed description of what to achieve",
                        "success_criteria": "How to know the goal is complete (list or string)",
                        "priority": "critical, high, medium, or low",
                        "pillar": "Which pillar: self_improvement, revenue, replication, goal_setting",
                        "deadline": "Optional ISO deadline",
                    },
                    estimated_cost=0,
                    success_probability=1.0,
                ),
                SkillAction(
                    name="add_task",
                    description="Add an executable task to an existing goal",
                    parameters={
                        "goal_id": "ID of the parent goal",
                        "title": "Short task title",
                        "description": "What needs to be done",
                        "depends_on": "List of task IDs this task depends on (optional)",
                        "skill_hint": "Skill ID to use for execution (optional)",
                        "effort": "Estimated effort: small, medium, large",
                    },
                    estimated_cost=0,
                    success_probability=1.0,
                ),
                SkillAction(
                    name="update_task",
                    description="Update a task's status (pending, in_progress, completed, failed, skipped)",
                    parameters={
                        "goal_id": "ID of the parent goal",
                        "task_id": "ID of the task to update",
                        "status": "New status: pending, in_progress, completed, failed, skipped",
                        "note": "Optional note about the update",
                    },
                    estimated_cost=0,
                    success_probability=1.0,
                ),
                SkillAction(
                    name="next_task",
                    description="Get the highest-priority next task to work on across all goals",
                    parameters={
                        "goal_id": "Optional: limit to a specific goal",
                    },
                    estimated_cost=0,
                    success_probability=1.0,
                ),
                SkillAction(
                    name="get_goal",
                    description="Get full details of a goal including all tasks and progress",
                    parameters={
                        "goal_id": "ID of the goal",
                    },
                    estimated_cost=0,
                    success_probability=1.0,
                ),
                SkillAction(
                    name="list_goals",
                    description="List all goals with progress summaries",
                    parameters={
                        "status": "Filter by status: active, completed, failed, paused, archived, all",
                        "pillar": "Filter by pillar (optional)",
                    },
                    estimated_cost=0,
                    success_probability=1.0,
                ),
                SkillAction(
                    name="update_goal",
                    description="Update a goal's status or details",
                    parameters={
                        "goal_id": "ID of the goal",
                        "status": "New status: active, completed, failed, paused, archived",
                        "note": "Optional note",
                    },
                    estimated_cost=0,
                    success_probability=1.0,
                ),
                SkillAction(
                    name="replan",
                    description="Re-evaluate and reprioritize tasks for a goal based on what failed",
                    parameters={
                        "goal_id": "ID of the goal to replan",
                        "reason": "Why replanning is needed",
                    },
                    estimated_cost=0,
                    success_probability=1.0,
                ),
                SkillAction(
                    name="progress",
                    description="Get an overall progress report across all active goals",
                    parameters={},
                    estimated_cost=0,
                    success_probability=1.0,
                ),
                SkillAction(
                    name="remove_task",
                    description="Remove a task from a goal",
                    parameters={
                        "goal_id": "ID of the parent goal",
                        "task_id": "ID of the task to remove",
                    },
                    estimated_cost=0,
                    success_probability=1.0,
                ),
            ],
        )

    def check_credentials(self) -> bool:
        return True

    async def execute(self, action: str, params: Dict) -> SkillResult:
        try:
            if action == "create_goal":
                return self._create_goal(params)
            elif action == "add_task":
                return self._add_task(params)
            elif action == "update_task":
                return self._update_task(params)
            elif action == "next_task":
                return self._next_task(params)
            elif action == "get_goal":
                return self._get_goal(params)
            elif action == "list_goals":
                return self._list_goals(params)
            elif action == "update_goal":
                return self._update_goal(params)
            elif action == "replan":
                return self._replan(params)
            elif action == "progress":
                return self._progress(params)
            elif action == "remove_task":
                return self._remove_task(params)
            else:
                return SkillResult(success=False, message=f"Unknown action: {action}")
        except Exception as e:
            return SkillResult(success=False, message=f"Planner error: {e}")

    def _create_goal(self, params: Dict) -> SkillResult:
        title = params.get("title", "").strip()
        if not title:
            return SkillResult(success=False, message="Goal title is required")

        description = params.get("description", "")
        success_criteria = params.get("success_criteria", "")
        priority = params.get("priority", "medium")
        pillar = params.get("pillar", "")
        deadline = params.get("deadline", "")

        # Validate priority
        valid_priorities = [p.value for p in Priority]
        if priority not in valid_priorities:
            priority = "medium"

        # Parse success criteria - accept string or list
        if isinstance(success_criteria, str):
            criteria_list = [c.strip() for c in success_criteria.split(";") if c.strip()] if ";" in success_criteria else [success_criteria] if success_criteria else []
        elif isinstance(success_criteria, list):
            criteria_list = success_criteria
        else:
            criteria_list = [str(success_criteria)] if success_criteria else []

        goal_id = str(uuid.uuid4())[:8]
        goal = {
            "id": goal_id,
            "title": title,
            "description": description,
            "success_criteria": criteria_list,
            "priority": priority,
            "pillar": pillar,
            "status": GoalStatus.ACTIVE.value,
            "deadline": deadline,
            "tasks": [],
            "notes": [],
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
        }

        data = self._load()
        data["goals"].append(goal)
        self._save(data)

        return SkillResult(
            success=True,
            message=f"Goal created: {title} [{goal_id}]",
            data={"goal_id": goal_id, "title": title, "priority": priority, "pillar": pillar},
        )

    def _add_task(self, params: Dict) -> SkillResult:
        goal_id = params.get("goal_id", "").strip()
        title = params.get("title", "").strip()
        if not goal_id:
            return SkillResult(success=False, message="goal_id is required")
        if not title:
            return SkillResult(success=False, message="Task title is required")

        data = self._load()
        goal = self._find_goal(data, goal_id)
        if not goal:
            return SkillResult(success=False, message=f"Goal not found: {goal_id}")

        depends_on = params.get("depends_on", [])
        if isinstance(depends_on, str):
            depends_on = [d.strip() for d in depends_on.split(",") if d.strip()]

        # Validate dependencies exist
        existing_task_ids = {t["id"] for t in goal["tasks"]}
        invalid_deps = [d for d in depends_on if d not in existing_task_ids]
        if invalid_deps:
            return SkillResult(success=False, message=f"Invalid dependency task IDs: {invalid_deps}")

        task_id = str(uuid.uuid4())[:8]
        task = {
            "id": task_id,
            "title": title,
            "description": params.get("description", ""),
            "depends_on": depends_on,
            "skill_hint": params.get("skill_hint", ""),
            "effort": params.get("effort", "medium"),
            "status": TaskStatus.PENDING.value,
            "notes": [],
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
        }

        # Check if task is blocked by incomplete dependencies
        if depends_on:
            incomplete = [
                d for d in depends_on
                if any(t["id"] == d and t["status"] not in (TaskStatus.COMPLETED.value, TaskStatus.SKIPPED.value) for t in goal["tasks"])
            ]
            if incomplete:
                task["status"] = TaskStatus.BLOCKED.value

        goal["tasks"].append(task)
        goal["updated_at"] = datetime.now().isoformat()
        self._save(data)

        return SkillResult(
            success=True,
            message=f"Task added: {title} [{task_id}] to goal [{goal_id}]",
            data={"task_id": task_id, "goal_id": goal_id, "status": task["status"]},
        )

    def _update_task(self, params: Dict) -> SkillResult:
        goal_id = params.get("goal_id", "").strip()
        task_id = params.get("task_id", "").strip()
        new_status = params.get("status", "").strip()
        note = params.get("note", "")

        if not goal_id or not task_id or not new_status:
            return SkillResult(success=False, message="goal_id, task_id, and status are required")

        valid_statuses = [s.value for s in TaskStatus]
        if new_status not in valid_statuses:
            return SkillResult(success=False, message=f"Invalid status. Use: {valid_statuses}")

        data = self._load()
        goal = self._find_goal(data, goal_id)
        if not goal:
            return SkillResult(success=False, message=f"Goal not found: {goal_id}")

        task = self._find_task(goal, task_id)
        if not task:
            return SkillResult(success=False, message=f"Task not found: {task_id}")

        old_status = task["status"]
        task["status"] = new_status
        task["updated_at"] = datetime.now().isoformat()
        if note:
            task["notes"].append({"text": note, "timestamp": datetime.now().isoformat()})

        # When a task completes, unblock dependent tasks
        if new_status in (TaskStatus.COMPLETED.value, TaskStatus.SKIPPED.value):
            self._unblock_dependents(goal, task_id)

        # Auto-complete goal if all tasks are done
        if new_status == TaskStatus.COMPLETED.value:
            self._check_goal_completion(goal)

        goal["updated_at"] = datetime.now().isoformat()
        self._save(data)

        return SkillResult(
            success=True,
            message=f"Task [{task_id}] updated: {old_status} -> {new_status}",
            data={
                "task_id": task_id,
                "goal_id": goal_id,
                "old_status": old_status,
                "new_status": new_status,
                "goal_progress": self._calc_progress(goal),
            },
        )

    def _next_task(self, params: Dict) -> SkillResult:
        data = self._load()
        goal_id = params.get("goal_id", "")

        candidates = []
        priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        effort_order = {"small": 0, "medium": 1, "large": 2}

        for goal in data["goals"]:
            if goal["status"] != GoalStatus.ACTIVE.value:
                continue
            if goal_id and goal["id"] != goal_id:
                continue

            for task in goal["tasks"]:
                if task["status"] != TaskStatus.PENDING.value:
                    continue
                candidates.append({
                    "goal_id": goal["id"],
                    "goal_title": goal["title"],
                    "goal_priority": goal["priority"],
                    "goal_pillar": goal.get("pillar", ""),
                    "task_id": task["id"],
                    "task_title": task["title"],
                    "task_description": task.get("description", ""),
                    "skill_hint": task.get("skill_hint", ""),
                    "effort": task.get("effort", "medium"),
                    "depends_on": task.get("depends_on", []),
                    "sort_key": (
                        priority_order.get(goal["priority"], 2),
                        effort_order.get(task.get("effort", "medium"), 1),
                    ),
                })

        if not candidates:
            return SkillResult(
                success=True,
                message="No pending tasks found. All goals may be complete or paused.",
                data={"task": None},
            )

        # Sort by goal priority then effort (prefer small tasks)
        candidates.sort(key=lambda c: c["sort_key"])
        best = candidates[0]
        del best["sort_key"]

        return SkillResult(
            success=True,
            message=f"Next task: [{best['task_id']}] {best['task_title']} (goal: {best['goal_title']})",
            data={"task": best, "total_candidates": len(candidates)},
        )

    def _get_goal(self, params: Dict) -> SkillResult:
        goal_id = params.get("goal_id", "").strip()
        if not goal_id:
            return SkillResult(success=False, message="goal_id is required")

        data = self._load()
        goal = self._find_goal(data, goal_id)
        if not goal:
            return SkillResult(success=False, message=f"Goal not found: {goal_id}")

        progress = self._calc_progress(goal)
        return SkillResult(
            success=True,
            message=f"Goal: {goal['title']} ({progress['percent_complete']}% complete)",
            data={"goal": goal, "progress": progress},
        )

    def _list_goals(self, params: Dict) -> SkillResult:
        data = self._load()
        status_filter = params.get("status", "all")
        pillar_filter = params.get("pillar", "")

        goals = data["goals"]
        if status_filter != "all":
            goals = [g for g in goals if g["status"] == status_filter]
        if pillar_filter:
            goals = [g for g in goals if g.get("pillar") == pillar_filter]

        summaries = []
        for g in goals:
            progress = self._calc_progress(g)
            summaries.append({
                "id": g["id"],
                "title": g["title"],
                "status": g["status"],
                "priority": g["priority"],
                "pillar": g.get("pillar", ""),
                "progress": progress,
                "task_count": len(g["tasks"]),
                "created_at": g["created_at"],
            })

        # Sort by priority
        priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        summaries.sort(key=lambda s: priority_order.get(s["priority"], 2))

        return SkillResult(
            success=True,
            message=f"Found {len(summaries)} goals",
            data={"goals": summaries},
        )

    def _update_goal(self, params: Dict) -> SkillResult:
        goal_id = params.get("goal_id", "").strip()
        new_status = params.get("status", "").strip()
        note = params.get("note", "")

        if not goal_id or not new_status:
            return SkillResult(success=False, message="goal_id and status are required")

        valid_statuses = [s.value for s in GoalStatus]
        if new_status not in valid_statuses:
            return SkillResult(success=False, message=f"Invalid status. Use: {valid_statuses}")

        data = self._load()
        goal = self._find_goal(data, goal_id)
        if not goal:
            return SkillResult(success=False, message=f"Goal not found: {goal_id}")

        old_status = goal["status"]
        goal["status"] = new_status
        goal["updated_at"] = datetime.now().isoformat()
        if note:
            goal["notes"].append({"text": note, "timestamp": datetime.now().isoformat()})

        self._save(data)

        return SkillResult(
            success=True,
            message=f"Goal [{goal_id}] updated: {old_status} -> {new_status}",
            data={"goal_id": goal_id, "old_status": old_status, "new_status": new_status},
        )

    def _replan(self, params: Dict) -> SkillResult:
        goal_id = params.get("goal_id", "").strip()
        reason = params.get("reason", "")

        if not goal_id:
            return SkillResult(success=False, message="goal_id is required")

        data = self._load()
        goal = self._find_goal(data, goal_id)
        if not goal:
            return SkillResult(success=False, message=f"Goal not found: {goal_id}")

        # Analyze current state
        failed_tasks = [t for t in goal["tasks"] if t["status"] == TaskStatus.FAILED.value]
        blocked_tasks = [t for t in goal["tasks"] if t["status"] == TaskStatus.BLOCKED.value]
        pending_tasks = [t for t in goal["tasks"] if t["status"] == TaskStatus.PENDING.value]
        completed_tasks = [t for t in goal["tasks"] if t["status"] == TaskStatus.COMPLETED.value]

        changes = []

        # Reset failed tasks to pending so they can be retried
        for task in failed_tasks:
            task["status"] = TaskStatus.PENDING.value
            task["updated_at"] = datetime.now().isoformat()
            task["notes"].append({
                "text": f"Reset to pending during replan. Reason: {reason}",
                "timestamp": datetime.now().isoformat(),
            })
            changes.append(f"Reset failed task [{task['id']}] {task['title']} to pending")

        # Unblock tasks whose dependencies are now met
        for task in blocked_tasks:
            deps = task.get("depends_on", [])
            all_met = all(
                any(t["id"] == d and t["status"] in (TaskStatus.COMPLETED.value, TaskStatus.SKIPPED.value) for t in goal["tasks"])
                for d in deps
            )
            if all_met:
                task["status"] = TaskStatus.PENDING.value
                task["updated_at"] = datetime.now().isoformat()
                changes.append(f"Unblocked task [{task['id']}] {task['title']}")

        # Add replan note to goal
        goal["notes"].append({
            "text": f"Replan triggered: {reason}. Changes: {len(changes)}",
            "timestamp": datetime.now().isoformat(),
        })
        goal["updated_at"] = datetime.now().isoformat()

        # Reactivate paused goals
        if goal["status"] == GoalStatus.PAUSED.value:
            goal["status"] = GoalStatus.ACTIVE.value
            changes.append("Reactivated paused goal")

        self._save(data)

        return SkillResult(
            success=True,
            message=f"Replanned goal [{goal_id}]: {len(changes)} changes",
            data={
                "goal_id": goal_id,
                "changes": changes,
                "summary": {
                    "failed_reset": len(failed_tasks),
                    "unblocked": len([c for c in changes if "Unblocked" in c]),
                    "pending": len(pending_tasks) + len(failed_tasks),
                    "completed": len(completed_tasks),
                },
            },
        )

    def _progress(self, params: Dict) -> SkillResult:
        data = self._load()

        active_goals = [g for g in data["goals"] if g["status"] == GoalStatus.ACTIVE.value]
        completed_goals = [g for g in data["goals"] if g["status"] == GoalStatus.COMPLETED.value]

        pillar_stats = {}
        total_tasks = 0
        total_completed = 0

        for goal in data["goals"]:
            pillar = goal.get("pillar", "other")
            if pillar not in pillar_stats:
                pillar_stats[pillar] = {"goals": 0, "completed_goals": 0, "tasks": 0, "completed_tasks": 0}
            pillar_stats[pillar]["goals"] += 1
            if goal["status"] == GoalStatus.COMPLETED.value:
                pillar_stats[pillar]["completed_goals"] += 1
            for task in goal["tasks"]:
                pillar_stats[pillar]["tasks"] += 1
                total_tasks += 1
                if task["status"] == TaskStatus.COMPLETED.value:
                    pillar_stats[pillar]["completed_tasks"] += 1
                    total_completed += 1

        overall_percent = round(total_completed / total_tasks * 100) if total_tasks > 0 else 0

        # Per-goal progress for active goals
        active_details = []
        for goal in active_goals:
            prog = self._calc_progress(goal)
            active_details.append({
                "id": goal["id"],
                "title": goal["title"],
                "priority": goal["priority"],
                "pillar": goal.get("pillar", ""),
                "progress": prog,
            })

        return SkillResult(
            success=True,
            message=f"Overall: {overall_percent}% ({total_completed}/{total_tasks} tasks). {len(active_goals)} active, {len(completed_goals)} completed goals.",
            data={
                "overall": {
                    "total_goals": len(data["goals"]),
                    "active_goals": len(active_goals),
                    "completed_goals": len(completed_goals),
                    "total_tasks": total_tasks,
                    "completed_tasks": total_completed,
                    "percent_complete": overall_percent,
                },
                "by_pillar": pillar_stats,
                "active_goals": active_details,
            },
        )

    def _remove_task(self, params: Dict) -> SkillResult:
        goal_id = params.get("goal_id", "").strip()
        task_id = params.get("task_id", "").strip()

        if not goal_id or not task_id:
            return SkillResult(success=False, message="goal_id and task_id are required")

        data = self._load()
        goal = self._find_goal(data, goal_id)
        if not goal:
            return SkillResult(success=False, message=f"Goal not found: {goal_id}")

        task = self._find_task(goal, task_id)
        if not task:
            return SkillResult(success=False, message=f"Task not found: {task_id}")

        # Remove from task list
        goal["tasks"] = [t for t in goal["tasks"] if t["id"] != task_id]

        # Remove from dependencies of other tasks
        for t in goal["tasks"]:
            if task_id in t.get("depends_on", []):
                t["depends_on"].remove(task_id)
                # Unblock if no more deps
                if t["status"] == TaskStatus.BLOCKED.value and not t["depends_on"]:
                    t["status"] = TaskStatus.PENDING.value

        goal["updated_at"] = datetime.now().isoformat()
        self._save(data)

        return SkillResult(
            success=True,
            message=f"Task [{task_id}] removed from goal [{goal_id}]",
            data={"task_id": task_id, "goal_id": goal_id},
        )

    # --- Helper methods ---

    def _find_goal(self, data: Dict, goal_id: str) -> Optional[Dict]:
        for g in data["goals"]:
            if g["id"] == goal_id:
                return g
        return None

    def _find_task(self, goal: Dict, task_id: str) -> Optional[Dict]:
        for t in goal["tasks"]:
            if t["id"] == task_id:
                return t
        return None

    def _calc_progress(self, goal: Dict) -> Dict:
        tasks = goal["tasks"]
        total = len(tasks)
        if total == 0:
            return {"total": 0, "completed": 0, "failed": 0, "pending": 0, "blocked": 0, "in_progress": 0, "percent_complete": 0}

        by_status = {}
        for s in TaskStatus:
            by_status[s.value] = len([t for t in tasks if t["status"] == s.value])

        completed = by_status.get(TaskStatus.COMPLETED.value, 0)
        skipped = by_status.get(TaskStatus.SKIPPED.value, 0)
        pct = round((completed + skipped) / total * 100)

        return {
            "total": total,
            "completed": completed,
            "failed": by_status.get(TaskStatus.FAILED.value, 0),
            "pending": by_status.get(TaskStatus.PENDING.value, 0),
            "blocked": by_status.get(TaskStatus.BLOCKED.value, 0),
            "in_progress": by_status.get(TaskStatus.IN_PROGRESS.value, 0),
            "skipped": skipped,
            "percent_complete": pct,
        }

    def _unblock_dependents(self, goal: Dict, completed_task_id: str):
        """When a task completes, check if dependent tasks can be unblocked."""
        for task in goal["tasks"]:
            if task["status"] != TaskStatus.BLOCKED.value:
                continue
            deps = task.get("depends_on", [])
            if completed_task_id not in deps:
                continue

            # Check if all deps are now complete
            all_met = all(
                any(t["id"] == d and t["status"] in (TaskStatus.COMPLETED.value, TaskStatus.SKIPPED.value) for t in goal["tasks"])
                for d in deps
            )
            if all_met:
                task["status"] = TaskStatus.PENDING.value
                task["updated_at"] = datetime.now().isoformat()

    def _check_goal_completion(self, goal: Dict):
        """Auto-complete a goal if all tasks are done."""
        tasks = goal["tasks"]
        if not tasks:
            return

        all_done = all(
            t["status"] in (TaskStatus.COMPLETED.value, TaskStatus.SKIPPED.value)
            for t in tasks
        )
        if all_done:
            goal["status"] = GoalStatus.COMPLETED.value
            goal["notes"].append({
                "text": "Goal auto-completed: all tasks finished",
                "timestamp": datetime.now().isoformat(),
            })
