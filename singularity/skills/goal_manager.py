#!/usr/bin/env python3
"""
GoalManager Skill - Autonomous goal tracking, prioritization, and execution planning.

This skill fills a critical gap in the Goal Setting pillar. While StrategySkill
assesses pillar maturity and recommends actions, GoalManager provides the concrete
execution layer: defining goals with deadlines and dependencies, breaking them into
milestones, tracking progress, and deciding what to work on next.

The loop:
  1. Agent assesses strategy (StrategySkill) → identifies priorities
  2. Agent creates goals with milestones (GoalManager)
  3. Agent queries "what next?" → gets highest-priority actionable goal
  4. Agent works on goal, updates progress
  5. Agent completes milestones → goal completion triggers next goals
  6. Repeat

Part of the Goal Setting pillar: structured planning with execution tracking.
"""

import json
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from .base import Skill, SkillResult, SkillManifest, SkillAction


GOALS_FILE = Path(__file__).parent.parent / "data" / "goals.json"
MAX_GOALS = 200
MAX_MILESTONES_PER_GOAL = 20


# Goal pillars for categorization
PILLARS = ["self_improvement", "revenue", "replication", "goal_setting", "other"]

# Priority levels
PRIORITIES = {"critical": 4, "high": 3, "medium": 2, "low": 1}

# Goal statuses
STATUSES = ["active", "blocked", "completed", "abandoned"]


class GoalManagerSkill(Skill):
    """
    Autonomous goal management with priority-based execution planning.

    Enables agents to:
    - Create goals with priority, pillar, deadline, and dependencies
    - Break goals into ordered milestones
    - Query "what should I work on next?" with smart prioritization
    - Track progress and auto-complete goals when all milestones done
    - Maintain a persistent goal history across sessions
    - Analyze goal completion patterns for self-improvement
    """

    def __init__(self, credentials: Dict[str, str] = None):
        super().__init__(credentials)
        self._ensure_data()

    def _ensure_data(self):
        GOALS_FILE.parent.mkdir(parents=True, exist_ok=True)
        if not GOALS_FILE.exists():
            self._save(self._default_state())

    def _default_state(self) -> Dict:
        return {
            "goals": [],
            "completed_goals": [],
            "session_log": [],
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
        }

    def _load(self) -> Dict:
        try:
            with open(GOALS_FILE, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return self._default_state()

    def _save(self, data: Dict):
        data["last_updated"] = datetime.now().isoformat()
        # Cap completed goals history
        if len(data.get("completed_goals", [])) > MAX_GOALS:
            data["completed_goals"] = data["completed_goals"][-MAX_GOALS:]
        if len(data.get("session_log", [])) > 500:
            data["session_log"] = data["session_log"][-500:]
        with open(GOALS_FILE, "w") as f:
            json.dump(data, f, indent=2, default=str)

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="goals",
            name="Goal Manager",
            version="1.0.0",
            category="meta",
            description="Autonomous goal tracking, prioritization, and execution planning",
            actions=[
                SkillAction(
                    name="create",
                    description="Create a new goal with priority, pillar, and optional deadline",
                    parameters={
                        "title": {"type": "string", "required": True, "description": "Short goal title"},
                        "description": {"type": "string", "required": False, "description": "Detailed description"},
                        "pillar": {"type": "string", "required": True, "description": "Pillar: self_improvement, revenue, replication, goal_setting, other"},
                        "priority": {"type": "string", "required": False, "description": "Priority: critical, high, medium, low (default: medium)"},
                        "deadline_hours": {"type": "number", "required": False, "description": "Hours until deadline (optional)"},
                        "depends_on": {"type": "array", "required": False, "description": "List of goal IDs this depends on"},
                        "milestones": {"type": "array", "required": False, "description": "List of milestone titles (strings)"},
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="next",
                    description="Get the highest-priority actionable goal to work on now",
                    parameters={
                        "pillar": {"type": "string", "required": False, "description": "Filter by pillar (optional)"},
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="list",
                    description="List all active goals, sorted by priority",
                    parameters={
                        "status": {"type": "string", "required": False, "description": "Filter by status (default: active)"},
                        "pillar": {"type": "string", "required": False, "description": "Filter by pillar"},
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="progress",
                    description="Update progress on a goal or complete a milestone",
                    parameters={
                        "goal_id": {"type": "string", "required": True, "description": "Goal ID"},
                        "milestone_index": {"type": "integer", "required": False, "description": "Index of milestone to complete (0-based)"},
                        "note": {"type": "string", "required": False, "description": "Progress note"},
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="complete",
                    description="Mark a goal as completed",
                    parameters={
                        "goal_id": {"type": "string", "required": True, "description": "Goal ID to complete"},
                        "outcome": {"type": "string", "required": False, "description": "Outcome notes"},
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="abandon",
                    description="Abandon a goal that is no longer relevant",
                    parameters={
                        "goal_id": {"type": "string", "required": True, "description": "Goal ID to abandon"},
                        "reason": {"type": "string", "required": False, "description": "Why this goal was abandoned"},
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="add_milestone",
                    description="Add a milestone to an existing goal",
                    parameters={
                        "goal_id": {"type": "string", "required": True, "description": "Goal ID"},
                        "title": {"type": "string", "required": True, "description": "Milestone title"},
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="analyze",
                    description="Analyze goal completion patterns and productivity",
                    parameters={},
                    estimated_cost=0,
                ),
                SkillAction(
                    name="focus",
                    description="Get a focused work plan: top goals per pillar with next actions",
                    parameters={},
                    estimated_cost=0,
                ),
                SkillAction(
                    name="cleanup",
                    description="Remove stale goals past deadline with no progress",
                    parameters={
                        "stale_hours": {"type": "number", "required": False, "description": "Hours after deadline to consider stale (default: 48)"},
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
            "create": self._create,
            "next": self._next,
            "list": self._list,
            "progress": self._progress,
            "complete": self._complete,
            "abandon": self._abandon,
            "add_milestone": self._add_milestone,
            "analyze": self._analyze,
            "focus": self._focus,
            "cleanup": self._cleanup,
        }
        handler = handlers.get(action)
        if handler:
            return handler(params)
        return SkillResult(success=False, message=f"Unknown action: {action}")

    def _create(self, params: Dict) -> SkillResult:
        """Create a new goal."""
        title = params.get("title", "").strip()
        if not title:
            return SkillResult(success=False, message="title is required")

        description = params.get("description", "")
        pillar = params.get("pillar", "other").strip().lower()
        priority = params.get("priority", "medium").strip().lower()
        deadline_hours = params.get("deadline_hours")
        depends_on = params.get("depends_on", [])
        milestone_titles = params.get("milestones", [])

        if pillar not in PILLARS:
            return SkillResult(
                success=False,
                message=f"Invalid pillar: {pillar}. Must be one of: {PILLARS}",
            )
        if priority not in PRIORITIES:
            return SkillResult(
                success=False,
                message=f"Invalid priority: {priority}. Must be one of: {list(PRIORITIES.keys())}",
            )

        goal_id = f"goal_{uuid.uuid4().hex[:8]}"
        now = datetime.now()

        # Build milestones
        milestones = []
        if isinstance(milestone_titles, list):
            for i, mt in enumerate(milestone_titles[:MAX_MILESTONES_PER_GOAL]):
                mt_str = str(mt).strip() if mt else ""
                if mt_str:
                    milestones.append({
                        "index": i,
                        "title": mt_str,
                        "completed": False,
                        "completed_at": None,
                    })

        goal = {
            "id": goal_id,
            "title": title,
            "description": description,
            "pillar": pillar,
            "priority": priority,
            "priority_score": PRIORITIES[priority],
            "status": "active",
            "milestones": milestones,
            "depends_on": depends_on if isinstance(depends_on, list) else [],
            "progress_notes": [],
            "created_at": now.isoformat(),
            "deadline": (now + timedelta(hours=float(deadline_hours))).isoformat() if deadline_hours else None,
            "completed_at": None,
        }

        data = self._load()

        # Validate dependencies exist
        active_ids = {g["id"] for g in data.get("goals", [])}
        invalid_deps = [d for d in goal["depends_on"] if d not in active_ids]
        if invalid_deps:
            return SkillResult(
                success=False,
                message=f"Unknown dependency goal IDs: {invalid_deps}",
            )

        # Check goal count limit
        if len(data.get("goals", [])) >= MAX_GOALS:
            return SkillResult(
                success=False,
                message=f"Maximum {MAX_GOALS} active goals reached. Complete or abandon some first.",
            )

        data.setdefault("goals", []).append(goal)
        self._save(data)

        return SkillResult(
            success=True,
            message=f"Goal created: [{priority.upper()}] {title} ({pillar})",
            data={
                "goal_id": goal_id,
                "title": title,
                "pillar": pillar,
                "priority": priority,
                "milestones": len(milestones),
                "deadline": goal["deadline"],
                "depends_on": goal["depends_on"],
            },
        )

    def _next(self, params: Dict) -> SkillResult:
        """Get the highest-priority actionable goal."""
        pillar_filter = params.get("pillar", "").strip().lower()
        data = self._load()
        goals = data.get("goals", [])

        if not goals:
            return SkillResult(
                success=True,
                message="No active goals. Create some with goals:create.",
                data={"next_goal": None},
            )

        # Filter to active goals
        active = [g for g in goals if g.get("status") == "active"]
        if pillar_filter:
            active = [g for g in active if g.get("pillar") == pillar_filter]

        if not active:
            return SkillResult(
                success=True,
                message=f"No active goals{' for ' + pillar_filter if pillar_filter else ''}.",
                data={"next_goal": None},
            )

        # Filter out blocked goals (dependencies not met)
        completed_ids = {g["id"] for g in data.get("completed_goals", [])}
        completed_ids.update(g["id"] for g in goals if g.get("status") == "completed")

        actionable = []
        for g in active:
            deps = g.get("depends_on", [])
            unmet = [d for d in deps if d not in completed_ids]
            if not unmet:
                actionable.append(g)

        if not actionable:
            blocked_goals = [g["title"] for g in active]
            return SkillResult(
                success=True,
                message=f"All {len(active)} active goals are blocked by dependencies.",
                data={"blocked_goals": blocked_goals},
            )

        # Score goals: priority_score * urgency_multiplier
        now = datetime.now()
        scored = []
        for g in actionable:
            score = g.get("priority_score", 2)

            # Urgency bonus for approaching deadlines
            if g.get("deadline"):
                try:
                    deadline = datetime.fromisoformat(g["deadline"])
                    hours_left = (deadline - now).total_seconds() / 3600
                    if hours_left < 0:
                        score += 5  # Overdue: highest urgency
                    elif hours_left < 24:
                        score += 3  # Due within 24h
                    elif hours_left < 72:
                        score += 1  # Due within 3 days
                except (ValueError, TypeError):
                    pass

            # Bonus for goals with progress (momentum)
            milestones = g.get("milestones", [])
            if milestones:
                completed_ms = sum(1 for m in milestones if m.get("completed"))
                if 0 < completed_ms < len(milestones):
                    score += 1  # In-progress bonus

            scored.append((score, g))

        scored.sort(key=lambda x: x[0], reverse=True)
        best_score, best_goal = scored[0]

        # Compute next milestone
        next_milestone = None
        milestones = best_goal.get("milestones", [])
        for m in milestones:
            if not m.get("completed"):
                next_milestone = m
                break

        return SkillResult(
            success=True,
            message=f"Next: [{best_goal['priority'].upper()}] {best_goal['title']}",
            data={
                "goal_id": best_goal["id"],
                "title": best_goal["title"],
                "description": best_goal.get("description", ""),
                "pillar": best_goal["pillar"],
                "priority": best_goal["priority"],
                "score": best_score,
                "deadline": best_goal.get("deadline"),
                "next_milestone": next_milestone,
                "milestones_done": sum(1 for m in milestones if m.get("completed")),
                "milestones_total": len(milestones),
                "alternatives": len(scored) - 1,
            },
        )

    def _list(self, params: Dict) -> SkillResult:
        """List goals filtered by status and pillar."""
        status_filter = params.get("status", "active").strip().lower()
        pillar_filter = params.get("pillar", "").strip().lower()

        data = self._load()

        if status_filter == "completed":
            goals = data.get("completed_goals", [])
        else:
            goals = [g for g in data.get("goals", []) if g.get("status") == status_filter]

        if pillar_filter:
            goals = [g for g in goals if g.get("pillar") == pillar_filter]

        # Sort by priority score descending
        goals.sort(key=lambda g: g.get("priority_score", 0), reverse=True)

        summary = []
        for g in goals:
            milestones = g.get("milestones", [])
            done = sum(1 for m in milestones if m.get("completed"))
            total = len(milestones)
            progress_str = f"{done}/{total}" if total > 0 else "no milestones"

            summary.append({
                "id": g["id"],
                "title": g["title"],
                "pillar": g["pillar"],
                "priority": g["priority"],
                "progress": progress_str,
                "deadline": g.get("deadline"),
            })

        return SkillResult(
            success=True,
            message=f"{len(goals)} {status_filter} goals" + (f" ({pillar_filter})" if pillar_filter else ""),
            data={"goals": summary, "count": len(goals)},
        )

    def _progress(self, params: Dict) -> SkillResult:
        """Update progress on a goal."""
        goal_id = params.get("goal_id", "").strip()
        milestone_index = params.get("milestone_index")
        note = params.get("note", "")

        if not goal_id:
            return SkillResult(success=False, message="goal_id is required")

        data = self._load()
        goal = self._find_goal(data, goal_id)
        if not goal:
            return SkillResult(success=False, message=f"Goal not found: {goal_id}")

        now = datetime.now()

        # Complete a milestone if specified
        if milestone_index is not None:
            milestones = goal.get("milestones", [])
            idx = int(milestone_index)
            if idx < 0 or idx >= len(milestones):
                return SkillResult(
                    success=False,
                    message=f"Invalid milestone index: {idx}. Goal has {len(milestones)} milestones.",
                )
            if milestones[idx].get("completed"):
                return SkillResult(
                    success=False,
                    message=f"Milestone {idx} already completed: {milestones[idx]['title']}",
                )

            milestones[idx]["completed"] = True
            milestones[idx]["completed_at"] = now.isoformat()

            done = sum(1 for m in milestones if m.get("completed"))
            total = len(milestones)

            # Auto-complete goal if all milestones done
            if done == total:
                return self._complete_goal(data, goal, f"All {total} milestones completed")

            # Add progress note
            if note:
                goal.setdefault("progress_notes", []).append({
                    "timestamp": now.isoformat(),
                    "note": note,
                    "milestone_completed": idx,
                })

            self._save(data)

            return SkillResult(
                success=True,
                message=f"Milestone {idx} completed: {milestones[idx]['title']} ({done}/{total})",
                data={
                    "goal_id": goal_id,
                    "milestone": milestones[idx]["title"],
                    "progress": f"{done}/{total}",
                    "auto_complete": False,
                },
            )

        # Just add a note
        if note:
            goal.setdefault("progress_notes", []).append({
                "timestamp": now.isoformat(),
                "note": note,
            })
            self._save(data)
            return SkillResult(
                success=True,
                message=f"Progress note added to '{goal['title']}'",
                data={"goal_id": goal_id, "note": note},
            )

        return SkillResult(
            success=False,
            message="Provide milestone_index to complete a milestone, or note to add progress",
        )

    def _complete(self, params: Dict) -> SkillResult:
        """Mark a goal as completed."""
        goal_id = params.get("goal_id", "").strip()
        outcome = params.get("outcome", "")

        if not goal_id:
            return SkillResult(success=False, message="goal_id is required")

        data = self._load()
        goal = self._find_goal(data, goal_id)
        if not goal:
            return SkillResult(success=False, message=f"Goal not found: {goal_id}")

        return self._complete_goal(data, goal, outcome)

    def _complete_goal(self, data: Dict, goal: Dict, outcome: str) -> SkillResult:
        """Internal: complete a goal and move to history."""
        now = datetime.now()
        goal["status"] = "completed"
        goal["completed_at"] = now.isoformat()
        goal["outcome"] = outcome

        # Calculate duration
        try:
            created = datetime.fromisoformat(goal["created_at"])
            duration_hours = round((now - created).total_seconds() / 3600, 1)
            goal["duration_hours"] = duration_hours
        except (ValueError, KeyError):
            duration_hours = None

        # Move to completed
        data.setdefault("completed_goals", []).append(goal)
        data["goals"] = [g for g in data.get("goals", []) if g["id"] != goal["id"]]

        # Log session event
        data.setdefault("session_log", []).append({
            "event": "goal_completed",
            "goal_id": goal["id"],
            "title": goal["title"],
            "pillar": goal["pillar"],
            "priority": goal["priority"],
            "duration_hours": duration_hours,
            "timestamp": now.isoformat(),
        })

        self._save(data)

        return SkillResult(
            success=True,
            message=f"Goal completed: {goal['title']}" + (f" ({duration_hours}h)" if duration_hours else ""),
            data={
                "goal_id": goal["id"],
                "title": goal["title"],
                "pillar": goal["pillar"],
                "duration_hours": duration_hours,
                "outcome": outcome,
            },
        )

    def _abandon(self, params: Dict) -> SkillResult:
        """Abandon a goal."""
        goal_id = params.get("goal_id", "").strip()
        reason = params.get("reason", "")

        if not goal_id:
            return SkillResult(success=False, message="goal_id is required")

        data = self._load()
        goal = self._find_goal(data, goal_id)
        if not goal:
            return SkillResult(success=False, message=f"Goal not found: {goal_id}")

        goal["status"] = "abandoned"
        goal["abandoned_at"] = datetime.now().isoformat()
        goal["abandon_reason"] = reason

        # Move to completed (as abandoned)
        data.setdefault("completed_goals", []).append(goal)
        data["goals"] = [g for g in data.get("goals", []) if g["id"] != goal["id"]]

        # Log
        data.setdefault("session_log", []).append({
            "event": "goal_abandoned",
            "goal_id": goal["id"],
            "title": goal["title"],
            "reason": reason,
            "timestamp": datetime.now().isoformat(),
        })

        self._save(data)

        return SkillResult(
            success=True,
            message=f"Goal abandoned: {goal['title']}" + (f" (reason: {reason})" if reason else ""),
            data={"goal_id": goal["id"], "title": goal["title"], "reason": reason},
        )

    def _add_milestone(self, params: Dict) -> SkillResult:
        """Add a milestone to an existing goal."""
        goal_id = params.get("goal_id", "").strip()
        title = params.get("title", "").strip()

        if not goal_id or not title:
            return SkillResult(success=False, message="goal_id and title are required")

        data = self._load()
        goal = self._find_goal(data, goal_id)
        if not goal:
            return SkillResult(success=False, message=f"Goal not found: {goal_id}")

        milestones = goal.setdefault("milestones", [])
        if len(milestones) >= MAX_MILESTONES_PER_GOAL:
            return SkillResult(
                success=False,
                message=f"Maximum {MAX_MILESTONES_PER_GOAL} milestones per goal",
            )

        new_index = len(milestones)
        milestones.append({
            "index": new_index,
            "title": title,
            "completed": False,
            "completed_at": None,
        })

        self._save(data)

        return SkillResult(
            success=True,
            message=f"Milestone added to '{goal['title']}': {title} (index {new_index})",
            data={"goal_id": goal_id, "milestone_index": new_index, "title": title},
        )

    def _analyze(self, params: Dict) -> SkillResult:
        """Analyze goal completion patterns."""
        data = self._load()
        active_goals = [g for g in data.get("goals", []) if g.get("status") == "active"]
        completed = [g for g in data.get("completed_goals", []) if g.get("status") == "completed"]
        abandoned = [g for g in data.get("completed_goals", []) if g.get("status") == "abandoned"]

        # Per-pillar stats
        pillar_stats = {}
        for pillar in PILLARS:
            p_active = [g for g in active_goals if g.get("pillar") == pillar]
            p_completed = [g for g in completed if g.get("pillar") == pillar]
            p_abandoned = [g for g in abandoned if g.get("pillar") == pillar]

            durations = [g.get("duration_hours", 0) for g in p_completed if g.get("duration_hours")]
            avg_duration = round(sum(durations) / len(durations), 1) if durations else 0

            total_finished = len(p_completed) + len(p_abandoned)
            completion_rate = round(len(p_completed) / total_finished * 100, 1) if total_finished > 0 else 0

            pillar_stats[pillar] = {
                "active": len(p_active),
                "completed": len(p_completed),
                "abandoned": len(p_abandoned),
                "completion_rate": completion_rate,
                "avg_duration_hours": avg_duration,
            }

        # Priority analysis
        priority_completion = {}
        for pri in PRIORITIES:
            p_done = [g for g in completed if g.get("priority") == pri]
            p_total = len(p_done) + len([g for g in abandoned if g.get("priority") == pri])
            priority_completion[pri] = {
                "completed": len(p_done),
                "total_finished": p_total,
                "rate": round(len(p_done) / p_total * 100, 1) if p_total > 0 else 0,
            }

        # Overdue goals
        now = datetime.now()
        overdue = []
        for g in active_goals:
            if g.get("deadline"):
                try:
                    dl = datetime.fromisoformat(g["deadline"])
                    if dl < now:
                        overdue.append({"id": g["id"], "title": g["title"], "hours_overdue": round((now - dl).total_seconds() / 3600, 1)})
                except (ValueError, TypeError):
                    pass

        return SkillResult(
            success=True,
            message=f"Analysis: {len(active_goals)} active, {len(completed)} completed, {len(abandoned)} abandoned",
            data={
                "active_count": len(active_goals),
                "completed_count": len(completed),
                "abandoned_count": len(abandoned),
                "pillar_stats": pillar_stats,
                "priority_completion": priority_completion,
                "overdue_goals": overdue,
                "overdue_count": len(overdue),
            },
        )

    def _focus(self, params: Dict) -> SkillResult:
        """Get a focused work plan: top goal per pillar with next action."""
        data = self._load()
        goals = data.get("goals", [])
        active = [g for g in goals if g.get("status") == "active"]

        if not active:
            return SkillResult(
                success=True,
                message="No active goals. Create goals to build a focus plan.",
                data={"focus_plan": []},
            )

        # Check dependencies
        completed_ids = {g["id"] for g in data.get("completed_goals", [])}
        completed_ids.update(g["id"] for g in goals if g.get("status") == "completed")

        focus_plan = []
        for pillar in PILLARS:
            pillar_goals = [g for g in active if g.get("pillar") == pillar]
            if not pillar_goals:
                continue

            # Find top actionable goal for this pillar
            best = None
            best_score = -1
            for g in pillar_goals:
                deps = g.get("depends_on", [])
                unmet = [d for d in deps if d not in completed_ids]
                if unmet:
                    continue  # Blocked

                score = g.get("priority_score", 2)
                if score > best_score:
                    best_score = score
                    best = g

            if not best:
                focus_plan.append({
                    "pillar": pillar,
                    "status": "blocked",
                    "goal_count": len(pillar_goals),
                })
                continue

            # Find next milestone
            next_ms = None
            milestones = best.get("milestones", [])
            for m in milestones:
                if not m.get("completed"):
                    next_ms = m["title"]
                    break

            done = sum(1 for m in milestones if m.get("completed"))

            focus_plan.append({
                "pillar": pillar,
                "status": "actionable",
                "goal_id": best["id"],
                "goal_title": best["title"],
                "priority": best["priority"],
                "next_action": next_ms or "No milestones defined - work on goal directly",
                "progress": f"{done}/{len(milestones)}" if milestones else "no milestones",
                "deadline": best.get("deadline"),
            })

        return SkillResult(
            success=True,
            message=f"Focus plan: {len([f for f in focus_plan if f.get('status') == 'actionable'])} actionable pillars",
            data={"focus_plan": focus_plan},
        )

    def _cleanup(self, params: Dict) -> SkillResult:
        """Remove stale overdue goals with no progress."""
        stale_hours = params.get("stale_hours", 48)

        data = self._load()
        goals = data.get("goals", [])
        now = datetime.now()

        stale = []
        kept = []
        for g in goals:
            if g.get("status") != "active":
                kept.append(g)
                continue

            # Check if overdue and no progress
            is_stale = False
            if g.get("deadline"):
                try:
                    dl = datetime.fromisoformat(g["deadline"])
                    hours_past = (now - dl).total_seconds() / 3600
                    if hours_past > float(stale_hours):
                        # Check for any progress
                        milestones = g.get("milestones", [])
                        has_progress = any(m.get("completed") for m in milestones) or g.get("progress_notes")
                        if not has_progress:
                            is_stale = True
                except (ValueError, TypeError):
                    pass

            if is_stale:
                g["status"] = "abandoned"
                g["abandon_reason"] = f"Stale: {stale_hours}h past deadline with no progress"
                g["abandoned_at"] = now.isoformat()
                data.setdefault("completed_goals", []).append(g)
                stale.append(g["title"])
            else:
                kept.append(g)

        data["goals"] = kept
        self._save(data)

        return SkillResult(
            success=True,
            message=f"Cleaned up {len(stale)} stale goals",
            data={"removed": stale, "remaining": len(kept)},
        )

    def _find_goal(self, data: Dict, goal_id: str) -> Optional[Dict]:
        """Find an active goal by ID."""
        for g in data.get("goals", []):
            if g["id"] == goal_id:
                return g
        return None
