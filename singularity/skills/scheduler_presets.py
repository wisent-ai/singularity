#!/usr/bin/env python3
"""
SchedulerPresetsSkill - One-command setup for common automation schedules.

The agent has many skills that should run periodically (alert polling,
reputation syncing, health checks, self-assessment, tuning cycles) but
setting each one up manually via SchedulerSkill is tedious and error-prone.

This skill provides pre-built automation presets that wire multiple
SchedulerSkill schedules in a single command:

1. **health_check** - Periodic health monitoring: polls alerts, checks
   incident status, runs diagnostics
2. **reputation_sync** - Auto-sync delegation outcomes to reputation scores
3. **self_improvement** - Self-assessment, tuning cycles, experiment analysis
4. **full_autonomy** - All presets combined for maximum autonomy

Each preset is a collection of SchedulerSkill recurring tasks with sensible
defaults. Presets can be enabled/disabled as a group, and intervals can be
customized.

Pillar: Self-Improvement + Goal Setting
- Eliminates manual setup overhead for autonomous behavior
- Makes the agent "set and forget" autonomous from first boot

Actions:
- apply: Enable a preset (schedules all its tasks)
- remove: Disable a preset (cancels all its tasks)
- list: Show all available presets and their status
- status: Show active presets and their scheduled tasks
- customize: Override interval for a preset's tasks
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from .base import Skill, SkillResult, SkillManifest, SkillAction

DATA_FILE = Path(__file__).parent.parent / "data" / "scheduler_presets.json"


def _now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


# ── Preset definitions ──

PRESETS = {
    "health_check": {
        "name": "Health Check",
        "description": "Periodic alert polling, incident checking, and system diagnostics",
        "category": "operations",
        "tasks": [
            {
                "name": "poll_alerts",
                "skill_id": "alert_incident_bridge",
                "action": "poll",
                "params": {},
                "interval_seconds": 300,  # 5 minutes
                "description": "Check observability alerts and auto-create incidents",
            },
            {
                "name": "check_health",
                "skill_id": "health_monitor",
                "action": "check",
                "params": {},
                "interval_seconds": 600,  # 10 minutes
                "description": "Run health check across all monitored systems",
            },
        ],
    },
    "reputation_sync": {
        "name": "Reputation Sync",
        "description": "Auto-sync task delegation outcomes to agent reputation scores",
        "category": "replication",
        "tasks": [
            {
                "name": "sync_reputation",
                "skill_id": "auto_reputation_bridge",
                "action": "poll",
                "params": {},
                "interval_seconds": 600,  # 10 minutes
                "description": "Scan completed delegations and update reputation",
            },
            {
                "name": "sync_task_reputation",
                "skill_id": "task_reputation_bridge",
                "action": "sync",
                "params": {},
                "interval_seconds": 600,  # 10 minutes
                "description": "Sync task outcomes to reputation via bridge",
            },
        ],
    },
    "self_improvement": {
        "name": "Self-Improvement Cycle",
        "description": "Periodic self-assessment, tuning, and experiment analysis",
        "category": "self_improvement",
        "tasks": [
            {
                "name": "self_assess",
                "skill_id": "self_assessment",
                "action": "profile",
                "params": {},
                "interval_seconds": 3600,  # 1 hour
                "description": "Generate capability profile and track score trends",
            },
            {
                "name": "auto_tune",
                "skill_id": "self_tuning",
                "action": "tune",
                "params": {},
                "interval_seconds": 1800,  # 30 minutes
                "description": "Run tuning cycle to auto-adjust parameters",
            },
            {
                "name": "analyze_experiments",
                "skill_id": "experiment",
                "action": "list",
                "params": {},
                "interval_seconds": 3600,  # 1 hour
                "description": "Review running experiments and check conclusions",
            },
        ],
    },
    "cost_optimization": {
        "name": "Cost Optimization",
        "description": "Periodic cost analysis and budget tuning",
        "category": "revenue",
        "tasks": [
            {
                "name": "optimize_costs",
                "skill_id": "cost_optimizer",
                "action": "analyze",
                "params": {},
                "interval_seconds": 3600,  # 1 hour
                "description": "Analyze spending patterns and find savings",
            },
            {
                "name": "check_usage",
                "skill_id": "usage_tracking",
                "action": "summary",
                "params": {},
                "interval_seconds": 1800,  # 30 minutes
                "description": "Track API usage and quota consumption",
            },
        ],
    },
    "full_autonomy": {
        "name": "Full Autonomy",
        "description": "All presets combined - maximum autonomous operation",
        "category": "all",
        "includes": ["health_check", "reputation_sync", "self_improvement", "cost_optimization"],
        "tasks": [],  # Composed from included presets
    },
}


class SchedulerPresetsSkill(Skill):
    """
    One-command setup for common automation schedules.

    Provides pre-built collections of SchedulerSkill recurring tasks
    that can be enabled/disabled as a group for common autonomous
    operation patterns.
    """

    def __init__(self, credentials: Dict[str, str] = None):
        super().__init__(credentials)
        self._store = None

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="scheduler_presets",
            name="Scheduler Presets",
            version="1.0.0",
            category="meta",
            description="One-command setup for common automation schedules (health checks, reputation sync, self-improvement, cost optimization)",
            actions=self.get_actions(),
            required_credentials=[],
            install_cost=0,
            author="singularity",
        )

    def get_actions(self) -> List[SkillAction]:
        return [
            SkillAction(
                name="apply",
                description="Enable a preset - schedules all its recurring tasks",
                parameters={
                    "preset": {"type": "string", "required": True,
                               "description": f"Preset name: {', '.join(PRESETS.keys())}"},
                    "dry_run": {"type": "boolean", "required": False,
                                "description": "Preview what would be scheduled (default: False)"},
                },
            ),
            SkillAction(
                name="remove",
                description="Disable a preset - cancels all its scheduled tasks",
                parameters={
                    "preset": {"type": "string", "required": True,
                               "description": "Preset name to disable"},
                },
            ),
            SkillAction(
                name="list",
                description="Show all available presets and their status",
                parameters={},
            ),
            SkillAction(
                name="status",
                description="Show active presets and their scheduled task details",
                parameters={},
            ),
            SkillAction(
                name="customize",
                description="Override the interval for a specific task within a preset",
                parameters={
                    "preset": {"type": "string", "required": True,
                               "description": "Preset name"},
                    "task_name": {"type": "string", "required": True,
                                  "description": "Task name within the preset"},
                    "interval_seconds": {"type": "number", "required": True,
                                         "description": "New interval in seconds"},
                },
            ),
        ]

    async def execute(self, action: str, params: Dict) -> SkillResult:
        handlers = {
            "apply": self._apply,
            "remove": self._remove,
            "list": self._list,
            "status": self._status,
            "customize": self._customize,
        }
        handler = handlers.get(action)
        if not handler:
            return SkillResult(
                success=False,
                message=f"Unknown action: {action}. Available: {', '.join(handlers.keys())}",
            )
        return await handler(params)

    # ── Persistence ──

    def _load(self) -> Dict:
        if self._store is not None:
            return self._store
        try:
            if DATA_FILE.exists():
                with open(DATA_FILE) as f:
                    self._store = json.load(f)
                    return self._store
        except (json.JSONDecodeError, OSError):
            pass
        self._store = self._default_state()
        return self._store

    def _save(self, state: Dict = None):
        if state is not None:
            self._store = state
        if self._store is None:
            return
        DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(DATA_FILE, "w") as f:
            json.dump(self._store, f, indent=2)

    def _default_state(self) -> Dict:
        return {
            "active_presets": {},  # preset_name -> {applied_at, task_ids: [...], overrides: {}}
            "history": [],
        }

    # ── Helpers ──

    def _resolve_tasks(self, preset_name: str) -> List[Dict]:
        """Resolve all tasks for a preset, including from included presets."""
        preset = PRESETS.get(preset_name)
        if not preset:
            return []

        tasks = list(preset.get("tasks", []))

        # Resolve included presets
        for included in preset.get("includes", []):
            included_preset = PRESETS.get(included)
            if included_preset:
                tasks.extend(included_preset.get("tasks", []))

        return tasks

    def _get_task_prefix(self, preset_name: str) -> str:
        """Generate a unique prefix for scheduled task names."""
        return f"preset_{preset_name}_"

    # ── Handlers ──

    async def _apply(self, params: Dict) -> SkillResult:
        """Enable a preset by scheduling all its tasks."""
        preset_name = params.get("preset", "").strip()
        dry_run = params.get("dry_run", False)

        if preset_name not in PRESETS:
            return SkillResult(
                success=False,
                message=f"Unknown preset: '{preset_name}'. Available: {', '.join(PRESETS.keys())}",
            )

        state = self._load()

        # Check if already active
        if preset_name in state["active_presets"]:
            return SkillResult(
                success=True,
                message=f"Preset '{preset_name}' is already active. Use 'remove' first to re-apply.",
                data={"preset": preset_name, "already_active": True},
            )

        tasks = self._resolve_tasks(preset_name)
        if not tasks:
            return SkillResult(
                success=False,
                message=f"Preset '{preset_name}' has no tasks to schedule.",
            )

        # Apply overrides if any
        overrides = state.get("active_presets", {}).get(preset_name, {}).get("overrides", {})

        scheduled = []
        failed = []
        prefix = self._get_task_prefix(preset_name)

        for task in tasks:
            task_name = f"{prefix}{task['name']}"
            interval = overrides.get(task["name"], task["interval_seconds"])

            schedule_params = {
                "name": task_name,
                "skill_id": task["skill_id"],
                "action": task["action"],
                "params": task.get("params", {}),
                "type": "recurring",
                "interval_seconds": interval,
            }

            if dry_run:
                scheduled.append({
                    "task_name": task_name,
                    "skill_id": task["skill_id"],
                    "action": task["action"],
                    "interval_seconds": interval,
                    "description": task.get("description", ""),
                })
                continue

            # Schedule via SchedulerSkill
            result = await self._schedule_task(schedule_params)
            if result and result.get("success"):
                scheduled.append({
                    "task_name": task_name,
                    "task_id": result.get("task_id", task_name),
                    "skill_id": task["skill_id"],
                    "action": task["action"],
                    "interval_seconds": interval,
                })
            else:
                failed.append({
                    "task_name": task_name,
                    "error": result.get("error", "Unknown error") if result else "No scheduler available",
                })

        if not dry_run:
            state["active_presets"][preset_name] = {
                "applied_at": _now_iso(),
                "task_ids": [s.get("task_id", s["task_name"]) for s in scheduled],
                "tasks_scheduled": len(scheduled),
                "tasks_failed": len(failed),
                "overrides": overrides,
            }
            state["history"].append({
                "action": "apply",
                "preset": preset_name,
                "timestamp": _now_iso(),
                "tasks_scheduled": len(scheduled),
                "tasks_failed": len(failed),
            })
            state["history"] = state["history"][-100:]
            self._save(state)

        prefix_msg = "[DRY RUN] " if dry_run else ""
        return SkillResult(
            success=True,
            message=f"{prefix_msg}Applied preset '{PRESETS[preset_name]['name']}': "
                    f"{len(scheduled)} task(s) scheduled"
                    + (f", {len(failed)} failed" if failed else ""),
            data={
                "preset": preset_name,
                "dry_run": dry_run,
                "scheduled": scheduled,
                "failed": failed,
            },
        )

    async def _remove(self, params: Dict) -> SkillResult:
        """Disable a preset by cancelling all its tasks."""
        preset_name = params.get("preset", "").strip()

        if preset_name not in PRESETS:
            return SkillResult(
                success=False,
                message=f"Unknown preset: '{preset_name}'.",
            )

        state = self._load()
        active = state["active_presets"].get(preset_name)

        if not active:
            return SkillResult(
                success=True,
                message=f"Preset '{preset_name}' is not active.",
                data={"preset": preset_name, "was_active": False},
            )

        # Cancel all scheduled tasks
        cancelled = 0
        for task_id in active.get("task_ids", []):
            result = await self._cancel_task(task_id)
            if result:
                cancelled += 1

        del state["active_presets"][preset_name]
        state["history"].append({
            "action": "remove",
            "preset": preset_name,
            "timestamp": _now_iso(),
            "tasks_cancelled": cancelled,
        })
        state["history"] = state["history"][-100:]
        self._save(state)

        return SkillResult(
            success=True,
            message=f"Removed preset '{PRESETS[preset_name]['name']}': {cancelled} task(s) cancelled.",
            data={"preset": preset_name, "cancelled": cancelled},
        )

    async def _list(self, params: Dict) -> SkillResult:
        """List all available presets."""
        state = self._load()
        active_names = set(state.get("active_presets", {}).keys())

        presets_info = []
        for name, preset in PRESETS.items():
            tasks = self._resolve_tasks(name)
            presets_info.append({
                "name": name,
                "display_name": preset["name"],
                "description": preset["description"],
                "category": preset["category"],
                "task_count": len(tasks),
                "active": name in active_names,
                "includes": preset.get("includes", []),
            })

        active_count = len(active_names)
        return SkillResult(
            success=True,
            message=f"{len(PRESETS)} presets available, {active_count} active.",
            data={"presets": presets_info},
        )

    async def _status(self, params: Dict) -> SkillResult:
        """Show active presets and their details."""
        state = self._load()
        active = state.get("active_presets", {})

        if not active:
            return SkillResult(
                success=True,
                message="No presets are currently active. Use 'apply' to enable one.",
                data={"active_presets": {}},
            )

        details = {}
        for name, info in active.items():
            preset = PRESETS.get(name, {})
            tasks = self._resolve_tasks(name)
            details[name] = {
                "display_name": preset.get("name", name),
                "applied_at": info["applied_at"],
                "tasks_scheduled": info["tasks_scheduled"],
                "tasks_failed": info.get("tasks_failed", 0),
                "task_ids": info["task_ids"],
                "overrides": info.get("overrides", {}),
                "tasks": [
                    {
                        "name": t["name"],
                        "skill_id": t["skill_id"],
                        "action": t["action"],
                        "interval_seconds": info.get("overrides", {}).get(t["name"], t["interval_seconds"]),
                    }
                    for t in tasks
                ],
            }

        return SkillResult(
            success=True,
            message=f"{len(active)} preset(s) active.",
            data={"active_presets": details},
        )

    async def _customize(self, params: Dict) -> SkillResult:
        """Override interval for a task in a preset."""
        preset_name = params.get("preset", "").strip()
        task_name = params.get("task_name", "").strip()
        interval = params.get("interval_seconds")

        if preset_name not in PRESETS:
            return SkillResult(
                success=False,
                message=f"Unknown preset: '{preset_name}'.",
            )

        if not task_name:
            return SkillResult(success=False, message="task_name is required")

        if interval is None or float(interval) < 10:
            return SkillResult(
                success=False,
                message="interval_seconds must be at least 10 seconds.",
            )

        interval = float(interval)

        # Verify task exists
        tasks = self._resolve_tasks(preset_name)
        task_names = [t["name"] for t in tasks]
        if task_name not in task_names:
            return SkillResult(
                success=False,
                message=f"Task '{task_name}' not found in preset '{preset_name}'. "
                        f"Available: {', '.join(task_names)}",
            )

        state = self._load()

        # Store the override
        if preset_name not in state["active_presets"]:
            state["active_presets"][preset_name] = {
                "applied_at": None,
                "task_ids": [],
                "tasks_scheduled": 0,
                "overrides": {},
            }

        state["active_presets"][preset_name].setdefault("overrides", {})[task_name] = interval
        self._save(state)

        return SkillResult(
            success=True,
            message=f"Custom interval set: '{task_name}' in '{preset_name}' → {interval}s. "
                    f"Re-apply preset to take effect.",
            data={
                "preset": preset_name,
                "task_name": task_name,
                "interval_seconds": interval,
            },
        )

    # ── Scheduler integration ──

    async def _schedule_task(self, params: Dict) -> Optional[Dict]:
        """Schedule a task via SchedulerSkill."""
        if self.context:
            try:
                result = await self.context.call_skill("scheduler", "schedule", params)
                if result.success:
                    return {
                        "success": True,
                        "task_id": result.data.get("task_id", params["name"]) if result.data else params["name"],
                    }
                return {"success": False, "error": result.message}
            except Exception as e:
                return {"success": False, "error": str(e)}

        # Without context, record intent but can't actually schedule
        return {"success": True, "task_id": params["name"], "note": "recorded_without_scheduler"}

    async def _cancel_task(self, task_id: str) -> bool:
        """Cancel a scheduled task via SchedulerSkill."""
        if self.context:
            try:
                result = await self.context.call_skill("scheduler", "cancel", {"task_id": task_id})
                return result.success
            except Exception:
                pass
        return True  # Assume success if no scheduler

    async def initialize(self) -> bool:
        self.initialized = True
        return True
