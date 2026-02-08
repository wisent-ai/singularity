#!/usr/bin/env python3
"""
SchedulerPresetsSkill - Pre-built automation schedule bundles.

Instead of manually wiring 5-10 individual scheduler entries for common
operational patterns, agents can activate a named preset that installs
a complete set of recurring tasks with sensible defaults.

Presets cover the four pillars:
- Self-Improvement: periodic self-assessment, feedback loops, self-tuning
- Revenue: usage report generation, invoice runs, service health checks
- Replication: agent network health polling, capability profile publishing
- Operations: alert polling, incident bridge, reputation sync, observability

Each preset is a bundle of SchedulerSkill entries with configurable
interval multipliers so agents can run them more/less frequently.

Pillar: Operations / Self-Improvement (enables autonomous background operations)
"""

import json
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

from .base import Skill, SkillManifest, SkillAction, SkillResult


# --- Preset Definitions ---

PRESETS: Dict[str, Dict[str, Any]] = {
    "self_improvement": {
        "name": "Self-Improvement Autopilot",
        "description": "Periodic self-assessment, feedback analysis, and self-tuning cycle",
        "pillar": "self_improvement",
        "tasks": [
            {
                "name": "Self-Assessment Profile",
                "skill_id": "self_assessment",
                "action": "profile",
                "params": {},
                "interval_seconds": 3600,  # hourly
                "description": "Generate capability profile to track skill health",
            },
            {
                "name": "Feedback Loop Analysis",
                "skill_id": "feedback_loop",
                "action": "analyze",
                "params": {},
                "interval_seconds": 1800,  # every 30 min
                "description": "Analyze recent performance data for adaptation opportunities",
            },
            {
                "name": "Self-Tuning Cycle",
                "skill_id": "self_tuning",
                "action": "tune",
                "params": {},
                "interval_seconds": 3600,  # hourly
                "description": "Run tuning rules to auto-adjust parameters from metrics",
            },
            {
                "name": "Publish Capability Profile",
                "skill_id": "self_assessment",
                "action": "publish",
                "params": {},
                "interval_seconds": 7200,  # every 2 hours
                "description": "Share updated capability profile with agent network",
            },
        ],
    },
    "operations": {
        "name": "Operations Monitoring",
        "description": "Alert polling, incident bridging, reputation sync, and observability",
        "pillar": "operations",
        "tasks": [
            {
                "name": "Alert Polling",
                "skill_id": "alert_incident_bridge",
                "action": "poll",
                "params": {},
                "interval_seconds": 300,  # every 5 min
                "description": "Check for firing alerts and create/resolve incidents",
            },
            {
                "name": "Reputation Sync",
                "skill_id": "auto_reputation_bridge",
                "action": "poll",
                "params": {},
                "interval_seconds": 600,  # every 10 min
                "description": "Sync task delegation outcomes to reputation scores",
            },
            {
                "name": "Observability Metrics Snapshot",
                "skill_id": "observability",
                "action": "query",
                "params": {"metric_name": "*", "period": "5m"},
                "interval_seconds": 300,  # every 5 min
                "description": "Collect and store latest observability metrics",
            },
        ],
    },
    "revenue": {
        "name": "Revenue Operations",
        "description": "Usage reporting, invoice generation, and service health monitoring",
        "pillar": "revenue",
        "tasks": [
            {
                "name": "Usage Report Generation",
                "skill_id": "usage_tracking",
                "action": "report",
                "params": {"period": "hourly"},
                "interval_seconds": 3600,  # hourly
                "description": "Generate usage report for billing and analytics",
            },
            {
                "name": "Invoice Run",
                "skill_id": "usage_tracking",
                "action": "invoice",
                "params": {},
                "interval_seconds": 86400,  # daily
                "description": "Generate invoices for customers with outstanding usage",
            },
            {
                "name": "Service Health Check",
                "skill_id": "health_monitor",
                "action": "check",
                "params": {},
                "interval_seconds": 600,  # every 10 min
                "description": "Check health of deployed revenue-generating services",
            },
        ],
    },
    "replication": {
        "name": "Replication Network",
        "description": "Agent network health, capability discovery, and knowledge sharing",
        "pillar": "replication",
        "tasks": [
            {
                "name": "Agent Network Discovery",
                "skill_id": "agent_network",
                "action": "discover",
                "params": {},
                "interval_seconds": 1800,  # every 30 min
                "description": "Discover new agents and update network topology",
            },
            {
                "name": "Knowledge Store Sync",
                "skill_id": "knowledge_sharing",
                "action": "query",
                "params": {"query": "recent", "limit": 10},
                "interval_seconds": 1800,  # every 30 min
                "description": "Pull latest knowledge entries from the shared store",
            },
            {
                "name": "Capability Profile Broadcast",
                "skill_id": "self_assessment",
                "action": "publish",
                "params": {},
                "interval_seconds": 3600,  # hourly
                "description": "Broadcast own capabilities for task delegation matching",
            },
        ],
    },
    "full_autonomy": {
        "name": "Full Autonomy Bundle",
        "description": "All presets combined - complete autonomous operation stack",
        "pillar": "all",
        "tasks": [],  # Populated dynamically from other presets
        "is_bundle": True,
        "includes": ["self_improvement", "operations", "revenue", "replication"],
    },
}


class SchedulerPresetsSkill(Skill):
    """
    Pre-built automation schedule bundles for common operational patterns.

    Instead of manually scheduling 10+ recurring tasks, agents activate
    a named preset to install a complete set of automation with sensible
    defaults. Supports interval multipliers for frequency tuning.
    """

    def __init__(self, credentials: Dict = None):
        super().__init__(credentials)
        self._data_dir = Path(__file__).parent.parent / "data"
        self._presets_file = self._data_dir / "scheduler_presets.json"
        self._active_presets: Dict[str, Dict] = {}
        self._installed_tasks: Dict[str, List[str]] = {}  # preset_id -> [task_ids]
        self._load_state()

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="scheduler_presets",
            name="Scheduler Presets",
            version="1.0.0",
            category="autonomy",
            description="Pre-built automation schedule bundles for common operational patterns",
            actions=[
                SkillAction(
                    name="activate",
                    description="Activate a preset schedule bundle, installing all its recurring tasks",
                    parameters={
                        "preset": {
                            "type": "string",
                            "required": True,
                            "description": f"Preset name: {', '.join(PRESETS.keys())}",
                        },
                        "interval_multiplier": {
                            "type": "number",
                            "required": False,
                            "description": "Scale all intervals (0.5 = 2x faster, 2.0 = half speed). Default 1.0",
                        },
                        "dry_run": {
                            "type": "boolean",
                            "required": False,
                            "description": "Preview what would be scheduled without actually installing",
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="deactivate",
                    description="Deactivate a preset and cancel all its scheduled tasks",
                    parameters={
                        "preset": {
                            "type": "string",
                            "required": True,
                            "description": "Preset name to deactivate",
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="list",
                    description="List all available presets and their activation status",
                    parameters={},
                    estimated_cost=0,
                ),
                SkillAction(
                    name="status",
                    description="Show detailed status of an active preset and its tasks",
                    parameters={
                        "preset": {
                            "type": "string",
                            "required": False,
                            "description": "Preset name (omit for all active presets)",
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="customize",
                    description="Customize a specific task within an active preset",
                    parameters={
                        "preset": {
                            "type": "string",
                            "required": True,
                            "description": "Preset name",
                        },
                        "task_name": {
                            "type": "string",
                            "required": True,
                            "description": "Name of the task within the preset to customize",
                        },
                        "interval_seconds": {
                            "type": "number",
                            "required": False,
                            "description": "New interval in seconds",
                        },
                        "params": {
                            "type": "object",
                            "required": False,
                            "description": "New parameters for the task action",
                        },
                        "enabled": {
                            "type": "boolean",
                            "required": False,
                            "description": "Enable or disable this specific task",
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="create_preset",
                    description="Create a custom preset from a list of task definitions",
                    parameters={
                        "name": {
                            "type": "string",
                            "required": True,
                            "description": "Unique preset identifier",
                        },
                        "display_name": {
                            "type": "string",
                            "required": True,
                            "description": "Human-readable preset name",
                        },
                        "description": {
                            "type": "string",
                            "required": True,
                            "description": "What this preset automates",
                        },
                        "tasks": {
                            "type": "array",
                            "required": True,
                            "description": "List of task defs: [{name, skill_id, action, params, interval_seconds}]",
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
            "activate": self._activate,
            "deactivate": self._deactivate,
            "list": self._list,
            "status": self._status,
            "customize": self._customize,
            "create_preset": self._create_preset,
        }
        handler = handlers.get(action)
        if not handler:
            return SkillResult(success=False, message=f"Unknown action: {action}")
        return await handler(params)

    def _get_preset_tasks(self, preset_id: str) -> List[Dict]:
        """Get the full task list for a preset, expanding bundles."""
        preset = PRESETS.get(preset_id) or self._active_presets.get(preset_id)
        if not preset:
            return []

        if preset.get("is_bundle"):
            tasks = []
            for included in preset.get("includes", []):
                tasks.extend(self._get_preset_tasks(included))
            return tasks

        return list(preset.get("tasks", []))

    async def _activate(self, params: Dict) -> SkillResult:
        """Activate a preset schedule bundle."""
        preset_id = params.get("preset", "").strip()
        multiplier = params.get("interval_multiplier", 1.0)
        dry_run = params.get("dry_run", False)

        if not preset_id:
            return SkillResult(success=False, message="preset is required")

        # Check if preset exists (built-in or custom)
        all_presets = {**PRESETS}
        all_presets.update({k: v for k, v in self._active_presets.items() if "tasks" in v})
        # Load custom presets from state
        state = self._load_state()
        for cp in state.get("custom_presets", {}).values():
            all_presets[cp.get("id", "")] = cp

        if preset_id not in all_presets:
            available = list(all_presets.keys())
            return SkillResult(
                success=False,
                message=f"Unknown preset '{preset_id}'. Available: {available}",
            )

        # Check if already active
        if preset_id in self._installed_tasks:
            return SkillResult(
                success=False,
                message=f"Preset '{preset_id}' is already active. Deactivate first to reinstall.",
            )

        multiplier = max(0.1, min(10.0, multiplier))  # clamp

        tasks = self._get_preset_tasks(preset_id)
        if not tasks:
            return SkillResult(success=False, message=f"Preset '{preset_id}' has no tasks")

        if dry_run:
            preview = []
            for t in tasks:
                interval = t["interval_seconds"] * multiplier
                preview.append({
                    "name": t["name"],
                    "skill_id": t["skill_id"],
                    "action": t["action"],
                    "interval_seconds": interval,
                    "interval_human": self._format_interval(interval),
                    "description": t.get("description", ""),
                })
            return SkillResult(
                success=True,
                message=f"Dry run: would install {len(preview)} recurring tasks for '{preset_id}'",
                data={"preset": preset_id, "tasks": preview, "dry_run": True},
            )

        # Install tasks via SchedulerSkill
        installed_ids = []
        errors = []

        for task_def in tasks:
            interval = task_def["interval_seconds"] * multiplier
            schedule_params = {
                "name": f"[{preset_id}] {task_def['name']}",
                "skill_id": task_def["skill_id"],
                "action": task_def["action"],
                "params": task_def.get("params", {}),
                "recurring": True,
                "interval_seconds": interval,
            }

            result = await self._call_scheduler("schedule", schedule_params)
            if result and result.success:
                task_id = result.data.get("id", "")
                installed_ids.append(task_id)
            else:
                errors.append({
                    "task": task_def["name"],
                    "error": result.message if result else "scheduler unavailable",
                })

        self._installed_tasks[preset_id] = installed_ids
        self._active_presets[preset_id] = {
            "activated_at": datetime.now().isoformat(),
            "multiplier": multiplier,
            "task_count": len(installed_ids),
            "task_ids": installed_ids,
        }
        self._save_state()

        if errors:
            return SkillResult(
                success=True,
                message=f"Activated '{preset_id}': {len(installed_ids)} tasks installed, {len(errors)} errors",
                data={
                    "preset": preset_id,
                    "installed": len(installed_ids),
                    "task_ids": installed_ids,
                    "errors": errors,
                },
            )

        return SkillResult(
            success=True,
            message=f"Activated '{preset_id}': {len(installed_ids)} recurring tasks installed (multiplier={multiplier}x)",
            data={
                "preset": preset_id,
                "installed": len(installed_ids),
                "task_ids": installed_ids,
                "multiplier": multiplier,
            },
        )

    async def _deactivate(self, params: Dict) -> SkillResult:
        """Deactivate a preset and cancel all its tasks."""
        preset_id = params.get("preset", "").strip()
        if not preset_id:
            return SkillResult(success=False, message="preset is required")

        if preset_id not in self._installed_tasks:
            return SkillResult(
                success=False,
                message=f"Preset '{preset_id}' is not active",
            )

        task_ids = self._installed_tasks[preset_id]
        cancelled = 0
        errors = []

        for task_id in task_ids:
            result = await self._call_scheduler("cancel", {"task_id": task_id})
            if result and result.success:
                cancelled += 1
            else:
                errors.append(task_id)

        del self._installed_tasks[preset_id]
        if preset_id in self._active_presets:
            del self._active_presets[preset_id]
        self._save_state()

        return SkillResult(
            success=True,
            message=f"Deactivated '{preset_id}': cancelled {cancelled}/{len(task_ids)} tasks",
            data={
                "preset": preset_id,
                "cancelled": cancelled,
                "total": len(task_ids),
                "errors": errors,
            },
        )

    async def _list(self, params: Dict) -> SkillResult:
        """List all available presets with status."""
        state = self._load_state()
        custom_presets = state.get("custom_presets", {})

        presets_info = []
        all_preset_ids = list(PRESETS.keys()) + list(custom_presets.keys())

        for pid in all_preset_ids:
            preset = PRESETS.get(pid) or custom_presets.get(pid, {})
            is_active = pid in self._installed_tasks
            task_count = len(self._get_preset_tasks(pid)) if pid in PRESETS else len(preset.get("tasks", []))

            info = {
                "id": pid,
                "name": preset.get("name", pid),
                "description": preset.get("description", ""),
                "pillar": preset.get("pillar", "custom"),
                "task_count": task_count,
                "active": is_active,
                "is_bundle": preset.get("is_bundle", False),
                "custom": pid in custom_presets,
            }

            if is_active:
                active_info = self._active_presets.get(pid, {})
                info["activated_at"] = active_info.get("activated_at", "")
                info["multiplier"] = active_info.get("multiplier", 1.0)

            presets_info.append(info)

        active_count = sum(1 for p in presets_info if p["active"])
        return SkillResult(
            success=True,
            message=f"{len(presets_info)} presets available ({active_count} active)",
            data={"presets": presets_info, "total": len(presets_info), "active": active_count},
        )

    async def _status(self, params: Dict) -> SkillResult:
        """Detailed status of active preset(s)."""
        preset_id = params.get("preset", "").strip()

        if preset_id:
            if preset_id not in self._installed_tasks:
                return SkillResult(success=False, message=f"Preset '{preset_id}' is not active")
            presets_to_check = [preset_id]
        else:
            presets_to_check = list(self._installed_tasks.keys())

        if not presets_to_check:
            return SkillResult(success=True, message="No active presets", data={"presets": []})

        results = []
        for pid in presets_to_check:
            task_ids = self._installed_tasks.get(pid, [])
            active_info = self._active_presets.get(pid, {})

            # Try to get task details from scheduler
            task_details = []
            scheduler_result = await self._call_scheduler("list", {"include_completed": True})
            if scheduler_result and scheduler_result.success:
                all_tasks = scheduler_result.data.get("tasks", [])
                for tid in task_ids:
                    task = next((t for t in all_tasks if t.get("id") == tid), None)
                    if task:
                        task_details.append({
                            "id": task["id"],
                            "name": task["name"],
                            "status": task.get("status", "unknown"),
                            "run_count": task.get("run_count", 0),
                            "last_success": task.get("last_success"),
                            "last_run_at": task.get("last_run_at"),
                            "enabled": task.get("enabled", True),
                        })

            results.append({
                "preset": pid,
                "activated_at": active_info.get("activated_at", ""),
                "multiplier": active_info.get("multiplier", 1.0),
                "task_count": len(task_ids),
                "tasks": task_details,
            })

        return SkillResult(
            success=True,
            message=f"Status for {len(results)} active preset(s)",
            data={"presets": results},
        )

    async def _customize(self, params: Dict) -> SkillResult:
        """Customize a specific task within an active preset."""
        preset_id = params.get("preset", "").strip()
        task_name = params.get("task_name", "").strip()
        new_interval = params.get("interval_seconds")
        new_params = params.get("params")
        enabled = params.get("enabled")

        if not preset_id:
            return SkillResult(success=False, message="preset is required")
        if not task_name:
            return SkillResult(success=False, message="task_name is required")
        if preset_id not in self._installed_tasks:
            return SkillResult(success=False, message=f"Preset '{preset_id}' is not active")

        # Find the task by name in scheduler
        task_ids = self._installed_tasks.get(preset_id, [])
        scheduler_result = await self._call_scheduler("list", {"include_completed": True})
        if not scheduler_result or not scheduler_result.success:
            return SkillResult(success=False, message="Cannot reach scheduler to find tasks")

        all_tasks = scheduler_result.data.get("tasks", [])
        target_task = None
        for tid in task_ids:
            task = next((t for t in all_tasks if t.get("id") == tid), None)
            if task and task_name.lower() in task.get("name", "").lower():
                target_task = task
                break

        if not target_task:
            task_names = []
            for tid in task_ids:
                task = next((t for t in all_tasks if t.get("id") == tid), None)
                if task:
                    task_names.append(task.get("name", tid))
            return SkillResult(
                success=False,
                message=f"Task '{task_name}' not found. Available: {task_names}",
            )

        changes = []

        # Handle enable/disable via pause/resume
        if enabled is not None:
            action = "resume" if enabled else "pause"
            result = await self._call_scheduler(action, {"task_id": target_task["id"]})
            if result and result.success:
                changes.append(f"{'enabled' if enabled else 'disabled'}")

        # Note: SchedulerSkill doesn't have a native "update" action,
        # so for interval/param changes we cancel and re-schedule
        if new_interval is not None or new_params is not None:
            # Cancel old
            await self._call_scheduler("cancel", {"task_id": target_task["id"]})

            # Re-schedule with new values
            schedule_params = {
                "name": target_task["name"],
                "skill_id": target_task["skill_id"],
                "action": target_task["action"],
                "params": new_params if new_params is not None else target_task.get("params", {}),
                "recurring": True,
                "interval_seconds": new_interval if new_interval is not None else target_task.get("interval_seconds", 60),
            }
            result = await self._call_scheduler("schedule", schedule_params)
            if result and result.success:
                new_id = result.data.get("id", "")
                # Update our tracking
                idx = task_ids.index(target_task["id"]) if target_task["id"] in task_ids else -1
                if idx >= 0:
                    task_ids[idx] = new_id
                    self._save_state()
                if new_interval is not None:
                    changes.append(f"interval={new_interval}s")
                if new_params is not None:
                    changes.append("params updated")

        if not changes:
            return SkillResult(success=False, message="No changes specified")

        return SkillResult(
            success=True,
            message=f"Customized '{task_name}': {', '.join(changes)}",
            data={"preset": preset_id, "task_name": task_name, "changes": changes},
        )

    async def _create_preset(self, params: Dict) -> SkillResult:
        """Create a custom preset."""
        name = params.get("name", "").strip()
        display_name = params.get("display_name", "").strip()
        description = params.get("description", "").strip()
        tasks = params.get("tasks", [])

        if not name:
            return SkillResult(success=False, message="name is required")
        if not display_name:
            return SkillResult(success=False, message="display_name is required")
        if not tasks:
            return SkillResult(success=False, message="tasks list is required (at least 1 task)")

        if name in PRESETS:
            return SkillResult(success=False, message=f"Cannot override built-in preset '{name}'")

        # Validate task definitions
        validated_tasks = []
        for i, t in enumerate(tasks):
            if not isinstance(t, dict):
                return SkillResult(success=False, message=f"Task {i} must be a dict")
            if not t.get("skill_id") or not t.get("action"):
                return SkillResult(success=False, message=f"Task {i} requires skill_id and action")

            validated_tasks.append({
                "name": t.get("name", f"Task {i+1}"),
                "skill_id": t["skill_id"],
                "action": t["action"],
                "params": t.get("params", {}),
                "interval_seconds": t.get("interval_seconds", 3600),
                "description": t.get("description", ""),
            })

        custom_preset = {
            "id": name,
            "name": display_name,
            "description": description,
            "pillar": "custom",
            "tasks": validated_tasks,
            "created_at": datetime.now().isoformat(),
        }

        state = self._load_state()
        custom_presets = state.get("custom_presets", {})
        custom_presets[name] = custom_preset
        state["custom_presets"] = custom_presets
        self._save_raw_state(state)

        return SkillResult(
            success=True,
            message=f"Created custom preset '{display_name}' with {len(validated_tasks)} tasks",
            data={"preset": custom_preset},
        )

    # --- Helpers ---

    async def _call_scheduler(self, action: str, params: Dict) -> Optional[SkillResult]:
        """Call SchedulerSkill via context or direct."""
        if self.context:
            try:
                return await self.context.call_skill("scheduler", action, params)
            except Exception:
                pass

        # Direct fallback: instantiate scheduler
        try:
            from .scheduler import SchedulerSkill
            sched = SchedulerSkill()
            if self.context:
                sched.set_context(self.context)
            return await sched.execute(action, params)
        except Exception as e:
            return SkillResult(success=False, message=f"Scheduler unavailable: {e}")

    def _format_interval(self, seconds: float) -> str:
        """Format seconds to human-readable interval."""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            return f"{seconds / 60:.0f}m"
        elif seconds < 86400:
            return f"{seconds / 3600:.1f}h"
        else:
            return f"{seconds / 86400:.1f}d"

    def _load_state(self) -> Dict:
        """Load persistent state."""
        try:
            if self._presets_file.exists():
                data = json.loads(self._presets_file.read_text())
                self._installed_tasks = data.get("installed_tasks", {})
                self._active_presets = data.get("active_presets", {})
                return data
        except Exception:
            pass
        return {"installed_tasks": {}, "active_presets": {}, "custom_presets": {}}

    def _save_state(self):
        """Save persistent state."""
        # Load only to preserve custom_presets, don't overwrite in-memory state
        try:
            if self._presets_file.exists():
                existing = json.loads(self._presets_file.read_text())
                custom_presets = existing.get("custom_presets", {})
            else:
                custom_presets = {}
        except Exception:
            custom_presets = {}

        state = {
            "installed_tasks": self._installed_tasks,
            "active_presets": self._active_presets,
            "custom_presets": custom_presets,
            "saved_at": datetime.now().isoformat(),
        }
        self._save_raw_state(state)

    def _save_raw_state(self, state: Dict):
        """Write state dict to disk."""
        try:
            self._data_dir.mkdir(parents=True, exist_ok=True)
            self._presets_file.write_text(json.dumps(state, indent=2))
        except Exception:
            pass
