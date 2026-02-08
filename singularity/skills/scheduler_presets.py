#!/usr/bin/env python3
"""
SchedulerPresetsSkill - Pre-built automation schedules for common agent operations.

Instead of manually wiring SchedulerSkill calls for each recurring task,
this skill provides one-command setup of common automation patterns:

- Health monitoring: periodic health checks across all skills
- Alert polling: automatic alert→incident bridge polling
- Self-assessment: periodic capability profiling and gap analysis
- Self-tuning: recurring parameter optimization cycles
- Reputation polling: auto-reputation updates from task completions
- Revenue reporting: periodic revenue/usage analytics
- Knowledge sync: periodic knowledge sharing between agents
- Full autonomy: all presets at once for fully autonomous operation

Each preset is a named collection of scheduler entries with sensible defaults
that can be customized. Presets can be applied, listed, removed, and their
status checked as a group.

Pillar: Operations / Self-Improvement - enables hands-free autonomous operation
by wiring together existing skills into recurring automation patterns.
"""

import json
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

from .base import Skill, SkillManifest, SkillAction, SkillResult


# Persistent storage
DATA_DIR = Path(__file__).parent.parent / "data"
PRESETS_FILE = DATA_DIR / "scheduler_presets.json"


@dataclass
class PresetSchedule:
    """A single scheduled entry within a preset."""
    name: str
    skill_id: str
    action: str
    params: Dict
    interval_seconds: float
    description: str


@dataclass
class PresetDefinition:
    """A complete preset with multiple scheduled entries."""
    preset_id: str
    name: str
    description: str
    pillar: str  # which pillar this serves
    schedules: List[PresetSchedule]
    category: str = "operations"


# ── Built-in preset definitions ──────────────────────────────────────────

BUILTIN_PRESETS: Dict[str, PresetDefinition] = {
    "health_monitoring": PresetDefinition(
        preset_id="health_monitoring",
        name="Health Monitoring",
        description="Periodic health checks across deployed services and skills",
        pillar="operations",
        schedules=[
            PresetSchedule(
                name="Skill Health Check",
                skill_id="self_assessment",
                action="benchmark",
                params={},
                interval_seconds=3600,  # every hour
                description="Benchmark all installed skills for health status",
            ),
            PresetSchedule(
                name="Diagnostics Scan",
                skill_id="diagnostics",
                action="scan",
                params={},
                interval_seconds=1800,  # every 30 min
                description="Run diagnostic scan for system issues",
            ),
        ],
    ),
    "alert_polling": PresetDefinition(
        preset_id="alert_polling",
        name="Alert Polling",
        description="Automatic alert checking and incident creation from observability alerts",
        pillar="operations",
        schedules=[
            PresetSchedule(
                name="Alert→Incident Poll",
                skill_id="alert_incident_bridge",
                action="poll",
                params={},
                interval_seconds=300,  # every 5 min
                description="Check for fired alerts and auto-create incidents",
            ),
            PresetSchedule(
                name="Observability Alert Check",
                skill_id="observability",
                action="check_alerts",
                params={},
                interval_seconds=120,  # every 2 min
                description="Evaluate alert rules against current metrics",
            ),
        ],
    ),
    "self_assessment": PresetDefinition(
        preset_id="self_assessment",
        name="Self-Assessment",
        description="Periodic capability profiling, gap analysis, and profile publishing",
        pillar="self_improvement",
        schedules=[
            PresetSchedule(
                name="Capability Profile",
                skill_id="self_assessment",
                action="profile",
                params={},
                interval_seconds=7200,  # every 2 hours
                description="Generate full capability profile with category scores",
            ),
            PresetSchedule(
                name="Gap Analysis",
                skill_id="self_assessment",
                action="gaps",
                params={},
                interval_seconds=14400,  # every 4 hours
                description="Identify missing capabilities and rank by impact",
            ),
            PresetSchedule(
                name="Publish Capabilities",
                skill_id="self_assessment",
                action="publish",
                params={},
                interval_seconds=7200,  # every 2 hours
                description="Share capability profile with other agents",
            ),
        ],
    ),
    "self_tuning": PresetDefinition(
        preset_id="self_tuning",
        name="Self-Tuning",
        description="Recurring parameter optimization based on observability metrics",
        pillar="self_improvement",
        schedules=[
            PresetSchedule(
                name="Auto-Tune Cycle",
                skill_id="self_tuning",
                action="tune",
                params={},
                interval_seconds=900,  # every 15 min
                description="Run tuning cycle - evaluate rules and adjust parameters",
            ),
        ],
    ),
    "reputation_polling": PresetDefinition(
        preset_id="reputation_polling",
        name="Reputation Polling",
        description="Auto-update agent reputation from completed task delegations",
        pillar="replication",
        schedules=[
            PresetSchedule(
                name="Reputation Bridge Poll",
                skill_id="auto_reputation_bridge",
                action="poll",
                params={},
                interval_seconds=600,  # every 10 min
                description="Process completed delegations and update reputation scores",
            ),
        ],
    ),
    "revenue_reporting": PresetDefinition(
        preset_id="revenue_reporting",
        name="Revenue Reporting",
        description="Periodic revenue analytics and usage reporting",
        pillar="revenue",
        schedules=[
            PresetSchedule(
                name="Usage Analytics",
                skill_id="usage_tracking",
                action="analytics",
                params={},
                interval_seconds=3600,  # every hour
                description="Generate usage analytics for all customers",
            ),
        ],
    ),
    "knowledge_sync": PresetDefinition(
        preset_id="knowledge_sync",
        name="Knowledge Sync",
        description="Periodic knowledge sharing and collective learning between agents",
        pillar="replication",
        schedules=[
            PresetSchedule(
                name="Knowledge Query",
                skill_id="knowledge_sharing",
                action="query",
                params={"category": "optimization", "min_confidence": 0.5},
                interval_seconds=3600,  # every hour
                description="Query knowledge store for new agent discoveries",
            ),
        ],
    ),
    "feedback_loop": PresetDefinition(
        preset_id="feedback_loop",
        name="Feedback Loop",
        description="Periodic performance analysis and behavioral adaptation",
        pillar="self_improvement",
        schedules=[
            PresetSchedule(
                name="Feedback Analysis",
                skill_id="feedback_loop",
                action="analyze",
                params={},
                interval_seconds=1800,  # every 30 min
                description="Analyze performance data and generate adaptations",
            ),
        ],
    ),
}

# "Full autonomy" is a meta-preset that includes all others
FULL_AUTONOMY_PRESETS = list(BUILTIN_PRESETS.keys())


class SchedulerPresetsSkill(Skill):
    """
    Pre-built automation schedules for common agent operations.

    Provides one-command setup of recurring automation patterns by wiring
    together existing skills through the SchedulerSkill. Instead of manually
    creating 10+ scheduler entries, use 'apply' with a preset name or
    'apply_all' for full autonomous operation.
    """

    def __init__(self, credentials: Dict = None):
        super().__init__(credentials)
        self._applied: Dict[str, Dict] = {}  # preset_id -> {task_ids, applied_at, ...}
        self._custom_presets: Dict[str, Dict] = {}
        self._load_state()

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="scheduler_presets",
            name="Scheduler Presets",
            version="1.0.0",
            category="operations",
            description="Pre-built automation schedules - one-command setup for health checks, alert polling, self-tuning, and more",
            actions=[
                SkillAction(
                    name="list_presets",
                    description="List all available presets with descriptions and schedules",
                    parameters={
                        "pillar": {
                            "type": "string",
                            "required": False,
                            "description": "Filter by pillar (self_improvement, revenue, replication, operations)",
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="apply",
                    description="Apply a preset - creates all its recurring scheduler entries",
                    parameters={
                        "preset_id": {
                            "type": "string",
                            "required": True,
                            "description": "ID of the preset to apply (e.g., 'health_monitoring', 'self_tuning')",
                        },
                        "interval_multiplier": {
                            "type": "number",
                            "required": False,
                            "description": "Multiply all intervals by this factor (e.g., 0.5 = twice as fast, 2 = half as often). Default 1.0",
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="apply_all",
                    description="Apply ALL presets at once for full autonomous operation",
                    parameters={
                        "interval_multiplier": {
                            "type": "number",
                            "required": False,
                            "description": "Multiply all intervals by this factor. Default 1.0",
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="remove",
                    description="Remove an applied preset and cancel all its scheduler entries",
                    parameters={
                        "preset_id": {
                            "type": "string",
                            "required": True,
                            "description": "ID of the preset to remove",
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="remove_all",
                    description="Remove all applied presets",
                    parameters={},
                    estimated_cost=0,
                ),
                SkillAction(
                    name="status",
                    description="Show status of all applied presets and their scheduler entries",
                    parameters={},
                    estimated_cost=0,
                ),
                SkillAction(
                    name="create_custom",
                    description="Create a custom preset from a list of schedule definitions",
                    parameters={
                        "preset_id": {
                            "type": "string",
                            "required": True,
                            "description": "Unique ID for the custom preset",
                        },
                        "name": {
                            "type": "string",
                            "required": True,
                            "description": "Human-readable name",
                        },
                        "description": {
                            "type": "string",
                            "required": False,
                            "description": "Description of what this preset does",
                        },
                        "schedules": {
                            "type": "array",
                            "required": True,
                            "description": "List of schedule entries: [{name, skill_id, action, params, interval_seconds, description}]",
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="recommend",
                    description="Recommend presets based on installed skills and current gaps",
                    parameters={},
                    estimated_cost=0,
                ),
            ],
            required_credentials=[],
        )

    def check_credentials(self) -> bool:
        return True

    async def execute(self, action: str, params: Dict) -> SkillResult:
        handlers = {
            "list_presets": self._list_presets,
            "apply": self._apply,
            "apply_all": self._apply_all,
            "remove": self._remove,
            "remove_all": self._remove_all,
            "status": self._status,
            "create_custom": self._create_custom,
            "recommend": self._recommend,
        }

        handler = handlers.get(action)
        if not handler:
            return SkillResult(success=False, message=f"Unknown action: {action}")
        return await handler(params)

    # ── Actions ──────────────────────────────────────────────────────

    async def _list_presets(self, params: Dict) -> SkillResult:
        """List all available presets."""
        pillar_filter = params.get("pillar")

        presets = []
        all_presets = {**BUILTIN_PRESETS}
        # Add custom presets
        for pid, pdata in self._custom_presets.items():
            all_presets[pid] = self._dict_to_preset(pdata)

        for pid, preset in all_presets.items():
            if pillar_filter and preset.pillar != pillar_filter:
                continue
            applied = pid in self._applied
            schedules_info = []
            for s in preset.schedules:
                schedules_info.append({
                    "name": s.name,
                    "skill_id": s.skill_id,
                    "action": s.action,
                    "interval_seconds": s.interval_seconds,
                    "interval_human": self._humanize_interval(s.interval_seconds),
                    "description": s.description,
                })
            presets.append({
                "preset_id": pid,
                "name": preset.name,
                "description": preset.description,
                "pillar": preset.pillar,
                "schedule_count": len(preset.schedules),
                "schedules": schedules_info,
                "applied": applied,
                "is_custom": pid in self._custom_presets,
            })

        return SkillResult(
            success=True,
            message=f"{len(presets)} presets available ({sum(1 for p in presets if p['applied'])} applied)",
            data={"presets": presets, "total": len(presets)},
        )

    async def _apply(self, params: Dict) -> SkillResult:
        """Apply a single preset."""
        preset_id = params.get("preset_id", "").strip()
        multiplier = params.get("interval_multiplier", 1.0)

        if not preset_id:
            return SkillResult(success=False, message="preset_id is required")

        if preset_id in self._applied:
            return SkillResult(
                success=False,
                message=f"Preset '{preset_id}' is already applied. Remove it first to re-apply.",
            )

        preset = self._get_preset(preset_id)
        if not preset:
            available = list(BUILTIN_PRESETS.keys()) + list(self._custom_presets.keys())
            return SkillResult(
                success=False,
                message=f"Unknown preset: '{preset_id}'. Available: {available}",
            )

        return await self._apply_preset(preset, multiplier)

    async def _apply_all(self, params: Dict) -> SkillResult:
        """Apply all presets for full autonomy."""
        multiplier = params.get("interval_multiplier", 1.0)
        results = []
        applied_count = 0
        skipped = []

        for preset_id in FULL_AUTONOMY_PRESETS:
            if preset_id in self._applied:
                skipped.append(preset_id)
                continue
            preset = BUILTIN_PRESETS[preset_id]
            result = await self._apply_preset(preset, multiplier)
            results.append({"preset_id": preset_id, "success": result.success, "message": result.message})
            if result.success:
                applied_count += 1

        msg = f"Applied {applied_count}/{len(FULL_AUTONOMY_PRESETS)} presets for full autonomy"
        if skipped:
            msg += f" (skipped {len(skipped)} already applied)"

        return SkillResult(
            success=True,
            message=msg,
            data={"results": results, "applied": applied_count, "skipped": skipped},
        )

    async def _remove(self, params: Dict) -> SkillResult:
        """Remove an applied preset."""
        preset_id = params.get("preset_id", "").strip()
        if not preset_id:
            return SkillResult(success=False, message="preset_id is required")

        if preset_id not in self._applied:
            return SkillResult(
                success=False,
                message=f"Preset '{preset_id}' is not currently applied",
            )

        applied_info = self._applied[preset_id]
        task_ids = applied_info.get("task_ids", [])
        cancelled = 0

        # Cancel scheduler entries via context or direct file access
        for tid in task_ids:
            success = await self._cancel_scheduler_task(tid)
            if success:
                cancelled += 1

        del self._applied[preset_id]
        self._save_state()

        return SkillResult(
            success=True,
            message=f"Removed preset '{preset_id}' - cancelled {cancelled}/{len(task_ids)} scheduler entries",
            data={"preset_id": preset_id, "cancelled": cancelled, "total_tasks": len(task_ids)},
        )

    async def _remove_all(self, params: Dict) -> SkillResult:
        """Remove all applied presets."""
        removed = 0
        total_cancelled = 0

        for preset_id in list(self._applied.keys()):
            result = await self._remove({"preset_id": preset_id})
            if result.success:
                removed += 1
                total_cancelled += result.data.get("cancelled", 0)

        return SkillResult(
            success=True,
            message=f"Removed {removed} presets, cancelled {total_cancelled} scheduler entries",
            data={"removed": removed, "cancelled": total_cancelled},
        )

    async def _status(self, params: Dict) -> SkillResult:
        """Show status of all applied presets."""
        statuses = []
        for preset_id, info in self._applied.items():
            preset = self._get_preset(preset_id)
            statuses.append({
                "preset_id": preset_id,
                "name": preset.name if preset else preset_id,
                "pillar": preset.pillar if preset else "unknown",
                "applied_at": info.get("applied_at", "unknown"),
                "task_count": len(info.get("task_ids", [])),
                "task_ids": info.get("task_ids", []),
                "interval_multiplier": info.get("multiplier", 1.0),
            })

        total_tasks = sum(s["task_count"] for s in statuses)
        return SkillResult(
            success=True,
            message=f"{len(statuses)} presets applied ({total_tasks} scheduler entries total)",
            data={"applied_presets": statuses, "total_presets": len(statuses), "total_tasks": total_tasks},
        )

    async def _create_custom(self, params: Dict) -> SkillResult:
        """Create a custom preset."""
        preset_id = params.get("preset_id", "").strip()
        name = params.get("name", "").strip()
        description = params.get("description", "Custom preset")
        schedules_raw = params.get("schedules", [])

        if not preset_id:
            return SkillResult(success=False, message="preset_id is required")
        if not name:
            return SkillResult(success=False, message="name is required")
        if preset_id in BUILTIN_PRESETS:
            return SkillResult(success=False, message=f"Cannot use builtin preset ID: {preset_id}")
        if not schedules_raw or not isinstance(schedules_raw, list):
            return SkillResult(success=False, message="schedules must be a non-empty list")

        # Validate schedule entries
        schedules = []
        for i, s in enumerate(schedules_raw):
            if not isinstance(s, dict):
                return SkillResult(success=False, message=f"Schedule entry {i} must be a dict")
            if not s.get("skill_id") or not s.get("action"):
                return SkillResult(success=False, message=f"Schedule entry {i} requires skill_id and action")
            interval = s.get("interval_seconds", 3600)
            if interval < 10:
                return SkillResult(success=False, message=f"Schedule entry {i}: interval must be >= 10 seconds")
            schedules.append({
                "name": s.get("name", f"Custom Task {i+1}"),
                "skill_id": s["skill_id"],
                "action": s["action"],
                "params": s.get("params", {}),
                "interval_seconds": interval,
                "description": s.get("description", ""),
            })

        self._custom_presets[preset_id] = {
            "preset_id": preset_id,
            "name": name,
            "description": description,
            "pillar": "custom",
            "schedules": schedules,
        }
        self._save_state()

        return SkillResult(
            success=True,
            message=f"Created custom preset '{name}' with {len(schedules)} schedule entries",
            data={"preset_id": preset_id, "schedule_count": len(schedules)},
        )

    async def _recommend(self, params: Dict) -> SkillResult:
        """Recommend presets based on what's installed and what's not applied."""
        recommendations = []

        # Check which skills are available
        available_skills = set()
        if self.context:
            available_skills = set(self.context.list_skills())

        for preset_id, preset in BUILTIN_PRESETS.items():
            if preset_id in self._applied:
                continue  # Already applied

            # Check if required skills are installed
            required_skills = set(s.skill_id for s in preset.schedules)
            missing = required_skills - available_skills
            installable = len(missing) == 0 or not available_skills  # if no context, assume all available

            recommendations.append({
                "preset_id": preset_id,
                "name": preset.name,
                "description": preset.description,
                "pillar": preset.pillar,
                "schedule_count": len(preset.schedules),
                "installable": installable,
                "missing_skills": list(missing) if missing and available_skills else [],
                "priority": self._preset_priority(preset_id),
            })

        # Sort by priority (lower = higher priority)
        recommendations.sort(key=lambda r: r["priority"])

        return SkillResult(
            success=True,
            message=f"{len(recommendations)} presets recommended ({sum(1 for r in recommendations if r['installable'])} installable now)",
            data={"recommendations": recommendations},
        )

    # ── Helpers ───────────────────────────────────────────────────────

    async def _apply_preset(self, preset: PresetDefinition, multiplier: float = 1.0) -> SkillResult:
        """Apply a preset by creating scheduler entries."""
        task_ids = []
        errors = []

        for schedule in preset.schedules:
            interval = max(10, schedule.interval_seconds * multiplier)
            task_id = await self._create_scheduler_entry(
                name=f"[{preset.name}] {schedule.name}",
                skill_id=schedule.skill_id,
                action=schedule.action,
                params=schedule.params,
                interval_seconds=interval,
            )
            if task_id:
                task_ids.append(task_id)
            else:
                errors.append(f"Failed to schedule: {schedule.name}")

        self._applied[preset.preset_id] = {
            "task_ids": task_ids,
            "applied_at": datetime.now().isoformat(),
            "multiplier": multiplier,
            "schedule_count": len(preset.schedules),
        }
        self._save_state()

        if errors:
            return SkillResult(
                success=True,
                message=f"Applied '{preset.name}' with {len(task_ids)}/{len(preset.schedules)} entries ({len(errors)} errors)",
                data={"preset_id": preset.preset_id, "task_ids": task_ids, "errors": errors},
            )

        return SkillResult(
            success=True,
            message=f"Applied '{preset.name}' - {len(task_ids)} recurring tasks scheduled",
            data={"preset_id": preset.preset_id, "task_ids": task_ids},
        )

    async def _create_scheduler_entry(self, name: str, skill_id: str, action: str,
                                        params: Dict, interval_seconds: float) -> Optional[str]:
        """Create a scheduler entry via SkillContext or direct file manipulation."""
        # Try via SkillContext first
        if self.context:
            try:
                result = await self.context.call_skill("scheduler", "schedule", {
                    "name": name,
                    "skill_id": skill_id,
                    "action": action,
                    "params": params,
                    "recurring": True,
                    "interval_seconds": interval_seconds,
                    "delay_seconds": 0,
                })
                if result.success and result.data:
                    return result.data.get("id")
            except Exception:
                pass

        # Fallback: direct file-based scheduler entry
        return self._create_scheduler_entry_direct(name, skill_id, action, params, interval_seconds)

    def _create_scheduler_entry_direct(self, name: str, skill_id: str, action: str,
                                         params: Dict, interval_seconds: float) -> Optional[str]:
        """Create scheduler entry by directly writing to scheduler.json."""
        try:
            scheduler_file = DATA_DIR / "scheduler.json"
            if scheduler_file.exists():
                data = json.loads(scheduler_file.read_text())
            else:
                data = {"tasks": {}}

            task_id = f"sched_{uuid.uuid4().hex[:8]}"
            now = time.time()

            data["tasks"][task_id] = {
                "id": task_id,
                "name": name,
                "skill_id": skill_id,
                "action": action,
                "params": params,
                "schedule_type": "recurring",
                "interval_seconds": interval_seconds,
                "created_at": datetime.now().isoformat(),
                "next_run_at": now + interval_seconds,
                "status": "pending",
                "run_count": 0,
                "max_runs": 0,
                "last_run_at": None,
                "last_result": None,
                "last_success": None,
                "enabled": True,
            }
            data["saved_at"] = datetime.now().isoformat()

            DATA_DIR.mkdir(parents=True, exist_ok=True)
            scheduler_file.write_text(json.dumps(data, indent=2))
            return task_id
        except Exception:
            return None

    async def _cancel_scheduler_task(self, task_id: str) -> bool:
        """Cancel a scheduler task."""
        # Try via SkillContext
        if self.context:
            try:
                result = await self.context.call_skill("scheduler", "cancel", {"task_id": task_id})
                return result.success
            except Exception:
                pass

        # Fallback: direct file
        try:
            scheduler_file = DATA_DIR / "scheduler.json"
            if not scheduler_file.exists():
                return False
            data = json.loads(scheduler_file.read_text())
            if task_id in data.get("tasks", {}):
                data["tasks"][task_id]["status"] = "cancelled"
                data["tasks"][task_id]["enabled"] = False
                scheduler_file.write_text(json.dumps(data, indent=2))
                return True
            return False
        except Exception:
            return False

    def _get_preset(self, preset_id: str) -> Optional[PresetDefinition]:
        """Get a preset by ID (builtin or custom)."""
        if preset_id in BUILTIN_PRESETS:
            return BUILTIN_PRESETS[preset_id]
        if preset_id in self._custom_presets:
            return self._dict_to_preset(self._custom_presets[preset_id])
        return None

    def _dict_to_preset(self, d: Dict) -> PresetDefinition:
        """Convert a dict to a PresetDefinition."""
        schedules = []
        for s in d.get("schedules", []):
            schedules.append(PresetSchedule(
                name=s.get("name", "Unknown"),
                skill_id=s.get("skill_id", ""),
                action=s.get("action", ""),
                params=s.get("params", {}),
                interval_seconds=s.get("interval_seconds", 3600),
                description=s.get("description", ""),
            ))
        return PresetDefinition(
            preset_id=d.get("preset_id", "unknown"),
            name=d.get("name", "Unknown"),
            description=d.get("description", ""),
            pillar=d.get("pillar", "custom"),
            schedules=schedules,
        )

    def _preset_priority(self, preset_id: str) -> int:
        """Priority ranking for recommendations (lower = higher priority)."""
        priority_order = [
            "health_monitoring",   # 1 - foundation
            "alert_polling",       # 2 - automated response
            "self_tuning",         # 3 - self-optimization
            "feedback_loop",       # 4 - learning
            "self_assessment",     # 5 - self-awareness
            "reputation_polling",  # 6 - multi-agent
            "revenue_reporting",   # 7 - business
            "knowledge_sync",     # 8 - sharing
        ]
        try:
            return priority_order.index(preset_id)
        except ValueError:
            return 99

    @staticmethod
    def _humanize_interval(seconds: float) -> str:
        """Convert seconds to human-readable interval."""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            return f"{seconds / 60:.0f}m"
        elif seconds < 86400:
            hours = seconds / 3600
            return f"{hours:.1f}h" if hours != int(hours) else f"{int(hours)}h"
        else:
            return f"{seconds / 86400:.1f}d"

    def _save_state(self):
        """Persist applied presets and custom presets to disk."""
        try:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            data = {
                "applied": self._applied,
                "custom_presets": self._custom_presets,
                "saved_at": datetime.now().isoformat(),
            }
            PRESETS_FILE.write_text(json.dumps(data, indent=2))
        except Exception:
            pass

    def _load_state(self):
        """Load state from disk."""
        try:
            if PRESETS_FILE.exists():
                data = json.loads(PRESETS_FILE.read_text())
                self._applied = data.get("applied", {})
                self._custom_presets = data.get("custom_presets", {})
        except Exception:
            self._applied = {}
            self._custom_presets = {}
