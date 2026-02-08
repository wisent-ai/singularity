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
- Adaptive thresholds: auto-tune circuit breaker thresholds per skill
- Revenue goals: auto-set/track/adjust revenue goals from forecast data
- Experiment management: auto-conclude experiments and review learnings
- Circuit sharing monitor: monitor cross-agent circuit states and fleet alerts
- Goal stall monitoring: periodic stall detection with automated alerts for stuck goals
- Revenue goal evaluation: periodic revenue goal status checks, reports, and history reviews
- Dashboard auto-check: periodic loop iteration dashboard health checks and alert scanning
- Fleet health auto-heal: periodic fleet health monitoring with automatic heal triggers
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
    depends_on: List[str] = field(default_factory=list)  # preset IDs this depends on


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
        depends_on=["health_monitoring"],
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
        depends_on=["self_assessment"],
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
    "adaptive_thresholds": PresetDefinition(
        preset_id="adaptive_thresholds",
        name="Adaptive Circuit Thresholds",
        description="Auto-tune circuit breaker thresholds per skill based on historical performance data",
        pillar="self_improvement",
        depends_on=["health_monitoring"],
        schedules=[
            PresetSchedule(
                name="Tune All Circuit Thresholds",
                skill_id="adaptive_circuit_thresholds",
                action="tune_all",
                params={},
                interval_seconds=1800,  # every 30 min
                description="Analyze all skill performance and update per-skill circuit breaker thresholds",
            ),
            PresetSchedule(
                name="Circuit Threshold Profiles",
                skill_id="adaptive_circuit_thresholds",
                action="profiles",
                params={},
                interval_seconds=7200,  # every 2 hours
                description="Review all skill performance profiles and current threshold overrides",
            ),
        ],
    ),
    "revenue_goals": PresetDefinition(
        preset_id="revenue_goals",
        name="Revenue Goal Management",
        description="Auto-set and track revenue goals from forecast data, adjust when conditions change",
        pillar="revenue",
        schedules=[
            PresetSchedule(
                name="Revenue Assessment",
                skill_id="revenue_goal_auto_setter",
                action="assess",
                params={},
                interval_seconds=3600,  # every hour
                description="Assess revenue state and generate goal recommendations from forecasts",
            ),
            PresetSchedule(
                name="Revenue Goal Tracking",
                skill_id="revenue_goal_auto_setter",
                action="track",
                params={},
                interval_seconds=1800,  # every 30 min
                description="Track progress of active revenue goals against actual data",
            ),
            PresetSchedule(
                name="Revenue Goal Adjustment",
                skill_id="revenue_goal_auto_setter",
                action="adjust",
                params={},
                interval_seconds=7200,  # every 2 hours
                description="Check for revenue condition changes and auto-adjust goals",
            ),
        ],
    ),
    "experiment_management": PresetDefinition(
        preset_id="experiment_management",
        name="Experiment Management",
        description="Auto-conclude experiments and review learnings for continuous self-improvement",
        pillar="self_improvement",
        depends_on=["feedback_loop"],
        schedules=[
            PresetSchedule(
                name="Conclude Experiments",
                skill_id="experiment",
                action="conclude_all",
                params={},
                interval_seconds=3600,  # every hour
                description="Evaluate all running experiments and conclude those with statistical significance",
            ),
            PresetSchedule(
                name="Review Experiment Learnings",
                skill_id="experiment",
                action="learnings",
                params={},
                interval_seconds=14400,  # every 4 hours
                description="Compile and review accumulated learnings from concluded experiments",
            ),
        ],
    ),
    "circuit_sharing_monitor": PresetDefinition(
        preset_id="circuit_sharing_monitor",
        name="Circuit Sharing Monitor",
        description="Monitor cross-agent circuit sharing state and emit fleet-wide alerts",
        pillar="replication",
        schedules=[
            PresetSchedule(
                name="Circuit Sharing Monitor",
                skill_id="circuit_sharing_events",
                action="monitor",
                params={},
                interval_seconds=300,  # every 5 min
                description="Check circuit sharing state for changes and emit events",
            ),
            PresetSchedule(
                name="Fleet Circuit Health Check",
                skill_id="circuit_sharing_events",
                action="fleet_check",
                params={},
                interval_seconds=600,  # every 10 min
                description="Analyze shared circuit states fleet-wide and alert if majority are open",
            ),
        ],
    ),
    "goal_stall_monitoring": PresetDefinition(
        preset_id="goal_stall_monitoring",
        name="Goal Stall Monitoring",
        description="Periodic stall detection for active goals - emit alerts when goals haven't progressed",
        pillar="goal_setting",
        schedules=[
            PresetSchedule(
                name="Goal Stall Check",
                skill_id="goal_progress_events",
                action="stall_check",
                params={},
                interval_seconds=14400,  # every 4 hours
                description="Detect goals that haven't progressed within threshold and emit stall events",
            ),
            PresetSchedule(
                name="Goal Progress Monitor",
                skill_id="goal_progress_events",
                action="monitor",
                params={},
                interval_seconds=1800,  # every 30 min
                description="Monitor active goals for status changes and emit progress events",
            ),
        ],
    ),
    "revenue_goal_evaluation": PresetDefinition(
        preset_id="revenue_goal_evaluation",
        name="Revenue Goal Evaluation",
        description="Periodic revenue goal evaluation - assess performance, track progress, auto-adjust targets",
        pillar="revenue",
        depends_on=["revenue_goals"],
        schedules=[
            PresetSchedule(
                name="Revenue Goal Status",
                skill_id="revenue_goal_auto_setter",
                action="status",
                params={},
                interval_seconds=1800,  # every 30 min
                description="Check current status of all active revenue goals",
            ),
            PresetSchedule(
                name="Revenue Goal Report",
                skill_id="revenue_goal_auto_setter",
                action="report",
                params={},
                interval_seconds=7200,  # every 2 hours
                description="Generate revenue goal performance report with trends",
            ),
            PresetSchedule(
                name="Revenue History Review",
                skill_id="revenue_goal_auto_setter",
                action="history",
                params={},
                interval_seconds=43200,  # every 12 hours
                description="Review revenue goal history for pattern analysis",
            ),
        ],
    ),
    "dashboard_auto_check": PresetDefinition(
        preset_id="dashboard_auto_check",
        name="Dashboard Auto-Check",
        description="Periodic loop iteration dashboard checks - detect degraded health and emit alerts",
        pillar="operations",
        depends_on=["health_monitoring"],
        schedules=[
            PresetSchedule(
                name="Dashboard Latest Check",
                skill_id="loop_iteration_dashboard",
                action="latest",
                params={},
                interval_seconds=600,  # every 10 min
                description="Fetch latest iteration dashboard to detect issues early",
            ),
            PresetSchedule(
                name="Dashboard Trend Analysis",
                skill_id="loop_iteration_dashboard",
                action="trends",
                params={},
                interval_seconds=3600,  # every hour
                description="Analyze iteration trends for success rate and performance degradation",
            ),
            PresetSchedule(
                name="Subsystem Health Check",
                skill_id="loop_iteration_dashboard",
                action="subsystem_health",
                params={},
                interval_seconds=1800,  # every 30 min
                description="Score each subsystem's health and detect weak components",
            ),
            PresetSchedule(
                name="Dashboard Alert Scan",
                skill_id="loop_iteration_dashboard",
                action="alerts",
                params={},
                interval_seconds=900,  # every 15 min
                description="Scan for degradation patterns - low success rate, slow iterations, failure streaks",
            ),
        ],
    ),
    "fleet_health_auto_heal": PresetDefinition(
        preset_id="fleet_health_auto_heal",
        name="Fleet Health Auto-Heal",
        description="Periodic fleet health monitoring with automatic heal triggers and fleet-wide checks",
        pillar="replication",
        depends_on=["circuit_sharing_monitor"],
        schedules=[
            PresetSchedule(
                name="Fleet Health Monitor",
                skill_id="fleet_health_events",
                action="monitor",
                params={},
                interval_seconds=300,  # every 5 min
                description="Monitor fleet for health changes and emit events for heal/scale/replace actions",
            ),
            PresetSchedule(
                name="Fleet Health Check",
                skill_id="fleet_health_events",
                action="fleet_check",
                params={},
                interval_seconds=600,  # every 10 min
                description="Proactive fleet-wide health check - detect critical conditions like capacity drops",
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
        # Health alert state
        self._alert_config = {
            "failure_streak_threshold": 3,      # Alert after N consecutive failures
            "success_rate_threshold": 50.0,     # Alert when success rate drops below %
            "recovery_streak_threshold": 2,     # Recover after N consecutive successes
            "emit_on_task_fail": True,          # Emit event on individual task failure
            "emit_on_preset_unhealthy": True,   # Emit event when preset becomes unhealthy
            "emit_on_preset_recovered": True,   # Emit event when preset recovers
            "event_source": "scheduler_presets_health",
        }
        self._alert_state: Dict[str, Dict] = {}  # task_id -> {streak, status, ...}
        self._alert_history: List[Dict] = []      # Recent alert events
        self._max_alert_history = 200
        self._load_state()

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="scheduler_presets",
            name="Scheduler Presets",
            version="3.0.0",
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
                SkillAction(
                    name="dashboard",
                    description="Rich operational dashboard: per-preset health, next run times, success rates, overdue tasks",
                    parameters={
                        "preset_id": {
                            "type": "string",
                            "required": False,
                            "description": "Show details for a specific preset only. Omit for all presets.",
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="dependency_graph",
                    description="Show the preset dependency graph with topological apply order and cycle detection",
                    parameters={
                        "preset_id": {
                            "type": "string",
                            "required": False,
                            "description": "Show dependencies for a specific preset. Omit for full graph.",
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="apply_with_deps",
                    description="Apply a preset and all its dependencies in topological order",
                    parameters={
                        "preset_id": {
                            "type": "string",
                            "required": True,
                            "description": "ID of the preset to apply (dependencies applied first)",
                        },
                        "interval_multiplier": {
                            "type": "number",
                            "required": False,
                            "description": "Multiply all intervals by this factor. Default 1.0",
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="health_alerts",
                    description="Scan applied presets for failures and emit EventBus alerts for unhealthy presets",
                    parameters={
                        "preset_id": {
                            "type": "string",
                            "required": False,
                            "description": "Check a specific preset only. Omit for all applied presets.",
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="configure_alerts",
                    description="Configure health alert thresholds and event emission settings",
                    parameters={
                        "failure_streak_threshold": {
                            "type": "integer",
                            "required": False,
                            "description": "Number of consecutive failures before alerting (default 3)",
                        },
                        "success_rate_threshold": {
                            "type": "number",
                            "required": False,
                            "description": "Success rate percentage below which a preset is unhealthy (default 50)",
                        },
                        "recovery_streak_threshold": {
                            "type": "integer",
                            "required": False,
                            "description": "Number of consecutive successes before marking recovered (default 2)",
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="alert_history",
                    description="View recent health alert events emitted for preset failures and recoveries",
                    parameters={
                        "limit": {
                            "type": "integer",
                            "required": False,
                            "description": "Max number of events to return (default 50)",
                        },
                        "preset_id": {
                            "type": "string",
                            "required": False,
                            "description": "Filter by preset ID",
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
            "list_presets": self._list_presets,
            "apply": self._apply,
            "apply_all": self._apply_all,
            "remove": self._remove,
            "remove_all": self._remove_all,
            "status": self._status,
            "create_custom": self._create_custom,
            "recommend": self._recommend,
            "dashboard": self._dashboard,
            "dependency_graph": self._dependency_graph,
            "apply_with_deps": self._apply_with_deps,
            "health_alerts": self._health_alerts,
            "configure_alerts": self._configure_alerts,
            "alert_history": self._get_alert_history,
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
        """Apply all presets for full autonomy, in dependency-sorted order."""
        multiplier = params.get("interval_multiplier", 1.0)
        results = []
        applied_count = 0
        skipped = []

        # Apply in topological order so dependencies are satisfied first
        ordered = self._topological_sort(FULL_AUTONOMY_PRESETS)

        for preset_id in ordered:
            if preset_id in self._applied:
                skipped.append(preset_id)
                continue
            preset = BUILTIN_PRESETS.get(preset_id)
            if not preset:
                continue
            result = await self._apply_preset(preset, multiplier)
            results.append({"preset_id": preset_id, "success": result.success, "message": result.message})
            if result.success:
                applied_count += 1

        msg = f"Applied {applied_count}/{len(FULL_AUTONOMY_PRESETS)} presets for full autonomy (dependency-ordered)"
        if skipped:
            msg += f" (skipped {len(skipped)} already applied)"

        return SkillResult(
            success=True,
            message=msg,
            data={"results": results, "applied": applied_count, "skipped": skipped, "apply_order": ordered},
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

    # ── Dependency Graph ──────────────────────────────────────────────

    async def _dependency_graph(self, params: Dict) -> SkillResult:
        """Show the preset dependency graph with topological order and cycle detection."""
        preset_id = params.get("preset_id")
        all_presets = {**BUILTIN_PRESETS}
        for pid, pdata in self._custom_presets.items():
            all_presets[pid] = self._dict_to_preset(pdata)

        # Build adjacency info
        graph = {}
        for pid, preset in all_presets.items():
            deps = getattr(preset, "depends_on", []) or []
            graph[pid] = {
                "name": preset.name,
                "pillar": preset.pillar,
                "depends_on": deps,
                "depended_by": [],
                "applied": pid in self._applied,
            }

        # Compute reverse edges (who depends on me)
        for pid, info in graph.items():
            for dep in info["depends_on"]:
                if dep in graph:
                    graph[dep]["depended_by"].append(pid)

        # Detect cycles
        cycles = self._detect_cycles(graph)

        # Compute topological order
        all_ids = list(all_presets.keys())
        topo_order = self._topological_sort(all_ids)

        # Compute depth (longest path from root)
        depths = {}
        for pid in topo_order:
            deps = graph[pid]["depends_on"]
            if not deps:
                depths[pid] = 0
            else:
                depths[pid] = max((depths.get(d, 0) for d in deps if d in depths), default=0) + 1
            graph[pid]["depth"] = depths[pid]

        # If specific preset requested, show its transitive deps
        if preset_id:
            if preset_id not in graph:
                return SkillResult(
                    success=False,
                    message=f"Unknown preset: '{preset_id}'",
                )
            transitive = self._transitive_deps(preset_id, all_presets)
            filtered_order = [p for p in topo_order if p in transitive or p == preset_id]
            return SkillResult(
                success=True,
                message=f"Dependency graph for '{preset_id}': {len(transitive)} dependencies",
                data={
                    "preset_id": preset_id,
                    "direct_deps": graph[preset_id]["depends_on"],
                    "transitive_deps": list(transitive),
                    "apply_order": filtered_order,
                    "depended_by": graph[preset_id]["depended_by"],
                    "depth": depths.get(preset_id, 0),
                },
            )

        # Full graph
        roots = [pid for pid, info in graph.items() if not info["depends_on"]]
        leaves = [pid for pid, info in graph.items() if not info["depended_by"]]

        return SkillResult(
            success=True,
            message=f"Dependency graph: {len(graph)} presets, {sum(len(v['depends_on']) for v in graph.values())} edges, {len(cycles)} cycles",
            data={
                "graph": graph,
                "topological_order": topo_order,
                "roots": roots,
                "leaves": leaves,
                "cycles": cycles,
                "max_depth": max(depths.values()) if depths else 0,
            },
        )

    async def _apply_with_deps(self, params: Dict) -> SkillResult:
        """Apply a preset and all its transitive dependencies in topological order."""
        preset_id = params.get("preset_id", "").strip()
        multiplier = params.get("interval_multiplier", 1.0)

        if not preset_id:
            return SkillResult(success=False, message="preset_id is required")

        all_presets = {**BUILTIN_PRESETS}
        for pid, pdata in self._custom_presets.items():
            all_presets[pid] = self._dict_to_preset(pdata)

        if preset_id not in all_presets:
            return SkillResult(success=False, message=f"Unknown preset: '{preset_id}'")

        # Get all transitive deps + self
        transitive = self._transitive_deps(preset_id, all_presets)
        all_needed = list(transitive) + [preset_id]

        # Check for cycles
        cycles = self._detect_cycles({
            pid: {"depends_on": getattr(all_presets.get(pid, PresetDefinition("", "", "", "", [])), "depends_on", []) or []}
            for pid in all_needed
        })
        if cycles:
            return SkillResult(
                success=False,
                message=f"Circular dependency detected: {cycles}",
                data={"cycles": cycles},
            )

        # Sort in dependency order
        ordered = self._topological_sort(all_needed)

        results = []
        applied_count = 0
        skipped = []

        for pid in ordered:
            if pid in self._applied:
                skipped.append(pid)
                continue
            preset = all_presets.get(pid)
            if not preset:
                continue
            result = await self._apply_preset(preset, multiplier)
            results.append({"preset_id": pid, "success": result.success, "message": result.message})
            if result.success:
                applied_count += 1

        msg = f"Applied '{preset_id}' with {len(transitive)} dependencies ({applied_count} new)"
        if skipped:
            msg += f" (skipped {len(skipped)} already applied)"

        return SkillResult(
            success=True,
            message=msg,
            data={
                "preset_id": preset_id,
                "apply_order": ordered,
                "results": results,
                "applied": applied_count,
                "skipped": skipped,
            },
        )

    def _topological_sort(self, preset_ids: List[str]) -> List[str]:
        """Topological sort of preset IDs based on their dependencies.

        Returns presets in order such that dependencies come before dependents.
        Uses Kahn's algorithm. Presets not in preset_ids are ignored.
        """
        all_presets = {**BUILTIN_PRESETS}
        for pid, pdata in self._custom_presets.items():
            all_presets[pid] = self._dict_to_preset(pdata)

        id_set = set(preset_ids)

        # Build in-degree map
        in_degree = {pid: 0 for pid in preset_ids}
        adj = {pid: [] for pid in preset_ids}

        for pid in preset_ids:
            preset = all_presets.get(pid)
            if not preset:
                continue
            deps = getattr(preset, "depends_on", []) or []
            for dep in deps:
                if dep in id_set:
                    in_degree[pid] = in_degree.get(pid, 0) + 1
                    adj.setdefault(dep, []).append(pid)

        # Kahn's algorithm
        queue = [pid for pid in preset_ids if in_degree.get(pid, 0) == 0]
        queue.sort()  # Deterministic ordering among same-depth presets
        result = []

        while queue:
            node = queue.pop(0)
            result.append(node)
            for neighbor in sorted(adj.get(node, [])):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
            queue.sort()

        # If not all nodes in result, there's a cycle - append remaining
        remaining = [pid for pid in preset_ids if pid not in set(result)]
        result.extend(sorted(remaining))

        return result

    def _transitive_deps(self, preset_id: str, all_presets: Dict) -> set:
        """Get all transitive dependencies of a preset (not including itself)."""
        visited = set()
        stack = [preset_id]

        while stack:
            current = stack.pop()
            preset = all_presets.get(current)
            if not preset:
                continue
            deps = getattr(preset, "depends_on", []) or []
            for dep in deps:
                if dep not in visited and dep != preset_id:
                    visited.add(dep)
                    stack.append(dep)

        return visited

    def _detect_cycles(self, graph: Dict) -> List[List[str]]:
        """Detect cycles in the dependency graph using DFS."""
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {pid: WHITE for pid in graph}
        cycles = []

        def dfs(node, path):
            color[node] = GRAY
            path.append(node)
            deps = graph[node].get("depends_on", [])
            for dep in deps:
                if dep not in color:
                    continue
                if color[dep] == GRAY:
                    # Found cycle
                    cycle_start = path.index(dep)
                    cycles.append(path[cycle_start:] + [dep])
                elif color[dep] == WHITE:
                    dfs(dep, path)
            path.pop()
            color[node] = BLACK

        for node in graph:
            if color[node] == WHITE:
                dfs(node, [])

        return cycles

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


    async def _dashboard(self, params: Dict) -> SkillResult:
        """Rich operational dashboard with per-preset health, next runs, success rates."""
        filter_preset = params.get("preset_id", "").strip() or None

        # Read scheduler state for task details and execution history
        scheduler_tasks, execution_history = self._read_scheduler_data()

        now = time.time()
        preset_reports = []
        total_tasks = 0
        total_healthy = 0
        total_overdue = 0
        total_disabled = 0
        total_executions = 0
        total_successes = 0

        applied_presets = dict(self._applied)
        if filter_preset:
            if filter_preset not in applied_presets:
                return SkillResult(
                    success=False,
                    message=f"Preset '{filter_preset}' is not applied",
                    data={"available": list(applied_presets.keys())},
                )
            applied_presets = {filter_preset: applied_presets[filter_preset]}

        for preset_id, info in applied_presets.items():
            preset = self._get_preset(preset_id)
            task_ids = info.get("task_ids", [])
            applied_at = info.get("applied_at", "unknown")
            multiplier = info.get("multiplier", 1.0)

            # Gather per-task details from scheduler
            task_details = []
            preset_executions = 0
            preset_successes = 0
            preset_overdue = 0
            preset_disabled = 0

            for tid in task_ids:
                task_data = scheduler_tasks.get(tid)
                if not task_data:
                    task_details.append({
                        "task_id": tid,
                        "status": "missing",
                        "name": "unknown",
                        "health": "missing",
                    })
                    continue

                task_name = task_data.get("name", "unknown")
                enabled = task_data.get("enabled", True)
                status = task_data.get("status", "unknown")
                next_run = task_data.get("next_run_at", 0)
                run_count = task_data.get("run_count", 0)
                last_run_at = task_data.get("last_run_at")
                last_success = task_data.get("last_success")
                interval = task_data.get("interval_seconds", 0)

                # Compute next run info
                if enabled and status == "pending" and next_run > 0:
                    remaining = next_run - now
                    if remaining > 0:
                        next_run_human = self._humanize_interval(remaining)
                        is_overdue = False
                    else:
                        next_run_human = f"overdue by {self._humanize_interval(abs(remaining))}"
                        is_overdue = True
                        preset_overdue += 1
                else:
                    next_run_human = "n/a"
                    is_overdue = False

                if not enabled:
                    preset_disabled += 1

                # Get execution history for this task
                task_history = [h for h in execution_history if h.get("task_id") == tid]
                task_exec_count = len(task_history)
                task_success_count = sum(1 for h in task_history if h.get("success"))
                success_rate = (task_success_count / task_exec_count * 100) if task_exec_count > 0 else None

                # Compute average duration
                durations = [h.get("duration_seconds", 0) for h in task_history if h.get("duration_seconds")]
                avg_duration = sum(durations) / len(durations) if durations else None

                preset_executions += task_exec_count
                preset_successes += task_success_count

                # Health assessment
                health = "healthy"
                if not enabled:
                    health = "disabled"
                elif status == "cancelled":
                    health = "cancelled"
                elif is_overdue:
                    health = "overdue"
                elif last_success is False:
                    health = "last_failed"
                elif success_rate is not None and success_rate < 50:
                    health = "degraded"

                detail = {
                    "task_id": tid,
                    "name": task_name,
                    "health": health,
                    "enabled": enabled,
                    "status": status,
                    "next_run_in": next_run_human,
                    "interval": self._humanize_interval(interval) if interval else "n/a",
                    "run_count": run_count,
                    "last_run_at": last_run_at,
                    "last_success": last_success,
                    "executions_in_history": task_exec_count,
                    "success_rate": f"{success_rate:.0f}%" if success_rate is not None else "n/a",
                    "avg_duration_seconds": round(avg_duration, 3) if avg_duration is not None else None,
                }
                task_details.append(detail)

            # Preset-level health
            task_count = len(task_ids)
            healthy_count = sum(1 for t in task_details if t["health"] == "healthy")
            preset_success_rate = (preset_successes / preset_executions * 100) if preset_executions > 0 else None

            if task_count == 0:
                preset_health = "empty"
            elif healthy_count == task_count:
                preset_health = "healthy"
            elif healthy_count == 0:
                preset_health = "unhealthy"
            else:
                preset_health = "degraded"

            total_tasks += task_count
            total_healthy += healthy_count
            total_overdue += preset_overdue
            total_disabled += preset_disabled
            total_executions += preset_executions
            total_successes += preset_successes

            preset_reports.append({
                "preset_id": preset_id,
                "name": preset.name if preset else preset_id,
                "pillar": preset.pillar if preset else "unknown",
                "health": preset_health,
                "applied_at": applied_at,
                "interval_multiplier": multiplier,
                "task_count": task_count,
                "healthy_tasks": healthy_count,
                "overdue_tasks": preset_overdue,
                "disabled_tasks": preset_disabled,
                "total_executions": preset_executions,
                "success_rate": f"{preset_success_rate:.0f}%" if preset_success_rate is not None else "n/a",
                "tasks": task_details,
            })

        # Overall summary
        overall_success_rate = (total_successes / total_executions * 100) if total_executions > 0 else None
        if total_tasks == 0:
            overall_health = "no_presets"
        elif total_healthy == total_tasks:
            overall_health = "all_healthy"
        elif total_healthy > total_tasks * 0.5:
            overall_health = "mostly_healthy"
        elif total_healthy > 0:
            overall_health = "degraded"
        else:
            overall_health = "unhealthy"

        summary = {
            "overall_health": overall_health,
            "presets_applied": len(preset_reports),
            "total_tasks": total_tasks,
            "healthy_tasks": total_healthy,
            "overdue_tasks": total_overdue,
            "disabled_tasks": total_disabled,
            "total_executions": total_executions,
            "overall_success_rate": f"{overall_success_rate:.0f}%" if overall_success_rate is not None else "n/a",
        }

        health_icon = {
            "all_healthy": "OK",
            "mostly_healthy": "WARN",
            "degraded": "DEGRADED",
            "unhealthy": "CRITICAL",
            "no_presets": "NONE",
        }.get(overall_health, "UNKNOWN")

        msg = (
            f"[{health_icon}] {len(preset_reports)} presets, "
            f"{total_tasks} tasks ({total_healthy} healthy, {total_overdue} overdue). "
            f"Executions: {total_executions} "
            f"(success rate: {summary['overall_success_rate']})"
        )

        return SkillResult(
            success=True,
            message=msg,
            data={
                "summary": summary,
                "presets": preset_reports,
            },
        )

    def _read_scheduler_data(self):
        """Read scheduler tasks and execution history from scheduler.json and skill context."""
        scheduler_tasks = {}
        execution_history = []

        # Try via skill context first
        if self.context:
            try:
                import asyncio
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # We're in an async context, can't await synchronously
                    pass
                else:
                    result = loop.run_until_complete(
                        self.context.call_skill("scheduler", "list", {"include_completed": True})
                    )
                    if result.success and result.data:
                        for t in result.data.get("tasks", []):
                            scheduler_tasks[t["id"]] = t
            except Exception:
                pass

        # Fallback / supplement: direct file read
        if not scheduler_tasks:
            try:
                scheduler_file = DATA_DIR / "scheduler.json"
                if scheduler_file.exists():
                    data = json.loads(scheduler_file.read_text())
                    scheduler_tasks = data.get("tasks", {})
            except Exception:
                pass

        # Read execution history from scheduler
        if self.context:
            try:
                import asyncio
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    pass
                else:
                    result = loop.run_until_complete(
                        self.context.call_skill("scheduler", "history", {"limit": 500})
                    )
                    if result.success and result.data:
                        execution_history = result.data.get("history", [])
            except Exception:
                pass

        # Fallback: read history from scheduler data file
        if not execution_history:
            try:
                history_file = DATA_DIR / "scheduler_history.json"
                if history_file.exists():
                    execution_history = json.loads(history_file.read_text())
            except Exception:
                pass

        return scheduler_tasks, execution_history

    # ── Health Alert Methods ──────────────────────────────────────────

    async def _health_alerts(self, params: Dict) -> SkillResult:
        """Scan applied presets for failures and emit EventBus alerts."""
        filter_preset = params.get("preset_id", "").strip() or None

        scheduler_tasks, execution_history = self._read_scheduler_data()
        now = time.time()

        applied_presets = dict(self._applied)
        if filter_preset:
            if filter_preset not in applied_presets:
                return SkillResult(
                    success=False,
                    message=f"Preset '{filter_preset}' is not applied",
                    data={"available": list(applied_presets.keys())},
                )
            applied_presets = {filter_preset: applied_presets[filter_preset]}

        alerts_emitted = []
        recoveries_emitted = []
        preset_health_summary = []

        for preset_id, info in applied_presets.items():
            preset = self._get_preset(preset_id)
            task_ids = info.get("task_ids", [])
            preset_failures = 0
            preset_total = 0
            preset_unhealthy_tasks = []

            for tid in task_ids:
                task_data = scheduler_tasks.get(tid, {})
                task_name = task_data.get("name", tid)
                last_success = task_data.get("last_success")

                # Compute recent execution history for this task
                task_history = [h for h in execution_history if h.get("task_id") == tid]
                task_exec_count = len(task_history)
                task_success_count = sum(1 for h in task_history if h.get("success"))

                # Initialize alert state for this task if not present
                if tid not in self._alert_state:
                    self._alert_state[tid] = {
                        "failure_streak": 0,
                        "success_streak": 0,
                        "status": "healthy",  # healthy, alerting, unknown
                        "preset_id": preset_id,
                        "task_name": task_name,
                        "last_checked": None,
                        "total_alerts_emitted": 0,
                    }

                state = self._alert_state[tid]
                state["last_checked"] = datetime.now().isoformat()
                state["preset_id"] = preset_id
                state["task_name"] = task_name

                # Determine current task health from latest executions
                # Look at the most recent N executions to determine streak
                recent = sorted(task_history, key=lambda h: h.get("executed_at", ""), reverse=True)
                streak_threshold = self._alert_config["failure_streak_threshold"]
                recovery_threshold = self._alert_config["recovery_streak_threshold"]

                # Count consecutive failures from most recent
                consecutive_failures = 0
                consecutive_successes = 0
                for h in recent:
                    if not h.get("success"):
                        if consecutive_successes == 0:
                            consecutive_failures += 1
                        else:
                            break
                    else:
                        if consecutive_failures == 0:
                            consecutive_successes += 1
                        else:
                            break

                # Also use last_success from scheduler task data as a quick indicator
                if last_success is False and consecutive_failures == 0:
                    consecutive_failures = 1

                state["failure_streak"] = consecutive_failures
                state["success_streak"] = consecutive_successes

                # Success rate check
                success_rate = (task_success_count / task_exec_count * 100) if task_exec_count > 0 else None
                rate_threshold = self._alert_config["success_rate_threshold"]

                was_alerting = state["status"] == "alerting"

                # Determine if this task should alert
                should_alert = False
                alert_reasons = []

                if consecutive_failures >= streak_threshold:
                    should_alert = True
                    alert_reasons.append(f"failure_streak={consecutive_failures}")

                if success_rate is not None and success_rate < rate_threshold and task_exec_count >= 3:
                    should_alert = True
                    alert_reasons.append(f"success_rate={success_rate:.0f}%<{rate_threshold:.0f}%")

                if should_alert:
                    state["status"] = "alerting"
                    preset_failures += 1
                    preset_unhealthy_tasks.append(tid)

                    # Emit task failure event
                    if self._alert_config["emit_on_task_fail"]:
                        event_data = {
                            "task_id": tid,
                            "task_name": task_name,
                            "preset_id": preset_id,
                            "preset_name": preset.name if preset else preset_id,
                            "failure_streak": consecutive_failures,
                            "success_rate": f"{success_rate:.0f}%" if success_rate is not None else "n/a",
                            "total_executions": task_exec_count,
                            "reasons": alert_reasons,
                            "severity": "critical" if consecutive_failures >= streak_threshold * 2 else "warning",
                        }
                        emitted = await self._emit_alert_event(
                            "preset.task_failed", event_data,
                            priority="high" if consecutive_failures >= streak_threshold * 2 else "normal",
                        )
                        if emitted:
                            state["total_alerts_emitted"] += 1
                            alerts_emitted.append({
                                "topic": "preset.task_failed",
                                "task_id": tid,
                                "task_name": task_name,
                                "preset_id": preset_id,
                                "reasons": alert_reasons,
                            })

                elif was_alerting and consecutive_successes >= recovery_threshold:
                    # Task recovered!
                    state["status"] = "healthy"
                    if self._alert_config["emit_on_preset_recovered"]:
                        event_data = {
                            "task_id": tid,
                            "task_name": task_name,
                            "preset_id": preset_id,
                            "preset_name": preset.name if preset else preset_id,
                            "recovery_streak": consecutive_successes,
                            "success_rate": f"{success_rate:.0f}%" if success_rate is not None else "n/a",
                        }
                        emitted = await self._emit_alert_event(
                            "preset.task_recovered", event_data, priority="normal",
                        )
                        if emitted:
                            recoveries_emitted.append({
                                "topic": "preset.task_recovered",
                                "task_id": tid,
                                "task_name": task_name,
                                "preset_id": preset_id,
                            })
                elif not should_alert:
                    state["status"] = "healthy"

                preset_total += 1

            # Determine preset-level health
            preset_status = "healthy"
            if preset_failures > 0:
                preset_status = "degraded" if preset_failures < preset_total else "unhealthy"

            # Emit preset-level unhealthy event if any tasks are alerting
            if preset_failures > 0 and self._alert_config["emit_on_preset_unhealthy"]:
                event_data = {
                    "preset_id": preset_id,
                    "preset_name": preset.name if preset else preset_id,
                    "pillar": preset.pillar if preset else "unknown",
                    "status": preset_status,
                    "unhealthy_tasks": preset_failures,
                    "total_tasks": preset_total,
                    "unhealthy_task_ids": preset_unhealthy_tasks,
                    "severity": "critical" if preset_status == "unhealthy" else "warning",
                }
                await self._emit_alert_event(
                    "preset.unhealthy", event_data,
                    priority="high" if preset_status == "unhealthy" else "normal",
                )

            preset_health_summary.append({
                "preset_id": preset_id,
                "name": preset.name if preset else preset_id,
                "status": preset_status,
                "healthy_tasks": preset_total - preset_failures,
                "unhealthy_tasks": preset_failures,
                "total_tasks": preset_total,
            })

        self._save_state()

        total_healthy = sum(1 for p in preset_health_summary if p["status"] == "healthy")
        total_degraded = sum(1 for p in preset_health_summary if p["status"] == "degraded")
        total_unhealthy = sum(1 for p in preset_health_summary if p["status"] == "unhealthy")

        summary_msg = (
            f"Scanned {len(preset_health_summary)} presets: "
            f"{total_healthy} healthy, {total_degraded} degraded, {total_unhealthy} unhealthy. "
            f"{len(alerts_emitted)} alerts emitted, {len(recoveries_emitted)} recoveries."
        )

        return SkillResult(
            success=True,
            message=summary_msg,
            data={
                "presets": preset_health_summary,
                "alerts_emitted": alerts_emitted,
                "recoveries_emitted": recoveries_emitted,
                "total_healthy": total_healthy,
                "total_degraded": total_degraded,
                "total_unhealthy": total_unhealthy,
            },
        )

    async def _configure_alerts(self, params: Dict) -> SkillResult:
        """Configure health alert thresholds."""
        changed = []
        valid_keys = {
            "failure_streak_threshold": (int, 1, 100),
            "success_rate_threshold": (float, 0, 100),
            "recovery_streak_threshold": (int, 1, 100),
            "emit_on_task_fail": (bool, None, None),
            "emit_on_preset_unhealthy": (bool, None, None),
            "emit_on_preset_recovered": (bool, None, None),
        }

        for key, (typ, min_val, max_val) in valid_keys.items():
            if key in params:
                val = params[key]
                if typ == bool:
                    val = bool(val)
                elif typ == int:
                    val = int(val)
                    if min_val is not None and val < min_val:
                        return SkillResult(success=False, message=f"{key} must be >= {min_val}")
                    if max_val is not None and val > max_val:
                        return SkillResult(success=False, message=f"{key} must be <= {max_val}")
                elif typ == float:
                    val = float(val)
                    if min_val is not None and val < min_val:
                        return SkillResult(success=False, message=f"{key} must be >= {min_val}")
                    if max_val is not None and val > max_val:
                        return SkillResult(success=False, message=f"{key} must be <= {max_val}")
                self._alert_config[key] = val
                changed.append(f"{key}={val}")

        if not changed:
            return SkillResult(
                success=True,
                message="No changes (pass threshold parameters to update)",
                data={"current_config": dict(self._alert_config)},
            )

        self._save_state()
        return SkillResult(
            success=True,
            message=f"Updated alert config: {', '.join(changed)}",
            data={"config": dict(self._alert_config), "changed": changed},
        )

    async def _get_alert_history(self, params: Dict) -> SkillResult:
        """View recent health alert events."""
        limit = min(params.get("limit", 50), self._max_alert_history)
        filter_preset = params.get("preset_id", "").strip() or None

        history = list(self._alert_history)
        if filter_preset:
            history = [h for h in history if h.get("data", {}).get("preset_id") == filter_preset]

        history = history[-limit:]

        # Compute summary stats
        total_task_failed = sum(1 for h in self._alert_history if h.get("topic") == "preset.task_failed")
        total_unhealthy = sum(1 for h in self._alert_history if h.get("topic") == "preset.unhealthy")
        total_recovered = sum(1 for h in self._alert_history if h.get("topic") == "preset.task_recovered")

        return SkillResult(
            success=True,
            message=f"{len(history)} alert events (of {len(self._alert_history)} total)",
            data={
                "events": history,
                "total_events": len(self._alert_history),
                "returned": len(history),
                "summary": {
                    "task_failed_events": total_task_failed,
                    "preset_unhealthy_events": total_unhealthy,
                    "task_recovered_events": total_recovered,
                },
            },
        )

    async def _emit_alert_event(self, topic: str, data: Dict, priority: str = "normal") -> bool:
        """Emit a health alert event via EventBus (EventSkill)."""
        event_record = {
            "topic": topic,
            "data": data,
            "priority": priority,
            "timestamp": datetime.now().isoformat(),
            "emitted": False,
        }

        try:
            source = self._alert_config.get("event_source", "scheduler_presets_health")
            if hasattr(self, "_skill_registry") and self._skill_registry:
                result = await self._skill_registry.execute_skill(
                    "event", "publish",
                    {"topic": topic, "data": data, "source": source, "priority": priority},
                )
                emitted = result.success if result else False
            elif self.context:
                result = await self.context.call_skill(
                    "event", "publish",
                    {"topic": topic, "data": data, "source": source, "priority": priority},
                )
                emitted = result.success if result else False
            else:
                # No event bus available - still record the alert locally
                emitted = False
        except Exception:
            emitted = False

        event_record["emitted"] = emitted
        self._alert_history.append(event_record)
        if len(self._alert_history) > self._max_alert_history:
            self._alert_history = self._alert_history[-self._max_alert_history:]

        return emitted

    def _preset_priority(self, preset_id: str) -> int:
        """Priority ranking for recommendations (lower = higher priority)."""
        priority_order = [
            "health_monitoring",         # 1 - foundation
            "alert_polling",             # 2 - automated response
            "dashboard_auto_check",      # 3 - observability
            "goal_stall_monitoring",     # 4 - goal health
            "self_tuning",               # 5 - self-optimization
            "feedback_loop",             # 6 - learning
            "self_assessment",           # 7 - self-awareness
            "revenue_goal_evaluation",   # 8 - revenue tracking
            "reputation_polling",        # 9 - multi-agent
            "revenue_reporting",         # 10 - business
            "revenue_goals",             # 11 - revenue goals
            "fleet_health_auto_heal",    # 12 - fleet health
            "circuit_sharing_monitor",   # 13 - circuit sharing
            "knowledge_sync",            # 14 - sharing
            "experiment_management",     # 15 - experiments
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
        """Persist applied presets, custom presets, and alert state to disk."""
        try:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            data = {
                "applied": self._applied,
                "custom_presets": self._custom_presets,
                "alert_config": self._alert_config,
                "alert_state": self._alert_state,
                "alert_history": self._alert_history[-self._max_alert_history:],
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
                saved_config = data.get("alert_config", {})
                if saved_config:
                    self._alert_config.update(saved_config)
                self._alert_state = data.get("alert_state", {})
                self._alert_history = data.get("alert_history", [])[-self._max_alert_history:]
        except Exception:
            self._applied = {}
            self._custom_presets = {}
