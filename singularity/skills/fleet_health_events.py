#!/usr/bin/env python3
"""
FleetHealthEventBridgeSkill - Emit EventBus events on fleet health management actions.

When FleetHealthManagerSkill heals, scales, replaces, or updates replicas, this
bridge emits structured EventBus events so downstream skills can react:

Events emitted:
- fleet_health.heal_started: A heal action (restart/replace) was initiated
- fleet_health.heal_completed: A heal action finished (success or failure)
- fleet_health.scale_up: Fleet scaled up (new replicas added)
- fleet_health.scale_down: Fleet scaled down (replicas removed)
- fleet_health.rolling_update: A rolling update batch was processed
- fleet_health.assessment: Fleet health assessment completed with recommendations
- fleet_health.policy_changed: Fleet management policy was updated
- fleet_health.fleet_alert: Critical fleet condition detected (e.g., too many unhealthy)

This enables reactive automation:
- AlertIncidentBridge can create incidents when heals fail repeatedly
- StrategySkill can reprioritize when fleet capacity changes
- CircuitSharingEvents can correlate circuit states with fleet health
- RevenueGoalAutoSetter can adjust targets when fleet degrades
- SchedulerPresets can trigger emergency maintenance on fleet alerts

Architecture:
  FleetHealthManager acts → FleetHealthEventBridge detects changes →
  EventBus emits events → downstream skills react

Pillar: Replication (primary) + Self-Improvement (fleet-wide reactive automation)
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import Skill, SkillAction, SkillManifest, SkillResult

DATA_DIR = Path(__file__).parent.parent / "data"
BRIDGE_STATE_FILE = DATA_DIR / "fleet_health_events.json"
MAX_EVENT_HISTORY = 200


def _now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


class FleetHealthEventBridgeSkill(Skill):
    """
    Bridge between FleetHealthManagerSkill and EventBus.

    Monitors fleet health management actions and emits structured events
    so the rest of the agent ecosystem can react to fleet changes.

    Actions:
    - monitor: Check fleet health state for changes and emit events
    - configure: Update event emission settings
    - status: View bridge health and emission statistics
    - history: View recent emitted events
    - emit_test: Emit a test event to verify EventBus integration
    - fleet_check: Analyze fleet health for critical conditions and emit alerts
    """

    def __init__(self, credentials: Dict[str, str] = None):
        super().__init__(credentials)
        self._load_state()

    def _load_state(self):
        """Load persisted bridge state from disk."""
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        if BRIDGE_STATE_FILE.exists():
            try:
                with open(BRIDGE_STATE_FILE) as f:
                    data = json.load(f)
                self._last_snapshot = data.get("last_snapshot", {})
                self._event_history = data.get("event_history", [])[-MAX_EVENT_HISTORY:]
                self._config = data.get("config", self._default_config())
                self._stats = data.get("stats", self._default_stats())
            except (json.JSONDecodeError, Exception):
                self._init_empty()
        else:
            self._init_empty()

    def _init_empty(self):
        self._last_snapshot: Dict = {}
        self._event_history: List[Dict] = []
        self._config = self._default_config()
        self._stats = self._default_stats()

    def _default_config(self) -> Dict:
        return {
            "emit_on_heal": True,
            "emit_on_scale": True,
            "emit_on_rolling_update": True,
            "emit_on_assessment": True,
            "emit_on_policy_change": True,
            "emit_on_fleet_alert": True,
            "unhealthy_threshold": 0.5,  # Alert when >50% of fleet is unhealthy
            "event_source": "fleet_health_event_bridge",
            "priority_heal": "high",
            "priority_scale": "normal",
            "priority_rolling_update": "normal",
            "priority_assessment": "low",
            "priority_policy_change": "normal",
            "priority_fleet_alert": "critical",
        }

    def _default_stats(self) -> Dict:
        return {
            "events_emitted": 0,
            "events_failed": 0,
            "monitors_run": 0,
            "fleet_checks_run": 0,
            "heals_detected": 0,
            "scales_detected": 0,
            "updates_detected": 0,
            "assessments_detected": 0,
            "policy_changes_detected": 0,
            "fleet_alerts_emitted": 0,
            "last_monitor_time": None,
            "last_fleet_check_time": None,
        }

    def _save_state(self):
        """Persist bridge state to disk."""
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        data = {
            "last_snapshot": self._last_snapshot,
            "event_history": self._event_history[-MAX_EVENT_HISTORY:],
            "config": self._config,
            "stats": self._stats,
            "last_updated": _now_iso(),
        }
        with open(BRIDGE_STATE_FILE, "w") as f:
            json.dump(data, f, indent=2, default=str)

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="fleet_health_events",
            name="Fleet Health Event Bridge",
            version="1.0.0",
            category="replication",
            description=(
                "Emit EventBus events when fleet health management actions occur "
                "(heal, scale, rolling update, assessment). Enables fleet-wide "
                "reactive automation."
            ),
            actions=[
                SkillAction(
                    name="monitor",
                    description=(
                        "Check fleet health manager for new incidents since last "
                        "monitor call and emit events for heals, scales, updates, "
                        "and policy changes."
                    ),
                    parameters={},
                    estimated_cost=0,
                ),
                SkillAction(
                    name="configure",
                    description="Update event emission settings and thresholds",
                    parameters={
                        "emit_on_heal": {
                            "type": "bool", "required": False,
                            "description": "Emit events when heal actions occur",
                        },
                        "emit_on_scale": {
                            "type": "bool", "required": False,
                            "description": "Emit events when fleet scales up/down",
                        },
                        "emit_on_rolling_update": {
                            "type": "bool", "required": False,
                            "description": "Emit events during rolling updates",
                        },
                        "emit_on_assessment": {
                            "type": "bool", "required": False,
                            "description": "Emit events on health assessments",
                        },
                        "emit_on_policy_change": {
                            "type": "bool", "required": False,
                            "description": "Emit events on policy updates",
                        },
                        "emit_on_fleet_alert": {
                            "type": "bool", "required": False,
                            "description": "Emit fleet-wide alert events",
                        },
                        "unhealthy_threshold": {
                            "type": "float", "required": False,
                            "description": "Alert when fraction of unhealthy agents exceeds this (0-1)",
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="status",
                    description="View bridge health, emission statistics, and detection counts",
                    parameters={},
                    estimated_cost=0,
                ),
                SkillAction(
                    name="history",
                    description="View recent emitted events",
                    parameters={
                        "limit": {
                            "type": "int", "required": False,
                            "description": "Max events to return (default 20)",
                        },
                        "topic_filter": {
                            "type": "str", "required": False,
                            "description": "Filter by event topic prefix",
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="emit_test",
                    description="Emit a test event to verify EventBus integration",
                    parameters={},
                    estimated_cost=0,
                ),
                SkillAction(
                    name="fleet_check",
                    description=(
                        "Analyze current fleet health for critical conditions "
                        "and emit alerts if thresholds are exceeded."
                    ),
                    parameters={},
                    estimated_cost=0,
                ),
            ],
            required_credentials=[],
        )

    def check_credentials(self) -> bool:
        return True

    async def execute(self, action: str, params: Dict) -> SkillResult:
        actions = {
            "monitor": self._monitor,
            "configure": self._configure,
            "status": self._status,
            "history": self._history,
            "emit_test": self._emit_test,
            "fleet_check": self._fleet_check,
        }
        handler = actions.get(action)
        if not handler:
            return SkillResult(
                success=False,
                message=f"Unknown action: {action}. Available: {list(actions.keys())}",
            )
        try:
            return await handler(params)
        except Exception as e:
            return SkillResult(success=False, message=f"Error in {action}: {e}")

    async def _monitor(self, params: Dict) -> SkillResult:
        """Check fleet health manager for new incidents and emit events."""
        self._stats["monitors_run"] += 1
        self._stats["last_monitor_time"] = _now_iso()

        events_emitted = 0
        events_detail = []

        # Get fleet health manager incidents
        incidents = await self._get_fleet_incidents()
        if incidents is None:
            self._save_state()
            return SkillResult(
                success=True,
                message="Fleet health manager skill not available, no events emitted",
                data={"events_emitted": 0},
            )

        # Get fleet status for context
        fleet_status = await self._get_fleet_status()

        # Track last processed incident timestamp
        last_processed = self._last_snapshot.get("last_incident_ts", "")

        # Process new incidents
        incident_list = incidents.get("incidents", [])
        new_incidents = [
            inc for inc in incident_list
            if inc.get("timestamp", "") > last_processed
        ]

        for incident in new_incidents:
            action_type = incident.get("action", "")
            agent_id = incident.get("agent_id", "")
            ts = incident.get("timestamp", _now_iso())

            # Emit heal events
            if action_type in ("heal_restart", "heal_replace") and self._config["emit_on_heal"]:
                self._stats["heals_detected"] += 1
                emitted = await self._emit_event(
                    "fleet_health.heal_completed",
                    {
                        "heal_type": action_type,
                        "agent_id": agent_id,
                        "success": incident.get("success", False),
                        "reason": incident.get("reason", ""),
                        "attempt": incident.get("attempt", 1),
                        "details": incident.get("details", {}),
                        "timestamp": ts,
                    },
                    self._config["priority_heal"],
                )
                if emitted:
                    events_emitted += 1
                    events_detail.append(f"heal:{action_type}:{agent_id}")

            # Emit scale events
            elif action_type in ("scale_up", "scale_down") and self._config["emit_on_scale"]:
                self._stats["scales_detected"] += 1
                topic = f"fleet_health.{action_type}"
                emitted = await self._emit_event(
                    topic,
                    {
                        "direction": action_type.split("_")[1],  # "up" or "down"
                        "agent_id": agent_id,
                        "reason": incident.get("reason", ""),
                        "fleet_size_before": incident.get("fleet_size_before"),
                        "fleet_size_after": incident.get("fleet_size_after"),
                        "details": incident.get("details", {}),
                        "timestamp": ts,
                    },
                    self._config["priority_scale"],
                )
                if emitted:
                    events_emitted += 1
                    events_detail.append(f"scale:{action_type}")

            # Emit rolling update events
            elif action_type == "rolling_update" and self._config["emit_on_rolling_update"]:
                self._stats["updates_detected"] += 1
                emitted = await self._emit_event(
                    "fleet_health.rolling_update",
                    {
                        "update_id": incident.get("update_id", ""),
                        "agent_id": agent_id,
                        "batch": incident.get("batch", 0),
                        "total_batches": incident.get("total_batches", 0),
                        "status": incident.get("status", "unknown"),
                        "details": incident.get("details", {}),
                        "timestamp": ts,
                    },
                    self._config["priority_rolling_update"],
                )
                if emitted:
                    events_emitted += 1
                    events_detail.append(f"rolling_update:{agent_id}")

            # Emit policy change events
            elif action_type == "policy_change" and self._config["emit_on_policy_change"]:
                self._stats["policy_changes_detected"] += 1
                emitted = await self._emit_event(
                    "fleet_health.policy_changed",
                    {
                        "changes": incident.get("changes", {}),
                        "reason": incident.get("reason", ""),
                        "timestamp": ts,
                    },
                    self._config["priority_policy_change"],
                )
                if emitted:
                    events_emitted += 1
                    events_detail.append("policy_changed")

        # Update snapshot watermark
        if incident_list:
            all_ts = [inc.get("timestamp", "") for inc in incident_list if inc.get("timestamp")]
            if all_ts:
                self._last_snapshot["last_incident_ts"] = max(all_ts)

        # Also emit assessment events if fleet status changed
        if fleet_status and self._config["emit_on_assessment"]:
            current_health = self._summarize_fleet(fleet_status)
            prev_health = self._last_snapshot.get("fleet_summary", {})

            if self._health_changed(prev_health, current_health):
                self._stats["assessments_detected"] += 1
                emitted = await self._emit_event(
                    "fleet_health.assessment",
                    {
                        "total_agents": current_health.get("total", 0),
                        "healthy": current_health.get("healthy", 0),
                        "unhealthy": current_health.get("unhealthy", 0),
                        "dead": current_health.get("dead", 0),
                        "unknown": current_health.get("unknown", 0),
                        "health_fraction": current_health.get("health_fraction", 0),
                        "previous_health_fraction": prev_health.get("health_fraction", 0),
                        "timestamp": _now_iso(),
                    },
                    self._config["priority_assessment"],
                )
                if emitted:
                    events_emitted += 1
                    events_detail.append("assessment")

            self._last_snapshot["fleet_summary"] = current_health

        self._save_state()

        return SkillResult(
            success=True,
            message=f"Monitor complete: {events_emitted} events emitted, {len(new_incidents)} new incidents processed",
            data={
                "events_emitted": events_emitted,
                "events_detail": events_detail,
                "new_incidents_processed": len(new_incidents),
            },
        )

    async def _fleet_check(self, params: Dict) -> SkillResult:
        """Analyze fleet health for critical conditions and emit alerts."""
        self._stats["fleet_checks_run"] += 1
        self._stats["last_fleet_check_time"] = _now_iso()

        fleet_status = await self._get_fleet_status()
        if fleet_status is None:
            self._save_state()
            return SkillResult(
                success=True,
                message="Fleet health manager skill not available",
                data={"fleet_alert_emitted": False},
            )

        summary = self._summarize_fleet(fleet_status)
        total = summary.get("total", 0)

        if total == 0:
            self._save_state()
            return SkillResult(
                success=True,
                message="No agents registered in fleet",
                data={"fleet_alert_emitted": False, "total_agents": 0},
            )

        unhealthy_fraction = 1.0 - summary.get("health_fraction", 1.0)
        threshold = self._config["unhealthy_threshold"]
        alert_emitted = False

        if unhealthy_fraction > threshold and self._config["emit_on_fleet_alert"]:
            self._stats["fleet_alerts_emitted"] += 1
            emitted = await self._emit_event(
                "fleet_health.fleet_alert",
                {
                    "alert_type": "high_unhealthy_rate",
                    "unhealthy_fraction": round(unhealthy_fraction, 3),
                    "threshold": threshold,
                    "total_agents": total,
                    "healthy": summary.get("healthy", 0),
                    "unhealthy": summary.get("unhealthy", 0),
                    "dead": summary.get("dead", 0),
                    "message": (
                        f"Fleet alert: {summary.get('unhealthy', 0) + summary.get('dead', 0)}/{total} "
                        f"agents unhealthy/dead ({unhealthy_fraction:.0%}) "
                        f"exceeds {threshold:.0%} threshold"
                    ),
                    "timestamp": _now_iso(),
                },
                self._config["priority_fleet_alert"],
            )
            alert_emitted = emitted

        self._save_state()

        return SkillResult(
            success=True,
            message=(
                f"Fleet check: {summary.get('healthy', 0)}/{total} healthy "
                f"({summary.get('health_fraction', 0):.0%}), "
                f"alert {'emitted' if alert_emitted else 'not needed'}"
            ),
            data={
                "health_fraction": summary.get("health_fraction", 0),
                "unhealthy_fraction": round(unhealthy_fraction, 3),
                "total_agents": total,
                "healthy": summary.get("healthy", 0),
                "unhealthy": summary.get("unhealthy", 0),
                "dead": summary.get("dead", 0),
                "threshold": threshold,
                "fleet_alert_emitted": alert_emitted,
            },
        )

    async def _configure(self, params: Dict) -> SkillResult:
        """Update event emission configuration."""
        updated = []

        for key in (
            "emit_on_heal",
            "emit_on_scale",
            "emit_on_rolling_update",
            "emit_on_assessment",
            "emit_on_policy_change",
            "emit_on_fleet_alert",
        ):
            if key in params:
                self._config[key] = bool(params[key])
                updated.append(f"{key}={params[key]}")

        if "unhealthy_threshold" in params:
            val = float(params["unhealthy_threshold"])
            self._config["unhealthy_threshold"] = max(0.0, min(1.0, val))
            updated.append(f"unhealthy_threshold={self._config['unhealthy_threshold']}")

        for key in (
            "priority_heal",
            "priority_scale",
            "priority_rolling_update",
            "priority_assessment",
            "priority_policy_change",
            "priority_fleet_alert",
        ):
            if key in params:
                self._config[key] = str(params[key])
                updated.append(f"{key}={params[key]}")

        if "event_source" in params:
            self._config["event_source"] = str(params["event_source"])
            updated.append(f"event_source={params['event_source']}")

        if not updated:
            return SkillResult(
                success=False,
                message="No valid configuration parameters provided",
            )

        self._save_state()
        return SkillResult(
            success=True,
            message=f"Updated: {', '.join(updated)}",
            data={"config": self._config},
        )

    async def _status(self, params: Dict) -> SkillResult:
        """View bridge health and statistics."""
        lines = ["=== Fleet Health Event Bridge Status ==="]
        lines.append(f"Events emitted: {self._stats['events_emitted']}")
        lines.append(f"Events failed: {self._stats['events_failed']}")
        lines.append(f"Monitors run: {self._stats['monitors_run']}")
        lines.append(f"Fleet checks run: {self._stats['fleet_checks_run']}")
        lines.append(f"Heals detected: {self._stats['heals_detected']}")
        lines.append(f"Scales detected: {self._stats['scales_detected']}")
        lines.append(f"Updates detected: {self._stats['updates_detected']}")
        lines.append(f"Assessments detected: {self._stats['assessments_detected']}")
        lines.append(f"Policy changes detected: {self._stats['policy_changes_detected']}")
        lines.append(f"Fleet alerts emitted: {self._stats['fleet_alerts_emitted']}")
        lines.append(f"Last monitor: {self._stats['last_monitor_time'] or 'never'}")
        lines.append(f"Last fleet check: {self._stats['last_fleet_check_time'] or 'never'}")

        # Include last known fleet summary
        summary = self._last_snapshot.get("fleet_summary", {})
        if summary:
            lines.append(f"Last fleet summary: {summary.get('healthy', 0)} healthy, "
                         f"{summary.get('unhealthy', 0)} unhealthy, "
                         f"{summary.get('dead', 0)} dead "
                         f"(total: {summary.get('total', 0)})")

        return SkillResult(
            success=True,
            message="\n".join(lines),
            data={
                "stats": self._stats,
                "config": self._config,
                "last_snapshot": self._last_snapshot,
                "event_history_count": len(self._event_history),
            },
        )

    async def _history(self, params: Dict) -> SkillResult:
        """View recent emitted events."""
        limit = int(params.get("limit", 20))
        topic_filter = params.get("topic_filter", "")

        entries = self._event_history
        if topic_filter:
            entries = [e for e in entries if e.get("topic", "").startswith(topic_filter)]

        recent = entries[-limit:]

        if not recent:
            return SkillResult(
                success=True,
                message="No events in history",
                data={"events": [], "total": 0},
            )

        lines = [f"=== Event History (last {len(recent)}) ==="]
        for entry in reversed(recent):
            ts = entry.get("timestamp", "?")
            topic = entry.get("topic", "?")
            success = "ok" if entry.get("emitted") else "FAILED"
            lines.append(f"  [{ts}] {topic} ({success})")

        return SkillResult(
            success=True,
            message="\n".join(lines),
            data={"events": recent, "total": len(entries)},
        )

    async def _emit_test(self, params: Dict) -> SkillResult:
        """Emit a test event to verify EventBus integration."""
        emitted = await self._emit_event(
            "fleet_health.test",
            {
                "message": "Test event from FleetHealthEventBridge",
                "timestamp": _now_iso(),
            },
            "normal",
        )

        return SkillResult(
            success=True,
            message=f"Test event {'emitted successfully' if emitted else 'failed (EventBus not available)'}",
            data={"emitted": emitted},
        )

    # --- Internal helpers ---

    def _summarize_fleet(self, fleet_status: Dict) -> Dict:
        """Extract health summary from fleet status data."""
        agents = fleet_status.get("agents", {})
        total = len(agents)
        healthy = 0
        unhealthy = 0
        dead = 0
        unknown = 0

        for agent_id, info in agents.items():
            status = info.get("status", "unknown")
            if status == "healthy":
                healthy += 1
            elif status == "unhealthy":
                unhealthy += 1
            elif status == "dead":
                dead += 1
            else:
                unknown += 1

        return {
            "total": total,
            "healthy": healthy,
            "unhealthy": unhealthy,
            "dead": dead,
            "unknown": unknown,
            "health_fraction": round(healthy / max(total, 1), 3),
        }

    def _health_changed(self, prev: Dict, current: Dict) -> bool:
        """Check if fleet health summary has meaningfully changed."""
        if not prev:
            return True
        # Detect any change in counts
        for key in ("total", "healthy", "unhealthy", "dead"):
            if prev.get(key) != current.get(key):
                return True
        return False

    async def _get_fleet_incidents(self) -> Optional[Dict]:
        """Get fleet health manager incidents via skill context."""
        try:
            if self.context:
                result = await self.context.call_skill(
                    "fleet_health_manager", "incidents", {"limit": 50}
                )
                if result and result.success:
                    return result.data
        except Exception:
            pass
        return None

    async def _get_fleet_status(self) -> Optional[Dict]:
        """Get fleet health manager status via skill context."""
        try:
            if self.context:
                result = await self.context.call_skill(
                    "fleet_health_manager", "status", {}
                )
                if result and result.success:
                    return result.data
        except Exception:
            pass
        return None

    async def _emit_event(self, topic: str, data: Dict, priority: str = "normal") -> bool:
        """Emit an event via the skill registry's EventSkill."""
        event_record = {
            "topic": topic,
            "data": data,
            "priority": priority,
            "timestamp": _now_iso(),
            "emitted": False,
        }

        try:
            if hasattr(self, "_skill_registry") and self._skill_registry:
                result = await self._skill_registry.execute_skill(
                    "event", "publish",
                    {
                        "topic": topic,
                        "data": data,
                        "source": self._config.get("event_source", "fleet_health_event_bridge"),
                        "priority": priority,
                    },
                )
                emitted = result.success if result else False
            elif self.context:
                result = await self.context.call_skill(
                    "event", "publish",
                    {
                        "topic": topic,
                        "data": data,
                        "source": self._config.get("event_source", "fleet_health_event_bridge"),
                        "priority": priority,
                    },
                )
                emitted = result.success if result else False
            else:
                emitted = False
        except Exception:
            emitted = False

        event_record["emitted"] = emitted
        self._event_history.append(event_record)
        if len(self._event_history) > MAX_EVENT_HISTORY:
            self._event_history = self._event_history[-MAX_EVENT_HISTORY:]

        if emitted:
            self._stats["events_emitted"] += 1
        else:
            self._stats["events_failed"] += 1

        return emitted
