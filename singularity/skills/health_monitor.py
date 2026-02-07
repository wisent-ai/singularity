#!/usr/bin/env python3
"""
AgentHealthMonitor Skill - Health checking and monitoring for replica agents.

This skill enables the parent agent to:
- Register replica agents with heartbeat expectations
- Track heartbeats from replicas (last seen, uptime, status)
- Detect unresponsive or degraded replicas
- Auto-restart or escalate when replicas go down
- Aggregate health metrics across the fleet
- Alert on anomalies (high error rates, resource exhaustion, drift)

This is a critical gap in the Replication pillar - without health monitoring,
replicas operate blindly and failures go undetected. With it, the parent
agent becomes a reliable fleet manager.

Works with:
- ReplicationSkill: knows which replicas exist
- InboxSkill: receives heartbeat messages from replicas
- TaskDelegator: can reassign work from unhealthy replicas
- KnowledgeSharingSkill: can share health insights across fleet
"""

import json
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from .base import Skill, SkillResult, SkillManifest, SkillAction


HEALTH_FILE = Path(__file__).parent.parent / "data" / "health_monitor.json"

# Agent health states
HEALTH_STATES = ["healthy", "degraded", "unresponsive", "dead", "unknown"]

# Alert severities
ALERT_SEVERITIES = ["info", "warning", "critical"]

# Limits
MAX_AGENTS = 100
MAX_ALERTS = 500
MAX_HEARTBEAT_HISTORY = 50


def _load_data(path: Path = None) -> Dict:
    """Load health monitor data from disk."""
    p = path or HEALTH_FILE
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return {
        "agents": {},
        "alerts": [],
        "config": {
            "heartbeat_interval_seconds": 60,
            "degraded_threshold_missed": 3,
            "dead_threshold_missed": 10,
            "auto_restart_enabled": True,
            "max_restart_attempts": 3,
        },
        "fleet_stats": {
            "total_heartbeats_received": 0,
            "total_restarts_triggered": 0,
            "total_alerts_fired": 0,
        },
    }


def _save_data(data: Dict, path: Path = None):
    """Save health monitor data to disk."""
    p = path or HEALTH_FILE
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(data, f, indent=2, default=str)


class AgentHealthMonitor(Skill):
    """
    Monitor health of replica agents through heartbeats, alerts, and fleet metrics.

    The parent agent registers replicas, receives their heartbeats,
    detects failures, and can trigger restarts or work reassignment.
    """

    def __init__(self, credentials=None, data_path: Path = None):
        super().__init__(credentials)
        self.data_path = data_path

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="health_monitor",
            name="Agent Health Monitor",
            version="1.0.0",
            category="replication",
            description="Health checking and monitoring for replica agents - heartbeats, alerts, fleet metrics",
            actions=[
                SkillAction(
                    name="register_agent",
                    description="Register a replica agent for health monitoring",
                    parameters={
                        "agent_id": {"type": "str", "required": True, "description": "Unique agent identifier"},
                        "agent_name": {"type": "str", "required": False, "description": "Human-readable name"},
                        "agent_type": {"type": "str", "required": False, "description": "Type: replica, worker, specialist"},
                        "expected_heartbeat_seconds": {"type": "int", "required": False, "description": "Expected heartbeat interval"},
                        "metadata": {"type": "dict", "required": False, "description": "Additional agent metadata"},
                    },
                ),
                SkillAction(
                    name="heartbeat",
                    description="Record a heartbeat from a replica agent",
                    parameters={
                        "agent_id": {"type": "str", "required": True, "description": "Agent sending heartbeat"},
                        "status": {"type": "str", "required": False, "description": "Agent's self-reported status"},
                        "metrics": {"type": "dict", "required": False, "description": "Agent metrics (cpu, memory, tasks, errors)"},
                        "message": {"type": "str", "required": False, "description": "Optional status message"},
                    },
                ),
                SkillAction(
                    name="check_health",
                    description="Check health of a specific agent or all agents",
                    parameters={
                        "agent_id": {"type": "str", "required": False, "description": "Specific agent to check (omit for all)"},
                    },
                ),
                SkillAction(
                    name="fleet_status",
                    description="Get aggregate health metrics for the entire fleet",
                    parameters={},
                ),
                SkillAction(
                    name="deregister_agent",
                    description="Remove an agent from health monitoring",
                    parameters={
                        "agent_id": {"type": "str", "required": True, "description": "Agent to deregister"},
                    },
                ),
                SkillAction(
                    name="get_alerts",
                    description="Get recent health alerts, optionally filtered",
                    parameters={
                        "severity": {"type": "str", "required": False, "description": "Filter by severity: info, warning, critical"},
                        "agent_id": {"type": "str", "required": False, "description": "Filter by agent"},
                        "limit": {"type": "int", "required": False, "description": "Max alerts to return"},
                    },
                ),
                SkillAction(
                    name="acknowledge_alert",
                    description="Acknowledge an alert to mark it as handled",
                    parameters={
                        "alert_id": {"type": "str", "required": True, "description": "Alert ID to acknowledge"},
                    },
                ),
                SkillAction(
                    name="update_config",
                    description="Update health monitoring configuration",
                    parameters={
                        "heartbeat_interval_seconds": {"type": "int", "required": False, "description": "Expected heartbeat interval"},
                        "degraded_threshold_missed": {"type": "int", "required": False, "description": "Missed heartbeats before degraded"},
                        "dead_threshold_missed": {"type": "int", "required": False, "description": "Missed heartbeats before dead"},
                        "auto_restart_enabled": {"type": "bool", "required": False, "description": "Enable auto-restart of dead agents"},
                        "max_restart_attempts": {"type": "int", "required": False, "description": "Max restart attempts before giving up"},
                    },
                ),
                SkillAction(
                    name="trigger_restart",
                    description="Manually trigger restart of an unresponsive agent",
                    parameters={
                        "agent_id": {"type": "str", "required": True, "description": "Agent to restart"},
                        "reason": {"type": "str", "required": False, "description": "Reason for restart"},
                    },
                ),
            ],
            required_credentials=[],
        )

    async def execute(self, action: str, params: Dict) -> SkillResult:
        actions = {
            "register_agent": self._register_agent,
            "heartbeat": self._heartbeat,
            "check_health": self._check_health,
            "fleet_status": self._fleet_status,
            "deregister_agent": self._deregister_agent,
            "get_alerts": self._get_alerts,
            "acknowledge_alert": self._acknowledge_alert,
            "update_config": self._update_config,
            "trigger_restart": self._trigger_restart,
        }
        handler = actions.get(action)
        if not handler:
            return SkillResult(success=False, message=f"Unknown action: {action}")
        return await handler(params)

    # --- Actions ---

    async def _register_agent(self, params: Dict) -> SkillResult:
        """Register a replica agent for health monitoring."""
        agent_id = params.get("agent_id")
        if not agent_id:
            return SkillResult(success=False, message="agent_id is required")

        data = _load_data(self.data_path)

        if len(data["agents"]) >= MAX_AGENTS and agent_id not in data["agents"]:
            return SkillResult(success=False, message=f"Max {MAX_AGENTS} agents reached")

        now = datetime.now().isoformat()
        agent_name = params.get("agent_name", agent_id)
        agent_type = params.get("agent_type", "replica")
        heartbeat_interval = params.get("expected_heartbeat_seconds",
                                         data["config"]["heartbeat_interval_seconds"])

        data["agents"][agent_id] = {
            "agent_id": agent_id,
            "agent_name": agent_name,
            "agent_type": agent_type,
            "registered_at": now,
            "expected_heartbeat_seconds": heartbeat_interval,
            "last_heartbeat": None,
            "heartbeat_count": 0,
            "health_state": "unknown",
            "last_status": None,
            "last_metrics": {},
            "last_message": None,
            "restart_count": 0,
            "heartbeat_history": [],
            "metadata": params.get("metadata", {}),
        }

        _save_data(data, self.data_path)

        return SkillResult(
            success=True,
            message=f"Agent '{agent_name}' ({agent_id}) registered for health monitoring",
            data={"agent_id": agent_id, "agent_name": agent_name},
        )

    async def _heartbeat(self, params: Dict) -> SkillResult:
        """Record a heartbeat from a replica agent."""
        agent_id = params.get("agent_id")
        if not agent_id:
            return SkillResult(success=False, message="agent_id is required")

        data = _load_data(self.data_path)

        if agent_id not in data["agents"]:
            return SkillResult(success=False, message=f"Agent '{agent_id}' not registered")

        agent = data["agents"][agent_id]
        now = datetime.now().isoformat()

        # Record heartbeat
        status = params.get("status", "ok")
        metrics = params.get("metrics", {})
        message = params.get("message")

        beat = {
            "timestamp": now,
            "status": status,
            "metrics": metrics,
            "message": message,
        }

        agent["last_heartbeat"] = now
        agent["heartbeat_count"] += 1
        agent["last_status"] = status
        agent["last_metrics"] = metrics
        agent["last_message"] = message
        agent["health_state"] = "healthy"

        # Keep limited history
        agent["heartbeat_history"].append(beat)
        if len(agent["heartbeat_history"]) > MAX_HEARTBEAT_HISTORY:
            agent["heartbeat_history"] = agent["heartbeat_history"][-MAX_HEARTBEAT_HISTORY:]

        data["fleet_stats"]["total_heartbeats_received"] += 1

        # Check for anomalies in metrics
        alerts_generated = self._check_metric_anomalies(data, agent_id, metrics)

        _save_data(data, self.data_path)

        return SkillResult(
            success=True,
            message=f"Heartbeat recorded for '{agent_id}' (count: {agent['heartbeat_count']})",
            data={
                "agent_id": agent_id,
                "heartbeat_count": agent["heartbeat_count"],
                "health_state": agent["health_state"],
                "alerts_generated": alerts_generated,
            },
        )

    async def _check_health(self, params: Dict) -> SkillResult:
        """Check health of a specific agent or all agents."""
        data = _load_data(self.data_path)
        agent_id = params.get("agent_id")

        if agent_id:
            if agent_id not in data["agents"]:
                return SkillResult(success=False, message=f"Agent '{agent_id}' not found")
            agents_to_check = {agent_id: data["agents"][agent_id]}
        else:
            agents_to_check = data["agents"]

        results = {}
        now = datetime.now()
        alerts_generated = 0

        for aid, agent in agents_to_check.items():
            old_state = agent["health_state"]
            new_state = self._compute_health_state(agent, data["config"], now)
            agent["health_state"] = new_state

            # Generate alerts on state transitions
            if old_state != new_state and old_state != "unknown":
                alert = self._create_alert(data, aid, old_state, new_state)
                alerts_generated += 1

                # Auto-restart if configured and agent is dead
                if (new_state == "dead"
                        and data["config"]["auto_restart_enabled"]
                        and agent["restart_count"] < data["config"]["max_restart_attempts"]):
                    agent["restart_count"] += 1
                    data["fleet_stats"]["total_restarts_triggered"] += 1
                    self._create_alert(
                        data, aid, "dead", "restarting",
                        severity="warning",
                        message=f"Auto-restart triggered (attempt {agent['restart_count']})",
                    )
                    # Actual restart would be done via ReplicationSkill
                    # We record the intent and let the caller act on it
                    if not results.get(aid):
                        results[aid] = {}
                    results[aid]["restart_triggered"] = True

            results[aid] = {
                "agent_name": agent["agent_name"],
                "health_state": new_state,
                "state_changed": old_state != new_state,
                "last_heartbeat": agent["last_heartbeat"],
                "heartbeat_count": agent["heartbeat_count"],
                "last_status": agent["last_status"],
                "restart_count": agent["restart_count"],
                "restart_triggered": results.get(aid, {}).get("restart_triggered", False),
            }

        _save_data(data, self.data_path)

        healthy = sum(1 for r in results.values() if r["health_state"] == "healthy")
        total = len(results)

        return SkillResult(
            success=True,
            message=f"Health check: {healthy}/{total} agents healthy, {alerts_generated} alerts generated",
            data={
                "agents": results,
                "summary": {
                    "total": total,
                    "healthy": healthy,
                    "degraded": sum(1 for r in results.values() if r["health_state"] == "degraded"),
                    "unresponsive": sum(1 for r in results.values() if r["health_state"] == "unresponsive"),
                    "dead": sum(1 for r in results.values() if r["health_state"] == "dead"),
                    "unknown": sum(1 for r in results.values() if r["health_state"] == "unknown"),
                },
                "alerts_generated": alerts_generated,
            },
        )

    async def _fleet_status(self, params: Dict) -> SkillResult:
        """Get aggregate fleet health metrics."""
        data = _load_data(self.data_path)

        if not data["agents"]:
            return SkillResult(
                success=True,
                message="No agents registered",
                data={"agents_count": 0, "fleet_stats": data["fleet_stats"]},
            )

        # Recompute health states
        now = datetime.now()
        state_counts = {s: 0 for s in HEALTH_STATES}
        agent_summaries = []
        total_heartbeats = 0
        total_restarts = 0

        for aid, agent in data["agents"].items():
            agent["health_state"] = self._compute_health_state(agent, data["config"], now)
            state_counts[agent["health_state"]] += 1
            total_heartbeats += agent["heartbeat_count"]
            total_restarts += agent["restart_count"]

            agent_summaries.append({
                "agent_id": aid,
                "agent_name": agent["agent_name"],
                "agent_type": agent["agent_type"],
                "health_state": agent["health_state"],
                "heartbeat_count": agent["heartbeat_count"],
                "last_heartbeat": agent["last_heartbeat"],
                "restart_count": agent["restart_count"],
            })

        _save_data(data, self.data_path)

        # Sort: unhealthy agents first
        state_priority = {"dead": 0, "unresponsive": 1, "degraded": 2, "unknown": 3, "healthy": 4}
        agent_summaries.sort(key=lambda a: state_priority.get(a["health_state"], 5))

        # Recent alerts count
        recent_alerts = [a for a in data["alerts"] if not a.get("acknowledged")]

        return SkillResult(
            success=True,
            message=f"Fleet: {len(data['agents'])} agents, {state_counts['healthy']} healthy, {state_counts['degraded']} degraded, {state_counts['dead']} dead",
            data={
                "agents_count": len(data["agents"]),
                "state_counts": state_counts,
                "agents": agent_summaries,
                "unacknowledged_alerts": len(recent_alerts),
                "fleet_stats": data["fleet_stats"],
                "total_heartbeats": total_heartbeats,
                "total_restarts": total_restarts,
            },
        )

    async def _deregister_agent(self, params: Dict) -> SkillResult:
        """Remove an agent from health monitoring."""
        agent_id = params.get("agent_id")
        if not agent_id:
            return SkillResult(success=False, message="agent_id is required")

        data = _load_data(self.data_path)
        if agent_id not in data["agents"]:
            return SkillResult(success=False, message=f"Agent '{agent_id}' not found")

        agent_name = data["agents"][agent_id]["agent_name"]
        del data["agents"][agent_id]
        _save_data(data, self.data_path)

        return SkillResult(
            success=True,
            message=f"Agent '{agent_name}' ({agent_id}) deregistered",
            data={"agent_id": agent_id},
        )

    async def _get_alerts(self, params: Dict) -> SkillResult:
        """Get recent health alerts."""
        data = _load_data(self.data_path)
        alerts = data["alerts"]

        # Filter by severity
        severity = params.get("severity")
        if severity:
            alerts = [a for a in alerts if a["severity"] == severity]

        # Filter by agent
        agent_id = params.get("agent_id")
        if agent_id:
            alerts = [a for a in alerts if a["agent_id"] == agent_id]

        # Limit
        limit = params.get("limit", 20)
        alerts = alerts[-limit:]

        return SkillResult(
            success=True,
            message=f"Found {len(alerts)} alerts",
            data={
                "alerts": alerts,
                "total_count": len(data["alerts"]),
            },
        )

    async def _acknowledge_alert(self, params: Dict) -> SkillResult:
        """Acknowledge an alert."""
        alert_id = params.get("alert_id")
        if not alert_id:
            return SkillResult(success=False, message="alert_id is required")

        data = _load_data(self.data_path)
        for alert in data["alerts"]:
            if alert["alert_id"] == alert_id:
                alert["acknowledged"] = True
                alert["acknowledged_at"] = datetime.now().isoformat()
                _save_data(data, self.data_path)
                return SkillResult(
                    success=True,
                    message=f"Alert {alert_id} acknowledged",
                    data={"alert_id": alert_id},
                )

        return SkillResult(success=False, message=f"Alert '{alert_id}' not found")

    async def _update_config(self, params: Dict) -> SkillResult:
        """Update health monitoring configuration."""
        data = _load_data(self.data_path)
        config = data["config"]
        updated = []

        for key in ["heartbeat_interval_seconds", "degraded_threshold_missed",
                     "dead_threshold_missed", "auto_restart_enabled", "max_restart_attempts"]:
            if key in params:
                config[key] = params[key]
                updated.append(key)

        if not updated:
            return SkillResult(success=False, message="No valid config parameters provided")

        _save_data(data, self.data_path)
        return SkillResult(
            success=True,
            message=f"Config updated: {', '.join(updated)}",
            data={"config": config, "updated": updated},
        )

    async def _trigger_restart(self, params: Dict) -> SkillResult:
        """Manually trigger restart of an agent."""
        agent_id = params.get("agent_id")
        if not agent_id:
            return SkillResult(success=False, message="agent_id is required")

        data = _load_data(self.data_path)
        if agent_id not in data["agents"]:
            return SkillResult(success=False, message=f"Agent '{agent_id}' not found")

        agent = data["agents"][agent_id]
        reason = params.get("reason", "manual restart")
        agent["restart_count"] += 1
        data["fleet_stats"]["total_restarts_triggered"] += 1

        self._create_alert(
            data, agent_id, agent["health_state"], "restarting",
            severity="warning",
            message=f"Manual restart: {reason} (attempt {agent['restart_count']})",
        )

        # If we have context, try to actually restart via ReplicationSkill
        restart_result = None
        if self.context:
            restart_result = await self.context.call_skill(
                "replication", "spawn_replica",
                {"config_override": agent.get("metadata", {}), "reason": f"restart: {reason}"}
            )

        _save_data(data, self.data_path)

        return SkillResult(
            success=True,
            message=f"Restart triggered for '{agent['agent_name']}' (attempt {agent['restart_count']}): {reason}",
            data={
                "agent_id": agent_id,
                "restart_count": agent["restart_count"],
                "restart_result": restart_result.data if restart_result else None,
            },
        )

    # --- Internal helpers ---

    def _compute_health_state(self, agent: Dict, config: Dict, now: datetime) -> str:
        """Compute health state based on heartbeat timing."""
        if not agent["last_heartbeat"]:
            return "unknown"

        last_beat = datetime.fromisoformat(agent["last_heartbeat"])
        interval = agent.get("expected_heartbeat_seconds", config["heartbeat_interval_seconds"])
        elapsed = (now - last_beat).total_seconds()
        missed = elapsed / max(interval, 1)

        if missed < 1.5:
            return "healthy"
        elif missed < config["degraded_threshold_missed"]:
            return "healthy"  # within tolerance
        elif missed < config["dead_threshold_missed"]:
            if agent.get("last_status") == "error":
                return "unresponsive"
            return "degraded"
        else:
            return "dead"

    def _create_alert(self, data: Dict, agent_id: str, from_state: str, to_state: str,
                      severity: str = None, message: str = None) -> Dict:
        """Create a health alert."""
        if severity is None:
            if to_state in ("dead", "unresponsive"):
                severity = "critical"
            elif to_state == "degraded":
                severity = "warning"
            else:
                severity = "info"

        alert = {
            "alert_id": f"alert_{uuid.uuid4().hex[:10]}",
            "agent_id": agent_id,
            "agent_name": data["agents"].get(agent_id, {}).get("agent_name", agent_id),
            "from_state": from_state,
            "to_state": to_state,
            "severity": severity,
            "message": message or f"Agent {agent_id} transitioned from {from_state} to {to_state}",
            "created_at": datetime.now().isoformat(),
            "acknowledged": False,
        }

        data["alerts"].append(alert)
        data["fleet_stats"]["total_alerts_fired"] += 1

        # Trim alerts to limit
        if len(data["alerts"]) > MAX_ALERTS:
            data["alerts"] = data["alerts"][-MAX_ALERTS:]

        return alert

    def _check_metric_anomalies(self, data: Dict, agent_id: str, metrics: Dict) -> int:
        """Check for anomalies in reported metrics and generate alerts."""
        alerts_count = 0

        # High error rate
        error_rate = metrics.get("error_rate", 0)
        if error_rate > 0.5:
            self._create_alert(
                data, agent_id, "healthy", "degraded",
                severity="warning",
                message=f"High error rate: {error_rate:.1%}",
            )
            data["agents"][agent_id]["health_state"] = "degraded"
            alerts_count += 1

        # Memory pressure
        memory_pct = metrics.get("memory_percent", 0)
        if memory_pct > 90:
            self._create_alert(
                data, agent_id, "healthy", "degraded",
                severity="warning",
                message=f"High memory usage: {memory_pct:.0f}%",
            )
            alerts_count += 1

        # CPU overload
        cpu_pct = metrics.get("cpu_percent", 0)
        if cpu_pct > 95:
            self._create_alert(
                data, agent_id, "healthy", "degraded",
                severity="warning",
                message=f"CPU overload: {cpu_pct:.0f}%",
            )
            alerts_count += 1

        # Task queue backup
        pending_tasks = metrics.get("pending_tasks", 0)
        if pending_tasks > 100:
            self._create_alert(
                data, agent_id, "healthy", "degraded",
                severity="info",
                message=f"Task queue backup: {pending_tasks} pending",
            )
            alerts_count += 1

        return alerts_count
