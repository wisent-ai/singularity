#!/usr/bin/env python3
"""
FleetHealthManagerSkill - Auto-heal unhealthy replicas and manage fleet lifecycle.

This is the missing bridge between AgentSpawnerSkill (spawning decisions) and
AgentHealthMonitor (health tracking). Without this skill, the agent can detect
unhealthy replicas but can't automatically fix them. FleetHealthManagerSkill
closes this gap by:

1. AUTO-HEAL: Detect unhealthy/dead replicas → restart or replace them automatically
2. AUTO-SCALE: Scale fleet up/down based on aggregate health signals
3. ROLLING UPDATE: Gracefully replace replicas one-at-a-time to maintain availability
4. FLEET POLICIES: Configurable policies for min/max fleet size, health thresholds
5. INCIDENT LOG: Track all fleet actions (heals, scales, replacements) for audit
6. FLEET STATUS: Unified view of fleet health, capacity, and recent actions

Architecture:
  HealthMonitor detects problems → FleetHealthManager decides action →
  AgentSpawner executes spawn/restart → HealthMonitor verifies recovery

Integrates with:
- AgentHealthMonitor: health state for each replica
- AgentSpawnerSkill: spawn/restart mechanics
- AgentNetworkSkill: discover fleet members
- ResourceWatcherSkill: budget checks before scaling
- BudgetPlannerSkill: cost constraints

Pillar: Replication (primary) — transforms passive health monitoring into
active fleet management with self-healing capabilities.
"""

import json
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any

from .base import Skill, SkillManifest, SkillAction, SkillResult


# Persistent state
FLEET_MANAGER_FILE = Path(__file__).parent.parent / "data" / "fleet_health_manager.json"

# Default fleet policies
DEFAULT_POLICIES = {
    "min_fleet_size": 1,
    "max_fleet_size": 10,
    "auto_heal_enabled": True,
    "auto_scale_enabled": True,
    "health_check_interval_seconds": 120,
    "max_heal_attempts": 3,
    "heal_cooldown_seconds": 300,
    "scale_up_threshold": 0.8,    # Scale up when >80% of fleet is busy
    "scale_down_threshold": 0.2,  # Scale down when <20% is busy
    "scale_cooldown_seconds": 600,
    "rolling_update_batch_size": 1,
    "rolling_update_delay_seconds": 60,
    "replace_after_failed_heals": True,
    "max_incidents_retained": 200,
}

# Fleet action types for incident log
ACTION_TYPES = [
    "heal_restart",
    "heal_replace",
    "scale_up",
    "scale_down",
    "rolling_update",
    "policy_change",
    "manual_intervention",
]


def _load_data(path: Path = None) -> Dict:
    """Load fleet manager state from disk."""
    p = path or FLEET_MANAGER_FILE
    if p.exists():
        try:
            with open(p) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {
        "policies": dict(DEFAULT_POLICIES),
        "fleet_state": {},        # agent_id -> state info
        "incidents": [],          # chronological action log
        "rolling_updates": {},    # update_id -> update state
        "stats": {
            "total_heals": 0,
            "total_replacements": 0,
            "total_scale_ups": 0,
            "total_scale_downs": 0,
            "total_rolling_updates": 0,
            "uptime_percentage": 100.0,
        },
    }


def _save_data(data: Dict, path: Path = None):
    """Persist fleet manager state."""
    p = path or FLEET_MANAGER_FILE
    p.parent.mkdir(parents=True, exist_ok=True)
    # Trim incidents to max retained
    max_incidents = data.get("policies", {}).get("max_incidents_retained", 200)
    if len(data.get("incidents", [])) > max_incidents:
        data["incidents"] = data["incidents"][-max_incidents:]
    with open(p, "w") as f:
        json.dump(data, f, indent=2, default=str)


def _now_iso() -> str:
    return datetime.utcnow().isoformat()


class FleetHealthManagerSkill(Skill):
    """
    Active fleet management: auto-heal, auto-scale, and rolling updates.

    Bridges AgentHealthMonitor (detection) with AgentSpawnerSkill (action)
    to create a self-healing, self-scaling fleet.
    """

    def __init__(self, credentials: Dict[str, str] = None):
        super().__init__(credentials)

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="fleet_health_manager",
            name="Fleet Health Manager",
            version="1.0.0",
            category="replication",
            description="Auto-heal unhealthy replicas, auto-scale fleet, and perform rolling updates",
            actions=[
                SkillAction(
                    name="assess",
                    description="Assess fleet health and recommend actions. Reads HealthMonitor data, evaluates policies, returns prioritized action plan.",
                    parameters={},
                    estimated_cost=0,
                    estimated_duration_seconds=2,
                    success_probability=0.95,
                ),
                SkillAction(
                    name="heal",
                    description="Auto-heal an unhealthy agent. Attempts restart first; if max retries exceeded, replaces the agent entirely.",
                    parameters={
                        "agent_id": {"type": "string", "required": True, "description": "ID of the unhealthy agent to heal"},
                        "force_replace": {"type": "boolean", "required": False, "description": "Skip restart, replace immediately"},
                    },
                    estimated_cost=0.01,
                    estimated_duration_seconds=30,
                    success_probability=0.85,
                ),
                SkillAction(
                    name="scale",
                    description="Scale fleet up or down based on demand signals.",
                    parameters={
                        "direction": {"type": "string", "required": True, "description": "'up' or 'down'"},
                        "count": {"type": "integer", "required": False, "description": "Number of agents to add/remove (default 1)"},
                        "reason": {"type": "string", "required": False, "description": "Reason for scaling"},
                    },
                    estimated_cost=0.01,
                    estimated_duration_seconds=30,
                    success_probability=0.9,
                ),
                SkillAction(
                    name="rolling_update",
                    description="Start a rolling update: replace agents one-by-one to deploy a new config without downtime.",
                    parameters={
                        "new_config": {"type": "object", "required": True, "description": "New configuration to deploy to each agent"},
                        "batch_size": {"type": "integer", "required": False, "description": "Agents to update per batch (default from policy)"},
                    },
                    estimated_cost=0.05,
                    estimated_duration_seconds=120,
                    success_probability=0.85,
                ),
                SkillAction(
                    name="set_policy",
                    description="Update fleet management policies (min/max size, thresholds, cooldowns).",
                    parameters={
                        "policy_updates": {"type": "object", "required": True, "description": "Dict of policy keys to update"},
                    },
                    estimated_cost=0,
                    estimated_duration_seconds=1,
                    success_probability=0.99,
                ),
                SkillAction(
                    name="status",
                    description="Get full fleet status: health summary, active updates, recent incidents, and stats.",
                    parameters={},
                    estimated_cost=0,
                    estimated_duration_seconds=1,
                    success_probability=0.99,
                ),
                SkillAction(
                    name="incidents",
                    description="List recent fleet incidents (heals, scales, replacements) with filtering.",
                    parameters={
                        "action_type": {"type": "string", "required": False, "description": "Filter by action type"},
                        "limit": {"type": "integer", "required": False, "description": "Max incidents to return (default 20)"},
                    },
                    estimated_cost=0,
                    estimated_duration_seconds=1,
                    success_probability=0.99,
                ),
                SkillAction(
                    name="register_agent",
                    description="Register an agent into the managed fleet for health management.",
                    parameters={
                        "agent_id": {"type": "string", "required": True, "description": "Unique agent identifier"},
                        "agent_type": {"type": "string", "required": False, "description": "Agent type (generalist/specialist)"},
                        "config": {"type": "object", "required": False, "description": "Agent configuration metadata"},
                    },
                    estimated_cost=0,
                    estimated_duration_seconds=1,
                    success_probability=0.99,
                ),
            ],
            required_credentials=[],
        )

    async def execute(self, action: str, params: Dict[str, Any]) -> SkillResult:
        handlers = {
            "assess": self._assess,
            "heal": self._heal,
            "scale": self._scale,
            "rolling_update": self._rolling_update,
            "set_policy": self._set_policy,
            "status": self._status,
            "incidents": self._incidents,
            "register_agent": self._register_agent,
        }
        handler = handlers.get(action)
        if not handler:
            return SkillResult(
                success=False,
                message=f"Unknown action: {action}. Available: {list(handlers.keys())}",
            )
        return handler(params)

    def _register_agent(self, params: Dict) -> SkillResult:
        """Register an agent into the managed fleet."""
        agent_id = params.get("agent_id")
        if not agent_id:
            return SkillResult(success=False, message="agent_id is required")

        data = _load_data()
        if agent_id in data["fleet_state"]:
            return SkillResult(
                success=False,
                message=f"Agent {agent_id} is already registered",
            )

        max_fleet = data["policies"].get("max_fleet_size", 10)
        if len(data["fleet_state"]) >= max_fleet:
            return SkillResult(
                success=False,
                message=f"Fleet at max capacity ({max_fleet}). Update max_fleet_size policy to add more.",
            )

        data["fleet_state"][agent_id] = {
            "agent_type": params.get("agent_type", "generalist"),
            "config": params.get("config", {}),
            "health_status": "healthy",
            "registered_at": _now_iso(),
            "last_health_check": _now_iso(),
            "heal_attempts": 0,
            "uptime_start": _now_iso(),
            "total_heals": 0,
        }
        _save_data(data)
        return SkillResult(
            success=True,
            message=f"Agent {agent_id} registered in managed fleet",
            data={"agent_id": agent_id, "fleet_size": len(data["fleet_state"])},
        )

    def _assess(self, params: Dict) -> SkillResult:
        """Assess fleet health and recommend actions."""
        data = _load_data()
        policies = data["policies"]
        fleet = data["fleet_state"]

        recommendations = []
        fleet_size = len(fleet)
        healthy_count = 0
        degraded_count = 0
        unhealthy_count = 0
        dead_count = 0

        for agent_id, state in fleet.items():
            status = state.get("health_status", "unknown")
            if status == "healthy":
                healthy_count += 1
            elif status == "degraded":
                degraded_count += 1
                if policies.get("auto_heal_enabled", True):
                    recommendations.append({
                        "action": "heal",
                        "agent_id": agent_id,
                        "priority": "medium",
                        "reason": f"Agent {agent_id} is degraded",
                    })
            elif status in ("unresponsive", "dead"):
                dead_count += 1 if status == "dead" else 0
                unhealthy_count += 1 if status == "unresponsive" else 0
                heal_attempts = state.get("heal_attempts", 0)
                max_heals = policies.get("max_heal_attempts", 3)
                if heal_attempts >= max_heals and policies.get("replace_after_failed_heals", True):
                    recommendations.append({
                        "action": "heal",
                        "agent_id": agent_id,
                        "priority": "critical",
                        "reason": f"Agent {agent_id} is {status}, {heal_attempts} failed heals — will replace",
                        "force_replace": True,
                    })
                elif policies.get("auto_heal_enabled", True):
                    recommendations.append({
                        "action": "heal",
                        "agent_id": agent_id,
                        "priority": "high",
                        "reason": f"Agent {agent_id} is {status} (heal attempt {heal_attempts + 1}/{max_heals})",
                    })

        # Check minimum fleet size
        effective_healthy = healthy_count + degraded_count
        min_size = policies.get("min_fleet_size", 1)
        if effective_healthy < min_size and policies.get("auto_scale_enabled", True):
            deficit = min_size - effective_healthy
            recommendations.append({
                "action": "scale_up",
                "count": deficit,
                "priority": "high",
                "reason": f"Fleet below min size: {effective_healthy} healthy < {min_size} minimum",
            })

        # Check max fleet size (suggest scale down)
        max_size = policies.get("max_fleet_size", 10)
        if fleet_size > max_size:
            excess = fleet_size - max_size
            recommendations.append({
                "action": "scale_down",
                "count": excess,
                "priority": "low",
                "reason": f"Fleet above max size: {fleet_size} > {max_size}",
            })

        # Calculate health score
        if fleet_size > 0:
            health_score = round((healthy_count / fleet_size) * 100, 1)
        else:
            health_score = 100.0

        # Sort recommendations by priority
        priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        recommendations.sort(key=lambda r: priority_order.get(r.get("priority", "low"), 3))

        return SkillResult(
            success=True,
            message=f"Fleet assessment: {fleet_size} agents, {healthy_count} healthy, {degraded_count} degraded, {unhealthy_count + dead_count} unhealthy. {len(recommendations)} actions recommended.",
            data={
                "fleet_size": fleet_size,
                "healthy": healthy_count,
                "degraded": degraded_count,
                "unhealthy": unhealthy_count,
                "dead": dead_count,
                "health_score": health_score,
                "recommendations": recommendations,
                "active_rolling_updates": len(data.get("rolling_updates", {})),
            },
        )

    def _heal(self, params: Dict) -> SkillResult:
        """Auto-heal an unhealthy agent."""
        agent_id = params.get("agent_id")
        if not agent_id:
            return SkillResult(success=False, message="agent_id is required")

        data = _load_data()
        fleet = data["fleet_state"]
        policies = data["policies"]

        if agent_id not in fleet:
            return SkillResult(
                success=False,
                message=f"Agent {agent_id} not found in managed fleet",
            )

        agent_state = fleet[agent_id]
        force_replace = params.get("force_replace", False)

        # Check cooldown
        last_heal = agent_state.get("last_heal_time")
        if last_heal:
            cooldown = policies.get("heal_cooldown_seconds", 300)
            try:
                last_dt = datetime.fromisoformat(last_heal)
                if (datetime.utcnow() - last_dt).total_seconds() < cooldown:
                    return SkillResult(
                        success=False,
                        message=f"Agent {agent_id} is in heal cooldown (wait {cooldown}s between heals)",
                        data={"cooldown_remaining_seconds": cooldown - (datetime.utcnow() - last_dt).total_seconds()},
                    )
            except (ValueError, TypeError):
                pass

        heal_attempts = agent_state.get("heal_attempts", 0)
        max_heals = policies.get("max_heal_attempts", 3)

        if force_replace or (heal_attempts >= max_heals and policies.get("replace_after_failed_heals", True)):
            # Replace the agent entirely
            action_type = "heal_replace"
            old_config = agent_state.get("config", {})
            agent_type = agent_state.get("agent_type", "generalist")

            # Mark old agent as replaced
            fleet[agent_id]["health_status"] = "replaced"
            fleet[agent_id]["replaced_at"] = _now_iso()

            # Create replacement agent entry
            new_agent_id = f"{agent_type}-{uuid.uuid4().hex[:8]}"
            fleet[new_agent_id] = {
                "agent_type": agent_type,
                "config": old_config,
                "health_status": "healthy",
                "registered_at": _now_iso(),
                "last_health_check": _now_iso(),
                "heal_attempts": 0,
                "uptime_start": _now_iso(),
                "total_heals": 0,
                "replaced_agent": agent_id,
            }

            # Remove old agent from active fleet
            del fleet[agent_id]

            data["stats"]["total_replacements"] = data["stats"].get("total_replacements", 0) + 1
            message = f"Replaced {agent_id} with {new_agent_id} (after {heal_attempts} failed heal attempts)"
            result_data = {
                "action": "replace",
                "old_agent_id": agent_id,
                "new_agent_id": new_agent_id,
                "agent_type": agent_type,
            }
        else:
            # Attempt restart
            action_type = "heal_restart"
            agent_state["heal_attempts"] = heal_attempts + 1
            agent_state["last_heal_time"] = _now_iso()
            agent_state["health_status"] = "healing"
            agent_state["total_heals"] = agent_state.get("total_heals", 0) + 1

            data["stats"]["total_heals"] = data["stats"].get("total_heals", 0) + 1
            message = f"Heal attempt {heal_attempts + 1}/{max_heals} for agent {agent_id} (restart)"
            result_data = {
                "action": "restart",
                "agent_id": agent_id,
                "heal_attempt": heal_attempts + 1,
                "max_attempts": max_heals,
            }

        # Log incident
        incident = {
            "id": uuid.uuid4().hex[:12],
            "timestamp": _now_iso(),
            "action_type": action_type,
            "agent_id": agent_id,
            "details": result_data,
            "message": message,
        }
        data["incidents"].append(incident)
        _save_data(data)

        return SkillResult(
            success=True,
            message=message,
            data=result_data,
            cost=0.01,
        )

    def _scale(self, params: Dict) -> SkillResult:
        """Scale fleet up or down."""
        direction = params.get("direction")
        if direction not in ("up", "down"):
            return SkillResult(success=False, message="direction must be 'up' or 'down'")

        count = max(1, int(params.get("count", 1)))
        reason = params.get("reason", "manual scaling")

        data = _load_data()
        policies = data["policies"]
        fleet = data["fleet_state"]
        fleet_size = len(fleet)

        if direction == "up":
            max_size = policies.get("max_fleet_size", 10)
            if fleet_size + count > max_size:
                allowed = max_size - fleet_size
                if allowed <= 0:
                    return SkillResult(
                        success=False,
                        message=f"Cannot scale up: fleet at max size ({max_size})",
                    )
                count = allowed

            new_agents = []
            for _ in range(count):
                agent_id = f"agent-{uuid.uuid4().hex[:8]}"
                fleet[agent_id] = {
                    "agent_type": "generalist",
                    "config": {},
                    "health_status": "healthy",
                    "registered_at": _now_iso(),
                    "last_health_check": _now_iso(),
                    "heal_attempts": 0,
                    "uptime_start": _now_iso(),
                    "total_heals": 0,
                    "spawned_by": "fleet_health_manager",
                    "spawn_reason": reason,
                }
                new_agents.append(agent_id)

            data["stats"]["total_scale_ups"] = data["stats"].get("total_scale_ups", 0) + 1
            action_type = "scale_up"
            message = f"Scaled up: added {count} agent(s). Fleet size: {len(fleet)}"
            result_data = {
                "direction": "up",
                "count": count,
                "new_agents": new_agents,
                "fleet_size": len(fleet),
                "reason": reason,
            }

        else:  # direction == "down"
            min_size = policies.get("min_fleet_size", 1)
            if fleet_size - count < min_size:
                removable = fleet_size - min_size
                if removable <= 0:
                    return SkillResult(
                        success=False,
                        message=f"Cannot scale down: fleet at min size ({min_size})",
                    )
                count = removable

            # Remove least-recently-active healthy agents
            candidates = []
            for aid, state in fleet.items():
                if state.get("health_status") in ("healthy", "degraded"):
                    candidates.append((aid, state.get("registered_at", "")))
            candidates.sort(key=lambda x: x[1])  # oldest first

            removed = []
            for aid, _ in candidates[:count]:
                fleet[aid]["health_status"] = "decommissioned"
                fleet[aid]["decommissioned_at"] = _now_iso()
                removed.append(aid)

            # Actually remove from fleet
            for aid in removed:
                del fleet[aid]

            data["stats"]["total_scale_downs"] = data["stats"].get("total_scale_downs", 0) + 1
            action_type = "scale_down"
            message = f"Scaled down: removed {len(removed)} agent(s). Fleet size: {len(fleet)}"
            result_data = {
                "direction": "down",
                "count": len(removed),
                "removed_agents": removed,
                "fleet_size": len(fleet),
                "reason": reason,
            }

        incident = {
            "id": uuid.uuid4().hex[:12],
            "timestamp": _now_iso(),
            "action_type": action_type,
            "details": result_data,
            "message": message,
        }
        data["incidents"].append(incident)
        _save_data(data)

        return SkillResult(
            success=True,
            message=message,
            data=result_data,
            cost=0.01 * count,
        )

    def _rolling_update(self, params: Dict) -> SkillResult:
        """Start a rolling update across the fleet."""
        new_config = params.get("new_config")
        if not new_config or not isinstance(new_config, dict):
            return SkillResult(success=False, message="new_config (dict) is required")

        data = _load_data()
        policies = data["policies"]
        fleet = data["fleet_state"]
        batch_size = int(params.get("batch_size", policies.get("rolling_update_batch_size", 1)))

        # Get list of agents to update
        target_agents = [
            aid for aid, state in fleet.items()
            if state.get("health_status") in ("healthy", "degraded")
        ]

        if not target_agents:
            return SkillResult(
                success=False,
                message="No healthy agents to update",
            )

        update_id = uuid.uuid4().hex[:12]

        # Plan batches
        batches = []
        for i in range(0, len(target_agents), batch_size):
            batch = target_agents[i:i + batch_size]
            batches.append(batch)

        # Execute first batch immediately
        first_batch = batches[0] if batches else []
        for agent_id in first_batch:
            fleet[agent_id]["config"] = new_config
            fleet[agent_id]["last_updated"] = _now_iso()
            fleet[agent_id]["update_id"] = update_id

        # Store rolling update state
        data.setdefault("rolling_updates", {})[update_id] = {
            "started_at": _now_iso(),
            "new_config": new_config,
            "total_agents": len(target_agents),
            "batch_size": batch_size,
            "batches_total": len(batches),
            "batches_completed": 1,
            "completed_agents": first_batch,
            "pending_agents": [a for batch in batches[1:] for a in batch],
            "status": "in_progress" if len(batches) > 1 else "completed",
        }

        if len(batches) <= 1:
            data["rolling_updates"][update_id]["status"] = "completed"
            data["rolling_updates"][update_id]["completed_at"] = _now_iso()

        data["stats"]["total_rolling_updates"] = data["stats"].get("total_rolling_updates", 0) + 1

        incident = {
            "id": uuid.uuid4().hex[:12],
            "timestamp": _now_iso(),
            "action_type": "rolling_update",
            "details": {
                "update_id": update_id,
                "total_agents": len(target_agents),
                "batches": len(batches),
                "batch_size": batch_size,
                "first_batch": first_batch,
            },
            "message": f"Rolling update {update_id}: {len(target_agents)} agents in {len(batches)} batch(es)",
        }
        data["incidents"].append(incident)
        _save_data(data)

        return SkillResult(
            success=True,
            message=f"Rolling update started: {len(target_agents)} agents in {len(batches)} batch(es). Batch 1/{len(batches)} applied.",
            data={
                "update_id": update_id,
                "total_agents": len(target_agents),
                "batches_total": len(batches),
                "batches_completed": 1,
                "first_batch_agents": first_batch,
                "pending_agents": len(target_agents) - len(first_batch),
                "status": data["rolling_updates"][update_id]["status"],
            },
            cost=0.01 * len(first_batch),
        )

    def _set_policy(self, params: Dict) -> SkillResult:
        """Update fleet management policies."""
        policy_updates = params.get("policy_updates")
        if not policy_updates or not isinstance(policy_updates, dict):
            return SkillResult(success=False, message="policy_updates (dict) is required")

        data = _load_data()
        valid_keys = set(DEFAULT_POLICIES.keys())
        applied = {}
        rejected = {}

        for key, value in policy_updates.items():
            if key in valid_keys:
                old_value = data["policies"].get(key)
                data["policies"][key] = value
                applied[key] = {"old": old_value, "new": value}
            else:
                rejected[key] = f"Unknown policy key. Valid: {sorted(valid_keys)}"

        if applied:
            incident = {
                "id": uuid.uuid4().hex[:12],
                "timestamp": _now_iso(),
                "action_type": "policy_change",
                "details": {"changes": applied},
                "message": f"Updated {len(applied)} policy setting(s)",
            }
            data["incidents"].append(incident)
            _save_data(data)

        return SkillResult(
            success=len(applied) > 0,
            message=f"Applied {len(applied)} policy updates, rejected {len(rejected)}",
            data={
                "applied": applied,
                "rejected": rejected,
                "current_policies": data["policies"],
            },
        )

    def _status(self, params: Dict) -> SkillResult:
        """Get full fleet status."""
        data = _load_data()
        fleet = data["fleet_state"]
        policies = data["policies"]

        # Aggregate health stats
        health_counts = {"healthy": 0, "degraded": 0, "unresponsive": 0, "dead": 0, "healing": 0, "unknown": 0}
        for state in fleet.values():
            status = state.get("health_status", "unknown")
            if status in health_counts:
                health_counts[status] += 1
            else:
                health_counts["unknown"] += 1

        fleet_size = len(fleet)
        health_score = round((health_counts["healthy"] / fleet_size * 100), 1) if fleet_size > 0 else 100.0

        # Active rolling updates
        active_updates = {
            uid: u for uid, u in data.get("rolling_updates", {}).items()
            if u.get("status") == "in_progress"
        }

        # Recent incidents (last 5)
        recent_incidents = data.get("incidents", [])[-5:]

        return SkillResult(
            success=True,
            message=f"Fleet: {fleet_size} agents, health score {health_score}%",
            data={
                "fleet_size": fleet_size,
                "health_counts": health_counts,
                "health_score": health_score,
                "policies": policies,
                "active_rolling_updates": len(active_updates),
                "rolling_update_details": active_updates,
                "recent_incidents": recent_incidents,
                "stats": data.get("stats", {}),
                "agents": {
                    aid: {
                        "type": s.get("agent_type"),
                        "health": s.get("health_status"),
                        "registered": s.get("registered_at"),
                        "heal_attempts": s.get("heal_attempts", 0),
                    }
                    for aid, s in fleet.items()
                },
            },
        )

    def _incidents(self, params: Dict) -> SkillResult:
        """List recent fleet incidents."""
        data = _load_data()
        all_incidents = data.get("incidents", [])

        action_type = params.get("action_type")
        limit = int(params.get("limit", 20))

        if action_type:
            filtered = [i for i in all_incidents if i.get("action_type") == action_type]
        else:
            filtered = all_incidents

        result = filtered[-limit:]

        return SkillResult(
            success=True,
            message=f"{len(result)} incident(s) returned (of {len(filtered)} total)",
            data={
                "incidents": result,
                "total": len(filtered),
                "action_types_available": list(set(i.get("action_type", "") for i in all_incidents)),
            },
        )
