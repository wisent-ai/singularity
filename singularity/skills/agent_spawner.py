#!/usr/bin/env python3
"""
AgentSpawnerSkill - Autonomous replication decision-making and lifecycle management.

This is the "brain" of the Replication pillar. While ReplicationSkill handles
low-level snapshot/spawn mechanics and AgentNetworkSkill handles discovery/RPC,
this skill makes the HIGH-LEVEL decisions:

1. WHEN to spawn - evaluates workload, capability gaps, and budget to decide
2. WHAT to spawn - selects optimal configuration (specialist vs generalist)
3. HOW to manage - monitors replica health, retires underperformers, scales down
4. WHY to replicate - maintains a spawning policy with configurable triggers

Trigger Types:
- workload: Queue depth exceeds threshold → spawn to handle overflow
- capability_gap: Task requires skill no current agent has → spawn specialist
- revenue: Demand for a service exceeds capacity → spawn to scale
- resilience: Fewer than N agents running → spawn for fault tolerance
- scheduled: Periodic spawning on a schedule (e.g., daily capacity expansion)

Integrates with:
- ReplicationSkill: actual spawning mechanics
- AgentNetworkSkill: discover existing agents to avoid over-spawning
- SelfAssessmentSkill: capability profiles for gap detection
- ResourceWatcherSkill: budget checks before spawning
- TaskDelegationSkill: workload metrics for demand-based spawning
- BudgetPlannerSkill: cost constraints

Pillar: Replication (primary), Goal Setting (spawning strategy)
"""

import json
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

from .base import Skill, SkillManifest, SkillAction, SkillResult


# --- Spawn Policy Defaults ---

DEFAULT_POLICIES = {
    "workload": {
        "enabled": True,
        "description": "Spawn when task queue exceeds threshold",
        "trigger": "queue_depth",
        "threshold": 10,
        "cooldown_seconds": 600,
        "max_spawns_per_hour": 3,
        "spawn_config": {"type": "generalist"},
    },
    "capability_gap": {
        "enabled": True,
        "description": "Spawn specialist when no agent has required capability",
        "trigger": "missing_capability",
        "cooldown_seconds": 1800,
        "max_spawns_per_hour": 2,
        "spawn_config": {"type": "specialist"},
    },
    "resilience": {
        "enabled": True,
        "description": "Maintain minimum agent count for fault tolerance",
        "trigger": "min_agents",
        "min_agents": 2,
        "cooldown_seconds": 300,
        "max_spawns_per_hour": 2,
        "spawn_config": {"type": "generalist"},
    },
    "revenue": {
        "enabled": False,
        "description": "Scale up when service demand exceeds capacity",
        "trigger": "demand_exceeds_capacity",
        "demand_threshold": 0.8,  # 80% utilization
        "cooldown_seconds": 900,
        "max_spawns_per_hour": 2,
        "spawn_config": {"type": "service_worker"},
    },
}

DEFAULT_SPAWN_BUDGET = 5.0  # max $ to spend on spawning per day
DEFAULT_MAX_REPLICAS = 10   # absolute cap on total replicas


class AgentSpawnerSkill(Skill):
    """
    Autonomous replication orchestrator.

    Makes high-level decisions about when, what, and how to spawn
    new agent replicas based on configurable policies.
    """

    def __init__(self, credentials: Dict = None):
        super().__init__(credentials)
        self._data_dir = Path(__file__).parent.parent / "data"
        self._state_file = self._data_dir / "agent_spawner.json"
        self._state = self._load_state()

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="agent_spawner",
            name="Agent Spawner",
            version="1.0.0",
            category="replication",
            description="Autonomous replication decision-making - decides when/what/how to spawn replicas",
            actions=[
                SkillAction(
                    name="evaluate",
                    description="Evaluate all spawn policies against current state and return recommendations",
                    parameters={
                        "dry_run": {
                            "type": "boolean",
                            "required": False,
                            "description": "If true, only report what would be spawned (default true)",
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="spawn",
                    description="Spawn a new agent replica with given configuration",
                    parameters={
                        "name": {
                            "type": "string",
                            "required": True,
                            "description": "Name for the new replica",
                        },
                        "type": {
                            "type": "string",
                            "required": False,
                            "description": "Replica type: generalist, specialist, service_worker (default: generalist)",
                        },
                        "skills": {
                            "type": "array",
                            "required": False,
                            "description": "Skills to install on the replica (for specialist type)",
                        },
                        "budget": {
                            "type": "number",
                            "required": False,
                            "description": "Budget allocation for this replica in USD",
                        },
                        "reason": {
                            "type": "string",
                            "required": False,
                            "description": "Why this replica is being spawned",
                        },
                    },
                    estimated_cost=0.10,
                ),
                SkillAction(
                    name="retire",
                    description="Retire (stop) a replica that is no longer needed",
                    parameters={
                        "replica_id": {
                            "type": "string",
                            "required": True,
                            "description": "ID of the replica to retire",
                        },
                        "reason": {
                            "type": "string",
                            "required": False,
                            "description": "Reason for retirement",
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="fleet",
                    description="View the current fleet of managed replicas",
                    parameters={
                        "include_retired": {
                            "type": "boolean",
                            "required": False,
                            "description": "Include retired replicas (default false)",
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="policies",
                    description="View or update spawn policies",
                    parameters={
                        "policy_id": {
                            "type": "string",
                            "required": False,
                            "description": "Specific policy to view/update",
                        },
                        "enabled": {
                            "type": "boolean",
                            "required": False,
                            "description": "Enable/disable a policy",
                        },
                        "threshold": {
                            "type": "number",
                            "required": False,
                            "description": "Update threshold value",
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="configure",
                    description="Configure global spawner settings",
                    parameters={
                        "max_replicas": {
                            "type": "integer",
                            "required": False,
                            "description": f"Maximum total replicas (default {DEFAULT_MAX_REPLICAS})",
                        },
                        "daily_budget": {
                            "type": "number",
                            "required": False,
                            "description": f"Daily spawn budget in USD (default {DEFAULT_SPAWN_BUDGET})",
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="history",
                    description="View spawning and retirement history",
                    parameters={
                        "limit": {
                            "type": "integer",
                            "required": False,
                            "description": "Max entries to return (default 20)",
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
            "evaluate": self._evaluate,
            "spawn": self._spawn,
            "retire": self._retire,
            "fleet": self._fleet,
            "policies": self._policies,
            "configure": self._configure,
            "history": self._history,
        }
        handler = handlers.get(action)
        if not handler:
            return SkillResult(success=False, message=f"Unknown action: {action}")
        return await handler(params)

    # --- Actions ---

    async def _evaluate(self, params: Dict) -> SkillResult:
        """Evaluate spawn policies and recommend actions."""
        dry_run = params.get("dry_run", True)
        recommendations = []
        active_replicas = self._get_active_replicas()
        active_count = len(active_replicas)
        config = self._state.get("config", {})
        max_replicas = config.get("max_replicas", DEFAULT_MAX_REPLICAS)

        if active_count >= max_replicas:
            return SkillResult(
                success=True,
                message=f"At replica cap ({active_count}/{max_replicas}). No spawning recommended.",
                data={"recommendations": [], "at_cap": True, "active_count": active_count},
            )

        policies = self._state.get("policies", DEFAULT_POLICIES)
        now = time.time()

        for pid, policy in policies.items():
            if not policy.get("enabled", False):
                continue

            # Check cooldown
            last_spawn = self._get_last_spawn_time(pid)
            cooldown = policy.get("cooldown_seconds", 600)
            if last_spawn and (now - last_spawn) < cooldown:
                continue

            # Check hourly spawn limit
            hourly_spawns = self._count_recent_spawns(pid, 3600)
            max_hourly = policy.get("max_spawns_per_hour", 3)
            if hourly_spawns >= max_hourly:
                continue

            # Evaluate trigger
            triggered, reason = await self._evaluate_trigger(pid, policy, active_replicas)
            if triggered:
                spawn_config = policy.get("spawn_config", {})
                recommendations.append({
                    "policy": pid,
                    "reason": reason,
                    "spawn_type": spawn_config.get("type", "generalist"),
                    "priority": self._policy_priority(pid),
                })

        # Sort by priority
        recommendations.sort(key=lambda r: r["priority"])

        if not dry_run and recommendations:
            # Execute top recommendation
            top = recommendations[0]
            spawn_result = await self._spawn({
                "name": f"auto-{top['policy']}-{uuid.uuid4().hex[:6]}",
                "type": top["spawn_type"],
                "reason": f"[auto:{top['policy']}] {top['reason']}",
            })
            return SkillResult(
                success=spawn_result.success,
                message=f"Auto-evaluated: {len(recommendations)} triggers fired. {spawn_result.message}",
                data={
                    "recommendations": recommendations,
                    "auto_spawned": spawn_result.success,
                    "spawn_result": spawn_result.data,
                },
            )

        return SkillResult(
            success=True,
            message=f"Evaluated {len(policies)} policies: {len(recommendations)} recommend spawning",
            data={
                "recommendations": recommendations,
                "active_count": active_count,
                "max_replicas": max_replicas,
                "dry_run": dry_run,
            },
        )

    async def _spawn(self, params: Dict) -> SkillResult:
        """Spawn a new replica."""
        name = params.get("name", "").strip()
        spawn_type = params.get("type", "generalist")
        skills = params.get("skills", [])
        budget = params.get("budget", 1.0)
        reason = params.get("reason", "manual spawn")

        if not name:
            return SkillResult(success=False, message="name is required")

        # Check limits
        active_replicas = self._get_active_replicas()
        config = self._state.get("config", {})
        max_replicas = config.get("max_replicas", DEFAULT_MAX_REPLICAS)

        if len(active_replicas) >= max_replicas:
            return SkillResult(
                success=False,
                message=f"At replica cap ({len(active_replicas)}/{max_replicas}). Retire one first.",
            )

        # Check daily budget
        daily_budget = config.get("daily_budget", DEFAULT_SPAWN_BUDGET)
        daily_spent = self._get_daily_spend()
        if daily_spent + budget > daily_budget:
            return SkillResult(
                success=False,
                message=f"Would exceed daily budget (${daily_spent:.2f} + ${budget:.2f} > ${daily_budget:.2f})",
            )

        # Try to spawn via ReplicationSkill
        replica_id = f"replica_{uuid.uuid4().hex[:8]}"
        spawn_success = False
        spawn_message = ""

        if self.context:
            try:
                # First try snapshot + spawn via ReplicationSkill
                result = await self.context.call_skill("replication", "spawn", {
                    "name": name,
                    "budget": budget,
                    "mutations": {"type": spawn_type, "skills": skills},
                })
                if result.success:
                    spawn_success = True
                    replica_id = result.data.get("replica_id", replica_id)
                    spawn_message = result.message
            except Exception as e:
                spawn_message = f"ReplicationSkill unavailable: {e}"

        if not spawn_success:
            # Record as "pending" - will need manual or external spawning
            spawn_message = spawn_message or "Recorded spawn request (no ReplicationSkill available)"

        # Record replica
        replica_record = {
            "id": replica_id,
            "name": name,
            "type": spawn_type,
            "skills": skills,
            "budget": budget,
            "reason": reason,
            "status": "running" if spawn_success else "pending",
            "spawned_at": datetime.now().isoformat(),
            "spawned_by": "agent_spawner",
            "health": "unknown",
            "last_health_check": None,
        }

        replicas = self._state.get("replicas", {})
        replicas[replica_id] = replica_record
        self._state["replicas"] = replicas

        # Record in history
        self._add_history_entry("spawn", {
            "replica_id": replica_id,
            "name": name,
            "type": spawn_type,
            "reason": reason,
            "budget": budget,
            "success": spawn_success,
        })

        self._save_state()

        # Try to register on agent network
        if self.context and spawn_success:
            try:
                await self.context.call_skill("agent_network", "register", {
                    "name": name,
                    "capabilities": skills,
                    "endpoint": f"local://{replica_id}",
                })
            except Exception:
                pass

        return SkillResult(
            success=True,
            message=f"{'Spawned' if spawn_success else 'Recorded'} replica '{name}' ({spawn_type}): {spawn_message}",
            data=replica_record,
        )

    async def _retire(self, params: Dict) -> SkillResult:
        """Retire a replica."""
        replica_id = params.get("replica_id", "").strip()
        reason = params.get("reason", "manual retirement")

        if not replica_id:
            return SkillResult(success=False, message="replica_id is required")

        replicas = self._state.get("replicas", {})
        replica = replicas.get(replica_id)
        if not replica:
            return SkillResult(success=False, message=f"Replica not found: {replica_id}")

        if replica.get("status") in ("retired", "terminated"):
            return SkillResult(success=False, message=f"Replica already retired")

        # Try to stop via ReplicationSkill
        stop_success = False
        if self.context and replica.get("status") == "running":
            try:
                result = await self.context.call_skill("replication", "terminate", {
                    "replica_id": replica_id,
                })
                stop_success = result.success
            except Exception:
                pass

        replica["status"] = "retired"
        replica["retired_at"] = datetime.now().isoformat()
        replica["retire_reason"] = reason
        self._state["replicas"] = replicas

        self._add_history_entry("retire", {
            "replica_id": replica_id,
            "name": replica.get("name", ""),
            "reason": reason,
            "was_running": stop_success,
        })

        self._save_state()

        return SkillResult(
            success=True,
            message=f"Retired replica '{replica.get('name', replica_id)}': {reason}",
            data={"replica_id": replica_id, "status": "retired"},
        )

    async def _fleet(self, params: Dict) -> SkillResult:
        """View managed replicas."""
        include_retired = params.get("include_retired", False)
        replicas = self._state.get("replicas", {})

        fleet = []
        for rid, replica in replicas.items():
            if not include_retired and replica.get("status") in ("retired", "terminated"):
                continue
            fleet.append(replica)

        active = [r for r in fleet if r.get("status") in ("running", "pending")]
        total_budget = sum(r.get("budget", 0) for r in active)

        return SkillResult(
            success=True,
            message=f"Fleet: {len(active)} active, {len(fleet)} total replicas (${total_budget:.2f} allocated)",
            data={
                "fleet": fleet,
                "active_count": len(active),
                "total_count": len(fleet),
                "total_budget": total_budget,
            },
        )

    async def _policies(self, params: Dict) -> SkillResult:
        """View or update spawn policies."""
        policy_id = params.get("policy_id", "").strip()
        enabled = params.get("enabled")
        threshold = params.get("threshold")

        policies = self._state.get("policies", dict(DEFAULT_POLICIES))

        if policy_id:
            if policy_id not in policies:
                return SkillResult(
                    success=False,
                    message=f"Unknown policy: {policy_id}. Available: {list(policies.keys())}",
                )

            changes = []
            if enabled is not None:
                policies[policy_id]["enabled"] = enabled
                changes.append(f"enabled={enabled}")
            if threshold is not None:
                policies[policy_id]["threshold"] = threshold
                changes.append(f"threshold={threshold}")

            if changes:
                self._state["policies"] = policies
                self._save_state()
                return SkillResult(
                    success=True,
                    message=f"Updated policy '{policy_id}': {', '.join(changes)}",
                    data={"policy": policies[policy_id]},
                )

            return SkillResult(
                success=True,
                message=f"Policy '{policy_id}'",
                data={"policy": policies[policy_id]},
            )

        # List all policies
        policy_list = []
        for pid, p in policies.items():
            policy_list.append({
                "id": pid,
                "enabled": p.get("enabled", False),
                "description": p.get("description", ""),
                "trigger": p.get("trigger", ""),
                "cooldown_seconds": p.get("cooldown_seconds", 0),
                "max_spawns_per_hour": p.get("max_spawns_per_hour", 0),
            })

        active_count = sum(1 for p in policy_list if p["enabled"])
        return SkillResult(
            success=True,
            message=f"{len(policy_list)} policies ({active_count} enabled)",
            data={"policies": policy_list},
        )

    async def _configure(self, params: Dict) -> SkillResult:
        """Configure global settings."""
        config = self._state.get("config", {})
        changes = []

        max_replicas = params.get("max_replicas")
        if max_replicas is not None:
            max_replicas = max(1, min(50, int(max_replicas)))
            config["max_replicas"] = max_replicas
            changes.append(f"max_replicas={max_replicas}")

        daily_budget = params.get("daily_budget")
        if daily_budget is not None:
            daily_budget = max(0.0, float(daily_budget))
            config["daily_budget"] = daily_budget
            changes.append(f"daily_budget=${daily_budget:.2f}")

        if not changes:
            return SkillResult(
                success=True,
                message="Current configuration",
                data={"config": {
                    "max_replicas": config.get("max_replicas", DEFAULT_MAX_REPLICAS),
                    "daily_budget": config.get("daily_budget", DEFAULT_SPAWN_BUDGET),
                }},
            )

        self._state["config"] = config
        self._save_state()

        return SkillResult(
            success=True,
            message=f"Updated config: {', '.join(changes)}",
            data={"config": config},
        )

    async def _history(self, params: Dict) -> SkillResult:
        """View spawning history."""
        limit = params.get("limit", 20)
        history = self._state.get("history", [])
        recent = history[-limit:][::-1]  # most recent first

        return SkillResult(
            success=True,
            message=f"{len(recent)} history entries (of {len(history)} total)",
            data={"history": recent, "total": len(history)},
        )

    # --- Trigger Evaluation ---

    async def _evaluate_trigger(self, policy_id: str, policy: Dict,
                                  active_replicas: List[Dict]) -> tuple:
        """Evaluate a single policy trigger. Returns (triggered: bool, reason: str)."""
        trigger = policy.get("trigger", "")

        if trigger == "queue_depth":
            return await self._check_queue_depth(policy)
        elif trigger == "missing_capability":
            return await self._check_capability_gap(policy, active_replicas)
        elif trigger == "min_agents":
            return self._check_min_agents(policy, active_replicas)
        elif trigger == "demand_exceeds_capacity":
            return await self._check_demand(policy)

        return False, ""

    async def _check_queue_depth(self, policy: Dict) -> tuple:
        """Check if task queue depth exceeds threshold."""
        threshold = policy.get("threshold", 10)

        if self.context:
            try:
                result = await self.context.call_skill("task_queue", "stats", {})
                if result.success:
                    queue_depth = result.data.get("pending", 0)
                    if queue_depth > threshold:
                        return True, f"Queue depth {queue_depth} > threshold {threshold}"
            except Exception:
                pass

        # Fallback: check task queue file
        try:
            tq_file = self._data_dir / "task_queue.json"
            if tq_file.exists():
                data = json.loads(tq_file.read_text())
                pending = sum(1 for t in data.get("tasks", [])
                              if t.get("status") == "pending")
                if pending > threshold:
                    return True, f"Queue depth {pending} > threshold {threshold}"
        except Exception:
            pass

        return False, ""

    async def _check_capability_gap(self, policy: Dict, active_replicas: List[Dict]) -> tuple:
        """Check if there's a capability gap no current agent can fill."""
        if self.context:
            try:
                result = await self.context.call_skill("self_assessment", "gaps", {})
                if result.success:
                    gaps = result.data.get("gaps", [])
                    critical_gaps = [g for g in gaps if g.get("impact_score", 0) > 0.7]
                    if critical_gaps:
                        top_gap = critical_gaps[0]
                        return True, f"Critical capability gap: {top_gap.get('skill', 'unknown')} (impact={top_gap.get('impact_score', 0):.2f})"
            except Exception:
                pass

        return False, ""

    def _check_min_agents(self, policy: Dict, active_replicas: List[Dict]) -> tuple:
        """Check if we have minimum agent count."""
        min_agents = policy.get("min_agents", 2)
        current = len(active_replicas) + 1  # +1 for self
        if current < min_agents:
            return True, f"Only {current} agents running (minimum: {min_agents})"
        return False, ""

    async def _check_demand(self, policy: Dict) -> tuple:
        """Check if service demand exceeds capacity."""
        threshold = policy.get("demand_threshold", 0.8)

        if self.context:
            try:
                result = await self.context.call_skill("usage_tracking", "analytics", {})
                if result.success:
                    utilization = result.data.get("utilization", 0)
                    if utilization > threshold:
                        return True, f"Utilization {utilization:.0%} > threshold {threshold:.0%}"
            except Exception:
                pass

        return False, ""

    # --- Helpers ---

    def _get_active_replicas(self) -> List[Dict]:
        """Get list of active (non-retired) replicas."""
        replicas = self._state.get("replicas", {})
        return [r for r in replicas.values()
                if r.get("status") in ("running", "pending")]

    def _get_last_spawn_time(self, policy_id: str) -> Optional[float]:
        """Get timestamp of last spawn for a policy."""
        history = self._state.get("history", [])
        for entry in reversed(history):
            if (entry.get("action") == "spawn" and
                entry.get("data", {}).get("reason", "").startswith(f"[auto:{policy_id}]")):
                try:
                    dt = datetime.fromisoformat(entry["timestamp"])
                    return dt.timestamp()
                except (ValueError, KeyError):
                    pass
        return None

    def _count_recent_spawns(self, policy_id: str, window_seconds: float) -> int:
        """Count spawns for a policy within a time window."""
        cutoff = time.time() - window_seconds
        count = 0
        for entry in self._state.get("history", []):
            if entry.get("action") != "spawn":
                continue
            try:
                ts = datetime.fromisoformat(entry["timestamp"]).timestamp()
                if ts > cutoff:
                    reason = entry.get("data", {}).get("reason", "")
                    if f"[auto:{policy_id}]" in reason:
                        count += 1
            except (ValueError, KeyError):
                pass
        return count

    def _get_daily_spend(self) -> float:
        """Calculate total spawn budget allocated today."""
        today = datetime.now().date().isoformat()
        total = 0.0
        for entry in self._state.get("history", []):
            if entry.get("action") != "spawn":
                continue
            if entry.get("timestamp", "").startswith(today):
                total += entry.get("data", {}).get("budget", 0)
        return total

    def _policy_priority(self, policy_id: str) -> int:
        """Priority for policy recommendations (lower = higher priority)."""
        order = ["resilience", "workload", "capability_gap", "revenue"]
        try:
            return order.index(policy_id)
        except ValueError:
            return 99

    def _add_history_entry(self, action: str, data: Dict):
        """Add a history entry."""
        history = self._state.get("history", [])
        history.append({
            "action": action,
            "timestamp": datetime.now().isoformat(),
            "data": data,
        })
        # Trim history
        if len(history) > 500:
            history = history[-500:]
        self._state["history"] = history

    def _load_state(self) -> Dict:
        """Load state from disk."""
        try:
            if self._state_file.exists():
                return json.loads(self._state_file.read_text())
        except Exception:
            pass
        return {
            "policies": dict(DEFAULT_POLICIES),
            "replicas": {},
            "config": {},
            "history": [],
        }

    def _save_state(self):
        """Persist state to disk."""
        try:
            self._data_dir.mkdir(parents=True, exist_ok=True)
            self._state["saved_at"] = datetime.now().isoformat()
            self._state_file.write_text(json.dumps(self._state, indent=2))
        except Exception:
            pass
