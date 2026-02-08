#!/usr/bin/env python3
"""
TaskDelegationSkill - Parent-to-child task assignment with budget tracking.

Bridges AgentNetwork (discovery/RPC), TaskDelegator (work coordination),
and ReplicationSkill (spawning) into a unified delegation workflow:

1. DELEGATE   - Assign a task to a specific agent or find the best one
2. SPAWN_FOR  - Spawn a new agent specifically for a task
3. CHECK      - Poll delegated task status
4. RECALL     - Cancel a delegated task and reclaim unspent budget
5. RESULTS    - Collect results from completed delegations
6. LEDGER     - View budget allocation and spending across delegations
7. BATCH      - Delegate multiple tasks at once with budget splitting
8. HISTORY    - Review past delegations and outcomes

This is the critical coordination layer that makes multi-agent work
practical. Without it, agents can discover each other (AgentNetwork)
and coordinate subtasks (TaskDelegator) but can't actually assign
work to other agents with budget accountability.

Pillars served:
- Replication: Parent agents can delegate work to spawned children
- Revenue: Complex customer requests decomposed across specialized agents
- Self-Improvement: Parallel experimentation via delegated tasks
- Goal Setting: Strategic work distribution based on agent capabilities
"""

import json
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from .base import Skill, SkillResult, SkillManifest, SkillAction

DELEGATION_FILE = Path(__file__).parent.parent / "data" / "task_delegations.json"

# Delegation statuses
DELEGATION_STATUSES = [
    "pending",      # Created but not yet sent
    "assigned",     # Sent to agent, awaiting acceptance
    "accepted",     # Agent accepted the task
    "in_progress",  # Agent is working on it
    "completed",    # Agent finished successfully
    "failed",       # Agent failed or timed out
    "recalled",     # Parent cancelled and reclaimed budget
]

# Limits
MAX_DELEGATIONS = 100
MAX_BATCH_SIZE = 10
DEFAULT_TIMEOUT_MINUTES = 120
MAX_BUDGET_PER_DELEGATION = 100.0


class TaskDelegationSkill(Skill):
    """
    Assign tasks to other agents with budget tracking and result collection.

    Enables parent agents to delegate work to children (via AgentNetwork RPC
    or spawned replicas), track progress, collect results, and manage budgets
    across delegations.
    """

    def __init__(self, credentials: Dict[str, str] = None):
        super().__init__(credentials)
        self._ensure_data()

    def _ensure_data(self):
        DELEGATION_FILE.parent.mkdir(parents=True, exist_ok=True)
        if not DELEGATION_FILE.exists():
            self._save(self._default_state())

    def _default_state(self) -> Dict:
        return {
            "delegations": [],
            "ledger": {
                "total_budget_allocated": 0.0,
                "total_budget_spent": 0.0,
                "total_budget_reclaimed": 0.0,
                "total_delegations_created": 0,
                "total_delegations_completed": 0,
                "total_delegations_failed": 0,
                "total_delegations_recalled": 0,
            },
            "created_at": datetime.now().isoformat(),
        }

    def _load(self) -> Dict:
        try:
            with open(DELEGATION_FILE, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return self._default_state()

    def _save(self, state: Dict):
        DELEGATION_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(DELEGATION_FILE, "w") as f:
            json.dump(state, f, indent=2, default=str)

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="task_delegation",
            name="Task Delegation",
            version="1.0.0",
            category="coordination",
            description="Assign tasks to other agents with budget tracking and result collection",
            required_credentials=[],
            install_cost=0,
            actions=[
                SkillAction(
                    name="delegate",
                    description="Assign a task to a specific agent or auto-find the best one",
                    parameters={
                        "task_name": {"type": "string", "required": True, "description": "Name of the task"},
                        "task_description": {"type": "string", "required": True, "description": "Detailed description of what needs to be done"},
                        "budget": {"type": "number", "required": True, "description": "Budget allocated for this task"},
                        "agent_id": {"type": "string", "required": False, "description": "Target agent ID (if known). If omitted, auto-routes via AgentNetwork"},
                        "required_capability": {"type": "string", "required": False, "description": "Capability needed (used for auto-routing)"},
                        "skill_id": {"type": "string", "required": False, "description": "Specific skill to execute on target agent"},
                        "action": {"type": "string", "required": False, "description": "Specific action to execute"},
                        "params": {"type": "object", "required": False, "description": "Parameters for the skill action"},
                        "timeout_minutes": {"type": "number", "required": False, "description": "Timeout in minutes (default 120)"},
                        "priority": {"type": "string", "required": False, "description": "Priority: low, normal, high, critical"},
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="spawn_for",
                    description="Spawn a new agent specifically for a task",
                    parameters={
                        "task_name": {"type": "string", "required": True, "description": "Name of the task"},
                        "task_description": {"type": "string", "required": True, "description": "What the spawned agent should do"},
                        "budget": {"type": "number", "required": True, "description": "Budget for the new agent"},
                        "mutations": {"type": "object", "required": False, "description": "Agent mutations (model, prompt overrides)"},
                        "timeout_minutes": {"type": "number", "required": False, "description": "Timeout in minutes"},
                    },
                    estimated_cost=0.01,
                ),
                SkillAction(
                    name="check",
                    description="Check status of a delegated task",
                    parameters={
                        "delegation_id": {"type": "string", "required": True, "description": "Delegation ID to check"},
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="recall",
                    description="Cancel a delegated task and reclaim unspent budget",
                    parameters={
                        "delegation_id": {"type": "string", "required": True, "description": "Delegation to cancel"},
                        "reason": {"type": "string", "required": False, "description": "Reason for recall"},
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="results",
                    description="Get results from completed delegations",
                    parameters={
                        "delegation_id": {"type": "string", "required": False, "description": "Specific delegation (or all completed)"},
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="ledger",
                    description="View budget allocation and spending across all delegations",
                    parameters={},
                    estimated_cost=0,
                ),
                SkillAction(
                    name="batch",
                    description="Delegate multiple tasks at once with budget splitting",
                    parameters={
                        "tasks": {"type": "array", "required": True, "description": "List of task defs [{task_name, task_description, budget, ...}]"},
                        "total_budget": {"type": "number", "required": False, "description": "Total budget to split (overrides per-task budgets)"},
                        "strategy": {"type": "string", "required": False, "description": "Split strategy: equal, weighted, priority_based"},
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="history",
                    description="Review past delegations and outcomes",
                    parameters={
                        "status": {"type": "string", "required": False, "description": "Filter by status"},
                        "limit": {"type": "number", "required": False, "description": "Max entries to return"},
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="report_completion",
                    description="Called by child agent to report task completion",
                    parameters={
                        "delegation_id": {"type": "string", "required": True, "description": "Delegation ID"},
                        "status": {"type": "string", "required": True, "description": "completed or failed"},
                        "result": {"type": "object", "required": False, "description": "Task result data"},
                        "budget_spent": {"type": "number", "required": False, "description": "Actual budget spent"},
                        "error": {"type": "string", "required": False, "description": "Error message if failed"},
                    },
                    estimated_cost=0,
                ),
            ],
        )

    async def execute(self, action: str, params: Dict) -> SkillResult:
        handlers = {
            "delegate": self._delegate,
            "spawn_for": self._spawn_for,
            "check": self._check,
            "recall": self._recall,
            "results": self._results,
            "ledger": self._ledger,
            "batch": self._batch,
            "history": self._history,
            "report_completion": self._report_completion,
        }

        handler = handlers.get(action)
        if not handler:
            return SkillResult(
                success=False,
                message=f"Unknown action: {action}. Available: {list(handlers.keys())}",
            )

        try:
            return await handler(params)
        except Exception as e:
            return SkillResult(success=False, message=f"Error in {action}: {str(e)}")

    async def _delegate(self, params: Dict) -> SkillResult:
        """Assign a task to a specific agent or auto-route to the best one."""
        task_name = params.get("task_name", "").strip()
        task_description = params.get("task_description", "").strip()
        budget = float(params.get("budget", 0))

        if not task_name:
            return SkillResult(success=False, message="task_name is required")
        if not task_description:
            return SkillResult(success=False, message="task_description is required")
        if budget <= 0:
            return SkillResult(success=False, message="budget must be positive")
        if budget > MAX_BUDGET_PER_DELEGATION:
            return SkillResult(
                success=False,
                message=f"Budget exceeds max ({MAX_BUDGET_PER_DELEGATION})",
            )

        state = self._load()

        # Enforce delegation limit
        active = [d for d in state["delegations"] if d["status"] not in ("completed", "failed", "recalled")]
        if len(active) >= MAX_DELEGATIONS:
            return SkillResult(
                success=False,
                message=f"Active delegation limit reached ({MAX_DELEGATIONS})",
            )

        agent_id = params.get("agent_id", "").strip()
        target_agent = None

        # Auto-route if no agent specified
        if not agent_id and self.context:
            capability = params.get("required_capability", "")
            if capability:
                # Try to find agent via AgentNetwork
                route_result = await self.context.call_skill(
                    "agent_network", "route",
                    {"capability": capability}
                )
                if route_result.success and route_result.data.get("matches"):
                    matches = route_result.data["matches"]
                    # Pick the best match (first one, already ranked)
                    target_agent = matches[0]
                    agent_id = target_agent.get("agent_id", "")

        delegation_id = f"dlg_{uuid.uuid4().hex[:10]}"
        timeout = int(params.get("timeout_minutes", DEFAULT_TIMEOUT_MINUTES))
        priority = params.get("priority", "normal")
        if priority not in ("low", "normal", "high", "critical"):
            priority = "normal"

        delegation = {
            "id": delegation_id,
            "task_name": task_name,
            "task_description": task_description,
            "budget": budget,
            "budget_spent": 0.0,
            "agent_id": agent_id,
            "agent_info": target_agent,
            "skill_id": params.get("skill_id", ""),
            "action": params.get("action", ""),
            "params": params.get("params", {}),
            "priority": priority,
            "status": "pending",
            "timeout_minutes": timeout,
            "created_at": datetime.now().isoformat(),
            "assigned_at": None,
            "completed_at": None,
            "result": None,
            "error": None,
            "spawned_replica": False,
        }

        # If we have an agent, try to send via RPC
        if agent_id and self.context:
            rpc_result = await self._send_to_agent(delegation)
            if rpc_result:
                delegation["status"] = "assigned"
                delegation["assigned_at"] = datetime.now().isoformat()
            else:
                delegation["status"] = "pending"
        elif not agent_id:
            delegation["status"] = "pending"

        state["delegations"].append(delegation)
        state["ledger"]["total_budget_allocated"] += budget
        state["ledger"]["total_delegations_created"] += 1
        self._save(state)

        msg = f"Task '{task_name}' delegated"
        if agent_id:
            msg += f" to agent '{agent_id}'"
        else:
            msg += " (no agent assigned yet - use spawn_for or assign manually)"
        msg += f" with budget ${budget:.2f}"

        return SkillResult(
            success=True,
            message=msg,
            data={
                "delegation_id": delegation_id,
                "task_name": task_name,
                "agent_id": agent_id,
                "budget": budget,
                "status": delegation["status"],
                "priority": priority,
            },
        )

    async def _spawn_for(self, params: Dict) -> SkillResult:
        """Spawn a new agent specifically to handle a task."""
        task_name = params.get("task_name", "").strip()
        task_description = params.get("task_description", "").strip()
        budget = float(params.get("budget", 0))

        if not task_name:
            return SkillResult(success=False, message="task_name is required")
        if not task_description:
            return SkillResult(success=False, message="task_description is required")
        if budget <= 0:
            return SkillResult(success=False, message="budget must be positive")

        state = self._load()
        delegation_id = f"dlg_{uuid.uuid4().hex[:10]}"
        mutations = params.get("mutations", {})
        timeout = int(params.get("timeout_minutes", DEFAULT_TIMEOUT_MINUTES))

        # Create the delegation record first
        delegation = {
            "id": delegation_id,
            "task_name": task_name,
            "task_description": task_description,
            "budget": budget,
            "budget_spent": 0.0,
            "agent_id": "",
            "agent_info": None,
            "skill_id": "",
            "action": "",
            "params": {},
            "priority": "normal",
            "status": "pending",
            "timeout_minutes": timeout,
            "created_at": datetime.now().isoformat(),
            "assigned_at": None,
            "completed_at": None,
            "result": None,
            "error": None,
            "spawned_replica": True,
            "mutations": mutations,
        }

        # Try to spawn via ReplicationSkill
        spawn_result = None
        if self.context:
            # First snapshot the current agent
            snap_result = await self.context.call_skill(
                "replication", "snapshot", {}
            )

            if snap_result.success:
                snapshot_id = snap_result.data.get("snapshot_id", "")

                # Apply task-specific mutations
                task_mutations = {
                    **mutations,
                    "task_delegation_id": delegation_id,
                    "task_description": task_description,
                }

                # Spawn the replica
                spawn_result = await self.context.call_skill(
                    "replication", "spawn",
                    {
                        "snapshot_id": snapshot_id,
                        "name": f"worker-{task_name[:20]}",
                        "budget": budget,
                        "mutations": task_mutations,
                    }
                )

                if spawn_result.success:
                    replica_id = spawn_result.data.get("replica_id", "")
                    delegation["agent_id"] = replica_id
                    delegation["status"] = "assigned"
                    delegation["assigned_at"] = datetime.now().isoformat()
                    delegation["agent_info"] = {
                        "replica_id": replica_id,
                        "spawned": True,
                        "mutations": task_mutations,
                    }

        state["delegations"].append(delegation)
        state["ledger"]["total_budget_allocated"] += budget
        state["ledger"]["total_delegations_created"] += 1
        self._save(state)

        if spawn_result and spawn_result.success:
            return SkillResult(
                success=True,
                message=f"Spawned agent for '{task_name}' with budget ${budget:.2f}",
                data={
                    "delegation_id": delegation_id,
                    "agent_id": delegation["agent_id"],
                    "status": "assigned",
                    "spawned": True,
                },
            )
        else:
            error_msg = ""
            if spawn_result:
                error_msg = spawn_result.message
            return SkillResult(
                success=True,
                message=f"Delegation created for '{task_name}' but spawn failed: {error_msg}. "
                        "Task is pending - assign manually or retry spawn.",
                data={
                    "delegation_id": delegation_id,
                    "status": "pending",
                    "spawned": False,
                    "spawn_error": error_msg,
                },
            )

    async def _check(self, params: Dict) -> SkillResult:
        """Check the status of a delegated task."""
        delegation_id = params.get("delegation_id", "").strip()
        if not delegation_id:
            return SkillResult(success=False, message="delegation_id is required")

        state = self._load()
        delegation = self._find_delegation(state, delegation_id)
        if not delegation:
            return SkillResult(success=False, message=f"Delegation '{delegation_id}' not found")

        # Check for timeout
        if delegation["status"] in ("assigned", "accepted", "in_progress"):
            if delegation.get("assigned_at"):
                assigned = datetime.fromisoformat(delegation["assigned_at"])
                elapsed = (datetime.now() - assigned).total_seconds() / 60
                if elapsed > delegation.get("timeout_minutes", DEFAULT_TIMEOUT_MINUTES):
                    delegation["status"] = "failed"
                    delegation["error"] = "Timeout exceeded"
                    delegation["completed_at"] = datetime.now().isoformat()
                    state["ledger"]["total_delegations_failed"] += 1
                    self._save(state)

        # Try to check via RPC if agent is assigned
        if delegation["status"] in ("assigned", "in_progress") and delegation.get("agent_id") and self.context:
            rpc_result = await self.context.call_skill(
                "agent_network", "rpc",
                {
                    "target_agent_id": delegation["agent_id"],
                    "skill_id": "task_delegation",
                    "action": "report_status",
                    "params": {"delegation_id": delegation_id},
                }
            )
            if rpc_result.success and rpc_result.data.get("response"):
                resp = rpc_result.data["response"]
                if resp.get("status"):
                    delegation["status"] = resp["status"]
                if resp.get("result"):
                    delegation["result"] = resp["result"]

        elapsed_str = ""
        if delegation.get("assigned_at"):
            assigned = datetime.fromisoformat(delegation["assigned_at"])
            elapsed_min = (datetime.now() - assigned).total_seconds() / 60
            elapsed_str = f" ({elapsed_min:.1f} min elapsed)"

        return SkillResult(
            success=True,
            message=f"Delegation '{delegation['task_name']}': {delegation['status']}{elapsed_str}",
            data={
                "delegation_id": delegation_id,
                "task_name": delegation["task_name"],
                "status": delegation["status"],
                "agent_id": delegation["agent_id"],
                "budget": delegation["budget"],
                "budget_spent": delegation["budget_spent"],
                "result": delegation.get("result"),
                "error": delegation.get("error"),
                "spawned_replica": delegation.get("spawned_replica", False),
            },
        )

    async def _recall(self, params: Dict) -> SkillResult:
        """Cancel a delegated task and reclaim unspent budget."""
        delegation_id = params.get("delegation_id", "").strip()
        if not delegation_id:
            return SkillResult(success=False, message="delegation_id is required")

        state = self._load()
        delegation = self._find_delegation(state, delegation_id)
        if not delegation:
            return SkillResult(success=False, message=f"Delegation '{delegation_id}' not found")

        if delegation["status"] in ("completed", "recalled"):
            return SkillResult(
                success=False,
                message=f"Cannot recall - delegation is already '{delegation['status']}'",
            )

        reason = params.get("reason", "Recalled by parent agent")
        budget_remaining = delegation["budget"] - delegation["budget_spent"]

        delegation["status"] = "recalled"
        delegation["completed_at"] = datetime.now().isoformat()
        delegation["error"] = reason

        state["ledger"]["total_budget_reclaimed"] += budget_remaining
        state["ledger"]["total_delegations_recalled"] += 1

        # Try to notify the agent via RPC
        if delegation.get("agent_id") and self.context:
            await self.context.call_skill(
                "agent_network", "rpc",
                {
                    "target_agent_id": delegation["agent_id"],
                    "skill_id": "task_delegation",
                    "action": "cancel_task",
                    "params": {
                        "delegation_id": delegation_id,
                        "reason": reason,
                    },
                }
            )

        self._save(state)

        return SkillResult(
            success=True,
            message=f"Delegation '{delegation['task_name']}' recalled. "
                    f"Budget reclaimed: ${budget_remaining:.2f}",
            data={
                "delegation_id": delegation_id,
                "budget_reclaimed": budget_remaining,
                "budget_spent": delegation["budget_spent"],
                "reason": reason,
            },
        )

    async def _results(self, params: Dict) -> SkillResult:
        """Get results from completed delegations."""
        state = self._load()
        delegation_id = params.get("delegation_id", "").strip()

        if delegation_id:
            delegation = self._find_delegation(state, delegation_id)
            if not delegation:
                return SkillResult(success=False, message=f"Delegation '{delegation_id}' not found")

            return SkillResult(
                success=True,
                message=f"Result for '{delegation['task_name']}': {delegation['status']}",
                data={
                    "delegation_id": delegation_id,
                    "task_name": delegation["task_name"],
                    "status": delegation["status"],
                    "result": delegation.get("result"),
                    "error": delegation.get("error"),
                    "budget": delegation["budget"],
                    "budget_spent": delegation["budget_spent"],
                },
            )

        # Return all completed delegations
        completed = [
            {
                "delegation_id": d["id"],
                "task_name": d["task_name"],
                "status": d["status"],
                "result": d.get("result"),
                "error": d.get("error"),
                "budget": d["budget"],
                "budget_spent": d["budget_spent"],
                "completed_at": d.get("completed_at"),
            }
            for d in state["delegations"]
            if d["status"] in ("completed", "failed", "recalled")
        ]

        return SkillResult(
            success=True,
            message=f"Found {len(completed)} completed delegation(s)",
            data={"results": completed, "count": len(completed)},
        )

    async def _ledger(self, params: Dict) -> SkillResult:
        """View budget allocation and spending across all delegations."""
        state = self._load()
        ledger = state.get("ledger", {})

        # Per-delegation breakdown
        active_delegations = []
        for d in state["delegations"]:
            if d["status"] not in ("completed", "failed", "recalled"):
                active_delegations.append({
                    "delegation_id": d["id"],
                    "task_name": d["task_name"],
                    "agent_id": d["agent_id"],
                    "budget": d["budget"],
                    "budget_spent": d["budget_spent"],
                    "status": d["status"],
                    "priority": d.get("priority", "normal"),
                })

        total_active_budget = sum(d["budget"] for d in active_delegations)
        total_active_spent = sum(d["budget_spent"] for d in active_delegations)

        return SkillResult(
            success=True,
            message=f"Ledger: ${ledger.get('total_budget_allocated', 0):.2f} allocated, "
                    f"${ledger.get('total_budget_spent', 0):.2f} spent, "
                    f"${ledger.get('total_budget_reclaimed', 0):.2f} reclaimed",
            data={
                "ledger": ledger,
                "active_delegations": active_delegations,
                "active_budget_total": total_active_budget,
                "active_budget_spent": total_active_spent,
            },
        )

    async def _batch(self, params: Dict) -> SkillResult:
        """Delegate multiple tasks at once."""
        tasks = params.get("tasks", [])
        if not tasks:
            return SkillResult(success=False, message="tasks list is required")
        if len(tasks) > MAX_BATCH_SIZE:
            return SkillResult(
                success=False,
                message=f"Maximum {MAX_BATCH_SIZE} tasks per batch",
            )

        total_budget = params.get("total_budget")
        strategy = params.get("strategy", "equal")

        # If total_budget specified, split across tasks
        if total_budget:
            total_budget = float(total_budget)
            if strategy == "equal":
                per_task = total_budget / len(tasks)
                for task in tasks:
                    task["budget"] = per_task
            elif strategy == "weighted":
                # Use existing budgets as weights
                weights = [float(t.get("budget", 1)) for t in tasks]
                total_weight = sum(weights)
                for task, weight in zip(tasks, weights):
                    task["budget"] = (weight / total_weight) * total_budget
            elif strategy == "priority_based":
                # Higher priority gets more budget
                priority_weights = {"critical": 4, "high": 3, "normal": 2, "low": 1}
                weights = [priority_weights.get(t.get("priority", "normal"), 2) for t in tasks]
                total_weight = sum(weights)
                for task, weight in zip(tasks, weights):
                    task["budget"] = (weight / total_weight) * total_budget

        results = []
        successes = 0
        for task in tasks:
            result = await self._delegate(task)
            results.append({
                "task_name": task.get("task_name", ""),
                "success": result.success,
                "delegation_id": result.data.get("delegation_id") if result.success else None,
                "message": result.message,
            })
            if result.success:
                successes += 1

        return SkillResult(
            success=successes > 0,
            message=f"Batch delegation: {successes}/{len(tasks)} tasks created",
            data={
                "results": results,
                "successes": successes,
                "failures": len(tasks) - successes,
                "strategy": strategy,
            },
        )

    async def _history(self, params: Dict) -> SkillResult:
        """Review past delegations."""
        state = self._load()
        delegations = state["delegations"]

        status_filter = params.get("status", "").strip()
        if status_filter:
            delegations = [d for d in delegations if d["status"] == status_filter]

        limit = int(params.get("limit", 20))
        delegations = delegations[-limit:]  # Most recent

        history = []
        for d in delegations:
            history.append({
                "delegation_id": d["id"],
                "task_name": d["task_name"],
                "agent_id": d["agent_id"],
                "status": d["status"],
                "budget": d["budget"],
                "budget_spent": d["budget_spent"],
                "priority": d.get("priority", "normal"),
                "spawned_replica": d.get("spawned_replica", False),
                "created_at": d["created_at"],
                "completed_at": d.get("completed_at"),
                "has_result": d.get("result") is not None,
                "error": d.get("error"),
            })

        return SkillResult(
            success=True,
            message=f"Delegation history: {len(history)} entries",
            data={"history": history, "count": len(history)},
        )

    async def _report_completion(self, params: Dict) -> SkillResult:
        """Called by child agent to report task completion."""
        delegation_id = params.get("delegation_id", "").strip()
        if not delegation_id:
            return SkillResult(success=False, message="delegation_id is required")

        new_status = params.get("status", "").strip()
        if new_status not in ("completed", "failed"):
            return SkillResult(success=False, message="status must be 'completed' or 'failed'")

        state = self._load()
        delegation = self._find_delegation(state, delegation_id)
        if not delegation:
            return SkillResult(success=False, message=f"Delegation '{delegation_id}' not found")

        if delegation["status"] in ("completed", "recalled"):
            return SkillResult(
                success=False,
                message=f"Delegation already '{delegation['status']}'",
            )

        delegation["status"] = new_status
        delegation["completed_at"] = datetime.now().isoformat()

        if new_status == "completed":
            delegation["result"] = params.get("result", {})
            state["ledger"]["total_delegations_completed"] += 1
        else:
            delegation["error"] = params.get("error", "Unknown error")
            state["ledger"]["total_delegations_failed"] += 1

        budget_spent = float(params.get("budget_spent", 0))
        if budget_spent > 0:
            delegation["budget_spent"] = min(budget_spent, delegation["budget"])
            state["ledger"]["total_budget_spent"] += delegation["budget_spent"]

        # Reclaim unspent budget
        unspent = delegation["budget"] - delegation["budget_spent"]
        if unspent > 0:
            state["ledger"]["total_budget_reclaimed"] += unspent

        self._save(state)

        # Auto-update agent reputation based on task outcome
        reputation_updated = await self._auto_update_reputation(delegation, new_status)

        msg = (
            f"Delegation '{delegation['task_name']}' reported as {new_status}. "
            f"Budget spent: ${delegation['budget_spent']:.2f}, "
            f"reclaimed: ${unspent:.2f}"
        )
        if reputation_updated:
            msg += f" (reputation updated for agent '{delegation['agent_id']}')"

        return SkillResult(
            success=True,
            message=msg,
            data={
                "delegation_id": delegation_id,
                "status": new_status,
                "budget_spent": delegation["budget_spent"],
                "budget_reclaimed": unspent,
                "reputation_updated": reputation_updated,
            },
        )

    async def _auto_update_reputation(self, delegation: Dict, status: str) -> bool:
        """
        Automatically update agent reputation when a delegation completes.
        
        Bridges TaskDelegationSkill -> AgentReputationSkill by calling
        record_task_outcome with computed metrics from the delegation.
        
        Returns True if reputation was successfully updated.
        """
        if not self.context:
            return False

        agent_id = delegation.get("agent_id", "")
        if not agent_id:
            return False

        # Compute budget efficiency: 0.0 = used all budget, 1.0 = used nothing
        budget_allocated = delegation.get("budget", 1.0)
        budget_spent = delegation.get("budget_spent", 0.0)
        if budget_allocated > 0:
            budget_efficiency = max(0.0, min(1.0, 1.0 - (budget_spent / budget_allocated)))
        else:
            budget_efficiency = 0.5

        # Compute on_time: check if completed before timeout
        on_time = True
        created_at = delegation.get("created_at")
        completed_at = delegation.get("completed_at")
        timeout_minutes = delegation.get("timeout_minutes", DEFAULT_TIMEOUT_MINUTES)
        if created_at and completed_at:
            try:
                created = datetime.fromisoformat(created_at)
                completed = datetime.fromisoformat(completed_at)
                elapsed_minutes = (completed - created).total_seconds() / 60.0
                on_time = elapsed_minutes <= timeout_minutes
            except (ValueError, TypeError):
                on_time = True  # Default to on-time if we can't parse

        try:
            result = await self.context.call_skill(
                "agent_reputation", "record_task_outcome",
                {
                    "agent_id": agent_id,
                    "success": status == "completed",
                    "budget_efficiency": round(budget_efficiency, 3),
                    "on_time": on_time,
                    "task_name": delegation.get("task_name", "unknown"),
                }
            )
            return result.success
        except Exception:
            # Reputation update is best-effort; don't break delegation
            return False

    async def _send_to_agent(self, delegation: Dict) -> bool:
        """Try to send delegation to an agent via RPC."""
        if not self.context:
            return False

        agent_id = delegation.get("agent_id", "")
        if not agent_id:
            return False

        result = await self.context.call_skill(
            "agent_network", "rpc",
            {
                "target_agent_id": agent_id,
                "skill_id": "task_delegation",
                "action": "accept_task",
                "params": {
                    "delegation_id": delegation["id"],
                    "task_name": delegation["task_name"],
                    "task_description": delegation["task_description"],
                    "budget": delegation["budget"],
                    "skill_id": delegation.get("skill_id", ""),
                    "action": delegation.get("action", ""),
                    "params": delegation.get("params", {}),
                },
            }
        )

        return result.success

    def _find_delegation(self, state: Dict, delegation_id: str) -> Optional[Dict]:
        for d in state["delegations"]:
            if d["id"] == delegation_id:
                return d
        return None

    async def initialize(self) -> bool:
        self.initialized = True
        return True
