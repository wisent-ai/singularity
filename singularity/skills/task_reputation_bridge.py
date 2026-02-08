#!/usr/bin/env python3
"""
TaskReputationBridgeSkill - Auto-updates agent reputation from task delegation outcomes.

When TaskDelegationSkill reports a task completion (success or failure), this
bridge automatically calls AgentReputationSkill.record_task_outcome to update
the assignee's reputation scores. Without this bridge, task outcomes and
reputation are disconnected — agents complete tasks but their reputation
never reflects their track record.

This closes the delegation → reputation feedback loop:
  delegate task → agent works → report_completion → auto-update reputation

Also tracks delegation performance analytics: which agents complete tasks
fastest, most budget-efficiently, and most reliably.

Pillar: Replication + Self-Improvement
- Replication: Agents automatically build reputation through work, enabling
  trust-based delegation decisions
- Self-Improvement: Closed feedback loop where task performance shapes
  future delegation routing

Actions:
- sync: Scan recent delegations and update reputation for any unsynced outcomes
- configure: Set scoring weights, auto-sync behavior, timeliness thresholds
- stats: View delegation-reputation sync statistics and per-agent summaries
- agent_report: Detailed delegation performance report for a specific agent
- history: View recent sync events
- reset_sync: Clear sync state to re-process all delegations
"""

import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from .base import Skill, SkillResult, SkillManifest, SkillAction

BRIDGE_FILE = Path(__file__).parent.parent / "data" / "task_reputation_bridge.json"
MAX_HISTORY = 200


def _now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


class TaskReputationBridgeSkill(Skill):
    """
    Auto-updates agent reputation from task delegation outcomes.
    Bridges TaskDelegationSkill and AgentReputationSkill to close the
    delegation-reputation feedback loop.
    """

    def __init__(self, credentials: Dict[str, str] = None):
        super().__init__(credentials)
        self._store = None

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="task_reputation_bridge",
            name="Task-Reputation Bridge",
            version="1.0.0",
            category="meta",
            description="Auto-updates agent reputation scores from task delegation outcomes",
            actions=[
                SkillAction(
                    name="sync",
                    description="Scan delegations and update reputation for any unsynced task outcomes",
                    parameters={
                        "dry_run": {"type": "boolean", "required": False, "description": "Preview updates without executing (default: False)"},
                        "max_delegations": {"type": "number", "required": False, "description": "Max delegations to process (default: 50)"},
                    },
                ),
                SkillAction(
                    name="configure",
                    description="Set scoring weights, timeliness thresholds, and sync behavior",
                    parameters={
                        "success_competence_base": {"type": "number", "required": False, "description": "Base competence boost for success (default: 2.0)"},
                        "success_competence_max": {"type": "number", "required": False, "description": "Max competence boost for budget-efficient success (default: 5.0)"},
                        "failure_competence_penalty": {"type": "number", "required": False, "description": "Competence penalty for failure (default: -3.0)"},
                        "on_time_reliability_boost": {"type": "number", "required": False, "description": "Reliability boost for on-time completion (default: 2.0)"},
                        "late_reliability_penalty": {"type": "number", "required": False, "description": "Reliability penalty for late completion (default: -1.0)"},
                        "timeliness_threshold_minutes": {"type": "number", "required": False, "description": "Minutes after delegation before considered late (default: 120)"},
                    },
                ),
                SkillAction(
                    name="stats",
                    description="View delegation-reputation sync statistics",
                    parameters={},
                ),
                SkillAction(
                    name="agent_report",
                    description="Detailed delegation performance report for a specific agent",
                    parameters={
                        "agent_id": {"type": "string", "required": True, "description": "Agent ID to report on"},
                    },
                ),
                SkillAction(
                    name="history",
                    description="View recent sync events",
                    parameters={
                        "limit": {"type": "number", "required": False, "description": "Max events to return (default: 20)"},
                        "agent_id": {"type": "string", "required": False, "description": "Filter by agent ID"},
                    },
                ),
                SkillAction(
                    name="reset_sync",
                    description="Clear sync state to re-process all delegations",
                    parameters={
                        "confirm": {"type": "boolean", "required": True, "description": "Must be True to confirm reset"},
                    },
                ),
            ],
            required_credentials=[],
        )

    # ── Persistence ───────────────────────────────────────────────

    def _default_state(self) -> Dict:
        return {
            "config": {
                "success_competence_base": 2.0,
                "success_competence_max": 5.0,
                "failure_competence_penalty": -3.0,
                "on_time_reliability_boost": 2.0,
                "late_reliability_penalty": -1.0,
                "timeliness_threshold_minutes": 120,
            },
            "synced_delegation_ids": [],  # IDs already processed
            "agent_stats": {},  # agent_id -> {completed, failed, total_budget, avg_efficiency, ...}
            "history": [],
            "stats": {
                "total_synced": 0,
                "total_successes": 0,
                "total_failures": 0,
                "sync_cycles": 0,
                "last_sync": None,
            },
            "metadata": {
                "created_at": _now_iso(),
                "version": "1.0.0",
            },
        }

    def _load(self) -> Dict:
        if self._store is not None:
            return self._store
        BRIDGE_FILE.parent.mkdir(parents=True, exist_ok=True)
        if BRIDGE_FILE.exists():
            try:
                with open(BRIDGE_FILE, "r") as f:
                    self._store = json.load(f)
                    return self._store
            except (json.JSONDecodeError, FileNotFoundError):
                pass
        self._store = self._default_state()
        self._save(self._store)
        return self._store

    def _save(self, data: Dict):
        self._store = data
        if len(data.get("history", [])) > MAX_HISTORY:
            data["history"] = data["history"][-MAX_HISTORY:]
        # Keep synced IDs bounded
        if len(data.get("synced_delegation_ids", [])) > 1000:
            data["synced_delegation_ids"] = data["synced_delegation_ids"][-1000:]
        BRIDGE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(BRIDGE_FILE, "w") as f:
            json.dump(data, f, indent=2, default=str)

    # ── Execute Dispatch ──────────────────────────────────────────

    async def execute(self, action: str, params: Dict[str, Any]) -> SkillResult:
        handlers = {
            "sync": self._sync,
            "configure": self._configure,
            "stats": self._stats,
            "agent_report": self._agent_report,
            "history": self._history,
            "reset_sync": self._reset_sync,
        }
        handler = handlers.get(action)
        if not handler:
            return SkillResult(success=False, message=f"Unknown action: {action}")
        return await handler(params)

    # ── Helpers ───────────────────────────────────────────────────

    async def _call_skill(self, skill_id: str, action: str, params: Dict) -> Optional[Dict]:
        """Call another skill through SkillContext if available."""
        if not self.context:
            return None
        try:
            result = await self.context.invoke_skill(skill_id, action, params)
            if hasattr(result, "data"):
                return result.data if result.success else None
            return result
        except Exception:
            return None

    def _get_delegations_from_file(self) -> Optional[List[Dict]]:
        """Fallback: Read delegation data directly from file."""
        try:
            from .task_delegation import DELEGATION_FILE
            if DELEGATION_FILE.exists():
                with open(DELEGATION_FILE, "r") as f:
                    data = json.load(f)
                    return data.get("delegations", [])
        except Exception:
            pass
        return None

    def _compute_budget_efficiency(self, delegation: Dict) -> float:
        """Compute budget efficiency: ratio of budget spent to budget allocated."""
        budget = delegation.get("budget", 0)
        spent = delegation.get("budget_spent", 0)
        if budget <= 0:
            return 0.5  # neutral when no budget info
        return min(1.0, max(0.0, spent / budget))

    def _check_timeliness(self, delegation: Dict, threshold_minutes: int) -> bool:
        """Check if delegation was completed on time."""
        created = delegation.get("created_at", "")
        completed = delegation.get("completed_at", "")
        timeout_mins = delegation.get("timeout_minutes", threshold_minutes)
        try:
            created_dt = datetime.fromisoformat(created.replace("Z", "+00:00")).replace(tzinfo=None)
            completed_dt = datetime.fromisoformat(completed.replace("Z", "+00:00")).replace(tzinfo=None)
            elapsed = (completed_dt - created_dt).total_seconds() / 60
            return elapsed <= timeout_mins
        except (ValueError, TypeError):
            return True  # assume on time if can't determine

    def _update_agent_stats(self, store: Dict, agent_id: str, success: bool,
                            budget_efficiency: float, on_time: bool):
        """Update per-agent running statistics."""
        stats = store.setdefault("agent_stats", {})
        if agent_id not in stats:
            stats[agent_id] = {
                "completed": 0,
                "failed": 0,
                "total_tasks": 0,
                "total_budget_efficiency": 0.0,
                "on_time_count": 0,
                "late_count": 0,
                "last_synced": None,
            }
        agent = stats[agent_id]
        agent["total_tasks"] += 1
        if success:
            agent["completed"] += 1
        else:
            agent["failed"] += 1
        agent["total_budget_efficiency"] += budget_efficiency
        if on_time:
            agent["on_time_count"] += 1
        else:
            agent["late_count"] += 1
        agent["last_synced"] = _now_iso()

    # ── Action: sync ─────────────────────────────────────────────

    async def _sync(self, params: Dict) -> SkillResult:
        """Scan delegations and update reputation for unsynced completed/failed tasks."""
        dry_run = params.get("dry_run", False)
        max_delegations = min(int(params.get("max_delegations", 50)), 200)
        store = self._load()
        config = store["config"]

        # Get delegation data
        delegations = None

        # Try via skill context first
        if self.context:
            result = await self._call_skill("task_delegation", "history", {"limit": max_delegations})
            if result:
                delegations = result.get("delegations", result.get("history", []))

        # Fallback: direct file access
        if delegations is None:
            delegations = self._get_delegations_from_file()

        if delegations is None:
            return SkillResult(
                success=False,
                message="Cannot access TaskDelegationSkill - ensure it is loaded",
                data={"error": "task_delegation unavailable"},
            )

        synced_ids = set(store.get("synced_delegation_ids", []))
        updates = []
        skipped = []
        errors = []

        for delegation in delegations:
            d_id = delegation.get("delegation_id", delegation.get("id", ""))
            status = delegation.get("status", "")

            # Only process completed/failed delegations not yet synced
            if status not in ("completed", "failed"):
                continue
            if d_id in synced_ids:
                skipped.append(d_id)
                continue

            agent_id = delegation.get("assigned_to", delegation.get("agent_id", ""))
            if not agent_id:
                errors.append({"delegation_id": d_id, "error": "no agent_id"})
                continue

            success = status == "completed"
            budget_efficiency = self._compute_budget_efficiency(delegation)
            on_time = self._check_timeliness(delegation, config["timeliness_threshold_minutes"])
            task_name = delegation.get("task_name", delegation.get("name", "unknown"))

            update_record = {
                "delegation_id": d_id,
                "agent_id": agent_id,
                "task_name": task_name,
                "success": success,
                "budget_efficiency": round(budget_efficiency, 3),
                "on_time": on_time,
            }

            if not dry_run:
                # Call AgentReputationSkill.record_task_outcome
                rep_result = await self._call_skill("agent_reputation", "record_task_outcome", {
                    "agent_id": agent_id,
                    "success": success,
                    "budget_efficiency": budget_efficiency,
                    "on_time": on_time,
                    "task_name": task_name,
                })

                if rep_result is not None:
                    update_record["reputation_update"] = rep_result
                    synced_ids.add(d_id)
                    self._update_agent_stats(store, agent_id, success, budget_efficiency, on_time)
                    store["stats"]["total_synced"] += 1
                    if success:
                        store["stats"]["total_successes"] += 1
                    else:
                        store["stats"]["total_failures"] += 1
                else:
                    update_record["error"] = "reputation update failed"
                    errors.append(update_record)
                    continue

                store["history"].append({
                    "event": "synced",
                    "delegation_id": d_id,
                    "agent_id": agent_id,
                    "success": success,
                    "budget_efficiency": round(budget_efficiency, 3),
                    "on_time": on_time,
                    "timestamp": _now_iso(),
                })
            else:
                update_record["dry_run"] = True

            updates.append(update_record)

        # Update state
        store["synced_delegation_ids"] = list(synced_ids)
        store["stats"]["sync_cycles"] += 1
        store["stats"]["last_sync"] = _now_iso()

        if not dry_run:
            self._save(store)

        prefix = "[DRY RUN] " if dry_run else ""
        summary_parts = []
        if updates:
            summary_parts.append(f"{len(updates)} reputation updates")
        if skipped:
            summary_parts.append(f"{len(skipped)} already synced")
        if errors:
            summary_parts.append(f"{len(errors)} errors")
        if not summary_parts:
            summary_parts.append("no new delegations to sync")

        return SkillResult(
            success=True,
            message=f"{prefix}Sync complete: {', '.join(summary_parts)}",
            data={
                "updates": updates,
                "skipped_count": len(skipped),
                "errors": errors,
                "dry_run": dry_run,
                "total_synced": store["stats"]["total_synced"],
            },
        )

    # ── Action: configure ────────────────────────────────────────

    async def _configure(self, params: Dict) -> SkillResult:
        """Configure scoring weights and thresholds."""
        store = self._load()
        config = store["config"]
        changes = []

        configurable = [
            "success_competence_base", "success_competence_max",
            "failure_competence_penalty", "on_time_reliability_boost",
            "late_reliability_penalty", "timeliness_threshold_minutes",
        ]

        for key in configurable:
            if key in params:
                val = float(params[key])
                if key == "timeliness_threshold_minutes" and (val < 1 or val > 10080):
                    return SkillResult(success=False, message=f"{key} must be 1-10080")
                config[key] = val
                changes.append(f"{key} = {val}")

        if not changes:
            return SkillResult(
                success=True,
                message="No changes specified",
                data={"config": config},
            )

        self._save(store)
        return SkillResult(
            success=True,
            message=f"Configuration updated: {'; '.join(changes)}",
            data={"config": config},
        )

    # ── Action: stats ────────────────────────────────────────────

    async def _stats(self, params: Dict) -> SkillResult:
        """View sync statistics."""
        store = self._load()
        agent_stats = store.get("agent_stats", {})

        # Compute per-agent summaries
        agent_summaries = []
        for agent_id, stats in agent_stats.items():
            total = stats["total_tasks"]
            success_rate = stats["completed"] / total if total > 0 else 0
            avg_efficiency = stats["total_budget_efficiency"] / total if total > 0 else 0
            on_time_rate = stats["on_time_count"] / total if total > 0 else 0

            agent_summaries.append({
                "agent_id": agent_id,
                "total_tasks": total,
                "completed": stats["completed"],
                "failed": stats["failed"],
                "success_rate": round(success_rate, 3),
                "avg_budget_efficiency": round(avg_efficiency, 3),
                "on_time_rate": round(on_time_rate, 3),
            })

        # Sort by total tasks descending
        agent_summaries.sort(key=lambda x: x["total_tasks"], reverse=True)

        return SkillResult(
            success=True,
            message=f"Task-Reputation Bridge: {store['stats']['total_synced']} synced, "
                    f"{len(agent_stats)} agents tracked, "
                    f"{store['stats']['sync_cycles']} sync cycles",
            data={
                "stats": store["stats"],
                "agent_summaries": agent_summaries,
                "config": store["config"],
            },
        )

    # ── Action: agent_report ─────────────────────────────────────

    async def _agent_report(self, params: Dict) -> SkillResult:
        """Detailed delegation performance report for a specific agent."""
        agent_id = params.get("agent_id", "").strip()
        if not agent_id:
            return SkillResult(success=False, message="agent_id is required")

        store = self._load()
        agent_stats = store.get("agent_stats", {}).get(agent_id)

        if not agent_stats:
            return SkillResult(
                success=True,
                message=f"No delegation data found for agent '{agent_id}'",
                data={"agent_id": agent_id, "found": False},
            )

        total = agent_stats["total_tasks"]
        success_rate = agent_stats["completed"] / total if total > 0 else 0
        avg_efficiency = agent_stats["total_budget_efficiency"] / total if total > 0 else 0
        on_time_rate = agent_stats["on_time_count"] / total if total > 0 else 0

        # Get recent history for this agent
        agent_history = [
            h for h in store.get("history", [])
            if h.get("agent_id") == agent_id
        ][-10:]

        # Try to get current reputation
        reputation = None
        if self.context:
            rep_data = await self._call_skill("agent_reputation", "get_reputation", {"agent_id": agent_id})
            if rep_data:
                reputation = rep_data

        return SkillResult(
            success=True,
            message=f"Agent '{agent_id}': {total} tasks, {success_rate:.0%} success rate, "
                    f"{avg_efficiency:.0%} budget efficiency, {on_time_rate:.0%} on-time",
            data={
                "agent_id": agent_id,
                "found": True,
                "total_tasks": total,
                "completed": agent_stats["completed"],
                "failed": agent_stats["failed"],
                "success_rate": round(success_rate, 3),
                "avg_budget_efficiency": round(avg_efficiency, 3),
                "on_time_rate": round(on_time_rate, 3),
                "on_time_count": agent_stats["on_time_count"],
                "late_count": agent_stats["late_count"],
                "recent_history": agent_history,
                "current_reputation": reputation,
                "last_synced": agent_stats.get("last_synced"),
            },
        )

    # ── Action: history ──────────────────────────────────────────

    async def _history(self, params: Dict) -> SkillResult:
        """View recent sync events."""
        store = self._load()
        limit = min(int(params.get("limit", 20)), MAX_HISTORY)
        agent_filter = params.get("agent_id", "").strip()

        history = store.get("history", [])
        if agent_filter:
            history = [h for h in history if h.get("agent_id") == agent_filter]

        recent = history[-limit:]

        return SkillResult(
            success=True,
            message=f"Showing {len(recent)} sync events" +
                    (f" for agent '{agent_filter}'" if agent_filter else ""),
            data={
                "events": recent,
                "total": len(history),
                "filter": agent_filter or None,
            },
        )

    # ── Action: reset_sync ───────────────────────────────────────

    async def _reset_sync(self, params: Dict) -> SkillResult:
        """Clear sync state to re-process all delegations."""
        if not params.get("confirm", False):
            return SkillResult(
                success=False,
                message="Must pass confirm=True to reset sync state",
            )

        store = self._load()
        old_count = len(store.get("synced_delegation_ids", []))
        store["synced_delegation_ids"] = []
        store["history"].append({
            "event": "reset",
            "cleared_count": old_count,
            "timestamp": _now_iso(),
        })
        self._save(store)

        return SkillResult(
            success=True,
            message=f"Sync state reset. Cleared {old_count} synced delegation IDs. "
                    f"Next sync will re-process all delegations.",
            data={"cleared_count": old_count},
        )
