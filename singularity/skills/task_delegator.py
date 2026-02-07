#!/usr/bin/env python3
"""
TaskDelegator Skill - Practical multi-agent task coordination.

While OrchestratorSkill creates autonomous agents with their own life,
TaskDelegator is the practical work distribution layer:

  - Break complex work into subtasks
  - Assign subtasks to worker agents or execute locally
  - Track progress with timeouts and deadlines
  - Collect and aggregate results
  - Handle failures with retries and fallbacks
  - Support fan-out/fan-in, pipeline, and map-reduce patterns

This is the missing coordination layer between "spawn agents" and
"get useful work done." Critical for:
  - Revenue: Handle complex customer requests by decomposing them
  - Self-improvement: Parallelize exploration and experimentation
  - Replication: Coordinate replicas working on shared goals

Execution patterns:
  1. Fan-out/Fan-in: Split work → parallel execution → merge results
  2. Pipeline: Sequential stages, each feeding into the next
  3. Map-Reduce: Apply same operation to many inputs → aggregate
"""

import json
import uuid
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from .base import Skill, SkillResult, SkillManifest, SkillAction


DELEGATOR_FILE = Path(__file__).parent.parent / "data" / "task_delegator.json"

# Task pool statuses
POOL_STATUSES = ["pending", "running", "completed", "failed", "cancelled"]

# Subtask statuses
SUBTASK_STATUSES = ["pending", "assigned", "running", "completed", "failed", "retrying", "cancelled"]

# Execution patterns
PATTERNS = ["fan_out", "pipeline", "map_reduce"]

# Limits
MAX_POOLS = 50
MAX_SUBTASKS_PER_POOL = 20
MAX_RETRIES = 3


class TaskDelegator(Skill):
    """
    Coordinate multi-agent work through structured task delegation.

    The parent agent decomposes work into subtask pools, assigns them
    to workers (other agents or self-execution via skills), tracks
    completion, and aggregates results.
    """

    def __init__(self, credentials: Dict[str, str] = None):
        super().__init__(credentials)
        self._workers: Dict[str, Callable] = {}  # worker_id -> execute_fn
        self._ensure_data()

    def _ensure_data(self):
        DELEGATOR_FILE.parent.mkdir(parents=True, exist_ok=True)
        if not DELEGATOR_FILE.exists():
            self._save(self._default_state())

    def _default_state(self) -> Dict:
        return {
            "pools": [],
            "completed_pools": [],
            "stats": {
                "total_pools_created": 0,
                "total_subtasks_created": 0,
                "total_subtasks_completed": 0,
                "total_subtasks_failed": 0,
                "total_retries": 0,
            },
            "created_at": datetime.now().isoformat(),
        }

    def _load(self) -> Dict:
        try:
            with open(DELEGATOR_FILE, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return self._default_state()

    def _save(self, state: Dict):
        DELEGATOR_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(DELEGATOR_FILE, "w") as f:
            json.dump(state, f, indent=2, default=str)

    def register_worker(self, worker_id: str, execute_fn: Callable):
        """Register a worker that can execute subtasks."""
        self._workers[worker_id] = execute_fn

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="task_delegator",
            name="Task Delegator",
            version="1.0.0",
            category="coordination",
            description="Decompose, delegate, and coordinate multi-agent work",
            required_credentials=[],
            install_cost=0,
            actions=[
                SkillAction(
                    name="create_pool",
                    description="Create a task pool with subtasks for coordinated execution",
                    parameters={
                        "name": {"type": "string", "required": True, "description": "Pool name"},
                        "description": {"type": "string", "required": False, "description": "What this pool accomplishes"},
                        "pattern": {"type": "string", "required": False, "description": "Execution pattern: fan_out (parallel), pipeline (sequential), map_reduce"},
                        "timeout_minutes": {"type": "number", "required": False, "description": "Pool-level timeout in minutes (default 60)"},
                        "subtasks": {"type": "array", "required": True, "description": "List of subtask definitions [{name, skill_id, action, params, depends_on}]"},
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="run_pool",
                    description="Start executing a task pool",
                    parameters={
                        "pool_id": {"type": "string", "required": True, "description": "Pool to execute"},
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="pool_status",
                    description="Check the status of a task pool and its subtasks",
                    parameters={
                        "pool_id": {"type": "string", "required": True, "description": "Pool to check"},
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="list_pools",
                    description="List all task pools",
                    parameters={
                        "status": {"type": "string", "required": False, "description": "Filter by status"},
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="report_subtask",
                    description="Report completion or failure of a subtask (used by workers)",
                    parameters={
                        "pool_id": {"type": "string", "required": True, "description": "Pool ID"},
                        "subtask_id": {"type": "string", "required": True, "description": "Subtask ID"},
                        "status": {"type": "string", "required": True, "description": "New status: completed or failed"},
                        "result": {"type": "object", "required": False, "description": "Subtask result data"},
                        "error": {"type": "string", "required": False, "description": "Error message if failed"},
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="cancel_pool",
                    description="Cancel a running task pool",
                    parameters={
                        "pool_id": {"type": "string", "required": True, "description": "Pool to cancel"},
                        "reason": {"type": "string", "required": False, "description": "Cancellation reason"},
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="retry_failed",
                    description="Retry failed subtasks in a pool",
                    parameters={
                        "pool_id": {"type": "string", "required": True, "description": "Pool ID"},
                        "subtask_id": {"type": "string", "required": False, "description": "Specific subtask to retry (or all failed)"},
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="get_results",
                    description="Get aggregated results from a completed pool",
                    parameters={
                        "pool_id": {"type": "string", "required": True, "description": "Pool to get results from"},
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="stats",
                    description="Get delegation statistics",
                    parameters={},
                    estimated_cost=0,
                ),
            ],
        )

    async def execute(self, action: str, params: Dict) -> SkillResult:
        handlers = {
            "create_pool": self._create_pool,
            "run_pool": self._run_pool,
            "pool_status": self._pool_status,
            "list_pools": self._list_pools,
            "report_subtask": self._report_subtask,
            "cancel_pool": self._cancel_pool,
            "retry_failed": self._retry_failed,
            "get_results": self._get_results,
            "stats": self._stats,
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

    async def _create_pool(self, params: Dict) -> SkillResult:
        """Create a new task pool with subtasks."""
        name = params.get("name", "").strip()
        if not name:
            return SkillResult(success=False, message="Pool name is required")

        subtask_defs = params.get("subtasks", [])
        if not subtask_defs:
            return SkillResult(success=False, message="At least one subtask is required")

        if len(subtask_defs) > MAX_SUBTASKS_PER_POOL:
            return SkillResult(
                success=False,
                message=f"Maximum {MAX_SUBTASKS_PER_POOL} subtasks per pool",
            )

        pattern = params.get("pattern", "fan_out")
        if pattern not in PATTERNS:
            return SkillResult(
                success=False,
                message=f"Invalid pattern. Use: {PATTERNS}",
            )

        state = self._load()

        # Clean old completed pools if at limit
        if len(state["pools"]) >= MAX_POOLS:
            state["pools"] = [
                p for p in state["pools"]
                if p["status"] not in ("completed", "failed", "cancelled")
            ]
            if len(state["pools"]) >= MAX_POOLS:
                return SkillResult(
                    success=False,
                    message=f"Pool limit reached ({MAX_POOLS}). Cancel or complete existing pools.",
                )

        pool_id = str(uuid.uuid4())[:12]

        # Build subtasks
        subtasks = []
        for i, sdef in enumerate(subtask_defs):
            if isinstance(sdef, str):
                # Simple string subtask — treat as a description
                sdef = {"name": sdef}

            subtask_id = str(uuid.uuid4())[:8]
            subtask = {
                "id": subtask_id,
                "name": sdef.get("name", f"subtask_{i}"),
                "skill_id": sdef.get("skill_id", ""),
                "action": sdef.get("action", ""),
                "params": sdef.get("params", {}),
                "depends_on": sdef.get("depends_on", []),
                "worker_id": sdef.get("worker_id"),
                "status": "pending",
                "result": None,
                "error": None,
                "retries": 0,
                "created_at": datetime.now().isoformat(),
                "started_at": None,
                "completed_at": None,
                "order": i,
            }
            subtasks.append(subtask)

        # For pipeline pattern, auto-set dependencies
        if pattern == "pipeline" and len(subtasks) > 1:
            for i in range(1, len(subtasks)):
                if not subtasks[i]["depends_on"]:
                    subtasks[i]["depends_on"] = [subtasks[i - 1]["id"]]

        pool = {
            "id": pool_id,
            "name": name,
            "description": params.get("description", ""),
            "pattern": pattern,
            "status": "pending",
            "timeout_minutes": int(params.get("timeout_minutes", 60)),
            "subtasks": subtasks,
            "created_at": datetime.now().isoformat(),
            "started_at": None,
            "completed_at": None,
            "result_aggregation": None,
        }

        state["pools"].append(pool)
        state["stats"]["total_pools_created"] += 1
        state["stats"]["total_subtasks_created"] += len(subtasks)
        self._save(state)

        return SkillResult(
            success=True,
            message=f"Pool '{name}' created with {len(subtasks)} subtasks ({pattern} pattern)",
            data={
                "pool_id": pool_id,
                "name": name,
                "pattern": pattern,
                "subtask_count": len(subtasks),
                "subtask_ids": [s["id"] for s in subtasks],
            },
        )

    async def _run_pool(self, params: Dict) -> SkillResult:
        """Start executing a task pool."""
        pool_id = params.get("pool_id", "").strip()
        if not pool_id:
            return SkillResult(success=False, message="pool_id is required")

        state = self._load()
        pool = self._find_pool(state, pool_id)
        if not pool:
            return SkillResult(success=False, message=f"Pool '{pool_id}' not found")

        if pool["status"] not in ("pending", "failed"):
            return SkillResult(
                success=False,
                message=f"Pool is '{pool['status']}', can only run pending/failed pools",
            )

        pool["status"] = "running"
        pool["started_at"] = datetime.now().isoformat()

        # Find ready subtasks (no unmet dependencies)
        ready = self._get_ready_subtasks(pool)
        started_names = []

        for subtask in ready:
            subtask["status"] = "running"
            subtask["started_at"] = datetime.now().isoformat()
            started_names.append(subtask["name"])

            # Try to execute via skill context if available
            if self.context and subtask.get("skill_id") and subtask.get("action"):
                try:
                    result = await self.context.call_skill(
                        subtask["skill_id"],
                        subtask["action"],
                        subtask.get("params", {}),
                    )
                    if result.success:
                        subtask["status"] = "completed"
                        subtask["result"] = result.data
                        subtask["completed_at"] = datetime.now().isoformat()
                        state["stats"]["total_subtasks_completed"] += 1
                    else:
                        subtask["status"] = "failed"
                        subtask["error"] = result.message
                        state["stats"]["total_subtasks_failed"] += 1
                except Exception as e:
                    subtask["status"] = "failed"
                    subtask["error"] = str(e)
                    state["stats"]["total_subtasks_failed"] += 1

        # Check if pool is now done
        self._check_pool_completion(pool)
        self._save(state)

        return SkillResult(
            success=True,
            message=f"Pool '{pool['name']}' running. {len(started_names)} subtask(s) started: {', '.join(started_names)}",
            data={
                "pool_id": pool_id,
                "status": pool["status"],
                "started": started_names,
                "subtask_summary": self._summarize_subtasks(pool),
            },
        )

    async def _pool_status(self, params: Dict) -> SkillResult:
        """Get detailed pool status."""
        pool_id = params.get("pool_id", "").strip()
        if not pool_id:
            return SkillResult(success=False, message="pool_id is required")

        state = self._load()
        pool = self._find_pool(state, pool_id)
        if not pool:
            return SkillResult(success=False, message=f"Pool '{pool_id}' not found")

        summary = self._summarize_subtasks(pool)

        # Check for timeout
        if pool["status"] == "running" and pool.get("started_at"):
            started = datetime.fromisoformat(pool["started_at"])
            elapsed = (datetime.now() - started).total_seconds() / 60
            if elapsed > pool.get("timeout_minutes", 60):
                pool["status"] = "failed"
                for st in pool["subtasks"]:
                    if st["status"] in ("pending", "assigned", "running"):
                        st["status"] = "cancelled"
                        st["error"] = "Pool timeout exceeded"
                self._save(state)
                summary = self._summarize_subtasks(pool)

        # Progress bar
        total = len(pool["subtasks"])
        done = summary.get("completed", 0) + summary.get("failed", 0)
        progress = (done / total * 100) if total > 0 else 0

        subtask_details = []
        for st in pool["subtasks"]:
            subtask_details.append({
                "id": st["id"],
                "name": st["name"],
                "status": st["status"],
                "has_result": st.get("result") is not None,
                "error": st.get("error"),
                "retries": st.get("retries", 0),
            })

        return SkillResult(
            success=True,
            message=f"Pool '{pool['name']}': {pool['status']} ({progress:.0f}% complete)",
            data={
                "pool_id": pool_id,
                "name": pool["name"],
                "status": pool["status"],
                "pattern": pool["pattern"],
                "progress_pct": round(progress, 1),
                "summary": summary,
                "subtasks": subtask_details,
            },
        )

    async def _list_pools(self, params: Dict) -> SkillResult:
        """List all pools."""
        state = self._load()
        pools = state["pools"]

        status_filter = params.get("status")
        if status_filter:
            pools = [p for p in pools if p["status"] == status_filter]

        pool_summaries = []
        for pool in pools:
            summary = self._summarize_subtasks(pool)
            total = len(pool["subtasks"])
            done = summary.get("completed", 0) + summary.get("failed", 0)
            progress = (done / total * 100) if total > 0 else 0

            pool_summaries.append({
                "id": pool["id"],
                "name": pool["name"],
                "status": pool["status"],
                "pattern": pool["pattern"],
                "progress_pct": round(progress, 1),
                "subtask_count": total,
                "created_at": pool["created_at"],
            })

        return SkillResult(
            success=True,
            message=f"Found {len(pool_summaries)} pool(s)",
            data={"pools": pool_summaries, "total": len(pool_summaries)},
        )

    async def _report_subtask(self, params: Dict) -> SkillResult:
        """Report completion/failure of a subtask."""
        pool_id = params.get("pool_id", "").strip()
        subtask_id = params.get("subtask_id", "").strip()
        new_status = params.get("status", "").strip()

        if not pool_id or not subtask_id:
            return SkillResult(success=False, message="pool_id and subtask_id are required")

        if new_status not in ("completed", "failed"):
            return SkillResult(success=False, message="status must be 'completed' or 'failed'")

        state = self._load()
        pool = self._find_pool(state, pool_id)
        if not pool:
            return SkillResult(success=False, message=f"Pool '{pool_id}' not found")

        subtask = self._find_subtask(pool, subtask_id)
        if not subtask:
            return SkillResult(success=False, message=f"Subtask '{subtask_id}' not found")

        subtask["status"] = new_status
        subtask["completed_at"] = datetime.now().isoformat()

        if new_status == "completed":
            subtask["result"] = params.get("result", {})
            state["stats"]["total_subtasks_completed"] += 1
        else:
            subtask["error"] = params.get("error", "Unknown error")
            state["stats"]["total_subtasks_failed"] += 1

        # After reporting, check if new subtasks can be started (dependency resolution)
        if pool["status"] == "running":
            newly_ready = self._get_ready_subtasks(pool)
            for ready_st in newly_ready:
                ready_st["status"] = "running"
                ready_st["started_at"] = datetime.now().isoformat()

        # Check pool completion
        self._check_pool_completion(pool)
        self._save(state)

        return SkillResult(
            success=True,
            message=f"Subtask '{subtask['name']}' reported as {new_status}. Pool status: {pool['status']}",
            data={
                "pool_id": pool_id,
                "subtask_id": subtask_id,
                "subtask_status": new_status,
                "pool_status": pool["status"],
                "summary": self._summarize_subtasks(pool),
            },
        )

    async def _cancel_pool(self, params: Dict) -> SkillResult:
        """Cancel a running pool."""
        pool_id = params.get("pool_id", "").strip()
        if not pool_id:
            return SkillResult(success=False, message="pool_id is required")

        state = self._load()
        pool = self._find_pool(state, pool_id)
        if not pool:
            return SkillResult(success=False, message=f"Pool '{pool_id}' not found")

        if pool["status"] in ("completed", "cancelled"):
            return SkillResult(
                success=False,
                message=f"Pool is already '{pool['status']}'",
            )

        reason = params.get("reason", "Cancelled by agent")
        pool["status"] = "cancelled"
        pool["completed_at"] = datetime.now().isoformat()

        cancelled_count = 0
        for st in pool["subtasks"]:
            if st["status"] in ("pending", "assigned", "running"):
                st["status"] = "cancelled"
                st["error"] = reason
                cancelled_count += 1

        self._save(state)

        return SkillResult(
            success=True,
            message=f"Pool '{pool['name']}' cancelled. {cancelled_count} subtask(s) cancelled.",
            data={
                "pool_id": pool_id,
                "cancelled_subtasks": cancelled_count,
                "reason": reason,
            },
        )

    async def _retry_failed(self, params: Dict) -> SkillResult:
        """Retry failed subtasks."""
        pool_id = params.get("pool_id", "").strip()
        subtask_id = params.get("subtask_id", "").strip()

        if not pool_id:
            return SkillResult(success=False, message="pool_id is required")

        state = self._load()
        pool = self._find_pool(state, pool_id)
        if not pool:
            return SkillResult(success=False, message=f"Pool '{pool_id}' not found")

        retried = []
        skipped = []

        subtasks_to_retry = pool["subtasks"]
        if subtask_id:
            subtask = self._find_subtask(pool, subtask_id)
            if not subtask:
                return SkillResult(success=False, message=f"Subtask '{subtask_id}' not found")
            subtasks_to_retry = [subtask]

        for st in subtasks_to_retry:
            if st["status"] != "failed":
                continue

            if st.get("retries", 0) >= MAX_RETRIES:
                skipped.append({"id": st["id"], "name": st["name"], "reason": "max retries reached"})
                continue

            st["status"] = "pending"
            st["error"] = None
            st["retries"] = st.get("retries", 0) + 1
            st["started_at"] = None
            st["completed_at"] = None
            state["stats"]["total_retries"] += 1
            retried.append(st["name"])

        # Reset pool status if it was failed
        if retried and pool["status"] == "failed":
            pool["status"] = "running"

        self._save(state)

        return SkillResult(
            success=True,
            message=f"Retried {len(retried)} subtask(s), skipped {len(skipped)}",
            data={
                "retried": retried,
                "skipped": skipped,
                "pool_status": pool["status"],
            },
        )

    async def _get_results(self, params: Dict) -> SkillResult:
        """Get aggregated results from a pool."""
        pool_id = params.get("pool_id", "").strip()
        if not pool_id:
            return SkillResult(success=False, message="pool_id is required")

        state = self._load()
        pool = self._find_pool(state, pool_id)
        if not pool:
            return SkillResult(success=False, message=f"Pool '{pool_id}' not found")

        results = {}
        errors = {}
        for st in pool["subtasks"]:
            if st["status"] == "completed" and st.get("result") is not None:
                results[st["name"]] = st["result"]
            elif st["status"] == "failed":
                errors[st["name"]] = st.get("error", "Unknown error")

        # Aggregate based on pattern
        aggregated = None
        if pool["pattern"] == "pipeline":
            # Pipeline: result is the output of the last completed subtask
            ordered = sorted(
                [st for st in pool["subtasks"] if st["status"] == "completed"],
                key=lambda s: s.get("order", 0),
            )
            if ordered:
                aggregated = ordered[-1].get("result")

        elif pool["pattern"] == "map_reduce":
            # Map-reduce: collect all results into a list
            ordered = sorted(
                [st for st in pool["subtasks"] if st["status"] == "completed"],
                key=lambda s: s.get("order", 0),
            )
            aggregated = [st.get("result") for st in ordered]

        else:  # fan_out
            # Fan-out: all results as a dict
            aggregated = results

        return SkillResult(
            success=True,
            message=f"Results from '{pool['name']}': {len(results)} completed, {len(errors)} failed",
            data={
                "pool_id": pool_id,
                "pool_status": pool["status"],
                "pattern": pool["pattern"],
                "results": results,
                "errors": errors,
                "aggregated": aggregated,
                "summary": self._summarize_subtasks(pool),
            },
        )

    async def _stats(self, params: Dict) -> SkillResult:
        """Get delegation statistics."""
        state = self._load()
        stats = state.get("stats", {})

        active_pools = [p for p in state["pools"] if p["status"] == "running"]
        pending_pools = [p for p in state["pools"] if p["status"] == "pending"]

        return SkillResult(
            success=True,
            message=f"Delegation stats: {stats.get('total_pools_created', 0)} pools, {stats.get('total_subtasks_completed', 0)} completed subtasks",
            data={
                "stats": stats,
                "active_pools": len(active_pools),
                "pending_pools": len(pending_pools),
                "registered_workers": list(self._workers.keys()),
            },
        )

    # === Helper methods ===

    def _find_pool(self, state: Dict, pool_id: str) -> Optional[Dict]:
        for p in state["pools"]:
            if p["id"] == pool_id:
                return p
        return None

    def _find_subtask(self, pool: Dict, subtask_id: str) -> Optional[Dict]:
        for st in pool["subtasks"]:
            if st["id"] == subtask_id:
                return st
        return None

    def _get_ready_subtasks(self, pool: Dict) -> List[Dict]:
        """Get subtasks whose dependencies are all met."""
        completed_ids = {
            st["id"] for st in pool["subtasks"] if st["status"] == "completed"
        }

        ready = []
        for st in pool["subtasks"]:
            if st["status"] != "pending":
                continue

            deps = st.get("depends_on", [])
            if all(dep_id in completed_ids for dep_id in deps):
                ready.append(st)

        return ready

    def _check_pool_completion(self, pool: Dict):
        """Check if a pool is complete (all subtasks done or failed)."""
        statuses = [st["status"] for st in pool["subtasks"]]

        # All completed
        if all(s == "completed" for s in statuses):
            pool["status"] = "completed"
            pool["completed_at"] = datetime.now().isoformat()
            return

        # All finished (completed, failed, or cancelled)
        terminal = {"completed", "failed", "cancelled"}
        if all(s in terminal for s in statuses):
            # If any failed, pool is failed; if all cancelled, pool is cancelled
            if any(s == "failed" for s in statuses):
                pool["status"] = "failed"
            else:
                pool["status"] = "cancelled"
            pool["completed_at"] = datetime.now().isoformat()
            return

    def _summarize_subtasks(self, pool: Dict) -> Dict:
        """Summarize subtask statuses."""
        summary = {}
        for st in pool["subtasks"]:
            status = st["status"]
            summary[status] = summary.get(status, 0) + 1
        summary["total"] = len(pool["subtasks"])
        return summary

    async def initialize(self) -> bool:
        self.initialized = True
        return True
