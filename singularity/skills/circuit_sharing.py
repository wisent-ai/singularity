#!/usr/bin/env python3
"""
CrossAgentCircuitSharingSkill - Share circuit breaker states across replicas.

When multiple agent replicas operate autonomously, each discovers skill failures
independently. This wastes budget: if replica A finds that an API is down, replicas
B, C, and D still burn budget learning the same lesson. Cross-agent circuit sharing
solves this by broadcasting circuit state changes across the fleet.

How it works:
1. **Export**: Serialize this agent's circuit breaker states into a shareable snapshot
2. **Import**: Merge another agent's circuit states into the local circuit breaker
3. **Shared Store**: Read/write circuit states to a shared JSON file that all replicas access
4. **Sync**: Pull from shared store + push local states in one operation
5. **Conflict Resolution**: When two agents disagree on a circuit's state, configurable
   merge strategies determine the winner (pessimistic, optimistic, or majority)

Merge strategies:
- **pessimistic** (default): If ANY agent has a circuit OPEN, keep it open locally.
  Safest for budget protection - one failure signal blocks everyone.
- **optimistic**: Only adopt OPEN state if local circuit also shows failures.
  Allows agents to independently verify before blocking.
- **newest**: Adopt whichever state was most recently updated.
  Good for fast convergence but can oscillate.

Integration with existing skills:
- Reads from CircuitBreakerSkill via SkillContext
- Can be triggered by SchedulerSkill for periodic sync
- Emits events via EventBus when remote circuit states are imported
- Works with AgentSpawnerSkill replicas sharing a mounted volume

Pillar: Replication (primary) + Self-Improvement (fleet learning)
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import Skill, SkillAction, SkillManifest, SkillResult

DATA_DIR = Path(__file__).parent.parent / "data"
SHARED_CIRCUIT_FILE = DATA_DIR / "shared_circuits.json"
SYNC_HISTORY_FILE = DATA_DIR / "circuit_sync_history.json"
MAX_SYNC_HISTORY = 100


class CrossAgentCircuitSharingSkill(Skill):
    """
    Share circuit breaker states across agent replicas.

    Actions:
    - export: Export local circuit states as a shareable snapshot
    - import_states: Import another agent's circuit states with merge strategy
    - sync: Pull shared states + push local states (bidirectional)
    - publish: Write local circuit states to shared store
    - pull: Read shared store and merge into local circuits
    - status: View sync status, last sync time, conflict history
    - configure: Update merge strategy and sync settings
    - history: View sync operation history
    """

    def __init__(self, credentials: Dict[str, str] = None):
        super().__init__(credentials)
        self._config = {
            "merge_strategy": "pessimistic",  # pessimistic | optimistic | newest
            "agent_id": "",  # Set from context or config
            "shared_store_path": str(SHARED_CIRCUIT_FILE),
            "auto_sync_on_state_change": True,
            "import_open_circuits": True,  # Import OPEN states from peers
            "import_half_open_circuits": False,  # Usually don't import transitional states
            "min_peer_window_size": 3,  # Peer needs at least N data points
            "trust_threshold": 0.5,  # Minimum trust to accept peer data (0-1)
        }
        self._sync_history: List[Dict] = []
        self._peer_states: Dict[str, Dict] = {}  # agent_id -> last known states
        self._last_sync_time: float = 0.0
        self._conflicts_resolved: int = 0
        self._states_imported: int = 0
        self._states_exported: int = 0
        self._load_history()

    def _load_history(self):
        """Load sync history from disk."""
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        try:
            with open(SYNC_HISTORY_FILE, "r") as f:
                data = json.load(f)
            self._sync_history = data.get("history", [])[-MAX_SYNC_HISTORY:]
            self._peer_states = data.get("peer_states", {})
            self._last_sync_time = data.get("last_sync_time", 0.0)
            self._conflicts_resolved = data.get("conflicts_resolved", 0)
            self._states_imported = data.get("states_imported", 0)
            self._states_exported = data.get("states_exported", 0)
            self._config.update(data.get("config", {}))
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            pass

    def _save_history(self):
        """Save sync history to disk."""
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        data = {
            "history": self._sync_history[-MAX_SYNC_HISTORY:],
            "peer_states": self._peer_states,
            "last_sync_time": self._last_sync_time,
            "conflicts_resolved": self._conflicts_resolved,
            "states_imported": self._states_imported,
            "states_exported": self._states_exported,
            "config": self._config,
            "last_updated": datetime.now().isoformat(),
        }
        with open(SYNC_HISTORY_FILE, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def _get_agent_id(self) -> str:
        """Get the agent ID from config or context."""
        if self._config.get("agent_id"):
            return self._config["agent_id"]
        if self.context:
            return self.context.agent_name
        return "unknown"

    def _get_local_circuits(self) -> Optional[Dict]:
        """Get circuit states from local CircuitBreakerSkill via context."""
        if not self.context:
            return None
        cb_skill = self.context.get_skill("circuit_breaker")
        if not cb_skill:
            return None
        # Access internal state directly for efficiency
        circuits = {}
        for skill_id, circuit in cb_skill._circuits.items():
            circuits[skill_id] = circuit.to_dict()
        return circuits

    def _apply_remote_state(self, skill_id: str, remote: Dict, source_agent: str) -> Dict:
        """
        Apply a remote circuit state to local circuit breaker.

        Returns a dict describing what happened.
        """
        if not self.context:
            return {"action": "skipped", "reason": "no context"}

        cb_skill = self.context.get_skill("circuit_breaker")
        if not cb_skill:
            return {"action": "skipped", "reason": "no circuit_breaker skill"}

        local_circuit = cb_skill._get_circuit(skill_id)
        local_state = local_circuit.state.value
        remote_state = remote.get("state", "closed")
        strategy = self._config["merge_strategy"]

        # Don't override manual overrides
        if local_state in ("forced_open", "forced_closed"):
            return {
                "action": "skipped",
                "reason": f"local is manually {local_state}",
                "local_state": local_state,
                "remote_state": remote_state,
            }

        # Check minimum data requirement from peer
        remote_window = remote.get("window_size", 0)
        if remote_window < self._config["min_peer_window_size"]:
            return {
                "action": "skipped",
                "reason": f"peer has insufficient data ({remote_window} < {self._config['min_peer_window_size']})",
                "local_state": local_state,
                "remote_state": remote_state,
            }

        # Determine if we should adopt the remote state
        should_adopt = False
        reason = ""

        if strategy == "pessimistic":
            # If remote is OPEN and local isn't, adopt OPEN
            if remote_state == "open" and local_state in ("closed", "half_open"):
                should_adopt = True
                reason = f"pessimistic: peer {source_agent} reports circuit open"
            # If remote is CLOSED and local is OPEN, check remote success data
            elif remote_state == "closed" and local_state == "open":
                remote_successes = remote.get("consecutive_successes", 0)
                if remote_successes >= 3:
                    should_adopt = True
                    reason = f"pessimistic: peer {source_agent} recovered ({remote_successes} successes)"

        elif strategy == "optimistic":
            # Only adopt OPEN if local also shows problems
            if remote_state == "open" and local_state == "closed":
                local_rate = local_circuit.failure_rate()
                if local_rate > 0.3:  # Local also failing
                    should_adopt = True
                    reason = f"optimistic: peer {source_agent} open + local failure rate {local_rate:.0%}"
            elif remote_state == "closed" and local_state == "open":
                should_adopt = True
                reason = f"optimistic: peer {source_agent} reports recovered"

        elif strategy == "newest":
            remote_change = remote.get("last_state_change")
            local_change_ts = local_circuit.last_state_change
            if remote_change and remote_state != local_state:
                # Parse ISO timestamp from remote
                try:
                    remote_ts = datetime.fromisoformat(remote_change).timestamp()
                except (ValueError, TypeError):
                    remote_ts = 0
                if remote_ts > local_change_ts:
                    should_adopt = True
                    reason = f"newest: peer {source_agent} state is more recent"

        if not should_adopt:
            return {
                "action": "kept_local",
                "reason": f"strategy '{strategy}' kept local state",
                "local_state": local_state,
                "remote_state": remote_state,
            }

        # Apply the state change
        from .circuit_breaker import CircuitState

        try:
            new_state = CircuitState(remote_state)
        except ValueError:
            return {"action": "error", "reason": f"invalid remote state: {remote_state}"}

        old_state = local_circuit.state
        local_circuit.state = new_state
        local_circuit.last_state_change = time.time()
        if new_state == CircuitState.OPEN:
            local_circuit.opened_count += 1
        cb_skill._save_state()

        self._states_imported += 1
        self._conflicts_resolved += 1

        return {
            "action": "adopted",
            "reason": reason,
            "old_state": old_state.value,
            "new_state": new_state.value,
            "source_agent": source_agent,
        }

    def _read_shared_store(self) -> Dict:
        """Read the shared circuit store file."""
        store_path = Path(self._config["shared_store_path"])
        try:
            with open(store_path, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {"agents": {}, "last_updated": None}

    def _write_shared_store(self, data: Dict):
        """Write to the shared circuit store file."""
        store_path = Path(self._config["shared_store_path"])
        store_path.parent.mkdir(parents=True, exist_ok=True)
        with open(store_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="circuit_sharing",
            name="Cross-Agent Circuit Sharing",
            version="1.0.0",
            category="replication",
            description="Share circuit breaker states across agent replicas for fleet-wide failure awareness",
            actions=[
                SkillAction(
                    name="export",
                    description="Export local circuit states as a shareable snapshot",
                    parameters={},
                    estimated_cost=0,
                ),
                SkillAction(
                    name="import_states",
                    description="Import another agent's circuit states with merge strategy",
                    parameters={
                        "agent_id": {"type": "string", "required": True, "description": "Source agent ID"},
                        "circuits": {"type": "object", "required": True, "description": "Circuit states dict from export"},
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="sync",
                    description="Pull shared states + push local states (bidirectional sync)",
                    parameters={},
                    estimated_cost=0,
                ),
                SkillAction(
                    name="publish",
                    description="Write local circuit states to shared store",
                    parameters={},
                    estimated_cost=0,
                ),
                SkillAction(
                    name="pull",
                    description="Read shared store and merge peer states into local circuits",
                    parameters={},
                    estimated_cost=0,
                ),
                SkillAction(
                    name="status",
                    description="View sync status, peer info, and conflict history",
                    parameters={},
                    estimated_cost=0,
                ),
                SkillAction(
                    name="configure",
                    description="Update merge strategy and sync settings",
                    parameters={
                        "merge_strategy": {"type": "string", "required": False, "description": "pessimistic|optimistic|newest"},
                        "agent_id": {"type": "string", "required": False, "description": "This agent's ID"},
                        "shared_store_path": {"type": "string", "required": False, "description": "Path to shared store file"},
                        "min_peer_window_size": {"type": "integer", "required": False, "description": "Min data points to trust peer"},
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="history",
                    description="View sync operation history",
                    parameters={
                        "limit": {"type": "integer", "required": False, "description": "Number of entries (default 10)"},
                    },
                    estimated_cost=0,
                ),
            ],
            required_credentials=[],
        )

    def check_credentials(self) -> bool:
        return True

    async def execute(self, action: str, params: Dict) -> SkillResult:
        actions = {
            "export": self._export,
            "import_states": self._import_states,
            "sync": self._sync,
            "publish": self._publish,
            "pull": self._pull,
            "status": self._status,
            "configure": self._configure,
            "history": self._history,
        }
        handler = actions.get(action)
        if not handler:
            return SkillResult(
                success=False,
                message=f"Unknown action: {action}. Available: {list(actions.keys())}",
            )
        try:
            return handler(params)
        except Exception as e:
            return SkillResult(success=False, message=f"Error in {action}: {e}")

    def _export(self, params: Dict) -> SkillResult:
        """Export local circuit states as a snapshot."""
        circuits = self._get_local_circuits()
        if circuits is None:
            return SkillResult(
                success=False,
                message="Cannot export: CircuitBreakerSkill not accessible via context",
            )

        agent_id = self._get_agent_id()
        snapshot = {
            "agent_id": agent_id,
            "timestamp": datetime.now().isoformat(),
            "circuit_count": len(circuits),
            "circuits": circuits,
        }

        self._states_exported += len(circuits)
        self._log_sync("export", agent_id, len(circuits), 0, [])
        self._save_history()

        open_count = sum(1 for c in circuits.values() if c.get("state") in ("open", "forced_open"))
        return SkillResult(
            success=True,
            message=f"Exported {len(circuits)} circuit states ({open_count} open) from agent '{agent_id}'",
            data=snapshot,
        )

    def _import_states(self, params: Dict) -> SkillResult:
        """Import circuit states from another agent."""
        source_agent = params.get("agent_id", "")
        circuits = params.get("circuits", {})

        if not source_agent:
            return SkillResult(success=False, message="agent_id is required")
        if not circuits:
            return SkillResult(success=False, message="circuits dict is required")

        results = []
        adopted = 0
        skipped = 0

        for skill_id, circuit_data in circuits.items():
            result = self._apply_remote_state(skill_id, circuit_data, source_agent)
            results.append({"skill_id": skill_id, **result})
            if result["action"] == "adopted":
                adopted += 1
            else:
                skipped += 1

        # Track peer states
        self._peer_states[source_agent] = {
            "last_import": datetime.now().isoformat(),
            "circuit_count": len(circuits),
            "states_adopted": adopted,
        }

        self._log_sync("import", source_agent, len(circuits), adopted, results)
        self._save_history()

        return SkillResult(
            success=True,
            message=f"Imported from '{source_agent}': {adopted} adopted, {skipped} skipped ({self._config['merge_strategy']} strategy)",
            data={
                "source_agent": source_agent,
                "total_circuits": len(circuits),
                "adopted": adopted,
                "skipped": skipped,
                "details": results,
            },
        )

    def _publish(self, params: Dict) -> SkillResult:
        """Write local circuit states to the shared store."""
        circuits = self._get_local_circuits()
        if circuits is None:
            return SkillResult(
                success=False,
                message="Cannot publish: CircuitBreakerSkill not accessible via context",
            )

        agent_id = self._get_agent_id()
        store = self._read_shared_store()

        store["agents"][agent_id] = {
            "circuits": circuits,
            "timestamp": datetime.now().isoformat(),
            "circuit_count": len(circuits),
        }
        store["last_updated"] = datetime.now().isoformat()

        self._write_shared_store(store)
        self._states_exported += len(circuits)
        self._log_sync("publish", agent_id, len(circuits), 0, [])
        self._save_history()

        return SkillResult(
            success=True,
            message=f"Published {len(circuits)} circuit states to shared store as '{agent_id}'",
            data={
                "agent_id": agent_id,
                "circuits_published": len(circuits),
                "total_agents_in_store": len(store["agents"]),
            },
        )

    def _pull(self, params: Dict) -> SkillResult:
        """Read shared store and merge peer states."""
        agent_id = self._get_agent_id()
        store = self._read_shared_store()

        if not store.get("agents"):
            return SkillResult(
                success=True,
                message="Shared store is empty, nothing to pull",
                data={"peers_found": 0},
            )

        total_adopted = 0
        total_skipped = 0
        peer_results = []

        for peer_id, peer_data in store["agents"].items():
            if peer_id == agent_id:
                continue  # Skip our own data

            peer_circuits = peer_data.get("circuits", {})
            adopted = 0
            skipped = 0
            details = []

            for skill_id, circuit_data in peer_circuits.items():
                result = self._apply_remote_state(skill_id, circuit_data, peer_id)
                details.append({"skill_id": skill_id, **result})
                if result["action"] == "adopted":
                    adopted += 1
                else:
                    skipped += 1

            total_adopted += adopted
            total_skipped += skipped
            peer_results.append({
                "peer_id": peer_id,
                "circuits": len(peer_circuits),
                "adopted": adopted,
                "skipped": skipped,
            })

            # Track peer states
            self._peer_states[peer_id] = {
                "last_pull": datetime.now().isoformat(),
                "circuit_count": len(peer_circuits),
                "states_adopted": adopted,
            }

        self._last_sync_time = time.time()
        self._log_sync("pull", agent_id, total_adopted + total_skipped, total_adopted, peer_results)
        self._save_history()

        peers_found = len([p for p in store["agents"] if p != agent_id])
        return SkillResult(
            success=True,
            message=f"Pulled from {peers_found} peers: {total_adopted} states adopted, {total_skipped} skipped",
            data={
                "peers_found": peers_found,
                "total_adopted": total_adopted,
                "total_skipped": total_skipped,
                "peer_results": peer_results,
            },
        )

    def _sync(self, params: Dict) -> SkillResult:
        """Bidirectional sync: pull from shared store then publish local states."""
        pull_result = self._pull(params)
        publish_result = self._publish(params)

        success = pull_result.success and publish_result.success
        parts = []
        if pull_result.success:
            parts.append(f"Pull: {pull_result.data.get('total_adopted', 0)} adopted from {pull_result.data.get('peers_found', 0)} peers")
        else:
            parts.append(f"Pull failed: {pull_result.message}")
        if publish_result.success:
            parts.append(f"Publish: {publish_result.data.get('circuits_published', 0)} circuits shared")
        else:
            parts.append(f"Publish failed: {publish_result.message}")

        self._last_sync_time = time.time()
        self._log_sync("sync", self._get_agent_id(), 0, pull_result.data.get("total_adopted", 0), [])
        self._save_history()

        return SkillResult(
            success=success,
            message=" | ".join(parts),
            data={
                "pull": pull_result.data,
                "publish": publish_result.data,
            },
        )

    def _status(self, params: Dict) -> SkillResult:
        """View sync status and peer info."""
        agent_id = self._get_agent_id()
        store = self._read_shared_store()
        peers_in_store = [p for p in store.get("agents", {}) if p != agent_id]

        # Build peer summary
        peer_summary = []
        for peer_id in peers_in_store:
            peer_data = store["agents"][peer_id]
            peer_circuits = peer_data.get("circuits", {})
            open_count = sum(1 for c in peer_circuits.values() if c.get("state") in ("open", "forced_open"))
            peer_summary.append({
                "peer_id": peer_id,
                "circuits": len(peer_circuits),
                "open_circuits": open_count,
                "last_updated": peer_data.get("timestamp", "unknown"),
            })

        msg_lines = ["=== Circuit Sharing Status ==="]
        msg_lines.append(f"Agent: {agent_id}")
        msg_lines.append(f"Strategy: {self._config['merge_strategy']}")
        msg_lines.append(f"Last sync: {datetime.fromtimestamp(self._last_sync_time).isoformat() if self._last_sync_time else 'never'}")
        msg_lines.append(f"States imported: {self._states_imported} | exported: {self._states_exported}")
        msg_lines.append(f"Conflicts resolved: {self._conflicts_resolved}")
        msg_lines.append(f"Known peers: {len(peers_in_store)}")
        for ps in peer_summary:
            msg_lines.append(f"  - {ps['peer_id']}: {ps['circuits']} circuits ({ps['open_circuits']} open)")

        return SkillResult(
            success=True,
            message="\n".join(msg_lines),
            data={
                "agent_id": agent_id,
                "merge_strategy": self._config["merge_strategy"],
                "last_sync_time": self._last_sync_time,
                "states_imported": self._states_imported,
                "states_exported": self._states_exported,
                "conflicts_resolved": self._conflicts_resolved,
                "peers": peer_summary,
                "config": self._config,
            },
        )

    def _configure(self, params: Dict) -> SkillResult:
        """Update configuration."""
        updated = []
        valid_strategies = ("pessimistic", "optimistic", "newest")

        if "merge_strategy" in params:
            strategy = params["merge_strategy"]
            if strategy not in valid_strategies:
                return SkillResult(
                    success=False,
                    message=f"Invalid strategy '{strategy}'. Must be one of: {valid_strategies}",
                )
            self._config["merge_strategy"] = strategy
            updated.append(f"merge_strategy={strategy}")

        for key in ("agent_id", "shared_store_path"):
            if key in params:
                self._config[key] = str(params[key])
                updated.append(f"{key}={params[key]}")

        for key in ("min_peer_window_size",):
            if key in params:
                self._config[key] = int(params[key])
                updated.append(f"{key}={params[key]}")

        for key in ("auto_sync_on_state_change", "import_open_circuits", "import_half_open_circuits"):
            if key in params:
                self._config[key] = bool(params[key])
                updated.append(f"{key}={params[key]}")

        if "trust_threshold" in params:
            val = float(params["trust_threshold"])
            self._config["trust_threshold"] = max(0.0, min(1.0, val))
            updated.append(f"trust_threshold={self._config['trust_threshold']}")

        if not updated:
            return SkillResult(
                success=False,
                message="No valid configuration parameters provided",
            )

        self._save_history()
        return SkillResult(
            success=True,
            message=f"Updated: {', '.join(updated)}",
            data={"config": self._config},
        )

    def _history(self, params: Dict) -> SkillResult:
        """View sync operation history."""
        limit = int(params.get("limit", 10))
        recent = self._sync_history[-limit:]

        if not recent:
            return SkillResult(
                success=True,
                message="No sync history yet",
                data={"history": []},
            )

        msg_lines = [f"=== Sync History (last {len(recent)}) ==="]
        for entry in reversed(recent):
            ts = entry.get("timestamp", "?")
            op = entry.get("operation", "?")
            agent = entry.get("agent_id", "?")
            total = entry.get("circuits_processed", 0)
            adopted = entry.get("states_adopted", 0)
            msg_lines.append(f"  [{ts}] {op} ({agent}): {total} processed, {adopted} adopted")

        return SkillResult(
            success=True,
            message="\n".join(msg_lines),
            data={"history": recent},
        )

    def _log_sync(self, operation: str, agent_id: str, circuits_processed: int, states_adopted: int, details: list):
        """Log a sync operation."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "operation": operation,
            "agent_id": agent_id,
            "circuits_processed": circuits_processed,
            "states_adopted": states_adopted,
            "strategy": self._config["merge_strategy"],
        }
        self._sync_history.append(entry)
        if len(self._sync_history) > MAX_SYNC_HISTORY:
            self._sync_history = self._sync_history[-MAX_SYNC_HISTORY:]
