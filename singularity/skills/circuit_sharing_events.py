#!/usr/bin/env python3
"""
CircuitSharingEventBridgeSkill - Emit EventBus events when circuit states are shared across agents.

When CrossAgentCircuitSharingSkill imports, syncs, or resolves conflicts with peer
circuit breaker states, this bridge emits structured EventBus events so downstream
skills can react:

Events emitted:
- circuit_sharing.state_adopted: A peer's circuit state was adopted locally
- circuit_sharing.sync_completed: A sync operation (pull/publish/sync) finished
- circuit_sharing.conflict_resolved: A merge conflict was resolved between local and peer states
- circuit_sharing.peer_discovered: A new peer appeared in the shared store
- circuit_sharing.fleet_alert: Significant fleet-wide pattern detected (e.g., many circuits open)

This enables reactive automation:
- AlertIncidentBridge can create fleet-wide incidents when many replicas report failures
- StrategySkill can adjust priorities when fleet capacity drops
- FleetHealthManager can auto-heal replicas affected by shared circuit openings
- ServiceMonitor can proactively degrade services before local failures hit

Architecture:
  CircuitSharingSkill imports/syncs → CircuitSharingEventBridge detects changes →
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
BRIDGE_STATE_FILE = DATA_DIR / "circuit_sharing_events.json"
MAX_EVENT_HISTORY = 200
MAX_KNOWN_PEERS = 50


def _now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


class CircuitSharingEventBridgeSkill(Skill):
    """
    Bridge between CrossAgentCircuitSharingSkill and EventBus.

    Monitors circuit sharing operations and emits structured events
    so the rest of the agent ecosystem can react to fleet-wide
    circuit state changes.

    Actions:
    - monitor: Check circuit sharing state for changes and emit events
    - configure: Update event emission settings
    - status: View bridge health and emission statistics
    - history: View recent emitted events
    - emit_test: Emit a test event to verify EventBus integration
    - fleet_check: Analyze shared store for fleet-wide patterns and emit alerts
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
                self._known_peers = data.get("known_peers", {})
                self._last_sync_snapshot = data.get("last_sync_snapshot", {})
                self._event_history = data.get("event_history", [])[-MAX_EVENT_HISTORY:]
                self._config = data.get("config", self._default_config())
                self._stats = data.get("stats", self._default_stats())
            except (json.JSONDecodeError, Exception):
                self._init_empty()
        else:
            self._init_empty()

    def _init_empty(self):
        self._known_peers: Dict[str, Dict] = {}
        self._last_sync_snapshot: Dict = {}
        self._event_history: List[Dict] = []
        self._config = self._default_config()
        self._stats = self._default_stats()

    def _default_config(self) -> Dict:
        return {
            "emit_on_state_adopted": True,
            "emit_on_sync_completed": True,
            "emit_on_conflict_resolved": True,
            "emit_on_peer_discovered": True,
            "emit_on_fleet_alert": True,
            "fleet_open_threshold": 0.5,  # Alert when >50% of circuits are open fleet-wide
            "event_source": "circuit_sharing_event_bridge",
            "priority_state_adopted": "high",
            "priority_sync_completed": "normal",
            "priority_conflict_resolved": "high",
            "priority_peer_discovered": "normal",
            "priority_fleet_alert": "critical",
        }

    def _default_stats(self) -> Dict:
        return {
            "events_emitted": 0,
            "events_failed": 0,
            "monitors_run": 0,
            "fleet_checks_run": 0,
            "peers_discovered": 0,
            "states_adopted_detected": 0,
            "conflicts_detected": 0,
            "fleet_alerts_emitted": 0,
            "last_monitor_time": None,
            "last_fleet_check_time": None,
        }

    def _save_state(self):
        """Persist bridge state to disk."""
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        data = {
            "known_peers": self._known_peers,
            "last_sync_snapshot": self._last_sync_snapshot,
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
            skill_id="circuit_sharing_events",
            name="Circuit Sharing Event Bridge",
            version="1.0.0",
            category="replication",
            description=(
                "Emit EventBus events when circuit breaker states are shared "
                "across agent replicas. Enables fleet-wide reactive automation."
            ),
            actions=[
                SkillAction(
                    name="monitor",
                    description=(
                        "Check circuit sharing state for changes since last "
                        "monitor call and emit events for adoptions, conflicts, "
                        "and new peers."
                    ),
                    parameters={},
                    estimated_cost=0,
                ),
                SkillAction(
                    name="configure",
                    description="Update event emission settings and thresholds",
                    parameters={
                        "emit_on_state_adopted": {
                            "type": "bool", "required": False,
                            "description": "Emit when a peer's circuit state is adopted locally",
                        },
                        "emit_on_sync_completed": {
                            "type": "bool", "required": False,
                            "description": "Emit when a sync operation completes",
                        },
                        "emit_on_conflict_resolved": {
                            "type": "bool", "required": False,
                            "description": "Emit when a merge conflict is resolved",
                        },
                        "emit_on_peer_discovered": {
                            "type": "bool", "required": False,
                            "description": "Emit when a new peer appears in shared store",
                        },
                        "emit_on_fleet_alert": {
                            "type": "bool", "required": False,
                            "description": "Emit fleet-wide alert events",
                        },
                        "fleet_open_threshold": {
                            "type": "float", "required": False,
                            "description": "Alert when fraction of circuits open exceeds this (0-1)",
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="status",
                    description="View bridge health, emission statistics, and known peers",
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
                        "Analyze shared circuit store for fleet-wide patterns "
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
        """Check circuit sharing state for changes and emit events."""
        self._stats["monitors_run"] += 1
        self._stats["last_monitor_time"] = _now_iso()

        events_emitted = 0
        events_detail = []

        # Get current circuit sharing status
        sharing_status = await self._get_sharing_status()
        if sharing_status is None:
            self._save_state()
            return SkillResult(
                success=True,
                message="Circuit sharing skill not available, no events emitted",
                data={"events_emitted": 0},
            )

        # Get current sync history from circuit sharing
        sharing_history = await self._get_sharing_history()

        # Detect new peers
        current_peers = sharing_status.get("peers", [])
        for peer in current_peers:
            peer_id = peer.get("peer_id", "")
            if peer_id and peer_id not in self._known_peers:
                self._known_peers[peer_id] = {
                    "discovered_at": _now_iso(),
                    "circuits": peer.get("circuits", 0),
                }
                self._stats["peers_discovered"] += 1
                if self._config["emit_on_peer_discovered"]:
                    emitted = await self._emit_event(
                        "circuit_sharing.peer_discovered",
                        {
                            "peer_id": peer_id,
                            "circuits": peer.get("circuits", 0),
                            "open_circuits": peer.get("open_circuits", 0),
                            "timestamp": _now_iso(),
                        },
                        self._config["priority_peer_discovered"],
                    )
                    if emitted:
                        events_emitted += 1
                        events_detail.append(f"peer_discovered:{peer_id}")
            elif peer_id:
                # Update known peer info
                self._known_peers[peer_id]["circuits"] = peer.get("circuits", 0)
                self._known_peers[peer_id]["last_seen"] = _now_iso()

        # Trim known peers
        if len(self._known_peers) > MAX_KNOWN_PEERS:
            sorted_peers = sorted(
                self._known_peers.items(),
                key=lambda x: x[1].get("last_seen", ""),
            )
            self._known_peers = dict(sorted_peers[-MAX_KNOWN_PEERS:])

        # Detect state adoptions and conflicts from sync history
        if sharing_history:
            history_entries = sharing_history.get("history", [])
            last_monitor = self._last_sync_snapshot.get("last_entry_ts", "")

            new_entries = [
                e for e in history_entries
                if e.get("timestamp", "") > last_monitor
            ]

            for entry in new_entries:
                op = entry.get("operation", "")
                adopted = entry.get("states_adopted", 0)
                agent_id = entry.get("agent_id", "")

                # Emit sync completed event
                if op in ("sync", "pull", "import") and self._config["emit_on_sync_completed"]:
                    emitted = await self._emit_event(
                        "circuit_sharing.sync_completed",
                        {
                            "operation": op,
                            "agent_id": agent_id,
                            "states_adopted": adopted,
                            "circuits_processed": entry.get("circuits_processed", 0),
                            "strategy": entry.get("strategy", "unknown"),
                            "timestamp": entry.get("timestamp", _now_iso()),
                        },
                        self._config["priority_sync_completed"],
                    )
                    if emitted:
                        events_emitted += 1
                        events_detail.append(f"sync_completed:{op}")

                # Emit state adopted events
                if adopted > 0 and self._config["emit_on_state_adopted"]:
                    self._stats["states_adopted_detected"] += adopted
                    emitted = await self._emit_event(
                        "circuit_sharing.state_adopted",
                        {
                            "source_agent": agent_id,
                            "states_adopted": adopted,
                            "operation": op,
                            "strategy": entry.get("strategy", "unknown"),
                            "timestamp": entry.get("timestamp", _now_iso()),
                        },
                        self._config["priority_state_adopted"],
                    )
                    if emitted:
                        events_emitted += 1
                        events_detail.append(f"state_adopted:{adopted}")

                # Emit conflict resolved events (adoption implies conflict resolution)
                if adopted > 0 and self._config["emit_on_conflict_resolved"]:
                    self._stats["conflicts_detected"] += adopted
                    emitted = await self._emit_event(
                        "circuit_sharing.conflict_resolved",
                        {
                            "source_agent": agent_id,
                            "conflicts_resolved": adopted,
                            "resolution_strategy": entry.get("strategy", "unknown"),
                            "operation": op,
                            "timestamp": entry.get("timestamp", _now_iso()),
                        },
                        self._config["priority_conflict_resolved"],
                    )
                    if emitted:
                        events_emitted += 1
                        events_detail.append(f"conflict_resolved:{adopted}")

            # Update snapshot watermark
            if history_entries:
                self._last_sync_snapshot["last_entry_ts"] = max(
                    e.get("timestamp", "") for e in history_entries
                )

        self._save_state()

        return SkillResult(
            success=True,
            message=f"Monitor complete: {events_emitted} events emitted, {len(current_peers)} peers tracked",
            data={
                "events_emitted": events_emitted,
                "events_detail": events_detail,
                "peers_tracked": len(self._known_peers),
            },
        )

    async def _fleet_check(self, params: Dict) -> SkillResult:
        """Analyze shared store for fleet-wide circuit patterns."""
        self._stats["fleet_checks_run"] += 1
        self._stats["last_fleet_check_time"] = _now_iso()

        sharing_status = await self._get_sharing_status()
        if sharing_status is None:
            self._save_state()
            return SkillResult(
                success=True,
                message="Circuit sharing skill not available",
                data={"fleet_alert_emitted": False},
            )

        peers = sharing_status.get("peers", [])
        if not peers:
            self._save_state()
            return SkillResult(
                success=True,
                message="No peers in shared store, nothing to analyze",
                data={"fleet_alert_emitted": False, "peers_count": 0},
            )

        # Aggregate circuit states across all peers
        total_circuits = 0
        total_open = 0
        peer_reports = []

        for peer in peers:
            circuits = peer.get("circuits", 0)
            open_circuits = peer.get("open_circuits", 0)
            total_circuits += circuits
            total_open += open_circuits
            peer_reports.append({
                "peer_id": peer.get("peer_id", "unknown"),
                "circuits": circuits,
                "open_circuits": open_circuits,
                "open_fraction": open_circuits / max(circuits, 1),
            })

        fleet_open_fraction = total_open / max(total_circuits, 1)
        threshold = self._config["fleet_open_threshold"]
        alert_emitted = False

        if fleet_open_fraction > threshold and self._config["emit_on_fleet_alert"]:
            self._stats["fleet_alerts_emitted"] += 1
            emitted = await self._emit_event(
                "circuit_sharing.fleet_alert",
                {
                    "alert_type": "high_open_rate",
                    "fleet_open_fraction": round(fleet_open_fraction, 3),
                    "threshold": threshold,
                    "total_circuits": total_circuits,
                    "total_open": total_open,
                    "peers_count": len(peers),
                    "peer_reports": peer_reports,
                    "message": (
                        f"Fleet alert: {total_open}/{total_circuits} circuits open "
                        f"({fleet_open_fraction:.0%}) exceeds {threshold:.0%} threshold"
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
                f"Fleet check: {total_open}/{total_circuits} circuits open "
                f"({fleet_open_fraction:.0%}), "
                f"alert {'emitted' if alert_emitted else 'not needed'}"
            ),
            data={
                "fleet_open_fraction": round(fleet_open_fraction, 3),
                "total_circuits": total_circuits,
                "total_open": total_open,
                "peers_count": len(peers),
                "threshold": threshold,
                "fleet_alert_emitted": alert_emitted,
                "peer_reports": peer_reports,
            },
        )

    async def _configure(self, params: Dict) -> SkillResult:
        """Update event emission configuration."""
        updated = []

        for key in (
            "emit_on_state_adopted",
            "emit_on_sync_completed",
            "emit_on_conflict_resolved",
            "emit_on_peer_discovered",
            "emit_on_fleet_alert",
        ):
            if key in params:
                self._config[key] = bool(params[key])
                updated.append(f"{key}={params[key]}")

        if "fleet_open_threshold" in params:
            val = float(params["fleet_open_threshold"])
            self._config["fleet_open_threshold"] = max(0.0, min(1.0, val))
            updated.append(f"fleet_open_threshold={self._config['fleet_open_threshold']}")

        for key in (
            "priority_state_adopted",
            "priority_sync_completed",
            "priority_conflict_resolved",
            "priority_peer_discovered",
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
        lines = ["=== Circuit Sharing Event Bridge Status ==="]
        lines.append(f"Events emitted: {self._stats['events_emitted']}")
        lines.append(f"Events failed: {self._stats['events_failed']}")
        lines.append(f"Monitors run: {self._stats['monitors_run']}")
        lines.append(f"Fleet checks run: {self._stats['fleet_checks_run']}")
        lines.append(f"Peers discovered: {self._stats['peers_discovered']}")
        lines.append(f"States adopted detected: {self._stats['states_adopted_detected']}")
        lines.append(f"Conflicts detected: {self._stats['conflicts_detected']}")
        lines.append(f"Fleet alerts emitted: {self._stats['fleet_alerts_emitted']}")
        lines.append(f"Last monitor: {self._stats['last_monitor_time'] or 'never'}")
        lines.append(f"Last fleet check: {self._stats['last_fleet_check_time'] or 'never'}")
        lines.append(f"Known peers: {len(self._known_peers)}")
        for pid, pinfo in self._known_peers.items():
            lines.append(f"  - {pid}: {pinfo.get('circuits', 0)} circuits, discovered {pinfo.get('discovered_at', '?')}")

        return SkillResult(
            success=True,
            message="\n".join(lines),
            data={
                "stats": self._stats,
                "config": self._config,
                "known_peers": self._known_peers,
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
            "circuit_sharing.test",
            {
                "message": "Test event from CircuitSharingEventBridge",
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

    async def _get_sharing_status(self) -> Optional[Dict]:
        """Get circuit sharing status via skill context."""
        try:
            if self.context:
                result = await self.context.call_skill(
                    "circuit_sharing", "status", {}
                )
                if result and result.success:
                    return result.data
        except Exception:
            pass
        return None

    async def _get_sharing_history(self) -> Optional[Dict]:
        """Get circuit sharing sync history via skill context."""
        try:
            if self.context:
                result = await self.context.call_skill(
                    "circuit_sharing", "history", {"limit": 50}
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
                        "source": self._config.get("event_source", "circuit_sharing_event_bridge"),
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
                        "source": self._config.get("event_source", "circuit_sharing_event_bridge"),
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
