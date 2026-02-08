#!/usr/bin/env python3
"""
SkillEventBridgeSkill - Wires EventBus into existing skills for reactive cross-skill automation.

The agent has a powerful EventBus and many capable skills, but they operate in isolation.
This skill bridges the gap by:

1. WIRE  - Inject EventBus event emission into key skill lifecycle points
2. REACT - Set up cross-skill reactive subscriptions (e.g., health issue → auto-incident)
3. CHAIN - Enable event chains where one skill's output triggers another's input

Bridges built:
- IncidentResponse → EventBus: Emit incident.detected/triaged/resolved/escalated events
- SelfHealing → EventBus: Emit health.scan/repair/quarantine events
- SelfHealing → IncidentResponse: Health issues auto-create incidents
- IncidentResponse → AgentReputation: Incident handling updates reputation
- Any skill → EventBus: Generic skill execution event emission

This is critical infrastructure that transforms isolated skills into a reactive system.

Pillar: Self-Improvement (reactive architecture enables emergent behavior)

Actions:
- wire: Set up event bridges between skills and EventBus
- unwire: Remove specific event bridges
- trigger: Manually emit an event through a bridge (for testing)
- status: View active bridges, recent events, subscription health
- bridges: List all available bridge definitions
- history: View recent bridge-triggered events
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from .base import Skill, SkillResult, SkillManifest, SkillAction

BRIDGE_FILE = Path(__file__).parent.parent / "data" / "skill_event_bridge.json"
MAX_EVENT_LOG = 200
MAX_BRIDGES = 50


# Bridge definitions - what events to emit and what reactions to set up
BRIDGE_DEFINITIONS = {
    "incident_lifecycle": {
        "description": "Emit EventBus events for all incident lifecycle transitions",
        "source_skill": "incident_response",
        "events": [
            {"topic": "incident.detected", "on_action": "detect", "extract": ["incident_id", "severity", "status"]},
            {"topic": "incident.triaged", "on_action": "triage", "extract": ["incident_id", "severity", "status", "assignee"]},
            {"topic": "incident.responding", "on_action": "respond", "extract": ["incident_id", "action_type"]},
            {"topic": "incident.escalated", "on_action": "escalate", "extract": ["incident_id", "target", "severity"]},
            {"topic": "incident.resolved", "on_action": "resolve", "extract": ["incident_id", "resolution", "mttr_seconds"]},
            {"topic": "incident.postmortem", "on_action": "postmortem", "extract": ["incident_id"]},
        ],
    },
    "health_lifecycle": {
        "description": "Emit EventBus events for self-healing lifecycle",
        "source_skill": "self_healing",
        "events": [
            {"topic": "health.scan_complete", "on_action": "scan", "extract": ["total_scanned", "issues_found"]},
            {"topic": "health.repair_applied", "on_action": "heal", "extract": ["skill_id", "strategy", "repair_success"]},
            {"topic": "health.auto_heal_complete", "on_action": "auto_heal", "extract": ["issues_found", "issues_healed"]},
            {"topic": "health.quarantined", "on_action": "quarantine", "extract": ["skill_id", "reason"]},
            {"topic": "health.released", "on_action": "release", "extract": ["skill_id"]},
        ],
    },
    "health_to_incident": {
        "description": "Auto-create incidents when self-healing detects failing subsystems",
        "source_skill": "self_healing",
        "target_skill": "incident_response",
        "trigger_event": "health.scan_complete",
        "condition": "issues_found > 0",
        "reaction": {
            "action": "detect",
            "param_map": {
                "title": "Auto-detected health issue: {issues_found} subsystem(s) degraded/failing",
                "description": "Self-healing scan found {issues_found} issues. Scan data: {scan_summary}",
                "source": "self_healing",
                "severity": "sev3",
            },
        },
    },
    "incident_to_reputation": {
        "description": "Update agent reputation when incidents are resolved",
        "source_skill": "incident_response",
        "target_skill": "agent_reputation",
        "trigger_event": "incident.resolved",
        "reaction": {
            "action": "record_event",
            "param_map": {
                "agent_id": "{assignee}",
                "dimension": "competence",
                "delta": 3,
                "reason": "Resolved incident {incident_id}: {resolution}",
            },
        },
    },
    "escalation_to_reputation": {
        "description": "Track escalation handling in agent reputation",
        "source_skill": "incident_response",
        "target_skill": "agent_reputation",
        "trigger_event": "incident.escalated",
        "reaction": {
            "action": "record_event",
            "param_map": {
                "agent_id": "{target}",
                "dimension": "leadership",
                "delta": 1,
                "reason": "Assigned escalated incident {incident_id}",
            },
        },
    },
}


class SkillEventBridgeSkill(Skill):
    """
    Wires EventBus into existing skills for reactive cross-skill automation.

    Transforms isolated skills into a reactive system where events from one
    skill automatically trigger actions in others, all flowing through the
    central EventBus for observability and replay.
    """

    def __init__(self, credentials: Dict[str, str] = None):
        super().__init__(credentials)
        self._store = None
        self._active_bridges: Dict[str, Dict] = {}

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="skill_event_bridge",
            name="Skill Event Bridge",
            version="1.0.0",
            category="meta",
            description="Wires EventBus into existing skills for reactive cross-skill event automation",
            actions=[
                SkillAction(
                    name="wire",
                    description="Activate event bridges between skills and EventBus",
                    parameters={
                        "bridge_ids": {"type": "list", "required": False, "description": "Specific bridge IDs to activate (omit for all)"},
                        "auto_react": {"type": "boolean", "required": False, "description": "Also set up reactive subscriptions (default: True)"},
                    },
                ),
                SkillAction(
                    name="unwire",
                    description="Deactivate specific event bridges",
                    parameters={
                        "bridge_ids": {"type": "list", "required": True, "description": "Bridge IDs to deactivate"},
                    },
                ),
                SkillAction(
                    name="trigger",
                    description="Manually emit an event through a bridge for testing",
                    parameters={
                        "topic": {"type": "string", "required": True, "description": "Event topic to emit"},
                        "data": {"type": "object", "required": False, "description": "Event data payload"},
                        "source": {"type": "string", "required": False, "description": "Event source identifier"},
                    },
                ),
                SkillAction(
                    name="status",
                    description="View active bridges, recent events, and subscription health",
                    parameters={},
                ),
                SkillAction(
                    name="bridges",
                    description="List all available bridge definitions and their status",
                    parameters={},
                ),
                SkillAction(
                    name="history",
                    description="View recent bridge-triggered events",
                    parameters={
                        "limit": {"type": "number", "required": False, "description": "Max events to return (default: 20)"},
                        "bridge_id": {"type": "string", "required": False, "description": "Filter by bridge ID"},
                    },
                ),
            ],
            required_credentials=[],
        )

    # ── Persistence ───────────────────────────────────────────────

    def _ensure_data(self):
        BRIDGE_FILE.parent.mkdir(parents=True, exist_ok=True)
        if not BRIDGE_FILE.exists():
            self._save(self._default_state())

    def _default_state(self) -> Dict:
        return {
            "active_bridges": {},
            "event_log": [],
            "reactions_executed": [],
            "stats": {
                "total_events_emitted": 0,
                "total_reactions_triggered": 0,
                "total_reaction_failures": 0,
                "events_by_bridge": {},
                "reactions_by_bridge": {},
            },
            "metadata": {
                "created_at": datetime.utcnow().isoformat(),
                "last_wired": None,
            },
        }

    def _load(self) -> Dict:
        if self._store is not None:
            return self._store
        self._ensure_data()
        try:
            with open(BRIDGE_FILE, "r") as f:
                self._store = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            self._store = self._default_state()
        return self._store

    def _save(self, data: Dict):
        self._store = data
        # Trim event log
        if len(data.get("event_log", [])) > MAX_EVENT_LOG:
            data["event_log"] = data["event_log"][-MAX_EVENT_LOG:]
        if len(data.get("reactions_executed", [])) > MAX_EVENT_LOG:
            data["reactions_executed"] = data["reactions_executed"][-MAX_EVENT_LOG:]
        BRIDGE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(BRIDGE_FILE, "w") as f:
            json.dump(data, f, indent=2, default=str)

    # ── Execute Dispatch ──────────────────────────────────────────

    async def execute(self, action: str, params: Dict[str, Any]) -> SkillResult:
        handlers = {
            "wire": self._wire,
            "unwire": self._unwire,
            "trigger": self._trigger,
            "status": self._status,
            "bridges": self._bridges,
            "history": self._history,
        }
        handler = handlers.get(action)
        if not handler:
            return SkillResult(success=False, message=f"Unknown action: {action}")
        return await handler(params)

    # ── Action: wire ──────────────────────────────────────────────

    async def _wire(self, params: Dict) -> SkillResult:
        """Activate event bridges."""
        bridge_ids = params.get("bridge_ids", list(BRIDGE_DEFINITIONS.keys()))
        auto_react = params.get("auto_react", True)

        store = self._load()
        wired = []
        skipped = []
        reactions_set = []

        for bid in bridge_ids:
            if bid not in BRIDGE_DEFINITIONS:
                skipped.append({"bridge_id": bid, "reason": "not found in definitions"})
                continue

            definition = BRIDGE_DEFINITIONS[bid]

            # Check if source skill is available
            source_skill = definition.get("source_skill")
            if source_skill and self.context:
                available = self.context.list_skills()
                if source_skill not in available:
                    skipped.append({"bridge_id": bid, "reason": f"source skill '{source_skill}' not available"})
                    continue

            # Mark bridge as active
            bridge_record = {
                "bridge_id": bid,
                "description": definition["description"],
                "source_skill": definition.get("source_skill"),
                "target_skill": definition.get("target_skill"),
                "activated_at": datetime.utcnow().isoformat(),
                "events_emitted": 0,
                "reactions_triggered": 0,
            }

            store["active_bridges"][bid] = bridge_record
            self._active_bridges[bid] = definition

            # Set up reactive subscriptions if this bridge has reactions
            if auto_react and "reaction" in definition and "trigger_event" in definition:
                target_skill = definition.get("target_skill")
                if target_skill and self.context:
                    available = self.context.list_skills()
                    if target_skill in available:
                        reactions_set.append({
                            "bridge_id": bid,
                            "trigger": definition["trigger_event"],
                            "target": f"{target_skill}.{definition['reaction']['action']}",
                        })

            wired.append(bid)
            store["stats"]["events_by_bridge"][bid] = store["stats"]["events_by_bridge"].get(bid, 0)
            store["stats"]["reactions_by_bridge"][bid] = store["stats"]["reactions_by_bridge"].get(bid, 0)

        store["metadata"]["last_wired"] = datetime.utcnow().isoformat()
        self._save(store)

        return SkillResult(
            success=True,
            message=f"Wired {len(wired)} bridge(s), {len(reactions_set)} reaction(s) set up. Skipped: {len(skipped)}.",
            data={
                "wired": wired,
                "skipped": skipped,
                "reactions_set": reactions_set,
                "total_active": len(store["active_bridges"]),
            },
        )

    # ── Action: unwire ────────────────────────────────────────────

    async def _unwire(self, params: Dict) -> SkillResult:
        """Deactivate event bridges."""
        bridge_ids = params.get("bridge_ids", [])
        if not bridge_ids:
            return SkillResult(success=False, message="bridge_ids required")

        store = self._load()
        unwired = []
        not_found = []

        for bid in bridge_ids:
            if bid in store["active_bridges"]:
                del store["active_bridges"][bid]
                self._active_bridges.pop(bid, None)
                unwired.append(bid)
            else:
                not_found.append(bid)

        self._save(store)

        return SkillResult(
            success=True,
            message=f"Unwired {len(unwired)} bridge(s). Not found: {len(not_found)}.",
            data={
                "unwired": unwired,
                "not_found": not_found,
                "total_active": len(store["active_bridges"]),
            },
        )

    # ── Action: trigger ───────────────────────────────────────────

    async def _trigger(self, params: Dict) -> SkillResult:
        """Manually emit an event and execute any matching reactions."""
        topic = params.get("topic")
        if not topic:
            return SkillResult(success=False, message="topic required")

        event_data = params.get("data", {})
        source = params.get("source", "skill_event_bridge")

        store = self._load()
        now = datetime.utcnow().isoformat()

        # Log the event
        event_record = {
            "topic": topic,
            "data": event_data,
            "source": source,
            "timestamp": now,
            "bridge_id": None,
            "manually_triggered": True,
        }

        # Try to emit via EventSkill if available
        event_emitted = False
        if self.context:
            try:
                result = await self.context.call_skill("event", "publish", {
                    "topic": topic,
                    "data": event_data,
                    "source": source,
                })
                event_emitted = result.success if result else False
            except Exception:
                pass

        store["event_log"].append(event_record)
        store["stats"]["total_events_emitted"] += 1

        # Check for matching reactions
        reactions_executed = []
        for bid, definition in self._active_bridges.items():
            if "trigger_event" not in definition:
                continue
            if definition["trigger_event"] != topic:
                continue

            reaction_result = await self._execute_reaction(bid, definition, event_data, store)
            if reaction_result:
                reactions_executed.append(reaction_result)

        self._save(store)

        return SkillResult(
            success=True,
            message=f"Event '{topic}' emitted. EventBus: {event_emitted}. {len(reactions_executed)} reaction(s) triggered.",
            data={
                "topic": topic,
                "event_bus_emitted": event_emitted,
                "reactions_executed": reactions_executed,
                "data": event_data,
            },
        )

    # ── Action: status ────────────────────────────────────────────

    async def _status(self, params: Dict) -> SkillResult:
        """View active bridges and stats."""
        store = self._load()

        active = store.get("active_bridges", {})
        stats = store.get("stats", {})
        recent_events = store.get("event_log", [])[-5:]
        recent_reactions = store.get("reactions_executed", [])[-5:]

        return SkillResult(
            success=True,
            message=f"{len(active)} active bridge(s). {stats.get('total_events_emitted', 0)} events emitted, "
                    f"{stats.get('total_reactions_triggered', 0)} reactions triggered.",
            data={
                "active_bridges": active,
                "stats": stats,
                "recent_events": recent_events,
                "recent_reactions": recent_reactions,
                "available_definitions": len(BRIDGE_DEFINITIONS),
            },
        )

    # ── Action: bridges ───────────────────────────────────────────

    async def _bridges(self, params: Dict) -> SkillResult:
        """List all available bridge definitions."""
        store = self._load()
        active_ids = set(store.get("active_bridges", {}).keys())

        definitions = []
        for bid, defn in BRIDGE_DEFINITIONS.items():
            definitions.append({
                "bridge_id": bid,
                "description": defn["description"],
                "source_skill": defn.get("source_skill"),
                "target_skill": defn.get("target_skill"),
                "active": bid in active_ids,
                "event_count": len(defn.get("events", [])),
                "has_reaction": "reaction" in defn,
                "trigger_event": defn.get("trigger_event"),
            })

        return SkillResult(
            success=True,
            message=f"{len(definitions)} bridge definitions available. {len(active_ids)} active.",
            data={
                "definitions": definitions,
                "active_count": len(active_ids),
                "total_count": len(definitions),
            },
        )

    # ── Action: history ───────────────────────────────────────────

    async def _history(self, params: Dict) -> SkillResult:
        """View recent bridge-triggered events."""
        limit = params.get("limit", 20)
        bridge_filter = params.get("bridge_id")

        store = self._load()
        events = store.get("event_log", [])
        reactions = store.get("reactions_executed", [])

        if bridge_filter:
            events = [e for e in events if e.get("bridge_id") == bridge_filter]
            reactions = [r for r in reactions if r.get("bridge_id") == bridge_filter]

        recent_events = events[-limit:]
        recent_reactions = reactions[-limit:]

        return SkillResult(
            success=True,
            message=f"{len(recent_events)} recent event(s), {len(recent_reactions)} recent reaction(s).",
            data={
                "events": recent_events,
                "reactions": recent_reactions,
            },
        )

    # ── Core: Emit bridge events after skill execution ────────────

    async def emit_bridge_events(self, skill_id: str, action: str, result_data: Dict) -> List[Dict]:
        """
        Called after a skill action to emit any bridged events.

        This is the main integration point - the agent or skill execution
        layer calls this after each skill action to check if any bridges
        should emit events.

        Returns list of emitted events.
        """
        store = self._load()
        emitted = []

        for bid, definition in self._active_bridges.items():
            source_skill = definition.get("source_skill")
            if source_skill != skill_id:
                continue

            # Check event definitions for this action
            for event_def in definition.get("events", []):
                if event_def.get("on_action") != action:
                    continue

                # Extract specified fields from result data
                event_data = {}
                for field in event_def.get("extract", []):
                    if field in result_data:
                        event_data[field] = result_data[field]

                event_data["_bridge_id"] = bid
                event_data["_source_action"] = f"{skill_id}.{action}"

                topic = event_def["topic"]

                # Log event
                event_record = {
                    "topic": topic,
                    "data": event_data,
                    "source": skill_id,
                    "timestamp": datetime.utcnow().isoformat(),
                    "bridge_id": bid,
                    "manually_triggered": False,
                }
                store["event_log"].append(event_record)
                store["stats"]["total_events_emitted"] += 1
                store["stats"]["events_by_bridge"][bid] = store["stats"]["events_by_bridge"].get(bid, 0) + 1

                # Update bridge stats
                if bid in store["active_bridges"]:
                    store["active_bridges"][bid]["events_emitted"] = (
                        store["active_bridges"][bid].get("events_emitted", 0) + 1
                    )

                # Emit to EventBus via EventSkill
                if self.context:
                    try:
                        await self.context.call_skill("event", "publish", {
                            "topic": topic,
                            "data": event_data,
                            "source": skill_id,
                        })
                    except Exception:
                        pass

                emitted.append({"topic": topic, "data": event_data, "bridge_id": bid})

                # Check if any reactive bridges match this event
                for react_bid, react_def in self._active_bridges.items():
                    if "trigger_event" not in react_def:
                        continue
                    if react_def["trigger_event"] == topic:
                        await self._execute_reaction(react_bid, react_def, event_data, store)

        self._save(store)
        return emitted

    async def _execute_reaction(self, bridge_id: str, definition: Dict, event_data: Dict, store: Dict) -> Optional[Dict]:
        """Execute a reactive bridge action."""
        reaction = definition.get("reaction")
        if not reaction:
            return None

        target_skill = definition.get("target_skill")
        if not target_skill or not self.context:
            return None

        # Check condition if present
        condition = definition.get("condition")
        if condition:
            if not self._evaluate_condition(condition, event_data):
                return None

        # Build parameters from param_map
        action = reaction["action"]
        params = {}
        for key, template in reaction.get("param_map", {}).items():
            if isinstance(template, str) and "{" in template:
                # Template string - substitute event data
                try:
                    params[key] = template.format(**event_data)
                except (KeyError, ValueError):
                    params[key] = template
            else:
                params[key] = template

        # Execute the reaction
        reaction_record = {
            "bridge_id": bridge_id,
            "target_skill": target_skill,
            "action": action,
            "params": params,
            "trigger_data": event_data,
            "timestamp": datetime.utcnow().isoformat(),
            "success": False,
            "error": None,
        }

        try:
            result = await self.context.call_skill(target_skill, action, params)
            reaction_record["success"] = result.success if result else False
            reaction_record["result_message"] = result.message if result else "No result"
        except Exception as e:
            reaction_record["error"] = str(e)

        store["reactions_executed"].append(reaction_record)
        store["stats"]["total_reactions_triggered"] += 1
        if not reaction_record["success"]:
            store["stats"]["total_reaction_failures"] += 1
        store["stats"]["reactions_by_bridge"][bridge_id] = (
            store["stats"]["reactions_by_bridge"].get(bridge_id, 0) + 1
        )

        # Update bridge stats
        if bridge_id in store["active_bridges"]:
            store["active_bridges"][bridge_id]["reactions_triggered"] = (
                store["active_bridges"][bridge_id].get("reactions_triggered", 0) + 1
            )

        return reaction_record

    def _evaluate_condition(self, condition: str, data: Dict) -> bool:
        """
        Evaluate a simple condition against event data.

        Supports: "field > N", "field < N", "field == value", "field != value"
        """
        try:
            parts = condition.split()
            if len(parts) != 3:
                return True  # Can't parse, allow through

            field, op, value = parts

            actual = data.get(field)
            if actual is None:
                return False

            # Try numeric comparison
            try:
                actual_num = float(actual)
                value_num = float(value)
                if op == ">":
                    return actual_num > value_num
                elif op == "<":
                    return actual_num < value_num
                elif op == ">=":
                    return actual_num >= value_num
                elif op == "<=":
                    return actual_num <= value_num
                elif op == "==":
                    return actual_num == value_num
                elif op == "!=":
                    return actual_num != value_num
            except (ValueError, TypeError):
                pass

            # String comparison
            if op == "==":
                return str(actual) == value
            elif op == "!=":
                return str(actual) != value

            return True
        except Exception:
            return True  # On error, allow through
