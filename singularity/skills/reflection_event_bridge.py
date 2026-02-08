#!/usr/bin/env python3
"""
ReflectionEventBridgeSkill - Wires AgentReflection into EventBus for reactive self-improvement.

The agent has two powerful systems that should work together:
1. **EventBus** - Emits events like action.failed, cycle.complete, etc.
2. **AgentReflection** - Reflects on actions, builds playbooks, extracts patterns

This bridge connects them:

LISTEN side (EventBus → Reflection):
- action.failed → Auto-reflect on what went wrong
- action.success → Optionally reflect on successes (configurable)
- cycle.complete → Trigger pattern extraction periodically

EMIT side (Reflection → EventBus):
- reflection.created → New reflection added (other skills can react)
- playbook.created → New playbook generated
- playbook.used → Playbook was applied to a task
- insight.added → New strategic insight recorded
- pattern.extracted → Patterns identified from reflection history

This creates a continuous self-improvement loop:
  action fails → auto-reflect → pattern emerges → playbook built → event emitted → future tasks use playbook

Pillar: Self-Improvement (closes the reactive self-improvement feedback loop)
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from .base import Skill, SkillResult, SkillManifest, SkillAction

BRIDGE_STATE_FILE = Path(__file__).parent.parent / "data" / "reflection_event_bridge.json"
MAX_AUTO_REFLECTIONS = 200
MAX_EVENT_LOG = 300


class ReflectionEventBridgeSkill(Skill):
    """
    Bridge between EventBus and AgentReflection for reactive self-improvement.

    When wired:
    - Subscribes to action.failed events and auto-triggers reflection
    - Emits events when reflections/playbooks/insights are created
    - Tracks auto-reflection stats and bridge health
    """

    def __init__(self, credentials: Dict[str, str] = None):
        super().__init__(credentials)
        self._load_state()

    def _load_state(self):
        """Load or initialize bridge state."""
        BRIDGE_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        if BRIDGE_STATE_FILE.exists():
            try:
                with open(BRIDGE_STATE_FILE) as f:
                    data = json.load(f)
                self._active_subscriptions = data.get("active_subscriptions", {})
                self._auto_reflections = data.get("auto_reflections", [])
                self._emitted_events = data.get("emitted_events", [])
                self._config = data.get("config", self._default_config())
                self._stats = data.get("stats", self._default_stats())
            except (json.JSONDecodeError, Exception):
                self._init_empty()
        else:
            self._init_empty()

    def _init_empty(self):
        self._active_subscriptions: Dict[str, Dict] = {}
        self._auto_reflections: List[Dict] = []
        self._emitted_events: List[Dict] = []
        self._config = self._default_config()
        self._stats = self._default_stats()

    def _default_config(self) -> Dict:
        return {
            "reflect_on_failures": True,
            "reflect_on_successes": False,
            "extract_patterns_every_n": 10,  # Extract patterns every N reflections
            "emit_on_reflect": True,
            "emit_on_playbook": True,
            "emit_on_insight": True,
            "emit_on_pattern": True,
            "auto_find_playbook_on_cycle": True,  # Find relevant playbook at cycle start
        }

    def _default_stats(self) -> Dict:
        return {
            "total_auto_reflections": 0,
            "failure_reflections": 0,
            "success_reflections": 0,
            "events_emitted": 0,
            "patterns_extracted": 0,
            "playbooks_suggested": 0,
            "bridge_wired_at": None,
            "last_reflection_at": None,
            "last_event_at": None,
        }

    def _save_state(self):
        """Persist bridge state."""
        data = {
            "active_subscriptions": self._active_subscriptions,
            "auto_reflections": self._auto_reflections[-MAX_AUTO_REFLECTIONS:],
            "emitted_events": self._emitted_events[-MAX_EVENT_LOG:],
            "config": self._config,
            "stats": self._stats,
        }
        BRIDGE_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(BRIDGE_STATE_FILE, "w") as f:
            json.dump(data, f, indent=2, default=str)

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="reflection_event_bridge",
            name="Reflection Event Bridge",
            version="1.0.0",
            category="self_improvement",
            description=(
                "Bridges AgentReflection and EventBus for reactive self-improvement. "
                "Auto-reflects on failures, emits events on new insights/playbooks, "
                "and creates a continuous improvement feedback loop."
            ),
            actions=self.get_actions(),
            required_credentials=[],
        )

    def get_actions(self) -> List[SkillAction]:
        return [
            SkillAction(
                name="wire",
                description=(
                    "Activate the bridge: subscribe to action events and start "
                    "auto-reflecting on failures. Optionally configure which events trigger reflection."
                ),
                parameters={
                    "reflect_on_failures": {
                        "type": "boolean", "required": False,
                        "description": "Auto-reflect on action.failed events (default: True)",
                    },
                    "reflect_on_successes": {
                        "type": "boolean", "required": False,
                        "description": "Also reflect on action.success events (default: False)",
                    },
                    "extract_patterns_every_n": {
                        "type": "integer", "required": False,
                        "description": "Extract patterns every N auto-reflections (default: 10)",
                    },
                },
                estimated_cost=0.0,
            ),
            SkillAction(
                name="unwire",
                description="Deactivate the bridge: stop auto-reflecting on events.",
                parameters={},
                estimated_cost=0.0,
            ),
            SkillAction(
                name="configure",
                description="Update bridge configuration without rewiring.",
                parameters={
                    "reflect_on_failures": {"type": "boolean", "required": False},
                    "reflect_on_successes": {"type": "boolean", "required": False,},
                    "extract_patterns_every_n": {"type": "integer", "required": False},
                    "emit_on_reflect": {"type": "boolean", "required": False},
                    "emit_on_playbook": {"type": "boolean", "required": False},
                    "emit_on_insight": {"type": "boolean", "required": False},
                    "auto_find_playbook_on_cycle": {"type": "boolean", "required": False},
                },
                estimated_cost=0.0,
            ),
            SkillAction(
                name="emit",
                description=(
                    "Manually emit a reflection-related event through the EventBus. "
                    "Useful for notifying other skills about reflection outcomes."
                ),
                parameters={
                    "event_type": {
                        "type": "string", "required": True,
                        "description": "Event type: reflection.created, playbook.created, insight.added, pattern.extracted",
                    },
                    "data": {
                        "type": "object", "required": False,
                        "description": "Event payload data",
                    },
                },
                estimated_cost=0.0,
            ),
            SkillAction(
                name="auto_reflect",
                description=(
                    "Trigger an automatic reflection based on event data. "
                    "Normally called internally when action.failed fires, "
                    "but can be triggered manually for testing."
                ),
                parameters={
                    "event_type": {
                        "type": "string", "required": True,
                        "description": "Type of event that triggered reflection (e.g., action.failed)",
                    },
                    "skill_id": {
                        "type": "string", "required": False,
                        "description": "Which skill's action failed/succeeded",
                    },
                    "action": {
                        "type": "string", "required": False,
                        "description": "Which action failed/succeeded",
                    },
                    "error": {
                        "type": "string", "required": False,
                        "description": "Error message if failure",
                    },
                    "outcome": {
                        "type": "string", "required": False,
                        "description": "Outcome description",
                    },
                    "tags": {
                        "type": "array", "required": False,
                        "description": "Tags for categorizing the reflection",
                    },
                },
                estimated_cost=0.0,
            ),
            SkillAction(
                name="status",
                description="View bridge status: active subscriptions, auto-reflection stats, config.",
                parameters={},
                estimated_cost=0.0,
            ),
            SkillAction(
                name="history",
                description="View recent auto-reflections and emitted events.",
                parameters={
                    "type": {
                        "type": "string", "required": False,
                        "description": "Filter: 'reflections', 'events', or 'all' (default: all)",
                    },
                    "limit": {
                        "type": "integer", "required": False,
                        "description": "Max items to return (default: 20)",
                    },
                },
                estimated_cost=0.0,
            ),
            SkillAction(
                name="suggest_playbook",
                description=(
                    "Given a task description, find the best matching playbook "
                    "and emit a playbook.suggested event for other skills to consume."
                ),
                parameters={
                    "task": {
                        "type": "string", "required": True,
                        "description": "Task description to find a playbook for",
                    },
                    "tags": {
                        "type": "array", "required": False,
                        "description": "Tags to help match playbooks",
                    },
                },
                estimated_cost=0.0,
            ),
        ]

    async def execute(self, action: str, params: Dict) -> SkillResult:
        actions = {
            "wire": self._wire,
            "unwire": self._unwire,
            "configure": self._configure,
            "emit": self._emit,
            "auto_reflect": self._auto_reflect,
            "status": self._status,
            "history": self._history,
            "suggest_playbook": self._suggest_playbook,
        }
        handler = actions.get(action)
        if not handler:
            return SkillResult(
                success=False,
                message=f"Unknown action: {action}. Available: {list(actions.keys())}",
            )
        return await handler(params)

    async def _wire(self, params: Dict) -> SkillResult:
        """Activate bridge: subscribe to events and start auto-reflecting."""
        # Update config from params
        if "reflect_on_failures" in params:
            self._config["reflect_on_failures"] = params["reflect_on_failures"]
        if "reflect_on_successes" in params:
            self._config["reflect_on_successes"] = params["reflect_on_successes"]
        if "extract_patterns_every_n" in params:
            self._config["extract_patterns_every_n"] = max(1, int(params["extract_patterns_every_n"]))

        subscriptions_created = []

        # Subscribe to action.failed
        if self._config["reflect_on_failures"]:
            sub_id = f"reb_failure_{int(time.time())}"
            self._active_subscriptions[sub_id] = {
                "pattern": "action.failed",
                "type": "failure_reflection",
                "created_at": datetime.utcnow().isoformat(),
            }
            subscriptions_created.append({"id": sub_id, "pattern": "action.failed"})

        # Subscribe to action.success
        if self._config["reflect_on_successes"]:
            sub_id = f"reb_success_{int(time.time())}"
            self._active_subscriptions[sub_id] = {
                "pattern": "action.success",
                "type": "success_reflection",
                "created_at": datetime.utcnow().isoformat(),
            }
            subscriptions_created.append({"id": sub_id, "pattern": "action.success"})

        # Subscribe to cycle.complete for periodic pattern extraction
        sub_id = f"reb_cycle_{int(time.time())}"
        self._active_subscriptions[sub_id] = {
            "pattern": "cycle.complete",
            "type": "pattern_extraction",
            "created_at": datetime.utcnow().isoformat(),
        }
        subscriptions_created.append({"id": sub_id, "pattern": "cycle.complete"})

        self._stats["bridge_wired_at"] = datetime.utcnow().isoformat()
        self._save_state()

        return SkillResult(
            success=True,
            message=f"Bridge wired with {len(subscriptions_created)} subscriptions",
            data={
                "subscriptions": subscriptions_created,
                "config": self._config,
                "total_active": len(self._active_subscriptions),
            },
        )

    async def _unwire(self, params: Dict) -> SkillResult:
        """Deactivate bridge: remove all subscriptions."""
        count = len(self._active_subscriptions)
        self._active_subscriptions.clear()
        self._save_state()

        return SkillResult(
            success=True,
            message=f"Bridge unwired: removed {count} subscriptions",
            data={"removed_count": count, "total_active": 0},
        )

    async def _configure(self, params: Dict) -> SkillResult:
        """Update bridge configuration."""
        updated = []
        for key in self._config:
            if key in params:
                old_val = self._config[key]
                self._config[key] = params[key]
                updated.append({"key": key, "old": old_val, "new": params[key]})

        self._save_state()
        return SkillResult(
            success=True,
            message=f"Updated {len(updated)} config values",
            data={"updated": updated, "config": self._config},
        )

    async def _emit(self, params: Dict) -> SkillResult:
        """Manually emit a reflection event via EventBus."""
        event_type = params.get("event_type", "")
        valid_types = [
            "reflection.created", "playbook.created", "playbook.used",
            "insight.added", "pattern.extracted", "playbook.suggested",
        ]
        if event_type not in valid_types:
            return SkillResult(
                success=False,
                message=f"Invalid event_type: {event_type}. Valid: {valid_types}",
            )

        event_data = params.get("data", {})
        record = self._record_emitted_event(event_type, event_data)

        # Try to emit via EventBus if context available
        emitted_to_bus = False
        if hasattr(self, "context") and self.context:
            try:
                result = await self.context.call_skill("event", "publish", {
                    "topic": event_type,
                    "data": event_data,
                    "source": "reflection_event_bridge",
                })
                emitted_to_bus = result.success if result else False
            except Exception:
                pass

        return SkillResult(
            success=True,
            message=f"Emitted {event_type}" + (" (via EventBus)" if emitted_to_bus else " (local only)"),
            data={"event": record, "emitted_to_bus": emitted_to_bus},
        )

    async def _auto_reflect(self, params: Dict) -> SkillResult:
        """Auto-reflect on an action event (failure or success)."""
        event_type = params.get("event_type", "action.failed")
        skill_id = params.get("skill_id", "unknown")
        action_name = params.get("action", "unknown")
        error = params.get("error", "")
        outcome = params.get("outcome", "")
        tags = params.get("tags", [])

        is_failure = "fail" in event_type.lower()

        # Build reflection params for AgentReflection
        if is_failure:
            task_desc = f"[AUTO] Failed action: {skill_id}.{action_name}"
            reflection_outcome = f"FAILURE: {error}" if error else "Action failed"
            what_happened = f"The {skill_id} skill's '{action_name}' action failed"
            if error:
                what_happened += f" with error: {error}"
            analysis = f"Auto-triggered reflection on failure of {skill_id}.{action_name}. Need to understand root cause and prevent recurrence."
            reflection_tags = ["auto_reflection", "failure", skill_id] + tags
        else:
            task_desc = f"[AUTO] Successful action: {skill_id}.{action_name}"
            reflection_outcome = outcome or "Action succeeded"
            what_happened = f"The {skill_id} skill's '{action_name}' action succeeded"
            analysis = f"Auto-triggered reflection on success of {skill_id}.{action_name}. Capture what went well for reuse."
            reflection_tags = ["auto_reflection", "success", skill_id] + tags

        # Call AgentReflection.reflect
        reflection_result = None
        if hasattr(self, "context") and self.context:
            try:
                reflection_result = await self.context.call_skill("agent_reflection", "reflect", {
                    "task": task_desc,
                    "actions_taken": what_happened,
                    "outcome": reflection_outcome,
                    "analysis": analysis,
                    "tags": reflection_tags,
                })
            except Exception as e:
                reflection_result = None

        # Record the auto-reflection
        record = {
            "event_type": event_type,
            "skill_id": skill_id,
            "action": action_name,
            "is_failure": is_failure,
            "error": error,
            "outcome": outcome,
            "reflection_success": reflection_result.success if reflection_result else False,
            "reflection_id": (
                reflection_result.data.get("reflection_id") if reflection_result and reflection_result.data else None
            ),
            "timestamp": datetime.utcnow().isoformat(),
        }
        self._auto_reflections.append(record)

        # Update stats
        self._stats["total_auto_reflections"] += 1
        if is_failure:
            self._stats["failure_reflections"] += 1
        else:
            self._stats["success_reflections"] += 1
        self._stats["last_reflection_at"] = datetime.utcnow().isoformat()

        # Emit reflection.created event
        if self._config.get("emit_on_reflect", True):
            await self._emit_event("reflection.created", {
                "auto": True,
                "event_type": event_type,
                "skill_id": skill_id,
                "action": action_name,
                "is_failure": is_failure,
                "reflection_id": record["reflection_id"],
            })

        # Check if we should extract patterns
        n = self._config.get("extract_patterns_every_n", 10)
        if n > 0 and self._stats["total_auto_reflections"] % n == 0:
            await self._trigger_pattern_extraction()

        self._save_state()

        return SkillResult(
            success=True,
            message=f"Auto-reflected on {event_type} for {skill_id}.{action_name}",
            data={
                "reflection": record,
                "stats": {
                    "total": self._stats["total_auto_reflections"],
                    "failures": self._stats["failure_reflections"],
                    "successes": self._stats["success_reflections"],
                },
            },
        )

    async def _status(self, params: Dict) -> SkillResult:
        """View bridge status."""
        is_active = len(self._active_subscriptions) > 0
        return SkillResult(
            success=True,
            message=f"Bridge {'ACTIVE' if is_active else 'INACTIVE'} - "
                    f"{self._stats['total_auto_reflections']} auto-reflections, "
                    f"{self._stats['events_emitted']} events emitted",
            data={
                "active": is_active,
                "subscriptions": self._active_subscriptions,
                "config": self._config,
                "stats": self._stats,
                "recent_reflections": self._auto_reflections[-5:],
                "recent_events": self._emitted_events[-5:],
            },
        )

    async def _history(self, params: Dict) -> SkillResult:
        """View recent auto-reflections and emitted events."""
        filter_type = params.get("type", "all")
        limit = min(params.get("limit", 20), 100)

        result_data = {}

        if filter_type in ("all", "reflections"):
            result_data["reflections"] = self._auto_reflections[-limit:]
            result_data["total_reflections"] = len(self._auto_reflections)

        if filter_type in ("all", "events"):
            result_data["events"] = self._emitted_events[-limit:]
            result_data["total_events"] = len(self._emitted_events)

        return SkillResult(
            success=True,
            message=f"History ({filter_type}): {len(result_data.get('reflections', []))} reflections, "
                    f"{len(result_data.get('events', []))} events",
            data=result_data,
        )

    async def _suggest_playbook(self, params: Dict) -> SkillResult:
        """Find best playbook for a task and emit suggestion event."""
        task = params.get("task", "")
        tags = params.get("tags", [])

        if not task:
            return SkillResult(success=False, message="Task description required")

        # Call AgentReflection.find_playbook
        playbook_result = None
        if hasattr(self, "context") and self.context:
            try:
                playbook_result = await self.context.call_skill(
                    "agent_reflection", "find_playbook",
                    {"task": task, "tags": tags},
                )
            except Exception:
                playbook_result = None

        if playbook_result and playbook_result.success and playbook_result.data:
            matches = playbook_result.data.get("matches", [])
            if matches:
                best = matches[0]
                self._stats["playbooks_suggested"] += 1

                # Emit playbook.suggested event
                await self._emit_event("playbook.suggested", {
                    "task": task,
                    "playbook_id": best.get("playbook_id", ""),
                    "playbook_name": best.get("name", ""),
                    "relevance_score": best.get("relevance_score", 0),
                    "effectiveness": best.get("effectiveness", 0),
                })

                self._save_state()
                return SkillResult(
                    success=True,
                    message=f"Found playbook: {best.get('name', 'unnamed')} "
                            f"(relevance: {best.get('relevance_score', 0):.1f})",
                    data={
                        "best_match": best,
                        "total_matches": len(matches),
                        "event_emitted": "playbook.suggested",
                    },
                )

        return SkillResult(
            success=True,
            message="No matching playbook found for this task",
            data={"matches": [], "task": task},
        )

    # --- Internal helpers ---

    async def _emit_event(self, topic: str, data: Dict):
        """Emit an event via EventBus and record it locally."""
        record = self._record_emitted_event(topic, data)

        if hasattr(self, "context") and self.context:
            try:
                await self.context.call_skill("event", "publish", {
                    "topic": topic,
                    "data": data,
                    "source": "reflection_event_bridge",
                })
            except Exception:
                pass

        return record

    def _record_emitted_event(self, topic: str, data: Dict) -> Dict:
        """Record an emitted event locally."""
        record = {
            "topic": topic,
            "data": data,
            "timestamp": datetime.utcnow().isoformat(),
        }
        self._emitted_events.append(record)
        self._stats["events_emitted"] += 1
        self._stats["last_event_at"] = datetime.utcnow().isoformat()
        return record

    async def _trigger_pattern_extraction(self):
        """Trigger pattern extraction via AgentReflection."""
        if not (hasattr(self, "context") and self.context):
            return

        try:
            result = await self.context.call_skill(
                "agent_reflection", "extract_patterns", {}
            )
            if result and result.success:
                self._stats["patterns_extracted"] += 1
                if self._config.get("emit_on_pattern", True):
                    await self._emit_event("pattern.extracted", {
                        "auto": True,
                        "trigger": "periodic",
                        "reflection_count": self._stats["total_auto_reflections"],
                    })
        except Exception:
            pass

    def estimate_cost(self, action: str, params: Dict) -> float:
        return 0.0
