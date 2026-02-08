#!/usr/bin/env python3
"""
FunctionMarketplaceDiscoveryEventsSkill - Emit events when marketplace state changes.

The FunctionMarketplaceSkill manages listings, imports, and ratings but declares
event topics it never actually emits.  The EventBus handles publish/subscribe
but has no knowledge of marketplace state transitions.  This bridge skill closes
the gap by periodically (or on-demand) snapshotting marketplace state, detecting
changes, and emitting well-typed events through the EventBus.

Reactive behaviours this enables:
  - Auto-import trending functions when import acceleration exceeds a threshold
  - Notify agents when a new function is published in a watched category
  - Trigger reputation updates when ratings change
  - Alert on price changes for functions an agent has imported
  - Build leaderboards of publishers with highest import velocity

Change-detection strategy:
  1. Take a snapshot of every listing's (import_count, rating_count, avg_rating,
     price_per_import, status) plus the set of listing_ids.
  2. On each check, re-read marketplace state via the ``function_marketplace``
     skill and diff against the snapshot.
  3. New listing_id  ➜  ``marketplace.new_listing``
  4. import_count up  ➜  ``marketplace.new_import``
  5. rating_count up  ➜  ``marketplace.new_rating``
  6. price changed    ➜  ``marketplace.price_change``
  7. Import acceleration above threshold  ➜  ``marketplace.trending``
  8. Any detected event matched by a watch rule  ➜  ``marketplace.watch.matched``

Events emitted:
  marketplace.new_listing   {listing_id, function_name, agent_name, category, tags, price}
  marketplace.new_import    {listing_id, function_name, importer_agent, publisher_agent}
  marketplace.new_rating    {listing_id, function_name, agent_name, rating, avg_rating}
  marketplace.trending      {listing_id, function_name, import_count, acceleration}
  marketplace.price_change  {listing_id, function_name, old_price, new_price}
  marketplace.watch.matched {rule_name, event_type, listing_id, details}

Pillar: Replication — agents reactively discover and share capabilities.
"""

import json
import hashlib
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import Skill, SkillAction, SkillManifest, SkillResult

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent.parent / "data"
STATE_FILE = DATA_DIR / "function_marketplace_discovery_events.json"

MAX_EVENT_LOG = 500
MAX_WATCH_RULES = 50

VALID_EVENT_TYPES = {
    "new_listing",
    "new_import",
    "rating_change",
    "trending",
}


def _now_iso() -> str:
    """Return current UTC timestamp in ISO-8601 format."""
    return datetime.utcnow().isoformat() + "Z"


# ---------------------------------------------------------------------------
# Skill implementation
# ---------------------------------------------------------------------------


class FunctionMarketplaceDiscoveryEventsSkill(Skill):
    """
    Bridge between FunctionMarketplaceSkill and EventBus.

    Monitors marketplace state for changes and emits events so that other
    skills can react to new publications, imports, rating changes, price
    adjustments, and trending functions.
    """

    def __init__(self, credentials: Dict[str, str] = None):
        """Initialise the skill and load persisted state from disk."""
        super().__init__(credentials)
        self._load_state()

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def _default_state(self) -> Dict[str, Any]:
        """Return the initial empty state structure."""
        return {
            "wire_state": {
                "active": False,
                "wired_at": None,
                "cycles_since_check": 0,
            },
            "snapshot": {
                "listings": {},
                "imports_count": 0,
                "taken_at": None,
            },
            "watch_rules": {},
            "event_log": [],
            "config": {
                "poll_interval_reflections": 5,
                "watch_categories": [],
                "watch_agents": [],
                "emit_on_new_listing": True,
                "emit_on_import": True,
                "emit_on_rating": True,
                "emit_on_trending": True,
                "trending_threshold": 2,
            },
            "stats": {
                "checks_performed": 0,
                "events_emitted": 0,
                "new_listings_detected": 0,
                "new_imports_detected": 0,
                "new_ratings_detected": 0,
                "trending_detected": 0,
                "watch_matches": 0,
                "last_check_at": None,
            },
        }

    def _load_state(self) -> None:
        """Load persisted state from disk, falling back to defaults."""
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        if STATE_FILE.exists():
            try:
                with open(STATE_FILE) as fh:
                    data = json.load(fh)
                self._wire_state = data.get("wire_state", self._default_state()["wire_state"])
                self._snapshot = data.get("snapshot", self._default_state()["snapshot"])
                self._watch_rules: Dict[str, Dict] = data.get("watch_rules", {})
                self._event_log: List[Dict] = data.get("event_log", [])
                self._config: Dict[str, Any] = {
                    **self._default_state()["config"],
                    **data.get("config", {}),
                }
                self._stats: Dict[str, Any] = {
                    **self._default_state()["stats"],
                    **data.get("stats", {}),
                }
            except (json.JSONDecodeError, KeyError, TypeError):
                self._init_empty()
        else:
            self._init_empty()

    def _init_empty(self) -> None:
        """Initialise all in-memory state to defaults."""
        defaults = self._default_state()
        self._wire_state: Dict[str, Any] = defaults["wire_state"]
        self._snapshot: Dict[str, Any] = defaults["snapshot"]
        self._watch_rules: Dict[str, Dict] = {}
        self._event_log: List[Dict] = []
        self._config: Dict[str, Any] = defaults["config"]
        self._stats: Dict[str, Any] = defaults["stats"]

    def _save_state(self) -> None:
        """Persist current state to disk, trimming the event log."""
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "wire_state": self._wire_state,
            "snapshot": self._snapshot,
            "watch_rules": self._watch_rules,
            "event_log": self._event_log[-MAX_EVENT_LOG:],
            "config": self._config,
            "stats": self._stats,
        }
        with open(STATE_FILE, "w") as fh:
            json.dump(data, fh, indent=2, default=str)

    # ------------------------------------------------------------------
    # Manifest
    # ------------------------------------------------------------------

    @property
    def manifest(self) -> SkillManifest:
        """Return the skill manifest describing identity, actions, and metadata."""
        return SkillManifest(
            skill_id="function_marketplace_discovery_events",
            name="Function Marketplace Discovery Events",
            version="1.0.0",
            category="replication",
            description=(
                "Monitors FunctionMarketplaceSkill for state changes and emits "
                "events (new listings, imports, ratings, trending, price changes) "
                "via EventBus to enable reactive marketplace behaviour."
            ),
            actions=self._build_actions(),
            required_credentials=[],
            author="singularity",
        )

    def _build_actions(self) -> List[SkillAction]:
        """Construct the list of SkillAction descriptors."""
        return [
            # 1. wire
            SkillAction(
                name="wire",
                description=(
                    "Activate discovery event monitoring.  Takes a snapshot of "
                    "current marketplace state for subsequent change detection."
                ),
                parameters={
                    "poll_interval_reflections": {
                        "type": "int",
                        "required": False,
                        "description": (
                            "Check marketplace every N agent cycles (default 5)."
                        ),
                    },
                    "watch_categories": {
                        "type": "list",
                        "required": False,
                        "description": (
                            "Only emit events for these categories (empty = all)."
                        ),
                    },
                    "watch_agents": {
                        "type": "list",
                        "required": False,
                        "description": (
                            "Only emit events from these publisher agents."
                        ),
                    },
                },
                estimated_cost=0.0,
            ),
            # 2. unwire
            SkillAction(
                name="unwire",
                description="Deactivate monitoring and clear subscriptions.",
                parameters={},
                estimated_cost=0.0,
            ),
            # 3. check
            SkillAction(
                name="check",
                description=(
                    "Manually check the marketplace for new activity since the "
                    "last snapshot.  Detects new listings, imports, ratings, "
                    "price changes, and status changes; emits appropriate events."
                ),
                parameters={},
                estimated_cost=0.0,
            ),
            # 4. watch
            SkillAction(
                name="watch",
                description=(
                    "Add a persistent watch rule.  When a matching marketplace "
                    "event is detected a marketplace.watch.matched event is also "
                    "emitted with the rule details."
                ),
                parameters={
                    "rule_name": {
                        "type": "str",
                        "required": True,
                        "description": "Unique name for this watch rule.",
                    },
                    "event_type": {
                        "type": "str",
                        "required": True,
                        "description": (
                            "Event type to match: new_listing, new_import, "
                            "rating_change, or trending."
                        ),
                    },
                    "category": {
                        "type": "str",
                        "required": False,
                        "description": "Only match listings in this category.",
                    },
                    "min_rating": {
                        "type": "float",
                        "required": False,
                        "description": "Only match listings with avg_rating >= this value.",
                    },
                    "agent_name": {
                        "type": "str",
                        "required": False,
                        "description": "Only match listings from this publisher.",
                    },
                    "tags": {
                        "type": "list",
                        "required": False,
                        "description": "Only match listings with at least one of these tags.",
                    },
                },
                estimated_cost=0.0,
            ),
            # 5. unwatch
            SkillAction(
                name="unwatch",
                description="Remove a watch rule by name.",
                parameters={
                    "rule_name": {
                        "type": "str",
                        "required": True,
                        "description": "Name of the watch rule to remove.",
                    },
                },
                estimated_cost=0.0,
            ),
            # 6. trending
            SkillAction(
                name="trending",
                description=(
                    "Analyse the marketplace for functions with accelerating "
                    "import counts.  Emits marketplace.trending events for each "
                    "trending function found."
                ),
                parameters={
                    "window": {
                        "type": "int",
                        "required": False,
                        "description": (
                            "Number of recent imports to consider (default 10)."
                        ),
                    },
                    "min_imports": {
                        "type": "int",
                        "required": False,
                        "description": (
                            "Minimum import count to be considered (default 2)."
                        ),
                    },
                },
                estimated_cost=0.0,
            ),
            # 7. configure
            SkillAction(
                name="configure",
                description="Update monitoring configuration without rewiring.",
                parameters={
                    "poll_interval_reflections": {
                        "type": "int",
                        "required": False,
                        "description": "Check marketplace every N agent cycles.",
                    },
                    "watch_categories": {
                        "type": "list",
                        "required": False,
                        "description": "Only emit events for these categories.",
                    },
                    "watch_agents": {
                        "type": "list",
                        "required": False,
                        "description": "Only emit events from these publishers.",
                    },
                    "emit_on_new_listing": {
                        "type": "bool",
                        "required": False,
                        "description": "Emit events for new listings.",
                    },
                    "emit_on_import": {
                        "type": "bool",
                        "required": False,
                        "description": "Emit events for new imports.",
                    },
                    "emit_on_rating": {
                        "type": "bool",
                        "required": False,
                        "description": "Emit events for new ratings.",
                    },
                    "emit_on_trending": {
                        "type": "bool",
                        "required": False,
                        "description": "Emit events for trending functions.",
                    },
                    "trending_threshold": {
                        "type": "int",
                        "required": False,
                        "description": (
                            "Minimum import acceleration to trigger trending event."
                        ),
                    },
                },
                estimated_cost=0.0,
            ),
            # 8. status
            SkillAction(
                name="status",
                description=(
                    "View current monitoring state, watch rules, snapshot age, "
                    "stats, and configuration."
                ),
                parameters={},
                estimated_cost=0.0,
            ),
        ]

    # ------------------------------------------------------------------
    # Execute dispatcher
    # ------------------------------------------------------------------

    async def execute(self, action: str, params: Dict) -> SkillResult:
        """
        Route an action name to the appropriate handler method.

        Args:
            action: One of the eight supported action names.
            params: Parameters dict for the action.

        Returns:
            SkillResult describing outcome.
        """
        handlers = {
            "wire": self._action_wire,
            "unwire": self._action_unwire,
            "check": self._action_check,
            "watch": self._action_watch,
            "unwatch": self._action_unwatch,
            "trending": self._action_trending,
            "configure": self._action_configure,
            "status": self._action_status,
        }
        handler = handlers.get(action)
        if not handler:
            return SkillResult(
                success=False,
                message=f"Unknown action: {action}. Available: {list(handlers.keys())}",
            )
        try:
            return await handler(params)
        except Exception as exc:
            return SkillResult(
                success=False,
                message=f"Error in {action}: {exc}",
            )

    # ------------------------------------------------------------------
    # Action: wire
    # ------------------------------------------------------------------

    async def _action_wire(self, params: Dict) -> SkillResult:
        """
        Activate discovery event monitoring.

        Applies optional configuration overrides, takes an initial snapshot
        of the marketplace, and marks the bridge as active.
        """
        # Apply optional config overrides
        if "poll_interval_reflections" in params:
            self._config["poll_interval_reflections"] = max(
                1, int(params["poll_interval_reflections"])
            )
        if "watch_categories" in params:
            val = params["watch_categories"]
            self._config["watch_categories"] = val if isinstance(val, list) else []
        if "watch_agents" in params:
            val = params["watch_agents"]
            self._config["watch_agents"] = val if isinstance(val, list) else []

        # Take initial snapshot
        snapshot_result = await self._take_snapshot()

        self._wire_state = {
            "active": True,
            "wired_at": _now_iso(),
            "cycles_since_check": 0,
        }

        self._save_state()

        return SkillResult(
            success=True,
            message=(
                "Marketplace discovery monitoring activated.  "
                f"Snapshot captured with {len(self._snapshot.get('listings', {}))} listings."
            ),
            data={
                "wire_state": self._wire_state,
                "config": self._config,
                "snapshot_listings": len(self._snapshot.get("listings", {})),
                "snapshot_taken_at": self._snapshot.get("taken_at"),
                "snapshot_ok": snapshot_result,
            },
        )

    # ------------------------------------------------------------------
    # Action: unwire
    # ------------------------------------------------------------------

    async def _action_unwire(self, params: Dict) -> SkillResult:
        """Deactivate monitoring and reset wire state."""
        was_active = self._wire_state.get("active", False)
        self._wire_state = {
            "active": False,
            "wired_at": None,
            "cycles_since_check": 0,
        }
        self._save_state()

        return SkillResult(
            success=True,
            message=(
                "Marketplace discovery monitoring deactivated."
                if was_active
                else "Monitoring was already inactive."
            ),
            data={"wire_state": self._wire_state},
        )

    # ------------------------------------------------------------------
    # Action: check
    # ------------------------------------------------------------------

    async def _action_check(self, params: Dict) -> SkillResult:
        """
        Compare current marketplace state to the stored snapshot, detect
        changes, emit events for each, update the snapshot, and return a
        summary of all detected changes.
        """
        # Fetch current marketplace state
        current_listings = await self._fetch_current_listings()
        if current_listings is None:
            return SkillResult(
                success=False,
                message=(
                    "Could not read marketplace state.  Ensure the "
                    "function_marketplace skill is available."
                ),
            )

        old_snapshot = self._snapshot.get("listings", {})
        changes: Dict[str, List[Dict]] = {
            "new_listings": [],
            "new_imports": [],
            "new_ratings": [],
            "price_changes": [],
            "status_changes": [],
            "trending": [],
        }
        events_emitted: List[Dict] = []

        # --- Detect new listings ---
        for lid, current in current_listings.items():
            if lid not in old_snapshot:
                # Respect category / agent filters
                if not self._passes_filters(current):
                    continue
                change = {
                    "listing_id": lid,
                    "function_name": current.get("function_name", ""),
                    "agent_name": current.get("agent_name", ""),
                    "category": current.get("category", ""),
                    "tags": current.get("tags", []),
                    "price": current.get("price_per_import", 0),
                }
                changes["new_listings"].append(change)
                self._stats["new_listings_detected"] += 1
                if self._config.get("emit_on_new_listing", True):
                    await self._emit_event("marketplace.new_listing", change)
                    events_emitted.append({"topic": "marketplace.new_listing", **change})
                # Check watch rules
                await self._check_watch_rules("new_listing", lid, current, change)

        # --- Detect import / rating / price / status changes on existing listings ---
        for lid, current in current_listings.items():
            if lid not in old_snapshot:
                continue  # Already handled above

            old = old_snapshot[lid]

            if not self._passes_filters(current):
                continue

            # New imports
            old_imports = old.get("import_count", 0)
            new_imports = current.get("import_count", 0)
            if new_imports > old_imports:
                import_delta = new_imports - old_imports
                change = {
                    "listing_id": lid,
                    "function_name": current.get("function_name", ""),
                    "importer_agent": "unknown",  # browse doesn't expose per-import info
                    "publisher_agent": current.get("agent_name", ""),
                    "old_import_count": old_imports,
                    "new_import_count": new_imports,
                    "delta": import_delta,
                }
                changes["new_imports"].append(change)
                self._stats["new_imports_detected"] += import_delta
                if self._config.get("emit_on_import", True):
                    await self._emit_event("marketplace.new_import", change)
                    events_emitted.append({"topic": "marketplace.new_import", **change})
                await self._check_watch_rules("new_import", lid, current, change)

            # New ratings
            old_rating_count = old.get("rating_count", 0)
            new_rating_count = current.get("rating_count", 0)
            if new_rating_count > old_rating_count:
                change = {
                    "listing_id": lid,
                    "function_name": current.get("function_name", ""),
                    "agent_name": current.get("agent_name", ""),
                    "rating": current.get("avg_rating", 0),
                    "avg_rating": current.get("avg_rating", 0),
                    "old_rating_count": old_rating_count,
                    "new_rating_count": new_rating_count,
                }
                changes["new_ratings"].append(change)
                self._stats["new_ratings_detected"] += (new_rating_count - old_rating_count)
                if self._config.get("emit_on_rating", True):
                    await self._emit_event("marketplace.new_rating", change)
                    events_emitted.append({"topic": "marketplace.new_rating", **change})
                await self._check_watch_rules("rating_change", lid, current, change)

            # Price changes
            old_price = old.get("price_per_import", 0)
            new_price = current.get("price_per_import", 0)
            if old_price != new_price:
                change = {
                    "listing_id": lid,
                    "function_name": current.get("function_name", ""),
                    "old_price": old_price,
                    "new_price": new_price,
                }
                changes["price_changes"].append(change)
                await self._emit_event("marketplace.price_change", change)
                events_emitted.append({"topic": "marketplace.price_change", **change})

            # Status changes
            old_status = old.get("status", "active")
            new_status = current.get("status", "active")
            if old_status != new_status:
                change = {
                    "listing_id": lid,
                    "function_name": current.get("function_name", ""),
                    "old_status": old_status,
                    "new_status": new_status,
                }
                changes["status_changes"].append(change)

        # --- Detect trending functions ---
        if self._config.get("emit_on_trending", True):
            trending = self._detect_trending(old_snapshot, current_listings)
            for t in trending:
                changes["trending"].append(t)
                self._stats["trending_detected"] += 1
                await self._emit_event("marketplace.trending", t)
                events_emitted.append({"topic": "marketplace.trending", **t})
                await self._check_watch_rules("trending", t["listing_id"], current_listings.get(t["listing_id"], {}), t)

        # Update snapshot
        self._snapshot = {
            "listings": {
                lid: self._listing_to_snapshot_entry(info)
                for lid, info in current_listings.items()
            },
            "imports_count": sum(
                info.get("import_count", 0) for info in current_listings.values()
            ),
            "taken_at": _now_iso(),
        }

        # Update stats
        self._stats["checks_performed"] += 1
        self._stats["last_check_at"] = _now_iso()
        self._wire_state["cycles_since_check"] = 0

        self._save_state()

        total_changes = sum(len(v) for v in changes.values())
        return SkillResult(
            success=True,
            message=(
                f"Marketplace check complete: {total_changes} change(s) detected, "
                f"{len(events_emitted)} event(s) emitted."
            ),
            data={
                "changes": changes,
                "total_changes": total_changes,
                "events_emitted_count": len(events_emitted),
                "snapshot_listings": len(self._snapshot.get("listings", {})),
                "snapshot_taken_at": self._snapshot.get("taken_at"),
            },
        )

    # ------------------------------------------------------------------
    # Action: watch
    # ------------------------------------------------------------------

    async def _action_watch(self, params: Dict) -> SkillResult:
        """
        Add a persistent watch rule that triggers additional
        ``marketplace.watch.matched`` events when a matching change is
        detected during a check.
        """
        rule_name = params.get("rule_name", "").strip()
        event_type = params.get("event_type", "").strip()

        if not rule_name:
            return SkillResult(
                success=False,
                message="rule_name is required.",
            )
        if event_type not in VALID_EVENT_TYPES:
            return SkillResult(
                success=False,
                message=(
                    f"Invalid event_type '{event_type}'.  "
                    f"Valid types: {sorted(VALID_EVENT_TYPES)}"
                ),
            )
        if len(self._watch_rules) >= MAX_WATCH_RULES and rule_name not in self._watch_rules:
            return SkillResult(
                success=False,
                message=f"Maximum of {MAX_WATCH_RULES} watch rules reached.  Remove a rule first.",
            )

        rule: Dict[str, Any] = {
            "rule_name": rule_name,
            "event_type": event_type,
            "created_at": _now_iso(),
        }

        # Optional filter fields
        if "category" in params:
            rule["category"] = params["category"]
        if "min_rating" in params:
            rule["min_rating"] = float(params["min_rating"])
        if "agent_name" in params:
            rule["agent_name"] = params["agent_name"]
        if "tags" in params:
            rule["tags"] = params["tags"] if isinstance(params["tags"], list) else []

        is_update = rule_name in self._watch_rules
        self._watch_rules[rule_name] = rule
        self._save_state()

        return SkillResult(
            success=True,
            message=(
                f"Watch rule '{rule_name}' {'updated' if is_update else 'created'} "
                f"for event_type={event_type}."
            ),
            data={
                "rule": rule,
                "total_rules": len(self._watch_rules),
                "updated": is_update,
            },
        )

    # ------------------------------------------------------------------
    # Action: unwatch
    # ------------------------------------------------------------------

    async def _action_unwatch(self, params: Dict) -> SkillResult:
        """Remove a watch rule by name."""
        rule_name = params.get("rule_name", "").strip()
        if not rule_name:
            return SkillResult(success=False, message="rule_name is required.")

        if rule_name not in self._watch_rules:
            return SkillResult(
                success=False,
                message=f"Watch rule '{rule_name}' not found.",
            )

        removed = self._watch_rules.pop(rule_name)
        self._save_state()

        return SkillResult(
            success=True,
            message=f"Watch rule '{rule_name}' removed.",
            data={
                "removed_rule": removed,
                "remaining_rules": len(self._watch_rules),
            },
        )

    # ------------------------------------------------------------------
    # Action: trending
    # ------------------------------------------------------------------

    async def _action_trending(self, params: Dict) -> SkillResult:
        """
        Analyse marketplace for functions with accelerating import counts.

        Compares the current import counts against the stored snapshot and
        identifies functions whose import delta exceeds ``min_imports``.
        Emits ``marketplace.trending`` events for each.
        """
        window = max(1, int(params.get("window", 10)))
        min_imports = max(1, int(params.get("min_imports", 2)))

        current_listings = await self._fetch_current_listings()
        if current_listings is None:
            return SkillResult(
                success=False,
                message="Could not read marketplace state.",
            )

        old_snapshot = self._snapshot.get("listings", {})
        trending_functions: List[Dict] = []

        # Sort listings by import_count descending, take top `window`
        sorted_listings = sorted(
            current_listings.values(),
            key=lambda x: x.get("import_count", 0),
            reverse=True,
        )[:window]

        for listing in sorted_listings:
            lid = listing.get("listing_id", "")
            current_imports = listing.get("import_count", 0)
            if current_imports < min_imports:
                continue

            old_entry = old_snapshot.get(lid, {})
            old_imports = old_entry.get("import_count", 0)
            acceleration = current_imports - old_imports

            if acceleration >= self._config.get("trending_threshold", 2):
                trending_entry = {
                    "listing_id": lid,
                    "function_name": listing.get("function_name", ""),
                    "import_count": current_imports,
                    "acceleration": acceleration,
                    "agent_name": listing.get("agent_name", ""),
                    "category": listing.get("category", ""),
                    "avg_rating": listing.get("avg_rating", 0),
                }
                trending_functions.append(trending_entry)
                self._stats["trending_detected"] += 1

                if self._config.get("emit_on_trending", True):
                    await self._emit_event("marketplace.trending", trending_entry)

                await self._check_watch_rules(
                    "trending", lid, listing, trending_entry
                )

        self._save_state()

        return SkillResult(
            success=True,
            message=f"Found {len(trending_functions)} trending function(s).",
            data={
                "trending": trending_functions,
                "window": window,
                "min_imports": min_imports,
                "threshold": self._config.get("trending_threshold", 2),
            },
        )

    # ------------------------------------------------------------------
    # Action: configure
    # ------------------------------------------------------------------

    async def _action_configure(self, params: Dict) -> SkillResult:
        """Update monitoring configuration values."""
        updated: List[Dict] = []

        config_keys = {
            "poll_interval_reflections",
            "watch_categories",
            "watch_agents",
            "emit_on_new_listing",
            "emit_on_import",
            "emit_on_rating",
            "emit_on_trending",
            "trending_threshold",
        }

        for key in config_keys:
            if key in params:
                old_val = self._config.get(key)
                new_val = params[key]
                # Validate int fields
                if key in ("poll_interval_reflections", "trending_threshold"):
                    new_val = max(1, int(new_val))
                # Validate list fields
                if key in ("watch_categories", "watch_agents"):
                    new_val = new_val if isinstance(new_val, list) else []
                # Validate bool fields
                if key.startswith("emit_on_"):
                    new_val = bool(new_val)
                self._config[key] = new_val
                updated.append({"key": key, "old": old_val, "new": new_val})

        self._save_state()

        return SkillResult(
            success=True,
            message=f"Updated {len(updated)} configuration value(s).",
            data={
                "updated": updated,
                "config": self._config,
            },
        )

    # ------------------------------------------------------------------
    # Action: status
    # ------------------------------------------------------------------

    async def _action_status(self, params: Dict) -> SkillResult:
        """Return current wire state, watch rules, snapshot info, stats, and config."""
        snapshot_age = None
        taken_at = self._snapshot.get("taken_at")
        if taken_at:
            try:
                taken_dt = datetime.fromisoformat(taken_at.rstrip("Z"))
                delta = datetime.utcnow() - taken_dt
                snapshot_age = f"{int(delta.total_seconds())}s"
            except (ValueError, TypeError):
                snapshot_age = "unknown"

        return SkillResult(
            success=True,
            message=(
                f"Discovery monitoring {'ACTIVE' if self._wire_state.get('active') else 'INACTIVE'} "
                f"| {self._stats.get('checks_performed', 0)} checks "
                f"| {self._stats.get('events_emitted', 0)} events emitted "
                f"| {len(self._watch_rules)} watch rules"
            ),
            data={
                "wire_state": self._wire_state,
                "watch_rules": self._watch_rules,
                "snapshot": {
                    "listing_count": len(self._snapshot.get("listings", {})),
                    "imports_count": self._snapshot.get("imports_count", 0),
                    "taken_at": taken_at,
                    "age": snapshot_age,
                },
                "stats": self._stats,
                "config": self._config,
                "event_log_size": len(self._event_log),
            },
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _fetch_current_listings(self) -> Optional[Dict[str, Dict]]:
        """
        Fetch the current marketplace listings via the ``function_marketplace``
        skill's ``browse`` and ``status`` actions.

        Returns:
            Dict mapping listing_id to listing info, or None on failure.
        """
        if not (hasattr(self, "context") and self.context):
            return None

        try:
            # Browse with a large limit to capture all listings
            browse_result = await self.context.call_skill(
                "function_marketplace",
                "browse",
                {"limit": 500},
            )
            if not browse_result or not browse_result.success:
                return None

            listings_list = browse_result.data.get("listings", [])
            listings_map: Dict[str, Dict] = {}
            for listing in listings_list:
                lid = listing.get("listing_id", "")
                if lid:
                    listings_map[lid] = listing

            return listings_map
        except Exception:
            return None

    async def _take_snapshot(self) -> bool:
        """
        Take a snapshot of the current marketplace state for later comparison.

        Returns:
            True if snapshot was captured successfully, False otherwise.
        """
        current_listings = await self._fetch_current_listings()
        if current_listings is None:
            # Initialise with empty snapshot — first check will treat
            # everything as new.
            self._snapshot = {
                "listings": {},
                "imports_count": 0,
                "taken_at": _now_iso(),
            }
            return False

        self._snapshot = {
            "listings": {
                lid: self._listing_to_snapshot_entry(info)
                for lid, info in current_listings.items()
            },
            "imports_count": sum(
                info.get("import_count", 0) for info in current_listings.values()
            ),
            "taken_at": _now_iso(),
        }
        return True

    @staticmethod
    def _listing_to_snapshot_entry(listing: Dict) -> Dict:
        """
        Extract the fields we track for change detection from a listing dict.

        Keeps only the minimal set of fields needed for diffing so the
        snapshot stays small.
        """
        return {
            "listing_id": listing.get("listing_id", ""),
            "function_name": listing.get("function_name", ""),
            "agent_name": listing.get("agent_name", ""),
            "category": listing.get("category", ""),
            "tags": listing.get("tags", []),
            "import_count": listing.get("import_count", 0),
            "avg_rating": listing.get("avg_rating", 0),
            "rating_count": listing.get("rating_count", 0),
            "price_per_import": listing.get("price_per_import", 0),
            "status": listing.get("status", "active"),
        }

    def _passes_filters(self, listing: Dict) -> bool:
        """
        Check whether a listing passes the configured category and agent
        filters.  If no filters are set all listings pass.

        Args:
            listing: Marketplace listing dict.

        Returns:
            True if the listing passes all configured filters.
        """
        watch_categories = self._config.get("watch_categories", [])
        if watch_categories:
            if listing.get("category", "") not in watch_categories:
                return False

        watch_agents = self._config.get("watch_agents", [])
        if watch_agents:
            if listing.get("agent_name", "") not in watch_agents:
                return False

        return True

    def _detect_trending(
        self,
        old_snapshot: Dict[str, Dict],
        current_listings: Dict[str, Dict],
    ) -> List[Dict]:
        """
        Identify functions whose import count has accelerated above the
        configured threshold since the last snapshot.

        Args:
            old_snapshot: Previous snapshot listings dict.
            current_listings: Current marketplace listings dict.

        Returns:
            List of trending function dicts.
        """
        threshold = self._config.get("trending_threshold", 2)
        trending: List[Dict] = []

        for lid, current in current_listings.items():
            if lid not in old_snapshot:
                continue
            old = old_snapshot[lid]
            old_imports = old.get("import_count", 0)
            new_imports = current.get("import_count", 0)
            acceleration = new_imports - old_imports

            if acceleration >= threshold:
                trending.append({
                    "listing_id": lid,
                    "function_name": current.get("function_name", ""),
                    "import_count": new_imports,
                    "acceleration": acceleration,
                    "agent_name": current.get("agent_name", ""),
                    "category": current.get("category", ""),
                    "avg_rating": current.get("avg_rating", 0),
                })

        return trending

    async def _emit_event(self, topic: str, data: Dict) -> bool:
        """
        Emit an event via the EventBus skill and record it in the local log.

        Args:
            topic: Event topic string (e.g. ``marketplace.new_listing``).
            data: Event payload dict.

        Returns:
            True if the event was successfully published to the bus,
            False if it was only recorded locally.
        """
        record = {
            "topic": topic,
            "data": data,
            "timestamp": _now_iso(),
        }
        self._event_log.append(record)
        # Trim to stay within budget
        if len(self._event_log) > MAX_EVENT_LOG:
            self._event_log = self._event_log[-MAX_EVENT_LOG:]

        self._stats["events_emitted"] += 1

        emitted_to_bus = False
        if hasattr(self, "context") and self.context:
            try:
                result = await self.context.call_skill(
                    "event",
                    "publish",
                    {
                        "event": topic,
                        "data": {
                            **data,
                            "source": "function_marketplace_discovery_events",
                        },
                    },
                )
                emitted_to_bus = result.success if result else False
            except Exception:
                pass

        return emitted_to_bus

    async def _check_watch_rules(
        self,
        event_type: str,
        listing_id: str,
        listing: Dict,
        change_details: Dict,
    ) -> None:
        """
        Check all watch rules against a detected change.  For each rule
        that matches, emit a ``marketplace.watch.matched`` event.

        Args:
            event_type: The type of change (new_listing, new_import, etc.).
            listing_id: The affected listing's ID.
            listing: The listing dict with current fields.
            change_details: The change payload that was already emitted.
        """
        for rule_name, rule in self._watch_rules.items():
            if rule.get("event_type") != event_type:
                continue

            # Check optional filter: category
            if "category" in rule:
                if listing.get("category", "") != rule["category"]:
                    continue

            # Check optional filter: min_rating
            if "min_rating" in rule:
                if listing.get("avg_rating", 0) < rule["min_rating"]:
                    continue

            # Check optional filter: agent_name
            if "agent_name" in rule:
                if listing.get("agent_name", "") != rule["agent_name"]:
                    continue

            # Check optional filter: tags (at least one overlap)
            if "tags" in rule and rule["tags"]:
                listing_tags = set(listing.get("tags", []))
                rule_tags = set(rule["tags"])
                if not listing_tags & rule_tags:
                    continue

            # Rule matched — emit watch.matched event
            match_payload = {
                "rule_name": rule_name,
                "event_type": event_type,
                "listing_id": listing_id,
                "details": change_details,
            }
            self._stats["watch_matches"] += 1
            await self._emit_event("marketplace.watch.matched", match_payload)

    def estimate_cost(self, action: str, params: Dict) -> float:
        """All actions in this skill are free (no external API calls)."""
        return 0.0
