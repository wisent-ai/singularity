#!/usr/bin/env python3
"""
Comprehensive tests for FunctionMarketplaceDiscoveryEventsSkill.

Tests cover: initialization, state persistence, wire/unwire, check (change
detection), watch/unwatch rules, trending analysis, configure, status,
snapshot comparison, watch rule matching, event emission, and manifest.
"""

import json
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from singularity.skills.function_marketplace_discovery_events import (
    FunctionMarketplaceDiscoveryEventsSkill,
    MAX_EVENT_LOG,
    MAX_WATCH_RULES,
    STATE_FILE,
    VALID_EVENT_TYPES,
    _now_iso,
)
from singularity.skills.base import SkillResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_listing(
    listing_id="fn_abc",
    function_name="my_func",
    agent_name="alice",
    category="utility",
    tags=None,
    import_count=0,
    avg_rating=0.0,
    rating_count=0,
    price_per_import=0.0,
    status="active",
):
    """Build a marketplace listing dict for testing."""
    return {
        "listing_id": listing_id,
        "function_name": function_name,
        "agent_name": agent_name,
        "category": category,
        "tags": tags or [],
        "import_count": import_count,
        "avg_rating": avg_rating,
        "rating_count": rating_count,
        "price_per_import": price_per_import,
        "status": status,
    }


def _listings_map(*listings):
    """Convert a list of listing dicts to a {listing_id: listing} map."""
    return {l["listing_id"]: l for l in listings}


def _make_browse_result(listings_list):
    """Build a SkillResult that looks like a marketplace browse response."""
    return SkillResult(
        success=True,
        message="ok",
        data={"listings": listings_list, "count": len(listings_list)},
    )


def _make_context(browse_listings=None):
    """
    Build a mock context whose ``call_skill`` returns browse results for
    the function_marketplace and success for event publish.
    """
    ctx = MagicMock()
    listings = browse_listings if browse_listings is not None else []

    async def _call_skill(skill_id, action, params=None):
        if skill_id == "function_marketplace" and action == "browse":
            return _make_browse_result(listings)
        if skill_id == "event" and action == "publish":
            return SkillResult(success=True, message="published")
        return SkillResult(success=False, message="unknown call")

    ctx.call_skill = AsyncMock(side_effect=_call_skill)
    return ctx


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def skill(tmp_path, monkeypatch):
    """Create a skill instance with state file redirected to tmp_path."""
    monkeypatch.setattr(
        "singularity.skills.function_marketplace_discovery_events.STATE_FILE",
        tmp_path / "discovery_events.json",
    )
    monkeypatch.setattr(
        "singularity.skills.function_marketplace_discovery_events.DATA_DIR",
        tmp_path,
    )
    return FunctionMarketplaceDiscoveryEventsSkill()


@pytest.fixture
def state_path(tmp_path, monkeypatch):
    """Return the redirected state file path (also patches the module constant)."""
    p = tmp_path / "discovery_events.json"
    monkeypatch.setattr(
        "singularity.skills.function_marketplace_discovery_events.STATE_FILE",
        p,
    )
    monkeypatch.setattr(
        "singularity.skills.function_marketplace_discovery_events.DATA_DIR",
        tmp_path,
    )
    return p


@pytest.fixture
def wired_skill(skill):
    """A skill instance that has been wired (active) with an empty snapshot."""
    skill._wire_state = {
        "active": True,
        "wired_at": _now_iso(),
        "cycles_since_check": 0,
    }
    skill._snapshot = {
        "listings": {},
        "imports_count": 0,
        "taken_at": _now_iso(),
    }
    return skill


# ===================================================================
# Test Class: Initialization
# ===================================================================


class TestInitialization:
    """Tests 1-3: default state, persistence round-trip, corrupted state."""

    def test_default_state_initialization(self, skill):
        """Test 1: A fresh skill has inactive wire state, empty snapshot, etc."""
        assert skill._wire_state["active"] is False
        assert skill._wire_state["wired_at"] is None
        assert skill._snapshot["listings"] == {}
        assert skill._watch_rules == {}
        assert skill._event_log == []
        assert skill._config["poll_interval_reflections"] == 5
        assert skill._stats["checks_performed"] == 0
        assert skill._stats["events_emitted"] == 0

    def test_state_persistence_round_trip(self, state_path):
        """Test 2: Save state, re-create skill, and confirm state is restored."""
        skill1 = FunctionMarketplaceDiscoveryEventsSkill()
        skill1._wire_state["active"] = True
        skill1._wire_state["wired_at"] = "2025-01-01T00:00:00Z"
        skill1._watch_rules["r1"] = {
            "rule_name": "r1",
            "event_type": "new_listing",
            "created_at": "2025-01-01T00:00:00Z",
        }
        skill1._stats["checks_performed"] = 7
        skill1._config["poll_interval_reflections"] = 10
        skill1._event_log.append({"topic": "test", "data": {}, "timestamp": "t"})
        skill1._save_state()

        skill2 = FunctionMarketplaceDiscoveryEventsSkill()
        assert skill2._wire_state["active"] is True
        assert skill2._wire_state["wired_at"] == "2025-01-01T00:00:00Z"
        assert "r1" in skill2._watch_rules
        assert skill2._stats["checks_performed"] == 7
        assert skill2._config["poll_interval_reflections"] == 10
        assert len(skill2._event_log) == 1

    def test_corrupted_state_file_falls_back_to_defaults(self, state_path):
        """Test 3: A corrupted JSON file causes the skill to start fresh."""
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text("NOT VALID JSON {{{}}}}")

        skill = FunctionMarketplaceDiscoveryEventsSkill()
        assert skill._wire_state["active"] is False
        assert skill._snapshot["listings"] == {}
        assert skill._watch_rules == {}


# ===================================================================
# Test Class: Action - wire
# ===================================================================


class TestWireAction:
    """Tests 4-7: wire with defaults, custom params, snapshot, already wired."""

    async def test_wire_sets_state_with_defaults(self, skill):
        """Test 4: Wire activates monitoring with default config."""
        # No context, snapshot will be empty
        result = await skill.execute("wire", {})
        assert result.success
        assert skill._wire_state["active"] is True
        assert skill._wire_state["wired_at"] is not None
        assert skill._config["poll_interval_reflections"] == 5
        assert skill._config["watch_categories"] == []
        assert skill._config["watch_agents"] == []

    async def test_wire_with_custom_parameters(self, skill):
        """Test 5: Wire with custom poll_interval, watch_categories, watch_agents."""
        result = await skill.execute("wire", {
            "poll_interval_reflections": 10,
            "watch_categories": ["utility", "ai"],
            "watch_agents": ["alice", "bob"],
        })
        assert result.success
        assert skill._config["poll_interval_reflections"] == 10
        assert skill._config["watch_categories"] == ["utility", "ai"]
        assert skill._config["watch_agents"] == ["alice", "bob"]

    async def test_wire_takes_initial_snapshot(self, skill):
        """Test 6: Wire takes a snapshot including current marketplace listings."""
        listings = [_make_listing("fn_1", "func_a"), _make_listing("fn_2", "func_b")]
        skill.context = _make_context(listings)

        result = await skill.execute("wire", {})
        assert result.success
        assert result.data["snapshot_listings"] == 2
        assert "fn_1" in skill._snapshot["listings"]
        assert "fn_2" in skill._snapshot["listings"]
        assert result.data["snapshot_ok"] is True

    async def test_wire_when_already_wired(self, wired_skill):
        """Test 7: Re-wiring an already wired skill succeeds (overwrites)."""
        result = await wired_skill.execute("wire", {"poll_interval_reflections": 3})
        assert result.success
        assert wired_skill._wire_state["active"] is True
        assert wired_skill._config["poll_interval_reflections"] == 3


# ===================================================================
# Test Class: Action - unwire
# ===================================================================


class TestUnwireAction:
    """Tests 8-9: unwire clears state, unwire when already unwired."""

    async def test_unwire_clears_state(self, wired_skill):
        """Test 8: Unwire deactivates monitoring."""
        assert wired_skill._wire_state["active"] is True
        result = await wired_skill.execute("unwire", {})
        assert result.success
        assert wired_skill._wire_state["active"] is False
        assert wired_skill._wire_state["wired_at"] is None
        assert "deactivated" in result.message.lower()

    async def test_unwire_when_already_unwired(self, skill):
        """Test 9: Unwire on inactive monitoring still succeeds with message."""
        result = await skill.execute("unwire", {})
        assert result.success
        assert "already inactive" in result.message.lower()


# ===================================================================
# Test Class: Action - check
# ===================================================================


class TestCheckAction:
    """Tests 10-19: new listings, imports, ratings, price changes, no changes,
    snapshot update, event emission, filters, stats."""

    async def test_check_detects_new_listings(self, wired_skill):
        """Test 10: Check detects a brand new listing."""
        new_listing = _make_listing("fn_new", "new_func", "alice")
        wired_skill.context = _make_context([new_listing])

        result = await wired_skill.execute("check", {})
        assert result.success
        assert len(result.data["changes"]["new_listings"]) == 1
        assert result.data["changes"]["new_listings"][0]["listing_id"] == "fn_new"

    async def test_check_detects_new_imports(self, wired_skill):
        """Test 11: Check detects import_count increase."""
        old = _make_listing("fn_1", "func_a", import_count=5)
        wired_skill._snapshot["listings"] = {
            "fn_1": FunctionMarketplaceDiscoveryEventsSkill._listing_to_snapshot_entry(old)
        }

        updated = _make_listing("fn_1", "func_a", import_count=8)
        wired_skill.context = _make_context([updated])

        result = await wired_skill.execute("check", {})
        assert result.success
        assert len(result.data["changes"]["new_imports"]) == 1
        change = result.data["changes"]["new_imports"][0]
        assert change["old_import_count"] == 5
        assert change["new_import_count"] == 8
        assert change["delta"] == 3

    async def test_check_detects_new_ratings(self, wired_skill):
        """Test 12: Check detects rating_count increase."""
        old = _make_listing("fn_1", "func_a", rating_count=2, avg_rating=3.5)
        wired_skill._snapshot["listings"] = {
            "fn_1": FunctionMarketplaceDiscoveryEventsSkill._listing_to_snapshot_entry(old)
        }

        updated = _make_listing("fn_1", "func_a", rating_count=4, avg_rating=4.0)
        wired_skill.context = _make_context([updated])

        result = await wired_skill.execute("check", {})
        assert result.success
        assert len(result.data["changes"]["new_ratings"]) == 1
        change = result.data["changes"]["new_ratings"][0]
        assert change["old_rating_count"] == 2
        assert change["new_rating_count"] == 4

    async def test_check_detects_price_changes(self, wired_skill):
        """Test 13: Check detects price_per_import change."""
        old = _make_listing("fn_1", "func_a", price_per_import=1.0)
        wired_skill._snapshot["listings"] = {
            "fn_1": FunctionMarketplaceDiscoveryEventsSkill._listing_to_snapshot_entry(old)
        }

        updated = _make_listing("fn_1", "func_a", price_per_import=2.5)
        wired_skill.context = _make_context([updated])

        result = await wired_skill.execute("check", {})
        assert result.success
        assert len(result.data["changes"]["price_changes"]) == 1
        change = result.data["changes"]["price_changes"][0]
        assert change["old_price"] == 1.0
        assert change["new_price"] == 2.5

    async def test_check_with_no_changes(self, wired_skill):
        """Test 14: Check with identical state produces zero changes."""
        listing = _make_listing("fn_1", "func_a", import_count=5, rating_count=2)
        wired_skill._snapshot["listings"] = {
            "fn_1": FunctionMarketplaceDiscoveryEventsSkill._listing_to_snapshot_entry(listing)
        }
        wired_skill.context = _make_context([listing])

        result = await wired_skill.execute("check", {})
        assert result.success
        assert result.data["total_changes"] == 0
        assert result.data["events_emitted_count"] == 0

    async def test_check_updates_snapshot(self, wired_skill):
        """Test 15: After check, the snapshot reflects the new state."""
        new_listing = _make_listing("fn_new", "new_func")
        wired_skill.context = _make_context([new_listing])

        await wired_skill.execute("check", {})
        assert "fn_new" in wired_skill._snapshot["listings"]
        assert wired_skill._snapshot["taken_at"] is not None

    async def test_check_emits_events_via_context(self, wired_skill):
        """Test 16: Check calls context.call_skill to publish events."""
        new_listing = _make_listing("fn_new", "new_func")
        wired_skill.context = _make_context([new_listing])

        await wired_skill.execute("check", {})

        # Should have called event publish at least once (for new_listing)
        calls = [
            c for c in wired_skill.context.call_skill.call_args_list
            if c.args[0] == "event" and c.args[1] == "publish"
        ]
        assert len(calls) >= 1

    async def test_check_respects_watch_categories_filter(self, wired_skill):
        """Test 17: When watch_categories is set, listings outside are ignored."""
        wired_skill._config["watch_categories"] = ["ai"]

        # This listing is in 'utility', not 'ai'
        new_listing = _make_listing("fn_1", "func", category="utility")
        wired_skill.context = _make_context([new_listing])

        result = await wired_skill.execute("check", {})
        assert result.success
        assert len(result.data["changes"]["new_listings"]) == 0

    async def test_check_respects_watch_agents_filter(self, wired_skill):
        """Test 18: When watch_agents is set, listings from other agents are ignored."""
        wired_skill._config["watch_agents"] = ["bob"]

        new_listing = _make_listing("fn_1", "func", agent_name="alice")
        wired_skill.context = _make_context([new_listing])

        result = await wired_skill.execute("check", {})
        assert result.success
        assert len(result.data["changes"]["new_listings"]) == 0

    async def test_check_updates_stats(self, wired_skill):
        """Test 19: Check increments stats counters."""
        new_listing = _make_listing("fn_new", "new_func")
        old_listing = _make_listing("fn_old", "old_func", import_count=0)
        updated_old = _make_listing("fn_old", "old_func", import_count=3)

        wired_skill._snapshot["listings"] = {
            "fn_old": FunctionMarketplaceDiscoveryEventsSkill._listing_to_snapshot_entry(
                old_listing
            )
        }
        wired_skill.context = _make_context([new_listing, updated_old])

        before_checks = wired_skill._stats["checks_performed"]
        await wired_skill.execute("check", {})

        assert wired_skill._stats["checks_performed"] == before_checks + 1
        assert wired_skill._stats["new_listings_detected"] >= 1
        assert wired_skill._stats["new_imports_detected"] >= 3
        assert wired_skill._stats["last_check_at"] is not None

    async def test_check_fails_without_context(self, wired_skill):
        """Check fails gracefully when no context is available."""
        wired_skill.context = None
        result = await wired_skill.execute("check", {})
        assert not result.success
        assert "marketplace" in result.message.lower()


# ===================================================================
# Test Class: Action - watch
# ===================================================================


class TestWatchAction:
    """Tests 20-26: add rules, matching, filters, limit enforcement."""

    async def test_watch_add_rule_with_all_parameters(self, skill):
        """Test 20: Create a watch rule with every optional parameter."""
        result = await skill.execute("watch", {
            "rule_name": "notify_ai",
            "event_type": "new_listing",
            "category": "ai",
            "min_rating": 3.5,
            "agent_name": "bob",
            "tags": ["ml", "nlp"],
        })
        assert result.success
        assert result.data["total_rules"] == 1
        rule = result.data["rule"]
        assert rule["rule_name"] == "notify_ai"
        assert rule["event_type"] == "new_listing"
        assert rule["category"] == "ai"
        assert rule["min_rating"] == 3.5
        assert rule["agent_name"] == "bob"
        assert rule["tags"] == ["ml", "nlp"]

    async def test_watch_rule_matching_on_new_listing(self, wired_skill):
        """Test 21: A watch rule for new_listing fires on new listings."""
        await wired_skill.execute("watch", {
            "rule_name": "catch_new",
            "event_type": "new_listing",
        })

        new_listing = _make_listing("fn_new", "func_new")
        wired_skill.context = _make_context([new_listing])

        result = await wired_skill.execute("check", {})
        assert result.success
        assert wired_skill._stats["watch_matches"] >= 1

    async def test_watch_rule_matching_on_new_import(self, wired_skill):
        """Test 22: A watch rule for new_import fires on import count increase."""
        await wired_skill.execute("watch", {
            "rule_name": "import_watcher",
            "event_type": "new_import",
        })

        old = _make_listing("fn_1", "func_a", import_count=2)
        wired_skill._snapshot["listings"] = {
            "fn_1": FunctionMarketplaceDiscoveryEventsSkill._listing_to_snapshot_entry(old)
        }

        updated = _make_listing("fn_1", "func_a", import_count=5)
        wired_skill.context = _make_context([updated])

        await wired_skill.execute("check", {})
        assert wired_skill._stats["watch_matches"] >= 1

    async def test_watch_with_category_filter(self, wired_skill):
        """Test 23: Watch rule with category filter only matches that category."""
        await wired_skill.execute("watch", {
            "rule_name": "ai_only",
            "event_type": "new_listing",
            "category": "ai",
        })

        # Listing in 'utility' should NOT match
        util_listing = _make_listing("fn_util", "func_util", category="utility")
        # Listing in 'ai' SHOULD match
        ai_listing = _make_listing("fn_ai", "func_ai", category="ai")
        wired_skill.context = _make_context([util_listing, ai_listing])

        await wired_skill.execute("check", {})
        assert wired_skill._stats["watch_matches"] == 1

    async def test_watch_with_min_rating_filter(self, wired_skill):
        """Test 24: Watch rule with min_rating only matches high-rated listings."""
        await wired_skill.execute("watch", {
            "rule_name": "high_rated",
            "event_type": "new_listing",
            "min_rating": 4.0,
        })

        low = _make_listing("fn_low", "func_low", avg_rating=2.0)
        high = _make_listing("fn_high", "func_high", avg_rating=4.5)
        wired_skill.context = _make_context([low, high])

        await wired_skill.execute("check", {})
        assert wired_skill._stats["watch_matches"] == 1

    async def test_watch_with_tags_filter(self, wired_skill):
        """Test 25: Watch rule with tags filter matches if at least one tag overlaps."""
        await wired_skill.execute("watch", {
            "rule_name": "ml_watch",
            "event_type": "new_listing",
            "tags": ["ml", "deep-learning"],
        })

        no_match = _make_listing("fn_1", "f1", tags=["web", "api"])
        match = _make_listing("fn_2", "f2", tags=["ml", "nlp"])
        wired_skill.context = _make_context([no_match, match])

        await wired_skill.execute("check", {})
        assert wired_skill._stats["watch_matches"] == 1

    async def test_watch_rule_limit_enforcement(self, skill):
        """Test 26: Cannot add more than MAX_WATCH_RULES rules."""
        for i in range(MAX_WATCH_RULES):
            result = await skill.execute("watch", {
                "rule_name": f"rule_{i}",
                "event_type": "new_listing",
            })
            assert result.success

        result = await skill.execute("watch", {
            "rule_name": "one_too_many",
            "event_type": "new_listing",
        })
        assert not result.success
        assert "maximum" in result.message.lower() or str(MAX_WATCH_RULES) in result.message

    async def test_watch_invalid_event_type(self, skill):
        """Watch with invalid event_type fails."""
        result = await skill.execute("watch", {
            "rule_name": "bad",
            "event_type": "nonexistent_event",
        })
        assert not result.success

    async def test_watch_missing_rule_name(self, skill):
        """Watch without rule_name fails."""
        result = await skill.execute("watch", {
            "event_type": "new_listing",
        })
        assert not result.success

    async def test_watch_update_existing_rule(self, skill):
        """Updating an existing rule by name succeeds and does not increase count."""
        await skill.execute("watch", {
            "rule_name": "r1",
            "event_type": "new_listing",
        })
        result = await skill.execute("watch", {
            "rule_name": "r1",
            "event_type": "new_import",
        })
        assert result.success
        assert result.data["updated"] is True
        assert result.data["total_rules"] == 1
        assert skill._watch_rules["r1"]["event_type"] == "new_import"


# ===================================================================
# Test Class: Action - unwatch
# ===================================================================


class TestUnwatchAction:
    """Tests 27-28: remove existing rule, remove non-existent rule."""

    async def test_unwatch_existing_rule(self, skill):
        """Test 27: Removing an existing watch rule succeeds."""
        await skill.execute("watch", {
            "rule_name": "r1",
            "event_type": "new_listing",
        })
        result = await skill.execute("unwatch", {"rule_name": "r1"})
        assert result.success
        assert "r1" not in skill._watch_rules
        assert result.data["remaining_rules"] == 0

    async def test_unwatch_nonexistent_rule(self, skill):
        """Test 28: Removing a rule that does not exist fails."""
        result = await skill.execute("unwatch", {"rule_name": "ghost"})
        assert not result.success
        assert "not found" in result.message.lower()

    async def test_unwatch_empty_rule_name(self, skill):
        """Unwatch with empty rule_name fails."""
        result = await skill.execute("unwatch", {"rule_name": ""})
        assert not result.success


# ===================================================================
# Test Class: Action - trending
# ===================================================================


class TestTrendingAction:
    """Tests 29-32: import acceleration, no data, min_imports, event emission."""

    async def test_trending_detects_import_acceleration(self, wired_skill):
        """Test 29: Trending identifies functions exceeding the threshold."""
        old = _make_listing("fn_1", "func_a", import_count=3)
        wired_skill._snapshot["listings"] = {
            "fn_1": FunctionMarketplaceDiscoveryEventsSkill._listing_to_snapshot_entry(old)
        }
        wired_skill._config["trending_threshold"] = 2

        current = _make_listing("fn_1", "func_a", import_count=10)
        wired_skill.context = _make_context([current])

        result = await wired_skill.execute("trending", {"min_imports": 1})
        assert result.success
        assert len(result.data["trending"]) == 1
        assert result.data["trending"][0]["acceleration"] == 7

    async def test_trending_with_no_data(self, wired_skill):
        """Test 30: Trending with empty marketplace returns zero results."""
        wired_skill.context = _make_context([])
        result = await wired_skill.execute("trending", {})
        assert result.success
        assert len(result.data["trending"]) == 0

    async def test_trending_respects_min_imports(self, wired_skill):
        """Test 31: Functions below min_imports are excluded from trending."""
        old = _make_listing("fn_1", "func_a", import_count=0)
        wired_skill._snapshot["listings"] = {
            "fn_1": FunctionMarketplaceDiscoveryEventsSkill._listing_to_snapshot_entry(old)
        }
        wired_skill._config["trending_threshold"] = 1

        current = _make_listing("fn_1", "func_a", import_count=1)
        wired_skill.context = _make_context([current])

        # min_imports=5 means 1 import is not enough
        result = await wired_skill.execute("trending", {"min_imports": 5})
        assert result.success
        assert len(result.data["trending"]) == 0

    async def test_trending_emits_events(self, wired_skill):
        """Test 32: Trending emits marketplace.trending events for matches."""
        old = _make_listing("fn_1", "func_a", import_count=0)
        wired_skill._snapshot["listings"] = {
            "fn_1": FunctionMarketplaceDiscoveryEventsSkill._listing_to_snapshot_entry(old)
        }
        wired_skill._config["trending_threshold"] = 2

        current = _make_listing("fn_1", "func_a", import_count=5)
        wired_skill.context = _make_context([current])

        before = wired_skill._stats["events_emitted"]
        await wired_skill.execute("trending", {"min_imports": 1})
        assert wired_skill._stats["events_emitted"] > before
        assert wired_skill._stats["trending_detected"] >= 1

    async def test_trending_fails_without_context(self, wired_skill):
        """Trending fails gracefully without context."""
        wired_skill.context = None
        result = await wired_skill.execute("trending", {})
        assert not result.success


# ===================================================================
# Test Class: Action - configure
# ===================================================================


class TestConfigureAction:
    """Tests 33-35: update values, preserve unspecified, invalid types."""

    async def test_configure_updates_values(self, skill):
        """Test 33: Configure changes specified keys."""
        result = await skill.execute("configure", {
            "poll_interval_reflections": 20,
            "emit_on_new_listing": False,
            "trending_threshold": 5,
        })
        assert result.success
        assert skill._config["poll_interval_reflections"] == 20
        assert skill._config["emit_on_new_listing"] is False
        assert skill._config["trending_threshold"] == 5

    async def test_configure_preserves_unspecified_values(self, skill):
        """Test 34: Keys not in params remain at their previous values."""
        original_emit_import = skill._config["emit_on_import"]
        original_emit_rating = skill._config["emit_on_rating"]

        await skill.execute("configure", {"poll_interval_reflections": 15})

        assert skill._config["emit_on_import"] == original_emit_import
        assert skill._config["emit_on_rating"] == original_emit_rating
        assert skill._config["poll_interval_reflections"] == 15

    async def test_configure_with_invalid_int_coerced(self, skill):
        """Test 35: Invalid int types for poll_interval are coerced (min 1)."""
        result = await skill.execute("configure", {
            "poll_interval_reflections": -5,
        })
        assert result.success
        assert skill._config["poll_interval_reflections"] == 1

    async def test_configure_list_validation(self, skill):
        """Configure with non-list for watch_categories falls back to empty list."""
        await skill.execute("configure", {
            "watch_categories": "not_a_list",
        })
        assert skill._config["watch_categories"] == []

    async def test_configure_bool_coercion(self, skill):
        """Configure coerces bool fields correctly."""
        await skill.execute("configure", {
            "emit_on_trending": 0,
        })
        assert skill._config["emit_on_trending"] is False

        await skill.execute("configure", {
            "emit_on_trending": 1,
        })
        assert skill._config["emit_on_trending"] is True


# ===================================================================
# Test Class: Action - status
# ===================================================================


class TestStatusAction:
    """Tests 36-37: status returns all state, status with populated data."""

    async def test_status_returns_all_state(self, skill):
        """Test 36: Status returns wire_state, watch_rules, snapshot, stats, config."""
        result = await skill.execute("status", {})
        assert result.success
        data = result.data
        assert "wire_state" in data
        assert "watch_rules" in data
        assert "snapshot" in data
        assert "stats" in data
        assert "config" in data
        assert "event_log_size" in data

    async def test_status_with_populated_data(self, wired_skill):
        """Test 37: Status reflects populated state accurately."""
        wired_skill._watch_rules["r1"] = {
            "rule_name": "r1",
            "event_type": "new_listing",
            "created_at": _now_iso(),
        }
        wired_skill._stats["checks_performed"] = 42
        wired_skill._stats["events_emitted"] = 15

        result = await wired_skill.execute("status", {})
        assert result.success
        assert "ACTIVE" in result.message
        assert result.data["wire_state"]["active"] is True
        assert "r1" in result.data["watch_rules"]
        assert result.data["stats"]["checks_performed"] == 42
        assert result.data["stats"]["events_emitted"] == 15

    async def test_status_inactive_message(self, skill):
        """Status shows INACTIVE when not wired."""
        result = await skill.execute("status", {})
        assert "INACTIVE" in result.message


# ===================================================================
# Test Class: Snapshot Comparison
# ===================================================================


class TestSnapshotComparison:
    """Tests 38-41: snapshot captures state, detects additions/removals/changes."""

    def test_listing_to_snapshot_entry_captures_state(self, skill):
        """Test 38: _listing_to_snapshot_entry extracts correct fields."""
        listing = _make_listing(
            "fn_1", "func_a", "alice", "ai", ["ml"], 10, 4.5, 3, 1.0, "active"
        )
        entry = FunctionMarketplaceDiscoveryEventsSkill._listing_to_snapshot_entry(listing)
        assert entry["listing_id"] == "fn_1"
        assert entry["function_name"] == "func_a"
        assert entry["agent_name"] == "alice"
        assert entry["category"] == "ai"
        assert entry["tags"] == ["ml"]
        assert entry["import_count"] == 10
        assert entry["avg_rating"] == 4.5
        assert entry["rating_count"] == 3
        assert entry["price_per_import"] == 1.0
        assert entry["status"] == "active"

    async def test_compare_detects_additions(self, wired_skill):
        """Test 39: A listing present in current but not snapshot is 'new'."""
        new = _make_listing("fn_new", "new_func")
        wired_skill.context = _make_context([new])

        result = await wired_skill.execute("check", {})
        assert len(result.data["changes"]["new_listings"]) == 1

    async def test_compare_detects_removals_as_no_event(self, wired_skill):
        """Test 40: A listing in snapshot but missing from current is not reported
        as an event (the skill only detects additions and changes, not removals)."""
        old = _make_listing("fn_old", "old_func")
        wired_skill._snapshot["listings"] = {
            "fn_old": FunctionMarketplaceDiscoveryEventsSkill._listing_to_snapshot_entry(old)
        }
        # Current has no listings
        wired_skill.context = _make_context([])

        result = await wired_skill.execute("check", {})
        assert result.success
        # Removed listing is NOT in any change category
        assert result.data["total_changes"] == 0
        # But the updated snapshot no longer contains the old listing
        assert "fn_old" not in wired_skill._snapshot["listings"]

    async def test_compare_detects_changes(self, wired_skill):
        """Test 41: Multiple types of changes on one listing are all detected."""
        old = _make_listing(
            "fn_1", "func_a", import_count=2, rating_count=1, price_per_import=1.0
        )
        wired_skill._snapshot["listings"] = {
            "fn_1": FunctionMarketplaceDiscoveryEventsSkill._listing_to_snapshot_entry(old)
        }

        updated = _make_listing(
            "fn_1", "func_a", import_count=5, rating_count=3, price_per_import=2.0
        )
        wired_skill.context = _make_context([updated])

        result = await wired_skill.execute("check", {})
        changes = result.data["changes"]
        assert len(changes["new_imports"]) == 1
        assert len(changes["new_ratings"]) == 1
        assert len(changes["price_changes"]) == 1


# ===================================================================
# Test Class: Watch Rule Matching (_check_watch_rules internals)
# ===================================================================


class TestWatchRuleMatching:
    """Tests 42-46: rule matching with various filters."""

    async def test_matches_with_category_filter(self, wired_skill):
        """Test 42: Rule with category filter only matches the specified category."""
        wired_skill._watch_rules["cat_rule"] = {
            "rule_name": "cat_rule",
            "event_type": "new_listing",
            "category": "ai",
            "created_at": _now_iso(),
        }

        # Listing NOT in 'ai' category
        listing_util = _make_listing("fn_u", "f_u", category="utility")
        await wired_skill._check_watch_rules("new_listing", "fn_u", listing_util, {})
        assert wired_skill._stats["watch_matches"] == 0

        # Listing in 'ai' category
        listing_ai = _make_listing("fn_a", "f_a", category="ai")
        await wired_skill._check_watch_rules("new_listing", "fn_a", listing_ai, {})
        assert wired_skill._stats["watch_matches"] == 1

    async def test_matches_with_agent_filter(self, wired_skill):
        """Test 43: Rule with agent_name filter only matches that agent."""
        wired_skill._watch_rules["agent_rule"] = {
            "rule_name": "agent_rule",
            "event_type": "new_listing",
            "agent_name": "bob",
            "created_at": _now_iso(),
        }

        listing_alice = _make_listing("fn_1", "f1", agent_name="alice")
        await wired_skill._check_watch_rules("new_listing", "fn_1", listing_alice, {})
        assert wired_skill._stats["watch_matches"] == 0

        listing_bob = _make_listing("fn_2", "f2", agent_name="bob")
        await wired_skill._check_watch_rules("new_listing", "fn_2", listing_bob, {})
        assert wired_skill._stats["watch_matches"] == 1

    async def test_matches_with_rating_filter(self, wired_skill):
        """Test 44: Rule with min_rating filter only matches above threshold."""
        wired_skill._watch_rules["rating_rule"] = {
            "rule_name": "rating_rule",
            "event_type": "new_listing",
            "min_rating": 4.0,
            "created_at": _now_iso(),
        }

        low_rated = _make_listing("fn_1", "f1", avg_rating=2.0)
        await wired_skill._check_watch_rules("new_listing", "fn_1", low_rated, {})
        assert wired_skill._stats["watch_matches"] == 0

        high_rated = _make_listing("fn_2", "f2", avg_rating=4.5)
        await wired_skill._check_watch_rules("new_listing", "fn_2", high_rated, {})
        assert wired_skill._stats["watch_matches"] == 1

    async def test_matches_with_tag_filter(self, wired_skill):
        """Test 45: Rule with tags filter matches when at least one tag overlaps."""
        wired_skill._watch_rules["tag_rule"] = {
            "rule_name": "tag_rule",
            "event_type": "new_listing",
            "tags": ["ml", "deep-learning"],
            "created_at": _now_iso(),
        }

        no_overlap = _make_listing("fn_1", "f1", tags=["web", "api"])
        await wired_skill._check_watch_rules("new_listing", "fn_1", no_overlap, {})
        assert wired_skill._stats["watch_matches"] == 0

        overlap = _make_listing("fn_2", "f2", tags=["ml", "nlp"])
        await wired_skill._check_watch_rules("new_listing", "fn_2", overlap, {})
        assert wired_skill._stats["watch_matches"] == 1

    async def test_matches_with_event_type_filter(self, wired_skill):
        """Test 46: Rule only matches its declared event_type."""
        wired_skill._watch_rules["import_rule"] = {
            "rule_name": "import_rule",
            "event_type": "new_import",
            "created_at": _now_iso(),
        }

        listing = _make_listing("fn_1", "f1")

        # Wrong event type
        await wired_skill._check_watch_rules("new_listing", "fn_1", listing, {})
        assert wired_skill._stats["watch_matches"] == 0

        # Correct event type
        await wired_skill._check_watch_rules("new_import", "fn_1", listing, {})
        assert wired_skill._stats["watch_matches"] == 1


# ===================================================================
# Test Class: Event Emission (_emit_event internals)
# ===================================================================


class TestEventEmission:
    """Tests 47-49: emit_event calls context, logs events, updates stats."""

    async def test_emit_event_calls_context_correctly(self, wired_skill):
        """Test 47: _emit_event calls context.call_skill('event', 'publish', ...)."""
        wired_skill.context = _make_context()

        await wired_skill._emit_event("marketplace.new_listing", {"listing_id": "fn_1"})

        event_calls = [
            c for c in wired_skill.context.call_skill.call_args_list
            if c.args[0] == "event" and c.args[1] == "publish"
        ]
        assert len(event_calls) == 1
        payload = event_calls[0].args[2]
        assert payload["event"] == "marketplace.new_listing"
        assert payload["data"]["listing_id"] == "fn_1"
        assert payload["data"]["source"] == "function_marketplace_discovery_events"

    async def test_emit_event_logs_events(self, wired_skill):
        """Test 48: _emit_event appends to _event_log."""
        wired_skill.context = _make_context()
        before = len(wired_skill._event_log)

        await wired_skill._emit_event("marketplace.new_listing", {"listing_id": "fn_1"})
        assert len(wired_skill._event_log) == before + 1
        assert wired_skill._event_log[-1]["topic"] == "marketplace.new_listing"

    async def test_emit_event_updates_stats(self, wired_skill):
        """Test 49: _emit_event increments events_emitted counter."""
        wired_skill.context = _make_context()
        before = wired_skill._stats["events_emitted"]

        await wired_skill._emit_event("marketplace.new_import", {"listing_id": "fn_1"})
        assert wired_skill._stats["events_emitted"] == before + 1

    async def test_emit_event_trims_log_to_max(self, wired_skill):
        """Event log is trimmed to MAX_EVENT_LOG entries."""
        wired_skill.context = _make_context()
        wired_skill._event_log = [
            {"topic": "t", "data": {}, "timestamp": "t"} for _ in range(MAX_EVENT_LOG)
        ]

        await wired_skill._emit_event("marketplace.new_listing", {"lid": "fn_1"})
        assert len(wired_skill._event_log) <= MAX_EVENT_LOG

    async def test_emit_event_without_context(self, wired_skill):
        """_emit_event still logs locally even without a context."""
        wired_skill.context = None
        result = await wired_skill._emit_event("marketplace.test", {"x": 1})
        assert result is False
        assert len(wired_skill._event_log) == 1
        assert wired_skill._stats["events_emitted"] == 1


# ===================================================================
# Test Class: Manifest
# ===================================================================


class TestManifest:
    """Tests 50-52: manifest fields and actions."""

    def test_manifest_has_correct_identity(self, skill):
        """Test 50: Manifest skill_id, name, version are correct."""
        m = skill.manifest
        assert m.skill_id == "function_marketplace_discovery_events"
        assert m.name == "Function Marketplace Discovery Events"
        assert m.version == "1.0.0"
        assert m.category == "replication"
        assert m.author == "singularity"

    def test_manifest_all_eight_actions_present(self, skill):
        """Test 51: All 8 actions exist with correct names."""
        action_names = [a.name for a in skill.manifest.actions]
        expected = [
            "wire", "unwire", "check", "watch",
            "unwatch", "trending", "configure", "status",
        ]
        assert len(action_names) == 8
        for name in expected:
            assert name in action_names, f"Missing action: {name}"

    def test_action_parameters_correct(self, skill):
        """Test 52: Key actions have expected parameter definitions."""
        actions = {a.name: a for a in skill.manifest.actions}

        # wire has poll_interval_reflections, watch_categories, watch_agents
        wire_params = actions["wire"].parameters
        assert "poll_interval_reflections" in wire_params
        assert "watch_categories" in wire_params
        assert "watch_agents" in wire_params

        # watch has rule_name (required), event_type (required), plus optional filters
        watch_params = actions["watch"].parameters
        assert watch_params["rule_name"]["required"] is True
        assert watch_params["event_type"]["required"] is True
        assert "category" in watch_params
        assert "min_rating" in watch_params
        assert "agent_name" in watch_params
        assert "tags" in watch_params

        # unwatch has rule_name
        assert "rule_name" in actions["unwatch"].parameters

        # trending has window, min_imports
        trending_params = actions["trending"].parameters
        assert "window" in trending_params
        assert "min_imports" in trending_params

        # configure has all config keys
        config_params = actions["configure"].parameters
        assert "poll_interval_reflections" in config_params
        assert "emit_on_new_listing" in config_params
        assert "emit_on_import" in config_params
        assert "trending_threshold" in config_params

        # unwire and status have empty parameters
        assert actions["unwire"].parameters == {}
        assert actions["status"].parameters == {}

    def test_all_actions_zero_cost(self, skill):
        """All actions in this skill have zero estimated cost."""
        for action in skill.manifest.actions:
            assert action.estimated_cost == 0.0


# ===================================================================
# Test Class: Execute dispatcher
# ===================================================================


class TestExecuteDispatcher:
    """Edge cases for the execute dispatcher."""

    async def test_unknown_action_returns_failure(self, skill):
        """Unknown action name returns a failure result."""
        result = await skill.execute("nonexistent", {})
        assert not result.success
        assert "unknown action" in result.message.lower()

    async def test_execute_catches_exceptions(self, skill):
        """If a handler raises, execute catches it and returns failure."""
        # Force an exception by passing params that cause an error in wire
        # e.g., poll_interval_reflections that cannot be int()
        with patch.object(
            skill, "_action_wire", side_effect=RuntimeError("boom")
        ):
            result = await skill.execute("wire", {})
            assert not result.success
            assert "boom" in result.message


# ===================================================================
# Test Class: Detect trending (internal helper)
# ===================================================================


class TestDetectTrending:
    """Tests for _detect_trending internal method."""

    def test_detect_trending_above_threshold(self, skill):
        """Listings with acceleration >= threshold appear in results."""
        old = {"fn_1": {"import_count": 3, "listing_id": "fn_1"}}
        current = _listings_map(
            _make_listing("fn_1", "func_a", import_count=10)
        )
        skill._config["trending_threshold"] = 5
        trending = skill._detect_trending(old, current)
        assert len(trending) == 1
        assert trending[0]["acceleration"] == 7

    def test_detect_trending_below_threshold(self, skill):
        """Listings with acceleration below threshold are excluded."""
        old = {"fn_1": {"import_count": 8, "listing_id": "fn_1"}}
        current = _listings_map(
            _make_listing("fn_1", "func_a", import_count=9)
        )
        skill._config["trending_threshold"] = 5
        trending = skill._detect_trending(old, current)
        assert len(trending) == 0

    def test_detect_trending_new_listing_ignored(self, skill):
        """New listings (not in old snapshot) are not considered trending."""
        old = {}
        current = _listings_map(
            _make_listing("fn_new", "new_func", import_count=100)
        )
        trending = skill._detect_trending(old, current)
        assert len(trending) == 0


# ===================================================================
# Test Class: Passes filters (internal helper)
# ===================================================================


class TestPassesFilters:
    """Tests for _passes_filters internal method."""

    def test_passes_when_no_filters_set(self, skill):
        """All listings pass when no filters configured."""
        listing = _make_listing("fn_1", "f1", category="anything", agent_name="anyone")
        assert skill._passes_filters(listing) is True

    def test_blocked_by_category_filter(self, skill):
        """Listing not in watch_categories is blocked."""
        skill._config["watch_categories"] = ["ai", "ml"]
        listing = _make_listing("fn_1", "f1", category="utility")
        assert skill._passes_filters(listing) is False

    def test_passed_by_category_filter(self, skill):
        """Listing in watch_categories passes."""
        skill._config["watch_categories"] = ["ai", "ml"]
        listing = _make_listing("fn_1", "f1", category="ai")
        assert skill._passes_filters(listing) is True

    def test_blocked_by_agent_filter(self, skill):
        """Listing not from a watched agent is blocked."""
        skill._config["watch_agents"] = ["bob"]
        listing = _make_listing("fn_1", "f1", agent_name="alice")
        assert skill._passes_filters(listing) is False

    def test_passed_by_agent_filter(self, skill):
        """Listing from a watched agent passes."""
        skill._config["watch_agents"] = ["bob"]
        listing = _make_listing("fn_1", "f1", agent_name="bob")
        assert skill._passes_filters(listing) is True

    def test_both_filters_applied(self, skill):
        """Both category and agent filters must pass."""
        skill._config["watch_categories"] = ["ai"]
        skill._config["watch_agents"] = ["bob"]

        # Wrong category
        assert skill._passes_filters(
            _make_listing("fn_1", "f1", category="utility", agent_name="bob")
        ) is False
        # Wrong agent
        assert skill._passes_filters(
            _make_listing("fn_2", "f2", category="ai", agent_name="alice")
        ) is False
        # Both match
        assert skill._passes_filters(
            _make_listing("fn_3", "f3", category="ai", agent_name="bob")
        ) is True


# ===================================================================
# Test Class: Take snapshot (internal helper)
# ===================================================================


class TestTakeSnapshot:
    """Tests for _take_snapshot internal method."""

    async def test_take_snapshot_captures_listing_state(self, skill):
        """Snapshot records listing data correctly."""
        listings = [
            _make_listing("fn_1", "func_a", import_count=5),
            _make_listing("fn_2", "func_b", import_count=3),
        ]
        skill.context = _make_context(listings)

        result = await skill._take_snapshot()
        assert result is True
        assert len(skill._snapshot["listings"]) == 2
        assert skill._snapshot["imports_count"] == 8
        assert skill._snapshot["taken_at"] is not None

    async def test_take_snapshot_without_context_gives_empty(self, skill):
        """Without context, snapshot is initialised empty."""
        skill.context = None
        result = await skill._take_snapshot()
        assert result is False
        assert skill._snapshot["listings"] == {}
        assert skill._snapshot["imports_count"] == 0


# ===================================================================
# Test Class: Estimate cost
# ===================================================================


class TestEstimateCost:
    """Test the estimate_cost method."""

    def test_estimate_cost_always_zero(self, skill):
        """All actions return zero cost."""
        for action_name in ["wire", "unwire", "check", "watch", "unwatch",
                            "trending", "configure", "status"]:
            assert skill.estimate_cost(action_name, {}) == 0.0


# ===================================================================
# Test Class: Check with trending during check
# ===================================================================


class TestCheckTrendingIntegration:
    """Test that the check action also detects trending as part of its run."""

    async def test_check_detects_trending_inline(self, wired_skill):
        """During check, trending functions are detected and emitted."""
        old = _make_listing("fn_1", "func_a", import_count=0)
        wired_skill._snapshot["listings"] = {
            "fn_1": FunctionMarketplaceDiscoveryEventsSkill._listing_to_snapshot_entry(old)
        }
        wired_skill._config["trending_threshold"] = 2

        current = _make_listing("fn_1", "func_a", import_count=10)
        wired_skill.context = _make_context([current])

        result = await wired_skill.execute("check", {})
        assert len(result.data["changes"]["trending"]) >= 1

    async def test_check_no_trending_when_emit_off(self, wired_skill):
        """When emit_on_trending is False, check skips trending detection."""
        old = _make_listing("fn_1", "func_a", import_count=0)
        wired_skill._snapshot["listings"] = {
            "fn_1": FunctionMarketplaceDiscoveryEventsSkill._listing_to_snapshot_entry(old)
        }
        wired_skill._config["emit_on_trending"] = False

        current = _make_listing("fn_1", "func_a", import_count=100)
        wired_skill.context = _make_context([current])

        result = await wired_skill.execute("check", {})
        assert len(result.data["changes"]["trending"]) == 0

    async def test_check_status_change_detected(self, wired_skill):
        """Check detects status changes on existing listings."""
        old = _make_listing("fn_1", "func_a", status="active")
        wired_skill._snapshot["listings"] = {
            "fn_1": FunctionMarketplaceDiscoveryEventsSkill._listing_to_snapshot_entry(old)
        }

        updated = _make_listing("fn_1", "func_a", status="unpublished")
        wired_skill.context = _make_context([updated])

        result = await wired_skill.execute("check", {})
        assert len(result.data["changes"]["status_changes"]) == 1
        change = result.data["changes"]["status_changes"][0]
        assert change["old_status"] == "active"
        assert change["new_status"] == "unpublished"
