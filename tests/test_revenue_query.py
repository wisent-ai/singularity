#!/usr/bin/env python3
"""Tests for RevenueQuerySkill - natural language revenue queries."""

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path

from singularity.skills.revenue_query import (
    RevenueQuerySkill,
    _tokenize,
    _match_score,
    INTENT_PATTERNS,
    INTENT_TO_ACTION,
    INTENT_DESCRIPTIONS,
)


@pytest.fixture
def skill(tmp_path):
    """Create a RevenueQuerySkill with temp data dir."""
    with patch("singularity.skills.revenue_query.DATA_DIR", tmp_path), \
         patch("singularity.skills.revenue_query.QUERY_STATE_FILE", tmp_path / "revenue_query.json"):
        s = RevenueQuerySkill()
        yield s


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


class TestTokenize:
    def test_basic(self):
        assert _tokenize("What is total revenue?") == ["what", "is", "total", "revenue"]

    def test_mixed_case(self):
        assert _tokenize("Revenue BY Source") == ["revenue", "by", "source"]

    def test_empty(self):
        assert _tokenize("") == []


class TestMatchScore:
    def test_full_match(self):
        assert _match_score(["total", "revenue"], ["total", "revenue"]) == 1.0

    def test_partial_match(self):
        assert _match_score(["total", "revenue", "today"], ["total", "revenue"]) == 1.0

    def test_no_match(self):
        assert _match_score(["hello", "world"], ["total", "revenue"]) == 0.0

    def test_empty_pattern(self):
        assert _match_score(["total"], []) == 0.0


class TestClassifyIntent:
    def test_overview(self, skill):
        skill._ensure_state()
        intent, score, _ = skill._classify_intent("What is total revenue?")
        assert intent == "overview"
        assert score >= 0.5

    def test_by_source(self, skill):
        skill._ensure_state()
        intent, score, _ = skill._classify_intent("Revenue breakdown by source")
        assert intent == "by_source"
        assert score >= 0.5

    def test_profitability(self, skill):
        skill._ensure_state()
        intent, score, _ = skill._classify_intent("Am I profitable?")
        assert intent == "profitability"

    def test_forecast(self, skill):
        skill._ensure_state()
        intent, score, _ = skill._classify_intent("Revenue forecast for next month")
        assert intent == "forecast"

    def test_customers(self, skill):
        skill._ensure_state()
        intent, score, _ = skill._classify_intent("Who are my customers?")
        assert intent == "customers"

    def test_trends(self, skill):
        skill._ensure_state()
        intent, score, _ = skill._classify_intent("Revenue trend over time")
        assert intent == "trends"

    def test_recommendations(self, skill):
        skill._ensure_state()
        intent, score, _ = skill._classify_intent("How to improve revenue?")
        assert intent == "recommendations"

    def test_override(self, skill):
        skill._ensure_state()
        skill._state["intent_overrides"]["hello world"] = "profitability"
        intent, score, _ = skill._classify_intent("hello world")
        assert intent == "profitability"
        assert score == 1.0


class TestAskAction:
    def test_empty_query(self, skill):
        result = run(skill.execute("ask", {"query": ""}))
        assert not result.success

    def test_ask_no_context(self, skill):
        result = run(skill.execute("ask", {"query": "What is total revenue?"}))
        assert result.success
        assert "overview" in result.message.lower() or "revenue" in result.message.lower()
        assert result.data.get("executed") is False

    def test_ambiguous_query(self, skill):
        result = run(skill.execute("ask", {"query": "xyz123abc"}))
        assert result.success
        assert result.data.get("ambiguous") is True or result.data.get("confidence", 1) < 0.5

    def test_stats_updated(self, skill):
        run(skill.execute("ask", {"query": "What is total revenue?"}))
        result = run(skill.execute("stats", {}))
        assert result.data["stats"]["total_queries"] >= 1


class TestClassifyAction:
    def test_classify(self, skill):
        result = run(skill.execute("classify", {"query": "Revenue by source"}))
        assert result.success
        assert result.data["intent"] == "by_source"

    def test_classify_empty(self, skill):
        result = run(skill.execute("classify", {"query": ""}))
        assert not result.success


class TestCorrectAction:
    def test_correct(self, skill):
        result = run(skill.execute("correct", {"query": "show me money", "intent": "profitability"}))
        assert result.success
        # Verify the correction is applied
        intent, score, _ = skill._classify_intent("show me money")
        assert intent == "profitability"
        assert score == 1.0

    def test_invalid_intent(self, skill):
        result = run(skill.execute("correct", {"query": "test", "intent": "invalid"}))
        assert not result.success


class TestExamplesAction:
    def test_examples(self, skill):
        result = run(skill.execute("examples", {}))
        assert result.success
        assert "overview" in result.data["examples"]
        assert len(result.data["examples"]) == len(INTENT_PATTERNS)


class TestHistoryAction:
    def test_empty_history(self, skill):
        result = run(skill.execute("history", {}))
        assert result.success
        assert result.data["history"] == []

    def test_history_after_queries(self, skill):
        run(skill.execute("ask", {"query": "Total revenue"}))
        result = run(skill.execute("history", {"limit": 5}))
        assert result.success
        assert len(result.data["history"]) >= 1


class TestIntentCoverage:
    def test_all_intents_have_actions(self):
        for intent in INTENT_PATTERNS:
            assert intent in INTENT_TO_ACTION, f"Missing action mapping for {intent}"

    def test_all_intents_have_descriptions(self):
        for intent in INTENT_PATTERNS:
            assert intent in INTENT_DESCRIPTIONS, f"Missing description for {intent}"


class TestManifest:
    def test_manifest(self, skill):
        m = skill.manifest
        assert m.skill_id == "revenue_query"
        assert len(m.actions) == 6

    def test_unknown_action(self, skill):
        result = run(skill.execute("nonexistent", {}))
        assert not result.success
