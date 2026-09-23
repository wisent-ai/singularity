"""Tests for AgentJournal - persistent cross-session memory."""

import json
import tempfile
from pathlib import Path

import pytest

from singularity.journal import AgentJournal


@pytest.fixture
def journal(tmp_path):
    """Create a journal with a temp file."""
    return AgentJournal(path=tmp_path / "test_journal.jsonl")


def test_start_session(journal):
    sid = journal.start_session("TestAgent", "TEST", "coder")
    assert sid.startswith("TEST-")
    assert journal._session_id == sid


def test_record_action(journal):
    journal.start_session("TestAgent", "TEST")
    journal.record_action("github:create_repo", {"name": "test"}, "success", "Created")
    assert len(journal._session_actions) == 1
    assert journal._session_actions[0]["status"] == "success"


def test_record_insight(journal):
    journal.start_session("TestAgent", "TEST")
    journal.record_insight("Shell commands are fastest for file ops")
    assert len(journal._session_insights) == 1


def test_record_goal(journal):
    journal.start_session("TestAgent", "TEST")
    journal.record_goal("Build a REST API", priority="high")
    assert len(journal._session_goals) == 1


def test_end_session(journal):
    journal.start_session("TestAgent", "TEST")
    journal.record_action("fs:write", {}, "success", "wrote file")
    journal.record_action("shell:bash", {}, "error", "command failed")
    journal.record_insight("Always check exit codes")
    journal.record_goal("Add error handling")
    journal.end_session(cycles=10, total_cost=0.05, balance_remaining=9.95)

    entries = journal._load_entries()
    end_entry = [e for e in entries if e["type"] == "session_end"][0]
    assert end_entry["cycles"] == 10
    assert end_entry["total_cost_usd"] == 0.05
    assert end_entry["actions_succeeded"] == 1
    assert end_entry["actions_failed"] == 1
    assert len(end_entry["insights"]) == 1
    assert len(end_entry["goals"]) == 1


def test_session_count(journal):
    assert journal.get_session_count() == 0
    journal.start_session("A", "A")
    journal.end_session(cycles=1)
    assert journal.get_session_count() == 1
    journal.start_session("B", "B")
    journal.end_session(cycles=2)
    assert journal.get_session_count() == 2


def test_get_context_summary_empty(journal):
    ctx = journal.get_context_summary()
    assert ctx == ""


def test_get_context_summary_with_data(journal):
    journal.start_session("Agent", "AGT")
    journal.record_action("fs:write", {}, "success", "ok")
    journal.record_insight("Writing files works well")
    journal.record_goal("Build more features")
    journal.end_session(cycles=5, total_cost=0.01)

    ctx = journal.get_context_summary()
    assert "AGENT JOURNAL" in ctx
    assert "AGT-" in ctx
    assert "Writing files works well" in ctx
    assert "Build more features" in ctx


def test_get_recent_goals(journal):
    journal.start_session("A", "A")
    journal.record_goal("Goal 1")
    journal.record_goal("Goal 2")
    journal.record_goal("Goal 3")
    goals = journal.get_recent_goals(2)
    assert len(goals) == 2
    assert goals == ["Goal 2", "Goal 3"]


def test_get_recent_insights(journal):
    journal.start_session("A", "A")
    journal.record_insight("Insight 1")
    journal.record_insight("Insight 2")
    insights = journal.get_recent_insights(1)
    assert len(insights) == 1
    assert insights == ["Insight 2"]


def test_clear(journal):
    journal.start_session("A", "A")
    journal.record_action("test", {}, "success")
    journal.clear()
    assert journal._load_entries() == []


def test_context_truncation(journal):
    journal.start_session("A", "A")
    for i in range(100):
        journal.record_insight(f"Insight number {i} " * 10)
    journal.end_session()
    ctx = journal.get_context_summary(max_chars=500)
    assert len(ctx) <= 500


def test_multiple_sessions_context(journal):
    for i in range(3):
        journal.start_session(f"Agent{i}", f"A{i}")
        journal.record_action(f"tool{i}", {}, "success")
        journal.record_insight(f"Session {i} insight")
        journal.end_session(cycles=i + 1, total_cost=0.01 * (i + 1))

    ctx = journal.get_context_summary()
    assert "3 of 3 total" in ctx
    assert "Session 0 insight" in ctx
    assert "Session 2 insight" in ctx


def test_jsonl_persistence(journal):
    journal.start_session("A", "A")
    journal.record_action("test", {}, "success", "ok")
    journal.end_session()

    # Create a new journal pointing to same file
    journal2 = AgentJournal(path=journal.path)
    assert journal2.get_session_count() == 1
    ctx = journal2.get_context_summary()
    assert "AGENT JOURNAL" in ctx


def test_duplicate_insights_deduplicated(journal):
    journal.start_session("A", "A")
    journal.record_insight("Same insight")
    journal.record_insight("Same insight")
    journal.record_insight("Different insight")
    journal.end_session()

    ctx = journal.get_context_summary()
    # Count occurrences in the Key Insights section
    insights_section = ctx.split("Key Insights")[1] if "Key Insights" in ctx else ""
    assert insights_section.count("Same insight") == 1


def test_top_tools_tracking(journal):
    journal.start_session("A", "A")
    for _ in range(5):
        journal.record_action("fs:write", {}, "success")
    for _ in range(3):
        journal.record_action("shell:bash", {}, "success")
    journal.record_action("github:push", {}, "success")
    journal.end_session()

    entries = journal._load_entries()
    end = [e for e in entries if e["type"] == "session_end"][0]
    assert end["top_tools"][0] == ["fs:write", 5]
    assert end["top_tools"][1] == ["shell:bash", 3]
