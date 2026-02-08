"""Tests for RuleConflictDetectionSkill - detect and resolve contradictions in learned rules."""

import json
import pytest
import asyncio
from pathlib import Path
from unittest.mock import patch
from singularity.skills.rule_conflict_detection import RuleConflictDetectionSkill, CONFLICT_DATA_FILE, RULES_FILE


@pytest.fixture
def tmp_env(tmp_path):
    """Patch data files to use tmp_path."""
    conflict_file = tmp_path / "rule_conflicts.json"
    rules_file = tmp_path / "learning_rules.json"
    with patch("singularity.skills.rule_conflict_detection.CONFLICT_DATA_FILE", conflict_file), \
         patch("singularity.skills.rule_conflict_detection.RULES_FILE", rules_file), \
         patch("singularity.skills.rule_conflict_detection.DATA_DIR", tmp_path):
        yield tmp_path, conflict_file, rules_file


@pytest.fixture
def skill(tmp_env):
    tmp_path, conflict_file, rules_file = tmp_env
    s = RuleConflictDetectionSkill()
    return s


def _write_rules(rules_file, rules):
    """Helper to write rules to the rules file."""
    data = {"rules": rules, "stats": {}, "config": {}, "distillation_history": []}
    rules_file.write_text(json.dumps(data))


def _make_rule(id, text, category="general", confidence=0.7, skill_id="", reinforcement_count=0, created_at="2025-01-01T00:00:00"):
    return {
        "id": id,
        "rule_text": text,
        "category": category,
        "confidence": confidence,
        "skill_id": skill_id,
        "reinforcement_count": reinforcement_count,
        "created_at": created_at,
        "last_reinforced": created_at,
        "source": "test",
    }


def test_scan_empty_rules(skill):
    result = asyncio.get_event_loop().run_until_complete(skill.execute("scan", {}))
    assert result.success
    assert result.data["rules_scanned"] == 0


def test_scan_no_conflicts(skill, tmp_env):
    _, _, rules_file = tmp_env
    rules = [
        _make_rule("r1", "Use Docker for deployment", skill_id="docker"),
        _make_rule("r2", "Python is good for scripting", skill_id="shell"),
    ]
    _write_rules(rules_file, rules)
    result = asyncio.get_event_loop().run_until_complete(skill.execute("scan", {}))
    assert result.success
    assert result.data["conflicts_found"] == 0


def test_scan_finds_sentiment_conflict(skill, tmp_env):
    _, _, rules_file = tmp_env
    rules = [
        _make_rule("r1", "Prefer Docker for reliable deployment use Docker", confidence=0.8),
        _make_rule("r2", "Avoid Docker for deployment it fails and is unreliable", confidence=0.4),
    ]
    _write_rules(rules_file, rules)
    result = asyncio.get_event_loop().run_until_complete(skill.execute("scan", {}))
    assert result.success
    assert result.data["conflicts_found"] >= 1
    conflict = result.data["conflicts"][0]
    assert "sentiment_opposition" in conflict["conflict_types"]


def test_scan_finds_category_conflict(skill, tmp_env):
    _, _, rules_file = tmp_env
    rules = [
        _make_rule("r1", "Docker deployment is good and works fast", category="success_pattern", confidence=0.8),
        _make_rule("r2", "Docker deployment is bad and fails slow", category="failure_pattern", confidence=0.4),
    ]
    _write_rules(rules_file, rules)
    result = asyncio.get_event_loop().run_until_complete(skill.execute("scan", {}))
    assert result.success
    assert result.data["conflicts_found"] >= 1


def test_scan_finds_skill_conflict(skill, tmp_env):
    _, _, rules_file = tmp_env
    rules = [
        _make_rule("r1", "This skill is good and effective", skill_id="vercel", confidence=0.9),
        _make_rule("r2", "This skill is bad and fails often", skill_id="vercel", confidence=0.3),
    ]
    _write_rules(rules_file, rules)
    result = asyncio.get_event_loop().run_until_complete(skill.execute("scan", {}))
    assert result.success
    assert result.data["conflicts_found"] >= 1
    assert "skill_opposition" in result.data["conflicts"][0]["conflict_types"]


def test_resolve_picks_higher_confidence(skill, tmp_env):
    _, conflict_file, rules_file = tmp_env
    rules = [
        _make_rule("r1", "Prefer Docker for reliable deployment use Docker", confidence=0.9),
        _make_rule("r2", "Avoid Docker for deployment it fails and is unreliable", confidence=0.3),
    ]
    _write_rules(rules_file, rules)
    # First scan
    scan_result = asyncio.get_event_loop().run_until_complete(skill.execute("scan", {}))
    assert scan_result.data["conflicts_found"] >= 1
    conflict_id = scan_result.data["conflicts"][0]["id"]
    # Resolve
    resolve_result = asyncio.get_event_loop().run_until_complete(
        skill.execute("resolve", {"conflict_id": conflict_id})
    )
    assert resolve_result.success
    assert resolve_result.data["winner_id"] == "r1"
    assert resolve_result.data["loser_id"] == "r2"
    # Check loser was weakened
    rules_data = json.loads(rules_file.read_text())
    loser = [r for r in rules_data["rules"] if r["id"] == "r2"][0]
    assert loser["confidence"] < 0.3


def test_resolve_manual_winner(skill, tmp_env):
    _, _, rules_file = tmp_env
    rules = [
        _make_rule("r1", "Prefer Docker for reliable deployment use Docker", confidence=0.9),
        _make_rule("r2", "Avoid Docker for deployment it fails and is unreliable", confidence=0.3),
    ]
    _write_rules(rules_file, rules)
    scan_result = asyncio.get_event_loop().run_until_complete(skill.execute("scan", {}))
    conflict_id = scan_result.data["conflicts"][0]["id"]
    # Override: manually pick the loser as winner
    resolve_result = asyncio.get_event_loop().run_until_complete(
        skill.execute("resolve", {"conflict_id": conflict_id, "winner": "r2"})
    )
    assert resolve_result.success
    assert resolve_result.data["winner_id"] == "r2"
    assert resolve_result.data["loser_id"] == "r1"


def test_resolve_retires_very_low_confidence(skill, tmp_env):
    _, _, rules_file = tmp_env
    rules = [
        _make_rule("r1", "Prefer Docker for reliable deployment use Docker", confidence=0.9),
        _make_rule("r2", "Avoid Docker for deployment it fails and is unreliable", confidence=0.05),
    ]
    _write_rules(rules_file, rules)
    scan_result = asyncio.get_event_loop().run_until_complete(skill.execute("scan", {}))
    conflict_id = scan_result.data["conflicts"][0]["id"]
    resolve_result = asyncio.get_event_loop().run_until_complete(
        skill.execute("resolve", {"conflict_id": conflict_id})
    )
    assert resolve_result.success
    assert resolve_result.data["loser_weakened"]["retired"]


def test_scan_and_resolve(skill, tmp_env):
    _, _, rules_file = tmp_env
    rules = [
        _make_rule("r1", "Prefer Docker for reliable deployment use Docker", confidence=0.85),
        _make_rule("r2", "Avoid Docker for deployment it fails and is unreliable", confidence=0.3),
    ]
    _write_rules(rules_file, rules)
    result = asyncio.get_event_loop().run_until_complete(skill.execute("scan_and_resolve", {}))
    assert result.success
    assert result.data["conflicts_found"] >= 1
    assert len(result.data["resolved"]) >= 1


def test_conflicts_list(skill, tmp_env):
    _, _, rules_file = tmp_env
    rules = [
        _make_rule("r1", "Prefer Docker for reliable deployment use Docker", confidence=0.85),
        _make_rule("r2", "Avoid Docker for deployment it fails and is unreliable", confidence=0.3),
    ]
    _write_rules(rules_file, rules)
    asyncio.get_event_loop().run_until_complete(skill.execute("scan", {}))
    result = asyncio.get_event_loop().run_until_complete(skill.execute("conflicts", {"status": "unresolved"}))
    assert result.success
    assert result.data["total"] >= 1


def test_status(skill):
    result = asyncio.get_event_loop().run_until_complete(skill.execute("status", {}))
    assert result.success
    assert "stats" in result.data
    assert "config" in result.data


def test_configure(skill):
    result = asyncio.get_event_loop().run_until_complete(
        skill.execute("configure", {"similarity_threshold": 0.5, "auto_resolve": False})
    )
    assert result.success
    assert result.data["config"]["similarity_threshold"] == 0.5
    assert result.data["config"]["auto_resolve"] is False


def test_unknown_action(skill):
    result = asyncio.get_event_loop().run_until_complete(skill.execute("invalid", {}))
    assert not result.success


def test_resolve_missing_conflict_id(skill):
    result = asyncio.get_event_loop().run_until_complete(skill.execute("resolve", {}))
    assert not result.success


def test_resolve_nonexistent_conflict(skill):
    result = asyncio.get_event_loop().run_until_complete(skill.execute("resolve", {"conflict_id": "nope"}))
    assert not result.success


def test_tiebreak_by_reinforcement(skill, tmp_env):
    _, _, rules_file = tmp_env
    rules = [
        _make_rule("r1", "Prefer Docker for reliable deployment use Docker", confidence=0.7, reinforcement_count=10),
        _make_rule("r2", "Avoid Docker for deployment it fails and is unreliable", confidence=0.65, reinforcement_count=1),
    ]
    _write_rules(rules_file, rules)
    result = asyncio.get_event_loop().run_until_complete(skill.execute("scan_and_resolve", {}))
    assert result.success
    if result.data["resolved"]:
        assert result.data["resolved"][0]["winner_id"] == "r1"


def test_skips_retired_rules(skill, tmp_env):
    _, _, rules_file = tmp_env
    rules = [
        _make_rule("r1", "Prefer Docker for reliable deployment use Docker", confidence=0.85),
        _make_rule("r2", "Avoid Docker for deployment it fails and is unreliable", confidence=0.3),
    ]
    rules[1]["status"] = "retired_by_conflict"
    _write_rules(rules_file, rules)
    result = asyncio.get_event_loop().run_until_complete(skill.execute("scan", {}))
    assert result.success
    assert result.data["conflicts_found"] == 0
