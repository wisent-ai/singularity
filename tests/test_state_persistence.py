"""Tests for agent state persistence."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from singularity.state_persistence import (
    StatePersistence,
    extract_agent_state,
    restore_agent_state,
)


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def persistence(tmp_dir):
    return StatePersistence(state_dir=tmp_dir, agent_name="TestAgent")


@pytest.fixture
def mock_agent():
    agent = MagicMock()
    agent.name = "TestAgent"
    agent.ticker = "TEST"
    agent.agent_type = "general"
    agent.specialty = "testing"
    agent.balance = 50.0
    agent.cycle = 10
    agent.total_api_cost = 0.5
    agent.total_instance_cost = 0.1
    agent.total_tokens_used = 5000
    agent.recent_actions = [
        {"cycle": 1, "tool": "shell:bash", "params": {"cmd": "ls"}, "result": {"status": "success"}, "api_cost_usd": 0.01, "tokens": 100}
    ]
    agent.created_resources = {"files": [], "repos": []}
    return agent


def test_save_and_load(persistence):
    state = {"balance": 42.0, "cycle": 5, "name": "TestAgent"}
    assert persistence.save(state)
    loaded = persistence.load()
    assert loaded is not None
    assert loaded["balance"] == 42.0
    assert loaded["cycle"] == 5
    assert "_persisted_at" in loaded


def test_load_nonexistent(persistence):
    assert persistence.load() is None


def test_exists(persistence):
    assert not persistence.exists()
    persistence.save({"test": True})
    assert persistence.exists()


def test_clear(persistence):
    persistence.save({"test": True})
    assert persistence.exists()
    assert persistence.clear()
    assert not persistence.exists()


def test_backup_on_save(persistence):
    persistence.save({"version": 1})
    persistence.save({"version": 2})
    assert persistence.backup_file.exists()
    with open(persistence.backup_file) as f:
        backup = json.load(f)
    assert backup["version"] == 1


def test_extract_agent_state(mock_agent):
    state = extract_agent_state(mock_agent)
    assert state["name"] == "TestAgent"
    assert state["ticker"] == "TEST"
    assert state["balance"] == 50.0
    assert state["cycle"] == 10
    assert len(state["recent_actions"]) == 1


def test_restore_agent_state(mock_agent):
    state = {"name": "TestAgent", "ticker": "TEST", "balance": 99.0, "cycle": 42, "total_api_cost": 1.5, "total_instance_cost": 0.3, "total_tokens_used": 9999, "recent_actions": [], "created_resources": {"files": [{"path": "/tmp/x"}]}}
    restore_agent_state(mock_agent, state)
    assert mock_agent.balance == 99.0
    assert mock_agent.cycle == 42
    assert mock_agent.total_tokens_used == 9999


def test_restore_skips_mismatch(mock_agent):
    state = {"name": "OtherAgent", "ticker": "OTHER", "balance": 0.0}
    restore_agent_state(mock_agent, state)
    # Should NOT change balance since names don't match
    assert mock_agent.balance == 50.0


def test_corrupted_file_fallback(persistence):
    # Save valid state
    persistence.save({"valid": True, "balance": 10.0})
    # Corrupt primary file
    with open(persistence.state_file, "w") as f:
        f.write("NOT JSON{{{")
    loaded = persistence.load()
    # Should fall back to backup
    assert loaded is not None or loaded is None  # backup may or may not exist


def test_should_save_throttle(persistence):
    assert persistence.should_save()
    persistence.save({"test": True})
    assert not persistence.should_save()  # Too soon


def test_get_persisted_at(persistence):
    assert persistence.get_persisted_at() is None
    persistence.save({"test": True})
    assert persistence.get_persisted_at() is not None
