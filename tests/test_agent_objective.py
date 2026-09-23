"""Tests for agent objective and project context support."""
import pytest
import os

os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")

from singularity.autonomous_agent import AutonomousAgent


def test_objective_from_param():
    agent = AutonomousAgent(objective="Build a website", llm_provider="none")
    assert agent.objective == "Build a website"


def test_objective_from_env(monkeypatch):
    monkeypatch.setenv("AGENT_OBJECTIVE", "Review code")
    agent = AutonomousAgent(llm_provider="none")
    assert agent.objective == "Review code"


def test_objective_param_overrides_env(monkeypatch):
    monkeypatch.setenv("AGENT_OBJECTIVE", "from env")
    agent = AutonomousAgent(objective="from param", llm_provider="none")
    assert agent.objective == "from param"


def test_project_context_inline():
    agent = AutonomousAgent(project_context="We use Python 3.12", llm_provider="none")
    ctx = agent._build_project_context()
    assert "We use Python 3.12" in ctx


def test_build_context_combined():
    agent = AutonomousAgent(
        objective="Fix the bug",
        project_context="Django app",
        llm_provider="none",
    )
    ctx = agent._build_project_context()
    assert "CURRENT OBJECTIVE" in ctx
    assert "Fix the bug" in ctx
    assert "PROJECT CONTEXT" in ctx
    assert "Django app" in ctx


def test_set_objective_runtime():
    agent = AutonomousAgent(llm_provider="none")
    assert agent.objective == ""
    agent.set_objective("New mission")
    assert agent.objective == "New mission"
    ctx = agent._build_project_context()
    assert "New mission" in ctx


def test_empty_context_when_no_objective():
    agent = AutonomousAgent(llm_provider="none")
    ctx = agent._build_project_context()
    assert ctx == ""
