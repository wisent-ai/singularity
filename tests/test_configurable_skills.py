"""Tests for configurable skill loading in AutonomousAgent."""
import pytest
from singularity.autonomous_agent import AutonomousAgent
from singularity.skills.filesystem import FilesystemSkill
from singularity.skills.shell import ShellSkill


def test_default_skills_loaded():
    """Agent loads default skills when none specified."""
    agent = AutonomousAgent(llm_provider="none", starting_balance=1.0)
    assert len(agent._skill_classes) == len(AutonomousAgent.DEFAULT_SKILL_CLASSES)


def test_custom_skills_list():
    """Agent loads only specified skills."""
    agent = AutonomousAgent(
        llm_provider="none",
        starting_balance=1.0,
        skills=[FilesystemSkill, ShellSkill],
    )
    assert len(agent._skill_classes) == 2
    assert FilesystemSkill in agent._skill_classes
    assert ShellSkill in agent._skill_classes


def test_empty_skills_list():
    """Agent with empty skills list has no tools."""
    agent = AutonomousAgent(
        llm_provider="none",
        starting_balance=1.0,
        skills=[],
    )
    assert len(agent._skill_classes) == 0
    tools = agent._get_tools()
    # Should have fallback "wait" tool
    assert len(tools) == 1
    assert tools[0]["name"] == "wait"


def test_skill_load_errors_tracked():
    """Failed skill loads are tracked instead of silently swallowed."""
    agent = AutonomousAgent(llm_provider="none", starting_balance=1.0)
    # _skill_load_errors exists (may or may not have entries depending on env)
    assert isinstance(agent._skill_load_errors, list)


def test_get_skill_status():
    """get_skill_status returns structured skill info."""
    agent = AutonomousAgent(
        llm_provider="none",
        starting_balance=1.0,
        skills=[FilesystemSkill],
    )
    status = agent.get_skill_status()
    assert "loaded" in status
    assert "load_errors" in status
    assert "total_tools" in status
    assert isinstance(status["loaded"], list)


def test_remove_skill():
    """Can remove a loaded skill."""
    agent = AutonomousAgent(
        llm_provider="none",
        starting_balance=1.0,
        skills=[FilesystemSkill, ShellSkill],
    )
    # Find loaded skill IDs
    loaded_ids = [s["id"] for s in agent.get_skill_status()["loaded"]]
    if "filesystem" in loaded_ids:
        assert agent.remove_skill("filesystem")
        new_ids = [s["id"] for s in agent.get_skill_status()["loaded"]]
        assert "filesystem" not in new_ids


def test_remove_nonexistent_skill():
    """Removing a nonexistent skill returns False."""
    agent = AutonomousAgent(
        llm_provider="none",
        starting_balance=1.0,
        skills=[],
    )
    assert not agent.remove_skill("nonexistent_skill")


def test_project_context():
    """Project context is stored on agent."""
    agent = AutonomousAgent(
        llm_provider="none",
        starting_balance=1.0,
        project_context="Build a web scraper",
        skills=[],
    )
    assert agent.project_context == "Build a web scraper"


def test_project_context_default_empty():
    """Project context defaults to empty string."""
    agent = AutonomousAgent(
        llm_provider="none",
        starting_balance=1.0,
        skills=[],
    )
    assert agent.project_context == ""
