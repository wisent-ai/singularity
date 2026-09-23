"""Tests for goal-directed agent behavior."""
import json
import pytest
from unittest.mock import patch, MagicMock
from singularity.autonomous_agent import AutonomousAgent


@pytest.fixture
def agent(tmp_path):
    """Create an agent with goals stored in tmp dir."""
    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "", "OPENAI_API_KEY": ""}):
        a = AutonomousAgent(name="TestAgent", llm_provider="none", starting_balance=10.0)
        a._goals_file = tmp_path / "goals.json"
        return a


def test_add_goal(agent):
    goal = agent.add_goal("Build a website", priority="high")
    assert goal["id"] == 1
    assert goal["description"] == "Build a website"
    assert goal["priority"] == "high"
    assert goal["status"] == "active"
    assert len(agent.goals) == 1


def test_add_multiple_goals(agent):
    agent.add_goal("Goal 1", priority="low")
    agent.add_goal("Goal 2", priority="critical")
    assert len(agent.goals) == 2
    assert agent.goals[0]["id"] == 1
    assert agent.goals[1]["id"] == 2


def test_complete_goal(agent):
    agent.add_goal("Finish task")
    result = agent.complete_goal(1, note="Done!")
    assert result["status"] == "completed"
    assert result["completed_at"] is not None
    assert result["progress_notes"][0]["note"] == "Done!"


def test_complete_nonexistent_goal(agent):
    assert agent.complete_goal(999) is None


def test_update_goal_progress(agent):
    agent.add_goal("Long project")
    result = agent.update_goal_progress(1, "50% done")
    assert len(result["progress_notes"]) == 1
    agent.update_goal_progress(1, "75% done")
    assert len(agent.goals[0]["progress_notes"]) == 2


def test_get_active_goals(agent):
    agent.add_goal("Active one")
    agent.add_goal("Will complete")
    agent.complete_goal(2)
    active = agent.get_active_goals()
    assert len(active) == 1
    assert active[0]["description"] == "Active one"


def test_format_goals_context_empty(agent):
    assert agent._format_goals_context() == ""


def test_format_goals_context_with_goals(agent):
    agent.add_goal("Deploy API", priority="critical")
    agent.add_goal("Write docs", priority="low")
    ctx = agent._format_goals_context()
    assert "Active Goals" in ctx
    assert "[CRITICAL]" in ctx
    assert "[LOW]" in ctx
    assert "Deploy API" in ctx


def test_goals_persist_to_file(agent):
    agent.add_goal("Persistent goal")
    assert agent._goals_file.exists()
    with open(agent._goals_file) as f:
        data = json.load(f)
    assert len(data) == 1
    assert data[0]["description"] == "Persistent goal"


def test_goals_load_from_file(agent, tmp_path):
    goals = [{"id": 1, "description": "Loaded", "priority": "high",
              "status": "active", "created_at": "", "completed_at": None, "progress_notes": []}]
    goals_file = tmp_path / "goals.json"
    with open(goals_file, 'w') as f:
        json.dump(goals, f)
    agent._goals_file = goals_file
    loaded = agent._load_goals()
    assert len(loaded) == 1
    assert loaded[0]["description"] == "Loaded"


@pytest.mark.asyncio
async def test_execute_set_goal(agent):
    from singularity.cognition import Action
    action = Action(tool="agent:set_goal", params={"description": "Test goal", "priority": "high"})
    result = await agent._execute(action)
    assert result["status"] == "success"
    assert len(agent.goals) == 1


@pytest.mark.asyncio
async def test_execute_list_goals(agent):
    from singularity.cognition import Action
    agent.add_goal("Goal A")
    agent.add_goal("Goal B")
    agent.complete_goal(2)
    action = Action(tool="agent:list_goals", params={})
    result = await agent._execute(action)
    assert result["status"] == "success"
    assert result["data"]["total"] == 2
    assert result["data"]["completed"] == 1


@pytest.mark.asyncio
async def test_execute_complete_goal(agent):
    from singularity.cognition import Action
    agent.add_goal("To complete")
    action = Action(tool="agent:complete_goal", params={"goal_id": 1, "note": "Done"})
    result = await agent._execute(action)
    assert result["status"] == "success"
    assert agent.goals[0]["status"] == "completed"


@pytest.mark.asyncio
async def test_execute_update_goal(agent):
    from singularity.cognition import Action
    agent.add_goal("In progress")
    action = Action(tool="agent:update_goal", params={"goal_id": 1, "note": "Making progress"})
    result = await agent._execute(action)
    assert result["status"] == "success"


def test_get_tools_includes_goal_actions(agent):
    tools = agent._get_tools()
    tool_names = [t["name"] for t in tools]
    assert "agent:set_goal" in tool_names
    assert "agent:complete_goal" in tool_names
    assert "agent:update_goal" in tool_names
    assert "agent:list_goals" in tool_names
