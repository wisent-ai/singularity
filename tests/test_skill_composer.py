"""Tests for SkillComposerSkill - dynamic skill composition from existing skills."""

import asyncio
import json
import pytest
from pathlib import Path
from unittest.mock import patch, AsyncMock, MagicMock

from singularity.skills.skill_composer import SkillComposerSkill, COMPOSER_FILE
from singularity.skills.base import SkillResult


@pytest.fixture(autouse=True)
def clean_state(tmp_path):
    """Use temp file for each test."""
    test_file = tmp_path / "compositions.json"
    with patch("singularity.skills.skill_composer.COMPOSER_FILE", test_file):
        yield test_file


@pytest.fixture
def skill(clean_state):
    with patch("singularity.skills.skill_composer.COMPOSER_FILE", clean_state):
        return SkillComposerSkill()


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def make_mock_context(results=None):
    """Create a mock skill context that returns SkillResults."""
    ctx = MagicMock()
    if results is None:
        results = [SkillResult(success=True, message="ok", data={"key": "value"})]
    call_count = {"n": 0}

    async def mock_execute(skill_id, action, params):
        idx = min(call_count["n"], len(results) - 1)
        call_count["n"] += 1
        return results[idx]

    ctx.execute_skill = AsyncMock(side_effect=mock_execute)
    return ctx


def test_manifest(skill):
    m = skill.manifest
    assert m.skill_id == "skill_composer"
    assert m.category == "self_improvement"
    action_names = [a.name for a in m.actions]
    assert "compose" in action_names
    assert "execute_composition" in action_names
    assert "list_compositions" in action_names
    assert "generate_code" in action_names


def test_compose_basic(skill):
    result = run(skill.execute("compose", {
        "name": "Review and Report",
        "description": "Review code then summarize findings",
        "steps": [
            {"skill_id": "revenue_services", "action": "code_review", "params": {"code": "x=1"}},
            {"skill_id": "revenue_services", "action": "text_summary", "params": {"text": "results"}},
        ],
    }))
    assert result.success
    assert result.data["composition_id"].startswith("comp_")
    assert result.data["steps_count"] == 2
    assert len(result.data["step_summary"]) == 2


def test_compose_validation_errors(skill):
    # Missing name
    result = run(skill.execute("compose", {"description": "test", "steps": [{"skill_id": "a", "action": "b"}]}))
    assert not result.success
    assert "Name" in result.message

    # Missing steps
    result = run(skill.execute("compose", {"name": "test", "description": "test", "steps": []}))
    assert not result.success

    # Invalid step (no action)
    result = run(skill.execute("compose", {"name": "t", "description": "d", "steps": [{"skill_id": "a"}]}))
    assert not result.success


def test_list_empty(skill):
    result = run(skill.execute("list_compositions", {}))
    assert result.success
    assert result.data["compositions"] == []


def test_compose_then_list(skill):
    run(skill.execute("compose", {
        "name": "Pipeline A",
        "description": "First pipeline",
        "steps": [{"skill_id": "s1", "action": "a1"}],
    }))
    run(skill.execute("compose", {
        "name": "Pipeline B",
        "description": "Second pipeline",
        "steps": [{"skill_id": "s2", "action": "a2"}],
    }))
    result = run(skill.execute("list_compositions", {}))
    assert result.success
    assert len(result.data["compositions"]) == 2


def test_get_composition(skill):
    comp = run(skill.execute("compose", {
        "name": "Test Comp",
        "description": "A test",
        "steps": [{"skill_id": "x", "action": "y", "params": {"key": "val"}}],
    }))
    comp_id = comp.data["composition_id"]

    result = run(skill.execute("get_composition", {"composition_id": comp_id}))
    assert result.success
    assert result.data["composition"]["name"] == "Test Comp"
    assert len(result.data["composition"]["steps"]) == 1


def test_delete_composition(skill):
    comp = run(skill.execute("compose", {
        "name": "To Delete",
        "description": "Will be deleted",
        "steps": [{"skill_id": "x", "action": "y"}],
    }))
    comp_id = comp.data["composition_id"]

    result = run(skill.execute("delete_composition", {"composition_id": comp_id}))
    assert result.success
    assert result.data["deleted_id"] == comp_id

    # Verify deleted
    result = run(skill.execute("get_composition", {"composition_id": comp_id}))
    assert not result.success


def test_execute_composition(skill):
    # Create composition
    comp = run(skill.execute("compose", {
        "name": "Two Step",
        "description": "Two step pipeline",
        "steps": [
            {"skill_id": "svc", "action": "step1"},
            {"skill_id": "svc", "action": "step2"},
        ],
    }))
    comp_id = comp.data["composition_id"]

    # Set up mock context
    ctx = make_mock_context([
        SkillResult(success=True, message="step1 done", data={"token": "abc123"}),
        SkillResult(success=True, message="step2 done", data={"result": "final"}),
    ])
    skill.set_context(ctx)

    result = run(skill.execute("execute_composition", {"composition_id": comp_id}))
    assert result.success
    assert result.data["steps_completed"] == 2
    assert result.data["total_steps"] == 2
    assert ctx.execute_skill.call_count == 2


def test_execute_stops_on_failure(skill):
    comp = run(skill.execute("compose", {
        "name": "Failing",
        "description": "Will fail at step 1",
        "steps": [
            {"skill_id": "s", "action": "a1"},
            {"skill_id": "s", "action": "a2"},
        ],
    }))
    comp_id = comp.data["composition_id"]

    ctx = make_mock_context([
        SkillResult(success=False, message="error!", data={}),
        SkillResult(success=True, message="ok", data={}),
    ])
    skill.set_context(ctx)

    result = run(skill.execute("execute_composition", {"composition_id": comp_id}))
    assert not result.success
    assert result.data["steps_completed"] == 1  # Stopped at step 0


def test_execute_continue_on_failure(skill):
    comp = run(skill.execute("compose", {
        "name": "Resilient",
        "description": "Continues even if step fails",
        "steps": [
            {"skill_id": "s", "action": "a1", "continue_on_failure": True},
            {"skill_id": "s", "action": "a2"},
        ],
    }))
    comp_id = comp.data["composition_id"]

    ctx = make_mock_context([
        SkillResult(success=False, message="error", data={}),
        SkillResult(success=True, message="ok", data={}),
    ])
    skill.set_context(ctx)

    result = run(skill.execute("execute_composition", {"composition_id": comp_id}))
    assert result.success
    assert result.data["steps_completed"] == 2


def test_execute_no_context(skill):
    comp = run(skill.execute("compose", {
        "name": "No Ctx",
        "description": "test",
        "steps": [{"skill_id": "s", "action": "a"}],
    }))
    # Don't set context
    result = run(skill.execute("execute_composition", {"composition_id": comp.data["composition_id"]}))
    assert not result.success  # Each step fails


def test_generate_code(skill):
    comp = run(skill.execute("compose", {
        "name": "Code Gen Test",
        "description": "Pipeline for code generation",
        "steps": [
            {"skill_id": "svc", "action": "review", "params": {"code": "x=1"}},
            {"skill_id": "svc", "action": "summarize"},
        ],
    }))
    comp_id = comp.data["composition_id"]

    result = run(skill.execute("generate_code", {"composition_id": comp_id}))
    assert result.success
    assert "class CodeGenTestSkill" in result.data["code"]
    assert "code_gen_test" in result.data["skill_id"]
    assert result.data["file_name"] == "code_gen_test.py"


def test_suggest_compositions(skill):
    # Not enough history
    result = run(skill.execute("suggest_compositions", {}))
    assert result.success
    assert len(result.data["suggestions"]) == 0


def test_execution_updates_stats(skill):
    comp = run(skill.execute("compose", {
        "name": "Stats Test",
        "description": "test",
        "steps": [{"skill_id": "s", "action": "a"}],
    }))
    comp_id = comp.data["composition_id"]

    ctx = make_mock_context([SkillResult(success=True, message="ok", data={})])
    skill.set_context(ctx)

    run(skill.execute("execute_composition", {"composition_id": comp_id}))
    run(skill.execute("execute_composition", {"composition_id": comp_id}))

    details = run(skill.execute("get_composition", {"composition_id": comp_id}))
    assert details.data["composition"]["execution_count"] == 2
    assert details.data["composition"]["success_count"] == 2
