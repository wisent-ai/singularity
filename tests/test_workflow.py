#!/usr/bin/env python3
"""Tests for WorkflowSkill - multi-step automated workflow execution."""

import pytest
import json
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path

from singularity.skills.workflow import WorkflowSkill, WORKFLOW_FILE, WorkflowStatus, StepStatus
from singularity.skills.base import SkillResult, SkillContext, SkillRegistry


@pytest.fixture(autouse=True)
def clean_workflow_file():
    """Ensure clean state for each test."""
    if WORKFLOW_FILE.exists():
        WORKFLOW_FILE.unlink()
    yield
    if WORKFLOW_FILE.exists():
        WORKFLOW_FILE.unlink()


@pytest.fixture
def skill():
    """Create WorkflowSkill instance."""
    return WorkflowSkill()


@pytest.fixture
def mock_context():
    """Create a mock SkillContext that simulates skill calls."""
    registry = MagicMock(spec=SkillRegistry)
    ctx = SkillContext(registry=registry, agent_name="TestAgent")

    async def mock_call_skill(skill_id, action, params=None):
        # Simulate different skill responses
        if skill_id == "memory" and action == "get":
            return SkillResult(success=True, message="Found", data={"value": "test_data"})
        if skill_id == "content" and action == "generate":
            return SkillResult(success=True, message="Generated", data={"text": "Hello world", "tokens": 42})
        if skill_id == "failing_skill":
            return SkillResult(success=False, message="Simulated failure")
        return SkillResult(success=True, message=f"OK: {skill_id}:{action}", data={"status": "done"})

    ctx.call_skill = AsyncMock(side_effect=mock_call_skill)
    return ctx


@pytest.fixture
def skill_with_context(skill, mock_context):
    """Skill with an attached mock context."""
    skill.set_context(mock_context)
    return skill


def test_manifest(skill):
    m = skill.manifest
    assert m.skill_id == "workflow"
    assert m.category == "automation"
    assert len(m.actions) == 9
    action_names = {a.name for a in m.actions}
    assert "create" in action_names
    assert "execute" in action_names
    assert "list" in action_names
    assert "stats" in action_names


@pytest.mark.asyncio
async def test_create_workflow(skill):
    result = await skill.execute("create", {
        "name": "test_pipeline",
        "description": "A test workflow",
        "steps": [
            {"skill_id": "memory", "action": "get", "params": {"key": "x"}},
            {"skill_id": "content", "action": "generate", "params": {"prompt": "hello"}},
        ],
    })
    assert result.success
    assert result.data["step_count"] == 2


@pytest.mark.asyncio
async def test_create_empty_name(skill):
    result = await skill.execute("create", {"name": "", "steps": [{"skill_id": "x", "action": "y"}]})
    assert not result.success


@pytest.mark.asyncio
async def test_create_no_steps(skill):
    result = await skill.execute("create", {"name": "empty", "steps": []})
    assert not result.success


@pytest.mark.asyncio
async def test_create_invalid_step(skill):
    result = await skill.execute("create", {"name": "bad", "steps": [{"skill_id": "x"}]})
    assert not result.success
    assert "action" in result.message


@pytest.mark.asyncio
async def test_list_empty(skill):
    result = await skill.execute("list", {})
    assert result.success
    assert result.data["count"] == 0


@pytest.mark.asyncio
async def test_list_workflows(skill):
    await skill.execute("create", {"name": "wf1", "steps": [{"skill_id": "a", "action": "b"}]})
    await skill.execute("create", {"name": "wf2", "steps": [{"skill_id": "c", "action": "d"}]})
    result = await skill.execute("list", {})
    assert result.success
    assert result.data["count"] == 2


@pytest.mark.asyncio
async def test_get_workflow(skill):
    await skill.execute("create", {"name": "my_wf", "steps": [{"skill_id": "a", "action": "b"}]})
    result = await skill.execute("get", {"name": "my_wf"})
    assert result.success
    assert result.data["name"] == "my_wf"


@pytest.mark.asyncio
async def test_get_nonexistent(skill):
    result = await skill.execute("get", {"name": "nope"})
    assert not result.success


@pytest.mark.asyncio
async def test_delete_workflow(skill):
    await skill.execute("create", {"name": "del_me", "steps": [{"skill_id": "a", "action": "b"}]})
    result = await skill.execute("delete", {"name": "del_me"})
    assert result.success
    result2 = await skill.execute("get", {"name": "del_me"})
    assert not result2.success


@pytest.mark.asyncio
async def test_execute_workflow(skill_with_context):
    await skill_with_context.execute("create", {
        "name": "exec_test",
        "steps": [
            {"skill_id": "memory", "action": "get", "params": {"key": "test"}},
            {"skill_id": "content", "action": "generate", "params": {"prompt": "hi"}},
        ],
    })
    result = await skill_with_context.execute("execute", {"name": "exec_test"})
    assert result.success
    assert result.data["steps_completed"] == 2
    assert result.data["steps_failed"] == 0


@pytest.mark.asyncio
async def test_execute_with_failure_stop(skill_with_context):
    await skill_with_context.execute("create", {
        "name": "fail_test",
        "steps": [
            {"skill_id": "failing_skill", "action": "do", "on_failure": "stop"},
            {"skill_id": "memory", "action": "get"},
        ],
    })
    result = await skill_with_context.execute("execute", {"name": "fail_test"})
    assert not result.success
    assert result.data["steps_failed"] == 1
    assert result.data["steps_completed"] == 0


@pytest.mark.asyncio
async def test_execute_with_failure_skip(skill_with_context):
    await skill_with_context.execute("create", {
        "name": "skip_test",
        "steps": [
            {"skill_id": "failing_skill", "action": "do", "on_failure": "skip"},
            {"skill_id": "memory", "action": "get"},
        ],
    })
    result = await skill_with_context.execute("execute", {"name": "skip_test"})
    assert result.success
    assert result.data["steps_completed"] == 1
    assert result.data["steps_failed"] == 1


@pytest.mark.asyncio
async def test_execute_no_context(skill):
    await skill.execute("create", {"name": "no_ctx", "steps": [{"skill_id": "a", "action": "b"}]})
    result = await skill.execute("execute", {"name": "no_ctx"})
    assert not result.success
    assert "context" in result.message.lower()


@pytest.mark.asyncio
async def test_execute_nonexistent(skill_with_context):
    result = await skill_with_context.execute("execute", {"name": "nope"})
    assert not result.success


@pytest.mark.asyncio
async def test_conditional_step(skill_with_context):
    await skill_with_context.execute("create", {
        "name": "cond_test",
        "steps": [
            {"skill_id": "memory", "action": "get", "params": {"key": "test"}},
            {
                "skill_id": "content", "action": "generate",
                "condition": {"ref": "$steps.0.value", "equals": "test_data"},
            },
        ],
    })
    result = await skill_with_context.execute("execute", {"name": "cond_test"})
    assert result.success
    assert result.data["steps_completed"] == 2


@pytest.mark.asyncio
async def test_conditional_step_skipped(skill_with_context):
    await skill_with_context.execute("create", {
        "name": "cond_skip",
        "steps": [
            {"skill_id": "memory", "action": "get"},
            {
                "skill_id": "content", "action": "generate",
                "condition": {"ref": "$steps.0.value", "equals": "WRONG"},
            },
        ],
    })
    result = await skill_with_context.execute("execute", {"name": "cond_skip"})
    assert result.success
    assert result.data["steps_skipped"] == 1


@pytest.mark.asyncio
async def test_input_references(skill_with_context):
    await skill_with_context.execute("create", {
        "name": "input_ref",
        "steps": [
            {"skill_id": "memory", "action": "get", "params": {"key": "$inputs.my_key"}},
        ],
    })
    result = await skill_with_context.execute("execute", {
        "name": "input_ref",
        "inputs": {"my_key": "resolved_value"},
    })
    assert result.success
    # Verify the call was made with the resolved parameter
    call_args = skill_with_context.context.call_skill.call_args_list[0]
    assert call_args[0][2]["key"] == "resolved_value"


@pytest.mark.asyncio
async def test_history(skill_with_context):
    await skill_with_context.execute("create", {
        "name": "hist_wf",
        "steps": [{"skill_id": "memory", "action": "get"}],
    })
    await skill_with_context.execute("execute", {"name": "hist_wf"})
    await skill_with_context.execute("execute", {"name": "hist_wf"})
    result = await skill_with_context.execute("history", {"name": "hist_wf"})
    assert result.success
    assert result.data["count"] == 2


@pytest.mark.asyncio
async def test_save_and_use_template(skill_with_context):
    await skill_with_context.execute("create", {
        "name": "source_wf",
        "description": "Template source",
        "steps": [{"skill_id": "memory", "action": "get", "params": {"key": "default"}}],
    })
    result = await skill_with_context.execute("save_template", {
        "name": "source_wf",
        "template_name": "my_template",
    })
    assert result.success

    result2 = await skill_with_context.execute("from_template", {
        "template_name": "my_template",
        "name": "new_wf",
        "overrides": {"0": {"key": "overridden"}},
    })
    assert result2.success
    assert result2.data["template"] == "my_template"


@pytest.mark.asyncio
async def test_stats(skill_with_context):
    await skill_with_context.execute("create", {
        "name": "stats_wf",
        "steps": [{"skill_id": "memory", "action": "get"}],
    })
    await skill_with_context.execute("execute", {"name": "stats_wf"})
    result = await skill_with_context.execute("stats", {})
    assert result.success
    assert result.data["total_executed"] >= 1
    assert result.data["workflow_count"] >= 1


@pytest.mark.asyncio
async def test_retry_on_failure(skill_with_context):
    call_count = 0
    original_call = skill_with_context.context.call_skill

    async def flaky_call(skill_id, action, params=None):
        nonlocal call_count
        call_count += 1
        if call_count <= 1:
            return SkillResult(success=False, message="Temporary failure")
        return SkillResult(success=True, message="OK", data={"recovered": True})

    skill_with_context.context.call_skill = AsyncMock(side_effect=flaky_call)

    await skill_with_context.execute("create", {
        "name": "retry_wf",
        "steps": [{"skill_id": "flaky", "action": "do", "max_retries": 2}],
    })
    result = await skill_with_context.execute("execute", {"name": "retry_wf"})
    assert result.success
    assert result.data["step_results"][0]["retries"] == 1
