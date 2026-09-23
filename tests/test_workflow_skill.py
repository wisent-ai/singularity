"""Tests for WorkflowSkill - multi-step workflow chaining."""

import pytest
import json
import shutil
from pathlib import Path
from singularity.skills.workflow import WorkflowSkill, WORKFLOW_DIR
from singularity.skills.base import SkillResult


@pytest.fixture
def clean_workflows(tmp_path, monkeypatch):
    """Use a temp dir for workflow storage."""
    test_dir = tmp_path / "workflows"
    test_dir.mkdir()
    import singularity.skills.workflow as wf_mod
    monkeypatch.setattr(wf_mod, "WORKFLOW_DIR", test_dir)
    return test_dir


@pytest.fixture
def skill(clean_workflows):
    s = WorkflowSkill()
    return s


@pytest.fixture
def skill_with_executor(skill):
    """Skill with a mock executor that tracks calls."""
    calls = []

    async def mock_executor(skill_id, action, params):
        calls.append({"skill_id": skill_id, "action": action, "params": params})
        if params.get("fail"):
            return SkillResult(success=False, message="Forced failure")
        return SkillResult(
            success=True,
            message=f"Executed {skill_id}:{action}",
            data={"result": f"output_from_{action}", "echo": params},
        )

    skill.set_skill_executor(mock_executor)
    return skill, calls


@pytest.mark.asyncio
async def test_create_workflow(skill):
    result = await skill.execute("create", {
        "id": "test_wf",
        "name": "Test Workflow",
        "description": "A test",
        "steps": [
            {"name": "s1", "skill_id": "shell", "action": "bash", "params": {"command": "echo hi"}},
            {"name": "s2", "skill_id": "filesystem", "action": "ls", "params": {"path": "/tmp"}},
        ],
        "tags": ["test"],
    })
    assert result.success
    assert result.data["steps"] == 2
    assert result.data["step_names"] == ["s1", "s2"]


@pytest.mark.asyncio
async def test_create_requires_fields(skill):
    r1 = await skill.execute("create", {"id": "", "name": "X", "steps": [{"name": "s", "skill_id": "a", "action": "b"}]})
    assert not r1.success
    r2 = await skill.execute("create", {"id": "x", "name": "", "steps": [{"name": "s", "skill_id": "a", "action": "b"}]})
    assert not r2.success
    r3 = await skill.execute("create", {"id": "x", "name": "X", "steps": []})
    assert not r3.success


@pytest.mark.asyncio
async def test_duplicate_step_names(skill):
    result = await skill.execute("create", {
        "id": "dup", "name": "Dup",
        "steps": [
            {"name": "s1", "skill_id": "a", "action": "b"},
            {"name": "s1", "skill_id": "c", "action": "d"},
        ],
    })
    assert not result.success
    assert "Duplicate" in result.message


@pytest.mark.asyncio
async def test_invalid_condition(skill):
    result = await skill.execute("create", {
        "id": "bad", "name": "Bad",
        "steps": [{"name": "s1", "skill_id": "a", "action": "b", "condition": "maybe"}],
    })
    assert not result.success
    assert "invalid condition" in result.message


@pytest.mark.asyncio
async def test_run_workflow(skill_with_executor):
    skill, calls = skill_with_executor
    await skill.execute("create", {
        "id": "run_test", "name": "Run Test",
        "steps": [
            {"name": "s1", "skill_id": "shell", "action": "bash", "params": {"command": "echo hi"}},
            {"name": "s2", "skill_id": "fs", "action": "ls", "params": {"path": "/tmp"}},
        ],
    })
    result = await skill.execute("run", {"workflow_id": "run_test"})
    assert result.success
    assert result.data["steps_run"] == 2
    assert result.data["steps_failed"] == 0
    assert len(calls) == 2
    assert calls[0]["skill_id"] == "shell"
    assert calls[1]["skill_id"] == "fs"


@pytest.mark.asyncio
async def test_run_nonexistent(skill_with_executor):
    skill, _ = skill_with_executor
    result = await skill.execute("run", {"workflow_id": "nope"})
    assert not result.success


@pytest.mark.asyncio
async def test_run_without_executor(skill):
    await skill.execute("create", {
        "id": "no_exec", "name": "No Exec",
        "steps": [{"name": "s1", "skill_id": "a", "action": "b"}],
    })
    result = await skill.execute("run", {"workflow_id": "no_exec"})
    assert not result.success
    assert "executor" in result.message.lower()


@pytest.mark.asyncio
async def test_conditional_on_success(skill_with_executor):
    skill, calls = skill_with_executor
    await skill.execute("create", {
        "id": "cond", "name": "Conditional",
        "steps": [
            {"name": "s1", "skill_id": "a", "action": "b"},
            {"name": "s2", "skill_id": "c", "action": "d", "condition": "on_success"},
            {"name": "s3", "skill_id": "e", "action": "f", "condition": "on_failure"},
        ],
    })
    result = await skill.execute("run", {"workflow_id": "cond"})
    assert result.success
    assert result.data["steps_run"] == 2  # s1 + s2
    assert result.data["steps_skipped"] == 1  # s3


@pytest.mark.asyncio
async def test_conditional_on_failure(skill_with_executor):
    skill, calls = skill_with_executor
    await skill.execute("create", {
        "id": "fail_cond", "name": "Fail Conditional",
        "steps": [
            {"name": "s1", "skill_id": "a", "action": "b", "params": {"fail": True}},
            {"name": "s2", "skill_id": "c", "action": "d", "condition": "on_success"},
            {"name": "s3", "skill_id": "e", "action": "f", "condition": "on_failure"},
        ],
    })
    result = await skill.execute("run", {"workflow_id": "fail_cond"})
    assert result.data["steps_run"] == 2  # s1 + s3
    assert result.data["steps_skipped"] == 1  # s2
    assert result.data["steps_failed"] == 1  # s1


@pytest.mark.asyncio
async def test_template_resolution(skill_with_executor):
    skill, calls = skill_with_executor
    await skill.execute("create", {
        "id": "tmpl", "name": "Template",
        "steps": [
            {"name": "s1", "skill_id": "a", "action": "get"},
            {"name": "s2", "skill_id": "b", "action": "use", "params": {"val": "{{s1.result}}"}},
        ],
    })
    result = await skill.execute("run", {"workflow_id": "tmpl"})
    assert result.success
    assert calls[1]["params"]["val"] == "output_from_get"


@pytest.mark.asyncio
async def test_variable_injection(skill_with_executor):
    skill, calls = skill_with_executor
    await skill.execute("create", {
        "id": "vars", "name": "Variables",
        "steps": [
            {"name": "s1", "skill_id": "a", "action": "b", "params": {"msg": "Hello {{var.name}}!"}},
        ],
    })
    result = await skill.execute("run", {"workflow_id": "vars", "variables": {"name": "World"}})
    assert result.success
    assert calls[0]["params"]["msg"] == "Hello World!"


@pytest.mark.asyncio
async def test_list_workflows(skill):
    await skill.execute("create", {"id": "w1", "name": "W1", "steps": [{"name": "s", "skill_id": "a", "action": "b"}], "tags": ["x"]})
    await skill.execute("create", {"id": "w2", "name": "W2", "steps": [{"name": "s", "skill_id": "a", "action": "b"}], "tags": ["y"]})
    r = await skill.execute("list", {})
    assert r.data["count"] == 2
    r_filtered = await skill.execute("list", {"tag": "x"})
    assert r_filtered.data["count"] == 1


@pytest.mark.asyncio
async def test_show_workflow(skill):
    await skill.execute("create", {"id": "show_me", "name": "Show", "steps": [{"name": "s", "skill_id": "a", "action": "b"}]})
    r = await skill.execute("show", {"workflow_id": "show_me"})
    assert r.success
    assert r.data["name"] == "Show"
    assert len(r.data["steps"]) == 1


@pytest.mark.asyncio
async def test_delete_workflow(skill):
    await skill.execute("create", {"id": "del_me", "name": "Del", "steps": [{"name": "s", "skill_id": "a", "action": "b"}]})
    r = await skill.execute("delete", {"workflow_id": "del_me"})
    assert r.success
    r2 = await skill.execute("show", {"workflow_id": "del_me"})
    assert not r2.success


@pytest.mark.asyncio
async def test_persistence(clean_workflows):
    s1 = WorkflowSkill()
    await s1.execute("create", {"id": "persist", "name": "Persist", "steps": [{"name": "s", "skill_id": "a", "action": "b"}]})
    # New instance should load from disk
    s2 = WorkflowSkill()
    r = await s2.execute("show", {"workflow_id": "persist"})
    assert r.success
    assert r.data["name"] == "Persist"


@pytest.mark.asyncio
async def test_run_count_tracking(skill_with_executor):
    skill, _ = skill_with_executor
    await skill.execute("create", {"id": "counted", "name": "Counted", "steps": [{"name": "s", "skill_id": "a", "action": "b"}]})
    await skill.execute("run", {"workflow_id": "counted"})
    await skill.execute("run", {"workflow_id": "counted"})
    r = await skill.execute("show", {"workflow_id": "counted"})
    assert r.data["run_count"] == 2


@pytest.mark.asyncio
async def test_retry_logic(skill):
    attempt_count = [0]

    async def flaky_executor(skill_id, action, params):
        attempt_count[0] += 1
        if attempt_count[0] < 3:
            return SkillResult(success=False, message="Temporary failure")
        return SkillResult(success=True, message="OK", data={"done": True})

    skill.set_skill_executor(flaky_executor)
    await skill.execute("create", {
        "id": "retry", "name": "Retry",
        "steps": [{"name": "s1", "skill_id": "a", "action": "b", "retries": 3}],
    })
    result = await skill.execute("run", {"workflow_id": "retry"})
    assert result.success
    assert result.data["step_results"]["s1"]["retries_used"] == 2


@pytest.mark.asyncio
async def test_unknown_action(skill):
    r = await skill.execute("nonexistent", {})
    assert not r.success


@pytest.mark.asyncio
async def test_manifest(skill):
    m = skill.manifest
    assert m.skill_id == "workflow"
    assert len(m.actions) == 5
    action_names = {a.name for a in m.actions}
    assert action_names == {"create", "run", "list", "show", "delete"}
