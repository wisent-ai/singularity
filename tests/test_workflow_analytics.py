"""Tests for WorkflowAnalyticsSkill."""

import pytest
import json
from pathlib import Path
from unittest.mock import patch
from singularity.skills.workflow_analytics import WorkflowAnalyticsSkill, WORKFLOWS_FILE


@pytest.fixture
def skill(tmp_path):
    """Create a WorkflowAnalyticsSkill with temp storage."""
    test_file = tmp_path / "workflow_analytics.json"
    with patch("singularity.skills.workflow_analytics.WORKFLOWS_FILE", test_file):
        s = WorkflowAnalyticsSkill()
        yield s


@pytest.mark.asyncio
async def test_manifest(skill):
    m = skill.manifest
    assert m.skill_id == "workflow_analytics"
    assert len(m.actions) == 9
    action_names = {a.name for a in m.actions}
    assert "start_workflow" in action_names
    assert "suggest_next" in action_names
    assert "anti_patterns" in action_names


@pytest.mark.asyncio
async def test_start_and_end_workflow(skill, tmp_path):
    test_file = tmp_path / "workflow_analytics.json"
    with patch("singularity.skills.workflow_analytics.WORKFLOWS_FILE", test_file):
        r = await skill.execute("start_workflow", {"name": "test_wf", "goal": "testing"})
        assert r.success
        assert "test_wf" in r.message

        r = await skill.execute("record_step", {"action": "github:clone", "success": True})
        assert r.success
        assert r.data["step"]["step_number"] == 1

        r = await skill.execute("record_step", {"action": "shell:run_tests", "success": True})
        assert r.success
        assert r.data["total_steps"] == 2

        r = await skill.execute("end_workflow", {"success": True, "notes": "all good"})
        assert r.success
        assert "SUCCESS" in r.message

        data = json.loads(test_file.read_text())
        assert len(data["workflows"]) == 1
        assert data["workflows"][0]["success"] is True
        assert data["workflows"][0]["action_sequence"] == ["github:clone", "shell:run_tests"]


@pytest.mark.asyncio
async def test_no_double_start(skill):
    r = await skill.execute("start_workflow", {"name": "wf1"})
    assert r.success
    r = await skill.execute("start_workflow", {"name": "wf2"})
    assert not r.success
    assert "already in progress" in r.message


@pytest.mark.asyncio
async def test_record_without_workflow(skill):
    r = await skill.execute("record_step", {"action": "foo", "success": True})
    assert not r.success
    assert "No workflow" in r.message


@pytest.mark.asyncio
async def test_end_without_workflow(skill):
    r = await skill.execute("end_workflow", {"success": True})
    assert not r.success


@pytest.mark.asyncio
async def test_find_patterns(skill, tmp_path):
    test_file = tmp_path / "workflow_analytics.json"
    with patch("singularity.skills.workflow_analytics.WORKFLOWS_FILE", test_file):
        # Record two workflows with overlapping patterns
        for success in [True, True]:
            await skill.execute("start_workflow", {"name": "deploy"})
            await skill.execute("record_step", {"action": "git:commit", "success": True})
            await skill.execute("record_step", {"action": "git:push", "success": True})
            await skill.execute("record_step", {"action": "deploy:run", "success": True})
            await skill.execute("end_workflow", {"success": success})

        r = await skill.execute("find_patterns", {"min_occurrences": 2})
        assert r.success
        assert len(r.data["patterns"]) > 0
        # The bigram [git:commit, git:push] should appear
        found = any(
            p["pattern"] == ["git:commit", "git:push"]
            for p in r.data["patterns"]
        )
        assert found


@pytest.mark.asyncio
async def test_anti_patterns(skill, tmp_path):
    test_file = tmp_path / "workflow_analytics.json"
    with patch("singularity.skills.workflow_analytics.WORKFLOWS_FILE", test_file):
        # Two failing workflows with same pattern
        for _ in range(2):
            await skill.execute("start_workflow", {"name": "bad_deploy"})
            await skill.execute("record_step", {"action": "skip_tests", "success": True})
            await skill.execute("record_step", {"action": "force_push", "success": True})
            await skill.execute("end_workflow", {"success": False})

        r = await skill.execute("anti_patterns", {"min_occurrences": 2})
        assert r.success
        assert len(r.data["anti_patterns"]) > 0
        assert r.data["anti_patterns"][0]["failure_rate"] == 100.0


@pytest.mark.asyncio
async def test_suggest_next(skill, tmp_path):
    test_file = tmp_path / "workflow_analytics.json"
    with patch("singularity.skills.workflow_analytics.WORKFLOWS_FILE", test_file):
        # Build history: git:commit -> git:push -> deploy:run (success)
        for _ in range(3):
            await skill.execute("start_workflow", {"name": "deploy"})
            await skill.execute("record_step", {"action": "git:commit", "success": True})
            await skill.execute("record_step", {"action": "git:push", "success": True})
            await skill.execute("record_step", {"action": "deploy:run", "success": True})
            await skill.execute("end_workflow", {"success": True})

        r = await skill.execute("suggest_next", {"recent_actions": ["git:commit", "git:push"]})
        assert r.success
        assert len(r.data["suggestions"]) > 0
        assert r.data["suggestions"][0]["action"] == "deploy:run"


@pytest.mark.asyncio
async def test_templates(skill, tmp_path):
    test_file = tmp_path / "workflow_analytics.json"
    with patch("singularity.skills.workflow_analytics.WORKFLOWS_FILE", test_file):
        r = await skill.execute("save_template", {
            "name": "standard_deploy",
            "steps": ["git:commit", "git:push", "deploy:run"],
            "description": "Standard deployment flow"
        })
        assert r.success

        r = await skill.execute("list_templates", {})
        assert r.success
        assert len(r.data["templates"]) == 1
        assert r.data["templates"][0]["name"] == "standard_deploy"

        # Update existing template
        r = await skill.execute("save_template", {
            "name": "standard_deploy",
            "steps": ["git:commit", "git:push", "test:run", "deploy:run"],
        })
        assert r.success

        r = await skill.execute("list_templates", {})
        assert len(r.data["templates"]) == 1
        assert len(r.data["templates"][0]["steps"]) == 4


@pytest.mark.asyncio
async def test_summary(skill, tmp_path):
    test_file = tmp_path / "workflow_analytics.json"
    with patch("singularity.skills.workflow_analytics.WORKFLOWS_FILE", test_file):
        r = await skill.execute("summary", {})
        assert r.success
        assert r.data["total_workflows"] == 0

        # Add workflows
        await skill.execute("start_workflow", {"name": "build"})
        await skill.execute("record_step", {"action": "compile", "success": True})
        await skill.execute("end_workflow", {"success": True})

        await skill.execute("start_workflow", {"name": "build"})
        await skill.execute("record_step", {"action": "compile", "success": False})
        await skill.execute("end_workflow", {"success": False})

        r = await skill.execute("summary", {})
        assert r.success
        assert r.data["total_workflows"] == 2
        assert r.data["successes"] == 1
        assert r.data["failures"] == 1
        assert r.data["success_rate"] == 50.0


@pytest.mark.asyncio
async def test_suggest_no_history(skill):
    r = await skill.execute("suggest_next", {"recent_actions": ["foo"]})
    assert r.success
    assert r.data["reason"] == "no_history"


@pytest.mark.asyncio
async def test_suggest_empty_actions(skill):
    r = await skill.execute("suggest_next", {"recent_actions": []})
    assert not r.success
