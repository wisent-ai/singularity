"""Tests for WorkflowTemplateBridgeSkill."""
import pytest
from unittest.mock import AsyncMock, MagicMock
from singularity.skills.workflow_template_bridge import WorkflowTemplateBridgeSkill
from singularity.skills.base import SkillResult


@pytest.fixture
def skill(tmp_path):
    s = WorkflowTemplateBridgeSkill(data_dir=str(tmp_path / "bridge"))
    return s


@pytest.fixture
def skill_with_context(tmp_path):
    """Skill with mocked context for inter-skill calls."""
    s = WorkflowTemplateBridgeSkill(data_dir=str(tmp_path / "bridge"))
    ctx = MagicMock()

    async def mock_call_skill(skill_id, action, params):
        if skill_id == "workflow_templates" and action == "instantiate":
            return SkillResult(success=True, message="Instantiated", data={
                "instance_id": "inst_abc123",
                "instance": {
                    "name": "Test Workflow",
                    "steps": [
                        {"skill": "github", "action": "get_pr", "params": {"repo": "test/repo"}},
                        {"skill": "code_review", "action": "review", "params": {"code": "step.0.diff"}},
                    ],
                },
                "estimated_cost": 0.05,
            })
        if skill_id == "workflow_templates" and action == "browse":
            return SkillResult(success=True, message="OK", data={
                "templates": [
                    {"id": "t1", "name": "Template 1", "category": "ci_cd", "description": "Test",
                     "required_skills": ["github"], "estimated_cost": 0.05, "tags": ["ci"]},
                    {"id": "t2", "name": "Template 2", "category": "billing", "description": "Test2",
                     "required_skills": ["payment"], "estimated_cost": 0.02, "tags": ["billing"]},
                ],
            })
        if skill_id == "event_driven_workflow" and action == "create_workflow":
            return SkillResult(success=True, message="Created", data={
                "workflow_id": "wf_xyz789",
                "name": params.get("name", "test"),
                "steps_count": len(params.get("steps", [])),
            })
        if skill_id == "event_driven_workflow" and action == "delete_workflow":
            return SkillResult(success=True, message="Deleted")
        if skill_id == "event_driven_workflow" and action == "bind_webhook":
            return SkillResult(success=True, message="Bound")
        if skill_id == "event_driven_workflow" and action == "stats":
            return SkillResult(success=True, message="Stats", data={"workflows": []})
        return SkillResult(success=False, message="Unknown")

    ctx.call_skill = AsyncMock(side_effect=mock_call_skill)
    s._context = ctx
    return s


@pytest.mark.asyncio
async def test_manifest(skill):
    m = skill.manifest
    assert m.skill_id == "workflow_template_bridge"
    assert len(m.actions) == 8
    action_names = [a.name for a in m.actions]
    assert "deploy" in action_names
    assert "quick_deploy" in action_names
    assert "catalog" in action_names


@pytest.mark.asyncio
async def test_deploy(skill_with_context):
    r = await skill_with_context.execute("deploy", {
        "template_id": "github_pr_review",
        "params": {"repo": "wisent-ai/singularity"},
    })
    assert r.success
    assert "deployment_id" in r.data
    assert r.data["workflow_id"] == "wf_xyz789"
    assert r.data["template_id"] == "github_pr_review"
    assert r.data["steps_count"] == 2


@pytest.mark.asyncio
async def test_deploy_missing_template_id(skill_with_context):
    r = await skill_with_context.execute("deploy", {})
    assert not r.success
    assert "template_id" in r.message


@pytest.mark.asyncio
async def test_deploy_with_event_bindings(skill_with_context):
    r = await skill_with_context.execute("deploy", {
        "template_id": "github_pr_review",
        "params": {"repo": "test/repo"},
        "event_bindings": [{"source": "github", "event": "pull_request.opened"}],
    })
    assert r.success
    assert r.data["event_bindings_count"] == 1


@pytest.mark.asyncio
async def test_list_empty(skill):
    r = await skill.execute("list", {})
    assert r.success
    assert r.data["total"] == 0


@pytest.mark.asyncio
async def test_list_after_deploy(skill_with_context):
    await skill_with_context.execute("deploy", {
        "template_id": "t1", "params": {},
    })
    r = await skill_with_context.execute("list", {})
    assert r.success
    assert r.data["total"] == 1
    assert r.data["active"] == 1


@pytest.mark.asyncio
async def test_undeploy(skill_with_context):
    dep = await skill_with_context.execute("deploy", {
        "template_id": "t1", "params": {},
    })
    dep_id = dep.data["deployment_id"]
    r = await skill_with_context.execute("undeploy", {"deployment_id": dep_id})
    assert r.success
    # Verify listed as stopped
    lst = await skill_with_context.execute("list", {"status": "stopped"})
    assert lst.data["total"] == 1


@pytest.mark.asyncio
async def test_undeploy_not_found(skill):
    r = await skill.execute("undeploy", {"deployment_id": "nonexistent"})
    assert not r.success


@pytest.mark.asyncio
async def test_status(skill_with_context):
    dep = await skill_with_context.execute("deploy", {
        "template_id": "t1", "params": {"repo": "x/y"},
    })
    dep_id = dep.data["deployment_id"]
    r = await skill_with_context.execute("status", {"deployment_id": dep_id})
    assert r.success
    assert r.data["status"] == "active"
    assert r.data["template_id"] == "t1"


@pytest.mark.asyncio
async def test_redeploy(skill_with_context):
    dep = await skill_with_context.execute("deploy", {
        "template_id": "t1", "params": {"repo": "old/repo"},
    })
    dep_id = dep.data["deployment_id"]
    r = await skill_with_context.execute("redeploy", {
        "deployment_id": dep_id,
        "params": {"repo": "new/repo"},
    })
    assert r.success
    assert r.data["parameters"]["repo"] == "new/repo"


@pytest.mark.asyncio
async def test_quick_deploy(skill_with_context):
    r = await skill_with_context.execute("quick_deploy", {
        "template_id": "github_pr_review",
        "params": {"repo": "test/repo"},
        "event_source": "github",
        "event_name": "push",
    })
    assert r.success
    assert r.data["event_bindings_count"] == 1


@pytest.mark.asyncio
async def test_catalog(skill_with_context):
    r = await skill_with_context.execute("catalog", {})
    assert r.success
    assert r.data["total"] == 2
    assert r.data["available"] == 2


@pytest.mark.asyncio
async def test_catalog_shows_deployed(skill_with_context):
    await skill_with_context.execute("deploy", {
        "template_id": "t1", "params": {},
    })
    r = await skill_with_context.execute("catalog", {})
    assert r.success
    assert r.data["deployed"] == 1
    assert r.data["available"] == 1


@pytest.mark.asyncio
async def test_bind(skill_with_context):
    dep = await skill_with_context.execute("deploy", {
        "template_id": "t1", "params": {},
    })
    dep_id = dep.data["deployment_id"]
    r = await skill_with_context.execute("bind", {
        "deployment_id": dep_id,
        "event_bindings": [{"source": "stripe", "event": "payment.success"}],
    })
    assert r.success
    assert r.data["total_bindings"] == 1


@pytest.mark.asyncio
async def test_convert_steps_with_inter_step_refs(skill):
    steps = [
        {"skill": "github", "action": "get_pr", "params": {"repo": "test"}},
        {"skill": "code_review", "action": "review", "params": {"code": "step.0.diff", "depth": "standard"}},
    ]
    converted = skill._convert_steps(steps)
    assert len(converted) == 2
    assert converted[0]["skill_id"] == "github"
    assert converted[1]["skill_id"] == "code_review"
    # Inter-step ref should become input_mapping
    assert "input_mapping" in converted[1]
    assert "step_1.data.diff" in converted[1]["input_mapping"]
    assert converted[1]["input_mapping"]["step_1.data.diff"] == "code"
    # Static param should remain
    assert converted[1]["params"]["depth"] == "standard"


@pytest.mark.asyncio
async def test_unknown_action(skill):
    r = await skill.execute("nonexistent", {})
    assert not r.success


@pytest.mark.asyncio
async def test_persistence(tmp_path):
    s1 = WorkflowTemplateBridgeSkill(data_dir=str(tmp_path / "bridge"))
    s1._deployments["dep_test"] = {
        "deployment_id": "dep_test", "template_id": "t1",
        "workflow_id": "wf_1", "workflow_name": "Test",
        "parameters": {}, "event_bindings": [],
        "status": "active", "deployed_at": "2025-01-01",
        "updated_at": "2025-01-01", "trigger_count": 0,
    }
    s1._save_data()
    s2 = WorkflowTemplateBridgeSkill(data_dir=str(tmp_path / "bridge"))
    assert "dep_test" in s2._deployments
