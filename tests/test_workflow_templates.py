"""Tests for WorkflowTemplateLibrarySkill."""
import pytest
from singularity.skills.workflow_templates import WorkflowTemplateLibrarySkill

@pytest.fixture
def skill(tmp_path):
    s = WorkflowTemplateLibrarySkill()
    s._data_path = tmp_path / "templates.json"
    return s

@pytest.mark.asyncio
async def test_browse_all(skill):
    r = await skill.execute("browse", {})
    assert r.success
    assert len(r.data["templates"]) == 10  # 10 built-in templates
    assert r.data["total_templates"] == 10

@pytest.mark.asyncio
async def test_browse_by_category(skill):
    r = await skill.execute("browse", {"category": "ci_cd"})
    assert r.success
    assert all(t["category"] == "ci_cd" for t in r.data["templates"])
    assert len(r.data["templates"]) == 2

@pytest.mark.asyncio
async def test_get_template(skill):
    r = await skill.execute("get", {"template_id": "github_pr_review"})
    assert r.success
    assert r.data["template"]["name"] == "GitHub PR Auto-Review"
    assert "parameters" in r.data["template"]
    assert "steps" in r.data["template"]

@pytest.mark.asyncio
async def test_instantiate_with_params(skill):
    r = await skill.execute("instantiate", {
        "template_id": "github_pr_review",
        "params": {"repo": "wisent-ai/singularity"},
    })
    assert r.success
    inst = r.data["instance"]
    assert inst["template_id"] == "github_pr_review"
    assert inst["parameters"]["repo"] == "wisent-ai/singularity"
    assert inst["status"] == "ready"
    assert len(inst["steps"]) == 3

@pytest.mark.asyncio
async def test_instantiate_missing_required(skill):
    r = await skill.execute("instantiate", {
        "template_id": "github_pr_review",
        "params": {},  # missing required "repo"
    })
    assert not r.success
    assert "repo" in str(r.data.get("missing", []))

@pytest.mark.asyncio
async def test_register_custom_template(skill):
    r = await skill.execute("register", {
        "name": "My Custom Workflow",
        "category": "custom",
        "description": "A test workflow",
        "parameters": {"input": {"type": "string", "required": True}},
        "steps": [{"skill": "shell", "action": "run", "params_from": {"cmd": "param.input"}}],
        "tags": ["test", "custom"],
    })
    assert r.success
    tid = r.data["template_id"]
    r2 = await skill.execute("get", {"template_id": tid})
    assert r2.success
    assert r2.data["template"]["name"] == "My Custom Workflow"

@pytest.mark.asyncio
async def test_search(skill):
    r = await skill.execute("search", {"query": "github deploy"})
    assert r.success
    assert len(r.data["results"]) > 0
    # github_pr_review and deploy_on_merge should match
    ids = [res["id"] for res in r.data["results"]]
    assert "github_pr_review" in ids or "deploy_on_merge" in ids

@pytest.mark.asyncio
async def test_rate_template(skill):
    r = await skill.execute("rate", {"template_id": "github_pr_review", "rating": 5, "agent_id": "agent_1", "comment": "Great!"})
    assert r.success
    assert r.data["avg_rating"] == 5.0
    # Second rating
    await skill.execute("rate", {"template_id": "github_pr_review", "rating": 3, "agent_id": "agent_2"})
    r2 = await skill.execute("get", {"template_id": "github_pr_review"})
    assert r2.data["avg_rating"] == 4.0

@pytest.mark.asyncio
async def test_popular(skill):
    # Instantiate one template to bump use count
    await skill.execute("instantiate", {"template_id": "github_pr_review", "params": {"repo": "test/repo"}})
    r = await skill.execute("popular", {"limit": 3})
    assert r.success
    assert r.data["popular"][0]["id"] == "github_pr_review"

@pytest.mark.asyncio
async def test_export(skill):
    r = await skill.execute("export", {"template_id": "customer_onboarding", "params": {"customer_name": "Alice", "customer_email": "alice@test.com"}})
    assert r.success
    export = r.data["export"]
    assert export["parameters"]["customer_name"] == "Alice"
    assert export["source_template"] == "customer_onboarding"

@pytest.mark.asyncio
async def test_manifest(skill):
    m = skill.manifest
    assert m.skill_id == "workflow_templates"
    assert len(m.actions) == 8
