"""Tests for AgentMetaSkill."""

import pytest
from singularity.skills.agent_meta import AgentMetaSkill
from singularity.skills.base import SkillRegistry, Skill, SkillManifest, SkillAction, SkillResult


class DummySkill(Skill):
    """A minimal skill for testing."""
    def __init__(self, credentials=None, sid="dummy", name="Dummy", cat="dev"):
        super().__init__(credentials)
        self._sid = sid
        self._name = name
        self._cat = cat

    @property
    def manifest(self):
        return SkillManifest(
            skill_id=self._sid, name=self._name, version="1.0.0",
            category=self._cat, description=f"{self._name} skill for testing",
            actions=[
                SkillAction(name="do_thing", description="Does a thing", parameters={"x": {"type": "string", "required": True, "description": "Input value"}}),
                SkillAction(name="do_other", description="Does another thing", parameters={}),
            ],
            required_credentials=[],
        )

    async def execute(self, action, params):
        return SkillResult(success=True, message="ok")


class FakeAgent:
    """Fake agent for testing."""
    def __init__(self):
        self.name = "TestAgent"
        self.ticker = "TEST"
        self.agent_type = "general"
        self.specialty = "testing"
        self.balance = 42.0
        self.total_api_cost = 1.5
        self.total_instance_cost = 0.05
        self.total_tokens_used = 5000
        self.cycle = 10
        self.running = True
        self.instance_type = "local"
        self.instance_cost_per_hour = 0.0
        self.cycle_interval = 5.0
        self.recent_actions = []
        self.created_resources = {"files": [], "repos": [], "payment_links": [], "products": []}
        self.skills = SkillRegistry()


def make_skill_with_registry():
    skill = AgentMetaSkill()
    reg = SkillRegistry()
    reg.install(DummySkill, {})
    skill.set_registry_ref(reg)
    return skill, reg


class FsSkill(DummySkill):
    def __init__(self, credentials=None):
        super().__init__(credentials, "fs", "Filesystem", "system")


def make_skill_with_agent():
    skill = AgentMetaSkill()
    agent = FakeAgent()
    reg = SkillRegistry()
    reg.install(DummySkill, {})
    reg.install(FsSkill, {})
    agent.skills = reg
    skill.set_agent_ref(agent)
    return skill, agent


@pytest.mark.asyncio
async def test_list_skills():
    skill, reg = make_skill_with_registry()
    r = await skill.execute("list_skills", {})
    assert r.success
    assert r.data["total"] >= 1
    assert r.data["total_actions"] >= 2


@pytest.mark.asyncio
async def test_list_skills_filter_category():
    skill, reg = make_skill_with_registry()
    r = await skill.execute("list_skills", {"category": "dev"})
    assert r.success
    assert r.data["total"] >= 1
    r2 = await skill.execute("list_skills", {"category": "nonexistent"})
    assert r2.success
    assert r2.data["total"] == 0


@pytest.mark.asyncio
async def test_list_skills_verbose():
    skill, reg = make_skill_with_registry()
    r = await skill.execute("list_skills", {"verbose": True})
    assert r.success
    assert "actions" in r.data["skills"][0]


@pytest.mark.asyncio
async def test_skill_help():
    skill, reg = make_skill_with_registry()
    r = await skill.execute("skill_help", {"skill_id": "dummy"})
    assert r.success
    assert r.data["skill_id"] == "dummy"
    assert len(r.data["actions"]) == 2


@pytest.mark.asyncio
async def test_skill_help_action():
    skill, reg = make_skill_with_registry()
    r = await skill.execute("skill_help", {"skill_id": "dummy", "action_name": "do_thing"})
    assert r.success
    assert r.data["action"] == "do_thing"
    assert "parameters" in r.data


@pytest.mark.asyncio
async def test_skill_help_not_found():
    skill, reg = make_skill_with_registry()
    r = await skill.execute("skill_help", {"skill_id": "nope"})
    assert not r.success


@pytest.mark.asyncio
async def test_search_capabilities():
    skill, reg = make_skill_with_registry()
    r = await skill.execute("search_capabilities", {"query": "thing"})
    assert r.success
    assert r.data["total"] >= 1


@pytest.mark.asyncio
async def test_search_no_query():
    skill, reg = make_skill_with_registry()
    r = await skill.execute("search_capabilities", {"query": ""})
    assert not r.success


@pytest.mark.asyncio
async def test_agent_status():
    skill, agent = make_skill_with_agent()
    r = await skill.execute("agent_status", {})
    assert r.success
    assert r.data["name"] == "TestAgent"
    assert r.data["balance_usd"] == 42.0
    assert r.data["installed_skills"] >= 2


@pytest.mark.asyncio
async def test_agent_status_no_agent():
    skill = AgentMetaSkill()
    r = await skill.execute("agent_status", {})
    assert not r.success


@pytest.mark.asyncio
async def test_action_history():
    skill, agent = make_skill_with_agent()
    agent.recent_actions = [
        {"cycle": 1, "tool": "dummy:do_thing", "result": {"status": "success"}, "api_cost_usd": 0.01, "tokens": 100},
        {"cycle": 2, "tool": "dummy:do_other", "result": {"status": "error", "message": "fail"}, "api_cost_usd": 0.02, "tokens": 200},
    ]
    r = await skill.execute("action_history", {"limit": 5})
    assert r.success
    assert r.data["stats"]["total_actions"] == 2
    assert r.data["stats"]["successes"] == 1
    assert r.data["stats"]["failures"] == 1


@pytest.mark.asyncio
async def test_error_summary():
    skill, agent = make_skill_with_agent()
    agent.recent_actions = [
        {"cycle": 1, "tool": "dummy:do_thing", "result": {"status": "success"}},
        {"cycle": 2, "tool": "dummy:do_thing", "result": {"status": "error", "message": "timeout"}},
        {"cycle": 3, "tool": "fs:read", "result": {"status": "failed", "message": "not found"}},
        {"cycle": 4, "tool": "dummy:do_other", "result": {"status": "error", "message": "crash"}},
    ]
    r = await skill.execute("error_summary", {})
    assert r.success
    assert r.data["total_errors"] == 3
    assert "dummy" in r.data["errors_by_skill"]
    assert r.data["errors_by_skill"]["dummy"]["count"] == 2


@pytest.mark.asyncio
async def test_capability_matrix():
    skill, agent = make_skill_with_agent()
    r = await skill.execute("capability_matrix", {})
    assert r.success
    assert "matrix" in r.data
    assert "capability_areas" in r.data
    assert isinstance(r.data["enabled_areas"], list)


@pytest.mark.asyncio
async def test_suggest_action():
    skill, agent = make_skill_with_agent()
    r = await skill.execute("suggest_action", {"goal": "do a thing"})
    assert r.success
    assert "suggestions" in r.data


@pytest.mark.asyncio
async def test_suggest_action_empty():
    skill, agent = make_skill_with_agent()
    r = await skill.execute("suggest_action", {"goal": ""})
    assert not r.success


@pytest.mark.asyncio
async def test_unknown_action():
    skill = AgentMetaSkill()
    r = await skill.execute("nonexistent", {})
    assert not r.success


@pytest.mark.asyncio
async def test_no_credentials_required():
    skill = AgentMetaSkill()
    assert skill.check_credentials()
    assert skill.manifest.required_credentials == []
