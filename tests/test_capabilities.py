"""Tests for CapabilityAdvertiser."""
import pytest
from singularity.capabilities import (
    CapabilityAdvertiser, AgentCapabilityManifest, _auto_tag,
)
from singularity.skills.base import SkillRegistry, Skill, SkillManifest, SkillAction, SkillResult


class MockSkillA(Skill):
    @property
    def manifest(self):
        return SkillManifest(
            skill_id="files", name="FileSkill", version="1.0",
            category="filesystem", description="File operations",
            actions=[
                SkillAction(name="read", description="Read a file", parameters={"path": {"type": "string", "required": True}}, estimated_cost=0.0),
                SkillAction(name="write", description="Write content to file", parameters={"path": {"type": "string"}, "content": {"type": "string"}}, estimated_cost=0.0),
            ],
            required_credentials=[],
        )
    async def execute(self, action, params):
        return SkillResult(success=True)


class MockSkillB(Skill):
    @property
    def manifest(self):
        return SkillManifest(
            skill_id="shell", name="ShellSkill", version="1.0",
            category="system", description="Shell commands",
            actions=[
                SkillAction(name="bash", description="Execute a bash command", parameters={"command": {"type": "string"}}, estimated_cost=0.0),
            ],
            required_credentials=[],
        )
    async def execute(self, action, params):
        return SkillResult(success=True)


@pytest.fixture
def registry():
    reg = SkillRegistry()
    reg.install(MockSkillA)
    reg.install(MockSkillB)
    return reg


@pytest.fixture
def advertiser(registry):
    return CapabilityAdvertiser(registry, "TestAgent", "TEST", "coder", "Python development")


def test_generate_manifest(advertiser):
    m = advertiser.generate_manifest()
    assert m.agent_name == "TestAgent"
    assert m.total_skills == 2
    assert m.total_actions == 3
    assert "filesystem" in m.categories
    assert "system" in m.categories
    assert len(m.manifest_hash) == 16


def test_manifest_to_json(advertiser):
    m = advertiser.generate_manifest()
    j = m.to_json()
    assert '"agent_name": "TestAgent"' in j
    assert '"total_actions": 3' in j


def test_empty_registry():
    adv = CapabilityAdvertiser(None, "Empty", "E", "none", "")
    m = adv.generate_manifest()
    assert m.total_skills == 0
    assert m.total_actions == 0


def test_capability_card(advertiser):
    card = advertiser.generate_capability_card()
    assert "TestAgent" in card
    assert "files:read" in card
    assert "shell:bash" in card


def test_match_request(advertiser):
    matches = advertiser.match_request("read a file")
    assert len(matches) > 0
    assert matches[0]["action"] == "files:read"


def test_match_request_shell(advertiser):
    matches = advertiser.match_request("execute bash command")
    names = [m["action"] for m in matches]
    assert "shell:bash" in names


def test_match_request_no_match(advertiser):
    matches = advertiser.match_request("xyzzy foobar", threshold=0.9)
    assert len(matches) == 0


def test_capabilities_for_prompt(advertiser):
    prompt = advertiser.get_capabilities_for_prompt()
    assert "3 actions" in prompt
    assert "2 skills" in prompt
    assert "files:read" in prompt


def test_diff_capabilities(advertiser):
    other = AgentCapabilityManifest(
        agent_name="Other", agent_ticker="OTH", agent_type="writer",
        specialty="writing", version="1.0", generated_at="",
        capabilities=[], total_skills=0, total_actions=0,
        categories=[], tags=[],
    )
    diff = advertiser.diff_capabilities(other)
    assert len(diff["only_self"]) == 3
    assert len(diff["only_other"]) == 0
    assert diff["overlap_pct"] == 0.0


def test_auto_tag():
    tags = _auto_tag("read_file", "Read a file from disk")
    assert "read" in tags
    assert "io" in tags
    assert "filesystem" in tags
