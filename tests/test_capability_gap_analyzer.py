"""Tests for CapabilityGapAnalyzerSkill."""
import pytest
import asyncio
from singularity.skills.capability_gap_analyzer import CapabilityGapAnalyzerSkill
from singularity.skills.base import Skill, SkillManifest, SkillAction, SkillResult


class MockSkill(Skill):
    """Mock skill for testing introspection."""
    def __init__(self, skill_id, name, category, actions=None):
        super().__init__()
        self._skill_id = skill_id
        self._name = name
        self._category = category
        self._actions = actions or []

    @property
    def manifest(self):
        return SkillManifest(
            skill_id=self._skill_id, name=self._name, version="1.0.0",
            category=self._category, description=f"Mock {self._name}",
            actions=[SkillAction(name=a, description=a, parameters={}) for a in self._actions],
            required_credentials=[],
        )

    async def execute(self, action, params):
        return SkillResult(success=True, message="mock")


def _fresh_skill(mock_skills=None):
    s = CapabilityGapAnalyzerSkill()
    # Reset persistent data to fresh state
    s._save(s._default_state())
    if mock_skills:
        s.set_agent_skills(mock_skills)
    return s


def _sample_skills():
    return [
        MockSkill("billing_pipeline", "BillingPipeline", "revenue", ["create_invoice"]),
        MockSkill("payment", "Payment", "revenue", ["charge"]),
        MockSkill("event", "EventBus", "infrastructure", ["publish", "subscribe"]),
        MockSkill("scheduler", "Scheduler", "infrastructure", ["schedule", "tick"]),
        MockSkill("strategy", "Strategy", "meta-cognition", ["assess", "diagnose"]),
        MockSkill("self_modify", "SelfModify", "self-improvement", ["edit_prompt"]),
        MockSkill("replication", "Replication", "replication", ["snapshot", "spawn"]),
        MockSkill("goal_manager", "GoalManager", "goal-setting", ["create", "next"]),
        MockSkill("feedback_loop", "FeedbackLoop", "self-improvement", ["record"]),
        MockSkill("database", "Database", "infrastructure", ["query"]),
        MockSkill("http_client", "HTTPClient", "infrastructure", ["get", "post"]),
    ]


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


@pytest.fixture
def skill():
    return _fresh_skill(_sample_skills())


def test_manifest():
    s = _fresh_skill()
    m = s.manifest
    assert m.skill_id == "capability_gap_analyzer"
    action_names = [a.name for a in m.actions]
    assert "inventory" in action_names
    assert "analyze_gaps" in action_names
    assert "generate_plan" in action_names
    assert len(m.actions) == 8


def test_inventory(skill):
    result = run(skill.execute("inventory", {}))
    assert result.success
    assert result.data["total_skills"] == 11
    assert result.data["total_actions"] > 0
    assert len(result.data["skills"]) == 11


def test_analyze_gaps(skill):
    result = run(skill.execute("analyze_gaps", {}))
    assert result.success
    assert "pillar_gaps" in result.data
    assert "missing_bridges" in result.data
    assert result.data["total_skills"] == 11


def test_analyze_gaps_focus_pillar(skill):
    result = run(skill.execute("analyze_gaps", {"focus_pillar": "revenue"}))
    assert result.success
    assert len(result.data["pillar_gaps"]) == 1
    assert "revenue" in result.data["pillar_gaps"]


def test_score_gaps(skill):
    # First run analysis
    run(skill.execute("analyze_gaps", {}))
    result = run(skill.execute("score_gaps", {}))
    assert result.success
    assert "scored_gaps" in result.data
    gaps = result.data["scored_gaps"]
    # Scores should be sorted descending
    for i in range(len(gaps) - 1):
        assert gaps[i]["impact_score"] >= gaps[i + 1]["impact_score"]


def test_score_gaps_no_analysis():
    s = _fresh_skill([])
    result = run(s.execute("score_gaps", {}))
    assert not result.success
    assert "analyze_gaps" in result.message.lower()


def test_generate_plan(skill):
    result = run(skill.execute("generate_plan", {"max_items": 3}))
    assert result.success
    plan = result.data
    assert "items" in plan
    assert len(plan["items"]) <= 3
    for item in plan["items"]:
        assert "priority" in item
        assert "action" in item
        assert "pillar" in item
        assert "impact_score" in item


def test_pillar_coverage(skill):
    result = run(skill.execute("pillar_coverage", {}))
    assert result.success
    assert "pillars" in result.data
    assert "weakest_pillar" in result.data
    for pillar_data in result.data["pillars"].values():
        assert "coverage_pct" in pillar_data
        assert "strengths" in pillar_data
        assert "weaknesses" in pillar_data


def test_integration_map(skill):
    result = run(skill.execute("integration_map", {}))
    assert result.success
    assert "existing_bridges" in result.data
    assert "missing_bridges" in result.data
    assert result.data["total_skills"] == 11


def test_mark_addressed(skill):
    result = run(skill.execute("mark_addressed", {
        "gap_id": "test_gap_1",
        "resolution": "Built TestSkill",
    }))
    assert result.success
    # Check history
    hist = run(skill.execute("history", {}))
    assert hist.success
    addressed = hist.data["addressed_gaps"]
    assert any(g["gap_id"] == "test_gap_1" for g in addressed)


def test_mark_addressed_missing_params(skill):
    result = run(skill.execute("mark_addressed", {}))
    assert not result.success


def test_history_empty():
    s = _fresh_skill([])
    result = run(s.execute("history", {"limit": 5}))
    assert result.success
    assert len(result.data["analyses"]) == 0


def test_plan_filters_addressed_gaps(skill):
    # Generate plan
    plan1 = run(skill.execute("generate_plan", {"max_items": 10}))
    assert plan1.success
    items = plan1.data["items"]
    if items:
        # Mark first gap as addressed
        gap_id = items[0]["gap_id"]
        run(skill.execute("mark_addressed", {
            "gap_id": gap_id, "resolution": "built it",
        }))
        # Re-analyze and re-plan
        run(skill.execute("analyze_gaps", {}))
        plan2 = run(skill.execute("generate_plan", {"max_items": 10}))
        assert plan2.success
        plan2_ids = [i["gap_id"] for i in plan2.data["items"]]
        assert gap_id not in plan2_ids


def test_unknown_action(skill):
    result = run(skill.execute("nonexistent", {}))
    assert not result.success
