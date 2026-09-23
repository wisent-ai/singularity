"""Tests for StrategySkill."""
import pytest
from singularity.skills.strategy import StrategySkill


@pytest.fixture
def skill():
    s = StrategySkill()
    prompt_parts = ["You are a helpful agent."]
    actions = []
    s.set_agent_hooks(
        get_actions=lambda: actions,
        get_prompt=lambda: "\n".join(prompt_parts),
        append_prompt=lambda text: prompt_parts.append(text),
        get_balance=lambda: 50.0,
    )
    s._actions_ref = actions  # for test manipulation
    s._prompt_ref = prompt_parts
    return s


def _add_actions(skill, successes=5, failures=2, errors=0):
    for i in range(successes):
        skill._actions_ref.append({"tool": "fs:view", "params": {"path": f"f{i}.py"}, "result": {"status": "success"}, "api_cost_usd": 0.01, "tokens": 100})
    for i in range(failures):
        skill._actions_ref.append({"tool": "shell:bash", "params": {"cmd": "bad"}, "result": {"status": "failed"}, "api_cost_usd": 0.01, "tokens": 100})
    for i in range(errors):
        skill._actions_ref.append({"tool": "shell:bash", "params": {"cmd": "err"}, "result": {"status": "error"}, "api_cost_usd": 0.01, "tokens": 100})


@pytest.mark.asyncio
async def test_analyze_empty(skill):
    result = await skill.execute("analyze", {})
    assert result.success
    assert result.data["action_count"] == 0


@pytest.mark.asyncio
async def test_analyze_with_actions(skill):
    _add_actions(skill, successes=8, failures=2)
    result = await skill.execute("analyze", {})
    assert result.success
    assert result.data["action_count"] == 10
    assert result.data["success_rate"] == 0.8
    assert result.data["score"] > 0


@pytest.mark.asyncio
async def test_score(skill):
    _add_actions(skill, successes=10, failures=0)
    result = await skill.execute("score", {})
    assert result.success
    assert result.data["score"] >= 60


@pytest.mark.asyncio
async def test_evolve(skill):
    _add_actions(skill, successes=3, failures=7)
    result = await skill.execute("evolve", {})
    assert result.success
    assert result.data["evolved"]
    assert len(result.data["strategies_applied"]) > 0
    # Verify prompt was modified
    full_prompt = "\n".join(skill._prompt_ref)
    assert "EVOLVED STRATEGY" in full_prompt


@pytest.mark.asyncio
async def test_evolve_with_focus(skill):
    _add_actions(skill, successes=5, failures=5)
    result = await skill.execute("evolve", {"focus": "cost"})
    assert result.success
    assert result.data["focus"] == "cost"


@pytest.mark.asyncio
async def test_save_and_get_insights(skill):
    result = await skill.execute("save_insight", {"insight": "Test insight", "category": "efficiency"})
    assert result.success
    result = await skill.execute("get_insights", {})
    assert result.success
    assert result.data["total"] >= 1


@pytest.mark.asyncio
async def test_loop_detection(skill):
    # Add same action 5 times
    for _ in range(5):
        skill._actions_ref.append({"tool": "shell:bash", "params": {"cmd": "ls"}, "result": {"status": "failed"}, "api_cost_usd": 0.01, "tokens": 100})
    result = await skill.execute("analyze", {})
    assert result.success
    assert result.data["detected_loops"] > 0


@pytest.mark.asyncio
async def test_check_credentials(skill):
    assert skill.check_credentials()
    s2 = StrategySkill()
    assert not s2.check_credentials()


@pytest.mark.asyncio
async def test_manifest(skill):
    m = skill.manifest
    assert m.skill_id == "strategy"
    assert len(m.actions) == 5
    action_names = [a.name for a in m.actions]
    assert "analyze" in action_names
    assert "evolve" in action_names
    assert "score" in action_names
