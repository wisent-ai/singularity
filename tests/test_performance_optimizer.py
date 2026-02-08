"""Tests for PerformanceOptimizerSkill."""
import os
import pytest
from singularity.skills.performance_optimizer import PerformanceOptimizerSkill, ImprovementCycle


def _make_outcomes(skill="shell", action="run", successes=3, failures=7):
    """Generate mock outcomes."""
    outcomes = []
    for i in range(successes):
        outcomes.append({"skill": skill, "action": action, "outcome": "success", "context": f"task_{i}", "timestamp": 1000 + i})
    for i in range(failures):
        outcomes.append({"skill": skill, "action": action, "outcome": "failure", "context": f"error_{i}", "timestamp": 2000 + i})
    return outcomes


@pytest.fixture
def skill(tmp_path):
    s = PerformanceOptimizerSkill()
    s._storage_dir = str(tmp_path)
    s._cycles_file = os.path.join(str(tmp_path), "cycles.json")
    s._analysis_file = os.path.join(str(tmp_path), "analysis_history.json")
    s._config_file = os.path.join(str(tmp_path), "config.json")
    s._min_outcomes_for_analysis = 3
    s._min_post_outcomes_to_conclude = 3
    # Wire up mock hooks
    s._current_prompt = "You are a helpful agent."
    s._mock_outcomes = _make_outcomes()
    s.set_hooks(
        get_outcomes=lambda: s._mock_outcomes,
        get_prompt=lambda: s._current_prompt,
        set_prompt=lambda p: setattr(s, "_current_prompt", p),
    )
    return s


@pytest.mark.asyncio
async def test_analyze(skill):
    r = await skill.execute("analyze", {"window": 50})
    assert r.success
    assert r.data["overall_success_rate"] == 0.3
    assert len(r.data["bottlenecks"]) > 0


@pytest.mark.asyncio
async def test_analyze_insufficient_data(skill):
    skill._mock_outcomes = [{"skill": "x", "action": "y", "outcome": "success"}]
    r = await skill.execute("analyze", {})
    assert r.success
    assert "Insufficient" in r.message


@pytest.mark.asyncio
async def test_hypothesize_requires_analysis(skill):
    r = await skill.execute("hypothesize", {})
    assert not r.success


@pytest.mark.asyncio
async def test_hypothesize_after_analysis(skill):
    await skill.execute("analyze", {})
    r = await skill.execute("hypothesize", {"max_hypotheses": 2})
    assert r.success
    assert len(r.data["hypotheses"]) > 0
    h = r.data["hypotheses"][0]
    assert "target_skill" in h
    assert "hypothesis" in h


@pytest.mark.asyncio
async def test_propose_cycle(skill):
    r = await skill.execute("propose_cycle", {
        "hypothesis": "Adding validation will help",
        "target_skill": "shell",
        "modification": "\nValidate before running.",
    })
    assert r.success
    assert r.data["cycle_id"] == "cycle_1"
    assert r.data["status"] == "proposed"


@pytest.mark.asyncio
async def test_apply_cycle(skill):
    await skill.execute("propose_cycle", {
        "hypothesis": "Test fix", "target_skill": "shell",
        "modification": "\n== FIX ==",
    })
    r = await skill.execute("apply_cycle", {"cycle_id": "cycle_1"})
    assert r.success
    assert "== FIX ==" in skill._current_prompt


@pytest.mark.asyncio
async def test_cannot_apply_two_cycles(skill):
    await skill.execute("propose_cycle", {"hypothesis": "A", "target_skill": "s", "modification": "\nA"})
    await skill.execute("propose_cycle", {"hypothesis": "B", "target_skill": "s", "modification": "\nB"})
    await skill.execute("apply_cycle", {"cycle_id": "cycle_1"})
    r = await skill.execute("apply_cycle", {"cycle_id": "cycle_2"})
    assert not r.success
    assert "already active" in r.message


@pytest.mark.asyncio
async def test_record_post_outcome_and_evaluate(skill):
    await skill.execute("propose_cycle", {"hypothesis": "Fix", "target_skill": "shell", "modification": "\nFix"})
    await skill.execute("apply_cycle", {"cycle_id": "cycle_1"})
    for _ in range(4):
        await skill.execute("record_post_outcome", {"cycle_id": "cycle_1", "outcome": "success"})
    r = await skill.execute("evaluate_cycle", {"cycle_id": "cycle_1"})
    assert r.success
    assert r.data["result"] == "improved"


@pytest.mark.asyncio
async def test_auto_revert_on_degradation(skill):
    skill._mock_outcomes = _make_outcomes(successes=8, failures=2)  # 80% baseline
    await skill.execute("propose_cycle", {"hypothesis": "Bad idea", "target_skill": "shell", "modification": "\nBad"})
    await skill.execute("apply_cycle", {"cycle_id": "cycle_1"})
    assert "\nBad" in skill._current_prompt
    for _ in range(3):
        r = await skill.execute("record_post_outcome", {"cycle_id": "cycle_1", "outcome": "failure"})
    assert r.data["auto_revert_triggered"]
    assert "\nBad" not in skill._current_prompt


@pytest.mark.asyncio
async def test_revert_cycle_manually(skill):
    await skill.execute("propose_cycle", {"hypothesis": "X", "target_skill": "s", "modification": "\nMOD"})
    await skill.execute("apply_cycle", {"cycle_id": "cycle_1"})
    assert "\nMOD" in skill._current_prompt
    r = await skill.execute("revert_cycle", {"cycle_id": "cycle_1"})
    assert r.success
    assert "\nMOD" not in skill._current_prompt


@pytest.mark.asyncio
async def test_list_cycles(skill):
    await skill.execute("propose_cycle", {"hypothesis": "A", "target_skill": "s", "modification": "\nA"})
    await skill.execute("propose_cycle", {"hypothesis": "B", "target_skill": "s", "modification": "\nB"})
    r = await skill.execute("list_cycles", {})
    assert r.success
    assert len(r.data["cycles"]) == 2


@pytest.mark.asyncio
async def test_list_cycles_with_filter(skill):
    await skill.execute("propose_cycle", {"hypothesis": "A", "target_skill": "s", "modification": "\nA"})
    await skill.execute("apply_cycle", {"cycle_id": "cycle_1"})
    r = await skill.execute("list_cycles", {"status_filter": "active"})
    assert len(r.data["cycles"]) == 1
    r2 = await skill.execute("list_cycles", {"status_filter": "proposed"})
    assert len(r2.data["cycles"]) == 0


@pytest.mark.asyncio
async def test_improvement_report(skill):
    await skill.execute("propose_cycle", {"hypothesis": "Fix", "target_skill": "shell", "modification": "\nFix"})
    await skill.execute("apply_cycle", {"cycle_id": "cycle_1"})
    for _ in range(4):
        await skill.execute("record_post_outcome", {"cycle_id": "cycle_1", "outcome": "success"})
    await skill.execute("evaluate_cycle", {"cycle_id": "cycle_1"})
    r = await skill.execute("improvement_report", {})
    assert r.success
    assert r.data["by_result"]["improved"] == 1


@pytest.mark.asyncio
async def test_auto_optimize(skill):
    r = await skill.execute("auto_optimize", {"window": 50})
    assert r.success
    assert r.data["proposed_cycle"] is not None
    assert r.data["proposed_cycle"]["cycle_id"] == "cycle_1"


@pytest.mark.asyncio
async def test_auto_optimize_no_bottlenecks(skill):
    skill._mock_outcomes = _make_outcomes(successes=10, failures=0)
    r = await skill.execute("auto_optimize", {})
    assert r.success
    assert r.data["proposed_cycle"] is None


def test_improvement_cycle_serialization():
    c = ImprovementCycle("c1", "hyp", "shell", "run", "mod", 0.5, 10)
    d = c.to_dict()
    c2 = ImprovementCycle.from_dict(d)
    assert c2.cycle_id == "c1"
    assert c2.baseline_success_rate == 0.5
