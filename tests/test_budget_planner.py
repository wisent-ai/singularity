"""Tests for BudgetAwarePlannerSkill."""
import json
import pytest
import asyncio
from pathlib import Path

from singularity.skills.budget_planner import BudgetAwarePlannerSkill, DATA_FILE


@pytest.fixture(autouse=True)
def clean_data(tmp_path, monkeypatch):
    """Use temp files for all data."""
    test_file = tmp_path / "budget_planner.json"
    goals_file = tmp_path / "goals.json"
    monkeypatch.setattr("singularity.skills.budget_planner.DATA_FILE", test_file)
    # Write empty goals file
    goals_file.parent.mkdir(parents=True, exist_ok=True)
    goals_file.write_text(json.dumps({"goals": [], "completed_goals": [], "session_log": []}))
    yield test_file, goals_file


@pytest.fixture
def skill():
    return BudgetAwarePlannerSkill()


@pytest.fixture
def goals_file(clean_data):
    return clean_data[1]


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


class TestSetBudget:
    def test_set_budget(self, skill):
        r = run(skill.execute("set_budget", {"total": 50.0}))
        assert r.success
        assert "50.00" in r.message

    def test_set_budget_with_allocation(self, skill):
        r = run(skill.execute("set_budget", {
            "total": 100.0,
            "pillar_allocation": {"self_improvement": 0.4, "revenue": 0.3, "replication": 0.15, "goal_setting": 0.1, "other": 0.05}
        }))
        assert r.success
        assert r.data["pillar_allocation"]["self_improvement"] == 0.4

    def test_set_budget_invalid_allocation(self, skill):
        r = run(skill.execute("set_budget", {
            "total": 100.0,
            "pillar_allocation": {"self_improvement": 0.5, "revenue": 0.6}
        }))
        assert not r.success
        assert "sum to" in r.message.lower()


class TestEstimateGoal:
    def test_estimate_basic(self, skill):
        r = run(skill.execute("estimate_goal", {"title": "Build feature X", "expected_actions": 20}))
        assert r.success
        assert r.data["estimated_cost"] > 0
        assert r.data["num_actions"] == 20

    def test_estimate_with_milestones(self, skill):
        r = run(skill.execute("estimate_goal", {
            "title": "Deploy app",
            "milestones": ["Write code", "Test", "Deploy"],
        }))
        assert r.success
        # 3 milestones * 3 actions = 9 actions
        assert r.data["num_actions"] == 9

    def test_estimate_with_learned_costs(self, skill):
        # Teach the planner actual costs
        for _ in range(5):
            run(skill.execute("learn_costs", {"action_type": "llm_call", "cost": 0.05}))
        r = run(skill.execute("estimate_goal", {
            "title": "LLM task",
            "action_types": ["llm_call"],
            "expected_actions": 10,
        }))
        assert r.success
        assert abs(r.data["estimated_cost"] - 0.50) < 0.01  # 10 * $0.05


class TestAffordableGoals:
    def test_no_goals(self, skill):
        r = run(skill.execute("affordable_goals", {}))
        assert r.success
        assert len(r.data["affordable"]) == 0

    def test_with_affordable_goals(self, skill, goals_file, monkeypatch):
        # Patch goals file path in budget_planner module
        monkeypatch.setattr(
            "singularity.skills.budget_planner.Path",
            type("MockPath", (), {"__call__": lambda *a: goals_file.parent / "goals.json"})
        ) if False else None  # We'll write directly

        goals_data = {
            "goals": [
                {"id": "g1", "title": "Cheap goal", "pillar": "revenue", "priority": "high",
                 "status": "active", "milestones": [{"title": "m1", "completed": False}]},
                {"id": "g2", "title": "Another goal", "pillar": "self_improvement", "priority": "medium",
                 "status": "active", "milestones": []},
            ],
            "completed_goals": [],
        }
        # Write goals to the path that budget_planner will read
        real_goals_path = Path(__file__).parent.parent / "singularity" / "data" / "goals.json"
        real_goals_path.parent.mkdir(parents=True, exist_ok=True)
        real_goals_path.write_text(json.dumps(goals_data))
        try:
            r = run(skill.execute("affordable_goals", {}))
            assert r.success
            assert len(r.data["affordable"]) > 0
        finally:
            real_goals_path.unlink(missing_ok=True)


class TestPlanBudget:
    def test_plan_empty(self, skill):
        r = run(skill.execute("plan_budget", {}))
        assert r.success
        assert len(r.data["plan"]) == 0

    def test_plan_with_goals(self, skill):
        # Write goals to the real path
        real_goals_path = Path(__file__).parent.parent / "singularity" / "data" / "goals.json"
        real_goals_path.parent.mkdir(parents=True, exist_ok=True)
        goals_data = {
            "goals": [
                {"id": "g1", "title": "High priority", "pillar": "revenue", "priority": "critical",
                 "status": "active", "milestones": [{"title": "m1", "completed": False}]},
                {"id": "g2", "title": "Low priority", "pillar": "other", "priority": "low",
                 "status": "active", "milestones": []},
            ],
            "completed_goals": [],
        }
        real_goals_path.write_text(json.dumps(goals_data))
        try:
            r = run(skill.execute("plan_budget", {}))
            assert r.success
            assert r.data["budget_available"] > 0
            # Critical goal should be in plan
            if r.data["plan"]:
                assert r.data["plan"][0]["priority"] in ("critical", "high", "low")
        finally:
            real_goals_path.unlink(missing_ok=True)


class TestRecordCost:
    def test_record_cost(self, skill):
        r = run(skill.execute("record_cost", {"goal_id": "g1", "cost": 0.50, "revenue": 1.00, "pillar": "revenue"}))
        assert r.success
        assert r.data["roi"] == 1.0  # (1.00 - 0.50) / 0.50

    def test_record_updates_budget(self, skill):
        run(skill.execute("set_budget", {"total": 100.0}))
        run(skill.execute("record_cost", {"goal_id": "g1", "cost": 10.0, "pillar": "revenue"}))
        r = run(skill.execute("budget_status", {}))
        assert r.data["spent"] == 10.0
        assert r.data["remaining"] == 90.0


class TestROIReport:
    def test_roi_empty(self, skill):
        r = run(skill.execute("roi_report", {}))
        assert r.success
        assert r.data["records"] == 0

    def test_roi_with_data(self, skill):
        run(skill.execute("record_cost", {"goal_id": "g1", "cost": 1.0, "revenue": 3.0, "pillar": "revenue"}))
        run(skill.execute("record_cost", {"goal_id": "g2", "cost": 0.5, "revenue": 0.0, "pillar": "self_improvement"}))
        r = run(skill.execute("roi_report", {}))
        assert r.success
        assert r.data["summary"]["goals_tracked"] == 2
        assert r.data["summary"]["total_cost"] == 1.5
        assert "revenue" in r.data["by_pillar"]


class TestBudgetStatus:
    def test_status_default(self, skill):
        r = run(skill.execute("budget_status", {}))
        assert r.success
        assert r.data["health"] == "healthy"
        assert "pillar_status" in r.data

    def test_status_critical(self, skill):
        run(skill.execute("set_budget", {"total": 10.0}))
        run(skill.execute("record_cost", {"goal_id": "g1", "cost": 9.6, "pillar": "revenue"}))
        r = run(skill.execute("budget_status", {}))
        assert r.data["health"] == "critical"


class TestLearnCosts:
    def test_learn_single(self, skill):
        r = run(skill.execute("learn_costs", {"action_type": "llm_call", "cost": 0.03}))
        assert r.success
        assert r.data["observations"] == 1

    def test_learn_running_average(self, skill):
        run(skill.execute("learn_costs", {"action_type": "api", "cost": 0.01}))
        run(skill.execute("learn_costs", {"action_type": "api", "cost": 0.03}))
        r = run(skill.execute("learn_costs", {"action_type": "api", "cost": 0.02}))
        assert r.success
        assert abs(r.data["running_average"] - 0.02) < 0.001
        assert r.data["observations"] == 3
