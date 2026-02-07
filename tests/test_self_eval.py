"""Tests for SelfEvalSkill - cognitive-level self-evaluation."""
import json
import pytest
import asyncio
from pathlib import Path

from singularity.skills.self_eval import SelfEvalSkill, EVAL_FILE


@pytest.fixture(autouse=True)
def clean_eval_file():
    """Remove eval file before/after each test."""
    if EVAL_FILE.exists():
        EVAL_FILE.unlink()
    yield
    if EVAL_FILE.exists():
        EVAL_FILE.unlink()


@pytest.fixture
def skill():
    return SelfEvalSkill()


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


class TestEvaluate:
    def test_basic_evaluation(self, skill):
        result = run(skill.execute("evaluate", {
            "task_description": "Write a sorting algorithm",
            "output_summary": "Implemented quicksort with O(n log n) average",
            "scores": {"correctness": 9, "efficiency": 8, "robustness": 7},
        }))
        assert result.success
        assert result.data["overall"] == 8.0
        assert result.data["scores"]["correctness"] == 9

    def test_scores_clamped(self, skill):
        result = run(skill.execute("evaluate", {
            "task_description": "Test",
            "output_summary": "Test output",
            "scores": {"correctness": 15, "efficiency": -3},
        }))
        assert result.success
        assert result.data["scores"]["correctness"] == 10
        assert result.data["scores"]["efficiency"] == 1

    def test_missing_task_description(self, skill):
        result = run(skill.execute("evaluate", {
            "output_summary": "Test",
            "scores": {"correctness": 5},
        }))
        assert not result.success

    def test_empty_scores(self, skill):
        result = run(skill.execute("evaluate", {
            "task_description": "Test",
            "output_summary": "Test",
            "scores": {},
        }))
        assert not result.success

    def test_persists_to_disk(self, skill):
        run(skill.execute("evaluate", {
            "task_description": "Persist test",
            "output_summary": "Output",
            "scores": {"correctness": 7},
        }))
        data = json.loads(EVAL_FILE.read_text())
        assert len(data["evaluations"]) == 1
        assert data["evaluations"][0]["task_description"] == "Persist test"


class TestAnalyze:
    def _add_evals(self, skill, count=5, base_score=6):
        for i in range(count):
            run(skill.execute("evaluate", {
                "task_description": f"Task {i}",
                "output_summary": f"Output {i}",
                "scores": {
                    "correctness": min(10, base_score + i % 3),
                    "efficiency": max(1, base_score - i % 2),
                    "robustness": base_score,
                },
                "skill_used": "shell" if i % 2 == 0 else "github",
            }))

    def test_analyze_empty(self, skill):
        result = run(skill.execute("analyze", {}))
        assert not result.success

    def test_analyze_with_data(self, skill):
        self._add_evals(skill)
        result = run(skill.execute("analyze", {}))
        assert result.success
        assert result.data["evaluation_count"] == 5
        assert "correctness" in result.data["dimension_stats"]
        assert "shell" in result.data["skill_stats"]

    def test_identifies_weaknesses(self, skill):
        # Add evaluations with low efficiency
        for i in range(5):
            run(skill.execute("evaluate", {
                "task_description": f"Task {i}",
                "output_summary": f"Output {i}",
                "scores": {"correctness": 9, "efficiency": 3, "robustness": 8},
            }))
        result = run(skill.execute("analyze", {}))
        assert result.success
        assert "efficiency" in result.data["weaknesses"]
        assert "correctness" in result.data["strengths"]


class TestDirectives:
    def _add_weak_evals(self, skill):
        for i in range(5):
            run(skill.execute("evaluate", {
                "task_description": f"Task {i}",
                "output_summary": f"Output {i}",
                "scores": {"correctness": 4, "efficiency": 3, "communication": 8},
            }))

    def test_generate_directives(self, skill):
        self._add_weak_evals(skill)
        result = run(skill.execute("generate_directives", {}))
        assert result.success
        assert len(result.data["new_directives"]) > 0
        # Should target weak dimensions
        dims = [d["target_dimension"] for d in result.data["new_directives"]]
        assert "efficiency" in dims or "correctness" in dims

    def test_get_directives(self, skill):
        self._add_weak_evals(skill)
        run(skill.execute("generate_directives", {}))
        result = run(skill.execute("get_directives", {}))
        assert result.success
        assert len(result.data["active"]) > 0

    def test_complete_directive(self, skill):
        self._add_weak_evals(skill)
        gen_result = run(skill.execute("generate_directives", {}))
        d_id = gen_result.data["new_directives"][0]["id"]
        result = run(skill.execute("complete_directive", {
            "directive_id": d_id,
            "outcome": "improved",
            "notes": "Added validation step",
        }))
        assert result.success
        # Verify it's no longer active
        dirs = run(skill.execute("get_directives", {}))
        active_ids = [d["id"] for d in dirs.data["active"]]
        assert d_id not in active_ids

    def test_invalid_outcome(self, skill):
        result = run(skill.execute("complete_directive", {
            "directive_id": "x",
            "outcome": "invalid",
        }))
        assert not result.success


class TestTrendAndReport:
    def test_trend_empty(self, skill):
        result = run(skill.execute("trend", {}))
        assert not result.success

    def test_trend_with_data(self, skill):
        for i in range(3):
            run(skill.execute("evaluate", {
                "task_description": f"T{i}",
                "output_summary": f"O{i}",
                "scores": {"correctness": 5 + i},
            }))
        result = run(skill.execute("trend", {}))
        assert result.success
        assert result.data["total_evaluations"] == 3

    def test_report(self, skill):
        for i in range(3):
            run(skill.execute("evaluate", {
                "task_description": f"T{i}",
                "output_summary": f"O{i}",
                "scores": {"correctness": 7, "efficiency": 5},
            }))
        result = run(skill.execute("report", {}))
        assert result.success
        assert "summary" in result.data
        assert "dimensions" in result.data

    def test_reset(self, skill):
        run(skill.execute("evaluate", {
            "task_description": "T",
            "output_summary": "O",
            "scores": {"correctness": 5},
        }))
        result = run(skill.execute("reset", {}))
        assert result.success
        assert result.data["cleared"] == 1
        # Verify empty
        data = json.loads(EVAL_FILE.read_text())
        assert len(data["evaluations"]) == 0
