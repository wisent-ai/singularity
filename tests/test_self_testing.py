#!/usr/bin/env python3
"""Tests for SelfTestingSkill."""

import pytest
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
from singularity.skills.self_testing import SelfTestingSkill, HISTORY_FILE


@pytest.fixture
def skill(tmp_path, monkeypatch):
    monkeypatch.setattr("singularity.skills.self_testing.HISTORY_FILE", tmp_path / "test_history.json")
    s = SelfTestingSkill()
    s._ensure_data()
    return s


@pytest.mark.asyncio
async def test_manifest(skill):
    m = skill.manifest
    assert m.skill_id == "self_testing"
    assert m.category == "dev"
    actions = [a.name for a in m.actions]
    assert "run_tests" in actions
    assert "diagnose" in actions
    assert "health" in actions
    assert "discover" in actions


@pytest.mark.asyncio
async def test_unknown_action(skill):
    result = await skill.execute("nonexistent", {})
    assert not result.success
    assert "Unknown action" in result.message


@pytest.mark.asyncio
async def test_parse_summary(skill):
    output = "======= 5 passed, 2 failed, 1 error, 3 skipped in 2.5s ======="
    counts = skill._parse_summary(output)
    assert counts["passed"] == 5
    assert counts["failed"] == 2
    assert counts["errors"] == 1
    assert counts["skipped"] == 3


@pytest.mark.asyncio
async def test_parse_failures(skill):
    output = """FAILED tests/test_foo.py::test_bar - AssertionError: expected 1 got 2
FAILED tests/test_baz.py::test_qux - TypeError: bad type
ERROR tests/test_err.py::test_boom - ImportError: no module"""
    failures = skill._parse_failures(output)
    assert len(failures) >= 3
    assert any(f["test"] == "test_bar" for f in failures)
    assert any(f["type"] == "error" for f in failures)


@pytest.mark.asyncio
async def test_diagnose_import_error(skill):
    failure = {"test": "test_foo", "file": "tests/test_foo.py", "error": "ModuleNotFoundError: No module named 'xyz'"}
    result = await skill.execute("diagnose", {"failure": failure})
    assert result.success
    assert result.data["category"] == "import_error"
    assert result.data["severity"] == "high"


@pytest.mark.asyncio
async def test_diagnose_assertion_error(skill):
    failure = {"test": "test_calc", "file": "tests/test_calc.py", "error": "AssertionError: assert 1 == 2"}
    result = await skill.execute("diagnose", {"failure": failure})
    assert result.success
    assert result.data["category"] == "assertion_failure"


@pytest.mark.asyncio
async def test_diagnose_string_input(skill):
    result = await skill.execute("diagnose", {"failure": "SyntaxError: invalid syntax"})
    assert result.success
    assert result.data["category"] == "syntax_error"


@pytest.mark.asyncio
async def test_health_empty(skill):
    result = await skill.execute("health", {})
    assert result.success
    assert "No test runs" in result.message


@pytest.mark.asyncio
async def test_health_with_data(skill, tmp_path, monkeypatch):
    history = [
        {"timestamp": "2024-01-01T00:00:00", "type": "full_suite", "total_tests": 10,
         "passed": 10, "failed": 0, "errors": 0, "skipped": 0,
         "duration_seconds": 5.0, "pass_rate": 100.0, "failed_tests": []},
        {"timestamp": "2024-01-02T00:00:00", "type": "full_suite", "total_tests": 10,
         "passed": 8, "failed": 2, "errors": 0, "skipped": 0,
         "duration_seconds": 6.0, "pass_rate": 80.0, "failed_tests": ["test_a", "test_b"]},
    ]
    skill._save_history(history)
    result = await skill.execute("health", {"limit": 10})
    assert result.success
    assert result.data["health"]["current_pass_rate"] == 80.0
    assert set(result.data["health"]["regressions"]) == {"test_a", "test_b"}


@pytest.mark.asyncio
async def test_discover(skill, tmp_path):
    # Create fake test files
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_one.py").write_text("def test_a(): pass\ndef test_b(): pass\n")
    (tests_dir / "test_two.py").write_text("async def test_c(): pass\n")
    (tests_dir / "not_a_test.py").write_text("def helper(): pass\n")

    result = await skill.execute("discover", {"project_root": str(tmp_path)})
    assert result.success
    assert result.data["test_file_count"] >= 2
    assert result.data["total_test_count"] >= 3


@pytest.mark.asyncio
async def test_run_file_not_found(skill):
    result = await skill.execute("run_file", {"file_path": "/nonexistent/test_foo.py"})
    assert not result.success
    assert "not found" in result.message


@pytest.mark.asyncio
async def test_run_file_missing_param(skill):
    result = await skill.execute("run_file", {})
    assert not result.success
    assert "required" in result.message.lower()
