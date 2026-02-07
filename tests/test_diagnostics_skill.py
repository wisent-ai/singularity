"""Tests for DiagnosticsSkill."""

import asyncio
import os
import pytest
from unittest.mock import MagicMock
from singularity.skills.diagnostics import DiagnosticsSkill, _format_duration


@pytest.fixture
def skill():
    return DiagnosticsSkill()


@pytest.fixture
def skill_with_registry():
    registry = MagicMock()
    mock_skill = MagicMock()
    mock_skill.manifest.skill_id = "test"
    mock_skill.manifest.name = "TestSkill"
    mock_skill.manifest.version = "1.0"
    mock_skill.manifest.category = "test"
    mock_skill.manifest.actions = []
    mock_skill.manifest.required_credentials = []
    mock_skill.initialized = True
    mock_skill._usage_count = 5
    mock_skill._total_cost = 0.01
    registry._skills = {"test": mock_skill}
    return DiagnosticsSkill(registry=registry)


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


class TestSystemInfo:
    def test_returns_system_info(self, skill):
        result = run(skill.execute("system_info", {}))
        assert result.success
        assert "python_version" in result.data
        assert "cpu_count" in result.data
        assert "platform" in result.data
        assert "pid" in result.data


class TestCheckResources:
    def test_returns_resource_data(self, skill):
        result = run(skill.execute("check_resources", {}))
        assert result.success
        assert isinstance(result.data, dict)

    def test_has_disk_info(self, skill):
        result = run(skill.execute("check_resources", {}))
        assert "disk" in result.data


class TestCheckCredentials:
    def test_reports_credential_status(self, skill):
        result = run(skill.execute("check_credentials", {}))
        assert result.success
        assert "available_count" in result.data
        assert "missing_count" in result.data

    def test_detects_set_env_var(self, skill):
        os.environ["GITHUB_TOKEN"] = "test_token_12345"
        try:
            result = run(skill.execute("check_credentials", {}))
            assert "GITHUB_TOKEN" in result.data["available"]
        finally:
            del os.environ["GITHUB_TOKEN"]


class TestCheckDependencies:
    def test_finds_installed_packages(self, skill):
        result = run(skill.execute("check_dependencies", {"packages": ["json", "os"]}))
        assert result.success
        assert result.data["installed_count"] >= 0

    def test_finds_missing_packages(self, skill):
        result = run(skill.execute("check_dependencies", {"packages": ["nonexistent_pkg_xyz"]}))
        assert "nonexistent_pkg_xyz" in result.data["missing"]


class TestSkillStatus:
    def test_without_registry(self, skill):
        result = run(skill.execute("skill_status", {}))
        assert result.success

    def test_with_registry(self, skill_with_registry):
        result = run(skill_with_registry.execute("skill_status", {}))
        assert result.success
        assert result.data["skill_count"] == 1
        assert result.data["skills"][0]["id"] == "test"


class TestHealthCheck:
    def test_returns_health_score(self, skill):
        result = run(skill.execute("health_check", {}))
        assert result.success
        assert "score" in result.data
        assert "health" in result.data
        assert result.data["health"] in ["HEALTHY", "DEGRADED", "UNHEALTHY", "CRITICAL"]

    def test_health_score_range(self, skill):
        result = run(skill.execute("health_check", {}))
        assert 0 <= result.data["score"] <= 100


class TestCapabilityGaps:
    def test_identifies_gaps(self, skill):
        result = run(skill.execute("capability_gaps", {}))
        assert result.success
        assert "gaps" in result.data
        assert "total" in result.data

    def test_gaps_sorted_by_severity(self, skill):
        result = run(skill.execute("capability_gaps", {}))
        severities = [g["severity"] for g in result.data["gaps"]]
        order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        vals = [order.get(s, 3) for s in severities]
        assert vals == sorted(vals)


class TestRuntimeStats:
    def test_returns_uptime(self, skill):
        result = run(skill.execute("runtime_stats", {}))
        assert result.success
        assert result.data["uptime_seconds"] >= 0
        assert "uptime_human" in result.data

    def test_with_registry(self, skill_with_registry):
        result = run(skill_with_registry.execute("runtime_stats", {}))
        assert "total_skill_usage" in result.data


class TestSnapshots:
    def test_take_snapshot(self, skill):
        result = run(skill.execute("snapshot", {"label": "test_snap"}))
        assert result.success
        assert result.data["label"] == "test_snap"
        assert result.data["index"] == 0

    def test_compare_needs_two(self, skill):
        result = run(skill.execute("compare_snapshots", {}))
        assert not result.success

    def test_compare_two_snapshots(self, skill):
        run(skill.execute("snapshot", {"label": "first"}))
        run(skill.execute("snapshot", {"label": "second"}))
        result = run(skill.execute("compare_snapshots", {}))
        assert result.success
        assert "changes" in result.data


class TestFormatDuration:
    def test_seconds(self):
        assert _format_duration(30) == "30.0s"

    def test_minutes(self):
        assert _format_duration(125) == "2m 5s"

    def test_hours(self):
        assert _format_duration(3725) == "1h 2m"


class TestUnknownAction:
    def test_unknown_action(self, skill):
        result = run(skill.execute("nonexistent", {}))
        assert not result.success

    def test_manifest(self, skill):
        m = skill.manifest
        assert m.skill_id == "diagnostics"
        assert len(m.actions) == 10
        assert m.required_credentials == []
