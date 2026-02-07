"""Tests for SkillDependencyAnalyzer."""

import pytest
import asyncio
from singularity.skills.skill_analyzer import SkillDependencyAnalyzer


@pytest.fixture
def analyzer():
    return SkillDependencyAnalyzer()


@pytest.mark.asyncio
async def test_scan(analyzer):
    result = await analyzer.execute("scan", {})
    assert result.success
    assert result.data["total_skills"] > 10
    assert result.data["total_actions"] > 0
    assert "categories" in result.data


@pytest.mark.asyncio
async def test_dependency_graph_full(analyzer):
    result = await analyzer.execute("dependency_graph", {})
    assert result.success
    assert "edges" in result.data
    assert "most_depended_on" in result.data


@pytest.mark.asyncio
async def test_dependency_graph_filtered(analyzer):
    await analyzer.execute("scan", {})
    result = await analyzer.execute("dependency_graph", {"skill_id": "base"})
    assert result.success
    assert result.data["skill"] == "base"


@pytest.mark.asyncio
async def test_dependency_graph_bad_skill(analyzer):
    await analyzer.execute("scan", {})
    result = await analyzer.execute("dependency_graph", {"skill_id": "nonexistent_xyz"})
    assert not result.success


@pytest.mark.asyncio
async def test_find_circular(analyzer):
    result = await analyzer.execute("find_circular", {})
    assert result.success
    assert "cycles" in result.data
    assert isinstance(result.data["count"], int)


@pytest.mark.asyncio
async def test_find_orphans(analyzer):
    result = await analyzer.execute("find_orphans", {})
    assert result.success
    assert "orphans" in result.data
    assert "leaf_skills" in result.data
    assert "root_skills" in result.data


@pytest.mark.asyncio
async def test_capability_matrix(analyzer):
    result = await analyzer.execute("capability_matrix", {})
    assert result.success
    assert "matrix" in result.data
    assert result.data["total_actions"] > 0
    assert "duplicate_action_names" in result.data


@pytest.mark.asyncio
async def test_capability_matrix_filtered(analyzer):
    result = await analyzer.execute("capability_matrix", {"category": "self-improvement"})
    assert result.success
    for entry in result.data["matrix"]:
        assert entry["category"] == "self-improvement"


@pytest.mark.asyncio
async def test_suggest_consolidation(analyzer):
    result = await analyzer.execute("suggest_consolidation", {})
    assert result.success
    assert "suggestions" in result.data
    assert "by_type" in result.data


@pytest.mark.asyncio
async def test_health_score(analyzer):
    result = await analyzer.execute("health_score", {})
    assert result.success
    assert 0 <= result.data["score"] <= 100
    assert result.data["rating"] in ("Excellent", "Good", "Fair", "Needs Work", "Poor")
    assert "issues" in result.data
    assert "metrics" in result.data


@pytest.mark.asyncio
async def test_skill_summary(analyzer):
    result = await analyzer.execute("skill_summary", {"skill_id": "base"})
    assert result.success
    assert result.data["skill_id"] == "base"
    assert "role" in result.data
    assert "role_description" in result.data


@pytest.mark.asyncio
async def test_skill_summary_missing(analyzer):
    result = await analyzer.execute("skill_summary", {})
    assert not result.success


@pytest.mark.asyncio
async def test_unknown_action(analyzer):
    result = await analyzer.execute("nonexistent", {})
    assert not result.success


@pytest.mark.asyncio
async def test_manifest(analyzer):
    m = analyzer.manifest
    assert m.skill_id == "skill_analyzer"
    assert m.category == "self-improvement"
    assert len(m.actions) == 8
