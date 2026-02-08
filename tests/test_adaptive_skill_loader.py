"""Tests for AdaptiveSkillLoaderSkill."""
import pytest, json
from unittest.mock import AsyncMock, MagicMock
from pathlib import Path
from singularity.skills.adaptive_skill_loader import (
    AdaptiveSkillLoaderSkill, ADAPTIVE_LOADER_FILE,
)


@pytest.fixture(autouse=True)
def clean_data():
    if ADAPTIVE_LOADER_FILE.exists():
        ADAPTIVE_LOADER_FILE.unlink()
    yield
    if ADAPTIVE_LOADER_FILE.exists():
        ADAPTIVE_LOADER_FILE.unlink()


@pytest.fixture
def skill():
    return AdaptiveSkillLoaderSkill()


def _make_reflections():
    return [
        {
            "task": "deploy the web service",
            "actions_taken": ["shell:run", "docker:build", "k8s:deploy"],
            "success": True,
            "tags": ["deployment", "skill:shell"],
            "timestamp": "2026-02-08T10:00:00",
        },
        {
            "task": "deploy the API",
            "actions_taken": ["shell:run", "docker:build"],
            "success": True,
            "tags": ["deployment"],
            "timestamp": "2026-02-08T11:00:00",
        },
        {
            "task": "review pull request code",
            "actions_taken": ["code_review:analyze", "github:comment"],
            "success": True,
            "tags": ["code_review"],
            "timestamp": "2026-02-08T12:00:00",
        },
        {
            "task": "fix failing tests",
            "actions_taken": ["shell:run", "code_review:analyze"],
            "success": False,
            "tags": ["debugging"],
            "timestamp": "2026-02-08T13:00:00",
        },
    ]


@pytest.fixture
def skill_with_context():
    s = AdaptiveSkillLoaderSkill()
    ctx = MagicMock()
    ctx.call_skill = AsyncMock(return_value=MagicMock(
        success=True,
        data={"reflections": _make_reflections()},
    ))
    s.context = ctx
    return s


def test_manifest(skill):
    m = skill.manifest
    assert m.skill_id == "adaptive_skill_loader"
    assert m.category == "self_improvement"


def test_get_actions(skill):
    actions = skill.get_actions()
    names = [a.name for a in actions]
    assert "analyze" in names
    assert "recommend" in names
    assert "record_usage" in names
    assert "hot_skills" in names
    assert "cold_skills" in names
    assert "status" in names


@pytest.mark.asyncio
async def test_record_usage(skill):
    result = await skill.execute("record_usage", {
        "skill_id": "shell",
        "task_type": "deployment",
        "success": True,
        "co_skills": ["docker", "k8s"],
    })
    assert result.success
    assert skill._profiles["shell"]["total_uses"] == 1
    assert skill._profiles["shell"]["success_rate"] == 1.0
    assert "docker" in skill._profiles["shell"]["co_skills"]


@pytest.mark.asyncio
async def test_record_multiple_usages(skill):
    await skill.execute("record_usage", {"skill_id": "shell", "task_type": "deploy", "success": True})
    await skill.execute("record_usage", {"skill_id": "shell", "task_type": "debug", "success": False})
    assert skill._profiles["shell"]["total_uses"] == 2
    assert skill._profiles["shell"]["success_rate"] == 0.5


@pytest.mark.asyncio
async def test_recommend_with_profiles(skill):
    await skill.execute("record_usage", {"skill_id": "shell", "task_type": "deploy service", "success": True})
    await skill.execute("record_usage", {"skill_id": "docker", "task_type": "deploy service", "success": True, "co_skills": ["shell"]})

    result = await skill.execute("recommend", {"task_description": "deploy the service"})
    assert result.success
    ids = [r["skill_id"] for r in result.data["recommendations"]]
    assert "shell" in ids or "docker" in ids


@pytest.mark.asyncio
async def test_recommend_empty(skill):
    result = await skill.execute("recommend", {"task_description": "something new"})
    assert result.success
    assert len(result.data["recommendations"]) == 0


@pytest.mark.asyncio
async def test_analyze_with_context(skill_with_context):
    result = await skill_with_context.execute("analyze", {})
    assert result.success
    assert result.data["profiles_built"] > 0
    assert "shell" in skill_with_context._profiles


@pytest.mark.asyncio
async def test_analyze_no_context(skill):
    result = await skill.execute("analyze", {})
    assert result.success
    assert result.data["profiles_built"] == 0


@pytest.mark.asyncio
async def test_profile(skill):
    await skill.execute("record_usage", {"skill_id": "shell", "task_type": "deploy", "success": True})
    result = await skill.execute("profile", {"skill_id": "shell"})
    assert result.success
    assert result.data["profile"]["total_uses"] == 1


@pytest.mark.asyncio
async def test_profile_not_found(skill):
    result = await skill.execute("profile", {"skill_id": "nonexistent"})
    assert not result.success


@pytest.mark.asyncio
async def test_hot_skills(skill):
    await skill.execute("record_usage", {"skill_id": "shell", "task_type": "deploy", "success": True})
    await skill.execute("record_usage", {"skill_id": "shell", "task_type": "debug", "success": True})
    await skill.execute("record_usage", {"skill_id": "docker", "task_type": "deploy", "success": True})

    result = await skill.execute("hot_skills", {"limit": 5})
    assert result.success
    assert result.data["hot_skills"][0]["skill_id"] == "shell"


@pytest.mark.asyncio
async def test_cold_skills(skill):
    await skill.execute("record_usage", {"skill_id": "shell", "task_type": "deploy", "success": True})
    await skill.execute("record_usage", {"skill_id": "docker", "task_type": "deploy", "success": False})

    result = await skill.execute("cold_skills", {"limit": 5})
    assert result.success
    assert len(result.data["cold_skills"]) > 0


@pytest.mark.asyncio
async def test_configure(skill):
    result = await skill.execute("configure", {"key": "max_recommended_skills", "value": 5})
    assert result.success
    assert skill._config["max_recommended_skills"] == 5


@pytest.mark.asyncio
async def test_configure_invalid_key(skill):
    result = await skill.execute("configure", {"key": "nonexistent", "value": 5})
    assert not result.success


@pytest.mark.asyncio
async def test_status(skill):
    result = await skill.execute("status", {})
    assert result.success
    assert "stats" in result.data
    assert "config" in result.data


@pytest.mark.asyncio
async def test_unknown_action(skill):
    result = await skill.execute("nonexistent", {})
    assert not result.success


@pytest.mark.asyncio
async def test_persistence(skill):
    await skill.execute("record_usage", {"skill_id": "shell", "task_type": "deploy", "success": True})
    s2 = AdaptiveSkillLoaderSkill()
    assert "shell" in s2._profiles


@pytest.mark.asyncio
async def test_co_occurrence_in_recommendations(skill):
    # Record shell always used with docker
    for _ in range(3):
        await skill.execute("record_usage", {
            "skill_id": "shell", "task_type": "deploy", "success": True, "co_skills": ["docker"],
        })
        await skill.execute("record_usage", {
            "skill_id": "docker", "task_type": "deploy", "success": True, "co_skills": ["shell"],
        })
    result = await skill.execute("recommend", {"task_description": "deploy application"})
    assert result.success
    ids = [r["skill_id"] for r in result.data["recommendations"]]
    # Both shell and docker should be recommended
    assert "shell" in ids
    assert "docker" in ids
