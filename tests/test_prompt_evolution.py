"""Tests for PromptEvolutionSkill."""
import os
import json
import shutil
import pytest
from singularity.skills.prompt_evolution import PromptEvolutionSkill


@pytest.fixture
def skill(tmp_path):
    s = PromptEvolutionSkill()
    s._storage_dir = str(tmp_path)
    s._versions_file = os.path.join(str(tmp_path), "versions.json")
    s._outcomes_file = os.path.join(str(tmp_path), "outcomes.json")
    s._config_file = os.path.join(str(tmp_path), "config.json")
    # Wire up mock prompt hooks
    s._current_prompt = "You are a helpful assistant. Be concise."
    s.set_prompt_hooks(
        get_prompt=lambda: s._current_prompt,
        set_prompt=lambda p: setattr(s, "_current_prompt", p),
    )
    return s


@pytest.mark.asyncio
async def test_snapshot(skill):
    r = await skill.execute("snapshot", {"label": "baseline", "notes": "initial"})
    assert r.success
    assert r.data["label"] == "baseline"
    assert r.data["total_versions"] == 1


@pytest.mark.asyncio
async def test_snapshot_duplicate_label(skill):
    await skill.execute("snapshot", {"label": "v1"})
    r = await skill.execute("snapshot", {"label": "v1"})
    assert not r.success
    assert "already exists" in r.message


@pytest.mark.asyncio
async def test_snapshot_empty_label(skill):
    r = await skill.execute("snapshot", {"label": ""})
    assert not r.success


@pytest.mark.asyncio
async def test_record_outcome(skill):
    await skill.execute("snapshot", {"label": "v1"})
    r = await skill.execute("record_outcome", {"outcome": "success", "context": "test task"})
    assert r.success
    assert r.data["version"] == "v1"


@pytest.mark.asyncio
async def test_record_outcome_invalid(skill):
    r = await skill.execute("record_outcome", {"outcome": "maybe"})
    assert not r.success


@pytest.mark.asyncio
async def test_record_outcome_with_score(skill):
    await skill.execute("snapshot", {"label": "v1"})
    r = await skill.execute("record_outcome", {"outcome": "success", "score": 0.85})
    assert r.success
    outcomes = skill._load_outcomes()
    assert outcomes[0]["score"] == 0.85


@pytest.mark.asyncio
async def test_list_versions(skill):
    await skill.execute("snapshot", {"label": "v1"})
    skill._current_prompt = "Updated prompt"
    await skill.execute("snapshot", {"label": "v2"})
    r = await skill.execute("list_versions", {})
    assert r.success
    assert len(r.data["versions"]) == 2


@pytest.mark.asyncio
async def test_rollback(skill):
    await skill.execute("snapshot", {"label": "original"})
    original_prompt = skill._current_prompt
    skill._current_prompt = "Changed prompt"
    await skill.execute("snapshot", {"label": "changed"})
    r = await skill.execute("rollback", {"version": "original"})
    assert r.success
    assert skill._current_prompt == original_prompt


@pytest.mark.asyncio
async def test_rollback_not_found(skill):
    r = await skill.execute("rollback", {"version": "nonexistent"})
    assert not r.success


@pytest.mark.asyncio
async def test_compare(skill):
    await skill.execute("snapshot", {"label": "v1"})
    await skill.execute("record_outcome", {"outcome": "success"})
    await skill.execute("record_outcome", {"outcome": "failure"})
    skill._current_prompt = "Better prompt"
    await skill.execute("snapshot", {"label": "v2"})
    await skill.execute("record_outcome", {"outcome": "success"})
    await skill.execute("record_outcome", {"outcome": "success"})
    r = await skill.execute("compare", {"version_a": "v2", "version_b": "v1"})
    assert r.success
    assert r.data["winner"] == "v2"


@pytest.mark.asyncio
async def test_compare_insufficient_data(skill):
    r = await skill.execute("compare", {})
    assert not r.success


@pytest.mark.asyncio
async def test_best_version(skill):
    await skill.execute("snapshot", {"label": "v1"})
    for _ in range(3):
        await skill.execute("record_outcome", {"outcome": "success"})
    skill._current_prompt = "Another"
    await skill.execute("snapshot", {"label": "v2"})
    for _ in range(3):
        await skill.execute("record_outcome", {"outcome": "failure"})
    r = await skill.execute("best_version", {"min_outcomes": 3})
    assert r.success
    assert r.data["best"]["label"] == "v1"


@pytest.mark.asyncio
async def test_best_version_no_data(skill):
    r = await skill.execute("best_version", {})
    assert r.success
    assert r.data["best"] is None


@pytest.mark.asyncio
async def test_mutate(skill):
    await skill.execute("snapshot", {"label": "v1"})
    r = await skill.execute("mutate", {
        "section": "Be concise.",
        "replacement": "Be verbose and detailed.",
        "label": "verbose-v1",
    })
    assert r.success
    assert "verbose and detailed" in skill._current_prompt


@pytest.mark.asyncio
async def test_mutate_section_not_found(skill):
    await skill.execute("snapshot", {"label": "v1"})
    r = await skill.execute("mutate", {
        "section": "nonexistent section",
        "replacement": "new text",
        "label": "v2",
    })
    assert not r.success


@pytest.mark.asyncio
async def test_active_version(skill):
    r = await skill.execute("active_version", {})
    assert r.success
    assert r.data["active_version"] is None

    await skill.execute("snapshot", {"label": "v1"})
    r = await skill.execute("active_version", {})
    assert r.success
    assert r.data["label"] == "v1"


@pytest.mark.asyncio
async def test_unknown_action(skill):
    r = await skill.execute("nonexistent", {})
    assert not r.success


@pytest.mark.asyncio
async def test_manifest(skill):
    m = skill.manifest
    assert m.skill_id == "prompt_evolution"
    assert len(m.actions) == 8
