"""Tests for SkillAutoPublisherSkill."""

import pytest
import json
from pathlib import Path
from singularity.skills.skill_auto_publisher import (
    SkillAutoPublisherSkill, PUBLISHER_FILE,
)


@pytest.fixture(autouse=True)
def clean_data():
    if PUBLISHER_FILE.exists():
        PUBLISHER_FILE.unlink()
    yield
    if PUBLISHER_FILE.exists():
        PUBLISHER_FILE.unlink()


@pytest.fixture
def skill():
    return SkillAutoPublisherSkill()


@pytest.mark.asyncio
async def test_scan_finds_skills(skill):
    r = await skill.execute("scan", {})
    assert r.success
    assert r.data["total_scanned"] > 0
    # Should find at least a few known skills
    all_ids = r.data["new"] + r.data["updated"] + r.data["published"]
    assert len(all_ids) > 0


@pytest.mark.asyncio
async def test_publish_one(skill):
    # First scan to find a skill
    scan = await skill.execute("scan", {})
    if not scan.data["new"]:
        pytest.skip("No new skills to publish")
    first_id = scan.data["new"][0]
    r = await skill.execute("publish_one", {"skill_id": first_id, "price": 1.0})
    assert r.success
    assert r.data["skill_id"] == first_id


@pytest.mark.asyncio
async def test_publish_one_not_found(skill):
    r = await skill.execute("publish_one", {"skill_id": "nonexistent_skill_xyz"})
    assert not r.success


@pytest.mark.asyncio
async def test_publish_all_dry_run(skill):
    r = await skill.execute("publish_all", {"dry_run": True})
    assert r.success
    assert "would_publish" in r.data
    assert len(r.data["would_publish"]) > 0


@pytest.mark.asyncio
async def test_publish_all(skill):
    r = await skill.execute("publish_all", {"agent_id": "test-agent"})
    assert r.success
    assert r.data["total_published"] > 0


@pytest.mark.asyncio
async def test_diff_shows_new(skill):
    r = await skill.execute("diff", {})
    assert r.success
    assert len(r.data["new"]) > 0


@pytest.mark.asyncio
async def test_diff_after_publish(skill):
    await skill.execute("publish_all", {})
    r = await skill.execute("diff", {})
    assert r.success
    assert len(r.data["new"]) == 0
    assert len(r.data["unchanged"]) > 0


@pytest.mark.asyncio
async def test_sync_dry_run(skill):
    r = await skill.execute("sync", {"dry_run": True})
    assert r.success
    assert "would_publish" in r.data


@pytest.mark.asyncio
async def test_sync_publishes(skill):
    r = await skill.execute("sync", {"agent_id": "test-agent"})
    assert r.success
    assert len(r.data["published"]) > 0


@pytest.mark.asyncio
async def test_unpublish(skill):
    await skill.execute("publish_all", {})
    scan = await skill.execute("scan", {})
    if not scan.data["published"]:
        pytest.skip("Nothing published")
    sid = scan.data["published"][0]
    r = await skill.execute("unpublish", {"skill_id": sid})
    assert r.success


@pytest.mark.asyncio
async def test_unpublish_not_published(skill):
    r = await skill.execute("unpublish", {"skill_id": "not_published_xyz"})
    assert not r.success


@pytest.mark.asyncio
async def test_status(skill):
    r = await skill.execute("status", {})
    assert r.success
    assert r.data["summary"]["total"] > 0
    assert r.data["summary"]["unpublished"] > 0


@pytest.mark.asyncio
async def test_status_after_publish(skill):
    await skill.execute("publish_all", {})
    r = await skill.execute("status", {})
    assert r.success
    assert r.data["summary"]["published"] > 0


@pytest.mark.asyncio
async def test_set_pricing(skill):
    r = await skill.execute("set_pricing", {
        "default_price": 0.50,
        "category_prices": {"revenue": 2.0},
        "free_categories": ["core"],
    })
    assert r.success
    assert r.data["pricing_rules"]["default_price"] == 0.50


@pytest.mark.asyncio
async def test_set_pricing_exclude(skill):
    r = await skill.execute("set_pricing", {
        "exclude_skills": ["skill_auto_publisher"],
    })
    assert r.success
    # Now scan - excluded skill should not appear as new
    r2 = await skill.execute("scan", {})
    assert "skill_auto_publisher" not in r2.data["new"]


@pytest.mark.asyncio
async def test_manifest(skill):
    m = skill.manifest
    assert m.skill_id == "skill_auto_publisher"
    assert len(m.actions) == 8


@pytest.mark.asyncio
async def test_unknown_action(skill):
    r = await skill.execute("bogus", {})
    assert not r.success


@pytest.mark.asyncio
async def test_initialize(skill):
    assert await skill.initialize()
