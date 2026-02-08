"""Tests for SkillMarketplaceHub - inter-agent skill exchange."""
import pytest
import json
from pathlib import Path
from unittest.mock import patch
from singularity.skills.skill_marketplace_hub import SkillMarketplaceHub, HUB_FILE


@pytest.fixture
def hub(tmp_path):
    with patch("singularity.skills.skill_marketplace_hub.HUB_FILE", tmp_path / "skill_hub.json"):
        h = SkillMarketplaceHub()
        yield h


@pytest.fixture
def published_hub(hub):
    """Hub with a pre-published skill."""
    import asyncio
    result = asyncio.get_event_loop().run_until_complete(hub.execute("publish", {
        "skill_id": "code_review",
        "name": "Code Reviewer Pro",
        "description": "AI-powered code review with security analysis",
        "version": "1.0.0",
        "category": "dev",
        "tags": ["code", "review", "security"],
        "price": 0.50,
        "author_agent_id": "agent_alpha",
    }))
    assert result.success
    return hub, result.data["listing_id"]


@pytest.mark.asyncio
async def test_publish_skill(hub):
    result = await hub.execute("publish", {
        "skill_id": "data_analyzer",
        "name": "Data Analyzer",
        "description": "Analyzes CSV/JSON data and generates reports",
        "version": "2.0.0",
        "category": "data",
        "tags": ["data", "analytics"],
        "price": 1.00,
        "author_agent_id": "agent_one",
    })
    assert result.success
    assert "listing_id" in result.data
    assert result.data["listing"]["price"] == 1.0


@pytest.mark.asyncio
async def test_publish_duplicate_rejected(hub):
    await hub.execute("publish", {
        "skill_id": "dup_skill", "name": "Dup", "description": "d", "version": "1.0", "author_agent_id": "a1",
    })
    result = await hub.execute("publish", {
        "skill_id": "dup_skill", "name": "Dup2", "description": "d2", "version": "1.1", "author_agent_id": "a1",
    })
    assert not result.success
    assert "already published" in result.message


@pytest.mark.asyncio
async def test_browse_with_filters(hub):
    for i, cat in enumerate(["dev", "dev", "data", "revenue"]):
        await hub.execute("publish", {
            "skill_id": f"skill_{i}", "name": f"Skill {i}", "description": f"Desc {i}",
            "version": "1.0", "category": cat, "price": i * 0.5, "author_agent_id": "a",
        })
    result = await hub.execute("browse", {"category": "dev"})
    assert result.success
    assert len(result.data["listings"]) == 2

    result2 = await hub.execute("browse", {"max_price": 0.5})
    assert result2.success
    assert all(l["price"] <= 0.5 for l in result2.data["listings"])


@pytest.mark.asyncio
async def test_search_relevance(hub):
    await hub.execute("publish", {
        "skill_id": "code_review", "name": "Code Review Expert",
        "description": "Reviews code for bugs", "version": "1.0",
        "tags": ["review", "code"], "author_agent_id": "a",
    })
    await hub.execute("publish", {
        "skill_id": "data_clean", "name": "Data Cleaner",
        "description": "Cleans messy data", "version": "1.0",
        "tags": ["data"], "author_agent_id": "a",
    })
    result = await hub.execute("search", {"query": "code review"})
    assert result.success
    assert len(result.data["results"]) >= 1
    assert result.data["results"][0]["name"] == "Code Review Expert"


@pytest.mark.asyncio
async def test_install_skill(published_hub):
    hub, listing_id = published_hub
    result = await hub.execute("install", {
        "listing_id": listing_id, "installer_agent_id": "agent_beta",
    })
    assert result.success
    assert result.data["install_record"]["price_paid"] == 0.50


@pytest.mark.asyncio
async def test_install_duplicate_rejected(published_hub):
    hub, listing_id = published_hub
    await hub.execute("install", {"listing_id": listing_id, "installer_agent_id": "b"})
    result = await hub.execute("install", {"listing_id": listing_id, "installer_agent_id": "b"})
    assert not result.success
    assert "Already installed" in result.message


@pytest.mark.asyncio
async def test_review_requires_install(published_hub):
    hub, listing_id = published_hub
    result = await hub.execute("review", {
        "listing_id": listing_id, "rating": 5, "reviewer_agent_id": "stranger",
    })
    assert not result.success
    assert "must install" in result.message


@pytest.mark.asyncio
async def test_review_and_rating(published_hub):
    hub, listing_id = published_hub
    await hub.execute("install", {"listing_id": listing_id, "installer_agent_id": "r1"})
    await hub.execute("install", {"listing_id": listing_id, "installer_agent_id": "r2"})
    await hub.execute("review", {"listing_id": listing_id, "rating": 5, "reviewer_agent_id": "r1"})
    result = await hub.execute("review", {"listing_id": listing_id, "rating": 3, "reviewer_agent_id": "r2"})
    assert result.success
    assert result.data["avg_rating"] == 4.0


@pytest.mark.asyncio
async def test_earnings_report(published_hub):
    hub, listing_id = published_hub
    await hub.execute("install", {"listing_id": listing_id, "installer_agent_id": "c1"})
    await hub.execute("install", {"listing_id": listing_id, "installer_agent_id": "c2"})
    result = await hub.execute("earnings_report", {"author_agent_id": "agent_alpha"})
    assert result.success
    assert result.data["total_revenue"] == 1.0


@pytest.mark.asyncio
async def test_update_listing(published_hub):
    hub, listing_id = published_hub
    result = await hub.execute("update_listing", {
        "listing_id": listing_id, "price": 0.75, "version": "1.1.0",
    })
    assert result.success
    assert "price" in result.data["updated_fields"]
    assert result.data["listing"]["price"] == 0.75


@pytest.mark.asyncio
async def test_my_listings_and_installs(published_hub):
    hub, listing_id = published_hub
    await hub.execute("install", {"listing_id": listing_id, "installer_agent_id": "buyer1"})
    my = await hub.execute("my_listings", {"author_agent_id": "agent_alpha"})
    assert my.success
    assert my.data["total_listings"] == 1

    inst = await hub.execute("my_installs", {"installer_agent_id": "buyer1"})
    assert inst.success
    assert inst.data["total_installed"] == 1


@pytest.mark.asyncio
async def test_manifest(hub):
    m = hub.manifest
    assert m.skill_id == "skill_marketplace_hub"
    assert len(m.actions) == 10
