#!/usr/bin/env python3
"""Tests for FunctionMarketplaceSkill."""

import pytest
import json
from unittest.mock import patch
from singularity.skills.function_marketplace import (
    FunctionMarketplaceSkill, MARKETPLACE_FILE, SERVERLESS_FILE, CATEGORIES,
)


@pytest.fixture
def skill(tmp_path, monkeypatch):
    monkeypatch.setattr("singularity.skills.function_marketplace.MARKETPLACE_FILE", tmp_path / "marketplace.json")
    monkeypatch.setattr("singularity.skills.function_marketplace.SERVERLESS_FILE", tmp_path / "serverless.json")
    return FunctionMarketplaceSkill()


@pytest.fixture
def sample_serverless_data():
    return {
        "functions": {
            "summarize_text": {
                "name": "summarize_text",
                "code": "async def handler(req): return {'status_code': 200, 'body': {'summary': 'hello'}}",
                "route": "/summarize",
                "method": "POST",
                "description": "Summarize any text input",
                "enabled": True,
            },
        },
        "stats": {},
    }


@pytest.mark.asyncio
async def test_publish_with_code(skill):
    result = await skill.execute("publish", {
        "function_name": "hello_world",
        "agent_name": "eve",
        "category": "utility",
        "tags": ["demo", "hello"],
        "code": "async def handler(req): return {'status_code': 200, 'body': 'hello'}",
        "description": "A simple hello function",
    })
    assert result.success
    assert result.data["listing_id"].startswith("fn_")


@pytest.mark.asyncio
async def test_publish_from_serverless(skill, sample_serverless_data, tmp_path):
    sf_path = tmp_path / "serverless.json"
    sf_path.write_text(json.dumps(sample_serverless_data))
    result = await skill.execute("publish", {
        "function_name": "summarize_text",
        "agent_name": "eve",
        "category": "text_processing",
    })
    assert result.success


@pytest.mark.asyncio
async def test_publish_duplicate(skill):
    await skill.execute("publish", {"function_name": "f1", "agent_name": "eve",
                                     "code": "code", "category": "utility"})
    result = await skill.execute("publish", {"function_name": "f1", "agent_name": "eve",
                                              "code": "code", "category": "utility"})
    assert not result.success
    assert "already published" in result.message.lower()


@pytest.mark.asyncio
async def test_browse_all(skill):
    await skill.execute("publish", {"function_name": "f1", "agent_name": "eve", "code": "c1"})
    await skill.execute("publish", {"function_name": "f2", "agent_name": "adam", "code": "c2"})
    result = await skill.execute("browse", {})
    assert result.success
    assert result.data["count"] == 2


@pytest.mark.asyncio
async def test_browse_search(skill):
    await skill.execute("publish", {"function_name": "text_summarizer", "agent_name": "eve",
                                     "code": "c", "description": "Summarizes text", "tags": ["nlp"]})
    await skill.execute("publish", {"function_name": "data_parser", "agent_name": "adam", "code": "c"})
    result = await skill.execute("browse", {"search": "summarize"})
    assert result.success
    assert result.data["count"] == 1
    assert result.data["listings"][0]["function_name"] == "text_summarizer"


@pytest.mark.asyncio
async def test_import_function(skill):
    pub = await skill.execute("publish", {"function_name": "util_fn", "agent_name": "eve",
                                           "code": "async def handler(r): pass",
                                           "price_per_import": 0.50})
    listing_id = pub.data["listing_id"]
    result = await skill.execute("import_function", {
        "listing_id": listing_id, "importer_agent": "adam",
    })
    assert result.success
    assert result.data["price_paid"] == 0.50
    # Verify listing import count
    listing = await skill.execute("get_listing", {"listing_id": listing_id})
    assert listing.data["import_count"] == 1
    assert listing.data["total_revenue"] == 0.50


@pytest.mark.asyncio
async def test_import_own_function_blocked(skill):
    pub = await skill.execute("publish", {"function_name": "my_fn", "agent_name": "eve", "code": "c"})
    result = await skill.execute("import_function", {
        "listing_id": pub.data["listing_id"], "importer_agent": "eve",
    })
    assert not result.success
    assert "own function" in result.message.lower()


@pytest.mark.asyncio
async def test_rate_function(skill):
    pub = await skill.execute("publish", {"function_name": "rate_me", "agent_name": "eve", "code": "c"})
    lid = pub.data["listing_id"]
    result = await skill.execute("rate", {"listing_id": lid, "agent_name": "adam",
                                           "rating": 4, "review": "Works well"})
    assert result.success
    assert result.data["avg_rating"] == 4.0


@pytest.mark.asyncio
async def test_rate_invalid(skill):
    pub = await skill.execute("publish", {"function_name": "r_fn", "agent_name": "eve", "code": "c"})
    result = await skill.execute("rate", {"listing_id": pub.data["listing_id"],
                                           "agent_name": "adam", "rating": 6})
    assert not result.success


@pytest.mark.asyncio
async def test_rate_own_blocked(skill):
    pub = await skill.execute("publish", {"function_name": "self_rate", "agent_name": "eve", "code": "c"})
    result = await skill.execute("rate", {"listing_id": pub.data["listing_id"],
                                           "agent_name": "eve", "rating": 5})
    assert not result.success


@pytest.mark.asyncio
async def test_featured(skill):
    pub = await skill.execute("publish", {"function_name": "popular", "agent_name": "eve", "code": "c"})
    result = await skill.execute("featured", {"limit": 5})
    assert result.success
    assert len(result.data["featured"]) == 1


@pytest.mark.asyncio
async def test_my_publications(skill):
    await skill.execute("publish", {"function_name": "f1", "agent_name": "eve", "code": "c"})
    await skill.execute("publish", {"function_name": "f2", "agent_name": "eve", "code": "c2"})
    await skill.execute("publish", {"function_name": "f3", "agent_name": "adam", "code": "c3"})
    result = await skill.execute("my_publications", {"agent_name": "eve"})
    assert result.success
    assert len(result.data["publications"]) == 2


@pytest.mark.asyncio
async def test_unpublish(skill):
    pub = await skill.execute("publish", {"function_name": "remove_me", "agent_name": "eve", "code": "c"})
    lid = pub.data["listing_id"]
    result = await skill.execute("unpublish", {"listing_id": lid, "agent_name": "eve"})
    assert result.success
    browse = await skill.execute("browse", {})
    assert browse.data["count"] == 0  # Unpublished not shown


@pytest.mark.asyncio
async def test_unpublish_wrong_agent(skill):
    pub = await skill.execute("publish", {"function_name": "protected", "agent_name": "eve", "code": "c"})
    result = await skill.execute("unpublish", {"listing_id": pub.data["listing_id"], "agent_name": "adam"})
    assert not result.success


@pytest.mark.asyncio
async def test_status(skill):
    await skill.execute("publish", {"function_name": "s1", "agent_name": "eve", "code": "c"})
    result = await skill.execute("status", {})
    assert result.success
    assert result.data["active_listings"] == 1
    assert result.data["total_published"] == 1
