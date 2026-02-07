"""Tests for KnowledgeSharingSkill - cross-agent knowledge propagation."""
import json
import pytest
from pathlib import Path

from singularity.skills.knowledge_sharing import (
    KnowledgeSharingSkill, KNOWLEDGE_FILE, KnowledgeCategory,
)


@pytest.fixture(autouse=True)
def clean_store(tmp_path, monkeypatch):
    """Use temp file for knowledge store."""
    kf = tmp_path / "knowledge_store.json"
    monkeypatch.setattr("singularity.skills.knowledge_sharing.KNOWLEDGE_FILE", kf)
    return kf


@pytest.fixture
def skill():
    s = KnowledgeSharingSkill()
    s.set_agent_id("agent_A")
    return s


@pytest.mark.asyncio
async def test_publish_and_query(skill):
    """Publish knowledge and query it back."""
    r = await skill.execute("publish", {
        "content": "Use batch API calls to reduce cost by 40%",
        "category": "optimization",
        "confidence": 0.9,
        "tags": ["cost", "api"],
    })
    assert r.success
    assert r.data["action"] == "published"

    q = await skill.execute("query", {"category": "optimization"})
    assert q.success
    assert len(q.data["entries"]) == 1
    assert "batch API" in q.data["entries"][0]["content"]


@pytest.mark.asyncio
async def test_publish_dedup(skill):
    """Same content+category = same entry, updated."""
    await skill.execute("publish", {"content": "Foo works", "category": "strategy", "confidence": 0.6})
    await skill.execute("publish", {"content": "Foo works", "category": "strategy", "confidence": 0.8})
    q = await skill.execute("query", {"category": "strategy"})
    assert len(q.data["entries"]) == 1
    assert q.data["entries"][0]["confidence"] == 0.8


@pytest.mark.asyncio
async def test_publish_lower_confidence_skipped(skill):
    """Lower confidence update is skipped."""
    await skill.execute("publish", {"content": "X is good", "category": "strategy", "confidence": 0.9})
    r = await skill.execute("publish", {"content": "X is good", "category": "strategy", "confidence": 0.5})
    assert r.data["action"] == "skipped"


@pytest.mark.asyncio
async def test_query_text_search(skill):
    """Text search filters by content."""
    await skill.execute("publish", {"content": "Alpha strategy works", "category": "strategy"})
    await skill.execute("publish", {"content": "Beta approach fails", "category": "warning"})
    q = await skill.execute("query", {"search": "alpha"})
    assert len(q.data["entries"]) == 1


@pytest.mark.asyncio
async def test_query_exclude_own(skill):
    """Exclude own entries when querying."""
    await skill.execute("publish", {"content": "My finding", "category": "strategy"})
    q = await skill.execute("query", {"exclude_own": True})
    assert len(q.data["entries"]) == 0


@pytest.mark.asyncio
async def test_endorse_boosts_confidence(skill):
    """Endorsing boosts confidence."""
    await skill.execute("publish", {"content": "Good idea", "category": "strategy", "confidence": 0.7})
    q = await skill.execute("query", {})
    eid = q.data["entries"][0]["id"]
    skill.set_agent_id("agent_B")
    r = await skill.execute("endorse", {"entry_id": eid, "endorsement": True})
    assert r.success
    q2 = await skill.execute("query", {})
    assert q2.data["entries"][0]["confidence"] == pytest.approx(0.75, abs=0.01)


@pytest.mark.asyncio
async def test_dispute_lowers_confidence(skill):
    """Disputing lowers confidence."""
    await skill.execute("publish", {"content": "Bad idea", "category": "warning", "confidence": 0.8})
    q = await skill.execute("query", {})
    eid = q.data["entries"][0]["id"]
    skill.set_agent_id("agent_B")
    await skill.execute("endorse", {"entry_id": eid, "endorsement": False})
    q2 = await skill.execute("query", {})
    assert q2.data["entries"][0]["confidence"] == pytest.approx(0.7, abs=0.01)


@pytest.mark.asyncio
async def test_digest(skill):
    """Digest returns recent items."""
    await skill.execute("publish", {"content": "Recent insight", "category": "strategy"})
    d = await skill.execute("digest", {"since_hours": 1})
    assert d.success
    assert d.data["total_recent"] == 1


@pytest.mark.asyncio
async def test_stats(skill):
    """Stats returns store summary."""
    await skill.execute("publish", {"content": "A", "category": "strategy"})
    await skill.execute("publish", {"content": "B", "category": "warning"})
    s = await skill.execute("stats", {})
    assert s.success
    assert s.data["total_entries"] == 2
    assert s.data["by_category"]["strategy"] == 1


@pytest.mark.asyncio
async def test_invalid_category(skill):
    """Invalid category is rejected."""
    r = await skill.execute("publish", {"content": "X", "category": "invalid"})
    assert not r.success


@pytest.mark.asyncio
async def test_subscribe(skill):
    """Subscribe to specific categories."""
    r = await skill.execute("subscribe", {"categories": ["strategy", "warning"]})
    assert r.success
    assert len(r.data["subscriptions"]) == 2
