#!/usr/bin/env python3
"""Tests for NaturalLanguageRouter skill."""

import json
import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
from pathlib import Path

from singularity.skills.nl_router import (
    NaturalLanguageRouter,
    CatalogEntry,
    _tokenize,
    _extract_keywords_from_desc,
    ROUTER_DATA,
)
from singularity.skills.base import SkillResult, SkillContext, SkillRegistry


@pytest.fixture
def router(tmp_path, monkeypatch):
    """Create a router with tmp storage."""
    data_file = tmp_path / "nl_router.json"
    monkeypatch.setattr("singularity.skills.nl_router.ROUTER_DATA", data_file)
    r = NaturalLanguageRouter()
    return r


@pytest.fixture
def mock_context():
    """Create a mock skill context with fake skills."""
    registry = MagicMock(spec=SkillRegistry)

    # Create mock skills with manifests
    code_review = MagicMock()
    code_review.manifest.skill_id = "code_review"
    code_review.manifest.name = "Code Review"
    code_review.manifest.category = "dev"
    action1 = MagicMock()
    action1.name = "analyze"
    action1.description = "Analyze code for bugs and quality issues"
    action2 = MagicMock()
    action2.name = "security_scan"
    action2.description = "Scan code for security vulnerabilities"
    code_review.get_actions.return_value = [action1, action2]

    content = MagicMock()
    content.manifest.skill_id = "content"
    content.manifest.name = "Content Creation"
    content.manifest.category = "content"
    action3 = MagicMock()
    action3.name = "write"
    action3.description = "Write an article or blog post on a topic"
    content.get_actions.return_value = [action3]

    shell = MagicMock()
    shell.manifest.skill_id = "shell"
    shell.manifest.name = "Shell"
    shell.manifest.category = "dev"
    action4 = MagicMock()
    action4.name = "execute"
    action4.description = "Execute a shell command in the terminal"
    shell.get_actions.return_value = [action4]

    skills_map = {"code_review": code_review, "content": content, "shell": shell}
    registry.skills = skills_map

    ctx = SkillContext(
        registry=registry,
        agent_name="TestAgent",
        agent_ticker="TEST",
    )
    # Mock the methods
    ctx.list_skills = MagicMock(return_value=list(skills_map.keys()))
    ctx.get_skill = MagicMock(side_effect=lambda sid: skills_map.get(sid))

    return ctx


def test_tokenize():
    assert "code" in _tokenize("Review this code")
    assert "the" not in _tokenize("Review the code")
    assert _tokenize("") == []


def test_catalog_entry_creation():
    entry = CatalogEntry("test", "action", "test action desc", "dev")
    assert entry.skill_id == "test"
    assert entry.success_rate == 0.5  # No data = neutral


def test_catalog_entry_success_rate():
    entry = CatalogEntry("test", "action", "test action desc", "dev")
    entry.success_count = 8
    entry.fail_count = 2
    assert entry.success_rate == 0.8


def test_catalog_entry_serialization():
    entry = CatalogEntry("test", "action", "test action desc", "dev", ["test", "action"])
    d = entry.to_dict()
    restored = CatalogEntry.from_dict(d)
    assert restored.skill_id == "test"
    assert restored.action == "action"
    assert restored.keywords == ["test", "action"]


def test_build_catalog(router, mock_context):
    router.context = mock_context
    router.build_catalog()
    assert len(router._catalog) >= 3  # At least 3 actions from mock skills


def test_route_code_review(router, mock_context):
    router.context = mock_context
    router.build_catalog()
    matches = router.route("analyze this code for bugs")
    assert len(matches) > 0
    assert matches[0]["skill_id"] == "code_review"


def test_route_content(router, mock_context):
    router.context = mock_context
    router.build_catalog()
    matches = router.route("write a blog article about AI")
    assert len(matches) > 0
    assert matches[0]["skill_id"] == "content"


def test_route_empty_query(router, mock_context):
    router.context = mock_context
    router.build_catalog()
    matches = router.route("")
    assert matches == []


def test_route_no_catalog(router):
    matches = router.route("test query")
    assert matches == []  # No context = no catalog


def test_record_outcome(router, mock_context):
    router.context = mock_context
    router.build_catalog()
    router.record_outcome("code_review", "analyze", True, "test query")
    # Find the entry
    for entry in router._catalog:
        if entry.skill_id == "code_review" and entry.action == "analyze":
            assert entry.success_count == 1
            assert entry.weight_boost > 0
            break


def test_record_outcome_failure(router, mock_context):
    router.context = mock_context
    router.build_catalog()
    router.record_outcome("code_review", "analyze", False, "test query")
    for entry in router._catalog:
        if entry.skill_id == "code_review" and entry.action == "analyze":
            assert entry.fail_count == 1
            assert entry.weight_boost < 0
            break


@pytest.mark.asyncio
async def test_execute_route(router, mock_context):
    router.context = mock_context
    router.build_catalog()
    result = await router.execute("route", {"query": "scan code for security issues"})
    assert result.success
    assert "matches" in result.data
    assert result.data["best"]["skill_id"] == "code_review"


@pytest.mark.asyncio
async def test_execute_catalog(router, mock_context):
    router.context = mock_context
    router.build_catalog()
    result = await router.execute("catalog", {})
    assert result.success
    assert result.data["total"] >= 3


@pytest.mark.asyncio
async def test_execute_stats_empty(router):
    result = await router.execute("stats", {})
    assert result.success
    assert result.data["total_routes"] == 0


@pytest.mark.asyncio
async def test_execute_rebuild(router, mock_context):
    router.context = mock_context
    result = await router.execute("rebuild", {})
    assert result.success
    assert "new_size" in result.data


@pytest.mark.asyncio
async def test_execute_unknown_action(router):
    result = await router.execute("nonexistent", {})
    assert not result.success


@pytest.mark.asyncio
async def test_manifest(router):
    m = router.manifest
    assert m.skill_id == "nl_router"
    assert len(m.actions) == 6


@pytest.mark.asyncio
async def test_route_and_execute(router, mock_context):
    router.context = mock_context
    router.build_catalog()
    mock_context.call_skill = AsyncMock(return_value=SkillResult(
        success=True, message="Code analyzed", data={"score": 85}
    ))
    result = await router.execute("route_and_execute", {"query": "analyze code quality"})
    assert result.success
    assert "routed_to" in result.data
    mock_context.call_skill.assert_called_once()
