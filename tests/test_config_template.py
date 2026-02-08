"""Tests for ConfigTemplateSkill."""

import json
import pytest
import asyncio
from pathlib import Path
from unittest.mock import MagicMock
from singularity.skills.config_template import ConfigTemplateSkill, BUILTIN_TEMPLATES, CONFIG_TEMPLATE_FILE


@pytest.fixture(autouse=True)
def clean_data(tmp_path, monkeypatch):
    test_file = tmp_path / "config_templates.json"
    monkeypatch.setattr(
        "singularity.skills.config_template.CONFIG_TEMPLATE_FILE", test_file
    )
    yield test_file


@pytest.fixture
def skill():
    return ConfigTemplateSkill()


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def test_list_includes_builtins(skill):
    result = run(skill.execute("list", {}))
    assert result.success
    assert result.data["total"] >= len(BUILTIN_TEMPLATES)
    ids = [t["id"] for t in result.data["templates"]]
    assert "code_reviewer" in ids
    assert "revenue_agent" in ids


def test_list_filter_by_category(skill):
    result = run(skill.execute("list", {"category": "revenue"}))
    assert result.success
    for t in result.data["templates"]:
        assert t["category"] == "revenue"


def test_list_filter_by_tag(skill):
    result = run(skill.execute("list", {"tag": "monitoring"}))
    assert result.success
    for t in result.data["templates"]:
        assert "monitoring" in t["tags"]


def test_get_builtin(skill):
    result = run(skill.execute("get", {"template_id": "code_reviewer"}))
    assert result.success
    tmpl = result.data["template"]
    assert tmpl["name"] == "Code Reviewer"
    assert tmpl["builtin"] is True
    assert "code_review" in tmpl["skills_enabled"]


def test_get_missing(skill):
    result = run(skill.execute("get", {"template_id": "nonexistent"}))
    assert not result.success


def test_create_custom(skill):
    result = run(skill.execute("create", {
        "name": "My Template",
        "description": "Test template",
        "skills_enabled": ["shell", "filesystem"],
        "skills_disabled": ["crypto"],
        "parameters": {"model_preference": "haiku"},
        "tags": ["test"],
    }))
    assert result.success
    tid = result.data["template_id"]
    # Verify it's retrievable
    get_result = run(skill.execute("get", {"template_id": tid}))
    assert get_result.success
    assert get_result.data["template"]["name"] == "My Template"


def test_create_requires_name(skill):
    result = run(skill.execute("create", {}))
    assert not result.success


def test_snapshot(skill):
    # Create mock context with registry
    mock_ctx = MagicMock()
    mock_ctx._registry._skills = {"shell": MagicMock(), "filesystem": MagicMock()}
    mock_ctx.agent_name = "TestAgent"
    mock_ctx.agent_ticker = "TEST"
    skill.set_context(mock_ctx)
    result = run(skill.execute("snapshot", {"name": "Current State"}))
    assert result.success
    tmpl = result.data["template"]
    assert "shell" in tmpl["skills_enabled"]
    assert tmpl["category"] == "snapshot"


def test_apply_dry_run(skill):
    result = run(skill.execute("apply", {"template_id": "ops_monitor", "dry_run": True}))
    assert result.success
    assert result.data["dry_run"] is True
    assert "changes" in result.data


def test_apply(skill):
    result = run(skill.execute("apply", {"template_id": "revenue_agent"}))
    assert result.success
    assert result.data["active_template"] == "revenue_agent"
    # Check status reflects the change
    status = run(skill.execute("status", {}))
    assert status.data["active_template"] == "revenue_agent"


def test_apply_missing_template(skill):
    result = run(skill.execute("apply", {"template_id": "nonexistent"}))
    assert not result.success


def test_diff_two_builtins(skill):
    result = run(skill.execute("diff", {
        "template_a": "code_reviewer",
        "template_b": "content_writer",
    }))
    assert result.success
    diff = result.data["diff"]
    assert "skills_enabled" in diff
    assert len(diff["skills_enabled"]["only_in_a"]) > 0
    assert diff["summary"]["total_skill_differences"] > 0


def test_diff_missing_template(skill):
    result = run(skill.execute("diff", {"template_a": "code_reviewer", "template_b": "nope"}))
    assert not result.success


def test_export_and_import(skill):
    # Export a built-in
    export_result = run(skill.execute("export", {"template_id": "data_analyst"}))
    assert export_result.success
    bundle = export_result.data["bundle"]
    assert bundle["format"] == "singularity_config_template_v1"
    # Import it back
    import_result = run(skill.execute("import_template", {"bundle": bundle}))
    assert import_result.success
    tid = import_result.data["template_id"]
    assert tid.startswith("imported_")


def test_import_invalid_format(skill):
    result = run(skill.execute("import_template", {"bundle": {"format": "wrong"}}))
    assert not result.success


def test_delete_custom(skill):
    create_result = run(skill.execute("create", {"name": "To Delete"}))
    tid = create_result.data["template_id"]
    del_result = run(skill.execute("delete", {"template_id": tid}))
    assert del_result.success
    get_result = run(skill.execute("get", {"template_id": tid}))
    assert not get_result.success


def test_delete_builtin_blocked(skill):
    result = run(skill.execute("delete", {"template_id": "code_reviewer"}))
    assert not result.success
    assert "built-in" in result.message.lower()


def test_status(skill):
    result = run(skill.execute("status", {}))
    assert result.success
    assert "builtin_template_count" in result.data
    assert result.data["builtin_template_count"] == len(BUILTIN_TEMPLATES)


def test_unknown_action(skill):
    result = run(skill.execute("bogus", {}))
    assert not result.success
