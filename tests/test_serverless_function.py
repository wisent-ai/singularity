#!/usr/bin/env python3
"""Tests for ServerlessFunctionSkill."""

import pytest
import json
from pathlib import Path
from singularity.skills.serverless_function import (
    ServerlessFunctionSkill,
    SUPPORTED_METHODS,
    _load_functions,
)


@pytest.fixture
def skill(tmp_path, monkeypatch):
    import singularity.skills.serverless_function as mod
    monkeypatch.setattr(mod, "DATA_DIR", tmp_path)
    monkeypatch.setattr(mod, "FUNCTIONS_FILE", tmp_path / "serverless_functions.json")
    monkeypatch.setattr(mod, "FUNCTION_CODE_DIR", tmp_path / "function_code")
    s = ServerlessFunctionSkill()
    s._state = s._default_state()
    return s


SAMPLE_HANDLER = '''async def handler(request):
    return {"status_code": 200, "headers": {}, "body": {"ok": True}}
'''

ECHO_HANDLER = '''async def handler(request):
    return {"status_code": 200, "headers": {}, "body": {"echo": request.get("body")}}
'''


def test_manifest(skill):
    m = skill.manifest
    assert m.skill_id == "serverless_function"
    assert m.category == "infrastructure"
    assert len(m.actions) == 10


def test_deploy_basic(skill):
    result = skill.execute("deploy", {
        "agent_name": "eve",
        "function_name": "hello",
        "route": "/api/hello",
        "handler_code": SAMPLE_HANDLER,
    })
    assert result.success
    assert result.data["route"] == "/api/hello"
    assert result.data["methods"] == ["POST"]
    assert result.data["agent_name"] == "eve"
    assert "function_id" in result.data


def test_deploy_route_conflict(skill):
    skill.execute("deploy", {
        "agent_name": "eve", "function_name": "f1",
        "route": "/api/test", "handler_code": SAMPLE_HANDLER,
    })
    result = skill.execute("deploy", {
        "agent_name": "adam", "function_name": "f2",
        "route": "/api/test", "handler_code": SAMPLE_HANDLER,
    })
    assert not result.success
    assert "already assigned" in result.message


def test_deploy_missing_params(skill):
    result = skill.execute("deploy", {"agent_name": "eve"})
    assert not result.success


def test_deploy_invalid_method(skill):
    result = skill.execute("deploy", {
        "agent_name": "eve", "function_name": "f", "route": "/x",
        "handler_code": SAMPLE_HANDLER, "methods": ["INVALID"],
    })
    assert not result.success


def test_deploy_no_async(skill):
    result = skill.execute("deploy", {
        "agent_name": "eve", "function_name": "f", "route": "/x",
        "handler_code": "def handler(r): return {}",
    })
    assert not result.success


def test_update_function(skill):
    r = skill.execute("deploy", {
        "agent_name": "eve", "function_name": "f1",
        "route": "/api/f1", "handler_code": SAMPLE_HANDLER,
    })
    fid = r.data["function_id"]
    result = skill.execute("update", {
        "function_id": fid,
        "description": "Updated desc",
        "price_per_call": 0.05,
    })
    assert result.success
    assert "description" in result.data["updated_fields"]
    assert "price_per_call" in result.data["updated_fields"]
    assert result.data["version"] == 1  # no code change


def test_update_code_bumps_version(skill):
    r = skill.execute("deploy", {
        "agent_name": "eve", "function_name": "f1",
        "route": "/api/f1", "handler_code": SAMPLE_HANDLER,
    })
    fid = r.data["function_id"]
    result = skill.execute("update", {
        "function_id": fid, "handler_code": ECHO_HANDLER,
    })
    assert result.success
    assert result.data["version"] == 2


def test_remove_function(skill):
    r = skill.execute("deploy", {
        "agent_name": "eve", "function_name": "f1",
        "route": "/api/f1", "handler_code": SAMPLE_HANDLER,
    })
    fid = r.data["function_id"]
    result = skill.execute("remove", {"function_id": fid})
    assert result.success
    # Verify it's gone
    listing = skill.execute("list", {})
    assert listing.data["total"] == 0


def test_enable_disable(skill):
    r = skill.execute("deploy", {
        "agent_name": "eve", "function_name": "f1",
        "route": "/api/f1", "handler_code": SAMPLE_HANDLER,
    })
    fid = r.data["function_id"]
    # Disable
    result = skill.execute("disable", {"function_id": fid})
    assert result.success
    assert result.data["status"] == "disabled"
    # Invoke should fail
    inv = skill.execute("invoke", {"function_id": fid})
    assert not inv.success
    # Enable
    result = skill.execute("enable", {"function_id": fid})
    assert result.success
    assert result.data["status"] == "active"


def test_invoke_function(skill):
    r = skill.execute("deploy", {
        "agent_name": "eve", "function_name": "echo",
        "route": "/api/echo", "handler_code": ECHO_HANDLER,
        "methods": ["POST"], "price_per_call": 0.01,
    })
    fid = r.data["function_id"]
    result = skill.execute("invoke", {
        "function_id": fid, "body": {"msg": "hello"},
    })
    assert result.success
    assert result.data["response"]["body"]["echo"] == {"msg": "hello"}
    assert result.data["revenue_earned"] == 0.01
    assert result.revenue == 0.01


def test_list_filter_by_agent(skill):
    skill.execute("deploy", {
        "agent_name": "eve", "function_name": "f1",
        "route": "/a", "handler_code": SAMPLE_HANDLER,
    })
    skill.execute("deploy", {
        "agent_name": "adam", "function_name": "f2",
        "route": "/b", "handler_code": SAMPLE_HANDLER,
    })
    result = skill.execute("list", {"agent_name": "eve"})
    assert result.data["total"] == 1
    assert result.data["functions"][0]["agent_name"] == "eve"


def test_inspect(skill):
    r = skill.execute("deploy", {
        "agent_name": "eve", "function_name": "f1",
        "route": "/api/f1", "handler_code": SAMPLE_HANDLER,
    })
    fid = r.data["function_id"]
    result = skill.execute("inspect", {"function_id": fid})
    assert result.success
    assert "source_code" in result.data
    assert result.data["source_code"] == SAMPLE_HANDLER


def test_generate_server(skill):
    skill.execute("deploy", {
        "agent_name": "eve", "function_name": "hello",
        "route": "/api/hello", "handler_code": SAMPLE_HANDLER,
        "methods": ["GET", "POST"],
    })
    skill.execute("deploy", {
        "agent_name": "eve", "function_name": "echo",
        "route": "/api/echo", "handler_code": ECHO_HANDLER,
    })
    result = skill.execute("generate_server", {"agent_name": "eve"})
    assert result.success
    assert result.data["function_count"] == 2
    assert "server_path" in result.data
    server_code = Path(result.data["server_path"]).read_text()
    assert "async def handler" in server_code
    assert "/api/hello" in server_code
    assert "/api/echo" in server_code


def test_stats(skill):
    r = skill.execute("deploy", {
        "agent_name": "eve", "function_name": "f1",
        "route": "/api/f1", "handler_code": SAMPLE_HANDLER,
        "price_per_call": 0.02,
    })
    fid = r.data["function_id"]
    skill.execute("invoke", {"function_id": fid})
    skill.execute("invoke", {"function_id": fid})
    result = skill.execute("stats", {"agent_name": "eve"})
    assert result.success
    assert result.data["total_invocations"] == 2
    assert result.data["total_revenue"] == 0.04
    assert result.data["active_functions"] == 1


def test_unknown_action(skill):
    result = skill.execute("nonexistent", {})
    assert not result.success
