"""Tests for APIGatewaySkill - key management, rate limiting, usage tracking."""

import asyncio
import hashlib
import pytest

from singularity.skills.api_gateway import APIGatewaySkill, RateLimiter


@pytest.fixture
def gw():
    skill = APIGatewaySkill()
    skill._persist_path = "/dev/null"  # Don't persist in tests
    return skill


@pytest.fixture
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


# ── Key Creation ────────────────────────────────────────────────

def test_create_key(gw):
    result = run(gw.execute("create_key", {"name": "Test Key", "owner": "alice"}))
    assert result.success
    assert result.data["key_id"].startswith("key_")
    assert result.data["api_key"].startswith("sg_")
    assert result.data["owner"] == "alice"
    assert "skills:read" in result.data["scopes"]


def test_create_key_custom_limits(gw):
    result = run(gw.execute("create_key", {
        "name": "Premium", "owner": "bob",
        "rate_limit": 120, "daily_limit": 5000,
        "scopes": ["admin"], "expires_in_days": 30,
        "metadata": {"plan": "premium"},
    }))
    assert result.success
    assert result.data["rate_limit"] == 120
    assert result.data["daily_limit"] == 5000
    assert result.data["expires_at"] is not None


def test_create_key_missing_params(gw):
    result = run(gw.execute("create_key", {"name": "No Owner"}))
    assert not result.success


# ── Key Revocation ──────────────────────────────────────────────

def test_revoke_key(gw):
    create = run(gw.execute("create_key", {"name": "Temp", "owner": "alice"}))
    key_id = create.data["key_id"]
    result = run(gw.execute("revoke_key", {"key_id": key_id}))
    assert result.success
    assert result.data["revoked_at"] is not None


def test_revoke_already_revoked(gw):
    create = run(gw.execute("create_key", {"name": "Temp", "owner": "alice"}))
    key_id = create.data["key_id"]
    run(gw.execute("revoke_key", {"key_id": key_id}))
    result = run(gw.execute("revoke_key", {"key_id": key_id}))
    assert not result.success


# ── List / Get Keys ────────────────────────────────────────────

def test_list_keys(gw):
    run(gw.execute("create_key", {"name": "K1", "owner": "alice"}))
    run(gw.execute("create_key", {"name": "K2", "owner": "bob"}))
    result = run(gw.execute("list_keys", {}))
    assert result.success
    assert result.data["total"] == 2


def test_list_keys_filter_owner(gw):
    run(gw.execute("create_key", {"name": "K1", "owner": "alice"}))
    run(gw.execute("create_key", {"name": "K2", "owner": "bob"}))
    result = run(gw.execute("list_keys", {"owner": "alice"}))
    assert result.data["total"] == 1


def test_get_key(gw):
    create = run(gw.execute("create_key", {"name": "Lookup", "owner": "alice"}))
    result = run(gw.execute("get_key", {"key_id": create.data["key_id"]}))
    assert result.success
    assert result.data["name"] == "Lookup"


# ── Update Key ──────────────────────────────────────────────────

def test_update_key(gw):
    create = run(gw.execute("create_key", {"name": "Mutable", "owner": "alice"}))
    key_id = create.data["key_id"]
    result = run(gw.execute("update_key", {
        "key_id": key_id, "rate_limit": 200, "scopes": ["admin"],
    }))
    assert result.success
    assert "rate_limit" in result.data["updated_fields"]
    get = run(gw.execute("get_key", {"key_id": key_id}))
    assert get.data["rate_limit"] == 200
    assert get.data["scopes"] == ["admin"]


# ── Access Control ──────────────────────────────────────────────

def test_check_access_valid(gw):
    create = run(gw.execute("create_key", {"name": "Access", "owner": "alice"}))
    raw_key = create.data["api_key"]
    result = run(gw.execute("check_access", {"api_key": raw_key, "required_scope": "skills:read"}))
    assert result.success
    assert result.data["allowed"] is True
    assert result.data["owner"] == "alice"


def test_check_access_invalid_key(gw):
    result = run(gw.execute("check_access", {"api_key": "sg_invalid_key_here"}))
    assert not result.success
    assert result.data["reason"] == "invalid_key"


def test_check_access_revoked(gw):
    create = run(gw.execute("create_key", {"name": "RevTest", "owner": "alice"}))
    raw_key = create.data["api_key"]
    run(gw.execute("revoke_key", {"key_id": create.data["key_id"]}))
    result = run(gw.execute("check_access", {"api_key": raw_key}))
    assert not result.success
    assert result.data["reason"] == "revoked"


def test_check_access_insufficient_scope(gw):
    create = run(gw.execute("create_key", {
        "name": "Limited", "owner": "alice", "scopes": ["skills:read"],
    }))
    raw_key = create.data["api_key"]
    result = run(gw.execute("check_access", {"api_key": raw_key, "required_scope": "admin"}))
    assert not result.success
    assert result.data["reason"] == "insufficient_scope"


def test_check_access_wildcard_scope(gw):
    create = run(gw.execute("create_key", {
        "name": "Wildcard", "owner": "alice", "scopes": ["skills:*"],
    }))
    raw_key = create.data["api_key"]
    result = run(gw.execute("check_access", {"api_key": raw_key, "required_scope": "skills:write"}))
    assert result.success
    assert result.data["allowed"] is True


# ── Rate Limiting ───────────────────────────────────────────────

def test_rate_limiter_basic():
    rl = RateLimiter()
    allowed, remaining, _ = rl.check("k1", 3)
    assert allowed
    rl.record("k1")
    rl.record("k1")
    rl.record("k1")
    allowed, remaining, _ = rl.check("k1", 3)
    assert not allowed


def test_rate_limit_via_check_access(gw):
    create = run(gw.execute("create_key", {
        "name": "RateLimited", "owner": "alice", "rate_limit": 2,
    }))
    raw_key = create.data["api_key"]
    # First two should succeed (check_access records in rate limiter)
    r1 = run(gw.execute("check_access", {"api_key": raw_key}))
    assert r1.data["allowed"]
    r2 = run(gw.execute("check_access", {"api_key": raw_key}))
    assert r2.data["allowed"]
    # Third should be rate limited
    r3 = run(gw.execute("check_access", {"api_key": raw_key}))
    assert not r3.success
    assert r3.data["reason"] == "rate_limited"


# ── Usage Tracking ──────────────────────────────────────────────

def test_record_and_get_usage(gw):
    create = run(gw.execute("create_key", {"name": "Usage", "owner": "alice"}))
    key_id = create.data["key_id"]
    run(gw.execute("record_usage", {"key_id": key_id, "endpoint": "/tasks", "cost": 0.01, "revenue": 0.05}))
    run(gw.execute("record_usage", {"key_id": key_id, "endpoint": "/tasks", "cost": 0.01, "revenue": 0.05}))
    run(gw.execute("record_usage", {"key_id": key_id, "endpoint": "/skills", "error": True}))
    result = run(gw.execute("get_usage", {"key_id": key_id}))
    assert result.success
    assert result.data["total_requests"] == 3
    assert result.data["total_errors"] == 1
    assert abs(result.data["total_revenue"] - 0.10) < 0.001
    assert result.data["endpoint_counts"]["/tasks"] == 2


# ── Billing ─────────────────────────────────────────────────────

def test_billing_summary(gw):
    c1 = run(gw.execute("create_key", {"name": "K1", "owner": "alice"}))
    c2 = run(gw.execute("create_key", {"name": "K2", "owner": "bob"}))
    run(gw.execute("record_usage", {"key_id": c1.data["key_id"], "revenue": 1.00, "cost": 0.20}))
    run(gw.execute("record_usage", {"key_id": c2.data["key_id"], "revenue": 2.00, "cost": 0.50}))
    result = run(gw.execute("get_billing", {}))
    assert result.success
    assert abs(result.data["total_revenue"] - 3.00) < 0.001
    assert abs(result.data["total_cost"] - 0.70) < 0.001
    assert result.data["active_keys"] == 2
    assert len(result.data["per_owner"]) == 2


# ── Key Rotation ────────────────────────────────────────────────

def test_rotate_key(gw):
    create = run(gw.execute("create_key", {"name": "Rotate", "owner": "alice"}))
    old_key = create.data["api_key"]
    key_id = create.data["key_id"]
    rotate = run(gw.execute("rotate_key", {"key_id": key_id}))
    assert rotate.success
    new_key = rotate.data["api_key"]
    assert old_key != new_key
    # Old key should no longer work
    result = run(gw.execute("check_access", {"api_key": old_key}))
    assert not result.success
    # New key should work
    result = run(gw.execute("check_access", {"api_key": new_key}))
    assert result.success


# ── Manifest ────────────────────────────────────────────────────

def test_manifest(gw):
    m = gw.manifest
    assert m.skill_id == "api_gateway"
    assert m.category == "revenue"
    assert len(m.actions) == 10
