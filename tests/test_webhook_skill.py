"""Tests for WebhookSkill - inbound webhook endpoint management."""
import hashlib
import hmac
import json
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from singularity.skills.webhook import WebhookSkill, WebhookEndpoint
from singularity.skills.base import SkillResult, SkillContext, SkillRegistry


@pytest.fixture
def skill(tmp_path):
    s = WebhookSkill(data_dir=str(tmp_path))
    return s


@pytest.fixture
def skill_with_context(tmp_path):
    s = WebhookSkill(data_dir=str(tmp_path))
    registry = SkillRegistry()
    ctx = SkillContext(registry=registry)
    ctx.call_skill = AsyncMock(return_value=SkillResult(
        success=True, message="Executed", data={"result": "ok"}
    ))
    s.set_context(ctx)
    return s


def make_signature(payload, secret):
    payload_bytes = json.dumps(payload, sort_keys=True).encode()
    return hmac.new(secret.encode(), payload_bytes, hashlib.sha256).hexdigest()


@pytest.mark.asyncio
async def test_register_endpoint(skill):
    result = await skill.execute("register", {
        "name": "github-push",
        "description": "Handle GitHub push events",
        "target_skill_id": "revenue_services",
        "target_action": "code_review",
    })
    assert result.success
    assert "github-push" in result.message
    assert result.data["url_path"] == "/webhooks/github-push"


@pytest.mark.asyncio
async def test_register_duplicate_fails(skill):
    await skill.execute("register", {
        "name": "test-hook", "target_skill_id": "s", "target_action": "a"
    })
    result = await skill.execute("register", {
        "name": "test-hook", "target_skill_id": "s", "target_action": "a"
    })
    assert not result.success
    assert "already exists" in result.message


@pytest.mark.asyncio
async def test_receive_basic(skill):
    await skill.execute("register", {
        "name": "basic", "target_skill_id": "test", "target_action": "run",
    })
    result = await skill.execute("receive", {
        "endpoint_name": "basic",
        "payload": {"key": "value"},
    })
    assert result.success
    assert "delivery_id" in result.data


@pytest.mark.asyncio
async def test_receive_with_context(skill_with_context):
    s = skill_with_context
    await s.execute("register", {
        "name": "ctx-hook", "target_skill_id": "revenue_services",
        "target_action": "code_review",
        "field_mapping": {"repo": "repository"},
    })
    result = await s.execute("receive", {
        "endpoint_name": "ctx-hook",
        "payload": {"repo": "wisent-ai/singularity"},
    })
    assert result.success
    s.context.call_skill.assert_called_once()


@pytest.mark.asyncio
async def test_signature_verification(skill):
    secret = "my-secret-key"
    await skill.execute("register", {
        "name": "secure", "target_skill_id": "s", "target_action": "a",
        "secret": secret,
    })
    payload = {"event": "push", "ref": "main"}
    sig = make_signature(payload, secret)
    result = await skill.execute("receive", {
        "endpoint_name": "secure", "payload": payload, "signature": sig,
    })
    assert result.success

    # Bad signature should fail
    result = await skill.execute("receive", {
        "endpoint_name": "secure", "payload": payload, "signature": "bad",
    })
    assert not result.success
    assert "signature" in result.message.lower()


@pytest.mark.asyncio
async def test_filters(skill):
    await skill.execute("register", {
        "name": "filtered", "target_skill_id": "s", "target_action": "a",
        "filters": {"action": "opened"},
    })
    # Matching filter
    result = await skill.execute("receive", {
        "endpoint_name": "filtered", "payload": {"action": "opened"},
    })
    assert result.success
    assert not result.data.get("filtered", False)

    # Non-matching filter
    result = await skill.execute("receive", {
        "endpoint_name": "filtered", "payload": {"action": "closed"},
    })
    assert result.success
    assert result.data.get("filtered") is True


@pytest.mark.asyncio
async def test_field_mapping(skill):
    await skill.execute("register", {
        "name": "mapped", "target_skill_id": "s", "target_action": "a",
        "field_mapping": {"repository.full_name": "repo", "head_commit.message": "desc"},
        "static_params": {"language": "python"},
    })
    result = await skill.execute("receive", {
        "endpoint_name": "mapped",
        "payload": {
            "repository": {"full_name": "wisent-ai/singularity"},
            "head_commit": {"message": "fix bug"},
        },
    })
    assert result.success
    params = result.data["params_sent"]
    assert params["repo"] == "wisent-ai/singularity"
    assert params["desc"] == "fix bug"
    assert params["language"] == "python"


@pytest.mark.asyncio
async def test_rate_limiting(skill):
    await skill.execute("register", {
        "name": "limited", "target_skill_id": "s", "target_action": "a",
        "max_calls_per_minute": 2,
    })
    await skill.execute("receive", {"endpoint_name": "limited", "payload": {}})
    await skill.execute("receive", {"endpoint_name": "limited", "payload": {}})
    result = await skill.execute("receive", {"endpoint_name": "limited", "payload": {}})
    assert not result.success
    assert "rate limit" in result.message.lower()


@pytest.mark.asyncio
async def test_list_and_delete(skill):
    await skill.execute("register", {
        "name": "ep1", "target_skill_id": "s", "target_action": "a"
    })
    await skill.execute("register", {
        "name": "ep2", "target_skill_id": "s", "target_action": "b"
    })
    result = await skill.execute("list_endpoints", {})
    assert result.success
    assert result.data["total"] == 2

    result = await skill.execute("delete_endpoint", {"name": "ep1"})
    assert result.success

    result = await skill.execute("list_endpoints", {})
    assert result.data["total"] == 1


@pytest.mark.asyncio
async def test_delivery_history(skill):
    await skill.execute("register", {
        "name": "hist", "target_skill_id": "s", "target_action": "a"
    })
    await skill.execute("receive", {"endpoint_name": "hist", "payload": {"n": 1}})
    await skill.execute("receive", {"endpoint_name": "hist", "payload": {"n": 2}})
    result = await skill.execute("get_deliveries", {"endpoint_name": "hist"})
    assert result.success
    assert result.data["total"] == 2


@pytest.mark.asyncio
async def test_update_endpoint(skill):
    await skill.execute("register", {
        "name": "upd", "target_skill_id": "s", "target_action": "a"
    })
    result = await skill.execute("update_endpoint", {
        "name": "upd", "enabled": False, "target_action": "b"
    })
    assert result.success
    assert "enabled" in result.data["updated_fields"]

    # Disabled endpoint should reject
    result = await skill.execute("receive", {"endpoint_name": "upd", "payload": {}})
    assert not result.success
    assert "disabled" in result.message.lower()


@pytest.mark.asyncio
async def test_test_endpoint(skill):
    await skill.execute("register", {
        "name": "tester", "target_skill_id": "s", "target_action": "a"
    })
    result = await skill.execute("test_endpoint", {"name": "tester"})
    assert result.success
    assert "delivery_id" in result.data


@pytest.mark.asyncio
async def test_persistence(tmp_path):
    s1 = WebhookSkill(data_dir=str(tmp_path))
    await s1.execute("register", {
        "name": "persist", "target_skill_id": "s", "target_action": "a"
    })
    # Load fresh instance
    s2 = WebhookSkill(data_dir=str(tmp_path))
    result = await s2.execute("get_endpoint", {"name": "persist"})
    assert result.success
    assert result.data["name"] == "persist"
