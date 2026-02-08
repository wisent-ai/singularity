"""Tests for HTTPRevenueBridgeSkill."""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from singularity.skills.http_revenue_bridge import HTTPRevenueBridgeSkill, PRICING, _load_store
from singularity.skills.base import SkillResult


@pytest.fixture
def skill(tmp_path):
    with patch("singularity.skills.http_revenue_bridge.BRIDGE_FILE", tmp_path / "bridge.json"):
        with patch("singularity.skills.http_revenue_bridge.DATA_DIR", tmp_path):
            s = HTTPRevenueBridgeSkill()
            s._store = _load_store()
            s._http_skill = MagicMock()
            yield s


def _ok(body="OK", status_code=200, headers=None):
    return SkillResult(success=True, message=f"GET -> {status_code}",
                       data={"body": body, "status_code": status_code, "headers": headers or {}})


def _fail():
    return SkillResult(success=False, message="Request failed")


@pytest.mark.asyncio
async def test_manifest(skill):
    m = skill.manifest
    assert m.skill_id == "http_revenue_bridge"
    assert m.category == "revenue"
    assert len(skill.get_actions()) == 8


@pytest.mark.asyncio
async def test_proxy_request_success(skill):
    skill._http_skill.execute = AsyncMock(return_value=_ok(body='{"data": 1}'))
    result = await skill.execute("proxy_request", {"customer_id": "cust1", "url": "https://api.example.com/data"})
    assert result.success
    assert result.data["charged"] == PRICING["proxy_request"]
    assert result.data["customer_id"] == "cust1"
    assert skill._store["revenue"]["total"] == PRICING["proxy_request"]


@pytest.mark.asyncio
async def test_proxy_request_missing_url(skill):
    result = await skill.execute("proxy_request", {"customer_id": "c1"})
    assert not result.success
    assert "url is required" in result.message


@pytest.mark.asyncio
async def test_proxy_request_failure(skill):
    skill._http_skill.execute = AsyncMock(return_value=_fail())
    result = await skill.execute("proxy_request", {"customer_id": "c1", "url": "https://bad.example.com"})
    assert not result.success
    assert skill._store["stats"]["failed_requests"] == 1


@pytest.mark.asyncio
async def test_setup_relay(skill):
    result = await skill.execute("setup_relay", {
        "customer_id": "cust1", "relay_name": "my-relay", "target_url": "https://hook.example.com/recv"
    })
    assert result.success
    assert "relay_id" in result.data
    assert len(skill._store["relays"]) == 1


@pytest.mark.asyncio
async def test_setup_relay_invalid_url(skill):
    result = await skill.execute("setup_relay", {"customer_id": "c1", "target_url": "not-a-url"})
    assert not result.success


@pytest.mark.asyncio
async def test_trigger_relay(skill):
    skill._http_skill.execute = AsyncMock(return_value=_ok())
    setup = await skill.execute("setup_relay", {
        "customer_id": "cust1", "relay_name": "test", "target_url": "https://hook.example.com"
    })
    relay_id = setup.data["relay_id"]
    result = await skill.execute("trigger_relay", {"relay_id": relay_id, "payload": {"event": "test"}})
    assert result.success
    assert result.data["charged"] == PRICING["webhook_relay"]


@pytest.mark.asyncio
async def test_trigger_relay_filter(skill):
    setup = await skill.execute("setup_relay", {
        "customer_id": "c1", "relay_name": "filtered", "target_url": "https://hook.example.com",
        "filter_fields": ["required_field"]
    })
    relay_id = setup.data["relay_id"]
    result = await skill.execute("trigger_relay", {"relay_id": relay_id, "payload": {"other": "data"}})
    assert result.success
    assert result.data.get("filtered") is True


@pytest.mark.asyncio
async def test_monitor_url(skill):
    result = await skill.execute("monitor_url", {
        "customer_id": "cust1", "url": "https://example.com", "name": "Example"
    })
    assert result.success
    assert "monitor_id" in result.data
    assert len(skill._store["monitors"]) == 1


@pytest.mark.asyncio
async def test_check_health(skill):
    skill._http_skill.execute = AsyncMock(return_value=_ok())
    await skill.execute("monitor_url", {"customer_id": "c1", "url": "https://example.com", "name": "Test"})
    result = await skill.execute("check_health", {"customer_id": "c1"})
    assert result.success
    assert result.data["total_checked"] == 1
    assert result.data["healthy"] == 1


@pytest.mark.asyncio
async def test_extract_data(skill):
    skill._http_skill.execute = AsyncMock(return_value=_ok(
        body='<title>Hello World</title><price>$9.99</price>'
    ))
    result = await skill.execute("extract_data", {
        "customer_id": "c1", "url": "https://example.com",
        "patterns": {"title": "<title>(.*?)</title>", "price": r"\$[\d.]+"}
    })
    assert result.success
    assert result.data["extracted"]["title"] == "Hello World"
    assert result.data["charged"] == PRICING["data_extraction"]


@pytest.mark.asyncio
async def test_list_services(skill):
    await skill.execute("monitor_url", {"customer_id": "c1", "url": "https://a.com", "name": "A"})
    await skill.execute("setup_relay", {"customer_id": "c1", "relay_name": "r", "target_url": "https://b.com"})
    result = await skill.execute("list_services", {"customer_id": "c1"})
    assert result.success
    assert result.data["total_monitors"] == 1
    assert result.data["total_relays"] == 1


@pytest.mark.asyncio
async def test_service_stats(skill):
    skill._http_skill.execute = AsyncMock(return_value=_ok())
    await skill.execute("proxy_request", {"customer_id": "c1", "url": "https://api.com"})
    result = await skill.execute("service_stats", {})
    assert result.success
    assert result.data["total_revenue"] > 0
    assert result.data["total_requests"] == 1
    assert len(result.data["services_available"]) == 4


@pytest.mark.asyncio
async def test_unknown_action(skill):
    result = await skill.execute("nonexistent", {})
    assert not result.success
    assert "Unknown action" in result.message


@pytest.mark.asyncio
async def test_revenue_tracking_across_services(skill):
    skill._http_skill.execute = AsyncMock(return_value=_ok(body='<h1>Test</h1>'))
    await skill.execute("proxy_request", {"customer_id": "c1", "url": "https://a.com"})
    await skill.execute("extract_data", {"customer_id": "c2", "url": "https://b.com", "patterns": {"h1": "<h1>(.*?)</h1>"}})
    rev = skill._store["revenue"]
    assert rev["total"] == PRICING["proxy_request"] + PRICING["data_extraction"]
    assert rev["by_customer"]["c1"] == PRICING["proxy_request"]
    assert rev["by_customer"]["c2"] == PRICING["data_extraction"]
    assert rev["by_service"]["proxy_request"] == PRICING["proxy_request"]
    assert rev["by_service"]["data_extraction"] == PRICING["data_extraction"]
