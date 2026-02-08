"""Tests for CloudflareDNSSkill - Cloudflare DNS management."""

import pytest
import json
from unittest.mock import AsyncMock, patch, MagicMock
from singularity.skills.cloudflare_dns import (
    CloudflareDNSSkill,
    _validate_record_type,
    _validate_ttl,
    _validate_record_content,
    _load_data,
    _save_data,
    RECORD_TYPES,
)


@pytest.fixture
def skill(tmp_path):
    s = CloudflareDNSSkill()
    s._data_file = tmp_path / "cloudflare_dns.json"
    return s


def test_manifest(skill):
    m = skill.manifest
    assert m.skill_id == "cloudflare_dns"
    assert m.category == "infrastructure"
    assert "CLOUDFLARE_API_TOKEN" in m.required_credentials
    action_names = [a.name for a in m.actions]
    assert "list_zones" in action_names
    assert "create_record" in action_names
    assert "wire_service" in action_names
    assert "bulk_create" in action_names


def test_validate_record_type():
    assert _validate_record_type("A") is True
    assert _validate_record_type("CNAME") is True
    assert _validate_record_type("TXT") is True
    assert _validate_record_type("MX") is True
    assert _validate_record_type("INVALID") is False


def test_validate_ttl():
    assert _validate_ttl(1) is True  # auto
    assert _validate_ttl(60) is True
    assert _validate_ttl(86400) is True
    assert _validate_ttl(30) is False  # too low
    assert _validate_ttl(100000) is False  # too high


def test_validate_record_content():
    assert _validate_record_content("A", "1.2.3.4") is None
    assert _validate_record_content("A", "999.1.1.1") is not None
    assert _validate_record_content("A", "not-an-ip") is not None
    assert _validate_record_content("CNAME", "example.com") is None
    assert _validate_record_content("CNAME", "has space") is not None
    assert _validate_record_content("AAAA", "::1") is None
    assert _validate_record_content("AAAA", "not-ipv6") is not None
    assert _validate_record_content("TXT", "any text is fine") is None
    assert _validate_record_content("A", "") is not None


def test_load_save_data(tmp_path):
    path = tmp_path / "test_cf.json"
    data = _load_data(path)
    assert "zones_cache" in data
    assert "config" in data
    data["zones_cache"]["z1"] = {"name": "example.com"}
    _save_data(data, path)
    loaded = _load_data(path)
    assert loaded["zones_cache"]["z1"]["name"] == "example.com"


@pytest.mark.asyncio
async def test_unknown_action(skill):
    result = await skill.execute("nonexistent", {})
    assert not result.success
    assert "Unknown action" in result.message


@pytest.mark.asyncio
async def test_create_record_validation(skill):
    # Missing zone_id
    result = await skill.execute("create_record", {"type": "A", "name": "test", "content": "1.2.3.4"})
    assert not result.success
    assert "zone_id" in result.message

    # Missing type
    result = await skill.execute("create_record", {"zone_id": "z1", "name": "test", "content": "1.2.3.4"})
    assert not result.success

    # Invalid record type
    result = await skill.execute("create_record", {"zone_id": "z1", "type": "INVALID", "name": "test", "content": "1.2.3.4"})
    assert not result.success

    # Invalid IP for A record
    result = await skill.execute("create_record", {"zone_id": "z1", "type": "A", "name": "test", "content": "bad-ip"})
    assert not result.success


@pytest.mark.asyncio
async def test_update_record_validation(skill):
    result = await skill.execute("update_record", {"zone_id": "z1"})
    assert not result.success
    assert "record_id" in result.message

    # No fields to update
    result = await skill.execute("update_record", {"zone_id": "z1", "record_id": "r1"})
    assert not result.success
    assert "No fields" in result.message


@pytest.mark.asyncio
async def test_delete_record_validation(skill):
    result = await skill.execute("delete_record", {})
    assert not result.success
    result = await skill.execute("delete_record", {"zone_id": "z1"})
    assert not result.success


@pytest.mark.asyncio
async def test_bulk_create_validation(skill):
    result = await skill.execute("bulk_create", {"zone_id": "z1"})
    assert not result.success
    assert "empty" in result.message

    # Too many records
    records = [{"type": "A", "name": f"r{i}", "content": "1.1.1.1"} for i in range(51)]
    result = await skill.execute("bulk_create", {"zone_id": "z1", "records": records})
    assert not result.success
    assert "Too many" in result.message

    # Invalid record in batch
    result = await skill.execute("bulk_create", {"zone_id": "z1", "records": [{"type": "INVALID", "name": "x", "content": "y"}]})
    assert not result.success


@pytest.mark.asyncio
async def test_configure(skill):
    result = await skill.execute("configure", {"default_ttl": 300, "default_proxied": False})
    assert result.success
    assert result.data["config"]["default_ttl"] == 300
    assert result.data["config"]["default_proxied"] is False

    # Invalid TTL
    result = await skill.execute("configure", {"default_ttl": 10})
    assert not result.success


@pytest.mark.asyncio
async def test_history_empty(skill):
    result = await skill.execute("history", {})
    assert result.success
    assert result.data["total"] == 0


@pytest.mark.asyncio
async def test_wire_service_validation(skill):
    result = await skill.execute("wire_service", {})
    assert not result.success
    result = await skill.execute("wire_service", {"zone_id": "z1"})
    assert not result.success
    result = await skill.execute("wire_service", {"zone_id": "z1", "subdomain": "api"})
    assert not result.success
