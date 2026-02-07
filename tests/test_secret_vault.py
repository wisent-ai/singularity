#!/usr/bin/env python3
"""Tests for SecretVaultSkill - secure credential management."""

import json
import pytest
import asyncio
from pathlib import Path
from unittest.mock import patch

from singularity.skills.secret_vault import (
    SecretVaultSkill, VAULT_FILE, VAULT_KEY_FILE, AUDIT_FILE,
    VaultEncryption, _generate_vault_key, SECRET_CATEGORIES,
)


@pytest.fixture
def skill(tmp_path):
    """Create a SecretVaultSkill with temp data directory."""
    vault_f = tmp_path / "secret_vault.json"
    key_f = tmp_path / ".vault_key"
    audit_f = tmp_path / "vault_audit.json"
    with patch("singularity.skills.secret_vault.VAULT_FILE", vault_f), \
         patch("singularity.skills.secret_vault.VAULT_KEY_FILE", key_f), \
         patch("singularity.skills.secret_vault.AUDIT_FILE", audit_f):
        s = SecretVaultSkill()
        yield s


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


class TestStoreAndRetrieve:
    def test_store_secret(self, skill):
        result = run(skill.execute("store_secret", {
            "name": "stripe_key",
            "value": "sk_test_abc123",
            "category": "api_key",
            "description": "Stripe API key",
        }))
        assert result.success
        assert result.data["name"] == "stripe_key"
        assert result.data["category"] == "api_key"

    def test_get_secret(self, skill):
        run(skill.execute("store_secret", {"name": "my_key", "value": "secret123"}))
        result = run(skill.execute("get_secret", {"name": "my_key"}))
        assert result.success
        assert result.data["value"] == "secret123"

    def test_get_missing_secret(self, skill):
        result = run(skill.execute("get_secret", {"name": "nonexistent"}))
        assert not result.success

    def test_store_duplicate_fails(self, skill):
        run(skill.execute("store_secret", {"name": "dup", "value": "v1"}))
        result = run(skill.execute("store_secret", {"name": "dup", "value": "v2"}))
        assert not result.success
        assert "already exists" in result.message

    def test_store_empty_name_fails(self, skill):
        result = run(skill.execute("store_secret", {"name": "", "value": "x"}))
        assert not result.success

    def test_store_invalid_category(self, skill):
        result = run(skill.execute("store_secret", {"name": "x", "value": "y", "category": "invalid"}))
        assert not result.success


class TestRotation:
    def test_rotate_secret(self, skill):
        run(skill.execute("store_secret", {"name": "token", "value": "old_val"}))
        result = run(skill.execute("rotate_secret", {"name": "token", "new_value": "new_val"}))
        assert result.success
        assert result.data["rotation_count"] == 1
        # Verify new value
        get_result = run(skill.execute("get_secret", {"name": "token"}))
        assert get_result.data["value"] == "new_val"

    def test_rotate_nonexistent(self, skill):
        result = run(skill.execute("rotate_secret", {"name": "nope", "new_value": "v"}))
        assert not result.success


class TestDeletion:
    def test_delete_secret(self, skill):
        run(skill.execute("store_secret", {"name": "del_me", "value": "val"}))
        result = run(skill.execute("delete_secret", {"name": "del_me"}))
        assert result.success
        get = run(skill.execute("get_secret", {"name": "del_me"}))
        assert not get.success

    def test_delete_nonexistent(self, skill):
        result = run(skill.execute("delete_secret", {"name": "nope"}))
        assert not result.success


class TestAccessControl:
    def test_allowed_skills_access(self, skill):
        run(skill.execute("store_secret", {
            "name": "restricted", "value": "secret",
            "allowed_skills": ["payment", "revenue_services"]
        }))
        ok = run(skill.execute("get_secret", {"name": "restricted", "accessor": "payment"}))
        assert ok.success
        denied = run(skill.execute("get_secret", {"name": "restricted", "accessor": "twitter"}))
        assert not denied.success

    def test_agent_bypasses_access_control(self, skill):
        run(skill.execute("store_secret", {
            "name": "restricted2", "value": "s", "allowed_skills": ["payment"]
        }))
        result = run(skill.execute("get_secret", {"name": "restricted2", "accessor": "agent"}))
        assert result.success


class TestExpiry:
    def test_expired_secret_denied(self, skill, tmp_path):
        run(skill.execute("store_secret", {"name": "exp", "value": "v", "expires_in_days": 1}))
        # Manually set expiry to the past
        vault_f = tmp_path / "secret_vault.json"
        data = json.loads(vault_f.read_text())
        data["secrets"]["exp"]["expires_at"] = "2020-01-01T00:00:00"
        vault_f.write_text(json.dumps(data))
        result = run(skill.execute("get_secret", {"name": "exp"}))
        assert not result.success
        assert "expired" in result.message

    def test_check_expiring(self, skill, tmp_path):
        run(skill.execute("store_secret", {"name": "soon", "value": "v", "expires_in_days": 3}))
        result = run(skill.execute("check_expiring", {"within_days": 7}))
        assert result.success
        assert len(result.data["expiring_soon"]) == 1


class TestListAndStatus:
    def test_list_secrets(self, skill):
        run(skill.execute("store_secret", {"name": "a", "value": "1", "category": "api_key"}))
        run(skill.execute("store_secret", {"name": "b", "value": "2", "category": "token"}))
        result = run(skill.execute("list_secrets", {}))
        assert result.success
        assert result.data["total"] == 2

    def test_list_filter_category(self, skill):
        run(skill.execute("store_secret", {"name": "a", "value": "1", "category": "api_key"}))
        run(skill.execute("store_secret", {"name": "b", "value": "2", "category": "token"}))
        result = run(skill.execute("list_secrets", {"category": "api_key"}))
        assert result.data["total"] == 1

    def test_vault_status(self, skill):
        run(skill.execute("store_secret", {"name": "x", "value": "y"}))
        result = run(skill.execute("vault_status", {}))
        assert result.success
        assert result.data["total_secrets"] == 1

    def test_audit_log(self, skill):
        run(skill.execute("store_secret", {"name": "s", "value": "v"}))
        run(skill.execute("get_secret", {"name": "s"}))
        result = run(skill.execute("audit_log", {}))
        assert result.success
        assert len(result.data["entries"]) >= 2

    def test_export_manifest(self, skill):
        run(skill.execute("store_secret", {"name": "k", "value": "v", "tags": ["core"]}))
        result = run(skill.execute("export_manifest", {"include_tags": ["core"]}))
        assert result.success
        assert result.data["total"] == 1
        # Values should NOT be in manifest
        assert "value" not in str(result.data["manifest"][0])


class TestEncryption:
    def test_encryption_roundtrip(self):
        key = _generate_vault_key()
        enc = VaultEncryption(key)
        plaintext = "super-secret-api-key-12345"
        ciphertext = enc.encrypt(plaintext)
        assert ciphertext != plaintext
        assert enc.decrypt(ciphertext) == plaintext
