"""Tests for SSLCertificateSkill."""

import json
import asyncio
import pytest
from pathlib import Path
from singularity.skills.ssl_certificate import (
    SSLCertificateSkill, DATA_FILE, RENEWAL_LOG_FILE,
    _validate_domain, _is_wildcard,
)


@pytest.fixture(autouse=True)
def clean_data():
    for f in [DATA_FILE, RENEWAL_LOG_FILE]:
        if f.exists():
            f.unlink()
    yield
    for f in [DATA_FILE, RENEWAL_LOG_FILE]:
        if f.exists():
            f.unlink()


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def make_skill():
    return SSLCertificateSkill()


def test_manifest():
    s = make_skill()
    m = s.manifest
    assert m.skill_id == "ssl_certificate"
    assert m.category == "infrastructure"
    actions = [a.name for a in m.actions]
    assert "provision" in actions
    assert "renew" in actions
    assert "revoke" in actions
    assert "status" in actions
    assert "audit" in actions
    assert "auto_secure" in actions
    assert "configure" in actions
    assert "upload" in actions
    assert "delete" in actions
    assert "check_renewal" in actions


def test_validate_domain():
    assert _validate_domain("example.com")
    assert _validate_domain("api.example.com")
    assert _validate_domain("*.example.com")
    assert not _validate_domain("")
    assert not _validate_domain("a")
    assert not _validate_domain("-bad.com")


def test_is_wildcard():
    assert _is_wildcard("*.example.com")
    assert not _is_wildcard("example.com")


def test_provision_letsencrypt():
    s = make_skill()
    r = run(s.execute("provision", {"domain": "api.example.com"}))
    assert r.success
    assert r.data["cert_id"].startswith("cert_")
    assert r.data["provider"] == "letsencrypt"
    assert r.data["domain"] == "api.example.com"
    assert r.data["validity_days"] == 90


def test_provision_self_signed():
    s = make_skill()
    r = run(s.execute("provision", {
        "domain": "dev.example.com",
        "provider": "self_signed",
    }))
    assert r.success
    assert r.data["provider"] == "self_signed"
    assert r.data["validity_days"] == 365


def test_provision_duplicate_blocked():
    s = make_skill()
    run(s.execute("provision", {"domain": "api.example.com"}))
    r = run(s.execute("provision", {"domain": "api.example.com"}))
    assert not r.success
    assert "already has active" in r.message


def test_provision_invalid_domain():
    s = make_skill()
    r = run(s.execute("provision", {"domain": "not valid!"}))
    assert not r.success


def test_renew():
    s = make_skill()
    r1 = run(s.execute("provision", {"domain": "api.example.com"}))
    cert_id = r1.data["cert_id"]
    r2 = run(s.execute("renew", {"cert_id": cert_id}))
    assert r2.success
    assert r2.data["renewal_count"] == 1


def test_renew_by_domain():
    s = make_skill()
    run(s.execute("provision", {"domain": "api.example.com"}))
    r = run(s.execute("renew", {"domain": "api.example.com"}))
    assert r.success


def test_revoke():
    s = make_skill()
    r1 = run(s.execute("provision", {"domain": "api.example.com"}))
    cert_id = r1.data["cert_id"]
    r2 = run(s.execute("revoke", {"cert_id": cert_id, "reason": "compromised"}))
    assert r2.success
    assert r2.data["reason"] == "compromised"


def test_status_empty():
    s = make_skill()
    r = run(s.execute("status", {}))
    assert r.success
    assert r.data["total_certificates"] == 0


def test_status_with_certs():
    s = make_skill()
    run(s.execute("provision", {"domain": "a.example.com"}))
    run(s.execute("provision", {"domain": "b.example.com"}))
    r = run(s.execute("status", {}))
    assert r.success
    assert r.data["total_certificates"] == 2
    assert r.data["active"] == 2


def test_audit():
    s = make_skill()
    run(s.execute("provision", {"domain": "api.example.com"}))
    r = run(s.execute("audit", {}))
    assert r.success
    assert r.data["health_score"] == 100


def test_configure():
    s = make_skill()
    r = run(s.execute("configure", {"renewal_threshold_days": 14, "auto_renew": False}))
    assert r.success
    assert r.data["config"]["renewal_threshold_days"] == 14
    assert r.data["config"]["auto_renew"] is False


def test_upload():
    s = make_skill()
    r = run(s.execute("upload", {
        "domain": "shop.example.com",
        "cert_pem": "-----BEGIN CERTIFICATE-----\nMIIB...\n-----END CERTIFICATE-----",
        "key_pem": "-----BEGIN PRIVATE KEY-----\nMIIE...\n-----END PRIVATE KEY-----",
    }))
    assert r.success
    assert r.data["provider"] == "manual"
    assert r.data["auto_renew"] is False


def test_delete():
    s = make_skill()
    r1 = run(s.execute("provision", {"domain": "api.example.com"}))
    cert_id = r1.data["cert_id"]
    r2 = run(s.execute("delete", {"cert_id": cert_id}))
    assert r2.success
    r3 = run(s.execute("status", {}))
    assert r3.data["total_certificates"] == 0


def test_check_renewal_dry_run():
    s = make_skill()
    run(s.execute("provision", {"domain": "api.example.com"}))
    r = run(s.execute("check_renewal", {"dry_run": True}))
    assert r.success
    # Just-provisioned cert shouldn't need renewal
    assert len(r.data["needs_renewal"]) == 0
