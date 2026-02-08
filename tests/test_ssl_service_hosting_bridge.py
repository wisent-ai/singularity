"""Tests for SSLServiceHostingBridgeSkill."""

import json
import pytest
from pathlib import Path
from unittest.mock import patch
from singularity.skills.ssl_service_hosting_bridge import (
    SSLServiceHostingBridgeSkill,
    BRIDGE_FILE,
)


@pytest.fixture(autouse=True)
def clean_data(tmp_path, monkeypatch):
    """Redirect all data files to tmp_path."""
    bridge_f = tmp_path / "ssl_service_hosting_bridge.json"
    monkeypatch.setattr("singularity.skills.ssl_service_hosting_bridge.BRIDGE_FILE", bridge_f)
    # Store tmp_path for test helpers
    yield tmp_path


def _write_services(tmp_path, services=None):
    if services is None:
        services = {
            "agent1-api": {"service_id": "agent1-api", "service_name": "api", "domain": "agent1.singularity.wisent.ai", "status": "active"},
            "agent2-web": {"service_id": "agent2-web", "service_name": "web", "domain": "agent2.singularity.wisent.ai", "status": "active"},
        }
    data = {"services": services, "routing_rules": {}, "domain_assignments": {}}
    (tmp_path / "hosted_services.json").write_text(json.dumps(data))


def _make_skill(tmp_path):
    """Create skill with patched file paths for helper methods."""
    skill = SSLServiceHostingBridgeSkill()

    svc_file = tmp_path / "hosted_services.json"
    ssl_file = tmp_path / "ssl_certificates.json"

    def _get_service_info(service_id):
        if not svc_file.exists():
            return None
        try:
            data = json.loads(svc_file.read_text())
            return data.get("services", {}).get(service_id)
        except Exception:
            return None

    def _get_all_services():
        if not svc_file.exists():
            return {}
        try:
            data = json.loads(svc_file.read_text())
            return data.get("services", {})
        except Exception:
            return {}

    def _load_ssl_data():
        if not ssl_file.exists():
            return {"certificates": {}, "domains": {}}
        try:
            return json.loads(ssl_file.read_text())
        except Exception:
            return {"certificates": {}, "domains": {}}

    original_provision = skill._provision_ssl

    def _provision_ssl(domain, provider, challenge):
        import uuid, hashlib
        from datetime import datetime, timedelta
        ssl_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            ssl_data = json.loads(ssl_file.read_text()) if ssl_file.exists() else {
                "certificates": {}, "domains": {},
                "renewal_config": {"auto_renew": True, "renewal_threshold_days": 30, "preferred_provider": "letsencrypt", "preferred_challenge": "http-01"},
                "stats": {"total_provisioned": 0, "total_renewed": 0, "total_failed": 0, "total_revoked": 0},
            }
        except Exception:
            ssl_data = {
                "certificates": {}, "domains": {},
                "renewal_config": {"auto_renew": True, "renewal_threshold_days": 30, "preferred_provider": "letsencrypt", "preferred_challenge": "http-01"},
                "stats": {"total_provisioned": 0, "total_renewed": 0, "total_failed": 0, "total_revoked": 0},
            }
        # Check existing
        for cid in reversed(ssl_data.get("domains", {}).get(domain, [])):
            cert = ssl_data["certificates"].get(cid)
            if cert and cert.get("status") == "active":
                return {"success": True, "cert_id": cid, "reused": True}
        cert_id = f"cert_{uuid.uuid4().hex[:12]}"
        now = datetime.utcnow()
        validity = 90 if provider == "letsencrypt" else 365
        ssl_data["certificates"][cert_id] = {
            "cert_id": cert_id, "domain": domain, "san_domains": [], "provider": provider,
            "challenge_type": challenge, "status": "active",
            "issued_at": str(now), "expires_at": str(now + timedelta(days=validity)),
            "fingerprint": hashlib.sha256(f"{domain}:{provider}".encode()).hexdigest()[:40],
            "serial_number": uuid.uuid4().hex[:20].upper(), "key_type": "ec-256",
            "auto_renew": True, "renewal_count": 0, "created_at": str(now),
        }
        ssl_data.setdefault("domains", {})[domain] = ssl_data["domains"].get(domain, []) + [cert_id]
        ssl_data["stats"]["total_provisioned"] += 1
        ssl_file.write_text(json.dumps(ssl_data, indent=2, default=str))
        return {"success": True, "cert_id": cert_id, "reused": False}

    def _revoke_ssl(cert_id):
        if not ssl_file.exists():
            return {"success": False, "error": "not found"}
        ssl_data = json.loads(ssl_file.read_text())
        cert = ssl_data.get("certificates", {}).get(cert_id)
        if not cert:
            return {"success": False, "error": "not found"}
        cert["status"] = "revoked"
        ssl_file.write_text(json.dumps(ssl_data, indent=2, default=str))
        return {"success": True, "cert_id": cert_id}

    skill._get_service_info = _get_service_info
    skill._get_all_services = _get_all_services
    skill._load_ssl_data = _load_ssl_data
    skill._provision_ssl = _provision_ssl
    skill._revoke_ssl = _revoke_ssl
    return skill


@pytest.mark.asyncio
async def test_on_register_auto_provisions(tmp_path):
    _write_services(tmp_path)
    skill = _make_skill(tmp_path)
    result = await skill.execute("on_register", {
        "service_id": "agent1-api", "domain": "agent1.singularity.wisent.ai", "service_name": "api",
    })
    assert result.success
    assert "auto-provisioned" in result.message
    assert result.data["binding"]["cert_status"] == "active"


@pytest.mark.asyncio
async def test_on_register_disabled(tmp_path):
    _write_services(tmp_path)
    skill = _make_skill(tmp_path)
    await skill.execute("configure", {"auto_provision": False})
    result = await skill.execute("on_register", {"service_id": "agent1-api", "domain": "agent1.singularity.wisent.ai"})
    assert result.success
    assert "disabled" in result.message


@pytest.mark.asyncio
async def test_on_register_already_wired(tmp_path):
    _write_services(tmp_path)
    skill = _make_skill(tmp_path)
    await skill.execute("on_register", {"service_id": "agent1-api", "domain": "agent1.singularity.wisent.ai"})
    result = await skill.execute("on_register", {"service_id": "agent1-api", "domain": "agent1.singularity.wisent.ai"})
    assert result.success
    assert result.data.get("already_wired") is True


@pytest.mark.asyncio
async def test_wire_and_unwire(tmp_path):
    _write_services(tmp_path)
    skill = _make_skill(tmp_path)
    r1 = await skill.execute("wire", {"service_id": "agent1-api"})
    assert r1.success
    r2 = await skill.execute("unwire", {"service_id": "agent1-api"})
    assert r2.success


@pytest.mark.asyncio
async def test_wire_all(tmp_path):
    _write_services(tmp_path)
    skill = _make_skill(tmp_path)
    result = await skill.execute("wire_all", {})
    assert result.success
    assert len(result.data["wired"]) == 2


@pytest.mark.asyncio
async def test_wire_all_dry_run(tmp_path):
    _write_services(tmp_path)
    skill = _make_skill(tmp_path)
    result = await skill.execute("wire_all", {"dry_run": True})
    assert result.success
    assert result.data["dry_run"] is True
    assert len(result.data["to_wire"]) == 2


@pytest.mark.asyncio
async def test_on_domain_change(tmp_path):
    _write_services(tmp_path)
    skill = _make_skill(tmp_path)
    await skill.execute("on_register", {"service_id": "agent1-api", "domain": "agent1.singularity.wisent.ai"})
    result = await skill.execute("on_domain_change", {
        "service_id": "agent1-api", "old_domain": "agent1.singularity.wisent.ai",
        "new_domain": "new.singularity.wisent.ai",
    })
    assert result.success
    assert "new.singularity.wisent.ai" in result.message


@pytest.mark.asyncio
async def test_on_deregister(tmp_path):
    _write_services(tmp_path)
    skill = _make_skill(tmp_path)
    await skill.execute("on_register", {"service_id": "agent1-api", "domain": "agent1.singularity.wisent.ai"})
    result = await skill.execute("on_deregister", {"service_id": "agent1-api"})
    assert result.success
    assert "removed" in result.message


@pytest.mark.asyncio
async def test_compliance_dashboard(tmp_path):
    _write_services(tmp_path)
    skill = _make_skill(tmp_path)
    await skill.execute("on_register", {"service_id": "agent1-api", "domain": "agent1.singularity.wisent.ai"})
    result = await skill.execute("compliance", {})
    assert result.success
    assert result.data["compliance_pct"] > 0
    assert len(result.data["secured"]) == 1
    assert len(result.data["unsecured"]) == 1


@pytest.mark.asyncio
async def test_health_check(tmp_path):
    _write_services(tmp_path)
    skill = _make_skill(tmp_path)
    await skill.execute("on_register", {"service_id": "agent1-api", "domain": "agent1.singularity.wisent.ai"})
    result = await skill.execute("health", {})
    assert result.success
    assert result.data["health_score"] >= 0


@pytest.mark.asyncio
async def test_configure(tmp_path):
    _write_services(tmp_path)
    skill = _make_skill(tmp_path)
    result = await skill.execute("configure", {"auto_provision": False, "default_provider": "self_signed"})
    assert result.success
    assert "auto_provision=False" in result.message


@pytest.mark.asyncio
async def test_status(tmp_path):
    _write_services(tmp_path)
    skill = _make_skill(tmp_path)
    await skill.execute("on_register", {"service_id": "agent1-api", "domain": "agent1.singularity.wisent.ai"})
    result = await skill.execute("status", {})
    assert result.success
    assert len(result.data["bindings"]) == 1


@pytest.mark.asyncio
async def test_excluded_service(tmp_path):
    _write_services(tmp_path)
    skill = _make_skill(tmp_path)
    await skill.execute("configure", {"excluded_services": ["agent1-api"]})
    result = await skill.execute("on_register", {"service_id": "agent1-api", "domain": "agent1.singularity.wisent.ai"})
    assert result.success
    assert "excluded" in result.message
