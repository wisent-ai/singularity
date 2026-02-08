#!/usr/bin/env python3
"""
SSLServiceHostingBridgeSkill - Auto-provision SSL when services are registered/updated.

SSLCertificateSkill manages certificates and ServiceHostingSkill manages service
registration, but they operate independently. This bridge connects them so that:

1. When a service is REGISTERED, SSL is auto-provisioned for its domain
2. When a service's domain CHANGES, old cert is tracked and new cert provisioned
3. When a service is DEREGISTERED, cert status is updated (optional revoke)
4. Certificate health is checked alongside service health
5. Bulk operations: secure all unsecured services in one command
6. Event integration: emits ssl_bridge.* events for observability
7. Compliance dashboard: shows which services are secured, which need attention

Without this bridge, new services are deployed without HTTPS, requiring manual
cert provisioning. With it, every deployed service gets SSL automatically.

Integration flow:
  ServiceHosting.register → Bridge detects new service → SSL.provision(domain)
  ServiceHosting.update(domain) → Bridge detects domain change → SSL.provision(new_domain)
  ServiceHosting.deregister → Bridge marks cert for cleanup → optional SSL.revoke

Pillar: Revenue Generation (HTTPS required for production services)
        Self-Improvement (automated infrastructure management)
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import Skill, SkillResult, SkillManifest, SkillAction

BRIDGE_FILE = Path(__file__).parent.parent / "data" / "ssl_service_hosting_bridge.json"
MAX_LOG_ENTRIES = 500

# Default SSL provider for auto-provisioning
DEFAULT_PROVIDER = "letsencrypt"
DEFAULT_CHALLENGE = "http-01"


class SSLServiceHostingBridgeSkill(Skill):
    """Bridge between SSLCertificateSkill and ServiceHostingSkill for auto-SSL."""

    def __init__(self, credentials: Dict[str, str] = None):
        self.credentials = credentials or {}
        self._ensure_data()

    def _ensure_data(self):
        BRIDGE_FILE.parent.mkdir(parents=True, exist_ok=True)
        if not BRIDGE_FILE.exists():
            self._save(self._default_state())

    def _default_state(self) -> Dict:
        return {
            "config": {
                "auto_provision": True,
                "auto_revoke_on_deregister": False,
                "default_provider": DEFAULT_PROVIDER,
                "default_challenge": DEFAULT_CHALLENGE,
                "wildcard_domains": [],
                "excluded_services": [],
                "emit_events": True,
            },
            "bindings": {},  # service_id -> {domain, cert_id, status, ...}
            "event_log": [],
            "stats": {
                "services_secured": 0,
                "auto_provisions": 0,
                "domain_changes": 0,
                "deregistrations": 0,
                "provision_failures": 0,
                "total_certs_managed": 0,
            },
        }

    def _load(self) -> Dict:
        try:
            return json.loads(BRIDGE_FILE.read_text())
        except Exception:
            return self._default_state()

    def _save(self, data: Dict):
        if len(data.get("event_log", [])) > MAX_LOG_ENTRIES:
            data["event_log"] = data["event_log"][-MAX_LOG_ENTRIES:]
        BRIDGE_FILE.parent.mkdir(parents=True, exist_ok=True)
        BRIDGE_FILE.write_text(json.dumps(data, indent=2, default=str))

    def _log_event(self, data: Dict, event_type: str, details: Dict):
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "event": event_type,
            **details,
        }
        data["event_log"].append(entry)

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            name="ssl_service_hosting_bridge",
            description="Auto-provision SSL certificates when services are registered in ServiceHosting",
            version="1.0.0",
            actions=self._get_actions(),
        )

    def _get_actions(self) -> List[SkillAction]:
        return [
            SkillAction(
                name="wire",
                description="Wire a service to auto-SSL: provision cert for its domain",
                parameters={"service_id": "str", "provider": "str (optional)", "challenge": "str (optional)"},
            ),
            SkillAction(
                name="wire_all",
                description="Wire all unsecured hosted services to SSL certificates",
                parameters={"provider": "str (optional)", "dry_run": "bool (optional)"},
            ),
            SkillAction(
                name="unwire",
                description="Remove SSL binding for a service (optionally revoke cert)",
                parameters={"service_id": "str", "revoke_cert": "bool (optional)"},
            ),
            SkillAction(
                name="on_register",
                description="Hook: called when a new service is registered - auto-provisions SSL",
                parameters={"service_id": "str", "domain": "str", "service_name": "str (optional)"},
            ),
            SkillAction(
                name="on_domain_change",
                description="Hook: called when a service's domain changes - provisions new cert",
                parameters={"service_id": "str", "old_domain": "str", "new_domain": "str"},
            ),
            SkillAction(
                name="on_deregister",
                description="Hook: called when a service is deregistered - handles cert cleanup",
                parameters={"service_id": "str"},
            ),
            SkillAction(
                name="compliance",
                description="Show SSL compliance dashboard: which services are secured vs not",
                parameters={},
            ),
            SkillAction(
                name="health",
                description="Check SSL health for all wired services (expiry, validity)",
                parameters={},
            ),
            SkillAction(
                name="configure",
                description="Configure bridge settings (auto_provision, provider, etc.)",
                parameters={"auto_provision": "bool", "default_provider": "str", "auto_revoke_on_deregister": "bool"},
            ),
            SkillAction(
                name="status",
                description="Show bridge status: bindings, stats, config",
                parameters={},
            ),
        ]

    def estimate_cost(self, action: str, parameters: Dict) -> float:
        return 0.0

    async def execute(self, action: str, parameters: Dict) -> SkillResult:
        actions = {
            "wire": self._wire,
            "wire_all": self._wire_all,
            "unwire": self._unwire,
            "on_register": self._on_register,
            "on_domain_change": self._on_domain_change,
            "on_deregister": self._on_deregister,
            "compliance": self._compliance,
            "health": self._health,
            "configure": self._configure,
            "status": self._status,
        }
        handler = actions.get(action)
        if not handler:
            return SkillResult(
                success=False,
                message=f"Unknown action: {action}. Available: {list(actions.keys())}",
            )
        return handler(parameters)

    # ── Core actions ────────────────────────────────────────────────────

    def _wire(self, params: Dict) -> SkillResult:
        """Wire a specific service to SSL: provision cert for its domain."""
        service_id = params.get("service_id")
        if not service_id:
            return SkillResult(success=False, message="service_id is required")

        data = self._load()

        # Check if already wired
        if service_id in data["bindings"]:
            existing = data["bindings"][service_id]
            return SkillResult(
                success=True,
                message=f"Service '{service_id}' already wired to domain '{existing['domain']}'",
                data={"binding": existing, "already_wired": True},
            )

        # Look up service in ServiceHosting data
        service_info = self._get_service_info(service_id)
        if not service_info:
            return SkillResult(
                success=False,
                message=f"Service '{service_id}' not found in ServiceHosting registry",
            )

        domain = service_info.get("domain", "")
        if not domain:
            return SkillResult(
                success=False,
                message=f"Service '{service_id}' has no domain assigned",
            )

        # Provision SSL
        provider = params.get("provider", data["config"]["default_provider"])
        challenge = params.get("challenge", data["config"]["default_challenge"])
        cert_result = self._provision_ssl(domain, provider, challenge)

        binding = {
            "service_id": service_id,
            "domain": domain,
            "service_name": service_info.get("service_name", service_id),
            "provider": provider,
            "cert_id": cert_result.get("cert_id"),
            "cert_status": "active" if cert_result.get("success") else "failed",
            "wired_at": datetime.utcnow().isoformat(),
            "last_checked": datetime.utcnow().isoformat(),
            "provision_error": cert_result.get("error"),
        }

        data["bindings"][service_id] = binding
        data["stats"]["services_secured"] += 1 if cert_result.get("success") else 0
        data["stats"]["total_certs_managed"] += 1 if cert_result.get("success") else 0
        if not cert_result.get("success"):
            data["stats"]["provision_failures"] += 1

        self._log_event(data, "ssl_bridge.wired", {
            "service_id": service_id,
            "domain": domain,
            "cert_id": cert_result.get("cert_id"),
            "success": cert_result.get("success", False),
        })

        self._save(data)

        if cert_result.get("success"):
            return SkillResult(
                success=True,
                message=f"Service '{service_id}' wired to SSL: cert provisioned for {domain}",
                data={"binding": binding},
            )
        else:
            return SkillResult(
                success=False,
                message=f"Failed to provision SSL for '{service_id}' domain {domain}: {cert_result.get('error')}",
                data={"binding": binding},
            )

    def _wire_all(self, params: Dict) -> SkillResult:
        """Wire all unsecured hosted services to SSL certificates."""
        data = self._load()
        dry_run = params.get("dry_run", False)
        provider = params.get("provider", data["config"]["default_provider"])

        # Get all services from ServiceHosting
        services = self._get_all_services()
        if not services:
            return SkillResult(
                success=True,
                message="No hosted services found",
                data={"total": 0, "already_wired": 0, "to_wire": 0},
            )

        already_wired = []
        to_wire = []
        excluded = data["config"].get("excluded_services", [])

        for svc_id, svc in services.items():
            if svc_id in data["bindings"]:
                already_wired.append(svc_id)
            elif svc_id in excluded:
                continue
            elif svc.get("domain"):
                to_wire.append({"service_id": svc_id, "domain": svc["domain"], "service_name": svc.get("service_name", svc_id)})

        if dry_run:
            return SkillResult(
                success=True,
                message=f"Dry run: {len(to_wire)} services need SSL, {len(already_wired)} already wired",
                data={
                    "to_wire": to_wire,
                    "already_wired": already_wired,
                    "excluded": excluded,
                    "dry_run": True,
                },
            )

        wired = []
        failed = []
        for svc in to_wire:
            cert_result = self._provision_ssl(svc["domain"], provider, data["config"]["default_challenge"])
            binding = {
                "service_id": svc["service_id"],
                "domain": svc["domain"],
                "service_name": svc["service_name"],
                "provider": provider,
                "cert_id": cert_result.get("cert_id"),
                "cert_status": "active" if cert_result.get("success") else "failed",
                "wired_at": datetime.utcnow().isoformat(),
                "last_checked": datetime.utcnow().isoformat(),
                "provision_error": cert_result.get("error"),
            }
            data["bindings"][svc["service_id"]] = binding

            if cert_result.get("success"):
                wired.append(svc["service_id"])
                data["stats"]["services_secured"] += 1
                data["stats"]["total_certs_managed"] += 1
            else:
                failed.append({"service_id": svc["service_id"], "error": cert_result.get("error")})
                data["stats"]["provision_failures"] += 1

            self._log_event(data, "ssl_bridge.wire_all", {
                "service_id": svc["service_id"],
                "domain": svc["domain"],
                "success": cert_result.get("success", False),
            })

        data["stats"]["auto_provisions"] += len(wired)
        self._save(data)

        return SkillResult(
            success=True,
            message=f"Bulk wire complete: {len(wired)} secured, {len(failed)} failed, {len(already_wired)} already wired",
            data={
                "wired": wired,
                "failed": failed,
                "already_wired": already_wired,
                "total_services": len(services),
            },
        )

    def _unwire(self, params: Dict) -> SkillResult:
        """Remove SSL binding for a service."""
        service_id = params.get("service_id")
        if not service_id:
            return SkillResult(success=False, message="service_id is required")

        data = self._load()
        if service_id not in data["bindings"]:
            return SkillResult(
                success=False,
                message=f"Service '{service_id}' is not wired to SSL",
            )

        binding = data["bindings"].pop(service_id)
        revoke = params.get("revoke_cert", data["config"]["auto_revoke_on_deregister"])

        revoke_result = None
        if revoke and binding.get("cert_id"):
            revoke_result = self._revoke_ssl(binding["cert_id"])

        self._log_event(data, "ssl_bridge.unwired", {
            "service_id": service_id,
            "domain": binding.get("domain"),
            "cert_id": binding.get("cert_id"),
            "revoked": revoke and revoke_result and revoke_result.get("success", False),
        })

        self._save(data)

        return SkillResult(
            success=True,
            message=f"Service '{service_id}' unwired from SSL (domain: {binding.get('domain')})",
            data={
                "removed_binding": binding,
                "cert_revoked": revoke and revoke_result and revoke_result.get("success", False),
            },
        )

    # ── Hooks (called by ServiceHostingSkill integration) ──────────────

    def _on_register(self, params: Dict) -> SkillResult:
        """Hook: auto-provision SSL when a service is registered."""
        service_id = params.get("service_id")
        domain = params.get("domain")
        service_name = params.get("service_name", service_id or "unknown")

        if not service_id:
            return SkillResult(success=False, message="service_id is required")
        if not domain:
            return SkillResult(success=False, message="domain is required")

        data = self._load()

        # Check if auto-provision is enabled
        if not data["config"]["auto_provision"]:
            self._log_event(data, "ssl_bridge.register_skipped", {
                "service_id": service_id,
                "reason": "auto_provision disabled",
            })
            self._save(data)
            return SkillResult(
                success=True,
                message=f"Auto-provision disabled, skipping SSL for '{service_id}'",
                data={"auto_provision": False, "service_id": service_id},
            )

        # Check if service is excluded
        if service_id in data["config"].get("excluded_services", []):
            return SkillResult(
                success=True,
                message=f"Service '{service_id}' is excluded from auto-SSL",
                data={"excluded": True},
            )

        # Check if already wired
        if service_id in data["bindings"]:
            return SkillResult(
                success=True,
                message=f"Service '{service_id}' already has SSL binding",
                data={"binding": data["bindings"][service_id], "already_wired": True},
            )

        # Check if domain is covered by a wildcard
        provider = data["config"]["default_provider"]
        challenge = data["config"]["default_challenge"]

        wildcard_match = self._check_wildcard_coverage(domain, data)
        if wildcard_match:
            # Domain already covered by wildcard cert
            binding = {
                "service_id": service_id,
                "domain": domain,
                "service_name": service_name,
                "provider": provider,
                "cert_id": wildcard_match["cert_id"],
                "cert_status": "active",
                "wired_at": datetime.utcnow().isoformat(),
                "last_checked": datetime.utcnow().isoformat(),
                "wildcard_covered": True,
                "provision_error": None,
            }
            data["bindings"][service_id] = binding
            data["stats"]["services_secured"] += 1
            data["stats"]["auto_provisions"] += 1
            self._log_event(data, "ssl_bridge.auto_provisioned", {
                "service_id": service_id,
                "domain": domain,
                "cert_id": wildcard_match["cert_id"],
                "wildcard": True,
            })
            self._save(data)
            return SkillResult(
                success=True,
                message=f"Service '{service_id}' covered by wildcard cert for {wildcard_match['wildcard_domain']}",
                data={"binding": binding},
            )

        # Provision new cert
        cert_result = self._provision_ssl(domain, provider, challenge)

        binding = {
            "service_id": service_id,
            "domain": domain,
            "service_name": service_name,
            "provider": provider,
            "cert_id": cert_result.get("cert_id"),
            "cert_status": "active" if cert_result.get("success") else "failed",
            "wired_at": datetime.utcnow().isoformat(),
            "last_checked": datetime.utcnow().isoformat(),
            "wildcard_covered": False,
            "provision_error": cert_result.get("error"),
        }

        data["bindings"][service_id] = binding
        data["stats"]["auto_provisions"] += 1
        if cert_result.get("success"):
            data["stats"]["services_secured"] += 1
            data["stats"]["total_certs_managed"] += 1
        else:
            data["stats"]["provision_failures"] += 1

        self._log_event(data, "ssl_bridge.auto_provisioned", {
            "service_id": service_id,
            "domain": domain,
            "cert_id": cert_result.get("cert_id"),
            "success": cert_result.get("success", False),
        })

        self._save(data)

        if cert_result.get("success"):
            return SkillResult(
                success=True,
                message=f"SSL auto-provisioned for new service '{service_id}' on {domain}",
                data={"binding": binding},
            )
        else:
            return SkillResult(
                success=False,
                message=f"Failed to auto-provision SSL for '{service_id}': {cert_result.get('error')}",
                data={"binding": binding},
            )

    def _on_domain_change(self, params: Dict) -> SkillResult:
        """Hook: handle domain change for a wired service."""
        service_id = params.get("service_id")
        old_domain = params.get("old_domain")
        new_domain = params.get("new_domain")

        if not service_id:
            return SkillResult(success=False, message="service_id is required")
        if not new_domain:
            return SkillResult(success=False, message="new_domain is required")

        data = self._load()
        data["stats"]["domain_changes"] += 1

        old_binding = data["bindings"].get(service_id)

        # Provision new cert for new domain
        provider = data["config"]["default_provider"]
        challenge = data["config"]["default_challenge"]
        cert_result = self._provision_ssl(new_domain, provider, challenge)

        binding = {
            "service_id": service_id,
            "domain": new_domain,
            "service_name": old_binding.get("service_name", service_id) if old_binding else service_id,
            "provider": provider,
            "cert_id": cert_result.get("cert_id"),
            "cert_status": "active" if cert_result.get("success") else "failed",
            "wired_at": datetime.utcnow().isoformat(),
            "last_checked": datetime.utcnow().isoformat(),
            "previous_domain": old_domain,
            "previous_cert_id": old_binding.get("cert_id") if old_binding else None,
            "provision_error": cert_result.get("error"),
        }

        data["bindings"][service_id] = binding
        if cert_result.get("success"):
            data["stats"]["total_certs_managed"] += 1
        else:
            data["stats"]["provision_failures"] += 1

        self._log_event(data, "ssl_bridge.domain_changed", {
            "service_id": service_id,
            "old_domain": old_domain,
            "new_domain": new_domain,
            "new_cert_id": cert_result.get("cert_id"),
            "success": cert_result.get("success", False),
        })

        self._save(data)

        if cert_result.get("success"):
            return SkillResult(
                success=True,
                message=f"Domain change handled: {old_domain} -> {new_domain}, new cert provisioned",
                data={"binding": binding},
            )
        else:
            return SkillResult(
                success=False,
                message=f"Domain changed but SSL provision failed: {cert_result.get('error')}",
                data={"binding": binding},
            )

    def _on_deregister(self, params: Dict) -> SkillResult:
        """Hook: handle service deregistration - cleanup SSL binding."""
        service_id = params.get("service_id")
        if not service_id:
            return SkillResult(success=False, message="service_id is required")

        data = self._load()
        data["stats"]["deregistrations"] += 1

        if service_id not in data["bindings"]:
            self._log_event(data, "ssl_bridge.deregister_noop", {
                "service_id": service_id,
                "reason": "not wired",
            })
            self._save(data)
            return SkillResult(
                success=True,
                message=f"Service '{service_id}' was not wired to SSL, nothing to clean up",
                data={"had_binding": False},
            )

        binding = data["bindings"].pop(service_id)
        revoke = data["config"]["auto_revoke_on_deregister"]

        revoke_result = None
        if revoke and binding.get("cert_id"):
            revoke_result = self._revoke_ssl(binding["cert_id"])

        self._log_event(data, "ssl_bridge.deregistered", {
            "service_id": service_id,
            "domain": binding.get("domain"),
            "cert_id": binding.get("cert_id"),
            "revoked": revoke and revoke_result and revoke_result.get("success", False),
        })

        self._save(data)

        return SkillResult(
            success=True,
            message=f"SSL binding removed for deregistered service '{service_id}'",
            data={
                "removed_binding": binding,
                "cert_revoked": revoke and revoke_result and revoke_result.get("success", False),
            },
        )

    # ── Observability ──────────────────────────────────────────────────

    def _compliance(self, params: Dict) -> SkillResult:
        """Show SSL compliance dashboard."""
        data = self._load()
        services = self._get_all_services()

        secured = []
        unsecured = []
        failed_ssl = []
        excluded = data["config"].get("excluded_services", [])

        for svc_id, svc in services.items():
            domain = svc.get("domain", "")
            if svc_id in data["bindings"]:
                binding = data["bindings"][svc_id]
                if binding.get("cert_status") == "active":
                    secured.append({
                        "service_id": svc_id,
                        "domain": domain,
                        "cert_id": binding.get("cert_id"),
                        "provider": binding.get("provider"),
                        "wired_at": binding.get("wired_at"),
                    })
                else:
                    failed_ssl.append({
                        "service_id": svc_id,
                        "domain": domain,
                        "cert_status": binding.get("cert_status"),
                        "error": binding.get("provision_error"),
                    })
            elif svc_id in excluded:
                continue
            elif domain:
                unsecured.append({
                    "service_id": svc_id,
                    "domain": domain,
                    "service_name": svc.get("service_name", svc_id),
                })

        total = len(secured) + len(unsecured) + len(failed_ssl)
        compliance_pct = round(len(secured) / total * 100, 1) if total > 0 else 100.0

        grade = "A" if compliance_pct >= 90 else "B" if compliance_pct >= 75 else "C" if compliance_pct >= 50 else "F"

        return SkillResult(
            success=True,
            message=f"SSL Compliance: {compliance_pct}% ({grade}) - {len(secured)} secured, {len(unsecured)} unsecured, {len(failed_ssl)} failed",
            data={
                "compliance_pct": compliance_pct,
                "grade": grade,
                "secured": secured,
                "unsecured": unsecured,
                "failed_ssl": failed_ssl,
                "excluded_count": len(excluded),
                "total_services": total,
            },
        )

    def _health(self, params: Dict) -> SkillResult:
        """Check SSL health for all wired services."""
        data = self._load()
        ssl_data = self._load_ssl_data()

        healthy = []
        expiring_soon = []
        expired = []
        missing_cert = []

        for svc_id, binding in data["bindings"].items():
            cert_id = binding.get("cert_id")
            if not cert_id:
                missing_cert.append({
                    "service_id": svc_id,
                    "domain": binding.get("domain"),
                    "status": "no_cert_id",
                })
                continue

            cert = ssl_data.get("certificates", {}).get(cert_id)
            if not cert:
                missing_cert.append({
                    "service_id": svc_id,
                    "domain": binding.get("domain"),
                    "cert_id": cert_id,
                    "status": "cert_not_found",
                })
                continue

            days_left = self._days_until_expiry(cert)
            entry = {
                "service_id": svc_id,
                "domain": binding.get("domain"),
                "cert_id": cert_id,
                "cert_status": cert.get("status"),
                "days_until_expiry": days_left,
                "provider": cert.get("provider"),
            }

            if cert.get("status") == "expired" or (days_left is not None and days_left <= 0):
                expired.append(entry)
            elif days_left is not None and days_left <= 30:
                expiring_soon.append(entry)
            else:
                healthy.append(entry)

        total = len(healthy) + len(expiring_soon) + len(expired) + len(missing_cert)
        health_score = 0
        if total > 0:
            health_score = round(
                (len(healthy) * 100 + len(expiring_soon) * 50) / total
            )

        return SkillResult(
            success=True,
            message=f"SSL Health: {health_score}/100 - {len(healthy)} healthy, {len(expiring_soon)} expiring, {len(expired)} expired, {len(missing_cert)} missing",
            data={
                "health_score": health_score,
                "healthy": healthy,
                "expiring_soon": expiring_soon,
                "expired": expired,
                "missing_cert": missing_cert,
                "total_bindings": total,
            },
        )

    def _configure(self, params: Dict) -> SkillResult:
        """Configure bridge settings."""
        data = self._load()
        config = data["config"]
        changes = []

        if "auto_provision" in params:
            config["auto_provision"] = bool(params["auto_provision"])
            changes.append(f"auto_provision={config['auto_provision']}")

        if "default_provider" in params:
            provider = params["default_provider"]
            valid = ["letsencrypt", "self_signed", "manual"]
            if provider not in valid:
                return SkillResult(success=False, message=f"Invalid provider: {provider}. Valid: {valid}")
            config["default_provider"] = provider
            changes.append(f"default_provider={provider}")

        if "default_challenge" in params:
            challenge = params["default_challenge"]
            valid = ["http-01", "dns-01"]
            if challenge not in valid:
                return SkillResult(success=False, message=f"Invalid challenge: {challenge}. Valid: {valid}")
            config["default_challenge"] = challenge
            changes.append(f"default_challenge={challenge}")

        if "auto_revoke_on_deregister" in params:
            config["auto_revoke_on_deregister"] = bool(params["auto_revoke_on_deregister"])
            changes.append(f"auto_revoke_on_deregister={config['auto_revoke_on_deregister']}")

        if "excluded_services" in params:
            config["excluded_services"] = list(params["excluded_services"])
            changes.append(f"excluded_services={config['excluded_services']}")

        if "wildcard_domains" in params:
            config["wildcard_domains"] = list(params["wildcard_domains"])
            changes.append(f"wildcard_domains={config['wildcard_domains']}")

        if not changes:
            return SkillResult(
                success=True,
                message="Current configuration (no changes)",
                data={"config": config},
            )

        self._save(data)

        return SkillResult(
            success=True,
            message=f"Configuration updated: {', '.join(changes)}",
            data={"config": config},
        )

    def _status(self, params: Dict) -> SkillResult:
        """Show bridge status."""
        data = self._load()
        bindings_summary = []
        for svc_id, binding in data["bindings"].items():
            bindings_summary.append({
                "service_id": svc_id,
                "domain": binding.get("domain"),
                "cert_status": binding.get("cert_status"),
                "provider": binding.get("provider"),
                "wired_at": binding.get("wired_at"),
            })

        recent_events = data["event_log"][-10:] if data["event_log"] else []

        return SkillResult(
            success=True,
            message=f"SSL-ServiceHosting Bridge: {len(data['bindings'])} bindings, {data['stats']['auto_provisions']} auto-provisions",
            data={
                "bindings": bindings_summary,
                "stats": data["stats"],
                "config": data["config"],
                "recent_events": recent_events,
            },
        )

    # ── Helper methods ─────────────────────────────────────────────────

    def _get_service_info(self, service_id: str) -> Optional[Dict]:
        """Look up a service from ServiceHosting registry."""
        services_file = Path(__file__).parent.parent / "data" / "hosted_services.json"
        if not services_file.exists():
            return None
        try:
            data = json.loads(services_file.read_text())
            return data.get("services", {}).get(service_id)
        except (json.JSONDecodeError, OSError):
            return None

    def _get_all_services(self) -> Dict:
        """Get all services from ServiceHosting registry."""
        services_file = Path(__file__).parent.parent / "data" / "hosted_services.json"
        if not services_file.exists():
            return {}
        try:
            data = json.loads(services_file.read_text())
            return data.get("services", {})
        except (json.JSONDecodeError, OSError):
            return {}

    def _load_ssl_data(self) -> Dict:
        """Load SSL certificate data."""
        ssl_file = Path(__file__).parent.parent / "data" / "ssl_certificates.json"
        if not ssl_file.exists():
            return {"certificates": {}, "domains": {}}
        try:
            return json.loads(ssl_file.read_text())
        except (json.JSONDecodeError, OSError):
            return {"certificates": {}, "domains": {}}

    def _provision_ssl(self, domain: str, provider: str, challenge: str) -> Dict:
        """Provision an SSL certificate for a domain.

        Delegates to SSLCertificateSkill data format. In production this would
        call SSLCertificateSkill.execute("provision", ...) via context. Here we
        write directly to the SSL data store for atomicity.
        """
        import uuid
        import hashlib

        ssl_file = Path(__file__).parent.parent / "data" / "ssl_certificates.json"
        ssl_file.parent.mkdir(parents=True, exist_ok=True)

        try:
            ssl_data = json.loads(ssl_file.read_text()) if ssl_file.exists() else {
                "certificates": {},
                "domains": {},
                "renewal_config": {
                    "auto_renew": True,
                    "renewal_threshold_days": 30,
                    "preferred_provider": "letsencrypt",
                    "preferred_challenge": "http-01",
                },
                "stats": {
                    "total_provisioned": 0,
                    "total_renewed": 0,
                    "total_failed": 0,
                    "total_revoked": 0,
                },
            }
        except (json.JSONDecodeError, OSError):
            ssl_data = {
                "certificates": {},
                "domains": {},
                "renewal_config": {
                    "auto_renew": True,
                    "renewal_threshold_days": 30,
                    "preferred_provider": "letsencrypt",
                    "preferred_challenge": "http-01",
                },
                "stats": {
                    "total_provisioned": 0,
                    "total_renewed": 0,
                    "total_failed": 0,
                    "total_revoked": 0,
                },
            }

        # Check if domain already has active cert
        existing_certs = ssl_data.get("domains", {}).get(domain, [])
        for cert_id in reversed(existing_certs):
            cert = ssl_data["certificates"].get(cert_id)
            if cert and cert.get("status") == "active":
                return {"success": True, "cert_id": cert_id, "reused": True}

        # Create new cert
        cert_id = f"cert_{uuid.uuid4().hex[:12]}"
        now = datetime.utcnow()

        if provider == "letsencrypt":
            validity_days = 90
        elif provider == "self_signed":
            validity_days = 365
        else:
            validity_days = 365

        from datetime import timedelta
        cert_record = {
            "cert_id": cert_id,
            "domain": domain,
            "san_domains": [],
            "provider": provider,
            "challenge_type": challenge,
            "status": "active",
            "issued_at": str(now),
            "expires_at": str(now + timedelta(days=validity_days)),
            "fingerprint": hashlib.sha256(f"{domain}:{provider}:{now}".encode()).hexdigest()[:40],
            "serial_number": uuid.uuid4().hex[:20].upper(),
            "key_type": "ec-256" if provider == "letsencrypt" else "rsa-2048",
            "auto_renew": provider != "manual",
            "renewal_count": 0,
            "created_at": str(now),
            "last_renewed_at": None,
            "has_chain": provider == "letsencrypt",
        }

        ssl_data["certificates"][cert_id] = cert_record

        if domain not in ssl_data["domains"]:
            ssl_data["domains"][domain] = []
        ssl_data["domains"][domain].append(cert_id)

        ssl_data["stats"]["total_provisioned"] += 1

        ssl_file.write_text(json.dumps(ssl_data, indent=2, default=str))

        return {"success": True, "cert_id": cert_id, "reused": False}

    def _revoke_ssl(self, cert_id: str) -> Dict:
        """Revoke an SSL certificate by updating its status."""
        ssl_file = Path(__file__).parent.parent / "data" / "ssl_certificates.json"
        if not ssl_file.exists():
            return {"success": False, "error": "SSL data not found"}

        try:
            ssl_data = json.loads(ssl_file.read_text())
        except (json.JSONDecodeError, OSError):
            return {"success": False, "error": "SSL data corrupted"}

        cert = ssl_data.get("certificates", {}).get(cert_id)
        if not cert:
            return {"success": False, "error": f"Certificate {cert_id} not found"}

        cert["status"] = "revoked"
        cert["revoked_at"] = str(datetime.utcnow())
        ssl_data["stats"]["total_revoked"] = ssl_data.get("stats", {}).get("total_revoked", 0) + 1

        ssl_file.write_text(json.dumps(ssl_data, indent=2, default=str))

        return {"success": True, "cert_id": cert_id}

    def _check_wildcard_coverage(self, domain: str, data: Dict) -> Optional[Dict]:
        """Check if a domain is covered by any configured wildcard certificate."""
        wildcard_domains = data["config"].get("wildcard_domains", [])
        ssl_data = self._load_ssl_data()

        for wc_domain in wildcard_domains:
            # Wildcard *.example.com covers sub.example.com
            if wc_domain.startswith("*."):
                base = wc_domain[2:]
                if domain.endswith(base) and domain != base:
                    # Check if we have an active cert for this wildcard
                    wc_certs = ssl_data.get("domains", {}).get(wc_domain, [])
                    for cert_id in reversed(wc_certs):
                        cert = ssl_data.get("certificates", {}).get(cert_id)
                        if cert and cert.get("status") == "active":
                            return {
                                "cert_id": cert_id,
                                "wildcard_domain": wc_domain,
                            }

        return None

    def _days_until_expiry(self, cert: Dict) -> Optional[int]:
        """Calculate days until certificate expiry."""
        expires_at = cert.get("expires_at")
        if not expires_at:
            return None
        try:
            expiry = datetime.fromisoformat(str(expires_at).replace("Z", "").split("+")[0])
            delta = expiry - datetime.utcnow()
            return delta.days
        except (ValueError, TypeError):
            return None
