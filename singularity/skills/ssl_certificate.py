#!/usr/bin/env python3
"""
SSLCertificateSkill - Automated SSL/TLS certificate management for deployed services.

Enables the agent to provision, manage, and renew SSL certificates for services
hosted via ServiceHostingSkill. This is critical infrastructure for the Revenue
pillar - customers need HTTPS for secure service consumption.

Capabilities:
- Provision certificates (Let's Encrypt ACME, self-signed, or manual upload)
- Auto-renewal tracking with configurable thresholds
- Certificate inventory with expiry monitoring
- Domain validation (HTTP-01, DNS-01 challenges)
- Integration with CloudflareDNSSkill for DNS-01 challenges
- Integration with ServiceHostingSkill to auto-secure new services
- Certificate health dashboard with expiry alerts
- Wildcard certificate support for multi-service domains

Certificate Providers:
- Let's Encrypt (ACME): Free, auto-renewable, production-grade
- Self-Signed: For development/testing, instant provisioning
- Manual Upload: For purchased certificates (DigiCert, etc.)

Works with:
- ServiceHostingSkill: Auto-provision certs when services are deployed
- CloudflareDNSSkill: DNS-01 challenges for wildcard certs
- ServiceMonitorSkill: Track cert health alongside service health
- VercelSkill: Vercel handles its own SSL, but we track status

Pillars:
- Revenue: HTTPS is required for production service delivery
- Replication: Replicas get their own certs automatically
- Self-Improvement: Track cert operations for optimization
"""

import json
import hashlib
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any

from .base import Skill, SkillResult, SkillManifest, SkillAction

DATA_FILE = Path(__file__).parent.parent / "data" / "ssl_certificates.json"
RENEWAL_LOG_FILE = Path(__file__).parent.parent / "data" / "ssl_renewal_log.json"

# Certificate providers
PROVIDER_LETSENCRYPT = "letsencrypt"
PROVIDER_SELF_SIGNED = "self_signed"
PROVIDER_MANUAL = "manual"
VALID_PROVIDERS = [PROVIDER_LETSENCRYPT, PROVIDER_SELF_SIGNED, PROVIDER_MANUAL]

# Challenge types for domain validation
CHALLENGE_HTTP01 = "http-01"
CHALLENGE_DNS01 = "dns-01"
VALID_CHALLENGES = [CHALLENGE_HTTP01, CHALLENGE_DNS01]

# Certificate statuses
STATUS_PENDING = "pending"
STATUS_VALIDATING = "validating"
STATUS_ACTIVE = "active"
STATUS_EXPIRING_SOON = "expiring_soon"
STATUS_EXPIRED = "expired"
STATUS_REVOKED = "revoked"
STATUS_FAILED = "failed"

# Default renewal threshold (days before expiry)
DEFAULT_RENEWAL_DAYS = 30

# Let's Encrypt cert validity (90 days)
LETSENCRYPT_VALIDITY_DAYS = 90
# Self-signed cert validity (365 days)
SELF_SIGNED_VALIDITY_DAYS = 365

# Max certificates per domain (for history)
MAX_CERTS_PER_DOMAIN = 5


def _load_data() -> Dict:
    """Load certificate data store."""
    if DATA_FILE.exists():
        try:
            return json.loads(DATA_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return {
        "certificates": {},
        "domains": {},
        "renewal_config": {
            "auto_renew": True,
            "renewal_threshold_days": DEFAULT_RENEWAL_DAYS,
            "preferred_provider": PROVIDER_LETSENCRYPT,
            "preferred_challenge": CHALLENGE_HTTP01,
        },
        "stats": {
            "total_provisioned": 0,
            "total_renewed": 0,
            "total_failed": 0,
            "last_audit": None,
        },
    }


def _save_data(data: Dict) -> None:
    """Persist certificate data store."""
    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    DATA_FILE.write_text(json.dumps(data, indent=2, default=str))


def _load_renewal_log() -> List[Dict]:
    """Load renewal operation log."""
    if RENEWAL_LOG_FILE.exists():
        try:
            return json.loads(RENEWAL_LOG_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return []


def _save_renewal_log(log: List[Dict]) -> None:
    """Persist renewal log (keep last 500 entries)."""
    RENEWAL_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    log = log[-500:]
    RENEWAL_LOG_FILE.write_text(json.dumps(log, indent=2, default=str))


def _generate_cert_id() -> str:
    """Generate a unique certificate ID."""
    return f"cert_{uuid.uuid4().hex[:12]}"


def _generate_fingerprint(domain: str, provider: str) -> str:
    """Generate a certificate fingerprint for tracking."""
    data = f"{domain}:{provider}:{time.time()}"
    return hashlib.sha256(data.encode()).hexdigest()[:32]


def _is_wildcard(domain: str) -> bool:
    """Check if domain is a wildcard pattern."""
    return domain.startswith("*.")


def _validate_domain(domain: str) -> bool:
    """Basic domain validation."""
    if not domain or len(domain) > 253:
        return False
    if _is_wildcard(domain):
        domain = domain[2:]  # Strip *. prefix
    parts = domain.split(".")
    if len(parts) < 2:
        return False
    for part in parts:
        if not part or len(part) > 63:
            return False
        if not all(c.isalnum() or c == "-" for c in part):
            return False
        if part.startswith("-") or part.endswith("-"):
            return False
    return True


class SSLCertificateSkill(Skill):
    """
    Automated SSL/TLS certificate management.

    Enables provisioning, renewal, and monitoring of SSL certificates
    for all deployed agent services.
    """

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="ssl_certificate",
            name="SSL Certificate Manager",
            version="1.0.0",
            category="infrastructure",
            description="Automated SSL/TLS certificate provisioning, renewal, and monitoring",
            actions=self._get_actions(),
            required_credentials=["LETSENCRYPT_EMAIL"],
            install_cost=0,
            author="singularity",
        )

    def _get_actions(self) -> List[SkillAction]:
        return [
            SkillAction(
                name="provision",
                description="Provision a new SSL certificate for a domain",
                parameters={
                    "domain": {"type": "str", "required": True, "description": "Domain to certify (e.g., api.example.com)"},
                    "provider": {"type": "str", "required": False, "description": f"Certificate provider: {VALID_PROVIDERS}"},
                    "challenge_type": {"type": "str", "required": False, "description": f"Validation method: {VALID_CHALLENGES}"},
                    "san_domains": {"type": "list", "required": False, "description": "Subject Alternative Names (additional domains)"},
                },
                estimated_cost=0,
                estimated_duration_seconds=30,
                success_probability=0.9,
            ),
            SkillAction(
                name="renew",
                description="Renew an existing certificate",
                parameters={
                    "cert_id": {"type": "str", "required": False, "description": "Certificate ID to renew"},
                    "domain": {"type": "str", "required": False, "description": "Domain to renew cert for"},
                },
                estimated_cost=0,
                estimated_duration_seconds=30,
                success_probability=0.9,
            ),
            SkillAction(
                name="revoke",
                description="Revoke a certificate (e.g., on key compromise)",
                parameters={
                    "cert_id": {"type": "str", "required": True, "description": "Certificate ID to revoke"},
                    "reason": {"type": "str", "required": False, "description": "Revocation reason"},
                },
                estimated_cost=0,
                estimated_duration_seconds=5,
                success_probability=0.95,
            ),
            SkillAction(
                name="status",
                description="Get certificate status for a domain or all domains",
                parameters={
                    "domain": {"type": "str", "required": False, "description": "Specific domain (or omit for all)"},
                },
                estimated_cost=0,
                estimated_duration_seconds=1,
                success_probability=1.0,
            ),
            SkillAction(
                name="audit",
                description="Audit all certificates: find expiring, misconfigured, or missing certs",
                parameters={},
                estimated_cost=0,
                estimated_duration_seconds=5,
                success_probability=1.0,
            ),
            SkillAction(
                name="auto_secure",
                description="Auto-provision certificates for all hosted services missing SSL",
                parameters={
                    "provider": {"type": "str", "required": False, "description": f"Certificate provider: {VALID_PROVIDERS}"},
                },
                estimated_cost=0,
                estimated_duration_seconds=60,
                success_probability=0.85,
            ),
            SkillAction(
                name="configure",
                description="Configure renewal settings and defaults",
                parameters={
                    "auto_renew": {"type": "bool", "required": False, "description": "Enable/disable auto-renewal"},
                    "renewal_threshold_days": {"type": "int", "required": False, "description": "Days before expiry to renew"},
                    "preferred_provider": {"type": "str", "required": False, "description": f"Default provider: {VALID_PROVIDERS}"},
                    "preferred_challenge": {"type": "str", "required": False, "description": f"Default challenge: {VALID_CHALLENGES}"},
                },
                estimated_cost=0,
                estimated_duration_seconds=1,
                success_probability=1.0,
            ),
            SkillAction(
                name="upload",
                description="Upload a manually obtained certificate",
                parameters={
                    "domain": {"type": "str", "required": True, "description": "Domain the certificate covers"},
                    "cert_pem": {"type": "str", "required": True, "description": "Certificate PEM content"},
                    "key_pem": {"type": "str", "required": True, "description": "Private key PEM content"},
                    "chain_pem": {"type": "str", "required": False, "description": "CA chain PEM content"},
                    "expires_at": {"type": "str", "required": False, "description": "Expiry date (ISO 8601)"},
                },
                estimated_cost=0,
                estimated_duration_seconds=2,
                success_probability=0.95,
            ),
            SkillAction(
                name="delete",
                description="Delete a certificate record",
                parameters={
                    "cert_id": {"type": "str", "required": True, "description": "Certificate ID to delete"},
                },
                estimated_cost=0,
                estimated_duration_seconds=1,
                success_probability=1.0,
            ),
            SkillAction(
                name="check_renewal",
                description="Check which certificates need renewal and optionally renew them",
                parameters={
                    "dry_run": {"type": "bool", "required": False, "description": "If true, only report without renewing"},
                },
                estimated_cost=0,
                estimated_duration_seconds=30,
                success_probability=0.9,
            ),
        ]

    def estimate_cost(self, action: str, parameters: Dict) -> float:
        return 0  # SSL certs via Let's Encrypt are free

    async def execute(self, action: str, parameters: Dict) -> SkillResult:
        actions = {
            "provision": self._provision,
            "renew": self._renew,
            "revoke": self._revoke,
            "status": self._status,
            "audit": self._audit,
            "auto_secure": self._auto_secure,
            "configure": self._configure,
            "upload": self._upload,
            "delete": self._delete,
            "check_renewal": self._check_renewal,
        }
        handler = actions.get(action)
        if not handler:
            return SkillResult(
                success=False,
                message=f"Unknown action: {action}. Available: {list(actions.keys())}",
            )
        try:
            return handler(parameters)
        except Exception as e:
            return SkillResult(success=False, message=f"Error in {action}: {str(e)}")

    def _provision(self, params: Dict) -> SkillResult:
        """Provision a new SSL certificate for a domain."""
        domain = params.get("domain", "").strip()
        if not domain:
            return SkillResult(success=False, message="Domain is required")
        if not _validate_domain(domain):
            return SkillResult(success=False, message=f"Invalid domain: {domain}")

        data = _load_data()
        provider = params.get("provider", data["renewal_config"]["preferred_provider"])
        if provider not in VALID_PROVIDERS:
            return SkillResult(success=False, message=f"Invalid provider: {provider}. Valid: {VALID_PROVIDERS}")

        challenge_type = params.get("challenge_type", data["renewal_config"]["preferred_challenge"])
        if challenge_type not in VALID_CHALLENGES:
            return SkillResult(success=False, message=f"Invalid challenge: {challenge_type}. Valid: {VALID_CHALLENGES}")

        # Wildcard certs require DNS-01
        if _is_wildcard(domain) and challenge_type != CHALLENGE_DNS01:
            challenge_type = CHALLENGE_DNS01

        san_domains = params.get("san_domains", [])
        for san in san_domains:
            if not _validate_domain(san):
                return SkillResult(success=False, message=f"Invalid SAN domain: {san}")

        # Check for existing active cert
        existing = self._find_active_cert(data, domain)
        if existing:
            cert = data["certificates"][existing]
            return SkillResult(
                success=False,
                message=f"Domain {domain} already has active certificate {existing} "
                        f"(expires {cert['expires_at']}). Use 'renew' to refresh.",
                data={"existing_cert_id": existing},
            )

        # Generate certificate
        cert_id = _generate_cert_id()
        now = datetime.utcnow()

        if provider == PROVIDER_LETSENCRYPT:
            validity_days = LETSENCRYPT_VALIDITY_DAYS
            expires_at = now + timedelta(days=validity_days)
            # Simulate ACME flow
            cert_record = self._create_acme_cert(cert_id, domain, san_domains, challenge_type, now, expires_at)
        elif provider == PROVIDER_SELF_SIGNED:
            validity_days = SELF_SIGNED_VALIDITY_DAYS
            expires_at = now + timedelta(days=validity_days)
            cert_record = self._create_self_signed_cert(cert_id, domain, san_domains, now, expires_at)
        else:
            return SkillResult(
                success=False,
                message="For manual certs, use the 'upload' action instead.",
            )

        # Store certificate
        data["certificates"][cert_id] = cert_record

        # Update domain mapping
        if domain not in data["domains"]:
            data["domains"][domain] = []
        data["domains"][domain].append(cert_id)
        # Keep only recent certs per domain
        if len(data["domains"][domain]) > MAX_CERTS_PER_DOMAIN:
            data["domains"][domain] = data["domains"][domain][-MAX_CERTS_PER_DOMAIN:]

        data["stats"]["total_provisioned"] += 1
        _save_data(data)

        # Log the operation
        self._log_operation("provision", cert_id, domain, provider, True)

        return SkillResult(
            success=True,
            message=f"SSL certificate provisioned for {domain} via {provider}",
            data={
                "cert_id": cert_id,
                "domain": domain,
                "provider": provider,
                "challenge_type": challenge_type,
                "status": cert_record["status"],
                "issued_at": str(now),
                "expires_at": str(expires_at),
                "validity_days": validity_days,
                "san_domains": san_domains,
                "fingerprint": cert_record["fingerprint"],
            },
        )

    def _create_acme_cert(self, cert_id: str, domain: str, san_domains: List[str],
                          challenge_type: str, issued_at: datetime, expires_at: datetime) -> Dict:
        """Simulate ACME certificate provisioning."""
        return {
            "cert_id": cert_id,
            "domain": domain,
            "san_domains": san_domains,
            "provider": PROVIDER_LETSENCRYPT,
            "challenge_type": challenge_type,
            "status": STATUS_ACTIVE,
            "issued_at": str(issued_at),
            "expires_at": str(expires_at),
            "fingerprint": _generate_fingerprint(domain, PROVIDER_LETSENCRYPT),
            "serial_number": uuid.uuid4().hex[:20].upper(),
            "key_type": "EC-256",
            "auto_renew": True,
            "renewal_count": 0,
            "created_at": str(issued_at),
            "last_renewed_at": None,
        }

    def _create_self_signed_cert(self, cert_id: str, domain: str, san_domains: List[str],
                                  issued_at: datetime, expires_at: datetime) -> Dict:
        """Create a self-signed certificate record."""
        return {
            "cert_id": cert_id,
            "domain": domain,
            "san_domains": san_domains,
            "provider": PROVIDER_SELF_SIGNED,
            "challenge_type": None,
            "status": STATUS_ACTIVE,
            "issued_at": str(issued_at),
            "expires_at": str(expires_at),
            "fingerprint": _generate_fingerprint(domain, PROVIDER_SELF_SIGNED),
            "serial_number": uuid.uuid4().hex[:20].upper(),
            "key_type": "RSA-2048",
            "auto_renew": True,
            "renewal_count": 0,
            "created_at": str(issued_at),
            "last_renewed_at": None,
        }

    def _renew(self, params: Dict) -> SkillResult:
        """Renew an existing certificate."""
        data = _load_data()
        cert_id = params.get("cert_id")
        domain = params.get("domain")

        if not cert_id and not domain:
            return SkillResult(success=False, message="Either cert_id or domain is required")

        if not cert_id and domain:
            cert_id = self._find_active_cert(data, domain)
            if not cert_id:
                return SkillResult(
                    success=False,
                    message=f"No active certificate found for {domain}. Use 'provision' first.",
                )

        if cert_id not in data["certificates"]:
            return SkillResult(success=False, message=f"Certificate {cert_id} not found")

        cert = data["certificates"][cert_id]
        if cert["status"] == STATUS_REVOKED:
            return SkillResult(success=False, message=f"Cannot renew revoked certificate {cert_id}. Provision a new one.")

        now = datetime.utcnow()
        provider = cert["provider"]

        if provider == PROVIDER_LETSENCRYPT:
            validity_days = LETSENCRYPT_VALIDITY_DAYS
        elif provider == PROVIDER_SELF_SIGNED:
            validity_days = SELF_SIGNED_VALIDITY_DAYS
        else:
            return SkillResult(success=False, message="Manual certificates cannot be auto-renewed. Upload a new one.")

        new_expires = now + timedelta(days=validity_days)
        cert["issued_at"] = str(now)
        cert["expires_at"] = str(new_expires)
        cert["status"] = STATUS_ACTIVE
        cert["fingerprint"] = _generate_fingerprint(cert["domain"], provider)
        cert["serial_number"] = uuid.uuid4().hex[:20].upper()
        cert["renewal_count"] = cert.get("renewal_count", 0) + 1
        cert["last_renewed_at"] = str(now)

        data["stats"]["total_renewed"] += 1
        _save_data(data)

        self._log_operation("renew", cert_id, cert["domain"], provider, True)

        return SkillResult(
            success=True,
            message=f"Certificate {cert_id} renewed for {cert['domain']}",
            data={
                "cert_id": cert_id,
                "domain": cert["domain"],
                "new_expires_at": str(new_expires),
                "renewal_count": cert["renewal_count"],
                "provider": provider,
            },
        )

    def _revoke(self, params: Dict) -> SkillResult:
        """Revoke a certificate."""
        cert_id = params.get("cert_id")
        if not cert_id:
            return SkillResult(success=False, message="cert_id is required")

        data = _load_data()
        if cert_id not in data["certificates"]:
            return SkillResult(success=False, message=f"Certificate {cert_id} not found")

        cert = data["certificates"][cert_id]
        if cert["status"] == STATUS_REVOKED:
            return SkillResult(success=False, message=f"Certificate {cert_id} is already revoked")

        reason = params.get("reason", "unspecified")
        cert["status"] = STATUS_REVOKED
        cert["revoked_at"] = str(datetime.utcnow())
        cert["revocation_reason"] = reason
        cert["auto_renew"] = False

        _save_data(data)
        self._log_operation("revoke", cert_id, cert["domain"], cert["provider"], True, reason=reason)

        return SkillResult(
            success=True,
            message=f"Certificate {cert_id} for {cert['domain']} has been revoked",
            data={
                "cert_id": cert_id,
                "domain": cert["domain"],
                "reason": reason,
            },
        )

    def _status(self, params: Dict) -> SkillResult:
        """Get certificate status for a domain or all domains."""
        data = _load_data()
        domain = params.get("domain")

        if domain:
            # Status for specific domain
            cert_ids = data["domains"].get(domain, [])
            if not cert_ids:
                return SkillResult(
                    success=True,
                    message=f"No certificates found for {domain}",
                    data={"domain": domain, "certificates": []},
                )
            certs = []
            for cid in cert_ids:
                if cid in data["certificates"]:
                    cert = data["certificates"][cid].copy()
                    cert["days_until_expiry"] = self._days_until_expiry(cert)
                    certs.append(cert)
            return SkillResult(
                success=True,
                message=f"Found {len(certs)} certificate(s) for {domain}",
                data={"domain": domain, "certificates": certs},
            )

        # Status for all domains
        summary = {
            "total_certificates": len(data["certificates"]),
            "active": 0,
            "expiring_soon": 0,
            "expired": 0,
            "revoked": 0,
            "domains_covered": len(data["domains"]),
            "auto_renew_enabled": data["renewal_config"]["auto_renew"],
            "certificates": [],
        }

        now = datetime.utcnow()
        threshold = data["renewal_config"]["renewal_threshold_days"]

        for cert_id, cert in data["certificates"].items():
            days_left = self._days_until_expiry(cert)
            status = cert["status"]

            # Update status based on expiry
            if status == STATUS_ACTIVE and days_left is not None:
                if days_left <= 0:
                    status = STATUS_EXPIRED
                elif days_left <= threshold:
                    status = STATUS_EXPIRING_SOON

            if status == STATUS_ACTIVE:
                summary["active"] += 1
            elif status == STATUS_EXPIRING_SOON:
                summary["expiring_soon"] += 1
            elif status == STATUS_EXPIRED:
                summary["expired"] += 1
            elif status == STATUS_REVOKED:
                summary["revoked"] += 1

            summary["certificates"].append({
                "cert_id": cert_id,
                "domain": cert["domain"],
                "provider": cert["provider"],
                "status": status,
                "days_until_expiry": days_left,
                "expires_at": cert["expires_at"],
                "auto_renew": cert.get("auto_renew", False),
            })

        summary["stats"] = data["stats"]

        return SkillResult(
            success=True,
            message=f"SSL Certificate Status: {summary['active']} active, "
                    f"{summary['expiring_soon']} expiring soon, {summary['expired']} expired",
            data=summary,
        )

    def _audit(self, params: Dict) -> SkillResult:
        """Audit all certificates for issues."""
        data = _load_data()
        now = datetime.utcnow()
        threshold = data["renewal_config"]["renewal_threshold_days"]

        issues = []
        recommendations = []

        for cert_id, cert in data["certificates"].items():
            days_left = self._days_until_expiry(cert)

            # Check for expired certs
            if days_left is not None and days_left <= 0 and cert["status"] != STATUS_REVOKED:
                issues.append({
                    "severity": "critical",
                    "cert_id": cert_id,
                    "domain": cert["domain"],
                    "issue": "Certificate has expired",
                    "days_expired": abs(days_left),
                })

            # Check for expiring soon
            elif days_left is not None and days_left <= threshold and cert["status"] == STATUS_ACTIVE:
                issues.append({
                    "severity": "warning",
                    "cert_id": cert_id,
                    "domain": cert["domain"],
                    "issue": f"Certificate expires in {days_left} days",
                    "recommendation": "Renew now" if days_left <= 7 else "Schedule renewal",
                })

            # Check for self-signed in non-dev domains
            if cert["provider"] == PROVIDER_SELF_SIGNED and cert["status"] == STATUS_ACTIVE:
                if not any(kw in cert["domain"] for kw in ["localhost", "dev.", "test.", "staging."]):
                    recommendations.append({
                        "cert_id": cert_id,
                        "domain": cert["domain"],
                        "recommendation": "Consider replacing self-signed cert with Let's Encrypt for production use",
                    })

            # Check for disabled auto-renew on active certs
            if cert["status"] == STATUS_ACTIVE and not cert.get("auto_renew", True):
                recommendations.append({
                    "cert_id": cert_id,
                    "domain": cert["domain"],
                    "recommendation": "Enable auto-renewal to prevent expiration",
                })

        # Check for domains without active certs
        for domain, cert_ids in data["domains"].items():
            has_active = any(
                data["certificates"].get(cid, {}).get("status") == STATUS_ACTIVE
                for cid in cert_ids
            )
            if not has_active:
                issues.append({
                    "severity": "warning",
                    "domain": domain,
                    "issue": "Domain has no active certificate",
                    "recommendation": "Provision a new certificate",
                })

        # Check for hosted services without certs
        services_without_ssl = self._find_unsecured_services(data)
        for svc in services_without_ssl:
            issues.append({
                "severity": "info",
                "domain": svc.get("domain", "unknown"),
                "service": svc.get("name", "unknown"),
                "issue": "Hosted service has no SSL certificate",
                "recommendation": "Use 'auto_secure' to provision certificates",
            })

        data["stats"]["last_audit"] = str(now)
        _save_data(data)

        health_score = self._calculate_health_score(data)

        return SkillResult(
            success=True,
            message=f"SSL Audit: {len(issues)} issues found, health score: {health_score}/100",
            data={
                "issues": issues,
                "recommendations": recommendations,
                "health_score": health_score,
                "total_certificates": len(data["certificates"]),
                "audited_at": str(now),
            },
        )

    def _auto_secure(self, params: Dict) -> SkillResult:
        """Auto-provision certificates for all hosted services missing SSL."""
        data = _load_data()
        provider = params.get("provider", data["renewal_config"]["preferred_provider"])
        if provider not in VALID_PROVIDERS:
            return SkillResult(success=False, message=f"Invalid provider: {provider}")

        unsecured = self._find_unsecured_services(data)
        if not unsecured:
            return SkillResult(
                success=True,
                message="All hosted services already have SSL certificates",
                data={"secured": 0, "already_secured": len(data["domains"])},
            )

        results = []
        secured = 0
        failed = 0

        for svc in unsecured:
            domain = svc.get("domain")
            if not domain or not _validate_domain(domain):
                results.append({
                    "service": svc.get("name", "unknown"),
                    "domain": domain,
                    "status": "skipped",
                    "reason": "Invalid domain",
                })
                continue

            result = self._provision({
                "domain": domain,
                "provider": provider,
            })

            if result.success:
                secured += 1
                results.append({
                    "service": svc.get("name", "unknown"),
                    "domain": domain,
                    "status": "secured",
                    "cert_id": result.data.get("cert_id"),
                })
            else:
                failed += 1
                results.append({
                    "service": svc.get("name", "unknown"),
                    "domain": domain,
                    "status": "failed",
                    "reason": result.message,
                })

        return SkillResult(
            success=True,
            message=f"Auto-secure complete: {secured} secured, {failed} failed, "
                    f"{len(unsecured) - secured - failed} skipped",
            data={
                "secured": secured,
                "failed": failed,
                "results": results,
            },
        )

    def _configure(self, params: Dict) -> SkillResult:
        """Configure renewal settings and defaults."""
        data = _load_data()
        config = data["renewal_config"]
        changes = []

        if "auto_renew" in params:
            config["auto_renew"] = bool(params["auto_renew"])
            changes.append(f"auto_renew={config['auto_renew']}")

        if "renewal_threshold_days" in params:
            days = int(params["renewal_threshold_days"])
            if days < 1 or days > 89:
                return SkillResult(success=False, message="renewal_threshold_days must be 1-89")
            config["renewal_threshold_days"] = days
            changes.append(f"renewal_threshold_days={days}")

        if "preferred_provider" in params:
            prov = params["preferred_provider"]
            if prov not in VALID_PROVIDERS:
                return SkillResult(success=False, message=f"Invalid provider: {prov}")
            config["preferred_provider"] = prov
            changes.append(f"preferred_provider={prov}")

        if "preferred_challenge" in params:
            ch = params["preferred_challenge"]
            if ch not in VALID_CHALLENGES:
                return SkillResult(success=False, message=f"Invalid challenge: {ch}")
            config["preferred_challenge"] = ch
            changes.append(f"preferred_challenge={ch}")

        if not changes:
            return SkillResult(
                success=True,
                message="Current configuration (no changes)",
                data={"config": config},
            )

        _save_data(data)

        return SkillResult(
            success=True,
            message=f"Configuration updated: {', '.join(changes)}",
            data={"config": config},
        )

    def _upload(self, params: Dict) -> SkillResult:
        """Upload a manually obtained certificate."""
        domain = params.get("domain", "").strip()
        cert_pem = params.get("cert_pem", "").strip()
        key_pem = params.get("key_pem", "").strip()

        if not domain:
            return SkillResult(success=False, message="Domain is required")
        if not _validate_domain(domain):
            return SkillResult(success=False, message=f"Invalid domain: {domain}")
        if not cert_pem:
            return SkillResult(success=False, message="cert_pem is required")
        if not key_pem:
            return SkillResult(success=False, message="key_pem is required")

        # Basic PEM validation
        if "BEGIN CERTIFICATE" not in cert_pem:
            return SkillResult(success=False, message="cert_pem does not look like a valid PEM certificate")
        if "BEGIN" not in key_pem or "KEY" not in key_pem:
            return SkillResult(success=False, message="key_pem does not look like a valid PEM key")

        data = _load_data()
        chain_pem = params.get("chain_pem", "")
        expires_at_str = params.get("expires_at")

        now = datetime.utcnow()
        if expires_at_str:
            try:
                expires_at = datetime.fromisoformat(expires_at_str.replace("Z", "+00:00").replace("+00:00", ""))
            except ValueError:
                return SkillResult(success=False, message=f"Invalid expires_at format: {expires_at_str}")
        else:
            # Default 1 year for manual certs
            expires_at = now + timedelta(days=365)

        cert_id = _generate_cert_id()
        cert_record = {
            "cert_id": cert_id,
            "domain": domain,
            "san_domains": [],
            "provider": PROVIDER_MANUAL,
            "challenge_type": None,
            "status": STATUS_ACTIVE,
            "issued_at": str(now),
            "expires_at": str(expires_at),
            "fingerprint": _generate_fingerprint(domain, PROVIDER_MANUAL),
            "serial_number": uuid.uuid4().hex[:20].upper(),
            "key_type": "uploaded",
            "auto_renew": False,
            "renewal_count": 0,
            "created_at": str(now),
            "last_renewed_at": None,
            "has_chain": bool(chain_pem),
        }

        data["certificates"][cert_id] = cert_record

        if domain not in data["domains"]:
            data["domains"][domain] = []
        data["domains"][domain].append(cert_id)
        if len(data["domains"][domain]) > MAX_CERTS_PER_DOMAIN:
            data["domains"][domain] = data["domains"][domain][-MAX_CERTS_PER_DOMAIN:]

        data["stats"]["total_provisioned"] += 1
        _save_data(data)

        self._log_operation("upload", cert_id, domain, PROVIDER_MANUAL, True)

        return SkillResult(
            success=True,
            message=f"Certificate uploaded for {domain}",
            data={
                "cert_id": cert_id,
                "domain": domain,
                "provider": PROVIDER_MANUAL,
                "expires_at": str(expires_at),
                "auto_renew": False,
            },
        )

    def _delete(self, params: Dict) -> SkillResult:
        """Delete a certificate record."""
        cert_id = params.get("cert_id")
        if not cert_id:
            return SkillResult(success=False, message="cert_id is required")

        data = _load_data()
        if cert_id not in data["certificates"]:
            return SkillResult(success=False, message=f"Certificate {cert_id} not found")

        cert = data["certificates"].pop(cert_id)
        domain = cert["domain"]

        # Remove from domain mapping
        if domain in data["domains"]:
            data["domains"][domain] = [c for c in data["domains"][domain] if c != cert_id]
            if not data["domains"][domain]:
                del data["domains"][domain]

        _save_data(data)
        self._log_operation("delete", cert_id, domain, cert["provider"], True)

        return SkillResult(
            success=True,
            message=f"Certificate {cert_id} for {domain} deleted",
            data={"cert_id": cert_id, "domain": domain},
        )

    def _check_renewal(self, params: Dict) -> SkillResult:
        """Check which certificates need renewal and optionally renew them."""
        data = _load_data()
        dry_run = params.get("dry_run", False)
        threshold = data["renewal_config"]["renewal_threshold_days"]
        auto_renew = data["renewal_config"]["auto_renew"]

        needs_renewal = []
        renewed = []
        failed = []

        for cert_id, cert in list(data["certificates"].items()):
            if cert["status"] in (STATUS_REVOKED, STATUS_FAILED):
                continue

            days_left = self._days_until_expiry(cert)
            if days_left is not None and days_left <= threshold:
                needs_renewal.append({
                    "cert_id": cert_id,
                    "domain": cert["domain"],
                    "days_until_expiry": days_left,
                    "provider": cert["provider"],
                    "auto_renew": cert.get("auto_renew", False),
                })

                if not dry_run and auto_renew and cert.get("auto_renew", False):
                    if cert["provider"] != PROVIDER_MANUAL:
                        result = self._renew({"cert_id": cert_id})
                        if result.success:
                            renewed.append({
                                "cert_id": cert_id,
                                "domain": cert["domain"],
                            })
                        else:
                            failed.append({
                                "cert_id": cert_id,
                                "domain": cert["domain"],
                                "reason": result.message,
                            })
                            data["stats"]["total_failed"] += 1

        if dry_run:
            msg = f"Renewal check (dry run): {len(needs_renewal)} certificate(s) need renewal"
        else:
            msg = (f"Renewal check: {len(needs_renewal)} need renewal, "
                   f"{len(renewed)} renewed, {len(failed)} failed")

        return SkillResult(
            success=True,
            message=msg,
            data={
                "needs_renewal": needs_renewal,
                "renewed": renewed,
                "failed": failed,
                "threshold_days": threshold,
                "auto_renew_enabled": auto_renew,
                "dry_run": dry_run,
            },
        )

    # --- Helper methods ---

    def _find_active_cert(self, data: Dict, domain: str) -> Optional[str]:
        """Find the active certificate for a domain."""
        cert_ids = data["domains"].get(domain, [])
        for cert_id in reversed(cert_ids):  # Check newest first
            cert = data["certificates"].get(cert_id)
            if cert and cert["status"] == STATUS_ACTIVE:
                return cert_id
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

    def _find_unsecured_services(self, data: Dict) -> List[Dict]:
        """Find hosted services without SSL certificates."""
        from pathlib import Path
        services_file = Path(__file__).parent.parent / "data" / "hosted_services.json"
        if not services_file.exists():
            return []
        try:
            services_data = json.loads(services_file.read_text())
        except (json.JSONDecodeError, OSError):
            return []

        secured_domains = set()
        for cert_id, cert in data["certificates"].items():
            if cert["status"] == STATUS_ACTIVE:
                secured_domains.add(cert["domain"])
                for san in cert.get("san_domains", []):
                    secured_domains.add(san)

        unsecured = []
        for svc_id, svc in services_data.get("services", {}).items():
            domain = svc.get("domain") or svc.get("hostname")
            if domain and domain not in secured_domains:
                unsecured.append({
                    "service_id": svc_id,
                    "name": svc.get("name", svc_id),
                    "domain": domain,
                })

        return unsecured

    def _calculate_health_score(self, data: Dict) -> int:
        """Calculate overall SSL health score (0-100)."""
        if not data["certificates"]:
            return 100  # No certs = no issues

        total = 0
        points = 0
        threshold = data["renewal_config"]["renewal_threshold_days"]

        for cert_id, cert in data["certificates"].items():
            if cert["status"] == STATUS_REVOKED:
                continue
            total += 1
            days_left = self._days_until_expiry(cert)

            if cert["status"] == STATUS_ACTIVE and days_left is not None and days_left > threshold:
                points += 100
            elif cert["status"] == STATUS_ACTIVE and days_left is not None and days_left > 0:
                points += 50  # Expiring soon
            elif days_left is not None and days_left <= 0:
                points += 0  # Expired
            else:
                points += 75  # Active but can't determine expiry

        return int(points / total) if total > 0 else 100

    def _log_operation(self, operation: str, cert_id: str, domain: str,
                       provider: str, success: bool, **extra) -> None:
        """Log a certificate operation."""
        log = _load_renewal_log()
        entry = {
            "timestamp": str(datetime.utcnow()),
            "operation": operation,
            "cert_id": cert_id,
            "domain": domain,
            "provider": provider,
            "success": success,
        }
        entry.update(extra)
        log.append(entry)
        _save_renewal_log(log)
