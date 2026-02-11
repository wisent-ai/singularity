"""
Namecheap Skill

Register domains, manage DNS records, and renew domains via the Namecheap XML API.
"""

import os
from typing import Dict

import httpx

from singularity.skills.base import Skill, SkillResult, SkillManifest, SkillAction


def _a(n, d, p, cost=0.0, dur=5, prob=0.90):
    return SkillAction(name=n, description=d, parameters=p, estimated_cost=cost,
                       estimated_duration_seconds=dur, success_probability=prob)


def _p(n, t, r, d):
    return {n: {"type": t, "required": r, "description": d}}


class NamecheapSkill(Skill):
    """
    Namecheap domain management via the Namecheap XML API.

    Required credentials:
    - NAMECHEAP_API_USER: API user (usually same as username)
    - NAMECHEAP_API_KEY: API key from Namecheap account
    - NAMECHEAP_USERNAME: Namecheap account username
    - NAMECHEAP_CLIENT_IP: Whitelisted client IP address
    """

    API_URL = "https://api.namecheap.com/xml.response"
    SANDBOX_URL = "https://api.sandbox.namecheap.com/xml.response"

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="namecheap",
            name="Namecheap Domains",
            version="1.0.0",
            category="domain",
            description="Register and manage domains via Namecheap",
            required_credentials=[
                "NAMECHEAP_API_USER",
                "NAMECHEAP_API_KEY",
                "NAMECHEAP_USERNAME",
                "NAMECHEAP_CLIENT_IP",
            ],
            install_cost=0,
            actions=[
                _a("check_domain", "Check if a domain is available for registration", {
                    **_p("domain", "string", True, "Domain name to check (e.g. example.com)"),
                }, dur=5, prob=0.95),
                _a("register_domain", "Register a new domain name", {
                    **_p("domain", "string", True, "Domain name to register (e.g. example.com)"),
                    **_p("years", "integer", False, "Registration period in years (default: 1)"),
                    **_p("nameservers", "string", False, "Comma-separated custom nameservers"),
                    **_p("add_whois_guard", "string", False, "Enable WhoisGuard privacy (true/false, default: true)"),
                }, cost=10.0, dur=30, prob=0.85),
                _a("get_domains", "List all domains in the Namecheap account", {
                    **_p("page", "integer", False, "Page number (default: 1)"),
                    **_p("page_size", "integer", False, "Results per page (default: 20, max: 100)"),
                    **_p("sort_by", "string", False, "Sort field: NAME, EXPIREDATE, CREATEDATE (default: NAME)"),
                }, dur=8, prob=0.95),
                _a("set_dns", "Set DNS host records for a domain", {
                    **_p("domain", "string", True, "Domain name (e.g. example.com)"),
                    **_p("records", "string", True, "JSON array of records: [{\"type\":\"A\",\"host\":\"@\",\"value\":\"1.2.3.4\",\"ttl\":300}]"),
                }, dur=10, prob=0.85),
                _a("get_dns", "Get current DNS host records for a domain", {
                    **_p("domain", "string", True, "Domain name (e.g. example.com)"),
                }, dur=5, prob=0.95),
                _a("renew_domain", "Renew an existing domain registration", {
                    **_p("domain", "string", True, "Domain name to renew (e.g. example.com)"),
                    **_p("years", "integer", False, "Renewal period in years (default: 1)"),
                }, cost=10.0, dur=15, prob=0.90),
            ]
        )

    def __init__(self, credentials: Dict[str, str] = None):
        super().__init__(credentials)
        self.http = httpx.AsyncClient()
        self._use_sandbox = os.environ.get("NAMECHEAP_SANDBOX", "").lower() in ("1", "true", "yes")

    @property
    def _api_url(self) -> str:
        """Return sandbox or production API URL."""
        return self.SANDBOX_URL if self._use_sandbox else self.API_URL

    def _get_credential(self, key: str) -> str:
        """Get a credential from instance credentials or environment."""
        return self.credentials.get(key) or os.environ.get(key, "")

    def _base_params(self) -> Dict:
        """Base query params required for all Namecheap API calls."""
        return {
            "ApiUser": self._get_credential("NAMECHEAP_API_USER"),
            "ApiKey": self._get_credential("NAMECHEAP_API_KEY"),
            "UserName": self._get_credential("NAMECHEAP_USERNAME"),
            "ClientIp": self._get_credential("NAMECHEAP_CLIENT_IP"),
        }

    def _split_domain(self, domain: str):
        """Split a domain into SLD and TLD parts."""
        parts = domain.strip().split(".")
        if len(parts) >= 2:
            sld = parts[0]
            tld = ".".join(parts[1:])
            return sld, tld
        return domain, "com"

    async def execute(self, action: str, params: Dict) -> SkillResult:
        from . import actions

        try:
            dispatch = {
                "check_domain": lambda: actions.check_domain(self, params),
                "register_domain": lambda: actions.register_domain(self, params),
                "get_domains": lambda: actions.get_domains(self, params),
                "set_dns": lambda: actions.set_dns(self, params),
                "get_dns": lambda: actions.get_dns(self, params),
                "renew_domain": lambda: actions.renew_domain(self, params),
            }
            handler = dispatch.get(action)
            if not handler:
                return SkillResult(success=False, message=f"Unknown action: {action}")
            return await handler()
        except Exception as e:
            return SkillResult(success=False, message=f"namecheap error: {str(e)}")

    async def close(self):
        await self.http.aclose()
