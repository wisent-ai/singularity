#!/usr/bin/env python3
"""
Email Provider Helpers

Domain management operations for Resend + Namecheap integration:
- Add/verify/list domains in Resend
- Automated DNS setup via Namecheap API
"""

import xml.etree.ElementTree as ET
import asyncio

from singularity.skills.base import SkillResult
from . import (
    NAMECHEAP_API_URL,
    NAMECHEAP_XML_NS,
    REQUIRED_NAMECHEAP_CREDENTIALS,
    RESEND_API_BASE,
)


class DomainManagementMixin:
    """Mixin providing domain management methods for email skill."""

    async def _add_domain(self, domain: str) -> SkillResult:
        """Add a domain to Resend"""
        if self._provider != "resend":
            return SkillResult(
                success=False,
                message="Domain management only supported with Resend",
            )

        api_key = self.credentials.get("RESEND_API_KEY")
        response = await self.http.post(
            f"{RESEND_API_BASE}/domains",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={"name": domain},
        )

        if response.status_code in [200, 201]:
            data = response.json()
            return SkillResult(
                success=True,
                message=f"Domain {domain} added to Resend",
                data={
                    "domain_id": data.get("id"),
                    "domain": domain,
                    "status": data.get("status"),
                    "records": data.get("records", []),
                },
            )
        return SkillResult(success=False, message=f"Failed to add domain: {response.text}")

    async def _get_domain_dns(self, domain_id: str) -> SkillResult:
        """Get DNS records for a domain"""
        if self._provider != "resend":
            return SkillResult(success=False, message="Domain management only supported with Resend")

        api_key = self.credentials.get("RESEND_API_KEY")
        response = await self.http.get(
            f"{RESEND_API_BASE}/domains/{domain_id}",
            headers={"Authorization": f"Bearer {api_key}"},
        )

        if response.status_code == 200:
            data = response.json()
            return SkillResult(
                success=True,
                message="DNS records for domain",
                data={
                    "domain_id": data.get("id"),
                    "domain": data.get("name"),
                    "status": data.get("status"),
                    "records": data.get("records", []),
                },
            )
        return SkillResult(success=False, message=f"Failed to get domain: {response.text}")

    async def _verify_domain(self, domain_id: str) -> SkillResult:
        """Verify a domain in Resend"""
        if self._provider != "resend":
            return SkillResult(success=False, message="Domain management only supported with Resend")

        api_key = self.credentials.get("RESEND_API_KEY")
        response = await self.http.post(
            f"{RESEND_API_BASE}/domains/{domain_id}/verify",
            headers={"Authorization": f"Bearer {api_key}"},
        )

        if response.status_code == 200:
            data = response.json()
            return SkillResult(
                success=True,
                message="Domain verification initiated",
                data={"domain_id": domain_id, "status": data.get("status", "pending")},
            )
        return SkillResult(success=False, message=f"Failed to verify domain: {response.text}")

    async def _list_domains(self) -> SkillResult:
        """List all domains in Resend"""
        if self._provider != "resend":
            return SkillResult(success=False, message="Domain management only supported with Resend")

        api_key = self.credentials.get("RESEND_API_KEY")
        response = await self.http.get(
            f"{RESEND_API_BASE}/domains",
            headers={"Authorization": f"Bearer {api_key}"},
        )

        if response.status_code == 200:
            data = response.json()
            domains = data.get("data", [])
            return SkillResult(
                success=True,
                message=f"Found {len(domains)} domains",
                data={"domains": domains},
            )
        return SkillResult(success=False, message=f"Failed to list domains: {response.text}")

    async def _setup_domain(self, domain: str) -> SkillResult:
        """Full automated domain setup: add to Resend, configure DNS via Namecheap, verify."""
        if self._provider != "resend":
            return SkillResult(success=False, message="Domain setup only supported with Resend")

        missing = [k for k in REQUIRED_NAMECHEAP_CREDENTIALS if not self.credentials.get(k)]
        if missing:
            return SkillResult(success=False, message=f"Missing Namecheap credentials: {missing}")

        # Step 1: Add domain to Resend
        add_result = await self._add_domain(domain)
        if not add_result.success:
            return add_result

        domain_id = add_result.data.get("domain_id")
        records = add_result.data.get("records", [])

        if not records:
            dns_result = await self._get_domain_dns(domain_id)
            if dns_result.success:
                records = dns_result.data.get("records", [])

        if not records:
            return SkillResult(success=False, message="No DNS records returned from Resend")

        # Step 2: Parse domain for Namecheap
        parts = domain.split(".")
        if len(parts) < 2:
            return SkillResult(success=False, message="Invalid domain format")

        sld, tld = parts[-2], parts[-1]

        # Step 3: Fetch existing DNS + merge Resend records + set on Namecheap
        existing_records = await self._fetch_namecheap_hosts(sld, tld)
        merged = _merge_dns_records(existing_records, records)
        set_result = await self._set_namecheap_hosts(sld, tld, merged)

        if not set_result:
            return SkillResult(success=False, message="Failed to set DNS records on Namecheap")

        # Step 4: Verify domain
        await asyncio.sleep(5)
        verify_result = await self._verify_domain(domain_id)

        return SkillResult(
            success=True,
            message=f"Domain {domain} setup complete. DNS records added, verification initiated.",
            data={
                "domain_id": domain_id,
                "domain": domain,
                "records_added": len(records),
                "verification_status": verify_result.data.get("status") if verify_result.success else "pending",
            },
        )

    async def _fetch_namecheap_hosts(self, sld: str, tld: str) -> list:
        """Get existing DNS records from Namecheap."""
        nc_params = {
            "ApiUser": self.credentials["NAMECHEAP_API_USER"],
            "ApiKey": self.credentials["NAMECHEAP_API_KEY"],
            "UserName": self.credentials["NAMECHEAP_USERNAME"],
            "ClientIp": self.credentials["NAMECHEAP_CLIENT_IP"],
            "Command": "namecheap.domains.dns.getHosts",
            "SLD": sld, "TLD": tld,
        }
        resp = await self.http.get(NAMECHEAP_API_URL, params=nc_params)
        root = ET.fromstring(resp.text)
        existing = []
        for h in root.findall(".//ns:host", NAMECHEAP_XML_NS):
            existing.append({
                "name": h.get("Name"), "type": h.get("Type"),
                "address": h.get("Address"),
                "mxpref": h.get("MXPref", "10"), "ttl": h.get("TTL", "1799"),
            })
        return existing

    async def _set_namecheap_hosts(self, sld: str, tld: str, records: list) -> bool:
        """Set all DNS records on Namecheap. Returns True on success."""
        params = {
            "ApiUser": self.credentials["NAMECHEAP_API_USER"],
            "ApiKey": self.credentials["NAMECHEAP_API_KEY"],
            "UserName": self.credentials["NAMECHEAP_USERNAME"],
            "ClientIp": self.credentials["NAMECHEAP_CLIENT_IP"],
            "Command": "namecheap.domains.dns.setHosts",
            "SLD": sld, "TLD": tld, "EmailType": "MX",
        }
        for i, rec in enumerate(records, 1):
            params[f"HostName{i}"] = rec["name"]
            params[f"RecordType{i}"] = rec["type"]
            params[f"Address{i}"] = rec["address"]
            params[f"TTL{i}"] = rec.get("ttl", "1799")
            if rec["type"] == "MX":
                params[f"MXPref{i}"] = rec.get("mxpref", "10")
        resp = await self.http.get(NAMECHEAP_API_URL, params=params)
        root = ET.fromstring(resp.text)
        return root.get("Status") == "OK"


def _merge_dns_records(existing: list, resend_records: list) -> list:
    """Merge existing Namecheap records with new Resend DNS records."""
    merged = list(existing)
    for rec in resend_records:
        rec_type = rec.get("type", rec.get("record_type", "")).upper()
        rec_name = rec.get("name", rec.get("host", ""))
        rec_value = rec.get("value", rec.get("data", ""))
        rec_priority = rec.get("priority", "10")
        merged.append({
            "name": rec_name, "type": rec_type, "address": rec_value,
            "mxpref": str(rec_priority), "ttl": "1799",
        })
    return merged
