#!/usr/bin/env python3
"""
Namecheap Domain Skill

Enables agents to:
- Check domain availability
- Register domains
- Manage DNS records
- Renew domains
"""

import httpx
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional
from .base import Skill, SkillResult, SkillManifest, SkillAction


class NamecheapSkill(Skill):
    """
    Skill for domain management via Namecheap API.

    Required credentials:
    - NAMECHEAP_API_USER: API username
    - NAMECHEAP_API_KEY: API key
    - NAMECHEAP_USERNAME: Namecheap account username
    - NAMECHEAP_CLIENT_IP: Whitelisted IP address
    """

    # Namecheap API endpoints
    SANDBOX_URL = "https://api.sandbox.namecheap.com/xml.response"
    PRODUCTION_URL = "https://api.namecheap.com/xml.response"

    # Maximum price for domain registration (prevents agents from buying expensive domains)
    MAX_DOMAIN_PRICE = 20.0

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="namecheap",
            name="Namecheap Domain Management",
            version="1.0.0",
            category="domain",
            description="Register and manage domains via Namecheap",
            required_credentials=[
                "NAMECHEAP_API_USER",
                "NAMECHEAP_API_KEY",
                "NAMECHEAP_USERNAME",
                "NAMECHEAP_CLIENT_IP"
            ],
            install_cost=0,
            actions=[
                SkillAction(
                    name="check_domain",
                    description="Check if a domain is available for registration",
                    parameters={
                        "domain": {"type": "string", "required": True, "description": "Domain to check (e.g., example.com)"}
                    },
                    estimated_cost=0,
                    estimated_duration_seconds=2,
                    success_probability=0.95
                ),
                SkillAction(
                    name="register_domain",
                    description="Register a new domain",
                    parameters={
                        "domain": {"type": "string", "required": True, "description": "Domain to register"},
                        "years": {"type": "integer", "required": False, "description": "Years to register (default: 1)"}
                    },
                    estimated_cost=12.0,  # Typical .com price
                    estimated_duration_seconds=30,
                    success_probability=0.9
                ),
                SkillAction(
                    name="get_domains",
                    description="List all domains in the account",
                    parameters={},
                    estimated_cost=0,
                    estimated_duration_seconds=5,
                    success_probability=0.95
                ),
                SkillAction(
                    name="set_dns",
                    description="Set DNS records for a domain",
                    parameters={
                        "domain": {"type": "string", "required": True, "description": "Domain name"},
                        "records": {"type": "array", "required": True, "description": "DNS records to set"}
                    },
                    estimated_cost=0,
                    estimated_duration_seconds=10,
                    success_probability=0.9
                ),
                SkillAction(
                    name="get_pricing",
                    description="Get domain pricing for TLDs",
                    parameters={
                        "tlds": {"type": "array", "required": False, "description": "TLDs to check (default: com, net, org)"}
                    },
                    estimated_cost=0,
                    estimated_duration_seconds=3,
                    success_probability=0.95
                )
            ]
        )

    def __init__(self, credentials: Dict[str, str] = None, sandbox: bool = False):
        super().__init__(credentials)
        self.sandbox = sandbox
        self.base_url = self.SANDBOX_URL if sandbox else self.PRODUCTION_URL
        self.http = httpx.AsyncClient(timeout=30)

    async def execute(self, action: str, params: Dict) -> SkillResult:
        """Execute a Namecheap action"""
        if not self.check_credentials():
            missing = self.get_missing_credentials()
            return SkillResult(
                success=False,
                message=f"Missing credentials: {missing}"
            )

        try:
            if action == "check_domain":
                return await self._check_domain(params.get("domain"))
            elif action == "register_domain":
                return await self._register_domain(
                    params.get("domain"),
                    params.get("years", 1)
                )
            elif action == "get_domains":
                return await self._get_domains()
            elif action == "set_dns":
                return await self._set_dns(
                    params.get("domain"),
                    params.get("records", [])
                )
            elif action == "get_pricing":
                return await self._get_pricing(
                    params.get("tlds", ["com", "net", "org"])
                )
            else:
                return SkillResult(
                    success=False,
                    message=f"Unknown action: {action}"
                )
        except Exception as e:
            return SkillResult(
                success=False,
                message=f"Namecheap error: {str(e)}"
            )

    def _get_base_params(self) -> Dict:
        """Get base API parameters"""
        return {
            "ApiUser": self.credentials.get("NAMECHEAP_API_USER"),
            "ApiKey": self.credentials.get("NAMECHEAP_API_KEY"),
            "UserName": self.credentials.get("NAMECHEAP_USERNAME"),
            "ClientIp": self.credentials.get("NAMECHEAP_CLIENT_IP")
        }

    async def _api_call(self, command: str, extra_params: Dict = None) -> ET.Element:
        """Make API call to Namecheap"""
        params = self._get_base_params()
        params["Command"] = command
        if extra_params:
            params.update(extra_params)

        response = await self.http.get(self.base_url, params=params)
        response.raise_for_status()

        root = ET.fromstring(response.text)
        return root

    async def _check_domain(self, domain: str) -> SkillResult:
        """Check domain availability"""
        if not domain:
            return SkillResult(success=False, message="Domain required")

        root = await self._api_call(
            "namecheap.domains.check",
            {"DomainList": domain}
        )

        # Parse response
        status = root.get("Status")
        if status != "OK":
            errors = root.findall(".//Error")
            error_msg = errors[0].text if errors else "Unknown error"
            return SkillResult(success=False, message=error_msg)

        # Handle XML namespace
        ns = {"ns": "http://api.namecheap.com/xml.response"}
        result = root.find(".//ns:DomainCheckResult", ns)
        if result is None:
            # Try without namespace (some responses don't have it)
            result = root.find(".//DomainCheckResult")
        if result is not None:
            available = result.get("Available", "false").lower() == "true"
            premium = result.get("IsPremiumName", "false").lower() == "true"

            return SkillResult(
                success=True,
                message=f"Domain {domain} is {'available' if available else 'not available'}",
                data={
                    "domain": domain,
                    "available": available,
                    "premium": premium
                }
            )

        return SkillResult(success=False, message="Could not parse response")

    async def _register_domain(self, domain: str, years: int = 1) -> SkillResult:
        """Register a domain"""
        if not domain:
            return SkillResult(success=False, message="Domain required")

        # Split domain into SLD and TLD
        parts = domain.split(".")
        if len(parts) < 2:
            return SkillResult(success=False, message="Invalid domain format")

        sld = parts[0]
        tld = ".".join(parts[1:])

        # Check price first
        pricing_result = await self._get_pricing([tld])
        if pricing_result.success and pricing_result.data.get("pricing"):
            price_info = pricing_result.data["pricing"].get(tld.lower())
            if price_info:
                price = price_info.get("price", 0) * years
                if price > self.MAX_DOMAIN_PRICE:
                    return SkillResult(
                        success=False,
                        message=f"Domain price ${price:.2f} exceeds maximum allowed ${self.MAX_DOMAIN_PRICE:.2f}"
                    )

        # Check availability first
        check_result = await self._check_domain(domain)
        if check_result.success and check_result.data:
            if not check_result.data.get("available"):
                return SkillResult(success=False, message=f"Domain {domain} is not available")
            if check_result.data.get("premium"):
                return SkillResult(success=False, message=f"Domain {domain} is premium - not allowed")

        # Registration requires contact info - using defaults
        # In production, these should come from agent config or user input
        params = {
            "DomainName": domain,
            "Years": str(years),
            # Registrant contact
            "RegistrantFirstName": "Agent",
            "RegistrantLastName": "Autonomous",
            "RegistrantAddress1": "123 AI Street",
            "RegistrantCity": "San Francisco",
            "RegistrantStateProvince": "CA",
            "RegistrantPostalCode": "94105",
            "RegistrantCountry": "US",
            "RegistrantPhone": "+1.4155551234",
            "RegistrantEmailAddress": "agent@wisent.com",
            # Tech contact (same)
            "TechFirstName": "Agent",
            "TechLastName": "Autonomous",
            "TechAddress1": "123 AI Street",
            "TechCity": "San Francisco",
            "TechStateProvince": "CA",
            "TechPostalCode": "94105",
            "TechCountry": "US",
            "TechPhone": "+1.4155551234",
            "TechEmailAddress": "agent@wisent.com",
            # Admin contact (same)
            "AdminFirstName": "Agent",
            "AdminLastName": "Autonomous",
            "AdminAddress1": "123 AI Street",
            "AdminCity": "San Francisco",
            "AdminStateProvince": "CA",
            "AdminPostalCode": "94105",
            "AdminCountry": "US",
            "AdminPhone": "+1.4155551234",
            "AdminEmailAddress": "agent@wisent.com",
            # Billing contact (same)
            "AuxBillingFirstName": "Agent",
            "AuxBillingLastName": "Autonomous",
            "AuxBillingAddress1": "123 AI Street",
            "AuxBillingCity": "San Francisco",
            "AuxBillingStateProvince": "CA",
            "AuxBillingPostalCode": "94105",
            "AuxBillingCountry": "US",
            "AuxBillingPhone": "+1.4155551234",
            "AuxBillingEmailAddress": "agent@wisent.com",
        }

        root = await self._api_call("namecheap.domains.create", params)

        status = root.get("Status")
        if status != "OK":
            errors = root.findall(".//Error")
            error_msg = errors[0].text if errors else "Registration failed"
            return SkillResult(success=False, message=error_msg)

        result = root.find(".//DomainCreateResult")
        if result is not None:
            registered = result.get("Registered", "false").lower() == "true"
            charged = float(result.get("ChargedAmount", "0"))

            if registered:
                return SkillResult(
                    success=True,
                    message=f"Successfully registered {domain}",
                    data={
                        "domain": domain,
                        "years": years,
                        "charged": charged,
                        "domain_id": result.get("DomainID"),
                        "order_id": result.get("OrderID")
                    },
                    cost=charged,
                    asset_created={
                        "type": "domain",
                        "name": domain,
                        "value": charged
                    }
                )

        return SkillResult(success=False, message="Registration failed")

    async def _get_domains(self) -> SkillResult:
        """List all domains"""
        root = await self._api_call(
            "namecheap.domains.getList",
            {"PageSize": "100"}
        )

        status = root.get("Status")
        if status != "OK":
            errors = root.findall(".//Error")
            error_msg = errors[0].text if errors else "Failed to get domains"
            return SkillResult(success=False, message=error_msg)

        domains = []
        for domain in root.findall(".//Domain"):
            domains.append({
                "name": domain.get("Name"),
                "expires": domain.get("Expires"),
                "created": domain.get("Created"),
                "auto_renew": domain.get("AutoRenew") == "true",
                "is_locked": domain.get("IsLocked") == "true"
            })

        return SkillResult(
            success=True,
            message=f"Found {len(domains)} domains",
            data={"domains": domains, "count": len(domains)}
        )

    async def _set_dns(self, domain: str, records: List[Dict]) -> SkillResult:
        """Set DNS records for a domain"""
        if not domain:
            return SkillResult(success=False, message="Domain required")

        # Split domain
        parts = domain.split(".")
        sld = parts[0]
        tld = ".".join(parts[1:])

        # Build parameters for each record
        params = {
            "SLD": sld,
            "TLD": tld
        }

        for i, record in enumerate(records, 1):
            params[f"HostName{i}"] = record.get("host", "@")
            params[f"RecordType{i}"] = record.get("type", "A")
            params[f"Address{i}"] = record.get("value", "")
            if record.get("ttl"):
                params[f"TTL{i}"] = str(record.get("ttl"))

        root = await self._api_call("namecheap.domains.dns.setHosts", params)

        status = root.get("Status")
        if status != "OK":
            errors = root.findall(".//Error")
            error_msg = errors[0].text if errors else "Failed to set DNS"
            return SkillResult(success=False, message=error_msg)

        return SkillResult(
            success=True,
            message=f"Set {len(records)} DNS records for {domain}",
            data={"domain": domain, "records_set": len(records)}
        )

    async def _get_pricing(self, tlds: List[str]) -> SkillResult:
        """Get domain pricing"""
        root = await self._api_call(
            "namecheap.users.getPricing",
            {
                "ProductType": "DOMAIN",
                "ProductCategory": "DOMAINS",
                "ActionName": "REGISTER"
            }
        )

        status = root.get("Status")
        if status != "OK":
            errors = root.findall(".//Error")
            error_msg = errors[0].text if errors else "Failed to get pricing"
            return SkillResult(success=False, message=error_msg)

        pricing = {}
        for product in root.findall(".//Product"):
            name = product.get("Name", "").lower()
            if name in [t.lower() for t in tlds]:
                price_elem = product.find(".//Price[@Duration='1']")
                if price_elem is not None:
                    pricing[name] = {
                        "price": float(price_elem.get("Price", "0")),
                        "currency": price_elem.get("Currency", "USD")
                    }

        return SkillResult(
            success=True,
            message=f"Got pricing for {len(pricing)} TLDs",
            data={"pricing": pricing}
        )

    async def close(self):
        """Clean up"""
        await self.http.aclose()
