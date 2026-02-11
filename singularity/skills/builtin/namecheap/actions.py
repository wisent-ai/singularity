"""
Namecheap XML API actions.
"""

import xml.etree.ElementTree as ET
from typing import Dict, List, Optional

from singularity.skills.base import SkillResult


def _parse_xml(text: str) -> Optional[ET.Element]:
    """Parse Namecheap XML response and return the root element."""
    try:
        return ET.fromstring(text)
    except ET.ParseError:
        return None


def _ns(tag: str) -> str:
    """Wrap a tag with the Namecheap XML namespace."""
    return f"{{http://api.namecheap.com/xml.response}}{tag}"


def _check_errors(root: ET.Element) -> Optional[str]:
    """Check for API errors in the XML response. Returns error message or None."""
    if root is None:
        return "Failed to parse API response"
    status = root.attrib.get("Status", "")
    if status == "ERROR":
        errors = root.find(_ns("Errors"))
        if errors is not None:
            error_msgs = []
            for err in errors.findall(_ns("Error")):
                error_msgs.append(f"[{err.attrib.get('Number', '?')}] {err.text or 'Unknown error'}")
            return "; ".join(error_msgs) if error_msgs else "Unknown API error"
        return "Unknown API error"
    return None


async def check_domain(skill, params: Dict) -> SkillResult:
    """Check if a domain is available for registration."""
    domain = params.get("domain")
    if not domain:
        return SkillResult(success=False, message="Missing required parameter: domain")

    api_params = {
        **skill._base_params(),
        "Command": "namecheap.domains.check",
        "DomainList": domain,
    }

    resp = await skill.http.get(skill._api_url, params=api_params)

    if resp.status_code != 200:
        return SkillResult(success=False, message=f"API request failed: HTTP {resp.status_code}")

    root = _parse_xml(resp.text)
    error = _check_errors(root)
    if error:
        return SkillResult(success=False, message=f"Namecheap API error: {error}")

    command_resp = root.find(_ns("CommandResponse"))
    if command_resp is None:
        return SkillResult(success=False, message="Invalid API response: missing CommandResponse")

    results = []
    for result in command_resp.findall(_ns("DomainCheckResult")):
        domain_name = result.attrib.get("Domain", "")
        available = result.attrib.get("Available", "false").lower() == "true"
        premium = result.attrib.get("IsPremiumName", "false").lower() == "true"
        price = result.attrib.get("PremiumRegistrationPrice", "")
        results.append({
            "domain": domain_name,
            "available": available,
            "premium": premium,
            "premium_price": price if premium else None,
        })

    if results:
        r = results[0]
        status = "available" if r["available"] else "taken"
        msg = f"{r['domain']} is {status}"
        if r["premium"]:
            msg += f" (premium: ${r['premium_price']})"
        return SkillResult(success=True, message=msg, data={"results": results})

    return SkillResult(success=False, message="No results returned for domain check")


async def register_domain(skill, params: Dict) -> SkillResult:
    """Register a new domain name."""
    domain = params.get("domain")
    if not domain:
        return SkillResult(success=False, message="Missing required parameter: domain")

    years = int(params.get("years", 1))
    nameservers = params.get("nameservers")
    add_whois_guard = params.get("add_whois_guard", "true").lower() == "true"

    sld, tld = skill._split_domain(domain)

    api_params = {
        **skill._base_params(),
        "Command": "namecheap.domains.create",
        "DomainName": domain,
        "Years": str(years),
        "AddFreeWhoisguard": "yes" if add_whois_guard else "no",
        "WGEnabled": "yes" if add_whois_guard else "no",
        # Registrant contact (use Namecheap account defaults)
        "RegistrantFirstName": skill._get_credential("NAMECHEAP_REGISTRANT_FIRST_NAME") or "Domain",
        "RegistrantLastName": skill._get_credential("NAMECHEAP_REGISTRANT_LAST_NAME") or "Admin",
        "RegistrantAddress1": skill._get_credential("NAMECHEAP_REGISTRANT_ADDRESS") or "123 Main St",
        "RegistrantCity": skill._get_credential("NAMECHEAP_REGISTRANT_CITY") or "Los Angeles",
        "RegistrantStateProvince": skill._get_credential("NAMECHEAP_REGISTRANT_STATE") or "CA",
        "RegistrantPostalCode": skill._get_credential("NAMECHEAP_REGISTRANT_ZIP") or "90001",
        "RegistrantCountry": skill._get_credential("NAMECHEAP_REGISTRANT_COUNTRY") or "US",
        "RegistrantPhone": skill._get_credential("NAMECHEAP_REGISTRANT_PHONE") or "+1.5555555555",
        "RegistrantEmailAddress": skill._get_credential("NAMECHEAP_REGISTRANT_EMAIL") or "admin@example.com",
        # Copy registrant to other contacts
        "TechFirstName": skill._get_credential("NAMECHEAP_REGISTRANT_FIRST_NAME") or "Domain",
        "TechLastName": skill._get_credential("NAMECHEAP_REGISTRANT_LAST_NAME") or "Admin",
        "TechAddress1": skill._get_credential("NAMECHEAP_REGISTRANT_ADDRESS") or "123 Main St",
        "TechCity": skill._get_credential("NAMECHEAP_REGISTRANT_CITY") or "Los Angeles",
        "TechStateProvince": skill._get_credential("NAMECHEAP_REGISTRANT_STATE") or "CA",
        "TechPostalCode": skill._get_credential("NAMECHEAP_REGISTRANT_ZIP") or "90001",
        "TechCountry": skill._get_credential("NAMECHEAP_REGISTRANT_COUNTRY") or "US",
        "TechPhone": skill._get_credential("NAMECHEAP_REGISTRANT_PHONE") or "+1.5555555555",
        "TechEmailAddress": skill._get_credential("NAMECHEAP_REGISTRANT_EMAIL") or "admin@example.com",
        "AdminFirstName": skill._get_credential("NAMECHEAP_REGISTRANT_FIRST_NAME") or "Domain",
        "AdminLastName": skill._get_credential("NAMECHEAP_REGISTRANT_LAST_NAME") or "Admin",
        "AdminAddress1": skill._get_credential("NAMECHEAP_REGISTRANT_ADDRESS") or "123 Main St",
        "AdminCity": skill._get_credential("NAMECHEAP_REGISTRANT_CITY") or "Los Angeles",
        "AdminStateProvince": skill._get_credential("NAMECHEAP_REGISTRANT_STATE") or "CA",
        "AdminPostalCode": skill._get_credential("NAMECHEAP_REGISTRANT_ZIP") or "90001",
        "AdminCountry": skill._get_credential("NAMECHEAP_REGISTRANT_COUNTRY") or "US",
        "AdminPhone": skill._get_credential("NAMECHEAP_REGISTRANT_PHONE") or "+1.5555555555",
        "AdminEmailAddress": skill._get_credential("NAMECHEAP_REGISTRANT_EMAIL") or "admin@example.com",
        "AuxBillingFirstName": skill._get_credential("NAMECHEAP_REGISTRANT_FIRST_NAME") or "Domain",
        "AuxBillingLastName": skill._get_credential("NAMECHEAP_REGISTRANT_LAST_NAME") or "Admin",
        "AuxBillingAddress1": skill._get_credential("NAMECHEAP_REGISTRANT_ADDRESS") or "123 Main St",
        "AuxBillingCity": skill._get_credential("NAMECHEAP_REGISTRANT_CITY") or "Los Angeles",
        "AuxBillingStateProvince": skill._get_credential("NAMECHEAP_REGISTRANT_STATE") or "CA",
        "AuxBillingPostalCode": skill._get_credential("NAMECHEAP_REGISTRANT_ZIP") or "90001",
        "AuxBillingCountry": skill._get_credential("NAMECHEAP_REGISTRANT_COUNTRY") or "US",
        "AuxBillingPhone": skill._get_credential("NAMECHEAP_REGISTRANT_PHONE") or "+1.5555555555",
        "AuxBillingEmailAddress": skill._get_credential("NAMECHEAP_REGISTRANT_EMAIL") or "admin@example.com",
    }

    if nameservers:
        api_params["Nameservers"] = nameservers

    resp = await skill.http.get(skill._api_url, params=api_params)

    if resp.status_code != 200:
        return SkillResult(success=False, message=f"API request failed: HTTP {resp.status_code}")

    root = _parse_xml(resp.text)
    error = _check_errors(root)
    if error:
        return SkillResult(success=False, message=f"Registration failed: {error}")

    command_resp = root.find(_ns("CommandResponse"))
    if command_resp is None:
        return SkillResult(success=False, message="Invalid API response: missing CommandResponse")

    result = command_resp.find(_ns("DomainCreateResult"))
    if result is not None:
        registered = result.attrib.get("Registered", "false").lower() == "true"
        if registered:
            return SkillResult(
                success=True,
                message=f"Domain {domain} registered for {years} year(s)",
                data={
                    "domain": domain,
                    "domain_id": result.attrib.get("DomainID"),
                    "order_id": result.attrib.get("OrderID"),
                    "transaction_id": result.attrib.get("TransactionID"),
                    "charged_amount": result.attrib.get("ChargedAmount"),
                    "years": years,
                    "whois_guard": add_whois_guard,
                },
                cost=float(result.attrib.get("ChargedAmount", "0")),
            )
        return SkillResult(success=False, message=f"Domain registration not confirmed for {domain}")

    return SkillResult(success=False, message="Unexpected API response during registration")


async def get_domains(skill, params: Dict) -> SkillResult:
    """List all domains in the Namecheap account."""
    page = int(params.get("page", 1))
    page_size = min(int(params.get("page_size", 20)), 100)
    sort_by = params.get("sort_by", "NAME")

    api_params = {
        **skill._base_params(),
        "Command": "namecheap.domains.getList",
        "Page": str(page),
        "PageSize": str(page_size),
        "SortBy": sort_by,
    }

    resp = await skill.http.get(skill._api_url, params=api_params)

    if resp.status_code != 200:
        return SkillResult(success=False, message=f"API request failed: HTTP {resp.status_code}")

    root = _parse_xml(resp.text)
    error = _check_errors(root)
    if error:
        return SkillResult(success=False, message=f"Namecheap API error: {error}")

    command_resp = root.find(_ns("CommandResponse"))
    if command_resp is None:
        return SkillResult(success=False, message="Invalid API response: missing CommandResponse")

    domain_list = command_resp.find(_ns("DomainGetListResult"))
    domains = []
    if domain_list is not None:
        for d in domain_list.findall(_ns("Domain")):
            domains.append({
                "id": d.attrib.get("ID"),
                "name": d.attrib.get("Name"),
                "user": d.attrib.get("User"),
                "created": d.attrib.get("Created"),
                "expires": d.attrib.get("Expires"),
                "is_expired": d.attrib.get("IsExpired", "false").lower() == "true",
                "is_locked": d.attrib.get("IsLocked", "false").lower() == "true",
                "auto_renew": d.attrib.get("AutoRenew", "false").lower() == "true",
                "whois_guard": d.attrib.get("WhoisGuard", ""),
            })

    # Get paging info
    paging = command_resp.find(_ns("Paging"))
    total = 0
    if paging is not None:
        total_elem = paging.find(_ns("TotalItems"))
        if total_elem is not None and total_elem.text:
            total = int(total_elem.text)

    return SkillResult(
        success=True,
        message=f"Found {total} domains (showing page {page})",
        data={"domains": domains, "total": total, "page": page, "page_size": page_size}
    )


async def set_dns(skill, params: Dict) -> SkillResult:
    """Set DNS host records for a domain."""
    domain = params.get("domain")
    records_str = params.get("records")
    if not domain or not records_str:
        return SkillResult(success=False, message="Missing required parameters: domain, records")

    import json
    try:
        records = json.loads(records_str)
    except json.JSONDecodeError as e:
        return SkillResult(success=False, message=f"Invalid records JSON: {e}")

    if not isinstance(records, list) or not records:
        return SkillResult(success=False, message="Records must be a non-empty JSON array")

    sld, tld = skill._split_domain(domain)

    api_params = {
        **skill._base_params(),
        "Command": "namecheap.domains.dns.setHosts",
        "SLD": sld,
        "TLD": tld,
    }

    # Add each record as numbered parameters
    for i, record in enumerate(records, 1):
        record_type = record.get("type", "A").upper()
        host = record.get("host", "@")
        value = record.get("value", "")
        ttl = record.get("ttl", 1800)
        mx_pref = record.get("mx_pref", 10)

        api_params[f"HostName{i}"] = host
        api_params[f"RecordType{i}"] = record_type
        api_params[f"Address{i}"] = value
        api_params[f"TTL{i}"] = str(ttl)
        if record_type == "MX":
            api_params[f"MXPref{i}"] = str(mx_pref)

    resp = await skill.http.get(skill._api_url, params=api_params)

    if resp.status_code != 200:
        return SkillResult(success=False, message=f"API request failed: HTTP {resp.status_code}")

    root = _parse_xml(resp.text)
    error = _check_errors(root)
    if error:
        return SkillResult(success=False, message=f"DNS set failed: {error}")

    command_resp = root.find(_ns("CommandResponse"))
    if command_resp is None:
        return SkillResult(success=False, message="Invalid API response: missing CommandResponse")

    result = command_resp.find(_ns("DomainDNSSetHostsResult"))
    if result is not None:
        is_success = result.attrib.get("IsSuccess", "false").lower() == "true"
        if is_success:
            return SkillResult(
                success=True,
                message=f"DNS records set for {domain} ({len(records)} records)",
                data={"domain": domain, "records_count": len(records), "records": records}
            )

    return SkillResult(success=False, message=f"Failed to set DNS records for {domain}")


async def get_dns(skill, params: Dict) -> SkillResult:
    """Get current DNS host records for a domain."""
    domain = params.get("domain")
    if not domain:
        return SkillResult(success=False, message="Missing required parameter: domain")

    sld, tld = skill._split_domain(domain)

    api_params = {
        **skill._base_params(),
        "Command": "namecheap.domains.dns.getHosts",
        "SLD": sld,
        "TLD": tld,
    }

    resp = await skill.http.get(skill._api_url, params=api_params)

    if resp.status_code != 200:
        return SkillResult(success=False, message=f"API request failed: HTTP {resp.status_code}")

    root = _parse_xml(resp.text)
    error = _check_errors(root)
    if error:
        return SkillResult(success=False, message=f"Namecheap API error: {error}")

    command_resp = root.find(_ns("CommandResponse"))
    if command_resp is None:
        return SkillResult(success=False, message="Invalid API response: missing CommandResponse")

    hosts_result = command_resp.find(_ns("DomainDNSGetHostsResult"))
    records = []
    if hosts_result is not None:
        for host in hosts_result.findall(_ns("host")):
            records.append({
                "host_id": host.attrib.get("HostId"),
                "host": host.attrib.get("Name"),
                "type": host.attrib.get("Type"),
                "value": host.attrib.get("Address"),
                "ttl": host.attrib.get("TTL"),
                "mx_pref": host.attrib.get("MXPref"),
                "is_active": host.attrib.get("IsActive", "false").lower() == "true",
            })

    return SkillResult(
        success=True,
        message=f"Found {len(records)} DNS records for {domain}",
        data={"domain": domain, "records": records}
    )


async def renew_domain(skill, params: Dict) -> SkillResult:
    """Renew an existing domain registration."""
    domain = params.get("domain")
    if not domain:
        return SkillResult(success=False, message="Missing required parameter: domain")

    years = int(params.get("years", 1))

    api_params = {
        **skill._base_params(),
        "Command": "namecheap.domains.renew",
        "DomainName": domain,
        "Years": str(years),
    }

    resp = await skill.http.get(skill._api_url, params=api_params)

    if resp.status_code != 200:
        return SkillResult(success=False, message=f"API request failed: HTTP {resp.status_code}")

    root = _parse_xml(resp.text)
    error = _check_errors(root)
    if error:
        return SkillResult(success=False, message=f"Renewal failed: {error}")

    command_resp = root.find(_ns("CommandResponse"))
    if command_resp is None:
        return SkillResult(success=False, message="Invalid API response: missing CommandResponse")

    result = command_resp.find(_ns("DomainRenewResult"))
    if result is not None:
        renewed = result.attrib.get("DomainName", "")
        order_id = result.attrib.get("OrderID", "")
        transaction_id = result.attrib.get("TransactionID", "")
        charged = result.attrib.get("ChargedAmount", "0")

        return SkillResult(
            success=True,
            message=f"Domain {domain} renewed for {years} year(s) (${charged})",
            data={
                "domain": renewed,
                "order_id": order_id,
                "transaction_id": transaction_id,
                "charged_amount": charged,
                "years": years,
            },
            cost=float(charged),
        )

    return SkillResult(success=False, message=f"Unexpected API response during renewal of {domain}")
