"""
Resend Skill - Browser Actions

Contains enable_inbound, get_inbound_mx, setup_domain_full, setup_namecheap_dns.
Also includes login helper.
"""

import asyncio
import re
from typing import Dict
from ..browser import BrowserSkill
from singularity.skills.base import SkillResult


async def _init_browser(skill):
    """Initialize browser if needed"""
    if skill.browser is None:
        skill.browser = BrowserSkill(credentials=skill.credentials, stealth=True)


async def _login(skill) -> bool:
    """Login to Resend dashboard via Google OAuth"""
    if skill._logged_in:
        return True

    await _init_browser(skill)

    email = skill.credentials.get("RESEND_EMAIL")
    password = skill.credentials.get("RESEND_PASSWORD")

    if not email or not password:
        return False

    await skill.browser.execute("goto", {"url": "https://resend.com/login"})
    await asyncio.sleep(2)

    await skill.browser.execute("click", {"selector": "button:has-text('Google'), a:has-text('Google')"})
    await asyncio.sleep(3)

    await skill.browser.execute("type", {"selector": "input[type='email']", "text": email})
    await skill.browser.execute("click", {"selector": "#identifierNext, button:has-text('Next')"})
    await asyncio.sleep(2)

    await skill.browser.execute("type", {"selector": "input[type='password']", "text": password})
    await skill.browser.execute("click", {"selector": "#passwordNext, button:has-text('Next')"})
    await asyncio.sleep(5)

    current_url = await skill.browser.execute("evaluate", {"script": "window.location.href"})
    url = current_url.data.get("result", "") if current_url.success else ""

    if "resend.com" in url and "login" not in url:
        skill._logged_in = True
        return True

    content = await skill.browser.execute("evaluate", {"script": "document.body.innerText"})
    page_text = content.data.get("result", "") if content.success else ""

    if "Log in" not in page_text and "Domains" in page_text:
        skill._logged_in = True
        return True

    return False


async def enable_inbound(skill, domain: str) -> SkillResult:
    """Enable inbound email for domain via browser"""
    if not await _login(skill):
        return SkillResult(success=False, message="Failed to login to Resend")

    await skill.browser.execute("goto", {"url": "https://resend.com/domains"})
    await asyncio.sleep(2)

    await skill.browser.execute("click", {"selector": f"text={domain}"})
    await asyncio.sleep(2)

    toggle_result = await skill.browser.execute("evaluate", {
        "script": """
            const toggles = document.querySelectorAll("button[role='switch']");
            for (const toggle of toggles) {
                const parent = toggle.closest('div');
                if (parent && parent.innerText.includes('Enable Receiving')) {
                    if (toggle.getAttribute('aria-checked') === 'false') {
                        toggle.click();
                        return 'clicked';
                    } else {
                        return 'already_enabled';
                    }
                }
            }
            return 'not_found';
        """
    })

    status = toggle_result.data.get("result", "unknown") if toggle_result.success else "error"
    await asyncio.sleep(3)

    content = await skill.browser.execute("evaluate", {"script": "document.body.innerText"})
    page_text = content.data.get("result", "") if content.success else ""

    mx_record = None
    mx_name = None

    mx_match = re.search(r'(inbound-smtp[^\s]*\.amazonaws\.com)', page_text, re.IGNORECASE)
    if mx_match:
        mx_record = mx_match.group(1)

    parts = domain.split('.')
    if len(parts) > 2:
        mx_name = '.'.join(parts[:-2])

    return SkillResult(
        success=True,
        message=f"Inbound {'enabled' if status == 'clicked' else status} for {domain}",
        data={
            "domain": domain,
            "mx_record": mx_record,
            "mx_name": mx_name,
            "mx_priority": 10,
            "toggle_status": status
        }
    )


async def get_inbound_mx(skill, domain: str) -> SkillResult:
    """Get MX records for inbound via browser"""
    if not await _login(skill):
        return SkillResult(success=False, message="Failed to login to Resend")

    await skill.browser.execute("goto", {"url": "https://resend.com/domains"})
    await asyncio.sleep(2)

    await skill.browser.execute("click", {"selector": f"a:has-text('{domain}')"})
    await asyncio.sleep(2)

    mx_record = await skill.browser.execute("evaluate", {
        "script": """
            const rows = document.querySelectorAll('tr, div');
            for (const row of rows) {
                if (row.innerText.includes('MX') && row.innerText.includes('inbound')) {
                    return row.innerText;
                }
            }
            return null;
        """
    })

    if mx_record.success and mx_record.data.get("result"):
        return SkillResult(
            success=True,
            message="MX record found",
            data={"mx_record": mx_record.data["result"]}
        )

    return SkillResult(success=False, message="MX record not found")


async def setup_domain_full(skill, domain: str) -> SkillResult:
    """Full domain setup: add, enable inbound, get records, setup DNS"""
    from . import api_actions

    results = []

    add_result = await api_actions.add_domain(skill, domain)
    results.append(("add_domain", add_result))

    inbound_result = await enable_inbound(skill, domain)
    results.append(("enable_inbound", inbound_result))

    records_result = await api_actions.get_domain_records(skill, domain)
    results.append(("get_records", records_result))

    mx_result = await get_inbound_mx(skill, domain)
    results.append(("get_mx", mx_result))

    if all(skill.credentials.get(k) for k in ["NAMECHEAP_API_KEY", "NAMECHEAP_API_USER"]):
        dns_result = await _setup_namecheap_dns(skill, domain, records_result.data, mx_result.data)
        results.append(("setup_dns", dns_result))

    return SkillResult(
        success=True,
        message=f"Domain {domain} setup complete",
        data={"steps": [(name, r.success, r.message) for name, r in results]}
    )


async def _setup_namecheap_dns(skill, domain: str, records: Dict, mx_data: Dict) -> SkillResult:
    """Add DNS records to Namecheap"""
    parts = domain.split(".")
    sld = parts[-2]
    tld = parts[-1]

    nc_params = {
        "ApiUser": skill.credentials["NAMECHEAP_API_USER"],
        "ApiKey": skill.credentials["NAMECHEAP_API_KEY"],
        "UserName": skill.credentials["NAMECHEAP_USERNAME"],
        "ClientIp": skill.credentials["NAMECHEAP_CLIENT_IP"],
        "Command": "namecheap.domains.dns.setHosts",
        "SLD": sld,
        "TLD": tld
    }

    idx = 1
    for rec in records.get("records", []):
        nc_params[f"HostName{idx}"] = rec.get("name", "@")
        nc_params[f"RecordType{idx}"] = rec.get("record", rec.get("type", "TXT"))
        nc_params[f"Address{idx}"] = rec.get("value", "")
        nc_params[f"TTL{idx}"] = "1799"
        idx += 1

    if mx_data.get("mx_record"):
        nc_params[f"HostName{idx}"] = "@"
        nc_params[f"RecordType{idx}"] = "MX"
        nc_params[f"Address{idx}"] = mx_data["mx_record"]
        nc_params[f"MXPref{idx}"] = "10"
        nc_params[f"TTL{idx}"] = "1799"

    resp = await skill.http.get(
        "https://api.namecheap.com/xml.response",
        params=nc_params
    )

    if "OK" in resp.text:
        return SkillResult(success=True, message="DNS records set")
    return SkillResult(success=False, message=f"DNS setup failed: {resp.text[:200]}")
