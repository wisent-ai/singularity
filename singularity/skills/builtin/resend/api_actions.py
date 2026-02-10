"""
Resend Skill - API Actions

Contains send_email, list_domains, add_domain, get_domain_records,
get_received_emails, wait_for_email, extract_code.
"""

import asyncio
import re
from typing import Optional
from singularity.skills.base import SkillResult


async def send_email(skill, to: str, subject: str, body: str, from_email: str = None) -> SkillResult:
    """Send email via API"""
    resp = await skill.http.post(
        "https://api.resend.com/emails",
        headers={"Authorization": f"Bearer {skill._api_key}"},
        json={
            "from": from_email or "noreply@resend.dev",
            "to": [to] if isinstance(to, str) else to,
            "subject": subject,
            "text": body
        }
    )

    if resp.status_code in [200, 201]:
        return SkillResult(success=True, message="Email sent", data=resp.json())
    return SkillResult(success=False, message=f"Failed: {resp.text}")


async def list_domains(skill) -> SkillResult:
    """List domains via API"""
    resp = await skill.http.get(
        "https://api.resend.com/domains",
        headers={"Authorization": f"Bearer {skill._api_key}"}
    )

    if resp.status_code == 200:
        domains = resp.json().get("data", [])
        return SkillResult(
            success=True,
            message=f"Found {len(domains)} domains",
            data={"domains": domains}
        )
    return SkillResult(success=False, message=f"Failed: {resp.text}")


async def add_domain(skill, domain: str) -> SkillResult:
    """Add domain via API"""
    resp = await skill.http.post(
        "https://api.resend.com/domains",
        headers={"Authorization": f"Bearer {skill._api_key}"},
        json={"name": domain}
    )

    if resp.status_code in [200, 201]:
        return SkillResult(success=True, message=f"Domain {domain} added", data=resp.json())
    return SkillResult(success=False, message=f"Failed: {resp.text}")


async def get_domain_records(skill, domain: str) -> SkillResult:
    """Get DNS records for domain"""
    domains_resp = await list_domains(skill)
    if not domains_resp.success:
        return domains_resp

    domain_id = None
    for d in domains_resp.data.get("domains", []):
        if d.get("name") == domain:
            domain_id = d.get("id")
            break

    if not domain_id:
        return SkillResult(success=False, message=f"Domain {domain} not found")

    resp = await skill.http.get(
        f"https://api.resend.com/domains/{domain_id}",
        headers={"Authorization": f"Bearer {skill._api_key}"}
    )

    if resp.status_code == 200:
        data = resp.json()
        return SkillResult(
            success=True,
            message=f"Records for {domain}",
            data={"records": data.get("records", []), "domain_id": domain_id}
        )
    return SkillResult(success=False, message=f"Failed: {resp.text}")


async def get_received_emails(skill, limit: int = 50) -> SkillResult:
    """Get received emails via API"""
    resp = await skill.http.get(
        "https://api.resend.com/emails/receiving",
        headers={"Authorization": f"Bearer {skill._api_key}"},
        params={"limit": limit}
    )

    if resp.status_code == 200:
        emails = resp.json().get("data", [])
        return SkillResult(
            success=True,
            message=f"Found {len(emails)} emails",
            data={"emails": emails}
        )
    return SkillResult(success=False, message=f"Failed: {resp.text}")


async def wait_for_email(skill, to: str, sender_contains: str = None, max_wait: int = 120) -> SkillResult:
    """Wait for email matching criteria"""
    import time
    start = time.time()

    while time.time() - start < max_wait:
        result = await get_received_emails(skill, 50)
        if not result.success:
            await asyncio.sleep(5)
            continue

        for email in result.data.get("emails", []):
            email_to = email.get("to", [])
            if isinstance(email_to, list):
                email_to = [e.get("email", e) if isinstance(e, dict) else e for e in email_to]
            else:
                email_to = [email_to]

            if to.lower() not in [e.lower() for e in email_to]:
                continue

            if sender_contains:
                email_from = email.get("from", {})
                if isinstance(email_from, dict):
                    email_from = email_from.get("email", "")
                if sender_contains.lower() not in str(email_from).lower():
                    continue

            return SkillResult(success=True, message="Email found", data=email)

        await asyncio.sleep(5)

    return SkillResult(success=False, message=f"No email received in {max_wait}s")


async def extract_code(skill, email_id: str) -> SkillResult:
    """Extract verification code from email"""
    resp = await skill.http.get(
        f"https://api.resend.com/emails/receiving/{email_id}",
        headers={"Authorization": f"Bearer {skill._api_key}"}
    )

    if resp.status_code != 200:
        return SkillResult(success=False, message=f"Failed to get email: {resp.text}")

    email = resp.json()
    body = email.get("text", "") or email.get("html", "") or ""

    match = re.search(r'\b(\d{6})\b', body)
    if match:
        return SkillResult(
            success=True,
            message=f"Code: {match.group(1)}",
            data={"code": match.group(1)}
        )

    return SkillResult(success=False, message="No code found")
