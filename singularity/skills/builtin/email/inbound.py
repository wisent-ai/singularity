"""
Email Skill - Inbound Operations

Contains get_received_emails, get_received_email, wait_for_email, extract_code.
"""

import asyncio
import re
import time
from typing import Optional
from singularity.skills.base import SkillResult


async def get_received_emails(skill, limit: int = 50) -> SkillResult:
    """List received emails from Resend Inbound"""
    if skill._provider != "resend":
        return SkillResult(success=False, message="Inbound email requires Resend")

    api_key = skill.credentials.get("RESEND_API_KEY")
    response = await skill.http.get(
        "https://api.resend.com/emails/receiving",
        headers={"Authorization": f"Bearer {api_key}"},
        params={"limit": limit}
    )

    if response.status_code != 200:
        return SkillResult(success=False, message=f"Failed to get emails: {response.text}")

    data = response.json()
    emails = data.get("data", [])

    return SkillResult(
        success=True,
        message=f"Found {len(emails)} received emails",
        data={"emails": emails, "count": len(emails)}
    )


async def get_received_email(skill, email_id: str) -> SkillResult:
    """Get a specific received email by ID"""
    if skill._provider != "resend":
        return SkillResult(success=False, message="Inbound email requires Resend")

    api_key = skill.credentials.get("RESEND_API_KEY")
    response = await skill.http.get(
        f"https://api.resend.com/emails/receiving/{email_id}",
        headers={"Authorization": f"Bearer {api_key}"}
    )

    if response.status_code != 200:
        return SkillResult(success=False, message=f"Failed to get email: {response.text}")

    email = response.json()

    return SkillResult(
        success=True,
        message=f"Email from {email.get('from', 'unknown')}",
        data=email
    )


async def wait_for_email(skill, to: str, sender_contains: str = None,
                         subject_contains: str = None,
                         max_wait: int = 120) -> SkillResult:
    """Wait for an email matching criteria"""
    if skill._provider != "resend":
        return SkillResult(success=False, message="Inbound email requires Resend")

    api_key = skill.credentials.get("RESEND_API_KEY")
    start_time = time.time()

    while time.time() - start_time < max_wait:
        response = await skill.http.get(
            "https://api.resend.com/emails/receiving",
            headers={"Authorization": f"Bearer {api_key}"},
            params={"limit": 50}
        )

        if response.status_code == 200:
            emails = response.json().get("data", [])

            for email in emails:
                # Check recipient
                email_to = email.get("to", [])
                if isinstance(email_to, list):
                    email_to = [e.get("email", e) if isinstance(e, dict) else e for e in email_to]
                else:
                    email_to = [email_to]

                if to.lower() not in [e.lower() for e in email_to]:
                    continue

                # Check sender
                if sender_contains:
                    email_from = email.get("from", {})
                    if isinstance(email_from, dict):
                        email_from = email_from.get("email", "")
                    if sender_contains.lower() not in email_from.lower():
                        continue

                # Check subject
                if subject_contains:
                    subject = email.get("subject", "")
                    if subject_contains.lower() not in subject.lower():
                        continue

                # Found matching email
                return SkillResult(
                    success=True,
                    message=f"Found email from {email.get('from')}",
                    data=email
                )

        await asyncio.sleep(5)

    return SkillResult(
        success=False,
        message=f"No matching email received within {max_wait} seconds"
    )


async def extract_code(skill, email_id: str, code_length: int = 6) -> SkillResult:
    """Extract verification code from email body"""
    # Get the email
    email_result = await get_received_email(skill, email_id)
    if not email_result.success:
        return email_result

    email = email_result.data
    body = email.get("text", "") or email.get("html", "") or ""

    # Try to find code - look for standalone digits of expected length
    pattern = rf'\b(\d{{{code_length}}})\b'
    match = re.search(pattern, body)

    if match:
        code = match.group(1)
        return SkillResult(
            success=True,
            message=f"Found {code_length}-digit code",
            data={"code": code, "email_id": email_id}
        )

    # Also try common patterns
    patterns = [
        rf'code[:\s]+(\d{{{code_length}}})',
        rf'verification[:\s]+(\d{{{code_length}}})',
        rf'confirm[:\s]+(\d{{{code_length}}})',
    ]

    for p in patterns:
        match = re.search(p, body, re.IGNORECASE)
        if match:
            code = match.group(1)
            return SkillResult(
                success=True,
                message=f"Found {code_length}-digit code",
                data={"code": code, "email_id": email_id}
            )

    return SkillResult(
        success=False,
        message=f"No {code_length}-digit code found in email"
    )
