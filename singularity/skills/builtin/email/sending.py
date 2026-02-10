"""
Email Skill - Sending Operations

Contains send_email, _send_resend, _send_sendgrid, send_template, send_bulk, check_delivery.
"""

from typing import Dict, List
from singularity.skills.base import SkillResult


async def send_email(skill, to: str, subject: str, body: str,
                     from_name: str = None, from_email: str = None,
                     html: bool = False) -> SkillResult:
    """Send an email"""
    if not to or not subject or not body:
        return SkillResult(success=False, message="To, subject, and body required")

    if skill._provider == "resend":
        return await _send_resend(skill, to, subject, body, from_name, from_email, html)
    elif skill._provider == "sendgrid":
        return await _send_sendgrid(skill, to, subject, body, from_name, from_email, html)
    else:
        return SkillResult(success=False, message="No email provider available")


async def _send_resend(skill, to: str, subject: str, body: str,
                       from_name: str = None, from_email: str = None,
                       html: bool = False) -> SkillResult:
    """Send via Resend API"""
    api_key = skill.credentials.get("RESEND_API_KEY")
    default_from = skill.credentials.get("EMAIL_FROM", "agent@wisent.ai")

    from_addr = from_email or default_from
    if from_name:
        from_addr = f"{from_name} <{from_addr}>"

    data = {
        "from": from_addr,
        "to": [to] if isinstance(to, str) else to,
        "subject": subject
    }

    if html:
        data["html"] = body
    else:
        data["text"] = body

    response = await skill.http.post(
        "https://api.resend.com/emails",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        },
        json=data
    )

    if response.status_code == 200:
        result = response.json()
        return SkillResult(
            success=True,
            message=f"Email sent to {to}",
            data={
                "email_id": result.get("id"),
                "to": to,
                "subject": subject
            },
            cost=0.001
        )
    else:
        return SkillResult(success=False, message=f"Resend error: {response.text}")


async def _send_sendgrid(skill, to: str, subject: str, body: str,
                         from_name: str = None, from_email: str = None,
                         html: bool = False) -> SkillResult:
    """Send via SendGrid API"""
    api_key = skill.credentials.get("SENDGRID_API_KEY")
    default_from = skill.credentials.get("EMAIL_FROM", "agent@wisent.ai")

    data = {
        "personalizations": [{"to": [{"email": to}]}],
        "from": {
            "email": from_email or default_from,
            "name": from_name or "Agent"
        },
        "subject": subject,
        "content": [{
            "type": "text/html" if html else "text/plain",
            "value": body
        }]
    }

    response = await skill.http.post(
        "https://api.sendgrid.com/v3/mail/send",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        },
        json=data
    )

    if response.status_code in [200, 202]:
        return SkillResult(
            success=True,
            message=f"Email sent to {to}",
            data={"to": to, "subject": subject},
            cost=0.001
        )
    else:
        return SkillResult(success=False, message=f"SendGrid error: {response.text}")


async def send_template(skill, to: str, template_id: str, variables: Dict) -> SkillResult:
    """Send email using template"""
    if skill._provider != "resend":
        return SkillResult(success=False, message="Templates only supported with Resend")

    return SkillResult(success=False, message="Template feature not yet implemented")


async def send_bulk(skill, recipients: List[Dict], subject: str, body: str) -> SkillResult:
    """Send bulk emails"""
    if not recipients:
        return SkillResult(success=False, message="Recipients required")

    sent = 0
    failed = 0
    errors = []

    for recipient in recipients:
        email = recipient.get("email")
        name = recipient.get("name")
        vars = recipient.get("variables", {})

        personalized_body = body
        personalized_subject = subject
        for key, value in vars.items():
            personalized_body = personalized_body.replace(f"{{{{{key}}}}}", str(value))
            personalized_subject = personalized_subject.replace(f"{{{{{key}}}}}", str(value))

        result = await send_email(
            skill, email, personalized_subject, personalized_body,
            from_name=None, from_email=None, html=False
        )

        if result.success:
            sent += 1
        else:
            failed += 1
            errors.append({"email": email, "error": result.message})

    return SkillResult(
        success=failed == 0,
        message=f"Sent {sent}/{len(recipients)} emails",
        data={
            "sent": sent,
            "failed": failed,
            "total": len(recipients),
            "errors": errors[:10]
        },
        cost=sent * 0.001
    )


async def check_delivery(skill, email_id: str) -> SkillResult:
    """Check email delivery status"""
    if skill._provider != "resend":
        return SkillResult(success=False, message="Delivery tracking only supported with Resend")

    api_key = skill.credentials.get("RESEND_API_KEY")

    response = await skill.http.get(
        f"https://api.resend.com/emails/{email_id}",
        headers={"Authorization": f"Bearer {api_key}"}
    )

    if response.status_code == 200:
        data = response.json()
        return SkillResult(
            success=True,
            message=f"Email status: {data.get('status')}",
            data={
                "id": data.get("id"),
                "status": data.get("status"),
                "to": data.get("to"),
                "subject": data.get("subject"),
                "created_at": data.get("created_at")
            }
        )
    else:
        return SkillResult(success=False, message=f"Failed to check status: {response.text}")
