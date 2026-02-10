#!/usr/bin/env python3
"""
Email Skill - Core Class

EmailSkill with manifest, execute, provider detection.
Domain management is provided by DomainManagementMixin in provider_helpers.py.
"""

import httpx
from typing import Dict, List, Optional
from singularity.skills.base import Skill, SkillResult, SkillManifest, SkillAction
from .provider_helpers import DomainManagementMixin


class EmailSkill(DomainManagementMixin, Skill):
    """
    Skill for sending and managing emails.

    Supports multiple providers:
    - Resend (preferred)
    - SendGrid
    - SMTP

    Required credentials (one of):
    - RESEND_API_KEY: Resend API key
    - SENDGRID_API_KEY: SendGrid API key
    - EMAIL_SMTP_HOST, EMAIL_SMTP_USER, EMAIL_SMTP_PASS: SMTP credentials
    """

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="email",
            name="Email Management",
            version="1.0.0",
            category="communication",
            description="Send and manage emails via Resend, SendGrid, or SMTP",
            required_credentials=[
                "RESEND_API_KEY",
                "NAMECHEAP_API_KEY",
                "NAMECHEAP_API_USER",
                "NAMECHEAP_USERNAME",
                "NAMECHEAP_CLIENT_IP"
            ],
            install_cost=0,
            actions=[
                SkillAction(
                    name="send_email",
                    description="Send an email",
                    parameters={
                        "to": {"type": "string", "required": True, "description": "Recipient email(s)"},
                        "subject": {"type": "string", "required": True, "description": "Email subject"},
                        "body": {"type": "string", "required": True, "description": "Email body (text or HTML)"},
                        "from_name": {"type": "string", "required": False, "description": "Sender name"},
                        "from_email": {"type": "string", "required": False, "description": "Sender email"},
                        "html": {"type": "boolean", "required": False, "description": "Is body HTML? (default: false)"}
                    },
                    estimated_cost=0.001,
                    estimated_duration_seconds=5,
                    success_probability=0.95
                ),
                SkillAction(
                    name="send_template",
                    description="Send email using a template",
                    parameters={
                        "to": {"type": "string", "required": True, "description": "Recipient email"},
                        "template_id": {"type": "string", "required": True, "description": "Template ID"},
                        "variables": {"type": "object", "required": False, "description": "Template variables"}
                    },
                    estimated_cost=0.001,
                    estimated_duration_seconds=5,
                    success_probability=0.95
                ),
                SkillAction(
                    name="send_bulk",
                    description="Send bulk emails",
                    parameters={
                        "recipients": {"type": "array", "required": True, "description": "List of {email, name, variables}"},
                        "subject": {"type": "string", "required": True, "description": "Email subject"},
                        "body": {"type": "string", "required": True, "description": "Email body"}
                    },
                    estimated_cost=0.001,
                    estimated_duration_seconds=30,
                    success_probability=0.9
                ),
                SkillAction(
                    name="check_delivery",
                    description="Check email delivery status",
                    parameters={
                        "email_id": {"type": "string", "required": True, "description": "Email ID to check"}
                    },
                    estimated_cost=0,
                    estimated_duration_seconds=2,
                    success_probability=0.95
                ),
                SkillAction(
                    name="add_domain",
                    description="Add a domain to Resend for email sending",
                    parameters={
                        "domain": {"type": "string", "required": True, "description": "Domain to add"}
                    },
                    estimated_cost=0,
                    estimated_duration_seconds=5,
                    success_probability=0.95
                ),
                SkillAction(
                    name="get_domain_dns",
                    description="Get DNS records needed to verify a domain",
                    parameters={
                        "domain_id": {"type": "string", "required": True, "description": "Resend domain ID"}
                    },
                    estimated_cost=0,
                    estimated_duration_seconds=2,
                    success_probability=0.95
                ),
                SkillAction(
                    name="verify_domain",
                    description="Verify a domain after DNS records are set",
                    parameters={
                        "domain_id": {"type": "string", "required": True, "description": "Resend domain ID to verify"}
                    },
                    estimated_cost=0,
                    estimated_duration_seconds=5,
                    success_probability=0.9
                ),
                SkillAction(
                    name="list_domains",
                    description="List all domains configured in Resend",
                    parameters={},
                    estimated_cost=0,
                    estimated_duration_seconds=2,
                    success_probability=0.95
                ),
                SkillAction(
                    name="setup_domain",
                    description="Full automated domain setup: add to Resend, configure DNS via Namecheap, verify",
                    parameters={
                        "domain": {"type": "string", "required": True, "description": "Domain to set up for email"}
                    },
                    estimated_cost=0,
                    estimated_duration_seconds=30,
                    success_probability=0.85
                ),
                SkillAction(
                    name="get_received_emails",
                    description="List received emails (Resend Inbound)",
                    parameters={
                        "limit": {"type": "integer", "required": False, "description": "Max emails to return (default: 50)"}
                    },
                    estimated_cost=0,
                    estimated_duration_seconds=2,
                    success_probability=0.95
                ),
                SkillAction(
                    name="get_received_email",
                    description="Get a specific received email by ID",
                    parameters={
                        "email_id": {"type": "string", "required": True, "description": "Email ID to retrieve"}
                    },
                    estimated_cost=0,
                    estimated_duration_seconds=2,
                    success_probability=0.95
                ),
                SkillAction(
                    name="wait_for_email",
                    description="Wait for an email matching criteria (for verification codes)",
                    parameters={
                        "to": {"type": "string", "required": True, "description": "Email address to check"},
                        "sender_contains": {"type": "string", "required": False, "description": "Sender must contain this string"},
                        "subject_contains": {"type": "string", "required": False, "description": "Subject must contain this string"},
                        "max_wait": {"type": "integer", "required": False, "description": "Timeout in seconds (default: 120)"}
                    },
                    estimated_cost=0,
                    estimated_duration_seconds=60,
                    success_probability=0.85
                ),
                SkillAction(
                    name="extract_code",
                    description="Extract verification code from email body",
                    parameters={
                        "email_id": {"type": "string", "required": True, "description": "Email ID to extract code from"},
                        "code_length": {"type": "integer", "required": False, "description": "Expected code length (default: 6)"}
                    },
                    estimated_cost=0,
                    estimated_duration_seconds=1,
                    success_probability=0.9
                )
            ]
        )

    def __init__(self, credentials: Dict[str, str] = None):
        super().__init__(credentials)
        self.http = httpx.AsyncClient()
        self._provider = self._detect_provider()

    def _detect_provider(self) -> str:
        """Detect which email provider to use"""
        if self.credentials.get("RESEND_API_KEY"):
            return "resend"
        elif self.credentials.get("SENDGRID_API_KEY"):
            return "sendgrid"
        elif self.credentials.get("EMAIL_SMTP_HOST"):
            return "smtp"
        return "none"

    def check_credentials(self) -> bool:
        """Check if any email provider is configured"""
        return self._provider != "none"

    async def execute(self, action: str, params: Dict) -> SkillResult:
        """Execute an email action"""
        if not self.check_credentials():
            return SkillResult(
                success=False,
                message="No email provider configured. Set RESEND_API_KEY, SENDGRID_API_KEY, or SMTP credentials."
            )

        try:
            from . import sending, inbound

            if action == "send_email":
                return await sending.send_email(
                    self, params.get("to"), params.get("subject"),
                    params.get("body"), params.get("from_name"),
                    params.get("from_email"), params.get("html", False)
                )
            elif action == "send_template":
                return await sending.send_template(
                    self, params.get("to"), params.get("template_id"),
                    params.get("variables", {})
                )
            elif action == "send_bulk":
                return await sending.send_bulk(
                    self, params.get("recipients"), params.get("subject"),
                    params.get("body")
                )
            elif action == "check_delivery":
                return await sending.check_delivery(self, params.get("email_id"))
            elif action == "add_domain":
                return await self._add_domain(params.get("domain"))
            elif action == "get_domain_dns":
                return await self._get_domain_dns(params.get("domain_id"))
            elif action == "verify_domain":
                return await self._verify_domain(params.get("domain_id"))
            elif action == "list_domains":
                return await self._list_domains()
            elif action == "setup_domain":
                return await self._setup_domain(params.get("domain"))
            elif action == "get_received_emails":
                return await inbound.get_received_emails(self, params.get("limit", 50))
            elif action == "get_received_email":
                return await inbound.get_received_email(self, params.get("email_id"))
            elif action == "wait_for_email":
                return await inbound.wait_for_email(
                    self, params.get("to"), params.get("sender_contains"),
                    params.get("subject_contains"), params.get("max_wait", 120)
                )
            elif action == "extract_code":
                return await inbound.extract_code(
                    self, params.get("email_id"), params.get("code_length", 6)
                )
            else:
                return SkillResult(success=False, message=f"Unknown action: {action}")
        except Exception as e:
            return SkillResult(success=False, message=f"Email error: {str(e)}")

    async def close(self):
        """Clean up"""
        await self.http.aclose()
