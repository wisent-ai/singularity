#!/usr/bin/env python3
"""
Resend Skill

Full Resend management including inbound email setup.
Uses both API and browser automation for complete control.
"""

import asyncio
from typing import Dict, Optional
from singularity.skills.base import Skill, SkillResult, SkillManifest, SkillAction
from ..browser import BrowserSkill
import httpx


class ResendSkill(Skill):
    """
    Complete Resend email management.

    Required credentials:
    - RESEND_API_KEY: For API operations
    - RESEND_EMAIL: Account email (for browser login)
    - RESEND_PASSWORD: Account password (for browser login)

    For DNS setup:
    - NAMECHEAP_API_KEY, NAMECHEAP_API_USER, NAMECHEAP_USERNAME, NAMECHEAP_CLIENT_IP
    """

    def __init__(self, credentials: Dict[str, str] = None):
        super().__init__(credentials)
        self.http = httpx.AsyncClient()
        self.browser: Optional[BrowserSkill] = None
        self._logged_in = False
        self._api_key = self.credentials.get("RESEND_API_KEY")

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="resend",
            name="Resend Email Management",
            version="1.0.0",
            category="email",
            description="Complete Resend management: send, receive, domain setup",
            required_credentials=["RESEND_API_KEY"],
            install_cost=0,
            actions=[
                SkillAction(name="send_email", description="Send an email",
                    parameters={"to": {"type": "string", "required": True},
                        "subject": {"type": "string", "required": True},
                        "body": {"type": "string", "required": True},
                        "from_email": {"type": "string", "required": False}},
                    estimated_cost=0.001, estimated_duration_seconds=5, success_probability=0.95),
                SkillAction(name="list_domains", description="List all domains",
                    parameters={}, estimated_cost=0, estimated_duration_seconds=2, success_probability=0.95),
                SkillAction(name="add_domain", description="Add a domain to Resend",
                    parameters={"domain": {"type": "string", "required": True}},
                    estimated_cost=0, estimated_duration_seconds=5, success_probability=0.9),
                SkillAction(name="get_domain_records", description="Get DNS records needed for a domain",
                    parameters={"domain": {"type": "string", "required": True}},
                    estimated_cost=0, estimated_duration_seconds=2, success_probability=0.95),
                SkillAction(name="enable_inbound", description="Enable inbound email receiving for a domain (via browser)",
                    parameters={"domain": {"type": "string", "required": True}},
                    estimated_cost=0, estimated_duration_seconds=30, success_probability=0.85),
                SkillAction(name="get_inbound_mx", description="Get MX records for inbound email (via browser)",
                    parameters={"domain": {"type": "string", "required": True}},
                    estimated_cost=0, estimated_duration_seconds=15, success_probability=0.85),
                SkillAction(name="get_received_emails", description="Get received emails",
                    parameters={"limit": {"type": "integer", "required": False}},
                    estimated_cost=0, estimated_duration_seconds=2, success_probability=0.95),
                SkillAction(name="wait_for_email", description="Wait for email matching criteria",
                    parameters={"to": {"type": "string", "required": True},
                        "sender_contains": {"type": "string", "required": False},
                        "max_wait": {"type": "integer", "required": False}},
                    estimated_cost=0, estimated_duration_seconds=60, success_probability=0.85),
                SkillAction(name="extract_code", description="Extract verification code from email",
                    parameters={"email_id": {"type": "string", "required": True}},
                    estimated_cost=0, estimated_duration_seconds=1, success_probability=0.9),
                SkillAction(name="setup_domain_full",
                    description="Full domain setup: add domain, enable inbound, configure DNS",
                    parameters={"domain": {"type": "string", "required": True}},
                    estimated_cost=0, estimated_duration_seconds=60, success_probability=0.8)
            ]
        )

    async def execute(self, action: str, params: Dict) -> SkillResult:
        from . import api_actions, browser_actions

        try:
            if action == "send_email":
                return await api_actions.send_email(self, params.get("to"), params.get("subject"),
                    params.get("body"), params.get("from_email"))
            elif action == "list_domains":
                return await api_actions.list_domains(self)
            elif action == "add_domain":
                return await api_actions.add_domain(self, params.get("domain"))
            elif action == "get_domain_records":
                return await api_actions.get_domain_records(self, params.get("domain"))
            elif action == "get_received_emails":
                return await api_actions.get_received_emails(self, params.get("limit", 50))
            elif action == "wait_for_email":
                return await api_actions.wait_for_email(self, params.get("to"),
                    params.get("sender_contains"), params.get("max_wait", 120))
            elif action == "extract_code":
                return await api_actions.extract_code(self, params.get("email_id"))
            elif action == "enable_inbound":
                return await browser_actions.enable_inbound(self, params.get("domain"))
            elif action == "get_inbound_mx":
                return await browser_actions.get_inbound_mx(self, params.get("domain"))
            elif action == "setup_domain_full":
                return await browser_actions.setup_domain_full(self, params.get("domain"))
            else:
                return SkillResult(success=False, message=f"Unknown action: {action}")
        except Exception as e:
            return SkillResult(success=False, message=str(e))

    async def close(self):
        """Cleanup"""
        await self.http.aclose()
        if self.browser:
            await self.browser.close()
