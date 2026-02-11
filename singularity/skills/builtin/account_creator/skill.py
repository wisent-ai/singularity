#!/usr/bin/env python3
"""
Account Creator Skill - Core Class

AccountCreator, TempEmailService, and credential generation helpers.
"""

import os
import asyncio
import random
import string
from datetime import datetime
from typing import Dict, List, Optional
from singularity.skills.base import Skill, SkillResult, SkillManifest, SkillAction

try:
    from ..browser import BrowserSkill
except ImportError:
    BrowserSkill = None  # type: ignore[assignment,misc]

try:
    from ..captcha import CaptchaSolver, get_enabled_sites, get_site_config, SITE_CONFIGS
except ImportError:
    CaptchaSolver = None  # type: ignore[assignment,misc]
    get_enabled_sites = None  # type: ignore[assignment]
    get_site_config = None  # type: ignore[assignment]
    SITE_CONFIGS = {}

try:
    from ..proxy import ProxySkill
except ImportError:
    ProxySkill = None  # type: ignore[assignment,misc]

try:
    from ..resend import ResendSkill
except ImportError:
    ResendSkill = None  # type: ignore[assignment,misc]


def generate_username(prefix: str = "") -> str:
    """Generate a random username"""
    adjectives = ["happy", "clever", "swift", "bright", "cool", "epic", "mega", "super", "ultra", "hyper"]
    nouns = ["trader", "wolf", "hawk", "phoenix", "dragon", "tiger", "lion", "bear", "eagle", "shark"]
    numbers = ''.join(random.choices(string.digits, k=random.randint(2, 4)))
    if prefix:
        return f"{prefix}{random.choice(adjectives)}{random.choice(nouns)}{numbers}"
    return f"{random.choice(adjectives)}{random.choice(nouns)}{numbers}"


def generate_password(length: int = 16) -> str:
    """Generate a strong random password"""
    chars = string.ascii_letters + string.digits + "!@#$%^&*"
    password = [
        random.choice(string.ascii_uppercase),
        random.choice(string.ascii_lowercase),
        random.choice(string.digits),
        random.choice("!@#$%^&*")
    ]
    password += random.choices(chars, k=length - 4)
    random.shuffle(password)
    return ''.join(password)


def generate_email(domain: str = None) -> str:
    """Generate a random email address using agent's own domain"""
    if domain is None:
        domain = os.environ.get("AGENT_DOMAIN")
        if not domain:
            raise ValueError(
                "Agent must have its own domain configured (AGENT_DOMAIN). "
                "Use AgentSetupSkill.setup_domain() first to purchase a domain."
            )
    username = ''.join(random.choices(string.ascii_lowercase + string.digits, k=12))
    return f"{username}@{domain}"


class TempEmailService:
    """Temp email service using mail.tm (free, no API key needed)"""

    def __init__(self):
        self.base_url = "https://api.mail.tm"
        self.email = None
        self.password = None
        self.token = None
        self.account_id = None

    async def create_email(self) -> str:
        """Create a new temp email address"""
        import httpx
        async with httpx.AsyncClient() as client:
            domains_resp = await client.get(f"{self.base_url}/domains")
            domains = domains_resp.json().get("hydra:member", [])
            if not domains:
                raise Exception("No domains available")
            domain = domains[0]["domain"]
            username = ''.join(random.choices(string.ascii_lowercase + string.digits, k=10))
            self.email = f"{username}@{domain}"
            self.password = ''.join(random.choices(string.ascii_letters + string.digits, k=16))
            create_resp = await client.post(
                f"{self.base_url}/accounts",
                json={"address": self.email, "password": self.password}
            )
            if create_resp.status_code != 201:
                raise Exception(f"Failed to create email: {create_resp.text}")
            self.account_id = create_resp.json()["id"]
            token_resp = await client.post(
                f"{self.base_url}/token",
                json={"address": self.email, "password": self.password}
            )
            if token_resp.status_code != 200:
                raise Exception(f"Failed to get token: {token_resp.text}")
            self.token = token_resp.json()["token"]
            return self.email

    async def get_messages(self) -> list:
        """Get all messages in inbox"""
        import httpx
        if not self.token:
            raise Exception("No email created yet - call create_email() first")
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{self.base_url}/messages",
                headers={"Authorization": f"Bearer {self.token}"}
            )
            if resp.status_code != 200:
                return []
            return resp.json().get("hydra:member", [])

    async def get_message(self, message_id: str) -> dict:
        """Get full message content"""
        import httpx
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{self.base_url}/messages/{message_id}",
                headers={"Authorization": f"Bearer {self.token}"}
            )
            if resp.status_code != 200:
                return {}
            return resp.json()

    async def wait_for_code(self, sender_contains: str = None, max_wait: int = 120) -> str:
        """Wait for verification code email and extract the code"""
        import re
        start_time = asyncio.get_event_loop().time()
        while asyncio.get_event_loop().time() - start_time < max_wait:
            messages = await self.get_messages()
            for msg in messages:
                if sender_contains:
                    from_addr = msg.get("from", {}).get("address", "").lower()
                    if sender_contains.lower() not in from_addr:
                        continue
                full_msg = await self.get_message(msg["id"])
                body = full_msg.get("text", "") or full_msg.get("html", "")
                code_match = re.search(r'\b(\d{6})\b', body)
                if code_match:
                    return code_match.group(1)
                code_match = re.search(r'\b(\d{4})\b', body)
                if code_match:
                    return code_match.group(1)
            await asyncio.sleep(5)
        return None


class AccountCreator(Skill):
    """Automates account creation on social platforms."""

    def __init__(self, credentials: Dict[str, str] = None, use_proxy: bool = True):
        super().__init__(credentials)
        self.browser: Optional[BrowserSkill] = None
        self.captcha: Optional[CaptchaSolver] = None
        self.proxy: Optional[ProxySkill] = None
        self.resend: Optional[ResendSkill] = None
        self.created_accounts: List[Dict] = []
        self.use_proxy = use_proxy
        self._proxy_config: Optional[Dict] = None
        self._email_domain = os.environ.get("AGENT_DOMAIN", "ralph.agents.trade.wisent.ai")

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="account_creator", name="Account Creator", version="1.0.0",
            category="automation",
            description="Automated account creation on social platforms with captcha solving",
            required_credentials=["TWOCAPTCHA_API_KEY"],
            install_cost=0,
            actions=[
                SkillAction(name="create_account", description="Create a single account on a platform",
                    parameters={
                        "site": {"type": "string", "required": True, "description": "Site domain"},
                        "username": {"type": "string", "required": False},
                        "email": {"type": "string", "required": False},
                        "password": {"type": "string", "required": False}
                    },
                    estimated_cost=0.01, estimated_duration_seconds=60, success_probability=0.7),
                SkillAction(name="create_bulk", description="Create multiple accounts on a platform",
                    parameters={
                        "site": {"type": "string", "required": True},
                        "count": {"type": "integer", "required": True},
                        "username_prefix": {"type": "string", "required": False}
                    },
                    estimated_cost=0.01, estimated_duration_seconds=60, success_probability=0.6),
                SkillAction(name="verify_email", description="Complete email verification for an account",
                    parameters={
                        "email": {"type": "string", "required": True},
                        "site": {"type": "string", "required": True}
                    },
                    estimated_cost=0, estimated_duration_seconds=30, success_probability=0.8),
                SkillAction(name="get_created_accounts", description="Get list of all created accounts",
                    parameters={}, estimated_cost=0, estimated_duration_seconds=1, success_probability=1.0),
                SkillAction(name="test_site", description="Test if account creation works on a site",
                    parameters={"site": {"type": "string", "required": True}},
                    estimated_cost=0.003, estimated_duration_seconds=30, success_probability=0.8)
            ]
        )

    async def _init_proxy(self, country: str = "US"):
        if self.use_proxy and self.proxy is None:
            self.proxy = ProxySkill(credentials=self.credentials)
            if self.proxy.check_credentials():
                result = await self.proxy.execute("get_proxy", {"country": country, "sticky": True})
                if result.success:
                    self._proxy_config = result.data.get("playwright_config")

    async def _init_browser(self, country: str = "US"):
        if self.browser is None:
            await self._init_proxy(country)
            self.browser = BrowserSkill(credentials=self.credentials, stealth=True, proxy=self._proxy_config)

    async def _init_captcha(self):
        if self.captcha is None:
            self.captcha = CaptchaSolver(credentials=self.credentials)

    async def _init_resend(self):
        if self.resend is None:
            self.resend = ResendSkill(credentials=self.credentials)

    async def _wait_for_verification_code(self, email: str, sender_contains: str = None, max_wait: int = 120) -> Optional[str]:
        """Wait for verification code email via Resend inbound"""
        await self._init_resend()
        import time
        start = time.time()
        while time.time() - start < max_wait:
            result = await self.resend.execute("get_received_emails", {"limit": 20})
            if not result.success:
                await asyncio.sleep(5)
                continue
            for email_data in result.data.get("emails", []):
                email_to = email_data.get("to", [])
                if isinstance(email_to, list):
                    email_to = [e.get("email", e) if isinstance(e, dict) else e for e in email_to]
                else:
                    email_to = [email_to]
                if email.lower() not in [e.lower() for e in email_to]:
                    continue
                if sender_contains:
                    email_from = email_data.get("from", "")
                    if isinstance(email_from, dict):
                        email_from = email_from.get("email", "")
                    if sender_contains.lower() not in str(email_from).lower():
                        continue
                email_id = email_data.get("id")
                if email_id:
                    extract_result = await self.resend.execute("extract_code", {"email_id": email_id})
                    if extract_result.success and extract_result.data.get("code"):
                        return extract_result.data["code"]
            await asyncio.sleep(5)
        return None

    async def execute(self, action: str, params: Dict) -> SkillResult:
        try:
            from . import platforms, linkedin, twitter
            if action == "create_account":
                return await platforms.create_account(
                    self, params.get("site"), params.get("username"),
                    params.get("email"), params.get("password"))
            elif action == "create_bulk":
                return await platforms.create_bulk(
                    self, params.get("site"), params.get("count", 1),
                    params.get("username_prefix", ""))
            elif action == "verify_email":
                return await platforms.verify_email(self, params.get("email"), params.get("site"))
            elif action == "get_created_accounts":
                return SkillResult(
                    success=True, message=f"Found {len(self.created_accounts)} created accounts",
                    data={"accounts": self.created_accounts})
            elif action == "test_site":
                return await twitter.test_site(self, params.get("site"))
            else:
                return SkillResult(success=False, message=f"Unknown action: {action}")
        except Exception as e:
            return SkillResult(success=False, message=f"Error: {str(e)}")

    async def close(self):
        if self.browser:
            await self.browser.close()
        if self.captcha:
            await self.captcha.close()
        if self.resend:
            await self.resend.close()
