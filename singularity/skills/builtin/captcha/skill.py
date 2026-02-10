#!/usr/bin/env python3
"""
Captcha Solver Skill - Core Class & Site Configuration

Contains CaptchaSolver class, manifest, execute routing, site configs,
and human-like interaction helpers.
"""

import os
import asyncio
import base64
import httpx
import random
import math
from typing import Dict, Optional, List
from singularity.skills.base import Skill, SkillResult, SkillManifest, SkillAction


async def human_like_mouse_move(page, start_x, start_y, end_x, end_y, steps=None):
    """Move mouse in a human-like curved path with variable speed."""
    if steps is None:
        distance = math.sqrt((end_x - start_x)**2 + (end_y - start_y)**2)
        steps = max(10, int(distance / 10))
    ctrl_x = (start_x + end_x) / 2 + random.uniform(-50, 50)
    ctrl_y = (start_y + end_y) / 2 + random.uniform(-50, 50)
    for i in range(steps + 1):
        t = i / steps
        x = (1-t)**2 * start_x + 2*(1-t)*t * ctrl_x + t**2 * end_x
        y = (1-t)**2 * start_y + 2*(1-t)*t * ctrl_y + t**2 * end_y
        x += random.uniform(-2, 2)
        y += random.uniform(-2, 2)
        await page.mouse.move(x, y)
        speed_factor = 1 - 4 * (t - 0.5)**2
        delay = random.uniform(5, 15) * (1 + 0.5 * (1 - speed_factor))
        await asyncio.sleep(delay / 1000)


async def human_like_click(page, x, y, current_x=None, current_y=None):
    """Move to element and click with human-like behavior."""
    if current_x is None: current_x = random.uniform(100, 500)
    if current_y is None: current_y = random.uniform(100, 300)
    await human_like_mouse_move(page, current_x, current_y, x, y)
    await asyncio.sleep(random.uniform(0.05, 0.15))
    await page.mouse.click(x + random.uniform(-3, 3), y + random.uniform(-3, 3))
    return (x, y)


SITE_CONFIGS = {
    "reddit.com": {
        "name": "Reddit", "signup_url": "https://www.reddit.com/register/",
        "login_url": "https://www.reddit.com/login/", "captcha_type": "recaptcha_invisible",
        "selectors": {"email": "input[type='email']", "code": "input[name='code']",
                      "username": "input[name='username']", "password": "input[name='password']",
                      "continue": "button:has-text('Continue')", "submit": "button:has-text('Continue')"},
        "requires_email_verification": True, "flow": ["email", "code", "username", "password"]},
    "tiktok.com": {
        "name": "TikTok", "signup_url": "https://www.tiktok.com/signup/phone-or-email/email",
        "login_url": "https://www.tiktok.com/login", "captcha_type": "slider",
        "selectors": {"email": "input[name='email']", "password": "input[type='password']",
                      "code": "input[placeholder*='6-digit']", "send_code": "button[data-e2e='send-code-button']",
                      "submit": "button[type='submit']", "month_select": "select",
                      "day_select": "select", "year_select": "select"},
        "requires_email_verification": True},
    "linkedin.com": {
        "name": "LinkedIn", "signup_url": "https://www.linkedin.com/signup",
        "login_url": "https://www.linkedin.com/login", "captcha_type": "none",
        "selectors": {"email": "input#email-address", "password": "input#password",
                      "first_name": "input#first-name", "last_name": "input#last-name",
                      "submit": "button#join-form-submit"}},
    "twitter.com": {
        "name": "Twitter/X", "signup_url": "https://twitter.com/i/flow/signup",
        "login_url": "https://twitter.com/i/flow/login", "captcha_type": "arkose",
        "selectors": {"username": "input[name='username']", "email": "input[name='email']",
                      "password": "input[name='password']",
                      "submit": "button[data-testid='LoginForm_Login_Button']"}}
}


def get_enabled_sites() -> List[str]:
    """Get list of enabled sites from environment variables"""
    sites = []
    for i in range(1, 10):
        site = os.environ.get(f"SITE_{i}", "").strip()
        if site:
            sites.append(site.lower().replace("www.", ""))
    return sites


def get_site_config(site: str) -> Optional[Dict]:
    """Get configuration for a specific site"""
    site = site.lower().replace("www.", "").replace("https://", "").replace("http://", "")
    return SITE_CONFIGS.get(site)


class CaptchaSolver(Skill):
    """Solve captchas using external services OR AI vision."""

    CAPSOLVER_API = "https://api.capsolver.com"
    TWOCAPTCHA_API = "https://2captcha.com"
    ANTICAPTCHA_API = "https://api.anti-captcha.com"
    NEXTCAPTCHA_API = "https://api.nextcaptcha.com"
    NOCAPTCHAAI_API = "https://token.nocaptchaai.com"
    NOPECHA_API = "https://api.nopecha.com"

    SERVICE_PRIORITY = [
        ("NOPECHA_API_KEY", "nopecha"), ("ANTICAPTCHA_API_KEY", "anticaptcha"),
        ("NEXTCAPTCHA_API_KEY", "nextcaptcha"), ("NOCAPTCHAAI_API_KEY", "nocaptchaai"),
        ("TWOCAPTCHA_API_KEY", "2captcha"), ("CAPSOLVER_API_KEY", "capsolver"),
    ]

    def __init__(self, credentials: Dict[str, str] = None, proxy: Dict[str, str] = None,
                 user_agent: str = None):
        super().__init__(credentials)
        self.http = httpx.AsyncClient()
        self._service = None
        self._api_key = None
        self._available_services = {}
        self._proxy = proxy
        self._user_agent = user_agent
        if proxy:
            print(f"  [CAPTCHA] Proxy configured: {proxy.get('address')}:{proxy.get('port')}")
        if user_agent:
            print(f"  [CAPTCHA] User-Agent configured: {user_agent[:50]}...")
        for key_name, service_name in self.SERVICE_PRIORITY:
            api_key = (credentials or {}).get(key_name) or os.environ.get(key_name)
            if api_key:
                self._available_services[service_name] = api_key
                if self._service is None:
                    self._service = service_name
                    self._api_key = api_key
        self._model_provider = os.environ.get("AGENT_MODEL_PROVIDER", "anthropic")
        self._model_name = os.environ.get("AGENT_MODEL_NAME", "claude-sonnet-4-20250514")
        self._api_base = os.environ.get("OPENAI_API_BASE", "http://localhost:11434/v1")
        

    def _switch_service(self, exclude: str = None) -> bool:
        for service_name, api_key in self._available_services.items():
            if service_name != exclude and service_name != self._service:
                self._service = service_name
                self._api_key = api_key
                return True
        return False

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="captcha", name="Captcha Solver", version="1.0.0",
            category="automation",
            description="Solve captchas using 2captcha or anti-captcha services",
            required_credentials=["CAPSOLVER_API_KEY"], install_cost=0,
            actions=[
                SkillAction(name="solve_recaptcha_v2", description="Solve reCAPTCHA v2/Enterprise",
                    parameters={"sitekey": {"type": "string", "required": True},
                                "url": {"type": "string", "required": True},
                                "invisible": {"type": "boolean", "required": False},
                                "enterprise": {"type": "boolean", "required": False},
                                "enterprise_payload": {"type": "object", "required": False}},
                    estimated_cost=0.003, estimated_duration_seconds=30, success_probability=0.95),
                SkillAction(name="solve_recaptcha_v3", description="Solve reCAPTCHA v3",
                    parameters={"sitekey": {"type": "string", "required": True},
                                "url": {"type": "string", "required": True},
                                "action": {"type": "string", "required": False},
                                "min_score": {"type": "number", "required": False}},
                    estimated_cost=0.003, estimated_duration_seconds=20, success_probability=0.9),
                SkillAction(name="solve_hcaptcha", description="Solve hCaptcha",
                    parameters={"sitekey": {"type": "string", "required": True},
                                "url": {"type": "string", "required": True}},
                    estimated_cost=0.003, estimated_duration_seconds=30, success_probability=0.95),
                SkillAction(name="solve_turnstile", description="Solve Cloudflare Turnstile",
                    parameters={"sitekey": {"type": "string", "required": True},
                                "url": {"type": "string", "required": True}},
                    estimated_cost=0.003, estimated_duration_seconds=20, success_probability=0.95),
                SkillAction(name="solve_image", description="Solve image captcha",
                    parameters={"image": {"type": "string", "required": True},
                                "case_sensitive": {"type": "boolean", "required": False},
                                "numeric": {"type": "boolean", "required": False}},
                    estimated_cost=0.001, estimated_duration_seconds=15, success_probability=0.9),
                SkillAction(name="solve_funcaptcha", description="Solve FunCaptcha/Arkose Labs",
                    parameters={"public_key": {"type": "string", "required": True},
                                "url": {"type": "string", "required": True},
                                "subdomain": {"type": "string", "required": False},
                                "blob": {"type": "string", "required": False}},
                    estimated_cost=0.002, estimated_duration_seconds=45, success_probability=0.85),
                SkillAction(name="extract_funcaptcha_params", description="Extract FunCaptcha params from page",
                    parameters={"page": {"type": "object", "required": True}},
                    estimated_cost=0, estimated_duration_seconds=2, success_probability=0.9),
                SkillAction(name="get_balance", description="Check account balance",
                    parameters={}, estimated_cost=0, estimated_duration_seconds=2, success_probability=0.99),
                SkillAction(name="get_enabled_sites", description="Get enabled sites from env",
                    parameters={}, estimated_cost=0, estimated_duration_seconds=1, success_probability=1.0),
                SkillAction(name="get_site_config", description="Get site configuration",
                    parameters={"site": {"type": "string", "required": True}},
                    estimated_cost=0, estimated_duration_seconds=1, success_probability=1.0),
                SkillAction(name="solve_for_site", description="Solve captcha for a configured site",
                    parameters={"site": {"type": "string", "required": True},
                                "sitekey": {"type": "string", "required": True}},
                    estimated_cost=0.003, estimated_duration_seconds=30, success_probability=0.9),
                SkillAction(name="solve_with_ai", description="Solve CAPTCHA using AI vision",
                    parameters={"page": {"type": "object", "required": True},
                                "max_attempts": {"type": "integer", "required": False}},
                    estimated_cost=0, estimated_duration_seconds=60, success_probability=0.7),
                SkillAction(name="detect_captcha", description="Detect CAPTCHA on page",
                    parameters={"page": {"type": "object", "required": True}},
                    estimated_cost=0, estimated_duration_seconds=2, success_probability=0.95)
            ])

    async def execute(self, action: str, params: Dict) -> SkillResult:
        if action == "get_enabled_sites":
            return self._get_enabled_sites_result()
        elif action == "get_site_config":
            return self._get_site_config_result(params.get("site", ""))
        if action in ("solve_with_ai", "detect_captcha"):
            from . import ai_vision
            page = params.get("page")
            if not page:
                return SkillResult(success=False, message="Playwright page object required")
            if action == "solve_with_ai":
                return await ai_vision.solve_with_ai(self, page, params.get("max_attempts", 3))
            return await ai_vision.detect_captcha(self, page)
        if not self._service or not self._api_key:
            return SkillResult(success=False,
                message="No captcha service configured. Set CAPSOLVER_API_KEY, TWOCAPTCHA_API_KEY, or ANTICAPTCHA_API_KEY")
        try:
            from . import solvers
            if action == "solve_recaptcha_v2":
                return await solvers.solve_recaptcha_v2(self, params.get("sitekey"),
                    params.get("url"), params.get("invisible", False),
                    params.get("enterprise", False), params.get("enterprise_payload"))
            elif action == "solve_recaptcha_v3":
                return await solvers.solve_recaptcha_v3(self, params.get("sitekey"),
                    params.get("url"), params.get("action", "verify"), params.get("min_score", 0.3))
            elif action == "solve_hcaptcha":
                return await solvers.solve_hcaptcha(self, params.get("sitekey"), params.get("url"))
            elif action == "solve_turnstile":
                return await solvers.solve_turnstile(self, params.get("sitekey"), params.get("url"))
            elif action == "solve_funcaptcha":
                return await solvers.solve_funcaptcha(self, params.get("public_key"),
                    params.get("url"), params.get("subdomain"), params.get("blob"))
            elif action == "extract_funcaptcha_params":
                page = params.get("page")
                if not page:
                    return SkillResult(success=False, message="Playwright page object required")
                fp = await solvers.extract_funcaptcha_params(page)
                return SkillResult(success=bool(fp.get("public_key")),
                    message="FunCaptcha params extracted" if fp.get("public_key") else "No FunCaptcha found",
                    data=fp)
            elif action == "solve_image":
                return await solvers.solve_image(self, params.get("image"),
                    params.get("case_sensitive", False), params.get("numeric", False))
            elif action == "get_balance":
                return await solvers.get_balance(self)
            elif action == "solve_for_site":
                return await solvers.solve_for_site(self, params.get("site"), params.get("sitekey"))
            else:
                return SkillResult(success=False, message=f"Unknown action: {action}")
        except Exception as e:
            return SkillResult(success=False, message=str(e))

    def _get_enabled_sites_result(self) -> SkillResult:
        sites = get_enabled_sites()
        details = []
        for site in sites:
            config = get_site_config(site)
            details.append({"domain": site, "name": config["name"] if config else site,
                            "captcha_type": config["captcha_type"] if config else "unknown",
                            "supported": config is not None})
        return SkillResult(success=True, message=f"Found {len(sites)} enabled sites",
                           data={"sites": details, "count": len(sites)})

    def _get_site_config_result(self, site: str) -> SkillResult:
        config = get_site_config(site)
        if not config:
            return SkillResult(success=False,
                message=f"No config for {site}. Supported: {', '.join(SITE_CONFIGS.keys())}")
        return SkillResult(success=True, message=f"Config for {config['name']}", data=config)

    async def close(self):
        await self.http.aclose()
