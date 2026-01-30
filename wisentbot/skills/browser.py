#!/usr/bin/env python3
"""
Browser Skill - Playwright-based web automation with stealth

Real browser control with anti-detection. No mocks.
"""

import random
from typing import Dict
from .base import Skill, SkillResult, SkillManifest, SkillAction

try:
    from playwright.async_api import async_playwright
    HAS_PLAYWRIGHT = True
except ImportError:
    HAS_PLAYWRIGHT = False

# Stealth scripts to inject
STEALTH_SCRIPTS = """
// Webdriver property
Object.defineProperty(navigator, 'webdriver', {get: () => undefined});

// Chrome property
window.chrome = {runtime: {}};

// Permissions
const originalQuery = window.navigator.permissions.query;
window.navigator.permissions.query = (parameters) => (
    parameters.name === 'notifications' ?
        Promise.resolve({state: Notification.permission}) :
        originalQuery(parameters)
);

// Plugins
Object.defineProperty(navigator, 'plugins', {
    get: () => [1, 2, 3, 4, 5]
});

// Languages
Object.defineProperty(navigator, 'languages', {
    get: () => ['en-US', 'en']
});

// Platform
Object.defineProperty(navigator, 'platform', {
    get: () => 'Win32'
});

// Hardware concurrency
Object.defineProperty(navigator, 'hardwareConcurrency', {
    get: () => 8
});

// WebGL vendor
const getParameter = WebGLRenderingContext.prototype.getParameter;
WebGLRenderingContext.prototype.getParameter = function(parameter) {
    if (parameter === 37445) return 'Intel Inc.';
    if (parameter === 37446) return 'Intel Iris OpenGL Engine';
    return getParameter.call(this, parameter);
};
"""

# Common user agents
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15",
]


class BrowserSkill(Skill):
    """
    Browser automation using Playwright with stealth mode.

    Can navigate, click, type, screenshot, scrape.
    Includes anti-detection measures.
    Supports residential proxies for avoiding detection.
    No API keys needed - just playwright installed.
    """

    def __init__(self, credentials: Dict[str, str] = None, stealth: bool = True, proxy: Dict = None):
        super().__init__(credentials)
        self.browser = None
        self.context = None
        self.page = None
        self.stealth = stealth
        self.proxy = proxy  # Playwright proxy config: {"server": "...", "username": "...", "password": "..."}

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="browser",
            name="Browser Automation",
            version="1.0.0",
            category="automation",
            description="Control a real browser - navigate, click, type, scrape",
            required_credentials=[],  # No credentials needed
            install_cost=0,
            actions=[
                SkillAction(
                    name="goto",
                    description="Navigate to a URL",
                    parameters={"url": "URL to navigate to"},
                    estimated_cost=0,
                    success_probability=0.95
                ),
                SkillAction(
                    name="click",
                    description="Click an element by selector",
                    parameters={"selector": "CSS selector to click"},
                    estimated_cost=0,
                    success_probability=0.8
                ),
                SkillAction(
                    name="type",
                    description="Type text into an input field",
                    parameters={"selector": "CSS selector", "text": "text to type"},
                    estimated_cost=0,
                    success_probability=0.85
                ),
                SkillAction(
                    name="get_text",
                    description="Get text content from an element",
                    parameters={"selector": "CSS selector"},
                    estimated_cost=0,
                    success_probability=0.9
                ),
                SkillAction(
                    name="screenshot",
                    description="Take a screenshot of the page",
                    parameters={"filename": "output filename"},
                    estimated_cost=0,
                    success_probability=0.95
                ),
                SkillAction(
                    name="get_page_content",
                    description="Get the full page HTML or text",
                    parameters={"format": "html or text"},
                    estimated_cost=0,
                    success_probability=0.95
                ),
                SkillAction(
                    name="fill_form",
                    description="Fill a form with multiple fields",
                    parameters={"fields": "dict of selector: value pairs"},
                    estimated_cost=0,
                    success_probability=0.8
                ),
                SkillAction(
                    name="wait_for",
                    description="Wait for an element to appear",
                    parameters={"selector": "CSS selector", "timeout": "timeout in ms"},
                    estimated_cost=0,
                    success_probability=0.85
                ),
                SkillAction(
                    name="evaluate",
                    description="Run JavaScript in the page",
                    parameters={"script": "JavaScript code to run"},
                    estimated_cost=0,
                    success_probability=0.9
                ),
                SkillAction(
                    name="keyboard_type",
                    description="Type text using keyboard (works with custom components)",
                    parameters={"text": "text to type", "delay": "delay between keystrokes in ms"},
                    estimated_cost=0,
                    success_probability=0.9
                ),
                SkillAction(
                    name="press",
                    description="Press a key (Enter, Tab, etc.)",
                    parameters={"key": "key to press"},
                    estimated_cost=0,
                    success_probability=0.95
                ),
            ]
        )

    def check_credentials(self) -> bool:
        """Browser skill just needs playwright installed"""
        return HAS_PLAYWRIGHT

    async def _ensure_browser(self):
        """Ensure browser is running with stealth configuration"""
        if not self.browser:
            self._playwright = await async_playwright().start()

            # Launch with stealth-friendly options
            launch_args = [
                '--disable-blink-features=AutomationControlled',
                '--disable-dev-shm-usage',
                '--no-sandbox',
                '--disable-setuid-sandbox',
                '--disable-infobars',
                '--window-size=1920,1080',
                '--start-maximized',
            ]

            # Add proxy args if using proxy at browser level
            launch_options = {
                'headless': True,
                'args': launch_args
            }

            self.browser = await self._playwright.chromium.launch(**launch_options)

            # Create context with realistic fingerprint
            user_agent = random.choice(USER_AGENTS) if self.stealth else None
            context_options = {
                'viewport': {'width': 1920, 'height': 1080},
                'locale': 'en-US',
                'timezone_id': 'America/New_York',
            }
            if user_agent:
                context_options['user_agent'] = user_agent

            # Add proxy to context if configured
            if self.proxy:
                context_options['proxy'] = self.proxy
                # Ignore SSL errors when using proxy (needed for Bright Data, etc.)
                context_options['ignore_https_errors'] = True

            self.context = await self.browser.new_context(**context_options)
            self.page = await self.context.new_page()

            # Inject stealth scripts before any navigation
            if self.stealth:
                await self.page.add_init_script(STEALTH_SCRIPTS)

    async def _close_browser(self):
        """Close browser"""
        if self.browser:
            await self.browser.close()
            await self._playwright.stop()
            self.browser = None
            self.context = None
            self.page = None

    async def execute(self, action: str, params: Dict) -> SkillResult:
        if not HAS_PLAYWRIGHT:
            return SkillResult(
                success=False,
                message="Playwright not installed. Run: pip install playwright && playwright install chromium"
            )

        try:
            await self._ensure_browser()

            if action == "goto":
                url = params.get("url", "")
                await self.page.goto(url, wait_until="domcontentloaded")
                title = await self.page.title()
                return SkillResult(
                    success=True,
                    message=f"Navigated to {url}",
                    data={"url": url, "title": title}
                )

            elif action == "click":
                selector = params.get("selector", "")
                await self.page.click(selector)
                return SkillResult(
                    success=True,
                    message=f"Clicked {selector}"
                )

            elif action == "type":
                selector = params.get("selector", "")
                text = params.get("text", "")
                await self.page.fill(selector, text)
                return SkillResult(
                    success=True,
                    message=f"Typed into {selector}"
                )

            elif action == "get_text":
                selector = params.get("selector", "")
                element = await self.page.query_selector(selector)
                if element:
                    text = await element.text_content()
                    return SkillResult(
                        success=True,
                        message="Got text",
                        data={"text": text}
                    )
                return SkillResult(success=False, message=f"Element not found: {selector}")

            elif action == "screenshot":
                filename = params.get("filename") or params.get("path", "screenshot.png")
                await self.page.screenshot(path=filename)
                return SkillResult(
                    success=True,
                    message=f"Screenshot saved to {filename}",
                    data={"path": filename}
                )

            elif action == "fill_placeholder":
                # Use get_by_placeholder for custom components
                placeholder = params.get("placeholder", "")
                text = params.get("text", "")
                locator = self.page.get_by_placeholder(placeholder)
                await locator.fill(text)
                return SkillResult(
                    success=True,
                    message=f"Filled field with placeholder '{placeholder}'"
                )

            elif action == "fill_label":
                # Use get_by_label for custom components
                label = params.get("label", "")
                text = params.get("text", "")
                locator = self.page.get_by_label(label)
                await locator.fill(text)
                return SkillResult(
                    success=True,
                    message=f"Filled field with label '{label}'"
                )

            elif action == "fill_role":
                # Use get_by_role for custom components
                role = params.get("role", "textbox")
                name = params.get("name", "")
                text = params.get("text", "")
                index = params.get("index")  # Optional: which element to fill (0-indexed)

                locator = self.page.get_by_role(role, name=name) if name else self.page.get_by_role(role)

                if index is not None:
                    locator = locator.nth(index)

                await locator.fill(text)
                return SkillResult(
                    success=True,
                    message=f"Filled {role} element" + (f" at index {index}" if index is not None else "")
                )

            elif action == "click_text":
                # Click element containing text
                text = params.get("text", "")
                locator = self.page.get_by_text(text)
                await locator.click()
                return SkillResult(
                    success=True,
                    message=f"Clicked element with text '{text}'"
                )

            elif action == "click_button":
                # Click button by text using get_by_role
                text = params.get("text", "")
                locator = self.page.get_by_role("button", name=text)
                await locator.click()
                return SkillResult(
                    success=True,
                    message=f"Clicked button '{text}'"
                )

            elif action == "get_page_content":
                fmt = params.get("format", "text")
                if fmt == "html":
                    content = await self.page.content()
                else:
                    content = await self.page.evaluate("document.body.innerText")
                return SkillResult(
                    success=True,
                    message="Got page content",
                    data={"content": content[:5000]}  # Limit size
                )

            elif action == "fill_form":
                fields = params.get("fields", {})
                for selector, value in fields.items():
                    await self.page.fill(selector, value)
                return SkillResult(
                    success=True,
                    message=f"Filled {len(fields)} fields"
                )

            elif action == "wait_for":
                selector = params.get("selector", "")
                timeout = int(params.get("timeout", 5000))
                await self.page.wait_for_selector(selector, timeout=timeout)
                return SkillResult(
                    success=True,
                    message=f"Element appeared: {selector}"
                )

            elif action == "evaluate":
                script = params.get("script", "")
                result = await self.page.evaluate(script)
                return SkillResult(
                    success=True,
                    message="Script executed",
                    data={"result": result}
                )

            elif action == "keyboard_type":
                text = params.get("text", "")
                delay = int(params.get("delay", 50))
                await self.page.keyboard.type(text, delay=delay)
                return SkillResult(
                    success=True,
                    message=f"Typed {len(text)} characters via keyboard"
                )

            elif action == "press":
                key = params.get("key", "Enter")
                await self.page.keyboard.press(key)
                return SkillResult(
                    success=True,
                    message=f"Pressed {key}"
                )

            else:
                return SkillResult(success=False, message=f"Unknown action: {action}")

        except Exception as e:
            return SkillResult(success=False, message=str(e))

    async def close(self):
        """Public method to close browser"""
        await self._close_browser()
