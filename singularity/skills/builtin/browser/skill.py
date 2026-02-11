"""
Browser Automation Skill — control a real browser.

Provides headless browser automation via Playwright. Navigate, click,
type, screenshot, scrape text, fill forms, and extract structured data.
Falls back to httpx for simple fetches when Playwright is not installed.
"""

import asyncio
import json
import re
from typing import Dict, List, Optional
from urllib.parse import urljoin, urlparse

from ...base import Skill, SkillResult, SkillAction, SkillManifest

# Check if playwright is available
try:
    from playwright.async_api import async_playwright, Browser, Page
    HAS_PLAYWRIGHT = True
except ImportError:
    HAS_PLAYWRIGHT = False

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False


def _a(name: str, desc: str, params: Optional[Dict] = None,
       prob: float = 0.85, dur: float = 15) -> SkillAction:
    return SkillAction(
        name=name, description=desc, parameters=params or {},
        estimated_cost=0, estimated_duration_seconds=dur,
        success_probability=prob,
    )


class BrowserSkill(Skill):
    """
    Browser automation skill using Playwright.

    Provides headless Chromium browser control for web scraping,
    form filling, navigation, screenshots, and data extraction.
    Falls back to httpx for simple HTTP requests.
    """

    def __init__(self, credentials: Dict[str, str] = None):
        super().__init__(credentials)
        self._playwright = None
        self._browser: Optional["Browser"] = None
        self._page: Optional["Page"] = None
        self._history: List[str] = []

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="browser",
            name="Browser Automation",
            version="1.0.0",
            category="automation",
            description=(
                "Control a real browser — navigate, click, type, scrape. "
                "Uses Playwright for full browser automation, with httpx fallback."
            ),
            required_credentials=[],
            install_cost=0,
            author="system",
            actions=[
                _a("navigate", "Navigate to a URL and return page content",
                   {"url": {"type": "string", "description": "URL to navigate to"},
                    "wait_for": {"type": "string", "description": "CSS selector to wait for (optional)"},
                    "timeout": {"type": "number", "description": "Timeout in seconds (default 30)"}},
                   prob=0.9, dur=10),
                _a("click", "Click an element on the page",
                   {"selector": {"type": "string", "description": "CSS selector of element to click"},
                    "text": {"type": "string", "description": "Click element containing this text (alternative to selector)"}},
                   prob=0.8, dur=5),
                _a("type", "Type text into an input field",
                   {"selector": {"type": "string", "description": "CSS selector of input field"},
                    "text": {"type": "string", "description": "Text to type"},
                    "clear": {"type": "boolean", "description": "Clear field first (default true)"}},
                   prob=0.85, dur=5),
                _a("screenshot", "Take a screenshot of the current page",
                   {"path": {"type": "string", "description": "File path to save (default: /tmp/screenshot.png)"},
                    "full_page": {"type": "boolean", "description": "Capture full scrollable page (default false)"}},
                   prob=0.9, dur=5),
                _a("scrape", "Extract text content from the current page",
                   {"selector": {"type": "string", "description": "CSS selector to scrape (optional, defaults to body)"},
                    "attribute": {"type": "string", "description": "Extract specific attribute instead of text (optional)"}},
                   prob=0.9, dur=5),
                _a("links", "Extract all links from the current page",
                   {"selector": {"type": "string", "description": "CSS selector scope (optional, defaults to body)"},
                    "pattern": {"type": "string", "description": "Regex filter for URLs (optional)"}},
                   prob=0.9, dur=5),
                _a("evaluate", "Execute JavaScript in the browser context",
                   {"script": {"type": "string", "description": "JavaScript code to evaluate"}},
                   prob=0.75, dur=10),
                _a("fill_form", "Fill a form with key-value pairs and optionally submit",
                   {"fields": {"type": "object", "description": "Dict of selector -> value pairs"},
                    "submit": {"type": "string", "description": "CSS selector of submit button (optional)"}},
                   prob=0.8, dur=10),
                _a("wait", "Wait for an element or condition",
                   {"selector": {"type": "string", "description": "CSS selector to wait for"},
                    "state": {"type": "string", "description": "State: visible, hidden, attached, detached (default visible)"},
                    "timeout": {"type": "number", "description": "Timeout in seconds (default 30)"}},
                   prob=0.8, dur=15),
                _a("history", "Get browsing history for this session",
                   {}, prob=1.0, dur=1),
                _a("close", "Close the browser session",
                   {}, prob=1.0, dur=2),
            ],
        )

    def check_credentials(self) -> bool:
        return True

    async def _ensure_browser(self) -> bool:
        """Launch browser if not already running."""
        if not HAS_PLAYWRIGHT:
            return False
        if self._browser and self._browser.is_connected():
            return True
        try:
            self._playwright = await async_playwright().start()
            self._browser = await self._playwright.chromium.launch(
                headless=True,
                args=["--no-sandbox", "--disable-dev-shm-usage"],
            )
            self._page = await self._browser.new_page()
            await self._page.set_extra_http_headers({
                "User-Agent": (
                    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                ),
            })
            return True
        except Exception:
            return False

    async def _close_browser(self):
        """Close browser and cleanup."""
        if self._browser:
            try:
                await self._browser.close()
            except Exception:
                pass
            self._browser = None
            self._page = None
        if self._playwright:
            try:
                await self._playwright.stop()
            except Exception:
                pass
            self._playwright = None

    async def execute(self, action: str, params: Dict) -> SkillResult:
        try:
            dispatch = {
                "navigate": self._navigate,
                "click": self._click,
                "type": self._type,
                "screenshot": self._screenshot,
                "scrape": self._scrape,
                "links": self._links,
                "evaluate": self._evaluate,
                "fill_form": self._fill_form,
                "wait": self._wait,
                "history": self._get_history,
                "close": self._close,
            }
            handler = dispatch.get(action)
            if not handler:
                return SkillResult(
                    success=False,
                    message=f"Unknown action: {action}. Available: {', '.join(dispatch.keys())}",
                )
            return await handler(params)
        except Exception as e:
            return SkillResult(success=False, message=f"Browser error: {e}")

    # ── Actions ──────────────────────────────────────────────────────

    async def _navigate(self, params: Dict) -> SkillResult:
        """Navigate to a URL."""
        url = params.get("url", "").strip()
        if not url:
            return SkillResult(success=False, message="'url' parameter is required.")
        if not url.startswith(("http://", "https://")):
            url = f"https://{url}"

        timeout = int(params.get("timeout", 30)) * 1000
        wait_for = params.get("wait_for", "")

        # Try Playwright first
        if await self._ensure_browser():
            try:
                response = await self._page.goto(url, timeout=timeout, wait_until="domcontentloaded")
                if wait_for:
                    await self._page.wait_for_selector(wait_for, timeout=timeout)

                self._history.append(url)
                title = await self._page.title()
                text = await self._page.inner_text("body")
                text = text[:5000] if len(text) > 5000 else text
                status = response.status if response else 0

                return SkillResult(
                    success=True,
                    message=f"Navigated to {url} — \"{title}\" (status {status})",
                    data={
                        "url": self._page.url,
                        "title": title,
                        "status": status,
                        "text": text,
                    },
                )
            except Exception as e:
                return SkillResult(success=False, message=f"Navigation failed: {e}")

        # Fallback to httpx
        if HAS_HTTPX:
            try:
                async with httpx.AsyncClient(follow_redirects=True, timeout=30) as client:
                    resp = await client.get(url)
                    content = resp.text[:5000]
                    # Extract title from HTML
                    title_match = re.search(r"<title>(.*?)</title>", content, re.IGNORECASE | re.DOTALL)
                    title = title_match.group(1).strip() if title_match else ""
                    self._history.append(url)
                    return SkillResult(
                        success=True,
                        message=f"Fetched {url} — \"{title}\" (status {resp.status_code}, httpx fallback)",
                        data={"url": str(resp.url), "title": title,
                              "status": resp.status_code, "text": content},
                    )
            except Exception as e:
                return SkillResult(success=False, message=f"Fetch failed: {e}")

        return SkillResult(
            success=False,
            message="Neither playwright nor httpx available. Install: pip install playwright httpx",
        )

    async def _click(self, params: Dict) -> SkillResult:
        """Click an element on the page."""
        if not self._page:
            return SkillResult(success=False, message="No page open. Navigate first.")

        selector = params.get("selector", "").strip()
        text = params.get("text", "").strip()

        try:
            if text:
                locator = self._page.get_by_text(text, exact=False).first
                await locator.click(timeout=10000)
                return SkillResult(success=True, message=f"Clicked element containing text: '{text}'")
            elif selector:
                await self._page.click(selector, timeout=10000)
                return SkillResult(success=True, message=f"Clicked: {selector}")
            else:
                return SkillResult(success=False, message="Provide 'selector' or 'text' parameter.")
        except Exception as e:
            return SkillResult(success=False, message=f"Click failed: {e}")

    async def _type(self, params: Dict) -> SkillResult:
        """Type text into a field."""
        if not self._page:
            return SkillResult(success=False, message="No page open. Navigate first.")

        selector = params.get("selector", "").strip()
        text = params.get("text", "")
        clear = params.get("clear", True)

        if not selector:
            return SkillResult(success=False, message="'selector' parameter is required.")

        try:
            if clear:
                await self._page.fill(selector, text, timeout=10000)
            else:
                await self._page.type(selector, text, timeout=10000)
            return SkillResult(
                success=True,
                message=f"Typed {len(text)} chars into {selector}",
            )
        except Exception as e:
            return SkillResult(success=False, message=f"Type failed: {e}")

    async def _screenshot(self, params: Dict) -> SkillResult:
        """Take a screenshot."""
        if not self._page:
            return SkillResult(success=False, message="No page open. Navigate first.")

        path = params.get("path", "/tmp/screenshot.png")
        full_page = params.get("full_page", False)

        try:
            await self._page.screenshot(path=path, full_page=full_page)
            return SkillResult(
                success=True,
                message=f"Screenshot saved to {path}",
                data={"path": path, "full_page": full_page},
                asset_created={"type": "file", "path": path},
            )
        except Exception as e:
            return SkillResult(success=False, message=f"Screenshot failed: {e}")

    async def _scrape(self, params: Dict) -> SkillResult:
        """Extract text from the page."""
        if not self._page:
            return SkillResult(success=False, message="No page open. Navigate first.")

        selector = params.get("selector", "body")
        attribute = params.get("attribute", "").strip()

        try:
            if attribute:
                elements = await self._page.query_selector_all(selector)
                values = []
                for el in elements[:100]:
                    val = await el.get_attribute(attribute)
                    if val:
                        values.append(val)
                return SkillResult(
                    success=True,
                    message=f"Extracted {len(values)} '{attribute}' values from '{selector}'",
                    data={"selector": selector, "attribute": attribute, "values": values},
                )
            else:
                text = await self._page.inner_text(selector)
                text = text[:10000] if len(text) > 10000 else text
                return SkillResult(
                    success=True,
                    message=f"Scraped {len(text)} chars from '{selector}'",
                    data={"selector": selector, "text": text},
                )
        except Exception as e:
            return SkillResult(success=False, message=f"Scrape failed: {e}")

    async def _links(self, params: Dict) -> SkillResult:
        """Extract all links from the page."""
        if not self._page:
            return SkillResult(success=False, message="No page open. Navigate first.")

        selector = params.get("selector", "body")
        pattern = params.get("pattern", "")

        try:
            elements = await self._page.query_selector_all(f"{selector} a[href]")
            links = []
            base_url = self._page.url

            for el in elements[:500]:
                href = await el.get_attribute("href")
                text = (await el.inner_text()).strip()[:100]
                if href:
                    absolute = urljoin(base_url, href)
                    if pattern and not re.search(pattern, absolute):
                        continue
                    links.append({"url": absolute, "text": text})

            return SkillResult(
                success=True,
                message=f"Found {len(links)} links on page",
                data={"links": links, "count": len(links)},
            )
        except Exception as e:
            return SkillResult(success=False, message=f"Link extraction failed: {e}")

    async def _evaluate(self, params: Dict) -> SkillResult:
        """Execute JavaScript in the browser."""
        if not self._page:
            return SkillResult(success=False, message="No page open. Navigate first.")

        script = params.get("script", "").strip()
        if not script:
            return SkillResult(success=False, message="'script' parameter is required.")

        try:
            result = await self._page.evaluate(script)
            result_str = json.dumps(result, default=str) if result is not None else "null"
            if len(result_str) > 5000:
                result_str = result_str[:5000] + "..."
            return SkillResult(
                success=True,
                message=f"JS result: {result_str[:200]}",
                data={"result": result},
            )
        except Exception as e:
            return SkillResult(success=False, message=f"JS execution failed: {e}")

    async def _fill_form(self, params: Dict) -> SkillResult:
        """Fill a form with multiple fields."""
        if not self._page:
            return SkillResult(success=False, message="No page open. Navigate first.")

        fields = params.get("fields", {})
        submit_selector = params.get("submit", "").strip()

        if not fields:
            return SkillResult(success=False, message="'fields' parameter is required (dict of selector -> value).")

        filled = 0
        errors = []
        for selector, value in fields.items():
            try:
                await self._page.fill(selector, str(value), timeout=5000)
                filled += 1
            except Exception as e:
                errors.append(f"{selector}: {e}")

        if submit_selector:
            try:
                await self._page.click(submit_selector, timeout=10000)
            except Exception as e:
                errors.append(f"Submit ({submit_selector}): {e}")

        success = filled > 0 and len(errors) == 0
        msg = f"Filled {filled}/{len(fields)} fields"
        if submit_selector:
            msg += ", submitted"
        if errors:
            msg += f" ({len(errors)} errors: {'; '.join(errors[:3])})"

        return SkillResult(success=success, message=msg,
                          data={"filled": filled, "total": len(fields), "errors": errors})

    async def _wait(self, params: Dict) -> SkillResult:
        """Wait for an element or condition."""
        if not self._page:
            return SkillResult(success=False, message="No page open. Navigate first.")

        selector = params.get("selector", "").strip()
        if not selector:
            return SkillResult(success=False, message="'selector' parameter is required.")

        state = params.get("state", "visible")
        timeout = int(params.get("timeout", 30)) * 1000

        try:
            await self._page.wait_for_selector(selector, state=state, timeout=timeout)
            return SkillResult(
                success=True,
                message=f"Element '{selector}' is now {state}",
            )
        except Exception as e:
            return SkillResult(success=False, message=f"Wait timed out: {e}")

    async def _get_history(self, params: Dict) -> SkillResult:
        """Return browsing history."""
        return SkillResult(
            success=True,
            message=f"{len(self._history)} URLs visited",
            data={"history": self._history, "count": len(self._history),
                  "current": self._page.url if self._page else None},
        )

    async def _close(self, params: Dict) -> SkillResult:
        """Close the browser session."""
        await self._close_browser()
        self._history.clear()
        return SkillResult(success=True, message="Browser session closed")
