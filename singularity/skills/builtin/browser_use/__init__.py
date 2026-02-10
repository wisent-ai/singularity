"""Browser-Use Skill - AI-powered browser automation."""

import os
from typing import Dict
from singularity.skills.base import Skill, SkillResult, SkillManifest, SkillAction
from . import handlers

try:
    from browser_use import Agent as BrowserAgent, Browser, BrowserConfig
    from langchain_anthropic import ChatAnthropic
    HAS_BROWSER_USE = True
except ImportError:
    HAS_BROWSER_USE = False


def _a(n, d, p, cost=0.05, dur=30, prob=0.8):
    return SkillAction(name=n, description=d, parameters=p, estimated_cost=cost,
                       estimated_duration_seconds=dur, success_probability=prob)


class BrowserUseSkill(Skill):
    """AI-powered browser automation using browser-use."""

    def __init__(self, credentials: Dict[str, str] = None, headless: bool = True):
        super().__init__(credentials)
        self.headless = headless
        self._browser = None
        self._browser_agent = None

    @property
    def manifest(self) -> SkillManifest:
        _p = lambda n, t, r, d: {n: {"type": t, "required": r, "description": d}}
        return SkillManifest(
            skill_id="browser_use", name="AI Browser", version="1.0.0", category="automation",
            description="AI-powered browser that understands natural language",
            required_credentials=["ANTHROPIC_API_KEY"], install_cost=0,
            actions=[
                _a("run_task", "Run a browser task described in natural language",
                   {**_p("task", "string", True, "Natural language task description"),
                    **_p("max_steps", "integer", False, "Maximum steps (default 10)"),
                    **_p("start_url", "string", False, "Starting URL (optional)")}),
                _a("extract_data", "Extract structured data from a webpage",
                   {**_p("url", "string", True, "URL to extract from"),
                    **_p("data_description", "string", True, "What data to extract")}, 0.03, 20, 0.85),
                _a("fill_form", "Fill out a form on a webpage",
                   {**_p("url", "string", True, "URL with the form"),
                    **_p("form_data", "object", True, "Data to fill in (field name -> value)"),
                    **_p("submit", "boolean", False, "Whether to submit (default true)")}, 0.04, 25, 0.75),
                _a("screenshot", "Take a screenshot of a webpage",
                   {**_p("url", "string", True, "URL to screenshot"),
                    **_p("filename", "string", False, "Output filename")}, 0.01, 10, 0.95),
                _a("search_and_click", "Search for something and click on a result",
                   {**_p("search_engine", "string", False, "google, bing, duckduckgo"),
                    **_p("query", "string", True, "Search query"),
                    **_p("click_result", "string", False, "Which result to click")}),
                _a("monitor_page", "Monitor a page for changes or specific content",
                   {**_p("url", "string", True, "URL to monitor"),
                    **_p("condition", "string", True, "What to look for"),
                    **_p("max_wait_seconds", "integer", False, "How long to wait (default 60)")}, 0.1, 60, 0.7),
            ])

    def check_credentials(self) -> bool:
        if not HAS_BROWSER_USE:
            return False
        api_key = self.credentials.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
        return bool(api_key)

    async def _get_browser(self):
        if self._browser is None:
            config = BrowserConfig(headless=self.headless, disable_security=True)
            self._browser = Browser(config=config)
        return self._browser

    async def _get_llm(self):
        api_key = self.credentials.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
        return ChatAnthropic(model="claude-sonnet-4-20250514", api_key=api_key,  max_tokens=4096)

    async def _run_browser_task(self, task: str, max_steps: int = 10, start_url: str = None) -> Dict:
        browser = await self._get_browser()
        llm = await self._get_llm()
        full_task = f"First go to {start_url}. Then: {task}" if start_url else task
        agent = BrowserAgent(task=full_task, llm=llm, browser=browser, max_actions_per_step=3)
        result = await agent.run(max_steps=max_steps)
        return {"success": not result.is_done() or result.is_successful(),
                "history": [str(h) for h in result.history()[-5:]] if result.history() else [],
                "final_url": result.final_result() if hasattr(result, 'final_result') else None,
                "extracted_content": result.extracted_content() if hasattr(result, 'extracted_content') else None}

    async def execute(self, action: str, params: Dict) -> SkillResult:
        if not HAS_BROWSER_USE:
            return SkillResult(success=False, message="browser-use not installed. Run: pip install browser-use langchain-anthropic")
        if not self.check_credentials():
            return SkillResult(success=False, message="ANTHROPIC_API_KEY required for AI browser control")
        try:
            dispatch = {
                "run_task": lambda: handlers.run_task(self, params),
                "extract_data": lambda: handlers.extract_data(self, params),
                "fill_form": lambda: handlers.fill_form(self, params),
                "screenshot": lambda: handlers.screenshot(self, params),
                "search_and_click": lambda: handlers.search_and_click(self, params),
                "monitor_page": lambda: handlers.monitor_page(self, params),
            }
            handler = dispatch.get(action)
            if not handler:
                return SkillResult(success=False, message=f"Unknown action: {action}")
            return await handler()
        except Exception as e:
            return SkillResult(success=False, message=f"Browser-use error: {str(e)}")

    async def close(self):
        if self._browser:
            await self._browser.close()
            self._browser = None
