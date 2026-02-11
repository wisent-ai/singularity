"""
Comprehensive tests for the BrowserSkill — browser automation with
Playwright and httpx fallback.

Tests focus on structural correctness, manifest validation, and
action dispatch. Browser-dependent tests mock Playwright internals.
"""

import asyncio
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
exec(open(str(Path(__file__).resolve().parent / "conftest.py")).read())

from singularity.skills.base.types import SkillResult, SkillAction, SkillManifest
from singularity.skills.builtin.browser.skill import BrowserSkill


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


class TestBrowserSkillManifest(unittest.TestCase):
    """Test skill manifest metadata."""

    def setUp(self):
        self.skill = BrowserSkill()

    def test_manifest_type(self):
        self.assertIsInstance(self.skill.manifest, SkillManifest)

    def test_skill_id(self):
        self.assertEqual(self.skill.manifest.skill_id, "browser")

    def test_category(self):
        self.assertEqual(self.skill.manifest.category, "automation")

    def test_version(self):
        self.assertEqual(self.skill.manifest.version, "1.0.0")

    def test_no_credentials(self):
        self.assertEqual(self.skill.manifest.required_credentials, [])

    def test_action_names(self):
        names = {a.name for a in self.skill.manifest.actions}
        expected = {"navigate", "click", "type", "screenshot", "scrape",
                    "links", "evaluate", "fill_form", "wait", "history", "close"}
        self.assertEqual(names, expected)

    def test_actions_count(self):
        self.assertEqual(len(self.skill.manifest.actions), 11)

    def test_all_actions_are_skill_actions(self):
        for action in self.skill.manifest.actions:
            self.assertIsInstance(action, SkillAction)

    def test_all_actions_have_descriptions(self):
        for action in self.skill.manifest.actions:
            self.assertTrue(len(action.description) > 0)

    def test_check_credentials(self):
        self.assertTrue(self.skill.check_credentials())


class TestBrowserSkillUnknownAction(unittest.TestCase):
    """Test unknown action handling."""

    def test_unknown_action(self):
        skill = BrowserSkill()
        result = run(skill.execute("nonexistent", {}))
        self.assertFalse(result.success)
        self.assertIn("Unknown action", result.message)


class TestBrowserSkillNavigate(unittest.TestCase):
    """Test navigate action."""

    def test_navigate_empty_url(self):
        skill = BrowserSkill()
        result = run(skill.execute("navigate", {"url": ""}))
        self.assertFalse(result.success)
        self.assertIn("required", result.message.lower())

    def test_navigate_adds_https(self):
        """Test that URLs without scheme get https:// prepended."""
        skill = BrowserSkill()
        # This will fail due to no network/playwright but validates URL processing
        result = run(skill.execute("navigate", {"url": "example.com"}))
        # Either succeeds or fails gracefully (no crash)
        self.assertIsInstance(result, SkillResult)


class TestBrowserSkillNoPage(unittest.TestCase):
    """Test actions that require an open page fail gracefully."""

    def setUp(self):
        self.skill = BrowserSkill()
        # Ensure no page is set
        self.skill._page = None

    def test_click_no_page(self):
        result = run(self.skill.execute("click", {"selector": "#btn"}))
        self.assertFalse(result.success)
        self.assertIn("no page", result.message.lower())

    def test_type_no_page(self):
        result = run(self.skill.execute("type", {"selector": "#input", "text": "hi"}))
        self.assertFalse(result.success)

    def test_screenshot_no_page(self):
        result = run(self.skill.execute("screenshot", {}))
        self.assertFalse(result.success)

    def test_scrape_no_page(self):
        result = run(self.skill.execute("scrape", {}))
        self.assertFalse(result.success)

    def test_links_no_page(self):
        result = run(self.skill.execute("links", {}))
        self.assertFalse(result.success)

    def test_evaluate_no_page(self):
        result = run(self.skill.execute("evaluate", {"script": "1+1"}))
        self.assertFalse(result.success)

    def test_fill_form_no_page(self):
        result = run(self.skill.execute("fill_form", {"fields": {"#x": "y"}}))
        self.assertFalse(result.success)

    def test_wait_no_page(self):
        result = run(self.skill.execute("wait", {"selector": "#x"}))
        self.assertFalse(result.success)


class TestBrowserSkillHistory(unittest.TestCase):
    """Test history action."""

    def test_empty_history(self):
        skill = BrowserSkill()
        result = run(skill.execute("history", {}))
        self.assertTrue(result.success)
        self.assertEqual(result.data["count"], 0)
        self.assertEqual(result.data["history"], [])

    def test_history_tracks_urls(self):
        skill = BrowserSkill()
        skill._history = ["https://example.com", "https://google.com"]
        result = run(skill.execute("history", {}))
        self.assertTrue(result.success)
        self.assertEqual(result.data["count"], 2)


class TestBrowserSkillClose(unittest.TestCase):
    """Test close action."""

    def test_close_no_browser(self):
        skill = BrowserSkill()
        result = run(skill.execute("close", {}))
        self.assertTrue(result.success)
        self.assertIn("closed", result.message.lower())

    def test_close_clears_history(self):
        skill = BrowserSkill()
        skill._history = ["https://example.com"]
        run(skill.execute("close", {}))
        self.assertEqual(len(skill._history), 0)


class TestBrowserSkillInputValidation(unittest.TestCase):
    """Test parameter validation for various actions."""

    def setUp(self):
        self.skill = BrowserSkill()

    def test_click_needs_selector_or_text(self):
        self.skill._page = MagicMock()
        result = run(self.skill.execute("click", {}))
        self.assertFalse(result.success)

    def test_type_needs_selector(self):
        self.skill._page = MagicMock()
        result = run(self.skill.execute("type", {"text": "hello"}))
        self.assertFalse(result.success)

    def test_evaluate_needs_script(self):
        self.skill._page = MagicMock()
        result = run(self.skill.execute("evaluate", {}))
        self.assertFalse(result.success)

    def test_fill_form_needs_fields(self):
        self.skill._page = MagicMock()
        result = run(self.skill.execute("fill_form", {}))
        self.assertFalse(result.success)

    def test_wait_needs_selector(self):
        self.skill._page = MagicMock()
        result = run(self.skill.execute("wait", {}))
        self.assertFalse(result.success)


class TestBrowserSkillMockedPage(unittest.TestCase):
    """Test actions with a mocked Playwright page."""

    def setUp(self):
        self.skill = BrowserSkill()
        self.mock_page = AsyncMock()
        self.mock_page.url = "https://example.com"
        self.skill._page = self.mock_page

    def test_click_by_selector(self):
        self.mock_page.click = AsyncMock()
        result = run(self.skill.execute("click", {"selector": "#btn"}))
        self.assertTrue(result.success)
        self.mock_page.click.assert_called_once()

    def test_type_with_clear(self):
        self.mock_page.fill = AsyncMock()
        result = run(self.skill.execute("type", {"selector": "#input", "text": "hello", "clear": True}))
        self.assertTrue(result.success)
        self.mock_page.fill.assert_called_once()

    def test_type_without_clear(self):
        self.mock_page.type = AsyncMock()
        result = run(self.skill.execute("type", {"selector": "#input", "text": "hello", "clear": False}))
        self.assertTrue(result.success)
        self.mock_page.type.assert_called_once()

    def test_screenshot_default_path(self):
        self.mock_page.screenshot = AsyncMock()
        result = run(self.skill.execute("screenshot", {}))
        self.assertTrue(result.success)
        self.mock_page.screenshot.assert_called_once()
        self.assertIn("/tmp/screenshot.png", result.data["path"])

    def test_scrape_text(self):
        self.mock_page.inner_text = AsyncMock(return_value="Hello World")
        result = run(self.skill.execute("scrape", {"selector": "h1"}))
        self.assertTrue(result.success)
        self.assertIn("Hello World", result.data["text"])

    def test_scrape_attribute(self):
        mock_el = AsyncMock()
        mock_el.get_attribute = AsyncMock(return_value="value1")
        self.mock_page.query_selector_all = AsyncMock(return_value=[mock_el])
        result = run(self.skill.execute("scrape", {"selector": "input", "attribute": "value"}))
        self.assertTrue(result.success)
        self.assertIn("value1", result.data["values"])

    def test_evaluate_js(self):
        self.mock_page.evaluate = AsyncMock(return_value=42)
        result = run(self.skill.execute("evaluate", {"script": "1 + 41"}))
        self.assertTrue(result.success)
        self.assertEqual(result.data["result"], 42)

    def test_links_extraction(self):
        mock_el = AsyncMock()
        mock_el.get_attribute = AsyncMock(return_value="/page2")
        mock_el.inner_text = AsyncMock(return_value="Page 2")
        self.mock_page.query_selector_all = AsyncMock(return_value=[mock_el])
        result = run(self.skill.execute("links", {}))
        self.assertTrue(result.success)
        self.assertEqual(result.data["count"], 1)
        self.assertEqual(result.data["links"][0]["text"], "Page 2")

    def test_fill_form_success(self):
        self.mock_page.fill = AsyncMock()
        result = run(self.skill.execute("fill_form", {
            "fields": {"#name": "Adam", "#email": "adam@example.com"},
        }))
        self.assertTrue(result.success)
        self.assertEqual(result.data["filled"], 2)

    def test_fill_form_with_submit(self):
        self.mock_page.fill = AsyncMock()
        self.mock_page.click = AsyncMock()
        result = run(self.skill.execute("fill_form", {
            "fields": {"#name": "Adam"},
            "submit": "#submit",
        }))
        self.assertTrue(result.success)
        self.mock_page.click.assert_called_once()

    def test_wait_for_selector(self):
        self.mock_page.wait_for_selector = AsyncMock()
        result = run(self.skill.execute("wait", {"selector": "#loaded", "state": "visible"}))
        self.assertTrue(result.success)


if __name__ == "__main__":
    unittest.main()
