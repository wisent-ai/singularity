#!/usr/bin/env python3
"""
Tests for WebScraperSkill.

Uses importlib to load modules directly to avoid pulling in
the full singularity dependency tree (httpx, dotenv, etc.).
Uses mock HTML and a mock HTTP client to avoid network dependencies.
"""

import asyncio
import importlib.util
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

# ── Direct imports (avoid full package init) ──────────────────

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SKILLS = os.path.join(_ROOT, "singularity", "skills")


def _import_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# Import base first, then web_scraper
_base_mod = _import_module("singularity.skills.base", os.path.join(_SKILLS, "base.py"))
_ws_mod = _import_module(
    "singularity.skills.web_scraper",
    os.path.join(_SKILLS, "web_scraper.py"),
)

WebScraperSkill = _ws_mod.WebScraperSkill
_MiniParser = _ws_mod._MiniParser
_ensure_dirs = _ws_mod._ensure_dirs
_load_json = _ws_mod._load_json
_save_json = _ws_mod._save_json


def run_async(coro):
    """Helper to run async functions in sync tests."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ── Sample HTML fixtures ──────────────────────────────────────

SAMPLE_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <title>Test Page - Products</title>
    <meta name="description" content="A test page for web scraping">
    <meta property="og:title" content="Test Products">
    <meta property="og:type" content="website">
    <meta property="og:image" content="https://example.com/img.png">
    <link rel="canonical" href="https://example.com/products">
    <script type="application/ld+json">
    {"@type": "WebPage", "name": "Test Products"}
    </script>
</head>
<body>
    <h1>Product Listing</h1>
    <h1>Best Products 2024</h1>

    <div class="product-card">
        <h3 class="product-name">Widget A</h3>
        <span class="price">$9.99</span>
        <a href="/products/widget-a" class="product-link">View</a>
    </div>

    <div class="product-card">
        <h3 class="product-name">Widget B</h3>
        <span class="price">$19.99</span>
        <a href="/products/widget-b" class="product-link">View</a>
    </div>

    <div class="product-card">
        <h3 class="product-name">Widget C</h3>
        <span class="price">$29.99</span>
        <a href="/products/widget-c" class="product-link">Details</a>
    </div>

    <table id="specs-table">
        <thead>
            <tr><th>Feature</th><th>Value</th><th>Rating</th></tr>
        </thead>
        <tbody>
            <tr><td>Weight</td><td>100g</td><td>4.5</td></tr>
            <tr><td>Size</td><td>10cm</td><td>4.0</td></tr>
            <tr><td>Color</td><td>Blue</td><td>4.8</td></tr>
        </tbody>
    </table>

    <nav>
        <a href="/">Home</a>
        <a href="/products">Products</a>
        <a href="/about">About</a>
        <a href="/contact">Contact</a>
        <a href="https://external.com/partner">Partner</a>
    </nav>
</body>
</html>"""


SAMPLE_TABLE_ONLY = """<html><body>
<table>
    <tr><td>Name</td><td>Age</td></tr>
    <tr><td>Alice</td><td>30</td></tr>
    <tr><td>Bob</td><td>25</td></tr>
</table>
</body></html>"""


# ── MiniParser tests ──────────────────────────────────────────

class TestMiniParser(unittest.TestCase):
    """Test the lightweight regex-based HTML parser."""

    def test_select_all_by_tag(self):
        results = _MiniParser.select_all(SAMPLE_HTML, "h1")
        self.assertEqual(len(results), 2)
        self.assertIn("Product Listing", _MiniParser.get_text(results[0]))

    def test_select_all_by_class(self):
        results = _MiniParser.select_all(SAMPLE_HTML, "span.price")
        self.assertEqual(len(results), 3)
        texts = [_MiniParser.get_text(r) for r in results]
        self.assertIn("$9.99", texts)
        self.assertIn("$19.99", texts)
        self.assertIn("$29.99", texts)

    def test_select_all_by_id(self):
        results = _MiniParser.select_all(SAMPLE_HTML, "table#specs-table")
        self.assertEqual(len(results), 1)

    def test_get_text_strips_tags(self):
        fragment = '<span class="price"><b>$9.99</b></span>'
        text = _MiniParser.get_text(fragment)
        self.assertEqual(text, "$9.99")

    def test_get_text_decodes_entities(self):
        fragment = "Hello &amp; World &lt;3&gt;"
        text = _MiniParser.get_text(fragment)
        self.assertEqual(text, "Hello & World <3>")

    def test_get_attr_href(self):
        results = _MiniParser.get_attr(SAMPLE_HTML, "a.product-link", "href")
        self.assertEqual(len(results), 3)
        self.assertIn("/products/widget-a", results)
        self.assertIn("/products/widget-b", results)

    def test_parse_selector_tag_class(self):
        parts = _MiniParser._parse_selector("div.product-card")
        self.assertEqual(parts["tag"], "div")
        self.assertEqual(parts["class"], "product-card")
        self.assertIsNone(parts["id"])

    def test_parse_selector_id(self):
        parts = _MiniParser._parse_selector("table#specs-table")
        self.assertEqual(parts["tag"], "table")
        self.assertEqual(parts["id"], "specs-table")

    def test_parse_selector_class_only(self):
        parts = _MiniParser._parse_selector(".price")
        self.assertEqual(parts["class"], "price")


# ── WebScraperSkill tests ─────────────────────────────────────

class TestWebScraperSkill(unittest.TestCase):
    """Test the WebScraperSkill with mocked HTTP."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.skill = WebScraperSkill(credentials={})
        # Patch data directory to use temp
        self._orig_cache = _ws_mod.CACHE_FILE
        self._orig_monitors = _ws_mod.MONITORS_FILE
        self._orig_history = _ws_mod.HISTORY_FILE
        self._orig_dir = _ws_mod.DATA_DIR
        _ws_mod.DATA_DIR = Path(self.tmpdir)
        _ws_mod.CACHE_FILE = Path(self.tmpdir) / "cache.json"
        _ws_mod.MONITORS_FILE = Path(self.tmpdir) / "monitors.json"
        _ws_mod.HISTORY_FILE = Path(self.tmpdir) / "scrape_history.json"

    def tearDown(self):
        _ws_mod.DATA_DIR = self._orig_dir
        _ws_mod.CACHE_FILE = self._orig_cache
        _ws_mod.MONITORS_FILE = self._orig_monitors
        _ws_mod.HISTORY_FILE = self._orig_history
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _mock_fetch(self, html_content=SAMPLE_HTML):
        """Create a mock for the _fetch method."""
        async def mock_fetch(url, use_cache=True):
            return html_content, False
        return mock_fetch

    # ── Manifest ──

    def test_manifest(self):
        m = self.skill.manifest
        self.assertEqual(m.skill_id, "web_scraper")
        self.assertEqual(m.category, "data")
        self.assertEqual(m.version, "1.0.0")
        self.assertEqual(m.author, "Adam (ADAM)")
        self.assertEqual(len(m.required_credentials), 0)

    def test_manifest_actions(self):
        m = self.skill.manifest
        action_names = [a.name for a in m.actions]
        self.assertIn("scrape", action_names)
        self.assertIn("scrape_table", action_names)
        self.assertIn("scrape_list", action_names)
        self.assertIn("extract_links", action_names)
        self.assertIn("extract_metadata", action_names)
        self.assertIn("monitor_changes", action_names)
        self.assertIn("list_monitors", action_names)
        self.assertIn("scrape_history", action_names)
        self.assertEqual(len(action_names), 8)

    def test_check_credentials(self):
        self.assertTrue(self.skill.check_credentials())

    # ── scrape action ──

    def test_scrape_basic(self):
        self.skill._fetch = self._mock_fetch()
        result = run_async(self.skill.execute("scrape", {
            "url": "https://example.com/products",
            "selectors": {"title": "h1", "prices": "span.price"},
        }))
        self.assertTrue(result.success)
        self.assertIn("title", result.data["results"])
        self.assertIn("prices", result.data["results"])
        self.assertEqual(len(result.data["results"]["title"]), 2)
        self.assertEqual(len(result.data["results"]["prices"]), 3)

    def test_scrape_with_attr(self):
        self.skill._fetch = self._mock_fetch()
        result = run_async(self.skill.execute("scrape", {
            "url": "https://example.com/products",
            "selectors": {"links": "a.product-link|href"},
        }))
        self.assertTrue(result.success)
        links = result.data["results"]["links"]
        self.assertEqual(len(links), 3)
        # Should resolve relative URLs
        self.assertTrue(links[0].startswith("https://"))

    def test_scrape_missing_url(self):
        result = run_async(self.skill.execute("scrape", {
            "selectors": {"title": "h1"},
        }))
        self.assertFalse(result.success)
        self.assertIn("url", result.message.lower())

    def test_scrape_missing_selectors(self):
        self.skill._fetch = self._mock_fetch()
        result = run_async(self.skill.execute("scrape", {
            "url": "https://example.com",
        }))
        self.assertFalse(result.success)
        self.assertIn("selectors", result.message.lower())

    def test_scrape_with_limit(self):
        self.skill._fetch = self._mock_fetch()
        result = run_async(self.skill.execute("scrape", {
            "url": "https://example.com/products",
            "selectors": {"prices": "span.price"},
            "limit": 1,
        }))
        self.assertTrue(result.success)
        self.assertEqual(len(result.data["results"]["prices"]), 1)

    # ── scrape_table action ──

    def test_scrape_table(self):
        self.skill._fetch = self._mock_fetch()
        result = run_async(self.skill.execute("scrape_table", {
            "url": "https://example.com/products",
            "table_selector": "table#specs-table",
        }))
        self.assertTrue(result.success)
        self.assertEqual(result.data["row_count"], 3)
        self.assertIn("Feature", result.data["headers"])
        self.assertIn("Value", result.data["headers"])
        self.assertEqual(result.data["rows"][0]["Feature"], "Weight")

    def test_scrape_table_no_thead(self):
        """Test table extraction when there's no thead."""
        self.skill._fetch = self._mock_fetch(SAMPLE_TABLE_ONLY)
        result = run_async(self.skill.execute("scrape_table", {
            "url": "https://example.com/table",
        }))
        self.assertTrue(result.success)
        self.assertEqual(result.data["row_count"], 2)

    def test_scrape_table_not_found(self):
        self.skill._fetch = self._mock_fetch("<html><body>No table here</body></html>")
        result = run_async(self.skill.execute("scrape_table", {
            "url": "https://example.com",
        }))
        self.assertFalse(result.success)
        self.assertIn("No tables found", result.message)

    # ── scrape_list action ──

    def test_scrape_list(self):
        self.skill._fetch = self._mock_fetch()
        result = run_async(self.skill.execute("scrape_list", {
            "url": "https://example.com/products",
            "item_selector": "div.product-card",
            "fields": {
                "name": "h3.product-name",
                "price": "span.price",
                "url": "a.product-link|href",
            },
        }))
        self.assertTrue(result.success)
        self.assertEqual(result.data["item_count"], 3)
        items = result.data["items"]
        self.assertEqual(items[0]["name"], "Widget A")
        self.assertEqual(items[0]["price"], "$9.99")
        self.assertTrue(items[0]["url"].endswith("/products/widget-a"))

    def test_scrape_list_with_limit(self):
        self.skill._fetch = self._mock_fetch()
        result = run_async(self.skill.execute("scrape_list", {
            "url": "https://example.com/products",
            "item_selector": "div.product-card",
            "fields": {"name": "h3.product-name"},
            "limit": 2,
        }))
        self.assertTrue(result.success)
        self.assertEqual(result.data["item_count"], 2)

    def test_scrape_list_not_found(self):
        self.skill._fetch = self._mock_fetch()
        result = run_async(self.skill.execute("scrape_list", {
            "url": "https://example.com",
            "item_selector": "div.nonexistent",
            "fields": {"name": "h3"},
        }))
        self.assertFalse(result.success)

    def test_scrape_list_missing_item_selector(self):
        result = run_async(self.skill.execute("scrape_list", {
            "url": "https://example.com",
            "fields": {"name": "h3"},
        }))
        self.assertFalse(result.success)

    # ── extract_links action ──

    def test_extract_links(self):
        self.skill._fetch = self._mock_fetch()
        result = run_async(self.skill.execute("extract_links", {
            "url": "https://example.com/products",
        }))
        self.assertTrue(result.success)
        self.assertGreater(result.data["unique_links"], 0)
        urls = [l["url"] for l in result.data["links"]]
        self.assertTrue(any("/about" in u for u in urls))

    def test_extract_links_with_pattern(self):
        self.skill._fetch = self._mock_fetch()
        result = run_async(self.skill.execute("extract_links", {
            "url": "https://example.com/products",
            "pattern": "/products/",
        }))
        self.assertTrue(result.success)
        for link in result.data["links"]:
            self.assertIn("/products/", link["url"])

    def test_extract_links_deduplicates(self):
        html_content = '<html><body><a href="/a">A</a><a href="/a">A again</a><a href="/b">B</a></body></html>'
        self.skill._fetch = self._mock_fetch(html_content)
        result = run_async(self.skill.execute("extract_links", {
            "url": "https://example.com",
        }))
        self.assertTrue(result.success)
        self.assertEqual(result.data["unique_links"], 2)

    # ── extract_metadata action ──

    def test_extract_metadata(self):
        self.skill._fetch = self._mock_fetch()
        result = run_async(self.skill.execute("extract_metadata", {
            "url": "https://example.com/products",
        }))
        self.assertTrue(result.success)
        data = result.data
        self.assertEqual(data["title"], "Test Page - Products")
        self.assertEqual(data["description"], "A test page for web scraping")
        self.assertEqual(data["canonical_url"], "https://example.com/products")
        self.assertIn("open_graph", data)
        self.assertEqual(data["open_graph"]["title"], "Test Products")
        self.assertIn("json_ld", data)
        self.assertEqual(data["json_ld"][0]["@type"], "WebPage")
        self.assertEqual(data["language"], "en")
        self.assertIn("h1_headings", data)
        self.assertEqual(len(data["h1_headings"]), 2)

    def test_extract_metadata_minimal(self):
        """Test with minimal HTML that has few metadata fields."""
        self.skill._fetch = self._mock_fetch("<html><body><p>Hello</p></body></html>")
        result = run_async(self.skill.execute("extract_metadata", {
            "url": "https://example.com",
        }))
        self.assertTrue(result.success)
        self.assertEqual(result.data["url"], "https://example.com")

    # ── monitor_changes action ──

    def test_monitor_changes_first_check(self):
        self.skill._fetch = self._mock_fetch()
        result = run_async(self.skill.execute("monitor_changes", {
            "url": "https://example.com",
            "label": "Example",
        }))
        self.assertTrue(result.success)
        self.assertIn("baseline", result.data["summary"].lower())
        self.assertFalse(result.data["changed"])

    def test_monitor_changes_no_change(self):
        self.skill._fetch = self._mock_fetch()
        # First check
        run_async(self.skill.execute("monitor_changes", {
            "url": "https://example.com",
        }))
        # Second check - same content
        result = run_async(self.skill.execute("monitor_changes", {
            "url": "https://example.com",
        }))
        self.assertTrue(result.success)
        self.assertFalse(result.data["changed"])
        self.assertIn("No change", result.data["summary"])

    def test_monitor_changes_detected(self):
        self.skill._fetch = self._mock_fetch("<html><body>Version 1</body></html>")
        run_async(self.skill.execute("monitor_changes", {
            "url": "https://example.com",
        }))

        # Change content
        self.skill._fetch = self._mock_fetch("<html><body>Version 2</body></html>")
        result = run_async(self.skill.execute("monitor_changes", {
            "url": "https://example.com",
        }))
        self.assertTrue(result.success)
        self.assertTrue(result.data["changed"])
        self.assertIn("CHANGED", result.message)

    def test_monitor_changes_with_selector(self):
        self.skill._fetch = self._mock_fetch()
        result = run_async(self.skill.execute("monitor_changes", {
            "url": "https://example.com",
            "selector": "h1",
        }))
        self.assertTrue(result.success)
        self.assertIn("content_hash", result.data)

    # ── list_monitors action ──

    def test_list_monitors_empty(self):
        result = run_async(self.skill.execute("list_monitors", {}))
        self.assertTrue(result.success)
        self.assertEqual(result.data["count"], 0)

    def test_list_monitors_after_setup(self):
        self.skill._fetch = self._mock_fetch()
        run_async(self.skill.execute("monitor_changes", {
            "url": "https://example.com",
            "label": "Example Site",
        }))
        result = run_async(self.skill.execute("list_monitors", {}))
        self.assertTrue(result.success)
        self.assertEqual(result.data["count"], 1)
        self.assertEqual(result.data["monitors"][0]["label"], "Example Site")

    # ── scrape_history action ──

    def test_scrape_history_empty(self):
        result = run_async(self.skill.execute("scrape_history", {}))
        self.assertTrue(result.success)
        self.assertEqual(len(result.data["history"]), 0)

    def test_scrape_history_after_scrapes(self):
        self.skill._fetch = self._mock_fetch()
        # Do a few scrapes
        run_async(self.skill.execute("scrape", {
            "url": "https://example.com",
            "selectors": {"title": "h1"},
        }))
        run_async(self.skill.execute("extract_links", {
            "url": "https://example.com",
        }))

        result = run_async(self.skill.execute("scrape_history", {}))
        self.assertTrue(result.success)
        self.assertEqual(len(result.data["history"]), 2)
        self.assertEqual(result.data["stats"]["total_scrapes"], 2)
        self.assertEqual(result.data["stats"]["successful"], 2)

    def test_scrape_history_limit(self):
        self.skill._fetch = self._mock_fetch()
        for i in range(5):
            run_async(self.skill.execute("scrape", {
                "url": f"https://example.com/{i}",
                "selectors": {"title": "h1"},
            }))

        result = run_async(self.skill.execute("scrape_history", {"limit": 2}))
        self.assertTrue(result.success)
        self.assertEqual(len(result.data["history"]), 2)
        # Stats should still reflect all scrapes
        self.assertEqual(result.data["stats"]["total_scrapes"], 5)

    # ── Unknown action ──

    def test_unknown_action(self):
        result = run_async(self.skill.execute("unknown_action", {}))
        self.assertFalse(result.success)
        self.assertIn("Unknown action", result.message)

    # ── to_dict ──

    def test_to_dict(self):
        d = self.skill.to_dict()
        self.assertEqual(d["skill_id"], "web_scraper")
        self.assertEqual(d["category"], "data")
        self.assertIn("actions", d)
        self.assertEqual(len(d["actions"]), 8)


# ── Storage utility tests ─────────────────────────────────────

class TestStorageUtils(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self._orig_dir = _ws_mod.DATA_DIR
        _ws_mod.DATA_DIR = Path(self.tmpdir)

    def tearDown(self):
        _ws_mod.DATA_DIR = self._orig_dir
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_save_and_load_json(self):
        path = Path(self.tmpdir) / "test.json"
        _save_json(path, {"key": "value"})
        data = _load_json(path)
        self.assertEqual(data["key"], "value")

    def test_load_missing_file(self):
        path = Path(self.tmpdir) / "cache.json"
        data = _load_json(path)
        self.assertEqual(data, {})

    def test_load_missing_list_file(self):
        path = Path(self.tmpdir) / "scrape_history.json"
        data = _load_json(path)
        self.assertEqual(data, [])


if __name__ == "__main__":
    unittest.main()
