#!/usr/bin/env python3
"""
WebScraperSkill - Structured web data extraction.

Extracts structured data from web pages using CSS selectors, URL patterns,
and content analysis. Unlike the general-purpose BrowserSkill (Playwright),
this skill is purpose-built for data extraction: scraping product listings,
extracting pricing data, monitoring page changes, and aggregating content.

Zero external dependencies beyond Python stdlib and httpx (already in project).
Falls back gracefully when optional dependencies (beautifulsoup4) are unavailable.

Author: Adam (ADAM) - autonomous AI agent
"""

import hashlib
import html
import json
import re
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse

try:
    import httpx

    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False

try:
    from bs4 import BeautifulSoup, Tag

    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False

from .base import Skill, SkillAction, SkillManifest, SkillResult

# --- Storage ---
DATA_DIR = Path(__file__).parent.parent / "data" / "web_scraper"
CACHE_FILE = DATA_DIR / "cache.json"
MONITORS_FILE = DATA_DIR / "monitors.json"
HISTORY_FILE = DATA_DIR / "scrape_history.json"

# --- Defaults ---
DEFAULT_TIMEOUT = 15  # seconds
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (compatible; SingularityBot/1.0; +https://github.com/wisent-ai/singularity)"
)
MAX_CACHE_AGE = 3600  # 1 hour
MAX_HISTORY_ENTRIES = 500
MAX_RESPONSE_SIZE = 5_000_000  # 5MB


def _ensure_dirs():
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def _load_json(path: Path) -> Any:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {} if "cache" in str(path) or "monitors" in str(path) else []


def _save_json(path: Path, data: Any):
    _ensure_dirs()
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


# ────────────────────────────────────────────────────────────────
# Lightweight HTML parser (no BS4 dependency)
# ────────────────────────────────────────────────────────────────

class _MiniParser:
    """Minimal CSS-selector-based HTML extractor using regex.

    Supports: tag, .class, #id, tag.class, tag#id, and simple
    attribute selectors like [attr=value]. Good enough for 80% of
    scraping needs without pulling in BeautifulSoup.
    """

    @staticmethod
    def _parse_selector(sel: str) -> Dict[str, Optional[str]]:
        """Parse a simple CSS selector into components."""
        result: Dict[str, Optional[str]] = {"tag": None, "id": None, "class": None, "attr": None, "attr_val": None}

        # Handle attribute selector [attr=value]
        attr_match = re.search(r'\[(\w+)=["\']?([^"\'\]]+)["\']?\]', sel)
        if attr_match:
            result["attr"] = attr_match.group(1)
            result["attr_val"] = attr_match.group(2)
            sel = sel[:attr_match.start()]

        # Handle #id
        id_match = re.search(r'#([\w-]+)', sel)
        if id_match:
            result["id"] = id_match.group(1)
            sel = sel[:id_match.start()] + sel[id_match.end():]

        # Handle .class
        class_match = re.search(r'\.([\w-]+)', sel)
        if class_match:
            result["class"] = class_match.group(1)
            sel = sel[:class_match.start()] + sel[class_match.end():]

        # Remaining is tag name
        tag = sel.strip()
        if tag:
            result["tag"] = tag

        return result

    @staticmethod
    def select_all(html_text: str, selector: str) -> List[str]:
        """Extract all elements matching a simple CSS selector."""
        parts = _MiniParser._parse_selector(selector)
        tag = parts["tag"] or r"[\w]+"

        # Build pattern
        attr_patterns = []
        if parts["id"]:
            attr_patterns.append(rf'id=["\']?{re.escape(parts["id"])}["\']?')
        if parts["class"]:
            attr_patterns.append(rf'class=["\'][^"\']*\b{re.escape(parts["class"])}\b[^"\']*["\']')
        if parts["attr"] and parts["attr_val"]:
            attr_patterns.append(rf'{re.escape(parts["attr"])}=["\']?{re.escape(parts["attr_val"])}["\']?')

        if attr_patterns:
            attrs_re = r'[^>]*?' + r'[^>]*?'.join(attr_patterns) + r'[^>]*?'
        else:
            attrs_re = r'[^>]*?'

        # Match opening tag through closing tag
        pattern = rf'<({tag}){attrs_re}>(.*?)</\1>'
        matches = re.findall(pattern, html_text, re.DOTALL | re.IGNORECASE)
        return [m[1] for m in matches]

    @staticmethod
    def get_text(html_fragment: str) -> str:
        """Strip HTML tags and decode entities."""
        text = re.sub(r'<[^>]+>', ' ', html_fragment)
        text = html.unescape(text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    @staticmethod
    def get_attr(html_text: str, selector: str, attr: str) -> List[str]:
        """Extract attribute values from matching elements."""
        parts = _MiniParser._parse_selector(selector)
        tag = parts["tag"] or r"[\w]+"

        # Build a pattern to match the opening tag
        attr_patterns = []
        if parts["id"]:
            attr_patterns.append(rf'id=["\']?{re.escape(parts["id"])}["\']?')
        if parts["class"]:
            attr_patterns.append(rf'class=["\'][^"\']*\b{re.escape(parts["class"])}\b[^"\']*["\']')

        if attr_patterns:
            attrs_re = r'[^>]*?' + r'[^>]*?'.join(attr_patterns) + r'[^>]*?'
        else:
            attrs_re = r'[^>]*?'

        # Find opening tags matching the selector
        tag_pattern = rf'<{tag}{attrs_re}>'
        opening_tags = re.findall(tag_pattern, html_text, re.DOTALL | re.IGNORECASE)

        # Extract the requested attribute from each matched tag
        results = []
        for tag_str in opening_tags:
            attr_match = re.search(rf'{re.escape(attr)}=["\']([^"\']*)["\']', tag_str)
            if attr_match:
                results.append(attr_match.group(1))
        return results


# ────────────────────────────────────────────────────────────────
# BeautifulSoup wrapper (when available)
# ────────────────────────────────────────────────────────────────

class _SoupParser:
    """Parser using BeautifulSoup for better accuracy."""

    @staticmethod
    def select_all(html_text: str, selector: str) -> List[str]:
        soup = BeautifulSoup(html_text, "html.parser")
        elements = soup.select(selector)
        return [str(el) for el in elements]

    @staticmethod
    def get_text(html_fragment: str) -> str:
        soup = BeautifulSoup(html_fragment, "html.parser")
        return soup.get_text(separator=" ", strip=True)

    @staticmethod
    def get_attr(html_text: str, selector: str, attr: str) -> List[str]:
        soup = BeautifulSoup(html_text, "html.parser")
        elements = soup.select(selector)
        return [el.get(attr, "") for el in elements if el.get(attr)]


def _get_parser():
    """Return the best available parser."""
    if HAS_BS4:
        return _SoupParser
    return _MiniParser


# ────────────────────────────────────────────────────────────────
# Main Skill
# ────────────────────────────────────────────────────────────────

class WebScraperSkill(Skill):
    """
    Structured web data extraction skill.

    Extracts structured data from web pages using CSS selectors.
    Supports caching, change monitoring, and multi-page scraping.
    Works with or without BeautifulSoup (graceful fallback to regex).
    """

    def __init__(self, credentials: Dict[str, str] = None):
        super().__init__(credentials)
        _ensure_dirs()

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="web_scraper",
            name="Web Scraper",
            version="1.0.0",
            category="data",
            description=(
                "Extract structured data from web pages using CSS selectors. "
                "Scrape listings, extract tables, monitor page changes, and "
                "aggregate content from multiple URLs."
            ),
            required_credentials=[],
            install_cost=0,
            author="Adam (ADAM)",
            actions=[
                SkillAction(
                    name="scrape",
                    description=(
                        "Fetch a URL and extract data using CSS selectors. "
                        "Returns text content or attribute values for each matching element."
                    ),
                    parameters={
                        "url": {"type": "string", "required": True, "description": "URL to scrape"},
                        "selectors": {
                            "type": "object",
                            "required": True,
                            "description": (
                                "Map of field names to CSS selectors. "
                                'E.g. {"title": "h1", "prices": ".price", "links": "a.product|href"} '
                                "Append |attr to extract an attribute instead of text."
                            ),
                        },
                        "limit": {
                            "type": "integer",
                            "required": False,
                            "description": "Max items to return per selector (default: 100)",
                        },
                        "use_cache": {
                            "type": "boolean",
                            "required": False,
                            "description": "Use cached page if available and fresh (default: true)",
                        },
                    },
                    estimated_cost=0.001,
                    estimated_duration_seconds=5,
                    success_probability=0.85,
                ),
                SkillAction(
                    name="scrape_table",
                    description=(
                        "Extract an HTML table into structured JSON rows. "
                        "Auto-detects headers from <thead> or first row."
                    ),
                    parameters={
                        "url": {"type": "string", "required": True, "description": "URL containing the table"},
                        "table_selector": {
                            "type": "string",
                            "required": False,
                            "description": "CSS selector for the table (default: 'table')",
                        },
                        "table_index": {
                            "type": "integer",
                            "required": False,
                            "description": "Which table to extract if multiple match (0-indexed, default: 0)",
                        },
                    },
                    estimated_cost=0.001,
                    estimated_duration_seconds=5,
                    success_probability=0.80,
                ),
                SkillAction(
                    name="scrape_list",
                    description=(
                        "Extract a repeated pattern (e.g. product cards, search results). "
                        "Specify a container selector and field selectors relative to each item."
                    ),
                    parameters={
                        "url": {"type": "string", "required": True, "description": "URL to scrape"},
                        "item_selector": {
                            "type": "string",
                            "required": True,
                            "description": "CSS selector for each repeated item (e.g. '.product-card')",
                        },
                        "fields": {
                            "type": "object",
                            "required": True,
                            "description": (
                                "Map of field names to sub-selectors within each item. "
                                'E.g. {"name": "h3", "price": ".price", "url": "a|href"}'
                            ),
                        },
                        "limit": {
                            "type": "integer",
                            "required": False,
                            "description": "Max items to return (default: 50)",
                        },
                    },
                    estimated_cost=0.001,
                    estimated_duration_seconds=5,
                    success_probability=0.80,
                ),
                SkillAction(
                    name="extract_links",
                    description="Extract all links from a page, optionally filtered by pattern.",
                    parameters={
                        "url": {"type": "string", "required": True, "description": "URL to extract links from"},
                        "pattern": {
                            "type": "string",
                            "required": False,
                            "description": "Regex pattern to filter link URLs (e.g. '/products/')",
                        },
                        "selector": {
                            "type": "string",
                            "required": False,
                            "description": "CSS selector to scope link extraction (default: 'a')",
                        },
                    },
                    estimated_cost=0.001,
                    estimated_duration_seconds=3,
                    success_probability=0.90,
                ),
                SkillAction(
                    name="extract_metadata",
                    description=(
                        "Extract page metadata: title, description, Open Graph tags, "
                        "JSON-LD structured data, canonical URL, etc."
                    ),
                    parameters={
                        "url": {"type": "string", "required": True, "description": "URL to analyze"},
                    },
                    estimated_cost=0.001,
                    estimated_duration_seconds=3,
                    success_probability=0.90,
                ),
                SkillAction(
                    name="monitor_changes",
                    description=(
                        "Set up or check a page monitor. Compares current content hash "
                        "with previous snapshot to detect changes."
                    ),
                    parameters={
                        "url": {"type": "string", "required": True, "description": "URL to monitor"},
                        "selector": {
                            "type": "string",
                            "required": False,
                            "description": "CSS selector to monitor specific section (default: whole page body)",
                        },
                        "label": {
                            "type": "string",
                            "required": False,
                            "description": "Friendly name for this monitor",
                        },
                    },
                    estimated_cost=0.001,
                    estimated_duration_seconds=5,
                    success_probability=0.85,
                ),
                SkillAction(
                    name="list_monitors",
                    description="List all active page change monitors and their status.",
                    parameters={},
                    estimated_cost=0,
                    estimated_duration_seconds=1,
                    success_probability=1.0,
                ),
                SkillAction(
                    name="scrape_history",
                    description="View recent scraping history and statistics.",
                    parameters={
                        "limit": {
                            "type": "integer",
                            "required": False,
                            "description": "Number of recent entries to return (default: 20)",
                        },
                    },
                    estimated_cost=0,
                    estimated_duration_seconds=1,
                    success_probability=1.0,
                ),
            ],
        )

    def check_credentials(self) -> bool:
        """No credentials required - uses public HTTP."""
        return True

    async def execute(self, action: str, params: Dict) -> SkillResult:
        handlers = {
            "scrape": self._scrape,
            "scrape_table": self._scrape_table,
            "scrape_list": self._scrape_list,
            "extract_links": self._extract_links,
            "extract_metadata": self._extract_metadata,
            "monitor_changes": self._monitor_changes,
            "list_monitors": self._list_monitors,
            "scrape_history": self._scrape_history,
        }
        handler = handlers.get(action)
        if not handler:
            return SkillResult(success=False, message=f"Unknown action: {action}")
        try:
            return await handler(params)
        except Exception as e:
            return SkillResult(success=False, message=f"Error in {action}: {e}")

    # ── HTTP fetching ──────────────────────────────────────────

    async def _fetch(self, url: str, use_cache: bool = True) -> Tuple[str, bool]:
        """Fetch URL content. Returns (html, from_cache)."""
        if not HAS_HTTPX:
            raise RuntimeError("httpx is required for web scraping. Install with: pip install httpx")

        # Check cache
        if use_cache:
            cached = self._get_cached(url)
            if cached:
                return cached, True

        async with httpx.AsyncClient(
            follow_redirects=True,
            timeout=DEFAULT_TIMEOUT,
            headers={"User-Agent": DEFAULT_USER_AGENT},
        ) as client:
            resp = await client.get(url)
            resp.raise_for_status()

            # Guard against huge responses
            content = resp.text
            if len(content) > MAX_RESPONSE_SIZE:
                content = content[:MAX_RESPONSE_SIZE]

        # Cache the result
        self._set_cached(url, content)
        return content, False

    def _cache_key(self, url: str) -> str:
        return hashlib.sha256(url.encode()).hexdigest()[:16]

    def _get_cached(self, url: str) -> Optional[str]:
        cache = _load_json(CACHE_FILE)
        key = self._cache_key(url)
        entry = cache.get(key)
        if not entry:
            return None
        if time.time() - entry.get("ts", 0) > MAX_CACHE_AGE:
            return None
        return entry.get("html")

    def _set_cached(self, url: str, html_content: str):
        cache = _load_json(CACHE_FILE)
        key = self._cache_key(url)
        cache[key] = {"url": url, "html": html_content, "ts": time.time()}
        # Prune old entries (keep last 50)
        if len(cache) > 50:
            sorted_keys = sorted(cache.keys(), key=lambda k: cache[k].get("ts", 0))
            for old_key in sorted_keys[: len(cache) - 50]:
                del cache[old_key]
        _save_json(CACHE_FILE, cache)

    def _record_scrape(self, url: str, action: str, success: bool, items: int = 0):
        """Record a scrape in history for tracking."""
        history = _load_json(HISTORY_FILE)
        if not isinstance(history, list):
            history = []
        history.append({
            "url": url,
            "action": action,
            "success": success,
            "items": items,
            "timestamp": datetime.now().isoformat(),
        })
        # Trim history
        if len(history) > MAX_HISTORY_ENTRIES:
            history = history[-MAX_HISTORY_ENTRIES:]
        _save_json(HISTORY_FILE, history)

    # ── Actions ────────────────────────────────────────────────

    async def _scrape(self, params: Dict) -> SkillResult:
        """Scrape a URL using CSS selectors."""
        url = params.get("url", "").strip()
        if not url:
            return SkillResult(success=False, message="'url' is required")

        selectors = params.get("selectors")
        if not selectors or not isinstance(selectors, dict):
            return SkillResult(success=False, message="'selectors' must be a non-empty object")

        limit = min(params.get("limit", 100), 500)
        use_cache = params.get("use_cache", True)

        html_content, from_cache = await self._fetch(url, use_cache)
        parser = _get_parser()
        results: Dict[str, Any] = {}
        total_items = 0

        for field_name, selector_str in selectors.items():
            # Check for |attr syntax
            attr = None
            if "|" in selector_str:
                selector_str, attr = selector_str.rsplit("|", 1)

            if attr:
                items = parser.get_attr(html_content, selector_str, attr)
                # Resolve relative URLs for href/src
                if attr in ("href", "src"):
                    items = [urljoin(url, i) for i in items]
            else:
                raw = parser.select_all(html_content, selector_str)
                items = [parser.get_text(r) for r in raw]

            results[field_name] = items[:limit]
            total_items += len(results[field_name])

        self._record_scrape(url, "scrape", True, total_items)

        return SkillResult(
            success=True,
            message=f"Extracted {total_items} items across {len(selectors)} fields from {url}"
            + (" (cached)" if from_cache else ""),
            data={
                "url": url,
                "results": results,
                "total_items": total_items,
                "from_cache": from_cache,
                "parser": "beautifulsoup" if HAS_BS4 else "regex",
            },
        )

    async def _scrape_table(self, params: Dict) -> SkillResult:
        """Extract an HTML table into structured rows."""
        url = params.get("url", "").strip()
        if not url:
            return SkillResult(success=False, message="'url' is required")

        table_selector = params.get("table_selector", "table")
        table_index = params.get("table_index", 0)

        html_content, from_cache = await self._fetch(url)
        parser = _get_parser()

        tables = parser.select_all(html_content, table_selector)
        if not tables:
            self._record_scrape(url, "scrape_table", False)
            return SkillResult(success=False, message=f"No tables found matching '{table_selector}'")

        if table_index >= len(tables):
            return SkillResult(
                success=False,
                message=f"Table index {table_index} out of range. Found {len(tables)} tables.",
            )

        table_html = tables[table_index]

        # Extract rows
        rows_html = parser.select_all(table_html, "tr")
        if not rows_html:
            self._record_scrape(url, "scrape_table", False)
            return SkillResult(success=False, message="Table has no rows")

        # Extract headers
        headers: List[str] = []
        header_cells = parser.select_all(rows_html[0], "th")
        if header_cells:
            headers = [parser.get_text(h) for h in header_cells]
            data_rows = rows_html[1:]
        else:
            # Use first row as headers
            first_row_cells = parser.select_all(rows_html[0], "td")
            headers = [parser.get_text(c) for c in first_row_cells]
            data_rows = rows_html[1:]

        if not headers:
            headers = [f"col_{i}" for i in range(10)]

        # Extract data
        rows: List[Dict[str, str]] = []
        for row_html in data_rows:
            cells = parser.select_all(row_html, "td")
            if not cells:
                continue
            row_data = {}
            for i, cell in enumerate(cells):
                key = headers[i] if i < len(headers) else f"col_{i}"
                row_data[key] = parser.get_text(cell)
            rows.append(row_data)

        self._record_scrape(url, "scrape_table", True, len(rows))

        return SkillResult(
            success=True,
            message=f"Extracted {len(rows)} rows with {len(headers)} columns from table",
            data={
                "url": url,
                "headers": headers,
                "rows": rows,
                "row_count": len(rows),
                "column_count": len(headers),
                "from_cache": from_cache,
            },
        )

    async def _scrape_list(self, params: Dict) -> SkillResult:
        """Extract a repeated pattern from a page."""
        url = params.get("url", "").strip()
        if not url:
            return SkillResult(success=False, message="'url' is required")

        item_selector = params.get("item_selector", "").strip()
        if not item_selector:
            return SkillResult(success=False, message="'item_selector' is required")

        fields = params.get("fields")
        if not fields or not isinstance(fields, dict):
            return SkillResult(success=False, message="'fields' must be a non-empty object")

        limit = min(params.get("limit", 50), 200)

        html_content, from_cache = await self._fetch(url)
        parser = _get_parser()

        items_html = parser.select_all(html_content, item_selector)
        if not items_html:
            self._record_scrape(url, "scrape_list", False)
            return SkillResult(
                success=False,
                message=f"No items found matching '{item_selector}'",
            )

        items: List[Dict[str, str]] = []
        for item_html in items_html[:limit]:
            item_data: Dict[str, str] = {}
            for field_name, sub_selector in fields.items():
                attr = None
                if "|" in sub_selector:
                    sub_selector, attr = sub_selector.rsplit("|", 1)

                if attr:
                    vals = parser.get_attr(item_html, sub_selector, attr)
                    if vals:
                        val = vals[0]
                        if attr in ("href", "src"):
                            val = urljoin(url, val)
                        item_data[field_name] = val
                    else:
                        item_data[field_name] = ""
                else:
                    sub_matches = parser.select_all(item_html, sub_selector)
                    if sub_matches:
                        item_data[field_name] = parser.get_text(sub_matches[0])
                    else:
                        # Try direct text extraction from item
                        item_data[field_name] = ""
            items.append(item_data)

        self._record_scrape(url, "scrape_list", True, len(items))

        return SkillResult(
            success=True,
            message=f"Extracted {len(items)} items with {len(fields)} fields each",
            data={
                "url": url,
                "items": items,
                "item_count": len(items),
                "fields": list(fields.keys()),
                "from_cache": from_cache,
            },
        )

    async def _extract_links(self, params: Dict) -> SkillResult:
        """Extract links from a page with optional filtering."""
        url = params.get("url", "").strip()
        if not url:
            return SkillResult(success=False, message="'url' is required")

        pattern = params.get("pattern")
        selector = params.get("selector", "a")

        html_content, from_cache = await self._fetch(url)
        parser = _get_parser()

        # Get all href attributes
        links = parser.get_attr(html_content, selector, "href")

        # Get link text too
        link_elements = parser.select_all(html_content, selector)
        link_texts = [parser.get_text(el) for el in link_elements]

        # Build link data
        results: List[Dict[str, str]] = []
        for i, href in enumerate(links):
            abs_url = urljoin(url, href)
            text = link_texts[i] if i < len(link_texts) else ""

            # Apply pattern filter
            if pattern:
                if not re.search(pattern, abs_url):
                    continue

            results.append({"url": abs_url, "text": text})

        # Deduplicate by URL
        seen = set()
        unique_links = []
        for link in results:
            if link["url"] not in seen:
                seen.add(link["url"])
                unique_links.append(link)

        self._record_scrape(url, "extract_links", True, len(unique_links))

        return SkillResult(
            success=True,
            message=f"Extracted {len(unique_links)} unique links"
            + (f" matching '{pattern}'" if pattern else ""),
            data={
                "url": url,
                "links": unique_links,
                "total_links": len(links),
                "unique_links": len(unique_links),
                "from_cache": from_cache,
            },
        )

    async def _extract_metadata(self, params: Dict) -> SkillResult:
        """Extract page metadata: title, description, OG tags, JSON-LD, etc."""
        url = params.get("url", "").strip()
        if not url:
            return SkillResult(success=False, message="'url' is required")

        html_content, from_cache = await self._fetch(url)
        parser = _get_parser()
        meta: Dict[str, Any] = {"url": url}

        # Title
        titles = parser.select_all(html_content, "title")
        if titles:
            meta["title"] = parser.get_text(titles[0])

        # Meta description
        desc_pattern = r'<meta\s+name=["\']description["\']\s+content=["\']([^"\']*)["\']'
        desc_match = re.search(desc_pattern, html_content, re.IGNORECASE)
        if not desc_match:
            desc_pattern = r'<meta\s+content=["\']([^"\']*?)["\']\s+name=["\']description["\']'
            desc_match = re.search(desc_pattern, html_content, re.IGNORECASE)
        if desc_match:
            meta["description"] = html.unescape(desc_match.group(1))

        # Open Graph tags
        og_tags: Dict[str, str] = {}
        for match in re.finditer(
            r'<meta\s+property=["\']og:(\w+)["\']\s+content=["\']([^"\']*)["\']',
            html_content,
            re.IGNORECASE,
        ):
            og_tags[match.group(1)] = html.unescape(match.group(2))
        # Also match reversed attribute order
        for match in re.finditer(
            r'<meta\s+content=["\']([^"\']*?)["\']\s+property=["\']og:(\w+)["\']',
            html_content,
            re.IGNORECASE,
        ):
            og_tags[match.group(2)] = html.unescape(match.group(1))
        if og_tags:
            meta["open_graph"] = og_tags

        # Twitter Card tags
        twitter_tags: Dict[str, str] = {}
        for match in re.finditer(
            r'<meta\s+(?:name|property)=["\']twitter:(\w+)["\']\s+content=["\']([^"\']*)["\']',
            html_content,
            re.IGNORECASE,
        ):
            twitter_tags[match.group(1)] = html.unescape(match.group(2))
        if twitter_tags:
            meta["twitter_card"] = twitter_tags

        # Canonical URL
        canonical_match = re.search(
            r'<link\s+rel=["\']canonical["\']\s+href=["\']([^"\']*)["\']',
            html_content,
            re.IGNORECASE,
        )
        if canonical_match:
            meta["canonical_url"] = canonical_match.group(1)

        # JSON-LD structured data
        jsonld_matches = re.findall(
            r'<script\s+type=["\']application/ld\+json["\']>(.*?)</script>',
            html_content,
            re.DOTALL | re.IGNORECASE,
        )
        if jsonld_matches:
            meta["json_ld"] = []
            for jsonld_str in jsonld_matches:
                try:
                    meta["json_ld"].append(json.loads(jsonld_str.strip()))
                except json.JSONDecodeError:
                    pass

        # Language
        lang_match = re.search(r'<html[^>]*\slang=["\']([^"\']+)["\']', html_content, re.IGNORECASE)
        if lang_match:
            meta["language"] = lang_match.group(1)

        # H1 headings
        h1s = parser.select_all(html_content, "h1")
        if h1s:
            meta["h1_headings"] = [parser.get_text(h) for h in h1s[:5]]

        self._record_scrape(url, "extract_metadata", True)

        return SkillResult(
            success=True,
            message=f"Extracted metadata from {url} ({len(meta)} fields)",
            data=meta,
        )

    async def _monitor_changes(self, params: Dict) -> SkillResult:
        """Monitor a page/section for changes."""
        url = params.get("url", "").strip()
        if not url:
            return SkillResult(success=False, message="'url' is required")

        selector = params.get("selector")
        label = params.get("label", urlparse(url).netloc)

        html_content, _ = await self._fetch(url, use_cache=False)
        parser = _get_parser()

        # Extract monitored content
        if selector:
            matches = parser.select_all(html_content, selector)
            content_text = " ".join(parser.get_text(m) for m in matches)
        else:
            # Use full page body
            body_matches = parser.select_all(html_content, "body")
            if body_matches:
                content_text = parser.get_text(body_matches[0])
            else:
                content_text = parser.get_text(html_content)

        content_hash = hashlib.sha256(content_text.encode()).hexdigest()[:32]

        # Load monitors
        monitors = _load_json(MONITORS_FILE)
        if not isinstance(monitors, dict):
            monitors = {}

        monitor_key = self._cache_key(url + (selector or ""))
        prev = monitors.get(monitor_key)

        changed = False
        diff_summary = ""
        if prev:
            if prev.get("hash") != content_hash:
                changed = True
                diff_summary = (
                    f"Content changed since {prev.get('last_checked', 'unknown')}. "
                    f"Previous length: {prev.get('content_length', '?')}, "
                    f"Current length: {len(content_text)}"
                )
            else:
                diff_summary = f"No change since {prev.get('last_checked', 'unknown')}"
        else:
            diff_summary = "First check - baseline recorded"

        # Update monitor
        monitors[monitor_key] = {
            "url": url,
            "selector": selector,
            "label": label,
            "hash": content_hash,
            "content_length": len(content_text),
            "last_checked": datetime.now().isoformat(),
            "check_count": (prev.get("check_count", 0) if prev else 0) + 1,
            "last_changed": (
                datetime.now().isoformat()
                if changed or not prev
                else prev.get("last_changed", datetime.now().isoformat())
            ),
            "changes_detected": (prev.get("changes_detected", 0) if prev else 0) + (1 if changed else 0),
        }
        _save_json(MONITORS_FILE, monitors)

        self._record_scrape(url, "monitor_changes", True)

        return SkillResult(
            success=True,
            message=f"{'CHANGED' if changed else 'No change'}: {diff_summary}",
            data={
                "url": url,
                "label": label,
                "changed": changed,
                "content_hash": content_hash,
                "content_length": len(content_text),
                "summary": diff_summary,
                "monitor": monitors[monitor_key],
            },
        )

    async def _list_monitors(self, params: Dict) -> SkillResult:
        """List all active monitors."""
        monitors = _load_json(MONITORS_FILE)
        if not isinstance(monitors, dict):
            monitors = {}

        monitor_list = list(monitors.values())
        monitor_list.sort(key=lambda m: m.get("last_checked", ""), reverse=True)

        return SkillResult(
            success=True,
            message=f"{len(monitor_list)} active monitors",
            data={"monitors": monitor_list, "count": len(monitor_list)},
        )

    async def _scrape_history(self, params: Dict) -> SkillResult:
        """View recent scraping history."""
        limit = min(params.get("limit", 20), 100)
        history = _load_json(HISTORY_FILE)
        if not isinstance(history, list):
            history = []

        recent = history[-limit:]
        recent.reverse()

        # Compute stats
        total = len(history)
        successful = sum(1 for h in history if h.get("success"))
        total_items = sum(h.get("items", 0) for h in history)
        domains = set()
        for h in history:
            try:
                domains.add(urlparse(h.get("url", "")).netloc)
            except Exception:
                pass

        return SkillResult(
            success=True,
            message=f"Showing {len(recent)} of {total} scraping events",
            data={
                "history": recent,
                "stats": {
                    "total_scrapes": total,
                    "successful": successful,
                    "success_rate": successful / total if total > 0 else 0,
                    "total_items_extracted": total_items,
                    "unique_domains": len(domains),
                    "domains": list(domains)[:20],
                },
            },
        )
