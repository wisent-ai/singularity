#!/usr/bin/env python3
"""
Tests for HTTPClientSkill.

Uses importlib to load modules directly. Mocks HTTP calls to avoid network deps.
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
from unittest.mock import AsyncMock, MagicMock, patch

# ── Direct imports ────────────────────────────────────────────
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SKILLS = os.path.join(_ROOT, "singularity", "skills")


def _import_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_base_mod = _import_module("singularity.skills.base", os.path.join(_SKILLS, "base.py"))
_hc_mod = _import_module("singularity.skills.http_client", os.path.join(_SKILLS, "http_client.py"))

HTTPClientSkill = _hc_mod.HTTPClientSkill


def run_async(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class TestHTTPClientSkill(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        _hc_mod.DATA_DIR = Path(self._tmp) / "http_client"
        _hc_mod.HISTORY_FILE = _hc_mod.DATA_DIR / "request_history.json"
        _hc_mod.SAVED_ENDPOINTS_FILE = _hc_mod.DATA_DIR / "saved_endpoints.json"
        self.skill = HTTPClientSkill()

    def tearDown(self):
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_manifest(self):
        m = self.skill.manifest
        self.assertEqual(m.skill_id, "http_client")
        self.assertEqual(m.category, "integration")

    def test_actions_list(self):
        actions = self.skill.manifest.actions
        names = [a.name for a in actions]
        self.assertIn("request", names)
        self.assertIn("get", names)
        self.assertIn("post_json", names)
        self.assertIn("save_endpoint", names)
        self.assertIn("call_endpoint", names)
        self.assertIn("list_endpoints", names)
        self.assertIn("history", names)
        self.assertIn("configure", names)

    def test_validate_url_blocks_http(self):
        err = self.skill._validate_url("http://example.com/api")
        self.assertIsNotNone(err)
        self.assertIn("HTTPS", err)

    def test_validate_url_allows_https(self):
        err = self.skill._validate_url("https://api.example.com/v1")
        self.assertIsNone(err)

    def test_validate_url_allows_localhost_http(self):
        err = self.skill._validate_url("http://localhost:8080/api")
        self.assertIsNone(err)

    def test_validate_url_blocks_metadata(self):
        err = self.skill._validate_url("http://169.254.169.254/latest/meta-data/")
        self.assertIsNotNone(err)

    def test_validate_url_bad_scheme(self):
        err = self.skill._validate_url("ftp://files.example.com")
        self.assertIsNotNone(err)

    def test_rate_limit(self):
        domain = "test.example.com"
        for _ in range(60):
            self.assertTrue(self.skill._check_rate_limit(domain, max_per_minute=60))
        self.assertFalse(self.skill._check_rate_limit(domain, max_per_minute=60))

    def test_configure_block_domain(self):
        result = run_async(self.skill.execute("configure", {"block_domain": "evil.com"}))
        self.assertTrue(result.success)
        self.assertIn("evil.com", result.data["blocked_domains"])

    def test_configure_allowlist(self):
        result = run_async(self.skill.execute("configure", {"allow_only": ["api.safe.com"]}))
        self.assertTrue(result.success)
        err = self.skill._validate_url("https://api.unsafe.com/v1")
        self.assertIsNotNone(err)
        err2 = self.skill._validate_url("https://api.safe.com/v1")
        self.assertIsNone(err2)

    def test_configure_clear_allowlist(self):
        self.skill._allowed_domains = {"only.this.com"}
        result = run_async(self.skill.execute("configure", {"clear_allowlist": True}))
        self.assertTrue(result.success)
        self.assertIsNone(self.skill._allowed_domains)

    def test_save_and_list_endpoints(self):
        result = run_async(self.skill.execute("save_endpoint", {
            "name": "my_api",
            "method": "POST",
            "url": "https://api.example.com/v1/data",
            "headers": {"Authorization": "Bearer tok123"},
        }))
        self.assertTrue(result.success)

        lst = run_async(self.skill.execute("list_endpoints", {}))
        self.assertTrue(lst.success)
        self.assertEqual(len(lst.data["endpoints"]), 1)
        self.assertEqual(lst.data["endpoints"][0]["name"], "my_api")

    def test_call_missing_endpoint(self):
        result = run_async(self.skill.execute("call_endpoint", {"name": "nonexistent"}))
        self.assertFalse(result.success)
        self.assertIn("not found", result.message)

    def test_history_empty(self):
        result = run_async(self.skill.execute("history", {}))
        self.assertTrue(result.success)
        self.assertEqual(result.data["stats"]["total_requests"], 0)

    def test_unknown_action(self):
        result = run_async(self.skill.execute("unknown_action", {}))
        self.assertFalse(result.success)

    @patch.object(_hc_mod, "HAS_HTTPX", False)
    def test_request_urllib_fallback_invalid_url(self):
        """Test that invalid URLs are caught before hitting urllib."""
        result = run_async(self.skill.execute("request", {
            "method": "GET", "url": "not-a-url"
        }))
        self.assertFalse(result.success)

    def test_request_unsupported_method(self):
        result = run_async(self.skill.execute("request", {
            "method": "TRACE", "url": "https://example.com"
        }))
        self.assertFalse(result.success)
        self.assertIn("Unsupported", result.message)

    def test_request_body_too_large(self):
        result = run_async(self.skill.execute("request", {
            "method": "POST",
            "url": "https://api.example.com/upload",
            "body": "x" * (_hc_mod.MAX_REQUEST_BODY_SIZE + 1),
        }))
        self.assertFalse(result.success)
        self.assertIn("too large", result.message)

    def test_history_recording(self):
        self.skill._record_history("GET", "https://api.example.com/data", 200, 0.5)
        self.skill._record_history("POST", "https://api.example.com/data", 500, 1.2, error="Server Error")
        result = run_async(self.skill.execute("history", {"limit": 10}))
        self.assertTrue(result.success)
        self.assertEqual(result.data["stats"]["total_requests"], 2)
        self.assertEqual(result.data["stats"]["successful"], 1)

    def test_history_domain_filter(self):
        self.skill._record_history("GET", "https://a.com/x", 200, 0.1)
        self.skill._record_history("GET", "https://b.com/y", 200, 0.2)
        result = run_async(self.skill.execute("history", {"domain_filter": "a.com"}))
        self.assertEqual(len(result.data["requests"]), 1)

    def test_get_shorthand_with_params(self):
        """Test that get action builds URL with query params."""
        # We can't make real requests, but we can test URL validation path
        result = run_async(self.skill.execute("get", {
            "url": "ftp://bad.scheme.com",
            "params": {"q": "test"},
        }))
        self.assertFalse(result.success)

    def test_estimate_cost(self):
        self.assertEqual(self.skill.estimate_cost("request", {}), 0.0)

    def test_save_endpoint_missing_name(self):
        result = run_async(self.skill.execute("save_endpoint", {
            "name": "", "method": "GET", "url": "https://api.com"
        }))
        self.assertFalse(result.success)

    def test_save_endpoint_missing_url(self):
        result = run_async(self.skill.execute("save_endpoint", {
            "name": "test", "method": "GET", "url": ""
        }))
        self.assertFalse(result.success)


if __name__ == "__main__":
    unittest.main()
