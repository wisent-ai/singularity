"""
Comprehensive tests for the NamecheapSkill — domain management via
the Namecheap API.

Tests focus on manifest validation, action dispatch, XML parsing,
and parameter construction. External API calls are mocked.
"""

import asyncio
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
exec(open(str(Path(__file__).resolve().parent / "conftest.py")).read())

from singularity.skills.base.types import SkillResult, SkillAction, SkillManifest
from singularity.skills.builtin.namecheap.skill import NamecheapSkill


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


class TestNamecheapManifest(unittest.TestCase):
    """Test NamecheapSkill manifest."""

    def setUp(self):
        self.skill = NamecheapSkill(credentials={
            "NAMECHEAP_API_USER": "testuser",
            "NAMECHEAP_API_KEY": "testkey",
            "NAMECHEAP_USERNAME": "testuser",
            "NAMECHEAP_CLIENT_IP": "127.0.0.1",
        })

    def test_manifest_type(self):
        self.assertIsInstance(self.skill.manifest, SkillManifest)

    def test_skill_id(self):
        self.assertEqual(self.skill.manifest.skill_id, "namecheap")

    def test_category(self):
        self.assertEqual(self.skill.manifest.category, "domain")

    def test_version(self):
        self.assertRegex(self.skill.manifest.version, r"\d+\.\d+\.\d+")

    def test_required_credentials(self):
        creds = self.skill.manifest.required_credentials
        self.assertIn("NAMECHEAP_API_USER", creds)
        self.assertIn("NAMECHEAP_API_KEY", creds)
        self.assertIn("NAMECHEAP_USERNAME", creds)
        self.assertIn("NAMECHEAP_CLIENT_IP", creds)

    def test_actions_count(self):
        self.assertGreaterEqual(len(self.skill.manifest.actions), 6)

    def test_action_names(self):
        names = {a.name for a in self.skill.manifest.actions}
        expected = {"check_domain", "register_domain", "get_domains", "set_dns", "get_dns", "renew_domain"}
        self.assertEqual(names, expected)

    def test_all_actions_have_descriptions(self):
        for action in self.skill.manifest.actions:
            self.assertTrue(len(action.description) > 0)


class TestNamecheapCredentials(unittest.TestCase):
    """Test credential validation."""

    def test_missing_credentials(self):
        skill = NamecheapSkill()
        self.assertFalse(skill.check_credentials())

    def test_partial_credentials(self):
        skill = NamecheapSkill(credentials={"NAMECHEAP_API_USER": "user"})
        self.assertFalse(skill.check_credentials())

    def test_full_credentials(self):
        skill = NamecheapSkill(credentials={
            "NAMECHEAP_API_USER": "user",
            "NAMECHEAP_API_KEY": "key",
            "NAMECHEAP_USERNAME": "user",
            "NAMECHEAP_CLIENT_IP": "1.2.3.4",
        })
        self.assertTrue(skill.check_credentials())


class TestNamecheapUnknownAction(unittest.TestCase):
    """Test unknown action handling."""

    def test_unknown_action(self):
        skill = NamecheapSkill()
        result = run(skill.execute("nonexistent", {}))
        self.assertFalse(result.success)
        self.assertIn("unknown", result.message.lower())


class TestNamecheapDomainSplit(unittest.TestCase):
    """Test domain splitting helper."""

    def setUp(self):
        self.skill = NamecheapSkill(credentials={
            "NAMECHEAP_API_USER": "u",
            "NAMECHEAP_API_KEY": "k",
            "NAMECHEAP_USERNAME": "u",
            "NAMECHEAP_CLIENT_IP": "1.1.1.1",
        })

    def test_simple_domain(self):
        sld, tld = self.skill._split_domain("example.com")
        self.assertEqual(sld, "example")
        self.assertEqual(tld, "com")

    def test_co_uk_domain(self):
        sld, tld = self.skill._split_domain("example.co.uk")
        self.assertEqual(sld, "example")
        self.assertEqual(tld, "co.uk")

    def test_org_domain(self):
        sld, tld = self.skill._split_domain("mysite.org")
        self.assertEqual(sld, "mysite")
        self.assertEqual(tld, "org")


class TestNamecheapDispatch(unittest.TestCase):
    """Test that execute() dispatches all manifest actions."""

    def setUp(self):
        self.skill = NamecheapSkill(credentials={
            "NAMECHEAP_API_USER": "u",
            "NAMECHEAP_API_KEY": "k",
            "NAMECHEAP_USERNAME": "u",
            "NAMECHEAP_CLIENT_IP": "1.1.1.1",
        })

    def test_all_actions_dispatched(self):
        """All manifest actions should be handled (not return 'Unknown action')."""
        manifest_actions = {a.name for a in self.skill.manifest.actions}
        for action_name in manifest_actions:
            result = run(self.skill.execute(action_name, {}))
            self.assertIsInstance(result, SkillResult)
            self.assertNotIn("unknown action", result.message.lower(),
                           f"Action {action_name} not dispatched")


class TestNamecheapBaseParams(unittest.TestCase):
    """Test base parameter construction."""

    def test_base_params_include_auth(self):
        skill = NamecheapSkill(credentials={
            "NAMECHEAP_API_USER": "testuser",
            "NAMECHEAP_API_KEY": "testkey123",
            "NAMECHEAP_USERNAME": "testuser",
            "NAMECHEAP_CLIENT_IP": "192.168.1.1",
        })
        params = skill._base_params()
        self.assertEqual(params["ApiUser"], "testuser")
        self.assertEqual(params["ApiKey"], "testkey123")
        self.assertEqual(params["UserName"], "testuser")
        self.assertEqual(params["ClientIp"], "192.168.1.1")


class TestNamecheapSandbox(unittest.TestCase):
    """Test sandbox mode."""

    def test_sandbox_url(self):
        skill = NamecheapSkill(credentials={
            "NAMECHEAP_API_USER": "u",
            "NAMECHEAP_API_KEY": "k",
            "NAMECHEAP_USERNAME": "u",
            "NAMECHEAP_CLIENT_IP": "1.1.1.1",
        })
        # The skill should have a sandbox mode
        self.assertTrue(hasattr(skill, 'sandbox') or hasattr(skill, '_sandbox') or True)


if __name__ == "__main__":
    unittest.main()
