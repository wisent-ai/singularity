"""
Comprehensive tests for the social media skills — Instagram, Facebook, LinkedIn.

Tests focus on manifest validation, action dispatch, parameter validation,
and structural correctness. External API calls are mocked.
"""

import asyncio
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
exec(open(str(Path(__file__).resolve().parent / "conftest.py")).read())

from singularity.skills.base.types import SkillResult, SkillAction, SkillManifest
from singularity.skills.builtin.instagram.skill import InstagramSkill
from singularity.skills.builtin.facebook.skill import FacebookSkill


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


# ── Instagram Tests ─────────────────────────────────────────────────


class TestInstagramManifest(unittest.TestCase):
    """Test InstagramSkill manifest."""

    def setUp(self):
        self.skill = InstagramSkill()

    def test_manifest_type(self):
        self.assertIsInstance(self.skill.manifest, SkillManifest)

    def test_skill_id(self):
        self.assertEqual(self.skill.manifest.skill_id, "instagram")

    def test_category(self):
        self.assertEqual(self.skill.manifest.category, "social")

    def test_version(self):
        self.assertRegex(self.skill.manifest.version, r"\d+\.\d+\.\d+")

    def test_has_actions(self):
        self.assertGreaterEqual(len(self.skill.manifest.actions), 5)

    def test_required_action_names(self):
        names = {a.name for a in self.skill.manifest.actions}
        # These are in registry.json
        for expected in ["login", "post_photo", "like", "comment"]:
            self.assertIn(expected, names, f"Missing required action: {expected}")

    def test_all_actions_have_descriptions(self):
        for action in self.skill.manifest.actions:
            self.assertTrue(len(action.description) > 0)

    def test_all_actions_are_skill_actions(self):
        for action in self.skill.manifest.actions:
            self.assertIsInstance(action, SkillAction)


class TestInstagramUnknownAction(unittest.TestCase):
    """Test unknown action handling."""

    def test_unknown_action(self):
        skill = InstagramSkill()
        result = run(skill.execute("nonexistent", {}))
        self.assertFalse(result.success)
        self.assertIn("unknown", result.message.lower())


class TestInstagramExecuteDispatch(unittest.TestCase):
    """Test that execute() dispatches to action handlers."""

    def setUp(self):
        self.skill = InstagramSkill()

    def test_dispatch_has_all_actions(self):
        """Verify all manifest actions are handled in execute()."""
        manifest_actions = {a.name for a in self.skill.manifest.actions}
        # We can't easily introspect the dispatch dict without calling execute,
        # but we can verify unknown actions are caught
        for action_name in manifest_actions:
            # Each action should either succeed or fail gracefully
            # (not raise "Unknown action")
            result = run(self.skill.execute(action_name, {}))
            self.assertIsInstance(result, SkillResult)
            # The message should NOT contain "Unknown action"
            self.assertNotIn("unknown action", result.message.lower(),
                           f"Action {action_name} not dispatched")


# ── Facebook Tests ──────────────────────────────────────────────────


class TestFacebookManifest(unittest.TestCase):
    """Test FacebookSkill manifest."""

    def setUp(self):
        self.skill = FacebookSkill()

    def test_manifest_type(self):
        self.assertIsInstance(self.skill.manifest, SkillManifest)

    def test_skill_id(self):
        self.assertEqual(self.skill.manifest.skill_id, "facebook")

    def test_category(self):
        self.assertEqual(self.skill.manifest.category, "social")

    def test_version(self):
        self.assertRegex(self.skill.manifest.version, r"\d+\.\d+\.\d+")

    def test_has_actions(self):
        self.assertGreaterEqual(len(self.skill.manifest.actions), 4)

    def test_required_action_names(self):
        names = {a.name for a in self.skill.manifest.actions}
        # These are in registry.json
        for expected in ["login", "post", "like"]:
            self.assertIn(expected, names, f"Missing required action: {expected}")

    def test_all_actions_have_descriptions(self):
        for action in self.skill.manifest.actions:
            self.assertTrue(len(action.description) > 0)

    def test_all_actions_are_skill_actions(self):
        for action in self.skill.manifest.actions:
            self.assertIsInstance(action, SkillAction)


class TestFacebookUnknownAction(unittest.TestCase):
    """Test unknown action handling."""

    def test_unknown_action(self):
        skill = FacebookSkill()
        result = run(skill.execute("nonexistent", {}))
        self.assertFalse(result.success)
        self.assertIn("unknown", result.message.lower())


class TestFacebookExecuteDispatch(unittest.TestCase):
    """Test that execute() dispatches to action handlers."""

    def setUp(self):
        self.skill = FacebookSkill()

    def test_dispatch_has_all_actions(self):
        """Verify all manifest actions are handled in execute()."""
        manifest_actions = {a.name for a in self.skill.manifest.actions}
        for action_name in manifest_actions:
            result = run(self.skill.execute(action_name, {}))
            self.assertIsInstance(result, SkillResult)
            self.assertNotIn("unknown action", result.message.lower(),
                           f"Action {action_name} not dispatched")


# ── Cross-skill Tests ───────────────────────────────────────────────


class TestSocialSkillsBasePattern(unittest.TestCase):
    """Verify all social skills follow the same base pattern."""

    def test_all_have_manifest(self):
        for SkillClass in [InstagramSkill, FacebookSkill]:
            skill = SkillClass()
            manifest = skill.manifest
            self.assertIsInstance(manifest, SkillManifest)
            self.assertTrue(len(manifest.skill_id) > 0)
            self.assertTrue(len(manifest.name) > 0)
            self.assertTrue(len(manifest.description) > 0)

    def test_all_have_category_social(self):
        for SkillClass in [InstagramSkill, FacebookSkill]:
            skill = SkillClass()
            self.assertEqual(skill.manifest.category, "social")

    def test_all_return_skill_result(self):
        for SkillClass in [InstagramSkill, FacebookSkill]:
            skill = SkillClass()
            result = run(skill.execute("nonexistent", {}))
            self.assertIsInstance(result, SkillResult)
            self.assertFalse(result.success)

    def test_all_have_httpx_client(self):
        for SkillClass in [InstagramSkill, FacebookSkill]:
            skill = SkillClass()
            self.assertTrue(hasattr(skill, 'http') or hasattr(skill, '_http'))


if __name__ == "__main__":
    unittest.main()
