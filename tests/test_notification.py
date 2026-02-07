#!/usr/bin/env python3
"""
Tests for NotificationSkill.

Uses importlib to load modules directly to avoid pulling in
the full singularity dependency tree. Uses mock senders to
avoid network calls.
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


# Import base first, then notification
_base_mod = _import_module("singularity.skills.base", os.path.join(_SKILLS, "base.py"))
_notif_mod = _import_module(
    "singularity.skills.notification",
    os.path.join(_SKILLS, "notification.py"),
)

NotificationSkill = _notif_mod.NotificationSkill
_load_json = _notif_mod._load_json
_save_json = _notif_mod._save_json


def run_async(coro):
    """Helper to run async functions in sync tests."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class TestNotificationSkill(unittest.TestCase):
    """Test NotificationSkill with mocked dispatch."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.skill = NotificationSkill(credentials={})
        # Patch data directory
        self._orig_dir = _notif_mod.DATA_DIR
        self._orig_config = _notif_mod.CONFIG_FILE
        self._orig_history = _notif_mod.HISTORY_FILE
        _notif_mod.DATA_DIR = Path(self.tmpdir)
        _notif_mod.CONFIG_FILE = Path(self.tmpdir) / "channels.json"
        _notif_mod.HISTORY_FILE = Path(self.tmpdir) / "history.json"

        # Mock dispatcher that always succeeds
        self._dispatch_calls = []
        original_dispatch = self.skill._dispatch

        async def mock_dispatch(channel, config, message, title=""):
            self._dispatch_calls.append({
                "channel": channel,
                "config": config,
                "message": message,
                "title": title,
            })
            return {"success": True}

        self.skill._dispatch = mock_dispatch

    def tearDown(self):
        _notif_mod.DATA_DIR = self._orig_dir
        _notif_mod.CONFIG_FILE = self._orig_config
        _notif_mod.HISTORY_FILE = self._orig_history
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _configure_telegram(self):
        """Helper to configure a test telegram channel."""
        return run_async(self.skill.execute("configure_channel", {
            "channel": "telegram",
            "config": {"bot_token": "test-token", "chat_id": "12345"},
            "min_severity": "info",
        }))

    def _configure_discord(self):
        """Helper to configure a test discord channel."""
        return run_async(self.skill.execute("configure_channel", {
            "channel": "discord",
            "config": {"webhook_url": "https://discord.com/api/webhooks/test"},
            "min_severity": "warning",
        }))

    # ── Manifest ──

    def test_manifest(self):
        m = self.skill.manifest
        self.assertEqual(m.skill_id, "notification")
        self.assertEqual(m.category, "communication")
        self.assertEqual(m.version, "1.0.0")
        self.assertEqual(m.author, "Adam (ADAM)")
        self.assertEqual(len(m.required_credentials), 0)

    def test_manifest_actions(self):
        m = self.skill.manifest
        action_names = [a.name for a in m.actions]
        self.assertIn("send", action_names)
        self.assertIn("alert", action_names)
        self.assertIn("configure_channel", action_names)
        self.assertIn("list_channels", action_names)
        self.assertIn("test_channel", action_names)
        self.assertIn("notification_history", action_names)
        self.assertIn("remove_channel", action_names)
        self.assertEqual(len(action_names), 7)

    def test_check_credentials(self):
        self.assertTrue(self.skill.check_credentials())

    # ── configure_channel ──

    def test_configure_telegram(self):
        result = self._configure_telegram()
        self.assertTrue(result.success)
        self.assertIn("telegram", result.message)

    def test_configure_discord(self):
        result = self._configure_discord()
        self.assertTrue(result.success)

    def test_configure_missing_channel(self):
        result = run_async(self.skill.execute("configure_channel", {
            "config": {"url": "http://test"},
        }))
        self.assertFalse(result.success)

    def test_configure_unknown_channel(self):
        result = run_async(self.skill.execute("configure_channel", {
            "channel": "pigeon",
            "config": {"url": "http://test"},
        }))
        self.assertFalse(result.success)
        self.assertIn("Unknown channel", result.message)

    def test_configure_missing_config(self):
        result = run_async(self.skill.execute("configure_channel", {
            "channel": "telegram",
        }))
        self.assertFalse(result.success)

    def test_configure_invalid_config_missing_fields(self):
        result = run_async(self.skill.execute("configure_channel", {
            "channel": "telegram",
            "config": {"bot_token": "test"},  # Missing chat_id
        }))
        self.assertFalse(result.success)
        self.assertIn("chat_id", result.message)

    def test_configure_with_min_severity(self):
        result = run_async(self.skill.execute("configure_channel", {
            "channel": "webhook",
            "config": {"url": "http://test.com/hook"},
            "min_severity": "critical",
        }))
        self.assertTrue(result.success)
        self.assertEqual(result.data["min_severity"], "critical")

    def test_configure_disabled(self):
        result = run_async(self.skill.execute("configure_channel", {
            "channel": "webhook",
            "config": {"url": "http://test.com/hook"},
            "enabled": False,
        }))
        self.assertTrue(result.success)
        self.assertFalse(result.data["enabled"])

    # ── list_channels ──

    def test_list_channels_empty(self):
        result = run_async(self.skill.execute("list_channels", {}))
        self.assertTrue(result.success)
        self.assertEqual(result.data["count"], 0)

    def test_list_channels_after_configure(self):
        self._configure_telegram()
        self._configure_discord()
        result = run_async(self.skill.execute("list_channels", {}))
        self.assertTrue(result.success)
        self.assertEqual(result.data["count"], 2)
        self.assertIn("telegram", result.data["channels"])
        self.assertIn("discord", result.data["channels"])

    def test_list_channels_sanitized(self):
        """Ensure tokens are not exposed in channel listing."""
        self._configure_telegram()
        result = run_async(self.skill.execute("list_channels", {}))
        ch = result.data["channels"]["telegram"]
        self.assertNotIn("bot_token", str(ch))
        self.assertNotIn("test-token", str(ch))

    # ── send ──

    def test_send_basic(self):
        self._configure_telegram()
        result = run_async(self.skill.execute("send", {
            "channel": "telegram",
            "message": "Hello world!",
        }))
        self.assertTrue(result.success)
        self.assertEqual(len(self._dispatch_calls), 1)
        self.assertEqual(self._dispatch_calls[0]["channel"], "telegram")
        self.assertIn("Hello world!", self._dispatch_calls[0]["message"])

    def test_send_with_severity(self):
        self._configure_telegram()
        result = run_async(self.skill.execute("send", {
            "channel": "telegram",
            "message": "Server down!",
            "severity": "critical",
        }))
        self.assertTrue(result.success)
        self.assertIn("[CRITICAL]", self._dispatch_calls[0]["message"])

    def test_send_with_title(self):
        self._configure_telegram()
        result = run_async(self.skill.execute("send", {
            "channel": "telegram",
            "message": "Test message",
            "title": "Important",
        }))
        self.assertTrue(result.success)
        self.assertEqual(self._dispatch_calls[0]["title"], "Important")

    def test_send_missing_message(self):
        self._configure_telegram()
        result = run_async(self.skill.execute("send", {
            "channel": "telegram",
        }))
        self.assertFalse(result.success)

    def test_send_missing_channel(self):
        result = run_async(self.skill.execute("send", {
            "message": "Hello",
        }))
        self.assertFalse(result.success)

    def test_send_unconfigured_channel(self):
        result = run_async(self.skill.execute("send", {
            "channel": "telegram",
            "message": "Hello",
        }))
        self.assertFalse(result.success)
        self.assertIn("not configured", result.message)

    def test_send_disabled_channel(self):
        run_async(self.skill.execute("configure_channel", {
            "channel": "webhook",
            "config": {"url": "http://test.com"},
            "enabled": False,
        }))
        result = run_async(self.skill.execute("send", {
            "channel": "webhook",
            "message": "Hello",
        }))
        self.assertFalse(result.success)
        self.assertIn("disabled", result.message)

    # ── alert ──

    def test_alert_to_all_channels(self):
        self._configure_telegram()
        self._configure_discord()
        result = run_async(self.skill.execute("alert", {
            "message": "System overload!",
            "severity": "critical",
            "title": "Critical Alert",
        }))
        self.assertTrue(result.success)
        self.assertEqual(result.data["channels_targeted"], 2)
        self.assertEqual(result.data["channels_succeeded"], 2)

    def test_alert_severity_routing(self):
        """Discord set to warning, so info alerts should only go to telegram."""
        self._configure_telegram()  # min_severity: info
        self._configure_discord()   # min_severity: warning
        result = run_async(self.skill.execute("alert", {
            "message": "FYI something happened",
            "severity": "info",
        }))
        self.assertTrue(result.success)
        # Only telegram should receive (info >= info threshold)
        # Discord has warning threshold, info doesn't meet it
        self.assertEqual(result.data["channels_targeted"], 1)

    def test_alert_critical_reaches_all(self):
        """Critical alerts should reach all channels regardless of threshold."""
        self._configure_telegram()
        self._configure_discord()
        result = run_async(self.skill.execute("alert", {
            "message": "Total failure!",
            "severity": "critical",
        }))
        self.assertTrue(result.success)
        self.assertEqual(result.data["channels_targeted"], 2)

    def test_alert_with_source(self):
        self._configure_telegram()
        result = run_async(self.skill.execute("alert", {
            "message": "Low balance detected",
            "severity": "warning",
            "source": "resource_watcher",
        }))
        self.assertTrue(result.success)
        self.assertIn("resource_watcher", self._dispatch_calls[0]["message"])

    def test_alert_no_channels(self):
        result = run_async(self.skill.execute("alert", {
            "message": "Hello",
            "severity": "info",
        }))
        self.assertFalse(result.success)
        self.assertIn("No channels configured", result.message)

    def test_alert_invalid_severity(self):
        result = run_async(self.skill.execute("alert", {
            "message": "Hello",
            "severity": "super_critical",
        }))
        self.assertFalse(result.success)
        self.assertIn("Invalid severity", result.message)

    def test_alert_missing_message(self):
        result = run_async(self.skill.execute("alert", {
            "severity": "info",
        }))
        self.assertFalse(result.success)

    # ── test_channel ──

    def test_test_channel(self):
        self._configure_telegram()
        result = run_async(self.skill.execute("test_channel", {
            "channel": "telegram",
        }))
        self.assertTrue(result.success)
        self.assertEqual(len(self._dispatch_calls), 1)
        self.assertIn("Test notification", self._dispatch_calls[0]["message"])

    def test_test_unconfigured_channel(self):
        result = run_async(self.skill.execute("test_channel", {
            "channel": "telegram",
        }))
        self.assertFalse(result.success)

    # ── notification_history ──

    def test_history_empty(self):
        result = run_async(self.skill.execute("notification_history", {}))
        self.assertTrue(result.success)
        self.assertEqual(len(result.data["history"]), 0)

    def test_history_after_sends(self):
        self._configure_telegram()
        run_async(self.skill.execute("send", {
            "channel": "telegram",
            "message": "First",
            "severity": "info",
        }))
        run_async(self.skill.execute("send", {
            "channel": "telegram",
            "message": "Second",
            "severity": "warning",
        }))

        result = run_async(self.skill.execute("notification_history", {}))
        self.assertTrue(result.success)
        self.assertEqual(len(result.data["history"]), 2)
        self.assertEqual(result.data["stats"]["total"], 2)
        self.assertEqual(result.data["stats"]["successful"], 2)

    def test_history_filter_by_channel(self):
        self._configure_telegram()
        self._configure_discord()
        run_async(self.skill.execute("send", {"channel": "telegram", "message": "A"}))
        run_async(self.skill.execute("send", {"channel": "discord", "message": "B"}))

        result = run_async(self.skill.execute("notification_history", {
            "channel": "telegram",
        }))
        self.assertTrue(result.success)
        self.assertEqual(len(result.data["history"]), 1)

    def test_history_filter_by_severity(self):
        self._configure_telegram()
        run_async(self.skill.execute("send", {"channel": "telegram", "message": "A", "severity": "info"}))
        run_async(self.skill.execute("send", {"channel": "telegram", "message": "B", "severity": "critical"}))

        result = run_async(self.skill.execute("notification_history", {
            "severity": "critical",
        }))
        self.assertTrue(result.success)
        self.assertEqual(len(result.data["history"]), 1)

    def test_history_limit(self):
        self._configure_telegram()
        for i in range(5):
            run_async(self.skill.execute("send", {"channel": "telegram", "message": f"msg {i}"}))

        result = run_async(self.skill.execute("notification_history", {"limit": 2}))
        self.assertTrue(result.success)
        self.assertEqual(len(result.data["history"]), 2)

    # ── remove_channel ──

    def test_remove_channel(self):
        self._configure_telegram()
        result = run_async(self.skill.execute("remove_channel", {
            "channel": "telegram",
        }))
        self.assertTrue(result.success)

        # Verify it's gone
        list_result = run_async(self.skill.execute("list_channels", {}))
        self.assertEqual(list_result.data["count"], 0)

    def test_remove_nonexistent_channel(self):
        result = run_async(self.skill.execute("remove_channel", {
            "channel": "telegram",
        }))
        self.assertFalse(result.success)
        self.assertIn("not found", result.message)

    # ── Unknown action ──

    def test_unknown_action(self):
        result = run_async(self.skill.execute("unknown_action", {}))
        self.assertFalse(result.success)
        self.assertIn("Unknown action", result.message)

    # ── to_dict ──

    def test_to_dict(self):
        d = self.skill.to_dict()
        self.assertEqual(d["skill_id"], "notification")
        self.assertEqual(d["category"], "communication")
        self.assertIn("actions", d)
        self.assertEqual(len(d["actions"]), 7)


# ── Channel validation tests ──────────────────────────────────

class TestChannelValidation(unittest.TestCase):
    """Test channel config validation logic."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.skill = NotificationSkill(credentials={})
        self._orig_dir = _notif_mod.DATA_DIR
        self._orig_config = _notif_mod.CONFIG_FILE
        self._orig_history = _notif_mod.HISTORY_FILE
        _notif_mod.DATA_DIR = Path(self.tmpdir)
        _notif_mod.CONFIG_FILE = Path(self.tmpdir) / "channels.json"
        _notif_mod.HISTORY_FILE = Path(self.tmpdir) / "history.json"

    def tearDown(self):
        _notif_mod.DATA_DIR = self._orig_dir
        _notif_mod.CONFIG_FILE = self._orig_config
        _notif_mod.HISTORY_FILE = self._orig_history
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_validate_telegram_valid(self):
        r = self.skill._validate_channel_config("telegram", {"bot_token": "t", "chat_id": "c"})
        self.assertTrue(r["valid"])

    def test_validate_telegram_missing_chat_id(self):
        r = self.skill._validate_channel_config("telegram", {"bot_token": "t"})
        self.assertFalse(r["valid"])

    def test_validate_discord_valid(self):
        r = self.skill._validate_channel_config("discord", {"webhook_url": "https://..."})
        self.assertTrue(r["valid"])

    def test_validate_discord_missing_url(self):
        r = self.skill._validate_channel_config("discord", {})
        self.assertFalse(r["valid"])

    def test_validate_slack_valid(self):
        r = self.skill._validate_channel_config("slack", {"webhook_url": "https://..."})
        self.assertTrue(r["valid"])

    def test_validate_sms_valid(self):
        r = self.skill._validate_channel_config("sms", {
            "account_sid": "a", "auth_token": "b",
            "from_number": "+1", "to_number": "+2",
        })
        self.assertTrue(r["valid"])

    def test_validate_sms_missing_fields(self):
        r = self.skill._validate_channel_config("sms", {"account_sid": "a"})
        self.assertFalse(r["valid"])

    def test_validate_webhook_valid(self):
        r = self.skill._validate_channel_config("webhook", {"url": "http://test.com"})
        self.assertTrue(r["valid"])


# ── Environment credential tests ──────────────────────────────

class TestEnvCredentials(unittest.TestCase):
    """Test environment-based channel auto-detection."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        _notif_mod.DATA_DIR = Path(self.tmpdir)
        _notif_mod.CONFIG_FILE = Path(self.tmpdir) / "channels.json"
        _notif_mod.HISTORY_FILE = Path(self.tmpdir) / "history.json"

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_telegram_from_env(self):
        skill = NotificationSkill(credentials={
            "TELEGRAM_BOT_TOKEN": "env-token",
            "TELEGRAM_CHAT_ID": "env-chat",
        })
        cfg = skill._get_env_config("telegram")
        self.assertIsNotNone(cfg)
        self.assertEqual(cfg["bot_token"], "env-token")
        self.assertEqual(cfg["chat_id"], "env-chat")

    def test_discord_from_env(self):
        skill = NotificationSkill(credentials={
            "DISCORD_WEBHOOK_URL": "https://discord.com/hook",
        })
        cfg = skill._get_env_config("discord")
        self.assertIsNotNone(cfg)
        self.assertEqual(cfg["webhook_url"], "https://discord.com/hook")

    def test_slack_from_env(self):
        skill = NotificationSkill(credentials={
            "SLACK_WEBHOOK_URL": "https://hooks.slack.com/test",
        })
        cfg = skill._get_env_config("slack")
        self.assertIsNotNone(cfg)

    def test_sms_from_env(self):
        skill = NotificationSkill(credentials={
            "TWILIO_ACCOUNT_SID": "sid",
            "TWILIO_AUTH_TOKEN": "token",
            "TWILIO_FROM_NUMBER": "+1234",
            "TWILIO_TO_NUMBER": "+5678",
        })
        cfg = skill._get_env_config("sms")
        self.assertIsNotNone(cfg)
        self.assertEqual(cfg["account_sid"], "sid")

    def test_no_env_returns_none(self):
        skill = NotificationSkill(credentials={})
        cfg = skill._get_env_config("telegram")
        self.assertIsNone(cfg)

    def test_partial_env_returns_none(self):
        skill = NotificationSkill(credentials={
            "TELEGRAM_BOT_TOKEN": "token",
            # Missing TELEGRAM_CHAT_ID
        })
        cfg = skill._get_env_config("telegram")
        self.assertIsNone(cfg)


if __name__ == "__main__":
    unittest.main()
