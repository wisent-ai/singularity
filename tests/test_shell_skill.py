"""
Comprehensive tests for the ShellSkill — the core system skill for
executing shell commands, fetching URLs, managing background processes.

Tests use asyncio with mock-free validation where possible (real subprocess
calls to safe commands like echo, pwd, which) and structural tests for
everything else.
"""

import asyncio
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock

# Add project root and bootstrap dependency mocks
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
exec(open(str(Path(__file__).resolve().parent / "conftest.py")).read())

from singularity.skills.base.types import SkillResult, SkillAction, SkillManifest
from singularity.skills.builtin.shell.skill import (
    ShellSkill, MAX_OUTPUT_BYTES, DEFAULT_TIMEOUT, MAX_TIMEOUT, BLOCKED_COMMANDS,
)


def run(coro):
    """Helper to run async test coroutines."""
    return asyncio.get_event_loop().run_until_complete(coro)


class TestShellSkillManifest(unittest.TestCase):
    """Test skill manifest metadata."""

    def setUp(self):
        self.skill = ShellSkill()

    def test_manifest_type(self):
        self.assertIsInstance(self.skill.manifest, SkillManifest)

    def test_skill_id(self):
        self.assertEqual(self.skill.manifest.skill_id, "shell")

    def test_category(self):
        self.assertEqual(self.skill.manifest.category, "system")

    def test_version(self):
        self.assertEqual(self.skill.manifest.version, "1.0.0")

    def test_no_credentials_required(self):
        self.assertEqual(self.skill.manifest.required_credentials, [])

    def test_actions_count(self):
        actions = self.skill.manifest.actions
        self.assertGreaterEqual(len(actions), 8)

    def test_action_names(self):
        names = {a.name for a in self.skill.manifest.actions}
        expected = {"run", "fetch", "which", "env", "background", "check_background", "kill", "pipe"}
        self.assertEqual(names, expected)

    def test_all_actions_have_descriptions(self):
        for action in self.skill.manifest.actions:
            self.assertTrue(len(action.description) > 0, f"Action {action.name} has empty description")

    def test_all_actions_are_skill_actions(self):
        for action in self.skill.manifest.actions:
            self.assertIsInstance(action, SkillAction)

    def test_check_credentials(self):
        self.assertTrue(self.skill.check_credentials())


class TestShellSkillRun(unittest.TestCase):
    """Test the 'run' action with real subprocess calls."""

    def setUp(self):
        self.skill = ShellSkill()

    def test_echo_command(self):
        result = run(self.skill.execute("run", {"command": "echo hello"}))
        self.assertIsInstance(result, SkillResult)
        self.assertTrue(result.success)
        self.assertIn("hello", result.data["stdout"])
        self.assertEqual(result.data["exit_code"], 0)

    def test_pwd_command(self):
        result = run(self.skill.execute("run", {"command": "pwd"}))
        self.assertTrue(result.success)
        self.assertTrue(len(result.data["stdout"]) > 0)

    def test_failing_command(self):
        result = run(self.skill.execute("run", {"command": "false"}))
        self.assertFalse(result.success)
        self.assertNotEqual(result.data["exit_code"], 0)

    def test_empty_command(self):
        result = run(self.skill.execute("run", {"command": ""}))
        self.assertFalse(result.success)
        self.assertIn("required", result.message.lower())

    def test_missing_command_param(self):
        result = run(self.skill.execute("run", {}))
        self.assertFalse(result.success)

    def test_timeout_enforced(self):
        result = run(self.skill.execute("run", {
            "command": "sleep 10",
            "timeout": 1,
        }))
        self.assertFalse(result.success)
        self.assertIn("timed out", result.message.lower())
        self.assertTrue(result.data.get("timed_out", False))

    def test_max_timeout_cap(self):
        # Even if user asks for 9999s, it should be capped at MAX_TIMEOUT
        # We test this by making sure a short command runs fine with high timeout
        result = run(self.skill.execute("run", {
            "command": "echo test",
            "timeout": 9999,
        }))
        self.assertTrue(result.success)

    def test_stderr_capture(self):
        result = run(self.skill.execute("run", {"command": "echo err >&2"}))
        # Command still succeeds (exit 0) but stderr is captured
        self.assertTrue(result.success)
        self.assertIn("err", result.data["stderr"])

    def test_custom_cwd(self):
        result = run(self.skill.execute("run", {
            "command": "pwd",
            "cwd": "/tmp",
        }))
        self.assertTrue(result.success)
        self.assertIn("/tmp", result.data["stdout"])

    def test_pipe_in_command(self):
        result = run(self.skill.execute("run", {
            "command": "echo hello world | wc -w",
        }))
        self.assertTrue(result.success)
        self.assertIn("2", result.data["stdout"])

    def test_multiline_output(self):
        result = run(self.skill.execute("run", {
            "command": "echo -e 'line1\nline2\nline3'",
        }))
        self.assertTrue(result.success)
        lines = result.data["stdout"].strip().split("\n")
        self.assertEqual(len(lines), 3)

    def test_exit_code_preserved(self):
        result = run(self.skill.execute("run", {"command": "exit 42"}))
        self.assertFalse(result.success)
        self.assertEqual(result.data["exit_code"], 42)


class TestShellSkillBlocking(unittest.TestCase):
    """Test command safety blocking."""

    def setUp(self):
        self.skill = ShellSkill()

    def test_rm_rf_root_blocked(self):
        result = run(self.skill.execute("run", {"command": "rm -rf /"}))
        self.assertFalse(result.success)
        self.assertIn("blocked", result.message.lower())

    def test_rm_rf_wildcard_blocked(self):
        result = run(self.skill.execute("run", {"command": "rm -rf /*"}))
        self.assertFalse(result.success)

    def test_fork_bomb_blocked(self):
        result = run(self.skill.execute("run", {"command": ":(){:|:&};:"}))
        self.assertFalse(result.success)

    def test_shutdown_blocked(self):
        result = run(self.skill.execute("run", {"command": "shutdown now"}))
        self.assertFalse(result.success)

    def test_safe_commands_not_blocked(self):
        safe = ["ls", "echo test", "cat /etc/hostname", "date", "uname -a"]
        for cmd in safe:
            result = run(self.skill.execute("run", {"command": cmd}))
            # Should not be blocked (may fail for other reasons, but not safety)
            self.assertNotIn("blocked", result.message.lower(), f"Safe command blocked: {cmd}")


class TestShellSkillDirectoryRestriction(unittest.TestCase):
    """Test allowed_dirs restriction."""

    def test_allowed_dir_permits_access(self):
        skill = ShellSkill(allowed_dirs=["/tmp"])
        result = run(skill.execute("run", {"command": "ls", "cwd": "/tmp"}))
        self.assertTrue(result.success)

    def test_disallowed_dir_blocked(self):
        skill = ShellSkill(allowed_dirs=["/tmp"])
        result = run(skill.execute("run", {"command": "ls", "cwd": "/etc"}))
        self.assertFalse(result.success)
        self.assertIn("outside", result.message.lower())

    def test_no_restriction_by_default(self):
        skill = ShellSkill()
        result = run(skill.execute("run", {"command": "ls", "cwd": "/etc"}))
        # Should work fine with no restrictions
        self.assertTrue(result.success)


class TestShellSkillWhich(unittest.TestCase):
    """Test the 'which' action."""

    def setUp(self):
        self.skill = ShellSkill()

    def test_which_python(self):
        result = run(self.skill.execute("which", {"command": "python3"}))
        self.assertTrue(result.success)
        self.assertTrue(result.data["exists"])
        self.assertIn("python", result.data["path"])

    def test_which_nonexistent(self):
        result = run(self.skill.execute("which", {"command": "nonexistent_binary_xyz_123"}))
        self.assertFalse(result.success)
        self.assertFalse(result.data["exists"])

    def test_which_empty(self):
        result = run(self.skill.execute("which", {"command": ""}))
        self.assertFalse(result.success)


class TestShellSkillEnv(unittest.TestCase):
    """Test the 'env' action."""

    def setUp(self):
        self.skill = ShellSkill()

    def test_env_returns_info(self):
        result = run(self.skill.execute("env", {}))
        self.assertTrue(result.success)
        self.assertIn("cwd", result.data)
        self.assertIn("user", result.data)
        self.assertIn("home", result.data)
        self.assertIn("path", result.data)

    def test_env_with_requested_vars(self):
        result = run(self.skill.execute("env", {"vars": ["HOME", "PATH", "NONEXISTENT_VAR"]}))
        self.assertTrue(result.success)
        self.assertIn("requested", result.data)
        self.assertIn("HOME", result.data["requested"])
        self.assertEqual(result.data["requested"]["NONEXISTENT_VAR"], "<not set>")


class TestShellSkillFetch(unittest.TestCase):
    """Test the 'fetch' action (URL fetching via curl)."""

    def setUp(self):
        self.skill = ShellSkill()

    def test_fetch_empty_url(self):
        result = run(self.skill.execute("fetch", {"url": ""}))
        self.assertFalse(result.success)

    def test_fetch_adds_https(self):
        # We just verify the command is constructed correctly by checking
        # the result data structure (even if the actual fetch might fail)
        result = run(self.skill.execute("fetch", {"url": "example.com"}))
        # May succeed or fail depending on network, but should not crash
        self.assertIsInstance(result, SkillResult)


class TestShellSkillBackground(unittest.TestCase):
    """Test background process management."""

    def setUp(self):
        self.skill = ShellSkill()

    def test_start_background(self):
        result = run(self.skill.execute("background", {
            "command": "sleep 0.1",
            "label": "test_bg",
        }))
        self.assertTrue(result.success)
        self.assertEqual(result.data["label"], "test_bg")
        self.assertIn("pid", result.data)

    def test_check_background_list(self):
        run(self.skill.execute("background", {"command": "sleep 60", "label": "bg1"}))
        result = run(self.skill.execute("check_background", {}))
        self.assertTrue(result.success)
        self.assertIn("bg1", result.data["processes"])
        # Clean up
        run(self.skill.execute("kill", {"label": "bg1"}))

    def test_kill_background(self):
        run(self.skill.execute("background", {"command": "sleep 60", "label": "killme"}))
        result = run(self.skill.execute("kill", {"label": "killme"}))
        self.assertTrue(result.success)
        self.assertIn("killed", result.message.lower())

    def test_kill_nonexistent(self):
        result = run(self.skill.execute("kill", {"label": "nope"}))
        self.assertFalse(result.success)

    def test_duplicate_label_blocked(self):
        run(self.skill.execute("background", {"command": "sleep 60", "label": "dup"}))
        result = run(self.skill.execute("background", {"command": "sleep 60", "label": "dup"}))
        self.assertFalse(result.success)
        self.assertIn("already running", result.message.lower())
        # Clean up
        run(self.skill.execute("kill", {"label": "dup"}))

    def test_background_empty_command(self):
        result = run(self.skill.execute("background", {"command": "", "label": "x"}))
        self.assertFalse(result.success)


class TestShellSkillPipe(unittest.TestCase):
    """Test the 'pipe' action."""

    def setUp(self):
        self.skill = ShellSkill()

    def test_simple_pipe(self):
        result = run(self.skill.execute("pipe", {
            "commands": ["echo hello world", "wc -w"],
        }))
        self.assertTrue(result.success)
        self.assertIn("2", result.data["stdout"])

    def test_pipe_too_few_commands(self):
        result = run(self.skill.execute("pipe", {"commands": ["echo hi"]}))
        self.assertFalse(result.success)
        self.assertIn("2+", result.message)

    def test_pipe_empty(self):
        result = run(self.skill.execute("pipe", {"commands": []}))
        self.assertFalse(result.success)

    def test_three_stage_pipe(self):
        result = run(self.skill.execute("pipe", {
            "commands": ["echo -e 'a\nb\nc\nd'", "sort", "head -2"],
        }))
        self.assertTrue(result.success)


class TestShellSkillUnknownAction(unittest.TestCase):
    """Test handling of unknown actions."""

    def setUp(self):
        self.skill = ShellSkill()

    def test_unknown_action(self):
        result = run(self.skill.execute("nonexistent", {}))
        self.assertFalse(result.success)
        self.assertIn("Unknown action", result.message)


class TestShellSkillHelpers(unittest.TestCase):
    """Test static helper methods."""

    def test_truncate_short(self):
        data = b"hello"
        result = ShellSkill._truncate(data)
        self.assertEqual(result, "hello")

    def test_truncate_long(self):
        data = b"x" * 200_000
        result = ShellSkill._truncate(data, max_bytes=100)
        self.assertIn("truncated", result)
        self.assertTrue(len(result) < 200_000)

    def test_is_blocked_true(self):
        self.assertTrue(ShellSkill._is_blocked("rm -rf /"))
        self.assertTrue(ShellSkill._is_blocked("  SHUTDOWN  "))

    def test_is_blocked_false(self):
        self.assertFalse(ShellSkill._is_blocked("ls -la"))
        self.assertFalse(ShellSkill._is_blocked("echo test"))

    def test_is_subpath_true(self):
        self.assertTrue(ShellSkill._is_subpath(Path("/tmp/foo"), Path("/tmp")))

    def test_is_subpath_false(self):
        self.assertFalse(ShellSkill._is_subpath(Path("/etc/foo"), Path("/tmp")))


if __name__ == "__main__":
    unittest.main()
