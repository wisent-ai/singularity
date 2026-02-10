"""
Tests for security and crash fixes.

Covers 4 bugs:
1. shell.py: _spawn() bypasses dangerous command checks that _bash() applies
2. crypto.py: private keys leaked in SkillResult.data, flowing to LLM context
3. browser.py: self._playwright not initialized in __init__, AttributeError on cleanup
4. content.py: self.model not initialized when no API key available
"""

import os
import sys

import pytest

# Ensure the project root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------------------------------
# 1. Shell security: _spawn() must apply same checks as _bash()
# ---------------------------------------------------------------------------

class TestShellSpawnSecurity:
    """_spawn() previously had NO dangerous command checks, unlike _bash().
    An agent could bypass all safety by using shell:spawn instead of shell:bash.
    """

    def _make_skill(self):
        from singularity.skills.shell import ShellSkill
        return ShellSkill()

    @pytest.mark.asyncio
    async def test_spawn_blocks_rm_rf(self):
        """_spawn() must block 'rm -rf /' just like _bash() does."""
        skill = self._make_skill()
        result = await skill.execute("spawn", {"command": "rm -rf /"})
        assert not result.success
        assert "Blocked" in result.message or "dangerous" in result.message.lower()

    @pytest.mark.asyncio
    async def test_spawn_blocks_mkfs(self):
        """_spawn() must block 'mkfs' commands."""
        skill = self._make_skill()
        result = await skill.execute("spawn", {"command": "mkfs /dev/sda1"})
        assert not result.success
        assert "Blocked" in result.message or "dangerous" in result.message.lower()

    @pytest.mark.asyncio
    async def test_spawn_blocks_fork_bomb(self):
        """_spawn() must block fork bomb."""
        skill = self._make_skill()
        result = await skill.execute("spawn", {"command": ":(){:|:&};:"})
        assert not result.success

    @pytest.mark.asyncio
    async def test_spawn_blocks_dd_zero(self):
        """_spawn() must block dd if=/dev/zero."""
        skill = self._make_skill()
        result = await skill.execute("spawn", {"command": "dd if=/dev/zero of=/dev/sda"})
        assert not result.success

    @pytest.mark.asyncio
    async def test_spawn_allows_safe_command(self):
        """_spawn() must allow safe commands."""
        skill = self._make_skill()
        result = await skill.execute("spawn", {"command": "echo hello"})
        assert result.success

    @pytest.mark.asyncio
    async def test_bash_still_blocks_dangerous(self):
        """Verify _bash() also still blocks dangerous commands (regression check)."""
        skill = self._make_skill()
        result = await skill.execute("bash", {"command": "rm -rf /"})
        assert not result.success
        assert "Blocked" in result.message or "dangerous" in result.message.lower()

    @pytest.mark.asyncio
    async def test_bash_allows_safe_command(self):
        """Verify _bash() still allows safe commands (regression check)."""
        skill = self._make_skill()
        result = await skill.execute("bash", {"command": "echo hello", "timeout": 5})
        assert result.success
        assert result.data["stdout"].strip() == "hello"

    def test_check_dangerous_command_method_exists(self):
        """The shared _check_dangerous_command method must exist."""
        skill = self._make_skill()
        assert hasattr(skill, "_check_dangerous_command")
        # Safe command returns None
        assert skill._check_dangerous_command("echo hello") is None
        # Dangerous command returns error string
        result = skill._check_dangerous_command("rm -rf /")
        assert result is not None
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# 2. Crypto: private key must NOT leak in SkillResult.data
# ---------------------------------------------------------------------------

class TestCryptoPrivateKeyLeak:
    """Private keys were previously included in SkillResult.data under '_private_key'.
    This data flows into LLM context (via recent_actions) and activity logs,
    exposing the key to third-party LLM providers and on-disk storage.
    """

    def test_create_wallet_no_private_key_in_result(self):
        """_create_wallet result must NOT contain private key."""
        try:
            from singularity.skills.crypto import CryptoSkill
        except ImportError:
            pytest.skip("web3/eth-account not installed")

        # Read the source code to verify the key is not in the result
        import inspect
        source = inspect.getsource(CryptoSkill._create_wallet)

        # The old code had: "_private_key": private_key
        # Verify that pattern is gone
        assert '"_private_key"' not in source, (
            "Private key should not be returned in SkillResult.data. "
            "It leaks into LLM context and activity logs."
        )
        assert "'_private_key'" not in source, (
            "Private key should not be returned in SkillResult.data. "
            "It leaks into LLM context and activity logs."
        )

    def test_private_key_still_stored_internally(self):
        """Private key must still be stored in self._wallets for actual use."""
        try:
            from singularity.skills.crypto import CryptoSkill
        except ImportError:
            pytest.skip("web3/eth-account not installed")

        import inspect
        source = inspect.getsource(CryptoSkill._create_wallet)

        # The key should still be stored in self._wallets
        assert "self._wallets" in source, (
            "Private key should still be stored in self._wallets for blockchain operations"
        )


# ---------------------------------------------------------------------------
# 3. Browser: self._playwright must be initialized in __init__
# ---------------------------------------------------------------------------

class TestBrowserPlaywrightInit:
    """self._playwright was not initialized in __init__, causing AttributeError
    when _close_browser() was called before _ensure_browser() had ever run.
    """

    def test_playwright_initialized_in_init(self):
        """BrowserSkill.__init__ must set self._playwright = None."""
        try:
            from singularity.skills.browser import BrowserSkill
        except ImportError:
            pytest.skip("Browser skill not available")

        skill = BrowserSkill()
        # Must exist and be None (not AttributeError)
        assert hasattr(skill, "_playwright"), (
            "self._playwright must be initialized in __init__ to prevent "
            "AttributeError when _close_browser() is called before _ensure_browser()"
        )
        assert skill._playwright is None

    def test_browser_initialized_in_init(self):
        """Verify browser, context, page are also None."""
        try:
            from singularity.skills.browser import BrowserSkill
        except ImportError:
            pytest.skip("Browser skill not available")

        skill = BrowserSkill()
        assert skill.browser is None
        assert skill.context is None
        assert skill.page is None

    @pytest.mark.asyncio
    async def test_close_browser_safe_when_never_opened(self):
        """_close_browser() must not crash when browser was never opened."""
        try:
            from singularity.skills.browser import BrowserSkill
        except ImportError:
            pytest.skip("Browser skill not available")

        skill = BrowserSkill()
        # This should not raise AttributeError
        await skill._close_browser()
        # After close, all should still be None
        assert skill._playwright is None
        assert skill.browser is None


# ---------------------------------------------------------------------------
# 4. Content: self.model must be initialized in all branches
# ---------------------------------------------------------------------------

class TestContentModelInit:
    """When no API key was available, _init_llm() set self.llm = None and
    self.llm_type = "none" but never initialized self.model, causing
    AttributeError if _generate() was called without set_llm() injection.
    """

    def test_model_initialized_without_api_key(self):
        """ContentCreationSkill must have self.model even with no API key."""
        try:
            from singularity.skills.content import ContentCreationSkill
        except ImportError:
            pytest.skip("Content skill not available")

        # Create without any API keys
        skill = ContentCreationSkill(credentials={})
        assert hasattr(skill, "model"), (
            "self.model must be initialized in _init_llm() else branch "
            "to prevent AttributeError when no API key is available"
        )

    def test_model_set_after_injection(self):
        """set_llm() should properly set model."""
        try:
            from singularity.skills.content import ContentCreationSkill
        except ImportError:
            pytest.skip("Content skill not available")

        skill = ContentCreationSkill(credentials={})
        skill.set_llm(None, "test_type", "test-model")
        assert skill.model == "test-model"
        assert skill.llm_type == "test_type"

    def test_check_credentials_false_without_api_key(self):
        """Without API key, check_credentials must return False."""
        try:
            from singularity.skills.content import ContentCreationSkill
        except ImportError:
            pytest.skip("Content skill not available")

        skill = ContentCreationSkill(credentials={})
        assert not skill.check_credentials()


# ---------------------------------------------------------------------------
# Smoke tests: verify key modules import without error
# ---------------------------------------------------------------------------

class TestSmokeImports:
    """Basic import checks to ensure the fixes don't break imports."""

    def test_import_shell_skill(self):
        from singularity.skills.shell import ShellSkill
        skill = ShellSkill()
        assert skill.manifest.skill_id == "shell"

    def test_import_cognition(self):
        from singularity.cognition import Action, CognitionEngine, Decision
        assert CognitionEngine is not None
        assert Action is not None
        assert Decision is not None

    def test_import_base_skill(self):
        from singularity.skills.base import Skill, SkillManifest, SkillResult
        assert Skill is not None
        assert SkillResult is not None
        assert SkillManifest is not None

    def test_import_request_skill(self):
        from singularity.skills.request import RequestSkill
        assert RequestSkill is not None

    def test_import_browser_skill(self):
        try:
            from singularity.skills.browser import BrowserSkill
            assert BrowserSkill is not None
        except ImportError:
            pytest.skip("Playwright not installed")

    def test_import_content_skill(self):
        try:
            from singularity.skills.content import ContentCreationSkill
            assert ContentCreationSkill is not None
        except ImportError:
            pytest.skip("Content skill dependencies not available")
