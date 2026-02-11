"""
Comprehensive tests for the skill system — base classes, registry, plugin loader.

Tests cover:
- Skill base class: manifest, execute, credentials, stats
- SkillResult, SkillAction, SkillManifest data types
- SkillRegistry: install, uninstall, execute, list, credential management
- PluginLoader: registry loading, lazy loading, wiring hooks, caching
- Discovery: directory scanning, SKILL.md parsing
- Validation: credential checking, requirements checking
"""

import pytest
import json
import os
from unittest.mock import MagicMock, AsyncMock, patch
from pathlib import Path
from dataclasses import dataclass

from singularity.skills.base.types import SkillResult, SkillAction, SkillManifest
from singularity.skills.base.skill import Skill
from singularity.skills.base.registry import SkillRegistry
from singularity.skills.loader.loader import PluginLoader
from singularity.skills.loader.registry import SkillMetadata, WIRING_HOOKS


# ============================================================
# Concrete Skill Implementation for Testing
# ============================================================

class EchoSkill(Skill):
    """Concrete skill for testing — echoes input back."""

    @property
    def manifest(self):
        return SkillManifest(
            skill_id="echo",
            name="Echo",
            version="1.0.0",
            category="testing",
            description="Echoes input back",
            actions=[
                SkillAction(
                    name="echo",
                    description="Echo the input",
                    parameters={"text": {"type": "string", "description": "Text to echo"}},
                    estimated_cost=0.0,
                ),
                SkillAction(
                    name="reverse",
                    description="Reverse the input",
                    parameters={"text": {"type": "string"}},
                    estimated_cost=0.001,
                ),
            ],
            required_credentials=["ECHO_API_KEY"],
        )

    async def execute(self, action, params):
        if action == "echo":
            return SkillResult(success=True, message=params.get("text", ""), data={"echoed": True})
        elif action == "reverse":
            text = params.get("text", "")
            return SkillResult(success=True, message=text[::-1])
        return SkillResult(success=False, message=f"Unknown action: {action}")


class FreeSkill(Skill):
    """Skill with no credential requirements."""

    @property
    def manifest(self):
        return SkillManifest(
            skill_id="free",
            name="Free",
            version="1.0.0",
            category="testing",
            description="No credentials needed",
            actions=[
                SkillAction(name="hello", description="Say hello", parameters={}),
            ],
            required_credentials=[],
        )

    async def execute(self, action, params):
        return SkillResult(success=True, message="Hello!")


# ============================================================
# SkillResult Tests
# ============================================================

class TestSkillResult:
    """Test SkillResult data class."""

    def test_success_result(self):
        result = SkillResult(success=True, message="Done", data={"key": "value"})
        assert result.success is True
        assert result.message == "Done"
        assert result.data["key"] == "value"
        assert result.cost == 0
        assert result.revenue == 0

    def test_failure_result(self):
        result = SkillResult(success=False, message="Failed")
        assert result.success is False
        assert result.data == {}

    def test_result_with_costs(self):
        result = SkillResult(success=True, message="Paid", cost=0.05, revenue=1.00)
        assert result.cost == 0.05
        assert result.revenue == 1.00

    def test_result_with_asset(self):
        result = SkillResult(
            success=True, message="Created",
            asset_created={"type": "payment_link", "url": "https://stripe.com/pay"}
        )
        assert result.asset_created["type"] == "payment_link"


# ============================================================
# SkillAction Tests
# ============================================================

class TestSkillAction:
    """Test SkillAction data class."""

    def test_basic_action(self):
        action = SkillAction(name="send", description="Send message",
                             parameters={"to": {"type": "string"}})
        assert action.name == "send"
        assert action.estimated_cost == 0
        assert action.success_probability == 0.8

    def test_action_with_full_fields(self):
        action = SkillAction(
            name="deploy", description="Deploy app",
            parameters={"url": {"type": "string"}},
            estimated_cost=0.50,
            estimated_duration_seconds=120,
            success_probability=0.95,
        )
        assert action.estimated_cost == 0.50
        assert action.estimated_duration_seconds == 120
        assert action.success_probability == 0.95


# ============================================================
# SkillManifest Tests
# ============================================================

class TestSkillManifest:
    """Test SkillManifest data class."""

    def test_manifest_creation(self):
        manifest = SkillManifest(
            skill_id="test", name="Test", version="1.0.0",
            category="testing", description="Test skill",
            actions=[], required_credentials=["API_KEY"],
        )
        assert manifest.skill_id == "test"
        assert manifest.install_cost == 0
        assert manifest.author == "system"


# ============================================================
# Skill Base Class Tests
# ============================================================

class TestSkillBase:
    """Test Skill abstract base class methods."""

    def test_init_defaults(self):
        skill = EchoSkill()
        assert skill.credentials == {}
        assert skill.initialized is False
        assert skill._usage_count == 0
        assert skill._total_cost == 0
        assert skill._total_revenue == 0

    def test_init_with_credentials(self):
        skill = EchoSkill(credentials={"ECHO_API_KEY": "test-key"})
        assert skill.credentials["ECHO_API_KEY"] == "test-key"

    def test_get_actions(self):
        skill = EchoSkill()
        actions = skill.get_actions()
        assert len(actions) == 2
        assert actions[0].name == "echo"

    def test_get_action_exists(self):
        skill = EchoSkill()
        action = skill.get_action("echo")
        assert action is not None
        assert action.name == "echo"

    def test_get_action_missing(self):
        skill = EchoSkill()
        action = skill.get_action("nonexistent")
        assert action is None

    def test_estimate_cost(self):
        skill = EchoSkill()
        assert skill.estimate_cost("echo", {}) == 0.0
        assert skill.estimate_cost("reverse", {}) == 0.001
        assert skill.estimate_cost("nonexistent", {}) == 0

    def test_check_credentials_valid(self):
        skill = EchoSkill(credentials={"ECHO_API_KEY": "test-key"})
        assert skill.check_credentials() is True

    def test_check_credentials_missing(self):
        skill = EchoSkill()
        assert skill.check_credentials() is False

    def test_check_credentials_empty_string(self):
        skill = EchoSkill(credentials={"ECHO_API_KEY": ""})
        assert skill.check_credentials() is False

    def test_get_missing_credentials(self):
        skill = EchoSkill()
        missing = skill.get_missing_credentials()
        assert "ECHO_API_KEY" in missing

    def test_get_missing_credentials_all_set(self):
        skill = EchoSkill(credentials={"ECHO_API_KEY": "key"})
        missing = skill.get_missing_credentials()
        assert missing == []

    @pytest.mark.asyncio
    async def test_initialize_with_credentials(self):
        skill = EchoSkill(credentials={"ECHO_API_KEY": "key"})
        assert await skill.initialize() is True
        assert skill.initialized is True

    @pytest.mark.asyncio
    async def test_initialize_without_credentials(self):
        skill = EchoSkill()
        assert await skill.initialize() is False
        assert skill.initialized is False

    def test_record_usage(self):
        skill = EchoSkill()
        skill.record_usage(cost=0.01, revenue=0.05)
        skill.record_usage(cost=0.02, revenue=0.10)
        assert skill._usage_count == 2
        assert skill._total_cost == pytest.approx(0.03)
        assert skill._total_revenue == pytest.approx(0.15)

    def test_stats_property(self):
        skill = EchoSkill()
        skill.record_usage(cost=0.10, revenue=0.50)
        stats = skill.stats
        assert stats["usage_count"] == 1
        assert stats["total_cost"] == 0.10
        assert stats["total_revenue"] == 0.50
        assert stats["profit"] == pytest.approx(0.40)

    def test_to_dict(self):
        skill = EchoSkill()
        d = skill.to_dict()
        assert d["skill_id"] == "echo"
        assert d["name"] == "Echo"
        assert d["category"] == "testing"
        assert len(d["actions"]) == 2
        assert d["initialized"] is False

    @pytest.mark.asyncio
    async def test_execute_echo(self):
        skill = EchoSkill()
        result = await skill.execute("echo", {"text": "hello"})
        assert result.success is True
        assert result.message == "hello"

    @pytest.mark.asyncio
    async def test_execute_reverse(self):
        skill = EchoSkill()
        result = await skill.execute("reverse", {"text": "hello"})
        assert result.success is True
        assert result.message == "olleh"

    @pytest.mark.asyncio
    async def test_execute_unknown_action(self):
        skill = EchoSkill()
        result = await skill.execute("unknown", {})
        assert result.success is False


# ============================================================
# Skill with No Required Credentials Tests
# ============================================================

class TestFreeSkill:
    """Test skill with no credential requirements."""

    def test_check_credentials_always_true(self):
        """Skill with empty required_credentials should always pass."""
        skill = FreeSkill()
        assert skill.check_credentials() is True

    def test_get_missing_credentials_empty(self):
        """Should return empty list when no credentials needed."""
        skill = FreeSkill()
        assert skill.get_missing_credentials() == []


# ============================================================
# SkillRegistry Tests
# ============================================================

class TestSkillRegistry:
    """Test SkillRegistry class."""

    def test_init_empty(self):
        registry = SkillRegistry()
        assert len(registry.skills) == 0
        assert len(registry.credentials) == 0

    def test_install_by_class(self):
        registry = SkillRegistry()
        result = registry.install(FreeSkill)
        assert result is True
        assert "free" in registry.skills

    def test_install_by_class_with_credentials(self):
        registry = SkillRegistry()
        result = registry.install(EchoSkill, {"ECHO_API_KEY": "key"})
        assert result is True
        assert registry.skills["echo"].credentials["ECHO_API_KEY"] == "key"

    def test_uninstall(self):
        registry = SkillRegistry()
        registry.install(FreeSkill)
        assert "free" in registry.skills
        result = registry.uninstall("free")
        assert result is True
        assert "free" not in registry.skills

    def test_uninstall_nonexistent(self):
        registry = SkillRegistry()
        result = registry.uninstall("nonexistent")
        assert result is False

    def test_get_skill(self):
        registry = SkillRegistry()
        registry.install(FreeSkill)
        skill = registry.get("free")
        assert skill is not None
        assert skill.manifest.skill_id == "free"

    def test_get_nonexistent(self):
        registry = SkillRegistry()
        assert registry.get("nonexistent") is None

    def test_list_skills(self):
        registry = SkillRegistry()
        registry.install(FreeSkill)
        registry.install(EchoSkill, {"ECHO_API_KEY": "key"})
        skills = registry.list_skills()
        assert len(skills) == 2
        skill_ids = [s["skill_id"] for s in skills]
        assert "free" in skill_ids
        assert "echo" in skill_ids

    def test_list_all_actions(self):
        registry = SkillRegistry()
        registry.install(EchoSkill, {"ECHO_API_KEY": "key"})
        actions = registry.list_all_actions()
        assert len(actions) == 2
        action_names = [a["action"] for a in actions]
        assert "echo" in action_names
        assert "reverse" in action_names

    def test_set_credentials_propagates(self):
        registry = SkillRegistry()
        registry.install(EchoSkill)
        registry.set_credentials({"ECHO_API_KEY": "new-key"})
        assert registry.skills["echo"].credentials["ECHO_API_KEY"] == "new-key"

    @pytest.mark.asyncio
    async def test_execute_success(self):
        registry = SkillRegistry()
        registry.install(EchoSkill, {"ECHO_API_KEY": "key"})
        result = await registry.execute("echo", "echo", {"text": "test"})
        assert result.success is True
        assert result.message == "test"

    @pytest.mark.asyncio
    async def test_execute_missing_skill(self):
        registry = SkillRegistry()
        result = await registry.execute("nonexistent", "action", {})
        assert result.success is False
        assert "not found" in result.message

    @pytest.mark.asyncio
    async def test_execute_auto_initializes(self):
        registry = SkillRegistry()
        registry.install(EchoSkill, {"ECHO_API_KEY": "key"})
        skill = registry.get("echo")
        assert skill.initialized is False
        result = await registry.execute("echo", "echo", {"text": "init"})
        assert result.success is True
        assert skill.initialized is True

    @pytest.mark.asyncio
    async def test_execute_fails_without_credentials(self):
        registry = SkillRegistry()
        registry.install(EchoSkill)
        result = await registry.execute("echo", "echo", {"text": "test"})
        assert result.success is False
        assert "credentials" in result.message.lower()

    @pytest.mark.asyncio
    async def test_execute_records_usage(self):
        registry = SkillRegistry()
        registry.install(FreeSkill)
        await registry.execute("free", "hello", {})
        skill = registry.get("free")
        assert skill._usage_count == 1

    def test_get_skills_for_llm(self):
        registry = SkillRegistry()
        registry.install(EchoSkill, {"ECHO_API_KEY": "key"})
        text = registry.get_skills_for_llm()
        assert "echo" in text
        assert "Echo" in text
        assert "reverse" in text

    def test_set_agent(self):
        registry = SkillRegistry()
        agent = MagicMock()
        registry.set_agent(agent)
        assert registry._agent is agent


# ============================================================
# PluginLoader Tests
# ============================================================

class TestPluginLoader:
    """Test PluginLoader class."""

    def test_init_without_registry(self, tmp_path):
        """Should initialize without crashing when no registry exists."""
        loader = PluginLoader(registry_path=str(tmp_path / "nonexistent.json"))
        assert len(loader._registry) == 0

    def test_init_with_registry(self, tmp_path):
        """Should load registry from JSON file."""
        registry_data = {
            "version": "1.0",
            "skills": {
                "test_skill": {
                    "module": "test.module",
                    "class": "TestSkill",
                    "manifest": {
                        "name": "Test",
                        "version": "1.0.0",
                        "category": "testing",
                        "description": "A test skill",
                        "required_credentials": ["API_KEY"],
                        "actions": [
                            {"name": "do", "description": "Do it", "parameters": {}}
                        ],
                    },
                },
            },
        }
        registry_file = tmp_path / "registry.json"
        registry_file.write_text(json.dumps(registry_data))
        loader = PluginLoader(registry_path=str(registry_file))
        assert "test_skill" in loader._registry
        assert loader._registry["test_skill"].name == "Test"

    def test_list_available(self, tmp_path):
        """Should list all available skills."""
        registry_data = {
            "version": "1.0",
            "skills": {
                "s1": {"module": "m1", "class": "C1", "manifest": {
                    "name": "S1", "version": "1.0.0", "category": "cat1",
                    "description": "d1", "required_credentials": [],
                }},
                "s2": {"module": "m2", "class": "C2", "manifest": {
                    "name": "S2", "version": "1.0.0", "category": "cat2",
                    "description": "d2", "required_credentials": [],
                }},
            },
        }
        registry_file = tmp_path / "registry.json"
        registry_file.write_text(json.dumps(registry_data))
        loader = PluginLoader(registry_path=str(registry_file))
        available = loader.list_available()
        assert len(available) == 2

    def test_list_available_by_category(self, tmp_path):
        """Should filter by category."""
        registry_data = {
            "version": "1.0",
            "skills": {
                "s1": {"module": "m1", "class": "C1", "manifest": {
                    "name": "S1", "version": "1.0.0", "category": "social",
                    "description": "d1", "required_credentials": [],
                }},
                "s2": {"module": "m2", "class": "C2", "manifest": {
                    "name": "S2", "version": "1.0.0", "category": "finance",
                    "description": "d2", "required_credentials": [],
                }},
            },
        }
        registry_file = tmp_path / "registry.json"
        registry_file.write_text(json.dumps(registry_data))
        loader = PluginLoader(registry_path=str(registry_file))
        social = loader.list_available(category="social")
        assert len(social) == 1
        assert social[0].category == "social"

    def test_get_manifest(self, tmp_path):
        """Should return metadata for a skill."""
        registry_data = {
            "version": "1.0",
            "skills": {
                "test": {"module": "m", "class": "C", "manifest": {
                    "name": "Test", "version": "2.0.0", "category": "test",
                    "description": "Test", "required_credentials": [],
                }},
            },
        }
        registry_file = tmp_path / "registry.json"
        registry_file.write_text(json.dumps(registry_data))
        loader = PluginLoader(registry_path=str(registry_file))
        manifest = loader.get_manifest("test")
        assert manifest is not None
        assert manifest.version == "2.0.0"

    def test_get_manifest_missing(self, tmp_path):
        """Should return None for unknown skill."""
        loader = PluginLoader(registry_path=str(tmp_path / "empty.json"))
        assert loader.get_manifest("nonexistent") is None

    def test_is_loaded(self):
        """Should track loaded state."""
        loader = PluginLoader.__new__(PluginLoader)
        loader._registry = {}
        loader._loaded_skills = {}
        assert loader.is_loaded("test") is False
        loader._loaded_skills["test"] = MagicMock()
        assert loader.is_loaded("test") is True

    def test_unload(self):
        """Should unload a loaded skill."""
        loader = PluginLoader.__new__(PluginLoader)
        loader._registry = {}
        loader._loaded_skills = {"test": MagicMock()}
        assert loader.unload("test") is True
        assert "test" not in loader._loaded_skills

    def test_unload_not_loaded(self):
        """Should return False for unloaded skill."""
        loader = PluginLoader.__new__(PluginLoader)
        loader._registry = {}
        loader._loaded_skills = {}
        assert loader.unload("test") is False

    def test_list_loaded(self):
        """Should list loaded skill IDs."""
        loader = PluginLoader.__new__(PluginLoader)
        loader._registry = {}
        loader._loaded_skills = {"a": MagicMock(), "b": MagicMock()}
        assert set(loader.list_loaded()) == {"a", "b"}

    def test_register(self):
        """Should add metadata to registry."""
        loader = PluginLoader.__new__(PluginLoader)
        loader._registry = {}
        loader._loaded_skills = {}
        metadata = SkillMetadata(
            skill_id="new", module="new.mod", class_name="NewSkill",
            name="New", version="1.0.0", category="new",
            description="Newly registered", required_credentials=[],
        )
        loader.register(metadata)
        assert "new" in loader._registry

    def test_save_registry(self, tmp_path):
        """Should save registry to JSON file."""
        loader = PluginLoader.__new__(PluginLoader)
        loader._registry = {
            "test": SkillMetadata(
                skill_id="test", module="test.mod", class_name="TestSkill",
                name="Test", version="1.0.0", category="testing",
                description="Test", required_credentials=["KEY"],
            ),
        }
        loader._loaded_skills = {}
        loader._default_registry_path = tmp_path / "saved_registry.json"
        loader.save_registry()
        assert loader._default_registry_path.exists()
        data = json.loads(loader._default_registry_path.read_text())
        assert "test" in data["skills"]


# ============================================================
# Validation Mixin Tests
# ============================================================

class TestValidation:
    """Test credential and requirement validation."""

    def test_check_credentials_all_present(self, tmp_path):
        """Should return True when all credentials present."""
        registry_data = {
            "version": "1.0",
            "skills": {
                "test": {"module": "m", "class": "C", "manifest": {
                    "name": "T", "version": "1.0.0", "category": "t",
                    "description": "t", "required_credentials": ["API_KEY", "SECRET"],
                }},
            },
        }
        registry_file = tmp_path / "registry.json"
        registry_file.write_text(json.dumps(registry_data))
        loader = PluginLoader(registry_path=str(registry_file))
        assert loader.check_credentials("test", {"API_KEY": "key", "SECRET": "sec"}) is True

    def test_check_credentials_missing(self, tmp_path):
        """Should return False when credentials are missing."""
        registry_data = {
            "version": "1.0",
            "skills": {
                "test": {"module": "m", "class": "C", "manifest": {
                    "name": "T", "version": "1.0.0", "category": "t",
                    "description": "t", "required_credentials": ["API_KEY"],
                }},
            },
        }
        registry_file = tmp_path / "registry.json"
        registry_file.write_text(json.dumps(registry_data))
        loader = PluginLoader(registry_path=str(registry_file))
        assert loader.check_credentials("test", {}) is False

    def test_check_credentials_empty_value(self, tmp_path):
        """Empty string credential should fail check."""
        registry_data = {
            "version": "1.0",
            "skills": {
                "test": {"module": "m", "class": "C", "manifest": {
                    "name": "T", "version": "1.0.0", "category": "t",
                    "description": "t", "required_credentials": ["API_KEY"],
                }},
            },
        }
        registry_file = tmp_path / "registry.json"
        registry_file.write_text(json.dumps(registry_data))
        loader = PluginLoader(registry_path=str(registry_file))
        assert loader.check_credentials("test", {"API_KEY": ""}) is False

    def test_check_credentials_no_requirements(self, tmp_path):
        """Skill with no required credentials should always pass."""
        registry_data = {
            "version": "1.0",
            "skills": {
                "free": {"module": "m", "class": "C", "manifest": {
                    "name": "F", "version": "1.0.0", "category": "t",
                    "description": "t", "required_credentials": [],
                }},
            },
        }
        registry_file = tmp_path / "registry.json"
        registry_file.write_text(json.dumps(registry_data))
        loader = PluginLoader(registry_path=str(registry_file))
        assert loader.check_credentials("free", {}) is True

    def test_get_missing_credentials(self, tmp_path):
        """Should return list of missing credential names."""
        registry_data = {
            "version": "1.0",
            "skills": {
                "test": {"module": "m", "class": "C", "manifest": {
                    "name": "T", "version": "1.0.0", "category": "t",
                    "description": "t", "required_credentials": ["A", "B", "C"],
                }},
            },
        }
        registry_file = tmp_path / "registry.json"
        registry_file.write_text(json.dumps(registry_data))
        loader = PluginLoader(registry_path=str(registry_file))
        missing = loader.get_missing_credentials("test", {"A": "val"})
        assert "B" in missing
        assert "C" in missing
        assert "A" not in missing

    def test_check_credentials_unknown_skill(self, tmp_path):
        """Should return False for unknown skill."""
        loader = PluginLoader(registry_path=str(tmp_path / "empty.json"))
        assert loader.check_credentials("unknown", {"KEY": "val"}) is False


# ============================================================
# Discovery Mixin Tests
# ============================================================

class TestDiscovery:
    """Test skill discovery from directories."""

    def test_class_to_skill_id_conversion(self):
        """Should convert CamelCase class names to snake_case skill IDs."""
        loader = PluginLoader.__new__(PluginLoader)
        loader._registry = {}
        loader._loaded_skills = {}
        assert loader._class_to_skill_id("TwitterSkill") == "twitter"
        assert loader._class_to_skill_id("ContentCreationSkill") == "content_creation"
        assert loader._class_to_skill_id("MCPClientSkill") == "mcp_client"
        assert loader._class_to_skill_id("SimpleSkill") == "simple"

    def test_discover_python_files(self, tmp_path):
        """Should discover skills from Python files in directory."""
        skill_file = tmp_path / "my_skill.py"
        skill_file.write_text("""
from singularity.skills.base import Skill

class MyCustomSkill(Skill):
    pass
""")
        loader = PluginLoader.__new__(PluginLoader)
        loader._registry = {}
        loader._loaded_skills = {}
        loader.discover([str(tmp_path)])
        assert "my_custom" in loader._registry

    def test_discover_skips_underscore_files(self, tmp_path):
        """Should skip files starting with underscore."""
        (tmp_path / "_private.py").write_text("class PrivateSkill(Skill): pass")
        loader = PluginLoader.__new__(PluginLoader)
        loader._registry = {}
        loader._loaded_skills = {}
        loader.discover([str(tmp_path)])
        assert len(loader._registry) == 0


# ============================================================
# Wiring Hooks Tests
# ============================================================

class TestWiringHooks:
    """Test wiring hook functions."""

    def test_cognition_hooks_wiring(self):
        """Should wire cognition hooks into skill."""
        skill = MagicMock()
        agent = MagicMock()

        # Set up cognition mock
        agent.cognition.get_system_prompt = MagicMock(return_value="prompt")
        agent.cognition.set_system_prompt = MagicMock()
        agent.cognition.append_to_prompt = MagicMock()
        agent.cognition.get_available_models = MagicMock()
        agent.cognition.get_current_model = MagicMock()
        agent.cognition.switch_model = MagicMock()
        agent.cognition.record_training_example = MagicMock()
        agent.cognition.get_training_examples = MagicMock()
        agent.cognition.clear_training_examples = MagicMock()
        agent.cognition.export_training_data = MagicMock()
        agent.cognition.start_finetune = MagicMock()
        agent.cognition.check_finetune_status = MagicMock()
        agent.cognition.use_finetuned_model = MagicMock()

        WIRING_HOOKS["cognition_hooks"](skill, agent)
        skill.set_cognition_hooks.assert_called_once()

    def test_llm_wiring(self):
        """Should wire LLM instance into skill."""
        skill = MagicMock()
        agent = MagicMock()
        agent.cognition.llm = "mock_llm"
        agent.cognition.llm_type = "anthropic"
        agent.cognition.llm_model = "claude-sonnet-4-20250514"

        WIRING_HOOKS["llm"](skill, agent)
        skill.set_llm.assert_called_once_with("mock_llm", "anthropic", "claude-sonnet-4-20250514")

    def test_agent_info_wiring(self):
        """Should wire agent name into skill."""
        skill = MagicMock()
        agent = MagicMock()
        agent.name = "TestAgent"

        WIRING_HOOKS["agent_info"](skill, agent)
        skill.set_agent_info.assert_called_once_with("TestAgent")
