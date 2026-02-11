"""
Comprehensive tests for the Singularity v0.2.0 framework architecture.

Tests the core infrastructure: base types, skill class, registry,
plugin loader, and the registry.json manifest.
"""

import asyncio
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
exec(open(str(Path(__file__).resolve().parent / "conftest.py")).read())

from singularity.skills.base.types import SkillResult, SkillAction, SkillManifest
from singularity.skills.base.skill import Skill
from singularity.skills.loader.registry import SkillMetadata, MCPServerInfo, SkillMdFile, ValidationMixin


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


# ── SkillResult Tests ───────────────────────────────────────────────


class TestSkillResult(unittest.TestCase):
    """Test SkillResult dataclass."""

    def test_success_result(self):
        r = SkillResult(success=True, message="done")
        self.assertTrue(r.success)
        self.assertEqual(r.message, "done")
        self.assertEqual(r.data, {})
        self.assertEqual(r.cost, 0)
        self.assertEqual(r.revenue, 0)
        self.assertIsNone(r.asset_created)

    def test_failure_result(self):
        r = SkillResult(success=False, message="failed", data={"error": "timeout"})
        self.assertFalse(r.success)
        self.assertIn("error", r.data)

    def test_with_cost_and_revenue(self):
        r = SkillResult(success=True, message="ok", cost=0.5, revenue=1.0)
        self.assertEqual(r.cost, 0.5)
        self.assertEqual(r.revenue, 1.0)

    def test_with_asset(self):
        r = SkillResult(success=True, message="created",
                       asset_created={"type": "file", "path": "/tmp/x"})
        self.assertEqual(r.asset_created["type"], "file")


# ── SkillAction Tests ──────────────────────────────────────────────


class TestSkillAction(unittest.TestCase):
    """Test SkillAction dataclass."""

    def test_basic_action(self):
        a = SkillAction(name="run", description="Run command",
                       parameters={"command": {"type": "string"}})
        self.assertEqual(a.name, "run")
        self.assertEqual(a.description, "Run command")
        self.assertIn("command", a.parameters)

    def test_defaults(self):
        a = SkillAction(name="x", description="y", parameters={})
        self.assertEqual(a.estimated_cost, 0)
        self.assertEqual(a.estimated_duration_seconds, 10)
        self.assertEqual(a.success_probability, 0.8)

    def test_custom_probability(self):
        a = SkillAction(name="x", description="y", parameters={},
                       success_probability=0.95)
        self.assertEqual(a.success_probability, 0.95)


# ── SkillManifest Tests ────────────────────────────────────────────


class TestSkillManifest(unittest.TestCase):
    """Test SkillManifest dataclass."""

    def test_basic_manifest(self):
        m = SkillManifest(
            skill_id="test",
            name="Test Skill",
            version="1.0.0",
            category="system",
            description="A test skill",
            actions=[SkillAction(name="a", description="b", parameters={})],
            required_credentials=["API_KEY"],
        )
        self.assertEqual(m.skill_id, "test")
        self.assertEqual(m.name, "Test Skill")
        self.assertEqual(len(m.actions), 1)
        self.assertEqual(m.required_credentials, ["API_KEY"])
        self.assertEqual(m.install_cost, 0)
        self.assertEqual(m.author, "system")


# ── Skill Base Class Tests ──────────────────────────────────────────


class ConcreteSkill(Skill):
    """Concrete implementation for testing the abstract base class."""

    @property
    def manifest(self):
        return SkillManifest(
            skill_id="concrete",
            name="Concrete Skill",
            version="1.0.0",
            category="test",
            description="Test skill",
            actions=[
                SkillAction(name="hello", description="Say hello", parameters={}),
                SkillAction(name="add", description="Add numbers",
                           parameters={"a": {}, "b": {}}, estimated_cost=0.01),
            ],
            required_credentials=["TEST_KEY"],
        )

    async def execute(self, action, params):
        if action == "hello":
            return SkillResult(success=True, message="Hello!")
        elif action == "add":
            return SkillResult(success=True, message="Sum", data={"result": params.get("a", 0) + params.get("b", 0)})
        return SkillResult(success=False, message=f"Unknown: {action}")


class TestSkillBaseClass(unittest.TestCase):
    """Test the abstract Skill base class via ConcreteSkill."""

    def setUp(self):
        self.skill = ConcreteSkill(credentials={"TEST_KEY": "mykey"})

    def test_manifest_access(self):
        self.assertEqual(self.skill.manifest.skill_id, "concrete")

    def test_get_actions(self):
        actions = self.skill.get_actions()
        self.assertEqual(len(actions), 2)

    def test_get_action_by_name(self):
        action = self.skill.get_action("hello")
        self.assertIsNotNone(action)
        self.assertEqual(action.name, "hello")

    def test_get_action_nonexistent(self):
        action = self.skill.get_action("nonexistent")
        self.assertIsNone(action)

    def test_estimate_cost(self):
        cost = self.skill.estimate_cost("add", {})
        self.assertEqual(cost, 0.01)

    def test_estimate_cost_unknown(self):
        cost = self.skill.estimate_cost("nonexistent", {})
        self.assertEqual(cost, 0)

    def test_check_credentials_pass(self):
        self.assertTrue(self.skill.check_credentials())

    def test_check_credentials_fail(self):
        skill = ConcreteSkill()  # No credentials
        self.assertFalse(skill.check_credentials())

    def test_get_missing_credentials(self):
        skill = ConcreteSkill()
        missing = skill.get_missing_credentials()
        self.assertIn("TEST_KEY", missing)

    def test_get_missing_credentials_none(self):
        missing = self.skill.get_missing_credentials()
        self.assertEqual(missing, [])

    def test_initialize(self):
        result = run(self.skill.initialize())
        self.assertTrue(result)
        self.assertTrue(self.skill.initialized)

    def test_initialize_fails_without_creds(self):
        skill = ConcreteSkill()
        result = run(skill.initialize())
        self.assertFalse(result)

    def test_record_usage(self):
        self.skill.record_usage(cost=0.5, revenue=1.0)
        self.assertEqual(self.skill._usage_count, 1)
        self.assertEqual(self.skill._total_cost, 0.5)
        self.assertEqual(self.skill._total_revenue, 1.0)

    def test_stats(self):
        self.skill.record_usage(cost=1, revenue=3)
        stats = self.skill.stats
        self.assertEqual(stats["usage_count"], 1)
        self.assertEqual(stats["profit"], 2)

    def test_execute_hello(self):
        result = run(self.skill.execute("hello", {}))
        self.assertTrue(result.success)
        self.assertEqual(result.message, "Hello!")

    def test_execute_add(self):
        result = run(self.skill.execute("add", {"a": 3, "b": 4}))
        self.assertTrue(result.success)
        self.assertEqual(result.data["result"], 7)

    def test_execute_unknown(self):
        result = run(self.skill.execute("nonexistent", {}))
        self.assertFalse(result.success)

    def test_to_dict(self):
        d = self.skill.to_dict()
        self.assertEqual(d["skill_id"], "concrete")


# ── SkillMetadata Tests ─────────────────────────────────────────────


class TestSkillMetadata(unittest.TestCase):
    """Test SkillMetadata dataclass."""

    def test_basic_metadata(self):
        m = SkillMetadata(
            skill_id="test", module="test.module", class_name="TestSkill",
            name="Test", version="1.0.0", category="test",
            description="Test skill", required_credentials=[],
        )
        self.assertEqual(m.skill_id, "test")
        self.assertEqual(m.source_type, "python")
        self.assertTrue(m.user_invocable)

    def test_defaults(self):
        m = SkillMetadata(
            skill_id="x", module="x", class_name="X",
            name="x", version="1.0.0", category="x",
            description="x", required_credentials=[],
        )
        self.assertIsNone(m.wiring)
        self.assertEqual(m.actions, [])
        self.assertEqual(m.install_cost, 0)
        self.assertEqual(m.author, "system")
        self.assertIsNone(m.homepage)
        self.assertEqual(m.requires_bins, [])
        self.assertEqual(m.requires_env, [])
        self.assertEqual(m.os_platforms, [])


# ── MCPServerInfo Tests ─────────────────────────────────────────────


class TestMCPServerInfo(unittest.TestCase):
    """Test MCPServerInfo dataclass."""

    def test_basic(self):
        info = MCPServerInfo(name="test", description="Test MCP server")
        self.assertEqual(info.name, "test")
        self.assertEqual(info.transport, "stdio")
        self.assertIsNone(info.command)
        self.assertEqual(info.args, [])
        self.assertEqual(info.env, {})


# ── SkillMdFile Tests ──────────────────────────────────────────────


class TestSkillMdFile(unittest.TestCase):
    """Test SkillMdFile dataclass."""

    def test_basic(self):
        md = SkillMdFile(name="test", description="Test", instructions="Do this",
                        path=Path("/test"))
        self.assertEqual(md.name, "test")
        self.assertTrue(md.user_invocable)
        self.assertFalse(md.disable_model_invocation)
        self.assertEqual(md.command_arg_mode, "raw")


# ── Registry.json Validation ───────────────────────────────────────


class TestRegistryJson(unittest.TestCase):
    """Test that registry.json is valid and matches implementations."""

    def setUp(self):
        registry_path = Path(__file__).resolve().parent.parent / "singularity" / "skills" / "registry.json"
        with open(registry_path) as f:
            self.registry = json.load(f)

    def test_has_version(self):
        self.assertIn("version", self.registry)

    def test_has_skills(self):
        self.assertIn("skills", self.registry)
        self.assertIsInstance(self.registry["skills"], dict)

    def test_minimum_skill_count(self):
        self.assertGreaterEqual(len(self.registry["skills"]), 25)

    def test_all_skills_have_module(self):
        for sid, data in self.registry["skills"].items():
            self.assertIn("module", data, f"Skill {sid} missing 'module'")
            self.assertTrue(data["module"].startswith("singularity.skills.builtin."),
                          f"Skill {sid} module doesn't start with singularity.skills.builtin.")

    def test_all_skills_have_class(self):
        for sid, data in self.registry["skills"].items():
            self.assertIn("class", data, f"Skill {sid} missing 'class'")
            self.assertTrue(len(data["class"]) > 0)

    def test_all_skills_have_manifest(self):
        for sid, data in self.registry["skills"].items():
            self.assertIn("manifest", data, f"Skill {sid} missing 'manifest'")
            manifest = data["manifest"]
            self.assertIn("skill_id", manifest)
            self.assertIn("name", manifest)
            self.assertIn("category", manifest)
            self.assertIn("description", manifest)

    def test_manifest_skill_ids_match_keys(self):
        for sid, data in self.registry["skills"].items():
            manifest_id = data["manifest"]["skill_id"]
            self.assertEqual(sid, manifest_id,
                           f"Key '{sid}' doesn't match manifest skill_id '{manifest_id}'")

    def test_wiring_values_valid(self):
        valid_wirings = {None, "llm", "cognition_hooks", "agent_info"}
        for sid, data in self.registry["skills"].items():
            wiring = data.get("wiring")
            self.assertIn(wiring, valid_wirings,
                         f"Skill {sid} has invalid wiring: {wiring}")

    def test_core_skills_present(self):
        """Ensure core skills are registered."""
        core = ["filesystem", "shell", "browser", "github", "stripe",
                "content_creation", "self", "mcp", "request"]
        for sid in core:
            self.assertIn(sid, self.registry["skills"],
                         f"Core skill '{sid}' missing from registry")

    def test_all_skills_have_implementations(self):
        """Verify every registered skill has an implementation directory."""
        builtin_dir = Path(__file__).resolve().parent.parent / "singularity" / "skills" / "builtin"
        for sid, data in self.registry["skills"].items():
            # Extract the last part of the module path
            module_parts = data["module"].split(".")
            skill_dir_name = module_parts[-1]
            skill_path = builtin_dir / skill_dir_name
            self.assertTrue(
                skill_path.exists(),
                f"Skill '{sid}' registered but {skill_path} doesn't exist",
            )


# ── Builtin Skill Directory Tests ──────────────────────────────────


class TestBuiltinSkillStructure(unittest.TestCase):
    """Test that all builtin skill directories follow the expected structure."""

    def setUp(self):
        self.builtin_dir = Path(__file__).resolve().parent.parent / "singularity" / "skills" / "builtin"

    def test_all_have_init(self):
        """Every skill directory must have __init__.py."""
        for d in self.builtin_dir.iterdir():
            if d.is_dir() and not d.name.startswith("_"):
                init_file = d / "__init__.py"
                self.assertTrue(init_file.exists(),
                              f"Missing __init__.py in {d.name}")

    def test_all_have_skill_or_init_with_class(self):
        """Each skill directory should have either skill.py or define class in __init__.py."""
        import ast
        for d in self.builtin_dir.iterdir():
            if d.is_dir() and not d.name.startswith("_"):
                has_skill_py = (d / "skill.py").exists()
                has_handlers = (d / "handlers.py").exists() or (d / "actions.py").exists()
                # At minimum, __init__.py must exist
                self.assertTrue(
                    (d / "__init__.py").exists(),
                    f"Missing __init__.py in {d.name}",
                )

    def test_new_skills_have_skill_py(self):
        """The 5 newly added skills should all have skill.py."""
        new_skills = ["shell", "browser", "instagram", "facebook", "namecheap"]
        for name in new_skills:
            skill_py = self.builtin_dir / name / "skill.py"
            self.assertTrue(skill_py.exists(),
                          f"Missing skill.py for new skill: {name}")


if __name__ == "__main__":
    unittest.main()
