"""Tests for the PeerDiscoverySkill.

Uses importlib to load modules directly to avoid pulling in
the full singularity dependency tree (httpx, dotenv, etc.).
"""

import asyncio
import importlib.util
import json
import os
import shutil
import sys
import tempfile
import unittest

# Direct import to avoid the singularity package __init__ which
# transitively imports heavy dependencies.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SKILLS = os.path.join(_ROOT, "singularity", "skills")


def _import_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# Import base first, then peer_discovery
_base_mod = _import_module("singularity.skills.base", os.path.join(_SKILLS, "base.py"))
_pd_mod = _import_module(
    "singularity.skills.peer_discovery",
    os.path.join(_SKILLS, "peer_discovery.py"),
)

PeerDiscoverySkill = _pd_mod.PeerDiscoverySkill
SkillContext = _base_mod.SkillContext
SkillRegistry = _base_mod.SkillRegistry


def run_async(coro):
    """Helper to run async coroutines in sync tests."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class TestPeerDiscoverySkill(unittest.TestCase):
    """Test the PeerDiscoverySkill actions."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

        self.skill_a = PeerDiscoverySkill(credentials={"data_dir": self.tmpdir})
        self.skill_a._my_agent_id = "agent_alice"
        self.skill_a._my_name = "Alice"
        self.skill_a._my_ticker = "ALICE"
        self.skill_a.initialized = True

        self.skill_b = PeerDiscoverySkill(credentials={"data_dir": self.tmpdir})
        self.skill_b._my_agent_id = "agent_bob"
        self.skill_b._my_name = "Bob"
        self.skill_b._my_ticker = "BOB"
        self.skill_b.initialized = True

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_manifest(self):
        m = self.skill_a.manifest
        self.assertEqual(m.skill_id, "peer_discovery")
        self.assertEqual(m.category, "network")
        self.assertTrue(len(m.actions) >= 8)
        action_names = [a.name for a in m.actions]
        self.assertIn("register", action_names)
        self.assertIn("find_agents", action_names)
        self.assertIn("send_message", action_names)
        self.assertIn("check_inbox", action_names)
        self.assertIn("broadcast", action_names)
        self.assertIn("publish_knowledge", action_names)
        self.assertIn("query_knowledge", action_names)

    def test_register(self):
        result = run_async(self.skill_a.execute("register", {
            "specialty": "code analysis",
            "capabilities": [
                {"skill_id": "code", "action": "review", "description": "Review code for bugs"},
            ],
        }))
        self.assertTrue(result.success)
        self.assertIn("Alice", result.message)
        self.assertEqual(result.data["agent_id"], "agent_alice")
        self.assertEqual(result.data["capabilities_count"], 1)

    def test_who_is_online(self):
        run_async(self.skill_a.execute("register", {}))
        run_async(self.skill_b.execute("register", {}))

        result = run_async(self.skill_a.execute("who_is_online", {}))
        self.assertTrue(result.success)
        agents = result.data["agents"]
        self.assertEqual(len(agents), 2)
        names = {a["name"] for a in agents}
        self.assertIn("Alice", names)
        self.assertIn("Bob", names)

    def test_find_agents_by_type(self):
        run_async(self.skill_a.execute("register", {"agent_type": "coder"}))
        run_async(self.skill_b.execute("register", {"agent_type": "writer"}))

        result = run_async(self.skill_b.execute("find_agents", {"agent_type": "coder"}))
        self.assertTrue(result.success)
        agents = result.data["agents"]
        self.assertEqual(len(agents), 1)
        self.assertEqual(agents[0]["name"], "Alice")

    def test_find_agents_by_task(self):
        run_async(self.skill_a.execute("register", {
            "capabilities": [
                {"skill_id": "code", "action": "review",
                 "description": "Review code for security bugs"},
            ],
        }))
        run_async(self.skill_b.execute("register", {
            "capabilities": [
                {"skill_id": "write", "action": "blog",
                 "description": "Write blog posts and articles"},
            ],
        }))

        result = run_async(self.skill_a.execute("find_agents", {
            "task": "write blog articles",
        }))
        self.assertTrue(result.success)
        agents = result.data["agents"]
        self.assertTrue(any(a["name"] == "Bob" for a in agents))

    def test_send_and_receive_message(self):
        result = run_async(self.skill_a.execute("send_message", {
            "to_agent": "agent_bob",
            "subject": "Hello Bob!",
            "body": {"text": "Want to collaborate?"},
        }))
        self.assertTrue(result.success)
        self.assertIn("message_id", result.data)

        result = run_async(self.skill_b.execute("check_inbox", {}))
        self.assertTrue(result.success)
        self.assertEqual(result.data["count"], 1)
        msg = result.data["messages"][0]
        self.assertEqual(msg["subject"], "Hello Bob!")
        self.assertEqual(msg["from_agent"], "agent_alice")

    def test_messages_drained_after_read(self):
        run_async(self.skill_a.execute("send_message", {
            "to_agent": "agent_bob",
            "subject": "Test",
        }))
        result = run_async(self.skill_b.execute("check_inbox", {}))
        self.assertEqual(result.data["count"], 1)
        result = run_async(self.skill_b.execute("check_inbox", {}))
        self.assertEqual(result.data["count"], 0)

    def test_filter_messages_by_sender(self):
        run_async(self.skill_a.execute("send_message", {
            "to_agent": "agent_bob",
            "subject": "From Alice",
        }))

        skill_c = PeerDiscoverySkill(credentials={"data_dir": self.tmpdir})
        skill_c._my_agent_id = "agent_charlie"
        skill_c._my_name = "Charlie"
        skill_c._my_ticker = "CHAR"
        skill_c.initialized = True
        run_async(skill_c.execute("send_message", {
            "to_agent": "agent_bob",
            "subject": "From Charlie",
        }))

        result = run_async(self.skill_b.execute("check_inbox", {
            "from_agent": "agent_alice",
        }))
        self.assertEqual(result.data["count"], 1)
        self.assertEqual(result.data["messages"][0]["subject"], "From Alice")

    def test_broadcast(self):
        run_async(self.skill_a.execute("register", {}))
        run_async(self.skill_b.execute("register", {}))

        skill_c = PeerDiscoverySkill(credentials={"data_dir": self.tmpdir})
        skill_c._my_agent_id = "agent_charlie"
        skill_c._my_name = "Charlie"
        skill_c._my_ticker = "CHAR"
        skill_c.initialized = True
        run_async(skill_c.execute("register", {}))

        result = run_async(self.skill_a.execute("broadcast", {
            "subject": "System update",
            "body": {"info": "New features available"},
        }))
        self.assertTrue(result.success)
        self.assertEqual(result.data["sent_count"], 2)

        result = run_async(self.skill_b.execute("check_inbox", {}))
        self.assertEqual(result.data["count"], 1)
        self.assertEqual(result.data["messages"][0]["subject"], "System update")

    def test_publish_and_query_knowledge(self):
        result = run_async(self.skill_a.execute("publish_knowledge", {
            "content": "Dynamic pricing increases margins by 15% in competitive markets",
            "category": "optimization",
            "confidence": 0.8,
            "tags": ["pricing", "revenue"],
        }))
        self.assertTrue(result.success)
        self.assertIn("entry_id", result.data)

        result = run_async(self.skill_b.execute("query_knowledge", {
            "category": "optimization",
        }))
        self.assertTrue(result.success)
        self.assertEqual(result.data["total"], 1)
        self.assertIn("pricing", result.data["entries"][0]["content"].lower())

    def test_knowledge_text_search(self):
        run_async(self.skill_a.execute("publish_knowledge", {
            "content": "Python is excellent for data science applications",
            "confidence": 0.7,
        }))
        run_async(self.skill_a.execute("publish_knowledge", {
            "content": "Rust is fast for systems programming",
            "confidence": 0.7,
        }))

        result = run_async(self.skill_b.execute("query_knowledge", {
            "search": "Python",
        }))
        self.assertTrue(result.success)
        self.assertEqual(result.data["total"], 1)

    def test_knowledge_deduplication(self):
        content = "Caching reduces costs by 40%"
        run_async(self.skill_a.execute("publish_knowledge", {
            "content": content,
            "confidence": 0.5,
        }))
        run_async(self.skill_b.execute("publish_knowledge", {
            "content": content,
            "confidence": 0.3,
        }))

        result = run_async(self.skill_a.execute("query_knowledge", {}))
        self.assertEqual(result.data["total"], 1)

    def test_knowledge_confidence_filter(self):
        run_async(self.skill_a.execute("publish_knowledge", {
            "content": "Low confidence fact",
            "confidence": 0.1,
        }))
        run_async(self.skill_a.execute("publish_knowledge", {
            "content": "High confidence fact",
            "confidence": 0.9,
        }))

        result = run_async(self.skill_b.execute("query_knowledge", {
            "min_confidence": 0.5,
        }))
        self.assertEqual(result.data["total"], 1)
        self.assertIn("High confidence", result.data["entries"][0]["content"])

    def test_send_message_validation(self):
        result = run_async(self.skill_a.execute("send_message", {
            "subject": "No recipient",
        }))
        self.assertFalse(result.success)
        self.assertIn("to_agent", result.message)

        result = run_async(self.skill_a.execute("send_message", {
            "to_agent": "agent_bob",
        }))
        self.assertFalse(result.success)
        self.assertIn("subject", result.message)

    def test_unknown_action(self):
        result = run_async(self.skill_a.execute("nonexistent", {}))
        self.assertFalse(result.success)
        self.assertIn("Unknown action", result.message)

    def test_publish_knowledge_validation(self):
        result = run_async(self.skill_a.execute("publish_knowledge", {}))
        self.assertFalse(result.success)
        self.assertIn("content is required", result.message)

    def test_broadcast_validation(self):
        result = run_async(self.skill_a.execute("broadcast", {}))
        self.assertFalse(result.success)
        self.assertIn("subject is required", result.message)


class TestPeerDiscoveryWithContext(unittest.TestCase):
    """Test PeerDiscoverySkill with a SkillContext."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.skill = PeerDiscoverySkill(credentials={"data_dir": self.tmpdir})
        registry = SkillRegistry()
        context = SkillContext(
            registry=registry,
            agent_name="TestAgent",
            agent_ticker="TEST",
            get_state_fn=lambda: {"instance_id": "agent_test_123"},
        )
        self.skill.set_context(context)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_initialize_from_context(self):
        result = run_async(self.skill.initialize())
        self.assertTrue(result)
        self.assertEqual(self.skill._my_name, "TestAgent")
        self.assertEqual(self.skill._my_ticker, "TEST")
        self.assertEqual(self.skill._my_agent_id, "agent_test_123")


if __name__ == "__main__":
    unittest.main()
