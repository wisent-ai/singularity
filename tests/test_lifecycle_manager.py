#!/usr/bin/env python3
"""
Tests for LifecycleManagerSkill — graceful shutdown, signal handling, health probes.

Uses unittest (no pytest dependency required).
"""

import asyncio
import json
import os
import signal
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

# Stub heavy dependencies before importing the skill
import types

# Comprehensive stubbing of all external dependencies used by singularity
_STUB_MODULES = [
    "dotenv", "httpx", "anthropic", "openai", "fastapi", "pydantic",
    "uvicorn", "stripe", "boto3", "aiohttp", "aiohttp.web",
    "fastapi.middleware", "fastapi.middleware.cors",
    "pydantic",
]

for mod_name in _STUB_MODULES:
    if mod_name not in sys.modules:
        mod = types.ModuleType(mod_name)
        # Add commonly accessed attributes
        if mod_name == "dotenv":
            mod.load_dotenv = lambda *a, **kw: None
        if mod_name == "fastapi":
            mod.FastAPI = type("FastAPI", (), {})
            mod.HTTPException = type("HTTPException", (Exception,), {})
            mod.BackgroundTasks = type("BackgroundTasks", (), {})
            mod.Depends = lambda x: x
            mod.Header = lambda **kw: None
        if mod_name == "fastapi.middleware.cors":
            mod.CORSMiddleware = type("CORSMiddleware", (), {})
        if mod_name == "pydantic":
            mod.BaseModel = type("BaseModel", (), {})
            mod.Field = lambda **kw: None
        sys.modules[mod_name] = mod

# Ensure singularity package is importable — use direct file import to avoid __init__.py
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import skill module directly to avoid singularity/__init__.py chain
import importlib.util
_base_spec = importlib.util.spec_from_file_location(
    "singularity.skills.base",
    str(Path(__file__).parent.parent / "singularity" / "skills" / "base.py"),
)
_base_mod = importlib.util.module_from_spec(_base_spec)
sys.modules["singularity.skills.base"] = _base_mod
_base_spec.loader.exec_module(_base_mod)

_lm_spec = importlib.util.spec_from_file_location(
    "singularity.skills.lifecycle_manager",
    str(Path(__file__).parent.parent / "singularity" / "skills" / "lifecycle_manager.py"),
)
_lm_mod = importlib.util.module_from_spec(_lm_spec)
sys.modules["singularity.skills.lifecycle_manager"] = _lm_mod
_lm_spec.loader.exec_module(_lm_mod)

from singularity.skills.lifecycle_manager import (
    LifecycleManagerSkill,
    LifecyclePhase,
    ShutdownHook,
)


def run_async(coro):
    """Helper to run async code in sync tests."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class TestShutdownHook(unittest.TestCase):
    """Test ShutdownHook dataclass."""

    def test_create_hook(self):
        hook = ShutdownHook(
            hook_id="test_hook",
            name="Test Hook",
            priority=50,
            description="A test hook",
        )
        self.assertEqual(hook.hook_id, "test_hook")
        self.assertEqual(hook.name, "Test Hook")
        self.assertEqual(hook.priority, 50)
        self.assertIsNone(hook.callback)
        self.assertIsNotNone(hook.registered_at)

    def test_hook_to_dict(self):
        hook = ShutdownHook(
            hook_id="h1",
            name="Hook One",
            priority=100,
            timeout_seconds=5.0,
            description="Cleanup databases",
        )
        d = hook.to_dict()
        self.assertEqual(d["hook_id"], "h1")
        self.assertEqual(d["name"], "Hook One")
        self.assertEqual(d["priority"], 100)
        self.assertEqual(d["timeout_seconds"], 5.0)
        self.assertFalse(d["has_callback"])

    def test_hook_with_callback(self):
        cb = lambda: None
        hook = ShutdownHook(hook_id="h2", name="H2", callback=cb)
        d = hook.to_dict()
        self.assertTrue(d["has_callback"])


class TestLifecycleManagerInit(unittest.TestCase):
    """Test LifecycleManagerSkill initialization."""

    def test_default_init(self):
        skill = LifecycleManagerSkill()
        self.assertEqual(skill._phase, LifecyclePhase.INITIALIZING)
        self.assertEqual(len(skill._hooks), 0)
        self.assertFalse(skill._shutdown_requested)
        self.assertFalse(skill._signals_installed)
        self.assertFalse(skill._ready)
        self.assertEqual(skill._active_tasks, 0)

    def test_manifest(self):
        skill = LifecycleManagerSkill()
        m = skill.manifest
        self.assertEqual(m.skill_id, "lifecycle_manager")
        self.assertEqual(m.version, "1.0.0")
        self.assertEqual(m.category, "infrastructure")
        self.assertEqual(len(m.actions), 7)
        action_names = [a.name for a in m.actions]
        self.assertIn("register_hook", action_names)
        self.assertIn("initiate_shutdown", action_names)
        self.assertIn("get_health", action_names)
        self.assertIn("drain", action_names)

    def test_custom_timeouts_from_credentials(self):
        skill = LifecycleManagerSkill(credentials={
            "SHUTDOWN_TIMEOUT": "45",
            "DRAIN_TIMEOUT": "90",
        })
        self.assertEqual(skill._shutdown_timeout, 45.0)
        self.assertEqual(skill._drain_timeout, 90.0)

    def test_default_timeouts(self):
        skill = LifecycleManagerSkill()
        self.assertEqual(skill._shutdown_timeout, 30.0)
        self.assertEqual(skill._drain_timeout, 60.0)


class TestLifecyclePhases(unittest.TestCase):
    """Test lifecycle phase transitions."""

    def test_mark_starting(self):
        skill = LifecycleManagerSkill()
        skill.mark_starting()
        self.assertEqual(skill._phase, LifecyclePhase.STARTING)
        self.assertIsNotNone(skill._start_time)

    def test_mark_started(self):
        skill = LifecycleManagerSkill()
        skill.mark_started()
        self.assertEqual(skill._phase, LifecyclePhase.RUNNING)
        self.assertTrue(skill._ready)

    def test_uptime_tracking(self):
        skill = LifecycleManagerSkill()
        self.assertEqual(skill.uptime_seconds, 0.0)
        skill.mark_started()
        time.sleep(0.05)
        self.assertGreater(skill.uptime_seconds, 0.01)


class TestRegisterHook(unittest.TestCase):
    """Test hook registration via execute()."""

    def test_register_hook_success(self):
        skill = LifecycleManagerSkill()
        skill.mark_started()
        result = run_async(skill.execute("register_hook", {
            "hook_id": "flush_db",
            "name": "Flush Database",
            "priority": 50,
            "timeout_seconds": 5,
            "description": "Flush pending writes",
        }))
        self.assertTrue(result.success)
        self.assertIn("Registered", result.message)
        self.assertEqual(len(skill._hooks), 1)
        self.assertEqual(skill._hooks["flush_db"].priority, 50)

    def test_register_hook_missing_fields(self):
        skill = LifecycleManagerSkill()
        result = run_async(skill.execute("register_hook", {"hook_id": ""}))
        self.assertFalse(result.success)
        self.assertIn("required", result.message)

    def test_register_hook_replace(self):
        skill = LifecycleManagerSkill()
        skill.mark_started()
        run_async(skill.execute("register_hook", {
            "hook_id": "h1", "name": "First"
        }))
        result = run_async(skill.execute("register_hook", {
            "hook_id": "h1", "name": "Updated"
        }))
        self.assertTrue(result.success)
        self.assertIn("Replaced", result.message)
        self.assertEqual(skill._hooks["h1"].name, "Updated")

    def test_cannot_register_during_shutdown(self):
        skill = LifecycleManagerSkill()
        skill._phase = LifecyclePhase.STOPPING
        result = run_async(skill.execute("register_hook", {
            "hook_id": "late", "name": "Late Hook"
        }))
        self.assertFalse(result.success)
        self.assertIn("shutdown", result.message.lower())

    def test_register_hook_with_callback(self):
        skill = LifecycleManagerSkill()
        called = []
        async def cleanup():
            called.append(True)

        hook_id = skill.register_hook_with_callback(
            hook_id="cb_hook",
            name="Callback Hook",
            callback=cleanup,
            priority=10,
        )
        self.assertEqual(hook_id, "cb_hook")
        self.assertIsNotNone(skill._hooks["cb_hook"].callback)


class TestUnregisterHook(unittest.TestCase):
    """Test hook unregistration."""

    def test_unregister_existing(self):
        skill = LifecycleManagerSkill()
        skill.mark_started()
        run_async(skill.execute("register_hook", {
            "hook_id": "h1", "name": "Hook 1"
        }))
        result = run_async(skill.execute("unregister_hook", {"hook_id": "h1"}))
        self.assertTrue(result.success)
        self.assertEqual(len(skill._hooks), 0)

    def test_unregister_nonexistent(self):
        skill = LifecycleManagerSkill()
        result = run_async(skill.execute("unregister_hook", {"hook_id": "nope"}))
        self.assertFalse(result.success)

    def test_unregister_missing_id(self):
        skill = LifecycleManagerSkill()
        result = run_async(skill.execute("unregister_hook", {}))
        self.assertFalse(result.success)


class TestGracefulShutdown(unittest.TestCase):
    """Test graceful shutdown orchestration."""

    def test_basic_shutdown(self):
        skill = LifecycleManagerSkill()
        skill.mark_started()
        stop_called = []
        skill.set_agent_hooks(stop_fn=lambda: stop_called.append(True))

        result = run_async(skill.execute("initiate_shutdown", {
            "reason": "Test shutdown",
        }))

        self.assertTrue(result.success)
        self.assertIn("Graceful shutdown complete", result.message)
        self.assertEqual(skill._phase, LifecyclePhase.STOPPED)
        self.assertTrue(skill._shutdown_requested)
        self.assertEqual(len(stop_called), 1)

    def test_shutdown_already_stopped(self):
        skill = LifecycleManagerSkill()
        skill._phase = LifecyclePhase.STOPPED
        result = run_async(skill.execute("initiate_shutdown", {}))
        self.assertTrue(result.success)
        self.assertIn("Already stopped", result.message)

    def test_hooks_execute_in_priority_order(self):
        skill = LifecycleManagerSkill()
        skill.mark_started()
        skill.set_agent_hooks(stop_fn=lambda: None)

        execution_order = []

        async def hook_a():
            execution_order.append("A")

        async def hook_b():
            execution_order.append("B")

        async def hook_c():
            execution_order.append("C")

        skill.register_hook_with_callback("a", "Hook A", hook_a, priority=200)
        skill.register_hook_with_callback("b", "Hook B", hook_b, priority=50)
        skill.register_hook_with_callback("c", "Hook C", hook_c, priority=100)

        run_async(skill.execute("initiate_shutdown", {"reason": "test"}))

        self.assertEqual(execution_order, ["B", "C", "A"])

    def test_hook_timeout_handling(self):
        skill = LifecycleManagerSkill()
        skill.mark_started()
        skill.set_agent_hooks(stop_fn=lambda: None)

        async def slow_hook():
            await asyncio.sleep(10)

        skill.register_hook_with_callback(
            "slow", "Slow Hook", slow_hook,
            priority=100, timeout_seconds=0.1,
        )

        result = run_async(skill.execute("initiate_shutdown", {"reason": "test"}))
        self.assertTrue(result.success)

        hook_results = result.data["hook_results"]
        self.assertEqual(len(hook_results), 1)
        self.assertEqual(hook_results[0]["status"], "timeout")

    def test_hook_error_handling(self):
        skill = LifecycleManagerSkill()
        skill.mark_started()
        skill.set_agent_hooks(stop_fn=lambda: None)

        async def bad_hook():
            raise RuntimeError("Cleanup failed!")

        skill.register_hook_with_callback("bad", "Bad Hook", bad_hook, priority=100)

        result = run_async(skill.execute("initiate_shutdown", {"reason": "test"}))
        self.assertTrue(result.success)  # Shutdown succeeds despite hook failure

        hook_results = result.data["hook_results"]
        self.assertEqual(hook_results[0]["status"], "failed")
        self.assertIn("Cleanup failed", hook_results[0]["error"])

    def test_hooks_without_callback_skipped(self):
        skill = LifecycleManagerSkill()
        skill.mark_started()
        skill.set_agent_hooks(stop_fn=lambda: None)

        run_async(skill.execute("register_hook", {
            "hook_id": "no_cb", "name": "No Callback"
        }))

        result = run_async(skill.execute("initiate_shutdown", {"reason": "test"}))
        hook_results = result.data["hook_results"]
        self.assertEqual(hook_results[0]["status"], "skipped")

    def test_global_shutdown_timeout(self):
        skill = LifecycleManagerSkill(credentials={"SHUTDOWN_TIMEOUT": "0.2"})
        skill.mark_started()
        skill.set_agent_hooks(stop_fn=lambda: None)

        async def slow():
            await asyncio.sleep(0.15)

        # Register enough hooks that total exceeds 0.2s
        skill.register_hook_with_callback("s1", "Slow 1", slow, priority=10, timeout_seconds=1)
        skill.register_hook_with_callback("s2", "Slow 2", slow, priority=20, timeout_seconds=1)
        skill.register_hook_with_callback("s3", "Slow 3", slow, priority=30, timeout_seconds=1)

        result = run_async(skill.execute("initiate_shutdown", {"reason": "test"}))
        hook_results = result.data["hook_results"]

        # At least one hook should have been skipped due to global timeout
        statuses = [r["status"] for r in hook_results]
        self.assertIn("skipped", statuses)

    def test_sync_callback_hook(self):
        skill = LifecycleManagerSkill()
        skill.mark_started()
        skill.set_agent_hooks(stop_fn=lambda: None)

        called = []
        def sync_cleanup():
            called.append(True)

        skill.register_hook_with_callback("sync", "Sync Hook", sync_cleanup, priority=100)
        result = run_async(skill.execute("initiate_shutdown", {"reason": "test"}))

        self.assertTrue(result.success)
        self.assertEqual(len(called), 1)

    def test_shutdown_emits_lifecycle_events(self):
        skill = LifecycleManagerSkill()
        skill.mark_started()
        skill.set_agent_hooks(stop_fn=lambda: None)

        # Mock EventBus
        mock_bus = MagicMock()
        mock_bus.publish = AsyncMock()
        skill._event_bus_ref = mock_bus

        run_async(skill.execute("initiate_shutdown", {"reason": "test"}))

        # Should have emitted agent.stopping and agent.stopped
        call_topics = [call.args[0] for call in mock_bus.publish.call_args_list]
        self.assertIn("agent.stopping", call_topics)
        self.assertIn("agent.stopped", call_topics)


class TestHealthProbe(unittest.TestCase):
    """Test health probe endpoints."""

    def test_health_initializing(self):
        skill = LifecycleManagerSkill()
        result = run_async(skill.execute("get_health", {}))
        self.assertTrue(result.success)
        self.assertTrue(result.data["alive"])
        self.assertFalse(result.data["ready"])

    def test_health_running(self):
        skill = LifecycleManagerSkill()
        skill.mark_started()
        result = run_async(skill.execute("get_health", {}))
        self.assertTrue(result.data["alive"])
        self.assertTrue(result.data["ready"])
        self.assertEqual(result.data["phase"], "running")

    def test_health_draining(self):
        skill = LifecycleManagerSkill()
        skill.mark_started()
        skill._phase = LifecyclePhase.DRAINING
        result = run_async(skill.execute("get_health", {}))
        self.assertTrue(result.data["alive"])
        self.assertFalse(result.data["ready"])

    def test_health_stopped(self):
        skill = LifecycleManagerSkill()
        skill._phase = LifecyclePhase.STOPPED
        result = run_async(skill.execute("get_health", {}))
        self.assertFalse(result.data["alive"])
        self.assertFalse(result.data["ready"])

    def test_health_shutdown_requested(self):
        skill = LifecycleManagerSkill()
        skill.mark_started()
        skill._shutdown_requested = True
        result = run_async(skill.execute("get_health", {}))
        self.assertTrue(result.data["alive"])
        self.assertFalse(result.data["ready"])


class TestGetStatus(unittest.TestCase):
    """Test status reporting."""

    def test_status_basic(self):
        skill = LifecycleManagerSkill()
        skill.mark_started()
        result = run_async(skill.execute("get_status", {}))
        self.assertTrue(result.success)
        self.assertEqual(result.data["phase"], "running")
        self.assertFalse(result.data["shutdown_requested"])
        self.assertEqual(result.data["registered_hooks"], 0)
        self.assertIn("config", result.data)
        self.assertEqual(result.data["config"]["shutdown_timeout"], 30.0)

    def test_status_with_hooks(self):
        skill = LifecycleManagerSkill()
        skill.mark_started()
        run_async(skill.execute("register_hook", {
            "hook_id": "h1", "name": "Hook 1"
        }))
        run_async(skill.execute("register_hook", {
            "hook_id": "h2", "name": "Hook 2"
        }))
        result = run_async(skill.execute("get_status", {}))
        self.assertEqual(result.data["registered_hooks"], 2)


class TestDrainMode(unittest.TestCase):
    """Test drain mode."""

    def test_drain_no_active_tasks(self):
        skill = LifecycleManagerSkill()
        skill.mark_started()
        result = run_async(skill.execute("drain", {}))
        self.assertTrue(result.success)
        self.assertIn("Drain complete", result.message)
        self.assertFalse(result.data["timed_out"])

    def test_drain_already_draining(self):
        skill = LifecycleManagerSkill()
        skill._phase = LifecyclePhase.DRAINING
        result = run_async(skill.execute("drain", {}))
        self.assertTrue(result.success)
        self.assertIn("Already draining", result.message)

    def test_drain_wrong_phase(self):
        skill = LifecycleManagerSkill()
        # Still in INITIALIZING phase
        result = run_async(skill.execute("drain", {}))
        self.assertFalse(result.success)
        self.assertIn("Cannot drain", result.message)

    def test_drain_with_tasks_completing(self):
        skill = LifecycleManagerSkill()
        skill.mark_started()
        skill._active_tasks = 1

        async def drain_with_task_end():
            # Simulate task ending shortly after drain starts
            async def end_task():
                await asyncio.sleep(0.1)
                skill.track_task_end()
            asyncio.ensure_future(end_task())
            return await skill.execute("drain", {"timeout_seconds": 2})

        result = run_async(drain_with_task_end())
        self.assertTrue(result.success)
        self.assertEqual(skill._active_tasks, 0)

    def test_drain_timeout(self):
        skill = LifecycleManagerSkill()
        skill.mark_started()
        skill._active_tasks = 5  # Stuck tasks

        result = run_async(skill.execute("drain", {"timeout_seconds": 0.2}))
        self.assertFalse(result.success)
        self.assertTrue(result.data["timed_out"])
        self.assertEqual(result.data["active_tasks"], 5)


class TestListHooks(unittest.TestCase):
    """Test hook listing."""

    def test_list_empty(self):
        skill = LifecycleManagerSkill()
        result = run_async(skill.execute("list_hooks", {}))
        self.assertTrue(result.success)
        self.assertEqual(len(result.data["hooks"]), 0)

    def test_list_sorted_by_priority(self):
        skill = LifecycleManagerSkill()
        skill.mark_started()
        run_async(skill.execute("register_hook", {
            "hook_id": "low", "name": "Low Priority", "priority": 200
        }))
        run_async(skill.execute("register_hook", {
            "hook_id": "high", "name": "High Priority", "priority": 10
        }))
        run_async(skill.execute("register_hook", {
            "hook_id": "mid", "name": "Mid Priority", "priority": 100
        }))

        result = run_async(skill.execute("list_hooks", {}))
        hooks = result.data["hooks"]
        self.assertEqual(len(hooks), 3)
        self.assertEqual(hooks[0]["hook_id"], "high")
        self.assertEqual(hooks[1]["hook_id"], "mid")
        self.assertEqual(hooks[2]["hook_id"], "low")


class TestTaskTracking(unittest.TestCase):
    """Test active task tracking for drain mode."""

    def test_track_task_lifecycle(self):
        skill = LifecycleManagerSkill()
        self.assertEqual(skill._active_tasks, 0)

        skill.track_task_start()
        self.assertEqual(skill._active_tasks, 1)

        skill.track_task_start()
        self.assertEqual(skill._active_tasks, 2)

        skill.track_task_end()
        self.assertEqual(skill._active_tasks, 1)

        skill.track_task_end()
        self.assertEqual(skill._active_tasks, 0)

    def test_track_task_end_no_underflow(self):
        skill = LifecycleManagerSkill()
        skill.track_task_end()  # Should not go below 0
        self.assertEqual(skill._active_tasks, 0)


class TestPropertyHelpers(unittest.TestCase):
    """Test convenience properties."""

    def test_is_draining(self):
        skill = LifecycleManagerSkill()
        self.assertFalse(skill.is_draining)
        skill._phase = LifecyclePhase.DRAINING
        self.assertTrue(skill.is_draining)

    def test_is_shutting_down(self):
        skill = LifecycleManagerSkill()
        self.assertFalse(skill.is_shutting_down)

        for phase in [LifecyclePhase.DRAINING, LifecyclePhase.STOPPING, LifecyclePhase.STOPPED]:
            skill._phase = phase
            self.assertTrue(skill.is_shutting_down, f"Expected shutting_down=True for {phase}")

        skill._phase = LifecyclePhase.RUNNING
        self.assertFalse(skill.is_shutting_down)


class TestSignalHandlers(unittest.TestCase):
    """Test signal handler installation/uninstallation."""

    def test_install_sync_signals(self):
        skill = LifecycleManagerSkill()
        self.assertFalse(skill._signals_installed)

        # Install with no loop (sync mode)
        skill.install_signal_handlers(loop=None)
        self.assertTrue(skill._signals_installed)

        # Uninstall restores original handlers
        skill.uninstall_signal_handlers()
        self.assertFalse(skill._signals_installed)

    def test_double_install_idempotent(self):
        skill = LifecycleManagerSkill()
        skill.install_signal_handlers(loop=None)
        skill.install_signal_handlers(loop=None)  # Should be no-op
        self.assertTrue(skill._signals_installed)
        skill.uninstall_signal_handlers()

    def test_uninstall_without_install(self):
        skill = LifecycleManagerSkill()
        skill.uninstall_signal_handlers()  # Should be no-op
        self.assertFalse(skill._signals_installed)


class TestUnknownAction(unittest.TestCase):
    """Test error handling for unknown actions."""

    def test_unknown_action(self):
        skill = LifecycleManagerSkill()
        result = run_async(skill.execute("nonexistent_action", {}))
        self.assertFalse(result.success)
        self.assertIn("Unknown action", result.message)


class TestStatePersistence(unittest.TestCase):
    """Test shutdown state persistence to disk."""

    def test_persist_and_read_shutdown_state(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Monkey-patch the data dir
            import singularity.skills.lifecycle_manager as lm_mod
            orig_data_dir = lm_mod.DATA_DIR
            orig_state_file = lm_mod.LIFECYCLE_STATE_FILE
            try:
                lm_mod.DATA_DIR = Path(tmpdir)
                lm_mod.LIFECYCLE_STATE_FILE = Path(tmpdir) / "lifecycle_state.json"

                skill = LifecycleManagerSkill()
                skill.mark_started()
                skill.set_agent_hooks(stop_fn=lambda: None)

                run_async(skill.execute("initiate_shutdown", {
                    "reason": "Persistence test",
                }))

                # Read back from disk
                info = skill.get_last_shutdown_info()
                self.assertIsNotNone(info)
                self.assertEqual(info["reason"], "Persistence test")
                self.assertIn("timestamp", info)
                self.assertIn("uptime_seconds", info)
            finally:
                lm_mod.DATA_DIR = orig_data_dir
                lm_mod.LIFECYCLE_STATE_FILE = orig_state_file


class TestSetAgentHooks(unittest.TestCase):
    """Test wiring the skill into the agent."""

    def test_set_hooks(self):
        skill = LifecycleManagerSkill()
        mock_stop = MagicMock()
        mock_bus = MagicMock()

        skill.set_agent_hooks(stop_fn=mock_stop, event_bus=mock_bus)
        self.assertEqual(skill._agent_stop_fn, mock_stop)
        self.assertEqual(skill._event_bus_ref, mock_bus)


class TestFullLifecycleIntegration(unittest.TestCase):
    """Integration test: full lifecycle from start to shutdown with multiple hooks."""

    def test_full_lifecycle(self):
        skill = LifecycleManagerSkill()
        stop_called = []
        mock_bus = MagicMock()
        mock_bus.publish = AsyncMock()

        skill.set_agent_hooks(
            stop_fn=lambda: stop_called.append(True),
            event_bus=mock_bus,
        )

        # Phase 1: Starting
        skill.mark_starting()
        self.assertEqual(skill._phase, LifecyclePhase.STARTING)

        # Phase 2: Running
        skill.mark_started()
        self.assertEqual(skill._phase, LifecyclePhase.RUNNING)

        # Register hooks
        execution_log = []

        async def flush_queues():
            execution_log.append("flush_queues")

        async def save_checkpoint():
            execution_log.append("save_checkpoint")

        async def close_connections():
            execution_log.append("close_connections")

        skill.register_hook_with_callback(
            "save_cp", "Save Checkpoint", save_checkpoint,
            priority=50, description="Save state snapshot",
        )
        skill.register_hook_with_callback(
            "flush", "Flush Queues", flush_queues,
            priority=100, description="Flush pending messages",
        )
        skill.register_hook_with_callback(
            "close", "Close Connections", close_connections,
            priority=150, description="Close HTTP connections",
        )

        # Verify health
        health = run_async(skill.execute("get_health", {}))
        self.assertTrue(health.data["alive"])
        self.assertTrue(health.data["ready"])

        # Simulate some active tasks
        skill.track_task_start()
        skill.track_task_start()
        skill.track_task_end()  # One still active

        # Phase 3: Shutdown
        result = run_async(skill.execute("initiate_shutdown", {
            "reason": "Planned maintenance",
        }))

        self.assertTrue(result.success)
        self.assertEqual(skill._phase, LifecyclePhase.STOPPED)
        self.assertEqual(execution_log, ["save_checkpoint", "flush_queues", "close_connections"])
        self.assertEqual(len(stop_called), 1)
        self.assertEqual(result.data["reason"], "Planned maintenance")

        # Verify health after shutdown
        health = run_async(skill.execute("get_health", {}))
        self.assertFalse(health.data["alive"])
        self.assertFalse(health.data["ready"])

        # Verify events were emitted
        event_topics = [call.args[0] for call in mock_bus.publish.call_args_list]
        self.assertIn("agent.stopping", event_topics)
        self.assertIn("agent.stopped", event_topics)


if __name__ == "__main__":
    unittest.main()
