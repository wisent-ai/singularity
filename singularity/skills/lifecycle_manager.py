#!/usr/bin/env python3
"""
LifecycleManagerSkill - Graceful shutdown, signal handling, and health probes for agents.

Autonomous agents running as long-lived processes (Docker containers, systemd services,
Kubernetes pods) need proper lifecycle management. Without it, a SIGTERM from Docker
or Kubernetes causes an immediate kill — losing in-flight tasks, unsaved state, and
potentially corrupting data files.

This skill provides:
- SIGTERM/SIGINT signal handlers that trigger ordered graceful shutdown
- Shutdown hook registry — skills register cleanup callbacks executed in priority order
- Configurable shutdown timeout with forced exit fallback
- Liveness and readiness health probes for orchestrators (K8s, Docker, etc.)
- Lifecycle event emission (agent.starting, agent.stopping, agent.stopped)
- Startup/shutdown timing metrics
- Drain mode — stop accepting new work while finishing in-flight tasks

Serves all four pillars:
- Self-Improvement: Clean shutdown preserves learned state, prevents corruption
- Revenue: Graceful drain ensures customer tasks complete before shutdown
- Replication: Orderly shutdown + checkpoint enables clean migration to replicas
- Goal Setting: Shutdown metrics reveal reliability issues and resource leaks

Actions:
- register_hook: Register a shutdown cleanup callback with priority
- unregister_hook: Remove a registered shutdown hook
- initiate_shutdown: Programmatically trigger graceful shutdown
- get_health: Get liveness/readiness health probe status
- get_status: Get current lifecycle state and metrics
- drain: Enter drain mode (stop new work, finish existing)
- list_hooks: List all registered shutdown hooks
"""

import asyncio
import json
import os
import signal
import time
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Coroutine, Dict, List, Optional, Union

from .base import Skill, SkillAction, SkillManifest, SkillResult

# Data directory for lifecycle state
DATA_DIR = Path(__file__).parent.parent / "data"
LIFECYCLE_STATE_FILE = DATA_DIR / "lifecycle_state.json"


class LifecyclePhase(str, Enum):
    """Agent lifecycle phases."""
    INITIALIZING = "initializing"
    STARTING = "starting"
    RUNNING = "running"
    DRAINING = "draining"
    STOPPING = "stopping"
    STOPPED = "stopped"


class ShutdownHook:
    """A registered shutdown cleanup callback."""

    def __init__(
        self,
        hook_id: str,
        name: str,
        callback: Optional[Callable] = None,
        priority: int = 100,
        timeout_seconds: float = 10.0,
        description: str = "",
    ):
        """
        Args:
            hook_id: Unique identifier for this hook.
            name: Human-readable name.
            callback: Async or sync callable to invoke during shutdown.
            priority: Execution order (lower = earlier). Defaults to 100.
                      Recommended ranges:
                        0-49:   Critical infrastructure (signal handlers, event bus)
                        50-99:  State persistence (checkpoints, data files)
                        100-149: Application logic (flush queues, close connections)
                        150-199: Cleanup (temp files, metrics reporting)
                        200+:   Best-effort (analytics, notifications)
            timeout_seconds: Max time to wait for this hook. 0 = no timeout.
            description: What this hook does (for diagnostics).
        """
        self.hook_id = hook_id
        self.name = name
        self.callback = callback
        self.priority = priority
        self.timeout_seconds = timeout_seconds
        self.description = description
        self.registered_at = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hook_id": self.hook_id,
            "name": self.name,
            "priority": self.priority,
            "timeout_seconds": self.timeout_seconds,
            "description": self.description,
            "registered_at": self.registered_at,
            "has_callback": self.callback is not None,
        }


class LifecycleManagerSkill(Skill):
    """
    Manages agent lifecycle: signal handling, graceful shutdown, and health probes.

    Wire this skill into the agent to get production-grade lifecycle management:
    - Docker/K8s SIGTERM triggers orderly shutdown instead of immediate kill
    - Skills can register cleanup hooks that run in priority order
    - Health endpoints enable proper load balancer and orchestrator integration
    - Drain mode lets in-flight work complete before shutdown
    """

    # Shutdown configuration defaults
    DEFAULT_SHUTDOWN_TIMEOUT = 30.0  # Max total shutdown time in seconds
    DEFAULT_DRAIN_TIMEOUT = 60.0     # Max drain time before forced shutdown
    DEFAULT_HOOK_TIMEOUT = 10.0      # Per-hook timeout

    def __init__(self, credentials: Dict[str, str] = None):
        super().__init__(credentials)
        self._phase = LifecyclePhase.INITIALIZING
        self._hooks: Dict[str, ShutdownHook] = {}
        self._shutdown_requested = False
        self._shutdown_reason = ""
        self._drain_start: Optional[float] = None
        self._start_time: Optional[float] = None
        self._shutdown_start: Optional[float] = None
        self._shutdown_complete: Optional[float] = None
        self._hook_results: List[Dict[str, Any]] = []
        self._signals_installed = False
        self._original_sigterm = None
        self._original_sigint = None
        self._agent_stop_fn: Optional[Callable] = None
        self._event_bus_ref = None  # Set by agent wiring
        self._active_tasks: int = 0  # Track in-flight tasks for drain mode
        self._ready = False  # Readiness flag (set after full startup)

        # Configurable timeouts from credentials/env
        self._shutdown_timeout = float(
            (credentials or {}).get("SHUTDOWN_TIMEOUT", self.DEFAULT_SHUTDOWN_TIMEOUT)
        )
        self._drain_timeout = float(
            (credentials or {}).get("DRAIN_TIMEOUT", self.DEFAULT_DRAIN_TIMEOUT)
        )

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="lifecycle_manager",
            name="Lifecycle Manager",
            version="1.0.0",
            category="infrastructure",
            description=(
                "Graceful shutdown, signal handling, and health probes. "
                "Ensures clean agent termination with ordered cleanup, "
                "proper state persistence, and orchestrator integration."
            ),
            actions=[
                SkillAction(
                    name="register_hook",
                    description=(
                        "Register a shutdown cleanup hook. Hooks run in priority order "
                        "(lower priority number = runs first) during graceful shutdown."
                    ),
                    parameters={
                        "hook_id": {"type": "string", "required": True,
                                    "description": "Unique hook identifier"},
                        "name": {"type": "string", "required": True,
                                 "description": "Human-readable hook name"},
                        "priority": {"type": "integer", "required": False,
                                     "description": "Execution order (lower=earlier, default=100)"},
                        "timeout_seconds": {"type": "number", "required": False,
                                            "description": "Per-hook timeout (default=10)"},
                        "description": {"type": "string", "required": False,
                                        "description": "What this hook does"},
                    },
                ),
                SkillAction(
                    name="unregister_hook",
                    description="Remove a registered shutdown hook by ID.",
                    parameters={
                        "hook_id": {"type": "string", "required": True,
                                    "description": "Hook ID to remove"},
                    },
                ),
                SkillAction(
                    name="initiate_shutdown",
                    description=(
                        "Programmatically trigger graceful shutdown. "
                        "Runs all hooks in order and signals the agent to stop."
                    ),
                    parameters={
                        "reason": {"type": "string", "required": False,
                                   "description": "Reason for shutdown"},
                    },
                ),
                SkillAction(
                    name="get_health",
                    description=(
                        "Health probe endpoint. Returns liveness and readiness status "
                        "for orchestrator integration (Kubernetes, Docker, etc.)."
                    ),
                    parameters={},
                ),
                SkillAction(
                    name="get_status",
                    description="Get current lifecycle state, phase, uptime, and metrics.",
                    parameters={},
                ),
                SkillAction(
                    name="drain",
                    description=(
                        "Enter drain mode: stop accepting new work while letting "
                        "in-flight tasks complete. Useful before planned shutdown."
                    ),
                    parameters={
                        "timeout_seconds": {"type": "number", "required": False,
                                            "description": "Max drain time (default=60)"},
                    },
                ),
                SkillAction(
                    name="list_hooks",
                    description="List all registered shutdown hooks with details.",
                    parameters={},
                ),
            ],
            required_credentials=[],
            install_cost=0,
            author="adam",
        )

    def set_agent_hooks(
        self,
        stop_fn: Callable,
        event_bus=None,
    ):
        """
        Wire lifecycle manager into the agent.

        Args:
            stop_fn: Callable that stops the agent (sets running=False)
            event_bus: Optional EventBus reference for lifecycle events
        """
        self._agent_stop_fn = stop_fn
        self._event_bus_ref = event_bus

    def mark_started(self):
        """Called by agent after full initialization to mark as running."""
        self._start_time = time.monotonic()
        self._phase = LifecyclePhase.RUNNING
        self._ready = True

    def mark_starting(self):
        """Called by agent during initialization."""
        self._phase = LifecyclePhase.STARTING
        self._start_time = time.monotonic()

    def install_signal_handlers(self, loop: Optional[asyncio.AbstractEventLoop] = None):
        """
        Install SIGTERM and SIGINT handlers for graceful shutdown.

        Should be called after the event loop is running. Handles both
        threaded (signal.signal) and async (loop.add_signal_handler) modes.

        Args:
            loop: Event loop to register handlers on. If None, uses signal.signal.
        """
        if self._signals_installed:
            return

        if loop is not None:
            try:
                # Async signal handlers (preferred for event loop integration)
                loop.add_signal_handler(
                    signal.SIGTERM,
                    lambda: asyncio.ensure_future(
                        self._signal_shutdown("SIGTERM"), loop=loop
                    ),
                )
                loop.add_signal_handler(
                    signal.SIGINT,
                    lambda: asyncio.ensure_future(
                        self._signal_shutdown("SIGINT"), loop=loop
                    ),
                )
                self._signals_installed = True
                return
            except (NotImplementedError, RuntimeError):
                # Windows or non-main thread — fall through to signal.signal
                pass

        # Fallback: synchronous signal handlers
        self._original_sigterm = signal.getsignal(signal.SIGTERM)
        self._original_sigint = signal.getsignal(signal.SIGINT)

        def _sync_handler(signum, frame):
            sig_name = "SIGTERM" if signum == signal.SIGTERM else "SIGINT"
            # Schedule async shutdown on the event loop
            try:
                loop = asyncio.get_running_loop()
                asyncio.ensure_future(self._signal_shutdown(sig_name), loop=loop)
            except RuntimeError:
                # No event loop — set flag for the agent loop to pick up
                self._shutdown_requested = True
                self._shutdown_reason = f"Signal {sig_name} (no event loop)"
                if self._agent_stop_fn:
                    self._agent_stop_fn()

        signal.signal(signal.SIGTERM, _sync_handler)
        signal.signal(signal.SIGINT, _sync_handler)
        self._signals_installed = True

    def uninstall_signal_handlers(self):
        """Restore original signal handlers."""
        if not self._signals_installed:
            return

        try:
            if self._original_sigterm is not None:
                signal.signal(signal.SIGTERM, self._original_sigterm)
            if self._original_sigint is not None:
                signal.signal(signal.SIGINT, self._original_sigint)
        except (OSError, ValueError):
            pass  # Not in main thread or process

        self._signals_installed = False

    async def _signal_shutdown(self, signal_name: str):
        """Handle shutdown signal asynchronously."""
        if self._shutdown_requested:
            # Second signal — force immediate exit
            self._phase = LifecyclePhase.STOPPED
            return

        await self.execute("initiate_shutdown", {
            "reason": f"Received {signal_name}",
        })

    def track_task_start(self):
        """Track that a new task has started (for drain mode)."""
        self._active_tasks += 1

    def track_task_end(self):
        """Track that a task has completed (for drain mode)."""
        self._active_tasks = max(0, self._active_tasks - 1)

    @property
    def is_draining(self) -> bool:
        """Whether the agent is in drain mode."""
        return self._phase == LifecyclePhase.DRAINING

    @property
    def is_shutting_down(self) -> bool:
        """Whether shutdown has been initiated."""
        return self._phase in (
            LifecyclePhase.STOPPING,
            LifecyclePhase.STOPPED,
            LifecyclePhase.DRAINING,
        )

    @property
    def uptime_seconds(self) -> float:
        """Agent uptime in seconds."""
        if self._start_time is None:
            return 0.0
        return time.monotonic() - self._start_time

    async def execute(self, action: str, params: Dict) -> SkillResult:
        """Execute a lifecycle management action."""

        if action == "register_hook":
            return self._register_hook(params)
        elif action == "unregister_hook":
            return self._unregister_hook(params)
        elif action == "initiate_shutdown":
            return await self._initiate_shutdown(params)
        elif action == "get_health":
            return self._get_health()
        elif action == "get_status":
            return self._get_status()
        elif action == "drain":
            return await self._drain(params)
        elif action == "list_hooks":
            return self._list_hooks()
        else:
            return SkillResult(
                success=False,
                message=f"Unknown action: {action}. "
                        f"Available: register_hook, unregister_hook, initiate_shutdown, "
                        f"get_health, get_status, drain, list_hooks",
            )

    def _register_hook(self, params: Dict) -> SkillResult:
        """Register a shutdown cleanup hook."""
        hook_id = params.get("hook_id", "")
        name = params.get("name", "")

        if not hook_id or not name:
            return SkillResult(
                success=False,
                message="hook_id and name are required",
            )

        if self._phase in (LifecyclePhase.STOPPING, LifecyclePhase.STOPPED):
            return SkillResult(
                success=False,
                message="Cannot register hooks during shutdown",
            )

        hook = ShutdownHook(
            hook_id=hook_id,
            name=name,
            priority=int(params.get("priority", 100)),
            timeout_seconds=float(params.get("timeout_seconds", self.DEFAULT_HOOK_TIMEOUT)),
            description=params.get("description", ""),
        )

        replaced = hook_id in self._hooks
        self._hooks[hook_id] = hook

        return SkillResult(
            success=True,
            message=f"{'Replaced' if replaced else 'Registered'} shutdown hook '{name}' "
                    f"(priority={hook.priority})",
            data=hook.to_dict(),
        )

    def register_hook_with_callback(
        self,
        hook_id: str,
        name: str,
        callback: Callable,
        priority: int = 100,
        timeout_seconds: float = 10.0,
        description: str = "",
    ) -> str:
        """
        Register a shutdown hook with an actual callback function.

        This is the programmatic API for skills to register their own cleanup.
        Unlike the execute("register_hook") action (which is for the LLM to call),
        this accepts a real callback function.

        Args:
            hook_id: Unique hook identifier.
            name: Human-readable name.
            callback: Async or sync callable to run during shutdown.
            priority: Execution order (lower=earlier).
            timeout_seconds: Per-hook timeout.
            description: What this hook does.

        Returns:
            The hook_id for later unregistration.
        """
        hook = ShutdownHook(
            hook_id=hook_id,
            name=name,
            callback=callback,
            priority=priority,
            timeout_seconds=timeout_seconds,
            description=description,
        )
        self._hooks[hook_id] = hook
        return hook_id

    def _unregister_hook(self, params: Dict) -> SkillResult:
        """Remove a registered shutdown hook."""
        hook_id = params.get("hook_id", "")
        if not hook_id:
            return SkillResult(success=False, message="hook_id is required")

        if hook_id in self._hooks:
            removed = self._hooks.pop(hook_id)
            return SkillResult(
                success=True,
                message=f"Removed shutdown hook '{removed.name}'",
            )
        else:
            return SkillResult(
                success=False,
                message=f"Hook '{hook_id}' not found",
            )

    async def _initiate_shutdown(self, params: Dict) -> SkillResult:
        """Orchestrate graceful shutdown."""
        if self._phase == LifecyclePhase.STOPPED:
            return SkillResult(
                success=True,
                message="Already stopped",
            )

        reason = params.get("reason", "Programmatic shutdown")
        self._shutdown_requested = True
        self._shutdown_reason = reason
        self._shutdown_start = time.monotonic()
        self._phase = LifecyclePhase.STOPPING

        # Emit stopping event
        await self._emit_lifecycle_event("agent.stopping", {
            "reason": reason,
            "hooks_registered": len(self._hooks),
            "active_tasks": self._active_tasks,
        })

        # Execute hooks in priority order (lower priority number first)
        sorted_hooks = sorted(self._hooks.values(), key=lambda h: h.priority)
        self._hook_results = []
        total_start = time.monotonic()

        for hook in sorted_hooks:
            # Check global shutdown timeout
            elapsed = time.monotonic() - total_start
            if elapsed >= self._shutdown_timeout:
                self._hook_results.append({
                    "hook_id": hook.hook_id,
                    "name": hook.name,
                    "status": "skipped",
                    "reason": "Global shutdown timeout exceeded",
                })
                continue

            hook_result = await self._execute_hook(hook)
            self._hook_results.append(hook_result)

        # Persist shutdown state
        self._persist_shutdown_state()

        # Emit stopped event
        self._shutdown_complete = time.monotonic()
        shutdown_duration = self._shutdown_complete - self._shutdown_start

        await self._emit_lifecycle_event("agent.stopped", {
            "reason": reason,
            "shutdown_duration_seconds": round(shutdown_duration, 3),
            "hooks_executed": len(self._hook_results),
            "hooks_succeeded": sum(
                1 for r in self._hook_results if r.get("status") == "success"
            ),
            "hooks_failed": sum(
                1 for r in self._hook_results if r.get("status") == "failed"
            ),
        })

        self._phase = LifecyclePhase.STOPPED

        # Signal the agent to stop its main loop
        if self._agent_stop_fn:
            self._agent_stop_fn()

        # Restore original signal handlers
        self.uninstall_signal_handlers()

        return SkillResult(
            success=True,
            message=f"Graceful shutdown complete in {shutdown_duration:.2f}s "
                    f"({len(self._hook_results)} hooks executed). Reason: {reason}",
            data={
                "reason": reason,
                "shutdown_duration_seconds": round(shutdown_duration, 3),
                "hook_results": self._hook_results,
                "phase": self._phase.value,
            },
        )

    async def _execute_hook(self, hook: ShutdownHook) -> Dict[str, Any]:
        """Execute a single shutdown hook with timeout protection."""
        result = {
            "hook_id": hook.hook_id,
            "name": hook.name,
            "priority": hook.priority,
        }

        if hook.callback is None:
            result["status"] = "skipped"
            result["reason"] = "No callback registered"
            return result

        start = time.monotonic()
        try:
            if hook.timeout_seconds > 0:
                # Run with timeout
                coro = hook.callback()
                if asyncio.iscoroutine(coro):
                    await asyncio.wait_for(coro, timeout=hook.timeout_seconds)
                # If callback is sync, it already executed
            else:
                # No timeout
                coro = hook.callback()
                if asyncio.iscoroutine(coro):
                    await coro

            elapsed = time.monotonic() - start
            result["status"] = "success"
            result["duration_seconds"] = round(elapsed, 3)

        except asyncio.TimeoutError:
            elapsed = time.monotonic() - start
            result["status"] = "timeout"
            result["duration_seconds"] = round(elapsed, 3)
            result["reason"] = f"Exceeded {hook.timeout_seconds}s timeout"

        except Exception as e:
            elapsed = time.monotonic() - start
            result["status"] = "failed"
            result["duration_seconds"] = round(elapsed, 3)
            result["error"] = str(e)[:200]

        return result

    def _get_health(self) -> SkillResult:
        """Return health probe status for orchestrators."""
        # Liveness: Is the process alive and responsive?
        alive = self._phase not in (LifecyclePhase.STOPPED,)

        # Readiness: Is the agent ready to accept work?
        ready = (
            self._phase == LifecyclePhase.RUNNING
            and self._ready
            and not self._shutdown_requested
        )

        health = {
            "alive": alive,
            "ready": ready,
            "phase": self._phase.value,
            "uptime_seconds": round(self.uptime_seconds, 1),
            "shutdown_requested": self._shutdown_requested,
            "active_tasks": self._active_tasks,
        }

        return SkillResult(
            success=True,
            message="healthy" if alive and ready else (
                "alive but not ready" if alive else "not alive"
            ),
            data=health,
        )

    def _get_status(self) -> SkillResult:
        """Return detailed lifecycle status and metrics."""
        status = {
            "phase": self._phase.value,
            "uptime_seconds": round(self.uptime_seconds, 1),
            "shutdown_requested": self._shutdown_requested,
            "shutdown_reason": self._shutdown_reason,
            "signals_installed": self._signals_installed,
            "active_tasks": self._active_tasks,
            "registered_hooks": len(self._hooks),
            "ready": self._ready,
            "config": {
                "shutdown_timeout": self._shutdown_timeout,
                "drain_timeout": self._drain_timeout,
            },
        }

        if self._shutdown_start is not None:
            status["shutdown_start"] = self._shutdown_start
            if self._shutdown_complete is not None:
                status["shutdown_duration_seconds"] = round(
                    self._shutdown_complete - self._shutdown_start, 3
                )

        if self._hook_results:
            status["last_shutdown_hooks"] = self._hook_results

        return SkillResult(
            success=True,
            message=f"Phase: {self._phase.value}, "
                    f"Uptime: {self.uptime_seconds:.0f}s, "
                    f"Hooks: {len(self._hooks)}",
            data=status,
        )

    async def _drain(self, params: Dict) -> SkillResult:
        """Enter drain mode — stop accepting new work, wait for in-flight to complete."""
        if self._phase == LifecyclePhase.DRAINING:
            return SkillResult(
                success=True,
                message="Already draining",
                data={"active_tasks": self._active_tasks},
            )

        if self._phase != LifecyclePhase.RUNNING:
            return SkillResult(
                success=False,
                message=f"Cannot drain in phase: {self._phase.value}",
            )

        self._phase = LifecyclePhase.DRAINING
        self._drain_start = time.monotonic()
        drain_timeout = float(params.get("timeout_seconds", self._drain_timeout))

        await self._emit_lifecycle_event("agent.draining", {
            "active_tasks": self._active_tasks,
            "drain_timeout": drain_timeout,
        })

        # Wait for active tasks to complete
        poll_interval = 0.5
        while self._active_tasks > 0:
            elapsed = time.monotonic() - self._drain_start
            if elapsed >= drain_timeout:
                return SkillResult(
                    success=False,
                    message=f"Drain timeout ({drain_timeout}s) exceeded with "
                            f"{self._active_tasks} tasks still active",
                    data={
                        "active_tasks": self._active_tasks,
                        "drain_duration_seconds": round(elapsed, 3),
                        "timed_out": True,
                    },
                )
            await asyncio.sleep(poll_interval)

        drain_duration = time.monotonic() - self._drain_start
        return SkillResult(
            success=True,
            message=f"Drain complete in {drain_duration:.2f}s — all tasks finished",
            data={
                "drain_duration_seconds": round(drain_duration, 3),
                "timed_out": False,
            },
        )

    def _list_hooks(self) -> SkillResult:
        """List all registered shutdown hooks."""
        sorted_hooks = sorted(self._hooks.values(), key=lambda h: h.priority)
        hooks_list = [h.to_dict() for h in sorted_hooks]

        return SkillResult(
            success=True,
            message=f"{len(hooks_list)} shutdown hooks registered",
            data={"hooks": hooks_list},
        )

    async def _emit_lifecycle_event(self, topic: str, data: Dict):
        """Emit a lifecycle event through the EventBus if available."""
        if self._event_bus_ref is not None:
            try:
                await self._event_bus_ref.publish(topic, data, source="lifecycle_manager")
            except Exception:
                pass  # Never let event emission break shutdown

    def _persist_shutdown_state(self):
        """Save shutdown state to disk for post-mortem analysis."""
        try:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            state = {
                "last_shutdown": {
                    "timestamp": datetime.now().isoformat(),
                    "reason": self._shutdown_reason,
                    "phase": self._phase.value,
                    "uptime_seconds": round(self.uptime_seconds, 1),
                    "hooks_registered": len(self._hooks),
                    "hook_results": self._hook_results,
                    "active_tasks_at_shutdown": self._active_tasks,
                },
            }
            with open(LIFECYCLE_STATE_FILE, "w") as f:
                json.dump(state, f, indent=2)
        except Exception:
            pass  # Best-effort persistence

    def get_last_shutdown_info(self) -> Optional[Dict]:
        """Read last shutdown state from disk (for post-restart diagnostics)."""
        try:
            if LIFECYCLE_STATE_FILE.exists():
                with open(LIFECYCLE_STATE_FILE, "r") as f:
                    return json.load(f).get("last_shutdown")
        except Exception:
            pass
        return None
