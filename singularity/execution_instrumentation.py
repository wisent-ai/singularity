#!/usr/bin/env python3
"""
Execution Instrumentation - Wires ObservabilitySkill + SkillEventBridge into the agent execution loop.

Every skill action executed by the AutonomousAgent is automatically instrumented:
1. **Metrics** - Latency, execution count, error count emitted to ObservabilitySkill
2. **Bridge Events** - SkillEventBridge.emit_bridge_events() called for reactive cross-skill automation
3. **EventBus Events** - skill.executed events published for real-time subscribers

This closes the critical "act -> measure -> adapt" feedback loop:
- Skills execute actions
- ObservabilitySkill records metrics (latency, success rate, error patterns)
- Alerts fire when thresholds are breached
- SkillEventBridge triggers reactive behaviors (e.g., high error rate -> incident)
- The agent can query its own performance and adapt

Pillar: Self-Improvement (completes the feedback loop from execution to measurement)
"""

import time
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .autonomous_agent import AutonomousAgent


class ExecutionInstrumentation:
    """
    Instruments skill execution with metrics, bridge events, and EventBus notifications.

    Usage:
        instrumentation = ExecutionInstrumentation(agent)
        # In agent's _execute method, wrap skill execution:
        result = await instrumentation.instrumented_execute(skill_id, action, params, execute_fn)
    """

    def __init__(self, agent: "AutonomousAgent"):
        self._agent = agent
        self._observability = None
        self._bridge = None
        self._initialized = False
        self._metrics_buffer: List[Dict] = []
        self._buffer_limit = 20
        self._total_instrumented = 0
        self._total_errors = 0

    def _lazy_init(self):
        """Lazily discover ObservabilitySkill and SkillEventBridgeSkill from agent's registry."""
        if self._initialized:
            return
        self._initialized = True
        # Avoid circular import
        from .skills.observability import ObservabilitySkill
        from .skills.skill_event_bridge import SkillEventBridgeSkill

        for skill in self._agent.skills.skills.values():
            if isinstance(skill, ObservabilitySkill):
                self._observability = skill
            elif isinstance(skill, SkillEventBridgeSkill):
                self._bridge = skill

    async def instrumented_execute(
        self,
        skill_id: str,
        action_name: str,
        params: Dict,
        execute_fn: Callable,
    ) -> Dict:
        """
        Wrap a skill execution with instrumentation.

        Args:
            skill_id: The skill being executed (e.g., "code_review")
            action_name: The action within the skill (e.g., "review")
            params: Parameters passed to the action
            execute_fn: Async callable that performs the actual execution

        Returns:
            The result dict from execute_fn, unchanged.
        """
        self._lazy_init()

        start_time = time.monotonic()
        result = None
        error_occurred = False

        try:
            result = await execute_fn()
            return result
        except Exception as e:
            error_occurred = True
            result = {"status": "error", "message": str(e)}
            raise
        finally:
            elapsed_ms = (time.monotonic() - start_time) * 1000
            success = (
                result is not None
                and isinstance(result, dict)
                and result.get("status") == "success"
            )
            self._total_instrumented += 1
            if error_occurred or not success:
                self._total_errors += 1

            # Non-blocking instrumentation - never let metric emission break execution
            try:
                await self._emit_metrics(skill_id, action_name, elapsed_ms, success, error_occurred)
            except Exception:
                pass

            try:
                await self._emit_bridge_events(skill_id, action_name, result or {})
            except Exception:
                pass

            try:
                await self._emit_execution_event(skill_id, action_name, elapsed_ms, success)
            except Exception:
                pass

    async def _emit_metrics(
        self,
        skill_id: str,
        action_name: str,
        elapsed_ms: float,
        success: bool,
        error: bool,
    ):
        """Emit execution metrics to ObservabilitySkill."""
        if not self._observability:
            return

        labels = {"skill": skill_id, "action": action_name}

        # Execution count (counter)
        self._observability._emit({
            "name": "skill.execution.count",
            "value": 1,
            "metric_type": "counter",
            "labels": labels,
        })

        # Latency (histogram)
        self._observability._emit({
            "name": "skill.execution.latency_ms",
            "value": elapsed_ms,
            "metric_type": "histogram",
            "labels": labels,
        })

        # Error count (counter) - only when error
        if error or not success:
            self._observability._emit({
                "name": "skill.execution.errors",
                "value": 1,
                "metric_type": "counter",
                "labels": labels,
            })

        # Success count (counter)
        if success:
            self._observability._emit({
                "name": "skill.execution.success",
                "value": 1,
                "metric_type": "counter",
                "labels": labels,
            })

    async def _emit_bridge_events(
        self,
        skill_id: str,
        action_name: str,
        result: Dict,
    ):
        """Call SkillEventBridge to emit any configured bridge events."""
        if not self._bridge:
            return

        result_data = result.get("data", {}) if isinstance(result, dict) else {}
        if not isinstance(result_data, dict):
            result_data = {}

        await self._bridge.emit_bridge_events(skill_id, action_name, result_data)

    async def _emit_execution_event(
        self,
        skill_id: str,
        action_name: str,
        elapsed_ms: float,
        success: bool,
    ):
        """Emit a skill.executed event to the EventBus."""
        await self._agent._emit_event(
            topic="skill.executed",
            data={
                "skill": skill_id,
                "action": action_name,
                "latency_ms": round(elapsed_ms, 2),
                "success": success,
            },
        )

    def get_stats(self) -> Dict:
        """Return instrumentation statistics."""
        return {
            "total_instrumented": self._total_instrumented,
            "total_errors": self._total_errors,
            "observability_connected": self._observability is not None,
            "bridge_connected": self._bridge is not None,
            "error_rate": (
                round(self._total_errors / self._total_instrumented, 4)
                if self._total_instrumented > 0
                else 0.0
            ),
        }
