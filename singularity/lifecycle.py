#!/usr/bin/env python3
"""
Agent Lifecycle Hooks

A plugin system for the agent's run loop. Hooks can run code at key points
in the agent's lifecycle without modifying the core run loop.

Lifecycle events:
    on_startup   - Agent is starting, before first cycle
    pre_cycle    - Before each decision/action cycle
    post_cycle   - After each cycle completes (with results)
    on_shutdown  - Agent is stopping

Built-in hooks:
    OutcomeTrackingHook  - Tracks action success/failure rates
    CycleMetricsHook     - Tracks timing, cost trends, efficiency
    AdaptiveIntervalHook - Adjusts cycle interval based on activity
"""

from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
import time


@dataclass
class StartupInfo:
    """Information passed to hooks on agent startup."""
    agent_name: str
    agent_ticker: str
    agent_type: str
    balance: float
    instance_type: str
    skill_count: int
    skill_ids: List[str]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class CycleInfo:
    """Information passed to hooks before each cycle."""
    cycle: int
    balance: float
    runway_cycles: float
    runway_hours: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class CycleResult:
    """Information passed to hooks after each cycle."""
    cycle: int
    tool: str
    params: Dict
    result: Dict
    success: bool
    api_cost: float
    tokens_used: int
    duration_seconds: float
    balance_after: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class ShutdownInfo:
    """Information passed to hooks on agent shutdown."""
    total_cycles: int
    total_api_cost: float
    total_tokens: int
    balance_remaining: float
    runtime_seconds: float
    reason: str = "normal"  # normal, out_of_funds, error, manual_stop
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class LifecycleHook(ABC):
    """
    Base class for agent lifecycle hooks.

    Subclass and override methods to run code at lifecycle events.
    All methods receive structured info about the agent's state.
    """

    @property
    def name(self) -> str:
        """Hook identifier."""
        return self.__class__.__name__

    def on_startup(self, info: StartupInfo) -> None:
        """Called once when the agent starts, before the first cycle."""
        pass

    def pre_cycle(self, info: CycleInfo) -> Optional[Dict[str, Any]]:
        """
        Called before each decision cycle.

        Returns:
            Optional dict of context to inject into the cycle.
            Keys can include:
                - 'context_hint': str to append to LLM context
                - 'skip_cycle': bool to skip this cycle
        """
        return None

    def post_cycle(self, result: CycleResult) -> None:
        """Called after each cycle completes."""
        pass

    def on_shutdown(self, info: ShutdownInfo) -> None:
        """Called once when the agent shuts down."""
        pass

    def get_status(self) -> Dict[str, Any]:
        """Return current hook status/metrics for inspection."""
        return {"name": self.name, "active": True}


class HookManager:
    """
    Manages lifecycle hooks and dispatches events.

    Usage:
        manager = HookManager()
        manager.add(OutcomeTrackingHook())
        manager.add(CycleMetricsHook())

        # In agent run loop:
        manager.dispatch_startup(info)
        while running:
            hints = manager.dispatch_pre_cycle(info)
            # ... execute cycle ...
            manager.dispatch_post_cycle(result)
        manager.dispatch_shutdown(info)
    """

    def __init__(self):
        self._hooks: List[LifecycleHook] = []

    def add(self, hook: LifecycleHook) -> None:
        """Add a hook to the manager."""
        self._hooks.append(hook)

    def remove(self, hook_name: str) -> bool:
        """Remove a hook by name. Returns True if removed."""
        for i, hook in enumerate(self._hooks):
            if hook.name == hook_name:
                self._hooks.pop(i)
                return True
        return False

    def get(self, hook_name: str) -> Optional[LifecycleHook]:
        """Get a hook by name."""
        for hook in self._hooks:
            if hook.name == hook_name:
                return hook
        return None

    @property
    def hooks(self) -> List[LifecycleHook]:
        """List all registered hooks."""
        return list(self._hooks)

    def dispatch_startup(self, info: StartupInfo) -> None:
        """Dispatch startup event to all hooks."""
        for hook in self._hooks:
            try:
                hook.on_startup(info)
            except Exception:
                pass  # Hooks should not crash the agent

    def dispatch_pre_cycle(self, info: CycleInfo) -> Dict[str, Any]:
        """
        Dispatch pre-cycle event to all hooks.

        Returns merged context hints from all hooks.
        """
        merged: Dict[str, Any] = {}
        hints_list: List[str] = []

        for hook in self._hooks:
            try:
                result = hook.pre_cycle(info)
                if result:
                    if 'context_hint' in result:
                        hints_list.append(result['context_hint'])
                    if result.get('skip_cycle'):
                        merged['skip_cycle'] = True
                    # Merge other keys
                    for k, v in result.items():
                        if k not in ('context_hint', 'skip_cycle'):
                            merged[k] = v
            except Exception:
                pass

        if hints_list:
            merged['context_hints'] = hints_list

        return merged

    def dispatch_post_cycle(self, result: CycleResult) -> None:
        """Dispatch post-cycle event to all hooks."""
        for hook in self._hooks:
            try:
                hook.post_cycle(result)
            except Exception:
                pass

    def dispatch_shutdown(self, info: ShutdownInfo) -> None:
        """Dispatch shutdown event to all hooks."""
        for hook in self._hooks:
            try:
                hook.on_shutdown(info)
            except Exception:
                pass

    def get_all_status(self) -> List[Dict[str, Any]]:
        """Get status from all hooks."""
        statuses = []
        for hook in self._hooks:
            try:
                statuses.append(hook.get_status())
            except Exception:
                statuses.append({"name": hook.name, "error": "status failed"})
        return statuses


# ============================================================
# Built-in Hooks
# ============================================================


class OutcomeTrackingHook(LifecycleHook):
    """
    Tracks action outcomes (success/failure) with rolling statistics.

    Provides:
    - Overall success rate
    - Per-skill success rates
    - Recent failure patterns
    - Streak tracking (consecutive successes/failures)
    """

    def __init__(self, window_size: int = 100):
        self._window_size = window_size
        self._outcomes: deque = deque(maxlen=window_size)
        self._skill_outcomes: Dict[str, deque] = {}
        self._total_success = 0
        self._total_failure = 0
        self._current_streak = 0  # positive = success streak, negative = failure
        self._best_streak = 0
        self._worst_streak = 0
        self._recent_failures: deque = deque(maxlen=10)

    def post_cycle(self, result: CycleResult) -> None:
        success = result.success
        self._outcomes.append(success)

        if success:
            self._total_success += 1
            self._current_streak = max(1, self._current_streak + 1) if self._current_streak >= 0 else 1
            self._best_streak = max(self._best_streak, self._current_streak)
        else:
            self._total_failure += 1
            self._current_streak = min(-1, self._current_streak - 1) if self._current_streak <= 0 else -1
            self._worst_streak = min(self._worst_streak, self._current_streak)
            self._recent_failures.append({
                "cycle": result.cycle,
                "tool": result.tool,
                "timestamp": result.timestamp,
            })

        # Track per-skill outcomes
        skill_id = result.tool.split(":")[0] if ":" in result.tool else result.tool
        if skill_id not in self._skill_outcomes:
            self._skill_outcomes[skill_id] = deque(maxlen=50)
        self._skill_outcomes[skill_id].append(success)

    def pre_cycle(self, info: CycleInfo) -> Optional[Dict[str, Any]]:
        # After 5+ cycles, if failure rate is high, add a hint
        if len(self._outcomes) >= 5:
            recent = list(self._outcomes)[-5:]
            recent_failures = sum(1 for x in recent if not x)
            if recent_failures >= 3:
                failing_skills = []
                for skill_id, outcomes in self._skill_outcomes.items():
                    recent_skill = list(outcomes)[-3:]
                    if recent_skill and sum(1 for x in recent_skill if not x) >= 2:
                        failing_skills.append(skill_id)
                hint = f"WARNING: {recent_failures}/5 recent actions failed."
                if failing_skills:
                    hint += f" Struggling skills: {', '.join(failing_skills)}."
                hint += " Consider trying a different approach."
                return {"context_hint": hint}
        return None

    @property
    def success_rate(self) -> float:
        """Overall success rate (0.0 to 1.0)."""
        total = self._total_success + self._total_failure
        if total == 0:
            return 0.0
        return self._total_success / total

    @property
    def windowed_success_rate(self) -> float:
        """Success rate within the rolling window."""
        if not self._outcomes:
            return 0.0
        return sum(1 for x in self._outcomes if x) / len(self._outcomes)

    def skill_success_rate(self, skill_id: str) -> float:
        """Success rate for a specific skill."""
        outcomes = self._skill_outcomes.get(skill_id)
        if not outcomes:
            return 0.0
        return sum(1 for x in outcomes if x) / len(outcomes)

    def get_status(self) -> Dict[str, Any]:
        skill_rates = {}
        for skill_id, outcomes in self._skill_outcomes.items():
            if outcomes:
                skill_rates[skill_id] = round(
                    sum(1 for x in outcomes if x) / len(outcomes), 3
                )

        return {
            "name": self.name,
            "active": True,
            "total_actions": self._total_success + self._total_failure,
            "success_rate": round(self.success_rate, 3),
            "windowed_success_rate": round(self.windowed_success_rate, 3),
            "current_streak": self._current_streak,
            "best_streak": self._best_streak,
            "worst_streak": self._worst_streak,
            "recent_failures": list(self._recent_failures),
            "skill_success_rates": skill_rates,
        }


class CycleMetricsHook(LifecycleHook):
    """
    Tracks cycle-level performance metrics.

    Provides:
    - Cycle duration statistics (min, max, avg, p95)
    - Cost efficiency tracking
    - Token usage trends
    - Throughput metrics
    """

    def __init__(self, window_size: int = 100):
        self._window_size = window_size
        self._durations: deque = deque(maxlen=window_size)
        self._costs: deque = deque(maxlen=window_size)
        self._tokens: deque = deque(maxlen=window_size)
        self._start_time: Optional[float] = None
        self._start_balance: float = 0.0
        self._total_cycles = 0

    def on_startup(self, info: StartupInfo) -> None:
        self._start_time = time.time()
        self._start_balance = info.balance

    def post_cycle(self, result: CycleResult) -> None:
        self._total_cycles += 1
        self._durations.append(result.duration_seconds)
        self._costs.append(result.api_cost)
        self._tokens.append(result.tokens_used)

    def pre_cycle(self, info: CycleInfo) -> Optional[Dict[str, Any]]:
        # Warn if costs are escalating
        if len(self._costs) >= 10:
            recent_5 = list(self._costs)[-5:]
            earlier_5 = list(self._costs)[-10:-5]
            avg_recent = sum(recent_5) / len(recent_5) if recent_5 else 0
            avg_earlier = sum(earlier_5) / len(earlier_5) if earlier_5 else 0
            if avg_earlier > 0 and avg_recent > avg_earlier * 2:
                return {
                    "context_hint": f"COST ALERT: Average cost per cycle increased from ${avg_earlier:.4f} to ${avg_recent:.4f}. Consider using cheaper actions."
                }
        return None

    @property
    def avg_duration(self) -> float:
        if not self._durations:
            return 0.0
        return sum(self._durations) / len(self._durations)

    @property
    def avg_cost(self) -> float:
        if not self._costs:
            return 0.0
        return sum(self._costs) / len(self._costs)

    @property
    def avg_tokens(self) -> float:
        if not self._tokens:
            return 0.0
        return sum(self._tokens) / len(self._tokens)

    @property
    def total_runtime(self) -> float:
        if self._start_time is None:
            return 0.0
        return time.time() - self._start_time

    @property
    def cost_efficiency(self) -> float:
        """Cost per successful action (lower is better). Returns -1 if no data."""
        if not self._costs:
            return -1.0
        return sum(self._costs) / len(self._costs)

    def get_status(self) -> Dict[str, Any]:
        durations = sorted(self._durations) if self._durations else []
        costs = sorted(self._costs) if self._costs else []

        def percentile(data, p):
            if not data:
                return 0.0
            k = (len(data) - 1) * p
            f = int(k)
            c = f + 1
            if c >= len(data):
                return data[f]
            return data[f] + (k - f) * (data[c] - data[f])

        return {
            "name": self.name,
            "active": True,
            "total_cycles": self._total_cycles,
            "runtime_seconds": round(self.total_runtime, 1),
            "duration": {
                "avg": round(self.avg_duration, 3),
                "min": round(min(durations), 3) if durations else 0,
                "max": round(max(durations), 3) if durations else 0,
                "p95": round(percentile(durations, 0.95), 3),
            },
            "cost": {
                "avg_per_cycle": round(self.avg_cost, 6),
                "total": round(sum(self._costs), 4) if self._costs else 0,
                "min": round(min(costs), 6) if costs else 0,
                "max": round(max(costs), 6) if costs else 0,
            },
            "tokens": {
                "avg_per_cycle": round(self.avg_tokens, 0),
                "total": sum(self._tokens) if self._tokens else 0,
            },
            "spend_rate_per_hour": round(
                sum(self._costs) / (self.total_runtime / 3600), 4
            ) if self.total_runtime > 0 and self._costs else 0,
        }


class AdaptiveIntervalHook(LifecycleHook):
    """
    Suggests cycle interval adjustments based on activity patterns.

    When actions consistently succeed quickly, suggests shorter intervals.
    When actions fail or are slow, suggests backing off.
    Does NOT modify the interval directly - just provides hints.
    """

    def __init__(self, min_interval: float = 1.0, max_interval: float = 60.0):
        self._min_interval = min_interval
        self._max_interval = max_interval
        self._recent_durations: deque = deque(maxlen=20)
        self._recent_successes: deque = deque(maxlen=20)
        self._suggested_interval: Optional[float] = None

    def post_cycle(self, result: CycleResult) -> None:
        self._recent_durations.append(result.duration_seconds)
        self._recent_successes.append(result.success)

        if len(self._recent_successes) >= 5:
            success_rate = sum(1 for x in self._recent_successes if x) / len(self._recent_successes)
            avg_duration = sum(self._recent_durations) / len(self._recent_durations)

            if success_rate > 0.8 and avg_duration < 5.0:
                # Things are going well - speed up
                self._suggested_interval = max(self._min_interval, avg_duration * 0.5)
            elif success_rate < 0.5:
                # Things are failing - slow down
                self._suggested_interval = min(self._max_interval, avg_duration * 2.0)
            else:
                self._suggested_interval = None

    def pre_cycle(self, info: CycleInfo) -> Optional[Dict[str, Any]]:
        if self._suggested_interval is not None:
            return {"suggested_interval": self._suggested_interval}
        return None

    def get_status(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "active": True,
            "suggested_interval": self._suggested_interval,
            "recent_success_rate": (
                round(sum(1 for x in self._recent_successes if x) / len(self._recent_successes), 3)
                if self._recent_successes else 0.0
            ),
            "avg_duration": (
                round(sum(self._recent_durations) / len(self._recent_durations), 3)
                if self._recent_durations else 0.0
            ),
        }
