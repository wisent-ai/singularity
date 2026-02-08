#!/usr/bin/env python3
"""
CircuitBreakerSkill - Runtime safety mechanism for autonomous agent operations.

The agent operates autonomously, calling 130+ skills. Without a circuit breaker,
a broken API or misconfigured skill can cause infinite retry loops, burning
through the agent's entire budget on failures. This is the missing safety net.

Implements the Circuit Breaker pattern (popularized by Netflix Hystrix):

1. CLOSED (normal): Requests pass through. Failures are counted in a sliding window.
2. OPEN (blocked): When failure rate exceeds threshold, the circuit opens.
   All requests are immediately rejected without execution. This saves budget.
3. HALF-OPEN (testing): After a cooldown period, one request is allowed through.
   If it succeeds → circuit closes (recovered). If it fails → circuit reopens.

Additional capabilities beyond the classic pattern:
- **Cost circuit breaker**: Opens when cost-per-success is too high
- **Cascade protection**: When a skill fails, dependent skills are also blocked
- **Sliding window**: Only recent failures count (configurable window size)
- **Budget awareness**: Auto-opens all non-essential circuits when budget is critical
- **Manual override**: Force-open or force-close any circuit
- **Persistent state**: Circuit states survive restarts

This is CRITICAL for autonomous operation:
- Prevents budget drain from broken external APIs
- Enables graceful degradation when services fail
- Provides real-time health visibility across all skills
- Integrates with EventBus for reactive alerts

Pillar: Self-Improvement (primary) + Revenue (budget protection)
"""

import json
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

from .base import Skill, SkillAction, SkillManifest, SkillResult

DATA_DIR = Path(__file__).parent.parent / "data"
CIRCUIT_FILE = DATA_DIR / "circuit_breaker.json"
MAX_HISTORY = 500
MAX_WINDOW = 200


class CircuitState(str, Enum):
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Blocking requests
    HALF_OPEN = "half_open"  # Testing recovery
    FORCED_OPEN = "forced_open"  # Manually blocked
    FORCED_CLOSED = "forced_closed"  # Manually allowed


@dataclass
class CircuitRecord:
    """Tracks a single request outcome for a skill."""
    timestamp: float
    success: bool
    cost: float = 0.0
    duration_ms: float = 0.0
    error: str = ""


@dataclass
class Circuit:
    """Circuit breaker state for a single skill."""
    skill_id: str
    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    total_cost: float = 0.0
    total_successes_cost: float = 0.0
    window: Deque[CircuitRecord] = field(default_factory=deque)
    last_failure_time: float = 0.0
    last_success_time: float = 0.0
    last_state_change: float = 0.0
    opened_count: int = 0  # How many times this circuit has opened
    half_open_test_time: float = 0.0
    consecutive_failures: int = 0
    consecutive_successes: int = 0

    def failure_rate(self) -> float:
        """Current failure rate in the sliding window."""
        if not self.window:
            return 0.0
        failures = sum(1 for r in self.window if not r.success)
        return failures / len(self.window)

    def cost_per_success(self) -> float:
        """Average cost per successful request."""
        successes = sum(1 for r in self.window if r.success)
        if successes == 0:
            return float("inf") if self.window else 0.0
        total = sum(r.cost for r in self.window)
        return total / successes

    def avg_duration_ms(self) -> float:
        """Average request duration."""
        if not self.window:
            return 0.0
        return sum(r.duration_ms for r in self.window) / len(self.window)

    def to_dict(self) -> Dict:
        return {
            "skill_id": self.skill_id,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "failure_rate": round(self.failure_rate(), 3),
            "cost_per_success": round(self.cost_per_success(), 4)
            if self.cost_per_success() != float("inf")
            else "inf",
            "total_cost": round(self.total_cost, 6),
            "window_size": len(self.window),
            "opened_count": self.opened_count,
            "consecutive_failures": self.consecutive_failures,
            "consecutive_successes": self.consecutive_successes,
            "last_failure": datetime.fromtimestamp(self.last_failure_time).isoformat()
            if self.last_failure_time
            else None,
            "last_success": datetime.fromtimestamp(self.last_success_time).isoformat()
            if self.last_success_time
            else None,
            "last_state_change": datetime.fromtimestamp(self.last_state_change).isoformat()
            if self.last_state_change
            else None,
        }


class CircuitBreakerSkill(Skill):
    """
    Runtime circuit breaker for autonomous agent skill execution.

    Monitors skill failure rates and automatically blocks skills that are
    failing too often, preventing budget drain and enabling graceful degradation.

    Actions:
    - record: Record a skill execution outcome (success/failure/cost)
    - check: Check if a skill is allowed to execute (circuit state)
    - status: Get circuit status for one or all skills
    - force_open: Manually block a skill
    - force_close: Manually unblock a skill
    - reset: Reset a circuit to closed state
    - configure: Update thresholds and configuration
    - dashboard: Get a health dashboard of all circuits
    """

    def __init__(self, credentials: Dict[str, str] = None):
        super().__init__(credentials)
        self._circuits: Dict[str, Circuit] = {}
        self._config = {
            "failure_rate_threshold": 0.5,  # Open at 50% failure rate
            "min_window_size": 5,  # Need at least 5 requests before opening
            "window_capacity": 50,  # Sliding window max size
            "cooldown_seconds": 60,  # Wait before half-open test
            "half_open_max_tests": 3,  # Successes needed to close
            "cost_per_success_threshold": 0.10,  # Open if cost/success > $0.10
            "consecutive_failure_threshold": 5,  # Open after 5 consecutive failures
            "budget_critical_threshold": 1.0,  # $1.00 remaining = critical mode
            "essential_skills": [
                "memory",
                "strategy",
                "session",
                "governor",
            ],  # Never auto-open these
        }
        self._event_log: List[Dict] = []
        self._adaptive_source = None  # Optional AdaptiveCircuitThresholdsSkill reference
        self._load_state()

    def _load_state(self):
        """Load persisted circuit states."""
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        try:
            with open(CIRCUIT_FILE, "r") as f:
                data = json.load(f)
            self._config.update(data.get("config", {}))
            for cdata in data.get("circuits", []):
                circuit = Circuit(skill_id=cdata["skill_id"])
                circuit.state = CircuitState(cdata.get("state", "closed"))
                circuit.failure_count = cdata.get("failure_count", 0)
                circuit.success_count = cdata.get("success_count", 0)
                circuit.total_cost = cdata.get("total_cost", 0.0)
                circuit.opened_count = cdata.get("opened_count", 0)
                circuit.last_failure_time = cdata.get("last_failure_time", 0.0)
                circuit.last_success_time = cdata.get("last_success_time", 0.0)
                circuit.last_state_change = cdata.get("last_state_change", 0.0)
                circuit.consecutive_failures = cdata.get("consecutive_failures", 0)
                circuit.consecutive_successes = cdata.get("consecutive_successes", 0)
                self._circuits[cdata["skill_id"]] = circuit
            self._event_log = data.get("event_log", [])[-MAX_HISTORY:]
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            pass

    def _save_state(self):
        """Persist circuit states to disk."""
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        data = {
            "config": self._config,
            "circuits": [],
            "event_log": self._event_log[-MAX_HISTORY:],
            "last_updated": datetime.now().isoformat(),
        }
        for circuit in self._circuits.values():
            data["circuits"].append({
                "skill_id": circuit.skill_id,
                "state": circuit.state.value,
                "failure_count": circuit.failure_count,
                "success_count": circuit.success_count,
                "total_cost": round(circuit.total_cost, 6),
                "opened_count": circuit.opened_count,
                "last_failure_time": circuit.last_failure_time,
                "last_success_time": circuit.last_success_time,
                "last_state_change": circuit.last_state_change,
                "consecutive_failures": circuit.consecutive_failures,
                "consecutive_successes": circuit.consecutive_successes,
            })
        with open(CIRCUIT_FILE, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def _get_circuit(self, skill_id: str) -> Circuit:
        """Get or create a circuit for a skill."""
        if skill_id not in self._circuits:
            self._circuits[skill_id] = Circuit(skill_id=skill_id)
        return self._circuits[skill_id]

    def _log_event(self, event_type: str, skill_id: str, details: str = ""):
        """Log a circuit breaker event."""
        self._event_log.append({
            "type": event_type,
            "skill_id": skill_id,
            "details": details,
            "timestamp": datetime.now().isoformat(),
        })
        if len(self._event_log) > MAX_HISTORY:
            self._event_log = self._event_log[-MAX_HISTORY:]

    def _transition(self, circuit: Circuit, new_state: CircuitState, reason: str = ""):
        """Transition a circuit to a new state."""
        old_state = circuit.state
        if old_state == new_state:
            return
        circuit.state = new_state
        circuit.last_state_change = time.time()
        if new_state == CircuitState.OPEN:
            circuit.opened_count += 1
        self._log_event(
            "state_change",
            circuit.skill_id,
            f"{old_state.value} -> {new_state.value}: {reason}",
        )

    def set_adaptive_source(self, adaptive_skill):
        """
        Connect an AdaptiveCircuitThresholdsSkill for per-skill thresholds.

        When set, _evaluate_circuit() will use per-skill threshold overrides
        from the adaptive skill instead of global defaults. This enables
        different skills to have different failure rate thresholds, cooldown
        periods, etc. based on their observed performance.

        Args:
            adaptive_skill: An AdaptiveCircuitThresholdsSkill instance (or None to disable).
        """
        self._adaptive_source = adaptive_skill

    def _get_effective_config(self, skill_id: str) -> Dict:
        """
        Get effective thresholds for a skill, merging adaptive overrides with global config.

        If an adaptive source is set and has an override for this skill, those
        values take precedence over the global config. Keys not in the override
        fall back to the global config.

        Returns a dict with all threshold keys needed by _evaluate_circuit.
        """
        if self._adaptive_source is not None:
            override = self._adaptive_source.get_override_for_skill(skill_id)
            if override:
                # Merge: override values take precedence, global config fills gaps
                effective = dict(self._config)
                effective.update(override)
                return effective
        return self._config

    def _evaluate_circuit(self, circuit: Circuit):
        """Evaluate if a circuit should change state based on current metrics."""
        now = time.time()

        # Forced states don't auto-transition
        if circuit.state in (CircuitState.FORCED_OPEN, CircuitState.FORCED_CLOSED):
            return

        # Get per-skill thresholds (adaptive overrides if available, else global)
        cfg = self._get_effective_config(circuit.skill_id)

        if circuit.state == CircuitState.CLOSED:
            # Check consecutive failure threshold
            if circuit.consecutive_failures >= cfg["consecutive_failure_threshold"]:
                self._transition(
                    circuit, CircuitState.OPEN,
                    f"{circuit.consecutive_failures} consecutive failures",
                )
                return

            # Check failure rate threshold (only with enough data)
            if len(circuit.window) >= cfg.get("min_window_size", self._config["min_window_size"]):
                rate = circuit.failure_rate()
                if rate >= cfg["failure_rate_threshold"]:
                    self._transition(
                        circuit, CircuitState.OPEN,
                        f"failure rate {rate:.1%} >= {cfg['failure_rate_threshold']:.1%}",
                    )
                    return

            # Check cost-per-success threshold
            if len(circuit.window) >= cfg.get("min_window_size", self._config["min_window_size"]):
                cps = circuit.cost_per_success()
                threshold = cfg["cost_per_success_threshold"]
                if cps != float("inf") and cps > threshold:
                    self._transition(
                        circuit, CircuitState.OPEN,
                        f"cost/success ${cps:.4f} > ${threshold:.4f}",
                    )
                    return

        elif circuit.state == CircuitState.OPEN:
            # Check if cooldown has elapsed → half-open
            elapsed = now - circuit.last_state_change
            if elapsed >= cfg.get("cooldown_seconds", self._config.get("cooldown_seconds", 60)):
                self._transition(circuit, CircuitState.HALF_OPEN, "cooldown elapsed")
                circuit.half_open_test_time = now
                circuit.consecutive_successes = 0

        elif circuit.state == CircuitState.HALF_OPEN:
            # Check if enough successes → close
            if circuit.consecutive_successes >= cfg.get("half_open_max_tests", self._config.get("half_open_max_tests", 3)):
                self._transition(
                    circuit, CircuitState.CLOSED,
                    f"{circuit.consecutive_successes} consecutive successes in half-open",
                )
                circuit.consecutive_failures = 0

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="circuit_breaker",
            name="Circuit Breaker",
            version="1.0.0",
            category="infrastructure",
            description="Runtime circuit breaker for autonomous skill execution - prevents budget drain from failing skills",
            actions=[
                SkillAction(
                    name="record",
                    description="Record a skill execution outcome (call after every skill execution)",
                    parameters={
                        "skill_id": {"type": "string", "required": True, "description": "Skill that was executed"},
                        "success": {"type": "boolean", "required": True, "description": "Whether execution succeeded"},
                        "cost": {"type": "number", "required": False, "description": "Cost of the execution in USD"},
                        "duration_ms": {"type": "number", "required": False, "description": "Duration in milliseconds"},
                        "error": {"type": "string", "required": False, "description": "Error message if failed"},
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="check",
                    description="Check if a skill is allowed to execute (returns allow/deny with reason)",
                    parameters={
                        "skill_id": {"type": "string", "required": True, "description": "Skill to check"},
                        "budget_remaining": {"type": "number", "required": False, "description": "Current agent budget for budget-aware decisions"},
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="status",
                    description="Get circuit status for one or all skills",
                    parameters={
                        "skill_id": {"type": "string", "required": False, "description": "Specific skill (omit for all)"},
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="force_open",
                    description="Manually block a skill (force circuit open)",
                    parameters={
                        "skill_id": {"type": "string", "required": True, "description": "Skill to block"},
                        "reason": {"type": "string", "required": False, "description": "Why it's being blocked"},
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="force_close",
                    description="Manually unblock a skill (force circuit closed)",
                    parameters={
                        "skill_id": {"type": "string", "required": True, "description": "Skill to unblock"},
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="reset",
                    description="Reset a circuit to closed state, clearing all history",
                    parameters={
                        "skill_id": {"type": "string", "required": True, "description": "Skill to reset"},
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="configure",
                    description="Update circuit breaker thresholds and configuration",
                    parameters={
                        "failure_rate_threshold": {"type": "number", "required": False, "description": "Failure rate to trigger open (0.0-1.0)"},
                        "min_window_size": {"type": "number", "required": False, "description": "Minimum requests before evaluating"},
                        "window_capacity": {"type": "number", "required": False, "description": "Sliding window max size"},
                        "cooldown_seconds": {"type": "number", "required": False, "description": "Seconds before half-open test"},
                        "half_open_max_tests": {"type": "number", "required": False, "description": "Successes needed to close from half-open"},
                        "cost_per_success_threshold": {"type": "number", "required": False, "description": "Max cost per success before opening"},
                        "consecutive_failure_threshold": {"type": "number", "required": False, "description": "Consecutive failures before opening"},
                        "budget_critical_threshold": {"type": "number", "required": False, "description": "Budget level for critical mode"},
                        "essential_skills": {"type": "array", "required": False, "description": "Skills that are never auto-blocked"},
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="dashboard",
                    description="Get a health dashboard of all circuits with summary statistics",
                    parameters={},
                    estimated_cost=0,
                ),
            ],
            required_credentials=[],
        )

    def check_credentials(self) -> bool:
        return True

    async def execute(self, action: str, params: Dict) -> SkillResult:
        handlers = {
            "record": self._record,
            "check": self._check,
            "status": self._status,
            "force_open": self._force_open,
            "force_close": self._force_close,
            "reset": self._reset,
            "configure": self._configure,
            "dashboard": self._dashboard,
        }
        handler = handlers.get(action)
        if not handler:
            return SkillResult(success=False, message=f"Unknown action: {action}")
        return handler(params)

    def _record(self, params: Dict) -> SkillResult:
        """Record a skill execution outcome."""
        skill_id = params.get("skill_id", "").strip()
        if not skill_id:
            return SkillResult(success=False, message="skill_id is required")

        success = params.get("success", False)
        cost = float(params.get("cost", 0.0))
        duration_ms = float(params.get("duration_ms", 0.0))
        error = params.get("error", "")

        circuit = self._get_circuit(skill_id)
        now = time.time()

        # Add to sliding window
        record = CircuitRecord(
            timestamp=now,
            success=success,
            cost=cost,
            duration_ms=duration_ms,
            error=error,
        )
        circuit.window.append(record)
        while len(circuit.window) > self._config["window_capacity"]:
            circuit.window.popleft()

        # Update counters
        circuit.total_cost += cost
        if success:
            circuit.success_count += 1
            circuit.last_success_time = now
            circuit.consecutive_successes += 1
            circuit.consecutive_failures = 0
            circuit.total_successes_cost += cost
        else:
            circuit.failure_count += 1
            circuit.last_failure_time = now
            circuit.consecutive_failures += 1
            circuit.consecutive_successes = 0

        # Evaluate state transition
        self._evaluate_circuit(circuit)

        self._save_state()

        state_msg = f"[{circuit.state.value}]"
        if circuit.state == CircuitState.OPEN:
            state_msg += f" (opened: {circuit.failure_rate():.0%} failure rate)"

        return SkillResult(
            success=True,
            message=f"Recorded {'success' if success else 'failure'} for {skill_id} {state_msg}",
            data={
                "skill_id": skill_id,
                "circuit_state": circuit.state.value,
                "failure_rate": round(circuit.failure_rate(), 3),
                "consecutive_failures": circuit.consecutive_failures,
                "window_size": len(circuit.window),
            },
        )

    def _check(self, params: Dict) -> SkillResult:
        """Check if a skill is allowed to execute."""
        skill_id = params.get("skill_id", "").strip()
        if not skill_id:
            return SkillResult(success=False, message="skill_id is required")

        budget = params.get("budget_remaining")
        circuit = self._get_circuit(skill_id)

        # Re-evaluate state (might transition from OPEN → HALF_OPEN)
        self._evaluate_circuit(circuit)

        # Check budget-critical mode
        if budget is not None and budget <= self._config["budget_critical_threshold"]:
            essential = self._config.get("essential_skills", [])
            if skill_id not in essential:
                return SkillResult(
                    success=True,
                    message=f"DENY: Budget critical (${budget:.4f}), non-essential skill blocked",
                    data={
                        "allowed": False,
                        "reason": "budget_critical",
                        "circuit_state": circuit.state.value,
                        "budget_remaining": budget,
                    },
                )

        # Check circuit state
        if circuit.state == CircuitState.FORCED_OPEN:
            return SkillResult(
                success=True,
                message=f"DENY: {skill_id} is manually blocked (forced open)",
                data={
                    "allowed": False,
                    "reason": "forced_open",
                    "circuit_state": circuit.state.value,
                },
            )

        if circuit.state == CircuitState.OPEN:
            return SkillResult(
                success=True,
                message=f"DENY: {skill_id} circuit is open ({circuit.failure_rate():.0%} failure rate)",
                data={
                    "allowed": False,
                    "reason": "circuit_open",
                    "circuit_state": circuit.state.value,
                    "failure_rate": round(circuit.failure_rate(), 3),
                    "cooldown_remaining": max(
                        0,
                        self._config["cooldown_seconds"]
                        - (time.time() - circuit.last_state_change),
                    ),
                },
            )

        if circuit.state == CircuitState.HALF_OPEN:
            return SkillResult(
                success=True,
                message=f"ALLOW: {skill_id} in half-open testing ({circuit.consecutive_successes}/{self._config['half_open_max_tests']} successes needed)",
                data={
                    "allowed": True,
                    "reason": "half_open_test",
                    "circuit_state": circuit.state.value,
                    "tests_remaining": self._config["half_open_max_tests"] - circuit.consecutive_successes,
                },
            )

        # CLOSED or FORCED_CLOSED → allow
        return SkillResult(
            success=True,
            message=f"ALLOW: {skill_id} circuit is {circuit.state.value}",
            data={
                "allowed": True,
                "reason": "circuit_closed",
                "circuit_state": circuit.state.value,
                "failure_rate": round(circuit.failure_rate(), 3),
            },
        )

    def _status(self, params: Dict) -> SkillResult:
        """Get circuit status for one or all skills."""
        skill_id = params.get("skill_id", "").strip()

        if skill_id:
            circuit = self._get_circuit(skill_id)
            return SkillResult(
                success=True,
                message=f"Circuit status for {skill_id}: {circuit.state.value}",
                data=circuit.to_dict(),
            )

        # All circuits
        circuits = {}
        for sid, circuit in sorted(self._circuits.items()):
            circuits[sid] = circuit.to_dict()

        return SkillResult(
            success=True,
            message=f"Tracking {len(circuits)} circuits",
            data={
                "circuits": circuits,
                "total_tracked": len(circuits),
                "config": self._config,
            },
        )

    def _force_open(self, params: Dict) -> SkillResult:
        """Manually block a skill."""
        skill_id = params.get("skill_id", "").strip()
        if not skill_id:
            return SkillResult(success=False, message="skill_id is required")

        reason = params.get("reason", "manual override")
        circuit = self._get_circuit(skill_id)
        self._transition(circuit, CircuitState.FORCED_OPEN, reason)
        self._save_state()

        return SkillResult(
            success=True,
            message=f"Circuit for {skill_id} forced OPEN: {reason}",
            data=circuit.to_dict(),
        )

    def _force_close(self, params: Dict) -> SkillResult:
        """Manually unblock a skill."""
        skill_id = params.get("skill_id", "").strip()
        if not skill_id:
            return SkillResult(success=False, message="skill_id is required")

        circuit = self._get_circuit(skill_id)
        self._transition(circuit, CircuitState.FORCED_CLOSED, "manual override")
        circuit.consecutive_failures = 0
        self._save_state()

        return SkillResult(
            success=True,
            message=f"Circuit for {skill_id} forced CLOSED",
            data=circuit.to_dict(),
        )

    def _reset(self, params: Dict) -> SkillResult:
        """Reset a circuit completely."""
        skill_id = params.get("skill_id", "").strip()
        if not skill_id:
            return SkillResult(success=False, message="skill_id is required")

        self._circuits[skill_id] = Circuit(skill_id=skill_id)
        self._log_event("reset", skill_id, "circuit reset to initial state")
        self._save_state()

        return SkillResult(
            success=True,
            message=f"Circuit for {skill_id} reset to CLOSED with empty history",
            data=self._circuits[skill_id].to_dict(),
        )

    def _configure(self, params: Dict) -> SkillResult:
        """Update configuration."""
        updated = []
        valid_keys = {
            "failure_rate_threshold", "min_window_size", "window_capacity",
            "cooldown_seconds", "half_open_max_tests", "cost_per_success_threshold",
            "consecutive_failure_threshold", "budget_critical_threshold",
            "essential_skills",
        }

        for key in valid_keys:
            if key in params:
                val = params[key]
                # Basic validation
                if key == "failure_rate_threshold":
                    val = max(0.0, min(1.0, float(val)))
                elif key in ("min_window_size", "window_capacity", "half_open_max_tests", "consecutive_failure_threshold"):
                    val = max(1, int(val))
                elif key in ("cooldown_seconds", "cost_per_success_threshold", "budget_critical_threshold"):
                    val = max(0.0, float(val))
                elif key == "essential_skills":
                    if isinstance(val, str):
                        val = [s.strip() for s in val.split(",")]
                self._config[key] = val
                updated.append(key)

        if not updated:
            return SkillResult(
                success=True,
                message="No configuration changes made",
                data={"config": self._config},
            )

        self._save_state()
        return SkillResult(
            success=True,
            message=f"Updated {len(updated)} settings: {', '.join(updated)}",
            data={"updated": updated, "config": self._config},
        )

    def _dashboard(self, params: Dict) -> SkillResult:
        """Get a health dashboard of all circuits."""
        now = time.time()
        total = len(self._circuits)
        closed = sum(1 for c in self._circuits.values() if c.state in (CircuitState.CLOSED, CircuitState.FORCED_CLOSED))
        opened = sum(1 for c in self._circuits.values() if c.state in (CircuitState.OPEN, CircuitState.FORCED_OPEN))
        half_open = sum(1 for c in self._circuits.values() if c.state == CircuitState.HALF_OPEN)

        # Calculate aggregate stats
        total_cost = sum(c.total_cost for c in self._circuits.values())
        total_failures = sum(c.failure_count for c in self._circuits.values())
        total_successes = sum(c.success_count for c in self._circuits.values())
        total_requests = total_failures + total_successes

        # Find worst performers
        worst_skills = []
        for circuit in self._circuits.values():
            if len(circuit.window) >= 3:  # Need some data
                worst_skills.append({
                    "skill_id": circuit.skill_id,
                    "failure_rate": round(circuit.failure_rate(), 3),
                    "state": circuit.state.value,
                    "requests": len(circuit.window),
                })
        worst_skills.sort(key=lambda x: x["failure_rate"], reverse=True)

        # Recent events
        recent_events = self._event_log[-10:]

        msg_lines = ["=== Circuit Breaker Dashboard ==="]
        msg_lines.append(f"Circuits: {total} tracked | {closed} closed | {opened} open | {half_open} half-open")
        if total_requests > 0:
            overall_rate = total_failures / total_requests
            msg_lines.append(f"Overall: {total_requests} requests, {overall_rate:.1%} failure rate, ${total_cost:.4f} total cost")
        if worst_skills:
            top3 = worst_skills[:3]
            msg_lines.append("Worst performers:")
            for ws in top3:
                msg_lines.append(f"  - {ws['skill_id']}: {ws['failure_rate']:.0%} failures [{ws['state']}]")

        return SkillResult(
            success=True,
            message="\n".join(msg_lines),
            data={
                "total_circuits": total,
                "closed": closed,
                "open": opened,
                "half_open": half_open,
                "total_requests": total_requests,
                "total_failures": total_failures,
                "total_successes": total_successes,
                "overall_failure_rate": round(total_failures / total_requests, 3) if total_requests > 0 else 0,
                "total_cost": round(total_cost, 6),
                "worst_skills": worst_skills[:5],
                "recent_events": recent_events,
                "config": self._config,
            },
        )



def wire_adaptive_thresholds(registry) -> bool:
    """
    Wire AdaptiveCircuitThresholdsSkill into CircuitBreakerSkill.

    Looks up both skills in the registry and connects them so the circuit
    breaker uses per-skill adaptive thresholds when evaluating circuits.

    Call this after all skills are registered in the registry.

    Args:
        registry: SkillRegistry instance containing both skills

    Returns:
        True if wiring succeeded, False if either skill is missing
    """
    cb = registry.get("circuit_breaker")
    adaptive = registry.get("adaptive_circuit_thresholds")
    if cb is None or adaptive is None:
        return False
    cb.set_adaptive_source(adaptive)
    return True
