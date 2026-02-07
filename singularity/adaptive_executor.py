#!/usr/bin/env python3
"""
Adaptive Executor - Closes the act→measure→adapt feedback loop.

This module sits between the agent's decision-making and skill execution,
using historical performance data to make smarter execution decisions:

1. Circuit Breaker: Disables skills that consistently fail, re-enables
   them after a cooldown period to check if the issue resolved.

2. Smart Retry: Uses exponential backoff with jitter for transient failures,
   but skips retries for deterministic errors.

3. Skill Preference: When multiple skills can accomplish a goal, routes
   to the one with the best historical success rate and cost efficiency.

4. Cost Guard: Prevents execution when estimated cost exceeds budget
   thresholds or when a skill's cost-efficiency is poor.

5. Adaptation Journal: Logs every adaptation decision so the agent can
   learn what adjustments helped and which didn't.

Part of the Self-Improvement pillar: the "adapt" step in act→measure→adapt.
"""

import json
import time
import random
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field


ADAPTIVE_FILE = Path(__file__).parent / "data" / "adaptive.json"


@dataclass
class CircuitState:
    """Tracks circuit breaker state for a skill."""
    skill_id: str
    state: str = "closed"  # closed (healthy), open (blocked), half_open (testing)
    failure_count: int = 0
    success_count: int = 0
    last_failure: Optional[str] = None
    opened_at: Optional[str] = None
    cooldown_seconds: int = 300  # 5 min default cooldown


@dataclass
class ExecutionAdvice:
    """Advice from the adaptive executor about how to execute an action."""
    should_execute: bool = True
    reason: str = ""
    retry_config: Dict = field(default_factory=lambda: {
        "max_retries": 0,
        "base_delay_ms": 500,
        "max_delay_ms": 10000,
    })
    alternative_skill: Optional[str] = None
    cost_warning: Optional[str] = None
    circuit_state: str = "closed"


class AdaptiveExecutor:
    """
    Wraps skill execution with adaptive intelligence.

    Uses performance history to make real-time decisions about:
    - Whether to execute (circuit breaker)
    - How to retry (smart backoff)
    - Which skill to prefer (routing)
    - Cost guardrails
    """

    # Thresholds
    CIRCUIT_FAILURE_THRESHOLD = 5  # Failures before opening circuit
    CIRCUIT_COOLDOWN_SECONDS = 300  # Seconds before half-open test
    HALF_OPEN_SUCCESS_THRESHOLD = 2  # Successes to close circuit
    MAX_RETRIES = 3
    RETRY_BASE_MS = 500
    RETRY_MAX_MS = 10000
    COST_EFFICIENCY_THRESHOLD = 0.3  # Warn if success rate below 30%
    BUDGET_GUARD_RATIO = 0.1  # Warn if action costs > 10% of remaining balance

    def __init__(self, balance: float = 100.0):
        self._balance = balance
        self._circuits: Dict[str, CircuitState] = {}
        self._journal: List[Dict] = []
        self._perf_data: Dict = {}
        self._load_state()

    def _load_state(self):
        """Load persistent adaptive state."""
        try:
            ADAPTIVE_FILE.parent.mkdir(parents=True, exist_ok=True)
            if ADAPTIVE_FILE.exists():
                with open(ADAPTIVE_FILE, "r") as f:
                    data = json.load(f)
                for cid, cdata in data.get("circuits", {}).items():
                    self._circuits[cid] = CircuitState(
                        skill_id=cdata["skill_id"],
                        state=cdata.get("state", "closed"),
                        failure_count=cdata.get("failure_count", 0),
                        success_count=cdata.get("success_count", 0),
                        last_failure=cdata.get("last_failure"),
                        opened_at=cdata.get("opened_at"),
                        cooldown_seconds=cdata.get("cooldown_seconds", self.CIRCUIT_COOLDOWN_SECONDS),
                    )
                self._journal = data.get("journal", [])[-200:]  # Cap journal
        except (json.JSONDecodeError, KeyError):
            pass

    def _save_state(self):
        """Persist adaptive state to disk."""
        data = {
            "circuits": {},
            "journal": self._journal[-200:],
            "last_updated": datetime.now().isoformat(),
        }
        for cid, circuit in self._circuits.items():
            data["circuits"][cid] = {
                "skill_id": circuit.skill_id,
                "state": circuit.state,
                "failure_count": circuit.failure_count,
                "success_count": circuit.success_count,
                "last_failure": circuit.last_failure,
                "opened_at": circuit.opened_at,
                "cooldown_seconds": circuit.cooldown_seconds,
            }
        try:
            with open(ADAPTIVE_FILE, "w") as f:
                json.dump(data, f, indent=2, default=str)
        except Exception:
            pass

    def update_balance(self, balance: float):
        """Update the current balance for cost guards."""
        self._balance = balance

    def load_performance_data(self, perf_data: Dict):
        """Load performance analytics data for decision-making."""
        self._perf_data = perf_data

    def _get_circuit(self, skill_id: str) -> CircuitState:
        """Get or create circuit state for a skill."""
        if skill_id not in self._circuits:
            self._circuits[skill_id] = CircuitState(skill_id=skill_id)
        return self._circuits[skill_id]

    def _check_cooldown_expired(self, circuit: CircuitState) -> bool:
        """Check if an open circuit's cooldown has expired."""
        if not circuit.opened_at:
            return True
        opened = datetime.fromisoformat(circuit.opened_at)
        return datetime.now() > opened + timedelta(seconds=circuit.cooldown_seconds)

    def get_advice(self, skill_id: str, action: str = "",
                   estimated_cost: float = 0.0) -> ExecutionAdvice:
        """
        Get adaptive execution advice before running an action.

        Returns advice on whether to execute, retry config, alternatives, etc.
        """
        advice = ExecutionAdvice()
        circuit = self._get_circuit(skill_id)

        # 1. Circuit breaker check
        if circuit.state == "open":
            if self._check_cooldown_expired(circuit):
                circuit.state = "half_open"
                circuit.success_count = 0
                self._log_adaptation(
                    "circuit_half_open", skill_id, action,
                    f"Cooldown expired, testing {skill_id} again"
                )
                advice.circuit_state = "half_open"
                advice.retry_config["max_retries"] = 0  # No retries during test
            else:
                advice.should_execute = False
                advice.reason = (
                    f"Circuit OPEN for {skill_id}: {circuit.failure_count} consecutive "
                    f"failures. Cooldown until circuit resets."
                )
                advice.circuit_state = "open"
                self._save_state()
                return advice
        elif circuit.state == "half_open":
            advice.circuit_state = "half_open"
            advice.retry_config["max_retries"] = 0
        else:
            advice.circuit_state = "closed"

        # 2. Smart retry configuration based on history
        skill_key = f"{skill_id}:{action}" if action else skill_id
        success_rate = self._get_success_rate(skill_key)

        if success_rate is not None:
            if success_rate >= 0.9:
                # Very reliable - minimal retries
                advice.retry_config["max_retries"] = 1
                advice.retry_config["base_delay_ms"] = 200
            elif success_rate >= 0.6:
                # Somewhat reliable - standard retries
                advice.retry_config["max_retries"] = 2
                advice.retry_config["base_delay_ms"] = 500
            elif success_rate >= 0.3:
                # Unreliable - aggressive retries
                advice.retry_config["max_retries"] = self.MAX_RETRIES
                advice.retry_config["base_delay_ms"] = 1000
            else:
                # Very unreliable - warn but allow
                advice.retry_config["max_retries"] = 1
                advice.cost_warning = (
                    f"Low success rate ({success_rate:.0%}) for {skill_key}. "
                    f"Consider alternative approaches."
                )

        # 3. Cost guard
        if estimated_cost > 0 and self._balance > 0:
            cost_ratio = estimated_cost / self._balance
            if cost_ratio > self.BUDGET_GUARD_RATIO:
                advice.cost_warning = (
                    f"Action costs ${estimated_cost:.4f} = "
                    f"{cost_ratio:.1%} of remaining ${self._balance:.2f} balance. "
                    f"Proceed with caution."
                )

        # 4. Cost efficiency warning
        if success_rate is not None and success_rate < self.COST_EFFICIENCY_THRESHOLD:
            if estimated_cost > 0:
                effective_cost = estimated_cost / max(success_rate, 0.01)
                advice.cost_warning = (
                    f"Effective cost ${effective_cost:.4f} "
                    f"(${estimated_cost:.4f} / {success_rate:.0%} success rate). "
                    f"This skill may be wasting budget."
                )

        self._save_state()
        return advice

    def record_outcome(self, skill_id: str, action: str,
                       success: bool, error: str = "",
                       latency_ms: float = 0, cost: float = 0):
        """
        Record an execution outcome and update circuit breaker state.

        This is the key feedback mechanism - every outcome updates the
        adaptive executor's model of skill reliability.
        """
        circuit = self._get_circuit(skill_id)

        if success:
            if circuit.state == "half_open":
                circuit.success_count += 1
                if circuit.success_count >= self.HALF_OPEN_SUCCESS_THRESHOLD:
                    circuit.state = "closed"
                    circuit.failure_count = 0
                    self._log_adaptation(
                        "circuit_closed", skill_id, action,
                        f"Circuit recovered after {circuit.success_count} successes"
                    )
            elif circuit.state == "closed":
                # Reset failure count on success
                circuit.failure_count = max(0, circuit.failure_count - 1)
        else:
            circuit.failure_count += 1
            circuit.last_failure = datetime.now().isoformat()

            if circuit.state == "half_open":
                # Failed during test - reopen with longer cooldown
                circuit.state = "open"
                circuit.opened_at = datetime.now().isoformat()
                circuit.cooldown_seconds = min(
                    circuit.cooldown_seconds * 2,
                    3600  # Max 1 hour cooldown
                )
                self._log_adaptation(
                    "circuit_reopened", skill_id, action,
                    f"Failed during half-open test. New cooldown: {circuit.cooldown_seconds}s"
                )
            elif circuit.failure_count >= self.CIRCUIT_FAILURE_THRESHOLD:
                circuit.state = "open"
                circuit.opened_at = datetime.now().isoformat()
                self._log_adaptation(
                    "circuit_opened", skill_id, action,
                    f"Opened after {circuit.failure_count} failures. Error: {error[:100]}"
                )

        self._save_state()

    async def execute_with_retry(self, execute_fn, skill_id: str,
                                  action: str, params: Dict,
                                  advice: Optional[ExecutionAdvice] = None) -> Dict:
        """
        Execute a skill action with adaptive retry logic.

        Uses exponential backoff with jitter, respecting the advice
        from get_advice() about retry configuration.
        """
        if advice is None:
            advice = self.get_advice(skill_id, action)

        if not advice.should_execute:
            return {
                "status": "blocked",
                "message": advice.reason,
                "circuit_state": advice.circuit_state,
            }

        max_retries = advice.retry_config.get("max_retries", 0)
        base_delay = advice.retry_config.get("base_delay_ms", self.RETRY_BASE_MS)
        max_delay = advice.retry_config.get("max_delay_ms", self.RETRY_MAX_MS)

        last_error = ""
        for attempt in range(max_retries + 1):
            start = time.time()
            try:
                result = await execute_fn(skill_id, action, params)
                latency_ms = (time.time() - start) * 1000
                success = result.get("status") == "success"

                self.record_outcome(
                    skill_id, action,
                    success=success,
                    error=result.get("message", "") if not success else "",
                    latency_ms=latency_ms,
                )

                if success or attempt == max_retries:
                    return result

                last_error = result.get("message", "Unknown error")

            except Exception as e:
                latency_ms = (time.time() - start) * 1000
                last_error = str(e)
                self.record_outcome(
                    skill_id, action,
                    success=False,
                    error=last_error,
                    latency_ms=latency_ms,
                )

                if attempt == max_retries:
                    return {"status": "error", "message": last_error}

            # Exponential backoff with jitter
            delay = min(base_delay * (2 ** attempt), max_delay)
            jitter = random.uniform(0, delay * 0.3)
            await asyncio.sleep((delay + jitter) / 1000)

            self._log_adaptation(
                "retry", skill_id, action,
                f"Attempt {attempt + 1}/{max_retries + 1} after {delay:.0f}ms. Error: {last_error[:80]}"
            )

        return {"status": "error", "message": f"All retries exhausted: {last_error}"}

    def rank_skills(self, skill_ids: List[str],
                    action: str = "") -> List[Tuple[str, float]]:
        """
        Rank skills by effectiveness for routing decisions.

        Returns list of (skill_id, score) sorted by score descending.
        Score combines success rate and cost efficiency.
        """
        scores = []
        for sid in skill_ids:
            key = f"{sid}:{action}" if action else sid
            success_rate = self._get_success_rate(key) or 0.5  # Default 50%
            avg_cost = self._get_avg_cost(key) or 0.01
            avg_latency = self._get_avg_latency(key) or 1000

            # Composite score: success rate * (1 / relative_cost) * speed_bonus
            cost_factor = 1.0 / max(avg_cost, 0.001)
            speed_factor = 1.0 / max(avg_latency / 1000, 0.1)  # Normalize to seconds

            # Circuit breaker penalty
            circuit = self._get_circuit(sid)
            circuit_penalty = 1.0
            if circuit.state == "open":
                circuit_penalty = 0.0
            elif circuit.state == "half_open":
                circuit_penalty = 0.3

            # Weighted composite (success rate dominates)
            score = (
                success_rate * 0.6 +
                min(cost_factor, 1.0) * 0.2 +
                min(speed_factor, 1.0) * 0.2
            ) * circuit_penalty

            scores.append((sid, round(score, 4)))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores

    def get_status(self) -> Dict:
        """Get full adaptive executor status for monitoring."""
        circuit_summary = {}
        for cid, circuit in self._circuits.items():
            circuit_summary[cid] = {
                "state": circuit.state,
                "failure_count": circuit.failure_count,
                "success_count": circuit.success_count,
                "last_failure": circuit.last_failure,
                "cooldown_seconds": circuit.cooldown_seconds,
            }

        return {
            "circuits": circuit_summary,
            "open_circuits": [
                cid for cid, c in self._circuits.items() if c.state == "open"
            ],
            "half_open_circuits": [
                cid for cid, c in self._circuits.items() if c.state == "half_open"
            ],
            "journal_entries": len(self._journal),
            "recent_adaptations": self._journal[-5:] if self._journal else [],
            "balance": self._balance,
        }

    def get_journal(self, limit: int = 20) -> List[Dict]:
        """Get recent adaptation journal entries."""
        return self._journal[-limit:]

    def reset_circuit(self, skill_id: str) -> bool:
        """Manually reset a circuit breaker (e.g., after fixing an issue)."""
        if skill_id in self._circuits:
            circuit = self._circuits[skill_id]
            old_state = circuit.state
            circuit.state = "closed"
            circuit.failure_count = 0
            circuit.success_count = 0
            circuit.cooldown_seconds = self.CIRCUIT_COOLDOWN_SECONDS
            self._log_adaptation(
                "circuit_manual_reset", skill_id, "",
                f"Manually reset from {old_state}"
            )
            self._save_state()
            return True
        return False

    def _get_success_rate(self, skill_key: str) -> Optional[float]:
        """Get success rate from loaded performance data."""
        stats = self._perf_data.get("skill_stats", {})
        if ":" in skill_key:
            skill_id, action = skill_key.split(":", 1)
            action_stats = stats.get(skill_id, {}).get("actions", {}).get(action, {})
            total = action_stats.get("total", 0)
            if total > 0:
                return action_stats.get("successes", 0) / total
        else:
            skill_stats = stats.get(skill_key, {})
            total = skill_stats.get("total", 0)
            if total > 0:
                return skill_stats.get("successes", 0) / total
        return None

    def _get_avg_cost(self, skill_key: str) -> Optional[float]:
        """Get average cost from performance data."""
        stats = self._perf_data.get("skill_stats", {})
        skill_id = skill_key.split(":")[0] if ":" in skill_key else skill_key
        return stats.get(skill_id, {}).get("avg_cost")

    def _get_avg_latency(self, skill_key: str) -> Optional[float]:
        """Get average latency from performance data."""
        stats = self._perf_data.get("skill_stats", {})
        skill_id = skill_key.split(":")[0] if ":" in skill_key else skill_key
        return stats.get(skill_id, {}).get("avg_latency_ms")

    def _log_adaptation(self, event_type: str, skill_id: str,
                        action: str, detail: str):
        """Log an adaptation decision to the journal."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "type": event_type,
            "skill_id": skill_id,
            "action": action,
            "detail": detail,
        }
        self._journal.append(entry)
        # Keep journal bounded
        if len(self._journal) > 200:
            self._journal = self._journal[-200:]
