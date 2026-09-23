"""
RetryExecutor — Smart execution with retry, circuit breaking, and error tracking.

Enhances the agent's _execute method to:
1. Retry transient failures with exponential backoff
2. Track per-skill error rates and success rates  
3. Circuit-break skills that fail consistently
4. Provide rich error context for LLM awareness
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


# Errors that are worth retrying (transient)
TRANSIENT_PATTERNS = [
    "timeout", "timed out", "rate limit", "429", "503", "502", "504",
    "connection", "network", "temporary", "retry", "overloaded",
    "server error", "service unavailable", "too many requests",
]


@dataclass
class SkillHealth:
    """Tracks health metrics for a single skill."""
    skill_id: str
    total_calls: int = 0
    successes: int = 0
    failures: int = 0
    consecutive_failures: int = 0
    last_error: str = ""
    last_error_time: float = 0.0
    circuit_open_until: float = 0.0  # timestamp when circuit breaker expires

    @property
    def success_rate(self) -> float:
        if self.total_calls == 0:
            return 1.0
        return self.successes / self.total_calls

    @property
    def is_circuit_open(self) -> bool:
        if self.circuit_open_until <= 0:
            return False
        return time.time() < self.circuit_open_until

    def record_success(self):
        self.total_calls += 1
        self.successes += 1
        self.consecutive_failures = 0
        # Reset circuit breaker on success
        self.circuit_open_until = 0.0

    def record_failure(self, error: str):
        self.total_calls += 1
        self.failures += 1
        self.consecutive_failures += 1
        self.last_error = error[:200]
        self.last_error_time = time.time()

    def open_circuit(self, duration_seconds: float = 60.0):
        """Open the circuit breaker for the given duration."""
        self.circuit_open_until = time.time() + duration_seconds


class RetryExecutor:
    """
    Wraps skill execution with retry logic, circuit breaking, and health tracking.
    
    Usage:
        executor = RetryExecutor(skills_registry)
        result = await executor.execute(action)
        summary = executor.get_health_summary()
    """

    def __init__(
        self,
        skills,
        max_retries: int = 2,
        base_delay: float = 1.0,
        max_delay: float = 10.0,
        circuit_break_threshold: int = 5,
        circuit_break_duration: float = 60.0,
    ):
        """
        Args:
            skills: SkillRegistry instance
            max_retries: Max retry attempts for transient errors
            base_delay: Base delay in seconds for exponential backoff
            max_delay: Maximum delay between retries
            circuit_break_threshold: Consecutive failures before circuit opens
            circuit_break_duration: Seconds to keep circuit open
        """
        self.skills = skills
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.circuit_break_threshold = circuit_break_threshold
        self.circuit_break_duration = circuit_break_duration
        self._health: Dict[str, SkillHealth] = {}

    def _get_health(self, skill_id: str) -> SkillHealth:
        if skill_id not in self._health:
            self._health[skill_id] = SkillHealth(skill_id=skill_id)
        return self._health[skill_id]

    def _is_transient(self, error_msg: str) -> bool:
        """Check if an error is likely transient and worth retrying."""
        lower = error_msg.lower()
        return any(pattern in lower for pattern in TRANSIENT_PATTERNS)

    async def execute(self, tool: str, params: Dict) -> Dict:
        """
        Execute a tool action with retry logic and circuit breaking.
        
        Args:
            tool: Tool identifier in "skill:action" format
            params: Action parameters
            
        Returns:
            Result dict with status, data, message, and metadata
        """
        if tool == "wait":
            return {"status": "waited"}

        if ":" not in tool:
            return {"status": "error", "message": f"Unknown tool: {tool}"}

        parts = tool.split(":", 1)
        skill_id = parts[0]
        action_name = parts[1] if len(parts) > 1 else ""

        skill = self.skills.get(skill_id)
        if not skill:
            return {"status": "error", "message": f"Skill not found: {skill_id}"}

        health = self._get_health(skill_id)

        # Check circuit breaker
        if health.is_circuit_open:
            return {
                "status": "error",
                "message": f"Skill '{skill_id}' is temporarily disabled due to repeated failures. "
                           f"Last error: {health.last_error}. "
                           f"Will retry in {health.circuit_open_until - time.time():.0f}s.",
                "_retry_info": {
                    "circuit_open": True,
                    "consecutive_failures": health.consecutive_failures,
                }
            }

        # Execute with retry
        last_error = ""
        attempts = 0

        for attempt in range(self.max_retries + 1):
            attempts = attempt + 1
            try:
                result = await skill.execute(action_name, params)
                
                if result.success:
                    health.record_success()
                    return {
                        "status": "success",
                        "data": result.data,
                        "message": result.message,
                        "_retry_info": {"attempts": attempts} if attempts > 1 else {},
                    }
                else:
                    # Skill returned failure (not an exception)
                    error_msg = result.message or "Action failed"
                    
                    if attempt < self.max_retries and self._is_transient(error_msg):
                        delay = min(self.base_delay * (2 ** attempt), self.max_delay)
                        await asyncio.sleep(delay)
                        last_error = error_msg
                        continue
                    
                    health.record_failure(error_msg)
                    self._check_circuit_break(health)
                    
                    return {
                        "status": "failed",
                        "data": result.data,
                        "message": error_msg,
                        "_retry_info": {
                            "attempts": attempts,
                            "skill_success_rate": f"{health.success_rate:.0%}",
                        }
                    }

            except Exception as e:
                error_msg = str(e)
                
                if attempt < self.max_retries and self._is_transient(error_msg):
                    delay = min(self.base_delay * (2 ** attempt), self.max_delay)
                    await asyncio.sleep(delay)
                    last_error = error_msg
                    continue
                
                health.record_failure(error_msg)
                self._check_circuit_break(health)
                
                return {
                    "status": "error",
                    "message": error_msg,
                    "_retry_info": {
                        "attempts": attempts,
                        "retried": attempts > 1,
                        "last_transient_error": last_error if last_error != error_msg else "",
                        "skill_success_rate": f"{health.success_rate:.0%}",
                    }
                }

        # Should not reach here, but just in case
        health.record_failure(last_error)
        self._check_circuit_break(health)
        return {
            "status": "error",
            "message": f"Failed after {attempts} attempts. Last error: {last_error}",
        }

    def _check_circuit_break(self, health: SkillHealth):
        """Open circuit breaker if threshold is exceeded."""
        if health.consecutive_failures >= self.circuit_break_threshold:
            health.open_circuit(self.circuit_break_duration)

    def get_health_summary(self) -> str:
        """Get a summary of skill health for LLM context."""
        if not self._health:
            return ""

        lines = []
        unhealthy = []
        
        for sid, h in sorted(self._health.items()):
            if h.total_calls == 0:
                continue
            if h.is_circuit_open:
                unhealthy.append(f"⚠ {sid}: DISABLED (circuit open, {h.consecutive_failures} consecutive failures)")
            elif h.success_rate < 0.5 and h.total_calls >= 3:
                unhealthy.append(f"⚠ {sid}: LOW SUCCESS RATE ({h.success_rate:.0%} over {h.total_calls} calls)")

        if unhealthy:
            lines.append("Skill Health Warnings:")
            lines.extend(unhealthy)

        return "\n".join(lines)

    def get_all_health(self) -> Dict[str, Dict]:
        """Get detailed health metrics for all tracked skills."""
        result = {}
        for sid, h in self._health.items():
            result[sid] = {
                "total_calls": h.total_calls,
                "successes": h.successes,
                "failures": h.failures,
                "success_rate": h.success_rate,
                "consecutive_failures": h.consecutive_failures,
                "last_error": h.last_error,
                "circuit_open": h.is_circuit_open,
            }
        return result

    def reset_health(self, skill_id: Optional[str] = None):
        """Reset health tracking for a skill or all skills."""
        if skill_id:
            self._health.pop(skill_id, None)
        else:
            self._health.clear()
