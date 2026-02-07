"""
ExecutionGuard - Safe skill execution with timeout, error categorization, and metrics.

Wraps skill.execute() calls with:
1. Timeout enforcement via asyncio.wait_for
2. Error categorization (transient vs permanent vs timeout)
3. Execution time tracking
4. Output size limits to prevent memory bloat
5. Per-skill execution statistics
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from collections import defaultdict


@dataclass
class ExecutionResult:
    """Result of a guarded skill execution."""
    success: bool
    data: Any = None
    message: str = ""
    error_type: Optional[str] = None  # "timeout", "transient", "permanent", "unknown"
    execution_time_ms: float = 0.0
    truncated: bool = False

    def to_dict(self) -> Dict:
        """Convert to dict for agent action history."""
        result = {
            "status": "success" if self.success else "failed",
            "data": self.data,
            "message": self.message,
        }
        if self.error_type:
            result["error_type"] = self.error_type
        if self.execution_time_ms > 0:
            result["execution_time_ms"] = round(self.execution_time_ms, 1)
        if self.truncated:
            result["truncated"] = True
        return result


@dataclass
class SkillStats:
    """Execution statistics for a single skill."""
    total_calls: int = 0
    successes: int = 0
    failures: int = 0
    timeouts: int = 0
    total_time_ms: float = 0.0
    last_error: Optional[str] = None

    @property
    def success_rate(self) -> float:
        if self.total_calls == 0:
            return 0.0
        return self.successes / self.total_calls

    @property
    def avg_time_ms(self) -> float:
        if self.total_calls == 0:
            return 0.0
        return self.total_time_ms / self.total_calls

    def to_dict(self) -> Dict:
        return {
            "total_calls": self.total_calls,
            "successes": self.successes,
            "failures": self.failures,
            "timeouts": self.timeouts,
            "success_rate": round(self.success_rate, 3),
            "avg_time_ms": round(self.avg_time_ms, 1),
            "last_error": self.last_error,
        }


# Errors that are likely transient (retry might help)
TRANSIENT_ERRORS = (
    ConnectionError,
    TimeoutError,
    asyncio.TimeoutError,
    OSError,
)

# Max output size in characters before truncation
MAX_OUTPUT_SIZE = 50_000


class ExecutionGuard:
    """
    Guards skill execution with timeout, error handling, and metrics.

    Usage:
        guard = ExecutionGuard(default_timeout=30.0)
        result = await guard.execute(skill, action_name, params)
        stats = guard.get_stats()
    """

    def __init__(
        self,
        default_timeout: float = 30.0,
        max_output_size: int = MAX_OUTPUT_SIZE,
        skill_timeouts: Optional[Dict[str, float]] = None,
    ):
        """
        Args:
            default_timeout: Default timeout in seconds for skill execution.
            max_output_size: Max chars in output before truncation.
            skill_timeouts: Per-skill timeout overrides. Keys are skill_ids.
        """
        self.default_timeout = default_timeout
        self.max_output_size = max_output_size
        self.skill_timeouts = skill_timeouts or {}
        self._stats: Dict[str, SkillStats] = defaultdict(SkillStats)

    def get_timeout(self, skill_id: str) -> float:
        """Get timeout for a specific skill."""
        return self.skill_timeouts.get(skill_id, self.default_timeout)

    async def execute(self, skill, action_name: str, params: Dict) -> ExecutionResult:
        """
        Execute a skill action with timeout and error handling.

        Args:
            skill: The skill instance to execute on.
            action_name: Name of the action to execute.
            params: Parameters for the action.

        Returns:
            ExecutionResult with success/failure info and metrics.
        """
        skill_id = skill.manifest.skill_id
        tool_name = f"{skill_id}:{action_name}"
        timeout = self.get_timeout(skill_id)
        stats = self._stats[skill_id]
        stats.total_calls += 1

        start = time.monotonic()

        try:
            # Execute with timeout
            result = await asyncio.wait_for(
                skill.execute(action_name, params),
                timeout=timeout,
            )
            elapsed_ms = (time.monotonic() - start) * 1000
            stats.total_time_ms += elapsed_ms

            # Process result
            data = result.data
            message = result.message
            truncated = False

            # Truncate oversized output
            if isinstance(data, str) and len(data) > self.max_output_size:
                data = data[:self.max_output_size] + f"\n... [truncated, {len(result.data)} chars total]"
                truncated = True
            elif isinstance(data, dict):
                data_str = str(data)
                if len(data_str) > self.max_output_size:
                    # Keep dict but add truncation note
                    data["_truncation_note"] = f"Output was {len(data_str)} chars"
                    truncated = True

            if result.success:
                stats.successes += 1
                return ExecutionResult(
                    success=True,
                    data=data,
                    message=message,
                    execution_time_ms=elapsed_ms,
                    truncated=truncated,
                )
            else:
                stats.failures += 1
                stats.last_error = message[:200] if message else "Unknown failure"
                return ExecutionResult(
                    success=False,
                    data=data,
                    message=message,
                    error_type=_classify_error_from_message(message),
                    execution_time_ms=elapsed_ms,
                    truncated=truncated,
                )

        except asyncio.TimeoutError:
            elapsed_ms = (time.monotonic() - start) * 1000
            stats.total_time_ms += elapsed_ms
            stats.failures += 1
            stats.timeouts += 1
            stats.last_error = f"Timeout after {timeout}s"
            return ExecutionResult(
                success=False,
                message=f"Action '{tool_name}' timed out after {timeout}s",
                error_type="timeout",
                execution_time_ms=elapsed_ms,
            )

        except TRANSIENT_ERRORS as e:
            elapsed_ms = (time.monotonic() - start) * 1000
            stats.total_time_ms += elapsed_ms
            stats.failures += 1
            stats.last_error = str(e)[:200]
            return ExecutionResult(
                success=False,
                message=f"Transient error in '{tool_name}': {e}",
                error_type="transient",
                execution_time_ms=elapsed_ms,
            )

        except Exception as e:
            elapsed_ms = (time.monotonic() - start) * 1000
            stats.total_time_ms += elapsed_ms
            stats.failures += 1
            stats.last_error = str(e)[:200]
            return ExecutionResult(
                success=False,
                message=f"Error in '{tool_name}': {type(e).__name__}: {e}",
                error_type="permanent",
                execution_time_ms=elapsed_ms,
            )

    def get_stats(self, skill_id: Optional[str] = None) -> Dict:
        """
        Get execution statistics.

        Args:
            skill_id: If provided, return stats for just this skill.
                      Otherwise, return stats for all skills.

        Returns:
            Dict of skill_id -> stats dict, or single stats dict.
        """
        if skill_id:
            return self._stats[skill_id].to_dict()
        return {sid: s.to_dict() for sid, s in self._stats.items()}

    def get_slow_skills(self, threshold_ms: float = 5000.0) -> Dict[str, float]:
        """Get skills whose average execution time exceeds the threshold."""
        return {
            sid: stats.avg_time_ms
            for sid, stats in self._stats.items()
            if stats.avg_time_ms > threshold_ms and stats.total_calls > 0
        }

    def get_unreliable_skills(self, threshold: float = 0.5) -> Dict[str, float]:
        """Get skills with success rate below the threshold."""
        return {
            sid: stats.success_rate
            for sid, stats in self._stats.items()
            if stats.success_rate < threshold and stats.total_calls >= 3
        }

    def reset_stats(self):
        """Clear all execution statistics."""
        self._stats.clear()

    def summary(self) -> str:
        """Get a human-readable summary of execution stats."""
        if not self._stats:
            return "No executions recorded."

        total_calls = sum(s.total_calls for s in self._stats.values())
        total_success = sum(s.successes for s in self._stats.values())
        total_timeouts = sum(s.timeouts for s in self._stats.values())

        lines = [
            f"ExecutionGuard: {total_calls} calls, "
            f"{total_success} ok, {total_calls - total_success} failed, "
            f"{total_timeouts} timeouts"
        ]

        for sid, stats in sorted(self._stats.items()):
            lines.append(
                f"  {sid}: {stats.total_calls} calls, "
                f"{stats.success_rate:.0%} ok, "
                f"{stats.avg_time_ms:.0f}ms avg"
            )

        return "\n".join(lines)


def _classify_error_from_message(message: str) -> str:
    """Classify an error type from its message string."""
    if not message:
        return "unknown"
    msg_lower = message.lower()
    if any(word in msg_lower for word in ["timeout", "timed out"]):
        return "timeout"
    if any(word in msg_lower for word in [
        "connection", "network", "dns", "refused",
        "temporary", "unavailable", "rate limit", "429", "503",
    ]):
        return "transient"
    if any(word in msg_lower for word in [
        "permission", "denied", "forbidden", "not found",
        "invalid", "unauthorized", "401", "403", "404",
    ]):
        return "permanent"
    return "unknown"
