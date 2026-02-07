"""Loop Detector — detects when the agent is stuck repeating actions.

Analyzes recent_actions for repetitive patterns and generates warnings
that get injected into the LLM context to break out of loops.

Self-Improvement pillar: enables act → measure → adapt feedback loop
by detecting unproductive repetition and signaling the LLM to change strategy.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class LoopAlert:
    """An alert about a detected loop pattern."""
    pattern_type: str  # "exact_repeat", "tool_repeat", "error_loop", "ping_pong"
    description: str
    severity: str  # "warning", "critical"
    suggested_action: str
    repeat_count: int = 0


class LoopDetector:
    """Detects repetitive patterns in agent action history.
    
    Usage:
        detector = LoopDetector()
        # After each cycle:
        alerts = detector.analyze(recent_actions)
        if alerts:
            warning_text = detector.format_warnings(alerts)
            # Inject warning_text into LLM context
    """

    def __init__(
        self,
        exact_repeat_threshold: int = 3,
        tool_repeat_threshold: int = 5,
        error_streak_threshold: int = 3,
        window_size: int = 10,
    ):
        self.exact_repeat_threshold = exact_repeat_threshold
        self.tool_repeat_threshold = tool_repeat_threshold
        self.error_streak_threshold = error_streak_threshold
        self.window_size = window_size
        self._total_alerts: int = 0

    @property
    def total_alerts(self) -> int:
        return self._total_alerts

    def analyze(self, recent_actions: List[Dict[str, Any]]) -> List[LoopAlert]:
        """Analyze recent actions for loop patterns.
        
        Args:
            recent_actions: List of action dicts with keys like
                'tool', 'params', 'result', 'cycle'.
        
        Returns:
            List of LoopAlert objects for detected patterns.
        """
        if not recent_actions:
            return []

        window = recent_actions[-self.window_size:]
        alerts: List[LoopAlert] = []

        # Check each pattern type
        alert = self._check_exact_repeats(window)
        if alert:
            alerts.append(alert)

        alert = self._check_tool_repeats(window)
        if alert:
            alerts.append(alert)

        alert = self._check_error_streak(window)
        if alert:
            alerts.append(alert)

        alert = self._check_ping_pong(window)
        if alert:
            alerts.append(alert)

        self._total_alerts += len(alerts)
        return alerts

    def _check_exact_repeats(self, actions: List[Dict]) -> Optional[LoopAlert]:
        """Detect when the exact same tool+params is called repeatedly."""
        if len(actions) < self.exact_repeat_threshold:
            return None

        # Look at the last N actions
        tail = actions[-self.exact_repeat_threshold:]
        signatures = []
        for a in tail:
            tool = a.get("tool", "")
            params = str(a.get("params", {}))
            signatures.append(f"{tool}|{params}")

        if len(set(signatures)) == 1:
            tool = tail[0].get("tool", "unknown")
            return LoopAlert(
                pattern_type="exact_repeat",
                description=f"Exact same action '{tool}' called {len(tail)} times in a row with identical parameters",
                severity="critical",
                suggested_action=f"STOP calling '{tool}' with the same parameters. Try a different approach or tool.",
                repeat_count=len(tail),
            )
        return None

    def _check_tool_repeats(self, actions: List[Dict]) -> Optional[LoopAlert]:
        """Detect when the same tool is called too many times (even with different params)."""
        if len(actions) < self.tool_repeat_threshold:
            return None

        tools = [a.get("tool", "") for a in actions[-self.tool_repeat_threshold:]]
        if len(set(tools)) == 1 and tools[0]:
            return LoopAlert(
                pattern_type="tool_repeat",
                description=f"Tool '{tools[0]}' used {len(tools)} times consecutively",
                severity="warning",
                suggested_action=f"Consider using a different tool. '{tools[0]}' has been used {len(tools)} times in a row.",
                repeat_count=len(tools),
            )
        return None

    def _check_error_streak(self, actions: List[Dict]) -> Optional[LoopAlert]:
        """Detect consecutive failures."""
        if len(actions) < self.error_streak_threshold:
            return None

        streak = 0
        for a in reversed(actions):
            result = a.get("result", {})
            status = result.get("status", "") if isinstance(result, dict) else ""
            if status in ("failed", "error"):
                streak += 1
            else:
                break

        if streak >= self.error_streak_threshold:
            # Collect error messages
            errors = []
            for a in actions[-streak:]:
                result = a.get("result", {})
                if isinstance(result, dict):
                    msg = result.get("message", "") or result.get("data", "")
                    if msg:
                        errors.append(str(msg)[:100])

            error_summary = "; ".join(errors[-3:]) if errors else "multiple failures"
            return LoopAlert(
                pattern_type="error_loop",
                description=f"{streak} consecutive failures: {error_summary}",
                severity="critical",
                suggested_action="Multiple actions are failing. Re-evaluate your approach. Check if prerequisites are met before retrying.",
                repeat_count=streak,
            )
        return None

    def _check_ping_pong(self, actions: List[Dict]) -> Optional[LoopAlert]:
        """Detect alternating between two actions (A-B-A-B pattern)."""
        if len(actions) < 4:
            return None

        # Check last 4+ actions for alternating pattern
        tools = [a.get("tool", "") for a in actions]
        
        # Look for A-B-A-B in the tail
        for length in range(4, min(len(tools) + 1, 9), 2):
            tail = tools[-length:]
            evens = set(tail[0::2])
            odds = set(tail[1::2])
            if len(evens) == 1 and len(odds) == 1 and evens != odds:
                tool_a = tail[0]
                tool_b = tail[1]
                return LoopAlert(
                    pattern_type="ping_pong",
                    description=f"Alternating between '{tool_a}' and '{tool_b}' for {length} cycles",
                    severity="warning",
                    suggested_action=f"You're going back and forth between '{tool_a}' and '{tool_b}'. Break the pattern — try a completely different approach.",
                    repeat_count=length,
                )
        return None

    def format_warnings(self, alerts: List[LoopAlert]) -> str:
        """Format alerts into a text block for LLM injection.
        
        Returns a string suitable for injection into the LLM prompt context.
        """
        if not alerts:
            return ""

        lines = ["⚠️ LOOP DETECTION WARNINGS:"]
        for alert in alerts:
            icon = "🔴" if alert.severity == "critical" else "🟡"
            lines.append(f"  {icon} [{alert.pattern_type}] {alert.description}")
            lines.append(f"     → {alert.suggested_action}")
        
        lines.append("")
        lines.append("You MUST change your approach. Repeating the same actions will waste budget.")
        return "\n".join(lines)

    def should_force_rethink(self, alerts: List[LoopAlert]) -> bool:
        """Return True if any alert is critical, suggesting the agent should pause."""
        return any(a.severity == "critical" for a in alerts)
