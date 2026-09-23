"""
Adaptive Intelligence — in-session learning for the agent.

Tracks tool success/failure rates, detects repeated failures (loops),
and generates context summaries injected into the LLM prompt so the
agent can make smarter decisions over time.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ToolStats:
    """Tracks success/failure counts for a single tool."""
    successes: int = 0
    failures: int = 0
    consecutive_failures: int = 0
    last_error: str = ""

    @property
    def total(self) -> int:
        return self.successes + self.failures

    @property
    def success_rate(self) -> float:
        if self.total == 0:
            return 1.0
        return self.successes / self.total


class AdaptiveIntelligence:
    """
    In-session adaptive learning engine.

    Wired into the agent run loop to:
    1. Record outcomes of every action (record_outcome)
    2. Detect repeated failures / loops (is_looping)
    3. Generate a context summary for the LLM (get_context)
    """

    def __init__(self, loop_threshold: int = 3, history_window: int = 6):
        self.tool_stats: Dict[str, ToolStats] = defaultdict(ToolStats)
        self.recent_tools: List[str] = []
        self.loop_threshold = loop_threshold
        self.history_window = history_window
        self.warnings: List[str] = []

    def record_outcome(self, tool: str, params: Dict, success: bool,
                       error_message: str = "") -> None:
        """Record the outcome of an action."""
        stats = self.tool_stats[tool]
        if success:
            stats.successes += 1
            stats.consecutive_failures = 0
        else:
            stats.failures += 1
            stats.consecutive_failures += 1
            stats.last_error = error_message[:200]

        self.recent_tools.append(tool)
        # Keep bounded
        if len(self.recent_tools) > 50:
            self.recent_tools = self.recent_tools[-50:]

        # Generate warnings for repeated failures
        if stats.consecutive_failures >= self.loop_threshold:
            warning = (f"WARNING: '{tool}' has failed {stats.consecutive_failures} "
                       f"times in a row. Last error: {stats.last_error}. "
                       f"Try a different approach.")
            if warning not in self.warnings:
                self.warnings = self.warnings[-4:]  # Keep last 5
                self.warnings.append(warning)

    def is_looping(self) -> bool:
        """Detect if the agent is stuck in a loop of repeated actions."""
        if len(self.recent_tools) < self.history_window:
            return False

        recent = self.recent_tools[-self.history_window:]

        # Check if all recent actions are the same tool
        if len(set(recent)) == 1:
            return True

        # Check for repeating pattern of length 2 or 3
        for pattern_len in (2, 3):
            if self.history_window >= pattern_len * 2:
                pattern = recent[-pattern_len:]
                preceding = recent[-(pattern_len * 2):-pattern_len]
                if pattern == preceding:
                    return True

        return False

    def get_failing_tools(self) -> List[str]:
        """Return tools with consecutive failure streaks."""
        return [
            tool for tool, stats in self.tool_stats.items()
            if stats.consecutive_failures >= 2
        ]

    def get_context(self) -> str:
        """
        Generate a context summary for the LLM prompt.
        Returns empty string if there's nothing noteworthy to report.
        """
        lines = []

        # Report tools with poor success rates (only if enough data)
        struggling = []
        for tool, stats in self.tool_stats.items():
            if stats.total >= 2 and stats.success_rate < 0.5:
                struggling.append(
                    f"  - {tool}: {stats.successes}/{stats.total} succeeded"
                    f" (last error: {stats.last_error})"
                )

        if struggling:
            lines.append("⚠ TOOLS WITH LOW SUCCESS RATE:")
            lines.extend(struggling)

        # Report active failure streaks
        streaking = []
        for tool, stats in self.tool_stats.items():
            if stats.consecutive_failures >= 2:
                streaking.append(
                    f"  - {tool}: {stats.consecutive_failures} consecutive failures"
                )

        if streaking:
            lines.append("🔴 ACTIVE FAILURE STREAKS (try a different approach!):")
            lines.extend(streaking)

        # Loop detection warning
        if self.is_looping():
            lines.append(
                "🔄 LOOP DETECTED: You are repeating the same actions. "
                "STOP and try a completely different strategy."
            )

        # Include recent warnings
        if self.warnings:
            lines.append("📋 RECENT WARNINGS:")
            for w in self.warnings[-3:]:
                lines.append(f"  - {w}")

        # Report overall stats summary (only after several cycles)
        total_actions = sum(s.total for s in self.tool_stats.values())
        total_successes = sum(s.successes for s in self.tool_stats.values())
        if total_actions >= 5:
            rate = total_successes / total_actions * 100
            lines.append(
                f"📊 Session stats: {total_successes}/{total_actions} actions "
                f"succeeded ({rate:.0f}%) across {len(self.tool_stats)} tools"
            )

        if not lines:
            return ""

        return "\n--- ADAPTIVE INTELLIGENCE ---\n" + "\n".join(lines) + "\n"
