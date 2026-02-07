"""
SkillHealthMonitor — Tracks per-skill success/failure rates and generates
warnings for the agent when skills are consistently failing.

Wired into the agent run loop to inject failure-awareness context into
the LLM prompt, enabling the agent to avoid broken tools and adapt.

Self-Improvement pillar: act → measure → adapt feedback loop.
"""

from collections import defaultdict
from typing import Dict, List, Optional


class SkillHealthMonitor:
    """
    Monitors skill execution health from recent_actions history.

    Tracks:
    - Per-skill success/failure counts
    - Consecutive failure streaks
    - Recently failed tools (to warn the agent)

    Generates a context string for injection into the LLM prompt.
    """

    def __init__(self, failure_threshold: int = 3, lookback: int = 20):
        """
        Args:
            failure_threshold: Number of consecutive failures before warning.
            lookback: How many recent actions to analyze.
        """
        self.failure_threshold = failure_threshold
        self.lookback = lookback

    def analyze(self, recent_actions: List[Dict]) -> Dict:
        """
        Analyze recent actions and return health metrics.

        Returns dict with:
            - skill_stats: {tool: {success: int, failure: int, rate: float}}
            - failing_skills: list of tools with consecutive failures >= threshold
            - last_errors: {tool: last_error_message}
            - consecutive_failures: current streak of any failures
        """
        actions = recent_actions[-self.lookback:]
        if not actions:
            return {
                "skill_stats": {},
                "failing_skills": [],
                "last_errors": {},
                "consecutive_failures": 0,
            }

        # Per-skill stats
        stats = defaultdict(lambda: {"success": 0, "failure": 0})
        last_errors = {}
        consecutive_by_skill = defaultdict(int)

        for action in actions:
            tool = action.get("tool", "unknown")
            status = action.get("result", {}).get("status", "unknown")

            if status == "success":
                stats[tool]["success"] += 1
                consecutive_by_skill[tool] = 0
            elif status in ("error", "failed"):
                stats[tool]["failure"] += 1
                consecutive_by_skill[tool] += 1
                error_msg = action.get("result", {}).get("message", "unknown error")
                last_errors[tool] = error_msg

        # Calculate success rates
        skill_stats = {}
        for tool, counts in stats.items():
            total = counts["success"] + counts["failure"]
            rate = counts["success"] / total if total > 0 else 0.0
            skill_stats[tool] = {
                "success": counts["success"],
                "failure": counts["failure"],
                "rate": round(rate, 2),
            }

        # Find skills exceeding failure threshold
        failing_skills = [
            tool for tool, count in consecutive_by_skill.items()
            if count >= self.failure_threshold
        ]

        # Overall consecutive failure streak (any tool)
        consecutive_failures = 0
        for action in reversed(actions):
            status = action.get("result", {}).get("status", "unknown")
            if status in ("error", "failed"):
                consecutive_failures += 1
            else:
                break

        return {
            "skill_stats": skill_stats,
            "failing_skills": failing_skills,
            "last_errors": last_errors,
            "consecutive_failures": consecutive_failures,
        }

    def generate_context(self, recent_actions: List[Dict]) -> str:
        """
        Generate a context string for injection into the LLM prompt.

        Returns empty string if everything is healthy.
        Returns warnings when failures are detected.
        """
        health = self.analyze(recent_actions)

        parts = []

        # Warn about consistently failing skills
        if health["failing_skills"]:
            failing_list = ", ".join(health["failing_skills"])
            parts.append(
                f"⚠️ FAILING TOOLS: {failing_list} — these have failed "
                f"{self.failure_threshold}+ times in a row. "
                "Try a different approach or tool."
            )
            for tool in health["failing_skills"]:
                if tool in health["last_errors"]:
                    parts.append(f"  Last error from {tool}: {health['last_errors'][tool][:200]}")

        # Warn about overall failure streak
        if health["consecutive_failures"] >= 2:
            parts.append(
                f"⚠️ {health['consecutive_failures']} consecutive failures. "
                "Stop and reconsider your approach before trying again."
            )

        # Show skill health summary if there are any failures
        skills_with_failures = {
            tool: s for tool, s in health["skill_stats"].items()
            if s["failure"] > 0
        }
        if skills_with_failures:
            summary_lines = []
            for tool, s in sorted(skills_with_failures.items(), key=lambda x: x[1]["rate"]):
                summary_lines.append(
                    f"  {tool}: {s['rate']*100:.0f}% success ({s['success']}/{s['success']+s['failure']})"
                )
            if summary_lines:
                parts.append("Tool health:\n" + "\n".join(summary_lines))

        return "\n".join(parts)
