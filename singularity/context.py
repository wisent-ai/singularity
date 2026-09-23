"""
ContextAggregator - Runtime intelligence for agent decision-making.

Collects and analyzes runtime data from agent actions, skill outcomes,
and cost trajectory, then generates structured context that gets injected
into the LLM prompt via AgentState.project_context.

This makes the agent self-aware of:
- What it has done this session (action history summary)
- Which skills are working vs failing (skill health)
- Patterns in its behavior (loop detection, repeated actions)
- Cost trajectory and efficiency metrics
"""

from collections import Counter, defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Any


class ContextAggregator:
    """Collects runtime context from agent execution for LLM prompt injection."""

    def __init__(self, max_history: int = 100):
        self._session_start = datetime.now()
        self._action_log: List[Dict[str, Any]] = []
        self._skill_health: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {"success": 0, "failure": 0, "error": 0}
        )
        self._cost_history: List[float] = []
        self._max_history = max_history

    def record_action(
        self,
        tool: str,
        params: Dict,
        result: Dict,
        cost: float = 0.0,
        cycle: int = 0,
    ):
        """Record an action outcome for context generation.

        Args:
            tool: The tool name (e.g., 'filesystem:write')
            params: Parameters passed to the tool
            result: Result dict with 'status' key
            cost: API cost for this action
            cycle: Current cycle number
        """
        status = result.get("status", "unknown")
        entry = {
            "tool": tool,
            "params_summary": self._summarize_params(params),
            "status": status,
            "cost": cost,
            "cycle": cycle,
            "timestamp": datetime.now().isoformat(),
        }
        self._action_log.append(entry)

        # Trim history
        if len(self._action_log) > self._max_history:
            self._action_log = self._action_log[-self._max_history:]

        # Track skill health
        skill_id = tool.split(":")[0] if ":" in tool else tool
        if status == "success":
            self._skill_health[skill_id]["success"] += 1
        elif status == "error":
            self._skill_health[skill_id]["error"] += 1
        else:
            self._skill_health[skill_id]["failure"] += 1

        # Track costs
        self._cost_history.append(cost)

    def get_context(self) -> Dict[str, Any]:
        """Generate context dict for AgentState.project_context.

        Returns a dict with session summary, skill health, action patterns,
        and cost trajectory - all designed to help the LLM make better decisions.
        """
        ctx: Dict[str, Any] = {}

        session = self._session_summary()
        if session:
            ctx["session_summary"] = session

        health = self._skill_health_report()
        if health:
            ctx["skill_health"] = health

        patterns = self._action_patterns()
        if patterns:
            ctx["action_patterns"] = patterns

        cost = self._cost_trajectory()
        if cost:
            ctx["cost_trajectory"] = cost

        return ctx

    def _session_summary(self) -> Dict[str, Any]:
        """Summary of current session."""
        if not self._action_log:
            return {}

        elapsed = (datetime.now() - self._session_start).total_seconds()
        total_cost = sum(self._cost_history)
        successes = sum(1 for a in self._action_log if a["status"] == "success")
        failures = sum(1 for a in self._action_log if a["status"] != "success")
        unique_tools = len(set(a["tool"] for a in self._action_log))

        summary: Dict[str, Any] = {
            "elapsed_minutes": round(elapsed / 60, 1),
            "total_actions": len(self._action_log),
            "successes": successes,
            "failures": failures,
            "success_rate": round(successes / len(self._action_log) * 100, 1)
            if self._action_log
            else 0,
            "unique_tools_used": unique_tools,
            "total_cost_usd": round(total_cost, 6),
        }

        # Last 3 actions for immediate context
        recent = []
        for a in self._action_log[-3:]:
            recent.append(f"{a['tool']} -> {a['status']}")
        if recent:
            summary["recent_actions"] = recent

        return summary

    def _skill_health_report(self) -> Dict[str, Any]:
        """Which skills are healthy vs failing."""
        if not self._skill_health:
            return {}

        report: Dict[str, Any] = {}
        failing_skills = []
        healthy_skills = []

        for skill_id, counts in self._skill_health.items():
            total = counts["success"] + counts["failure"] + counts["error"]
            if total == 0:
                continue

            rate = counts["success"] / total * 100
            if rate < 50 and total >= 2:
                failing_skills.append(
                    f"{skill_id} ({rate:.0f}% success, {total} attempts)"
                )
            elif rate >= 80:
                healthy_skills.append(skill_id)

        if failing_skills:
            report["failing_skills"] = failing_skills
            report["warning"] = (
                "These skills are failing frequently. "
                "Consider using alternative approaches or checking parameters."
            )
        if healthy_skills:
            report["healthy_skills"] = healthy_skills

        return report

    def _action_patterns(self) -> Dict[str, Any]:
        """Detect patterns in actions (loops, repeated failures)."""
        if len(self._action_log) < 3:
            return {}

        patterns: Dict[str, Any] = {}

        # Check for repeated identical actions (loop detection)
        recent_tools = [a["tool"] for a in self._action_log[-6:]]
        if len(recent_tools) >= 4:
            # Check for simple repetition (ABAB or AAAA)
            last_two = recent_tools[-2:]
            if len(set(recent_tools[-4:])) <= 2:
                count = sum(
                    1 for t in recent_tools if t in set(recent_tools[-2:])
                )
                if count >= 4:
                    patterns["possible_loop"] = {
                        "repeated_tools": list(set(recent_tools[-4:])),
                        "suggestion": (
                            "You may be stuck in a loop. "
                            "Try a completely different approach or tool."
                        ),
                    }

        # Check for consecutive failures
        recent_statuses = [a["status"] for a in self._action_log[-5:]]
        consecutive_fails = 0
        for s in reversed(recent_statuses):
            if s != "success":
                consecutive_fails += 1
            else:
                break

        if consecutive_fails >= 3:
            patterns["consecutive_failures"] = {
                "count": consecutive_fails,
                "suggestion": (
                    f"Last {consecutive_fails} actions failed. "
                    "Step back and try a different strategy."
                ),
            }

        # Most used tools
        tool_counts = Counter(a["tool"] for a in self._action_log)
        most_common = tool_counts.most_common(3)
        if most_common:
            patterns["most_used_tools"] = [
                {"tool": t, "count": c} for t, c in most_common
            ]

        return patterns

    def _cost_trajectory(self) -> Dict[str, Any]:
        """Cost trend analysis."""
        if len(self._cost_history) < 2:
            return {}

        total = sum(self._cost_history)
        avg = total / len(self._cost_history) if self._cost_history else 0

        trajectory: Dict[str, Any] = {
            "avg_cost_per_action": round(avg, 6),
            "total_session_cost": round(total, 6),
        }

        # Compare first half vs second half cost
        mid = len(self._cost_history) // 2
        if mid > 0:
            first_half_avg = sum(self._cost_history[:mid]) / mid
            second_half_avg = sum(self._cost_history[mid:]) / (
                len(self._cost_history) - mid
            )
            if first_half_avg > 0:
                change = ((second_half_avg - first_half_avg) / first_half_avg) * 100
                if abs(change) > 20:
                    trajectory["cost_trend"] = (
                        f"{'increasing' if change > 0 else 'decreasing'} "
                        f"({change:+.0f}%)"
                    )

        return trajectory

    def _summarize_params(self, params: Dict) -> str:
        """Create a brief summary of params for logging."""
        if not params:
            return ""
        parts = []
        for k, v in list(params.items())[:3]:
            val_str = str(v)
            if len(val_str) > 50:
                val_str = val_str[:47] + "..."
            parts.append(f"{k}={val_str}")
        summary = ", ".join(parts)
        if len(params) > 3:
            summary += f" (+{len(params)-3} more)"
        return summary
