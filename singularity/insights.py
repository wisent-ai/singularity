"""
ActionInsights - Analyzes agent action history to generate self-awareness.

Computes performance metrics from recent_actions and produces a concise
summary that gets injected into the LLM prompt context, enabling the agent
to make better decisions based on its own track record.

Part of the Self-Improvement pillar: act → measure → adapt.
"""

from collections import Counter, defaultdict
from typing import Dict, List, Optional


class ActionInsights:
    """Analyzes agent action history and produces actionable insights."""

    def __init__(self, recent_actions: List[Dict]):
        self.actions = recent_actions or []

    def compute(self) -> Dict:
        """Compute all insights from action history.
        
        Returns dict with:
            total_actions, success_rate, total_cost, avg_cost_per_action,
            skill_stats, error_patterns, current_streak, recommendations
        """
        if not self.actions:
            return {
                "total_actions": 0,
                "success_rate": 0.0,
                "total_cost": 0.0,
                "avg_cost_per_action": 0.0,
                "skill_stats": {},
                "error_patterns": [],
                "current_streak": {"type": "none", "count": 0},
                "recommendations": [],
            }

        total = len(self.actions)
        successes = sum(
            1 for a in self.actions
            if a.get("result", {}).get("status") == "success"
        )
        failures = sum(
            1 for a in self.actions
            if a.get("result", {}).get("status") in ("failed", "error")
        )
        success_rate = successes / total if total > 0 else 0.0

        # Cost analysis
        total_cost = sum(a.get("api_cost_usd", 0) for a in self.actions)
        avg_cost = total_cost / total if total > 0 else 0.0

        # Per-skill stats
        skill_stats = self._compute_skill_stats()

        # Error patterns
        error_patterns = self._compute_error_patterns()

        # Current streak
        current_streak = self._compute_streak()

        # Recommendations
        recommendations = self._generate_recommendations(
            success_rate, skill_stats, error_patterns, current_streak
        )

        return {
            "total_actions": total,
            "success_rate": round(success_rate, 3),
            "total_cost": round(total_cost, 6),
            "avg_cost_per_action": round(avg_cost, 6),
            "skill_stats": skill_stats,
            "error_patterns": error_patterns,
            "current_streak": current_streak,
            "recommendations": recommendations,
        }

    def _compute_skill_stats(self) -> Dict:
        """Compute per-skill success/failure counts."""
        stats = defaultdict(lambda: {"total": 0, "success": 0, "failed": 0, "cost": 0.0})

        for action in self.actions:
            tool = action.get("tool", "unknown")
            skill_id = tool.split(":")[0] if ":" in tool else tool
            status = action.get("result", {}).get("status", "unknown")

            stats[skill_id]["total"] += 1
            if status == "success":
                stats[skill_id]["success"] += 1
            elif status in ("failed", "error"):
                stats[skill_id]["failed"] += 1
            stats[skill_id]["cost"] += action.get("api_cost_usd", 0)

        # Convert to regular dict and add success_rate
        result = {}
        for skill_id, s in stats.items():
            rate = s["success"] / s["total"] if s["total"] > 0 else 0.0
            result[skill_id] = {
                "total": s["total"],
                "success": s["success"],
                "failed": s["failed"],
                "cost": round(s["cost"], 6),
                "success_rate": round(rate, 3),
            }

        return result

    def _compute_error_patterns(self) -> List[Dict]:
        """Find repeated error messages."""
        error_messages = []
        for action in self.actions:
            result = action.get("result", {})
            if result.get("status") in ("failed", "error"):
                msg = result.get("message", "unknown error")
                # Truncate long messages for pattern matching
                msg_key = msg[:100] if len(msg) > 100 else msg
                error_messages.append({
                    "tool": action.get("tool", "unknown"),
                    "message": msg_key,
                })

        if not error_messages:
            return []

        # Group by message
        counter = Counter(e["message"] for e in error_messages)
        patterns = []
        for msg, count in counter.most_common(5):
            # Find which tools hit this error
            tools = list(set(
                e["tool"] for e in error_messages if e["message"] == msg
            ))
            patterns.append({
                "message": msg,
                "count": count,
                "tools": tools,
            })

        return patterns

    def _compute_streak(self) -> Dict:
        """Compute current success/failure streak from most recent actions."""
        if not self.actions:
            return {"type": "none", "count": 0}

        # Walk backwards from most recent
        streak_type = None
        count = 0

        for action in reversed(self.actions):
            status = action.get("result", {}).get("status", "unknown")
            if status == "success":
                current = "success"
            elif status in ("failed", "error"):
                current = "failure"
            else:
                current = "other"

            if streak_type is None:
                streak_type = current
                count = 1
            elif current == streak_type:
                count += 1
            else:
                break

        return {"type": streak_type or "none", "count": count}

    def _generate_recommendations(
        self,
        success_rate: float,
        skill_stats: Dict,
        error_patterns: List[Dict],
        streak: Dict,
    ) -> List[str]:
        """Generate actionable recommendations based on insights."""
        recs = []

        # Low success rate warning
        if success_rate < 0.5 and len(self.actions) >= 3:
            recs.append(
                f"Low success rate ({success_rate:.0%}). Consider switching strategy or using different tools."
            )

        # Repeated errors
        for pattern in error_patterns:
            if pattern["count"] >= 2:
                recs.append(
                    f"Repeated error ({pattern['count']}x): \"{pattern['message'][:60]}\" — avoid or fix this."
                )

        # Failing skill
        for skill_id, stats in skill_stats.items():
            if stats["total"] >= 2 and stats["success_rate"] < 0.3:
                recs.append(
                    f"Skill '{skill_id}' has {stats['success_rate']:.0%} success rate. Consider alternatives."
                )

        # Failure streak
        if streak["type"] == "failure" and streak["count"] >= 3:
            recs.append(
                f"On a {streak['count']}-action failure streak. Step back and reassess approach."
            )

        # Success streak encouragement
        if streak["type"] == "success" and streak["count"] >= 5:
            recs.append(
                f"On a {streak['count']}-action success streak. Current approach is working well."
            )

        # Cost awareness
        total_cost = sum(a.get("api_cost_usd", 0) for a in self.actions)
        if total_cost > 1.0:
            recs.append(
                f"Total API cost so far: ${total_cost:.4f}. Monitor spend carefully."
            )

        return recs

    def format_for_prompt(self) -> str:
        """Format insights as a concise string for LLM prompt injection.
        
        Returns empty string if no actions yet (no noise in prompt).
        """
        insights = self.compute()

        if insights["total_actions"] == 0:
            return ""

        lines = [
            "=== Performance Insights ===",
            f"Actions: {insights['total_actions']} | "
            f"Success rate: {insights['success_rate']:.0%} | "
            f"API cost: ${insights['total_cost']:.4f}",
        ]

        # Streak info
        streak = insights["current_streak"]
        if streak["count"] >= 2:
            lines.append(
                f"Current streak: {streak['count']} {streak['type']}s in a row"
            )

        # Skill breakdown (top 5 by usage)
        if insights["skill_stats"]:
            sorted_skills = sorted(
                insights["skill_stats"].items(),
                key=lambda x: x[1]["total"],
                reverse=True,
            )[:5]
            skill_parts = []
            for skill_id, stats in sorted_skills:
                skill_parts.append(
                    f"{skill_id}: {stats['success']}/{stats['total']}"
                )
            lines.append("Skills: " + " | ".join(skill_parts))

        # Error patterns (top 2)
        if insights["error_patterns"]:
            for pattern in insights["error_patterns"][:2]:
                lines.append(
                    f"⚠ Repeated error ({pattern['count']}x): {pattern['message'][:80]}"
                )

        # Recommendations
        if insights["recommendations"]:
            for rec in insights["recommendations"][:3]:
                lines.append(f"→ {rec}")

        return "\n".join(lines)
