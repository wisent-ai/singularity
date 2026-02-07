#!/usr/bin/env python3
"""
StrategySkill - Autonomous self-improvement through action analysis.

Provides the feedback loop: act → measure outcome → adapt strategy.

The agent can analyze its own action history to find patterns of
success and failure, then automatically evolve its system prompt
with learned strategies, rules, and insights.

This is the core self-improvement mechanism that makes the agent
get better over time without human intervention.
"""

import json
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional

from .base import Skill, SkillAction, SkillManifest, SkillResult

# Persistent storage for cross-session learning
INSIGHTS_FILE = Path(__file__).parent.parent / "data" / "strategy_insights.json"


class StrategySkill(Skill):
    """Skill for autonomous performance analysis and strategy evolution."""

    def __init__(self, credentials: Dict = None):
        super().__init__(credentials)
        # Hooks into agent
        self._get_actions_fn: Optional[Callable[[], List[Dict]]] = None
        self._get_prompt_fn: Optional[Callable[[], str]] = None
        self._append_prompt_fn: Optional[Callable[[str], None]] = None
        self._get_balance_fn: Optional[Callable[[], float]] = None
        # In-memory tracking
        self._analysis_history: List[Dict] = []

    def set_agent_hooks(
        self,
        get_actions: Callable[[], List[Dict]],
        get_prompt: Callable[[], str],
        append_prompt: Callable[[str], None],
        get_balance: Callable[[], float] = None,
    ):
        """Connect this skill to the agent's internals."""
        self._get_actions_fn = get_actions
        self._get_prompt_fn = get_prompt
        self._append_prompt_fn = append_prompt
        self._get_balance_fn = get_balance

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="strategy",
            name="Strategy & Self-Improvement",
            version="1.0.0",
            category="meta",
            description="Analyze your performance, find patterns, and evolve your strategy",
            actions=[
                SkillAction(
                    name="analyze",
                    description="Analyze recent actions for success/failure patterns, waste, and efficiency",
                    parameters={},
                    estimated_cost=0,
                ),
                SkillAction(
                    name="evolve",
                    description="Analyze performance and apply a strategy improvement to your system prompt",
                    parameters={
                        "focus": {
                            "type": "string",
                            "required": False,
                            "description": "Optional focus area: 'efficiency', 'success_rate', 'cost', 'revenue'",
                        }
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="get_insights",
                    description="Get all accumulated strategy insights from past analyses",
                    parameters={},
                    estimated_cost=0,
                ),
                SkillAction(
                    name="save_insight",
                    description="Manually record a strategic insight for future reference",
                    parameters={
                        "insight": {
                            "type": "string",
                            "required": True,
                            "description": "The insight to record",
                        },
                        "category": {
                            "type": "string",
                            "required": False,
                            "description": "Category: 'efficiency', 'cost', 'revenue', 'behavior', 'general'",
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="score",
                    description="Get a numerical performance score (0-100) based on recent actions",
                    parameters={},
                    estimated_cost=0,
                ),
            ],
            required_credentials=[],
        )

    def check_credentials(self) -> bool:
        return self._get_actions_fn is not None

    async def execute(self, action: str, params: Dict) -> SkillResult:
        if not self._get_actions_fn:
            return SkillResult(
                success=False,
                message="Strategy skill not connected to agent. Call set_agent_hooks first.",
            )

        if action == "analyze":
            return self._analyze()
        elif action == "evolve":
            return self._evolve(params.get("focus"))
        elif action == "get_insights":
            return self._get_insights()
        elif action == "save_insight":
            return self._save_insight(
                params.get("insight", ""), params.get("category", "general")
            )
        elif action == "score":
            return self._score()
        else:
            return SkillResult(success=False, message=f"Unknown action: {action}")

    def _get_action_history(self) -> List[Dict]:
        """Get the agent's recent action history."""
        if not self._get_actions_fn:
            return []
        return self._get_actions_fn()

    def _analyze(self) -> SkillResult:
        """Analyze recent actions for performance patterns."""
        actions = self._get_action_history()

        if not actions:
            return SkillResult(
                success=True,
                message="No actions to analyze yet",
                data={"action_count": 0, "recommendations": ["Take some actions first"]},
            )

        analysis = self._compute_analysis(actions)

        # Save analysis to history
        analysis["timestamp"] = datetime.now().isoformat()
        self._analysis_history.append(analysis)
        # Keep last 50 analyses
        self._analysis_history = self._analysis_history[-50:]

        return SkillResult(
            success=True,
            message=f"Analyzed {len(actions)} actions: score={analysis['score']}/100",
            data=analysis,
        )

    def _compute_analysis(self, actions: List[Dict]) -> Dict:
        """Compute detailed analysis of action history."""
        total = len(actions)
        if total == 0:
            return {"score": 0, "action_count": 0, "recommendations": []}

        # Count success/failure by skill
        skill_stats = defaultdict(lambda: {"success": 0, "failure": 0, "error": 0, "total": 0})
        total_cost = 0.0
        total_tokens = 0
        statuses = Counter()
        consecutive_failures = 0
        max_consecutive_failures = 0
        repeated_actions = Counter()

        for action in actions:
            tool = action.get("tool", "unknown")
            status = action.get("result", {}).get("status", "unknown")
            cost = action.get("api_cost_usd", 0) or 0
            tokens = action.get("tokens", 0) or 0

            skill_id = tool.split(":")[0] if ":" in tool else tool

            skill_stats[skill_id]["total"] += 1
            skill_stats[skill_id][status] = skill_stats[skill_id].get(status, 0) + 1

            statuses[status] += 1
            total_cost += cost
            total_tokens += tokens

            # Track consecutive failures
            if status in ("failed", "error"):
                consecutive_failures += 1
                max_consecutive_failures = max(max_consecutive_failures, consecutive_failures)
            else:
                consecutive_failures = 0

            # Track repeated identical actions (potential loops)
            action_key = f"{tool}:{json.dumps(action.get('params', {}), sort_keys=True)[:100]}"
            repeated_actions[action_key] += 1

        success_count = statuses.get("success", 0)
        failure_count = statuses.get("failed", 0) + statuses.get("error", 0)
        success_rate = success_count / total if total > 0 else 0

        # Detect repeated/looping actions
        loops = {k: v for k, v in repeated_actions.items() if v >= 3}

        # Generate recommendations
        recommendations = self._generate_recommendations(
            success_rate=success_rate,
            skill_stats=dict(skill_stats),
            max_consecutive_failures=max_consecutive_failures,
            loops=loops,
            total_cost=total_cost,
            total=total,
        )

        # Compute overall score (0-100)
        score = self._compute_score(
            success_rate=success_rate,
            max_consecutive_failures=max_consecutive_failures,
            loop_count=len(loops),
            total=total,
        )

        # Find worst-performing skills
        worst_skills = []
        for skill_id, stats in skill_stats.items():
            if stats["total"] >= 2:
                skill_success = stats.get("success", 0) / stats["total"]
                if skill_success < 0.5:
                    worst_skills.append({
                        "skill": skill_id,
                        "success_rate": round(skill_success, 2),
                        "total_uses": stats["total"],
                    })
        worst_skills.sort(key=lambda x: x["success_rate"])

        return {
            "action_count": total,
            "success_rate": round(success_rate, 3),
            "success_count": success_count,
            "failure_count": failure_count,
            "total_cost_usd": round(total_cost, 6),
            "total_tokens": total_tokens,
            "cost_per_action": round(total_cost / total, 6) if total > 0 else 0,
            "max_consecutive_failures": max_consecutive_failures,
            "detected_loops": len(loops),
            "loop_details": {k: v for k, v in list(loops.items())[:5]},
            "skill_stats": {k: dict(v) for k, v in skill_stats.items()},
            "worst_skills": worst_skills[:5],
            "score": score,
            "recommendations": recommendations,
        }

    def _compute_score(
        self,
        success_rate: float,
        max_consecutive_failures: int,
        loop_count: int,
        total: int,
    ) -> int:
        """Compute a 0-100 performance score."""
        if total == 0:
            return 0

        # Base score from success rate (0-60 points)
        score = success_rate * 60

        # Penalty for consecutive failures (up to -20 points)
        failure_penalty = min(max_consecutive_failures * 5, 20)
        score -= failure_penalty

        # Penalty for detected loops (up to -15 points)
        loop_penalty = min(loop_count * 5, 15)
        score -= loop_penalty

        # Bonus for action diversity (up to 10 points)
        if total >= 5:
            score += 10

        # Bonus for having enough actions to analyze (up to 5 points)
        if total >= 10:
            score += 5

        return max(0, min(100, int(score)))

    def _generate_recommendations(
        self,
        success_rate: float,
        skill_stats: Dict,
        max_consecutive_failures: int,
        loops: Dict,
        total_cost: float,
        total: int,
    ) -> List[str]:
        """Generate actionable strategy recommendations."""
        recommendations = []

        # Success rate recommendations
        if success_rate < 0.5:
            recommendations.append(
                "CRITICAL: Success rate below 50%. Consider switching strategies or "
                "using different skills for your current task."
            )
        elif success_rate < 0.75:
            recommendations.append(
                "Success rate could improve. Review which skills are failing and why."
            )

        # Consecutive failure detection
        if max_consecutive_failures >= 3:
            recommendations.append(
                f"Detected {max_consecutive_failures} consecutive failures. "
                "Stop and reassess your approach when actions keep failing."
            )

        # Loop detection
        if loops:
            loop_actions = list(loops.keys())[:3]
            recommendations.append(
                f"Detected {len(loops)} repeated action pattern(s). "
                "Avoid repeating the same action - try a different approach. "
                f"Loops: {', '.join(a.split(':')[0] for a in loop_actions)}"
            )

        # Cost efficiency
        if total > 0 and total_cost / total > 0.05:
            recommendations.append(
                f"High cost per action (${total_cost/total:.4f}). "
                "Consider batching operations or using cheaper alternatives."
            )

        # Skill-specific recommendations
        for skill_id, stats in skill_stats.items():
            if stats["total"] >= 3:
                skill_success = stats.get("success", 0) / stats["total"]
                if skill_success < 0.3:
                    recommendations.append(
                        f"Skill '{skill_id}' has {skill_success:.0%} success rate "
                        f"over {stats['total']} uses. Consider avoiding it or "
                        "checking its prerequisites."
                    )

        if not recommendations:
            recommendations.append(
                "Performance looks good. Keep monitoring and adjusting."
            )

        return recommendations

    def _evolve(self, focus: Optional[str] = None) -> SkillResult:
        """Analyze performance and apply a strategy improvement to the prompt."""
        actions = self._get_action_history()

        if not actions:
            return SkillResult(
                success=True,
                message="No actions to analyze - nothing to evolve from yet",
                data={"evolved": False},
            )

        analysis = self._compute_analysis(actions)

        # Generate strategy update
        strategy_lines = []

        # Add timestamp
        ts = datetime.now().strftime("%Y-%m-%d %H:%M")

        # Focus-specific strategies
        if focus == "efficiency":
            strategy_lines.extend(self._efficiency_strategy(analysis))
        elif focus == "success_rate":
            strategy_lines.extend(self._success_strategy(analysis))
        elif focus == "cost":
            strategy_lines.extend(self._cost_strategy(analysis))
        elif focus == "revenue":
            strategy_lines.extend(self._revenue_strategy(analysis))
        else:
            # Auto-detect what needs improvement
            if analysis["success_rate"] < 0.7:
                strategy_lines.extend(self._success_strategy(analysis))
            if analysis["detected_loops"] > 0:
                strategy_lines.extend(self._efficiency_strategy(analysis))
            if analysis["cost_per_action"] > 0.03:
                strategy_lines.extend(self._cost_strategy(analysis))

        if not strategy_lines:
            strategy_lines.append(
                f"Performance score: {analysis['score']}/100. Current strategy is working well."
            )

        # Build the prompt addition
        strategy_text = "\n".join(f"- {line}" for line in strategy_lines)
        prompt_addition = (
            f"\n\n=== EVOLVED STRATEGY ({ts}) ===\n"
            f"Based on analysis of {analysis['action_count']} actions "
            f"(score: {analysis['score']}/100, success: {analysis['success_rate']:.0%}):\n"
            f"{strategy_text}"
        )

        # Apply to prompt
        if self._append_prompt_fn:
            self._append_prompt_fn(prompt_addition)

        # Save insights
        for line in strategy_lines:
            self._persist_insight(line, focus or "auto")

        return SkillResult(
            success=True,
            message=f"Strategy evolved based on {analysis['action_count']} actions (score: {analysis['score']}/100)",
            data={
                "evolved": True,
                "score": analysis["score"],
                "strategies_applied": strategy_lines,
                "focus": focus or "auto",
                "analysis_summary": {
                    "success_rate": analysis["success_rate"],
                    "failure_count": analysis["failure_count"],
                    "detected_loops": analysis["detected_loops"],
                    "cost_per_action": analysis["cost_per_action"],
                },
            },
        )

    def _efficiency_strategy(self, analysis: Dict) -> List[str]:
        """Generate efficiency-focused strategies."""
        strategies = []
        if analysis["detected_loops"] > 0:
            strategies.append(
                "ANTI-LOOP: If an action fails twice in a row, switch to a completely different approach."
            )
            strategies.append(
                "Before repeating an action, check if the prerequisites have changed."
            )
        if analysis["max_consecutive_failures"] >= 3:
            strategies.append(
                "CIRCUIT BREAKER: After 3 consecutive failures, pause and re-plan from scratch."
            )
        strategies.append(
            "Prefer actions that accomplish multiple goals at once."
        )
        return strategies

    def _success_strategy(self, analysis: Dict) -> List[str]:
        """Generate success-rate-focused strategies."""
        strategies = []
        worst = analysis.get("worst_skills", [])
        if worst:
            skill_names = ", ".join(s["skill"] for s in worst[:3])
            strategies.append(
                f"AVOID or FIX: Skills with low success rates: {skill_names}"
            )
        if analysis["success_rate"] < 0.5:
            strategies.append(
                "CRITICAL: More actions are failing than succeeding. "
                "Simplify approach - do fewer things but do them right."
            )
        strategies.append(
            "Before each action, verify parameters are correct and prerequisites are met."
        )
        return strategies

    def _cost_strategy(self, analysis: Dict) -> List[str]:
        """Generate cost-focused strategies."""
        strategies = []
        if analysis["cost_per_action"] > 0.05:
            strategies.append(
                "HIGH COST: Consider using smaller/cheaper models for routine tasks."
            )
        strategies.append(
            "Batch related operations together to reduce per-action overhead."
        )
        strategies.append(
            "Plan before acting - think first, then execute with fewer wasted cycles."
        )
        return strategies

    def _revenue_strategy(self, analysis: Dict) -> List[str]:
        """Generate revenue-focused strategies."""
        strategies = []
        strategies.append(
            "Prioritize actions that directly generate revenue or build revenue-generating assets."
        )
        strategies.append(
            "Track ROI: every action should either generate revenue or improve capability to generate revenue."
        )
        strategies.append(
            "Look for opportunities to offer services: code review, content creation, data analysis."
        )
        return strategies

    def _score(self) -> SkillResult:
        """Get a numerical performance score."""
        actions = self._get_action_history()
        if not actions:
            return SkillResult(
                success=True,
                message="No actions yet - score is 0",
                data={"score": 0, "action_count": 0},
            )

        analysis = self._compute_analysis(actions)
        return SkillResult(
            success=True,
            message=f"Performance score: {analysis['score']}/100 ({len(actions)} actions analyzed)",
            data={
                "score": analysis["score"],
                "success_rate": analysis["success_rate"],
                "action_count": len(actions),
                "total_cost_usd": analysis["total_cost_usd"],
            },
        )

    # === Insight persistence ===

    def _load_insights(self) -> List[Dict]:
        """Load insights from persistent storage."""
        try:
            if INSIGHTS_FILE.exists():
                with open(INSIGHTS_FILE, "r") as f:
                    data = json.load(f)
                return data.get("insights", [])
        except (json.JSONDecodeError, IOError):
            pass
        return []

    def _persist_insight(self, insight: str, category: str = "general"):
        """Save an insight to persistent storage."""
        try:
            insights = self._load_insights()

            # Avoid duplicates
            existing_texts = {i.get("text", "") for i in insights}
            if insight in existing_texts:
                return

            insights.append({
                "text": insight,
                "category": category,
                "timestamp": datetime.now().isoformat(),
            })

            # Keep last 100 insights
            insights = insights[-100:]

            INSIGHTS_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(INSIGHTS_FILE, "w") as f:
                json.dump({"insights": insights, "updated": datetime.now().isoformat()}, f, indent=2)
        except (IOError, OSError):
            pass  # Non-critical - skip if can't write

    def _get_insights(self) -> SkillResult:
        """Get all accumulated insights."""
        insights = self._load_insights()

        # Group by category
        by_category = defaultdict(list)
        for insight in insights:
            cat = insight.get("category", "general")
            by_category[cat].append(insight.get("text", ""))

        return SkillResult(
            success=True,
            message=f"Found {len(insights)} accumulated insights",
            data={
                "total": len(insights),
                "by_category": dict(by_category),
                "recent": [i.get("text", "") for i in insights[-10:]],
            },
        )

    def _save_insight(self, insight: str, category: str = "general") -> SkillResult:
        """Manually save a strategic insight."""
        if not insight.strip():
            return SkillResult(success=False, message="Insight cannot be empty")

        self._persist_insight(insight.strip(), category)
        return SkillResult(
            success=True,
            message=f"Insight saved ({category}): {insight[:80]}...",
            data={"insight": insight, "category": category},
        )
