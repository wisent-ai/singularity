#!/usr/bin/env python3
"""
RevenueGoalAutoSetterSkill - Auto-set revenue goals from forecast data.

This skill bridges RevenueAnalyticsDashboard forecast data with GoalManager to
automatically create, update, and track revenue-related goals. Without this bridge,
the agent has revenue forecasts and a goal system, but no automated connection
between them. The agent must manually decide "what revenue targets should I set?"

With this skill, the agent can:
1. **Assess** - Pull forecast data and current revenue metrics
2. **Generate Goals** - Auto-create revenue goals based on forecasts and gaps
3. **Track** - Monitor goal progress against actual revenue data
4. **Adjust** - Auto-update goals when forecasts change significantly
5. **Report** - Show how revenue goals track against forecasts

The feedback loop:
  Forecast revenue -> Set goals -> Execute revenue skills -> Measure actual ->
  Compare to forecast -> Adjust goals -> Repeat

Pillar: Revenue Generation (primary), Goal Setting (autonomous target-setting)
"""

import json
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any

from .base import Skill, SkillResult, SkillManifest, SkillAction

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_FILE = DATA_DIR / "revenue_goal_auto_setter.json"
DASHBOARD_FILE = DATA_DIR / "revenue_analytics_dashboard.json"
GOALS_FILE = DATA_DIR / "goals.json"

# Default configuration
DEFAULT_CONFIG = {
    "auto_create_goals": False,       # If True, auto-create goals on assess
    "goal_stretch_factor": 1.2,       # Set goals 20% above forecast
    "min_daily_rate_for_goal": 0.001, # Min daily rate to create a growth goal
    "breakeven_priority": "critical", # Priority for break-even goals
    "growth_priority": "high",        # Priority for growth goals
    "optimization_priority": "medium",# Priority for optimization goals
    "default_deadline_days": 7,       # Default goal deadline
    "reassess_cooldown_hours": 6,     # Min hours between reassessments
    "significant_change_pct": 0.25,   # 25% change triggers goal update
    "max_active_revenue_goals": 5,    # Max concurrent revenue goals
}

# Goal templates based on revenue state
GOAL_TEMPLATES = {
    "breakeven": {
        "title": "Reach daily break-even: ${target}/day",
        "description": "Revenue must cover compute costs of ${cost}/day. "
                       "Current rate: ${current}/day. Gap: ${gap}/day.",
        "priority": "breakeven_priority",
        "milestones": [
            "Identify highest-ROI revenue service",
            "Optimize pricing for top service",
            "Increase service utilization",
            "Achieve break-even target",
        ],
    },
    "growth": {
        "title": "Grow revenue to ${target}/day",
        "description": "Forecast projects ${forecast}/day in {days} days. "
                       "Stretch target: ${target}/day ({stretch_pct}% above forecast).",
        "priority": "growth_priority",
        "milestones": [
            "Review revenue source mix",
            "Expand highest-margin service",
            "Hit 50% of target",
            "Achieve full target",
        ],
    },
    "diversification": {
        "title": "Diversify: activate {count} new revenue sources",
        "description": "Currently {active} of {total} revenue sources active. "
                       "Diversification reduces risk of single-source dependency.",
        "priority": "optimization_priority",
        "milestones": [
            "Identify inactive but viable revenue sources",
            "Configure and test first new source",
            "Achieve first revenue from new source",
        ],
    },
    "margin_improvement": {
        "title": "Improve profit margin to {target_pct}%",
        "description": "Current margin: {current_pct}%. Target: {target_pct}%. "
                       "Focus on cost reduction and pricing optimization.",
        "priority": "optimization_priority",
        "milestones": [
            "Analyze cost drivers per service",
            "Implement cost reduction for top expense",
            "Review and adjust pricing",
            "Achieve target margin",
        ],
    },
}


def _now_iso():
    return datetime.now().isoformat()


class RevenueGoalAutoSetterSkill(Skill):
    """
    Automatically creates and manages revenue goals from forecast data.

    Bridges RevenueAnalyticsDashboard forecasts with GoalManager to ensure
    the agent always has data-driven, actionable revenue targets.
    """

    def __init__(self, credentials: Dict[str, str] = None):
        super().__init__(credentials)
        self._ensure_data()

    def _ensure_data(self):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        if not DATA_FILE.exists():
            self._save(self._default_state())

    def _default_state(self) -> Dict:
        return {
            "config": DEFAULT_CONFIG.copy(),
            "created_goals": [],       # History of auto-created goals
            "assessments": [],         # Revenue assessment history
            "goal_tracking": {},       # goal_id -> tracking data
            "last_assess_at": None,
            "stats": {
                "total_goals_created": 0,
                "goals_achieved": 0,
                "goals_missed": 0,
                "total_assessments": 0,
            },
        }

    def _load(self) -> Dict:
        try:
            with open(DATA_FILE, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return self._default_state()

    def _save(self, data: Dict):
        data["last_updated"] = _now_iso()
        with open(DATA_FILE, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def _load_dashboard(self) -> Dict:
        try:
            with open(DASHBOARD_FILE, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _load_goals(self) -> Dict:
        try:
            with open(GOALS_FILE, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {"goals": []}

    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="revenue_goal_auto_setter",
            name="Revenue Goal Auto-Setter",
            version="1.0.0",
            category="meta",
            description="Auto-set revenue goals from RevenueAnalyticsDashboard forecast data",
            actions=[
                SkillAction(
                    name="assess",
                    description="Assess current revenue state and recommend goals",
                    parameters={
                        "auto_create": {
                            "type": "boolean",
                            "required": False,
                            "description": "Override config to auto-create goals (default: use config)",
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="create_goals",
                    description="Create recommended revenue goals in GoalManager",
                    parameters={
                        "goal_types": {
                            "type": "array",
                            "required": False,
                            "description": "Filter: only create these goal types (breakeven, growth, diversification, margin_improvement)",
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="track",
                    description="Track progress of auto-created revenue goals against actuals",
                    parameters={},
                    estimated_cost=0,
                ),
                SkillAction(
                    name="adjust",
                    description="Check if goals need adjustment based on new forecast data",
                    parameters={
                        "force": {
                            "type": "boolean",
                            "required": False,
                            "description": "Force adjustment even if within cooldown",
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="report",
                    description="Generate a report of revenue goal performance",
                    parameters={},
                    estimated_cost=0,
                ),
                SkillAction(
                    name="configure",
                    description="Update auto-setter configuration",
                    parameters={
                        "auto_create_goals": {"type": "boolean", "required": False},
                        "goal_stretch_factor": {"type": "number", "required": False},
                        "breakeven_priority": {"type": "string", "required": False},
                        "growth_priority": {"type": "string", "required": False},
                        "reassess_cooldown_hours": {"type": "number", "required": False},
                        "significant_change_pct": {"type": "number", "required": False},
                        "max_active_revenue_goals": {"type": "integer", "required": False},
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="status",
                    description="Show current auto-setter status and stats",
                    parameters={},
                    estimated_cost=0,
                ),
                SkillAction(
                    name="history",
                    description="Show history of auto-created goals and their outcomes",
                    parameters={
                        "limit": {"type": "integer", "required": False, "description": "Max records (default 20)"},
                    },
                    estimated_cost=0,
                ),
            ],
            required_credentials=[],
        )

    def check_credentials(self) -> bool:
        return True

    async def execute(self, action: str, params: Dict) -> SkillResult:
        handlers = {
            "assess": self._assess,
            "create_goals": self._create_goals,
            "track": self._track,
            "adjust": self._adjust,
            "report": self._report,
            "configure": self._configure,
            "status": self._status,
            "history": self._history,
        }
        handler = handlers.get(action)
        if handler:
            return handler(params)
        return SkillResult(success=False, message=f"Unknown action: {action}")

    # ── Core: Assess ──────────────────────────────────────────────────

    def _assess(self, params: Dict) -> SkillResult:
        """Assess current revenue state and generate goal recommendations."""
        state = self._load()
        config = state.get("config", DEFAULT_CONFIG)

        # Check cooldown
        last = state.get("last_assess_at")
        if last:
            cooldown_h = config.get("reassess_cooldown_hours", 6)
            try:
                last_dt = datetime.fromisoformat(last)
                if (datetime.now() - last_dt).total_seconds() < cooldown_h * 3600:
                    if not params.get("auto_create"):
                        # Return last assessment if within cooldown
                        if state.get("assessments"):
                            return SkillResult(
                                success=True,
                                message="Within cooldown. Returning last assessment.",
                                data=state["assessments"][-1],
                            )
            except (ValueError, TypeError):
                pass

        # Pull data from dashboard
        dash = self._load_dashboard()
        snapshots = dash.get("snapshots", [])
        dash_config = dash.get("config", {})

        # Compute revenue metrics
        metrics = self._compute_metrics(snapshots, dash_config)

        # Generate recommendations
        recommendations = self._generate_recommendations(metrics, config)

        # Store assessment
        assessment = {
            "timestamp": _now_iso(),
            "metrics": metrics,
            "recommendations": recommendations,
            "recommendation_count": len(recommendations),
        }

        state["assessments"] = (state.get("assessments") or [])[-49:] + [assessment]
        state["last_assess_at"] = _now_iso()
        state["stats"]["total_assessments"] = state["stats"].get("total_assessments", 0) + 1
        self._save(state)

        # Auto-create if configured
        auto_create = params.get("auto_create", config.get("auto_create_goals", False))
        created = []
        if auto_create and recommendations:
            result = self._create_goals({"_recommendations": recommendations})
            if result.success:
                created = result.data.get("created", [])

        return SkillResult(
            success=True,
            message=f"Assessment complete: {len(recommendations)} recommendations"
                    + (f", {len(created)} goals created" if created else ""),
            data={
                "metrics": metrics,
                "recommendations": recommendations,
                "goals_created": created,
            },
        )

    def _compute_metrics(self, snapshots: List[Dict], dash_config: Dict) -> Dict:
        """Compute revenue metrics from dashboard snapshots."""
        compute_cost_daily = dash_config.get("compute_cost_per_hour", 0.10) * 24
        target_daily = dash_config.get("revenue_target_daily", 1.00)

        if not snapshots:
            return {
                "current_revenue": 0,
                "daily_rate": 0,
                "growth_direction": "unknown",
                "compute_cost_daily": compute_cost_daily,
                "target_daily": target_daily,
                "at_breakeven": False,
                "profit_margin": 0,
                "sources_active": 0,
                "sources_total": 0,
                "snapshot_count": 0,
                "days_to_breakeven": -1,
                "forecast_7d": 0,
            }

        recent = snapshots[-20:]
        revenues = [s.get("total_revenue", 0) for s in recent]
        current = revenues[-1] if revenues else 0

        # Linear regression for growth rate
        n = len(revenues)
        daily_rate = 0
        if n >= 2:
            x_mean = (n - 1) / 2
            y_mean = sum(revenues) / n
            num = sum((i - x_mean) * (revenues[i] - y_mean) for i in range(n))
            den = sum((i - x_mean) ** 2 for i in range(n))
            slope = num / den if den > 0 else 0
            daily_rate = slope * 24  # Assuming hourly snapshots

        # Source diversity
        latest = snapshots[-1]
        by_source = latest.get("by_source", {})
        sources_active = sum(1 for s in by_source.values() if s.get("revenue", 0) > 0)
        sources_total = len(by_source) if by_source else 0

        # Profit margin
        total_cost = latest.get("total_cost", 0)
        profit_margin = 0
        if current > 0:
            profit_margin = ((current - total_cost) / current) * 100

        # Days to breakeven
        if daily_rate > 0 and current < compute_cost_daily:
            days_to_breakeven = max(0, round((compute_cost_daily - current) / daily_rate))
        elif current >= compute_cost_daily:
            days_to_breakeven = 0
        else:
            days_to_breakeven = -1

        # 7-day forecast
        forecast_7d = max(0, current + daily_rate * 7)

        return {
            "current_revenue": round(current, 6),
            "daily_rate": round(daily_rate, 6),
            "growth_direction": "positive" if daily_rate > 0 else "negative" if daily_rate < 0 else "flat",
            "compute_cost_daily": round(compute_cost_daily, 4),
            "target_daily": target_daily,
            "at_breakeven": current >= compute_cost_daily,
            "profit_margin": round(profit_margin, 2),
            "sources_active": sources_active,
            "sources_total": sources_total,
            "snapshot_count": len(snapshots),
            "days_to_breakeven": days_to_breakeven,
            "forecast_7d": round(forecast_7d, 6),
        }

    def _generate_recommendations(self, metrics: Dict, config: Dict) -> List[Dict]:
        """Generate goal recommendations based on metrics."""
        recs = []

        cost_daily = metrics["compute_cost_daily"]
        current = metrics["current_revenue"]
        daily_rate = metrics["daily_rate"]
        stretch = config.get("goal_stretch_factor", 1.2)
        deadline_days = config.get("default_deadline_days", 7)

        # 1. Break-even goal (if not at breakeven)
        if not metrics["at_breakeven"]:
            gap = cost_daily - current
            recs.append({
                "type": "breakeven",
                "priority": config.get("breakeven_priority", "critical"),
                "target": round(cost_daily, 4),
                "current": round(current, 6),
                "gap": round(gap, 4),
                "urgency": "high",
                "rationale": f"Revenue (${current:.4f}/day) below compute cost (${cost_daily:.4f}/day). "
                             f"Gap: ${gap:.4f}/day.",
                "template_vars": {
                    "target": f"{cost_daily:.4f}",
                    "cost": f"{cost_daily:.4f}",
                    "current": f"{current:.4f}",
                    "gap": f"{gap:.4f}",
                },
            })

        # 2. Growth goal (if positive growth or at breakeven)
        min_rate = config.get("min_daily_rate_for_goal", 0.001)
        if daily_rate >= min_rate or metrics["at_breakeven"]:
            forecast_7d = metrics["forecast_7d"]
            target = round(forecast_7d * stretch, 4)
            stretch_pct = int((stretch - 1) * 100)
            recs.append({
                "type": "growth",
                "priority": config.get("growth_priority", "high"),
                "target": target,
                "forecast": round(forecast_7d, 4),
                "stretch_pct": stretch_pct,
                "rationale": f"Forecast: ${forecast_7d:.4f}/day in 7 days. "
                             f"Stretch target: ${target:.4f}/day ({stretch_pct}% above).",
                "template_vars": {
                    "target": f"{target:.4f}",
                    "forecast": f"{forecast_7d:.4f}",
                    "days": "7",
                    "stretch_pct": str(stretch_pct),
                },
            })

        # 3. Diversification goal (if few sources active)
        total = metrics.get("sources_total", 0)
        active = metrics.get("sources_active", 0)
        if total > 0 and active < total and (total - active) >= 2:
            count = min(3, total - active)
            recs.append({
                "type": "diversification",
                "priority": config.get("optimization_priority", "medium"),
                "target_new_sources": count,
                "active": active,
                "total": total,
                "rationale": f"Only {active}/{total} revenue sources active. "
                             f"Activate {count} more for diversification.",
                "template_vars": {
                    "count": str(count),
                    "active": str(active),
                    "total": str(total),
                },
            })

        # 4. Margin improvement goal (if margin < 50%)
        margin = metrics.get("profit_margin", 0)
        if current > 0 and margin < 50:
            target_pct = min(70, margin + 20)
            recs.append({
                "type": "margin_improvement",
                "priority": config.get("optimization_priority", "medium"),
                "current_margin": round(margin, 2),
                "target_margin": round(target_pct, 2),
                "rationale": f"Profit margin at {margin:.1f}%. Target: {target_pct:.1f}%.",
                "template_vars": {
                    "current_pct": f"{margin:.1f}",
                    "target_pct": f"{target_pct:.1f}",
                },
            })

        return recs

    # ── Create Goals ──────────────────────────────────────────────────

    def _create_goals(self, params: Dict) -> SkillResult:
        """Create revenue goals in GoalManager from recommendations."""
        state = self._load()
        config = state.get("config", DEFAULT_CONFIG)

        # Get recommendations from params or last assessment
        recs = params.get("_recommendations")
        if not recs:
            assessments = state.get("assessments", [])
            if not assessments:
                return SkillResult(
                    success=False,
                    message="No assessment available. Run 'assess' first.",
                )
            recs = assessments[-1].get("recommendations", [])

        if not recs:
            return SkillResult(
                success=True,
                message="No recommendations to create goals from.",
                data={"created": []},
            )

        # Filter by type if specified
        goal_types = params.get("goal_types")
        if goal_types:
            recs = [r for r in recs if r.get("type") in goal_types]

        # Check max active goals
        max_active = config.get("max_active_revenue_goals", 5)
        existing = self._count_active_revenue_goals()
        available_slots = max(0, max_active - existing)
        if available_slots == 0:
            return SkillResult(
                success=True,
                message=f"Already at max active revenue goals ({max_active}). Complete or abandon existing goals first.",
                data={"created": [], "existing_active": existing, "max": max_active},
            )

        # Create goals
        created = []
        goals_data = self._load_goals()
        goals_list = goals_data.get("goals", [])
        deadline_days = config.get("default_deadline_days", 7)

        for rec in recs[:available_slots]:
            goal_type = rec.get("type", "unknown")
            template = GOAL_TEMPLATES.get(goal_type)
            if not template:
                continue

            # Check if we already have an active goal of this type
            if self._has_active_goal_of_type(goal_type, state):
                continue

            # Build goal from template
            tvars = rec.get("template_vars", {})
            title = template["title"]
            desc = template["description"]
            for k, v in tvars.items():
                title = title.replace(f"${{{k}}}", str(v)).replace(f"{{{k}}}", str(v))
                desc = desc.replace(f"${{{k}}}", str(v)).replace(f"{{{k}}}", str(v))

            priority_key = template.get("priority", "medium")
            priority = config.get(priority_key, priority_key)

            goal_id = f"goal_{uuid.uuid4().hex[:8]}"
            now = datetime.now()

            milestones = []
            for i, mt in enumerate(template.get("milestones", [])):
                milestones.append({
                    "index": i,
                    "title": mt,
                    "completed": False,
                    "completed_at": None,
                })

            goal = {
                "id": goal_id,
                "title": title,
                "description": desc,
                "pillar": "revenue",
                "priority": priority,
                "priority_score": {"critical": 4, "high": 3, "medium": 2, "low": 1}.get(priority, 2),
                "status": "active",
                "created_at": now.isoformat(),
                "deadline": (now + timedelta(days=deadline_days)).isoformat(),
                "depends_on": [],
                "milestones": milestones,
                "progress_notes": [],
                "completed_at": None,
                "outcome": None,
                "metadata": {
                    "auto_created_by": "revenue_goal_auto_setter",
                    "goal_type": goal_type,
                    "recommendation": rec,
                },
            }

            goals_list.append(goal)

            # Track in our state
            created_record = {
                "goal_id": goal_id,
                "goal_type": goal_type,
                "title": title,
                "priority": priority,
                "target": rec.get("target", rec.get("target_new_sources", rec.get("target_margin"))),
                "created_at": _now_iso(),
                "status": "active",
            }
            state["created_goals"] = (state.get("created_goals") or [])[-99:] + [created_record]
            state["goal_tracking"][goal_id] = {
                "goal_type": goal_type,
                "target": created_record["target"],
                "created_at": _now_iso(),
                "checks": [],
                "status": "active",
            }
            state["stats"]["total_goals_created"] = state["stats"].get("total_goals_created", 0) + 1

            created.append({"goal_id": goal_id, "type": goal_type, "title": title, "priority": priority})

        # Save goals and state
        if created:
            goals_data["goals"] = goals_list
            with open(GOALS_FILE, "w") as f:
                json.dump(goals_data, f, indent=2, default=str)
        self._save(state)

        return SkillResult(
            success=True,
            message=f"Created {len(created)} revenue goals",
            data={"created": created, "available_slots_remaining": available_slots - len(created)},
        )

    def _count_active_revenue_goals(self) -> int:
        """Count active revenue goals in GoalManager."""
        goals_data = self._load_goals()
        return sum(
            1 for g in goals_data.get("goals", [])
            if g.get("pillar") == "revenue" and g.get("status") == "active"
        )

    def _has_active_goal_of_type(self, goal_type: str, state: Dict) -> bool:
        """Check if we already have an active auto-created goal of this type."""
        for record in state.get("created_goals", []):
            if record.get("goal_type") == goal_type and record.get("status") == "active":
                # Verify it's still active in GoalManager
                goals_data = self._load_goals()
                for g in goals_data.get("goals", []):
                    if g.get("id") == record.get("goal_id") and g.get("status") == "active":
                        return True
                # Goal no longer active in GoalManager, update our record
                record["status"] = "completed_externally"
        return False

    # ── Track ─────────────────────────────────────────────────────────

    def _track(self, params: Dict) -> SkillResult:
        """Track progress of auto-created revenue goals against actuals."""
        state = self._load()
        tracking = state.get("goal_tracking", {})
        goals_data = self._load_goals()
        goals_map = {g["id"]: g for g in goals_data.get("goals", [])}

        # Get current metrics
        dash = self._load_dashboard()
        snapshots = dash.get("snapshots", [])
        dash_config = dash.get("config", {})
        metrics = self._compute_metrics(snapshots, dash_config)

        tracked = []
        for goal_id, track_data in tracking.items():
            if track_data.get("status") != "active":
                continue

            goal = goals_map.get(goal_id)
            if not goal:
                track_data["status"] = "goal_not_found"
                continue

            if goal.get("status") != "active":
                track_data["status"] = goal.get("status", "unknown")
                if goal.get("status") == "completed":
                    state["stats"]["goals_achieved"] = state["stats"].get("goals_achieved", 0) + 1
                continue

            # Check progress based on goal type
            goal_type = track_data.get("goal_type")
            target = track_data.get("target", 0)
            progress = self._compute_goal_progress(goal_type, target, metrics)

            check = {
                "timestamp": _now_iso(),
                "progress_pct": progress["pct"],
                "current_value": progress["current"],
                "target_value": target,
                "on_track": progress["on_track"],
            }
            track_data.setdefault("checks", [])
            track_data["checks"] = track_data["checks"][-19:] + [check]

            milestones_done = sum(1 for m in goal.get("milestones", []) if m.get("completed"))
            milestones_total = len(goal.get("milestones", []))

            tracked.append({
                "goal_id": goal_id,
                "type": goal_type,
                "title": goal.get("title", ""),
                "progress_pct": progress["pct"],
                "on_track": progress["on_track"],
                "milestones": f"{milestones_done}/{milestones_total}",
                "current": progress["current"],
                "target": target,
            })

        self._save(state)

        return SkillResult(
            success=True,
            message=f"Tracked {len(tracked)} active revenue goals",
            data={"tracked_goals": tracked, "current_metrics": metrics},
        )

    def _compute_goal_progress(self, goal_type: str, target: Any, metrics: Dict) -> Dict:
        """Compute progress percentage for a goal type."""
        if goal_type == "breakeven":
            current = metrics.get("current_revenue", 0)
            target_val = float(target) if target else metrics.get("compute_cost_daily", 1)
            pct = min(100, (current / target_val * 100) if target_val > 0 else 0)
            return {"pct": round(pct, 1), "current": current, "on_track": metrics.get("days_to_breakeven", -1) >= 0}

        elif goal_type == "growth":
            current = metrics.get("current_revenue", 0)
            target_val = float(target) if target else 1
            pct = min(100, (current / target_val * 100) if target_val > 0 else 0)
            on_track = metrics.get("daily_rate", 0) > 0
            return {"pct": round(pct, 1), "current": current, "on_track": on_track}

        elif goal_type == "diversification":
            active = metrics.get("sources_active", 0)
            target_val = int(target) if target else 1
            # We want target_val NEW sources, so progress is how many are active vs when we started
            pct = min(100, (active / max(1, target_val + active) * 100))
            return {"pct": round(pct, 1), "current": active, "on_track": True}

        elif goal_type == "margin_improvement":
            current_margin = metrics.get("profit_margin", 0)
            target_val = float(target) if target else 50
            pct = min(100, (current_margin / target_val * 100) if target_val > 0 else 0)
            return {"pct": round(pct, 1), "current": current_margin, "on_track": current_margin > 0}

        return {"pct": 0, "current": 0, "on_track": False}

    # ── Adjust ────────────────────────────────────────────────────────

    def _adjust(self, params: Dict) -> SkillResult:
        """Check if goals need adjustment based on new forecast data."""
        state = self._load()
        config = state.get("config", DEFAULT_CONFIG)

        # Run a fresh assessment
        dash = self._load_dashboard()
        snapshots = dash.get("snapshots", [])
        dash_config = dash.get("config", {})
        metrics = self._compute_metrics(snapshots, dash_config)

        # Compare against last assessment
        assessments = state.get("assessments", [])
        if not assessments:
            return SkillResult(
                success=True,
                message="No prior assessment. Run 'assess' first.",
                data={"adjustments": []},
            )

        last_metrics = assessments[-1].get("metrics", {})
        threshold = config.get("significant_change_pct", 0.25)

        adjustments = []

        # Check revenue change
        old_rev = last_metrics.get("current_revenue", 0)
        new_rev = metrics.get("current_revenue", 0)
        if old_rev > 0:
            change_pct = abs(new_rev - old_rev) / old_rev
            if change_pct >= threshold:
                adjustments.append({
                    "type": "revenue_change",
                    "old_value": round(old_rev, 6),
                    "new_value": round(new_rev, 6),
                    "change_pct": round(change_pct * 100, 1),
                    "direction": "up" if new_rev > old_rev else "down",
                    "action": "reassess_goals",
                })

        # Check growth rate change
        old_rate = last_metrics.get("daily_rate", 0)
        new_rate = metrics.get("daily_rate", 0)
        if old_rate != 0:
            rate_change = abs(new_rate - old_rate) / abs(old_rate)
            if rate_change >= threshold:
                adjustments.append({
                    "type": "growth_rate_change",
                    "old_value": round(old_rate, 6),
                    "new_value": round(new_rate, 6),
                    "change_pct": round(rate_change * 100, 1),
                    "direction": "accelerating" if new_rate > old_rate else "decelerating",
                    "action": "update_growth_targets",
                })

        # Check breakeven status change
        was_breakeven = last_metrics.get("at_breakeven", False)
        is_breakeven = metrics.get("at_breakeven", False)
        if was_breakeven != is_breakeven:
            adjustments.append({
                "type": "breakeven_change",
                "old_value": was_breakeven,
                "new_value": is_breakeven,
                "action": "create_breakeven_goal" if not is_breakeven else "celebrate_and_set_growth",
            })

        # If significant changes, trigger reassessment
        needs_reassess = len(adjustments) > 0 or params.get("force", False)
        if needs_reassess:
            assess_result = self._assess({"auto_create": config.get("auto_create_goals", False)})
            return SkillResult(
                success=True,
                message=f"Found {len(adjustments)} significant changes. Reassessment triggered.",
                data={
                    "adjustments": adjustments,
                    "reassessment": assess_result.data,
                    "needs_action": True,
                },
            )

        return SkillResult(
            success=True,
            message="No significant changes. Goals remain valid.",
            data={"adjustments": [], "needs_action": False},
        )

    # ── Report ────────────────────────────────────────────────────────

    def _report(self, params: Dict) -> SkillResult:
        """Generate a comprehensive revenue goal performance report."""
        state = self._load()
        stats = state.get("stats", {})
        tracking = state.get("goal_tracking", {})
        assessments = state.get("assessments", [])

        # Current metrics
        dash = self._load_dashboard()
        snapshots = dash.get("snapshots", [])
        dash_config = dash.get("config", {})
        metrics = self._compute_metrics(snapshots, dash_config)

        # Goal outcomes
        active_count = sum(1 for t in tracking.values() if t.get("status") == "active")
        achieved_count = stats.get("goals_achieved", 0)
        missed_count = stats.get("goals_missed", 0)
        total_created = stats.get("total_goals_created", 0)

        # Achievement rate
        completed = achieved_count + missed_count
        achievement_rate = (achieved_count / completed * 100) if completed > 0 else 0

        # Latest assessment summary
        latest_recs = []
        if assessments:
            latest_recs = assessments[-1].get("recommendations", [])

        report = {
            "summary": {
                "total_goals_created": total_created,
                "currently_active": active_count,
                "achieved": achieved_count,
                "missed": missed_count,
                "achievement_rate": round(achievement_rate, 1),
                "total_assessments": stats.get("total_assessments", 0),
            },
            "current_state": {
                "revenue": metrics.get("current_revenue", 0),
                "daily_rate": metrics.get("daily_rate", 0),
                "at_breakeven": metrics.get("at_breakeven", False),
                "profit_margin": metrics.get("profit_margin", 0),
                "forecast_7d": metrics.get("forecast_7d", 0),
            },
            "latest_recommendations": [
                {"type": r.get("type"), "priority": r.get("priority"), "rationale": r.get("rationale", "")}
                for r in latest_recs
            ],
            "active_goal_tracking": [
                {
                    "goal_id": gid,
                    "type": td.get("goal_type"),
                    "target": td.get("target"),
                    "latest_check": td.get("checks", [{}])[-1] if td.get("checks") else {},
                }
                for gid, td in tracking.items()
                if td.get("status") == "active"
            ],
        }

        return SkillResult(
            success=True,
            message=f"Revenue Goal Report: {active_count} active, {achieved_count} achieved, "
                    f"{achievement_rate:.0f}% success rate",
            data=report,
        )

    # ── Configure ─────────────────────────────────────────────────────

    def _configure(self, params: Dict) -> SkillResult:
        """Update auto-setter configuration."""
        state = self._load()
        config = state.get("config", DEFAULT_CONFIG)
        updated = []

        valid_keys = set(DEFAULT_CONFIG.keys())
        for key, value in params.items():
            if key in valid_keys:
                config[key] = value
                updated.append(key)

        if not updated:
            return SkillResult(
                success=True,
                message="No valid configuration keys provided.",
                data={"config": config, "valid_keys": sorted(valid_keys)},
            )

        state["config"] = config
        self._save(state)

        return SkillResult(
            success=True,
            message=f"Updated config: {', '.join(updated)}",
            data={"config": config, "updated": updated},
        )

    # ── Status ────────────────────────────────────────────────────────

    def _status(self, params: Dict) -> SkillResult:
        """Show current auto-setter status."""
        state = self._load()
        config = state.get("config", DEFAULT_CONFIG)
        stats = state.get("stats", {})
        tracking = state.get("goal_tracking", {})

        active = sum(1 for t in tracking.values() if t.get("status") == "active")

        return SkillResult(
            success=True,
            message=f"Revenue Goal Auto-Setter: {active} active goals, "
                    f"{stats.get('total_goals_created', 0)} total created",
            data={
                "config": config,
                "stats": stats,
                "active_goals": active,
                "last_assess_at": state.get("last_assess_at"),
                "total_assessments": len(state.get("assessments", [])),
            },
        )

    # ── History ───────────────────────────────────────────────────────

    def _history(self, params: Dict) -> SkillResult:
        """Show history of auto-created goals."""
        state = self._load()
        limit = params.get("limit", 20)
        created = state.get("created_goals", [])
        recent = created[-limit:]

        return SkillResult(
            success=True,
            message=f"Showing {len(recent)} of {len(created)} auto-created goals",
            data={"goals": list(reversed(recent)), "total": len(created)},
        )
