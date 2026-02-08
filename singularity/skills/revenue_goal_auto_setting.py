#!/usr/bin/env python3
"""
RevenueGoalAutoSettingSkill - Auto-set revenue goals from analytics forecasts.

RevenueAnalyticsDashboardSkill tracks revenue, costs, profit margins, and can
forecast daily growth rates and days to breakeven. GoalManagerSkill creates and
tracks pillar-aligned goals with milestones and priorities. But there is no
automated connection between revenue analytics and goal setting — agents must
manually create revenue goals based on data they read themselves.

This bridge connects them so that:

1. GENERATE: Analyze forecast data and auto-create revenue goals with milestones
2. REVIEW: Show all auto-generated revenue goals and their progress vs actuals
3. ADJUST: Re-evaluate goals when actuals diverge from forecasts by a threshold
4. TRACK: Compare goal milestones against actual revenue snapshots
5. RECOMMEND: Suggest goal priorities based on profitability analysis (high-margin first)
6. CONFIGURE: Set target margins, growth rates, and auto-adjustment thresholds

Integration flow:
  RevenueAnalytics.forecast → Bridge analyzes → GoalManager.create(revenue goals)
  RevenueAnalytics.profitability → Bridge ranks → GoalManager goals sorted by ROI
  RevenueAnalytics.snapshot → Bridge compares → GoalManager.progress if milestone hit

Without this bridge, agents fly blind on revenue targets. With it, forecast data
automatically becomes actionable goals, goals auto-adjust when forecasts change,
and profitability analysis drives goal prioritization.

Pillars: Revenue Generation (automated revenue targeting)
         Goal Setting (data-driven goal creation)
         Self-Improvement (continuous adjustment based on actuals)
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from .base import Skill, SkillAction, SkillManifest, SkillResult

BRIDGE_FILE = Path(__file__).parent.parent / "data" / "revenue_goal_auto_setting.json"
MAX_LOG_ENTRIES = 500
MAX_GOALS_TRACKED = 200


class RevenueGoalAutoSettingSkill(Skill):
    """Bridge between RevenueAnalyticsDashboard and GoalManager for auto revenue goals."""

    def __init__(self, credentials: Dict[str, str] = None):
        self.credentials = credentials or {}
        self._ensure_data()

    def _ensure_data(self):
        BRIDGE_FILE.parent.mkdir(parents=True, exist_ok=True)
        if not BRIDGE_FILE.exists():
            self._save(self._default_state())

    def _default_state(self) -> Dict:
        return {
            "config": {
                "auto_generate_enabled": True,
                "target_daily_growth_rate": 0.05,
                "target_profit_margin_pct": 20.0,
                "adjustment_threshold_pct": 25.0,
                "min_revenue_for_goal": 0.01,
                "default_goal_deadline_hours": 168,  # 1 week
                "high_margin_threshold": 40.0,
                "low_margin_threshold": 10.0,
                "max_active_goals": 5,
            },
            "generated_goals": [],  # [{goal_id, title, target_revenue, source, ...}]
            "adjustments": [],  # [{timestamp, goal_id, old_target, new_target, reason}]
            "recommendations": [],  # [{timestamp, source, action, reason, priority}]
            "event_log": [],
            "stats": {
                "goals_generated": 0,
                "goals_adjusted": 0,
                "goals_achieved": 0,
                "goals_missed": 0,
                "recommendations_made": 0,
                "total_target_revenue": 0.0,
                "total_achieved_revenue": 0.0,
            },
        }

    def _load(self) -> Dict:
        try:
            return json.loads(BRIDGE_FILE.read_text())
        except Exception:
            return self._default_state()

    def _save(self, data: Dict):
        if len(data.get("event_log", [])) > MAX_LOG_ENTRIES:
            data["event_log"] = data["event_log"][-MAX_LOG_ENTRIES:]
        if len(data.get("generated_goals", [])) > MAX_GOALS_TRACKED:
            data["generated_goals"] = data["generated_goals"][-MAX_GOALS_TRACKED:]
        if len(data.get("adjustments", [])) > MAX_GOALS_TRACKED:
            data["adjustments"] = data["adjustments"][-MAX_GOALS_TRACKED:]
        if len(data.get("recommendations", [])) > MAX_GOALS_TRACKED:
            data["recommendations"] = data["recommendations"][-MAX_GOALS_TRACKED:]
        BRIDGE_FILE.parent.mkdir(parents=True, exist_ok=True)
        BRIDGE_FILE.write_text(json.dumps(data, indent=2, default=str))

    def _log_event(self, data: Dict, event_type: str, details: Dict):
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "event": event_type,
            **details,
        }
        data["event_log"].append(entry)

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="revenue_goal_auto_setting",
            name="Revenue Goal Auto-Setting",
            description=(
                "Auto-set revenue goals from RevenueAnalyticsDashboard forecast data. "
                "Creates, tracks, and adjusts goals based on growth rates and profitability."
            ),
            version="1.0.0",
            category="revenue",
            actions=self._get_actions(),
            required_credentials=[],
        )

    def _get_actions(self) -> List[SkillAction]:
        return [
            SkillAction(
                name="generate",
                description=(
                    "Analyze revenue forecast data and auto-create goals with milestones. "
                    "Pass forecast and profitability data to create prioritized revenue goals."
                ),
                parameters={
                    "forecast": "dict - forecast data with daily_growth_rate, forecasted_days",
                    "profitability": "dict (optional) - profitability data with margin sources",
                    "overview": "dict (optional) - revenue overview with total_revenue, total_profit",
                    "dry_run": "bool (optional) - preview goals without creating",
                },
            ),
            SkillAction(
                name="review",
                description=(
                    "Show all auto-generated revenue goals, their targets, and progress. "
                    "Includes achievement rates and comparison with actuals."
                ),
                parameters={
                    "status_filter": "str (optional) - active | achieved | missed | adjusted",
                },
            ),
            SkillAction(
                name="adjust",
                description=(
                    "Re-evaluate goals when actuals diverge from forecasts. "
                    "Pass current revenue data to auto-adjust targets."
                ),
                parameters={
                    "goal_id": "str (optional) - specific goal to adjust; adjusts all if omitted",
                    "current_revenue": "float - actual current revenue",
                    "new_forecast": "dict (optional) - updated forecast data",
                    "dry_run": "bool (optional) - preview adjustments without applying",
                },
            ),
            SkillAction(
                name="track",
                description=(
                    "Compare goal milestones against actual revenue. "
                    "Mark milestones as achieved when targets are met."
                ),
                parameters={
                    "actual_revenue": "float - current total revenue",
                    "by_source": "dict (optional) - revenue breakdown by source/skill",
                },
            ),
            SkillAction(
                name="recommend",
                description=(
                    "Suggest goal priorities based on profitability analysis. "
                    "Identifies high-margin opportunities and low performers to deprioritize."
                ),
                parameters={
                    "profitability": "dict - profitability data from RevenueAnalytics",
                    "by_source": "dict (optional) - per-source revenue breakdown",
                },
            ),
            SkillAction(
                name="configure",
                description="Update auto-setting configuration (thresholds, targets, limits).",
                parameters={
                    "auto_generate_enabled": "bool (optional)",
                    "target_daily_growth_rate": "float (optional)",
                    "target_profit_margin_pct": "float (optional)",
                    "adjustment_threshold_pct": "float (optional)",
                    "min_revenue_for_goal": "float (optional)",
                    "default_goal_deadline_hours": "float (optional)",
                    "max_active_goals": "int (optional)",
                },
            ),
            SkillAction(
                name="status",
                description="Show bridge status: active goals, stats, recent events, configuration.",
                parameters={},
            ),
        ]

    def estimate_cost(self, action: str, parameters: Dict) -> float:
        return 0.0

    async def execute(self, action: str, parameters: Dict) -> SkillResult:
        actions = {
            "generate": self._generate,
            "review": self._review,
            "adjust": self._adjust,
            "track": self._track,
            "recommend": self._recommend,
            "configure": self._configure,
            "status": self._status,
        }
        handler = actions.get(action)
        if not handler:
            return SkillResult(
                success=False,
                message=f"Unknown action: {action}. Available: {list(actions.keys())}",
            )
        return handler(parameters)

    # ------------------------------------------------------------------
    # Generate: create revenue goals from forecast data
    # ------------------------------------------------------------------

    def _generate(self, params: Dict) -> SkillResult:
        """Analyze forecast and create revenue goals with milestones."""
        forecast = params.get("forecast")
        if not forecast:
            return SkillResult(
                success=False, message="Required: forecast (dict with forecast data)"
            )

        dry_run = params.get("dry_run", False)
        data = self._load()
        config = data["config"]

        if not config["auto_generate_enabled"] and not dry_run:
            return SkillResult(
                success=True,
                message="Auto-generation disabled. Use configure to enable or pass dry_run=True.",
                data={"auto_generate_enabled": False},
            )

        # Extract forecast metrics
        daily_growth = forecast.get("daily_growth_rate", 0)
        days_to_breakeven = forecast.get("days_to_breakeven")
        forecasted_days = forecast.get("forecasted_days", [])

        overview = params.get("overview", {})
        current_revenue = overview.get("total_revenue", 0)
        profit_margin = overview.get("profit_margin_pct", 0)

        # Determine goal priority from forecast health
        if daily_growth >= config["target_daily_growth_rate"]:
            priority = "medium"
            growth_status = "on_track"
        elif daily_growth > 0:
            priority = "high"
            growth_status = "below_target"
        else:
            priority = "critical"
            growth_status = "declining"

        # Count current active goals
        active_goals = [g for g in data["generated_goals"] if g.get("status") == "active"]
        if len(active_goals) >= config["max_active_goals"] and not dry_run:
            return SkillResult(
                success=False,
                message=(
                    f"Max active goals ({config['max_active_goals']}) reached. "
                    f"Complete or remove existing goals first."
                ),
                data={"active_goals": len(active_goals), "max": config["max_active_goals"]},
            )

        # Build milestones from forecast
        milestones = []
        target_revenue = current_revenue

        if forecasted_days:
            for day_data in forecasted_days[:7]:  # Max 7 milestones (1 week)
                day_rev = day_data.get("revenue", 0)
                if day_rev > 0:
                    target_revenue = max(target_revenue, day_rev)
                    day_label = day_data.get("day", len(milestones) + 1)
                    milestones.append(f"Day {day_label}: ${day_rev:.4f} revenue")
        elif daily_growth > 0:
            # Generate milestones from growth rate
            projected = max(current_revenue, config["min_revenue_for_goal"])
            for i in range(1, 8):
                projected *= 1 + daily_growth
                milestones.append(f"Day {i}: ${projected:.4f} revenue")
            target_revenue = projected

        if not milestones:
            milestones = [
                f"Reach ${config['min_revenue_for_goal']:.4f} total revenue",
                f"Achieve {config['target_profit_margin_pct']}% profit margin",
                "Maintain positive daily growth for 3 days",
            ]
            target_revenue = config["min_revenue_for_goal"]

        # Build goal title
        if target_revenue > 0:
            title = f"Revenue target: ${target_revenue:.4f} ({growth_status})"
        else:
            title = f"Revenue growth: achieve positive daily growth ({growth_status})"

        # Build goal record
        now = datetime.utcnow().isoformat()
        goal_id = f"rev-goal-{now.replace(':', '-').replace('.', '-')}"

        goal_record = {
            "goal_id": goal_id,
            "title": title,
            "target_revenue": target_revenue,
            "starting_revenue": current_revenue,
            "daily_growth_rate": daily_growth,
            "priority": priority,
            "growth_status": growth_status,
            "milestones": milestones,
            "milestones_achieved": 0,
            "deadline_hours": config["default_goal_deadline_hours"],
            "status": "active",
            "created_at": now,
            "profit_margin_at_creation": profit_margin,
            "forecast_snapshot": {
                "daily_growth_rate": daily_growth,
                "days_to_breakeven": days_to_breakeven,
            },
        }

        if dry_run:
            self._log_event(
                data,
                "generate_dry_run",
                {
                    "goal_id": goal_id,
                    "target_revenue": target_revenue,
                    "priority": priority,
                },
            )
            self._save(data)
            return SkillResult(
                success=True,
                message=f"DRY RUN: Would create '{title}' with {len(milestones)} milestones",
                data={
                    "goal": goal_record,
                    "dry_run": True,
                },
            )

        data["generated_goals"].append(goal_record)
        data["stats"]["goals_generated"] += 1
        data["stats"]["total_target_revenue"] += target_revenue

        self._log_event(
            data,
            "goal_generated",
            {
                "goal_id": goal_id,
                "title": title,
                "target_revenue": target_revenue,
                "priority": priority,
                "milestones_count": len(milestones),
            },
        )

        self._save(data)

        return SkillResult(
            success=True,
            message=f"Revenue goal created: '{title}' ({priority} priority, {len(milestones)} milestones)",
            data={
                "goal": goal_record,
                "active_goals": len(active_goals) + 1,
            },
        )

    # ------------------------------------------------------------------
    # Review: show generated goals and progress
    # ------------------------------------------------------------------

    def _review(self, params: Dict) -> SkillResult:
        """Show all auto-generated revenue goals and their status."""
        data = self._load()
        status_filter = params.get("status_filter")

        goals = data["generated_goals"]
        if status_filter:
            goals = [g for g in goals if g.get("status") == status_filter]

        # Compute summary stats
        active = sum(1 for g in data["generated_goals"] if g.get("status") == "active")
        achieved = sum(1 for g in data["generated_goals"] if g.get("status") == "achieved")
        missed = sum(1 for g in data["generated_goals"] if g.get("status") == "missed")
        adjusted = sum(1 for g in data["generated_goals"] if g.get("adjusted"))

        total_target = sum(g.get("target_revenue", 0) for g in data["generated_goals"])
        total_achieved = sum(
            g.get("achieved_revenue", 0)
            for g in data["generated_goals"]
            if g.get("status") == "achieved"
        )

        achievement_rate = achieved / (achieved + missed) * 100 if (achieved + missed) > 0 else 0

        return SkillResult(
            success=True,
            message=(
                f"Revenue goals: {active} active, {achieved} achieved, "
                f"{missed} missed ({achievement_rate:.0f}% success rate)"
            ),
            data={
                "goals": goals,
                "summary": {
                    "active": active,
                    "achieved": achieved,
                    "missed": missed,
                    "adjusted": adjusted,
                    "total_goals": len(data["generated_goals"]),
                    "total_target_revenue": total_target,
                    "total_achieved_revenue": total_achieved,
                    "achievement_rate_pct": round(achievement_rate, 1),
                },
                "filter_applied": status_filter,
            },
        )

    # ------------------------------------------------------------------
    # Adjust: re-evaluate goals based on new data
    # ------------------------------------------------------------------

    def _adjust(self, params: Dict) -> SkillResult:
        """Re-evaluate goals when actuals diverge from forecasts."""
        current_revenue = params.get("current_revenue")
        if current_revenue is None:
            return SkillResult(success=False, message="Required: current_revenue (float)")

        dry_run = params.get("dry_run", False)
        specific_goal_id = params.get("goal_id")
        new_forecast = params.get("new_forecast", {})
        data = self._load()
        config = data["config"]
        threshold = config["adjustment_threshold_pct"]

        adjustments_made = []
        goals_to_check = data["generated_goals"]

        if specific_goal_id:
            goals_to_check = [g for g in goals_to_check if g["goal_id"] == specific_goal_id]
            if not goals_to_check:
                return SkillResult(
                    success=False,
                    message=f"Goal '{specific_goal_id}' not found.",
                )

        for goal in goals_to_check:
            if goal.get("status") != "active":
                continue

            target = goal.get("target_revenue", 0)
            starting = goal.get("starting_revenue", 0)

            if target <= 0:
                continue

            # Calculate divergence: how far is actual from expected trajectory
            expected_progress = target - starting
            actual_progress = current_revenue - starting

            if expected_progress > 0:
                divergence_pct = abs(actual_progress - expected_progress) / expected_progress * 100
            else:
                divergence_pct = 0

            if divergence_pct < threshold:
                continue

            # Calculate new target
            if actual_progress > expected_progress:
                # Ahead of forecast - raise target
                new_growth = new_forecast.get("daily_growth_rate", goal.get("daily_growth_rate", 0))
                new_target = current_revenue * (1 + new_growth * 7)
                reason = "ahead_of_forecast"
                new_priority = "medium"
            else:
                # Behind forecast - lower target or increase urgency
                new_target = current_revenue + (expected_progress * 0.75)
                reason = "behind_forecast"
                new_priority = "critical"

            adjustment = {
                "timestamp": datetime.utcnow().isoformat(),
                "goal_id": goal["goal_id"],
                "old_target": target,
                "new_target": round(new_target, 6),
                "divergence_pct": round(divergence_pct, 1),
                "reason": reason,
                "old_priority": goal.get("priority"),
                "new_priority": new_priority,
                "dry_run": dry_run,
            }

            if not dry_run:
                goal["target_revenue"] = round(new_target, 6)
                goal["priority"] = new_priority
                goal["adjusted"] = True
                goal["last_adjusted_at"] = datetime.utcnow().isoformat()
                data["adjustments"].append(adjustment)
                data["stats"]["goals_adjusted"] += 1

            adjustments_made.append(adjustment)

        self._log_event(
            data,
            "adjust" if not dry_run else "adjust_dry_run",
            {
                "adjustments_count": len(adjustments_made),
                "current_revenue": current_revenue,
            },
        )

        self._save(data)

        return SkillResult(
            success=True,
            message=(
                f"{'DRY RUN: ' if dry_run else ''}"
                f"{len(adjustments_made)} goals adjusted based on ${current_revenue:.4f} actual revenue"
            ),
            data={
                "adjustments": adjustments_made,
                "dry_run": dry_run,
                "threshold_pct": threshold,
            },
        )

    # ------------------------------------------------------------------
    # Track: compare milestones against actuals
    # ------------------------------------------------------------------

    def _track(self, params: Dict) -> SkillResult:
        """Compare goal milestones against actual revenue."""
        actual_revenue = params.get("actual_revenue")
        if actual_revenue is None:
            return SkillResult(success=False, message="Required: actual_revenue (float)")

        data = self._load()

        tracked = []
        newly_achieved = 0

        for goal in data["generated_goals"]:
            if goal.get("status") != "active":
                continue

            target = goal.get("target_revenue", 0)

            # Check if revenue target is met
            if actual_revenue >= target and target > 0:
                goal["status"] = "achieved"
                goal["achieved_at"] = datetime.utcnow().isoformat()
                goal["achieved_revenue"] = actual_revenue
                goal["milestones_achieved"] = len(goal.get("milestones", []))
                data["stats"]["goals_achieved"] += 1
                data["stats"]["total_achieved_revenue"] += actual_revenue
                newly_achieved += 1
                tracked.append(
                    {
                        "goal_id": goal["goal_id"],
                        "title": goal["title"],
                        "status": "achieved",
                        "target": target,
                        "actual": actual_revenue,
                    }
                )
            else:
                # Track milestone progress
                milestones = goal.get("milestones", [])
                achieved_count = 0
                for i, ms in enumerate(milestones):
                    # Parse milestone target if it contains a dollar amount
                    try:
                        ms_target = float(ms.split("$")[1].split(" ")[0] if "$" in ms else "0")
                    except (IndexError, ValueError):
                        ms_target = 0

                    if ms_target > 0 and actual_revenue >= ms_target:
                        achieved_count += 1

                goal["milestones_achieved"] = achieved_count
                progress_pct = achieved_count / len(milestones) * 100 if milestones else 0

                tracked.append(
                    {
                        "goal_id": goal["goal_id"],
                        "title": goal["title"],
                        "status": "in_progress",
                        "target": target,
                        "actual": actual_revenue,
                        "milestones_achieved": achieved_count,
                        "milestones_total": len(milestones),
                        "progress_pct": round(progress_pct, 1),
                    }
                )

        self._log_event(
            data,
            "tracked",
            {
                "actual_revenue": actual_revenue,
                "goals_tracked": len(tracked),
                "newly_achieved": newly_achieved,
            },
        )

        self._save(data)

        return SkillResult(
            success=True,
            message=(
                f"Tracked {len(tracked)} goals against ${actual_revenue:.4f}: "
                f"{newly_achieved} achieved, {len(tracked) - newly_achieved} in progress"
            ),
            data={
                "tracked_goals": tracked,
                "actual_revenue": actual_revenue,
                "newly_achieved": newly_achieved,
            },
        )

    # ------------------------------------------------------------------
    # Recommend: suggest priorities based on profitability
    # ------------------------------------------------------------------

    def _recommend(self, params: Dict) -> SkillResult:
        """Suggest goal priorities based on profitability analysis."""
        profitability = params.get("profitability")
        if not profitability:
            return SkillResult(success=False, message="Required: profitability (dict)")

        data = self._load()
        config = data["config"]

        recommendations = []
        high_threshold = config["high_margin_threshold"]
        low_threshold = config["low_margin_threshold"]

        overall_margin = profitability.get("overall_margin_pct", 0)
        best_sources = profitability.get("best_margin_sources", [])
        worst_sources = profitability.get("worst_margin_sources", [])

        # Recommend focusing on high-margin sources
        for source in best_sources:
            source_name = source if isinstance(source, str) else source.get("source", "unknown")
            source_margin = (
                source.get("margin_pct", high_threshold)
                if isinstance(source, dict)
                else high_threshold
            )
            if source_margin >= high_threshold:
                rec = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "source": source_name,
                    "action": "scale_up",
                    "reason": f"High margin ({source_margin:.1f}%) — increase investment",
                    "priority": "high",
                    "margin_pct": source_margin,
                }
                recommendations.append(rec)

        # Recommend reducing investment in low-margin sources
        for source in worst_sources:
            source_name = source if isinstance(source, str) else source.get("source", "unknown")
            source_margin = (
                source.get("margin_pct", low_threshold)
                if isinstance(source, dict)
                else low_threshold
            )
            if source_margin <= low_threshold:
                rec = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "source": source_name,
                    "action": "deprioritize",
                    "reason": f"Low margin ({source_margin:.1f}%) — reduce investment or optimize",
                    "priority": "low",
                    "margin_pct": source_margin,
                }
                recommendations.append(rec)

        # Overall margin recommendation
        target_margin = config["target_profit_margin_pct"]
        if overall_margin < target_margin:
            rec = {
                "timestamp": datetime.utcnow().isoformat(),
                "source": "overall",
                "action": "improve_margin",
                "reason": (
                    f"Overall margin {overall_margin:.1f}% below target "
                    f"{target_margin:.1f}% — focus on cost reduction"
                ),
                "priority": "high",
                "margin_pct": overall_margin,
            }
            recommendations.append(rec)
        elif overall_margin >= target_margin * 1.5:
            rec = {
                "timestamp": datetime.utcnow().isoformat(),
                "source": "overall",
                "action": "reinvest",
                "reason": (
                    f"Overall margin {overall_margin:.1f}% exceeds target by 50%+ — "
                    f"reinvest surplus into growth"
                ),
                "priority": "medium",
                "margin_pct": overall_margin,
            }
            recommendations.append(rec)

        data["recommendations"].extend(recommendations)
        data["stats"]["recommendations_made"] += len(recommendations)

        self._log_event(
            data,
            "recommendations_generated",
            {
                "count": len(recommendations),
                "overall_margin": overall_margin,
            },
        )

        self._save(data)

        return SkillResult(
            success=True,
            message=f"{len(recommendations)} recommendations generated (margin: {overall_margin:.1f}%)",
            data={
                "recommendations": recommendations,
                "overall_margin_pct": overall_margin,
                "target_margin_pct": target_margin,
            },
        )

    # ------------------------------------------------------------------
    # Configure
    # ------------------------------------------------------------------

    def _configure(self, params: Dict) -> SkillResult:
        """Update auto-setting configuration."""
        data = self._load()
        updated = {}

        configurable = [
            "auto_generate_enabled",
            "target_daily_growth_rate",
            "target_profit_margin_pct",
            "adjustment_threshold_pct",
            "min_revenue_for_goal",
            "default_goal_deadline_hours",
            "high_margin_threshold",
            "low_margin_threshold",
            "max_active_goals",
        ]

        for key in configurable:
            if key in params:
                val = params[key]
                # Validate numeric values
                if key in (
                    "target_daily_growth_rate",
                    "target_profit_margin_pct",
                    "adjustment_threshold_pct",
                    "min_revenue_for_goal",
                    "default_goal_deadline_hours",
                    "high_margin_threshold",
                    "low_margin_threshold",
                ) and (not isinstance(val, (int, float)) or val < 0):
                    return SkillResult(
                        success=False,
                        message=f"{key} must be a non-negative number, got: {val}",
                    )
                if key == "max_active_goals" and (not isinstance(val, int) or val < 1):
                    return SkillResult(
                        success=False,
                        message=f"max_active_goals must be a positive integer, got: {val}",
                    )

                old_val = data["config"].get(key)
                data["config"][key] = val
                updated[key] = {"old": old_val, "new": val}

        if not updated:
            return SkillResult(
                success=True,
                message="No changes. Configurable: " + ", ".join(configurable),
                data={"current_config": data["config"]},
            )

        self._log_event(data, "config_updated", {"changes": updated})
        self._save(data)

        return SkillResult(
            success=True,
            message=f"Configuration updated: {', '.join(updated.keys())}",
            data={"updated": updated, "config": data["config"]},
        )

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def _status(self, params: Dict) -> SkillResult:
        """Show bridge status overview."""
        data = self._load()

        active_goals = [g for g in data["generated_goals"] if g.get("status") == "active"]
        recent_events = data["event_log"][-5:] if data["event_log"] else []
        recent_adjustments = data["adjustments"][-3:] if data["adjustments"] else []
        recent_recommendations = data["recommendations"][-3:] if data["recommendations"] else []

        return SkillResult(
            success=True,
            message=(
                f"Revenue goal bridge: {len(active_goals)} active goals, "
                f"{data['stats']['goals_generated']} total generated, "
                f"{data['stats']['goals_achieved']} achieved"
            ),
            data={
                "active_goals_count": len(active_goals),
                "active_goals": [
                    {
                        "goal_id": g["goal_id"],
                        "title": g["title"],
                        "target_revenue": g.get("target_revenue"),
                        "priority": g.get("priority"),
                        "milestones_achieved": g.get("milestones_achieved", 0),
                        "milestones_total": len(g.get("milestones", [])),
                    }
                    for g in active_goals
                ],
                "config": data["config"],
                "stats": data["stats"],
                "recent_events": recent_events,
                "recent_adjustments": recent_adjustments,
                "recent_recommendations": recent_recommendations,
            },
        )
