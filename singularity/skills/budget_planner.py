#!/usr/bin/env python3
"""
BudgetAwarePlannerSkill - Budget-constrained goal planning and execution.

This skill bridges the gap between resource awareness (ResourceWatcher, CostOptimizer)
and goal management (GoalManager) by ensuring agents only pursue goals they can afford.

Key capabilities:
1. Cost Estimation: Estimate goal costs from historical action data
2. Affordability Filtering: Only recommend goals within current budget
3. Value Maximization: Sequence goals to maximize ROI within budget constraints
4. ROI Tracking: Measure actual cost vs value for completed goals
5. Budget Allocation: Distribute budget across pillars proportionally
6. Constraint Propagation: Auto-block/deprioritize goals when budget shrinks

Without this, agents set ambitious goals but run out of budget mid-execution.
With this, agents plan within their means and maximize value per dollar.

Pillars served: Goal Setting (primary), Revenue (ROI tracking),
                Self-Improvement (learning which goals are cost-effective)
"""

import json
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any

from .base import Skill, SkillResult, SkillManifest, SkillAction

DATA_FILE = Path(__file__).parent.parent / "data" / "budget_planner.json"
MAX_PLANS = 200
MAX_HISTORY = 500

# Default budget allocation per pillar (fraction of total)
DEFAULT_PILLAR_ALLOCATION = {
    "self_improvement": 0.25,
    "revenue": 0.30,
    "replication": 0.20,
    "goal_setting": 0.15,
    "other": 0.10,
}

# Default cost estimates per action type when no historical data exists
DEFAULT_ACTION_COSTS = {
    "llm_call": 0.02,
    "api_request": 0.005,
    "file_operation": 0.001,
    "deployment": 0.10,
    "simple_compute": 0.002,
    "unknown": 0.01,
}


class BudgetAwarePlannerSkill(Skill):
    """
    Budget-constrained goal planning that integrates resource awareness
    with goal management for cost-effective autonomous operation.
    """

    def __init__(self, credentials: Dict[str, str] = None):
        super().__init__(credentials)
        self._ensure_data()

    def _ensure_data(self):
        DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
        if not DATA_FILE.exists():
            self._save(self._default_state())

    def _default_state(self) -> Dict:
        return {
            "budget": {
                "total": 100.0,
                "spent": 0.0,
                "pillar_allocation": DEFAULT_PILLAR_ALLOCATION.copy(),
                "pillar_spent": {p: 0.0 for p in DEFAULT_PILLAR_ALLOCATION},
            },
            "cost_estimates": {},  # goal_id -> estimated cost
            "goal_roi": [],  # completed goal ROI records
            "plans": [],  # budget-constrained execution plans
            "action_cost_history": {},  # action_type -> [costs] for estimation
            "config": {
                "min_roi_threshold": 0.5,  # minimum expected ROI to recommend
                "budget_safety_margin": 0.10,  # keep 10% reserve
                "max_concurrent_goals": 5,
                "auto_block_over_budget": True,
            },
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
        }

    def _load(self) -> Dict:
        try:
            with open(DATA_FILE, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return self._default_state()

    def _save(self, data: Dict):
        data["last_updated"] = datetime.now().isoformat()
        if len(data.get("goal_roi", [])) > MAX_HISTORY:
            data["goal_roi"] = data["goal_roi"][-MAX_HISTORY:]
        if len(data.get("plans", [])) > MAX_PLANS:
            data["plans"] = data["plans"][-MAX_PLANS:]
        with open(DATA_FILE, "w") as f:
            json.dump(data, f, indent=2, default=str)

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="budget_planner",
            name="Budget-Aware Planner",
            version="1.0.0",
            category="meta",
            description="Budget-constrained goal planning: estimate costs, filter by affordability, maximize ROI",
            required_credentials=[],
            actions=[
                SkillAction(
                    name="estimate_goal",
                    description="Estimate the cost to achieve a goal based on its milestones and historical data",
                    parameters={
                        "goal_id": {"type": "string", "required": False, "description": "Existing goal ID to estimate"},
                        "title": {"type": "string", "required": False, "description": "Goal title (for new goal estimation)"},
                        "milestones": {"type": "array", "required": False, "description": "List of milestone descriptions"},
                        "action_types": {"type": "array", "required": False, "description": "Expected action types (llm_call, api_request, etc.)"},
                        "expected_actions": {"type": "integer", "required": False, "description": "Estimated number of actions needed"},
                    },
                    estimated_cost=0,
                    estimated_duration_seconds=1,
                    success_probability=0.95,
                ),
                SkillAction(
                    name="affordable_goals",
                    description="List goals filtered by what the agent can currently afford",
                    parameters={
                        "pillar": {"type": "string", "required": False, "description": "Filter by pillar"},
                        "include_estimates": {"type": "boolean", "required": False, "description": "Include cost estimates (default true)"},
                    },
                    estimated_cost=0,
                    estimated_duration_seconds=1,
                    success_probability=0.95,
                ),
                SkillAction(
                    name="plan_budget",
                    description="Create an optimal budget-constrained execution plan for active goals",
                    parameters={
                        "budget_override": {"type": "number", "required": False, "description": "Override available budget for planning"},
                        "pillar_focus": {"type": "string", "required": False, "description": "Give extra budget to this pillar"},
                    },
                    estimated_cost=0,
                    estimated_duration_seconds=2,
                    success_probability=0.90,
                ),
                SkillAction(
                    name="record_cost",
                    description="Record actual cost for a goal (for ROI tracking)",
                    parameters={
                        "goal_id": {"type": "string", "required": True, "description": "Goal ID"},
                        "cost": {"type": "number", "required": True, "description": "Cost in USD"},
                        "revenue": {"type": "number", "required": False, "description": "Revenue generated (default 0)"},
                        "pillar": {"type": "string", "required": False, "description": "Pillar for budget tracking"},
                    },
                    estimated_cost=0,
                    estimated_duration_seconds=0,
                    success_probability=0.99,
                ),
                SkillAction(
                    name="set_budget",
                    description="Set total budget and pillar allocations",
                    parameters={
                        "total": {"type": "number", "required": True, "description": "Total budget in USD"},
                        "pillar_allocation": {"type": "object", "required": False, "description": "Per-pillar fractions (must sum to 1.0)"},
                    },
                    estimated_cost=0,
                    estimated_duration_seconds=0,
                    success_probability=0.99,
                ),
                SkillAction(
                    name="roi_report",
                    description="Analyze ROI across completed goals by pillar and priority",
                    parameters={
                        "period_hours": {"type": "number", "required": False, "description": "Analysis period in hours"},
                    },
                    estimated_cost=0,
                    estimated_duration_seconds=1,
                    success_probability=0.95,
                ),
                SkillAction(
                    name="budget_status",
                    description="Get current budget status with per-pillar breakdown",
                    parameters={},
                    estimated_cost=0,
                    estimated_duration_seconds=0,
                    success_probability=0.99,
                ),
                SkillAction(
                    name="learn_costs",
                    description="Record historical action costs to improve future estimates",
                    parameters={
                        "action_type": {"type": "string", "required": True, "description": "Action type (llm_call, api_request, etc.)"},
                        "cost": {"type": "number", "required": True, "description": "Actual cost observed"},
                    },
                    estimated_cost=0,
                    estimated_duration_seconds=0,
                    success_probability=0.99,
                ),
            ],
        )

    async def execute(self, action: str, params: Dict[str, Any] = None) -> SkillResult:
        params = params or {}
        handlers = {
            "estimate_goal": self._estimate_goal,
            "affordable_goals": self._affordable_goals,
            "plan_budget": self._plan_budget,
            "record_cost": self._record_cost,
            "set_budget": self._set_budget,
            "roi_report": self._roi_report,
            "budget_status": self._budget_status,
            "learn_costs": self._learn_costs,
        }
        handler = handlers.get(action)
        if not handler:
            return SkillResult(success=False, message=f"Unknown action: {action}")
        return handler(params)

    # ── Actions ───────────────────────────────────────────────────

    def _estimate_goal(self, params: Dict) -> SkillResult:
        """Estimate cost to achieve a goal."""
        data = self._load()
        goal_id = params.get("goal_id", "")
        title = params.get("title", goal_id or "unnamed")
        milestones = params.get("milestones", [])
        action_types = params.get("action_types", [])
        expected_actions = params.get("expected_actions")

        # Determine number of actions needed
        if expected_actions:
            num_actions = int(expected_actions)
        elif milestones:
            # Estimate ~3 actions per milestone (typical for a development task)
            num_actions = len(milestones) * 3
        else:
            # Default estimate for goals without milestones
            num_actions = 10

        # Estimate cost per action from historical data
        cost_history = data.get("action_cost_history", {})
        if action_types:
            per_action_costs = []
            for at in action_types:
                hist = cost_history.get(at, [])
                if hist:
                    per_action_costs.append(sum(hist) / len(hist))
                else:
                    per_action_costs.append(DEFAULT_ACTION_COSTS.get(at, DEFAULT_ACTION_COSTS["unknown"]))
            avg_cost_per_action = sum(per_action_costs) / len(per_action_costs) if per_action_costs else DEFAULT_ACTION_COSTS["unknown"]
        else:
            # Use overall average from all history
            all_costs = []
            for costs in cost_history.values():
                all_costs.extend(costs)
            if all_costs:
                avg_cost_per_action = sum(all_costs) / len(all_costs)
            else:
                avg_cost_per_action = DEFAULT_ACTION_COSTS["unknown"]

        estimated_cost = num_actions * avg_cost_per_action

        # Add confidence based on data quality
        total_data_points = sum(len(v) for v in cost_history.values())
        if total_data_points >= 50:
            confidence = "high"
            margin = 1.2  # 20% margin
        elif total_data_points >= 10:
            confidence = "medium"
            margin = 1.5  # 50% margin
        else:
            confidence = "low"
            margin = 2.0  # 100% margin (double)

        estimate = {
            "goal_id": goal_id,
            "title": title,
            "estimated_cost": round(estimated_cost, 4),
            "estimated_cost_high": round(estimated_cost * margin, 4),
            "num_actions": num_actions,
            "avg_cost_per_action": round(avg_cost_per_action, 6),
            "confidence": confidence,
            "data_points_used": total_data_points,
        }

        # Store estimate
        if goal_id:
            data["cost_estimates"][goal_id] = estimate
            self._save(data)

        # Check affordability
        budget = data["budget"]
        remaining = budget["total"] - budget["spent"]
        safety = budget["total"] * data["config"]["budget_safety_margin"]
        available = remaining - safety

        estimate["affordable"] = estimated_cost <= available
        estimate["budget_remaining"] = round(available, 4)

        msg = (
            f"Goal '{title}': estimated ${estimated_cost:.4f} "
            f"(${estimated_cost * margin:.4f} worst case, {confidence} confidence) - "
            f"{'AFFORDABLE' if estimate['affordable'] else 'OVER BUDGET'}"
        )

        return SkillResult(success=True, message=msg, data=estimate)

    def _affordable_goals(self, params: Dict) -> SkillResult:
        """List goals filtered by affordability."""
        data = self._load()
        pillar_filter = params.get("pillar", "").strip().lower()
        include_estimates = params.get("include_estimates", True)

        budget = data["budget"]
        remaining = budget["total"] - budget["spent"]
        safety = budget["total"] * data["config"]["budget_safety_margin"]
        available = remaining - safety

        # Load goals from GoalManager data file
        goals_file = Path(__file__).parent.parent / "data" / "goals.json"
        try:
            with open(goals_file, "r") as f:
                goals_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return SkillResult(
                success=True,
                message="No goals found. Create goals with goals:create first.",
                data={"affordable": [], "over_budget": [], "available_budget": round(available, 4)},
            )

        active_goals = [g for g in goals_data.get("goals", []) if g.get("status") == "active"]
        if pillar_filter:
            active_goals = [g for g in active_goals if g.get("pillar") == pillar_filter]

        affordable = []
        over_budget = []

        for goal in active_goals:
            gid = goal["id"]

            # Get or compute estimate
            if gid in data["cost_estimates"]:
                est = data["cost_estimates"][gid]
                estimated_cost = est["estimated_cost"]
            elif include_estimates:
                milestones = goal.get("milestones", [])
                num_actions = max(len(milestones) * 3, 5)
                all_costs = []
                for costs in data.get("action_cost_history", {}).values():
                    all_costs.extend(costs)
                avg = sum(all_costs) / len(all_costs) if all_costs else DEFAULT_ACTION_COSTS["unknown"]
                estimated_cost = num_actions * avg
            else:
                estimated_cost = 0

            # Check pillar budget
            pillar = goal.get("pillar", "other")
            pillar_alloc = budget["pillar_allocation"].get(pillar, 0.1)
            pillar_budget = budget["total"] * pillar_alloc
            pillar_spent = budget["pillar_spent"].get(pillar, 0.0)
            pillar_remaining = pillar_budget - pillar_spent

            goal_info = {
                "goal_id": gid,
                "title": goal["title"],
                "pillar": pillar,
                "priority": goal.get("priority", "medium"),
                "estimated_cost": round(estimated_cost, 4),
                "pillar_remaining": round(pillar_remaining, 4),
                "milestones_done": sum(1 for m in goal.get("milestones", []) if m.get("completed")),
                "milestones_total": len(goal.get("milestones", [])),
            }

            if estimated_cost <= available and estimated_cost <= pillar_remaining:
                goal_info["status"] = "affordable"
                affordable.append(goal_info)
            else:
                goal_info["status"] = "over_budget"
                goal_info["shortfall"] = round(max(estimated_cost - available, estimated_cost - pillar_remaining, 0), 4)
                over_budget.append(goal_info)

        # Sort affordable by priority
        priority_scores = {"critical": 4, "high": 3, "medium": 2, "low": 1}
        affordable.sort(key=lambda g: priority_scores.get(g["priority"], 0), reverse=True)

        return SkillResult(
            success=True,
            message=f"{len(affordable)} affordable goals, {len(over_budget)} over budget (${available:.2f} available)",
            data={
                "affordable": affordable,
                "over_budget": over_budget,
                "total_available": round(available, 4),
                "pillar_budgets": {
                    p: round(budget["total"] * budget["pillar_allocation"].get(p, 0.1) - budget["pillar_spent"].get(p, 0.0), 4)
                    for p in DEFAULT_PILLAR_ALLOCATION
                },
            },
        )

    def _plan_budget(self, params: Dict) -> SkillResult:
        """Create a budget-constrained execution plan to maximize value."""
        data = self._load()
        budget_override = params.get("budget_override")
        pillar_focus = params.get("pillar_focus", "").strip().lower()

        budget = data["budget"]
        total_available = float(budget_override) if budget_override else (budget["total"] - budget["spent"])
        safety = budget["total"] * data["config"]["budget_safety_margin"]
        available = total_available - safety

        if available <= 0:
            return SkillResult(
                success=True,
                message="No budget available for planning. Increase budget or reduce spending.",
                data={"plan": [], "budget_available": 0},
            )

        # Adjust pillar allocations if focus specified
        allocations = budget["pillar_allocation"].copy()
        if pillar_focus and pillar_focus in allocations:
            boost = 0.15  # Give focused pillar 15% more
            allocations[pillar_focus] = min(allocations[pillar_focus] + boost, 0.60)
            others = [p for p in allocations if p != pillar_focus]
            reduction = boost / len(others) if others else 0
            for p in others:
                allocations[p] = max(allocations[p] - reduction, 0.05)

        # Load goals
        goals_file = Path(__file__).parent.parent / "data" / "goals.json"
        try:
            with open(goals_file, "r") as f:
                goals_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return SkillResult(
                success=True,
                message="No goals found for planning.",
                data={"plan": [], "budget_available": round(available, 4)},
            )

        active_goals = [g for g in goals_data.get("goals", []) if g.get("status") == "active"]
        if not active_goals:
            return SkillResult(
                success=True,
                message="No active goals to plan. Create goals with goals:create first.",
                data={"plan": [], "budget_available": round(available, 4)},
            )

        # Score each goal: value = priority_score * pillar_weight / estimated_cost (bang for buck)
        priority_scores = {"critical": 4, "high": 3, "medium": 2, "low": 1}
        scored_goals = []

        for goal in active_goals:
            gid = goal["id"]
            pillar = goal.get("pillar", "other")
            priority = goal.get("priority", "medium")

            # Estimate cost
            if gid in data["cost_estimates"]:
                est_cost = data["cost_estimates"][gid]["estimated_cost"]
            else:
                milestones = goal.get("milestones", [])
                num_actions = max(len(milestones) * 3, 5)
                all_costs = []
                for costs in data.get("action_cost_history", {}).values():
                    all_costs.extend(costs)
                avg = sum(all_costs) / len(all_costs) if all_costs else DEFAULT_ACTION_COSTS["unknown"]
                est_cost = num_actions * avg

            if est_cost <= 0:
                est_cost = 0.01

            # Value score: higher priority + in-progress bonus + deadline urgency
            base_score = priority_scores.get(priority, 2)

            # In-progress bonus (momentum)
            milestones = goal.get("milestones", [])
            done = sum(1 for m in milestones if m.get("completed"))
            if milestones and 0 < done < len(milestones):
                base_score += 1

            # Deadline urgency
            if goal.get("deadline"):
                try:
                    dl = datetime.fromisoformat(goal["deadline"])
                    hours_left = (dl - datetime.now()).total_seconds() / 3600
                    if hours_left < 0:
                        base_score += 3  # Overdue
                    elif hours_left < 24:
                        base_score += 2  # Due soon
                    elif hours_left < 72:
                        base_score += 1
                except (ValueError, TypeError):
                    pass

            # ROI score = value / cost (bang for buck)
            value_per_dollar = base_score / est_cost

            scored_goals.append({
                "goal_id": gid,
                "title": goal["title"],
                "pillar": pillar,
                "priority": priority,
                "priority_score": base_score,
                "estimated_cost": round(est_cost, 4),
                "value_per_dollar": round(value_per_dollar, 4),
                "milestones_done": done,
                "milestones_total": len(milestones),
                "deadline": goal.get("deadline"),
            })

        # Greedy knapsack: pick highest value-per-dollar goals within budget
        scored_goals.sort(key=lambda g: g["value_per_dollar"], reverse=True)

        plan = []
        remaining_budget = available
        pillar_remaining = {p: available * allocations.get(p, 0.1) for p in DEFAULT_PILLAR_ALLOCATION}
        total_planned_cost = 0
        deferred = []

        for goal in scored_goals:
            pillar = goal["pillar"]
            cost = goal["estimated_cost"]

            # Check both total budget and pillar allocation
            pr = pillar_remaining.get(pillar, 0)
            if cost <= remaining_budget and cost <= pr:
                plan.append({
                    **goal,
                    "status": "planned",
                    "budget_after": round(remaining_budget - cost, 4),
                })
                remaining_budget -= cost
                pillar_remaining[pillar] = pr - cost
                total_planned_cost += cost
            else:
                deferred.append({
                    **goal,
                    "status": "deferred",
                    "reason": "over_budget" if cost > remaining_budget else "pillar_budget_exceeded",
                    "shortfall": round(max(cost - remaining_budget, cost - pr, 0), 4),
                })

        # Limit to max concurrent
        max_concurrent = data["config"]["max_concurrent_goals"]
        if len(plan) > max_concurrent:
            deferred.extend([{**g, "status": "deferred", "reason": "concurrency_limit"} for g in plan[max_concurrent:]])
            plan = plan[:max_concurrent]
            total_planned_cost = sum(g["estimated_cost"] for g in plan)

        # Save plan
        plan_record = {
            "id": f"plan_{uuid.uuid4().hex[:8]}",
            "created_at": datetime.now().isoformat(),
            "budget_available": round(available, 4),
            "planned_cost": round(total_planned_cost, 4),
            "goals_planned": len(plan),
            "goals_deferred": len(deferred),
            "pillar_focus": pillar_focus or None,
        }
        data["plans"].append(plan_record)
        self._save(data)

        return SkillResult(
            success=True,
            message=f"Plan: {len(plan)} goals (${total_planned_cost:.4f}) within ${available:.2f} budget, {len(deferred)} deferred",
            data={
                "plan": plan,
                "deferred": deferred,
                "budget_available": round(available, 4),
                "budget_used": round(total_planned_cost, 4),
                "budget_remaining": round(available - total_planned_cost, 4),
                "plan_id": plan_record["id"],
                "pillar_remaining": {k: round(v, 4) for k, v in pillar_remaining.items()},
            },
        )

    def _record_cost(self, params: Dict) -> SkillResult:
        """Record actual cost for a goal (for ROI tracking)."""
        goal_id = params.get("goal_id", "").strip()
        cost = float(params.get("cost", 0))
        revenue = float(params.get("revenue", 0))
        pillar = params.get("pillar", "other").strip().lower()

        if not goal_id:
            return SkillResult(success=False, message="goal_id is required")

        data = self._load()
        budget = data["budget"]

        # Update spending
        budget["spent"] += cost
        pillar_spent = budget["pillar_spent"]
        pillar_spent[pillar] = pillar_spent.get(pillar, 0.0) + cost

        # Record ROI
        roi_record = {
            "goal_id": goal_id,
            "cost": cost,
            "revenue": revenue,
            "profit": revenue - cost,
            "roi": round((revenue - cost) / cost, 4) if cost > 0 else 0,
            "pillar": pillar,
            "timestamp": datetime.now().isoformat(),
        }
        data["goal_roi"].append(roi_record)

        # Compare to estimate
        estimate = data["cost_estimates"].get(goal_id)
        accuracy = None
        if estimate:
            est_cost = estimate.get("estimated_cost", 0)
            if est_cost > 0:
                accuracy = round(cost / est_cost, 4)
                roi_record["estimation_accuracy"] = accuracy

        self._save(data)

        msg = f"Recorded: goal {goal_id} cost=${cost:.4f} revenue=${revenue:.4f} ROI={roi_record['roi']:.2f}"
        if accuracy is not None:
            msg += f" (estimate accuracy: {accuracy:.0%})"

        return SkillResult(success=True, message=msg, data=roi_record)

    def _set_budget(self, params: Dict) -> SkillResult:
        """Set total budget and pillar allocations."""
        total = float(params.get("total", 0))
        pillar_allocation = params.get("pillar_allocation")

        if total <= 0:
            return SkillResult(success=False, message="Budget total must be positive")

        data = self._load()
        data["budget"]["total"] = total

        if pillar_allocation and isinstance(pillar_allocation, dict):
            # Validate allocations sum to ~1.0
            alloc_sum = sum(float(v) for v in pillar_allocation.values())
            if abs(alloc_sum - 1.0) > 0.05:
                return SkillResult(
                    success=False,
                    message=f"Pillar allocations must sum to ~1.0 (got {alloc_sum:.2f})",
                )
            data["budget"]["pillar_allocation"] = {
                k: float(v) for k, v in pillar_allocation.items()
            }

        self._save(data)

        budget = data["budget"]
        alloc_summary = {
            p: f"${total * budget['pillar_allocation'].get(p, 0):.2f}"
            for p in DEFAULT_PILLAR_ALLOCATION
        }

        return SkillResult(
            success=True,
            message=f"Budget set to ${total:.2f} with allocations: {alloc_summary}",
            data={
                "total": total,
                "pillar_allocation": budget["pillar_allocation"],
                "pillar_budgets": {p: round(total * v, 4) for p, v in budget["pillar_allocation"].items()},
            },
        )

    def _roi_report(self, params: Dict) -> SkillResult:
        """Analyze ROI across completed goals."""
        data = self._load()
        records = data.get("goal_roi", [])
        period_hours = params.get("period_hours")

        if period_hours:
            cutoff = (datetime.now() - timedelta(hours=float(period_hours))).isoformat()
            records = [r for r in records if r.get("timestamp", "") >= cutoff]

        if not records:
            return SkillResult(
                success=True,
                message="No ROI data yet. Record costs with budget_planner:record_cost.",
                data={"summary": {}, "by_pillar": {}, "records": 0},
            )

        # Overall summary
        total_cost = sum(r["cost"] for r in records)
        total_revenue = sum(r["revenue"] for r in records)
        total_profit = total_revenue - total_cost
        avg_roi = sum(r["roi"] for r in records) / len(records)

        # By pillar
        pillar_stats = {}
        for r in records:
            p = r.get("pillar", "other")
            if p not in pillar_stats:
                pillar_stats[p] = {"cost": 0, "revenue": 0, "count": 0, "roi_sum": 0}
            pillar_stats[p]["cost"] += r["cost"]
            pillar_stats[p]["revenue"] += r["revenue"]
            pillar_stats[p]["count"] += 1
            pillar_stats[p]["roi_sum"] += r["roi"]

        for p, s in pillar_stats.items():
            s["avg_roi"] = round(s["roi_sum"] / s["count"], 4) if s["count"] > 0 else 0
            s["profit"] = round(s["revenue"] - s["cost"], 4)
            s["cost"] = round(s["cost"], 4)
            s["revenue"] = round(s["revenue"], 4)
            del s["roi_sum"]

        # Estimation accuracy
        records_with_accuracy = [r for r in records if "estimation_accuracy" in r]
        avg_accuracy = None
        if records_with_accuracy:
            avg_accuracy = round(
                sum(r["estimation_accuracy"] for r in records_with_accuracy) / len(records_with_accuracy),
                4,
            )

        # Best and worst ROI goals
        sorted_by_roi = sorted(records, key=lambda r: r["roi"], reverse=True)
        best = sorted_by_roi[:3] if len(sorted_by_roi) >= 3 else sorted_by_roi
        worst = sorted_by_roi[-3:] if len(sorted_by_roi) >= 3 else []

        summary = {
            "total_cost": round(total_cost, 4),
            "total_revenue": round(total_revenue, 4),
            "total_profit": round(total_profit, 4),
            "avg_roi": round(avg_roi, 4),
            "goals_tracked": len(records),
            "estimation_accuracy": avg_accuracy,
        }

        msg = (
            f"ROI Report: {len(records)} goals, ${total_cost:.4f} cost, "
            f"${total_revenue:.4f} revenue, avg ROI {avg_roi:.2f}"
        )

        return SkillResult(
            success=True,
            message=msg,
            data={
                "summary": summary,
                "by_pillar": pillar_stats,
                "best_roi": best,
                "worst_roi": worst,
            },
        )

    def _budget_status(self, params: Dict) -> SkillResult:
        """Get current budget status with per-pillar breakdown."""
        data = self._load()
        budget = data["budget"]

        total = budget["total"]
        spent = budget["spent"]
        remaining = total - spent
        safety = total * data["config"]["budget_safety_margin"]
        available = remaining - safety
        pct_used = (spent / total * 100) if total > 0 else 0

        # Per-pillar breakdown
        pillar_status = {}
        for pillar, alloc_frac in budget["pillar_allocation"].items():
            pillar_total = total * alloc_frac
            pillar_spent = budget["pillar_spent"].get(pillar, 0.0)
            pillar_remaining = pillar_total - pillar_spent
            pillar_status[pillar] = {
                "allocated": round(pillar_total, 4),
                "spent": round(pillar_spent, 4),
                "remaining": round(pillar_remaining, 4),
                "pct_used": round(pillar_spent / pillar_total * 100, 1) if pillar_total > 0 else 0,
            }

        # Health assessment
        if pct_used >= 95:
            health = "critical"
        elif pct_used >= 80:
            health = "warning"
        elif pct_used >= 60:
            health = "caution"
        else:
            health = "healthy"

        # Count stored estimates
        estimates_count = len(data.get("cost_estimates", {}))
        roi_count = len(data.get("goal_roi", []))

        return SkillResult(
            success=True,
            message=f"Budget {health}: ${available:.2f} available of ${total:.2f} ({pct_used:.1f}% used)",
            data={
                "total": round(total, 4),
                "spent": round(spent, 4),
                "remaining": round(remaining, 4),
                "available_after_safety": round(available, 4),
                "safety_reserve": round(safety, 4),
                "pct_used": round(pct_used, 1),
                "health": health,
                "pillar_status": pillar_status,
                "estimates_stored": estimates_count,
                "roi_records": roi_count,
                "config": data["config"],
            },
        )

    def _learn_costs(self, params: Dict) -> SkillResult:
        """Record historical action cost to improve future estimates."""
        action_type = params.get("action_type", "").strip()
        cost = float(params.get("cost", 0))

        if not action_type:
            return SkillResult(success=False, message="action_type is required")

        data = self._load()
        history = data.setdefault("action_cost_history", {})
        costs_list = history.setdefault(action_type, [])
        costs_list.append(cost)

        # Keep last 100 observations per action type
        if len(costs_list) > 100:
            history[action_type] = costs_list[-100:]

        avg = sum(history[action_type]) / len(history[action_type])

        self._save(data)

        return SkillResult(
            success=True,
            message=f"Learned: {action_type} costs avg ${avg:.6f} ({len(history[action_type])} observations)",
            data={
                "action_type": action_type,
                "cost_recorded": cost,
                "running_average": round(avg, 6),
                "observations": len(history[action_type]),
            },
        )
