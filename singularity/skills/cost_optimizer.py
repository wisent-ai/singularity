#!/usr/bin/env python3
"""
CostOptimizerSkill - Autonomous cost tracking, analysis, and optimization.

Critical missing piece for the Revenue pillar: the agent needs to understand
its own cost structure to become profitable. This skill provides:

1. Cost Tracking: Per-action, per-skill, per-session cost recording
2. Revenue Tracking: Links costs to revenue generated per service
3. Profitability Analysis: Cost-to-revenue ratios, margin calculations
4. Cost Hotspots: Identifies the most expensive operations
5. Budget Management: Sets limits, warns on overruns, enforces caps
6. Optimization Suggestions: Recommends cost-saving strategies
7. Trend Projection: Forecasts future costs based on historical data

Without this, the agent is flying blind on profitability. With it, the agent
can autonomously optimize for maximum value per dollar spent.

Pillars served: Revenue (primary), Self-Improvement (cost-aware decisions),
                Goal Setting (budget-constrained planning)
"""

import json
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from .base import Skill, SkillResult, SkillManifest, SkillAction


COST_FILE = Path(__file__).parent.parent / "data" / "cost_optimizer.json"
MAX_ENTRIES = 5000
MAX_SUGGESTIONS = 100


class CostOptimizerSkill(Skill):
    """
    Autonomous cost tracking, profitability analysis, and optimization.

    Enables the agent to:
    - Record costs and revenue per action/skill/session
    - Analyze cost hotspots and profitability by service
    - Set and enforce budget limits with warnings
    - Generate optimization suggestions based on cost patterns
    - Project future costs from historical trends
    - Make cost-aware decisions about which actions to take
    """

    def __init__(self, credentials: Dict[str, str] = None):
        super().__init__(credentials)
        self._ensure_data()

    def _ensure_data(self):
        COST_FILE.parent.mkdir(parents=True, exist_ok=True)
        if not COST_FILE.exists():
            self._save(self._default_state())

    def _default_state(self) -> Dict:
        return {
            "entries": [],           # Cost/revenue log entries
            "budgets": {},           # Budget limits by category
            "suggestions": [],       # Optimization suggestions
            "sessions": {},          # Per-session cost summaries
            "skill_totals": {},      # Running totals per skill
            "config": {
                "warn_threshold_pct": 80,    # Warn at 80% of budget
                "block_threshold_pct": 100,  # Block at 100% of budget
                "enforce_budgets": False,     # Start non-enforcing
                "retention_days": 90,         # Keep entries for 90 days
            },
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
        }

    def _load(self) -> Dict:
        try:
            with open(COST_FILE, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return self._default_state()

    def _save(self, data: Dict):
        data["last_updated"] = datetime.now().isoformat()
        # Cap entries
        if len(data.get("entries", [])) > MAX_ENTRIES:
            data["entries"] = data["entries"][-MAX_ENTRIES:]
        if len(data.get("suggestions", [])) > MAX_SUGGESTIONS:
            data["suggestions"] = data["suggestions"][-MAX_SUGGESTIONS:]
        with open(COST_FILE, "w") as f:
            json.dump(data, f, indent=2, default=str)

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="cost_optimizer",
            name="Cost Optimizer",
            version="1.0.0",
            category="meta",
            description="Autonomous cost tracking, profitability analysis, and optimization",
            actions=[
                SkillAction(
                    name="record",
                    description="Record a cost and/or revenue entry for an action",
                    parameters={
                        "skill_id": {"type": "string", "required": True, "description": "Skill that incurred the cost"},
                        "action": {"type": "string", "required": True, "description": "Action name"},
                        "cost": {"type": "number", "required": False, "description": "Cost in USD (default 0)"},
                        "revenue": {"type": "number", "required": False, "description": "Revenue in USD (default 0)"},
                        "tokens_used": {"type": "number", "required": False, "description": "LLM tokens consumed"},
                        "duration_ms": {"type": "number", "required": False, "description": "Execution time in ms"},
                        "session_id": {"type": "string", "required": False, "description": "Session identifier"},
                        "metadata": {"type": "object", "required": False, "description": "Extra context"},
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="analyze",
                    description="Get full cost/revenue analysis with hotspots and profitability",
                    parameters={
                        "period_hours": {"type": "number", "required": False, "description": "Analysis period in hours (default: all time)"},
                        "group_by": {"type": "string", "required": False, "description": "Group by: skill, action, session (default: skill)"},
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="set_budget",
                    description="Set a budget limit for a category (skill, total, session)",
                    parameters={
                        "category": {"type": "string", "required": True, "description": "Budget category (skill name, 'total', or 'session')"},
                        "limit_usd": {"type": "number", "required": True, "description": "Maximum spend in USD"},
                        "period": {"type": "string", "required": False, "description": "Period: daily, weekly, monthly, total (default: monthly)"},
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="check_budget",
                    description="Check if an action is within budget before executing",
                    parameters={
                        "skill_id": {"type": "string", "required": True, "description": "Skill to check"},
                        "estimated_cost": {"type": "number", "required": True, "description": "Estimated cost of action"},
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="optimize",
                    description="Generate optimization suggestions based on cost patterns",
                    parameters={},
                    estimated_cost=0,
                ),
                SkillAction(
                    name="project",
                    description="Project future costs based on historical trends",
                    parameters={
                        "days_ahead": {"type": "number", "required": False, "description": "Days to project (default: 30)"},
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="summary",
                    description="Quick cost/revenue summary for the current period",
                    parameters={},
                    estimated_cost=0,
                ),
            ],
            required_credentials=[],
        )

    async def execute(self, action: str, params: Dict[str, Any]) -> SkillResult:
        """Execute a cost optimizer action."""
        handlers = {
            "record": self._record,
            "analyze": self._analyze,
            "set_budget": self._set_budget,
            "check_budget": self._check_budget,
            "optimize": self._optimize,
            "project": self._project,
            "summary": self._summary,
        }
        handler = handlers.get(action)
        if not handler:
            return SkillResult(success=False, message=f"Unknown action: {action}")
        return handler(params)

    def _record(self, params: Dict) -> SkillResult:
        """Record a cost/revenue entry."""
        skill_id = params.get("skill_id", "unknown")
        action = params.get("action", "unknown")
        cost = float(params.get("cost", 0))
        revenue = float(params.get("revenue", 0))
        tokens_used = int(params.get("tokens_used", 0))
        duration_ms = float(params.get("duration_ms", 0))
        session_id = params.get("session_id", "default")
        metadata = params.get("metadata", {})

        data = self._load()

        entry = {
            "id": f"cost_{uuid.uuid4().hex[:8]}",
            "skill_id": skill_id,
            "action": action,
            "cost": cost,
            "revenue": revenue,
            "profit": revenue - cost,
            "tokens_used": tokens_used,
            "duration_ms": duration_ms,
            "session_id": session_id,
            "metadata": metadata,
            "timestamp": datetime.now().isoformat(),
        }

        data["entries"].append(entry)

        # Update skill totals
        if skill_id not in data["skill_totals"]:
            data["skill_totals"][skill_id] = {
                "total_cost": 0, "total_revenue": 0,
                "total_tokens": 0, "action_count": 0,
            }
        totals = data["skill_totals"][skill_id]
        totals["total_cost"] += cost
        totals["total_revenue"] += revenue
        totals["total_tokens"] += tokens_used
        totals["action_count"] += 1

        # Update session tracking
        if session_id not in data["sessions"]:
            data["sessions"][session_id] = {
                "total_cost": 0, "total_revenue": 0,
                "started_at": datetime.now().isoformat(),
            }
        data["sessions"][session_id]["total_cost"] += cost
        data["sessions"][session_id]["total_revenue"] += revenue

        self._save(data)

        return SkillResult(
            success=True,
            message=f"Recorded: {skill_id}/{action} cost=${cost:.4f} revenue=${revenue:.4f} profit=${revenue - cost:.4f}",
            data=entry,
        )

    def _analyze(self, params: Dict) -> SkillResult:
        """Full cost/revenue analysis with hotspots and profitability."""
        data = self._load()
        entries = data.get("entries", [])
        period_hours = params.get("period_hours")
        group_by = params.get("group_by", "skill")

        # Filter by period
        if period_hours:
            cutoff = (datetime.now() - timedelta(hours=float(period_hours))).isoformat()
            entries = [e for e in entries if e.get("timestamp", "") >= cutoff]

        if not entries:
            return SkillResult(
                success=True,
                message="No cost entries found for the specified period.",
                data={"entries_count": 0, "groups": {}, "hotspots": [], "profitability": {}},
            )

        # Group entries
        groups = {}
        for entry in entries:
            if group_by == "skill":
                key = entry.get("skill_id", "unknown")
            elif group_by == "action":
                key = f"{entry.get('skill_id', '?')}/{entry.get('action', '?')}"
            elif group_by == "session":
                key = entry.get("session_id", "default")
            else:
                key = entry.get("skill_id", "unknown")

            if key not in groups:
                groups[key] = {
                    "total_cost": 0, "total_revenue": 0, "total_profit": 0,
                    "count": 0, "total_tokens": 0, "total_duration_ms": 0,
                }
            g = groups[key]
            g["total_cost"] += entry.get("cost", 0)
            g["total_revenue"] += entry.get("revenue", 0)
            g["total_profit"] += entry.get("profit", 0)
            g["count"] += 1
            g["total_tokens"] += entry.get("tokens_used", 0)
            g["total_duration_ms"] += entry.get("duration_ms", 0)

        # Calculate per-unit metrics
        for key, g in groups.items():
            g["avg_cost"] = g["total_cost"] / g["count"] if g["count"] > 0 else 0
            g["avg_revenue"] = g["total_revenue"] / g["count"] if g["count"] > 0 else 0
            g["margin_pct"] = (
                (g["total_profit"] / g["total_revenue"] * 100)
                if g["total_revenue"] > 0 else (-100 if g["total_cost"] > 0 else 0)
            )
            g["cost_per_token"] = (
                g["total_cost"] / g["total_tokens"]
                if g["total_tokens"] > 0 else 0
            )

        # Identify hotspots (top 5 most expensive)
        hotspots = sorted(groups.items(), key=lambda x: x[1]["total_cost"], reverse=True)[:5]
        hotspot_list = [{"name": k, **v} for k, v in hotspots]

        # Overall profitability
        total_cost = sum(g["total_cost"] for g in groups.values())
        total_revenue = sum(g["total_revenue"] for g in groups.values())
        total_profit = total_revenue - total_cost

        profitability = {
            "total_cost": round(total_cost, 4),
            "total_revenue": round(total_revenue, 4),
            "total_profit": round(total_profit, 4),
            "margin_pct": round(total_profit / total_revenue * 100, 1) if total_revenue > 0 else -100,
            "entries_analyzed": len(entries),
            "unique_skills": len(set(e.get("skill_id") for e in entries)),
        }

        return SkillResult(
            success=True,
            message=f"Analysis: {len(entries)} entries, ${total_cost:.4f} cost, ${total_revenue:.4f} revenue, ${total_profit:.4f} profit ({profitability['margin_pct']}% margin)",
            data={
                "groups": {k: {kk: round(vv, 4) if isinstance(vv, float) else vv for kk, vv in v.items()} for k, v in groups.items()},
                "hotspots": hotspot_list,
                "profitability": profitability,
            },
        )

    def _set_budget(self, params: Dict) -> SkillResult:
        """Set a budget limit for a category."""
        category = params.get("category", "total")
        limit_usd = float(params.get("limit_usd", 0))
        period = params.get("period", "monthly")

        if limit_usd <= 0:
            return SkillResult(success=False, message="Budget limit must be positive")

        data = self._load()
        data["budgets"][category] = {
            "limit_usd": limit_usd,
            "period": period,
            "created_at": datetime.now().isoformat(),
        }
        self._save(data)

        return SkillResult(
            success=True,
            message=f"Budget set: {category} = ${limit_usd:.2f}/{period}",
            data={"category": category, "limit_usd": limit_usd, "period": period},
        )

    def _check_budget(self, params: Dict) -> SkillResult:
        """Check if an action is within budget."""
        skill_id = params.get("skill_id", "unknown")
        estimated_cost = float(params.get("estimated_cost", 0))

        data = self._load()
        budgets = data.get("budgets", {})
        entries = data.get("entries", [])
        config = data.get("config", {})
        warn_pct = config.get("warn_threshold_pct", 80)
        block_pct = config.get("block_threshold_pct", 100)
        enforce = config.get("enforce_budgets", False)

        warnings = []
        blocked = False

        # Check skill-specific budget
        if skill_id in budgets:
            budget = budgets[skill_id]
            spent = self._get_period_spend(entries, skill_id, budget["period"])
            after_spend = spent + estimated_cost
            pct = (after_spend / budget["limit_usd"]) * 100 if budget["limit_usd"] > 0 else 0

            if pct >= block_pct:
                blocked = True
                warnings.append(f"BLOCKED: {skill_id} would be at {pct:.0f}% of ${budget['limit_usd']:.2f} {budget['period']} budget")
            elif pct >= warn_pct:
                warnings.append(f"WARNING: {skill_id} would be at {pct:.0f}% of ${budget['limit_usd']:.2f} {budget['period']} budget")

        # Check total budget
        if "total" in budgets:
            budget = budgets["total"]
            spent = self._get_period_spend(entries, None, budget["period"])
            after_spend = spent + estimated_cost
            pct = (after_spend / budget["limit_usd"]) * 100 if budget["limit_usd"] > 0 else 0

            if pct >= block_pct:
                blocked = True
                warnings.append(f"BLOCKED: Total spend would be at {pct:.0f}% of ${budget['limit_usd']:.2f} {budget['period']} budget")
            elif pct >= warn_pct:
                warnings.append(f"WARNING: Total spend would be at {pct:.0f}% of ${budget['limit_usd']:.2f} {budget['period']} budget")

        allowed = not (blocked and enforce)

        return SkillResult(
            success=True,
            message=f"Budget check: {'ALLOWED' if allowed else 'BLOCKED'}" + (f" - {'; '.join(warnings)}" if warnings else ""),
            data={
                "allowed": allowed,
                "blocked": blocked,
                "enforced": enforce,
                "warnings": warnings,
                "estimated_cost": estimated_cost,
                "skill_id": skill_id,
            },
        )

    def _get_period_spend(self, entries: List, skill_id: Optional[str], period: str) -> float:
        """Calculate total spend for a period, optionally filtered by skill."""
        now = datetime.now()
        if period == "daily":
            cutoff = (now - timedelta(days=1)).isoformat()
        elif period == "weekly":
            cutoff = (now - timedelta(weeks=1)).isoformat()
        elif period == "monthly":
            cutoff = (now - timedelta(days=30)).isoformat()
        else:  # total - no cutoff
            cutoff = ""

        total = 0.0
        for entry in entries:
            if cutoff and entry.get("timestamp", "") < cutoff:
                continue
            if skill_id and entry.get("skill_id") != skill_id:
                continue
            total += entry.get("cost", 0)
        return total

    def _optimize(self, params: Dict) -> SkillResult:
        """Generate optimization suggestions based on cost patterns."""
        data = self._load()
        entries = data.get("entries", [])
        skill_totals = data.get("skill_totals", {})

        if not entries:
            return SkillResult(
                success=True,
                message="No cost data yet. Record some entries first.",
                data={"suggestions": []},
            )

        suggestions = []

        # 1. Identify skills with high cost but zero revenue (pure cost centers)
        for skill_id, totals in skill_totals.items():
            if totals["total_cost"] > 0 and totals["total_revenue"] == 0:
                if totals["action_count"] >= 3:
                    suggestions.append({
                        "type": "cost_center",
                        "priority": "high" if totals["total_cost"] > 1.0 else "medium",
                        "skill_id": skill_id,
                        "message": f"{skill_id} has spent ${totals['total_cost']:.4f} with zero revenue over {totals['action_count']} actions. Consider reducing usage or finding ways to monetize.",
                        "total_cost": totals["total_cost"],
                    })

        # 2. Identify skills with poor margin (revenue < 2x cost)
        for skill_id, totals in skill_totals.items():
            if totals["total_revenue"] > 0 and totals["total_cost"] > 0:
                margin = (totals["total_revenue"] - totals["total_cost"]) / totals["total_revenue"]
                if margin < 0.5:  # Less than 50% margin
                    suggestions.append({
                        "type": "low_margin",
                        "priority": "high",
                        "skill_id": skill_id,
                        "message": f"{skill_id} has only {margin*100:.0f}% margin (${totals['total_revenue']:.4f} revenue vs ${totals['total_cost']:.4f} cost). Consider raising prices or reducing costs.",
                        "margin_pct": round(margin * 100, 1),
                    })

        # 3. Identify high-token-count actions (suggest caching/batching)
        action_tokens = {}
        for entry in entries:
            key = f"{entry.get('skill_id')}/{entry.get('action')}"
            if key not in action_tokens:
                action_tokens[key] = {"total_tokens": 0, "count": 0, "total_cost": 0}
            action_tokens[key]["total_tokens"] += entry.get("tokens_used", 0)
            action_tokens[key]["count"] += 1
            action_tokens[key]["total_cost"] += entry.get("cost", 0)

        for action_key, stats in action_tokens.items():
            avg_tokens = stats["total_tokens"] / stats["count"] if stats["count"] > 0 else 0
            if avg_tokens > 5000 and stats["count"] >= 3:
                suggestions.append({
                    "type": "high_tokens",
                    "priority": "medium",
                    "action": action_key,
                    "message": f"{action_key} uses avg {avg_tokens:.0f} tokens/call ({stats['count']} calls). Consider caching results, using shorter prompts, or batching requests.",
                    "avg_tokens": avg_tokens,
                })

        # 4. Identify slow/expensive actions (suggest optimization or alternatives)
        for action_key, stats in action_tokens.items():
            avg_cost = stats["total_cost"] / stats["count"] if stats["count"] > 0 else 0
            if avg_cost > 0.10 and stats["count"] >= 2:
                suggestions.append({
                    "type": "expensive_action",
                    "priority": "high",
                    "action": action_key,
                    "message": f"{action_key} costs avg ${avg_cost:.4f}/call. Consider using a cheaper model, reducing prompt size, or caching.",
                    "avg_cost": avg_cost,
                })

        # 5. Check for duplicate/redundant actions
        recent = entries[-100:] if len(entries) >= 100 else entries
        action_sequences = {}
        for i, entry in enumerate(recent):
            key = f"{entry.get('skill_id')}/{entry.get('action')}"
            if key not in action_sequences:
                action_sequences[key] = []
            action_sequences[key].append(i)

        for action_key, indices in action_sequences.items():
            if len(indices) >= 5:
                # Check for rapid repetition (within short time spans)
                consecutive = sum(1 for i in range(len(indices)-1) if indices[i+1] - indices[i] == 1)
                if consecutive >= 3:
                    suggestions.append({
                        "type": "repetitive",
                        "priority": "medium",
                        "action": action_key,
                        "message": f"{action_key} was called {len(indices)} times in recent history with {consecutive} consecutive calls. Consider batching or caching.",
                        "call_count": len(indices),
                    })

        # Sort by priority
        priority_order = {"high": 0, "medium": 1, "low": 2}
        suggestions.sort(key=lambda s: priority_order.get(s.get("priority", "low"), 2))

        # Store suggestions
        data["suggestions"] = suggestions
        self._save(data)

        return SkillResult(
            success=True,
            message=f"Generated {len(suggestions)} optimization suggestions ({sum(1 for s in suggestions if s['priority'] == 'high')} high priority)",
            data={"suggestions": suggestions, "count": len(suggestions)},
        )

    def _project(self, params: Dict) -> SkillResult:
        """Project future costs based on historical trends."""
        data = self._load()
        entries = data.get("entries", [])
        days_ahead = int(params.get("days_ahead", 30))

        if len(entries) < 3:
            return SkillResult(
                success=True,
                message="Need at least 3 cost entries to project trends.",
                data={"projected": None, "reason": "insufficient_data"},
            )

        # Group entries by day
        daily_costs = {}
        daily_revenue = {}
        for entry in entries:
            ts = entry.get("timestamp", "")
            if not ts:
                continue
            day = ts[:10]  # YYYY-MM-DD
            daily_costs[day] = daily_costs.get(day, 0) + entry.get("cost", 0)
            daily_revenue[day] = daily_revenue.get(day, 0) + entry.get("revenue", 0)

        if not daily_costs:
            return SkillResult(
                success=True,
                message="No dated entries found for projection.",
                data={"projected": None},
            )

        # Calculate daily averages
        days_with_data = len(daily_costs)
        total_cost = sum(daily_costs.values())
        total_revenue = sum(daily_revenue.values())
        avg_daily_cost = total_cost / days_with_data
        avg_daily_revenue = total_revenue / days_with_data

        # Simple linear projection
        projected_cost = avg_daily_cost * days_ahead
        projected_revenue = avg_daily_revenue * days_ahead
        projected_profit = projected_revenue - projected_cost

        # Calculate trend (is cost increasing or decreasing?)
        sorted_days = sorted(daily_costs.keys())
        if len(sorted_days) >= 2:
            first_half = sorted_days[:len(sorted_days)//2]
            second_half = sorted_days[len(sorted_days)//2:]
            first_avg = sum(daily_costs[d] for d in first_half) / len(first_half)
            second_avg = sum(daily_costs[d] for d in second_half) / len(second_half)
            cost_trend = "increasing" if second_avg > first_avg * 1.1 else (
                "decreasing" if second_avg < first_avg * 0.9 else "stable"
            )
        else:
            cost_trend = "unknown"

        projection = {
            "days_ahead": days_ahead,
            "days_of_data": days_with_data,
            "avg_daily_cost": round(avg_daily_cost, 4),
            "avg_daily_revenue": round(avg_daily_revenue, 4),
            "projected_cost": round(projected_cost, 4),
            "projected_revenue": round(projected_revenue, 4),
            "projected_profit": round(projected_profit, 4),
            "cost_trend": cost_trend,
            "break_even_daily_revenue": round(avg_daily_cost, 4),
        }

        profitable = projected_profit > 0
        msg = (
            f"{days_ahead}-day projection: ${projected_cost:.2f} cost, ${projected_revenue:.2f} revenue, "
            f"${projected_profit:.2f} {'profit' if profitable else 'loss'} (trend: {cost_trend})"
        )

        return SkillResult(success=True, message=msg, data=projection)

    def _summary(self, params: Dict) -> SkillResult:
        """Quick cost/revenue summary."""
        data = self._load()
        entries = data.get("entries", [])
        skill_totals = data.get("skill_totals", {})
        budgets = data.get("budgets", {})

        total_cost = sum(t["total_cost"] for t in skill_totals.values())
        total_revenue = sum(t["total_revenue"] for t in skill_totals.values())
        total_profit = total_revenue - total_cost
        total_actions = sum(t["action_count"] for t in skill_totals.values())
        total_tokens = sum(t["total_tokens"] for t in skill_totals.values())

        # Budget status
        budget_status = {}
        for cat, budget in budgets.items():
            spent = self._get_period_spend(entries, cat if cat != "total" else None, budget["period"])
            budget_status[cat] = {
                "limit": budget["limit_usd"],
                "spent": round(spent, 4),
                "remaining": round(budget["limit_usd"] - spent, 4),
                "pct_used": round(spent / budget["limit_usd"] * 100, 1) if budget["limit_usd"] > 0 else 0,
                "period": budget["period"],
            }

        # Top 3 most expensive skills
        top_costs = sorted(
            skill_totals.items(),
            key=lambda x: x[1]["total_cost"],
            reverse=True
        )[:3]

        # Top 3 most profitable skills
        top_profit = sorted(
            skill_totals.items(),
            key=lambda x: x[1]["total_revenue"] - x[1]["total_cost"],
            reverse=True
        )[:3]

        summary_data = {
            "total_cost": round(total_cost, 4),
            "total_revenue": round(total_revenue, 4),
            "total_profit": round(total_profit, 4),
            "total_actions": total_actions,
            "total_tokens": total_tokens,
            "margin_pct": round(total_profit / total_revenue * 100, 1) if total_revenue > 0 else 0,
            "cost_per_action": round(total_cost / total_actions, 4) if total_actions > 0 else 0,
            "unique_skills": len(skill_totals),
            "top_costs": [{"skill": k, "cost": round(v["total_cost"], 4)} for k, v in top_costs],
            "top_profit": [{"skill": k, "profit": round(v["total_revenue"] - v["total_cost"], 4)} for k, v in top_profit],
            "budget_status": budget_status,
            "entry_count": len(entries),
        }

        msg_parts = [
            f"Cost: ${total_cost:.4f}",
            f"Revenue: ${total_revenue:.4f}",
            f"Profit: ${total_profit:.4f}",
            f"Actions: {total_actions}",
            f"Skills: {len(skill_totals)}",
        ]
        if total_revenue > 0:
            msg_parts.append(f"Margin: {summary_data['margin_pct']}%")

        return SkillResult(
            success=True,
            message=" | ".join(msg_parts),
            data=summary_data,
        )
