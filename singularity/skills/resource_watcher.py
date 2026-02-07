#!/usr/bin/env python3
"""
ResourceWatcherSkill - Real-time resource monitoring, budget prediction, and cost optimization.

Unlike GovernorSkill (which enforces static limits) and PerformanceTracker (which records
metrics after the fact), ResourceWatcher is **proactive and predictive**:

1. Tracks resource consumption as it happens (API costs, tokens, wall time)
2. Computes burn rate and predicts when budgets will be exhausted
3. Generates alerts at configurable thresholds (50%, 75%, 90%)
4. Computes cost-per-action analytics so agents know which skills are expensive
5. Recommends optimizations (model downgrades, action batching, skill substitution)
6. Provides real-time budget health summaries for LLM context injection

Pillars:
- Self-Improvement: Agents learn which actions cost most and optimize
- Revenue: Track cost-per-service-call to ensure profitable pricing
- Replication: Budget awareness prevents replicas from burning money
- Goal Setting: Budget forecasting informs what's achievable
"""

import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .base import Skill, SkillAction, SkillManifest, SkillResult

DATA_FILE = Path(__file__).parent.parent / "data" / "resource_watcher.json"

# Default alert thresholds (fraction of budget consumed)
DEFAULT_THRESHOLDS = [0.50, 0.75, 0.90, 0.95]

# Model cost estimates per 1K tokens (input/output) in USD
MODEL_COSTS = {
    "claude-sonnet-4-20250514": {"input": 0.003, "output": 0.015},
    "claude-3-5-sonnet-20241022": {"input": 0.003, "output": 0.015},
    "claude-3-haiku-20240307": {"input": 0.00025, "output": 0.00125},
    "gpt-4o": {"input": 0.005, "output": 0.015},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
    "local": {"input": 0.0, "output": 0.0},
}


class ResourceWatcherSkill(Skill):
    """
    Real-time resource monitoring with predictive budget analytics.

    The agent calls resource_watcher:record after each action to log costs,
    resource_watcher:status to get a live budget summary, and
    resource_watcher:forecast to predict budget exhaustion.
    """

    def __init__(self, credentials: Dict[str, str] = None):
        super().__init__(credentials)
        self._get_balance_fn: Optional[Callable[[], float]] = None
        self._get_model_fn: Optional[Callable[[], str]] = None
        self._data = self._load_data()

    def set_agent_hooks(
        self,
        get_balance: Callable[[], float] = None,
        get_model: Callable[[], str] = None,
    ):
        """Connect to agent for live balance and model queries."""
        self._get_balance_fn = get_balance
        self._get_model_fn = get_model

    # ── Persistence ───────────────────────────────────────────────

    def _load_data(self) -> Dict:
        DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
        if DATA_FILE.exists():
            try:
                return json.loads(DATA_FILE.read_text())
            except (json.JSONDecodeError, OSError):
                pass
        return self._default_data()

    def _default_data(self) -> Dict:
        return {
            "session_start": datetime.utcnow().isoformat(),
            "total_budget": 100.0,
            "alert_thresholds": DEFAULT_THRESHOLDS,
            "fired_alerts": [],
            "consumption_log": [],  # [{timestamp, skill, action, cost, tokens, duration_ms}]
            "cumulative_cost": 0.0,
            "cumulative_tokens": 0,
            "action_stats": {},  # {skill:action -> {count, total_cost, total_tokens, avg_duration_ms}}
            "model_usage": {},  # {model -> {calls, tokens, cost}}
            "optimizations_applied": [],
        }

    def _save(self):
        DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
        DATA_FILE.write_text(json.dumps(self._data, indent=2, default=str))

    # ── Manifest ──────────────────────────────────────────────────

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="resource_watcher",
            name="Resource Watcher",
            version="1.0.0",
            category="monitoring",
            description="Real-time resource monitoring, burn rate prediction, and cost optimization",
            required_credentials=[],
            install_cost=0,
            actions=[
                SkillAction(
                    name="record",
                    description="Record resource consumption for an action",
                    parameters={
                        "skill_id": {"type": "string", "required": True, "description": "Skill that executed"},
                        "action": {"type": "string", "required": True, "description": "Action name"},
                        "cost": {"type": "number", "required": False, "description": "USD cost (0 if unknown)"},
                        "tokens": {"type": "integer", "required": False, "description": "Tokens consumed"},
                        "duration_ms": {"type": "number", "required": False, "description": "Wall time in ms"},
                        "model": {"type": "string", "required": False, "description": "Model used"},
                    },
                    estimated_cost=0,
                    estimated_duration_seconds=0,
                    success_probability=0.99,
                ),
                SkillAction(
                    name="status",
                    description="Get live budget health summary (balance, burn rate, time remaining)",
                    parameters={},
                    estimated_cost=0,
                    estimated_duration_seconds=0,
                    success_probability=0.99,
                ),
                SkillAction(
                    name="forecast",
                    description="Predict when budget will be exhausted at current burn rate",
                    parameters={
                        "budget_override": {"type": "number", "required": False, "description": "Override total budget for forecast"},
                    },
                    estimated_cost=0,
                    estimated_duration_seconds=0,
                    success_probability=0.99,
                ),
                SkillAction(
                    name="top_costs",
                    description="Show most expensive actions ranked by total cost",
                    parameters={
                        "limit": {"type": "integer", "required": False, "description": "Number of top actions (default 10)"},
                    },
                    estimated_cost=0,
                    estimated_duration_seconds=0,
                    success_probability=0.99,
                ),
                SkillAction(
                    name="set_budget",
                    description="Set or update total budget and alert thresholds",
                    parameters={
                        "total_budget": {"type": "number", "required": True, "description": "Total budget in USD"},
                        "thresholds": {"type": "array", "required": False, "description": "Alert thresholds (e.g. [0.5, 0.75, 0.9])"},
                    },
                    estimated_cost=0,
                    estimated_duration_seconds=0,
                    success_probability=0.99,
                ),
                SkillAction(
                    name="recommend",
                    description="Get cost optimization recommendations based on usage patterns",
                    parameters={},
                    estimated_cost=0,
                    estimated_duration_seconds=0,
                    success_probability=0.95,
                ),
                SkillAction(
                    name="alerts",
                    description="List all fired budget alerts",
                    parameters={},
                    estimated_cost=0,
                    estimated_duration_seconds=0,
                    success_probability=0.99,
                ),
                SkillAction(
                    name="reset",
                    description="Reset all tracking data for a fresh session",
                    parameters={
                        "keep_budget": {"type": "boolean", "required": False, "description": "Keep budget settings (default true)"},
                    },
                    estimated_cost=0,
                    estimated_duration_seconds=0,
                    success_probability=0.99,
                ),
            ],
        )

    # ── Execute ───────────────────────────────────────────────────

    async def execute(self, action: str, params: Dict[str, Any] = None) -> SkillResult:
        params = params or {}
        handlers = {
            "record": self._record,
            "status": self._status,
            "forecast": self._forecast,
            "top_costs": self._top_costs,
            "set_budget": self._set_budget,
            "recommend": self._recommend,
            "alerts": self._alerts,
            "reset": self._reset,
        }
        handler = handlers.get(action)
        if not handler:
            return SkillResult(success=False, message=f"Unknown action: {action}")
        return await handler(params)

    # ── Actions ───────────────────────────────────────────────────

    async def _record(self, params: Dict) -> SkillResult:
        """Record a resource consumption event."""
        skill_id = params.get("skill_id", "unknown")
        action = params.get("action", "unknown")
        cost = float(params.get("cost", 0))
        tokens = int(params.get("tokens", 0))
        duration_ms = float(params.get("duration_ms", 0))
        model = params.get("model", "")

        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "skill": skill_id,
            "action": action,
            "cost": cost,
            "tokens": tokens,
            "duration_ms": duration_ms,
            "model": model,
        }

        # Append to log (keep last 1000 entries)
        self._data["consumption_log"].append(entry)
        if len(self._data["consumption_log"]) > 1000:
            self._data["consumption_log"] = self._data["consumption_log"][-1000:]

        # Update cumulative totals
        self._data["cumulative_cost"] += cost
        self._data["cumulative_tokens"] += tokens

        # Update per-action stats
        key = f"{skill_id}:{action}"
        stats = self._data["action_stats"].get(key, {
            "count": 0, "total_cost": 0, "total_tokens": 0,
            "total_duration_ms": 0, "avg_cost": 0, "avg_tokens": 0, "avg_duration_ms": 0,
        })
        stats["count"] += 1
        stats["total_cost"] += cost
        stats["total_tokens"] += tokens
        stats["total_duration_ms"] += duration_ms
        stats["avg_cost"] = stats["total_cost"] / stats["count"]
        stats["avg_tokens"] = stats["total_tokens"] / stats["count"]
        stats["avg_duration_ms"] = stats["total_duration_ms"] / stats["count"]
        self._data["action_stats"][key] = stats

        # Update model usage
        if model:
            mu = self._data["model_usage"].get(model, {"calls": 0, "tokens": 0, "cost": 0})
            mu["calls"] += 1
            mu["tokens"] += tokens
            mu["cost"] += cost
            self._data["model_usage"][model] = mu

        # Check alert thresholds
        alerts_fired = self._check_alerts()

        self._save()

        return SkillResult(
            success=True,
            message=f"Recorded {key}: ${cost:.4f}, {tokens} tokens",
            data={"entry": entry, "alerts_fired": alerts_fired},
        )

    async def _status(self, params: Dict) -> SkillResult:
        """Return live budget health summary."""
        total_budget = self._data["total_budget"]
        cumulative = self._data["cumulative_cost"]
        remaining = total_budget - cumulative
        pct_used = (cumulative / total_budget * 100) if total_budget > 0 else 0

        # Compute burn rate (cost per minute)
        burn_rate = self._compute_burn_rate()

        # Time remaining estimate
        if burn_rate > 0:
            minutes_remaining = remaining / burn_rate
            hours_remaining = minutes_remaining / 60
        else:
            minutes_remaining = float("inf")
            hours_remaining = float("inf")

        # Current balance from agent if available
        live_balance = None
        if self._get_balance_fn:
            try:
                live_balance = self._get_balance_fn()
            except Exception:
                pass

        total_actions = sum(s["count"] for s in self._data["action_stats"].values())
        cost_per_action = cumulative / total_actions if total_actions > 0 else 0

        summary = {
            "total_budget": total_budget,
            "spent": round(cumulative, 4),
            "remaining": round(remaining, 4),
            "pct_used": round(pct_used, 1),
            "burn_rate_per_min": round(burn_rate, 6),
            "burn_rate_per_hour": round(burn_rate * 60, 4),
            "minutes_remaining": round(minutes_remaining, 1) if minutes_remaining != float("inf") else "unlimited",
            "hours_remaining": round(hours_remaining, 1) if hours_remaining != float("inf") else "unlimited",
            "total_tokens": self._data["cumulative_tokens"],
            "total_actions": total_actions,
            "avg_cost_per_action": round(cost_per_action, 6),
            "active_alerts": len(self._data["fired_alerts"]),
        }

        if live_balance is not None:
            summary["live_agent_balance"] = round(live_balance, 4)

        health = "healthy"
        if pct_used >= 95:
            health = "critical"
        elif pct_used >= 75:
            health = "warning"
        elif pct_used >= 50:
            health = "caution"

        summary["health"] = health

        return SkillResult(
            success=True,
            message=f"Budget {health}: ${remaining:.2f} remaining ({pct_used:.1f}% used), "
                    f"burn rate ${burn_rate * 60:.4f}/hr",
            data=summary,
        )

    async def _forecast(self, params: Dict) -> SkillResult:
        """Predict budget exhaustion based on burn rate."""
        budget = float(params.get("budget_override", self._data["total_budget"]))
        cumulative = self._data["cumulative_cost"]
        remaining = budget - cumulative
        burn_rate = self._compute_burn_rate()

        if burn_rate <= 0:
            return SkillResult(
                success=True,
                message="No spending detected yet — cannot forecast.",
                data={"burn_rate": 0, "status": "no_data"},
            )

        minutes_left = remaining / burn_rate
        exhaustion_time = datetime.utcnow() + timedelta(minutes=minutes_left)

        # Forecast at different time horizons
        horizons = {}
        for hours in [1, 4, 8, 24]:
            projected_spend = burn_rate * 60 * hours
            projected_total = cumulative + projected_spend
            will_exceed = projected_total > budget
            horizons[f"{hours}h"] = {
                "projected_spend": round(projected_spend, 4),
                "projected_total": round(projected_total, 4),
                "will_exceed_budget": will_exceed,
                "remaining_after": round(budget - projected_total, 4),
            }

        return SkillResult(
            success=True,
            message=f"At current rate (${burn_rate * 60:.4f}/hr), budget exhausts at "
                    f"{exhaustion_time.strftime('%Y-%m-%d %H:%M UTC')} "
                    f"({minutes_left:.0f} min from now)",
            data={
                "burn_rate_per_min": round(burn_rate, 6),
                "burn_rate_per_hour": round(burn_rate * 60, 4),
                "remaining_budget": round(remaining, 4),
                "minutes_until_exhaustion": round(minutes_left, 1),
                "exhaustion_time": exhaustion_time.isoformat(),
                "horizons": horizons,
            },
        )

    async def _top_costs(self, params: Dict) -> SkillResult:
        """Rank actions by total cost."""
        limit = int(params.get("limit", 10))
        stats = self._data["action_stats"]

        ranked = sorted(stats.items(), key=lambda x: x[1]["total_cost"], reverse=True)[:limit]

        top = []
        for key, s in ranked:
            top.append({
                "action": key,
                "count": s["count"],
                "total_cost": round(s["total_cost"], 4),
                "avg_cost": round(s["avg_cost"], 6),
                "total_tokens": s["total_tokens"],
                "avg_duration_ms": round(s["avg_duration_ms"], 1),
            })

        return SkillResult(
            success=True,
            message=f"Top {len(top)} most expensive actions",
            data={"top_actions": top, "total_unique_actions": len(stats)},
        )

    async def _set_budget(self, params: Dict) -> SkillResult:
        """Set or update budget and thresholds."""
        budget = float(params["total_budget"])
        thresholds = params.get("thresholds", DEFAULT_THRESHOLDS)

        if budget <= 0:
            return SkillResult(success=False, message="Budget must be positive")

        # Validate thresholds
        valid_thresholds = []
        for t in thresholds:
            t = float(t)
            if 0 < t < 1:
                valid_thresholds.append(t)
        valid_thresholds.sort()

        self._data["total_budget"] = budget
        self._data["alert_thresholds"] = valid_thresholds or DEFAULT_THRESHOLDS
        # Reset fired alerts since thresholds changed
        self._data["fired_alerts"] = []
        self._save()

        return SkillResult(
            success=True,
            message=f"Budget set to ${budget:.2f} with alerts at {valid_thresholds}",
            data={"total_budget": budget, "thresholds": valid_thresholds},
        )

    async def _recommend(self, params: Dict) -> SkillResult:
        """Generate cost optimization recommendations."""
        recommendations = []
        stats = self._data["action_stats"]
        model_usage = self._data["model_usage"]

        # 1. Identify expensive actions that could use cheaper models
        for key, s in stats.items():
            if s["avg_cost"] > 0.01 and s["count"] >= 3:
                recommendations.append({
                    "type": "expensive_action",
                    "action": key,
                    "avg_cost": round(s["avg_cost"], 4),
                    "count": s["count"],
                    "suggestion": f"Action '{key}' averages ${s['avg_cost']:.4f}/call. "
                                  f"Consider batching or using a cheaper model.",
                    "potential_savings": round(s["total_cost"] * 0.5, 4),
                })

        # 2. Model downgrade suggestions
        for model, usage in model_usage.items():
            costs = MODEL_COSTS.get(model)
            if costs and costs.get("output", 0) > 0.005:  # Expensive model
                cheaper = self._find_cheaper_model(model)
                if cheaper:
                    savings_pct = 1 - (MODEL_COSTS[cheaper]["output"] / costs["output"])
                    recommendations.append({
                        "type": "model_downgrade",
                        "current_model": model,
                        "suggested_model": cheaper,
                        "current_cost": round(usage["cost"], 4),
                        "potential_savings_pct": round(savings_pct * 100, 1),
                        "suggestion": f"Switch from {model} to {cheaper} for ~{savings_pct*100:.0f}% savings "
                                      f"on non-critical tasks.",
                    })

        # 3. Burn rate warning
        burn_rate = self._compute_burn_rate()
        remaining = self._data["total_budget"] - self._data["cumulative_cost"]
        if burn_rate > 0:
            hours_left = remaining / (burn_rate * 60)
            if hours_left < 2:
                recommendations.append({
                    "type": "budget_critical",
                    "hours_remaining": round(hours_left, 1),
                    "suggestion": "Budget will be exhausted in under 2 hours. "
                                  "Reduce action frequency or switch to cheaper operations.",
                })

        # 4. Token waste detection
        for key, s in stats.items():
            if s["avg_tokens"] > 5000 and s["count"] >= 2:
                recommendations.append({
                    "type": "token_heavy",
                    "action": key,
                    "avg_tokens": int(s["avg_tokens"]),
                    "suggestion": f"Action '{key}' uses ~{int(s['avg_tokens'])} tokens/call. "
                                  f"Consider shorter prompts or summarization.",
                })

        # Sort by potential impact
        recommendations.sort(
            key=lambda r: r.get("potential_savings", r.get("potential_savings_pct", 0)),
            reverse=True,
        )

        return SkillResult(
            success=True,
            message=f"Generated {len(recommendations)} optimization recommendations",
            data={"recommendations": recommendations},
        )

    async def _alerts(self, params: Dict) -> SkillResult:
        """List all fired alerts."""
        return SkillResult(
            success=True,
            message=f"{len(self._data['fired_alerts'])} alerts fired",
            data={"alerts": self._data["fired_alerts"]},
        )

    async def _reset(self, params: Dict) -> SkillResult:
        """Reset tracking data."""
        keep_budget = params.get("keep_budget", True)
        old_budget = self._data["total_budget"]
        old_thresholds = self._data["alert_thresholds"]

        self._data = self._default_data()

        if keep_budget:
            self._data["total_budget"] = old_budget
            self._data["alert_thresholds"] = old_thresholds

        self._save()
        return SkillResult(
            success=True,
            message="Resource tracking reset",
            data={"budget_preserved": keep_budget},
        )

    # ── Helpers ───────────────────────────────────────────────────

    def _compute_burn_rate(self) -> float:
        """Compute cost per minute based on recent consumption."""
        log = self._data["consumption_log"]
        if len(log) < 2:
            return 0.0

        # Use timestamps to compute actual elapsed time
        try:
            first_ts = datetime.fromisoformat(log[0]["timestamp"])
            last_ts = datetime.fromisoformat(log[-1]["timestamp"])
            elapsed_minutes = (last_ts - first_ts).total_seconds() / 60
        except (ValueError, TypeError):
            return 0.0

        if elapsed_minutes <= 0:
            return 0.0

        total_cost = sum(entry.get("cost", 0) for entry in log)
        return total_cost / elapsed_minutes

    def _check_alerts(self) -> List[Dict]:
        """Check if any alert thresholds have been crossed."""
        budget = self._data["total_budget"]
        if budget <= 0:
            return []

        pct_used = self._data["cumulative_cost"] / budget
        fired_thresholds = {a["threshold"] for a in self._data["fired_alerts"]}
        new_alerts = []

        for threshold in self._data["alert_thresholds"]:
            if pct_used >= threshold and threshold not in fired_thresholds:
                alert = {
                    "threshold": threshold,
                    "pct_label": f"{threshold * 100:.0f}%",
                    "timestamp": datetime.utcnow().isoformat(),
                    "spent": round(self._data["cumulative_cost"], 4),
                    "budget": budget,
                    "message": f"ALERT: {threshold*100:.0f}% of budget consumed "
                               f"(${self._data['cumulative_cost']:.2f} / ${budget:.2f})",
                }
                self._data["fired_alerts"].append(alert)
                new_alerts.append(alert)

        return new_alerts

    def _find_cheaper_model(self, current_model: str) -> Optional[str]:
        """Find a cheaper model alternative."""
        current_cost = MODEL_COSTS.get(current_model, {}).get("output", 0)
        if current_cost == 0:
            return None

        # Find models that cost less but still have reasonable capability
        cheaper_models = []
        for model, costs in MODEL_COSTS.items():
            if model != current_model and costs["output"] < current_cost:
                cheaper_models.append((model, costs["output"]))

        if not cheaper_models:
            return None

        # Return the cheapest capable model (not local)
        cheaper_models.sort(key=lambda x: x[1], reverse=True)
        for model, cost in cheaper_models:
            if model != "local" and cost > 0:
                return model

        return cheaper_models[0][0] if cheaper_models else None

    # ── Public API for integration ────────────────────────────────

    def get_budget_context(self) -> str:
        """
        Return a brief budget context string for injection into LLM prompts.
        Called by the agent loop to keep the LLM aware of resource constraints.
        """
        budget = self._data["total_budget"]
        spent = self._data["cumulative_cost"]
        remaining = budget - spent
        pct = (spent / budget * 100) if budget > 0 else 0
        burn_rate = self._compute_burn_rate()

        if burn_rate > 0:
            mins_left = remaining / burn_rate
            time_str = f", ~{mins_left:.0f}min remaining"
        else:
            time_str = ""

        return f"[Budget: ${remaining:.2f}/{budget:.2f} remaining ({pct:.0f}% used){time_str}]"
