#!/usr/bin/env python3
"""
ForecastStrategyBridgeSkill - Wire revenue forecasts into strategic planning.

Connects RevenueForecastSkill projections to StrategySkill so the agent can
make data-driven strategic decisions about which pillar to invest in.

Without this bridge:
  - Forecasts exist in isolation with no effect on planning
  - Strategy assessments are manually set without revenue data
  - The agent cannot answer: "Should I invest in revenue growth or cost reduction?"

With this bridge:
  - Revenue forecasts automatically update the revenue pillar score
  - Break-even timeline informs urgency of revenue vs other work
  - Trend reversals trigger strategic re-assessment
  - Scenario analysis feeds into risk-adjusted recommendations
  - The agent gets a unified view: forecast + strategy + action plan

Actions:
  - sync: Pull latest forecast data and update strategy pillar scores
  - analyze: Deep analysis combining forecast + strategy for action plan
  - threshold: Configure score thresholds and auto-sync triggers
  - history: View sync history and score changes over time
  - status: Current bridge state and last sync results
  - auto_assess: Run forecast → trend → breakeven → strategy update in one shot

Pillar: Goal Setting (primary) + Revenue (supporting)
  - Goal Setting: Data-driven prioritization of work across pillars
  - Revenue: Forecast data directly informs strategic revenue decisions
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import Skill, SkillAction, SkillManifest, SkillResult

DATA_DIR = Path(__file__).parent.parent / "data"
STATE_FILE = DATA_DIR / "forecast_strategy_bridge.json"
FORECAST_FILE = DATA_DIR / "revenue_forecast.json"
STRATEGY_FILE = DATA_DIR / "strategy.json"
MAX_HISTORY = 200


def _now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _now_ts() -> float:
    return time.time()


class ForecastStrategyBridgeSkill(Skill):
    """
    Bridges revenue forecasts into strategic decision-making.
    Reads forecast data and updates strategy pillar scores automatically.
    """

    def __init__(self, credentials: Dict[str, str] = None):
        super().__init__(credentials)
        self._state = self._load_state()

    # ── State persistence ──────────────────────────────────────────

    def _load_state(self) -> dict:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        if STATE_FILE.exists():
            try:
                return json.loads(STATE_FILE.read_text())
            except (json.JSONDecodeError, OSError):
                pass
        return self._default_state()

    def _default_state(self) -> dict:
        return {
            "config": {
                # Thresholds for converting forecast metrics to pillar scores
                "profitable_score_boost": 20,   # bonus when already profitable
                "converging_score_base": 40,     # base score when converging to breakeven
                "not_converging_penalty": -15,   # penalty when trend is flat/declining
                "growth_trend_bonus": 10,        # bonus for growth trend
                "decline_trend_penalty": -10,    # penalty for decline trend
                "strong_trend_multiplier": 1.5,  # multiplier for strong trends
                "reversal_alert": True,          # flag trend reversals
                "auto_sync_on_forecast": False,  # auto-sync when forecast runs
                "base_revenue_score": 30,        # starting score before adjustments
                "max_score": 95,                 # cap to leave room for improvement
                "min_score": 5,                  # floor to avoid zero
            },
            "last_sync": None,
            "sync_history": [],
            "stats": {
                "total_syncs": 0,
                "total_auto_assesses": 0,
                "score_increases": 0,
                "score_decreases": 0,
                "trend_reversals_detected": 0,
            },
            "last_forecast_data": None,
            "last_strategy_update": None,
            "created_at": _now_iso(),
        }

    def _save_state(self):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        STATE_FILE.write_text(json.dumps(self._state, indent=2, default=str))

    # ── Manifest ───────────────────────────────────────────────────

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="forecast_strategy_bridge",
            name="Forecast-Strategy Bridge",
            category="revenue",
            description=(
                "Bridges revenue forecasts into strategic planning. "
                "Automatically updates strategy pillar scores based on "
                "forecast data, trend analysis, and break-even projections."
            ),
            version="1.0.0",
            required_credentials=[],
            actions=[
                SkillAction(
                    name="sync",
                    description="Pull latest forecast data and update strategy revenue pillar score",
                    parameters={
                        "force": "bool - Force sync even if data unchanged (default: false)",
                    },
                ),
                SkillAction(
                    name="analyze",
                    description="Deep analysis combining forecast + strategy for unified action plan",
                    parameters={
                        "include_scenarios": "bool - Include scenario analysis (default: true)",
                        "cost_per_period": "float - Override cost per period for breakeven calc",
                    },
                ),
                SkillAction(
                    name="threshold",
                    description="Configure score thresholds and auto-sync triggers",
                    parameters={
                        "profitable_score_boost": "int - Score boost when profitable (default: 20)",
                        "converging_score_base": "int - Base score when converging (default: 40)",
                        "growth_trend_bonus": "int - Bonus for growth trend (default: 10)",
                        "auto_sync_on_forecast": "bool - Auto-sync when forecast runs",
                    },
                ),
                SkillAction(
                    name="history",
                    description="View sync history and score changes over time",
                    parameters={
                        "limit": "int - Number of entries to return (default: 10)",
                    },
                ),
                SkillAction(
                    name="status",
                    description="Current bridge state, last sync results, and configuration",
                    parameters={},
                ),
                SkillAction(
                    name="auto_assess",
                    description="Run full forecast→trend→breakeven→strategy update pipeline",
                    parameters={
                        "cost_per_period": "float - Cost per period for breakeven (default: from forecast config)",
                    },
                ),
            ],
        )

    def check_credentials(self) -> bool:
        return True

    # ── Execute ────────────────────────────────────────────────────

    async def execute(self, action: str, params: Dict = None) -> SkillResult:
        params = params or {}
        handlers = {
            "sync": self._sync,
            "analyze": self._analyze,
            "threshold": self._threshold,
            "history": self._history,
            "status": self._status,
            "auto_assess": self._auto_assess,
        }
        handler = handlers.get(action)
        if handler:
            return await handler(params)
        return SkillResult(success=False, message=f"Unknown action: {action}")

    # ── Data readers ───────────────────────────────────────────────

    def _read_forecast_state(self) -> Optional[Dict]:
        """Read the latest state from RevenueForecastSkill."""
        if not FORECAST_FILE.exists():
            return None
        try:
            return json.loads(FORECAST_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            return None

    def _read_strategy_state(self) -> Optional[Dict]:
        """Read the current state from StrategySkill."""
        if not STRATEGY_FILE.exists():
            return None
        try:
            return json.loads(STRATEGY_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            return None

    def _write_strategy_state(self, data: Dict):
        """Write updated state to StrategySkill's data file."""
        STRATEGY_FILE.write_text(json.dumps(data, indent=2, default=str))

    # ── Score calculation ──────────────────────────────────────────

    def _calculate_revenue_score(self, forecast_data: Dict) -> Dict:
        """
        Convert forecast metrics into a revenue pillar score (0-100).

        Factors:
        1. Has revenue data at all? (base score)
        2. Trend direction (growth vs decline)
        3. Trend strength (strong vs weak)
        4. Break-even status (profitable, converging, not converging)
        5. Forecast trajectory (improving vs worsening)
        """
        cfg = self._state["config"]
        score = cfg["base_revenue_score"]
        factors = []
        capabilities = []
        gaps = []

        # Extract data from forecast state
        series = forecast_data.get("time_series", [])
        forecasts = forecast_data.get("forecasts", [])
        backtest = forecast_data.get("backtest_results", {})
        stats = forecast_data.get("stats", {})
        config = forecast_data.get("config", {})

        # Factor 1: Data availability
        data_points = len(series)
        if data_points == 0:
            return {
                "score": cfg["min_score"],
                "factors": [{"name": "no_data", "impact": 0, "detail": "No revenue data available"}],
                "capabilities": [],
                "gaps": ["No revenue data recorded - need to start tracking revenue"],
            }

        if data_points >= 10:
            score += 5
            factors.append({"name": "data_depth", "impact": 5, "detail": f"{data_points} data points available"})
            capabilities.append(f"Revenue tracking active ({data_points} data points)")
        elif data_points >= 3:
            score += 2
            factors.append({"name": "data_depth", "impact": 2, "detail": f"{data_points} data points (limited)"})
            capabilities.append(f"Revenue tracking started ({data_points} data points)")
        else:
            gaps.append(f"Only {data_points} data points - need more for reliable forecasting")

        # Factor 2: Revenue values
        values = [s.get("value", 0) for s in series if isinstance(s, dict)]
        if not values:
            values = [s for s in series if isinstance(s, (int, float))]

        if values:
            latest_value = values[-1] if values else 0
            avg_value = sum(values) / len(values) if values else 0

            if latest_value > 0:
                score += 10
                factors.append({"name": "has_revenue", "impact": 10, "detail": f"Latest revenue: ${latest_value:.4f}"})
                capabilities.append(f"Generating revenue (${latest_value:.4f}/period)")
            else:
                gaps.append("Zero revenue - need to activate revenue-generating services")

            # Trend detection from values
            if len(values) >= 3:
                recent_avg = sum(values[-3:]) / 3
                if len(values) >= 6:
                    earlier_avg = sum(values[-6:-3]) / 3
                else:
                    earlier_avg = values[0]

                if earlier_avg > 0:
                    change_pct = ((recent_avg - earlier_avg) / earlier_avg) * 100
                else:
                    change_pct = 100 if recent_avg > 0 else 0

                if change_pct > 10:
                    trend_bonus = cfg["growth_trend_bonus"]
                    if change_pct > 50:
                        trend_bonus = int(trend_bonus * cfg["strong_trend_multiplier"])
                    score += trend_bonus
                    factors.append({"name": "growth_trend", "impact": trend_bonus, "detail": f"Revenue growing {change_pct:+.1f}%"})
                    capabilities.append(f"Revenue growth trend ({change_pct:+.1f}%)")
                elif change_pct < -10:
                    trend_penalty = cfg["decline_trend_penalty"]
                    if change_pct < -50:
                        trend_penalty = int(trend_penalty * cfg["strong_trend_multiplier"])
                    score += trend_penalty  # negative
                    factors.append({"name": "decline_trend", "impact": trend_penalty, "detail": f"Revenue declining {change_pct:+.1f}%"})
                    gaps.append(f"Revenue declining ({change_pct:+.1f}%) - investigate and reverse")
                    self._state["stats"]["trend_reversals_detected"] += 1
                else:
                    factors.append({"name": "stable_trend", "impact": 0, "detail": f"Revenue stable ({change_pct:+.1f}%)"})

        # Factor 3: Break-even / profitability
        cost_per_period = config.get("cost_per_period", 0)
        if cost_per_period > 0 and values:
            latest_value = values[-1]
            if latest_value >= cost_per_period:
                boost = cfg["profitable_score_boost"]
                score += boost
                margin = ((latest_value - cost_per_period) / cost_per_period) * 100
                factors.append({"name": "profitable", "impact": boost, "detail": f"Profitable! {margin:.1f}% margin"})
                capabilities.append(f"Profitable ({margin:.1f}% margin)")
            elif len(values) >= 2:
                # Check if converging
                from_values = values[-min(len(values), 5):]
                if len(from_values) >= 2:
                    slope = (from_values[-1] - from_values[0]) / max(len(from_values) - 1, 1)
                    if slope > 0:
                        gap = cost_per_period - latest_value
                        periods = int(gap / slope) if slope > 0 else 999
                        base = cfg["converging_score_base"]
                        # Closer to breakeven = higher score
                        proximity_bonus = max(0, 10 - periods)
                        total = base // 3 + proximity_bonus
                        score += total
                        factors.append({"name": "converging", "impact": total, "detail": f"~{periods} periods to break-even"})
                        capabilities.append(f"Converging to profitability (~{periods} periods)")
                    else:
                        penalty = cfg["not_converging_penalty"]
                        score += penalty  # negative
                        factors.append({"name": "not_converging", "impact": penalty, "detail": "Not converging to break-even"})
                        gaps.append("Revenue not converging to break-even - need growth strategy")

        # Factor 4: Forecasting capability
        if stats.get("total_forecasts", 0) > 0:
            score += 5
            factors.append({"name": "forecasting_active", "impact": 5, "detail": f"{stats['total_forecasts']} forecasts generated"})
            capabilities.append("Revenue forecasting active")
        else:
            gaps.append("No forecasts generated yet - run forecast:forecast")

        if backtest:
            score += 5
            factors.append({"name": "backtesting_done", "impact": 5, "detail": "Model accuracy validated via backtest"})
            capabilities.append("Forecast model backtested and validated")

        # Clamp score
        score = max(cfg["min_score"], min(cfg["max_score"], score))

        return {
            "score": score,
            "factors": factors,
            "capabilities": capabilities,
            "gaps": gaps,
        }

    # ── Action handlers ────────────────────────────────────────────

    async def _sync(self, params: Dict) -> SkillResult:
        """Pull forecast data and update strategy revenue pillar score."""
        force = params.get("force", False)

        # Read forecast data
        forecast_data = self._read_forecast_state()
        if not forecast_data:
            return SkillResult(
                success=False,
                message="No forecast data found. Run forecast:record or forecast:forecast first.",
                data={"hint": "Use revenue_forecast skill to record data points before syncing."},
            )

        # Calculate score
        score_result = self._calculate_revenue_score(forecast_data)
        new_score = score_result["score"]

        # Read and update strategy
        strategy_data = self._read_strategy_state()
        if not strategy_data:
            # Create minimal strategy state if none exists
            strategy_data = {
                "pillars": {
                    "revenue": {
                        "name": "Revenue",
                        "score": 0.0,
                        "capabilities": [],
                        "gaps": [],
                        "last_assessed": None,
                    },
                    "self_improvement": {"name": "Self Improvement", "score": 0.0, "capabilities": [], "gaps": [], "last_assessed": None},
                    "replication": {"name": "Replication", "score": 0.0, "capabilities": [], "gaps": [], "last_assessed": None},
                    "goal_setting": {"name": "Goal Setting", "score": 0.0, "capabilities": [], "gaps": [], "last_assessed": None},
                },
                "journal": [],
                "work_log": [],
                "recommendations": [],
                "session_count": 0,
                "created_at": _now_iso(),
                "last_updated": _now_iso(),
            }

        old_score = strategy_data["pillars"]["revenue"].get("score", 0)

        # Check if anything changed
        if not force and old_score == new_score and self._state["last_sync"]:
            return SkillResult(
                success=True,
                message=f"Revenue score unchanged at {new_score}. Use force=true to resync.",
                data={"score": new_score, "changed": False},
            )

        # Update revenue pillar
        strategy_data["pillars"]["revenue"]["score"] = new_score
        strategy_data["pillars"]["revenue"]["capabilities"] = score_result["capabilities"]
        strategy_data["pillars"]["revenue"]["gaps"] = score_result["gaps"]
        strategy_data["pillars"]["revenue"]["last_assessed"] = _now_iso()
        strategy_data["last_updated"] = _now_iso()

        # Write updated strategy
        self._write_strategy_state(strategy_data)

        # Track sync
        direction = "up" if new_score > old_score else ("down" if new_score < old_score else "unchanged")
        if new_score > old_score:
            self._state["stats"]["score_increases"] += 1
        elif new_score < old_score:
            self._state["stats"]["score_decreases"] += 1

        sync_entry = {
            "timestamp": _now_iso(),
            "old_score": old_score,
            "new_score": new_score,
            "direction": direction,
            "factors": score_result["factors"],
            "capabilities_count": len(score_result["capabilities"]),
            "gaps_count": len(score_result["gaps"]),
        }
        self._state["sync_history"].append(sync_entry)
        if len(self._state["sync_history"]) > MAX_HISTORY:
            self._state["sync_history"] = self._state["sync_history"][-MAX_HISTORY:]

        self._state["last_sync"] = sync_entry
        self._state["last_forecast_data"] = {
            "series_length": len(forecast_data.get("time_series", [])),
            "forecasts_count": len(forecast_data.get("forecasts", [])),
            "has_backtest": bool(forecast_data.get("backtest_results")),
        }
        self._state["last_strategy_update"] = _now_iso()
        self._state["stats"]["total_syncs"] += 1
        self._save_state()

        factor_summary = ", ".join(f"{f['name']}({f['impact']:+d})" for f in score_result["factors"])
        return SkillResult(
            success=True,
            message=(
                f"Revenue pillar score: {old_score} → {new_score} ({direction}). "
                f"Factors: {factor_summary}. "
                f"{len(score_result['capabilities'])} capabilities, {len(score_result['gaps'])} gaps."
            ),
            data={
                "old_score": old_score,
                "new_score": new_score,
                "direction": direction,
                "factors": score_result["factors"],
                "capabilities": score_result["capabilities"],
                "gaps": score_result["gaps"],
                "changed": old_score != new_score,
            },
        )

    async def _analyze(self, params: Dict) -> SkillResult:
        """Deep analysis combining forecast + strategy for action plan."""
        include_scenarios = params.get("include_scenarios", True)
        cost_override = params.get("cost_per_period")

        # Read both data sources
        forecast_data = self._read_forecast_state()
        strategy_data = self._read_strategy_state()

        if not forecast_data:
            return SkillResult(
                success=False,
                message="No forecast data available. Record revenue data first.",
            )

        analysis = {
            "timestamp": _now_iso(),
            "revenue_assessment": {},
            "strategic_context": {},
            "action_plan": [],
            "risk_factors": [],
        }

        # Revenue assessment
        score_result = self._calculate_revenue_score(forecast_data)
        analysis["revenue_assessment"] = {
            "score": score_result["score"],
            "factors": score_result["factors"],
            "capabilities": score_result["capabilities"],
            "gaps": score_result["gaps"],
        }

        # Strategic context from all pillars
        if strategy_data:
            pillar_scores = {}
            for name, info in strategy_data.get("pillars", {}).items():
                pillar_scores[name] = info.get("score", 0)
            analysis["strategic_context"] = {
                "pillar_scores": pillar_scores,
                "weakest_pillar": min(pillar_scores, key=pillar_scores.get) if pillar_scores else None,
                "strongest_pillar": max(pillar_scores, key=pillar_scores.get) if pillar_scores else None,
                "average_score": sum(pillar_scores.values()) / len(pillar_scores) if pillar_scores else 0,
            }

        # Generate action plan based on combined analysis
        action_plan = []
        series = forecast_data.get("time_series", [])
        values = []
        for s in series:
            if isinstance(s, dict):
                values.append(s.get("value", 0))
            elif isinstance(s, (int, float)):
                values.append(s)

        config = forecast_data.get("config", {})
        cost = float(cost_override) if cost_override else config.get("cost_per_period", 0)

        if not values:
            action_plan.append({
                "priority": "critical",
                "action": "Start recording revenue data",
                "rationale": "No revenue data exists - cannot make data-driven decisions",
                "pillar": "revenue",
            })
        else:
            latest = values[-1] if values else 0

            # Profitability actions
            if cost > 0:
                if latest >= cost:
                    action_plan.append({
                        "priority": "medium",
                        "action": "Maintain revenue momentum and explore growth",
                        "rationale": f"Currently profitable (${latest:.4f} > ${cost:.4f})",
                        "pillar": "revenue",
                    })
                else:
                    action_plan.append({
                        "priority": "high",
                        "action": "Focus on revenue growth to reach break-even",
                        "rationale": f"Revenue ${latest:.4f} below costs ${cost:.4f}",
                        "pillar": "revenue",
                    })

            # Trend-based actions
            if len(values) >= 3:
                recent = sum(values[-3:]) / 3
                if len(values) >= 6:
                    earlier = sum(values[-6:-3]) / 3
                else:
                    earlier = values[0]
                if earlier > 0:
                    change = ((recent - earlier) / earlier) * 100
                    if change < -20:
                        action_plan.append({
                            "priority": "critical",
                            "action": "Investigate revenue decline and implement recovery plan",
                            "rationale": f"Revenue dropping {change:.1f}% - urgent attention needed",
                            "pillar": "revenue",
                        })
                    elif change > 20:
                        action_plan.append({
                            "priority": "low",
                            "action": "Document what's driving growth and double down",
                            "rationale": f"Revenue growing {change:.1f}% - capitalize on momentum",
                            "pillar": "revenue",
                        })

            # Cross-pillar recommendations
            if strategy_data:
                pillar_scores = analysis["strategic_context"].get("pillar_scores", {})
                weakest = analysis["strategic_context"].get("weakest_pillar")
                if weakest and weakest != "revenue":
                    weak_score = pillar_scores.get(weakest, 0)
                    rev_score = pillar_scores.get("revenue", 0)
                    if rev_score > weak_score + 20:
                        action_plan.append({
                            "priority": "high",
                            "action": f"Shift focus to {weakest.replace('_', ' ')} (score: {weak_score})",
                            "rationale": f"Revenue ({rev_score}) is well ahead of {weakest} ({weak_score}). Balance pillars.",
                            "pillar": weakest,
                        })

        # Risk factors
        risk_factors = []
        if len(values) < 5:
            risk_factors.append({
                "risk": "insufficient_data",
                "severity": "medium",
                "detail": f"Only {len(values)} data points - forecasts may be unreliable",
            })
        if cost > 0 and values and values[-1] < cost * 0.5:
            risk_factors.append({
                "risk": "deep_loss",
                "severity": "high",
                "detail": f"Revenue less than 50% of costs - significant cash burn",
            })

        analysis["action_plan"] = action_plan
        analysis["risk_factors"] = risk_factors

        # Sort action plan by priority
        priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        analysis["action_plan"].sort(key=lambda x: priority_order.get(x["priority"], 4))

        plan_summary = "; ".join(f"[{a['priority'].upper()}] {a['action']}" for a in action_plan[:3])
        return SkillResult(
            success=True,
            message=f"Analysis complete. Score: {score_result['score']}. Plan: {plan_summary}",
            data=analysis,
        )

    async def _threshold(self, params: Dict) -> SkillResult:
        """Configure score thresholds and auto-sync triggers."""
        cfg = self._state["config"]
        updated = []

        configurable = [
            "profitable_score_boost", "converging_score_base",
            "not_converging_penalty", "growth_trend_bonus",
            "decline_trend_penalty", "strong_trend_multiplier",
            "reversal_alert", "auto_sync_on_forecast",
            "base_revenue_score", "max_score", "min_score",
        ]

        for key in configurable:
            if key in params:
                old_val = cfg.get(key)
                new_val = params[key]
                # Type coercion
                if isinstance(old_val, bool):
                    new_val = bool(new_val)
                elif isinstance(old_val, int):
                    new_val = int(new_val)
                elif isinstance(old_val, float):
                    new_val = float(new_val)
                cfg[key] = new_val
                updated.append(f"{key}: {old_val} → {new_val}")

        if not updated:
            return SkillResult(
                success=True,
                message=f"Current config: {json.dumps(cfg, indent=2)}",
                data={"config": cfg, "updated": []},
            )

        self._save_state()
        return SkillResult(
            success=True,
            message=f"Updated {len(updated)} settings: {', '.join(updated)}",
            data={"config": cfg, "updated": updated},
        )

    async def _history(self, params: Dict) -> SkillResult:
        """View sync history and score changes."""
        limit = int(params.get("limit", 10))
        limit = max(1, min(50, limit))

        history = self._state.get("sync_history", [])
        recent = history[-limit:]

        if not recent:
            return SkillResult(
                success=True,
                message="No sync history yet. Run forecast_strategy_bridge:sync first.",
                data={"history": [], "total": 0},
            )

        # Calculate trends from history
        scores = [h["new_score"] for h in history]
        trend = "stable"
        if len(scores) >= 3:
            recent_avg = sum(scores[-3:]) / 3
            if len(scores) >= 6:
                earlier_avg = sum(scores[-6:-3]) / 3
            else:
                earlier_avg = scores[0]
            if recent_avg > earlier_avg + 2:
                trend = "improving"
            elif recent_avg < earlier_avg - 2:
                trend = "declining"

        return SkillResult(
            success=True,
            message=f"Showing {len(recent)}/{len(history)} sync entries. Score trend: {trend}",
            data={
                "history": recent,
                "total": len(history),
                "score_trend": trend,
                "latest_score": scores[-1] if scores else None,
            },
        )

    async def _status(self, params: Dict) -> SkillResult:
        """Current bridge state and last sync results."""
        last_sync = self._state.get("last_sync")
        stats = self._state["stats"]
        config = self._state["config"]

        # Read current scores
        strategy_data = self._read_strategy_state()
        current_revenue_score = None
        if strategy_data:
            current_revenue_score = strategy_data.get("pillars", {}).get("revenue", {}).get("score")

        status = {
            "last_sync": last_sync,
            "current_revenue_score": current_revenue_score,
            "stats": stats,
            "config_summary": {
                "auto_sync": config["auto_sync_on_forecast"],
                "base_score": config["base_revenue_score"],
                "score_range": f"{config['min_score']}-{config['max_score']}",
            },
            "forecast_data_available": self._state.get("last_forecast_data"),
            "last_strategy_update": self._state.get("last_strategy_update"),
        }

        sync_msg = f"Last sync: {last_sync['timestamp']}, score: {last_sync['new_score']}" if last_sync else "Never synced"
        return SkillResult(
            success=True,
            message=f"Bridge status: {sync_msg}. Total syncs: {stats['total_syncs']}.",
            data=status,
        )

    async def _auto_assess(self, params: Dict) -> SkillResult:
        """Run full forecast→trend→breakeven→strategy update pipeline."""
        cost_override = params.get("cost_per_period")

        forecast_data = self._read_forecast_state()
        if not forecast_data:
            return SkillResult(
                success=False,
                message="No forecast data available. Record revenue data first.",
            )

        # Step 1: Calculate score from forecast data
        score_result = self._calculate_revenue_score(forecast_data)

        # Step 2: Compute trend info from series
        series = forecast_data.get("time_series", [])
        values = []
        for s in series:
            if isinstance(s, dict):
                values.append(s.get("value", 0))
            elif isinstance(s, (int, float)):
                values.append(s)

        trend_info = {"direction": "unknown", "data_points": len(values)}
        if len(values) >= 2:
            slope = (values[-1] - values[0]) / max(len(values) - 1, 1)
            if slope > 0.001:
                trend_info["direction"] = "growth"
            elif slope < -0.001:
                trend_info["direction"] = "decline"
            else:
                trend_info["direction"] = "stable"
            trend_info["slope"] = round(slope, 6)

        # Step 3: Compute breakeven info
        config = forecast_data.get("config", {})
        cost = float(cost_override) if cost_override else config.get("cost_per_period", 0)
        breakeven_info = {"cost_per_period": cost}
        if cost > 0 and values:
            latest = values[-1]
            breakeven_info["is_profitable"] = latest >= cost
            breakeven_info["current_revenue"] = round(latest, 4)
            breakeven_info["gap"] = round(latest - cost, 4)
            if latest < cost and len(values) >= 2:
                slope = (values[-1] - values[0]) / max(len(values) - 1, 1)
                if slope > 0:
                    periods = int((cost - latest) / slope)
                    breakeven_info["estimated_periods"] = periods
                else:
                    breakeven_info["estimated_periods"] = None
                    breakeven_info["note"] = "Not converging"

        # Step 4: Update strategy
        sync_result = await self._sync({"force": True})

        # Step 5: Generate strategic recommendation
        recommendation = self._generate_recommendation(score_result, trend_info, breakeven_info)

        self._state["stats"]["total_auto_assesses"] += 1
        self._save_state()

        return SkillResult(
            success=True,
            message=(
                f"Auto-assess complete. Revenue score: {score_result['score']}. "
                f"Trend: {trend_info['direction']}. "
                f"{'Profitable' if breakeven_info.get('is_profitable') else 'Not profitable'}. "
                f"Recommendation: {recommendation['summary']}"
            ),
            data={
                "score": score_result,
                "trend": trend_info,
                "breakeven": breakeven_info,
                "sync_result": sync_result.data if sync_result.success else None,
                "recommendation": recommendation,
            },
        )

    def _generate_recommendation(self, score_result: Dict, trend_info: Dict, breakeven_info: Dict) -> Dict:
        """Generate a strategic recommendation based on combined analysis."""
        score = score_result["score"]
        direction = trend_info.get("direction", "unknown")
        profitable = breakeven_info.get("is_profitable", False)

        if score >= 70 and profitable:
            return {
                "summary": "Revenue strong - invest in other pillars",
                "priority": "revenue_maintenance",
                "detail": "Revenue pillar is healthy. Consider shifting focus to weaker pillars.",
                "suggested_focus": "weakest_pillar",
            }
        elif score >= 50 and direction == "growth":
            return {
                "summary": "Revenue growing - maintain momentum",
                "priority": "revenue_growth",
                "detail": "Positive trajectory. Continue current revenue strategy while monitoring.",
                "suggested_focus": "revenue",
            }
        elif direction == "decline":
            return {
                "summary": "Revenue declining - urgent intervention needed",
                "priority": "revenue_critical",
                "detail": "Revenue trend is negative. Prioritize revenue recovery over other work.",
                "suggested_focus": "revenue",
            }
        elif score < 30:
            return {
                "summary": "Revenue weak - prioritize revenue generation",
                "priority": "revenue_build",
                "detail": "Revenue pillar needs significant investment. Focus on service deployment.",
                "suggested_focus": "revenue",
            }
        else:
            return {
                "summary": "Revenue moderate - balanced investment recommended",
                "priority": "balanced",
                "detail": "Revenue is neither critical nor strong. Balance with other pillar work.",
                "suggested_focus": "balanced",
            }
