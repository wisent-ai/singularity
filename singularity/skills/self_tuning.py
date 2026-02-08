#!/usr/bin/env python3
"""
SelfTuningSkill - Auto-adjusts agent parameters based on observability metrics.

This is the critical self-improvement feedback loop that uses runtime performance
data to automatically tune the agent's internal systems. Currently supports:

1. **LLM Router Tuning** - Analyzes model success rates, latency, and cost from
   ObservabilitySkill metrics and LLM router history, then adjusts model preferences
   to favor better-performing models and demote underperformers.

2. **Skill Execution Tuning** - Identifies skills with high error rates or latency
   and generates recommendations for improvement.

3. **Budget Tuning** - Adjusts spending budgets based on actual utilization patterns.

This completes the observe → analyze → tune feedback loop:
  metrics collected → patterns detected → parameters adjusted → better performance

Pillar: Self-Improvement (autonomous performance optimization without human input)

Actions:
- analyze: Gather metrics and analyze current performance across all tunable systems
- tune_router: Auto-adjust LLM router based on model performance analysis
- tune_budget: Adjust budget limits based on actual spending patterns
- recommend: Generate tuning recommendations without applying them
- history: View past tuning actions and their effects
- configure: Set tuning sensitivity, thresholds, and auto-apply behavior
- status: Current tuning state and health
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from .base import Skill, SkillResult, SkillManifest, SkillAction

DATA_FILE = Path(__file__).parent.parent / "data" / "self_tuning.json"
MAX_HISTORY = 200


def _now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _now_ts() -> float:
    return time.time()


class SelfTuningSkill(Skill):
    """
    Automatically tunes agent parameters based on observability metrics.

    Reads performance data from ObservabilitySkill and LLM router history,
    analyzes patterns, and adjusts configuration to optimize performance,
    cost, and reliability.
    """

    def __init__(self, credentials: Dict[str, str] = None):
        super().__init__(credentials)
        self._store = None

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="self_tuning",
            name="Self-Tuning Agent",
            version="1.0.0",
            category="meta",
            description="Auto-adjusts agent parameters (LLM routing, budgets) based on observability metrics",
            actions=self.get_actions(),
            required_credentials=[],
            install_cost=0,
            author="singularity",
        )

    def get_actions(self) -> List[SkillAction]:
        return [
            SkillAction(
                name="analyze",
                description="Gather and analyze current performance metrics across all tunable systems",
                parameters={
                    "window_minutes": {"type": "number", "required": False,
                                       "description": "Analysis time window in minutes (default: 60)"},
                },
            ),
            SkillAction(
                name="tune_router",
                description="Auto-adjust LLM router model preferences based on performance data",
                parameters={
                    "dry_run": {"type": "boolean", "required": False,
                                "description": "Preview adjustments without applying (default: False)"},
                },
            ),
            SkillAction(
                name="tune_budget",
                description="Adjust spending budgets based on actual utilization patterns",
                parameters={
                    "dry_run": {"type": "boolean", "required": False,
                                "description": "Preview adjustments without applying (default: False)"},
                    "target_utilization": {"type": "number", "required": False,
                                           "description": "Target budget utilization 0.0-1.0 (default: 0.8)"},
                },
            ),
            SkillAction(
                name="recommend",
                description="Generate tuning recommendations without applying them",
                parameters={},
            ),
            SkillAction(
                name="history",
                description="View past tuning actions and their effects",
                parameters={
                    "limit": {"type": "integer", "required": False,
                              "description": "Max history entries (default: 20)"},
                },
            ),
            SkillAction(
                name="configure",
                description="Set tuning sensitivity, thresholds, and behavior",
                parameters={
                    "min_samples": {"type": "integer", "required": False,
                                    "description": "Min data points before tuning (default: 5)"},
                    "success_rate_threshold": {"type": "number", "required": False,
                                               "description": "Below this success rate, demote model (default: 0.7)"},
                    "promote_threshold": {"type": "number", "required": False,
                                          "description": "Above this success rate, promote model (default: 0.9)"},
                    "max_adjustment_pct": {"type": "number", "required": False,
                                           "description": "Max weight adjustment per tune cycle (default: 0.2)"},
                    "emit_events": {"type": "boolean", "required": False,
                                    "description": "Emit EventBus events for tuning actions (default: True)"},
                },
            ),
            SkillAction(
                name="status",
                description="Current tuning state, health, and last actions",
                parameters={},
            ),
        ]

    async def execute(self, action: str, params: Dict) -> SkillResult:
        handlers = {
            "analyze": self._analyze,
            "tune_router": self._tune_router,
            "tune_budget": self._tune_budget,
            "recommend": self._recommend,
            "history": self._history,
            "configure": self._configure,
            "status": self._status,
        }
        handler = handlers.get(action)
        if not handler:
            return SkillResult(
                success=False,
                message=f"Unknown action: {action}. Available: {', '.join(handlers.keys())}",
            )
        return await handler(params)

    # ── Persistence ──

    def _load(self) -> Dict:
        if self._store is not None:
            return self._store
        try:
            if DATA_FILE.exists():
                with open(DATA_FILE) as f:
                    self._store = json.load(f)
                    return self._store
        except (json.JSONDecodeError, OSError):
            pass
        self._store = self._default_state()
        return self._store

    def _save(self, state: Dict = None):
        if state is not None:
            self._store = state
        if self._store is None:
            return
        DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(DATA_FILE, "w") as f:
            json.dump(self._store, f, indent=2)

    def _default_state(self) -> Dict:
        return {
            "config": {
                "min_samples": 5,
                "success_rate_threshold": 0.7,
                "promote_threshold": 0.9,
                "max_adjustment_pct": 0.2,
                "emit_events": True,
            },
            "model_weights": {},  # model_key -> weight (1.0 = default)
            "tuning_history": [],
            "stats": {
                "total_tune_cycles": 0,
                "total_adjustments": 0,
                "last_analyze_at": None,
                "last_tune_at": None,
            },
        }

    # ── Handlers ──

    async def _analyze(self, params: Dict) -> SkillResult:
        """Gather metrics and analyze performance."""
        window_minutes = float(params.get("window_minutes", 60))

        analysis = {
            "timestamp": _now_iso(),
            "window_minutes": window_minutes,
            "router_analysis": None,
            "skill_analysis": None,
            "budget_analysis": None,
        }

        # 1. Analyze LLM router performance
        router_data = await self._get_router_data()
        if router_data:
            analysis["router_analysis"] = self._analyze_router(router_data)

        # 2. Analyze skill execution metrics
        skill_metrics = await self._get_skill_metrics(window_minutes)
        if skill_metrics:
            analysis["skill_analysis"] = self._analyze_skills(skill_metrics)

        # 3. Analyze budget utilization
        budget_data = await self._get_budget_data()
        if budget_data:
            analysis["budget_analysis"] = self._analyze_budget(budget_data)

        state = self._load()
        state["stats"]["last_analyze_at"] = _now_iso()
        self._save(state)

        has_data = any([
            analysis["router_analysis"],
            analysis["skill_analysis"],
            analysis["budget_analysis"],
        ])

        return SkillResult(
            success=True,
            message=f"Analysis complete. "
                    f"Router: {'available' if analysis['router_analysis'] else 'no data'}. "
                    f"Skills: {'available' if analysis['skill_analysis'] else 'no data'}. "
                    f"Budget: {'available' if analysis['budget_analysis'] else 'no data'}.",
            data=analysis,
        )

    async def _tune_router(self, params: Dict) -> SkillResult:
        """Auto-adjust LLM router based on performance analysis."""
        dry_run = params.get("dry_run", False)
        state = self._load()
        config = state["config"]

        router_data = await self._get_router_data()
        if not router_data:
            return SkillResult(
                success=False,
                message="No router performance data available. Need LLM router history to tune.",
            )

        analysis = self._analyze_router(router_data)
        if not analysis or not analysis.get("models"):
            return SkillResult(
                success=False,
                message="Insufficient model performance data for tuning.",
            )

        adjustments = []
        weights = dict(state.get("model_weights", {}))

        for model_key, stats in analysis["models"].items():
            total = stats.get("total", 0)
            if total < config["min_samples"]:
                continue

            success_rate = stats.get("success_rate", 0)
            current_weight = weights.get(model_key, 1.0)
            max_adj = config["max_adjustment_pct"]

            if success_rate < config["success_rate_threshold"]:
                # Demote: reduce weight
                severity = (config["success_rate_threshold"] - success_rate) / config["success_rate_threshold"]
                adjustment = -min(max_adj, severity * max_adj)
                new_weight = max(0.1, current_weight + adjustment)
            elif success_rate >= config["promote_threshold"]:
                # Promote: increase weight
                adjustment = min(max_adj, (success_rate - config["promote_threshold"]) * max_adj * 2)
                new_weight = min(2.0, current_weight + adjustment)
            else:
                # Neutral: nudge toward 1.0
                if current_weight > 1.0:
                    new_weight = max(1.0, current_weight - 0.05)
                elif current_weight < 1.0:
                    new_weight = min(1.0, current_weight + 0.05)
                else:
                    continue

            new_weight = round(new_weight, 3)
            if abs(new_weight - current_weight) < 0.001:
                continue

            adjustments.append({
                "model": model_key,
                "old_weight": round(current_weight, 3),
                "new_weight": new_weight,
                "reason": f"success_rate={success_rate:.2f}, samples={total}",
                "direction": "demote" if new_weight < current_weight else "promote",
            })
            weights[model_key] = new_weight

        record = {
            "type": "router_tune",
            "timestamp": _now_iso(),
            "dry_run": dry_run,
            "adjustments": adjustments,
            "model_count": len(analysis["models"]),
        }

        if not dry_run and adjustments:
            state["model_weights"] = weights
            state["stats"]["total_tune_cycles"] += 1
            state["stats"]["total_adjustments"] += len(adjustments)
            state["stats"]["last_tune_at"] = _now_iso()
            state["tuning_history"].append(record)
            state["tuning_history"] = state["tuning_history"][-MAX_HISTORY:]
            self._save(state)

            if config.get("emit_events", True):
                await self._emit_event("self_tuning.router_tuned", {
                    "adjustments": adjustments,
                    "total_adjustments": len(adjustments),
                })

        prefix = "[DRY RUN] " if dry_run else ""
        if not adjustments:
            return SkillResult(
                success=True,
                message=f"{prefix}No adjustments needed. All models within acceptable thresholds.",
                data={"adjustments": [], "analysis": analysis},
            )

        demoted = sum(1 for a in adjustments if a["direction"] == "demote")
        promoted = sum(1 for a in adjustments if a["direction"] == "promote")

        return SkillResult(
            success=True,
            message=f"{prefix}Applied {len(adjustments)} adjustment(s): "
                    f"{promoted} promoted, {demoted} demoted.",
            data={
                "adjustments": adjustments,
                "weights": weights,
                "analysis": analysis,
            },
        )

    async def _tune_budget(self, params: Dict) -> SkillResult:
        """Adjust budget based on utilization patterns."""
        dry_run = params.get("dry_run", False)
        target_util = float(params.get("target_utilization", 0.8))
        target_util = max(0.1, min(1.0, target_util))

        budget_data = await self._get_budget_data()
        if not budget_data:
            return SkillResult(
                success=False,
                message="No budget data available. Need LLM router budget data.",
            )

        limit = budget_data.get("budget_limit_usd", 0)
        spent = budget_data.get("spent_this_period", 0)

        if limit <= 0:
            return SkillResult(
                success=True,
                message="No budget limit set. Nothing to tune.",
                data={"budget_mode": False},
            )

        utilization = spent / limit if limit > 0 else 0
        recommended_limit = (spent / target_util) if spent > 0 else limit

        # Clamp to reasonable bounds (50% to 200% of current)
        recommended_limit = max(limit * 0.5, min(limit * 2.0, recommended_limit))
        recommended_limit = round(recommended_limit, 2)

        adjustment = {
            "current_limit": limit,
            "current_spent": round(spent, 4),
            "current_utilization": round(utilization, 3),
            "target_utilization": target_util,
            "recommended_limit": recommended_limit,
        }

        state = self._load()
        record = {
            "type": "budget_tune",
            "timestamp": _now_iso(),
            "dry_run": dry_run,
            "adjustment": adjustment,
        }

        if not dry_run and abs(recommended_limit - limit) > 0.01:
            # Apply via LLM router
            applied = await self._apply_budget(recommended_limit)
            record["applied"] = applied

            state["tuning_history"].append(record)
            state["tuning_history"] = state["tuning_history"][-MAX_HISTORY:]
            state["stats"]["total_tune_cycles"] += 1
            state["stats"]["total_adjustments"] += 1
            state["stats"]["last_tune_at"] = _now_iso()
            self._save(state)

            if state["config"].get("emit_events", True):
                await self._emit_event("self_tuning.budget_tuned", adjustment)

        prefix = "[DRY RUN] " if dry_run else ""
        direction = "increase" if recommended_limit > limit else "decrease"

        return SkillResult(
            success=True,
            message=f"{prefix}Budget {direction}: ${limit:.2f} → ${recommended_limit:.2f} "
                    f"(utilization: {utilization:.0%} → target: {target_util:.0%})",
            data=adjustment,
        )

    async def _recommend(self, params: Dict) -> SkillResult:
        """Generate recommendations without applying."""
        recommendations = []

        # Router recommendations
        router_data = await self._get_router_data()
        if router_data:
            analysis = self._analyze_router(router_data)
            if analysis:
                state = self._load()
                config = state["config"]
                for model_key, stats in analysis.get("models", {}).items():
                    total = stats.get("total", 0)
                    if total < config["min_samples"]:
                        continue
                    sr = stats.get("success_rate", 0)
                    if sr < config["success_rate_threshold"]:
                        recommendations.append({
                            "system": "llm_router",
                            "type": "demote_model",
                            "model": model_key,
                            "reason": f"Low success rate: {sr:.1%} ({total} samples)",
                            "severity": "high" if sr < 0.5 else "medium",
                        })
                    elif sr >= config["promote_threshold"]:
                        recommendations.append({
                            "system": "llm_router",
                            "type": "promote_model",
                            "model": model_key,
                            "reason": f"High success rate: {sr:.1%} ({total} samples)",
                            "severity": "low",
                        })

        # Budget recommendations
        budget_data = await self._get_budget_data()
        if budget_data:
            limit = budget_data.get("budget_limit_usd", 0)
            spent = budget_data.get("spent_this_period", 0)
            if limit > 0:
                util = spent / limit
                if util > 0.95:
                    recommendations.append({
                        "system": "budget",
                        "type": "increase_budget",
                        "reason": f"Budget {util:.0%} utilized, risk of exhaustion",
                        "severity": "high",
                    })
                elif util < 0.3:
                    recommendations.append({
                        "system": "budget",
                        "type": "decrease_budget",
                        "reason": f"Budget only {util:.0%} utilized, over-allocated",
                        "severity": "low",
                    })

        # Skill recommendations
        skill_metrics = await self._get_skill_metrics(60)
        if skill_metrics:
            analysis = self._analyze_skills(skill_metrics)
            for skill_id, stats in analysis.get("skills", {}).items():
                if stats.get("error_rate", 0) > 0.3:
                    recommendations.append({
                        "system": "skills",
                        "type": "investigate_errors",
                        "skill": skill_id,
                        "reason": f"High error rate: {stats['error_rate']:.0%}",
                        "severity": "high",
                    })

        if not recommendations:
            return SkillResult(
                success=True,
                message="No tuning recommendations. All systems operating within thresholds.",
                data={"recommendations": []},
            )

        high = sum(1 for r in recommendations if r["severity"] == "high")
        return SkillResult(
            success=True,
            message=f"Generated {len(recommendations)} recommendation(s) ({high} high severity).",
            data={"recommendations": recommendations},
        )

    async def _history(self, params: Dict) -> SkillResult:
        """View tuning history."""
        state = self._load()
        limit = min(int(params.get("limit", 20)), MAX_HISTORY)
        entries = state["tuning_history"][-limit:]
        return SkillResult(
            success=True,
            message=f"Showing {len(entries)} tuning actions.",
            data={"history": entries, "total": len(state["tuning_history"])},
        )

    async def _configure(self, params: Dict) -> SkillResult:
        """Update tuning configuration."""
        state = self._load()
        config = state["config"]
        updated = []

        config_keys = [
            "min_samples", "success_rate_threshold", "promote_threshold",
            "max_adjustment_pct", "emit_events",
        ]

        for key in config_keys:
            if key in params:
                old_val = config[key]
                config[key] = params[key]
                updated.append(f"{key}: {old_val} -> {params[key]}")

        if not updated:
            return SkillResult(
                success=True,
                message="No configuration changes requested.",
                data={"config": config},
            )

        self._save(state)
        return SkillResult(
            success=True,
            message=f"Updated {len(updated)} setting(s): {'; '.join(updated)}",
            data={"config": config, "updated": updated},
        )

    async def _status(self, params: Dict) -> SkillResult:
        """Return tuning status."""
        state = self._load()
        return SkillResult(
            success=True,
            message=f"Self-tuning active. {state['stats']['total_tune_cycles']} tune cycles, "
                    f"{state['stats']['total_adjustments']} total adjustments. "
                    f"Last tune: {state['stats']['last_tune_at'] or 'never'}.",
            data={
                "config": state["config"],
                "stats": state["stats"],
                "model_weights": state["model_weights"],
            },
        )

    # ── Data retrieval ──

    async def _get_router_data(self) -> Optional[Dict]:
        """Get LLM router performance data."""
        # Try via skill context
        if self.context:
            try:
                result = await self.context.call_skill("llm_router", "status", {})
                if result.success and result.data:
                    return result.data
            except Exception:
                pass

        # Fallback: read router file directly
        try:
            router_file = Path(__file__).parent.parent / "data" / "llm_router.json"
            if router_file.exists():
                with open(router_file) as f:
                    return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass

        return None

    async def _get_skill_metrics(self, window_minutes: float) -> Optional[Dict]:
        """Get skill execution metrics from ObservabilitySkill."""
        if self.context:
            try:
                result = await self.context.call_skill("observability", "query", {
                    "name": "skill.execution.count",
                    "aggregation": "sum",
                    "window_minutes": window_minutes,
                })
                if result.success and result.data:
                    return result.data
            except Exception:
                pass

        # Fallback: read metrics file
        try:
            metrics_file = Path(__file__).parent.parent / "data" / "observability_metrics.json"
            if metrics_file.exists():
                with open(metrics_file) as f:
                    data = json.load(f)
                    return data
        except (json.JSONDecodeError, OSError):
            pass

        return None

    async def _get_budget_data(self) -> Optional[Dict]:
        """Get budget data from LLM router."""
        router_data = await self._get_router_data()
        if router_data and router_data.get("budget_mode"):
            return {
                "budget_mode": True,
                "budget_limit_usd": router_data.get("budget_limit_usd", 0),
                "spent_this_period": router_data.get("spent_this_period", 0),
            }
        # Also check the raw file
        if router_data:
            return {
                "budget_mode": router_data.get("budget_mode", False),
                "budget_limit_usd": router_data.get("budget_limit_usd", 0),
                "spent_this_period": router_data.get("spent_this_period", 0),
            }
        return None

    # ── Analysis ──

    def _analyze_router(self, router_data: Dict) -> Optional[Dict]:
        """Analyze LLM router performance data."""
        model_perf = router_data.get("model_performance", {})
        if not model_perf:
            return None

        models = {}
        total_tasks = 0
        total_successes = 0

        for model_key, perf in model_perf.items():
            total = perf.get("total", 0)
            successes = perf.get("successes", 0)
            failures = perf.get("failures", 0)
            quality_count = perf.get("quality_count", 0)
            total_quality = perf.get("total_quality", 0)

            total_tasks += total
            total_successes += successes

            models[model_key] = {
                "total": total,
                "successes": successes,
                "failures": failures,
                "success_rate": round(successes / total, 3) if total > 0 else 0,
                "avg_quality": round(total_quality / quality_count, 3) if quality_count > 0 else None,
            }

        return {
            "models": models,
            "total_tasks": total_tasks,
            "overall_success_rate": round(total_successes / total_tasks, 3) if total_tasks > 0 else 0,
        }

    def _analyze_skills(self, metrics_data: Dict) -> Dict:
        """Analyze skill execution metrics."""
        skills = {}
        series = metrics_data.get("series", {})

        for key, series_data in series.items():
            # Parse skill name from series key
            labels = series_data.get("labels", {})
            skill_id = labels.get("skill", "")
            if not skill_id:
                continue

            name = series_data.get("name", "")
            points = series_data.get("points", [])
            total_value = sum(p.get("v", 0) for p in points)

            if skill_id not in skills:
                skills[skill_id] = {"executions": 0, "errors": 0, "latency_sum": 0, "latency_count": 0}

            if "count" in name or "execution" in name:
                skills[skill_id]["executions"] += total_value
            if "error" in name:
                skills[skill_id]["errors"] += total_value
            if "latency" in name:
                skills[skill_id]["latency_sum"] += total_value
                skills[skill_id]["latency_count"] += len(points)

        # Compute derived stats
        for skill_id, stats in skills.items():
            executions = stats["executions"]
            stats["error_rate"] = round(stats["errors"] / executions, 3) if executions > 0 else 0
            stats["avg_latency_ms"] = round(
                stats["latency_sum"] / stats["latency_count"], 1
            ) if stats["latency_count"] > 0 else None

        return {"skills": skills}

    def _analyze_budget(self, budget_data: Dict) -> Dict:
        """Analyze budget utilization."""
        limit = budget_data.get("budget_limit_usd", 0)
        spent = budget_data.get("spent_this_period", 0)
        return {
            "budget_mode": budget_data.get("budget_mode", False),
            "limit": limit,
            "spent": round(spent, 4),
            "utilization": round(spent / limit, 3) if limit > 0 else 0,
            "remaining": round(limit - spent, 4) if limit > 0 else 0,
        }

    # ── Actions ──

    async def _apply_budget(self, new_limit: float) -> bool:
        """Apply new budget via LLM router."""
        if self.context:
            try:
                result = await self.context.call_skill("llm_router", "set_budget", {
                    "limit_usd": new_limit,
                })
                return result.success
            except Exception:
                pass
        return False

    async def _emit_event(self, topic: str, data: Dict):
        """Emit event via EventBus."""
        if not self.context:
            return
        try:
            await self.context.call_skill("event", "publish", {
                "topic": topic,
                "data": data,
            })
        except Exception:
            pass

    async def initialize(self) -> bool:
        self.initialized = True
        return True
