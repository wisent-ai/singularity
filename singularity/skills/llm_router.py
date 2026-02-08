#!/usr/bin/env python3
"""
CostAwareLLMRouter - Intelligent model routing for cost optimization.

Analyzes tasks and routes them to the cheapest LLM model capable of handling
the required complexity level. Tracks cost savings, learns from outcomes,
and adapts routing decisions over time.

Pillars served:
- Self-Improvement: Adaptive model selection based on outcome feedback
- Revenue: Cost reduction = higher profit margins on services
"""

import json
import math
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Any

from .base import Skill, SkillManifest, SkillAction, SkillResult

DATA_DIR = Path(__file__).parent.parent / "data"
ROUTER_FILE = DATA_DIR / "llm_router.json"

# Model tiers - ordered from cheapest to most expensive
MODEL_TIERS = {
    "budget": {
        "models": [
            {"provider": "openai", "model": "gpt-4o-mini", "cost_per_1m_input": 0.15, "cost_per_1m_output": 0.6},
            {"provider": "anthropic", "model": "claude-3-haiku-20240307", "cost_per_1m_input": 0.25, "cost_per_1m_output": 1.25},
            {"provider": "vertex", "model": "gemini-2.0-flash-001", "cost_per_1m_input": 0.35, "cost_per_1m_output": 1.5},
        ],
        "max_complexity": "simple",
        "description": "Fast, cheap models for straightforward tasks",
    },
    "standard": {
        "models": [
            {"provider": "openai", "model": "gpt-4o", "cost_per_1m_input": 2.5, "cost_per_1m_output": 10.0},
            {"provider": "vertex", "model": "gemini-1.5-pro-002", "cost_per_1m_input": 1.25, "cost_per_1m_output": 5.0},
        ],
        "max_complexity": "medium",
        "description": "Capable models for moderate-complexity tasks",
    },
    "premium": {
        "models": [
            {"provider": "anthropic", "model": "claude-sonnet-4-20250514", "cost_per_1m_input": 3.0, "cost_per_1m_output": 15.0},
        ],
        "max_complexity": "complex",
        "description": "Most capable models for complex reasoning tasks",
    },
}

# Task complexity indicators
COMPLEXITY_SIGNALS = {
    "simple": {
        "patterns": [
            r"\b(list|format|summarize|extract|convert|translate)\b",
            r"\b(yes|no|true|false)\b.*\?",
            r"\b(what is|define|lookup)\b",
        ],
        "max_estimated_tokens": 500,
        "description": "Lookups, formatting, simple Q&A, extraction",
    },
    "medium": {
        "patterns": [
            r"\b(analyze|compare|explain|review|debug)\b",
            r"\b(write|create|generate)\b.*\b(function|code|script)\b",
            r"\b(suggest|recommend|evaluate)\b",
        ],
        "max_estimated_tokens": 2000,
        "description": "Analysis, code generation, recommendations",
    },
    "complex": {
        "patterns": [
            r"\b(architect|design|refactor)\b",
            r"\b(multi[- ]step|plan|strategy)\b",
            r"\b(reason|prove|derive|optimize)\b",
            r"\b(security|vulnerability|audit)\b",
        ],
        "max_estimated_tokens": 4000,
        "description": "Architecture, complex reasoning, multi-step planning",
    },
}


class CostAwareLLMRouter(Skill):
    """
    Intelligent LLM model router that minimizes cost while maintaining quality.

    Classifies tasks by complexity, routes to cheapest adequate model,
    tracks outcomes, and learns routing preferences over time.
    """

    def __init__(self, credentials: Dict = None):
        super().__init__(credentials)
        self._ensure_data()

    def _ensure_data(self):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        if not ROUTER_FILE.exists():
            self._save_state(self._default_state())

    def _default_state(self) -> Dict:
        return {
            "routing_history": [],
            "model_performance": {},
            "task_type_stats": {},
            "total_cost_actual": 0.0,
            "total_cost_if_premium": 0.0,
            "total_savings": 0.0,
            "budget_mode": False,
            "budget_limit_usd": 1.0,
            "spent_this_period": 0.0,
            "period_start": time.time(),
            "created_at": time.time(),
        }

    def _load_state(self) -> Dict:
        try:
            with open(ROUTER_FILE, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return self._default_state()

    def _save_state(self, state: Dict):
        ROUTER_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(ROUTER_FILE, "w") as f:
            json.dump(state, f, indent=2)

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="llm_router",
            name="Cost-Aware LLM Router",
            version="1.0.0",
            category="meta",
            description="Routes tasks to cheapest adequate LLM model, tracks savings, learns from outcomes",
            actions=[
                SkillAction(
                    name="route",
                    description="Classify a task and recommend the optimal model",
                    parameters={
                        "task": {
                            "type": "string",
                            "required": True,
                            "description": "The task description to route",
                        },
                        "required_quality": {
                            "type": "string",
                            "required": False,
                            "description": "Minimum quality: 'any', 'good', 'best' (default: 'good')",
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="record_outcome",
                    description="Record the outcome of a routed task for learning",
                    parameters={
                        "routing_id": {
                            "type": "string",
                            "required": True,
                            "description": "The routing ID from the route action",
                        },
                        "success": {
                            "type": "boolean",
                            "required": True,
                            "description": "Whether the task was completed successfully",
                        },
                        "quality_score": {
                            "type": "number",
                            "required": False,
                            "description": "Quality score 0.0-1.0 (optional)",
                        },
                        "actual_tokens": {
                            "type": "integer",
                            "required": False,
                            "description": "Actual tokens used (optional)",
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="savings_report",
                    description="Get a report on cost savings from intelligent routing",
                    parameters={},
                    estimated_cost=0,
                ),
                SkillAction(
                    name="model_leaderboard",
                    description="See performance rankings of all models by task type",
                    parameters={
                        "task_type": {
                            "type": "string",
                            "required": False,
                            "description": "Filter by task type (simple/medium/complex)",
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="set_budget",
                    description="Set a spending budget limit for automatic cost control",
                    parameters={
                        "limit_usd": {
                            "type": "number",
                            "required": True,
                            "description": "Budget limit in USD per period",
                        },
                        "reset_period": {
                            "type": "boolean",
                            "required": False,
                            "description": "Reset the current spending period (default: false)",
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="override",
                    description="Force a specific model for the next task (bypass routing)",
                    parameters={
                        "provider": {
                            "type": "string",
                            "required": True,
                            "description": "Provider name (anthropic, openai, vertex)",
                        },
                        "model": {
                            "type": "string",
                            "required": True,
                            "description": "Model name to use",
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="status",
                    description="Get current router status including budget and model stats",
                    parameters={},
                    estimated_cost=0,
                ),
            ],
            required_credentials=[],
        )

    def check_credentials(self) -> bool:
        return True

    async def execute(self, action: str, params: Dict) -> SkillResult:
        handlers = {
            "route": self._route,
            "record_outcome": self._record_outcome,
            "savings_report": self._savings_report,
            "model_leaderboard": self._model_leaderboard,
            "set_budget": self._set_budget,
            "override": self._override,
            "status": self._status,
        }
        handler = handlers.get(action)
        if not handler:
            return SkillResult(success=False, message=f"Unknown action: {action}")
        return handler(params)

    # --- Task Classification ---

    def classify_complexity(self, task: str) -> Dict:
        """Classify task complexity based on content analysis."""
        task_lower = task.lower()
        scores = {"simple": 0.0, "medium": 0.0, "complex": 0.0}

        # Pattern matching
        for level, config in COMPLEXITY_SIGNALS.items():
            for pattern in config["patterns"]:
                if re.search(pattern, task_lower):
                    scores[level] += 1.0

        # Length-based heuristics
        word_count = len(task.split())
        if word_count < 20:
            scores["simple"] += 0.5
        elif word_count < 80:
            scores["medium"] += 0.5
        else:
            scores["complex"] += 0.5

        # Code indicators add complexity
        code_indicators = task.count("```") + task.count("def ") + task.count("class ")
        if code_indicators > 2:
            scores["complex"] += 0.5
        elif code_indicators > 0:
            scores["medium"] += 0.5

        # Multi-step indicators
        step_indicators = len(re.findall(r"\b(then|next|after|finally|step \d)\b", task_lower))
        if step_indicators > 2:
            scores["complex"] += 0.5
        elif step_indicators > 0:
            scores["medium"] += 0.3

        # Determine winner
        if not any(scores.values()):
            classification = "medium"  # Default to medium if no signals
        else:
            classification = max(scores, key=scores.get)

        return {
            "classification": classification,
            "scores": scores,
            "confidence": max(scores.values()) / (sum(scores.values()) or 1),
            "word_count": word_count,
        }

    def _select_model(self, complexity: str, quality: str, state: Dict) -> Dict:
        """Select the cheapest adequate model for the given complexity and quality."""
        # Quality mapping to minimum tier
        quality_to_min_tier = {
            "any": "budget",
            "good": "budget",  # budget models can handle simple tasks well
            "best": "premium",
        }

        # Complexity to minimum tier
        complexity_to_min_tier = {
            "simple": "budget",
            "medium": "standard",
            "complex": "premium",
        }

        tier_order = ["budget", "standard", "premium"]

        # Start from the tier required by complexity
        min_tier_by_complexity = complexity_to_min_tier.get(complexity, "standard")
        min_tier_by_quality = quality_to_min_tier.get(quality, "budget")

        # Take the higher of the two requirements
        min_tier_idx = max(
            tier_order.index(min_tier_by_complexity),
            tier_order.index(min_tier_by_quality),
        )

        # Budget mode: try to use cheaper models if possible
        if state.get("budget_mode"):
            budget_limit = state.get("budget_limit_usd", 1.0)
            spent = state.get("spent_this_period", 0.0)
            remaining = budget_limit - spent
            if remaining < 0.01:
                # Force budget tier
                min_tier_idx = 0

        selected_tier = tier_order[min_tier_idx]
        tier_config = MODEL_TIERS[selected_tier]

        # Check learned performance data to pick best model within tier
        model_perf = state.get("model_performance", {})
        best_model = None
        best_score = -1.0

        for model_info in tier_config["models"]:
            model_key = f"{model_info['provider']}:{model_info['model']}"
            perf = model_perf.get(model_key, {})
            total = perf.get("total", 0)
            successes = perf.get("successes", 0)

            if total == 0:
                # No data - give slight exploration bonus
                score = 0.6 + (0.1 / (1 + model_info["cost_per_1m_input"]))
            else:
                # Wilson score lower bound for confidence
                success_rate = successes / total
                z = 1.96  # 95% confidence
                denominator = 1 + z * z / total
                center = success_rate + z * z / (2 * total)
                spread = z * math.sqrt(
                    (success_rate * (1 - success_rate) + z * z / (4 * total)) / total
                )
                wilson_lower = (center - spread) / denominator

                # Combine quality with cost efficiency
                cost_factor = 1.0 / (1.0 + model_info["cost_per_1m_input"])
                score = wilson_lower * 0.7 + cost_factor * 0.3

            if score > best_score:
                best_score = score
                best_model = model_info

        if not best_model:
            # Fallback to first model in tier
            best_model = tier_config["models"][0]

        return {
            "tier": selected_tier,
            "provider": best_model["provider"],
            "model": best_model["model"],
            "cost_per_1m_input": best_model["cost_per_1m_input"],
            "cost_per_1m_output": best_model["cost_per_1m_output"],
            "selection_score": round(best_score, 4),
        }

    # --- Actions ---

    def _route(self, params: Dict) -> SkillResult:
        """Route a task to the optimal model."""
        task = params.get("task", "").strip()
        if not task:
            return SkillResult(success=False, message="Task description required")

        quality = params.get("required_quality", "good").lower()
        if quality not in ("any", "good", "best"):
            quality = "good"

        state = self._load_state()

        # Classify task
        classification = self.classify_complexity(task)

        # Select model
        selection = self._select_model(classification["classification"], quality, state)

        # Calculate what premium would cost for savings tracking
        premium_model = MODEL_TIERS["premium"]["models"][0]
        estimated_tokens = {"simple": 500, "medium": 1500, "complex": 3000}
        est_tokens = estimated_tokens.get(classification["classification"], 1500)
        est_premium_cost = (est_tokens / 1_000_000) * (premium_model["cost_per_1m_input"] + premium_model["cost_per_1m_output"])
        est_actual_cost = (est_tokens / 1_000_000) * (selection["cost_per_1m_input"] + selection["cost_per_1m_output"])

        # Create routing record
        routing_id = f"r_{int(time.time())}_{hash(task) % 10000:04d}"
        record = {
            "routing_id": routing_id,
            "task_preview": task[:200],
            "complexity": classification["classification"],
            "confidence": round(classification["confidence"], 3),
            "quality_requested": quality,
            "selected_tier": selection["tier"],
            "selected_provider": selection["provider"],
            "selected_model": selection["model"],
            "estimated_cost": round(est_actual_cost, 6),
            "premium_cost": round(est_premium_cost, 6),
            "estimated_savings": round(est_premium_cost - est_actual_cost, 6),
            "timestamp": time.time(),
            "outcome": None,
        }

        history = state.get("routing_history", [])
        history.append(record)
        # Keep last 500 records
        if len(history) > 500:
            history = history[-500:]
        state["routing_history"] = history
        self._save_state(state)

        savings_pct = ((est_premium_cost - est_actual_cost) / est_premium_cost * 100) if est_premium_cost > 0 else 0

        return SkillResult(
            success=True,
            message=(
                f"Route: {classification['classification']} task → "
                f"{selection['provider']}:{selection['model']} "
                f"(tier: {selection['tier']}, est. savings: {savings_pct:.0f}%)"
            ),
            data={
                "routing_id": routing_id,
                "complexity": classification,
                "recommendation": {
                    "provider": selection["provider"],
                    "model": selection["model"],
                    "tier": selection["tier"],
                },
                "cost_estimate": {
                    "estimated_cost": round(est_actual_cost, 6),
                    "premium_cost": round(est_premium_cost, 6),
                    "estimated_savings": round(est_premium_cost - est_actual_cost, 6),
                    "savings_percent": round(savings_pct, 1),
                },
            },
        )

    def _record_outcome(self, params: Dict) -> SkillResult:
        """Record outcome for a routed task to improve future routing."""
        routing_id = params.get("routing_id", "").strip()
        if not routing_id:
            return SkillResult(success=False, message="routing_id required")

        success = params.get("success")
        if success is None:
            return SkillResult(success=False, message="success (boolean) required")
        success = bool(success)

        quality_score = params.get("quality_score")
        if quality_score is not None:
            try:
                quality_score = float(quality_score)
                quality_score = max(0.0, min(1.0, quality_score))
            except (ValueError, TypeError):
                quality_score = None

        actual_tokens = params.get("actual_tokens")

        state = self._load_state()

        # Find the routing record
        history = state.get("routing_history", [])
        record = None
        for r in history:
            if r.get("routing_id") == routing_id:
                record = r
                break

        if not record:
            return SkillResult(success=False, message=f"Routing ID '{routing_id}' not found")

        # Update record
        record["outcome"] = {
            "success": success,
            "quality_score": quality_score,
            "actual_tokens": actual_tokens,
            "recorded_at": time.time(),
        }

        # Update model performance stats
        model_key = f"{record['selected_provider']}:{record['selected_model']}"
        model_perf = state.setdefault("model_performance", {})
        perf = model_perf.setdefault(model_key, {
            "total": 0, "successes": 0, "failures": 0,
            "total_quality": 0.0, "quality_count": 0,
            "total_tokens": 0,
        })
        perf["total"] += 1
        if success:
            perf["successes"] += 1
        else:
            perf["failures"] += 1
        if quality_score is not None:
            perf["total_quality"] += quality_score
            perf["quality_count"] += 1
        if actual_tokens:
            perf["total_tokens"] += actual_tokens

        # Update task type stats
        complexity = record.get("complexity", "medium")
        task_stats = state.setdefault("task_type_stats", {})
        ts = task_stats.setdefault(complexity, {
            "total": 0, "successes": 0,
            "models_used": {},
        })
        ts["total"] += 1
        if success:
            ts["successes"] += 1
        models_used = ts.setdefault("models_used", {})
        mu = models_used.setdefault(model_key, {"total": 0, "successes": 0})
        mu["total"] += 1
        if success:
            mu["successes"] += 1

        # Update cost tracking
        est_cost = record.get("estimated_cost", 0)
        premium_cost = record.get("premium_cost", 0)
        state["total_cost_actual"] = state.get("total_cost_actual", 0) + est_cost
        state["total_cost_if_premium"] = state.get("total_cost_if_premium", 0) + premium_cost
        state["total_savings"] = state.get("total_savings", 0) + (premium_cost - est_cost)
        state["spent_this_period"] = state.get("spent_this_period", 0) + est_cost

        # Check budget
        budget_warning = None
        if state.get("budget_mode"):
            remaining = state.get("budget_limit_usd", 1.0) - state.get("spent_this_period", 0)
            if remaining < 0.01:
                budget_warning = "Budget exhausted! Switching to budget-only models."
                state["budget_mode"] = True

        self._save_state(state)

        msg = f"Outcome recorded for {model_key}: {'success' if success else 'failure'}"
        if budget_warning:
            msg += f" | WARNING: {budget_warning}"

        return SkillResult(
            success=True,
            message=msg,
            data={
                "routing_id": routing_id,
                "model": model_key,
                "outcome_success": success,
                "quality_score": quality_score,
                "model_stats": {
                    "total": perf["total"],
                    "success_rate": perf["successes"] / perf["total"] if perf["total"] > 0 else 0,
                },
                "budget_warning": budget_warning,
            },
        )

    def _savings_report(self, params: Dict) -> SkillResult:
        """Generate a cost savings report."""
        state = self._load_state()

        total_actual = state.get("total_cost_actual", 0)
        total_premium = state.get("total_cost_if_premium", 0)
        total_savings = state.get("total_savings", 0)
        savings_pct = (total_savings / total_premium * 100) if total_premium > 0 else 0

        history = state.get("routing_history", [])
        completed = [r for r in history if r.get("outcome") is not None]
        total_routed = len(history)
        total_completed = len(completed)

        # Tier breakdown
        tier_counts = {}
        for r in history:
            tier = r.get("selected_tier", "unknown")
            tier_counts[tier] = tier_counts.get(tier, 0) + 1

        # Success rate by tier
        tier_success = {}
        for r in completed:
            tier = r.get("selected_tier", "unknown")
            ts = tier_success.setdefault(tier, {"total": 0, "successes": 0})
            ts["total"] += 1
            if r.get("outcome", {}).get("success"):
                ts["successes"] += 1

        tier_rates = {
            tier: round(s["successes"] / s["total"], 3) if s["total"] > 0 else 0
            for tier, s in tier_success.items()
        }

        msg_lines = ["=== LLM Routing Savings Report ==="]
        msg_lines.append(f"Total routed: {total_routed} tasks ({total_completed} with outcomes)")
        msg_lines.append(f"Actual cost:  ${total_actual:.4f}")
        msg_lines.append(f"Premium cost: ${total_premium:.4f}")
        msg_lines.append(f"Total saved:  ${total_savings:.4f} ({savings_pct:.1f}%)")
        msg_lines.append(f"Tier usage: {tier_counts}")
        msg_lines.append(f"Tier success rates: {tier_rates}")

        return SkillResult(
            success=True,
            message="\n".join(msg_lines),
            data={
                "total_routed": total_routed,
                "total_completed": total_completed,
                "total_cost_actual": round(total_actual, 6),
                "total_cost_if_premium": round(total_premium, 6),
                "total_savings": round(total_savings, 6),
                "savings_percent": round(savings_pct, 1),
                "tier_distribution": tier_counts,
                "tier_success_rates": tier_rates,
            },
        )

    def _model_leaderboard(self, params: Dict) -> SkillResult:
        """Rank models by performance."""
        task_type = params.get("task_type", "").strip().lower()

        state = self._load_state()
        model_perf = state.get("model_performance", {})

        if not model_perf:
            return SkillResult(
                success=True,
                message="No model performance data yet. Route some tasks first.",
                data={"leaderboard": []},
            )

        leaderboard = []
        for model_key, perf in model_perf.items():
            total = perf.get("total", 0)
            if total == 0:
                continue

            success_rate = perf.get("successes", 0) / total
            avg_quality = None
            if perf.get("quality_count", 0) > 0:
                avg_quality = perf["total_quality"] / perf["quality_count"]

            # Find cost info
            provider, model = model_key.split(":", 1)
            cost_info = None
            for tier_config in MODEL_TIERS.values():
                for m in tier_config["models"]:
                    if m["provider"] == provider and m["model"] == model:
                        cost_info = m
                        break

            cost_per_1k = 0
            if cost_info:
                cost_per_1k = (cost_info["cost_per_1m_input"] + cost_info["cost_per_1m_output"]) / 1000

            # Value score: quality per dollar
            value_score = success_rate / (cost_per_1k + 0.001)

            leaderboard.append({
                "model": model_key,
                "total_tasks": total,
                "success_rate": round(success_rate, 3),
                "avg_quality": round(avg_quality, 3) if avg_quality is not None else None,
                "cost_per_1k_tokens": round(cost_per_1k, 4),
                "value_score": round(value_score, 2),
            })

        # Sort by value score
        leaderboard.sort(key=lambda x: x["value_score"], reverse=True)

        # Filter by task type if requested
        if task_type and task_type in state.get("task_type_stats", {}):
            ts = state["task_type_stats"][task_type]
            models_used = ts.get("models_used", {})
            filtered = []
            for entry in leaderboard:
                mu = models_used.get(entry["model"])
                if mu:
                    entry["task_type_total"] = mu["total"]
                    entry["task_type_success_rate"] = round(
                        mu["successes"] / mu["total"], 3
                    ) if mu["total"] > 0 else 0
                    filtered.append(entry)
            leaderboard = filtered

        msg = f"Model leaderboard ({len(leaderboard)} models)"
        if task_type:
            msg += f" for '{task_type}' tasks"

        return SkillResult(
            success=True,
            message=msg,
            data={"leaderboard": leaderboard, "task_type_filter": task_type or None},
        )

    def _set_budget(self, params: Dict) -> SkillResult:
        """Set a spending budget."""
        limit = params.get("limit_usd")
        if limit is None:
            return SkillResult(success=False, message="limit_usd required")

        try:
            limit = float(limit)
            if limit <= 0:
                return SkillResult(success=False, message="limit_usd must be positive")
        except (ValueError, TypeError):
            return SkillResult(success=False, message="limit_usd must be a number")

        reset = bool(params.get("reset_period", False))

        state = self._load_state()
        state["budget_mode"] = True
        state["budget_limit_usd"] = limit
        if reset:
            state["spent_this_period"] = 0.0
            state["period_start"] = time.time()

        self._save_state(state)

        remaining = limit - state.get("spent_this_period", 0)

        return SkillResult(
            success=True,
            message=f"Budget set to ${limit:.2f}/period. Remaining: ${remaining:.4f}",
            data={
                "budget_limit": limit,
                "spent_this_period": state.get("spent_this_period", 0),
                "remaining": round(remaining, 4),
                "budget_mode": True,
            },
        )

    def _override(self, params: Dict) -> SkillResult:
        """Force a specific model for manual override."""
        provider = params.get("provider", "").strip()
        model = params.get("model", "").strip()

        if not provider or not model:
            return SkillResult(success=False, message="provider and model required")

        return SkillResult(
            success=True,
            message=f"Override: use {provider}:{model} for next task",
            data={
                "provider": provider,
                "model": model,
                "note": "Apply this override in your CognitionEngine configuration",
            },
        )

    def _status(self, params: Dict) -> SkillResult:
        """Get router status."""
        state = self._load_state()

        model_perf = state.get("model_performance", {})
        model_summary = {}
        for key, perf in model_perf.items():
            total = perf.get("total", 0)
            model_summary[key] = {
                "total": total,
                "success_rate": round(perf.get("successes", 0) / total, 3) if total > 0 else 0,
            }

        history = state.get("routing_history", [])
        recent = history[-5:] if history else []
        recent_summary = [
            {
                "routing_id": r.get("routing_id"),
                "complexity": r.get("complexity"),
                "model": f"{r.get('selected_provider')}:{r.get('selected_model')}",
                "has_outcome": r.get("outcome") is not None,
            }
            for r in recent
        ]

        budget_info = None
        if state.get("budget_mode"):
            budget_info = {
                "limit": state.get("budget_limit_usd", 1.0),
                "spent": round(state.get("spent_this_period", 0), 4),
                "remaining": round(
                    state.get("budget_limit_usd", 1.0) - state.get("spent_this_period", 0), 4
                ),
            }

        return SkillResult(
            success=True,
            message=f"Router: {len(history)} tasks routed, {len(model_perf)} models tracked",
            data={
                "total_routed": len(history),
                "total_savings": round(state.get("total_savings", 0), 4),
                "model_performance": model_summary,
                "recent_routings": recent_summary,
                "budget": budget_info,
                "tiers_available": list(MODEL_TIERS.keys()),
            },
        )
