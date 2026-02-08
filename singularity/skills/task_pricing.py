#!/usr/bin/env python3
"""
TaskPricingSkill - Autonomous dynamic pricing for agent services.

Critical Revenue Generation infrastructure. The agent can offer services
via ServiceAPI and process payments, but it cannot autonomously PRICE its
work. Without pricing, the agent can't generate revenue autonomously.

This skill provides:
1. ESTIMATE - Predict task cost from description and required skills
2. QUOTE - Generate detailed quotes with line items and margins
3. RECORD - Log actual costs after execution for calibration
4. LEARN - Auto-adjust pricing models from prediction vs actual data
5. DYNAMIC PRICING - Adjust prices based on demand, complexity, urgency
6. MARGIN MANAGEMENT - Ensure profitability with configurable margins

Pricing model:
  base_cost = sum(skill_estimated_costs) + token_cost_estimate
  complexity_multiplier = f(description_length, skill_count, urgency)
  price = base_cost * complexity_multiplier * margin_multiplier * demand_factor

Revenue flow integration:
  1. Customer submits task via ServiceAPI
  2. TaskPricingSkill generates a quote
  3. Customer accepts quote
  4. Task executes, actual costs recorded
  5. TaskPricingSkill compares prediction vs actual
  6. Model auto-calibrates for future accuracy

Pillar: Revenue Generation (primary), Self-Improvement (learns from errors)
"""

import json
import math
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from .base import Skill, SkillResult, SkillManifest, SkillAction


PRICING_FILE = Path(__file__).parent.parent / "data" / "task_pricing.json"
MAX_QUOTES = 2000
MAX_HISTORY = 5000


# Default per-skill cost estimates (USD) when no historical data
DEFAULT_SKILL_COSTS = {
    "code_review": 0.02,
    "content": 0.015,
    "summarize_text": 0.01,
    "data_analysis": 0.025,
    "seo_audit": 0.01,
    "api_docs": 0.02,
    "github": 0.005,
    "shell": 0.001,
    "filesystem": 0.001,
    "browser": 0.05,
    "email": 0.002,
    "twitter": 0.003,
    "deployment": 0.03,
    "web_scraper": 0.04,
    "mcp_client": 0.01,
    "request": 0.005,
}

# Token cost estimates per model family (USD per 1K tokens)
TOKEN_COSTS = {
    "claude-sonnet": {"input": 0.003, "output": 0.015},
    "claude-haiku": {"input": 0.00025, "output": 0.00125},
    "claude-opus": {"input": 0.015, "output": 0.075},
    "gpt-4": {"input": 0.03, "output": 0.06},
    "gpt-3.5": {"input": 0.0005, "output": 0.0015},
    "local": {"input": 0.0, "output": 0.0},
    "default": {"input": 0.003, "output": 0.015},
}

# Complexity keywords that increase price
COMPLEXITY_KEYWORDS = {
    "high": ["complex", "advanced", "sophisticated", "comprehensive", "enterprise",
             "multi-step", "integration", "architecture", "security audit", "full stack"],
    "medium": ["analyze", "review", "optimize", "refactor", "migrate",
               "debug", "test", "deploy", "automate", "transform"],
    "low": ["simple", "basic", "quick", "small", "minor",
            "format", "rename", "list", "check", "count"],
}

# Urgency levels and their multipliers
URGENCY_MULTIPLIERS = {
    "critical": 2.5,
    "high": 1.8,
    "normal": 1.0,
    "low": 0.8,
    "batch": 0.6,
}


class TaskPricingSkill(Skill):
    """
    Autonomous dynamic pricing engine for agent services.

    Estimates costs, generates quotes, records actuals, and auto-calibrates
    pricing models from prediction accuracy data. Enables the agent to
    autonomously price any task and maintain profitability.
    """

    def __init__(self, credentials: Dict[str, str] = None):
        super().__init__(credentials)
        self._ensure_data()

    def _ensure_data(self):
        PRICING_FILE.parent.mkdir(parents=True, exist_ok=True)
        if not PRICING_FILE.exists():
            self._save(self._default_state())

    def _default_state(self) -> Dict:
        return {
            "quotes": [],
            "history": [],
            "skill_cost_overrides": {},
            "calibration": {
                "total_estimates": 0,
                "total_actuals": 0,
                "prediction_errors": [],
                "avg_error_pct": 0.0,
                "bias": 0.0,  # positive = overestimate, negative = underestimate
                "correction_factor": 1.0,
            },
            "config": {
                "default_margin_pct": 30.0,
                "min_price": 0.01,
                "max_price": 100.0,
                "demand_factor": 1.0,
                "model": "default",
                "auto_calibrate": True,
                "calibration_window": 50,
            },
            "revenue_summary": {
                "total_quoted": 0.0,
                "total_accepted": 0.0,
                "total_actual_cost": 0.0,
                "total_revenue": 0.0,
                "total_profit": 0.0,
                "quote_count": 0,
                "acceptance_rate": 0.0,
            },
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
        }

    def _load(self) -> Dict:
        try:
            with open(PRICING_FILE, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return self._default_state()

    def _save(self, data: Dict):
        # Trim old data
        if len(data.get("quotes", [])) > MAX_QUOTES:
            data["quotes"] = data["quotes"][-MAX_QUOTES:]
        if len(data.get("history", [])) > MAX_HISTORY:
            data["history"] = data["history"][-MAX_HISTORY:]
        if len(data.get("calibration", {}).get("prediction_errors", [])) > 200:
            data["calibration"]["prediction_errors"] = data["calibration"]["prediction_errors"][-200:]
        data["last_updated"] = datetime.now().isoformat()
        PRICING_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(PRICING_FILE, "w") as f:
            json.dump(data, f, indent=2, default=str)

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="task_pricing",
            name="Task Pricing",
            version="1.0.0",
            category="revenue",
            description="Dynamic pricing engine - estimate costs, generate quotes, record actuals, auto-calibrate pricing models",
            required_credentials=[],
            actions=[
                SkillAction(
                    name="estimate",
                    description="Estimate the cost and generate a price for a task based on description and required skills",
                    parameters={
                        "description": {"type": "string", "required": True, "description": "Task description"},
                        "skills_needed": {"type": "list", "required": False, "description": "List of skill IDs needed"},
                        "urgency": {"type": "string", "required": False, "description": "Urgency: critical, high, normal, low, batch"},
                        "estimated_tokens": {"type": "number", "required": False, "description": "Estimated LLM tokens needed"},
                    },
                    estimated_cost=0.0,
                    estimated_duration_seconds=1,
                    success_probability=0.99,
                ),
                SkillAction(
                    name="quote",
                    description="Generate a formal quote with line items, valid for a specified duration",
                    parameters={
                        "description": {"type": "string", "required": True, "description": "Task description"},
                        "skills_needed": {"type": "list", "required": False, "description": "List of skill IDs needed"},
                        "urgency": {"type": "string", "required": False, "description": "Urgency level"},
                        "estimated_tokens": {"type": "number", "required": False, "description": "Estimated tokens"},
                        "valid_hours": {"type": "number", "required": False, "description": "Quote validity in hours (default 24)"},
                        "customer_id": {"type": "string", "required": False, "description": "Customer identifier for tracking"},
                    },
                    estimated_cost=0.0,
                    estimated_duration_seconds=1,
                    success_probability=0.99,
                ),
                SkillAction(
                    name="record_actual",
                    description="Record actual cost after task execution for calibration",
                    parameters={
                        "quote_id": {"type": "string", "required": True, "description": "Quote ID to record against"},
                        "actual_cost": {"type": "number", "required": True, "description": "Actual cost incurred"},
                        "actual_tokens": {"type": "number", "required": False, "description": "Actual tokens used"},
                        "success": {"type": "boolean", "required": False, "description": "Whether task completed successfully"},
                        "revenue_collected": {"type": "number", "required": False, "description": "Revenue collected from customer"},
                    },
                    estimated_cost=0.0,
                    estimated_duration_seconds=1,
                    success_probability=0.99,
                ),
                SkillAction(
                    name="accept_quote",
                    description="Mark a quote as accepted by the customer",
                    parameters={
                        "quote_id": {"type": "string", "required": True, "description": "Quote ID to accept"},
                    },
                    estimated_cost=0.0,
                    estimated_duration_seconds=1,
                    success_probability=0.99,
                ),
                SkillAction(
                    name="pricing_report",
                    description="Get pricing accuracy report with calibration stats, revenue summary, and improvement suggestions",
                    parameters={
                        "last_n": {"type": "number", "required": False, "description": "Report on last N quotes (default all)"},
                    },
                    estimated_cost=0.0,
                    estimated_duration_seconds=1,
                    success_probability=0.99,
                ),
                SkillAction(
                    name="adjust_config",
                    description="Adjust pricing configuration (margin, min/max price, demand factor)",
                    parameters={
                        "margin_pct": {"type": "number", "required": False, "description": "Default margin percentage"},
                        "min_price": {"type": "number", "required": False, "description": "Minimum price floor"},
                        "max_price": {"type": "number", "required": False, "description": "Maximum price ceiling"},
                        "demand_factor": {"type": "number", "required": False, "description": "Demand multiplier (>1 = high demand)"},
                        "model": {"type": "string", "required": False, "description": "LLM model family for token cost calc"},
                        "auto_calibrate": {"type": "boolean", "required": False, "description": "Enable auto-calibration"},
                    },
                    estimated_cost=0.0,
                    estimated_duration_seconds=1,
                    success_probability=0.99,
                ),
                SkillAction(
                    name="set_skill_cost",
                    description="Override the default cost estimate for a specific skill",
                    parameters={
                        "skill_id": {"type": "string", "required": True, "description": "Skill ID to override"},
                        "cost": {"type": "number", "required": True, "description": "New cost estimate in USD"},
                    },
                    estimated_cost=0.0,
                    estimated_duration_seconds=1,
                    success_probability=0.99,
                ),
                SkillAction(
                    name="bulk_estimate",
                    description="Estimate costs for multiple tasks at once (batch pricing)",
                    parameters={
                        "tasks": {"type": "list", "required": True, "description": "List of task objects with description and optional skills_needed"},
                    },
                    estimated_cost=0.0,
                    estimated_duration_seconds=2,
                    success_probability=0.95,
                ),
            ],
        )

    async def execute(self, action: str, params: Dict[str, Any]) -> SkillResult:
        try:
            if action == "estimate":
                return self._estimate(params)
            elif action == "quote":
                return self._generate_quote(params)
            elif action == "record_actual":
                return self._record_actual(params)
            elif action == "accept_quote":
                return self._accept_quote(params)
            elif action == "pricing_report":
                return self._pricing_report(params)
            elif action == "adjust_config":
                return self._adjust_config(params)
            elif action == "set_skill_cost":
                return self._set_skill_cost(params)
            elif action == "bulk_estimate":
                return self._bulk_estimate(params)
            else:
                return SkillResult(success=False, message=f"Unknown action: {action}")
        except Exception as e:
            return SkillResult(success=False, message=f"TaskPricing error: {str(e)}")

    def _detect_complexity(self, description: str) -> str:
        """Detect task complexity from description text."""
        desc_lower = description.lower()

        high_score = sum(1 for kw in COMPLEXITY_KEYWORDS["high"] if kw in desc_lower)
        med_score = sum(1 for kw in COMPLEXITY_KEYWORDS["medium"] if kw in desc_lower)
        low_score = sum(1 for kw in COMPLEXITY_KEYWORDS["low"] if kw in desc_lower)

        # Also factor in description length
        word_count = len(description.split())
        if word_count > 100:
            high_score += 2
        elif word_count > 50:
            med_score += 1

        if high_score > med_score and high_score > low_score:
            return "high"
        elif med_score >= high_score and med_score > low_score:
            return "medium"
        return "low"

    def _complexity_multiplier(self, complexity: str) -> float:
        """Convert complexity level to price multiplier."""
        return {"high": 2.0, "medium": 1.3, "low": 1.0}.get(complexity, 1.0)

    def _get_skill_cost(self, skill_id: str) -> float:
        """Get cost estimate for a skill, checking overrides first."""
        data = self._load()
        overrides = data.get("skill_cost_overrides", {})
        if skill_id in overrides:
            return overrides[skill_id]
        return DEFAULT_SKILL_COSTS.get(skill_id, 0.01)

    def _estimate_token_cost(self, token_count: int, model: str = "default") -> float:
        """Estimate the LLM token cost."""
        costs = TOKEN_COSTS.get(model, TOKEN_COSTS["default"])
        # Assume 60/40 input/output split
        input_tokens = int(token_count * 0.6)
        output_tokens = int(token_count * 0.4)
        return (input_tokens * costs["input"] / 1000) + (output_tokens * costs["output"] / 1000)

    def _calculate_price(
        self,
        description: str,
        skills_needed: List[str],
        urgency: str = "normal",
        estimated_tokens: int = 0,
        model: str = "default",
    ) -> Dict:
        """Core pricing calculation. Returns detailed breakdown."""
        data = self._load()
        config = data.get("config", {})
        calibration = data.get("calibration", {})

        # 1. Base skill costs
        skill_costs = {}
        for skill_id in skills_needed:
            skill_costs[skill_id] = self._get_skill_cost(skill_id)
        total_skill_cost = sum(skill_costs.values())

        # 2. Token cost estimate
        if estimated_tokens <= 0:
            # Heuristic: ~500 tokens per skill call + 200 base
            estimated_tokens = 200 + len(skills_needed) * 500
        token_cost = self._estimate_token_cost(estimated_tokens, model)

        # 3. Complexity analysis
        complexity = self._detect_complexity(description)
        cmx = self._complexity_multiplier(complexity)

        # 4. Urgency multiplier
        umx = URGENCY_MULTIPLIERS.get(urgency, 1.0)

        # 5. Base cost
        base_cost = (total_skill_cost + token_cost) * cmx * umx

        # 6. Apply calibration correction
        correction = calibration.get("correction_factor", 1.0)
        calibrated_cost = base_cost * correction

        # 7. Apply margin
        margin_pct = config.get("default_margin_pct", 30.0)
        margin_amount = calibrated_cost * (margin_pct / 100.0)

        # 8. Apply demand factor
        demand = config.get("demand_factor", 1.0)

        # 9. Final price
        raw_price = (calibrated_cost + margin_amount) * demand
        min_price = config.get("min_price", 0.01)
        max_price = config.get("max_price", 100.0)
        final_price = max(min_price, min(max_price, raw_price))

        return {
            "estimated_cost": round(calibrated_cost, 6),
            "price": round(final_price, 4),
            "breakdown": {
                "skill_costs": {k: round(v, 6) for k, v in skill_costs.items()},
                "total_skill_cost": round(total_skill_cost, 6),
                "token_cost": round(token_cost, 6),
                "estimated_tokens": estimated_tokens,
                "complexity": complexity,
                "complexity_multiplier": cmx,
                "urgency": urgency,
                "urgency_multiplier": umx,
                "base_cost": round(base_cost, 6),
                "calibration_correction": round(correction, 4),
                "calibrated_cost": round(calibrated_cost, 6),
                "margin_pct": margin_pct,
                "margin_amount": round(margin_amount, 6),
                "demand_factor": demand,
                "min_price": min_price,
                "max_price": max_price,
            },
        }

    def _estimate(self, params: Dict) -> SkillResult:
        """Estimate task cost and price."""
        description = params.get("description", "")
        if not description:
            return SkillResult(success=False, message="description is required")

        skills_needed = params.get("skills_needed", [])
        if isinstance(skills_needed, str):
            skills_needed = [s.strip() for s in skills_needed.split(",") if s.strip()]

        urgency = params.get("urgency", "normal")
        estimated_tokens = params.get("estimated_tokens", 0)

        data = self._load()
        model = data.get("config", {}).get("model", "default")

        result = self._calculate_price(
            description, skills_needed, urgency, estimated_tokens, model
        )

        return SkillResult(
            success=True,
            message=f"Estimated price: ${result['price']:.4f} (cost: ${result['estimated_cost']:.4f}, complexity: {result['breakdown']['complexity']}, urgency: {urgency})",
            data=result,
        )

    def _generate_quote(self, params: Dict) -> SkillResult:
        """Generate a formal quote with ID and expiration."""
        description = params.get("description", "")
        if not description:
            return SkillResult(success=False, message="description is required")

        skills_needed = params.get("skills_needed", [])
        if isinstance(skills_needed, str):
            skills_needed = [s.strip() for s in skills_needed.split(",") if s.strip()]

        urgency = params.get("urgency", "normal")
        estimated_tokens = params.get("estimated_tokens", 0)
        valid_hours = params.get("valid_hours", 24)
        customer_id = params.get("customer_id", "anonymous")

        data = self._load()
        model = data.get("config", {}).get("model", "default")

        pricing = self._calculate_price(
            description, skills_needed, urgency, estimated_tokens, model
        )

        quote_id = f"QT-{uuid.uuid4().hex[:8].upper()}"
        now = datetime.now()
        expires = now + timedelta(hours=valid_hours)

        quote = {
            "quote_id": quote_id,
            "customer_id": customer_id,
            "description": description,
            "skills_needed": skills_needed,
            "urgency": urgency,
            "estimated_cost": pricing["estimated_cost"],
            "price": pricing["price"],
            "breakdown": pricing["breakdown"],
            "status": "pending",  # pending, accepted, rejected, expired, completed
            "created_at": now.isoformat(),
            "expires_at": expires.isoformat(),
            "valid_hours": valid_hours,
            "actual_cost": None,
            "actual_tokens": None,
            "revenue_collected": None,
        }

        data["quotes"].append(quote)
        summary = data.get("revenue_summary", {})
        summary["quote_count"] = summary.get("quote_count", 0) + 1
        summary["total_quoted"] = summary.get("total_quoted", 0.0) + pricing["price"]
        data["revenue_summary"] = summary
        self._save(data)

        return SkillResult(
            success=True,
            message=f"Quote {quote_id} generated: ${pricing['price']:.4f} (valid {valid_hours}h, expires {expires.strftime('%Y-%m-%d %H:%M')})",
            data=quote,
        )

    def _accept_quote(self, params: Dict) -> SkillResult:
        """Mark a quote as accepted."""
        quote_id = params.get("quote_id", "")
        if not quote_id:
            return SkillResult(success=False, message="quote_id is required")

        data = self._load()
        for q in data.get("quotes", []):
            if q["quote_id"] == quote_id:
                if q["status"] != "pending":
                    return SkillResult(success=False, message=f"Quote {quote_id} is {q['status']}, cannot accept")

                # Check expiration
                expires = datetime.fromisoformat(q["expires_at"])
                if datetime.now() > expires:
                    q["status"] = "expired"
                    self._save(data)
                    return SkillResult(success=False, message=f"Quote {quote_id} has expired")

                q["status"] = "accepted"
                q["accepted_at"] = datetime.now().isoformat()
                summary = data.get("revenue_summary", {})
                summary["total_accepted"] = summary.get("total_accepted", 0.0) + q["price"]
                data["revenue_summary"] = summary
                self._save(data)

                return SkillResult(
                    success=True,
                    message=f"Quote {quote_id} accepted at ${q['price']:.4f}",
                    data=q,
                )

        return SkillResult(success=False, message=f"Quote {quote_id} not found")

    def _record_actual(self, params: Dict) -> SkillResult:
        """Record actual execution cost for a quote and calibrate."""
        quote_id = params.get("quote_id", "")
        actual_cost = params.get("actual_cost")
        if not quote_id or actual_cost is None:
            return SkillResult(success=False, message="quote_id and actual_cost are required")

        actual_cost = float(actual_cost)
        actual_tokens = params.get("actual_tokens", 0)
        success = params.get("success", True)
        revenue_collected = params.get("revenue_collected", 0.0)

        data = self._load()
        quote = None
        for q in data.get("quotes", []):
            if q["quote_id"] == quote_id:
                quote = q
                break

        if not quote:
            return SkillResult(success=False, message=f"Quote {quote_id} not found")

        # Record actuals
        quote["actual_cost"] = actual_cost
        quote["actual_tokens"] = actual_tokens
        quote["success"] = success
        quote["revenue_collected"] = revenue_collected
        quote["completed_at"] = datetime.now().isoformat()
        quote["status"] = "completed"

        # Calculate prediction error
        estimated = quote.get("estimated_cost", 0)
        if estimated > 0:
            error_pct = ((actual_cost - estimated) / estimated) * 100
        else:
            error_pct = 0.0

        profit = revenue_collected - actual_cost

        # Add to history
        history_entry = {
            "quote_id": quote_id,
            "estimated_cost": estimated,
            "actual_cost": actual_cost,
            "price": quote.get("price", 0),
            "revenue_collected": revenue_collected,
            "profit": profit,
            "error_pct": round(error_pct, 2),
            "complexity": quote.get("breakdown", {}).get("complexity", "unknown"),
            "skills_used": quote.get("skills_needed", []),
            "success": success,
            "timestamp": datetime.now().isoformat(),
        }
        data["history"].append(history_entry)

        # Update revenue summary
        summary = data.get("revenue_summary", {})
        summary["total_actual_cost"] = summary.get("total_actual_cost", 0.0) + actual_cost
        summary["total_revenue"] = summary.get("total_revenue", 0.0) + revenue_collected
        summary["total_profit"] = summary.get("total_profit", 0.0) + profit
        data["revenue_summary"] = summary

        # Auto-calibrate if enabled
        calibration = data.get("calibration", {})
        calibration["total_estimates"] = calibration.get("total_estimates", 0) + 1
        calibration["total_actuals"] = calibration.get("total_actuals", 0) + 1
        calibration["prediction_errors"].append(error_pct)

        config = data.get("config", {})
        if config.get("auto_calibrate", True):
            window = config.get("calibration_window", 50)
            recent_errors = calibration["prediction_errors"][-window:]
            if len(recent_errors) >= 5:
                avg_error = sum(recent_errors) / len(recent_errors)
                calibration["avg_error_pct"] = round(avg_error, 2)
                calibration["bias"] = round(avg_error, 2)

                # Adjust correction factor: if we consistently underestimate (negative bias),
                # increase the correction factor; if overestimate, decrease it
                # Use gentle adjustment (10% of error) to avoid oscillation
                adjustment = 1.0 + (avg_error / 100.0) * 0.1
                old_factor = calibration.get("correction_factor", 1.0)
                new_factor = old_factor * adjustment
                # Clamp to reasonable range
                calibration["correction_factor"] = round(max(0.5, min(3.0, new_factor)), 4)

        data["calibration"] = calibration
        self._save(data)

        return SkillResult(
            success=True,
            message=(
                f"Recorded actuals for {quote_id}: cost=${actual_cost:.4f} vs estimated=${estimated:.4f} "
                f"(error={error_pct:+.1f}%), profit=${profit:.4f}"
            ),
            data={
                "quote_id": quote_id,
                "estimated_cost": estimated,
                "actual_cost": actual_cost,
                "error_pct": round(error_pct, 2),
                "profit": round(profit, 4),
                "revenue_collected": revenue_collected,
                "correction_factor": calibration.get("correction_factor", 1.0),
            },
        )

    def _pricing_report(self, params: Dict) -> SkillResult:
        """Generate pricing accuracy and revenue report."""
        data = self._load()
        last_n = params.get("last_n", 0)

        history = data.get("history", [])
        if last_n and last_n > 0:
            history = history[-last_n:]

        if not history:
            return SkillResult(
                success=True,
                message="No pricing history yet. Generate quotes and record actuals to build data.",
                data={
                    "entries": 0,
                    "calibration": data.get("calibration", {}),
                    "config": data.get("config", {}),
                    "revenue_summary": data.get("revenue_summary", {}),
                },
            )

        # Compute stats
        errors = [h["error_pct"] for h in history if "error_pct" in h]
        profits = [h.get("profit", 0) for h in history]
        revenues = [h.get("revenue_collected", 0) for h in history]
        costs = [h.get("actual_cost", 0) for h in history]
        successes = [h for h in history if h.get("success", True)]

        avg_error = sum(errors) / len(errors) if errors else 0
        median_error = sorted(errors)[len(errors) // 2] if errors else 0
        total_profit = sum(profits)
        total_revenue = sum(revenues)
        total_cost = sum(costs)
        margin_pct = ((total_revenue - total_cost) / total_revenue * 100) if total_revenue > 0 else 0
        success_rate = len(successes) / len(history) * 100 if history else 0

        # Accuracy buckets
        within_10 = sum(1 for e in errors if abs(e) <= 10)
        within_25 = sum(1 for e in errors if abs(e) <= 25)
        within_50 = sum(1 for e in errors if abs(e) <= 50)

        # Complexity breakdown
        by_complexity = {}
        for h in history:
            c = h.get("complexity", "unknown")
            if c not in by_complexity:
                by_complexity[c] = {"count": 0, "total_profit": 0, "avg_error": []}
            by_complexity[c]["count"] += 1
            by_complexity[c]["total_profit"] += h.get("profit", 0)
            if "error_pct" in h:
                by_complexity[c]["avg_error"].append(h["error_pct"])

        for c in by_complexity:
            errs = by_complexity[c]["avg_error"]
            by_complexity[c]["avg_error"] = round(sum(errs) / len(errs), 2) if errs else 0
            by_complexity[c]["total_profit"] = round(by_complexity[c]["total_profit"], 4)

        # Suggestions
        suggestions = []
        if avg_error > 20:
            suggestions.append("Pricing estimates are too low on average. Consider increasing correction_factor or margins.")
        elif avg_error < -20:
            suggestions.append("Pricing estimates are too high on average. Consider decreasing margins to improve acceptance rate.")
        if margin_pct < 10 and total_revenue > 0:
            suggestions.append(f"Margin is only {margin_pct:.1f}%. Consider raising default_margin_pct to maintain profitability.")
        if within_10 / len(errors) * 100 < 50 if errors else True:
            suggestions.append("Less than 50% of estimates are within 10% of actual. More calibration data needed.")

        report = {
            "entries_analyzed": len(history),
            "accuracy": {
                "avg_error_pct": round(avg_error, 2),
                "median_error_pct": round(median_error, 2),
                "within_10pct": within_10,
                "within_25pct": within_25,
                "within_50pct": within_50,
                "total": len(errors),
            },
            "financial": {
                "total_revenue": round(total_revenue, 4),
                "total_cost": round(total_cost, 4),
                "total_profit": round(total_profit, 4),
                "margin_pct": round(margin_pct, 2),
                "success_rate": round(success_rate, 1),
            },
            "by_complexity": by_complexity,
            "calibration": data.get("calibration", {}),
            "revenue_summary": data.get("revenue_summary", {}),
            "suggestions": suggestions,
        }

        return SkillResult(
            success=True,
            message=(
                f"Pricing report: {len(history)} entries, avg error {avg_error:+.1f}%, "
                f"margin {margin_pct:.1f}%, total profit ${total_profit:.4f}"
            ),
            data=report,
        )

    def _adjust_config(self, params: Dict) -> SkillResult:
        """Adjust pricing configuration."""
        data = self._load()
        config = data.get("config", {})
        changes = []

        if "margin_pct" in params:
            old = config.get("default_margin_pct", 30.0)
            config["default_margin_pct"] = float(params["margin_pct"])
            changes.append(f"margin: {old}% -> {params['margin_pct']}%")

        if "min_price" in params:
            config["min_price"] = float(params["min_price"])
            changes.append(f"min_price: ${params['min_price']}")

        if "max_price" in params:
            config["max_price"] = float(params["max_price"])
            changes.append(f"max_price: ${params['max_price']}")

        if "demand_factor" in params:
            old = config.get("demand_factor", 1.0)
            config["demand_factor"] = float(params["demand_factor"])
            changes.append(f"demand: {old} -> {params['demand_factor']}")

        if "model" in params:
            config["model"] = params["model"]
            changes.append(f"model: {params['model']}")

        if "auto_calibrate" in params:
            config["auto_calibrate"] = bool(params["auto_calibrate"])
            changes.append(f"auto_calibrate: {params['auto_calibrate']}")

        if not changes:
            return SkillResult(success=False, message="No configuration changes provided")

        data["config"] = config
        self._save(data)

        return SkillResult(
            success=True,
            message=f"Pricing config updated: {', '.join(changes)}",
            data=config,
        )

    def _set_skill_cost(self, params: Dict) -> SkillResult:
        """Override the default cost estimate for a specific skill."""
        skill_id = params.get("skill_id", "")
        cost = params.get("cost")
        if not skill_id or cost is None:
            return SkillResult(success=False, message="skill_id and cost are required")

        data = self._load()
        overrides = data.get("skill_cost_overrides", {})
        old_cost = overrides.get(skill_id, DEFAULT_SKILL_COSTS.get(skill_id, "not set"))
        overrides[skill_id] = float(cost)
        data["skill_cost_overrides"] = overrides
        self._save(data)

        return SkillResult(
            success=True,
            message=f"Skill '{skill_id}' cost override: {old_cost} -> ${cost}",
            data={"skill_id": skill_id, "old_cost": old_cost, "new_cost": cost},
        )

    def _bulk_estimate(self, params: Dict) -> SkillResult:
        """Estimate prices for multiple tasks at once."""
        tasks = params.get("tasks", [])
        if not tasks:
            return SkillResult(success=False, message="tasks list is required")

        data = self._load()
        model = data.get("config", {}).get("model", "default")

        estimates = []
        total_price = 0.0
        total_cost = 0.0

        for task in tasks:
            if isinstance(task, str):
                task = {"description": task}
            desc = task.get("description", "")
            skills = task.get("skills_needed", [])
            if isinstance(skills, str):
                skills = [s.strip() for s in skills.split(",") if s.strip()]
            urgency = task.get("urgency", "normal")
            tokens = task.get("estimated_tokens", 0)

            pricing = self._calculate_price(desc, skills, urgency, tokens, model)
            estimates.append({
                "description": desc[:100],
                "price": pricing["price"],
                "estimated_cost": pricing["estimated_cost"],
                "complexity": pricing["breakdown"]["complexity"],
            })
            total_price += pricing["price"]
            total_cost += pricing["estimated_cost"]

        # Apply batch discount (5% for 3+, 10% for 10+)
        batch_discount = 0.0
        if len(estimates) >= 10:
            batch_discount = 0.10
        elif len(estimates) >= 3:
            batch_discount = 0.05

        discounted_total = total_price * (1.0 - batch_discount)

        return SkillResult(
            success=True,
            message=f"Bulk estimate for {len(estimates)} tasks: ${discounted_total:.4f} (discount: {batch_discount*100:.0f}%)",
            data={
                "estimates": estimates,
                "total_price": round(total_price, 4),
                "total_estimated_cost": round(total_cost, 4),
                "batch_discount_pct": batch_discount * 100,
                "discounted_total": round(discounted_total, 4),
                "task_count": len(estimates),
            },
        )
