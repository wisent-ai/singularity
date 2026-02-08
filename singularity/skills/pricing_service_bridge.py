#!/usr/bin/env python3
"""
PricingServiceBridgeSkill - Auto-price tasks submitted via ServiceAPI.

TaskPricingSkill can price work and ServiceAPI can accept tasks, but they
operate in isolation. This bridge connects them so that:

1. Tasks submitted via API are AUTO-QUOTED before execution
2. Actual costs are recorded after execution for pricing calibration
3. Quote-gated mode: tasks only execute after quote acceptance
4. Revenue tracking: bridge aggregates API task revenue in real-time
5. Events emitted: pricing.quoted, pricing.accepted, pricing.completed

Revenue flow:
  Customer → ServiceAPI → Bridge auto-quotes → Customer accepts →
  Task executes → Bridge records actual → Pricing model calibrates

Without this bridge, TaskPricingSkill and ServiceAPI are disconnected.
With it, the agent has end-to-end automated revenue generation.

Pillar: Revenue Generation (primary), Self-Improvement (pricing calibration)
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import Skill, SkillResult, SkillManifest, SkillAction

BRIDGE_FILE = Path(__file__).parent.parent / "data" / "pricing_service_bridge.json"
MAX_RECORDS = 2000


class PricingServiceBridgeSkill(Skill):
    """Bridge between TaskPricingSkill and ServiceAPI for automated revenue."""

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
                "auto_quote": True,
                "quote_gated": False,
                "default_urgency": "normal",
                "auto_record_actuals": True,
                "emit_events": True,
                "min_price_threshold": 0.0,
            },
            "task_quotes": {},
            "revenue_log": [],
            "stats": {
                "tasks_quoted": 0,
                "tasks_executed": 0,
                "tasks_gated": 0,
                "total_quoted_usd": 0.0,
                "total_actual_usd": 0.0,
                "total_revenue_usd": 0.0,
                "total_profit_usd": 0.0,
                "avg_margin_pct": 0.0,
            },
        }

    def _load(self) -> Dict:
        try:
            return json.loads(BRIDGE_FILE.read_text())
        except Exception:
            return self._default_state()

    def _save(self, data: Dict):
        # Trim records
        if len(data.get("revenue_log", [])) > MAX_RECORDS:
            data["revenue_log"] = data["revenue_log"][-MAX_RECORDS:]
        if len(data.get("task_quotes", {})) > MAX_RECORDS:
            keys = sorted(data["task_quotes"].keys())
            excess = len(keys) - MAX_RECORDS
            for k in keys[:excess]:
                del data["task_quotes"][k]
        BRIDGE_FILE.write_text(json.dumps(data, indent=2, default=str))

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="pricing_service_bridge",
            name="Pricing-ServiceAPI Bridge",
            version="1.0.0",
            category="revenue",
            description="Auto-price API tasks, gate execution on accepted quotes, track revenue end-to-end",
            actions=[
                SkillAction(
                    name="quote_task",
                    description="Generate a quote for a task before execution",
                    parameters={
                        "task_id": {"type": "string", "required": True, "description": "ServiceAPI task ID"},
                        "skill_id": {"type": "string", "required": True, "description": "Skill to execute"},
                        "action": {"type": "string", "required": True, "description": "Action to execute"},
                        "description": {"type": "string", "required": False, "description": "Task description for pricing"},
                        "urgency": {"type": "string", "required": False, "description": "normal|high|critical|batch"},
                        "customer_id": {"type": "string", "required": False, "description": "Customer identifier"},
                    },
                ),
                SkillAction(
                    name="accept_task_quote",
                    description="Accept a task quote and allow execution",
                    parameters={
                        "task_id": {"type": "string", "required": True, "description": "ServiceAPI task ID"},
                    },
                ),
                SkillAction(
                    name="record_completion",
                    description="Record task completion and actual cost for calibration",
                    parameters={
                        "task_id": {"type": "string", "required": True, "description": "ServiceAPI task ID"},
                        "actual_cost": {"type": "number", "required": False, "description": "Actual cost incurred"},
                        "actual_tokens": {"type": "integer", "required": False, "description": "Tokens used"},
                        "execution_time_ms": {"type": "number", "required": False, "description": "Execution time"},
                    },
                ),
                SkillAction(
                    name="revenue_dashboard",
                    description="Aggregated revenue dashboard across API tasks",
                    parameters={
                        "time_range_hours": {"type": "number", "required": False, "description": "Hours to look back"},
                    },
                ),
                SkillAction(
                    name="task_quote_status",
                    description="Check quote status for a specific task",
                    parameters={
                        "task_id": {"type": "string", "required": True, "description": "ServiceAPI task ID"},
                    },
                ),
                SkillAction(
                    name="configure",
                    description="Update bridge configuration",
                    parameters={
                        "auto_quote": {"type": "boolean", "required": False, "description": "Auto-quote incoming tasks"},
                        "quote_gated": {"type": "boolean", "required": False, "description": "Block execution until quote accepted"},
                        "default_urgency": {"type": "string", "required": False, "description": "Default urgency for unspecified tasks"},
                        "auto_record_actuals": {"type": "boolean", "required": False, "description": "Auto-record actual costs on completion"},
                        "min_price_threshold": {"type": "number", "required": False, "description": "Min price below which tasks run free"},
                    },
                ),
                SkillAction(
                    name="pending_quotes",
                    description="List all tasks with pending (unaccepted) quotes",
                    parameters={},
                ),
                SkillAction(
                    name="status",
                    description="Bridge status and health",
                    parameters={},
                ),
            ],
            required_credentials=[],
        )

    async def execute(self, action: str, params: Dict[str, Any]) -> SkillResult:
        actions = {
            "quote_task": self._quote_task,
            "accept_task_quote": self._accept_task_quote,
            "record_completion": self._record_completion,
            "revenue_dashboard": self._revenue_dashboard,
            "task_quote_status": self._task_quote_status,
            "configure": self._configure,
            "pending_quotes": self._pending_quotes,
            "status": self._status,
        }
        handler = actions.get(action)
        if not handler:
            return SkillResult(success=False, message=f"Unknown action: {action}")
        return handler(params)

    # ── Hook methods for ServiceAPI integration ───────────────────────

    def hook_pre_execute(self, task_id: str, skill_id: str, action: str,
                         params: Dict, customer_id: str = "api_user") -> Dict:
        """Called by ServiceAPI BEFORE task execution.

        Returns: {"allow": bool, "quote": dict|None, "reason": str}
        Used by ServiceAPI to gate execution on accepted quotes.
        """
        data = self._load()
        config = data.get("config", {})

        if not config.get("auto_quote", True):
            return {"allow": True, "quote": None, "reason": "auto_quote disabled"}

        # Generate description from params
        description = params.get("description", f"{skill_id}.{action}")

        # Generate quote
        quote_result = self._quote_task({
            "task_id": task_id,
            "skill_id": skill_id,
            "action": action,
            "description": description,
            "urgency": params.get("urgency", config.get("default_urgency", "normal")),
            "customer_id": customer_id,
        })

        quote = quote_result.data if quote_result.success else None

        # Check if gated
        if config.get("quote_gated", False) and quote:
            if quote.get("status") != "accepted":
                return {
                    "allow": False,
                    "quote": quote,
                    "reason": f"Quote {quote.get('quote_id', '?')} pending acceptance",
                }

        return {"allow": True, "quote": quote, "reason": "ok"}

    def hook_post_execute(self, task_id: str, success: bool,
                          execution_time_ms: float = 0,
                          result_data: Dict = None) -> Optional[Dict]:
        """Called by ServiceAPI AFTER task execution.

        Records actual cost and updates revenue tracking.
        """
        data = self._load()
        config = data.get("config", {})

        if not config.get("auto_record_actuals", True):
            return None

        task_quote = data.get("task_quotes", {}).get(task_id)
        if not task_quote:
            return None

        return self._record_completion({
            "task_id": task_id,
            "execution_time_ms": execution_time_ms,
        }).data

    # ── Core actions ──────────────────────────────────────────────────

    def _quote_task(self, params: Dict) -> SkillResult:
        """Generate a quote for an API task."""
        task_id = params.get("task_id", "")
        skill_id = params.get("skill_id", "")
        action = params.get("action", "")
        description = params.get("description", f"{skill_id}.{action}")
        urgency = params.get("urgency", "normal")
        customer_id = params.get("customer_id", "api_user")

        if not task_id:
            return SkillResult(success=False, message="task_id is required")
        if not skill_id:
            return SkillResult(success=False, message="skill_id is required")

        data = self._load()

        # Check if already quoted
        existing = data.get("task_quotes", {}).get(task_id)
        if existing:
            return SkillResult(
                success=True,
                message=f"Task {task_id} already quoted: ${existing['price']:.4f}",
                data=existing,
            )

        # Use pricing skill if available via context, else estimate locally
        pricing = self._estimate_price(skill_id, action, description, urgency)

        quote_record = {
            "task_id": task_id,
            "quote_id": f"QT-{task_id[:8].upper()}",
            "skill_id": skill_id,
            "action": action,
            "description": description,
            "urgency": urgency,
            "customer_id": customer_id,
            "estimated_cost": pricing["estimated_cost"],
            "price": pricing["price"],
            "breakdown": pricing["breakdown"],
            "status": "pending",
            "quoted_at": datetime.now().isoformat(),
            "accepted_at": None,
            "completed_at": None,
            "actual_cost": None,
            "execution_time_ms": None,
            "revenue": None,
            "profit": None,
        }

        data.setdefault("task_quotes", {})[task_id] = quote_record
        stats = data.get("stats", {})
        stats["tasks_quoted"] = stats.get("tasks_quoted", 0) + 1
        stats["total_quoted_usd"] = stats.get("total_quoted_usd", 0.0) + pricing["price"]
        data["stats"] = stats
        self._save(data)

        # Emit event if possible
        self._emit_event("pricing.quoted", {
            "task_id": task_id, "quote_id": quote_record["quote_id"],
            "price": pricing["price"], "skill_id": skill_id,
        })

        return SkillResult(
            success=True,
            message=f"Task {task_id} quoted: ${pricing['price']:.4f} ({urgency})",
            data=quote_record,
        )

    def _accept_task_quote(self, params: Dict) -> SkillResult:
        """Accept a task's quote to allow execution."""
        task_id = params.get("task_id", "")
        if not task_id:
            return SkillResult(success=False, message="task_id is required")

        data = self._load()
        quote = data.get("task_quotes", {}).get(task_id)
        if not quote:
            return SkillResult(success=False, message=f"No quote found for task {task_id}")

        if quote["status"] == "accepted":
            return SkillResult(success=True, message=f"Quote already accepted", data=quote)

        if quote["status"] not in ("pending",):
            return SkillResult(success=False, message=f"Quote is {quote['status']}, cannot accept")

        quote["status"] = "accepted"
        quote["accepted_at"] = datetime.now().isoformat()
        data["task_quotes"][task_id] = quote
        self._save(data)

        self._emit_event("pricing.accepted", {
            "task_id": task_id, "quote_id": quote["quote_id"],
            "price": quote["price"],
        })

        return SkillResult(
            success=True,
            message=f"Quote {quote['quote_id']} accepted at ${quote['price']:.4f}",
            data=quote,
        )

    def _record_completion(self, params: Dict) -> SkillResult:
        """Record task completion and calculate revenue."""
        task_id = params.get("task_id", "")
        if not task_id:
            return SkillResult(success=False, message="task_id is required")

        data = self._load()
        quote = data.get("task_quotes", {}).get(task_id)
        if not quote:
            return SkillResult(success=False, message=f"No quote found for task {task_id}")

        actual_cost = params.get("actual_cost")
        execution_time_ms = params.get("execution_time_ms", 0)
        actual_tokens = params.get("actual_tokens", 0)

        # If no actual cost provided, estimate from execution time
        if actual_cost is None:
            # Rough estimate: $0.003/sec execution cost (LLM + compute)
            actual_cost = (execution_time_ms / 1000.0) * 0.003
            actual_cost = max(actual_cost, 0.0001)

        revenue = quote["price"]
        profit = revenue - actual_cost
        margin_pct = (profit / revenue * 100) if revenue > 0 else 0.0

        quote["status"] = "completed"
        quote["completed_at"] = datetime.now().isoformat()
        quote["actual_cost"] = round(actual_cost, 6)
        quote["execution_time_ms"] = execution_time_ms
        quote["actual_tokens"] = actual_tokens
        quote["revenue"] = round(revenue, 6)
        quote["profit"] = round(profit, 6)
        quote["margin_pct"] = round(margin_pct, 2)

        data["task_quotes"][task_id] = quote

        # Update stats
        stats = data.get("stats", {})
        stats["tasks_executed"] = stats.get("tasks_executed", 0) + 1
        stats["total_actual_usd"] = stats.get("total_actual_usd", 0.0) + actual_cost
        stats["total_revenue_usd"] = stats.get("total_revenue_usd", 0.0) + revenue
        stats["total_profit_usd"] = stats.get("total_profit_usd", 0.0) + profit

        # Recalculate average margin
        completed = [q for q in data.get("task_quotes", {}).values()
                     if q.get("status") == "completed" and q.get("margin_pct") is not None]
        if completed:
            stats["avg_margin_pct"] = round(
                sum(q["margin_pct"] for q in completed) / len(completed), 2
            )
        data["stats"] = stats

        # Add to revenue log
        log_entry = {
            "task_id": task_id,
            "quote_id": quote["quote_id"],
            "skill_id": quote.get("skill_id", ""),
            "action": quote.get("action", ""),
            "price": revenue,
            "actual_cost": actual_cost,
            "profit": profit,
            "margin_pct": margin_pct,
            "completed_at": quote["completed_at"],
        }
        data.setdefault("revenue_log", []).append(log_entry)
        self._save(data)

        # Emit event
        self._emit_event("pricing.completed", {
            "task_id": task_id, "revenue": revenue,
            "profit": profit, "margin_pct": margin_pct,
        })

        # Try to record in TaskPricingSkill for calibration
        self._forward_to_pricing_skill(quote, actual_cost, actual_tokens)

        return SkillResult(
            success=True,
            message=f"Task {task_id} completed: revenue=${revenue:.4f}, cost=${actual_cost:.4f}, profit=${profit:.4f} ({margin_pct:.1f}%)",
            data=quote,
        )

    def _revenue_dashboard(self, params: Dict) -> SkillResult:
        """Aggregated revenue dashboard."""
        data = self._load()
        stats = data.get("stats", {})
        time_range_hours = params.get("time_range_hours", 24)

        # Filter revenue log by time range
        cutoff = datetime.now().timestamp() - (time_range_hours * 3600)
        recent_log = []
        for entry in data.get("revenue_log", []):
            try:
                entry_time = datetime.fromisoformat(entry.get("completed_at", "")).timestamp()
                if entry_time >= cutoff:
                    recent_log.append(entry)
            except (ValueError, TypeError):
                pass

        # Per-skill breakdown
        skill_breakdown = {}
        for entry in recent_log:
            sid = entry.get("skill_id", "unknown")
            if sid not in skill_breakdown:
                skill_breakdown[sid] = {
                    "tasks": 0, "revenue": 0.0, "cost": 0.0, "profit": 0.0,
                }
            skill_breakdown[sid]["tasks"] += 1
            skill_breakdown[sid]["revenue"] += entry.get("price", 0)
            skill_breakdown[sid]["cost"] += entry.get("actual_cost", 0)
            skill_breakdown[sid]["profit"] += entry.get("profit", 0)

        # Round values
        for sid in skill_breakdown:
            for k in ("revenue", "cost", "profit"):
                skill_breakdown[sid][k] = round(skill_breakdown[sid][k], 4)

        # Pending quotes
        pending = [q for q in data.get("task_quotes", {}).values()
                   if q.get("status") == "pending"]

        dashboard = {
            "time_range_hours": time_range_hours,
            "period_summary": {
                "tasks_completed": len(recent_log),
                "revenue": round(sum(e.get("price", 0) for e in recent_log), 4),
                "cost": round(sum(e.get("actual_cost", 0) for e in recent_log), 4),
                "profit": round(sum(e.get("profit", 0) for e in recent_log), 4),
            },
            "all_time": {
                "tasks_quoted": stats.get("tasks_quoted", 0),
                "tasks_executed": stats.get("tasks_executed", 0),
                "total_quoted_usd": round(stats.get("total_quoted_usd", 0), 4),
                "total_revenue_usd": round(stats.get("total_revenue_usd", 0), 4),
                "total_actual_usd": round(stats.get("total_actual_usd", 0), 4),
                "total_profit_usd": round(stats.get("total_profit_usd", 0), 4),
                "avg_margin_pct": stats.get("avg_margin_pct", 0),
            },
            "skill_breakdown": skill_breakdown,
            "pending_quotes": len(pending),
            "conversion_rate": round(
                stats.get("tasks_executed", 0) / max(stats.get("tasks_quoted", 0), 1) * 100, 1
            ),
        }

        return SkillResult(
            success=True,
            message=f"Revenue dashboard ({time_range_hours}h): {len(recent_log)} tasks, ${dashboard['period_summary']['revenue']:.4f} revenue, ${dashboard['period_summary']['profit']:.4f} profit",
            data=dashboard,
        )

    def _task_quote_status(self, params: Dict) -> SkillResult:
        """Check quote status for a specific task."""
        task_id = params.get("task_id", "")
        if not task_id:
            return SkillResult(success=False, message="task_id is required")

        data = self._load()
        quote = data.get("task_quotes", {}).get(task_id)
        if not quote:
            return SkillResult(success=False, message=f"No quote for task {task_id}")

        return SkillResult(
            success=True,
            message=f"Task {task_id}: status={quote['status']}, price=${quote['price']:.4f}",
            data=quote,
        )

    def _configure(self, params: Dict) -> SkillResult:
        """Update bridge configuration."""
        data = self._load()
        config = data.get("config", {})
        changed = []

        for key in ("auto_quote", "quote_gated", "default_urgency",
                     "auto_record_actuals", "emit_events", "min_price_threshold"):
            if key in params:
                old = config.get(key)
                config[key] = params[key]
                changed.append(f"{key}: {old} → {params[key]}")

        data["config"] = config
        self._save(data)

        return SkillResult(
            success=True,
            message=f"Config updated: {', '.join(changed) if changed else 'no changes'}",
            data=config,
        )

    def _pending_quotes(self, params: Dict) -> SkillResult:
        """List tasks with pending quotes."""
        data = self._load()
        pending = [
            q for q in data.get("task_quotes", {}).values()
            if q.get("status") == "pending"
        ]
        pending.sort(key=lambda q: q.get("quoted_at", ""), reverse=True)

        return SkillResult(
            success=True,
            message=f"{len(pending)} pending quotes",
            data={"pending": pending, "count": len(pending)},
        )

    def _status(self, params: Dict) -> SkillResult:
        """Bridge status and health."""
        data = self._load()
        config = data.get("config", {})
        stats = data.get("stats", {})

        # Count by status
        status_counts = {}
        for q in data.get("task_quotes", {}).values():
            s = q.get("status", "unknown")
            status_counts[s] = status_counts.get(s, 0) + 1

        return SkillResult(
            success=True,
            message="Pricing-ServiceAPI bridge active",
            data={
                "config": config,
                "stats": stats,
                "quote_status_counts": status_counts,
                "revenue_log_entries": len(data.get("revenue_log", [])),
            },
        )

    # ── Internal helpers ──────────────────────────────────────────────

    def _estimate_price(self, skill_id: str, action: str,
                        description: str, urgency: str) -> Dict:
        """Estimate price using TaskPricingSkill if available, else local fallback."""
        # Try using TaskPricingSkill via context
        if hasattr(self, '_context') and self._context:
            pricing_skill = self._context.get_skill("task_pricing")
            if pricing_skill:
                try:
                    import asyncio
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        # Can't await in sync context - use sync method
                        result = pricing_skill._estimate({
                            "description": description,
                            "skills_needed": [skill_id],
                            "urgency": urgency,
                        })
                        if result.success:
                            return result.data
                except Exception:
                    pass

        # Local fallback pricing
        return self._local_estimate(skill_id, action, description, urgency)

    def _local_estimate(self, skill_id: str, action: str,
                        description: str, urgency: str) -> Dict:
        """Simple local price estimation when TaskPricingSkill unavailable."""
        # Base costs per skill type
        base_costs = {
            "code_review": 0.02, "content": 0.015, "data_transform": 0.025,
            "web_scraper": 0.04, "browser": 0.05, "email": 0.002,
            "github": 0.005, "shell": 0.001, "filesystem": 0.001,
            "deployment": 0.03, "twitter": 0.003,
        }
        base = base_costs.get(skill_id, 0.01)

        # Complexity from description length
        desc_len = len(description)
        if desc_len > 500:
            complexity = 2.0
        elif desc_len > 200:
            complexity = 1.5
        elif desc_len > 50:
            complexity = 1.2
        else:
            complexity = 1.0

        # Urgency multiplier
        urgency_mult = {
            "batch": 0.7, "low": 0.85, "normal": 1.0,
            "high": 1.5, "critical": 2.5,
        }.get(urgency, 1.0)

        estimated_cost = base * complexity
        margin = 1.4  # 40% margin
        price = estimated_cost * complexity * urgency_mult * margin

        return {
            "estimated_cost": round(estimated_cost, 6),
            "price": round(price, 6),
            "breakdown": {
                "base_cost": base,
                "complexity": complexity,
                "urgency_multiplier": urgency_mult,
                "margin": margin,
                "skill_id": skill_id,
                "action": action,
            },
        }

    def _emit_event(self, topic: str, data: Dict):
        """Emit event to EventBus if available via context."""
        try:
            if hasattr(self, '_context') and self._context:
                bus = self._context.get_event_bus()
                if bus:
                    bus.publish(topic, data)
        except Exception:
            pass  # Events are best-effort

    def _forward_to_pricing_skill(self, quote: Dict, actual_cost: float,
                                  actual_tokens: int = 0):
        """Forward completion data to TaskPricingSkill for model calibration."""
        try:
            if hasattr(self, '_context') and self._context:
                pricing_skill = self._context.get_skill("task_pricing")
                if pricing_skill:
                    pricing_skill._record_actual({
                        "quote_id": quote.get("quote_id", ""),
                        "actual_cost": actual_cost,
                        "actual_tokens": actual_tokens,
                    })
        except Exception:
            pass  # Calibration forwarding is best-effort
