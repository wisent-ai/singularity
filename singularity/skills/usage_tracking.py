#!/usr/bin/env python3
"""
Usage Tracking Skill - Per-customer API usage metering, rate limiting, and billing.

This is the missing revenue protection and optimization layer. Without usage
tracking, the agent cannot:
  - Charge customers based on actual usage (metered billing)
  - Protect services from abuse (rate limiting / quotas)
  - Identify most valuable customers and services
  - Generate invoices for payment collection

Revenue flow integration:
  1. Customer authenticates via API key → linked to a customer account
  2. Each API call is metered (skill, action, tokens, cost)
  3. Rate limits and quotas are enforced per-customer
  4. Usage rolls up into billing periods with invoices
  5. Analytics show which customers/services are most profitable

Part of the Revenue Generation pillar: the billing and protection layer.
"""

import json
import uuid
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

from .base import Skill, SkillResult, SkillManifest, SkillAction


USAGE_FILE = Path(__file__).parent.parent / "data" / "usage_tracking.json"
MAX_RECORDS_PER_CUSTOMER = 5000
MAX_CUSTOMERS = 500


# Default rate limits
DEFAULT_RATE_LIMITS = {
    "requests_per_minute": 60,
    "requests_per_hour": 1000,
    "requests_per_day": 10000,
    "max_concurrent": 5,
}

# Billing period options
BILLING_PERIODS = ["hourly", "daily", "weekly", "monthly"]


class UsageTrackingSkill(Skill):
    """
    Per-customer usage metering, rate limiting, quotas, and billing.

    Enables the agent to:
    - Register customers with API keys and usage tiers
    - Track every API call (skill, action, cost, latency)
    - Enforce rate limits and quotas per customer
    - Generate usage reports and invoices
    - Identify top customers and most popular services
    - Set custom pricing tiers (free, basic, premium)
    """

    def __init__(self, credentials: Dict[str, str] = None):
        super().__init__(credentials)
        self._ensure_data()

    def _ensure_data(self):
        USAGE_FILE.parent.mkdir(parents=True, exist_ok=True)
        if not USAGE_FILE.exists():
            self._save(self._default_state())

    def _default_state(self) -> Dict:
        return {
            "customers": {},
            "api_keys": {},  # api_key -> customer_id mapping
            "usage_records": {},  # customer_id -> list of records
            "invoices": [],
            "tiers": {
                "free": {
                    "requests_per_minute": 10,
                    "requests_per_hour": 100,
                    "requests_per_day": 500,
                    "max_concurrent": 2,
                    "price_per_request": 0.0,
                    "monthly_quota": 500,
                },
                "basic": {
                    "requests_per_minute": 30,
                    "requests_per_hour": 500,
                    "requests_per_day": 5000,
                    "max_concurrent": 5,
                    "price_per_request": 0.001,
                    "monthly_quota": 5000,
                },
                "premium": {
                    "requests_per_minute": 120,
                    "requests_per_hour": 5000,
                    "requests_per_day": 50000,
                    "max_concurrent": 20,
                    "price_per_request": 0.0005,
                    "monthly_quota": 50000,
                },
            },
            "created_at": datetime.now().isoformat(),
        }

    def _load(self) -> Dict:
        try:
            with open(USAGE_FILE, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            state = self._default_state()
            self._save(state)
            return state

    def _save(self, state: Dict):
        try:
            with open(USAGE_FILE, "w") as f:
                json.dump(state, f, indent=2)
        except Exception:
            pass

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="usage_tracking",
            name="Usage Tracking",
            version="1.0.0",
            category="revenue",
            description="Per-customer API usage metering, rate limiting, and billing",
            required_credentials=[],
            actions=[
                SkillAction(
                    name="register_customer",
                    description="Register a new customer with an API key and usage tier",
                    parameters={
                        "name": {"type": "string", "required": True, "description": "Customer name"},
                        "email": {"type": "string", "required": False, "description": "Customer email"},
                        "tier": {"type": "string", "required": False, "description": "Pricing tier (free/basic/premium)"},
                    },
                ),
                SkillAction(
                    name="record_usage",
                    description="Record an API call for a customer (called by ServiceAPI middleware)",
                    parameters={
                        "api_key": {"type": "string", "required": True, "description": "Customer API key"},
                        "skill_id": {"type": "string", "required": True, "description": "Skill that was called"},
                        "action": {"type": "string", "required": True, "description": "Action that was executed"},
                        "cost": {"type": "number", "required": False, "description": "Cost of the request"},
                        "latency_ms": {"type": "number", "required": False, "description": "Request latency in ms"},
                        "success": {"type": "boolean", "required": False, "description": "Whether request succeeded"},
                    },
                ),
                SkillAction(
                    name="check_rate_limit",
                    description="Check if a customer is within their rate limits",
                    parameters={
                        "api_key": {"type": "string", "required": True, "description": "Customer API key"},
                    },
                ),
                SkillAction(
                    name="get_usage_report",
                    description="Get usage report for a customer or all customers",
                    parameters={
                        "customer_id": {"type": "string", "required": False, "description": "Customer ID (omit for all)"},
                        "period": {"type": "string", "required": False, "description": "Time period (hourly/daily/weekly/monthly)"},
                    },
                ),
                SkillAction(
                    name="generate_invoice",
                    description="Generate an invoice for a customer's usage in a period",
                    parameters={
                        "customer_id": {"type": "string", "required": True, "description": "Customer ID"},
                        "period_start": {"type": "string", "required": False, "description": "Period start (ISO format)"},
                        "period_end": {"type": "string", "required": False, "description": "Period end (ISO format)"},
                    },
                ),
                SkillAction(
                    name="get_analytics",
                    description="Get analytics on service popularity, top customers, and revenue trends",
                    parameters={},
                ),
                SkillAction(
                    name="update_tier",
                    description="Update a customer's pricing tier",
                    parameters={
                        "customer_id": {"type": "string", "required": True, "description": "Customer ID"},
                        "tier": {"type": "string", "required": True, "description": "New tier (free/basic/premium)"},
                    },
                ),
            ],
        )

    async def execute(self, action: str, params: Dict[str, Any]) -> SkillResult:
        actions = {
            "register_customer": self._register_customer,
            "record_usage": self._record_usage,
            "check_rate_limit": self._check_rate_limit,
            "get_usage_report": self._get_usage_report,
            "generate_invoice": self._generate_invoice,
            "get_analytics": self._get_analytics,
            "update_tier": self._update_tier,
        }

        handler = actions.get(action)
        if not handler:
            return SkillResult(
                success=False,
                message=f"Unknown action: {action}. Available: {list(actions.keys())}",
            )

        try:
            return handler(params)
        except Exception as e:
            return SkillResult(success=False, message=f"Error: {str(e)}")

    def _register_customer(self, params: Dict) -> SkillResult:
        """Register a new customer with API key and tier."""
        name = params.get("name")
        if not name:
            return SkillResult(success=False, message="Customer name is required")

        email = params.get("email", "")
        tier = params.get("tier", "free")

        state = self._load()

        if tier not in state["tiers"]:
            return SkillResult(
                success=False,
                message=f"Invalid tier: {tier}. Available: {list(state['tiers'].keys())}",
            )

        customer_id = f"cust_{uuid.uuid4().hex[:12]}"
        api_key = f"sk_{uuid.uuid4().hex}"

        if len(state["customers"]) >= MAX_CUSTOMERS:
            return SkillResult(success=False, message=f"Maximum customers ({MAX_CUSTOMERS}) reached")

        state["customers"][customer_id] = {
            "customer_id": customer_id,
            "name": name,
            "email": email,
            "tier": tier,
            "api_key": api_key,
            "created_at": datetime.now().isoformat(),
            "active": True,
            "total_requests": 0,
            "total_cost": 0.0,
            "total_revenue": 0.0,
        }
        state["api_keys"][api_key] = customer_id
        state["usage_records"][customer_id] = []

        self._save(state)

        return SkillResult(
            success=True,
            message=f"Customer '{name}' registered on '{tier}' tier",
            data={
                "customer_id": customer_id,
                "api_key": api_key,
                "tier": tier,
                "rate_limits": state["tiers"][tier],
            },
        )

    def _record_usage(self, params: Dict) -> SkillResult:
        """Record an API usage event for a customer."""
        api_key = params.get("api_key")
        if not api_key:
            return SkillResult(success=False, message="API key is required")

        state = self._load()
        customer_id = state["api_keys"].get(api_key)
        if not customer_id:
            return SkillResult(success=False, message="Invalid API key")

        customer = state["customers"].get(customer_id)
        if not customer or not customer.get("active", True):
            return SkillResult(success=False, message="Customer account inactive")

        tier = customer.get("tier", "free")
        tier_config = state["tiers"].get(tier, state["tiers"]["free"])
        price_per_request = tier_config.get("price_per_request", 0)

        record = {
            "record_id": f"rec_{uuid.uuid4().hex[:8]}",
            "timestamp": datetime.now().isoformat(),
            "skill_id": params.get("skill_id", "unknown"),
            "action": params.get("action", "unknown"),
            "cost": params.get("cost", 0),
            "revenue": price_per_request,
            "latency_ms": params.get("latency_ms", 0),
            "success": params.get("success", True),
        }

        # Append and trim
        records = state["usage_records"].get(customer_id, [])
        records.append(record)
        if len(records) > MAX_RECORDS_PER_CUSTOMER:
            records = records[-MAX_RECORDS_PER_CUSTOMER:]
        state["usage_records"][customer_id] = records

        # Update customer totals
        customer["total_requests"] = customer.get("total_requests", 0) + 1
        customer["total_cost"] = customer.get("total_cost", 0) + record["cost"]
        customer["total_revenue"] = customer.get("total_revenue", 0) + record["revenue"]

        self._save(state)

        return SkillResult(
            success=True,
            message="Usage recorded",
            data={"record_id": record["record_id"], "revenue": record["revenue"]},
            revenue=record["revenue"],
        )

    def _check_rate_limit(self, params: Dict) -> SkillResult:
        """Check if a customer is within their rate limits. Returns allowed/blocked."""
        api_key = params.get("api_key")
        if not api_key:
            return SkillResult(success=False, message="API key is required")

        state = self._load()
        customer_id = state["api_keys"].get(api_key)
        if not customer_id:
            return SkillResult(
                success=False,
                message="Invalid API key",
                data={"allowed": False, "reason": "invalid_key"},
            )

        customer = state["customers"].get(customer_id)
        if not customer or not customer.get("active", True):
            return SkillResult(
                success=False,
                message="Customer account inactive",
                data={"allowed": False, "reason": "inactive_account"},
            )

        tier = customer.get("tier", "free")
        tier_config = state["tiers"].get(tier, state["tiers"]["free"])

        records = state["usage_records"].get(customer_id, [])
        now = datetime.now()

        # Count requests in different windows
        minute_ago = (now - timedelta(minutes=1)).isoformat()
        hour_ago = (now - timedelta(hours=1)).isoformat()
        day_ago = (now - timedelta(days=1)).isoformat()

        requests_last_minute = sum(1 for r in records if r["timestamp"] > minute_ago)
        requests_last_hour = sum(1 for r in records if r["timestamp"] > hour_ago)
        requests_last_day = sum(1 for r in records if r["timestamp"] > day_ago)

        # Check limits
        limits = {
            "requests_per_minute": (requests_last_minute, tier_config.get("requests_per_minute", 60)),
            "requests_per_hour": (requests_last_hour, tier_config.get("requests_per_hour", 1000)),
            "requests_per_day": (requests_last_day, tier_config.get("requests_per_day", 10000)),
        }

        violated = None
        for limit_name, (current, maximum) in limits.items():
            if current >= maximum:
                violated = limit_name
                break

        # Check monthly quota
        month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0).isoformat()
        monthly_requests = sum(1 for r in records if r["timestamp"] > month_start)
        monthly_quota = tier_config.get("monthly_quota", 10000)

        if monthly_requests >= monthly_quota:
            violated = "monthly_quota"

        allowed = violated is None

        return SkillResult(
            success=True,
            message="Within rate limits" if allowed else f"Rate limit exceeded: {violated}",
            data={
                "allowed": allowed,
                "customer_id": customer_id,
                "tier": tier,
                "usage": {
                    "last_minute": requests_last_minute,
                    "last_hour": requests_last_hour,
                    "last_day": requests_last_day,
                    "this_month": monthly_requests,
                },
                "limits": {
                    "per_minute": tier_config.get("requests_per_minute", 60),
                    "per_hour": tier_config.get("requests_per_hour", 1000),
                    "per_day": tier_config.get("requests_per_day", 10000),
                    "monthly_quota": monthly_quota,
                },
                "violated": violated,
            },
        )

    def _get_usage_report(self, params: Dict) -> SkillResult:
        """Get usage report for a customer or all customers."""
        state = self._load()
        customer_id = params.get("customer_id")
        period = params.get("period", "daily")

        now = datetime.now()
        if period == "hourly":
            cutoff = (now - timedelta(hours=1)).isoformat()
        elif period == "daily":
            cutoff = (now - timedelta(days=1)).isoformat()
        elif period == "weekly":
            cutoff = (now - timedelta(weeks=1)).isoformat()
        elif period == "monthly":
            cutoff = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0).isoformat()
        else:
            cutoff = (now - timedelta(days=1)).isoformat()

        if customer_id:
            # Single customer report
            if customer_id not in state["customers"]:
                return SkillResult(success=False, message=f"Customer {customer_id} not found")
            report = self._build_customer_report(state, customer_id, cutoff)
            return SkillResult(
                success=True,
                message=f"Usage report for {state['customers'][customer_id]['name']}",
                data=report,
            )
        else:
            # All customers report
            reports = []
            for cid in state["customers"]:
                report = self._build_customer_report(state, cid, cutoff)
                reports.append(report)

            total_requests = sum(r["total_requests"] for r in reports)
            total_revenue = sum(r["total_revenue"] for r in reports)
            total_cost = sum(r["total_cost"] for r in reports)

            return SkillResult(
                success=True,
                message=f"Usage report for {len(reports)} customers ({period})",
                data={
                    "period": period,
                    "customer_count": len(reports),
                    "total_requests": total_requests,
                    "total_revenue": round(total_revenue, 4),
                    "total_cost": round(total_cost, 4),
                    "net_profit": round(total_revenue - total_cost, 4),
                    "customers": reports,
                },
            )

    def _build_customer_report(self, state: Dict, customer_id: str, cutoff: str) -> Dict:
        """Build usage report for a single customer within a time period."""
        customer = state["customers"][customer_id]
        records = state["usage_records"].get(customer_id, [])
        period_records = [r for r in records if r["timestamp"] > cutoff]

        # Aggregate by skill
        by_skill = defaultdict(lambda: {"requests": 0, "cost": 0, "revenue": 0, "errors": 0})
        total_latency = 0
        for r in period_records:
            skill_id = r.get("skill_id", "unknown")
            by_skill[skill_id]["requests"] += 1
            by_skill[skill_id]["cost"] += r.get("cost", 0)
            by_skill[skill_id]["revenue"] += r.get("revenue", 0)
            if not r.get("success", True):
                by_skill[skill_id]["errors"] += 1
            total_latency += r.get("latency_ms", 0)

        total_requests = len(period_records)
        total_revenue = sum(s["revenue"] for s in by_skill.values())
        total_cost = sum(s["cost"] for s in by_skill.values())
        avg_latency = total_latency / total_requests if total_requests > 0 else 0

        return {
            "customer_id": customer_id,
            "name": customer["name"],
            "tier": customer.get("tier", "free"),
            "total_requests": total_requests,
            "total_revenue": round(total_revenue, 4),
            "total_cost": round(total_cost, 4),
            "avg_latency_ms": round(avg_latency, 2),
            "error_rate": round(
                sum(s["errors"] for s in by_skill.values()) / total_requests * 100
                if total_requests > 0 else 0, 2
            ),
            "by_skill": dict(by_skill),
        }

    def _generate_invoice(self, params: Dict) -> SkillResult:
        """Generate an invoice for a customer's usage in a billing period."""
        customer_id = params.get("customer_id")
        if not customer_id:
            return SkillResult(success=False, message="Customer ID is required")

        state = self._load()
        if customer_id not in state["customers"]:
            return SkillResult(success=False, message=f"Customer {customer_id} not found")

        customer = state["customers"][customer_id]
        now = datetime.now()

        # Determine billing period
        period_end = params.get("period_end", now.isoformat())
        period_start = params.get("period_start")
        if not period_start:
            # Default to current month
            period_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0).isoformat()

        records = state["usage_records"].get(customer_id, [])
        period_records = [
            r for r in records
            if period_start <= r["timestamp"] <= period_end
        ]

        tier = customer.get("tier", "free")
        tier_config = state["tiers"].get(tier, state["tiers"]["free"])
        price_per_request = tier_config.get("price_per_request", 0)

        # Calculate line items by skill
        line_items = []
        by_skill = defaultdict(int)
        for r in period_records:
            by_skill[r.get("skill_id", "unknown")] += 1

        for skill_id, count in by_skill.items():
            line_items.append({
                "description": f"{skill_id} API calls",
                "quantity": count,
                "unit_price": price_per_request,
                "total": round(count * price_per_request, 4),
            })

        subtotal = sum(item["total"] for item in line_items)

        invoice = {
            "invoice_id": f"inv_{uuid.uuid4().hex[:10]}",
            "customer_id": customer_id,
            "customer_name": customer["name"],
            "tier": tier,
            "period_start": period_start,
            "period_end": period_end,
            "generated_at": now.isoformat(),
            "line_items": line_items,
            "subtotal": round(subtotal, 4),
            "total": round(subtotal, 4),
            "total_requests": len(period_records),
            "status": "pending",
        }

        state["invoices"].append(invoice)
        # Keep only recent invoices
        if len(state["invoices"]) > 1000:
            state["invoices"] = state["invoices"][-1000:]
        self._save(state)

        return SkillResult(
            success=True,
            message=f"Invoice generated: ${invoice['total']:.4f} for {invoice['total_requests']} requests",
            data=invoice,
            revenue=invoice["total"],
        )

    def _get_analytics(self, params: Dict) -> SkillResult:
        """Get analytics on service popularity, top customers, and revenue trends."""
        state = self._load()

        customers = state["customers"]
        if not customers:
            return SkillResult(
                success=True,
                message="No customer data yet",
                data={"customer_count": 0, "total_revenue": 0},
            )

        # Top customers by revenue
        top_by_revenue = sorted(
            customers.values(),
            key=lambda c: c.get("total_revenue", 0),
            reverse=True,
        )[:10]

        # Top customers by requests
        top_by_requests = sorted(
            customers.values(),
            key=lambda c: c.get("total_requests", 0),
            reverse=True,
        )[:10]

        # Service popularity across all customers
        service_popularity = defaultdict(lambda: {"requests": 0, "revenue": 0, "customers": set()})
        for cid, records in state["usage_records"].items():
            for r in records:
                skill_id = r.get("skill_id", "unknown")
                service_popularity[skill_id]["requests"] += 1
                service_popularity[skill_id]["revenue"] += r.get("revenue", 0)
                service_popularity[skill_id]["customers"].add(cid)

        # Convert sets to counts for JSON serialization
        popular_services = []
        for skill_id, data in sorted(
            service_popularity.items(), key=lambda x: x[1]["requests"], reverse=True
        ):
            popular_services.append({
                "skill_id": skill_id,
                "total_requests": data["requests"],
                "total_revenue": round(data["revenue"], 4),
                "unique_customers": len(data["customers"]),
            })

        # Tier distribution
        tier_dist = defaultdict(int)
        for c in customers.values():
            tier_dist[c.get("tier", "free")] += 1

        # Totals
        total_revenue = sum(c.get("total_revenue", 0) for c in customers.values())
        total_cost = sum(c.get("total_cost", 0) for c in customers.values())
        total_requests = sum(c.get("total_requests", 0) for c in customers.values())

        return SkillResult(
            success=True,
            message=f"Analytics: {len(customers)} customers, {total_requests} requests, ${total_revenue:.4f} revenue",
            data={
                "summary": {
                    "customer_count": len(customers),
                    "total_requests": total_requests,
                    "total_revenue": round(total_revenue, 4),
                    "total_cost": round(total_cost, 4),
                    "net_profit": round(total_revenue - total_cost, 4),
                },
                "tier_distribution": dict(tier_dist),
                "top_customers_by_revenue": [
                    {
                        "customer_id": c["customer_id"],
                        "name": c["name"],
                        "total_revenue": round(c.get("total_revenue", 0), 4),
                        "total_requests": c.get("total_requests", 0),
                    }
                    for c in top_by_revenue
                ],
                "top_customers_by_requests": [
                    {
                        "customer_id": c["customer_id"],
                        "name": c["name"],
                        "total_requests": c.get("total_requests", 0),
                        "tier": c.get("tier", "free"),
                    }
                    for c in top_by_requests
                ],
                "popular_services": popular_services[:10],
            },
        )

    def _update_tier(self, params: Dict) -> SkillResult:
        """Update a customer's pricing tier."""
        customer_id = params.get("customer_id")
        tier = params.get("tier")

        if not customer_id or not tier:
            return SkillResult(success=False, message="Customer ID and tier are required")

        state = self._load()

        if customer_id not in state["customers"]:
            return SkillResult(success=False, message=f"Customer {customer_id} not found")

        if tier not in state["tiers"]:
            return SkillResult(
                success=False,
                message=f"Invalid tier: {tier}. Available: {list(state['tiers'].keys())}",
            )

        old_tier = state["customers"][customer_id].get("tier", "free")
        state["customers"][customer_id]["tier"] = tier
        self._save(state)

        return SkillResult(
            success=True,
            message=f"Customer upgraded from '{old_tier}' to '{tier}'",
            data={
                "customer_id": customer_id,
                "old_tier": old_tier,
                "new_tier": tier,
                "new_limits": state["tiers"][tier],
            },
        )

    async def estimate_cost(self, action: str, params: Dict) -> float:
        """Usage tracking is free - it's internal infrastructure."""
        return 0.0
