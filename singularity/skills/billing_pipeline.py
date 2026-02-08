#!/usr/bin/env python3
"""
Billing Pipeline Skill - Automated end-to-end billing from usage to payment.

This is the critical missing link in the revenue chain. Individual pieces exist:
  - UsageTrackingSkill: meters per-customer API usage
  - PaymentSkill: creates invoices and payment links
  - MarketplaceSkill: tracks orders and revenue

But nothing CONNECTS them automatically. Without BillingPipelineSkill, someone
must manually: check usage → calculate charges → create invoice → send to customer.
That defeats the purpose of an autonomous revenue-generating agent.

BillingPipelineSkill automates the full billing lifecycle:
  1. Collect usage data from UsageTrackingSkill per billing period
  2. Calculate charges based on customer tier pricing
  3. Apply discounts, credits, and minimum charges
  4. Generate itemized invoices via PaymentSkill
  5. Track invoice status and send reminders
  6. Close billing periods and roll over credits
  7. Generate billing analytics and revenue forecasts

Actions:
  - run_billing_cycle: Execute a full billing cycle for all customers
  - bill_customer: Generate invoice for a specific customer
  - apply_credit: Add credit to a customer account
  - apply_discount: Set a discount for a customer
  - billing_status: Show current billing period status
  - billing_history: View past billing cycles and revenue
  - configure: Set billing period, thresholds, auto-send options
  - forecast: Predict revenue based on usage trends

Revenue flow:
  UsageTracking → BillingPipeline → PaymentSkill → Revenue

Part of the Revenue Generation pillar: the automated billing engine.
"""

import json
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any

from .base import Skill, SkillResult, SkillManifest, SkillAction


BILLING_FILE = Path(__file__).parent.parent / "data" / "billing_pipeline.json"


class BillingPipelineSkill(Skill):
    """
    Automated billing pipeline connecting usage metering to invoice generation.

    Runs billing cycles that:
    - Pull usage data per customer
    - Calculate charges with tier-based pricing
    - Apply credits and discounts
    - Generate itemized invoices
    - Track billing history for forecasting
    """

    def __init__(self, credentials: Dict[str, str] = None):
        super().__init__(credentials)
        self._ensure_data()

    def _ensure_data(self):
        BILLING_FILE.parent.mkdir(parents=True, exist_ok=True)
        if not BILLING_FILE.exists():
            self._save(self._default_state())

    def _default_state(self) -> Dict:
        return {
            "config": {
                "billing_period": "monthly",  # daily, weekly, monthly
                "auto_send_invoices": True,
                "minimum_charge": 0.01,  # Don't invoice below this
                "grace_period_days": 7,  # Days before overdue
                "auto_apply_credits": True,
                "currency": "USD",
            },
            "customers": {},  # customer_id -> billing profile
            "billing_cycles": [],  # completed billing cycles
            "current_cycle": None,  # active billing cycle
            "credits": {},  # customer_id -> credit balance
            "discounts": {},  # customer_id -> discount info
            "created_at": datetime.now().isoformat(),
        }

    def _load(self) -> Dict:
        try:
            with open(BILLING_FILE, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            state = self._default_state()
            self._save(state)
            return state

    def _save(self, state: Dict):
        try:
            with open(BILLING_FILE, "w") as f:
                json.dump(state, f, indent=2)
        except Exception:
            pass

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="billing_pipeline",
            name="Billing Pipeline",
            version="1.0.0",
            category="revenue",
            description="Automated end-to-end billing from usage metering to invoice generation",
            required_credentials=[],
            actions=[
                SkillAction(
                    name="run_billing_cycle",
                    description="Execute a full billing cycle - calculate charges and generate invoices for all customers",
                    parameters={
                        "period_end": {"type": "string", "required": False, "description": "End of billing period (ISO format, defaults to now)"},
                        "dry_run": {"type": "boolean", "required": False, "description": "Preview charges without generating invoices"},
                    },
                ),
                SkillAction(
                    name="bill_customer",
                    description="Generate an invoice for a specific customer based on their usage",
                    parameters={
                        "customer_id": {"type": "string", "required": True, "description": "Customer ID to bill"},
                        "usage_records": {"type": "array", "required": False, "description": "Usage records to bill (auto-fetched if omitted)"},
                    },
                ),
                SkillAction(
                    name="apply_credit",
                    description="Add billing credit to a customer account",
                    parameters={
                        "customer_id": {"type": "string", "required": True, "description": "Customer ID"},
                        "amount": {"type": "number", "required": True, "description": "Credit amount in USD"},
                        "reason": {"type": "string", "required": False, "description": "Reason for credit"},
                    },
                ),
                SkillAction(
                    name="apply_discount",
                    description="Set a percentage or fixed discount for a customer",
                    parameters={
                        "customer_id": {"type": "string", "required": True, "description": "Customer ID"},
                        "discount_type": {"type": "string", "required": True, "description": "Type: 'percentage' or 'fixed'"},
                        "value": {"type": "number", "required": True, "description": "Discount value (percent or dollar amount)"},
                        "expires_at": {"type": "string", "required": False, "description": "Expiration date (ISO format)"},
                    },
                ),
                SkillAction(
                    name="billing_status",
                    description="Show current billing period status - active customers, accrued charges, pending invoices",
                    parameters={},
                ),
                SkillAction(
                    name="billing_history",
                    description="View past billing cycles with revenue totals and trends",
                    parameters={
                        "limit": {"type": "integer", "required": False, "description": "Number of cycles to show (default 10)"},
                    },
                ),
                SkillAction(
                    name="configure",
                    description="Update billing configuration (period, thresholds, auto-send)",
                    parameters={
                        "billing_period": {"type": "string", "required": False, "description": "Billing period: daily, weekly, monthly"},
                        "auto_send_invoices": {"type": "boolean", "required": False, "description": "Auto-send invoices when generated"},
                        "minimum_charge": {"type": "number", "required": False, "description": "Minimum amount to invoice"},
                        "grace_period_days": {"type": "integer", "required": False, "description": "Days before marking overdue"},
                    },
                ),
                SkillAction(
                    name="forecast",
                    description="Predict revenue for next billing period based on usage trends",
                    parameters={
                        "periods_ahead": {"type": "integer", "required": False, "description": "Number of periods to forecast (default 3)"},
                    },
                ),
            ],
        )

    async def execute(self, action: str, params: Dict[str, Any]) -> SkillResult:
        actions = {
            "run_billing_cycle": self._run_billing_cycle,
            "bill_customer": self._bill_customer,
            "apply_credit": self._apply_credit,
            "apply_discount": self._apply_discount,
            "billing_status": self._billing_status,
            "billing_history": self._billing_history,
            "configure": self._configure,
            "forecast": self._forecast,
        }

        handler = actions.get(action)
        if not handler:
            return SkillResult(
                success=False,
                message=f"Unknown action: {action}. Available: {list(actions.keys())}",
            )

        try:
            return await handler(params)
        except Exception as e:
            return SkillResult(success=False, message=f"Billing error: {str(e)}")

    async def _run_billing_cycle(self, params: Dict) -> SkillResult:
        """Execute a full billing cycle for all customers."""
        state = self._load()
        dry_run = params.get("dry_run", False)
        period_end = params.get("period_end", datetime.now().isoformat())

        # Determine billing period start
        config = state["config"]
        period_start = self._calculate_period_start(period_end, config["billing_period"])

        cycle_id = str(uuid.uuid4())[:8]
        invoices_generated = []
        total_revenue = 0.0
        customers_billed = 0
        customers_skipped = 0

        # Bill each customer
        for customer_id, profile in state.get("customers", {}).items():
            usage = profile.get("usage_records", [])

            # Filter usage records within this billing period
            period_usage = [
                r for r in usage
                if period_start <= r.get("timestamp", "") <= period_end
            ]

            if not period_usage:
                customers_skipped += 1
                continue

            # Calculate charges
            charge_result = self._calculate_charges(
                customer_id, period_usage, profile, state
            )

            if charge_result["total"] < config["minimum_charge"]:
                # Roll over to next period
                customers_skipped += 1
                continue

            # Generate invoice
            invoice = {
                "invoice_id": f"INV-{cycle_id}-{customer_id[:6]}",
                "customer_id": customer_id,
                "cycle_id": cycle_id,
                "period_start": period_start,
                "period_end": period_end,
                "line_items": charge_result["line_items"],
                "subtotal": charge_result["subtotal"],
                "discount": charge_result["discount"],
                "credits_applied": charge_result["credits_applied"],
                "total": charge_result["total"],
                "currency": config["currency"],
                "status": "draft" if dry_run else "sent",
                "created_at": datetime.now().isoformat(),
                "due_date": (
                    datetime.now() + timedelta(days=config["grace_period_days"])
                ).isoformat(),
            }

            invoices_generated.append(invoice)
            total_revenue += charge_result["total"]
            customers_billed += 1

            # Deduct applied credits
            if not dry_run and charge_result["credits_applied"] > 0:
                credits = state.get("credits", {})
                if customer_id in credits:
                    credits[customer_id]["balance"] -= charge_result["credits_applied"]
                    credits[customer_id]["balance"] = max(0, credits[customer_id]["balance"])

        if not dry_run:
            # Record the billing cycle
            cycle_record = {
                "cycle_id": cycle_id,
                "period_start": period_start,
                "period_end": period_end,
                "billing_period": config["billing_period"],
                "customers_billed": customers_billed,
                "customers_skipped": customers_skipped,
                "total_revenue": total_revenue,
                "invoices": invoices_generated,
                "completed_at": datetime.now().isoformat(),
            }
            state.setdefault("billing_cycles", []).append(cycle_record)

            # Clear billed usage records
            for invoice in invoices_generated:
                cid = invoice["customer_id"]
                if cid in state.get("customers", {}):
                    # Keep only records outside the billed period
                    state["customers"][cid]["usage_records"] = [
                        r for r in state["customers"][cid].get("usage_records", [])
                        if not (period_start <= r.get("timestamp", "") <= period_end)
                    ]

            self._save(state)

        return SkillResult(
            success=True,
            message=f"{'[DRY RUN] ' if dry_run else ''}Billing cycle {cycle_id}: "
                    f"{customers_billed} customers billed, {customers_skipped} skipped, "
                    f"${total_revenue:.2f} total revenue",
            data={
                "cycle_id": cycle_id,
                "dry_run": dry_run,
                "period_start": period_start,
                "period_end": period_end,
                "customers_billed": customers_billed,
                "customers_skipped": customers_skipped,
                "total_revenue": total_revenue,
                "invoices": invoices_generated,
            },
            revenue=0 if dry_run else total_revenue,
        )

    async def _bill_customer(self, params: Dict) -> SkillResult:
        """Generate an invoice for a specific customer."""
        customer_id = params.get("customer_id")
        if not customer_id:
            return SkillResult(success=False, message="customer_id is required")

        state = self._load()
        config = state["config"]
        customers = state.get("customers", {})

        if customer_id not in customers:
            return SkillResult(
                success=False,
                message=f"Customer {customer_id} not found. Register them first.",
            )

        profile = customers[customer_id]

        # Use provided usage records or pull from profile
        usage_records = params.get("usage_records") or profile.get("usage_records", [])

        if not usage_records:
            return SkillResult(
                success=False,
                message=f"No usage records found for customer {customer_id}",
            )

        # Calculate charges
        charge_result = self._calculate_charges(customer_id, usage_records, profile, state)

        if charge_result["total"] < config["minimum_charge"]:
            return SkillResult(
                success=True,
                message=f"Charges for {customer_id} (${charge_result['total']:.4f}) "
                        f"below minimum (${config['minimum_charge']:.2f}). No invoice generated.",
                data={"charges": charge_result, "invoice_generated": False},
            )

        # Generate invoice
        invoice_id = f"INV-{str(uuid.uuid4())[:8]}"
        invoice = {
            "invoice_id": invoice_id,
            "customer_id": customer_id,
            "line_items": charge_result["line_items"],
            "subtotal": charge_result["subtotal"],
            "discount": charge_result["discount"],
            "credits_applied": charge_result["credits_applied"],
            "total": charge_result["total"],
            "currency": config["currency"],
            "status": "sent" if config["auto_send_invoices"] else "draft",
            "created_at": datetime.now().isoformat(),
            "due_date": (
                datetime.now() + timedelta(days=config["grace_period_days"])
            ).isoformat(),
        }

        # Save invoice to customer profile
        profile.setdefault("invoices", []).append(invoice)

        # Deduct credits
        if charge_result["credits_applied"] > 0:
            credits = state.get("credits", {})
            if customer_id in credits:
                credits[customer_id]["balance"] -= charge_result["credits_applied"]
                credits[customer_id]["balance"] = max(0, credits[customer_id]["balance"])

        # Clear billed usage
        profile["usage_records"] = []
        self._save(state)

        return SkillResult(
            success=True,
            message=f"Invoice {invoice_id} generated for {customer_id}: ${charge_result['total']:.2f}",
            data={"invoice": invoice},
            revenue=charge_result["total"],
        )

    async def _apply_credit(self, params: Dict) -> SkillResult:
        """Add billing credit to a customer account."""
        customer_id = params.get("customer_id")
        amount = params.get("amount", 0)
        reason = params.get("reason", "Manual credit")

        if not customer_id:
            return SkillResult(success=False, message="customer_id is required")
        if amount <= 0:
            return SkillResult(success=False, message="Credit amount must be positive")

        state = self._load()
        credits = state.setdefault("credits", {})

        if customer_id not in credits:
            credits[customer_id] = {
                "balance": 0.0,
                "total_credited": 0.0,
                "history": [],
            }

        credits[customer_id]["balance"] += amount
        credits[customer_id]["total_credited"] += amount
        credits[customer_id]["history"].append({
            "amount": amount,
            "reason": reason,
            "applied_at": datetime.now().isoformat(),
        })

        self._save(state)

        return SkillResult(
            success=True,
            message=f"${amount:.2f} credit added to {customer_id}. "
                    f"New balance: ${credits[customer_id]['balance']:.2f}",
            data={
                "customer_id": customer_id,
                "credit_added": amount,
                "new_balance": credits[customer_id]["balance"],
            },
        )

    async def _apply_discount(self, params: Dict) -> SkillResult:
        """Set a discount for a customer."""
        customer_id = params.get("customer_id")
        discount_type = params.get("discount_type")
        value = params.get("value", 0)
        expires_at = params.get("expires_at")

        if not customer_id:
            return SkillResult(success=False, message="customer_id is required")
        if discount_type not in ("percentage", "fixed"):
            return SkillResult(
                success=False,
                message="discount_type must be 'percentage' or 'fixed'",
            )
        if value <= 0:
            return SkillResult(success=False, message="Discount value must be positive")
        if discount_type == "percentage" and value > 100:
            return SkillResult(success=False, message="Percentage discount cannot exceed 100")

        state = self._load()
        discounts = state.setdefault("discounts", {})

        discounts[customer_id] = {
            "type": discount_type,
            "value": value,
            "created_at": datetime.now().isoformat(),
            "expires_at": expires_at,
        }

        self._save(state)

        if discount_type == "percentage":
            desc = f"{value}% off"
        else:
            desc = f"${value:.2f} off"

        return SkillResult(
            success=True,
            message=f"Discount applied to {customer_id}: {desc}"
                    + (f" (expires {expires_at})" if expires_at else ""),
            data={"customer_id": customer_id, "discount": discounts[customer_id]},
        )

    async def _billing_status(self, params: Dict) -> SkillResult:
        """Show current billing period status."""
        state = self._load()
        config = state["config"]
        customers = state.get("customers", {})
        credits = state.get("credits", {})
        discounts = state.get("discounts", {})

        # Calculate current accrued charges
        active_customers = 0
        total_accrued = 0.0
        customer_summaries = []

        for cid, profile in customers.items():
            usage = profile.get("usage_records", [])
            if not usage:
                continue

            active_customers += 1
            charges = self._calculate_charges(cid, usage, profile, state)
            total_accrued += charges["total"]

            customer_summaries.append({
                "customer_id": cid,
                "name": profile.get("name", "Unknown"),
                "requests": len(usage),
                "accrued_charges": charges["total"],
                "credit_balance": credits.get(cid, {}).get("balance", 0),
                "has_discount": cid in discounts,
            })

        # Sort by highest charges first
        customer_summaries.sort(key=lambda x: x["accrued_charges"], reverse=True)

        # Count past cycles
        past_cycles = state.get("billing_cycles", [])

        return SkillResult(
            success=True,
            message=f"Billing status: {active_customers} active customers, "
                    f"${total_accrued:.2f} accrued, "
                    f"{len(past_cycles)} completed cycles",
            data={
                "config": config,
                "active_customers": active_customers,
                "total_accrued": total_accrued,
                "customer_summaries": customer_summaries[:20],
                "completed_cycles": len(past_cycles),
                "total_customers": len(customers),
            },
        )

    async def _billing_history(self, params: Dict) -> SkillResult:
        """View past billing cycles with revenue totals."""
        state = self._load()
        limit = params.get("limit", 10)
        cycles = state.get("billing_cycles", [])

        # Most recent first
        recent = cycles[-limit:] if cycles else []
        recent.reverse()

        total_all_time = sum(c.get("total_revenue", 0) for c in cycles)
        avg_per_cycle = total_all_time / len(cycles) if cycles else 0

        return SkillResult(
            success=True,
            message=f"Billing history: {len(cycles)} total cycles, "
                    f"${total_all_time:.2f} all-time revenue, "
                    f"${avg_per_cycle:.2f} avg/cycle",
            data={
                "total_cycles": len(cycles),
                "total_revenue": total_all_time,
                "average_revenue_per_cycle": avg_per_cycle,
                "recent_cycles": [
                    {
                        "cycle_id": c.get("cycle_id"),
                        "period": f"{c.get('period_start', '?')} to {c.get('period_end', '?')}",
                        "customers_billed": c.get("customers_billed", 0),
                        "revenue": c.get("total_revenue", 0),
                        "completed_at": c.get("completed_at"),
                    }
                    for c in recent
                ],
            },
        )

    async def _configure(self, params: Dict) -> SkillResult:
        """Update billing configuration."""
        state = self._load()
        config = state["config"]
        updated = []

        valid_periods = ("daily", "weekly", "monthly")
        if "billing_period" in params:
            if params["billing_period"] not in valid_periods:
                return SkillResult(
                    success=False,
                    message=f"Invalid period. Must be one of: {valid_periods}",
                )
            config["billing_period"] = params["billing_period"]
            updated.append(f"billing_period={params['billing_period']}")

        if "auto_send_invoices" in params:
            config["auto_send_invoices"] = bool(params["auto_send_invoices"])
            updated.append(f"auto_send={params['auto_send_invoices']}")

        if "minimum_charge" in params:
            config["minimum_charge"] = float(params["minimum_charge"])
            updated.append(f"min_charge=${params['minimum_charge']:.2f}")

        if "grace_period_days" in params:
            config["grace_period_days"] = int(params["grace_period_days"])
            updated.append(f"grace={params['grace_period_days']}d")

        if not updated:
            return SkillResult(
                success=True,
                message="No changes. Current config shown.",
                data={"config": config},
            )

        self._save(state)

        return SkillResult(
            success=True,
            message=f"Billing config updated: {', '.join(updated)}",
            data={"config": config},
        )

    async def _forecast(self, params: Dict) -> SkillResult:
        """Predict revenue for upcoming billing periods based on trends."""
        state = self._load()
        periods_ahead = params.get("periods_ahead", 3)
        cycles = state.get("billing_cycles", [])

        if len(cycles) < 2:
            # Not enough data - estimate from current accrued
            customers = state.get("customers", {})
            current_accrued = 0.0
            for cid, profile in customers.items():
                usage = profile.get("usage_records", [])
                if usage:
                    charges = self._calculate_charges(cid, usage, profile, state)
                    current_accrued += charges["total"]

            forecasts = [
                {
                    "period": i + 1,
                    "predicted_revenue": current_accrued,
                    "confidence": "low",
                    "basis": "current_accrual",
                }
                for i in range(periods_ahead)
            ]

            return SkillResult(
                success=True,
                message=f"Forecast (low confidence - need 2+ billing cycles): "
                        f"~${current_accrued:.2f}/period",
                data={"forecasts": forecasts, "data_points": len(cycles)},
            )

        # Calculate trend from historical cycles
        revenues = [c.get("total_revenue", 0) for c in cycles]
        recent = revenues[-6:]  # Use last 6 cycles max

        # Simple linear trend
        n = len(recent)
        avg_revenue = sum(recent) / n
        if n >= 2:
            # Calculate growth rate
            growth_rates = []
            for i in range(1, len(recent)):
                if recent[i - 1] > 0:
                    rate = (recent[i] - recent[i - 1]) / recent[i - 1]
                    growth_rates.append(rate)
            avg_growth = sum(growth_rates) / len(growth_rates) if growth_rates else 0
        else:
            avg_growth = 0

        forecasts = []
        predicted = recent[-1] if recent else avg_revenue
        for i in range(periods_ahead):
            predicted = predicted * (1 + avg_growth)
            predicted = max(0, predicted)  # No negative revenue
            forecasts.append({
                "period": i + 1,
                "predicted_revenue": round(predicted, 2),
                "growth_rate": round(avg_growth * 100, 1),
                "confidence": "medium" if n >= 4 else "low",
                "basis": "trend_analysis",
            })

        total_forecast = sum(f["predicted_revenue"] for f in forecasts)

        return SkillResult(
            success=True,
            message=f"Revenue forecast ({periods_ahead} periods): ${total_forecast:.2f} total, "
                    f"{avg_growth * 100:.1f}% avg growth rate",
            data={
                "forecasts": forecasts,
                "data_points": len(cycles),
                "avg_revenue": round(avg_revenue, 2),
                "avg_growth_rate": round(avg_growth * 100, 1),
                "total_forecast": round(total_forecast, 2),
            },
        )

    # ─── Internal helpers ────────────────────────────────────────────────

    def _calculate_charges(
        self,
        customer_id: str,
        usage_records: List[Dict],
        profile: Dict,
        state: Dict,
    ) -> Dict:
        """Calculate charges for a set of usage records."""
        config = state["config"]

        # Group usage by service
        service_usage: Dict[str, List[Dict]] = {}
        for record in usage_records:
            key = f"{record.get('skill', 'unknown')}:{record.get('action', 'unknown')}"
            service_usage.setdefault(key, []).append(record)

        # Calculate line items
        line_items = []
        subtotal = 0.0

        for service_key, records in service_usage.items():
            # Get per-request price from profile tier or default
            tier = profile.get("tier", "basic")
            price_per_request = self._get_tier_price(tier)

            # Use cost from records if available, otherwise use tier pricing
            total_cost = sum(r.get("cost", price_per_request) for r in records)
            request_count = len(records)

            line_items.append({
                "service": service_key,
                "quantity": request_count,
                "unit_price": round(total_cost / request_count, 6) if request_count else 0,
                "total": round(total_cost, 4),
            })
            subtotal += total_cost

        subtotal = round(subtotal, 4)

        # Apply discount
        discount_amount = 0.0
        discount_info = state.get("discounts", {}).get(customer_id)
        if discount_info:
            # Check expiration
            expires = discount_info.get("expires_at")
            if expires and expires < datetime.now().isoformat():
                pass  # Expired, no discount
            elif discount_info["type"] == "percentage":
                discount_amount = subtotal * (discount_info["value"] / 100)
            elif discount_info["type"] == "fixed":
                discount_amount = min(discount_info["value"], subtotal)

        discount_amount = round(discount_amount, 4)

        # Apply credits
        credits_applied = 0.0
        after_discount = subtotal - discount_amount
        if config.get("auto_apply_credits", True):
            credit_balance = state.get("credits", {}).get(customer_id, {}).get("balance", 0)
            if credit_balance > 0:
                credits_applied = min(credit_balance, after_discount)
                credits_applied = round(credits_applied, 4)

        total = round(max(0, after_discount - credits_applied), 4)

        return {
            "line_items": line_items,
            "subtotal": subtotal,
            "discount": discount_amount,
            "credits_applied": credits_applied,
            "total": total,
        }

    @staticmethod
    def _get_tier_price(tier: str) -> float:
        """Get per-request price for a tier."""
        prices = {
            "free": 0.0,
            "basic": 0.001,
            "premium": 0.0005,
            "enterprise": 0.0003,
        }
        return prices.get(tier, 0.001)

    @staticmethod
    def _calculate_period_start(period_end: str, billing_period: str) -> str:
        """Calculate the start of the billing period."""
        try:
            end = datetime.fromisoformat(period_end)
        except (ValueError, TypeError):
            end = datetime.now()

        if billing_period == "daily":
            start = end - timedelta(days=1)
        elif billing_period == "weekly":
            start = end - timedelta(weeks=1)
        else:  # monthly
            start = end - timedelta(days=30)

        return start.isoformat()

    def register_customer(self, customer_id: str, name: str, tier: str = "basic") -> Dict:
        """Register a customer for billing (called by other skills)."""
        state = self._load()
        customers = state.setdefault("customers", {})

        if customer_id not in customers:
            customers[customer_id] = {
                "name": name,
                "tier": tier,
                "usage_records": [],
                "invoices": [],
                "registered_at": datetime.now().isoformat(),
            }
            self._save(state)

        return customers[customer_id]

    def record_usage(
        self, customer_id: str, skill: str, action: str,
        cost: float = 0.0, success: bool = True
    ):
        """Record a usage event for billing (called by ServiceAPI middleware)."""
        state = self._load()
        customers = state.get("customers", {})

        if customer_id not in customers:
            # Auto-register unknown customers
            customers[customer_id] = {
                "name": f"Customer {customer_id[:8]}",
                "tier": "basic",
                "usage_records": [],
                "invoices": [],
                "registered_at": datetime.now().isoformat(),
            }
            state["customers"] = customers

        customers[customer_id]["usage_records"].append({
            "skill": skill,
            "action": action,
            "cost": cost,
            "success": success,
            "timestamp": datetime.now().isoformat(),
        })

        self._save(state)
