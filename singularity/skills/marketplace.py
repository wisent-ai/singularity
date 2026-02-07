#!/usr/bin/env python3
"""
Marketplace Skill - Service catalog, pricing, order management, and revenue tracking.

This skill is the Revenue Generation engine. While ServiceAPI exposes the agent
as a callable REST endpoint, MarketplaceSkill manages the BUSINESS layer:

  - Define sellable services with pricing and SLAs
  - Accept and track customer orders
  - Calculate revenue, costs, and profit margins
  - Auto-discover new service opportunities from installed skills
  - Adjust pricing based on demand and performance data

The revenue loop:
  1. Agent inventories its skills → auto-generates service catalog
  2. Agent publishes services with pricing
  3. Customers place orders via ServiceAPI
  4. Agent fulfills orders using its skills
  5. Agent tracks revenue and adjusts pricing
  6. Agent identifies high-margin services → doubles down

Part of the Revenue Generation pillar: the business logic layer.
"""

import json
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from .base import Skill, SkillResult, SkillManifest, SkillAction


MARKETPLACE_FILE = Path(__file__).parent.parent / "data" / "marketplace.json"
MAX_SERVICES = 100
MAX_ORDERS = 1000


# Service statuses
SERVICE_STATUSES = ["active", "paused", "retired"]

# Order statuses
ORDER_STATUSES = ["pending", "in_progress", "completed", "failed", "refunded", "cancelled"]

# Pricing models
PRICING_MODELS = ["fixed", "per_unit", "hourly", "tiered"]


class MarketplaceSkill(Skill):
    """
    Revenue generation through service catalog and order management.

    Enables the agent to:
    - Create and manage sellable service offerings
    - Auto-discover services from installed skills
    - Accept and fulfill customer orders
    - Track revenue, costs, and profit per service
    - Adjust pricing based on demand signals
    - Generate revenue reports and insights
    """

    def __init__(self, credentials: Dict[str, str] = None):
        super().__init__(credentials)
        self._ensure_data()

    def _ensure_data(self):
        MARKETPLACE_FILE.parent.mkdir(parents=True, exist_ok=True)
        if not MARKETPLACE_FILE.exists():
            self._save(self._default_state())

    def _default_state(self) -> Dict:
        return {
            "services": [],
            "orders": [],
            "revenue_log": [],
            "pricing_adjustments": [],
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
        }

    def _load(self) -> Dict:
        try:
            with open(MARKETPLACE_FILE, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return self._default_state()

    def _save(self, state: Dict):
        state["last_updated"] = datetime.now().isoformat()
        MARKETPLACE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(MARKETPLACE_FILE, "w") as f:
            json.dump(state, f, indent=2, default=str)

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="marketplace",
            name="Service Marketplace",
            version="1.0.0",
            category="revenue",
            description="Manage sellable services, orders, pricing, and revenue tracking",
            required_credentials=[],
            install_cost=0,
            actions=[
                SkillAction(
                    name="create_service",
                    description="Create a new sellable service offering",
                    parameters={
                        "name": {"type": "string", "required": True, "description": "Service name"},
                        "description": {"type": "string", "required": True, "description": "What the service does"},
                        "skill_id": {"type": "string", "required": True, "description": "Underlying skill that fulfills this service"},
                        "action": {"type": "string", "required": True, "description": "Skill action to execute"},
                        "price": {"type": "number", "required": True, "description": "Price in USD"},
                        "pricing_model": {"type": "string", "required": False, "description": "Pricing model: fixed, per_unit, hourly, tiered"},
                        "estimated_cost": {"type": "number", "required": False, "description": "Estimated cost to fulfill"},
                        "sla_minutes": {"type": "number", "required": False, "description": "SLA delivery time in minutes"},
                        "tags": {"type": "array", "required": False, "description": "Categorization tags"},
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="list_services",
                    description="List all services in the catalog",
                    parameters={
                        "status": {"type": "string", "required": False, "description": "Filter by status: active, paused, retired"},
                        "tag": {"type": "string", "required": False, "description": "Filter by tag"},
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="update_service",
                    description="Update a service's details or pricing",
                    parameters={
                        "service_id": {"type": "string", "required": True, "description": "Service ID to update"},
                        "price": {"type": "number", "required": False, "description": "New price"},
                        "status": {"type": "string", "required": False, "description": "New status"},
                        "description": {"type": "string", "required": False, "description": "New description"},
                        "sla_minutes": {"type": "number", "required": False, "description": "New SLA"},
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="create_order",
                    description="Create a new customer order for a service",
                    parameters={
                        "service_id": {"type": "string", "required": True, "description": "Service to order"},
                        "customer_id": {"type": "string", "required": True, "description": "Customer identifier"},
                        "params": {"type": "object", "required": False, "description": "Parameters for the service"},
                        "priority": {"type": "string", "required": False, "description": "Order priority: normal, rush"},
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="fulfill_order",
                    description="Execute the underlying skill to fulfill an order",
                    parameters={
                        "order_id": {"type": "string", "required": True, "description": "Order ID to fulfill"},
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="complete_order",
                    description="Mark an order as completed with actual cost",
                    parameters={
                        "order_id": {"type": "string", "required": True, "description": "Order ID to complete"},
                        "actual_cost": {"type": "number", "required": False, "description": "Actual cost incurred"},
                        "notes": {"type": "string", "required": False, "description": "Completion notes"},
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="revenue_report",
                    description="Generate revenue, cost, and profit report",
                    parameters={
                        "days": {"type": "number", "required": False, "description": "Report period in days (default 30)"},
                        "service_id": {"type": "string", "required": False, "description": "Filter by service"},
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="discover_services",
                    description="Auto-discover potential services from installed skills",
                    parameters={},
                    estimated_cost=0,
                ),
                SkillAction(
                    name="adjust_pricing",
                    description="Adjust service pricing based on demand and margins",
                    parameters={
                        "service_id": {"type": "string", "required": True, "description": "Service to adjust"},
                        "strategy": {"type": "string", "required": False, "description": "Strategy: margin_target, demand_based, competitive"},
                        "target_margin": {"type": "number", "required": False, "description": "Target margin percentage (0-100)"},
                    },
                    estimated_cost=0,
                ),
            ],
        )

    async def execute(self, action: str, params: Dict) -> SkillResult:
        actions = {
            "create_service": self._create_service,
            "list_services": self._list_services,
            "update_service": self._update_service,
            "create_order": self._create_order,
            "fulfill_order": self._fulfill_order,
            "complete_order": self._complete_order,
            "revenue_report": self._revenue_report,
            "discover_services": self._discover_services,
            "adjust_pricing": self._adjust_pricing,
        }

        if action not in actions:
            return SkillResult(
                success=False,
                message=f"Unknown action: {action}. Available: {list(actions.keys())}",
            )

        try:
            return await actions[action](params)
        except Exception as e:
            return SkillResult(success=False, message=f"Error in {action}: {str(e)}")

    async def _create_service(self, params: Dict) -> SkillResult:
        """Create a new sellable service."""
        name = params.get("name")
        description = params.get("description")
        skill_id = params.get("skill_id")
        action = params.get("action")
        price = params.get("price")

        if not all([name, description, skill_id, action, price is not None]):
            return SkillResult(
                success=False,
                message="Required: name, description, skill_id, action, price",
            )

        if price < 0:
            return SkillResult(success=False, message="Price must be non-negative")

        state = self._load()

        if len(state["services"]) >= MAX_SERVICES:
            return SkillResult(success=False, message=f"Service limit reached ({MAX_SERVICES})")

        # Check for duplicate names
        existing_names = {s["name"].lower() for s in state["services"]}
        if name.lower() in existing_names:
            return SkillResult(success=False, message=f"Service '{name}' already exists")

        service = {
            "id": str(uuid.uuid4())[:12],
            "name": name,
            "description": description,
            "skill_id": skill_id,
            "action": action,
            "price": float(price),
            "pricing_model": params.get("pricing_model", "fixed"),
            "estimated_cost": float(params.get("estimated_cost", 0)),
            "sla_minutes": int(params.get("sla_minutes", 60)),
            "tags": params.get("tags", []),
            "status": "active",
            "created_at": datetime.now().isoformat(),
            "orders_count": 0,
            "revenue_total": 0.0,
            "cost_total": 0.0,
        }

        if service["pricing_model"] not in PRICING_MODELS:
            return SkillResult(
                success=False,
                message=f"Invalid pricing model. Use: {PRICING_MODELS}",
            )

        state["services"].append(service)
        self._save(state)

        margin = ((price - service["estimated_cost"]) / price * 100) if price > 0 else 0

        return SkillResult(
            success=True,
            message=f"Service '{name}' created at ${price:.2f} (est. margin: {margin:.0f}%)",
            data={"service": service},
            revenue=0,
        )

    async def _list_services(self, params: Dict) -> SkillResult:
        """List services with optional filtering."""
        state = self._load()
        services = state["services"]

        status_filter = params.get("status")
        if status_filter:
            services = [s for s in services if s["status"] == status_filter]

        tag_filter = params.get("tag")
        if tag_filter:
            services = [s for s in services if tag_filter in s.get("tags", [])]

        # Calculate metrics for each service
        for svc in services:
            revenue = svc.get("revenue_total", 0)
            cost = svc.get("cost_total", 0)
            orders = svc.get("orders_count", 0)
            svc["profit"] = revenue - cost
            svc["margin_pct"] = ((revenue - cost) / revenue * 100) if revenue > 0 else 0
            svc["avg_revenue_per_order"] = revenue / orders if orders > 0 else 0

        return SkillResult(
            success=True,
            message=f"Found {len(services)} service(s)",
            data={"services": services, "total": len(services)},
        )

    async def _update_service(self, params: Dict) -> SkillResult:
        """Update a service's details."""
        service_id = params.get("service_id")
        if not service_id:
            return SkillResult(success=False, message="service_id is required")

        state = self._load()
        service = self._find_service(state, service_id)
        if not service:
            return SkillResult(success=False, message=f"Service '{service_id}' not found")

        updates = {}
        if "price" in params:
            old_price = service["price"]
            service["price"] = float(params["price"])
            updates["price"] = f"${old_price:.2f} → ${service['price']:.2f}"

            # Log pricing adjustment
            state["pricing_adjustments"].append({
                "service_id": service_id,
                "old_price": old_price,
                "new_price": service["price"],
                "timestamp": datetime.now().isoformat(),
                "reason": "manual_update",
            })

        if "status" in params:
            if params["status"] not in SERVICE_STATUSES:
                return SkillResult(
                    success=False,
                    message=f"Invalid status. Use: {SERVICE_STATUSES}",
                )
            old_status = service["status"]
            service["status"] = params["status"]
            updates["status"] = f"{old_status} → {service['status']}"

        if "description" in params:
            service["description"] = params["description"]
            updates["description"] = "updated"

        if "sla_minutes" in params:
            service["sla_minutes"] = int(params["sla_minutes"])
            updates["sla_minutes"] = str(service["sla_minutes"])

        if not updates:
            return SkillResult(success=False, message="No valid fields to update")

        self._save(state)

        return SkillResult(
            success=True,
            message=f"Service updated: {updates}",
            data={"service": service, "updates": updates},
        )

    async def _create_order(self, params: Dict) -> SkillResult:
        """Create a customer order."""
        service_id = params.get("service_id")
        customer_id = params.get("customer_id")

        if not service_id or not customer_id:
            return SkillResult(success=False, message="service_id and customer_id are required")

        state = self._load()
        service = self._find_service(state, service_id)
        if not service:
            return SkillResult(success=False, message=f"Service '{service_id}' not found")

        if service["status"] != "active":
            return SkillResult(
                success=False,
                message=f"Service is {service['status']}, not accepting orders",
            )

        if len(state["orders"]) >= MAX_ORDERS:
            # Remove oldest completed orders
            state["orders"] = [
                o for o in state["orders"] if o["status"] not in ("completed", "failed", "cancelled")
            ][:MAX_ORDERS]

        order = {
            "id": str(uuid.uuid4())[:12],
            "service_id": service_id,
            "service_name": service["name"],
            "customer_id": customer_id,
            "params": params.get("params", {}),
            "priority": params.get("priority", "normal"),
            "price": service["price"],
            "status": "pending",
            "created_at": datetime.now().isoformat(),
            "started_at": None,
            "completed_at": None,
            "actual_cost": None,
            "notes": None,
        }

        state["orders"].append(order)
        service["orders_count"] = service.get("orders_count", 0) + 1
        self._save(state)

        return SkillResult(
            success=True,
            message=f"Order {order['id']} created for '{service['name']}' at ${order['price']:.2f}",
            data={"order": order},
        )

    async def _fulfill_order(self, params: Dict) -> SkillResult:
        """Fulfill an order by executing the underlying skill."""
        order_id = params.get("order_id")
        if not order_id:
            return SkillResult(success=False, message="order_id is required")

        state = self._load()
        order = self._find_order(state, order_id)
        if not order:
            return SkillResult(success=False, message=f"Order '{order_id}' not found")

        if order["status"] != "pending":
            return SkillResult(
                success=False,
                message=f"Order is '{order['status']}', expected 'pending'",
            )

        service = self._find_service(state, order["service_id"])
        if not service:
            return SkillResult(success=False, message="Service no longer exists")

        # Mark order as in progress
        order["status"] = "in_progress"
        order["started_at"] = datetime.now().isoformat()
        self._save(state)

        # Execute the underlying skill via context
        if self.context:
            result = await self.context.call_skill(
                service["skill_id"],
                service["action"],
                order.get("params", {}),
            )

            if result.success:
                return SkillResult(
                    success=True,
                    message=f"Order {order_id} fulfilled successfully",
                    data={"order": order, "result": result.data},
                    cost=result.cost,
                )
            else:
                order["status"] = "failed"
                order["notes"] = f"Fulfillment failed: {result.message}"
                self._save(state)
                return SkillResult(
                    success=False,
                    message=f"Fulfillment failed: {result.message}",
                    data={"order": order},
                )
        else:
            # No context - return order for manual fulfillment
            return SkillResult(
                success=True,
                message=f"Order {order_id} marked in_progress (no skill context for auto-fulfillment)",
                data={"order": order},
            )

    async def _complete_order(self, params: Dict) -> SkillResult:
        """Mark an order as completed and record revenue."""
        order_id = params.get("order_id")
        if not order_id:
            return SkillResult(success=False, message="order_id is required")

        state = self._load()
        order = self._find_order(state, order_id)
        if not order:
            return SkillResult(success=False, message=f"Order '{order_id}' not found")

        if order["status"] not in ("pending", "in_progress"):
            return SkillResult(
                success=False,
                message=f"Cannot complete order in '{order['status']}' status",
            )

        actual_cost = float(params.get("actual_cost", 0))
        order["status"] = "completed"
        order["completed_at"] = datetime.now().isoformat()
        order["actual_cost"] = actual_cost
        order["notes"] = params.get("notes", "")

        # Update service revenue/cost totals
        service = self._find_service(state, order["service_id"])
        if service:
            service["revenue_total"] = service.get("revenue_total", 0) + order["price"]
            service["cost_total"] = service.get("cost_total", 0) + actual_cost

        # Log revenue event
        state["revenue_log"].append({
            "order_id": order_id,
            "service_id": order["service_id"],
            "revenue": order["price"],
            "cost": actual_cost,
            "profit": order["price"] - actual_cost,
            "customer_id": order["customer_id"],
            "timestamp": datetime.now().isoformat(),
        })

        self._save(state)

        profit = order["price"] - actual_cost
        margin = (profit / order["price"] * 100) if order["price"] > 0 else 0

        return SkillResult(
            success=True,
            message=f"Order {order_id} completed. Revenue: ${order['price']:.2f}, Cost: ${actual_cost:.2f}, Profit: ${profit:.2f} ({margin:.0f}%)",
            data={"order": order, "profit": profit, "margin_pct": margin},
            revenue=order["price"],
            cost=actual_cost,
        )

    async def _revenue_report(self, params: Dict) -> SkillResult:
        """Generate a revenue report."""
        state = self._load()
        days = int(params.get("days", 30))
        service_filter = params.get("service_id")

        cutoff = (datetime.now() - timedelta(days=days)).isoformat()

        # Filter revenue log
        entries = state.get("revenue_log", [])
        entries = [e for e in entries if e.get("timestamp", "") >= cutoff]

        if service_filter:
            entries = [e for e in entries if e.get("service_id") == service_filter]

        total_revenue = sum(e.get("revenue", 0) for e in entries)
        total_cost = sum(e.get("cost", 0) for e in entries)
        total_profit = total_revenue - total_cost
        order_count = len(entries)

        # Per-service breakdown
        by_service = {}
        for e in entries:
            sid = e.get("service_id", "unknown")
            if sid not in by_service:
                by_service[sid] = {"revenue": 0, "cost": 0, "orders": 0}
            by_service[sid]["revenue"] += e.get("revenue", 0)
            by_service[sid]["cost"] += e.get("cost", 0)
            by_service[sid]["orders"] += 1

        for sid, data in by_service.items():
            data["profit"] = data["revenue"] - data["cost"]
            data["margin_pct"] = (
                (data["profit"] / data["revenue"] * 100) if data["revenue"] > 0 else 0
            )

        # Top services by profit
        top_services = sorted(
            by_service.items(), key=lambda x: x[1]["profit"], reverse=True
        )[:5]

        # Pending orders
        pending_orders = [
            o for o in state["orders"]
            if o["status"] in ("pending", "in_progress")
        ]
        pending_revenue = sum(o["price"] for o in pending_orders)

        report = {
            "period_days": days,
            "total_revenue": total_revenue,
            "total_cost": total_cost,
            "total_profit": total_profit,
            "margin_pct": (total_profit / total_revenue * 100) if total_revenue > 0 else 0,
            "order_count": order_count,
            "avg_order_value": total_revenue / order_count if order_count > 0 else 0,
            "by_service": by_service,
            "top_services": [{"service_id": s[0], **s[1]} for s in top_services],
            "pending_orders": len(pending_orders),
            "pending_revenue": pending_revenue,
        }

        overall_margin = report["margin_pct"]
        return SkillResult(
            success=True,
            message=f"Revenue report ({days}d): ${total_revenue:.2f} revenue, ${total_profit:.2f} profit ({overall_margin:.0f}% margin), {order_count} orders",
            data={"report": report},
        )

    async def _discover_services(self, params: Dict) -> SkillResult:
        """Auto-discover potential services from installed skills."""
        if not self.context:
            return SkillResult(
                success=False,
                message="No skill context available for discovery",
            )

        state = self._load()
        existing_keys = {
            (s["skill_id"], s["action"]) for s in state["services"]
        }

        suggestions = []
        for skill_id in self.context.list_skills():
            # Skip internal/meta skills
            if skill_id in ("marketplace", "strategy", "self_eval", "self_modify", "replication"):
                continue

            skill = self.context.get_skill(skill_id)
            if not skill:
                continue

            for action in skill.get_actions():
                key = (skill_id, action.name)
                if key in existing_keys:
                    continue

                # Suggest a price based on estimated cost with markup
                suggested_price = max(action.estimated_cost * 3, 0.10)  # 3x markup, min $0.10

                suggestions.append({
                    "skill_id": skill_id,
                    "action": action.name,
                    "description": action.description,
                    "estimated_cost": action.estimated_cost,
                    "suggested_price": round(suggested_price, 2),
                    "parameters": action.parameters,
                })

        return SkillResult(
            success=True,
            message=f"Discovered {len(suggestions)} potential services from installed skills",
            data={"suggestions": suggestions, "count": len(suggestions)},
        )

    async def _adjust_pricing(self, params: Dict) -> SkillResult:
        """Adjust pricing based on strategy."""
        service_id = params.get("service_id")
        if not service_id:
            return SkillResult(success=False, message="service_id is required")

        state = self._load()
        service = self._find_service(state, service_id)
        if not service:
            return SkillResult(success=False, message=f"Service '{service_id}' not found")

        strategy = params.get("strategy", "margin_target")
        old_price = service["price"]
        new_price = old_price

        if strategy == "margin_target":
            target_margin = float(params.get("target_margin", 50))
            avg_cost = (
                service["cost_total"] / service["orders_count"]
                if service.get("orders_count", 0) > 0
                else service.get("estimated_cost", 0)
            )
            if avg_cost > 0:
                # price = cost / (1 - margin/100)
                new_price = avg_cost / (1 - target_margin / 100)
            else:
                return SkillResult(
                    success=False,
                    message="No cost data available for margin-based pricing",
                )

        elif strategy == "demand_based":
            orders = service.get("orders_count", 0)
            if orders > 10:
                new_price = old_price * 1.15  # 15% increase for high demand
            elif orders > 5:
                new_price = old_price * 1.05  # 5% increase
            elif orders == 0:
                new_price = old_price * 0.80  # 20% discount for no demand
            else:
                new_price = old_price * 0.95  # 5% discount for low demand

        elif strategy == "competitive":
            # Simple competitive: undercut by 10% from current
            new_price = old_price * 0.90

        else:
            return SkillResult(
                success=False,
                message=f"Unknown strategy: {strategy}. Use: margin_target, demand_based, competitive",
            )

        new_price = round(max(new_price, 0.01), 2)  # Min $0.01
        service["price"] = new_price

        adjustment = {
            "service_id": service_id,
            "old_price": old_price,
            "new_price": new_price,
            "strategy": strategy,
            "change_pct": ((new_price - old_price) / old_price * 100) if old_price > 0 else 0,
            "timestamp": datetime.now().isoformat(),
        }
        state["pricing_adjustments"].append(adjustment)
        self._save(state)

        direction = "↑" if new_price > old_price else "↓" if new_price < old_price else "="
        return SkillResult(
            success=True,
            message=f"Price adjusted {direction}: ${old_price:.2f} → ${new_price:.2f} ({strategy})",
            data={"adjustment": adjustment, "service": service},
        )

    def _find_service(self, state: Dict, service_id: str) -> Optional[Dict]:
        """Find a service by ID or name."""
        for s in state["services"]:
            if s["id"] == service_id or s["name"].lower() == service_id.lower():
                return s
        return None

    def _find_order(self, state: Dict, order_id: str) -> Optional[Dict]:
        """Find an order by ID."""
        for o in state["orders"]:
            if o["id"] == order_id:
                return o
        return None

    async def initialize(self) -> bool:
        """Initialize the marketplace skill."""
        self.initialized = True
        return True
