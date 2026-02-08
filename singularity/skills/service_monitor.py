#!/usr/bin/env python3
"""
ServiceMonitorSkill - Aggregate monitoring of deployed revenue-generating services.

Bridges the gap between raw metrics (ObservabilitySkill) and business decisions
by providing service-level monitoring focused on what matters for revenue:

  - Service uptime and availability tracking
  - Request volume and throughput per service
  - Revenue and cost tracking per service with profit margins
  - SLA compliance monitoring (latency, error rate, availability)
  - Customer usage patterns and top-customer identification
  - Auto-generated health reports and scaling recommendations
  - Alert integration for SLA breaches

This is the "dashboard" layer that turns raw numbers into actionable insights
for the revenue pillar. Without it, the agent can't answer critical questions:
  "Which services are most profitable?"
  "Are we meeting our SLAs?"
  "Which services should we scale up or retire?"

Works with:
  - MarketplaceSkill: gets service catalog and order data
  - RevenueServiceSkill: gets execution logs
  - PaymentSkill: gets payment/revenue data
  - ObservabilitySkill: queries time-series metrics
  - HealthMonitor: replica health data
  - EventBus: emits service.sla_breach, service.down events

Part of Revenue Generation pillar: operational intelligence for services.
"""

import json
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from .base import Skill, SkillResult, SkillManifest, SkillAction

MONITOR_FILE = Path(__file__).parent.parent / "data" / "service_monitor.json"

# Service health states
HEALTHY = "healthy"
DEGRADED = "degraded"
DOWN = "down"
UNKNOWN = "unknown"

# SLA types
SLA_AVAILABILITY = "availability"  # % uptime
SLA_LATENCY = "latency"           # p95 response time
SLA_ERROR_RATE = "error_rate"     # % failed requests
SLA_THROUGHPUT = "throughput"     # min requests/min

SLA_TYPES = [SLA_AVAILABILITY, SLA_LATENCY, SLA_ERROR_RATE, SLA_THROUGHPUT]

# Limits
MAX_SERVICES = 100
MAX_INCIDENTS = 500
MAX_SNAPSHOTS = 1000
MAX_CUSTOMERS_PER_SERVICE = 200


def _now_iso() -> str:
    return datetime.utcnow().isoformat()


def _default_data() -> Dict:
    return {
        "services": {},
        "incidents": [],
        "snapshots": [],
        "config": {
            "snapshot_interval_seconds": 300,  # 5 min
            "incident_auto_resolve_seconds": 1800,  # 30 min
            "default_sla": {
                "availability": 99.0,   # 99% uptime
                "latency_p95_ms": 5000, # 5s p95
                "error_rate_max": 5.0,  # 5% error rate
                "min_throughput_rpm": 0, # no min by default
            },
        },
        "stats": {
            "total_requests_tracked": 0,
            "total_revenue_tracked": 0.0,
            "total_incidents": 0,
            "total_sla_breaches": 0,
            "last_snapshot_at": None,
        },
    }


class ServiceMonitorSkill(Skill):
    """
    Aggregate monitoring for revenue-generating services.

    Provides service-level visibility: uptime, throughput, revenue,
    SLA compliance, and scaling recommendations across all deployed services.
    """

    def __init__(self, credentials: Dict[str, str] = None, data_path: Path = None):
        super().__init__(credentials)
        self._data_path = data_path or MONITOR_FILE
        self._ensure_data()

    def _ensure_data(self):
        self._data_path.parent.mkdir(parents=True, exist_ok=True)
        if not self._data_path.exists():
            self._save(_default_data())

    def _load(self) -> Dict:
        try:
            with open(self._data_path) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return _default_data()

    def _save(self, data: Dict):
        try:
            with open(self._data_path, "w") as f:
                json.dump(data, f, indent=2, default=str)
        except IOError:
            pass

    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="service_monitor",
            name="Service Monitor",
            version="1.0.0",
            category="revenue",
            description="Aggregate monitoring of deployed services: uptime, revenue, SLA compliance, and scaling recommendations",
            required_credentials=[],
            actions=self.get_actions(),
        )

    def get_actions(self) -> List[SkillAction]:
        return [
            SkillAction(
                name="register",
                description="Register a service for monitoring with SLA targets",
                parameters={
                    "service_id": {"type": "string", "required": True, "description": "Unique service identifier"},
                    "name": {"type": "string", "required": True, "description": "Human-readable service name"},
                    "sla": {"type": "object", "required": False, "description": "SLA targets (availability, latency_p95_ms, error_rate_max, min_throughput_rpm)"},
                    "cost_per_request": {"type": "number", "required": False, "description": "Estimated cost per request (USD)"},
                    "price_per_request": {"type": "number", "required": False, "description": "Price charged per request (USD)"},
                },
            ),
            SkillAction(
                name="record",
                description="Record a service request outcome (success/failure, latency, revenue)",
                parameters={
                    "service_id": {"type": "string", "required": True, "description": "Service identifier"},
                    "success": {"type": "boolean", "required": True, "description": "Whether the request succeeded"},
                    "latency_ms": {"type": "number", "required": False, "description": "Request latency in ms"},
                    "revenue": {"type": "number", "required": False, "description": "Revenue from this request (USD)"},
                    "cost": {"type": "number", "required": False, "description": "Cost of this request (USD)"},
                    "customer_id": {"type": "string", "required": False, "description": "Customer identifier"},
                },
            ),
            SkillAction(
                name="status",
                description="Get current status and health of a specific service or all services",
                parameters={
                    "service_id": {"type": "string", "required": False, "description": "Service ID (omit for all services)"},
                },
            ),
            SkillAction(
                name="dashboard",
                description="Generate a comprehensive dashboard with all services, revenue, SLA, and recommendations",
                parameters={
                    "time_window_hours": {"type": "number", "required": False, "description": "Time window for metrics (default 24h)"},
                },
            ),
            SkillAction(
                name="sla_check",
                description="Check SLA compliance for a service or all services, fire incidents for breaches",
                parameters={
                    "service_id": {"type": "string", "required": False, "description": "Service ID (omit for all)"},
                },
            ),
            SkillAction(
                name="incidents",
                description="List active and recent incidents",
                parameters={
                    "status": {"type": "string", "required": False, "description": "Filter by status: active, resolved, all (default: active)"},
                    "service_id": {"type": "string", "required": False, "description": "Filter by service"},
                },
            ),
            SkillAction(
                name="top_services",
                description="Rank services by revenue, profit, volume, or error rate",
                parameters={
                    "sort_by": {"type": "string", "required": False, "description": "Sort metric: revenue, profit, volume, error_rate (default: revenue)"},
                    "limit": {"type": "integer", "required": False, "description": "Number of results (default 10)"},
                },
            ),
            SkillAction(
                name="recommend",
                description="Generate scaling and optimization recommendations based on service performance",
                parameters={},
            ),
            SkillAction(
                name="configure",
                description="Update monitoring configuration (snapshot interval, SLA defaults, incident settings)",
                parameters={
                    "key": {"type": "string", "required": True, "description": "Config key to update"},
                    "value": {"type": "any", "required": True, "description": "New value"},
                },
            ),
        ]

    async def execute(self, action: str, params: Dict[str, Any]) -> SkillResult:
        actions = {
            "register": self._register,
            "record": self._record,
            "status": self._status,
            "dashboard": self._dashboard,
            "sla_check": self._sla_check,
            "incidents": self._incidents,
            "top_services": self._top_services,
            "recommend": self._recommend,
            "configure": self._configure,
        }
        handler = actions.get(action)
        if not handler:
            return SkillResult(success=False, message=f"Unknown action: {action}")
        return handler(params)

    def _register(self, params: Dict) -> SkillResult:
        """Register a service for monitoring."""
        service_id = params.get("service_id", "").strip()
        name = params.get("name", "").strip()
        if not service_id or not name:
            return SkillResult(success=False, message="service_id and name are required")

        data = self._load()
        if len(data["services"]) >= MAX_SERVICES and service_id not in data["services"]:
            return SkillResult(success=False, message=f"Maximum {MAX_SERVICES} services reached")

        default_sla = data["config"]["default_sla"]
        custom_sla = params.get("sla", {})
        sla = {**default_sla, **(custom_sla if isinstance(custom_sla, dict) else {})}

        already_exists = service_id in data["services"]
        data["services"][service_id] = {
            "name": name,
            "registered_at": data["services"].get(service_id, {}).get("registered_at", _now_iso()),
            "sla": sla,
            "cost_per_request": params.get("cost_per_request", 0.001),
            "price_per_request": params.get("price_per_request", 0.01),
            "health": UNKNOWN,
            "metrics": data["services"].get(service_id, {}).get("metrics", {
                "total_requests": 0,
                "successful_requests": 0,
                "failed_requests": 0,
                "total_revenue": 0.0,
                "total_cost": 0.0,
                "latencies_ms": [],  # Rolling window of recent latencies
                "requests_per_minute": [],  # Rolling window of RPM snapshots
                "customers": {},  # customer_id -> request_count
                "last_request_at": None,
                "uptime_checks": [],  # (timestamp, was_healthy) pairs
            }),
        }

        self._save(data)
        action = "updated" if already_exists else "registered"
        return SkillResult(
            success=True,
            message=f"Service '{name}' ({service_id}) {action} for monitoring",
            data={"service_id": service_id, "sla": sla, "action": action},
        )

    def _record(self, params: Dict) -> SkillResult:
        """Record a service request outcome."""
        service_id = params.get("service_id", "").strip()
        if not service_id:
            return SkillResult(success=False, message="service_id is required")

        success = params.get("success", True)
        latency_ms = params.get("latency_ms")
        revenue = params.get("revenue", 0.0)
        cost = params.get("cost", 0.0)
        customer_id = params.get("customer_id")

        data = self._load()
        if service_id not in data["services"]:
            return SkillResult(success=False, message=f"Service '{service_id}' not registered. Use 'register' first.")

        svc = data["services"][service_id]
        m = svc["metrics"]

        m["total_requests"] += 1
        if success:
            m["successful_requests"] += 1
        else:
            m["failed_requests"] += 1

        m["total_revenue"] += revenue
        m["total_cost"] += cost
        m["last_request_at"] = _now_iso()

        # Track latency (keep last 100)
        if latency_ms is not None:
            m["latencies_ms"].append(latency_ms)
            if len(m["latencies_ms"]) > 100:
                m["latencies_ms"] = m["latencies_ms"][-100:]

        # Track customer usage
        if customer_id:
            customers = m.get("customers", {})
            customers[customer_id] = customers.get(customer_id, 0) + 1
            # Trim to max
            if len(customers) > MAX_CUSTOMERS_PER_SERVICE:
                # Keep top customers by count
                sorted_c = sorted(customers.items(), key=lambda x: x[1], reverse=True)
                customers = dict(sorted_c[:MAX_CUSTOMERS_PER_SERVICE])
            m["customers"] = customers

        # Update health based on recent performance
        svc["health"] = self._compute_health(svc)

        # Update global stats
        data["stats"]["total_requests_tracked"] += 1
        data["stats"]["total_revenue_tracked"] += revenue

        self._save(data)
        return SkillResult(
            success=True,
            message=f"Recorded {'success' if success else 'failure'} for {service_id}",
            data={
                "service_id": service_id,
                "health": svc["health"],
                "total_requests": m["total_requests"],
                "revenue_total": m["total_revenue"],
            },
            revenue=revenue,
            cost=cost,
        )

    def _compute_health(self, svc: Dict) -> str:
        """Compute service health from recent metrics."""
        m = svc["metrics"]
        total = m["total_requests"]
        if total == 0:
            return UNKNOWN

        error_rate = (m["failed_requests"] / total) * 100 if total > 0 else 0
        sla = svc.get("sla", {})
        max_error = sla.get("error_rate_max", 5.0)

        # Check latency SLA
        latency_ok = True
        if m["latencies_ms"] and sla.get("latency_p95_ms"):
            p95 = self._percentile(m["latencies_ms"], 95)
            if p95 > sla["latency_p95_ms"]:
                latency_ok = False

        if error_rate > max_error * 2 or (not latency_ok and error_rate > max_error):
            return DOWN
        elif error_rate > max_error or not latency_ok:
            return DEGRADED
        return HEALTHY

    def _percentile(self, values: List[float], pct: float) -> float:
        """Compute percentile from a list of values."""
        if not values:
            return 0.0
        sorted_v = sorted(values)
        idx = int(len(sorted_v) * pct / 100)
        idx = min(idx, len(sorted_v) - 1)
        return sorted_v[idx]

    def _status(self, params: Dict) -> SkillResult:
        """Get status of one or all services."""
        data = self._load()
        service_id = params.get("service_id")

        if service_id:
            svc = data["services"].get(service_id)
            if not svc:
                return SkillResult(success=False, message=f"Service '{service_id}' not found")
            m = svc["metrics"]
            total = m["total_requests"]
            return SkillResult(
                success=True,
                message=f"Service '{svc['name']}' is {svc['health']}",
                data={
                    "service_id": service_id,
                    "name": svc["name"],
                    "health": svc["health"],
                    "total_requests": total,
                    "success_rate": round((m["successful_requests"] / total) * 100, 2) if total > 0 else 0,
                    "error_rate": round((m["failed_requests"] / total) * 100, 2) if total > 0 else 0,
                    "total_revenue": round(m["total_revenue"], 4),
                    "total_cost": round(m["total_cost"], 4),
                    "profit": round(m["total_revenue"] - m["total_cost"], 4),
                    "avg_latency_ms": round(sum(m["latencies_ms"]) / len(m["latencies_ms"]), 1) if m["latencies_ms"] else None,
                    "p95_latency_ms": round(self._percentile(m["latencies_ms"], 95), 1) if m["latencies_ms"] else None,
                    "unique_customers": len(m.get("customers", {})),
                    "last_request_at": m["last_request_at"],
                    "sla": svc["sla"],
                },
            )

        # All services overview
        services = []
        for sid, svc in data["services"].items():
            m = svc["metrics"]
            total = m["total_requests"]
            services.append({
                "service_id": sid,
                "name": svc["name"],
                "health": svc["health"],
                "total_requests": total,
                "error_rate": round((m["failed_requests"] / total) * 100, 2) if total > 0 else 0,
                "revenue": round(m["total_revenue"], 4),
                "profit": round(m["total_revenue"] - m["total_cost"], 4),
            })

        return SkillResult(
            success=True,
            message=f"{len(services)} services monitored",
            data={
                "services": services,
                "total_revenue": round(data["stats"]["total_revenue_tracked"], 4),
                "total_requests": data["stats"]["total_requests_tracked"],
            },
        )

    def _dashboard(self, params: Dict) -> SkillResult:
        """Generate comprehensive monitoring dashboard."""
        data = self._load()
        time_window_hours = params.get("time_window_hours", 24)

        services_summary = []
        total_revenue = 0.0
        total_cost = 0.0
        total_requests = 0
        total_errors = 0
        health_counts = {HEALTHY: 0, DEGRADED: 0, DOWN: 0, UNKNOWN: 0}

        for sid, svc in data["services"].items():
            m = svc["metrics"]
            total = m["total_requests"]
            revenue = m["total_revenue"]
            cost = m["total_cost"]
            profit = revenue - cost
            margin = (profit / revenue * 100) if revenue > 0 else 0.0

            total_revenue += revenue
            total_cost += cost
            total_requests += total
            total_errors += m["failed_requests"]
            health_counts[svc.get("health", UNKNOWN)] += 1

            services_summary.append({
                "service_id": sid,
                "name": svc["name"],
                "health": svc["health"],
                "requests": total,
                "success_rate": round((m["successful_requests"] / total) * 100, 2) if total > 0 else 0,
                "revenue": round(revenue, 4),
                "cost": round(cost, 4),
                "profit": round(profit, 4),
                "margin_pct": round(margin, 1),
                "avg_latency_ms": round(sum(m["latencies_ms"]) / len(m["latencies_ms"]), 1) if m["latencies_ms"] else None,
                "p95_latency_ms": round(self._percentile(m["latencies_ms"], 95), 1) if m["latencies_ms"] else None,
                "unique_customers": len(m.get("customers", {})),
                "sla_status": self._check_sla_single(svc),
            })

        # Sort by revenue descending
        services_summary.sort(key=lambda s: s["revenue"], reverse=True)

        overall_error_rate = round((total_errors / total_requests) * 100, 2) if total_requests > 0 else 0
        total_profit = total_revenue - total_cost
        overall_margin = round((total_profit / total_revenue) * 100, 1) if total_revenue > 0 else 0

        return SkillResult(
            success=True,
            message=f"Dashboard: {len(services_summary)} services, ${total_revenue:.2f} revenue, {health_counts[HEALTHY]} healthy",
            data={
                "overview": {
                    "total_services": len(services_summary),
                    "health_breakdown": health_counts,
                    "total_requests": total_requests,
                    "total_errors": total_errors,
                    "overall_error_rate": overall_error_rate,
                    "total_revenue": round(total_revenue, 4),
                    "total_cost": round(total_cost, 4),
                    "total_profit": round(total_profit, 4),
                    "overall_margin_pct": overall_margin,
                    "total_incidents": data["stats"]["total_incidents"],
                    "total_sla_breaches": data["stats"]["total_sla_breaches"],
                },
                "services": services_summary,
                "time_window_hours": time_window_hours,
                "generated_at": _now_iso(),
            },
        )

    def _check_sla_single(self, svc: Dict) -> Dict:
        """Check SLA compliance for a single service."""
        m = svc["metrics"]
        sla = svc.get("sla", {})
        total = m["total_requests"]
        results = {}

        # Availability check
        if total > 0 and "availability" in sla:
            actual_avail = (m["successful_requests"] / total) * 100
            target = sla["availability"]
            results["availability"] = {
                "target": target,
                "actual": round(actual_avail, 2),
                "compliant": actual_avail >= target,
            }

        # Latency check
        if m["latencies_ms"] and "latency_p95_ms" in sla:
            p95 = self._percentile(m["latencies_ms"], 95)
            target = sla["latency_p95_ms"]
            results["latency_p95"] = {
                "target_ms": target,
                "actual_ms": round(p95, 1),
                "compliant": p95 <= target,
            }

        # Error rate check
        if total > 0 and "error_rate_max" in sla:
            actual_err = (m["failed_requests"] / total) * 100
            target = sla["error_rate_max"]
            results["error_rate"] = {
                "target_max_pct": target,
                "actual_pct": round(actual_err, 2),
                "compliant": actual_err <= target,
            }

        all_compliant = all(r.get("compliant", True) for r in results.values())
        return {"compliant": all_compliant, "checks": results}

    def _sla_check(self, params: Dict) -> SkillResult:
        """Check SLA compliance and create incidents for breaches."""
        data = self._load()
        service_id = params.get("service_id")

        services_to_check = {}
        if service_id:
            if service_id not in data["services"]:
                return SkillResult(success=False, message=f"Service '{service_id}' not found")
            services_to_check[service_id] = data["services"][service_id]
        else:
            services_to_check = data["services"]

        results = {}
        breaches = []
        for sid, svc in services_to_check.items():
            sla_result = self._check_sla_single(svc)
            results[sid] = sla_result

            if not sla_result["compliant"]:
                # Create incident for SLA breach
                violated = [k for k, v in sla_result["checks"].items() if not v.get("compliant", True)]
                incident = {
                    "id": str(uuid.uuid4())[:8],
                    "service_id": sid,
                    "type": "sla_breach",
                    "violated": violated,
                    "details": sla_result["checks"],
                    "created_at": _now_iso(),
                    "status": "active",
                    "resolved_at": None,
                }
                data["incidents"].append(incident)
                data["stats"]["total_incidents"] += 1
                data["stats"]["total_sla_breaches"] += 1
                breaches.append({"service_id": sid, "violated": violated, "incident_id": incident["id"]})

        # Trim incidents
        if len(data["incidents"]) > MAX_INCIDENTS:
            data["incidents"] = data["incidents"][-MAX_INCIDENTS:]

        self._save(data)

        n_compliant = sum(1 for r in results.values() if r["compliant"])
        n_total = len(results)
        return SkillResult(
            success=True,
            message=f"SLA check: {n_compliant}/{n_total} services compliant, {len(breaches)} breaches",
            data={
                "results": results,
                "breaches": breaches,
                "compliant_count": n_compliant,
                "total_checked": n_total,
            },
        )

    def _incidents(self, params: Dict) -> SkillResult:
        """List incidents filtered by status and/or service."""
        data = self._load()
        status_filter = params.get("status", "active")
        service_filter = params.get("service_id")

        incidents = data.get("incidents", [])

        if status_filter and status_filter != "all":
            incidents = [i for i in incidents if i.get("status") == status_filter]
        if service_filter:
            incidents = [i for i in incidents if i.get("service_id") == service_filter]

        # Return most recent first
        incidents = list(reversed(incidents[-50:]))

        return SkillResult(
            success=True,
            message=f"{len(incidents)} incidents found",
            data={"incidents": incidents, "filter": {"status": status_filter, "service_id": service_filter}},
        )

    def _top_services(self, params: Dict) -> SkillResult:
        """Rank services by a chosen metric."""
        data = self._load()
        sort_by = params.get("sort_by", "revenue")
        limit = min(params.get("limit", 10), MAX_SERVICES)

        services = []
        for sid, svc in data["services"].items():
            m = svc["metrics"]
            total = m["total_requests"]
            revenue = m["total_revenue"]
            cost = m["total_cost"]
            profit = revenue - cost
            error_rate = (m["failed_requests"] / total * 100) if total > 0 else 0

            services.append({
                "service_id": sid,
                "name": svc["name"],
                "health": svc["health"],
                "revenue": round(revenue, 4),
                "profit": round(profit, 4),
                "margin_pct": round((profit / revenue * 100), 1) if revenue > 0 else 0,
                "volume": total,
                "error_rate": round(error_rate, 2),
                "unique_customers": len(m.get("customers", {})),
            })

        # Sort
        sort_key = {
            "revenue": lambda s: s["revenue"],
            "profit": lambda s: s["profit"],
            "volume": lambda s: s["volume"],
            "error_rate": lambda s: s["error_rate"],
        }.get(sort_by, lambda s: s["revenue"])

        # For error_rate, higher is worse, so we still sort descending to show worst first
        services.sort(key=sort_key, reverse=True)
        services = services[:limit]

        return SkillResult(
            success=True,
            message=f"Top {len(services)} services by {sort_by}",
            data={"ranking": services, "sort_by": sort_by},
        )

    def _recommend(self, params: Dict) -> SkillResult:
        """Generate optimization and scaling recommendations."""
        data = self._load()
        recommendations = []

        for sid, svc in data["services"].items():
            m = svc["metrics"]
            total = m["total_requests"]
            if total == 0:
                recommendations.append({
                    "service_id": sid,
                    "type": "attention",
                    "priority": "low",
                    "message": f"Service '{svc['name']}' has no requests. Consider promoting or retiring.",
                })
                continue

            revenue = m["total_revenue"]
            cost = m["total_cost"]
            profit = revenue - cost
            error_rate = (m["failed_requests"] / total) * 100

            # High error rate
            sla_max = svc.get("sla", {}).get("error_rate_max", 5.0)
            if error_rate > sla_max:
                recommendations.append({
                    "service_id": sid,
                    "type": "fix",
                    "priority": "high",
                    "message": f"Service '{svc['name']}' has {error_rate:.1f}% error rate (SLA: {sla_max}%). Investigate and fix failures.",
                })

            # Negative profit
            if profit < 0 and total > 10:
                recommendations.append({
                    "service_id": sid,
                    "type": "pricing",
                    "priority": "high",
                    "message": f"Service '{svc['name']}' is unprofitable (${profit:.2f}). Increase pricing or reduce costs.",
                })

            # High profit, consider scaling
            margin = (profit / revenue * 100) if revenue > 0 else 0
            if margin > 50 and total > 20:
                recommendations.append({
                    "service_id": sid,
                    "type": "scale",
                    "priority": "medium",
                    "message": f"Service '{svc['name']}' has {margin:.0f}% margin with {total} requests. Consider promoting to increase volume.",
                })

            # Low margin but high volume
            if 0 < margin < 10 and total > 50:
                recommendations.append({
                    "service_id": sid,
                    "type": "optimize",
                    "priority": "medium",
                    "message": f"Service '{svc['name']}' has high volume ({total}) but low margin ({margin:.0f}%). Optimize costs or raise prices.",
                })

            # Latency issues
            if m["latencies_ms"]:
                p95 = self._percentile(m["latencies_ms"], 95)
                sla_lat = svc.get("sla", {}).get("latency_p95_ms", 5000)
                if p95 > sla_lat:
                    recommendations.append({
                        "service_id": sid,
                        "type": "performance",
                        "priority": "high",
                        "message": f"Service '{svc['name']}' p95 latency is {p95:.0f}ms (SLA: {sla_lat}ms). Optimize performance.",
                    })

            # Customer concentration risk
            customers = m.get("customers", {})
            if customers and total > 20:
                top_customer_reqs = max(customers.values()) if customers else 0
                concentration = (top_customer_reqs / total) * 100
                if concentration > 80:
                    recommendations.append({
                        "service_id": sid,
                        "type": "diversify",
                        "priority": "medium",
                        "message": f"Service '{svc['name']}' has {concentration:.0f}% customer concentration. Diversify customer base.",
                    })

        # Sort by priority
        priority_order = {"high": 0, "medium": 1, "low": 2}
        recommendations.sort(key=lambda r: priority_order.get(r["priority"], 3))

        return SkillResult(
            success=True,
            message=f"{len(recommendations)} recommendations generated",
            data={"recommendations": recommendations},
        )

    def _configure(self, params: Dict) -> SkillResult:
        """Update monitoring configuration."""
        key = params.get("key", "").strip()
        value = params.get("value")
        if not key:
            return SkillResult(success=False, message="key is required")

        data = self._load()
        config = data["config"]

        # Allow nested key access with dot notation
        if "." in key:
            parts = key.split(".", 1)
            if parts[0] in config and isinstance(config[parts[0]], dict):
                config[parts[0]][parts[1]] = value
            else:
                return SkillResult(success=False, message=f"Unknown config section: {parts[0]}")
        elif key in config:
            config[key] = value
        else:
            return SkillResult(success=False, message=f"Unknown config key: {key}. Valid: {list(config.keys())}")

        self._save(data)
        return SkillResult(
            success=True,
            message=f"Config '{key}' updated to {value}",
            data={"key": key, "value": value},
        )
