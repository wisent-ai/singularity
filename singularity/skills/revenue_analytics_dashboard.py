#!/usr/bin/env python3
"""
RevenueAnalyticsDashboardSkill - Unified revenue analytics across all services.

The agent has multiple revenue-generating skills (TaskPricing, PricingBridge,
RevenueServices, UsageTracking, Marketplace, ServiceHosting, RevenueCatalog)
but each tracks revenue independently. There is no unified view of:
- Total revenue across all sources
- Which services/skills are most profitable
- Revenue trends over time
- Cost efficiency and margin health
- Customer concentration risk
- Revenue forecasting

This dashboard aggregates revenue data from ALL revenue skills into a single
analytics view, enabling the agent to make informed decisions about:
- Which services to invest in vs deprecate
- Pricing adjustments needed
- Revenue trajectory vs compute costs
- Customer acquisition priorities

Data sources:
  - task_pricing.json: quotes, revenue_summary, calibration
  - pricing_service_bridge.json: task_quotes, revenue_log, stats
  - revenue_services_log.json: execution log entries
  - usage_tracking.json: customers, usage_records, invoices
  - marketplace.json: orders, revenue_log
  - hosted_services.json: services with revenue data
  - revenue_catalog.json: products, deployments
  - database_revenue_bridge.json: paid data analysis, schema design, reports
  - http_revenue_bridge.json: paid HTTP proxy, webhook relay, health checks
  - usage.json (api_marketplace): external API brokering, subscriptions

Pillar: Revenue Generation (primary), Goal Setting (data-driven prioritization)
"""

import json
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any

from .base import Skill, SkillResult, SkillManifest, SkillAction

DATA_DIR = Path(__file__).parent.parent / "data"
DASHBOARD_FILE = DATA_DIR / "revenue_analytics_dashboard.json"

# All revenue data source files
SOURCE_FILES = {
    "task_pricing": DATA_DIR / "task_pricing.json",
    "pricing_bridge": DATA_DIR / "pricing_service_bridge.json",
    "revenue_services": DATA_DIR / "revenue_services_log.json",
    "usage_tracking": DATA_DIR / "usage_tracking.json",
    "marketplace": DATA_DIR / "marketplace.json",
    "hosted_services": DATA_DIR / "hosted_services.json",
    "revenue_catalog": DATA_DIR / "revenue_catalog.json",
    "database_revenue_bridge": DATA_DIR / "database_revenue_bridge.json",
    "http_revenue_bridge": DATA_DIR / "http_revenue_bridge.json",
    "api_marketplace": DATA_DIR / "usage.json",
}

MAX_SNAPSHOTS = 200


def _now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _load_json(path: Path) -> Optional[Any]:
    try:
        if path.exists():
            return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        pass
    return None


def _save_json(path: Path, data: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str))


def _parse_ts(ts_str: str) -> Optional[float]:
    """Parse ISO timestamp string to epoch seconds."""
    if not ts_str:
        return None
    try:
        ts_str = ts_str.rstrip("Z")
        return datetime.fromisoformat(ts_str).timestamp()
    except (ValueError, TypeError):
        return None


class RevenueAnalyticsDashboardSkill(Skill):
    """
    Unified revenue analytics across all revenue-generating skills.

    Aggregates data from TaskPricing, PricingBridge, RevenueServices,
    UsageTracking, Marketplace, ServiceHosting, and RevenueCatalog into
    a single comprehensive analytics view.
    """

    def __init__(self, credentials: Dict[str, str] = None):
        super().__init__(credentials)
        self._ensure_data()

    def _ensure_data(self):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        if not DASHBOARD_FILE.exists():
            _save_json(DASHBOARD_FILE, self._default_state())

    def _default_state(self) -> Dict:
        return {
            "snapshots": [],
            "config": {
                "snapshot_interval_hours": 1,
                "compute_cost_per_hour": 0.10,
                "revenue_target_daily": 1.00,
            },
            "created_at": _now_iso(),
            "last_updated": _now_iso(),
        }

    def _load(self) -> Dict:
        data = _load_json(DASHBOARD_FILE)
        return data if data else self._default_state()

    def _save(self, data: Dict):
        if len(data.get("snapshots", [])) > MAX_SNAPSHOTS:
            data["snapshots"] = data["snapshots"][-MAX_SNAPSHOTS:]
        data["last_updated"] = _now_iso()
        _save_json(DASHBOARD_FILE, data)

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="revenue_analytics_dashboard",
            name="Revenue Analytics Dashboard",
            version="1.0.0",
            category="revenue",
            description="Unified revenue analytics aggregated across all revenue-generating skills",
            required_credentials=[],
            actions=[
                SkillAction(
                    name="overview",
                    description="Comprehensive revenue overview aggregated from all sources",
                    parameters={},
                ),
                SkillAction(
                    name="by_source",
                    description="Revenue breakdown by source skill",
                    parameters={},
                ),
                SkillAction(
                    name="profitability",
                    description="Profitability analysis: margins, costs, efficiency",
                    parameters={},
                ),
                SkillAction(
                    name="customers",
                    description="Customer analytics: count, concentration, top customers",
                    parameters={},
                ),
                SkillAction(
                    name="trends",
                    description="Revenue trends and trajectory analysis",
                    parameters={
                        "window_hours": {"type": "number", "required": False, "description": "Hours to analyze (default 24)"},
                    },
                ),
                SkillAction(
                    name="forecast",
                    description="Revenue forecast based on historical data",
                    parameters={
                        "days_ahead": {"type": "integer", "required": False, "description": "Days to forecast (default 7)"},
                    },
                ),
                SkillAction(
                    name="snapshot",
                    description="Take a point-in-time revenue snapshot for trend tracking",
                    parameters={},
                ),
                SkillAction(
                    name="recommendations",
                    description="AI-generated revenue optimization recommendations",
                    parameters={},
                ),
                SkillAction(
                    name="configure",
                    description="Update dashboard configuration",
                    parameters={
                        "compute_cost_per_hour": {"type": "number", "required": False, "description": "Compute cost per hour in USD"},
                        "revenue_target_daily": {"type": "number", "required": False, "description": "Daily revenue target in USD"},
                    },
                ),
                SkillAction(
                    name="status",
                    description="Dashboard status and data source health",
                    parameters={},
                ),
            ],
        )

    async def execute(self, action: str, params: Dict[str, Any]) -> SkillResult:
        actions = {
            "overview": self._overview,
            "by_source": self._by_source,
            "profitability": self._profitability,
            "customers": self._customers,
            "trends": self._trends,
            "forecast": self._forecast,
            "snapshot": self._snapshot,
            "recommendations": self._recommendations,
            "configure": self._configure,
            "status": self._status,
        }
        handler = actions.get(action)
        if not handler:
            return SkillResult(success=False, message=f"Unknown action: {action}")
        return handler(params)

    # ── Data collection from all sources ──────────────────────────────

    def _collect_all_revenue_data(self) -> Dict:
        """Collect and normalize revenue data from all sources."""
        collected = {
            "sources_available": [],
            "sources_missing": [],
            "total_revenue": 0.0,
            "total_cost": 0.0,
            "total_profit": 0.0,
            "total_transactions": 0,
            "by_source": {},
            "customers": set(),
            "transactions": [],
        }

        # 1. TaskPricing
        tp = _load_json(SOURCE_FILES["task_pricing"])
        if tp:
            collected["sources_available"].append("task_pricing")
            rs = tp.get("revenue_summary", {})
            rev = rs.get("total_revenue", 0.0)
            cost = rs.get("total_actual_cost", 0.0)
            count = rs.get("quote_count", 0)
            collected["by_source"]["task_pricing"] = {
                "revenue": rev, "cost": cost, "profit": rev - cost,
                "transactions": count, "acceptance_rate": rs.get("acceptance_rate", 0),
            }
            collected["total_revenue"] += rev
            collected["total_cost"] += cost
            collected["total_transactions"] += count
        else:
            collected["sources_missing"].append("task_pricing")

        # 2. PricingBridge
        pb = _load_json(SOURCE_FILES["pricing_bridge"])
        if pb:
            collected["sources_available"].append("pricing_bridge")
            stats = pb.get("stats", {})
            rev = stats.get("total_revenue_usd", 0.0)
            cost = stats.get("total_actual_usd", 0.0)
            count = stats.get("tasks_executed", 0)
            collected["by_source"]["pricing_bridge"] = {
                "revenue": rev, "cost": cost, "profit": stats.get("total_profit_usd", rev - cost),
                "transactions": count,
                "quotes_issued": stats.get("tasks_quoted", 0),
                "avg_margin_pct": stats.get("avg_margin_pct", 0),
            }
            collected["total_revenue"] += rev
            collected["total_cost"] += cost
            collected["total_transactions"] += count
            # Extract customers
            for q in pb.get("task_quotes", {}).values():
                cid = q.get("customer_id")
                if cid:
                    collected["customers"].add(cid)
        else:
            collected["sources_missing"].append("pricing_bridge")

        # 3. RevenueServices
        rs_data = _load_json(SOURCE_FILES["revenue_services"])
        if rs_data:
            collected["sources_available"].append("revenue_services")
            entries = rs_data if isinstance(rs_data, list) else rs_data.get("log", [])
            rev = sum(e.get("revenue", 0) for e in entries)
            cost = sum(e.get("cost", 0) for e in entries)
            collected["by_source"]["revenue_services"] = {
                "revenue": rev, "cost": cost, "profit": rev - cost,
                "transactions": len(entries),
                "services_used": len(set(e.get("service", "") for e in entries)),
            }
            collected["total_revenue"] += rev
            collected["total_cost"] += cost
            collected["total_transactions"] += len(entries)
        else:
            collected["sources_missing"].append("revenue_services")

        # 4. UsageTracking
        ut = _load_json(SOURCE_FILES["usage_tracking"])
        if ut:
            collected["sources_available"].append("usage_tracking")
            customers = ut.get("customers", {})
            total_rev = sum(c.get("total_revenue", 0) for c in customers.values())
            total_cost = sum(c.get("total_cost", 0) for c in customers.values())
            total_reqs = sum(c.get("total_requests", 0) for c in customers.values())
            collected["by_source"]["usage_tracking"] = {
                "revenue": total_rev, "cost": total_cost, "profit": total_rev - total_cost,
                "transactions": total_reqs,
                "customer_count": len(customers),
                "invoices_generated": len(ut.get("invoices", [])),
            }
            collected["total_revenue"] += total_rev
            collected["total_cost"] += total_cost
            collected["total_transactions"] += total_reqs
            for cid in customers:
                collected["customers"].add(cid)
        else:
            collected["sources_missing"].append("usage_tracking")

        # 5. Marketplace
        mp = _load_json(SOURCE_FILES["marketplace"])
        if mp:
            collected["sources_available"].append("marketplace")
            orders = mp.get("orders", [])
            rev_log = mp.get("revenue_log", [])
            rev = sum(e.get("price", e.get("revenue", 0)) for e in rev_log)
            cost = sum(e.get("actual_cost", e.get("cost", 0)) for e in rev_log)
            collected["by_source"]["marketplace"] = {
                "revenue": rev, "cost": cost, "profit": rev - cost,
                "transactions": len(orders),
                "services_listed": len(mp.get("services", [])),
            }
            collected["total_revenue"] += rev
            collected["total_cost"] += cost
            collected["total_transactions"] += len(orders)
        else:
            collected["sources_missing"].append("marketplace")

        # 6. ServiceHosting
        sh = _load_json(SOURCE_FILES["hosted_services"])
        if sh:
            collected["sources_available"].append("hosted_services")
            services = sh.get("services", {})
            rev = sum(s.get("total_revenue", 0) for s in services.values())
            reqs = sum(s.get("total_requests", 0) for s in services.values())
            collected["by_source"]["hosted_services"] = {
                "revenue": rev, "cost": 0.0, "profit": rev,
                "transactions": reqs,
                "services_hosted": len(services),
                "active_services": sum(1 for s in services.values() if s.get("status") == "active"),
            }
            collected["total_revenue"] += rev
            collected["total_transactions"] += reqs
        else:
            collected["sources_missing"].append("hosted_services")

        # 7. RevenueCatalog
        rc = _load_json(SOURCE_FILES["revenue_catalog"])
        if rc:
            collected["sources_available"].append("revenue_catalog")
            stats = rc.get("stats", {})
            products = rc.get("products", {})
            deployments = rc.get("deployments", [])
            rev = sum(d.get("revenue_earned", 0) for d in deployments)
            collected["by_source"]["revenue_catalog"] = {
                "revenue": rev, "cost": 0.0, "profit": rev,
                "transactions": sum(d.get("requests_served", 0) for d in deployments),
                "products_defined": len(products),
                "total_deployed": stats.get("total_deployed", 0),
            }
            collected["total_revenue"] += rev
            collected["total_transactions"] += sum(d.get("requests_served", 0) for d in deployments)
        else:
            collected["sources_missing"].append("revenue_catalog")

        # 8. DatabaseRevenueBridge
        drb = _load_json(SOURCE_FILES["database_revenue_bridge"])
        if drb:
            collected["sources_available"].append("database_revenue_bridge")
            rev_data = drb.get("revenue", {})
            stats = drb.get("stats", {})
            rev = rev_data.get("total", 0.0)
            jobs = drb.get("jobs", [])
            # Estimate cost as 10% of revenue (compute cost for DB queries)
            cost = rev * 0.1
            collected["by_source"]["database_revenue_bridge"] = {
                "revenue": rev, "cost": cost, "profit": rev - cost,
                "transactions": stats.get("total_requests", 0),
                "successful_requests": stats.get("successful_requests", 0),
                "failed_requests": stats.get("failed_requests", 0),
                "success_rate": (
                    round(stats.get("successful_requests", 0) / stats["total_requests"] * 100, 1)
                    if stats.get("total_requests", 0) > 0 else 0
                ),
                "revenue_by_service": rev_data.get("by_service", {}),
                "reports_generated": len(drb.get("reports", {})),
                "schemas_created": len(drb.get("schemas_created", {})),
            }
            collected["total_revenue"] += rev
            collected["total_cost"] += cost
            collected["total_transactions"] += stats.get("total_requests", 0)
            for cid in rev_data.get("by_customer", {}):
                collected["customers"].add(cid)
        else:
            collected["sources_missing"].append("database_revenue_bridge")

        # 9. HTTPRevenueBridge
        hrb = _load_json(SOURCE_FILES["http_revenue_bridge"])
        if hrb:
            collected["sources_available"].append("http_revenue_bridge")
            rev_data = hrb.get("revenue", {})
            stats = hrb.get("stats", {})
            rev = rev_data.get("total", 0.0)
            cost = rev * 0.05  # Low cost for HTTP proxying
            collected["by_source"]["http_revenue_bridge"] = {
                "revenue": rev, "cost": cost, "profit": rev - cost,
                "transactions": stats.get("total_requests", 0),
                "successful_requests": stats.get("successful_requests", 0),
                "failed_requests": stats.get("failed_requests", 0),
                "success_rate": (
                    round(stats.get("successful_requests", 0) / stats["total_requests"] * 100, 1)
                    if stats.get("total_requests", 0) > 0 else 0
                ),
                "revenue_by_service": rev_data.get("by_service", {}),
            }
            collected["total_revenue"] += rev
            collected["total_cost"] += cost
            collected["total_transactions"] += stats.get("total_requests", 0)
            for cid in rev_data.get("by_customer", {}):
                collected["customers"].add(cid)
        else:
            collected["sources_missing"].append("http_revenue_bridge")

        # 10. APIMarketplace
        amp = _load_json(SOURCE_FILES["api_marketplace"])
        if amp:
            collected["sources_available"].append("api_marketplace")
            rev_data = amp.get("revenue", {})
            rev = rev_data.get("total", 0.0)
            cost = rev * 0.15  # Higher cost for external API calls
            collected["by_source"]["api_marketplace"] = {
                "revenue": rev, "cost": cost, "profit": rev - cost,
                "transactions": amp.get("total_calls", 0),
                "revenue_by_api": rev_data.get("by_api", {}),
            }
            collected["total_revenue"] += rev
            collected["total_cost"] += cost
            collected["total_transactions"] += amp.get("total_calls", 0)
            for cid in rev_data.get("by_customer", {}):
                collected["customers"].add(cid)
        else:
            collected["sources_missing"].append("api_marketplace")

        # Convert set to count
        collected["customer_count"] = len(collected["customers"])
        collected["customers"] = list(collected["customers"])
        collected["total_profit"] = collected["total_revenue"] - collected["total_cost"]

        return collected

    # ── Actions ───────────────────────────────────────────────────────

    def _overview(self, params: Dict) -> SkillResult:
        """Comprehensive revenue overview."""
        data = self._collect_all_revenue_data()
        config = self._load().get("config", {})

        profit_margin = 0.0
        if data["total_revenue"] > 0:
            profit_margin = (data["total_profit"] / data["total_revenue"]) * 100

        # Revenue vs compute cost comparison
        compute_cost_hr = config.get("compute_cost_per_hour", 0.10)
        target_daily = config.get("revenue_target_daily", 1.00)

        overview = {
            "total_revenue": round(data["total_revenue"], 6),
            "total_cost": round(data["total_cost"], 6),
            "total_profit": round(data["total_profit"], 6),
            "profit_margin_pct": round(profit_margin, 1),
            "total_transactions": data["total_transactions"],
            "customer_count": data["customer_count"],
            "sources_active": len(data["sources_available"]),
            "sources_missing": len(data["sources_missing"]),
            "source_names": data["sources_available"],
            "compute_cost_per_hour": compute_cost_hr,
            "revenue_target_daily": target_daily,
            "self_sustaining": data["total_revenue"] > compute_cost_hr * 24,
        }

        return SkillResult(
            success=True,
            message=f"Revenue: ${data['total_revenue']:.4f} | Cost: ${data['total_cost']:.4f} | "
                    f"Profit: ${data['total_profit']:.4f} ({profit_margin:.1f}%) | "
                    f"{data['total_transactions']} txns across {len(data['sources_available'])} sources",
            data=overview,
        )

    def _by_source(self, params: Dict) -> SkillResult:
        """Revenue breakdown by source skill."""
        data = self._collect_all_revenue_data()

        # Sort by revenue descending
        sorted_sources = sorted(
            data["by_source"].items(),
            key=lambda x: x[1].get("revenue", 0),
            reverse=True,
        )

        breakdown = {}
        for name, metrics in sorted_sources:
            rev = metrics.get("revenue", 0)
            share_pct = (rev / data["total_revenue"] * 100) if data["total_revenue"] > 0 else 0
            margin = 0.0
            if rev > 0:
                margin = (metrics.get("profit", 0) / rev) * 100
            breakdown[name] = {
                **metrics,
                "revenue_share_pct": round(share_pct, 1),
                "margin_pct": round(margin, 1),
            }
            # Round monetary values
            for k in ("revenue", "cost", "profit"):
                if k in breakdown[name]:
                    breakdown[name][k] = round(breakdown[name][k], 6)

        top_source = sorted_sources[0][0] if sorted_sources else "none"

        return SkillResult(
            success=True,
            message=f"Revenue across {len(breakdown)} sources. "
                    f"Top: {top_source} (${sorted_sources[0][1].get('revenue', 0):.4f})" if sorted_sources else "No revenue data",
            data={
                "sources": breakdown,
                "total_revenue": round(data["total_revenue"], 6),
                "sources_available": data["sources_available"],
                "sources_missing": data["sources_missing"],
            },
        )

    def _profitability(self, params: Dict) -> SkillResult:
        """Profitability analysis."""
        data = self._collect_all_revenue_data()
        config = self._load().get("config", {})

        compute_cost_hr = config.get("compute_cost_per_hour", 0.10)
        compute_cost_daily = compute_cost_hr * 24

        # Per-source profitability
        source_profit = {}
        for name, metrics in data["by_source"].items():
            rev = metrics.get("revenue", 0)
            cost = metrics.get("cost", 0)
            profit = metrics.get("profit", rev - cost)
            margin = (profit / rev * 100) if rev > 0 else 0
            source_profit[name] = {
                "revenue": round(rev, 6),
                "cost": round(cost, 6),
                "profit": round(profit, 6),
                "margin_pct": round(margin, 1),
                "profitable": profit > 0,
            }

        # Identify best and worst performers
        sorted_by_margin = sorted(
            source_profit.items(),
            key=lambda x: x[1]["margin_pct"],
            reverse=True,
        )
        best = sorted_by_margin[0] if sorted_by_margin else None
        worst = sorted_by_margin[-1] if sorted_by_margin else None

        overall_margin = 0.0
        if data["total_revenue"] > 0:
            overall_margin = (data["total_profit"] / data["total_revenue"]) * 100

        # Revenue per transaction
        rev_per_txn = 0.0
        if data["total_transactions"] > 0:
            rev_per_txn = data["total_revenue"] / data["total_transactions"]

        profitability = {
            "overall_margin_pct": round(overall_margin, 1),
            "revenue_per_transaction": round(rev_per_txn, 6),
            "compute_cost_daily": round(compute_cost_daily, 4),
            "net_after_compute": round(data["total_profit"] - compute_cost_daily, 6),
            "break_even_txns_per_day": round(compute_cost_daily / rev_per_txn, 0) if rev_per_txn > 0 else float("inf"),
            "source_profitability": source_profit,
            "best_margin": {"source": best[0], **best[1]} if best else None,
            "worst_margin": {"source": worst[0], **worst[1]} if worst else None,
            "profitable_sources": sum(1 for s in source_profit.values() if s["profitable"]),
            "total_sources": len(source_profit),
        }

        return SkillResult(
            success=True,
            message=f"Overall margin: {overall_margin:.1f}% | "
                    f"Rev/txn: ${rev_per_txn:.4f} | "
                    f"Net after compute: ${profitability['net_after_compute']:.4f}/day",
            data=profitability,
        )

    def _customers(self, params: Dict) -> SkillResult:
        """Customer analytics."""
        data = self._collect_all_revenue_data()

        # Aggregate per-customer data from usage_tracking
        customer_details = {}
        ut = _load_json(SOURCE_FILES["usage_tracking"])
        if ut:
            for cid, info in ut.get("customers", {}).items():
                customer_details[cid] = {
                    "tier": info.get("tier", "unknown"),
                    "total_revenue": info.get("total_revenue", 0),
                    "total_requests": info.get("total_requests", 0),
                    "registered_at": info.get("registered_at", ""),
                }

        # Also from pricing bridge
        pb = _load_json(SOURCE_FILES["pricing_bridge"])
        if pb:
            for q in pb.get("task_quotes", {}).values():
                cid = q.get("customer_id", "")
                if cid and cid != "api_user":
                    if cid not in customer_details:
                        customer_details[cid] = {
                            "tier": "api", "total_revenue": 0,
                            "total_requests": 0, "registered_at": "",
                        }
                    if q.get("status") == "completed":
                        customer_details[cid]["total_revenue"] += q.get("revenue", q.get("price", 0))
                        customer_details[cid]["total_requests"] += 1

        # Also from database revenue bridge
        drb = _load_json(SOURCE_FILES["database_revenue_bridge"])
        if drb:
            for cid, rev in drb.get("revenue", {}).get("by_customer", {}).items():
                if cid not in customer_details:
                    customer_details[cid] = {
                        "tier": "data_services", "total_revenue": 0,
                        "total_requests": 0, "registered_at": "",
                    }
                customer_details[cid]["total_revenue"] += rev
                customer_jobs = [j for j in drb.get("jobs", []) if j.get("customer_id") == cid]
                customer_details[cid]["total_requests"] += len(customer_jobs)

        # Also from HTTP revenue bridge
        hrb = _load_json(SOURCE_FILES["http_revenue_bridge"])
        if hrb:
            for cid, rev in hrb.get("revenue", {}).get("by_customer", {}).items():
                if cid not in customer_details:
                    customer_details[cid] = {
                        "tier": "http_services", "total_revenue": 0,
                        "total_requests": 0, "registered_at": "",
                    }
                customer_details[cid]["total_revenue"] += rev

        # Also from API marketplace
        amp = _load_json(SOURCE_FILES["api_marketplace"])
        if amp:
            for cid, rev in amp.get("revenue", {}).get("by_customer", {}).items():
                if cid not in customer_details:
                    customer_details[cid] = {
                        "tier": "api_marketplace", "total_revenue": 0,
                        "total_requests": 0, "registered_at": "",
                    }
                customer_details[cid]["total_revenue"] += rev

        # Sort by revenue
        sorted_customers = sorted(
            customer_details.items(),
            key=lambda x: x[1].get("total_revenue", 0),
            reverse=True,
        )
        top_5 = sorted_customers[:5]

        # Concentration risk
        total_cust_rev = sum(c[1].get("total_revenue", 0) for c in sorted_customers)
        top_1_share = 0.0
        if sorted_customers and total_cust_rev > 0:
            top_1_share = sorted_customers[0][1].get("total_revenue", 0) / total_cust_rev * 100

        analytics = {
            "total_customers": len(customer_details),
            "total_customer_revenue": round(total_cust_rev, 6),
            "top_customers": [
                {"customer_id": cid, **{k: round(v, 6) if isinstance(v, float) else v for k, v in info.items()}}
                for cid, info in top_5
            ],
            "concentration_risk": {
                "top_1_share_pct": round(top_1_share, 1),
                "high_concentration": top_1_share > 50,
                "recommendation": "Diversify customer base" if top_1_share > 50 else "Healthy distribution",
            },
            "tiers": {},
        }

        # Tier breakdown
        tier_counts = defaultdict(int)
        for cid, info in customer_details.items():
            tier_counts[info.get("tier", "unknown")] += 1
        analytics["tiers"] = dict(tier_counts)

        return SkillResult(
            success=True,
            message=f"{len(customer_details)} customers | "
                    f"Top-1 concentration: {top_1_share:.1f}%",
            data=analytics,
        )

    def _trends(self, params: Dict) -> SkillResult:
        """Revenue trends from snapshot history."""
        dash_data = self._load()
        snapshots = dash_data.get("snapshots", [])
        window_hours = params.get("window_hours", 24)

        if len(snapshots) < 2:
            return SkillResult(
                success=True,
                message="Need at least 2 snapshots for trend analysis. Use 'snapshot' action first.",
                data={"snapshots_available": len(snapshots), "trend": "insufficient_data"},
            )

        cutoff = time.time() - (window_hours * 3600)
        recent = [s for s in snapshots if _parse_ts(s.get("timestamp", "")) and _parse_ts(s["timestamp"]) >= cutoff]

        if len(recent) < 2:
            recent = snapshots[-10:]  # Fallback to last 10

        # Calculate deltas
        first = recent[0]
        last = recent[-1]
        rev_delta = last.get("total_revenue", 0) - first.get("total_revenue", 0)
        cost_delta = last.get("total_cost", 0) - first.get("total_cost", 0)
        profit_delta = last.get("total_profit", 0) - first.get("total_profit", 0)
        txn_delta = last.get("total_transactions", 0) - first.get("total_transactions", 0)

        # Direction
        if rev_delta > 0:
            direction = "growing"
        elif rev_delta < 0:
            direction = "declining"
        else:
            direction = "flat"

        # Revenue velocity (per hour)
        time_span_hours = max(1, len(recent))
        rev_per_hour = rev_delta / time_span_hours if time_span_hours > 0 else 0

        trends = {
            "window_hours": window_hours,
            "snapshots_analyzed": len(recent),
            "direction": direction,
            "revenue_delta": round(rev_delta, 6),
            "cost_delta": round(cost_delta, 6),
            "profit_delta": round(profit_delta, 6),
            "transaction_delta": txn_delta,
            "revenue_per_hour": round(rev_per_hour, 6),
            "first_snapshot": first.get("timestamp"),
            "last_snapshot": last.get("timestamp"),
            "timeline": [
                {
                    "timestamp": s.get("timestamp"),
                    "revenue": round(s.get("total_revenue", 0), 6),
                    "profit": round(s.get("total_profit", 0), 6),
                    "transactions": s.get("total_transactions", 0),
                }
                for s in recent
            ],
        }

        return SkillResult(
            success=True,
            message=f"Trend: {direction} | Rev delta: ${rev_delta:.4f} | "
                    f"${rev_per_hour:.4f}/hr over {len(recent)} snapshots",
            data=trends,
        )

    def _forecast(self, params: Dict) -> SkillResult:
        """Revenue forecast based on historical snapshots."""
        dash_data = self._load()
        snapshots = dash_data.get("snapshots", [])
        config = dash_data.get("config", {})
        days_ahead = params.get("days_ahead", 7)

        if len(snapshots) < 3:
            return SkillResult(
                success=True,
                message="Need at least 3 snapshots for forecasting. Use 'snapshot' action periodically.",
                data={"snapshots_available": len(snapshots), "forecast": "insufficient_data"},
            )

        # Calculate growth rate from snapshots
        recent = snapshots[-20:]
        revenues = [s.get("total_revenue", 0) for s in recent]

        # Simple linear regression
        n = len(revenues)
        x_mean = (n - 1) / 2
        y_mean = sum(revenues) / n
        numerator = sum((i - x_mean) * (revenues[i] - y_mean) for i in range(n))
        denominator = sum((i - x_mean) ** 2 for i in range(n))

        slope = numerator / denominator if denominator > 0 else 0
        intercept = y_mean - slope * x_mean

        # Project forward
        current_rev = revenues[-1]
        daily_rate = slope * 24  # Assuming hourly snapshots
        forecasted = []
        for d in range(1, days_ahead + 1):
            projected = current_rev + daily_rate * d
            forecasted.append({
                "day": d,
                "projected_revenue": round(max(0, projected), 6),
            })

        compute_cost_daily = config.get("compute_cost_per_hour", 0.10) * 24
        target = config.get("revenue_target_daily", 1.00)

        # Days to break-even
        if daily_rate > 0 and current_rev < compute_cost_daily:
            days_to_breakeven = max(0, round((compute_cost_daily - current_rev) / daily_rate))
        elif current_rev >= compute_cost_daily:
            days_to_breakeven = 0
        else:
            days_to_breakeven = -1  # Never at current rate

        forecast = {
            "current_revenue": round(current_rev, 6),
            "daily_growth_rate": round(daily_rate, 6),
            "growth_direction": "positive" if slope > 0 else "negative" if slope < 0 else "flat",
            "forecasted_days": forecasted,
            "days_to_breakeven": days_to_breakeven,
            "compute_cost_daily": round(compute_cost_daily, 4),
            "revenue_target_daily": target,
            "days_to_target": round((target - current_rev) / daily_rate) if daily_rate > 0 else -1,
            "snapshots_used": len(recent),
        }

        return SkillResult(
            success=True,
            message=f"Forecast: ${daily_rate:.4f}/day growth | "
                    f"Break-even: {'achieved' if days_to_breakeven == 0 else f'{days_to_breakeven} days' if days_to_breakeven > 0 else 'not at current rate'}",
            data=forecast,
        )

    def _snapshot(self, params: Dict) -> SkillResult:
        """Take a point-in-time revenue snapshot."""
        collected = self._collect_all_revenue_data()
        dash_data = self._load()

        snapshot = {
            "timestamp": _now_iso(),
            "total_revenue": round(collected["total_revenue"], 6),
            "total_cost": round(collected["total_cost"], 6),
            "total_profit": round(collected["total_profit"], 6),
            "total_transactions": collected["total_transactions"],
            "customer_count": collected["customer_count"],
            "sources_active": len(collected["sources_available"]),
            "by_source": {
                name: {
                    "revenue": round(m.get("revenue", 0), 6),
                    "transactions": m.get("transactions", 0),
                }
                for name, m in collected["by_source"].items()
            },
        }

        dash_data.setdefault("snapshots", []).append(snapshot)
        self._save(dash_data)

        snap_count = len(dash_data["snapshots"])

        return SkillResult(
            success=True,
            message=f"Snapshot #{snap_count}: ${collected['total_revenue']:.4f} revenue, "
                    f"{collected['total_transactions']} txns, {len(collected['sources_available'])} sources",
            data={"snapshot": snapshot, "total_snapshots": snap_count},
        )

    def _recommendations(self, params: Dict) -> SkillResult:
        """Generate revenue optimization recommendations."""
        data = self._collect_all_revenue_data()
        config = self._load().get("config", {})
        recs = []

        # 1. Source activation
        if data["sources_missing"]:
            recs.append({
                "priority": "high",
                "category": "activation",
                "recommendation": f"Activate missing revenue sources: {', '.join(data['sources_missing'])}",
                "impact": "More revenue streams reduce concentration risk",
            })

        # 2. Profitability check
        for name, metrics in data["by_source"].items():
            rev = metrics.get("revenue", 0)
            cost = metrics.get("cost", 0)
            if cost > 0 and rev > 0:
                margin = (rev - cost) / rev * 100
                if margin < 20:
                    recs.append({
                        "priority": "medium",
                        "category": "pricing",
                        "recommendation": f"Increase prices for {name} (margin only {margin:.1f}%)",
                        "impact": "Target 30%+ margin for sustainability",
                    })

        # 3. Zero-revenue sources
        for name, metrics in data["by_source"].items():
            if metrics.get("revenue", 0) == 0 and metrics.get("transactions", 0) == 0:
                recs.append({
                    "priority": "medium",
                    "category": "growth",
                    "recommendation": f"Drive traffic to {name} - registered but no revenue",
                    "impact": "Idle infrastructure has cost without return",
                })

        # 4. Customer concentration
        if data["customer_count"] <= 1 and data["total_revenue"] > 0:
            recs.append({
                "priority": "high",
                "category": "risk",
                "recommendation": "Diversify customer base - revenue depends on <=1 customer",
                "impact": "Single customer loss would eliminate all revenue",
            })

        # 5. Revenue target check
        target = config.get("revenue_target_daily", 1.00)
        if data["total_revenue"] < target:
            recs.append({
                "priority": "high",
                "category": "growth",
                "recommendation": f"Revenue ${data['total_revenue']:.4f} below target ${target:.2f}/day",
                "impact": "Focus on highest-margin services and customer acquisition",
            })

        # 6. Compute cost coverage
        compute_daily = config.get("compute_cost_per_hour", 0.10) * 24
        if data["total_revenue"] < compute_daily:
            recs.append({
                "priority": "critical",
                "category": "sustainability",
                "recommendation": f"Revenue (${data['total_revenue']:.4f}) doesn't cover compute (${compute_daily:.2f}/day)",
                "impact": "Agent is losing money - must increase revenue or reduce costs",
            })

        if not recs:
            recs.append({
                "priority": "info",
                "category": "status",
                "recommendation": "No critical issues detected. Consider expanding service offerings.",
                "impact": "Growth opportunity",
            })

        # Sort by priority
        priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}
        recs.sort(key=lambda r: priority_order.get(r["priority"], 5))

        return SkillResult(
            success=True,
            message=f"{len(recs)} recommendations | "
                    f"Critical: {sum(1 for r in recs if r['priority'] == 'critical')}, "
                    f"High: {sum(1 for r in recs if r['priority'] == 'high')}",
            data={"recommendations": recs, "count": len(recs)},
        )

    def _configure(self, params: Dict) -> SkillResult:
        """Update dashboard configuration."""
        dash_data = self._load()
        config = dash_data.setdefault("config", {})
        changes = []

        if "compute_cost_per_hour" in params:
            config["compute_cost_per_hour"] = float(params["compute_cost_per_hour"])
            changes.append(f"compute_cost_per_hour=${config['compute_cost_per_hour']:.4f}")
        if "revenue_target_daily" in params:
            config["revenue_target_daily"] = float(params["revenue_target_daily"])
            changes.append(f"revenue_target_daily=${config['revenue_target_daily']:.4f}")

        self._save(dash_data)

        return SkillResult(
            success=True,
            message=f"Config updated: {', '.join(changes) if changes else 'no changes'}",
            data={"config": config},
        )

    def _status(self, params: Dict) -> SkillResult:
        """Dashboard status and data source health."""
        dash_data = self._load()
        collected = self._collect_all_revenue_data()

        source_health = {}
        for name, path in SOURCE_FILES.items():
            exists = path.exists()
            size = path.stat().st_size if exists else 0
            source_health[name] = {
                "available": exists,
                "file_size_bytes": size,
                "has_data": name in collected["by_source"],
            }

        return SkillResult(
            success=True,
            message=f"Dashboard: {len(collected['sources_available'])}/{len(SOURCE_FILES)} sources active, "
                    f"{len(dash_data.get('snapshots', []))} snapshots stored",
            data={
                "sources": source_health,
                "snapshots_stored": len(dash_data.get("snapshots", [])),
                "config": dash_data.get("config", {}),
                "last_updated": dash_data.get("last_updated"),
            },
        )
