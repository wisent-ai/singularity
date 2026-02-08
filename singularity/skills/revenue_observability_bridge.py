#!/usr/bin/env python3
"""
RevenueObservabilityBridgeSkill - Wire ALL revenue sources into ObservabilitySkill metrics.

The agent has multiple revenue-generating subsystems, each storing data independently:
- DatabaseRevenueBridgeSkill (database_revenue_bridge.json)
- HTTPRevenueBridgeSkill (http_revenue_bridge.json)
- BillingPipelineSkill (billing_pipeline.json)
- RevenueAnalyticsDashboardSkill (revenue_analytics_dashboard.json)
- RevenueServices (revenue_services_log.json)
- TaskPricing (task_pricing.json)
- Marketplace (marketplace.json)
- UsageTracking (usage_tracking.json)

Without this bridge, ObservabilitySkill has zero revenue metrics — making it
impossible to set up revenue alerts, track revenue trends over time, or
correlate revenue changes with other system metrics.

This bridge:
1. Reads revenue data from ALL sources (including newer bridges)
2. Emits structured metrics to ObservabilitySkill (revenue.total, revenue.by_source, etc.)
3. Auto-creates alert rules for revenue anomalies
4. Provides a `sync` action for periodic metric emission (schedulable)
5. Tracks sync history for consistency monitoring

Emitted metrics:
  - revenue.total (gauge): Total revenue across all sources
  - revenue.by_source (gauge, label: source): Revenue per source
  - revenue.requests.total (counter): Total paid requests
  - revenue.requests.success_rate (gauge): Success rate percentage
  - revenue.requests.by_source (counter, label: source): Requests per source
  - revenue.customers.active (gauge): Number of active customers
  - revenue.avg_per_request (gauge): Average revenue per request
  - revenue.pipeline.invoices (gauge): Invoices in billing pipeline
  - revenue.pipeline.outstanding (gauge): Outstanding invoice amount

Alert rules auto-created:
  - revenue.total drops below configurable threshold
  - revenue.requests.success_rate drops below 80%

Pillar: Revenue (primary) + Goal Setting (data-driven revenue prioritization)
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

from .base import Skill, SkillResult, SkillManifest, SkillAction

DATA_DIR = Path(__file__).parent.parent / "data"
BRIDGE_STATE_FILE = DATA_DIR / "revenue_observability_bridge.json"

# All revenue data source files
REVENUE_SOURCES = {
    "database_revenue_bridge": DATA_DIR / "database_revenue_bridge.json",
    "http_revenue_bridge": DATA_DIR / "http_revenue_bridge.json",
    "billing_pipeline": DATA_DIR / "billing_pipeline.json",
    "revenue_analytics": DATA_DIR / "revenue_analytics_dashboard.json",
    "revenue_services": DATA_DIR / "revenue_services_log.json",
    "task_pricing": DATA_DIR / "task_pricing.json",
    "marketplace": DATA_DIR / "marketplace.json",
    "usage_tracking": DATA_DIR / "usage_tracking.json",
}

MAX_SYNC_HISTORY = 200


def _now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _load_json(path: Path) -> Optional[Dict]:
    """Safely load a JSON file, returning None if missing/corrupt."""
    try:
        if path.exists():
            return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        pass
    return None


def _load_state() -> Dict:
    data = _load_json(BRIDGE_STATE_FILE)
    if data:
        return data
    return {
        "config": {
            "auto_alerts": True,
            "alert_revenue_min": 0.0,
            "alert_success_rate_min": 80.0,
        },
        "sync_history": [],
        "alerts_created": [],
        "stats": {
            "total_syncs": 0,
            "last_sync": None,
            "metrics_emitted": 0,
        },
    }


def _save_state(state: Dict):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if len(state.get("sync_history", [])) > MAX_SYNC_HISTORY:
        state["sync_history"] = state["sync_history"][-MAX_SYNC_HISTORY:]
    try:
        with open(BRIDGE_STATE_FILE, "w") as f:
            json.dump(state, f, indent=2, default=str)
    except IOError:
        pass


def _extract_revenue_from_source(name: str, data: Dict) -> Dict:
    """Extract revenue metrics from a source's data structure."""
    result = {
        "revenue": 0.0,
        "requests": 0,
        "successful_requests": 0,
        "failed_requests": 0,
        "customers": set(),
        "by_service": {},
    }

    if data is None:
        return result

    # RevenueServices log format (list of entries)
    if isinstance(data, list):
        for entry in data:
            if isinstance(entry, dict):
                result["revenue"] += float(entry.get("price", 0))
                result["requests"] += 1
                if entry.get("customer_id"):
                    result["customers"].add(entry["customer_id"])
        return result
    # DatabaseRevenueBridge / HTTPRevenueBridge format
    rev_block = data.get("revenue", {})
    if isinstance(rev_block, dict):
        result["revenue"] = float(rev_block.get("total", 0))
        by_svc = rev_block.get("by_service", {})
        if isinstance(by_svc, dict):
            result["by_service"] = {k: float(v) for k, v in by_svc.items()}
        by_cust = rev_block.get("by_customer", {})
        if isinstance(by_cust, dict):
            result["customers"] = set(by_cust.keys())

    stats = data.get("stats", {})
    if isinstance(stats, dict):
        result["requests"] = int(stats.get("total_requests", 0))
        result["successful_requests"] = int(stats.get("successful_requests", 0))
        result["failed_requests"] = int(stats.get("failed_requests", 0))

    # BillingPipeline format
    if "billing_cycles" in data:
        cycles = data.get("billing_cycles", [])
        for cycle in cycles:
            if isinstance(cycle, dict):
                result["revenue"] += float(cycle.get("total_revenue", 0))
        customers = data.get("customers", {})
        if isinstance(customers, dict):
            result["customers"] = set(customers.keys())

    # Marketplace format
    if "orders" in data and "revenue_log" in data:
        rev_log = data.get("revenue_log", [])
        for entry in rev_log:
            if isinstance(entry, dict):
                result["revenue"] += float(entry.get("amount", 0))
        result["requests"] = len(data.get("orders", []))

    # UsageTracking format
    if "customers" in data and "usage_records" in data:
        for cust_id, cust_data in data.get("customers", {}).items():
            result["customers"].add(cust_id)
            if isinstance(cust_data, dict):
                result["requests"] += int(cust_data.get("total_calls", 0))

    # TaskPricing format
    if "revenue_summary" in data:
        summary = data.get("revenue_summary", {})
        if isinstance(summary, dict):
            result["revenue"] = max(result["revenue"], float(summary.get("total_earned", 0)))
            result["requests"] = max(result["requests"], int(summary.get("tasks_completed", 0)))


    # Revenue analytics dashboard snapshot format
    if "snapshots" in data:
        snapshots = data.get("snapshots", [])
        if snapshots:
            latest = snapshots[-1] if isinstance(snapshots[-1], dict) else {}
            result["revenue"] = max(result["revenue"], float(latest.get("total_revenue", 0)))

    return result


class RevenueObservabilityBridgeSkill(Skill):
    """
    Bridges ALL revenue data sources into ObservabilitySkill metrics.

    Reads from every revenue-generating skill's data file and emits
    structured metrics for time-series tracking, alerting, and dashboards.
    """

    def __init__(self, credentials: Dict[str, str] = None):
        super().__init__(credentials)
        self._state = _load_state()
        self._observability = None

    def set_observability(self, obs_skill):
        """Inject ObservabilitySkill for metric emission."""
        self._observability = obs_skill

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="revenue_observability_bridge",
            name="revenue_observability_bridge",
            description="Wire all revenue sources into ObservabilitySkill metrics for tracking, alerting, and trend analysis",
            version="1.0.0",
            category="revenue",
            required_credentials=[],            actions=[
                SkillAction(
                    name="sync",
                    description="Collect revenue data from all sources and emit as ObservabilitySkill metrics",
                    parameters={"force": "bool - sync even if recently synced (default true)"},
                ),
                SkillAction(
                    name="sources",
                    description="List all revenue data sources and their availability",
                    parameters={},
                ),
                SkillAction(
                    name="snapshot",
                    description="Get current aggregated revenue snapshot without emitting metrics",
                    parameters={},
                ),
                SkillAction(
                    name="setup_alerts",
                    description="Create standard revenue alert rules in ObservabilitySkill",
                    parameters={
                        "revenue_min": "float - alert if total revenue below this (default 0.0)",
                        "success_rate_min": "float - alert if success rate below this % (default 80.0)",
                    },
                ),
                SkillAction(
                    name="history",
                    description="View sync history - when syncs happened and what was emitted",
                    parameters={"limit": "int - max history entries to return (default 10)"},
                ),
                SkillAction(
                    name="configure",
                    description="Configure bridge settings (auto_alerts, thresholds)",
                    parameters={
                        "auto_alerts": "bool - auto-create alerts on first sync",
                        "alert_revenue_min": "float - min revenue alert threshold",
                        "alert_success_rate_min": "float - min success rate alert threshold",
                    },
                ),
                SkillAction(
                    name="status",
                    description="Show bridge status, configuration, and stats",
                    parameters={},
                ),
            ],
        )

    async def execute(self, action: str, params: Dict[str, Any]) -> SkillResult:
        handlers = {
            "sync": self._sync,
            "sources": self._sources,
            "snapshot": self._snapshot,
            "setup_alerts": self._setup_alerts,
            "history": self._history,
            "configure": self._configure,
            "status": self._status,
        }
        handler = handlers.get(action)
        if not handler:
            return SkillResult(success=False, message=f"Unknown action: {action}")
        return handler(params)

    def _collect_all(self) -> Dict[str, Dict]:
        """Collect revenue data from all sources."""
        collected = {}
        for name, path in REVENUE_SOURCES.items():
            raw = _load_json(path)
            collected[name] = _extract_revenue_from_source(name, raw)
        return collected

    def _aggregate(self, collected: Dict[str, Dict]) -> Dict:
        """Aggregate all collected revenue data into unified metrics."""
        total_revenue = 0.0
        total_requests = 0
        total_success = 0
        total_failed = 0
        all_customers = set()
        by_source = {}

        for name, data in collected.items():
            rev = data.get("revenue", 0.0)
            reqs = data.get("requests", 0)
            success = data.get("successful_requests", 0)
            failed = data.get("failed_requests", 0)
            customers = data.get("customers", set())

            total_revenue += rev
            total_requests += reqs
            total_success += success
            total_failed += failed
            all_customers.update(customers)

            if rev > 0 or reqs > 0:
                by_source[name] = {
                    "revenue": rev,
                    "requests": reqs,
                    "successful_requests": success,
                    "failed_requests": failed,
                    "customers": len(customers),
                }

        success_rate = (total_success / total_requests * 100) if total_requests > 0 else 100.0
        avg_per_request = (total_revenue / total_requests) if total_requests > 0 else 0.0

        return {
            "total_revenue": total_revenue,
            "total_requests": total_requests,
            "total_success": total_success,
            "total_failed": total_failed,
            "success_rate": round(success_rate, 2),
            "avg_per_request": round(avg_per_request, 6),
            "active_customers": len(all_customers),
            "active_sources": len(by_source),
            "by_source": by_source,
        }

    def _emit_metrics(self, aggregated: Dict) -> int:
        """Emit metrics to ObservabilitySkill. Returns count of metrics emitted."""
        if not self._observability:
            return 0

        count = 0

        def emit(name, value, metric_type="gauge", labels=None):
            nonlocal count
            params = {
                "name": name,
                "value": value,
                "metric_type": metric_type,
                "labels": labels or {},
            }
            result = self._observability._emit(params)
            if result.success:
                count += 1

        # Totals
        emit("revenue.total", aggregated["total_revenue"])
        emit("revenue.requests.total", aggregated["total_requests"], "counter")
        emit("revenue.requests.success_rate", aggregated["success_rate"])
        emit("revenue.avg_per_request", aggregated["avg_per_request"])
        emit("revenue.customers.active", aggregated["active_customers"])
        emit("revenue.sources.active", aggregated["active_sources"])

        # Per-source breakdown
        for source_name, source_data in aggregated.get("by_source", {}).items():
            labels = {"source": source_name}
            emit("revenue.by_source", source_data["revenue"], "gauge", labels)
            emit("revenue.requests.by_source", source_data["requests"], "counter", labels)

        return count

    def _sync(self, params: Dict) -> SkillResult:
        """Collect from all sources and emit to ObservabilitySkill."""
        collected = self._collect_all()
        aggregated = self._aggregate(collected)
        metrics_emitted = self._emit_metrics(aggregated)

        # Record sync
        sync_entry = {
            "timestamp": _now_iso(),
            "metrics_emitted": metrics_emitted,
            "total_revenue": aggregated["total_revenue"],
            "total_requests": aggregated["total_requests"],
            "active_sources": aggregated["active_sources"],
            "active_customers": aggregated["active_customers"],
            "has_observability": self._observability is not None,
        }
        self._state["sync_history"].append(sync_entry)
        self._state["stats"]["total_syncs"] += 1
        self._state["stats"]["last_sync"] = _now_iso()
        self._state["stats"]["metrics_emitted"] += metrics_emitted

        # Auto-create alerts on first sync if configured
        if (self._state["config"].get("auto_alerts", True)
                and not self._state["alerts_created"]
                and self._observability):
            self._setup_alerts(params={})

        _save_state(self._state)

        return SkillResult(
            success=True,
            message=f"Synced revenue data: ${aggregated['total_revenue']:.4f} total, "
                    f"{aggregated['total_requests']} requests, "
                    f"{metrics_emitted} metrics emitted to observability",
            data={
                "aggregated": aggregated,
                "metrics_emitted": metrics_emitted,
                "sources_checked": len(REVENUE_SOURCES),
            },
        )

    def _sources(self, params: Dict) -> SkillResult:
        """List all revenue sources and their availability."""
        sources = []
        for name, path in REVENUE_SOURCES.items():
            raw = _load_json(path)
            extracted = _extract_revenue_from_source(name, raw)
            sources.append({
                "name": name,
                "file": str(path.name),
                "available": raw is not None,
                "revenue": extracted.get("revenue", 0.0),
                "requests": extracted.get("requests", 0),
            })

        available = sum(1 for s in sources if s["available"])
        return SkillResult(
            success=True,
            message=f"{available}/{len(sources)} revenue sources available",
            data={"sources": sources, "available": available, "total": len(sources)},
        )

    def _snapshot(self, params: Dict) -> SkillResult:
        """Get current aggregated revenue snapshot."""
        collected = self._collect_all()
        aggregated = self._aggregate(collected)
        return SkillResult(
            success=True,
            message=f"Revenue snapshot: ${aggregated['total_revenue']:.4f} total, "
                    f"{aggregated['active_sources']} active sources, "
                    f"{aggregated['active_customers']} customers",
            data=aggregated,
        )

    def _setup_alerts(self, params: Dict) -> SkillResult:
        """Create standard revenue alerts in ObservabilitySkill."""
        if not self._observability:
            return SkillResult(
                success=False,
                message="ObservabilitySkill not connected. Call set_observability() first.",
            )

        revenue_min = float(params.get("revenue_min", self._state["config"]["alert_revenue_min"]))
        success_min = float(params.get("success_rate_min", self._state["config"]["alert_success_rate_min"]))

        alerts_created = []

        # Revenue total alert
        result = self._observability._alert_create({
            "name": "revenue_below_minimum",
            "metric": "revenue.total",
            "condition": "below",
            "threshold": revenue_min,
            "window_minutes": 60,
            "cooldown_minutes": 30,
            "labels": {},
        })
        if result.success:
            alerts_created.append("revenue_below_minimum")

        # Success rate alert
        result = self._observability._alert_create({
            "name": "revenue_low_success_rate",
            "metric": "revenue.requests.success_rate",
            "condition": "below",
            "threshold": success_min,
            "window_minutes": 60,
            "cooldown_minutes": 30,
            "labels": {},
        })
        if result.success:
            alerts_created.append("revenue_low_success_rate")

        self._state["alerts_created"] = alerts_created
        _save_state(self._state)

        return SkillResult(
            success=True,
            message=f"Created {len(alerts_created)} revenue alert rules: {', '.join(alerts_created)}",
            data={"alerts_created": alerts_created},
        )

    def _history(self, params: Dict) -> SkillResult:
        """View sync history."""
        limit = int(params.get("limit", 10))
        history = self._state.get("sync_history", [])
        recent = history[-limit:] if limit > 0 else history

        return SkillResult(
            success=True,
            message=f"Showing {len(recent)} of {len(history)} sync history entries",
            data={"history": recent, "total": len(history)},
        )

    def _configure(self, params: Dict) -> SkillResult:
        """Configure bridge settings."""
        config = self._state["config"]
        changed = []

        if "auto_alerts" in params:
            config["auto_alerts"] = bool(params["auto_alerts"])
            changed.append("auto_alerts")
        if "alert_revenue_min" in params:
            config["alert_revenue_min"] = float(params["alert_revenue_min"])
            changed.append("alert_revenue_min")
        if "alert_success_rate_min" in params:
            config["alert_success_rate_min"] = float(params["alert_success_rate_min"])
            changed.append("alert_success_rate_min")

        if not changed:
            return SkillResult(
                success=True,
                message="No changes - provide auto_alerts, alert_revenue_min, or alert_success_rate_min",
                data={"config": config},
            )

        _save_state(self._state)
        return SkillResult(
            success=True,
            message=f"Updated config: {', '.join(changed)}",
            data={"config": config},
        )

    def _status(self, params: Dict) -> SkillResult:
        """Show bridge status."""
        return SkillResult(
            success=True,
            message="Revenue Observability Bridge status",
            data={
                "config": self._state["config"],
                "stats": self._state["stats"],
                "alerts_created": self._state["alerts_created"],
                "observability_connected": self._observability is not None,
                "sources_count": len(REVENUE_SOURCES),
            },
        )
