#!/usr/bin/env python3
"""
ServiceMonitoringDashboardSkill - Unified operational dashboard aggregating
health, uptime, revenue, and performance metrics across all agent subsystems.

This is the #1 priority from MEMORY.md: "Service Monitoring Dashboard -
Aggregate health, uptime, revenue metrics into a unified view."

Without a unified dashboard, the agent has metrics scattered across many skills
(ObservabilitySkill, HealthMonitor, ServiceAPI, RuntimeMetrics, etc.) with no
single place to answer "how is the agent doing overall?" This skill solves that.

Capabilities:
- **overview**: One-call summary of agent health, services, revenue, fleet status
- **services**: List all deployed services with uptime, latency, error rates
- **revenue**: Revenue dashboard - earnings, costs, profit, trends
- **fleet**: Fleet health across all replicas (aggregated from HealthMonitor)
- **alerts**: Active alerts across all monitoring systems
- **uptime**: Uptime report for services over configurable time windows
- **trends**: Trend analysis - what's improving, what's degrading
- **report**: Generate a full status report (suitable for logging or sharing)
- **configure**: Set dashboard preferences (refresh intervals, thresholds)
- **status**: Dashboard own status and data freshness

Integrates with:
- ObservabilitySkill: pulls time-series metrics
- AgentHealthMonitor: replica health data
- ServiceAPI: service endpoint status
- StrategySkill: pillar maturity scores
- RuntimeMetrics: real-time performance
- UsageTrackingSkill: cost data

Pillars:
- Revenue: unified view of earnings and service health
- Self-Improvement: see what's working and what needs attention
- Replication: monitor fleet health at a glance
- Goal Setting: track progress across all pillars
"""

import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from .base import Skill, SkillResult, SkillManifest, SkillAction

DATA_FILE = Path(__file__).parent.parent / "data" / "service_dashboard.json"

# Service status levels
STATUS_HEALTHY = "healthy"
STATUS_DEGRADED = "degraded"
STATUS_DOWN = "down"
STATUS_UNKNOWN = "unknown"

# Dashboard severity colors
SEVERITY_OK = "ok"
SEVERITY_WARNING = "warning"
SEVERITY_CRITICAL = "critical"

# Trend directions
TREND_UP = "improving"
TREND_DOWN = "degrading"
TREND_STABLE = "stable"

# Limits
MAX_SERVICES = 200
MAX_SNAPSHOTS = 1000
MAX_ALERTS_DISPLAY = 50


def _load_data(path: Path = None) -> Dict:
    """Load dashboard state from disk."""
    p = path or DATA_FILE
    if p.exists():
        try:
            with open(p) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return _default_data()


def _default_data() -> Dict:
    return {
        "services": {},
        "snapshots": [],
        "config": {
            "snapshot_interval_seconds": 300,
            "uptime_window_hours": 24,
            "degraded_latency_ms": 1000,
            "critical_latency_ms": 5000,
            "degraded_error_rate": 0.05,
            "critical_error_rate": 0.20,
        },
        "stats": {
            "total_snapshots": 0,
            "last_snapshot_at": None,
        },
    }


def _save_data(data: Dict, path: Path = None) -> None:
    """Save dashboard state to disk."""
    p = path or DATA_FILE
    p.parent.mkdir(parents=True, exist_ok=True)
    if len(data.get("snapshots", [])) > MAX_SNAPSHOTS:
        data["snapshots"] = data["snapshots"][-MAX_SNAPSHOTS:]
    with open(p, "w") as f:
        json.dump(data, f, indent=2, default=str)


def _compute_uptime(snapshots: List[Dict], service_id: str, window_hours: float) -> float:
    """Compute uptime percentage for a service over a time window."""
    if not snapshots:
        return 100.0
    cutoff = time.time() - (window_hours * 3600)
    relevant = [s for s in snapshots if s.get("timestamp", 0) >= cutoff
                and s.get("service_id") == service_id]
    if not relevant:
        return 100.0
    healthy_count = sum(1 for s in relevant if s.get("status") == STATUS_HEALTHY)
    return round((healthy_count / len(relevant)) * 100, 2)


def _compute_trend(values: List[float]) -> str:
    """Compute trend direction from a series of values."""
    if len(values) < 2:
        return TREND_STABLE
    first_half = values[:len(values) // 2]
    second_half = values[len(values) // 2:]
    if not first_half or not second_half:
        return TREND_STABLE
    avg_first = sum(first_half) / len(first_half)
    avg_second = sum(second_half) / len(second_half)
    if avg_first == 0:
        return TREND_STABLE
    change_pct = (avg_second - avg_first) / abs(avg_first) if avg_first != 0 else 0
    if change_pct > 0.1:
        return TREND_UP
    elif change_pct < -0.1:
        return TREND_DOWN
    return TREND_STABLE


def _determine_severity(error_rate: float, latency_ms: float, config: Dict) -> str:
    """Determine severity based on error rate and latency."""
    if (error_rate >= config.get("critical_error_rate", 0.20) or
            latency_ms >= config.get("critical_latency_ms", 5000)):
        return SEVERITY_CRITICAL
    if (error_rate >= config.get("degraded_error_rate", 0.05) or
            latency_ms >= config.get("degraded_latency_ms", 1000)):
        return SEVERITY_WARNING
    return SEVERITY_OK


def _service_status(error_rate: float, latency_ms: float, config: Dict) -> str:
    """Determine service status from metrics."""
    sev = _determine_severity(error_rate, latency_ms, config)
    if sev == SEVERITY_CRITICAL:
        return STATUS_DOWN
    elif sev == SEVERITY_WARNING:
        return STATUS_DEGRADED
    return STATUS_HEALTHY


class ServiceMonitoringDashboardSkill(Skill):
    """
    Unified monitoring dashboard aggregating health, uptime, revenue,
    and performance across all agent subsystems.
    """

    def __init__(self, credentials: Dict[str, str] = None, data_path: Path = None):
        super().__init__(credentials)
        self._data_path = data_path

    def _load(self) -> Dict:
        return _load_data(self._data_path)

    def _save(self, data: Dict) -> None:
        _save_data(data, self._data_path)

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="service_monitoring_dashboard",
            name="Service Monitoring Dashboard",
            version="1.0.0",
            category="operations",
            description="Unified operational dashboard aggregating health, uptime, revenue, and performance metrics across all agent subsystems",
            required_credentials=[],
            actions=[
                SkillAction(
                    name="overview",
                    description="One-call summary of overall agent health, service counts, revenue, fleet status, and active alerts",
                    parameters={},
                ),
                SkillAction(
                    name="register_service",
                    description="Register a service to monitor",
                    parameters={
                        "service_id": {"type": "string", "required": True, "description": "Unique service identifier"},
                        "name": {"type": "string", "required": True, "description": "Human-readable service name"},
                        "url": {"type": "string", "required": False, "description": "Service endpoint URL"},
                        "type": {"type": "string", "required": False, "description": "Service type (api, worker, web, scheduled)"},
                    },
                ),
                SkillAction(
                    name="record_check",
                    description="Record a health check result for a service",
                    parameters={
                        "service_id": {"type": "string", "required": True, "description": "Service to record check for"},
                        "latency_ms": {"type": "number", "required": False, "description": "Response latency in ms"},
                        "error_rate": {"type": "number", "required": False, "description": "Error rate 0.0-1.0"},
                        "request_count": {"type": "number", "required": False, "description": "Number of requests in period"},
                        "revenue": {"type": "number", "required": False, "description": "Revenue earned in period"},
                        "cost": {"type": "number", "required": False, "description": "Cost incurred in period"},
                    },
                ),
                SkillAction(
                    name="services",
                    description="List all monitored services with current status, uptime, and metrics",
                    parameters={
                        "status_filter": {"type": "string", "required": False, "description": "Filter by status: healthy, degraded, down"},
                    },
                ),
                SkillAction(
                    name="revenue",
                    description="Revenue dashboard - total earnings, costs, profit, per-service breakdown",
                    parameters={
                        "window_hours": {"type": "number", "required": False, "description": "Time window in hours (default: 24)"},
                    },
                ),
                SkillAction(
                    name="uptime",
                    description="Uptime report for all services over a configurable time window",
                    parameters={
                        "window_hours": {"type": "number", "required": False, "description": "Time window in hours (default: 24)"},
                        "service_id": {"type": "string", "required": False, "description": "Specific service (default: all)"},
                    },
                ),
                SkillAction(
                    name="trends",
                    description="Trend analysis - identify improving, stable, and degrading metrics",
                    parameters={
                        "service_id": {"type": "string", "required": False, "description": "Specific service (default: all)"},
                    },
                ),
                SkillAction(
                    name="report",
                    description="Generate a full text status report suitable for logging or sharing",
                    parameters={},
                ),
                SkillAction(
                    name="configure",
                    description="Update dashboard thresholds and preferences",
                    parameters={
                        "snapshot_interval_seconds": {"type": "number", "required": False, "description": "Snapshot interval"},
                        "uptime_window_hours": {"type": "number", "required": False, "description": "Default uptime window"},
                        "degraded_latency_ms": {"type": "number", "required": False, "description": "Latency threshold for degraded"},
                        "degraded_error_rate": {"type": "number", "required": False, "description": "Error rate threshold for degraded"},
                        "critical_latency_ms": {"type": "number", "required": False, "description": "Latency threshold for critical"},
                        "critical_error_rate": {"type": "number", "required": False, "description": "Error rate threshold for critical"},
                    },
                ),
                SkillAction(
                    name="status",
                    description="Dashboard own status - data freshness, service count, snapshot count",
                    parameters={},
                ),
            ],
        )

    async def execute(self, action: str, params: Dict) -> SkillResult:
        actions = {
            "overview": self._overview,
            "register_service": self._register_service,
            "record_check": self._record_check,
            "services": self._services,
            "revenue": self._revenue,
            "uptime": self._uptime,
            "trends": self._trends,
            "report": self._report,
            "configure": self._configure,
            "status": self._status,
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
            return SkillResult(success=False, message=f"Error in {action}: {e}")

    async def _register_service(self, params: Dict) -> SkillResult:
        """Register a new service for monitoring."""
        service_id = params.get("service_id", "").strip()
        name = params.get("name", "").strip()
        if not service_id or not name:
            return SkillResult(success=False, message="service_id and name are required")

        data = self._load()
        if len(data["services"]) >= MAX_SERVICES and service_id not in data["services"]:
            return SkillResult(success=False, message=f"Max services ({MAX_SERVICES}) reached")

        now = time.time()
        data["services"][service_id] = {
            "name": name,
            "url": params.get("url", ""),
            "type": params.get("type", "api"),
            "registered_at": now,
            "last_check_at": None,
            "status": STATUS_UNKNOWN,
            "current_latency_ms": 0,
            "current_error_rate": 0.0,
            "total_requests": 0,
            "total_revenue": 0.0,
            "total_cost": 0.0,
            "check_count": 0,
        }
        self._save(data)
        return SkillResult(
            success=True,
            message=f"Service '{name}' registered for monitoring",
            data={"service_id": service_id, "service": data["services"][service_id]},
        )

    async def _record_check(self, params: Dict) -> SkillResult:
        """Record a health check result for a service."""
        service_id = params.get("service_id", "").strip()
        if not service_id:
            return SkillResult(success=False, message="service_id is required")

        data = self._load()
        if service_id not in data["services"]:
            return SkillResult(success=False, message=f"Service '{service_id}' not found. Register it first.")

        svc = data["services"][service_id]
        now = time.time()

        latency_ms = float(params.get("latency_ms", 0))
        error_rate = float(params.get("error_rate", 0.0))
        request_count = int(params.get("request_count", 0))
        revenue = float(params.get("revenue", 0.0))
        cost = float(params.get("cost", 0.0))

        # Update service metrics
        svc["last_check_at"] = now
        svc["current_latency_ms"] = latency_ms
        svc["current_error_rate"] = error_rate
        svc["total_requests"] += request_count
        svc["total_revenue"] += revenue
        svc["total_cost"] += cost
        svc["check_count"] += 1
        svc["status"] = _service_status(error_rate, latency_ms, data["config"])

        # Record snapshot for uptime/trend tracking
        snapshot = {
            "service_id": service_id,
            "timestamp": now,
            "status": svc["status"],
            "latency_ms": latency_ms,
            "error_rate": error_rate,
            "request_count": request_count,
            "revenue": revenue,
            "cost": cost,
        }
        data["snapshots"].append(snapshot)
        data["stats"]["total_snapshots"] += 1
        data["stats"]["last_snapshot_at"] = now

        self._save(data)
        return SkillResult(
            success=True,
            message=f"Check recorded for '{svc['name']}': {svc['status']}",
            data={"service_id": service_id, "status": svc["status"], "snapshot": snapshot},
        )

    async def _overview(self, params: Dict) -> SkillResult:
        """Generate a unified overview of agent operational health."""
        data = self._load()
        services = data["services"]

        # Service health counts
        status_counts = {STATUS_HEALTHY: 0, STATUS_DEGRADED: 0, STATUS_DOWN: 0, STATUS_UNKNOWN: 0}
        total_revenue = 0.0
        total_cost = 0.0
        total_requests = 0

        for svc in services.values():
            status_counts[svc.get("status", STATUS_UNKNOWN)] = status_counts.get(svc.get("status", STATUS_UNKNOWN), 0) + 1
            total_revenue += svc.get("total_revenue", 0.0)
            total_cost += svc.get("total_cost", 0.0)
            total_requests += svc.get("total_requests", 0)

        # Overall health determination
        if status_counts[STATUS_DOWN] > 0:
            overall_health = SEVERITY_CRITICAL
        elif status_counts[STATUS_DEGRADED] > 0:
            overall_health = SEVERITY_WARNING
        elif len(services) == 0:
            overall_health = SEVERITY_OK
        else:
            overall_health = SEVERITY_OK

        # Fleet info from context if available
        fleet_info = await self._get_fleet_info()

        overview = {
            "overall_health": overall_health,
            "services": {
                "total": len(services),
                "by_status": status_counts,
            },
            "revenue": {
                "total_revenue": round(total_revenue, 2),
                "total_cost": round(total_cost, 2),
                "profit": round(total_revenue - total_cost, 2),
            },
            "requests": {
                "total": total_requests,
            },
            "fleet": fleet_info,
            "data_freshness": {
                "last_snapshot_at": data["stats"].get("last_snapshot_at"),
                "total_snapshots": data["stats"].get("total_snapshots", 0),
            },
        }

        return SkillResult(
            success=True,
            message=f"Dashboard: {overall_health.upper()} | {len(services)} services | "
                    f"${total_revenue:.2f} revenue | ${total_revenue - total_cost:.2f} profit",
            data=overview,
        )

    async def _get_fleet_info(self) -> Dict:
        """Pull fleet info from HealthMonitor via context if available."""
        if not self.context:
            return {"available": False, "reason": "no context"}
        try:
            result = await self.context.call_skill("health_monitor", "fleet_status", {})
            if result.success:
                return {"available": True, **result.data}
        except Exception:
            pass
        return {"available": False, "reason": "health_monitor not available"}

    async def _services(self, params: Dict) -> SkillResult:
        """List all monitored services with current metrics."""
        data = self._load()
        status_filter = params.get("status_filter", "").strip()

        services_list = []
        config = data["config"]
        window_hours = config.get("uptime_window_hours", 24)

        for sid, svc in data["services"].items():
            if status_filter and svc.get("status") != status_filter:
                continue
            uptime = _compute_uptime(data["snapshots"], sid, window_hours)
            services_list.append({
                "service_id": sid,
                "name": svc["name"],
                "type": svc.get("type", "api"),
                "status": svc.get("status", STATUS_UNKNOWN),
                "severity": _determine_severity(
                    svc.get("current_error_rate", 0),
                    svc.get("current_latency_ms", 0),
                    config,
                ),
                "latency_ms": svc.get("current_latency_ms", 0),
                "error_rate": svc.get("current_error_rate", 0.0),
                "uptime_pct": uptime,
                "total_requests": svc.get("total_requests", 0),
                "revenue": round(svc.get("total_revenue", 0.0), 2),
                "cost": round(svc.get("total_cost", 0.0), 2),
                "profit": round(svc.get("total_revenue", 0.0) - svc.get("total_cost", 0.0), 2),
                "check_count": svc.get("check_count", 0),
            })

        # Sort: critical first, then by name
        severity_order = {SEVERITY_CRITICAL: 0, SEVERITY_WARNING: 1, SEVERITY_OK: 2}
        services_list.sort(key=lambda s: (severity_order.get(s["severity"], 3), s["name"]))

        return SkillResult(
            success=True,
            message=f"{len(services_list)} services" + (f" (filtered: {status_filter})" if status_filter else ""),
            data={"services": services_list, "total": len(services_list)},
        )

    async def _revenue(self, params: Dict) -> SkillResult:
        """Revenue dashboard with per-service breakdown and totals."""
        data = self._load()
        window_hours = float(params.get("window_hours", 24))
        cutoff = time.time() - (window_hours * 3600)

        # Per-service revenue from snapshots within window
        service_revenue = {}
        for snap in data["snapshots"]:
            if snap.get("timestamp", 0) < cutoff:
                continue
            sid = snap.get("service_id", "")
            if sid not in service_revenue:
                svc = data["services"].get(sid, {})
                service_revenue[sid] = {
                    "name": svc.get("name", sid),
                    "revenue": 0.0,
                    "cost": 0.0,
                    "requests": 0,
                }
            service_revenue[sid]["revenue"] += snap.get("revenue", 0.0)
            service_revenue[sid]["cost"] += snap.get("cost", 0.0)
            service_revenue[sid]["requests"] += snap.get("request_count", 0)

        # Add profit
        for entry in service_revenue.values():
            entry["profit"] = round(entry["revenue"] - entry["cost"], 2)
            entry["revenue"] = round(entry["revenue"], 2)
            entry["cost"] = round(entry["cost"], 2)

        total_revenue = sum(e["revenue"] for e in service_revenue.values())
        total_cost = sum(e["cost"] for e in service_revenue.values())
        total_profit = round(total_revenue - total_cost, 2)

        # Sort by revenue descending
        breakdown = sorted(service_revenue.values(), key=lambda e: e["revenue"], reverse=True)

        return SkillResult(
            success=True,
            message=f"Revenue ({window_hours}h): ${total_revenue:.2f} revenue, "
                    f"${total_cost:.2f} cost, ${total_profit:.2f} profit",
            data={
                "window_hours": window_hours,
                "total_revenue": round(total_revenue, 2),
                "total_cost": round(total_cost, 2),
                "total_profit": total_profit,
                "services": breakdown,
            },
        )

    async def _uptime(self, params: Dict) -> SkillResult:
        """Uptime report for services over a time window."""
        data = self._load()
        window_hours = float(params.get("window_hours", data["config"].get("uptime_window_hours", 24)))
        service_id_filter = params.get("service_id", "").strip()

        uptime_report = []
        for sid, svc in data["services"].items():
            if service_id_filter and sid != service_id_filter:
                continue
            pct = _compute_uptime(data["snapshots"], sid, window_hours)
            uptime_report.append({
                "service_id": sid,
                "name": svc["name"],
                "uptime_pct": pct,
                "status": svc.get("status", STATUS_UNKNOWN),
                "check_count": svc.get("check_count", 0),
            })

        uptime_report.sort(key=lambda r: r["uptime_pct"])

        avg_uptime = (sum(r["uptime_pct"] for r in uptime_report) / len(uptime_report)) if uptime_report else 100.0

        return SkillResult(
            success=True,
            message=f"Uptime ({window_hours}h): {avg_uptime:.1f}% avg across {len(uptime_report)} services",
            data={
                "window_hours": window_hours,
                "average_uptime_pct": round(avg_uptime, 2),
                "services": uptime_report,
            },
        )

    async def _trends(self, params: Dict) -> SkillResult:
        """Analyze trends in latency, error rates, and revenue."""
        data = self._load()
        service_id_filter = params.get("service_id", "").strip()

        trends_data = []
        for sid, svc in data["services"].items():
            if service_id_filter and sid != service_id_filter:
                continue
            # Get snapshots for this service
            svc_snaps = [s for s in data["snapshots"] if s.get("service_id") == sid]
            if len(svc_snaps) < 2:
                trends_data.append({
                    "service_id": sid,
                    "name": svc["name"],
                    "latency_trend": TREND_STABLE,
                    "error_rate_trend": TREND_STABLE,
                    "revenue_trend": TREND_STABLE,
                    "data_points": len(svc_snaps),
                })
                continue

            # Sort by timestamp
            svc_snaps.sort(key=lambda s: s.get("timestamp", 0))

            latencies = [s.get("latency_ms", 0) for s in svc_snaps]
            error_rates = [s.get("error_rate", 0) for s in svc_snaps]
            revenues = [s.get("revenue", 0) for s in svc_snaps]

            # For latency and error rate, "improving" means going DOWN
            latency_trend_raw = _compute_trend(latencies)
            error_trend_raw = _compute_trend(error_rates)
            revenue_trend = _compute_trend(revenues)

            # Flip for latency/error: lower is better
            flip = {TREND_UP: TREND_DOWN, TREND_DOWN: TREND_UP, TREND_STABLE: TREND_STABLE}
            latency_trend = flip[latency_trend_raw]
            error_rate_trend = flip[error_trend_raw]

            trends_data.append({
                "service_id": sid,
                "name": svc["name"],
                "latency_trend": latency_trend,
                "error_rate_trend": error_rate_trend,
                "revenue_trend": revenue_trend,
                "data_points": len(svc_snaps),
            })

        improving = [t for t in trends_data if t["latency_trend"] == TREND_UP or t["revenue_trend"] == TREND_UP]
        degrading = [t for t in trends_data if t["latency_trend"] == TREND_DOWN or t["error_rate_trend"] == TREND_DOWN]

        return SkillResult(
            success=True,
            message=f"Trends: {len(improving)} improving, {len(degrading)} degrading, "
                    f"{len(trends_data) - len(improving) - len(degrading)} stable",
            data={
                "services": trends_data,
                "summary": {
                    "improving_count": len(improving),
                    "degrading_count": len(degrading),
                    "stable_count": len(trends_data) - len(improving) - len(degrading),
                },
            },
        )

    async def _report(self, params: Dict) -> SkillResult:
        """Generate a full-text operational status report."""
        # Gather data from other actions
        overview_result = await self._overview({})
        services_result = await self._services({})
        revenue_result = await self._revenue({})
        uptime_result = await self._uptime({})

        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        lines = [
            f"=== OPERATIONAL STATUS REPORT ===",
            f"Generated: {now_str}",
            f"",
        ]

        # Overview section
        if overview_result.success:
            ov = overview_result.data
            lines.append(f"## Overall Health: {ov['overall_health'].upper()}")
            lines.append(f"Services: {ov['services']['total']} total")
            sc = ov['services']['by_status']
            lines.append(f"  Healthy: {sc.get('healthy', 0)} | Degraded: {sc.get('degraded', 0)} | Down: {sc.get('down', 0)}")
            lines.append(f"Revenue: ${ov['revenue']['total_revenue']:.2f} | Cost: ${ov['revenue']['total_cost']:.2f} | Profit: ${ov['revenue']['profit']:.2f}")
            lines.append("")

        # Services section
        if services_result.success and services_result.data.get("services"):
            lines.append("## Services")
            for svc in services_result.data["services"]:
                status_icon = {"healthy": "+", "degraded": "~", "down": "!", "unknown": "?"}
                icon = status_icon.get(svc["status"], "?")
                lines.append(f"  [{icon}] {svc['name']} ({svc['service_id']})")
                lines.append(f"      Status: {svc['status']} | Latency: {svc['latency_ms']}ms | "
                             f"Errors: {svc['error_rate']:.1%} | Uptime: {svc['uptime_pct']:.1f}%")
            lines.append("")

        # Revenue section
        if revenue_result.success:
            rv = revenue_result.data
            lines.append(f"## Revenue ({rv['window_hours']}h)")
            lines.append(f"  Total: ${rv['total_revenue']:.2f} | Cost: ${rv['total_cost']:.2f} | Profit: ${rv['total_profit']:.2f}")
            if rv.get("services"):
                for entry in rv["services"][:5]:
                    lines.append(f"  - {entry['name']}: ${entry['revenue']:.2f} revenue ({entry['requests']} reqs)")
            lines.append("")

        # Uptime section
        if uptime_result.success:
            up = uptime_result.data
            lines.append(f"## Uptime ({up['window_hours']}h)")
            lines.append(f"  Average: {up['average_uptime_pct']:.1f}%")
            for svc in up.get("services", [])[:5]:
                lines.append(f"  - {svc['name']}: {svc['uptime_pct']:.1f}%")
            lines.append("")

        lines.append("=== END REPORT ===")
        report_text = "\n".join(lines)

        return SkillResult(
            success=True,
            message=f"Status report generated ({len(lines)} lines)",
            data={"report": report_text, "generated_at": now_str},
        )

    async def _configure(self, params: Dict) -> SkillResult:
        """Update dashboard configuration."""
        data = self._load()
        config = data["config"]

        updatable = [
            "snapshot_interval_seconds", "uptime_window_hours",
            "degraded_latency_ms", "critical_latency_ms",
            "degraded_error_rate", "critical_error_rate",
        ]
        updated = {}
        for key in updatable:
            if key in params and params[key] is not None:
                config[key] = float(params[key])
                updated[key] = config[key]

        if not updated:
            return SkillResult(
                success=True,
                message="No configuration changes (current config returned)",
                data={"config": config},
            )

        self._save(data)
        return SkillResult(
            success=True,
            message=f"Updated {len(updated)} config keys: {list(updated.keys())}",
            data={"config": config, "updated": updated},
        )

    async def _status(self, params: Dict) -> SkillResult:
        """Dashboard own status and data freshness."""
        data = self._load()
        last_snap = data["stats"].get("last_snapshot_at")
        freshness = "never"
        if last_snap:
            age_seconds = time.time() - last_snap
            if age_seconds < 60:
                freshness = f"{age_seconds:.0f}s ago"
            elif age_seconds < 3600:
                freshness = f"{age_seconds / 60:.0f}m ago"
            else:
                freshness = f"{age_seconds / 3600:.1f}h ago"

        return SkillResult(
            success=True,
            message=f"Dashboard: {len(data['services'])} services, "
                    f"{data['stats']['total_snapshots']} snapshots, "
                    f"last update: {freshness}",
            data={
                "service_count": len(data["services"]),
                "snapshot_count": data["stats"]["total_snapshots"],
                "last_snapshot_at": last_snap,
                "data_freshness": freshness,
                "config": data["config"],
            },
        )

    async def initialize(self) -> bool:
        """Always initializes successfully (no credentials required)."""
        self.initialized = True
        return True
