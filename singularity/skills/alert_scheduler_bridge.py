#!/usr/bin/env python3
"""
AlertSchedulerBridgeSkill - Wire revenue monitoring into autonomous scheduled operation.

Connects RevenueAlertEscalationSkill + RevenueForecastSkill to SchedulerSkill so
the agent continuously and autonomously monitors its revenue health without
manual prompting.

Without this bridge, revenue alerts and forecasts only run when explicitly requested.
With it, the agent autonomously:
  1. Schedules periodic revenue health checks (alert rule evaluation)
  2. Schedules periodic revenue forecast regeneration
  3. Auto-triggers forecasts when alerts fire (reactive analysis)
  4. Tracks monitoring uptime and reliability
  5. Provides a unified monitoring dashboard
  6. Supports configurable check intervals and forecast cadence

Revenue monitoring flow becomes fully autonomous:
  [Scheduler ticks] → AlertEscalation.check → Alert fired?
                                             → Yes → ForecastSkill.forecast (reactive)
                                                   → ForecastSkill.breakeven (impact)
                                             → No  → Health OK, log snapshot
  [Scheduler ticks] → ForecastSkill.forecast → Periodic projections

Actions:
  - setup: Configure and activate autonomous monitoring schedule
  - status: Show monitoring status (next checks, last results, uptime)
  - run_now: Trigger an immediate monitoring cycle
  - configure: Update check intervals, forecast cadence, reactive triggers
  - history: View monitoring run history with alert/forecast results
  - pause: Temporarily pause all monitoring
  - resume: Resume paused monitoring
  - dashboard: Unified view of alerts + forecasts + health

Pillar: Revenue (primary) + Self-Improvement (supporting)
  - Revenue: Continuous autonomous protection of revenue streams
  - Self-Improvement: Agent monitors its own business health without prompting
"""

import json
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import Skill, SkillAction, SkillManifest, SkillResult

DATA_DIR = Path(__file__).parent.parent / "data"
STATE_FILE = DATA_DIR / "alert_scheduler_bridge.json"
MAX_HISTORY = 200


def _now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _now_ts() -> float:
    return time.time()


class AlertSchedulerBridgeSkill(Skill):
    """
    Bridges revenue monitoring (alerts + forecasts) to the scheduler for
    autonomous, continuous operation.
    """

    def __init__(self, credentials: Dict[str, str] = None):
        super().__init__(credentials)
        self._state = self._load_state()

    # ── State persistence ──────────────────────────────────────────

    def _load_state(self) -> dict:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        if STATE_FILE.exists():
            try:
                return json.loads(STATE_FILE.read_text())
            except (json.JSONDecodeError, OSError):
                pass
        return self._default_state()

    def _default_state(self) -> dict:
        return {
            "enabled": False,
            "config": {
                "alert_check_interval_seconds": 300,     # 5 min
                "forecast_interval_seconds": 3600,        # 1 hour
                "reactive_forecast_on_alert": True,       # auto-forecast when alert fires
                "reactive_breakeven_on_drop": True,       # auto-breakeven when revenue drops
                "forecast_periods": 7,
                "forecast_model": "exponential_smoothing",
                "breakeven_cost_per_period": 1.0,
            },
            "schedule_ids": {
                "alert_check": None,
                "forecast": None,
            },
            "stats": {
                "total_checks": 0,
                "total_forecasts": 0,
                "alerts_fired": 0,
                "reactive_forecasts": 0,
                "last_check_at": None,
                "last_forecast_at": None,
                "last_alert_at": None,
                "monitoring_started_at": None,
                "uptime_checks": 0,
                "failed_checks": 0,
            },
            "history": [],
        }

    def _save_state(self):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        STATE_FILE.write_text(json.dumps(self._state, indent=2, default=str))

    # ── Manifest ───────────────────────────────────────────────────

    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="alert_scheduler_bridge",
            name="alert_scheduler_bridge",
            category="revenue",
            description="Autonomous revenue monitoring via scheduled alert checks and forecasts",
            version="1.0.0",
            required_credentials=[],
            actions=[
                SkillAction(
                    name="setup",
                    description="Configure and activate autonomous monitoring. Schedules periodic alert checks and forecast regeneration.",
                    parameters={
                        "alert_interval": "Alert check interval in seconds (default: 300)",
                        "forecast_interval": "Forecast regeneration interval in seconds (default: 3600)",
                        "reactive_forecast": "Auto-forecast when alerts fire (default: true)",
                        "forecast_periods": "Number of periods to forecast (default: 7)",
                        "forecast_model": "Forecast model to use (default: exponential_smoothing)",
                        "breakeven_cost": "Cost per period for breakeven analysis (default: 1.0)",
                    },
                ),
                SkillAction(
                    name="status",
                    description="Show current monitoring status including next checks, last results, and uptime",
                    parameters={},
                ),
                SkillAction(
                    name="run_now",
                    description="Trigger an immediate monitoring cycle (alert check + optional forecast)",
                    parameters={
                        "include_forecast": "Also run a forecast (default: true)",
                        "metrics": "Optional metrics dict to check against alert rules",
                    },
                ),
                SkillAction(
                    name="configure",
                    description="Update monitoring configuration without restarting schedules",
                    parameters={
                        "alert_interval": "Alert check interval in seconds",
                        "forecast_interval": "Forecast regeneration interval in seconds",
                        "reactive_forecast": "Auto-forecast when alerts fire",
                        "forecast_periods": "Number of periods to forecast",
                        "forecast_model": "Forecast model to use",
                        "breakeven_cost": "Cost per period for breakeven analysis",
                    },
                ),
                SkillAction(
                    name="history",
                    description="View monitoring run history with alert/forecast results",
                    parameters={
                        "limit": "Number of history entries to return (default: 20)",
                        "type": "Filter by type: 'alert', 'forecast', 'reactive', or 'all' (default: all)",
                    },
                ),
                SkillAction(
                    name="pause",
                    description="Temporarily pause all autonomous monitoring",
                    parameters={},
                ),
                SkillAction(
                    name="resume",
                    description="Resume paused monitoring",
                    parameters={},
                ),
                SkillAction(
                    name="dashboard",
                    description="Unified monitoring dashboard: alerts + forecasts + health + uptime",
                    parameters={},
                ),
            ],
        )

    # ── Execute dispatch ───────────────────────────────────────────

    async def execute(self, action: str, params: Dict[str, Any]) -> SkillResult:
        handlers = {
            "setup": self._setup,
            "status": self._status,
            "run_now": self._run_now,
            "configure": self._configure,
            "history": self._history,
            "pause": self._pause,
            "resume": self._resume,
            "dashboard": self._dashboard,
        }
        handler = handlers.get(action)
        if not handler:
            return SkillResult(
                success=False,
                message=f"Unknown action: {action}. Available: {', '.join(handlers.keys())}",
            )
        return await handler(params)

    # ── Action: setup ──────────────────────────────────────────────

    async def _setup(self, params: Dict[str, Any]) -> SkillResult:
        """Configure and activate autonomous monitoring."""
        config = self._state["config"]

        # Apply user overrides
        if "alert_interval" in params:
            config["alert_check_interval_seconds"] = max(30, int(params["alert_interval"]))
        if "forecast_interval" in params:
            config["forecast_interval_seconds"] = max(60, int(params["forecast_interval"]))
        if "reactive_forecast" in params:
            config["reactive_forecast_on_alert"] = bool(params["reactive_forecast"])
        if "forecast_periods" in params:
            config["forecast_periods"] = max(1, int(params["forecast_periods"]))
        if "forecast_model" in params:
            config["forecast_model"] = str(params["forecast_model"])
        if "breakeven_cost" in params:
            config["breakeven_cost_per_period"] = float(params["breakeven_cost"])

        # Create schedule entries (actual scheduling happens via scheduler skill)
        schedule_specs = self._build_schedule_specs()

        self._state["enabled"] = True
        self._state["stats"]["monitoring_started_at"] = _now_iso()
        self._save_state()

        return SkillResult(
            success=True,
            message=(
                f"Monitoring activated: alert checks every {config['alert_check_interval_seconds']}s, "
                f"forecasts every {config['forecast_interval_seconds']}s, "
                f"reactive forecasts {'ON' if config['reactive_forecast_on_alert'] else 'OFF'}"
            ),
            data={
                "enabled": True,
                "config": config,
                "schedules": schedule_specs,
                "instructions": (
                    "To activate scheduling, create these tasks via the scheduler skill:\n"
                    + "\n".join(
                        f"  scheduler:schedule {json.dumps(s)}" for s in schedule_specs
                    )
                ),
            },
        )

    def _build_schedule_specs(self) -> List[dict]:
        """Build scheduler task specs for alert checks and forecasts."""
        config = self._state["config"]
        return [
            {
                "name": "revenue_alert_check",
                "skill_id": "revenue_alert_escalation",
                "action": "check",
                "params": {},
                "interval_seconds": config["alert_check_interval_seconds"],
                "recurring": True,
                "max_runs": 0,
            },
            {
                "name": "revenue_forecast_refresh",
                "skill_id": "revenue_forecast",
                "action": "forecast",
                "params": {
                    "periods": config["forecast_periods"],
                    "model": config["forecast_model"],
                },
                "interval_seconds": config["forecast_interval_seconds"],
                "recurring": True,
                "max_runs": 0,
            },
        ]

    # ── Action: status ─────────────────────────────────────────────

    async def _status(self, params: Dict[str, Any]) -> SkillResult:
        """Show current monitoring status."""
        stats = self._state["stats"]
        config = self._state["config"]
        enabled = self._state["enabled"]

        reliability = 0.0
        total = stats["uptime_checks"]
        if total > 0:
            reliability = ((total - stats["failed_checks"]) / total) * 100

        return SkillResult(
            success=True,
            message=(
                f"Monitoring {'ACTIVE' if enabled else 'PAUSED'} | "
                f"Checks: {stats['total_checks']} | Forecasts: {stats['total_forecasts']} | "
                f"Alerts fired: {stats['alerts_fired']} | Reliability: {reliability:.1f}%"
            ),
            data={
                "enabled": enabled,
                "config": config,
                "stats": stats,
                "reliability_pct": reliability,
                "schedule_ids": self._state["schedule_ids"],
            },
        )

    # ── Action: run_now ────────────────────────────────────────────

    async def _run_now(self, params: Dict[str, Any]) -> SkillResult:
        """Trigger immediate monitoring cycle."""
        include_forecast = params.get("include_forecast", True)
        metrics = params.get("metrics")
        config = self._state["config"]
        results = {}

        # 1. Run alert check
        alert_result = await self._run_alert_check(metrics)
        results["alert_check"] = alert_result

        # 2. Check if alerts were fired
        alerts_fired = alert_result.get("alerts_fired", 0)
        has_revenue_drop = alert_result.get("has_revenue_drop", False)

        # 3. Reactive forecast on alert
        if alerts_fired > 0 and config["reactive_forecast_on_alert"]:
            reactive = await self._run_reactive_forecast(has_revenue_drop)
            results["reactive_forecast"] = reactive

        # 4. Scheduled forecast if requested
        if include_forecast:
            forecast_result = await self._run_forecast()
            results["forecast"] = forecast_result

        # Record in history
        self._record_run("manual", results)

        fired_msg = f", {alerts_fired} alerts fired" if alerts_fired > 0 else ""
        return SkillResult(
            success=True,
            message=f"Monitoring cycle complete{fired_msg}",
            data=results,
        )

    async def _run_alert_check(self, metrics: Optional[dict] = None) -> dict:
        """Execute revenue alert check. Uses SkillContext if available."""
        self._state["stats"]["total_checks"] += 1
        self._state["stats"]["uptime_checks"] += 1
        self._state["stats"]["last_check_at"] = _now_iso()

        check_params = {}
        if metrics:
            check_params["metrics"] = metrics

        # Try to call via skill context
        result = await self._call_skill("revenue_alert_escalation", "check", check_params)

        if result is None:
            self._state["stats"]["failed_checks"] += 1
            self._save_state()
            return {
                "success": False,
                "error": "revenue_alert_escalation skill not available via context",
                "alerts_fired": 0,
                "has_revenue_drop": False,
            }

        alerts_fired = 0
        has_revenue_drop = False
        if result.get("success") and result.get("data"):
            fired = result["data"].get("fired", [])
            alerts_fired = len(fired)
            has_revenue_drop = any(
                a.get("rule") == "revenue_drop" for a in fired
            )
            if alerts_fired > 0:
                self._state["stats"]["alerts_fired"] += alerts_fired
                self._state["stats"]["last_alert_at"] = _now_iso()

        self._save_state()
        return {
            "success": True,
            "alerts_fired": alerts_fired,
            "has_revenue_drop": has_revenue_drop,
            "active_alerts": result.get("data", {}).get("active_count", 0),
            "detail": result.get("data"),
        }

    async def _run_forecast(self) -> dict:
        """Execute revenue forecast."""
        config = self._state["config"]
        self._state["stats"]["total_forecasts"] += 1
        self._state["stats"]["last_forecast_at"] = _now_iso()

        result = await self._call_skill(
            "revenue_forecast",
            "forecast",
            {
                "periods": config["forecast_periods"],
                "model": config["forecast_model"],
            },
        )

        if result is None:
            self._save_state()
            return {"success": False, "error": "revenue_forecast skill not available"}

        self._save_state()
        return {"success": True, "detail": result.get("data")}

    async def _run_reactive_forecast(self, include_breakeven: bool) -> dict:
        """Run reactive forecast + breakeven when alert fires."""
        config = self._state["config"]
        self._state["stats"]["reactive_forecasts"] += 1
        results = {}

        # Forecast
        forecast_result = await self._call_skill(
            "revenue_forecast",
            "forecast",
            {"periods": config["forecast_periods"], "model": config["forecast_model"]},
        )
        results["forecast"] = forecast_result.get("data") if forecast_result else None

        # Breakeven if revenue dropped
        if include_breakeven and config["reactive_breakeven_on_drop"]:
            breakeven_result = await self._call_skill(
                "revenue_forecast",
                "breakeven",
                {"cost_per_period": config["breakeven_cost_per_period"]},
            )
            results["breakeven"] = breakeven_result.get("data") if breakeven_result else None

        # Trend analysis
        trend_result = await self._call_skill("revenue_forecast", "trend", {})
        results["trend"] = trend_result.get("data") if trend_result else None

        self._save_state()
        return {"success": True, "reactive": True, "detail": results}

    async def _call_skill(
        self, skill_id: str, action: str, params: dict
    ) -> Optional[dict]:
        """Call another skill via SkillContext or return None if unavailable."""
        if hasattr(self, "context") and self.context:
            try:
                result = await self.context.call_skill(skill_id, action, params)
                if hasattr(result, "success"):
                    return {
                        "success": result.success,
                        "message": getattr(result, "message", ""),
                        "data": getattr(result, "data", None),
                    }
                return result
            except Exception:
                return None
        return None

    # ── Action: configure ──────────────────────────────────────────

    async def _configure(self, params: Dict[str, Any]) -> SkillResult:
        """Update monitoring configuration."""
        config = self._state["config"]
        updated = []

        field_map = {
            "alert_interval": ("alert_check_interval_seconds", int, 30),
            "forecast_interval": ("forecast_interval_seconds", int, 60),
            "reactive_forecast": ("reactive_forecast_on_alert", bool, None),
            "forecast_periods": ("forecast_periods", int, 1),
            "forecast_model": ("forecast_model", str, None),
            "breakeven_cost": ("breakeven_cost_per_period", float, None),
        }

        for param_key, (config_key, cast_fn, min_val) in field_map.items():
            if param_key in params:
                val = cast_fn(params[param_key])
                if min_val is not None and isinstance(val, (int, float)):
                    val = max(min_val, val)
                config[config_key] = val
                updated.append(f"{param_key}={val}")

        if not updated:
            return SkillResult(
                success=False,
                message="No configuration parameters provided",
                data={"available": list(field_map.keys())},
            )

        self._save_state()
        return SkillResult(
            success=True,
            message=f"Updated: {', '.join(updated)}",
            data={"config": config},
        )

    # ── Action: history ────────────────────────────────────────────

    async def _history(self, params: Dict[str, Any]) -> SkillResult:
        """View monitoring run history."""
        limit = int(params.get("limit", 20))
        type_filter = params.get("type", "all")

        history = self._state["history"]
        if type_filter != "all":
            history = [h for h in history if h.get("type") == type_filter]

        recent = history[-limit:]
        recent.reverse()  # newest first

        return SkillResult(
            success=True,
            message=f"Showing {len(recent)} of {len(history)} monitoring runs",
            data={"runs": recent, "total": len(history), "filter": type_filter},
        )

    # ── Action: pause ──────────────────────────────────────────────

    async def _pause(self, params: Dict[str, Any]) -> SkillResult:
        """Pause autonomous monitoring."""
        if not self._state["enabled"]:
            return SkillResult(success=False, message="Monitoring is already paused")

        self._state["enabled"] = False
        self._save_state()

        return SkillResult(
            success=True,
            message="Monitoring paused. Use 'resume' to reactivate.",
            data={"enabled": False, "schedule_ids": self._state["schedule_ids"]},
        )

    # ── Action: resume ─────────────────────────────────────────────

    async def _resume(self, params: Dict[str, Any]) -> SkillResult:
        """Resume paused monitoring."""
        if self._state["enabled"]:
            return SkillResult(success=False, message="Monitoring is already active")

        self._state["enabled"] = True
        self._save_state()

        return SkillResult(
            success=True,
            message="Monitoring resumed",
            data={"enabled": True},
        )

    # ── Action: dashboard ──────────────────────────────────────────

    async def _dashboard(self, params: Dict[str, Any]) -> SkillResult:
        """Unified monitoring dashboard."""
        stats = self._state["stats"]
        config = self._state["config"]
        enabled = self._state["enabled"]

        # Compute reliability
        total = stats["uptime_checks"]
        reliability = ((total - stats["failed_checks"]) / total * 100) if total > 0 else 0.0

        # Compute monitoring uptime duration
        uptime_hours = 0.0
        if stats["monitoring_started_at"]:
            try:
                start = datetime.fromisoformat(
                    stats["monitoring_started_at"].replace("Z", "+00:00")
                )
                uptime_hours = (
                    datetime.now(start.tzinfo) - start
                ).total_seconds() / 3600
            except (ValueError, TypeError):
                pass

        # Recent history summary
        history = self._state["history"]
        recent = history[-10:] if history else []
        recent_alerts = sum(1 for h in recent if h.get("alerts_fired", 0) > 0)

        # Build dashboard
        dashboard = {
            "status": "ACTIVE" if enabled else "PAUSED",
            "monitoring": {
                "total_checks": stats["total_checks"],
                "total_forecasts": stats["total_forecasts"],
                "alerts_fired_total": stats["alerts_fired"],
                "reactive_forecasts": stats["reactive_forecasts"],
                "reliability_pct": round(reliability, 1),
                "uptime_hours": round(uptime_hours, 1),
            },
            "last_activity": {
                "last_check": stats["last_check_at"],
                "last_forecast": stats["last_forecast_at"],
                "last_alert": stats["last_alert_at"],
            },
            "schedule": {
                "alert_check_every": f"{config['alert_check_interval_seconds']}s",
                "forecast_every": f"{config['forecast_interval_seconds']}s",
                "reactive_forecast": config["reactive_forecast_on_alert"],
            },
            "recent_10_runs": {
                "total": len(recent),
                "with_alerts": recent_alerts,
                "runs": recent,
            },
        }

        return SkillResult(
            success=True,
            message=(
                f"Dashboard: {'ACTIVE' if enabled else 'PAUSED'} | "
                f"{stats['total_checks']} checks, {stats['alerts_fired']} alerts, "
                f"{reliability:.1f}% reliability"
            ),
            data=dashboard,
        )

    # ── Internal helpers ───────────────────────────────────────────

    def _record_run(self, trigger_type: str, results: dict):
        """Record a monitoring run in history."""
        alerts_fired = 0
        alert_check = results.get("alert_check", {})
        if isinstance(alert_check, dict):
            alerts_fired = alert_check.get("alerts_fired", 0)

        run_type = "manual"
        if trigger_type == "scheduled":
            run_type = "scheduled"
        elif alerts_fired > 0 and results.get("reactive_forecast"):
            run_type = "reactive"

        entry = {
            "id": f"mon_{uuid.uuid4().hex[:8]}",
            "timestamp": _now_iso(),
            "trigger": trigger_type,
            "type": run_type,
            "alerts_fired": alerts_fired,
            "has_forecast": "forecast" in results,
            "has_reactive": "reactive_forecast" in results,
            "success": all(
                r.get("success", False)
                for r in results.values()
                if isinstance(r, dict)
            ),
        }

        self._state["history"].append(entry)
        # Trim history
        if len(self._state["history"]) > MAX_HISTORY:
            self._state["history"] = self._state["history"][-MAX_HISTORY:]
        self._save_state()
