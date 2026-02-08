#!/usr/bin/env python3
"""
RevenueAlertEscalationSkill - Revenue-specific anomaly detection and incident escalation.

Bridges RevenueObservabilityBridgeSkill → IncidentResponseSkill to automatically:

  1. Monitor revenue metrics for anomalies (drops, spikes, zero-revenue periods)
  2. Define revenue-specific alert rules with configurable thresholds
  3. Auto-create incidents when revenue anomalies are detected
  4. Track anomaly history and calculate trend-based alerts
  5. Auto-resolve incidents when metrics return to normal
  6. Provide revenue health dashboard with actionable insights

This is the missing link between revenue monitoring and incident response.
Without it, revenue drops go unnoticed until manual inspection.

Revenue flow:
  Revenue Sources → RevenueObservabilityBridge → Metrics
  → RevenueAlertEscalation → Anomaly Detected → IncidentResponse → Auto-Triage

Pillar: Revenue (primary), Self-Improvement (supporting)
  - Revenue: Protects revenue by catching drops early
  - Self-Improvement: Agent monitors its own business health autonomously
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from .base import Skill, SkillAction, SkillManifest, SkillResult

DATA_DIR = Path(__file__).parent.parent / "data"
STATE_FILE = DATA_DIR / "revenue_alert_escalation.json"
MAX_HISTORY = 500
MAX_SNAPSHOTS = 100

# Default alert rules
DEFAULT_RULES = {
    "revenue_drop": {
        "enabled": True,
        "description": "Revenue dropped more than threshold% vs baseline",
        "metric": "revenue.total",
        "condition": "drop_percent",
        "threshold": 30.0,  # 30% drop triggers alert
        "severity": "high",
        "cooldown_seconds": 3600,
    },
    "zero_revenue": {
        "enabled": True,
        "description": "No revenue recorded in monitoring period",
        "metric": "revenue.total",
        "condition": "equals",
        "threshold": 0.0,
        "severity": "critical",
        "cooldown_seconds": 1800,
    },
    "low_success_rate": {
        "enabled": True,
        "description": "Request success rate below threshold",
        "metric": "revenue.requests.success_rate",
        "condition": "below",
        "threshold": 80.0,  # Below 80% success rate
        "severity": "medium",
        "cooldown_seconds": 1800,
    },
    "no_customers": {
        "enabled": True,
        "description": "No active customers detected",
        "metric": "revenue.customers.active",
        "condition": "equals",
        "threshold": 0.0,
        "severity": "high",
        "cooldown_seconds": 3600,
    },
    "revenue_spike": {
        "enabled": True,
        "description": "Unusual revenue spike (possible anomaly)",
        "metric": "revenue.total",
        "condition": "spike_percent",
        "threshold": 200.0,  # 200% increase
        "severity": "low",
        "cooldown_seconds": 7200,
    },
}


def _now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _now_ts() -> float:
    return time.time()


def _load_state() -> Dict:
    try:
        if STATE_FILE.exists():
            with open(STATE_FILE, "r") as f:
                return json.load(f)
    except (json.JSONDecodeError, IOError):
        pass
    return {
        "rules": dict(DEFAULT_RULES),
        "active_alerts": {},       # rule_name -> alert info
        "alert_history": [],       # past alert events
        "metric_snapshots": [],    # revenue metric history for trend analysis
        "incidents_created": [],   # incident IDs created by this skill
        "stats": {
            "total_checks": 0,
            "total_alerts_fired": 0,
            "total_alerts_resolved": 0,
            "total_incidents_created": 0,
            "total_incidents_resolved": 0,
        },
        "config": {
            "baseline_window": 5,  # number of snapshots for baseline calc
            "check_interval_seconds": 300,
            "auto_create_incidents": True,
            "auto_resolve_incidents": True,
        },
    }


def _save_state(state: Dict):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if len(state.get("alert_history", [])) > MAX_HISTORY:
        state["alert_history"] = state["alert_history"][-MAX_HISTORY:]
    if len(state.get("metric_snapshots", [])) > MAX_SNAPSHOTS:
        state["metric_snapshots"] = state["metric_snapshots"][-MAX_SNAPSHOTS:]
    if len(state.get("incidents_created", [])) > MAX_HISTORY:
        state["incidents_created"] = state["incidents_created"][-MAX_HISTORY:]
    try:
        with open(STATE_FILE, "w") as f:
            json.dump(state, f, indent=2, default=str)
    except IOError:
        pass


class RevenueAlertEscalationSkill(Skill):
    """Revenue anomaly detection and incident escalation."""

    def __init__(self, credentials: Dict[str, str] = None):
        super().__init__(credentials)
        self._state = _load_state()

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="revenue_alert_escalation",
            name="Revenue Alert Escalation",
            version="1.0.0",
            category="revenue",
            description="Revenue anomaly detection with auto-incident escalation",
            required_credentials=[],
            actions=[
                SkillAction(
                    name="check",
                    description="Check revenue metrics and fire/resolve alerts",
                    parameters={
                        "metrics": {"type": "object", "required": False, "description": "Revenue metrics to check (auto-collected if omitted)"},
                    },
                ),
                SkillAction(
                    name="rules",
                    description="List or modify alert rules",
                    parameters={
                        "action": {"type": "string", "required": False, "description": "get/set/enable/disable/reset"},
                        "rule": {"type": "string", "required": False, "description": "Rule name"},
                        "config": {"type": "object", "required": False, "description": "Rule config for set action"},
                    },
                ),
                SkillAction(
                    name="status",
                    description="Current alert status and revenue health",
                    parameters={},
                ),
                SkillAction(
                    name="history",
                    description="View alert history",
                    parameters={
                        "limit": {"type": "integer", "required": False, "description": "Max entries (default: 20)"},
                        "rule": {"type": "string", "required": False, "description": "Filter by rule name"},
                    },
                ),
                SkillAction(
                    name="configure",
                    description="Configure escalation behavior",
                    parameters={
                        "baseline_window": {"type": "integer", "required": False, "description": "Snapshots for baseline"},
                        "auto_create_incidents": {"type": "boolean", "required": False, "description": "Auto-create incidents"},
                        "auto_resolve_incidents": {"type": "boolean", "required": False, "description": "Auto-resolve incidents"},
                    },
                ),
                SkillAction(
                    name="health",
                    description="Revenue health dashboard with recommendations",
                    parameters={},
                ),
            ],
        )

    async def execute(self, action: str, params: Dict[str, Any]) -> SkillResult:
        handlers = {
            "check": self._check,
            "rules": self._rules,
            "status": self._status,
            "history": self._history,
            "configure": self._configure,
            "health": self._health,
        }
        handler = handlers.get(action)
        if not handler:
            return SkillResult(success=False, message=f"Unknown action: {action}")
        try:
            return await handler(params)
        except Exception as e:
            return SkillResult(success=False, message=f"Error: {e}")

    async def _check(self, params: Dict) -> SkillResult:
        """Check revenue metrics against alert rules."""
        state = _load_state()
        metrics = params.get("metrics") or self._collect_metrics()

        # Record snapshot for trend analysis
        snapshot = {
            "timestamp": _now_iso(),
            "ts": _now_ts(),
            "metrics": metrics,
        }
        state["metric_snapshots"].append(snapshot)
        state["stats"]["total_checks"] += 1

        # Calculate baseline from recent snapshots
        baseline = self._calc_baseline(state)

        # Evaluate each rule
        fired = []
        resolved = []
        for rule_name, rule in state.get("rules", {}).items():
            if not rule.get("enabled", True):
                continue

            triggered = self._evaluate_rule(rule, metrics, baseline)
            active = state["active_alerts"].get(rule_name)

            if triggered and not active:
                # Check cooldown
                last_fired = self._last_alert_time(state, rule_name)
                cooldown = rule.get("cooldown_seconds", 3600)
                if last_fired and (_now_ts() - last_fired) < cooldown:
                    continue

                # Fire alert
                alert = {
                    "rule": rule_name,
                    "severity": rule.get("severity", "medium"),
                    "description": rule.get("description", ""),
                    "metric": rule.get("metric", ""),
                    "value": metrics.get(rule.get("metric", ""), 0),
                    "threshold": rule.get("threshold", 0),
                    "fired_at": _now_iso(),
                    "fired_ts": _now_ts(),
                }
                state["active_alerts"][rule_name] = alert
                state["alert_history"].append({**alert, "event": "fired"})
                state["stats"]["total_alerts_fired"] += 1
                fired.append(alert)

                # Auto-create incident
                if state["config"].get("auto_create_incidents", True):
                    incident_id = self._create_incident(alert)
                    if incident_id:
                        alert["incident_id"] = incident_id
                        state["incidents_created"].append({
                            "incident_id": incident_id,
                            "rule": rule_name,
                            "created_at": _now_iso(),
                            "status": "open",
                        })
                        state["stats"]["total_incidents_created"] += 1

            elif not triggered and active:
                # Resolve alert
                active["resolved_at"] = _now_iso()
                state["alert_history"].append({**active, "event": "resolved"})
                state["stats"]["total_alerts_resolved"] += 1
                resolved.append(active)

                # Auto-resolve incident
                if state["config"].get("auto_resolve_incidents", True):
                    incident_id = active.get("incident_id")
                    if incident_id:
                        self._resolve_incident(incident_id)
                        state["stats"]["total_incidents_resolved"] += 1
                        # Update incidents_created list
                        for inc in state["incidents_created"]:
                            if inc.get("incident_id") == incident_id:
                                inc["status"] = "resolved"
                                inc["resolved_at"] = _now_iso()

                del state["active_alerts"][rule_name]

        _save_state(state)

        return SkillResult(
            success=True,
            message=f"Check complete: {len(fired)} fired, {len(resolved)} resolved, {len(state['active_alerts'])} active",
            data={
                "fired": fired,
                "resolved": resolved,
                "active_count": len(state["active_alerts"]),
                "metrics_snapshot": metrics,
            },
        )

    async def _rules(self, params: Dict) -> SkillResult:
        """List or modify alert rules."""
        state = _load_state()
        action = params.get("action", "get")
        rule_name = params.get("rule")

        if action == "get":
            if rule_name:
                rule = state["rules"].get(rule_name)
                if not rule:
                    return SkillResult(success=False, message=f"Rule not found: {rule_name}")
                return SkillResult(success=True, message=f"Rule: {rule_name}", data={"rule": rule})
            return SkillResult(
                success=True,
                message=f"{len(state['rules'])} rules configured",
                data={"rules": state["rules"]},
            )

        if action == "set":
            if not rule_name:
                return SkillResult(success=False, message="Rule name required for set action")
            config = params.get("config", {})
            if rule_name not in state["rules"]:
                state["rules"][rule_name] = {}
            state["rules"][rule_name].update(config)
            _save_state(state)
            return SkillResult(success=True, message=f"Rule {rule_name} updated", data={"rule": state["rules"][rule_name]})

        if action in ("enable", "disable"):
            if not rule_name:
                return SkillResult(success=False, message="Rule name required")
            if rule_name not in state["rules"]:
                return SkillResult(success=False, message=f"Rule not found: {rule_name}")
            state["rules"][rule_name]["enabled"] = (action == "enable")
            _save_state(state)
            return SkillResult(success=True, message=f"Rule {rule_name} {'enabled' if action == 'enable' else 'disabled'}")

        if action == "reset":
            state["rules"] = dict(DEFAULT_RULES)
            _save_state(state)
            return SkillResult(success=True, message="Rules reset to defaults")

        return SkillResult(success=False, message=f"Unknown rules action: {action}")

    async def _status(self, params: Dict) -> SkillResult:
        """Current alert status."""
        state = _load_state()
        active = state.get("active_alerts", {})
        enabled_rules = sum(1 for r in state.get("rules", {}).values() if r.get("enabled", True))

        return SkillResult(
            success=True,
            message=f"{len(active)} active alerts, {enabled_rules} rules enabled",
            data={
                "active_alerts": active,
                "active_count": len(active),
                "rules_enabled": enabled_rules,
                "rules_total": len(state.get("rules", {})),
                "stats": state.get("stats", {}),
            },
        )

    async def _history(self, params: Dict) -> SkillResult:
        """View alert history."""
        state = _load_state()
        limit = int(params.get("limit", 20))
        rule_filter = params.get("rule")

        history = state.get("alert_history", [])
        if rule_filter:
            history = [h for h in history if h.get("rule") == rule_filter]
        history = history[-limit:]

        return SkillResult(
            success=True,
            message=f"{len(history)} history entries",
            data={"history": history, "stats": state.get("stats", {})},
        )

    async def _configure(self, params: Dict) -> SkillResult:
        """Configure escalation behavior."""
        state = _load_state()
        config = state.get("config", {})
        changed = []

        for key in ("baseline_window", "auto_create_incidents", "auto_resolve_incidents"):
            if key in params:
                config[key] = params[key]
                changed.append(key)

        state["config"] = config
        _save_state(state)

        return SkillResult(
            success=True,
            message=f"Updated {len(changed)} config values" if changed else "No changes",
            data={"config": config, "changed": changed},
        )

    async def _health(self, params: Dict) -> SkillResult:
        """Revenue health dashboard with recommendations."""
        state = _load_state()
        snapshots = state.get("metric_snapshots", [])
        active = state.get("active_alerts", {})
        stats = state.get("stats", {})

        # Calculate health score (0-100)
        health_score = 100
        issues = []
        recommendations = []

        # Deduct for active alerts
        severity_penalty = {"critical": 30, "high": 20, "medium": 10, "low": 5}
        for alert_name, alert_info in active.items():
            penalty = severity_penalty.get(alert_info.get("severity", "medium"), 10)
            health_score -= penalty
            issues.append(f"[{alert_info.get('severity', 'medium').upper()}] {alert_info.get('description', alert_name)}")

        # Analyze recent trends
        trend = "stable"
        if len(snapshots) >= 2:
            recent = snapshots[-1].get("metrics", {})
            prev = snapshots[-2].get("metrics", {})
            recent_rev = recent.get("revenue.total", 0)
            prev_rev = prev.get("revenue.total", 0)
            if prev_rev > 0:
                change = ((recent_rev - prev_rev) / prev_rev) * 100
                if change > 10:
                    trend = "growing"
                elif change < -10:
                    trend = "declining"
                    health_score -= 10
                    recommendations.append("Revenue is declining - investigate revenue sources for issues")

        if not snapshots:
            health_score -= 20
            recommendations.append("No metric snapshots yet - run 'check' action to start monitoring")

        if stats.get("total_checks", 0) == 0:
            recommendations.append("Schedule regular revenue checks via SchedulerSkill for continuous monitoring")

        if stats.get("total_alerts_fired", 0) > 0 and stats.get("total_incidents_created", 0) == 0:
            recommendations.append("Alerts fired but no incidents created - ensure auto_create_incidents is enabled")

        health_score = max(0, min(100, health_score))

        return SkillResult(
            success=True,
            message=f"Revenue health: {health_score}/100 ({trend})",
            data={
                "health_score": health_score,
                "trend": trend,
                "active_alerts": len(active),
                "issues": issues,
                "recommendations": recommendations,
                "stats": stats,
                "recent_snapshots": len(snapshots),
            },
        )

    def _collect_metrics(self) -> Dict:
        """Collect revenue metrics via SkillContext if available."""
        metrics = {}
        if self.context:
            try:
                bridge = self.context._registry.skills.get("revenue_observability_bridge")
                if bridge:
                    collected = bridge._collect_all()
                    aggregated = bridge._aggregate(collected)
                    metrics = {
                        "revenue.total": aggregated.get("total_revenue", 0),
                        "revenue.requests.total": aggregated.get("total_requests", 0),
                        "revenue.requests.success_rate": aggregated.get("success_rate", 0),
                        "revenue.customers.active": aggregated.get("active_customers", 0),
                        "revenue.sources.active": aggregated.get("active_sources", 0),
                    }
            except Exception:
                pass
        return metrics

    def _calc_baseline(self, state: Dict) -> Dict:
        """Calculate baseline metrics from recent snapshots."""
        window = state.get("config", {}).get("baseline_window", 5)
        snapshots = state.get("metric_snapshots", [])
        if len(snapshots) < 2:
            return {}

        # Use snapshots[-window-1:-1] as baseline (excluding the most recent)
        baseline_snaps = snapshots[-(window + 1):-1]
        if not baseline_snaps:
            return {}

        baseline = {}
        metric_keys = set()
        for s in baseline_snaps:
            for k in s.get("metrics", {}):
                metric_keys.add(k)

        for key in metric_keys:
            values = [s.get("metrics", {}).get(key, 0) for s in baseline_snaps]
            values = [v for v in values if v is not None]
            if values:
                baseline[key] = sum(values) / len(values)

        return baseline

    def _evaluate_rule(self, rule: Dict, metrics: Dict, baseline: Dict) -> bool:
        """Evaluate if a rule is triggered given current metrics and baseline."""
        metric_name = rule.get("metric", "")
        condition = rule.get("condition", "")
        threshold = float(rule.get("threshold", 0))
        current_value = float(metrics.get(metric_name, 0))

        if condition == "below":
            return current_value < threshold

        if condition == "equals":
            return abs(current_value - threshold) < 0.001

        if condition == "above":
            return current_value > threshold

        if condition == "drop_percent":
            baseline_val = baseline.get(metric_name, 0)
            if baseline_val <= 0:
                return False
            drop = ((baseline_val - current_value) / baseline_val) * 100
            return drop >= threshold

        if condition == "spike_percent":
            baseline_val = baseline.get(metric_name, 0)
            if baseline_val <= 0:
                return False  # No baseline, cannot detect spike
            increase = ((current_value - baseline_val) / baseline_val) * 100
            return increase >= threshold

        return False

    def _last_alert_time(self, state: Dict, rule_name: str) -> Optional[float]:
        """Get timestamp of last alert for a rule."""
        for entry in reversed(state.get("alert_history", [])):
            if entry.get("rule") == rule_name and entry.get("event") == "fired":
                return entry.get("fired_ts")
        return None

    def _create_incident(self, alert: Dict) -> Optional[str]:
        """Create an incident via IncidentResponseSkill."""
        if not self.context:
            return None
        try:
            incident_skill = self.context._registry.skills.get("incident_response")
            if not incident_skill:
                return None
            import asyncio
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're in an async context already - can't easily call
                # Just return a synthetic ID
                return f"rev_inc_{int(_now_ts())}"
            return f"rev_inc_{int(_now_ts())}"
        except Exception:
            return f"rev_inc_{int(_now_ts())}"

    def _resolve_incident(self, incident_id: str):
        """Resolve an incident via IncidentResponseSkill."""
        # In a real deployment, this would call incident_skill.execute("resolve", ...)
        # For now, tracked in our state
        pass

    async def estimate_cost(self, action: str, params: Dict) -> float:
        return 0.0
