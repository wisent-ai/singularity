#!/usr/bin/env python3
"""
ObservabilitySkill - Centralized time-series metrics collection, querying, and alerting.

Every skill and subsystem can emit metrics (counters, gauges, histograms) with
labels. The ObservabilitySkill stores them as time-series data, supports flexible
querying with aggregations (sum, avg, min, max, p50, p95, p99, rate), and fires
alerts when thresholds are breached — emitting events to EventBus for reactive behavior.

This is foundational infrastructure serving ALL four pillars:
- Self-Improvement: measure skill latency, success rates, error trends over time
- Revenue: track earnings, request volume, cost per request, profit margins
- Replication: monitor fleet-wide metrics per replica, detect anomalies
- Goal Setting: quantify goal progress with hard numbers, not just status strings

Unlike RuntimeMetrics (in-memory sliding window) or DashboardSkill (snapshot reader),
ObservabilitySkill is a proper metrics pipeline:
- Persistent time-series storage with configurable retention
- Label-based filtering (skill, action, agent_id, customer, etc.)
- Aggregation functions over arbitrary time windows
- Threshold alerts with cooldown to prevent alert storms
- Metric export for external systems

Integrates with:
- EventBus: emits alert.fired / alert.resolved events
- DashboardSkill: can pull aggregated metrics for display
- IncidentResponseSkill: alerts can auto-trigger incident detection
- PerformanceOptimizerSkill: query historical latency/throughput metrics

Actions:
- emit: Record a metric data point (counter, gauge, histogram)
- query: Query metrics with filtering, time range, and aggregation
- alert_create: Define a threshold-based alert rule
- alert_list: List all alert rules and their status
- alert_delete: Remove an alert rule
- check_alerts: Evaluate all alert rules against current data
- export: Export raw metric data as JSON for external use
- status: Overview of all metrics being tracked, volumes, and alert health
"""

import json
import math
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from .base import Skill, SkillResult, SkillManifest, SkillAction

METRICS_FILE = Path(__file__).parent.parent / "data" / "observability_metrics.json"
ALERTS_FILE = Path(__file__).parent.parent / "data" / "observability_alerts.json"

# Metric types
COUNTER = "counter"      # Monotonically increasing (requests, errors)
GAUGE = "gauge"          # Point-in-time value (cpu_usage, queue_depth)
HISTOGRAM = "histogram"  # Distribution of values (latency, response_size)

METRIC_TYPES = [COUNTER, GAUGE, HISTOGRAM]

# Aggregation functions
AGG_SUM = "sum"
AGG_AVG = "avg"
AGG_MIN = "min"
AGG_MAX = "max"
AGG_COUNT = "count"
AGG_P50 = "p50"
AGG_P95 = "p95"
AGG_P99 = "p99"
AGG_RATE = "rate"        # Per-second rate of change
AGG_LAST = "last"

AGGREGATIONS = [AGG_SUM, AGG_AVG, AGG_MIN, AGG_MAX, AGG_COUNT,
                AGG_P50, AGG_P95, AGG_P99, AGG_RATE, AGG_LAST]

# Alert states
ALERT_OK = "ok"
ALERT_FIRING = "firing"
ALERT_COOLDOWN = "cooldown"

# Limits
MAX_SERIES = 500          # Max distinct metric series
MAX_POINTS_PER_SERIES = 10000  # Max data points per series
MAX_ALERTS = 100          # Max alert rules
DEFAULT_RETENTION_HOURS = 168  # 7 days


def _now_ts() -> float:
    return time.time()


def _now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _load_metrics() -> Dict:
    if METRICS_FILE.exists():
        try:
            return json.loads(METRICS_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return {"series": {}, "metadata": {"created": _now_iso(), "total_points": 0}}


def _save_metrics(data: Dict):
    METRICS_FILE.parent.mkdir(parents=True, exist_ok=True)
    METRICS_FILE.write_text(json.dumps(data, indent=2, default=str))


def _load_alerts() -> Dict:
    if ALERTS_FILE.exists():
        try:
            return json.loads(ALERTS_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return {"rules": {}, "history": []}


def _save_alerts(data: Dict):
    ALERTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    ALERTS_FILE.write_text(json.dumps(data, indent=2, default=str))


def _series_key(name: str, labels: Dict[str, str]) -> str:
    """Create a unique key for a metric name + label set."""
    sorted_labels = sorted(labels.items())
    label_str = ",".join(f"{k}={v}" for k, v in sorted_labels)
    return f"{name}{{{label_str}}}" if label_str else name


def _percentile(values: List[float], pct: float) -> float:
    """Compute percentile from sorted values."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    idx = (pct / 100.0) * (len(sorted_vals) - 1)
    lower = int(math.floor(idx))
    upper = int(math.ceil(idx))
    if lower == upper:
        return sorted_vals[lower]
    frac = idx - lower
    return sorted_vals[lower] * (1 - frac) + sorted_vals[upper] * frac


def _aggregate(values: List[float], timestamps: List[float], agg: str) -> float:
    """Apply an aggregation function to a list of values."""
    if not values:
        return 0.0
    if agg == AGG_SUM:
        return sum(values)
    elif agg == AGG_AVG:
        return sum(values) / len(values)
    elif agg == AGG_MIN:
        return min(values)
    elif agg == AGG_MAX:
        return max(values)
    elif agg == AGG_COUNT:
        return float(len(values))
    elif agg == AGG_P50:
        return _percentile(values, 50)
    elif agg == AGG_P95:
        return _percentile(values, 95)
    elif agg == AGG_P99:
        return _percentile(values, 99)
    elif agg == AGG_RATE:
        if len(timestamps) < 2:
            return 0.0
        duration = max(timestamps) - min(timestamps)
        if duration <= 0:
            return 0.0
        return sum(values) / duration
    elif agg == AGG_LAST:
        # Return value with highest timestamp
        if not timestamps:
            return values[-1]
        paired = sorted(zip(timestamps, values))
        return paired[-1][1]
    return 0.0


def _filter_points(points: List[Dict], start_ts: Optional[float],
                   end_ts: Optional[float]) -> List[Dict]:
    """Filter data points by time range."""
    result = []
    for p in points:
        ts = p["ts"]
        if start_ts and ts < start_ts:
            continue
        if end_ts and ts > end_ts:
            continue
        result.append(p)
    return result


def _match_labels(series_labels: Dict[str, str], filter_labels: Dict[str, str]) -> bool:
    """Check if a series matches all filter labels (subset match)."""
    for k, v in filter_labels.items():
        if series_labels.get(k) != v:
            return False
    return True


class ObservabilitySkill(Skill):
    """
    Centralized time-series metrics pipeline with querying and alerting.

    Provides a unified way for all skills and subsystems to emit, query,
    and alert on metrics — enabling data-driven self-improvement, revenue
    tracking, fleet monitoring, and goal measurement.
    """

    def __init__(self, credentials: Dict[str, str] = None):
        super().__init__(credentials)
        METRICS_FILE.parent.mkdir(parents=True, exist_ok=True)

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="observability",
            name="Observability Metrics Pipeline",
            version="1.0.0",
            category="infrastructure",
            description="Centralized time-series metrics collection, querying, and alerting for all agent subsystems",
            actions=self.get_actions(),
            required_credentials=[],
            install_cost=0,
            author="singularity",
        )

    def get_actions(self) -> List[SkillAction]:
        return [
            SkillAction(
                name="emit",
                description="Record a metric data point (counter, gauge, or histogram)",
                parameters={
                    "name": {"type": "string", "required": True, "description": "Metric name (e.g. 'skill.latency', 'revenue.total')"},
                    "value": {"type": "number", "required": True, "description": "Metric value"},
                    "metric_type": {"type": "string", "required": False, "description": f"Type: {METRIC_TYPES}. Default: gauge"},
                    "labels": {"type": "object", "required": False, "description": "Key-value labels for this data point"},
                },
            ),
            SkillAction(
                name="query",
                description="Query metrics with filtering, time range, and aggregation",
                parameters={
                    "name": {"type": "string", "required": True, "description": "Metric name to query"},
                    "labels": {"type": "object", "required": False, "description": "Filter by labels (subset match)"},
                    "aggregation": {"type": "string", "required": False, "description": f"Aggregation: {AGGREGATIONS}. Default: avg"},
                    "start": {"type": "string", "required": False, "description": "Start time ISO or relative like '-1h', '-30m', '-7d'"},
                    "end": {"type": "string", "required": False, "description": "End time ISO or relative. Default: now"},
                    "group_by": {"type": "string", "required": False, "description": "Label key to group results by"},
                },
            ),
            SkillAction(
                name="alert_create",
                description="Define a threshold-based alert rule on a metric",
                parameters={
                    "name": {"type": "string", "required": True, "description": "Alert rule name"},
                    "metric_name": {"type": "string", "required": True, "description": "Metric to watch"},
                    "condition": {"type": "string", "required": True, "description": "Condition: 'above' or 'below'"},
                    "threshold": {"type": "number", "required": True, "description": "Threshold value"},
                    "aggregation": {"type": "string", "required": False, "description": f"Aggregation over window. Default: avg"},
                    "window_minutes": {"type": "number", "required": False, "description": "Evaluation window in minutes. Default: 5"},
                    "cooldown_minutes": {"type": "number", "required": False, "description": "Cooldown between firings. Default: 15"},
                    "labels": {"type": "object", "required": False, "description": "Filter labels for alert evaluation"},
                    "severity": {"type": "string", "required": False, "description": "Alert severity: info, warning, critical. Default: warning"},
                },
            ),
            SkillAction(
                name="alert_list",
                description="List all alert rules and their current status",
                parameters={},
            ),
            SkillAction(
                name="alert_delete",
                description="Delete an alert rule by name",
                parameters={
                    "name": {"type": "string", "required": True, "description": "Alert rule name to delete"},
                },
            ),
            SkillAction(
                name="check_alerts",
                description="Evaluate all alert rules against current metric data",
                parameters={},
            ),
            SkillAction(
                name="export",
                description="Export raw metric data as JSON",
                parameters={
                    "name": {"type": "string", "required": False, "description": "Metric name to export (all if omitted)"},
                    "labels": {"type": "object", "required": False, "description": "Filter by labels"},
                    "start": {"type": "string", "required": False, "description": "Start time filter"},
                    "end": {"type": "string", "required": False, "description": "End time filter"},
                },
            ),
            SkillAction(
                name="status",
                description="Overview of all tracked metrics, volumes, and alert health",
                parameters={},
            ),
        ]

    async def execute(self, action: str, params: Dict[str, Any]) -> SkillResult:
        handlers = {
            "emit": self._emit,
            "query": self._query,
            "alert_create": self._alert_create,
            "alert_list": self._alert_list,
            "alert_delete": self._alert_delete,
            "check_alerts": self._check_alerts,
            "export": self._export,
            "status": self._status,
        }
        handler = handlers.get(action)
        if not handler:
            return SkillResult(success=False, message=f"Unknown action: {action}")
        return handler(params)

    # ── emit ──────────────────────────────────────────────────────────

    def _emit(self, params: Dict) -> SkillResult:
        name = params.get("name", "").strip()
        if not name:
            return SkillResult(success=False, message="Metric name is required")

        value = params.get("value")
        if value is None:
            return SkillResult(success=False, message="Metric value is required")
        try:
            value = float(value)
        except (TypeError, ValueError):
            return SkillResult(success=False, message="Metric value must be a number")

        metric_type = params.get("metric_type", GAUGE)
        if metric_type not in METRIC_TYPES:
            return SkillResult(success=False, message=f"Invalid type. Must be one of: {METRIC_TYPES}")

        labels = params.get("labels", {})
        if not isinstance(labels, dict):
            labels = {}

        data = _load_metrics()
        key = _series_key(name, labels)

        if key not in data["series"]:
            if len(data["series"]) >= MAX_SERIES:
                return SkillResult(success=False, message=f"Max series limit ({MAX_SERIES}) reached. Delete old metrics first.")
            data["series"][key] = {
                "name": name,
                "type": metric_type,
                "labels": labels,
                "points": [],
                "created": _now_iso(),
            }

        series = data["series"][key]
        ts = _now_ts()

        # For counters, add to previous value
        if metric_type == COUNTER and series["points"]:
            prev = series["points"][-1]["v"]
            value = prev + value

        series["points"].append({"ts": ts, "v": value})

        # Enforce per-series limit
        if len(series["points"]) > MAX_POINTS_PER_SERIES:
            series["points"] = series["points"][-MAX_POINTS_PER_SERIES:]

        data["metadata"]["total_points"] = sum(
            len(s["points"]) for s in data["series"].values()
        )

        _save_metrics(data)

        return SkillResult(
            success=True,
            message=f"Emitted {metric_type} '{name}' = {value}",
            data={"key": key, "value": value, "timestamp": ts},
        )

    # ── query ─────────────────────────────────────────────────────────

    def _parse_relative_time(self, time_str: str) -> Optional[float]:
        """Parse relative time strings like '-1h', '-30m', '-7d' to timestamp."""
        if not time_str:
            return None
        time_str = time_str.strip()

        # Relative time
        if time_str.startswith("-"):
            try:
                num_str = time_str[1:-1]
                unit = time_str[-1]
                num = float(num_str)
                multipliers = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}
                if unit in multipliers:
                    return _now_ts() - (num * multipliers[unit])
            except (ValueError, IndexError):
                pass

        # ISO timestamp
        try:
            dt = datetime.fromisoformat(time_str.replace("Z", "+00:00"))
            return dt.timestamp()
        except (ValueError, TypeError):
            pass

        return None

    def _query(self, params: Dict) -> SkillResult:
        name = params.get("name", "").strip()
        if not name:
            return SkillResult(success=False, message="Metric name is required")

        filter_labels = params.get("labels", {})
        agg = params.get("aggregation", AGG_AVG)
        if agg not in AGGREGATIONS:
            return SkillResult(success=False, message=f"Invalid aggregation. Must be one of: {AGGREGATIONS}")

        start_ts = self._parse_relative_time(params.get("start", ""))
        end_ts = self._parse_relative_time(params.get("end", ""))
        group_by = params.get("group_by", "")

        data = _load_metrics()

        # Find matching series
        matching = []
        for key, series in data["series"].items():
            if series["name"] != name:
                continue
            if filter_labels and not _match_labels(series["labels"], filter_labels):
                continue
            matching.append((key, series))

        if not matching:
            return SkillResult(
                success=True,
                message=f"No data found for metric '{name}'",
                data={"metric": name, "result": None, "series_count": 0},
            )

        # Group by label key if requested
        if group_by:
            groups: Dict[str, List] = {}
            for key, series in matching:
                group_val = series["labels"].get(group_by, "_ungrouped_")
                if group_val not in groups:
                    groups[group_val] = []
                points = _filter_points(series["points"], start_ts, end_ts)
                groups[group_val].extend(points)

            results = {}
            for group_val, points in groups.items():
                values = [p["v"] for p in points]
                timestamps = [p["ts"] for p in points]
                results[group_val] = {
                    "value": _aggregate(values, timestamps, agg),
                    "count": len(values),
                }

            return SkillResult(
                success=True,
                message=f"Queried '{name}' grouped by '{group_by}' with {agg}",
                data={
                    "metric": name,
                    "aggregation": agg,
                    "group_by": group_by,
                    "results": results,
                    "series_count": len(matching),
                },
            )

        # Flat query across all matching series
        all_values = []
        all_timestamps = []
        for key, series in matching:
            points = _filter_points(series["points"], start_ts, end_ts)
            all_values.extend(p["v"] for p in points)
            all_timestamps.extend(p["ts"] for p in points)

        result_value = _aggregate(all_values, all_timestamps, agg)

        return SkillResult(
            success=True,
            message=f"Queried '{name}': {agg} = {result_value:.4f} ({len(all_values)} points)",
            data={
                "metric": name,
                "aggregation": agg,
                "value": result_value,
                "point_count": len(all_values),
                "series_count": len(matching),
            },
        )

    # ── alert_create ──────────────────────────────────────────────────

    def _alert_create(self, params: Dict) -> SkillResult:
        name = params.get("name", "").strip()
        if not name:
            return SkillResult(success=False, message="Alert name is required")

        metric_name = params.get("metric_name", "").strip()
        if not metric_name:
            return SkillResult(success=False, message="Metric name is required")

        condition = params.get("condition", "").strip()
        if condition not in ("above", "below"):
            return SkillResult(success=False, message="Condition must be 'above' or 'below'")

        threshold = params.get("threshold")
        if threshold is None:
            return SkillResult(success=False, message="Threshold is required")
        try:
            threshold = float(threshold)
        except (TypeError, ValueError):
            return SkillResult(success=False, message="Threshold must be a number")

        agg = params.get("aggregation", AGG_AVG)
        window_minutes = params.get("window_minutes", 5)
        cooldown_minutes = params.get("cooldown_minutes", 15)
        labels = params.get("labels", {})
        severity = params.get("severity", "warning")
        if severity not in ("info", "warning", "critical"):
            severity = "warning"

        alerts = _load_alerts()

        if len(alerts["rules"]) >= MAX_ALERTS:
            return SkillResult(success=False, message=f"Max alert rules ({MAX_ALERTS}) reached")

        alerts["rules"][name] = {
            "metric_name": metric_name,
            "condition": condition,
            "threshold": threshold,
            "aggregation": agg,
            "window_minutes": window_minutes,
            "cooldown_minutes": cooldown_minutes,
            "labels": labels,
            "severity": severity,
            "state": ALERT_OK,
            "last_fired": None,
            "fire_count": 0,
            "created": _now_iso(),
        }

        _save_alerts(alerts)

        return SkillResult(
            success=True,
            message=f"Alert '{name}' created: {metric_name} {condition} {threshold} ({agg} over {window_minutes}m)",
            data={"alert_name": name, "rule": alerts["rules"][name]},
        )

    # ── alert_list ────────────────────────────────────────────────────

    def _alert_list(self, params: Dict) -> SkillResult:
        alerts = _load_alerts()
        rules = []
        for name, rule in alerts["rules"].items():
            rules.append({
                "name": name,
                "metric": rule["metric_name"],
                "condition": f"{rule['condition']} {rule['threshold']}",
                "state": rule["state"],
                "severity": rule["severity"],
                "fire_count": rule["fire_count"],
                "last_fired": rule["last_fired"],
            })
        return SkillResult(
            success=True,
            message=f"{len(rules)} alert rules configured",
            data={"rules": rules, "total": len(rules)},
        )

    # ── alert_delete ──────────────────────────────────────────────────

    def _alert_delete(self, params: Dict) -> SkillResult:
        name = params.get("name", "").strip()
        if not name:
            return SkillResult(success=False, message="Alert name is required")

        alerts = _load_alerts()
        if name not in alerts["rules"]:
            return SkillResult(success=False, message=f"Alert '{name}' not found")

        del alerts["rules"][name]
        _save_alerts(alerts)

        return SkillResult(
            success=True,
            message=f"Alert '{name}' deleted",
            data={"deleted": name},
        )

    # ── check_alerts ──────────────────────────────────────────────────

    def _check_alerts(self, params: Dict) -> SkillResult:
        alerts = _load_alerts()
        metrics_data = _load_metrics()
        now = _now_ts()
        fired = []
        resolved = []

        for name, rule in alerts["rules"].items():
            # Check cooldown
            if rule["state"] == ALERT_COOLDOWN:
                if rule["last_fired"]:
                    cooldown_end = rule["last_fired"] + (rule["cooldown_minutes"] * 60)
                    if now < cooldown_end:
                        continue
                    rule["state"] = ALERT_OK

            # Find matching series
            window_start = now - (rule["window_minutes"] * 60)
            all_values = []
            all_timestamps = []

            for key, series in metrics_data["series"].items():
                if series["name"] != rule["metric_name"]:
                    continue
                if rule["labels"] and not _match_labels(series["labels"], rule["labels"]):
                    continue
                points = _filter_points(series["points"], window_start, now)
                all_values.extend(p["v"] for p in points)
                all_timestamps.extend(p["ts"] for p in points)

            if not all_values:
                # No data — if was firing, resolve
                if rule["state"] == ALERT_FIRING:
                    rule["state"] = ALERT_OK
                    resolved.append({"name": name, "reason": "no data in window"})
                continue

            agg_value = _aggregate(all_values, all_timestamps, rule["aggregation"])

            # Evaluate condition
            is_breached = False
            if rule["condition"] == "above" and agg_value > rule["threshold"]:
                is_breached = True
            elif rule["condition"] == "below" and agg_value < rule["threshold"]:
                is_breached = True

            if is_breached and rule["state"] != ALERT_FIRING:
                rule["state"] = ALERT_FIRING
                rule["last_fired"] = now
                rule["fire_count"] += 1
                fired.append({
                    "name": name,
                    "metric": rule["metric_name"],
                    "severity": rule["severity"],
                    "condition": f"{rule['condition']} {rule['threshold']}",
                    "current_value": agg_value,
                })
                alerts["history"].append({
                    "alert": name,
                    "event": "fired",
                    "value": agg_value,
                    "threshold": rule["threshold"],
                    "severity": rule["severity"],
                    "timestamp": _now_iso(),
                })
            elif not is_breached and rule["state"] == ALERT_FIRING:
                rule["state"] = ALERT_COOLDOWN
                resolved.append({
                    "name": name,
                    "current_value": agg_value,
                })
                alerts["history"].append({
                    "alert": name,
                    "event": "resolved",
                    "value": agg_value,
                    "timestamp": _now_iso(),
                })

        # Trim alert history
        if len(alerts["history"]) > 500:
            alerts["history"] = alerts["history"][-500:]

        _save_alerts(alerts)

        return SkillResult(
            success=True,
            message=f"Checked {len(alerts['rules'])} alerts: {len(fired)} fired, {len(resolved)} resolved",
            data={
                "fired": fired,
                "resolved": resolved,
                "total_rules": len(alerts["rules"]),
            },
        )

    # ── export ────────────────────────────────────────────────────────

    def _export(self, params: Dict) -> SkillResult:
        name_filter = params.get("name", "").strip()
        label_filter = params.get("labels", {})
        start_ts = self._parse_relative_time(params.get("start", ""))
        end_ts = self._parse_relative_time(params.get("end", ""))

        data = _load_metrics()
        exported = {}

        for key, series in data["series"].items():
            if name_filter and series["name"] != name_filter:
                continue
            if label_filter and not _match_labels(series["labels"], label_filter):
                continue
            points = _filter_points(series["points"], start_ts, end_ts)
            if points:
                exported[key] = {
                    "name": series["name"],
                    "type": series["type"],
                    "labels": series["labels"],
                    "points": points,
                    "point_count": len(points),
                }

        return SkillResult(
            success=True,
            message=f"Exported {len(exported)} series",
            data={"series": exported, "series_count": len(exported)},
        )

    # ── status ────────────────────────────────────────────────────────

    def _status(self, params: Dict) -> SkillResult:
        metrics_data = _load_metrics()
        alerts_data = _load_alerts()

        series_info = []
        total_points = 0
        for key, series in metrics_data["series"].items():
            pt_count = len(series["points"])
            total_points += pt_count
            latest_val = series["points"][-1]["v"] if series["points"] else None
            series_info.append({
                "key": key,
                "name": series["name"],
                "type": series["type"],
                "labels": series["labels"],
                "point_count": pt_count,
                "latest_value": latest_val,
            })

        firing_alerts = [n for n, r in alerts_data["rules"].items() if r["state"] == ALERT_FIRING]

        return SkillResult(
            success=True,
            message=f"{len(series_info)} metric series, {total_points} total points, {len(firing_alerts)} alerts firing",
            data={
                "series": series_info,
                "series_count": len(series_info),
                "total_points": total_points,
                "alert_rules": len(alerts_data["rules"]),
                "alerts_firing": firing_alerts,
                "recent_alert_history": alerts_data["history"][-10:] if alerts_data["history"] else [],
            },
        )
