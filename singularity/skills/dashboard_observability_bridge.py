#!/usr/bin/env python3
"""
DashboardObservabilityBridgeSkill - Connect ObservabilitySkill metrics into DashboardSkill.

DashboardSkill reads raw data files for static snapshots. ObservabilitySkill has rich
time-series metrics with aggregation, querying, and alerting. But they don't talk to
each other. This bridge connects them so the dashboard can:

1. Pull live time-series metrics into dashboard sections (latency, throughput, error rates)
2. Show ObservabilitySkill alert status alongside dashboard health data
3. Generate metric sparklines for trend visualization
4. Auto-emit dashboard snapshot metrics INTO ObservabilitySkill for historical tracking
5. Create pre-built metric widgets for common dashboard views

The bidirectional flow:
  ObservabilitySkill metrics → Dashboard (richer sections, trends, sparklines)
  Dashboard snapshots → ObservabilitySkill (track dashboard scores over time)

Pillars:
- Goal Setting: Dashboard now shows real metric trends, not just point-in-time values
- Self-Improvement: Historical metric trends reveal degradation patterns early
- Revenue: Revenue metrics with proper time-series charting
- Replication: Fleet metrics with per-agent breakdown

Actions:
- sync: Pull ObservabilitySkill metrics into dashboard format
- push_snapshot: Push current dashboard scores into ObservabilitySkill as metrics
- widget: Generate a metric widget (sparkline, gauge, counter) for a specific metric
- alerts_summary: Aggregate ObservabilitySkill alert status for dashboard display
- configure: Set which metrics to auto-pull for each dashboard section
- trend: Get trend data for a metric over time with sparkline generation
- auto_sync: Run full bidirectional sync (pull metrics + push scores)
- status: Show bridge configuration and sync health
"""

import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any

from .base import Skill, SkillResult, SkillManifest, SkillAction

# Data files
DATA_DIR = Path(__file__).parent.parent / "data"
BRIDGE_CONFIG_FILE = DATA_DIR / "dashboard_obs_bridge.json"
METRICS_FILE = DATA_DIR / "observability_metrics.json"
ALERTS_FILE = DATA_DIR / "observability_alerts.json"
DASHBOARD_HISTORY_FILE = DATA_DIR / "dashboard_history.json"

# Default metric mappings for each dashboard section
DEFAULT_METRIC_MAPPINGS = {
    "performance": {
        "metrics": [
            {"name": "skill.latency", "aggregation": "avg", "label": "Avg Latency (ms)", "multiplier": 1000},
            {"name": "skill.latency", "aggregation": "p95", "label": "P95 Latency (ms)", "multiplier": 1000},
            {"name": "skill.success", "aggregation": "avg", "label": "Success Rate (%)", "multiplier": 100},
            {"name": "skill.executions", "aggregation": "sum", "label": "Total Executions"},
            {"name": "skill.errors", "aggregation": "sum", "label": "Total Errors"},
        ],
        "window_hours": 1,
    },
    "revenue": {
        "metrics": [
            {"name": "revenue.earned", "aggregation": "sum", "label": "Revenue ($)"},
            {"name": "revenue.cost", "aggregation": "sum", "label": "Cost ($)"},
            {"name": "revenue.requests", "aggregation": "sum", "label": "Paid Requests"},
            {"name": "revenue.profit", "aggregation": "sum", "label": "Profit ($)"},
        ],
        "window_hours": 24,
    },
    "fleet": {
        "metrics": [
            {"name": "fleet.health_score", "aggregation": "last", "label": "Fleet Health"},
            {"name": "fleet.agent_count", "aggregation": "last", "label": "Agent Count"},
            {"name": "fleet.incidents", "aggregation": "sum", "label": "Incidents"},
        ],
        "window_hours": 1,
    },
    "resources": {
        "metrics": [
            {"name": "resource.budget_used", "aggregation": "last", "label": "Budget Used ($)"},
            {"name": "resource.burn_rate", "aggregation": "avg", "label": "Burn Rate ($/hr)"},
        ],
        "window_hours": 24,
    },
}

# Sparkline characters for text-based trend visualization
SPARK_CHARS = "▁▂▃▄▅▆▇█"

MAX_SYNC_LOG = 200


def _now_ts() -> float:
    return time.time()


def _now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _load_json(path: Path) -> Optional[Dict]:
    try:
        if path.exists():
            return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        pass
    return None


def _save_json(path: Path, data: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str))


def _sparkline(values: List[float]) -> str:
    """Generate a text sparkline from a list of values."""
    if not values:
        return ""
    min_val = min(values)
    max_val = max(values)
    val_range = max_val - min_val
    if val_range == 0:
        return SPARK_CHARS[4] * len(values)
    chars = []
    for v in values:
        idx = int((v - min_val) / val_range * (len(SPARK_CHARS) - 1))
        idx = max(0, min(len(SPARK_CHARS) - 1, idx))
        chars.append(SPARK_CHARS[idx])
    return "".join(chars)


def _relative_time_to_ts(relative: str) -> float:
    """Parse relative time like '-1h', '-30m', '-7d' to timestamp."""
    now = _now_ts()
    if not relative or not relative.startswith("-"):
        return now
    try:
        num = int(relative[1:-1])
        unit = relative[-1]
        if unit == "m":
            return now - num * 60
        elif unit == "h":
            return now - num * 3600
        elif unit == "d":
            return now - num * 86400
    except (ValueError, IndexError):
        pass
    return now


class DashboardObservabilityBridgeSkill(Skill):
    """
    Bridge between DashboardSkill and ObservabilitySkill for rich metric-driven dashboards.

    Enables bidirectional data flow: ObservabilitySkill time-series metrics feed into
    dashboard sections, and dashboard scores are emitted as metrics for historical tracking.
    """

    def __init__(self, credentials: Dict[str, str] = None):
        super().__init__(credentials)
        self._ensure_data()

    def _ensure_data(self):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        if not BRIDGE_CONFIG_FILE.exists():
            _save_json(BRIDGE_CONFIG_FILE, {
                "mappings": DEFAULT_METRIC_MAPPINGS,
                "auto_sync_enabled": True,
                "sync_log": [],
                "last_sync": None,
                "push_pillar_scores": True,
                "sparkline_points": 20,
            })

    def _load_config(self) -> Dict:
        return _load_json(BRIDGE_CONFIG_FILE) or {
            "mappings": DEFAULT_METRIC_MAPPINGS,
            "auto_sync_enabled": True,
            "sync_log": [],
            "last_sync": None,
            "push_pillar_scores": True,
            "sparkline_points": 20,
        }

    def _save_config(self, config: Dict):
        _save_json(BRIDGE_CONFIG_FILE, config)

    def _load_metrics(self) -> Dict:
        return _load_json(METRICS_FILE) or {"series": {}, "metadata": {}}

    def _save_metrics(self, data: Dict):
        _save_json(METRICS_FILE, data)

    def _load_alerts(self) -> Dict:
        return _load_json(ALERTS_FILE) or {"rules": {}, "history": []}

    def _log_sync(self, config: Dict, action: str, details: str):
        log = config.get("sync_log", [])
        log.append({
            "timestamp": _now_iso(),
            "action": action,
            "details": details,
        })
        if len(log) > MAX_SYNC_LOG:
            log = log[-MAX_SYNC_LOG:]
        config["sync_log"] = log

    # ── Manifest ──────────────────────────────────────────────────────

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="dashboard_observability_bridge",
            name="Dashboard-Observability Bridge",
            version="1.0.0",
            category="infrastructure",
            description="Connect ObservabilitySkill time-series metrics into DashboardSkill for rich trend-driven displays",
            actions=self.get_actions(),
            required_credentials=[],
            install_cost=0,
            author="singularity",
        )

    def get_actions(self) -> List[SkillAction]:
        return [
            SkillAction(
                name="sync",
                description="Pull ObservabilitySkill metrics into dashboard-ready format for all configured sections",
                parameters={
                    "section": {"type": "string", "required": False,
                                "description": "Specific section to sync (performance|revenue|fleet|resources). Default: all"},
                },
                estimated_cost=0,
                estimated_duration_seconds=1,
            ),
            SkillAction(
                name="push_snapshot",
                description="Push current dashboard pillar scores into ObservabilitySkill as time-series metrics",
                parameters={
                    "scores": {"type": "object", "required": True,
                               "description": "Pillar scores dict: {pillar_name: {score: int, grade: str}}"},
                },
                estimated_cost=0,
                estimated_duration_seconds=1,
            ),
            SkillAction(
                name="widget",
                description="Generate a metric widget (sparkline, gauge, counter) for a specific metric",
                parameters={
                    "metric_name": {"type": "string", "required": True, "description": "Metric name to visualize"},
                    "widget_type": {"type": "string", "required": False,
                                    "description": "Widget type: sparkline|gauge|counter|trend. Default: sparkline"},
                    "window": {"type": "string", "required": False,
                               "description": "Time window: '-1h', '-6h', '-24h', '-7d'. Default: -1h"},
                    "labels": {"type": "object", "required": False, "description": "Label filters"},
                    "aggregation": {"type": "string", "required": False, "description": "Aggregation: avg, sum, max, etc."},
                },
                estimated_cost=0,
                estimated_duration_seconds=1,
            ),
            SkillAction(
                name="alerts_summary",
                description="Aggregate ObservabilitySkill alert status for dashboard display",
                parameters={},
                estimated_cost=0,
                estimated_duration_seconds=1,
            ),
            SkillAction(
                name="configure",
                description="Set which metrics to auto-pull for each dashboard section",
                parameters={
                    "section": {"type": "string", "required": True,
                                "description": "Dashboard section: performance|revenue|fleet|resources"},
                    "metrics": {"type": "array", "required": False,
                                "description": "List of metric configs: [{name, aggregation, label, multiplier?}]"},
                    "window_hours": {"type": "number", "required": False,
                                     "description": "Default query window in hours"},
                },
                estimated_cost=0,
                estimated_duration_seconds=1,
            ),
            SkillAction(
                name="trend",
                description="Get trend data for a metric over time with sparkline and direction indicator",
                parameters={
                    "metric_name": {"type": "string", "required": True, "description": "Metric name"},
                    "window": {"type": "string", "required": False,
                               "description": "Time window: '-1h', '-6h', '-24h', '-7d'. Default: -1h"},
                    "labels": {"type": "object", "required": False, "description": "Label filters"},
                    "points": {"type": "number", "required": False,
                               "description": "Number of points for sparkline. Default: 20"},
                },
                estimated_cost=0,
                estimated_duration_seconds=1,
            ),
            SkillAction(
                name="auto_sync",
                description="Run full bidirectional sync: pull metrics into dashboard AND push scores into ObservabilitySkill",
                parameters={
                    "scores": {"type": "object", "required": False,
                               "description": "Optional pillar scores to push. If omitted, only pull is performed."},
                },
                estimated_cost=0,
                estimated_duration_seconds=2,
            ),
            SkillAction(
                name="status",
                description="Show bridge configuration, sync health, and metric coverage",
                parameters={},
                estimated_cost=0,
                estimated_duration_seconds=1,
            ),
        ]

    # ── Main execute ──────────────────────────────────────────────────

    async def execute(self, action: str, params: Dict[str, Any] = None) -> SkillResult:
        params = params or {}
        actions = {
            "sync": self._sync,
            "push_snapshot": self._push_snapshot,
            "widget": self._widget,
            "alerts_summary": self._alerts_summary,
            "configure": self._configure,
            "trend": self._trend,
            "auto_sync": self._auto_sync,
            "status": self._status,
        }
        if action not in actions:
            return SkillResult(success=False,
                               message=f"Unknown action: {action}. Available: {list(actions.keys())}")
        try:
            return await actions[action](params)
        except Exception as e:
            return SkillResult(success=False, message=f"Bridge error: {str(e)}")

    # ── Core metric querying from ObservabilitySkill data ─────────────

    def _query_metric(self, metrics_data: Dict, name: str, labels: Optional[Dict] = None,
                      start_ts: Optional[float] = None, end_ts: Optional[float] = None) -> List[Dict]:
        """Query metric data points from ObservabilitySkill's stored series."""
        series = metrics_data.get("series", {})
        results = []

        for series_key, series_info in series.items():
            # Match metric name
            metric_name = series_info.get("name", "")
            if metric_name != name:
                continue

            # Match labels if specified
            if labels:
                series_labels = series_info.get("labels", {})
                if not all(series_labels.get(k) == v for k, v in labels.items()):
                    continue

            # Filter points by time range
            points = series_info.get("points", [])
            for p in points:
                ts = p.get("ts", 0)
                if start_ts and ts < start_ts:
                    continue
                if end_ts and ts > end_ts:
                    continue
                results.append({"ts": ts, "value": p.get("value", 0), "labels": series_info.get("labels", {})})

        # Sort by timestamp
        results.sort(key=lambda x: x["ts"])
        return results

    def _aggregate_values(self, values: List[float], agg: str) -> float:
        """Apply aggregation to a list of values."""
        if not values:
            return 0.0
        if agg == "sum":
            return sum(values)
        elif agg == "avg":
            return sum(values) / len(values)
        elif agg == "min":
            return min(values)
        elif agg == "max":
            return max(values)
        elif agg == "count":
            return float(len(values))
        elif agg == "last":
            return values[-1]
        elif agg == "p95":
            sorted_vals = sorted(values)
            idx = int(0.95 * (len(sorted_vals) - 1))
            return sorted_vals[idx]
        elif agg == "p99":
            sorted_vals = sorted(values)
            idx = int(0.99 * (len(sorted_vals) - 1))
            return sorted_vals[idx]
        elif agg == "rate":
            return sum(values) / max(len(values), 1)
        return sum(values) / len(values)

    def _compute_trend_direction(self, values: List[float]) -> str:
        """Determine trend direction from a sequence of values."""
        if len(values) < 2:
            return "stable"
        midpoint = len(values) // 2
        first_half_avg = sum(values[:midpoint]) / midpoint if midpoint > 0 else 0
        second_half_avg = sum(values[midpoint:]) / (len(values) - midpoint) if (len(values) - midpoint) > 0 else 0

        if first_half_avg == 0:
            if second_half_avg > 0:
                return "rising"
            return "stable"

        pct_change = (second_half_avg - first_half_avg) / abs(first_half_avg)
        if pct_change > 0.05:
            return "rising"
        elif pct_change < -0.05:
            return "falling"
        return "stable"

    def _bucket_values(self, points: List[Dict], num_buckets: int) -> List[float]:
        """Split time-series points into N evenly-spaced buckets and average each."""
        if not points or num_buckets <= 0:
            return []

        if len(points) <= num_buckets:
            return [p["value"] for p in points]

        bucket_size = len(points) / num_buckets
        buckets = []
        for i in range(num_buckets):
            start_idx = int(i * bucket_size)
            end_idx = int((i + 1) * bucket_size)
            bucket_points = points[start_idx:end_idx]
            if bucket_points:
                avg = sum(p["value"] for p in bucket_points) / len(bucket_points)
                buckets.append(avg)
            else:
                buckets.append(0.0)
        return buckets

    # ── Actions ───────────────────────────────────────────────────────

    async def _sync(self, params: Dict) -> SkillResult:
        """Pull ObservabilitySkill metrics into dashboard-ready format."""
        config = self._load_config()
        metrics_data = self._load_metrics()
        mappings = config.get("mappings", DEFAULT_METRIC_MAPPINGS)
        target_section = params.get("section")

        now = _now_ts()
        result_sections = {}
        total_metrics_pulled = 0

        for section_name, section_config in mappings.items():
            if target_section and section_name != target_section:
                continue

            window_hours = section_config.get("window_hours", 1)
            start_ts = now - window_hours * 3600
            section_metrics = []

            for metric_config in section_config.get("metrics", []):
                name = metric_config["name"]
                agg = metric_config.get("aggregation", "avg")
                label = metric_config.get("label", name)
                multiplier = metric_config.get("multiplier", 1)

                # Query the metric
                points = self._query_metric(metrics_data, name, start_ts=start_ts, end_ts=now)
                values = [p["value"] for p in points]

                # Compute aggregated value
                agg_value = self._aggregate_values(values, agg) * multiplier

                # Compute sparkline and trend
                sparkline_points = config.get("sparkline_points", 20)
                bucket_values = self._bucket_values(points, sparkline_points)
                spark = _sparkline(bucket_values)
                trend = self._compute_trend_direction(bucket_values)

                section_metrics.append({
                    "metric_name": name,
                    "label": label,
                    "value": round(agg_value, 4),
                    "aggregation": agg,
                    "data_points": len(points),
                    "sparkline": spark,
                    "trend": trend,
                    "window_hours": window_hours,
                })
                total_metrics_pulled += 1

            result_sections[section_name] = {
                "metrics": section_metrics,
                "window_hours": window_hours,
                "synced_at": _now_iso(),
            }

        # Log sync
        self._log_sync(config, "sync", f"Pulled {total_metrics_pulled} metrics across {len(result_sections)} sections")
        config["last_sync"] = _now_iso()
        self._save_config(config)

        return SkillResult(
            success=True,
            message=f"Synced {total_metrics_pulled} metrics across {len(result_sections)} sections from ObservabilitySkill",
            data={"sections": result_sections, "total_metrics": total_metrics_pulled},
        )

    async def _push_snapshot(self, params: Dict) -> SkillResult:
        """Push pillar scores into ObservabilitySkill as time-series metrics."""
        scores = params.get("scores", {})
        if not scores:
            return SkillResult(success=False, message="No scores provided. Pass pillar scores dict.")

        metrics_data = self._load_metrics()
        series = metrics_data.get("series", {})
        now = _now_ts()
        pushed = 0

        for pillar_name, pillar_info in scores.items():
            score = pillar_info.get("score", 0) if isinstance(pillar_info, dict) else pillar_info
            metric_name = f"dashboard.pillar.{pillar_name}"
            labels = {"pillar": pillar_name, "source": "dashboard"}
            series_key = f"{metric_name}{{pillar={pillar_name},source=dashboard}}"

            if series_key not in series:
                series[series_key] = {
                    "name": metric_name,
                    "type": "gauge",
                    "labels": labels,
                    "points": [],
                    "created": _now_iso(),
                }

            series[series_key]["points"].append({
                "ts": now,
                "value": float(score),
            })

            # Trim to last 10000 points
            if len(series[series_key]["points"]) > 10000:
                series[series_key]["points"] = series[series_key]["points"][-10000:]

            pushed += 1

        metrics_data["series"] = series
        self._save_metrics(metrics_data)

        config = self._load_config()
        self._log_sync(config, "push_snapshot", f"Pushed {pushed} pillar scores as metrics")
        self._save_config(config)

        return SkillResult(
            success=True,
            message=f"Pushed {pushed} pillar scores into ObservabilitySkill as time-series metrics",
            data={"pushed_count": pushed, "pillars": list(scores.keys())},
        )

    async def _widget(self, params: Dict) -> SkillResult:
        """Generate a visual widget for a metric."""
        metric_name = params.get("metric_name", "")
        if not metric_name:
            return SkillResult(success=False, message="metric_name is required")

        widget_type = params.get("widget_type", "sparkline")
        window = params.get("window", "-1h")
        labels = params.get("labels")
        agg = params.get("aggregation", "avg")

        metrics_data = self._load_metrics()
        start_ts = _relative_time_to_ts(window)
        now = _now_ts()

        points = self._query_metric(metrics_data, metric_name, labels=labels,
                                    start_ts=start_ts, end_ts=now)
        values = [p["value"] for p in points]

        if widget_type == "sparkline":
            config = self._load_config()
            num_points = config.get("sparkline_points", 20)
            bucket_values = self._bucket_values(points, num_points)
            spark = _sparkline(bucket_values)
            trend = self._compute_trend_direction(bucket_values)
            current = round(self._aggregate_values(values, "last"), 4) if values else 0

            widget_data = {
                "type": "sparkline",
                "metric": metric_name,
                "sparkline": spark,
                "trend": trend,
                "current_value": current,
                "data_points": len(points),
                "window": window,
                "display": f"{metric_name}: {current} {spark} ({trend})",
            }

        elif widget_type == "gauge":
            current = round(self._aggregate_values(values, agg), 4) if values else 0
            # Gauge assumes 0-100 range
            bar_filled = min(20, max(0, int(current / 5)))
            bar = "#" * bar_filled + "-" * (20 - bar_filled)

            widget_data = {
                "type": "gauge",
                "metric": metric_name,
                "value": current,
                "bar": f"[{bar}]",
                "data_points": len(points),
                "window": window,
                "display": f"{metric_name}: [{bar}] {current}",
            }

        elif widget_type == "counter":
            total = round(self._aggregate_values(values, "sum"), 4) if values else 0
            avg_val = round(self._aggregate_values(values, "avg"), 4) if values else 0

            widget_data = {
                "type": "counter",
                "metric": metric_name,
                "total": total,
                "average": avg_val,
                "count": len(points),
                "window": window,
                "display": f"{metric_name}: total={total} avg={avg_val} count={len(points)}",
            }

        elif widget_type == "trend":
            bucket_values = self._bucket_values(points, 10)
            trend = self._compute_trend_direction(bucket_values)
            current = round(self._aggregate_values(values, "last"), 4) if values else 0
            change_pct = 0
            if len(bucket_values) >= 2 and bucket_values[0] != 0:
                change_pct = round((bucket_values[-1] - bucket_values[0]) / abs(bucket_values[0]) * 100, 1)

            arrow = "→" if trend == "stable" else ("↑" if trend == "rising" else "↓")

            widget_data = {
                "type": "trend",
                "metric": metric_name,
                "current": current,
                "trend": trend,
                "change_pct": change_pct,
                "arrow": arrow,
                "data_points": len(points),
                "window": window,
                "display": f"{metric_name}: {current} {arrow} {change_pct:+.1f}%",
            }
        else:
            return SkillResult(success=False,
                               message=f"Unknown widget_type: {widget_type}. Use: sparkline|gauge|counter|trend")

        return SkillResult(
            success=True,
            message=widget_data["display"],
            data=widget_data,
        )

    async def _alerts_summary(self, params: Dict) -> SkillResult:
        """Aggregate ObservabilitySkill alert status for dashboard display."""
        alerts_data = self._load_alerts()
        rules = alerts_data.get("rules", {})
        history = alerts_data.get("history", [])

        total_rules = len(rules)
        firing = []
        ok = []
        cooldown = []

        for rule_name, rule in rules.items():
            state = rule.get("state", "ok")
            severity = rule.get("severity", "warning")
            info = {
                "name": rule_name,
                "metric": rule.get("metric_name", ""),
                "condition": rule.get("condition", ""),
                "threshold": rule.get("threshold", 0),
                "severity": severity,
                "state": state,
            }
            if state == "firing":
                firing.append(info)
            elif state == "cooldown":
                cooldown.append(info)
            else:
                ok.append(info)

        # Recent alert history (last 20)
        recent_history = history[-20:] if history else []

        # Health assessment
        if firing:
            critical_count = sum(1 for a in firing if a["severity"] == "critical")
            if critical_count > 0:
                health = "critical"
                health_score = max(0, 100 - critical_count * 30 - (len(firing) - critical_count) * 10)
            else:
                health = "degraded"
                health_score = max(0, 100 - len(firing) * 15)
        elif cooldown:
            health = "recovering"
            health_score = max(50, 100 - len(cooldown) * 5)
        else:
            health = "healthy"
            health_score = 100

        summary = {
            "total_rules": total_rules,
            "firing_count": len(firing),
            "ok_count": len(ok),
            "cooldown_count": len(cooldown),
            "firing_alerts": firing,
            "health": health,
            "health_score": health_score,
            "recent_history": recent_history,
        }

        status_msg = f"Alert health: {health} ({health_score}/100) - {len(firing)} firing, {len(ok)} ok, {len(cooldown)} cooldown"
        return SkillResult(success=True, message=status_msg, data=summary)

    async def _configure(self, params: Dict) -> SkillResult:
        """Configure metric mappings for a dashboard section."""
        section = params.get("section", "")
        if not section:
            return SkillResult(success=False, message="section is required")

        config = self._load_config()
        mappings = config.get("mappings", {})

        if section not in mappings:
            mappings[section] = {"metrics": [], "window_hours": 1}

        new_metrics = params.get("metrics")
        if new_metrics is not None:
            mappings[section]["metrics"] = new_metrics

        new_window = params.get("window_hours")
        if new_window is not None:
            mappings[section]["window_hours"] = new_window

        config["mappings"] = mappings
        self._log_sync(config, "configure", f"Updated config for section: {section}")
        self._save_config(config)

        return SkillResult(
            success=True,
            message=f"Updated metric mappings for section '{section}': {len(mappings[section]['metrics'])} metrics, {mappings[section]['window_hours']}h window",
            data={"section": section, "config": mappings[section]},
        )

    async def _trend(self, params: Dict) -> SkillResult:
        """Get trend data for a metric with sparkline visualization."""
        metric_name = params.get("metric_name", "")
        if not metric_name:
            return SkillResult(success=False, message="metric_name is required")

        window = params.get("window", "-1h")
        labels = params.get("labels")
        config = self._load_config()
        num_points = params.get("points", config.get("sparkline_points", 20))

        metrics_data = self._load_metrics()
        start_ts = _relative_time_to_ts(window)
        now = _now_ts()

        points = self._query_metric(metrics_data, metric_name, labels=labels,
                                    start_ts=start_ts, end_ts=now)
        values = [p["value"] for p in points]

        bucket_values = self._bucket_values(points, num_points)
        spark = _sparkline(bucket_values)
        trend = self._compute_trend_direction(bucket_values)

        # Compute stats
        current = round(values[-1], 4) if values else 0
        avg_val = round(sum(values) / len(values), 4) if values else 0
        min_val = round(min(values), 4) if values else 0
        max_val = round(max(values), 4) if values else 0

        change_pct = 0
        if len(bucket_values) >= 2 and bucket_values[0] != 0:
            change_pct = round((bucket_values[-1] - bucket_values[0]) / abs(bucket_values[0]) * 100, 1)

        arrow = "→" if trend == "stable" else ("↑" if trend == "rising" else "↓")

        trend_data = {
            "metric": metric_name,
            "sparkline": spark,
            "trend": trend,
            "arrow": arrow,
            "change_pct": change_pct,
            "current": current,
            "average": avg_val,
            "min": min_val,
            "max": max_val,
            "data_points": len(points),
            "buckets": bucket_values,
            "window": window,
            "display": f"{metric_name} {spark} {current} {arrow} {change_pct:+.1f}% ({trend})",
        }

        return SkillResult(
            success=True,
            message=trend_data["display"],
            data=trend_data,
        )

    async def _auto_sync(self, params: Dict) -> SkillResult:
        """Full bidirectional sync: pull metrics + push scores."""
        config = self._load_config()

        # Step 1: Pull all metrics
        sync_result = await self._sync({})

        # Step 2: Push scores if provided
        scores = params.get("scores")
        push_result = None
        if scores:
            push_result = await self._push_snapshot({"scores": scores})

        self._log_sync(config, "auto_sync", f"Pull: {sync_result.data.get('total_metrics', 0)} metrics. Push: {'yes' if push_result else 'no'}")
        config["last_sync"] = _now_iso()
        self._save_config(config)

        result_data = {
            "pull": sync_result.data,
            "push": push_result.data if push_result else None,
            "synced_at": _now_iso(),
        }

        msg_parts = [f"Pull: {sync_result.data.get('total_metrics', 0)} metrics"]
        if push_result:
            msg_parts.append(f"Push: {push_result.data.get('pushed_count', 0)} pillar scores")

        return SkillResult(
            success=True,
            message=f"Auto-sync complete. {' | '.join(msg_parts)}",
            data=result_data,
        )

    async def _status(self, params: Dict) -> SkillResult:
        """Show bridge configuration and sync health."""
        config = self._load_config()
        mappings = config.get("mappings", {})
        metrics_data = self._load_metrics()
        series = metrics_data.get("series", {})

        # Count available metrics
        available_metric_names = set()
        for s_info in series.values():
            available_metric_names.add(s_info.get("name", ""))

        # Check mapping coverage
        section_status = {}
        total_configured = 0
        total_available = 0

        for section_name, section_config in mappings.items():
            configured_metrics = [m["name"] for m in section_config.get("metrics", [])]
            total_configured += len(configured_metrics)
            available_in_section = [m for m in configured_metrics if m in available_metric_names]
            total_available += len(available_in_section)

            section_status[section_name] = {
                "configured_metrics": len(configured_metrics),
                "available_metrics": len(available_in_section),
                "missing_metrics": [m for m in configured_metrics if m not in available_metric_names],
                "window_hours": section_config.get("window_hours", 1),
                "coverage_pct": round(len(available_in_section) / len(configured_metrics) * 100, 1) if configured_metrics else 100,
            }

        sync_log = config.get("sync_log", [])
        recent_syncs = sync_log[-5:] if sync_log else []

        status_data = {
            "auto_sync_enabled": config.get("auto_sync_enabled", True),
            "last_sync": config.get("last_sync"),
            "push_pillar_scores": config.get("push_pillar_scores", True),
            "sparkline_points": config.get("sparkline_points", 20),
            "sections": section_status,
            "total_configured_metrics": total_configured,
            "total_available_metrics": total_available,
            "overall_coverage_pct": round(total_available / total_configured * 100, 1) if total_configured > 0 else 0,
            "observability_series_count": len(series),
            "observability_metric_names": sorted(available_metric_names),
            "recent_syncs": recent_syncs,
        }

        coverage = status_data["overall_coverage_pct"]
        return SkillResult(
            success=True,
            message=f"Bridge status: {total_configured} metrics configured, {total_available} available ({coverage}% coverage). Last sync: {config.get('last_sync', 'never')}",
            data=status_data,
        )

    def estimate_cost(self, action: str, params: Dict = None) -> float:
        return 0.0
