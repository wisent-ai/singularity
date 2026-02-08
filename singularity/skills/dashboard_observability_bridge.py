#!/usr/bin/env python3
"""
DashboardObservabilityBridgeSkill - Auto-pull ObservabilitySkill metrics into DashboardSkill.

DashboardSkill reads flat JSON data files for its reports, and ObservabilitySkill
stores rich time-series metrics with labels/aggregations. They operate independently.
This bridge connects them so the dashboard automatically includes:

1. **Metric summaries** - Top metrics with latest values, trends, and sparklines
2. **Alert status** - Active/resolved alerts from ObservabilitySkill shown in dashboard
3. **Pillar metrics** - Observability metrics aggregated per-pillar for scoring
4. **Trend detection** - Compare metric windows to detect improving/degrading trends
5. **Auto-refresh** - Periodically pull fresh metrics into dashboard data

Event topics emitted:
  - dashboard.metrics_refreshed  - Fresh metrics pulled into dashboard
  - dashboard.alert_synced       - Alert status synced from observability
  - dashboard.trend_detected     - Significant metric trend detected
  - dashboard.health_updated     - Overall health score recalculated from metrics

Pillar: Goal Setting (agent sees quantitative metrics for better prioritization)
"""

import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional

from .base import Skill, SkillResult, SkillManifest, SkillAction

BRIDGE_STATE_FILE = Path(__file__).parent.parent / "data" / "dashboard_observability_bridge.json"
METRICS_FILE = Path(__file__).parent.parent / "data" / "observability_metrics.json"
ALERTS_FILE = Path(__file__).parent.parent / "data" / "observability_alerts.json"
DASHBOARD_METRICS_FILE = Path(__file__).parent.parent / "data" / "dashboard_metrics.json"

MAX_EVENT_LOG = 200

# Metric-to-pillar mapping: which metric name prefixes belong to which pillar
PILLAR_METRIC_PREFIXES = {
    "self_improvement": [
        "skill.success", "skill.failure", "skill.latency",
        "prompt.score", "experiment.", "self_modify.", "self_eval.",
        "performance.", "error_recovery.",
    ],
    "revenue": [
        "revenue.", "payment.", "usage.", "service.request",
        "api.call", "customer.", "earnings.", "cost.",
    ],
    "replication": [
        "replica.", "fleet.", "spawn.", "agent_network.",
        "peer.", "delegation.",
    ],
    "goal_setting": [
        "goal.", "strategy.", "planner.", "milestone.",
        "priority.", "decision.",
    ],
}

# Trend thresholds
TREND_IMPROVING_THRESHOLD = 0.10  # 10% improvement
TREND_DEGRADING_THRESHOLD = -0.10  # 10% degradation


class DashboardObservabilityBridgeSkill(Skill):
    """Bridge ObservabilitySkill metrics into DashboardSkill for unified monitoring."""

    def __init__(self, credentials: Dict[str, str] = None):
        super().__init__(credentials)
        self._state = self._load_state()

    # ── Persistence ──────────────────────────────────────────────────

    def _default_state(self) -> Dict:
        return {
            "wired": False,
            "last_refresh": None,
            "refresh_interval_seconds": 300,  # 5 minutes
            "metric_summaries": [],
            "alert_snapshot": [],
            "pillar_metric_scores": {},
            "trends": [],
            "event_log": [],
            "config": {
                "max_metrics_in_summary": 20,
                "trend_window_seconds": 3600,  # Compare last hour
                "trend_comparison_window": 3600,  # vs previous hour
                "auto_refresh": True,
            },
        }

    def _load_state(self) -> Dict:
        if BRIDGE_STATE_FILE.exists():
            try:
                with open(BRIDGE_STATE_FILE) as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
        return self._default_state()

    def _save_state(self):
        BRIDGE_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(BRIDGE_STATE_FILE, "w") as f:
            json.dump(self._state, f, indent=2)

    def _log_event(self, event_type: str, details: Dict):
        entry = {
            "type": event_type,
            "timestamp": datetime.utcnow().isoformat(),
            "details": details,
        }
        self._state["event_log"].append(entry)
        if len(self._state["event_log"]) > MAX_EVENT_LOG:
            self._state["event_log"] = self._state["event_log"][-MAX_EVENT_LOG:]

    def _load_metrics_data(self) -> Dict:
        """Load raw metrics from ObservabilitySkill's data file."""
        if METRICS_FILE.exists():
            try:
                with open(METRICS_FILE) as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
        return {"series": {}}

    def _load_alerts_data(self) -> Dict:
        """Load alerts from ObservabilitySkill's data file."""
        if ALERTS_FILE.exists():
            try:
                with open(ALERTS_FILE) as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
        return {"rules": {}}

    def _save_dashboard_metrics(self, data: Dict):
        """Write enriched metrics for DashboardSkill to consume."""
        DASHBOARD_METRICS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(DASHBOARD_METRICS_FILE, "w") as f:
            json.dump(data, f, indent=2)

    # ── Manifest ─────────────────────────────────────────────────────

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="dashboard_observability_bridge",
            name="Dashboard-Observability Bridge",
            description=(
                "Bridges ObservabilitySkill time-series metrics into DashboardSkill "
                "for unified monitoring. Auto-pulls metric summaries, alert status, "
                "pillar-specific scores, and trend detection into the dashboard."
            ),
            version="1.0.0",
            category="infrastructure",
            required_credentials=[],
            actions=self.get_actions(),
        )

    def get_actions(self) -> List[SkillAction]:
        return [
            SkillAction(
                name="wire",
                description="Activate the bridge - start pulling metrics into dashboard",
                parameters=[],
            ),
            SkillAction(
                name="unwire",
                description="Deactivate the bridge",
                parameters=[],
            ),
            SkillAction(
                name="refresh",
                description="Manually refresh metrics into dashboard now",
                parameters=[],
            ),
            SkillAction(
                name="metric_summary",
                description="Get top metrics with latest values and trends",
                parameters=[
                    {"name": "pillar", "type": "string", "description": "Filter by pillar (optional)"},
                    {"name": "limit", "type": "integer", "description": "Max metrics to show (default 20)"},
                ],
            ),
            SkillAction(
                name="alert_status",
                description="Show current alert status from ObservabilitySkill",
                parameters=[],
            ),
            SkillAction(
                name="pillar_scores",
                description="Calculate pillar health scores from observability metrics",
                parameters=[],
            ),
            SkillAction(
                name="trends",
                description="Detect significant metric trends (improving/degrading)",
                parameters=[
                    {"name": "window_seconds", "type": "integer", "description": "Time window for trend analysis (default 3600)"},
                ],
            ),
            SkillAction(
                name="configure",
                description="Update bridge configuration",
                parameters=[
                    {"name": "refresh_interval_seconds", "type": "integer", "description": "Auto-refresh interval"},
                    {"name": "max_metrics_in_summary", "type": "integer", "description": "Max metrics in summary"},
                    {"name": "trend_window_seconds", "type": "integer", "description": "Window for trend detection"},
                    {"name": "auto_refresh", "type": "boolean", "description": "Enable/disable auto-refresh"},
                ],
            ),
            SkillAction(
                name="history",
                description="View bridge event log",
                parameters=[
                    {"name": "limit", "type": "integer", "description": "Max events to show (default 20)"},
                    {"name": "event_type", "type": "string", "description": "Filter by event type"},
                ],
            ),
            SkillAction(
                name="status",
                description="Show bridge status and configuration",
                parameters=[],
            ),
        ]

    # ── Execute dispatch ─────────────────────────────────────────────

    def execute(self, action: str, params: Dict[str, Any] = None) -> SkillResult:
        params = params or {}
        dispatch = {
            "wire": self._wire,
            "unwire": self._unwire,
            "refresh": self._refresh,
            "metric_summary": self._metric_summary,
            "alert_status": self._alert_status,
            "pillar_scores": self._pillar_scores,
            "trends": self._trends,
            "configure": self._configure,
            "history": self._history,
            "status": self._status,
        }
        handler = dispatch.get(action)
        if not handler:
            return SkillResult(
                success=False,
                message=f"Unknown action: {action}. Available: {list(dispatch.keys())}",
            )
        return handler(params)

    # ── Actions ──────────────────────────────────────────────────────

    def _wire(self, params: Dict) -> SkillResult:
        """Activate the bridge."""
        self._state["wired"] = True
        # Immediately do a first refresh
        refresh_result = self._refresh(params)
        self._log_event("bridge.wired", {"auto_refresh": self._state["config"]["auto_refresh"]})
        self._save_state()
        return SkillResult(
            success=True,
            message=f"Dashboard-Observability bridge activated. {refresh_result.message}",
            data={
                "wired": True,
                "metrics_loaded": len(self._state["metric_summaries"]),
                "alerts_loaded": len(self._state["alert_snapshot"]),
                "initial_refresh": refresh_result.data,
            },
        )

    def _unwire(self, params: Dict) -> SkillResult:
        """Deactivate the bridge."""
        self._state["wired"] = False
        self._log_event("bridge.unwired", {})
        self._save_state()
        return SkillResult(
            success=True,
            message="Dashboard-Observability bridge deactivated.",
            data={"wired": False},
        )

    def _refresh(self, params: Dict) -> SkillResult:
        """Pull fresh metrics from ObservabilitySkill into dashboard format."""
        metrics_data = self._load_metrics_data()
        alerts_data = self._load_alerts_data()
        now = time.time()

        # Build metric summaries
        summaries = self._build_metric_summaries(metrics_data, now)
        self._state["metric_summaries"] = summaries

        # Sync alert status
        alert_snapshot = self._build_alert_snapshot(alerts_data)
        self._state["alert_snapshot"] = alert_snapshot

        # Calculate pillar scores from metrics
        pillar_scores = self._calculate_pillar_scores(metrics_data, now)
        self._state["pillar_metric_scores"] = pillar_scores

        # Detect trends
        trends = self._detect_trends(metrics_data, now)
        self._state["trends"] = trends

        # Write enriched data for DashboardSkill
        dashboard_data = {
            "last_updated": datetime.utcnow().isoformat(),
            "metric_summaries": summaries[:self._state["config"]["max_metrics_in_summary"]],
            "alerts": alert_snapshot,
            "pillar_scores": pillar_scores,
            "trends": trends,
            "total_series": len(metrics_data.get("series", {})),
            "total_alerts": len(alerts_data.get("rules", {})),
        }
        self._save_dashboard_metrics(dashboard_data)

        self._state["last_refresh"] = datetime.utcnow().isoformat()
        self._log_event("dashboard.metrics_refreshed", {
            "metrics_count": len(summaries),
            "alerts_count": len(alert_snapshot),
            "trends_count": len(trends),
        })
        self._save_state()

        firing_count = sum(1 for a in alert_snapshot if a.get("state") == "firing")
        trend_count = len(trends)

        return SkillResult(
            success=True,
            message=f"Refreshed {len(summaries)} metrics, {len(alert_snapshot)} alerts ({firing_count} firing), {trend_count} trends detected.",
            data=dashboard_data,
        )

    def _metric_summary(self, params: Dict) -> SkillResult:
        """Get top metrics with latest values and trends."""
        pillar_filter = params.get("pillar")
        limit = params.get("limit", 20)

        summaries = self._state.get("metric_summaries", [])

        if pillar_filter:
            prefixes = PILLAR_METRIC_PREFIXES.get(pillar_filter, [])
            if not prefixes:
                return SkillResult(
                    success=False,
                    message=f"Unknown pillar: {pillar_filter}. Choose from: {list(PILLAR_METRIC_PREFIXES.keys())}",
                )
            summaries = [
                s for s in summaries
                if any(s["name"].startswith(p) for p in prefixes)
            ]

        summaries = summaries[:limit]

        return SkillResult(
            success=True,
            message=f"Showing {len(summaries)} metrics" + (f" for pillar '{pillar_filter}'" if pillar_filter else ""),
            data={"metrics": summaries, "total": len(summaries)},
        )

    def _alert_status(self, params: Dict) -> SkillResult:
        """Show current alert status."""
        snapshot = self._state.get("alert_snapshot", [])
        firing = [a for a in snapshot if a.get("state") == "firing"]
        resolved = [a for a in snapshot if a.get("state") != "firing"]

        return SkillResult(
            success=True,
            message=f"{len(firing)} alerts firing, {len(resolved)} resolved/inactive. Total: {len(snapshot)}.",
            data={
                "firing": firing,
                "resolved": resolved,
                "total": len(snapshot),
                "firing_count": len(firing),
            },
        )

    def _pillar_scores(self, params: Dict) -> SkillResult:
        """Calculate pillar health from observability metrics."""
        metrics_data = self._load_metrics_data()
        now = time.time()
        scores = self._calculate_pillar_scores(metrics_data, now)
        self._state["pillar_metric_scores"] = scores
        self._save_state()

        overall = sum(s["score"] for s in scores.values()) // max(len(scores), 1)
        return SkillResult(
            success=True,
            message=f"Pillar metric scores calculated. Overall: {overall}/100.",
            data={"pillar_scores": scores, "overall": overall},
        )

    def _trends(self, params: Dict) -> SkillResult:
        """Detect significant metric trends."""
        window = params.get("window_seconds", self._state["config"]["trend_window_seconds"])
        metrics_data = self._load_metrics_data()
        now = time.time()
        trends = self._detect_trends(metrics_data, now, window)
        self._state["trends"] = trends
        self._save_state()

        improving = [t for t in trends if t["direction"] == "improving"]
        degrading = [t for t in trends if t["direction"] == "degrading"]

        return SkillResult(
            success=True,
            message=f"Detected {len(trends)} significant trends: {len(improving)} improving, {len(degrading)} degrading.",
            data={
                "trends": trends,
                "improving_count": len(improving),
                "degrading_count": len(degrading),
            },
        )

    def _configure(self, params: Dict) -> SkillResult:
        """Update bridge configuration."""
        updated = []
        config = self._state["config"]

        if "refresh_interval_seconds" in params:
            config["refresh_interval_seconds"] = int(params["refresh_interval_seconds"])
            self._state["refresh_interval_seconds"] = config["refresh_interval_seconds"]
            updated.append("refresh_interval_seconds")
        if "max_metrics_in_summary" in params:
            config["max_metrics_in_summary"] = int(params["max_metrics_in_summary"])
            updated.append("max_metrics_in_summary")
        if "trend_window_seconds" in params:
            config["trend_window_seconds"] = int(params["trend_window_seconds"])
            updated.append("trend_window_seconds")
        if "auto_refresh" in params:
            config["auto_refresh"] = bool(params["auto_refresh"])
            updated.append("auto_refresh")

        if not updated:
            return SkillResult(
                success=False,
                message="No configuration parameters provided.",
                data={"available": ["refresh_interval_seconds", "max_metrics_in_summary", "trend_window_seconds", "auto_refresh"]},
            )

        self._log_event("bridge.configured", {"updated": updated})
        self._save_state()
        return SkillResult(
            success=True,
            message=f"Updated configuration: {', '.join(updated)}",
            data={"config": config, "updated": updated},
        )

    def _history(self, params: Dict) -> SkillResult:
        """View bridge event log."""
        limit = params.get("limit", 20)
        event_type = params.get("event_type")
        events = self._state.get("event_log", [])

        if event_type:
            events = [e for e in events if e["type"] == event_type]

        events = events[-limit:]
        return SkillResult(
            success=True,
            message=f"Showing {len(events)} events.",
            data={"events": events, "total": len(events)},
        )

    def _status(self, params: Dict) -> SkillResult:
        """Show bridge status."""
        return SkillResult(
            success=True,
            message=f"Bridge {'active' if self._state['wired'] else 'inactive'}. "
                    f"Last refresh: {self._state.get('last_refresh', 'never')}. "
                    f"{len(self._state.get('metric_summaries', []))} metrics tracked.",
            data={
                "wired": self._state["wired"],
                "last_refresh": self._state.get("last_refresh"),
                "config": self._state["config"],
                "metrics_count": len(self._state.get("metric_summaries", [])),
                "alerts_count": len(self._state.get("alert_snapshot", [])),
                "trends_count": len(self._state.get("trends", [])),
                "event_log_size": len(self._state.get("event_log", [])),
                "pillar_scores": self._state.get("pillar_metric_scores", {}),
            },
        )

    # ── Internal logic ───────────────────────────────────────────────

    def _build_metric_summaries(self, metrics_data: Dict, now: float) -> List[Dict]:
        """Build metric summaries from raw ObservabilitySkill data."""
        series = metrics_data.get("series", {})
        summaries = []

        for key, series_data in series.items():
            points = series_data.get("points", [])
            if not points:
                continue

            name = series_data.get("name", key)
            metric_type = series_data.get("type", "gauge")
            labels = series_data.get("labels", {})

            # Get latest value
            latest_point = points[-1]
            latest_value = latest_point.get("value", 0)
            latest_ts = latest_point.get("timestamp", now)

            # Calculate stats from recent points (last hour)
            hour_ago = now - 3600
            recent = [p for p in points if p.get("timestamp", 0) >= hour_ago]
            recent_values = [p.get("value", 0) for p in recent]

            avg_value = sum(recent_values) / len(recent_values) if recent_values else latest_value
            min_value = min(recent_values) if recent_values else latest_value
            max_value = max(recent_values) if recent_values else latest_value

            # Determine pillar
            pillar = self._classify_pillar(name)

            # Build sparkline (last 10 values normalized)
            spark_values = [p.get("value", 0) for p in points[-10:]]
            if spark_values:
                spark_min = min(spark_values)
                spark_max = max(spark_values)
                spark_range = spark_max - spark_min if spark_max != spark_min else 1
                sparkline = [(v - spark_min) / spark_range for v in spark_values]
            else:
                sparkline = []

            summaries.append({
                "name": name,
                "type": metric_type,
                "labels": labels,
                "latest_value": latest_value,
                "latest_timestamp": latest_ts,
                "avg_1h": round(avg_value, 4),
                "min_1h": round(min_value, 4),
                "max_1h": round(max_value, 4),
                "point_count_1h": len(recent),
                "total_points": len(points),
                "pillar": pillar,
                "sparkline": [round(v, 2) for v in sparkline],
                "age_seconds": round(now - latest_ts, 1),
            })

        # Sort by recency (most recent first)
        summaries.sort(key=lambda s: s.get("age_seconds", float("inf")))
        return summaries

    def _build_alert_snapshot(self, alerts_data: Dict) -> List[Dict]:
        """Build alert snapshot from ObservabilitySkill alerts."""
        rules = alerts_data.get("rules", {})
        snapshot = []

        for rule_id, rule in rules.items():
            snapshot.append({
                "rule_id": rule_id,
                "name": rule.get("name", rule_id),
                "metric": rule.get("metric", ""),
                "condition": rule.get("condition", ""),
                "threshold": rule.get("threshold", 0),
                "state": rule.get("state", "inactive"),
                "last_checked": rule.get("last_checked"),
                "last_fired": rule.get("last_fired"),
                "fire_count": rule.get("fire_count", 0),
                "labels": rule.get("labels", {}),
            })

        # Sort: firing first, then by fire_count
        snapshot.sort(key=lambda a: (0 if a["state"] == "firing" else 1, -a["fire_count"]))
        return snapshot

    def _classify_pillar(self, metric_name: str) -> str:
        """Classify a metric name into a pillar."""
        for pillar, prefixes in PILLAR_METRIC_PREFIXES.items():
            for prefix in prefixes:
                if metric_name.startswith(prefix):
                    return pillar
        return "general"

    def _calculate_pillar_scores(self, metrics_data: Dict, now: float) -> Dict[str, Dict]:
        """Calculate pillar health scores from observability metrics."""
        series = metrics_data.get("series", {})
        hour_ago = now - 3600

        pillar_metrics: Dict[str, List[Dict]] = {
            "self_improvement": [],
            "revenue": [],
            "replication": [],
            "goal_setting": [],
        }

        # Classify metrics into pillars
        for key, series_data in series.items():
            name = series_data.get("name", key)
            pillar = self._classify_pillar(name)
            if pillar in pillar_metrics:
                points = series_data.get("points", [])
                recent = [p for p in points if p.get("timestamp", 0) >= hour_ago]
                pillar_metrics[pillar].append({
                    "name": name,
                    "type": series_data.get("type", "gauge"),
                    "total_points": len(points),
                    "recent_points": len(recent),
                    "recent_values": [p.get("value", 0) for p in recent],
                })

        scores = {}
        for pillar, metrics in pillar_metrics.items():
            score = self._score_pillar_from_metrics(pillar, metrics)
            scores[pillar] = score

        return scores

    def _score_pillar_from_metrics(self, pillar: str, metrics: List[Dict]) -> Dict:
        """Score a pillar 0-100 based on its metrics."""
        if not metrics:
            return {"score": 0, "grade": "F", "metric_count": 0, "detail": "No metrics recorded"}

        # Base score from having metrics at all
        score = 20

        # Bonus for number of metrics being tracked (more coverage = better)
        metric_count = len(metrics)
        score += min(20, metric_count * 4)  # Up to 20 for 5+ metrics

        # Bonus for recent activity (metrics with recent data points)
        active_metrics = sum(1 for m in metrics if m["recent_points"] > 0)
        activity_ratio = active_metrics / metric_count if metric_count > 0 else 0
        score += int(activity_ratio * 20)  # Up to 20 for all metrics active

        # Look for success-rate type metrics and reward high values
        success_metrics = [m for m in metrics if "success" in m["name"].lower()]
        if success_metrics:
            for sm in success_metrics:
                if sm["recent_values"]:
                    avg_success = sum(sm["recent_values"]) / len(sm["recent_values"])
                    score += min(20, int(avg_success * 20))  # Up to 20 for high success rate
                    break
        else:
            # No success metrics, give partial credit for data volume
            total_recent = sum(m["recent_points"] for m in metrics)
            score += min(15, total_recent)

        # Check for error/failure metrics and penalize high rates
        error_metrics = [m for m in metrics if "error" in m["name"].lower() or "failure" in m["name"].lower()]
        for em in error_metrics:
            if em["recent_values"]:
                avg_errors = sum(em["recent_values"]) / len(em["recent_values"])
                score -= min(15, int(avg_errors * 10))

        score = max(0, min(100, score))

        return {
            "score": score,
            "grade": self._grade(score),
            "metric_count": metric_count,
            "active_metrics": active_metrics,
            "detail": f"{metric_count} metrics tracked, {active_metrics} active in last hour",
        }

    def _detect_trends(self, metrics_data: Dict, now: float, window: int = None) -> List[Dict]:
        """Detect significant trends by comparing two time windows."""
        if window is None:
            window = self._state["config"]["trend_window_seconds"]

        series = metrics_data.get("series", {})
        trends = []

        recent_end = now
        recent_start = now - window
        prev_end = recent_start
        prev_start = recent_start - window

        for key, series_data in series.items():
            name = series_data.get("name", key)
            points = series_data.get("points", [])

            # Get values in each window
            recent_values = [
                p.get("value", 0) for p in points
                if recent_start <= p.get("timestamp", 0) <= recent_end
            ]
            prev_values = [
                p.get("value", 0) for p in points
                if prev_start <= p.get("timestamp", 0) <= prev_end
            ]

            if not recent_values or not prev_values:
                continue

            recent_avg = sum(recent_values) / len(recent_values)
            prev_avg = sum(prev_values) / len(prev_values)

            if prev_avg == 0:
                if recent_avg > 0:
                    change_pct = 1.0  # New activity
                else:
                    continue
            else:
                change_pct = (recent_avg - prev_avg) / abs(prev_avg)

            # Only report significant changes
            if abs(change_pct) < abs(TREND_IMPROVING_THRESHOLD):
                continue

            # Determine if improving or degrading based on metric semantics
            is_error_metric = any(
                kw in name.lower() for kw in ["error", "failure", "latency", "cost"]
            )

            if is_error_metric:
                direction = "improving" if change_pct < 0 else "degrading"
            else:
                direction = "improving" if change_pct > 0 else "degrading"

            trends.append({
                "metric": name,
                "direction": direction,
                "change_pct": round(change_pct * 100, 1),
                "recent_avg": round(recent_avg, 4),
                "previous_avg": round(prev_avg, 4),
                "recent_points": len(recent_values),
                "previous_points": len(prev_values),
                "pillar": self._classify_pillar(name),
                "window_seconds": window,
            })

        # Sort by absolute change magnitude (most significant first)
        trends.sort(key=lambda t: abs(t["change_pct"]), reverse=True)
        return trends

    @staticmethod
    def _grade(score: int) -> str:
        if score >= 90:
            return "A"
        elif score >= 80:
            return "B"
        elif score >= 70:
            return "C"
        elif score >= 60:
            return "D"
        else:
            return "F"

    def estimate_cost(self, action: str, params: Dict = None) -> float:
        return 0.0
