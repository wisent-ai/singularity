#!/usr/bin/env python3
"""
LoopIterationDashboardSkill - Unified view of all autonomous loop iteration stats.

Aggregates data from the autonomous loop journal, scheduler, circuit breaker,
fleet health, goal progress, reputation, and revenue subsystems into a single
coherent dashboard per iteration. This gives the agent (and operators) clear
visibility into what happened during each loop cycle.

Without this skill, understanding a loop iteration requires reading multiple
data files and correlating timestamps manually. With it, the agent can:
- View a comprehensive summary of any iteration
- Track trends across iterations (success rates, durations, revenue)
- Identify degradation patterns (increasing failures, slower iterations)
- Compare iteration performance over time windows
- Get health scores per subsystem per iteration
- Export iteration reports for analysis

Pillar: Self-Improvement (primary - observability enables optimization),
        Goal Setting (iteration-level visibility for better planning)

Actions:
- latest: Get the most recent iteration's full dashboard
- history: View iteration summaries over a time window
- trends: Analyze trends across recent iterations (success rate, duration, revenue)
- compare: Compare two iterations side-by-side
- subsystem_health: Get per-subsystem health scores from recent iterations
- alerts: Get alerts for degradation patterns detected across iterations
- configure: Set dashboard thresholds and alert sensitivity
"""

import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
from .base import Skill, SkillResult, SkillManifest, SkillAction

DATA_DIR = Path(__file__).parent.parent / "data"
LOOP_STATE_FILE = DATA_DIR / "autonomous_loop.json"
CIRCUIT_FILE = DATA_DIR / "circuit_breaker.json"
GOALS_FILE = DATA_DIR / "goals.json"
SCHEDULER_FILE = DATA_DIR / "scheduler.json"
FLEET_FILE = DATA_DIR / "fleet_health.json"
REPUTATION_FILE = DATA_DIR / "reputation.json"
DASHBOARD_STATE_FILE = DATA_DIR / "loop_iteration_dashboard.json"

MAX_SNAPSHOTS = 500


def _now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _load_json(path: Path) -> Optional[Dict]:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _save_json(path: Path, data: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


DEFAULT_CONFIG = {
    "enabled": True,
    # Alert thresholds
    "success_rate_alert_threshold": 0.6,  # Alert if success rate drops below 60%
    "avg_duration_alert_threshold": 120.0,  # Alert if avg duration exceeds 120s
    "failure_streak_alert_threshold": 3,  # Alert after 3 consecutive failures
    "revenue_decline_alert_pct": 20.0,  # Alert if revenue drops 20% vs prior window
    # Trend analysis window
    "trend_window_size": 20,  # Analyze last N iterations for trends
    # Subsystem weights for overall health score
    "subsystem_weights": {
        "loop_execution": 0.25,
        "circuit_breaker": 0.15,
        "scheduler": 0.15,
        "fleet_health": 0.15,
        "goal_progress": 0.15,
        "reputation": 0.15,
    },
}


class LoopIterationDashboardSkill(Skill):
    """Unified view of all autonomous loop iteration stats."""

    def __init__(self, credentials: Dict[str, str] = None):
        super().__init__(credentials)
        self._ensure_data()

    def _ensure_data(self):
        DASHBOARD_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        if not DASHBOARD_STATE_FILE.exists():
            _save_json(DASHBOARD_STATE_FILE, self._default_state())

    def _default_state(self) -> Dict:
        return {
            "config": dict(DEFAULT_CONFIG),
            "snapshots": [],
            "alert_history": [],
            "created_at": _now_iso(),
        }

    def _load_state(self) -> Dict:
        return _load_json(DASHBOARD_STATE_FILE) or self._default_state()

    def _save_state(self, state: Dict):
        _save_json(DASHBOARD_STATE_FILE, state)

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="loop_iteration_dashboard",
            name="Loop Iteration Dashboard",
            version="1.0.0",
            category="observability",
            description="Unified view of all autonomous loop iteration stats with trend analysis and alerts",
            actions=[
                SkillAction(
                    name="latest",
                    description="Get the most recent iteration's full dashboard view",
                    parameters={},
                ),
                SkillAction(
                    name="history",
                    description="View iteration summaries over a time window",
                    parameters={
                        "limit": {"type": "int", "required": False, "description": "Max entries to return (default 10)"},
                    },
                ),
                SkillAction(
                    name="trends",
                    description="Analyze trends across recent iterations",
                    parameters={
                        "window": {"type": "int", "required": False, "description": "Number of iterations to analyze (default from config)"},
                    },
                ),
                SkillAction(
                    name="compare",
                    description="Compare two iterations side-by-side",
                    parameters={
                        "iteration_a": {"type": "str", "required": True, "description": "First iteration ID"},
                        "iteration_b": {"type": "str", "required": True, "description": "Second iteration ID"},
                    },
                ),
                SkillAction(
                    name="subsystem_health",
                    description="Per-subsystem health scores from recent iterations",
                    parameters={
                        "window": {"type": "int", "required": False, "description": "Iterations to analyze"},
                    },
                ),
                SkillAction(
                    name="alerts",
                    description="Get degradation alerts detected across iterations",
                    parameters={
                        "limit": {"type": "int", "required": False, "description": "Max alerts to return"},
                    },
                ),
                SkillAction(
                    name="configure",
                    description="Set dashboard thresholds and alert sensitivity",
                    parameters={
                        "config": {"type": "dict", "required": True, "description": "Config overrides"},
                    },
                ),
            ],
            required_credentials=[],
        )

    def check_credentials(self) -> bool:
        return True

    async def execute(self, action: str, params: Dict) -> SkillResult:
        actions = {
            "latest": self._latest,
            "history": self._history,
            "trends": self._trends,
            "compare": self._compare,
            "subsystem_health": self._subsystem_health,
            "alerts": self._alerts,
            "configure": self._configure,
        }
        handler = actions.get(action)
        if not handler:
            return SkillResult(success=False, message=f"Unknown action: {action}")
        return await handler(params)

    # ── Data Loading ─────────────────────────────────────────────────

    def _get_journal(self) -> List[Dict]:
        """Load autonomous loop journal entries."""
        loop_data = _load_json(LOOP_STATE_FILE)
        if not loop_data:
            return []
        return loop_data.get("journal", [])

    def _get_loop_stats(self) -> Dict:
        """Load autonomous loop aggregate stats."""
        loop_data = _load_json(LOOP_STATE_FILE)
        if not loop_data:
            return {}
        return loop_data.get("stats", {})

    def _get_circuit_data(self) -> Dict:
        """Load circuit breaker state."""
        return _load_json(CIRCUIT_FILE) or {}

    def _get_scheduler_data(self) -> Dict:
        """Load scheduler state."""
        return _load_json(SCHEDULER_FILE) or {}

    def _get_fleet_data(self) -> Dict:
        """Load fleet health data."""
        return _load_json(FLEET_FILE) or {}

    def _get_reputation_data(self) -> Dict:
        """Load reputation data."""
        return _load_json(REPUTATION_FILE) or {}

    def _get_goals_data(self) -> Dict:
        """Load goals data."""
        return _load_json(GOALS_FILE) or {}

    # ── Iteration Enrichment ────────────────────────────────────────

    def _enrich_iteration(self, entry: Dict) -> Dict:
        """Enrich a journal entry with subsystem context for dashboard display."""
        phases = entry.get("phases", {})
        enriched = {
            "iteration_id": entry.get("iteration_id", "unknown"),
            "started_at": entry.get("started_at", ""),
            "completed_at": entry.get("completed_at", ""),
            "outcome": entry.get("outcome", "unknown"),
            "duration_seconds": entry.get("duration_seconds", 0),
            "task": {
                "description": entry.get("task_description", phases.get("decide", {}).get("task", "")),
                "pillar": entry.get("pillar", phases.get("decide", {}).get("pillar", "")),
                "skill": phases.get("act", {}).get("skill_used", ""),
            },
            "phases": {
                "assess": {
                    "ran": "assess" in phases,
                    "weakest_pillar": phases.get("assess", {}).get("weakest_pillar", ""),
                    "score": phases.get("assess", {}).get("overall_score", None),
                },
                "decide": {
                    "ran": "decide" in phases,
                    "task": phases.get("decide", {}).get("task", ""),
                    "priority": phases.get("decide", {}).get("priority", ""),
                },
                "plan": {
                    "ran": "plan" in phases,
                    "steps": len(phases.get("plan", {}).get("steps", [])),
                },
                "act": {
                    "ran": "act" in phases,
                    "success": phases.get("act", {}).get("success", False),
                    "actions_taken": phases.get("act", {}).get("actions_taken", 0),
                    "actions_succeeded": phases.get("act", {}).get("actions_succeeded", 0),
                },
                "measure": {
                    "ran": "measure" in phases,
                    "revenue": phases.get("measure", {}).get("revenue", 0),
                    "cost": phases.get("measure", {}).get("cost", 0),
                },
                "learn": {
                    "ran": "learn" in phases,
                    "feedback": phases.get("learn", {}).get("feedback_recorded", False),
                },
            },
        }
        return enriched

    # ── Health Scoring ──────────────────────────────────────────────

    def _score_loop_execution(self, journal: List[Dict], window: int) -> Dict:
        """Score the loop execution subsystem based on recent iterations."""
        recent = journal[-window:] if journal else []
        if not recent:
            return {"score": 50, "detail": "No iterations yet", "iterations": 0}

        successes = sum(1 for e in recent if e.get("outcome") == "success")
        total = len(recent)
        success_rate = successes / total if total > 0 else 0
        avg_duration = sum(e.get("duration_seconds", 0) for e in recent) / total if total > 0 else 0

        score = int(success_rate * 80 + min(20, max(0, 20 - avg_duration / 10)))
        score = max(0, min(100, score))

        return {
            "score": score,
            "success_rate": round(success_rate, 3),
            "avg_duration_seconds": round(avg_duration, 2),
            "total_iterations": total,
            "successes": successes,
            "failures": total - successes,
        }

    def _score_circuit_breaker(self) -> Dict:
        """Score circuit breaker health."""
        cb_data = self._get_circuit_data()
        circuits = cb_data.get("circuits", {})
        if not circuits:
            return {"score": 100, "detail": "No circuits tracked", "total": 0}

        total = len(circuits)
        open_count = sum(1 for c in circuits.values() if c.get("state") == "open")
        half_open = sum(1 for c in circuits.values() if c.get("state") == "half_open")
        closed = total - open_count - half_open

        if total == 0:
            score = 100
        else:
            score = int((closed / total) * 80 + (half_open / total) * 40)
            score = max(0, min(100, score))

        return {
            "score": score,
            "total_circuits": total,
            "closed": closed,
            "half_open": half_open,
            "open": open_count,
        }

    def _score_scheduler(self) -> Dict:
        """Score scheduler health."""
        sched = self._get_scheduler_data()
        tasks = sched.get("tasks", [])
        if not tasks:
            return {"score": 100, "detail": "No scheduled tasks", "total": 0}

        total = len(tasks)
        active = sum(1 for t in tasks if t.get("status") == "active")
        paused = sum(1 for t in tasks if t.get("status") == "paused")
        completed = sum(1 for t in tasks if t.get("status") == "completed")

        # High score if most tasks are active or completed
        if total == 0:
            score = 100
        else:
            score = int(((active + completed) / total) * 100)
            score = max(0, min(100, score))

        return {
            "score": score,
            "total_tasks": total,
            "active": active,
            "paused": paused,
            "completed": completed,
        }

    def _score_fleet_health(self) -> Dict:
        """Score fleet health subsystem."""
        fleet = self._get_fleet_data()
        agents = fleet.get("agents", fleet.get("fleet", []))
        if not agents:
            return {"score": 100, "detail": "No fleet agents tracked", "total": 0}

        if isinstance(agents, dict):
            agents = list(agents.values())

        total = len(agents)
        healthy = sum(1 for a in agents if a.get("status") in ("healthy", "running", "active"))
        unhealthy = total - healthy

        score = int((healthy / total) * 100) if total > 0 else 100
        return {
            "score": max(0, min(100, score)),
            "total_agents": total,
            "healthy": healthy,
            "unhealthy": unhealthy,
        }

    def _score_goal_progress(self) -> Dict:
        """Score goal progress subsystem."""
        goals = self._get_goals_data()
        goal_list = goals.get("goals", [])
        if not goal_list:
            return {"score": 50, "detail": "No goals set", "total": 0}

        total = len(goal_list)
        completed = sum(1 for g in goal_list if g.get("status") == "completed")
        active = sum(1 for g in goal_list if g.get("status") in ("active", "in_progress"))
        stalled = sum(1 for g in goal_list if g.get("status") == "stalled")
        blocked = sum(1 for g in goal_list if g.get("status") == "blocked")

        if total == 0:
            score = 50
        else:
            score = int(((completed * 1.0 + active * 0.7) / total) * 100)
            score = max(0, min(100, score - stalled * 10 - blocked * 15))

        return {
            "score": score,
            "total_goals": total,
            "completed": completed,
            "active": active,
            "stalled": stalled,
            "blocked": blocked,
        }

    def _score_reputation(self) -> Dict:
        """Score reputation subsystem."""
        rep = self._get_reputation_data()
        if not rep:
            return {"score": 50, "detail": "No reputation data", "total_reviews": 0}

        score_val = rep.get("score", rep.get("overall_score", 50))
        reviews = rep.get("total_reviews", rep.get("review_count", 0))

        return {
            "score": max(0, min(100, int(score_val))),
            "total_reviews": reviews,
            "raw_score": score_val,
        }

    # ── Alert Detection ─────────────────────────────────────────────

    def _detect_alerts(self, journal: List[Dict], config: Dict) -> List[Dict]:
        """Detect degradation patterns from recent iterations."""
        alerts = []
        window = config.get("trend_window_size", 20)
        recent = journal[-window:] if journal else []

        if not recent:
            return alerts

        # 1. Success rate alert
        total = len(recent)
        successes = sum(1 for e in recent if e.get("outcome") == "success")
        success_rate = successes / total if total > 0 else 0
        threshold = config.get("success_rate_alert_threshold", 0.6)
        if success_rate < threshold:
            alerts.append({
                "type": "low_success_rate",
                "severity": "critical" if success_rate < threshold * 0.5 else "warning",
                "message": f"Success rate {success_rate:.1%} is below threshold {threshold:.1%}",
                "value": round(success_rate, 3),
                "threshold": threshold,
                "detected_at": _now_iso(),
            })

        # 2. Duration alert
        avg_dur = sum(e.get("duration_seconds", 0) for e in recent) / total if total > 0 else 0
        dur_threshold = config.get("avg_duration_alert_threshold", 120.0)
        if avg_dur > dur_threshold:
            alerts.append({
                "type": "slow_iterations",
                "severity": "warning",
                "message": f"Avg iteration duration {avg_dur:.1f}s exceeds threshold {dur_threshold:.1f}s",
                "value": round(avg_dur, 2),
                "threshold": dur_threshold,
                "detected_at": _now_iso(),
            })

        # 3. Failure streak alert
        streak_threshold = config.get("failure_streak_alert_threshold", 3)
        streak = 0
        for entry in reversed(recent):
            if entry.get("outcome") != "success":
                streak += 1
            else:
                break
        if streak >= streak_threshold:
            alerts.append({
                "type": "failure_streak",
                "severity": "critical",
                "message": f"Last {streak} iterations failed (threshold: {streak_threshold})",
                "value": streak,
                "threshold": streak_threshold,
                "detected_at": _now_iso(),
            })

        # 4. Revenue decline alert
        if len(recent) >= 4:
            half = len(recent) // 2
            first_half = recent[:half]
            second_half = recent[half:]
            rev_first = sum(
                e.get("phases", {}).get("measure", {}).get("revenue", 0) for e in first_half
            )
            rev_second = sum(
                e.get("phases", {}).get("measure", {}).get("revenue", 0) for e in second_half
            )
            if rev_first > 0:
                decline_pct = ((rev_first - rev_second) / rev_first) * 100
                rev_threshold = config.get("revenue_decline_alert_pct", 20.0)
                if decline_pct > rev_threshold:
                    alerts.append({
                        "type": "revenue_decline",
                        "severity": "warning",
                        "message": f"Revenue declined {decline_pct:.1f}% vs prior window (threshold: {rev_threshold:.1f}%)",
                        "value": round(decline_pct, 1),
                        "threshold": rev_threshold,
                        "detected_at": _now_iso(),
                    })

        return alerts

    # ── Actions ──────────────────────────────────────────────────────

    async def _latest(self, params: Dict) -> SkillResult:
        """Get the most recent iteration's full dashboard."""
        journal = self._get_journal()
        if not journal:
            return SkillResult(
                success=True,
                message="No loop iterations recorded yet",
                data={"iteration": None, "stats": {}, "subsystem_health": {}},
            )

        latest_entry = journal[-1]
        enriched = self._enrich_iteration(latest_entry)
        stats = self._get_loop_stats()
        config = self._load_state().get("config", DEFAULT_CONFIG)
        window = config.get("trend_window_size", 20)

        # Build subsystem health snapshot
        health = {
            "loop_execution": self._score_loop_execution(journal, window),
            "circuit_breaker": self._score_circuit_breaker(),
            "scheduler": self._score_scheduler(),
            "fleet_health": self._score_fleet_health(),
            "goal_progress": self._score_goal_progress(),
            "reputation": self._score_reputation(),
        }

        # Compute weighted overall health
        weights = config.get("subsystem_weights", DEFAULT_CONFIG["subsystem_weights"])
        overall = sum(
            health[k]["score"] * weights.get(k, 0.15) for k in health
        )
        overall = max(0, min(100, int(overall)))

        # Detect alerts
        alerts = self._detect_alerts(journal, config)

        return SkillResult(
            success=True,
            message=f"Latest iteration: {enriched['iteration_id']} | "
                    f"Outcome: {enriched['outcome']} | "
                    f"Overall health: {overall}/100 | "
                    f"Alerts: {len(alerts)}",
            data={
                "iteration": enriched,
                "loop_stats": stats,
                "subsystem_health": health,
                "overall_health_score": overall,
                "alerts": alerts,
            },
        )

    async def _history(self, params: Dict) -> SkillResult:
        """View iteration summaries."""
        limit = params.get("limit", 10)
        journal = self._get_journal()

        if not journal:
            return SkillResult(success=True, message="No iterations yet", data={"iterations": []})

        recent = journal[-limit:]
        summaries = []
        for entry in recent:
            summaries.append({
                "iteration_id": entry.get("iteration_id", ""),
                "started_at": entry.get("started_at", ""),
                "outcome": entry.get("outcome", ""),
                "duration_seconds": entry.get("duration_seconds", 0),
                "pillar": entry.get("pillar", ""),
                "task": entry.get("task_description", "")[:80],
            })

        total = len(journal)
        successes = sum(1 for e in journal if e.get("outcome") == "success")
        return SkillResult(
            success=True,
            message=f"Showing {len(summaries)}/{total} iterations | "
                    f"Overall success rate: {successes/total:.1%}" if total > 0 else "No iterations",
            data={
                "iterations": summaries,
                "total_iterations": total,
                "total_successes": successes,
            },
        )

    async def _trends(self, params: Dict) -> SkillResult:
        """Analyze trends across recent iterations."""
        config = self._load_state().get("config", DEFAULT_CONFIG)
        window = params.get("window", config.get("trend_window_size", 20))
        journal = self._get_journal()

        if len(journal) < 2:
            return SkillResult(
                success=True,
                message="Need at least 2 iterations for trend analysis",
                data={"trends": {}},
            )

        recent = journal[-window:]
        total = len(recent)

        # Success rate trend (compare first half vs second half)
        half = total // 2
        first_half = recent[:half]
        second_half = recent[half:]

        sr_first = sum(1 for e in first_half if e.get("outcome") == "success") / len(first_half) if first_half else 0
        sr_second = sum(1 for e in second_half if e.get("outcome") == "success") / len(second_half) if second_half else 0
        sr_delta = sr_second - sr_first

        # Duration trend
        dur_first = sum(e.get("duration_seconds", 0) for e in first_half) / len(first_half) if first_half else 0
        dur_second = sum(e.get("duration_seconds", 0) for e in second_half) / len(second_half) if second_half else 0
        dur_delta = dur_second - dur_first

        # Revenue trend
        rev_first = sum(
            e.get("phases", {}).get("measure", {}).get("revenue", 0) for e in first_half
        )
        rev_second = sum(
            e.get("phases", {}).get("measure", {}).get("revenue", 0) for e in second_half
        )
        rev_delta = rev_second - rev_first

        # Pillar distribution
        pillar_counts = {}
        for e in recent:
            p = e.get("pillar", "unknown")
            pillar_counts[p] = pillar_counts.get(p, 0) + 1

        # Direction indicators
        def direction(delta):
            if delta > 0.05:
                return "improving"
            elif delta < -0.05:
                return "declining"
            return "stable"

        trends = {
            "window_size": total,
            "success_rate": {
                "first_half": round(sr_first, 3),
                "second_half": round(sr_second, 3),
                "delta": round(sr_delta, 3),
                "direction": direction(sr_delta),
            },
            "avg_duration_seconds": {
                "first_half": round(dur_first, 2),
                "second_half": round(dur_second, 2),
                "delta": round(dur_delta, 2),
                "direction": "improving" if dur_delta < -5 else ("declining" if dur_delta > 5 else "stable"),
            },
            "revenue": {
                "first_half": round(rev_first, 4),
                "second_half": round(rev_second, 4),
                "delta": round(rev_delta, 4),
                "direction": direction(rev_delta),
            },
            "pillar_distribution": pillar_counts,
        }

        msg_parts = [
            f"Trends over {total} iterations:",
            f"Success rate: {sr_second:.1%} ({trends['success_rate']['direction']})",
            f"Avg duration: {dur_second:.1f}s ({trends['avg_duration_seconds']['direction']})",
        ]

        return SkillResult(
            success=True,
            message=" | ".join(msg_parts),
            data={"trends": trends},
        )

    async def _compare(self, params: Dict) -> SkillResult:
        """Compare two iterations side-by-side."""
        id_a = params.get("iteration_a", "")
        id_b = params.get("iteration_b", "")

        if not id_a or not id_b:
            return SkillResult(success=False, message="Both iteration_a and iteration_b are required")

        journal = self._get_journal()
        entry_a = next((e for e in journal if e.get("iteration_id") == id_a), None)
        entry_b = next((e for e in journal if e.get("iteration_id") == id_b), None)

        if not entry_a:
            return SkillResult(success=False, message=f"Iteration {id_a} not found")
        if not entry_b:
            return SkillResult(success=False, message=f"Iteration {id_b} not found")

        enriched_a = self._enrich_iteration(entry_a)
        enriched_b = self._enrich_iteration(entry_b)

        # Compute deltas
        dur_a = enriched_a.get("duration_seconds", 0)
        dur_b = enriched_b.get("duration_seconds", 0)
        rev_a = enriched_a["phases"]["measure"]["revenue"]
        rev_b = enriched_b["phases"]["measure"]["revenue"]

        comparison = {
            "iteration_a": enriched_a,
            "iteration_b": enriched_b,
            "deltas": {
                "duration_seconds": round(dur_b - dur_a, 2),
                "revenue": round(rev_b - rev_a, 4),
                "outcome_match": enriched_a["outcome"] == enriched_b["outcome"],
            },
        }

        return SkillResult(
            success=True,
            message=f"Compared {id_a} vs {id_b} | "
                    f"Duration delta: {dur_b - dur_a:+.1f}s | "
                    f"Revenue delta: {rev_b - rev_a:+.4f}",
            data=comparison,
        )

    async def _subsystem_health(self, params: Dict) -> SkillResult:
        """Get per-subsystem health scores."""
        config = self._load_state().get("config", DEFAULT_CONFIG)
        window = params.get("window", config.get("trend_window_size", 20))
        journal = self._get_journal()

        health = {
            "loop_execution": self._score_loop_execution(journal, window),
            "circuit_breaker": self._score_circuit_breaker(),
            "scheduler": self._score_scheduler(),
            "fleet_health": self._score_fleet_health(),
            "goal_progress": self._score_goal_progress(),
            "reputation": self._score_reputation(),
        }

        weights = config.get("subsystem_weights", DEFAULT_CONFIG["subsystem_weights"])
        overall = sum(
            health[k]["score"] * weights.get(k, 0.15) for k in health
        )
        overall = max(0, min(100, int(overall)))

        # Identify weakest subsystem
        weakest = min(health, key=lambda k: health[k]["score"])
        strongest = max(health, key=lambda k: health[k]["score"])

        return SkillResult(
            success=True,
            message=f"Overall health: {overall}/100 | "
                    f"Weakest: {weakest} ({health[weakest]['score']}) | "
                    f"Strongest: {strongest} ({health[strongest]['score']})",
            data={
                "subsystem_health": health,
                "overall_health_score": overall,
                "weakest_subsystem": weakest,
                "strongest_subsystem": strongest,
            },
        )

    async def _alerts(self, params: Dict) -> SkillResult:
        """Get degradation alerts."""
        limit = params.get("limit", 20)
        config = self._load_state().get("config", DEFAULT_CONFIG)
        journal = self._get_journal()

        current_alerts = self._detect_alerts(journal, config)

        # Also load historical alerts
        state = self._load_state()
        historical = state.get("alert_history", [])[-limit:]

        # Save current alerts to history
        if current_alerts:
            state["alert_history"] = (state.get("alert_history", []) + current_alerts)[-MAX_SNAPSHOTS:]
            self._save_state(state)

        critical = sum(1 for a in current_alerts if a.get("severity") == "critical")
        warnings = sum(1 for a in current_alerts if a.get("severity") == "warning")

        return SkillResult(
            success=True,
            message=f"Current alerts: {len(current_alerts)} ({critical} critical, {warnings} warnings)",
            data={
                "current_alerts": current_alerts,
                "alert_history": historical[-limit:],
                "critical_count": critical,
                "warning_count": warnings,
            },
        )

    async def _configure(self, params: Dict) -> SkillResult:
        """Update dashboard configuration."""
        overrides = params.get("config", {})
        if not overrides:
            return SkillResult(success=False, message="No config overrides provided")

        state = self._load_state()
        config = state.get("config", dict(DEFAULT_CONFIG))

        updated = []
        for key, value in overrides.items():
            if key in DEFAULT_CONFIG:
                old = config.get(key)
                config[key] = value
                updated.append(f"{key}: {old} -> {value}")

        state["config"] = config
        self._save_state(state)

        return SkillResult(
            success=True,
            message=f"Updated {len(updated)} config values",
            data={"updated": updated, "current_config": config},
        )
