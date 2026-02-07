#!/usr/bin/env python3
"""
Performance Tracker Skill - Persistent cross-session performance analytics.

This is the critical "measure" component of the act → measure → adapt
self-improvement feedback loop. Unlike RuntimeMetrics (which is in-memory
and resets each session), PerformanceTracker persists all action outcomes
to disk and computes analytics across sessions.

The agent can:
- Record action outcomes (success/fail, latency, cost) automatically
- Query per-skill and per-action success rates over time windows
- Detect degradation trends (is performance getting worse?)
- Get cost-efficiency rankings (best ROI per skill)
- Generate actionable insights for self-improvement decisions
- Compare performance between sessions

Part of the Self-Improvement pillar: enables data-driven adaptation.
"""

import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict
from .base import Skill, SkillResult, SkillManifest, SkillAction


PERF_FILE = Path(__file__).parent.parent / "data" / "performance.json"
MAX_RECORDS = 5000  # Cap stored records to prevent unbounded growth


class PerformanceTracker(Skill):
    """
    Persistent performance analytics for self-improvement.

    Records every action outcome and provides analytics that help
    the agent make better decisions about which skills to use,
    when to retry vs. skip, and where to focus improvement efforts.
    """

    def __init__(self, credentials: Dict[str, str] = None):
        super().__init__(credentials)
        self._ensure_data()

    def _ensure_data(self):
        PERF_FILE.parent.mkdir(parents=True, exist_ok=True)
        if not PERF_FILE.exists():
            self._save(self._default_state())

    def _default_state(self) -> Dict:
        return {
            "records": [],
            "sessions": [],
            "insights": [],
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
        }

    def _load(self) -> Dict:
        try:
            with open(PERF_FILE, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return self._default_state()

    def _save(self, data: Dict):
        data["last_updated"] = datetime.now().isoformat()
        # Trim records if over cap
        if len(data.get("records", [])) > MAX_RECORDS:
            data["records"] = data["records"][-MAX_RECORDS:]
        if len(data.get("insights", [])) > 200:
            data["insights"] = data["insights"][-200:]
        with open(PERF_FILE, "w") as f:
            json.dump(data, f, indent=2, default=str)

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="performance",
            name="Performance Tracker",
            version="1.0.0",
            category="meta",
            description="Persistent cross-session performance analytics for self-improvement",
            required_credentials=[],
            install_cost=0,
            actions=[
                SkillAction(
                    name="record",
                    description="Record an action outcome (usually called automatically by agent loop)",
                    parameters={
                        "skill_id": {"type": "string", "required": True, "description": "Skill that was executed"},
                        "action": {"type": "string", "required": True, "description": "Action name"},
                        "success": {"type": "boolean", "required": True, "description": "Whether the action succeeded"},
                        "latency_ms": {"type": "number", "required": False, "description": "Execution time in ms"},
                        "cost_usd": {"type": "number", "required": False, "description": "API cost in USD"},
                        "error": {"type": "string", "required": False, "description": "Error message if failed"},
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="skill_report",
                    description="Get performance report for a specific skill (success rate, latency, cost, trends)",
                    parameters={
                        "skill_id": {"type": "string", "required": True, "description": "Skill to report on"},
                        "hours": {"type": "number", "required": False, "description": "Lookback window in hours (default: all time)"},
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="rankings",
                    description="Rank all skills by success rate, cost efficiency, or usage frequency",
                    parameters={
                        "sort_by": {"type": "string", "required": False, "description": "Sort by: success_rate, cost_efficiency, usage (default: success_rate)"},
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="trends",
                    description="Detect performance trends - which skills are improving or degrading",
                    parameters={},
                    estimated_cost=0,
                ),
                SkillAction(
                    name="insights",
                    description="Generate actionable insights from performance data for self-improvement",
                    parameters={},
                    estimated_cost=0,
                ),
                SkillAction(
                    name="session_summary",
                    description="Get a summary comparing current session to historical performance",
                    parameters={},
                    estimated_cost=0,
                ),
                SkillAction(
                    name="start_session",
                    description="Mark the start of a new agent session for session-over-session comparison",
                    parameters={
                        "session_id": {"type": "string", "required": False, "description": "Optional session ID"},
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="reset",
                    description="Clear all performance data (use with caution)",
                    parameters={},
                    estimated_cost=0,
                ),
            ],
        )

    def check_credentials(self) -> bool:
        return True  # No credentials needed

    async def execute(self, action: str, params: Dict) -> SkillResult:
        if action == "record":
            return self._record(params)
        elif action == "skill_report":
            return self._skill_report(params.get("skill_id", ""), params.get("hours"))
        elif action == "rankings":
            return self._rankings(params.get("sort_by", "success_rate"))
        elif action == "trends":
            return self._trends()
        elif action == "insights":
            return self._generate_insights()
        elif action == "session_summary":
            return self._session_summary()
        elif action == "start_session":
            return self._start_session(params.get("session_id"))
        elif action == "reset":
            return self._reset()
        else:
            return SkillResult(success=False, message=f"Unknown action: {action}")

    # --- Core recording ---

    def _record(self, params: Dict) -> SkillResult:
        """Record an action outcome."""
        skill_id = params.get("skill_id", "")
        action = params.get("action", "")
        success = params.get("success", False)

        if not skill_id or not action:
            return SkillResult(success=False, message="skill_id and action are required")

        data = self._load()
        record = {
            "timestamp": datetime.now().isoformat(),
            "skill_id": skill_id,
            "action": action,
            "success": bool(success),
            "latency_ms": params.get("latency_ms", 0),
            "cost_usd": params.get("cost_usd", 0.0),
            "error": params.get("error", ""),
        }
        data["records"].append(record)
        self._save(data)

        return SkillResult(
            success=True,
            message=f"Recorded: {skill_id}:{action} -> {'OK' if success else 'FAIL'}",
            data=record,
        )

    def record_outcome(
        self,
        skill_id: str,
        action: str,
        success: bool,
        latency_ms: float = 0,
        cost_usd: float = 0.0,
        error: str = "",
    ):
        """Programmatic API for agent loop integration (no async needed)."""
        data = self._load()
        record = {
            "timestamp": datetime.now().isoformat(),
            "skill_id": skill_id,
            "action": action,
            "success": bool(success),
            "latency_ms": latency_ms,
            "cost_usd": cost_usd,
            "error": error,
        }
        data["records"].append(record)
        self._save(data)

    # --- Analytics ---

    def _filter_records(self, data: Dict, skill_id: str = None, hours: float = None) -> List[Dict]:
        """Filter records by skill and time window."""
        records = data.get("records", [])
        if skill_id:
            records = [r for r in records if r["skill_id"] == skill_id]
        if hours is not None:
            cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()
            records = [r for r in records if r["timestamp"] >= cutoff]
        return records

    def _compute_stats(self, records: List[Dict]) -> Dict:
        """Compute aggregate statistics from a list of records."""
        if not records:
            return {
                "total": 0,
                "successes": 0,
                "failures": 0,
                "success_rate": 0.0,
                "avg_latency_ms": 0.0,
                "total_cost_usd": 0.0,
                "avg_cost_usd": 0.0,
                "error_types": {},
            }

        successes = sum(1 for r in records if r.get("success"))
        failures = len(records) - successes
        latencies = [r.get("latency_ms", 0) for r in records if r.get("latency_ms", 0) > 0]
        costs = [r.get("cost_usd", 0) for r in records]

        # Count error types
        error_types: Dict[str, int] = defaultdict(int)
        for r in records:
            err = r.get("error", "")
            if err:
                # Normalize error to first 80 chars
                key = err[:80]
                error_types[key] += 1

        return {
            "total": len(records),
            "successes": successes,
            "failures": failures,
            "success_rate": successes / len(records) if records else 0.0,
            "avg_latency_ms": sum(latencies) / len(latencies) if latencies else 0.0,
            "total_cost_usd": sum(costs),
            "avg_cost_usd": sum(costs) / len(costs) if costs else 0.0,
            "error_types": dict(error_types),
        }

    def _skill_report(self, skill_id: str, hours: float = None) -> SkillResult:
        """Get detailed performance report for a skill."""
        if not skill_id:
            return SkillResult(success=False, message="skill_id is required")

        data = self._load()
        records = self._filter_records(data, skill_id=skill_id, hours=hours)

        if not records:
            return SkillResult(
                success=True,
                message=f"No records found for skill '{skill_id}'",
                data={"skill_id": skill_id, "total": 0},
            )

        stats = self._compute_stats(records)

        # Per-action breakdown
        actions: Dict[str, List[Dict]] = defaultdict(list)
        for r in records:
            actions[r["action"]].append(r)

        action_stats = {}
        for action_name, action_records in actions.items():
            action_stats[action_name] = self._compute_stats(action_records)

        report = {
            "skill_id": skill_id,
            "time_window_hours": hours,
            "overall": stats,
            "by_action": action_stats,
        }

        return SkillResult(
            success=True,
            message=f"Performance report for '{skill_id}': {stats['success_rate']:.0%} success rate over {stats['total']} actions",
            data=report,
        )

    def _rankings(self, sort_by: str = "success_rate") -> SkillResult:
        """Rank all skills by the given metric."""
        data = self._load()
        records = data.get("records", [])

        if not records:
            return SkillResult(success=True, message="No records yet", data={"rankings": []})

        # Group by skill
        by_skill: Dict[str, List[Dict]] = defaultdict(list)
        for r in records:
            by_skill[r["skill_id"]].append(r)

        rankings = []
        for skill_id, skill_records in by_skill.items():
            stats = self._compute_stats(skill_records)
            # Cost efficiency = successes per dollar (higher is better)
            cost_efficiency = (
                stats["successes"] / stats["total_cost_usd"]
                if stats["total_cost_usd"] > 0
                else float("inf") if stats["successes"] > 0 else 0.0
            )
            rankings.append({
                "skill_id": skill_id,
                "total": stats["total"],
                "success_rate": stats["success_rate"],
                "avg_latency_ms": stats["avg_latency_ms"],
                "total_cost_usd": stats["total_cost_usd"],
                "cost_efficiency": cost_efficiency,
            })

        # Sort
        if sort_by == "cost_efficiency":
            rankings.sort(key=lambda x: x["cost_efficiency"], reverse=True)
        elif sort_by == "usage":
            rankings.sort(key=lambda x: x["total"], reverse=True)
        else:  # success_rate
            rankings.sort(key=lambda x: (x["success_rate"], x["total"]), reverse=True)

        return SkillResult(
            success=True,
            message=f"Ranked {len(rankings)} skills by {sort_by}",
            data={"sort_by": sort_by, "rankings": rankings},
        )

    def _trends(self) -> SkillResult:
        """Detect performance trends - improving or degrading skills."""
        data = self._load()
        records = data.get("records", [])

        if len(records) < 10:
            return SkillResult(
                success=True,
                message="Need at least 10 records to detect trends",
                data={"trends": [], "insufficient_data": True},
            )

        # Split records into first-half and second-half for trend comparison
        by_skill: Dict[str, List[Dict]] = defaultdict(list)
        for r in records:
            by_skill[r["skill_id"]].append(r)

        trends = []
        for skill_id, skill_records in by_skill.items():
            if len(skill_records) < 4:
                continue

            mid = len(skill_records) // 2
            first_half = skill_records[:mid]
            second_half = skill_records[mid:]

            first_stats = self._compute_stats(first_half)
            second_stats = self._compute_stats(second_half)

            rate_delta = second_stats["success_rate"] - first_stats["success_rate"]
            latency_delta = second_stats["avg_latency_ms"] - first_stats["avg_latency_ms"]

            # Classify trend
            if rate_delta > 0.1:
                direction = "improving"
            elif rate_delta < -0.1:
                direction = "degrading"
            else:
                direction = "stable"

            trends.append({
                "skill_id": skill_id,
                "direction": direction,
                "success_rate_delta": round(rate_delta, 3),
                "latency_delta_ms": round(latency_delta, 1),
                "first_half_rate": round(first_stats["success_rate"], 3),
                "second_half_rate": round(second_stats["success_rate"], 3),
                "total_records": len(skill_records),
            })

        # Sort: degrading first (most urgent), then stable, then improving
        order = {"degrading": 0, "stable": 1, "improving": 2}
        trends.sort(key=lambda t: (order.get(t["direction"], 1), -abs(t["success_rate_delta"])))

        degrading = [t for t in trends if t["direction"] == "degrading"]
        improving = [t for t in trends if t["direction"] == "improving"]

        msg = f"Analyzed {len(trends)} skills: {len(degrading)} degrading, {len(improving)} improving"
        return SkillResult(success=True, message=msg, data={"trends": trends})

    def _generate_insights(self) -> SkillResult:
        """Generate actionable insights from performance data."""
        data = self._load()
        records = data.get("records", [])

        if not records:
            return SkillResult(
                success=True,
                message="No performance data yet",
                data={"insights": []},
            )

        insights = []

        # Group by skill
        by_skill: Dict[str, List[Dict]] = defaultdict(list)
        for r in records:
            by_skill[r["skill_id"]].append(r)

        # Insight 1: Unreliable skills (< 50% success rate with enough data)
        for skill_id, skill_records in by_skill.items():
            if len(skill_records) >= 5:
                stats = self._compute_stats(skill_records)
                if stats["success_rate"] < 0.5:
                    insights.append({
                        "type": "unreliable_skill",
                        "severity": "high",
                        "message": f"Skill '{skill_id}' has only {stats['success_rate']:.0%} success rate over {stats['total']} attempts. Consider avoiding or investigating root cause.",
                        "skill_id": skill_id,
                        "success_rate": stats["success_rate"],
                        "total": stats["total"],
                    })

        # Insight 2: Expensive skills (high cost per success)
        for skill_id, skill_records in by_skill.items():
            stats = self._compute_stats(skill_records)
            if stats["total_cost_usd"] > 0 and stats["successes"] > 0:
                cost_per_success = stats["total_cost_usd"] / stats["successes"]
                if cost_per_success > 0.1:  # More than 10 cents per success
                    insights.append({
                        "type": "expensive_skill",
                        "severity": "medium",
                        "message": f"Skill '{skill_id}' costs ${cost_per_success:.4f} per successful action. Look for cheaper alternatives.",
                        "skill_id": skill_id,
                        "cost_per_success": cost_per_success,
                    })

        # Insight 3: Repeated errors (same error message appearing multiple times)
        all_errors: Dict[str, int] = defaultdict(int)
        for r in records:
            if r.get("error"):
                all_errors[r["error"][:80]] += 1
        for error, count in all_errors.items():
            if count >= 3:
                insights.append({
                    "type": "repeated_error",
                    "severity": "medium",
                    "message": f"Error occurred {count} times: '{error}'. Fix the root cause.",
                    "error": error,
                    "count": count,
                })

        # Insight 4: Underutilized successful skills
        total_actions = len(records)
        for skill_id, skill_records in by_skill.items():
            stats = self._compute_stats(skill_records)
            usage_pct = stats["total"] / total_actions if total_actions > 0 else 0
            if stats["success_rate"] > 0.9 and usage_pct < 0.05 and stats["total"] >= 3:
                insights.append({
                    "type": "underutilized_skill",
                    "severity": "low",
                    "message": f"Skill '{skill_id}' has {stats['success_rate']:.0%} success rate but only {usage_pct:.0%} of usage. Use it more.",
                    "skill_id": skill_id,
                    "success_rate": stats["success_rate"],
                    "usage_pct": usage_pct,
                })

        # Insight 5: Session-over-session comparison
        sessions = data.get("sessions", [])
        if len(sessions) >= 2:
            prev_session = sessions[-2]
            curr_session = sessions[-1]
            prev_rate = prev_session.get("success_rate", 0)
            curr_rate = curr_session.get("success_rate", 0)
            if curr_rate < prev_rate - 0.1:
                insights.append({
                    "type": "session_regression",
                    "severity": "high",
                    "message": f"Current session success rate ({curr_rate:.0%}) is worse than previous ({prev_rate:.0%}). Investigate what changed.",
                    "previous_rate": prev_rate,
                    "current_rate": curr_rate,
                })

        # Sort by severity
        severity_order = {"high": 0, "medium": 1, "low": 2}
        insights.sort(key=lambda i: severity_order.get(i["severity"], 1))

        # Persist insights
        data["insights"].extend([{
            **i,
            "generated_at": datetime.now().isoformat(),
        } for i in insights])
        self._save(data)

        return SkillResult(
            success=True,
            message=f"Generated {len(insights)} insights from {len(records)} records",
            data={"insights": insights, "total_records": len(records)},
        )

    # --- Session tracking ---

    def _start_session(self, session_id: str = None) -> SkillResult:
        """Mark the start of a new session."""
        data = self._load()
        session = {
            "session_id": session_id or datetime.now().strftime("%Y%m%d_%H%M%S"),
            "started_at": datetime.now().isoformat(),
            "records_at_start": len(data.get("records", [])),
            "success_rate": 0.0,
            "total_actions": 0,
        }
        data.setdefault("sessions", []).append(session)
        self._save(data)
        return SkillResult(
            success=True,
            message=f"Session '{session['session_id']}' started with {session['records_at_start']} prior records",
            data=session,
        )

    def _session_summary(self) -> SkillResult:
        """Compare current session to historical performance."""
        data = self._load()
        sessions = data.get("sessions", [])
        records = data.get("records", [])

        if not sessions:
            # No session tracking started, give overall summary
            stats = self._compute_stats(records)
            return SkillResult(
                success=True,
                message=f"No sessions tracked. Overall: {stats['total']} actions, {stats['success_rate']:.0%} success rate",
                data={"overall": stats, "sessions": []},
            )

        current_session = sessions[-1]
        start_idx = current_session.get("records_at_start", 0)
        session_records = records[start_idx:]

        current_stats = self._compute_stats(session_records)

        # Update current session with latest stats
        current_session["success_rate"] = current_stats["success_rate"]
        current_session["total_actions"] = current_stats["total"]
        self._save(data)

        # Historical comparison
        historical_records = records[:start_idx]
        historical_stats = self._compute_stats(historical_records)

        comparison = {
            "current_session": {
                "session_id": current_session["session_id"],
                "started_at": current_session["started_at"],
                **current_stats,
            },
            "historical": historical_stats,
            "total_sessions": len(sessions),
        }

        # Compute delta
        if historical_stats["total"] > 0:
            rate_delta = current_stats["success_rate"] - historical_stats["success_rate"]
            comparison["success_rate_delta"] = round(rate_delta, 3)
            if rate_delta > 0.05:
                comparison["verdict"] = "improving"
            elif rate_delta < -0.05:
                comparison["verdict"] = "degrading"
            else:
                comparison["verdict"] = "stable"
        else:
            comparison["verdict"] = "first_session"

        return SkillResult(
            success=True,
            message=f"Session: {current_stats['total']} actions, {current_stats['success_rate']:.0%} success. Verdict: {comparison['verdict']}",
            data=comparison,
        )

    def _reset(self) -> SkillResult:
        """Clear all performance data."""
        self._save(self._default_state())
        return SkillResult(
            success=True,
            message="Performance data reset",
            data={},
        )

    # --- Context for LLM injection ---

    def get_context_summary(self) -> str:
        """
        Generate a concise performance summary for injection into LLM context.

        This enables the agent to be aware of its historical performance
        and make informed decisions about which skills to use.
        """
        data = self._load()
        records = data.get("records", [])

        if not records:
            return ""

        stats = self._compute_stats(records)
        lines = [
            f"Historical Performance ({stats['total']} actions across sessions):",
            f"  Overall success rate: {stats['success_rate']:.0%}",
        ]

        # Top skills by usage
        by_skill: Dict[str, List[Dict]] = defaultdict(list)
        for r in records:
            by_skill[r["skill_id"]].append(r)

        skill_summaries = []
        for skill_id, skill_records in by_skill.items():
            s = self._compute_stats(skill_records)
            skill_summaries.append((skill_id, s))

        skill_summaries.sort(key=lambda x: x[1]["total"], reverse=True)

        # Show top 5 most used
        for skill_id, s in skill_summaries[:5]:
            status = "OK" if s["success_rate"] >= 0.8 else "WARN" if s["success_rate"] >= 0.5 else "BAD"
            lines.append(f"  [{status}] {skill_id}: {s['success_rate']:.0%} success ({s['total']} uses)")

        # Highlight any degrading trends
        if len(records) >= 10:
            # Simple trend: last 25% vs first 25%
            quarter = max(len(records) // 4, 1)
            early = self._compute_stats(records[:quarter])
            recent = self._compute_stats(records[-quarter:])
            if early["success_rate"] > 0 and recent["success_rate"] < early["success_rate"] - 0.1:
                lines.append(f"  ⚠ Performance trending down: {early['success_rate']:.0%} → {recent['success_rate']:.0%}")

        return "\n".join(lines)
