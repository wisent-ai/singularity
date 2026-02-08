#!/usr/bin/env python3
"""
SkillPerformanceProfiler - Analyze skill usage, ROI, and optimize the skill portfolio.

With 50+ skills loaded, the agent needs to know:
- Which skills are actually used vs. dead weight?
- Which skills generate revenue vs. only cost money?
- Which skills have the best success rates?
- What's the optimal skill set for this agent's goals?

This skill profiles the entire skill portfolio, tracks usage over time,
computes ROI metrics, detects unused skills, and recommends changes to
the skill configuration (pruning, prioritization, investment).

Pillar: Self-Improvement (optimize own capabilities based on data)
Also serves Revenue (focus on high-ROI skills) and Goal Setting (align
skills with strategic priorities).

Actions:
  - profile: Generate a full skill portfolio profile with usage/cost/success metrics
  - unused: List skills that are loaded but have never been used
  - roi: Compute ROI ranking - which skills produce the most value per cost?
  - recommend: Get recommendations for skill portfolio changes
  - record: Record a skill execution event (called automatically by agent loop)
  - trends: Show skill usage trends over recent sessions
  - budget: Estimate cost allocation across skills and suggest rebalancing
  - reset: Clear profiling data and start fresh
"""

import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict
from .base import Skill, SkillResult, SkillManifest, SkillAction


PROFILER_FILE = Path(__file__).parent.parent / "data" / "skill_profiler.json"
MAX_EVENTS = 5000
MAX_SESSIONS = 100


class SkillPerformanceProfiler(Skill):
    """
    Profiles the agent's skill portfolio to optimize which skills to
    load, prioritize, and invest in improving.

    Unlike PerformanceTracker (which records raw action outcomes),
    SkillPerformanceProfiler operates at the skill level — analyzing
    the entire portfolio as a system and recommending changes.
    """

    def __init__(self, credentials: Dict[str, str] = None):
        super().__init__(credentials)
        self._installed_skills: List[str] = []
        self._ensure_data()

    def set_installed_skills(self, skill_ids: List[str]):
        """Called by agent to tell profiler what skills are loaded."""
        self._installed_skills = list(skill_ids)

    def _ensure_data(self):
        PROFILER_FILE.parent.mkdir(parents=True, exist_ok=True)
        if not PROFILER_FILE.exists():
            self._save(self._default_state())

    def _default_state(self) -> Dict:
        return {
            "events": [],
            "sessions": [],
            "skill_metadata": {},
            "recommendations_log": [],
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
        }

    def _load(self) -> Dict:
        try:
            with open(PROFILER_FILE, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return self._default_state()

    def _save(self, data: Dict):
        data["last_updated"] = datetime.now().isoformat()
        if len(data.get("events", [])) > MAX_EVENTS:
            data["events"] = data["events"][-MAX_EVENTS:]
        if len(data.get("sessions", [])) > MAX_SESSIONS:
            data["sessions"] = data["sessions"][-MAX_SESSIONS:]
        if len(data.get("recommendations_log", [])) > MAX_SESSIONS:
            data["recommendations_log"] = data["recommendations_log"][-MAX_SESSIONS:]
        with open(PROFILER_FILE, "w") as f:
            json.dump(data, f, indent=2, default=str)

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="skill_profiler",
            name="Skill Performance Profiler",
            version="1.0.0",
            category="meta",
            description=(
                "Profile the agent's skill portfolio — track usage, compute ROI, "
                "detect unused skills, and recommend portfolio optimization."
            ),
            actions=[
                SkillAction(
                    name="profile",
                    description="Generate full skill portfolio profile with usage, cost, and success metrics",
                    parameters={
                        "window_hours": {
                            "type": "number",
                            "required": False,
                            "description": "Time window in hours (default: all time)",
                        },
                    },
                ),
                SkillAction(
                    name="unused",
                    description="List skills that are loaded but have zero recorded usage",
                    parameters={},
                ),
                SkillAction(
                    name="roi",
                    description="Rank skills by return on investment (value per cost)",
                    parameters={
                        "top_n": {
                            "type": "number",
                            "required": False,
                            "description": "Number of top skills to return (default: 10)",
                        },
                    },
                ),
                SkillAction(
                    name="recommend",
                    description="Get recommendations for skill portfolio changes (prune, prioritize, invest)",
                    parameters={},
                ),
                SkillAction(
                    name="record",
                    description="Record a skill execution event",
                    parameters={
                        "skill_id": {
                            "type": "string",
                            "required": True,
                            "description": "ID of the skill that was executed",
                        },
                        "action": {
                            "type": "string",
                            "required": True,
                            "description": "Action that was executed",
                        },
                        "success": {
                            "type": "boolean",
                            "required": True,
                            "description": "Whether the execution succeeded",
                        },
                        "latency_ms": {
                            "type": "number",
                            "required": False,
                            "description": "Execution time in milliseconds",
                        },
                        "cost_usd": {
                            "type": "number",
                            "required": False,
                            "description": "Cost of execution in USD",
                        },
                        "revenue_usd": {
                            "type": "number",
                            "required": False,
                            "description": "Revenue generated in USD",
                        },
                    },
                ),
                SkillAction(
                    name="trends",
                    description="Show skill usage trends over recent sessions",
                    parameters={
                        "sessions": {
                            "type": "number",
                            "required": False,
                            "description": "Number of recent sessions to analyze (default: 10)",
                        },
                    },
                ),
                SkillAction(
                    name="budget",
                    description="Estimate cost allocation across skills and suggest rebalancing",
                    parameters={},
                ),
                SkillAction(
                    name="reset",
                    description="Clear all profiling data and start fresh",
                    parameters={},
                ),
            ],
            required_credentials=[],
        )

    async def execute(self, action: str, params: Dict = None) -> SkillResult:
        params = params or {}
        handlers = {
            "profile": self._profile,
            "unused": self._unused,
            "roi": self._roi,
            "recommend": self._recommend,
            "record": self._record,
            "trends": self._trends,
            "budget": self._budget,
            "reset": self._reset,
        }
        handler = handlers.get(action)
        if not handler:
            return SkillResult(
                success=False,
                message=f"Unknown action: {action}. Available: {list(handlers.keys())}",
            )
        return await handler(params)

    # ---- Actions ----

    async def _record(self, params: Dict) -> SkillResult:
        """Record a skill execution event."""
        skill_id = params.get("skill_id", "")
        action = params.get("action", "")
        success = params.get("success", False)

        if not skill_id or not action:
            return SkillResult(
                success=False,
                message="skill_id and action are required",
            )

        data = self._load()
        event = {
            "skill_id": skill_id,
            "action": action,
            "success": bool(success),
            "latency_ms": params.get("latency_ms", 0),
            "cost_usd": params.get("cost_usd", 0.0),
            "revenue_usd": params.get("revenue_usd", 0.0),
            "timestamp": datetime.now().isoformat(),
        }
        data["events"].append(event)
        self._save(data)

        return SkillResult(
            success=True,
            message=f"Recorded {skill_id}:{action} (success={success})",
            data=event,
        )

    async def _profile(self, params: Dict) -> SkillResult:
        """Generate full skill portfolio profile."""
        data = self._load()
        events = data.get("events", [])

        window_hours = params.get("window_hours")
        if window_hours:
            cutoff = (datetime.now() - timedelta(hours=float(window_hours))).isoformat()
            events = [e for e in events if e.get("timestamp", "") >= cutoff]

        # Aggregate per skill
        skill_stats = defaultdict(lambda: {
            "total_calls": 0,
            "successes": 0,
            "failures": 0,
            "total_cost": 0.0,
            "total_revenue": 0.0,
            "total_latency_ms": 0,
            "actions_used": set(),
            "first_used": None,
            "last_used": None,
        })

        for event in events:
            sid = event["skill_id"]
            stats = skill_stats[sid]
            stats["total_calls"] += 1
            if event.get("success"):
                stats["successes"] += 1
            else:
                stats["failures"] += 1
            stats["total_cost"] += event.get("cost_usd", 0)
            stats["total_revenue"] += event.get("revenue_usd", 0)
            stats["total_latency_ms"] += event.get("latency_ms", 0)
            stats["actions_used"].add(event.get("action", ""))
            ts = event.get("timestamp", "")
            if stats["first_used"] is None or ts < stats["first_used"]:
                stats["first_used"] = ts
            if stats["last_used"] is None or ts > stats["last_used"]:
                stats["last_used"] = ts

        # Build profile
        profiles = []
        for sid, stats in sorted(skill_stats.items(), key=lambda x: x[1]["total_calls"], reverse=True):
            success_rate = stats["successes"] / stats["total_calls"] if stats["total_calls"] > 0 else 0
            avg_latency = stats["total_latency_ms"] / stats["total_calls"] if stats["total_calls"] > 0 else 0
            net_value = stats["total_revenue"] - stats["total_cost"]
            profiles.append({
                "skill_id": sid,
                "total_calls": stats["total_calls"],
                "success_rate": round(success_rate, 3),
                "avg_latency_ms": round(avg_latency, 1),
                "total_cost_usd": round(stats["total_cost"], 4),
                "total_revenue_usd": round(stats["total_revenue"], 4),
                "net_value_usd": round(net_value, 4),
                "actions_used": sorted(stats["actions_used"]),
                "first_used": stats["first_used"],
                "last_used": stats["last_used"],
            })

        total_calls = sum(p["total_calls"] for p in profiles)
        total_cost = sum(p["total_cost_usd"] for p in profiles)
        total_revenue = sum(p["total_revenue_usd"] for p in profiles)
        skills_used = len(profiles)
        skills_installed = len(self._installed_skills)

        return SkillResult(
            success=True,
            message=f"Portfolio: {skills_used}/{skills_installed} skills used, {total_calls} total calls",
            data={
                "summary": {
                    "skills_installed": skills_installed,
                    "skills_used": skills_used,
                    "skills_unused": skills_installed - skills_used,
                    "utilization_rate": round(skills_used / skills_installed, 3) if skills_installed > 0 else 0,
                    "total_calls": total_calls,
                    "total_cost_usd": round(total_cost, 4),
                    "total_revenue_usd": round(total_revenue, 4),
                    "net_value_usd": round(total_revenue - total_cost, 4),
                },
                "skills": profiles,
            },
        )

    async def _unused(self, params: Dict) -> SkillResult:
        """List skills that are loaded but never used."""
        data = self._load()
        used_skills = set()
        for event in data.get("events", []):
            used_skills.add(event.get("skill_id", ""))

        unused = [s for s in self._installed_skills if s not in used_skills]
        unused.sort()

        return SkillResult(
            success=True,
            message=f"{len(unused)} skills loaded but never used",
            data={
                "unused_skills": unused,
                "used_skills": sorted(used_skills),
                "total_installed": len(self._installed_skills),
                "utilization_rate": round(
                    len(used_skills) / len(self._installed_skills), 3
                ) if self._installed_skills else 0,
            },
        )

    async def _roi(self, params: Dict) -> SkillResult:
        """Rank skills by return on investment."""
        data = self._load()
        top_n = int(params.get("top_n", 10))

        # Aggregate
        skill_totals = defaultdict(lambda: {
            "calls": 0, "successes": 0, "cost": 0.0, "revenue": 0.0,
        })
        for event in data.get("events", []):
            sid = event["skill_id"]
            skill_totals[sid]["calls"] += 1
            if event.get("success"):
                skill_totals[sid]["successes"] += 1
            skill_totals[sid]["cost"] += event.get("cost_usd", 0)
            skill_totals[sid]["revenue"] += event.get("revenue_usd", 0)

        # Compute ROI
        rankings = []
        for sid, totals in skill_totals.items():
            net = totals["revenue"] - totals["cost"]
            roi = net / totals["cost"] if totals["cost"] > 0 else (float("inf") if net > 0 else 0)
            success_rate = totals["successes"] / totals["calls"] if totals["calls"] > 0 else 0
            rankings.append({
                "skill_id": sid,
                "calls": totals["calls"],
                "success_rate": round(success_rate, 3),
                "cost_usd": round(totals["cost"], 4),
                "revenue_usd": round(totals["revenue"], 4),
                "net_usd": round(net, 4),
                "roi": round(roi, 2) if roi != float("inf") else "infinite",
            })

        # Sort by net value, then ROI
        rankings.sort(key=lambda x: (x["net_usd"], x["roi"] if x["roi"] != "infinite" else 999999), reverse=True)
        rankings = rankings[:top_n]

        return SkillResult(
            success=True,
            message=f"Top {len(rankings)} skills by ROI",
            data={"rankings": rankings},
        )

    async def _recommend(self, params: Dict) -> SkillResult:
        """Generate portfolio optimization recommendations."""
        data = self._load()
        events = data.get("events", [])

        # Aggregate stats
        skill_stats = defaultdict(lambda: {
            "calls": 0, "successes": 0, "failures": 0,
            "cost": 0.0, "revenue": 0.0,
        })
        for event in events:
            sid = event["skill_id"]
            skill_stats[sid]["calls"] += 1
            if event.get("success"):
                skill_stats[sid]["successes"] += 1
            else:
                skill_stats[sid]["failures"] += 1
            skill_stats[sid]["cost"] += event.get("cost_usd", 0)
            skill_stats[sid]["revenue"] += event.get("revenue_usd", 0)

        used_skills = set(skill_stats.keys())
        unused = [s for s in self._installed_skills if s not in used_skills]

        recommendations = []

        # 1. Prune unused skills
        if unused:
            recommendations.append({
                "type": "prune",
                "priority": "medium",
                "message": f"Consider unloading {len(unused)} unused skills to reduce memory and complexity",
                "skills": unused[:10],  # Top 10
            })

        # 2. Fix failing skills
        for sid, stats in skill_stats.items():
            if stats["calls"] >= 3:
                fail_rate = stats["failures"] / stats["calls"]
                if fail_rate > 0.5:
                    recommendations.append({
                        "type": "fix_or_disable",
                        "priority": "high",
                        "message": f"Skill '{sid}' has {fail_rate:.0%} failure rate over {stats['calls']} calls",
                        "skill": sid,
                        "failure_rate": round(fail_rate, 3),
                    })

        # 3. Invest in high-value skills
        revenue_skills = [
            (sid, stats) for sid, stats in skill_stats.items()
            if stats["revenue"] > 0
        ]
        if revenue_skills:
            revenue_skills.sort(key=lambda x: x[1]["revenue"], reverse=True)
            top_earners = [sid for sid, _ in revenue_skills[:3]]
            recommendations.append({
                "type": "invest",
                "priority": "high",
                "message": f"Top revenue skills: {', '.join(top_earners)}. Prioritize improving these.",
                "skills": top_earners,
            })

        # 4. Expensive but low-value skills
        for sid, stats in skill_stats.items():
            if stats["cost"] > 0.1 and stats["revenue"] == 0:
                recommendations.append({
                    "type": "review_cost",
                    "priority": "medium",
                    "message": f"Skill '{sid}' has cost ${stats['cost']:.4f} but generated $0 revenue",
                    "skill": sid,
                    "cost": round(stats["cost"], 4),
                })

        # 5. Underutilized high-success skills
        for sid, stats in skill_stats.items():
            success_rate = stats["successes"] / stats["calls"] if stats["calls"] > 0 else 0
            if success_rate > 0.9 and stats["calls"] < 5 and stats["calls"] >= 1:
                recommendations.append({
                    "type": "explore",
                    "priority": "low",
                    "message": f"Skill '{sid}' has {success_rate:.0%} success but only {stats['calls']} calls — worth exploring more",
                    "skill": sid,
                })

        # 6. Portfolio health score
        total_installed = len(self._installed_skills)
        total_used = len(used_skills)
        utilization = total_used / total_installed if total_installed > 0 else 0
        total_success = sum(s["successes"] for s in skill_stats.values())
        total_calls = sum(s["calls"] for s in skill_stats.values())
        overall_success_rate = total_success / total_calls if total_calls > 0 else 0

        health_score = round(
            (utilization * 30)  # 30% weight: how many skills are used
            + (overall_success_rate * 40)  # 40% weight: how reliable skills are
            + (min(total_calls / 100, 1.0) * 30),  # 30% weight: volume of usage
            1,
        )

        # Save recommendations
        rec_entry = {
            "timestamp": datetime.now().isoformat(),
            "health_score": health_score,
            "recommendations_count": len(recommendations),
            "utilization_rate": round(utilization, 3),
        }
        data["recommendations_log"].append(rec_entry)
        self._save(data)

        return SkillResult(
            success=True,
            message=f"Portfolio health: {health_score}/100 — {len(recommendations)} recommendations",
            data={
                "health_score": health_score,
                "utilization_rate": round(utilization, 3),
                "overall_success_rate": round(overall_success_rate, 3),
                "total_skills_installed": total_installed,
                "total_skills_used": total_used,
                "recommendations": recommendations,
            },
        )

    async def _trends(self, params: Dict) -> SkillResult:
        """Show skill usage trends over recent sessions."""
        data = self._load()
        events = data.get("events", [])
        num_sessions = int(params.get("sessions", 10))

        if not events:
            return SkillResult(
                success=True,
                message="No events recorded yet",
                data={"trends": [], "total_events": 0},
            )

        # Group events by day as a proxy for sessions
        day_buckets = defaultdict(lambda: defaultdict(int))
        for event in events:
            ts = event.get("timestamp", "")
            day = ts[:10] if len(ts) >= 10 else "unknown"
            sid = event.get("skill_id", "")
            day_buckets[day][sid] += 1

        # Get recent days
        sorted_days = sorted(day_buckets.keys(), reverse=True)[:num_sessions]
        sorted_days.reverse()

        # Build trend data
        all_skills = set()
        for day_data in day_buckets.values():
            all_skills.update(day_data.keys())

        trends = []
        for day in sorted_days:
            day_data = day_buckets[day]
            trends.append({
                "date": day,
                "total_calls": sum(day_data.values()),
                "skills_used": len(day_data),
                "top_skills": sorted(
                    day_data.items(), key=lambda x: x[1], reverse=True
                )[:5],
            })

        # Compute trend direction
        if len(trends) >= 2:
            recent_calls = trends[-1]["total_calls"]
            older_calls = trends[0]["total_calls"]
            trend_direction = "increasing" if recent_calls > older_calls else (
                "decreasing" if recent_calls < older_calls else "stable"
            )
        else:
            trend_direction = "insufficient_data"

        return SkillResult(
            success=True,
            message=f"Trends over {len(trends)} periods — direction: {trend_direction}",
            data={
                "trends": trends,
                "trend_direction": trend_direction,
                "total_events": len(events),
                "unique_skills_ever": len(all_skills),
            },
        )

    async def _budget(self, params: Dict) -> SkillResult:
        """Estimate cost allocation across skills and suggest rebalancing."""
        data = self._load()
        events = data.get("events", [])

        if not events:
            return SkillResult(
                success=True,
                message="No events recorded — no cost data available",
                data={"allocations": [], "total_cost": 0},
            )

        # Aggregate costs
        skill_costs = defaultdict(float)
        skill_revenue = defaultdict(float)
        for event in events:
            sid = event.get("skill_id", "")
            skill_costs[sid] += event.get("cost_usd", 0)
            skill_revenue[sid] += event.get("revenue_usd", 0)

        total_cost = sum(skill_costs.values())
        total_revenue = sum(skill_revenue.values())

        allocations = []
        for sid in sorted(skill_costs.keys(), key=lambda s: skill_costs[s], reverse=True):
            cost = skill_costs[sid]
            revenue = skill_revenue[sid]
            pct = (cost / total_cost * 100) if total_cost > 0 else 0
            allocations.append({
                "skill_id": sid,
                "cost_usd": round(cost, 4),
                "revenue_usd": round(revenue, 4),
                "net_usd": round(revenue - cost, 4),
                "cost_share_pct": round(pct, 1),
            })

        # Suggest rebalancing
        suggestions = []
        for alloc in allocations:
            if alloc["cost_share_pct"] > 30 and alloc["net_usd"] < 0:
                suggestions.append(
                    f"Skill '{alloc['skill_id']}' consumes {alloc['cost_share_pct']}% of budget but has negative ROI — reduce usage"
                )
            if alloc["revenue_usd"] > 0 and alloc["cost_share_pct"] < 10:
                suggestions.append(
                    f"Skill '{alloc['skill_id']}' generates revenue with only {alloc['cost_share_pct']}% of budget — consider increasing investment"
                )

        return SkillResult(
            success=True,
            message=f"Budget: ${total_cost:.4f} spent, ${total_revenue:.4f} earned across {len(allocations)} skills",
            data={
                "total_cost_usd": round(total_cost, 4),
                "total_revenue_usd": round(total_revenue, 4),
                "net_usd": round(total_revenue - total_cost, 4),
                "allocations": allocations,
                "suggestions": suggestions,
            },
        )

    async def _reset(self, params: Dict) -> SkillResult:
        """Clear all profiling data."""
        self._save(self._default_state())
        return SkillResult(
            success=True,
            message="Profiling data cleared",
            data={"status": "reset"},
        )
