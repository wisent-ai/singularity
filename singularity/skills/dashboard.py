#!/usr/bin/env python3
"""
DashboardSkill - Self-monitoring dashboard for comprehensive agent status awareness.

Aggregates data from all major agent subsystems into a unified status report:
- Performance metrics (success rates, latency, trends)
- Resource/budget health (burn rate, remaining budget, alerts)
- Goal progress (active goals, completion rates, blocked items)
- Service status (hosted services, uptime, request counts)
- Revenue summary (earnings, top services, cost efficiency)
- Skill inventory (total skills, most/least used, health)
- Replication fleet status (peer count, health)

The dashboard is the agent's "self-awareness" layer. By seeing its own state
holistically, the agent can make better decisions about what to prioritize,
when to replicate, and where to invest effort.

Pillars:
- Goal Setting: Agent sees its own state to decide priorities intelligently
- Self-Improvement: Identifies bottlenecks, degradation, underperforming skills
- Revenue: Tracks earnings vs costs, identifies profitable services
- Replication: Fleet health visibility for spawn/terminate decisions

Actions:
- full_report: Complete dashboard with all sections
- summary: Quick 1-paragraph agent health summary
- pillar_scorecard: Score each of the 4 pillars 0-100
- export_html: Generate a standalone HTML dashboard page
- section_report: Get data for a specific section only
"""

import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from .base import Skill, SkillResult, SkillManifest, SkillAction


# Data file paths for various subsystems
DATA_DIR = Path(__file__).parent.parent / "data"
PERF_FILE = DATA_DIR / "performance.json"
RESOURCE_FILE = DATA_DIR / "resource_watcher.json"
GOALS_FILE = DATA_DIR / "goals.json"
SERVICE_FILE = DATA_DIR / "service_hosting.json"
USAGE_FILE = DATA_DIR / "usage_tracking.json"
HEALTH_FILE = DATA_DIR / "health_monitor.json"
KNOWLEDGE_FILE = DATA_DIR / "knowledge_sharing.json"
DASHBOARD_FILE = DATA_DIR / "dashboard_history.json"

MAX_HISTORY = 100  # Keep last N dashboard snapshots


class DashboardSkill(Skill):
    """
    Unified self-monitoring dashboard aggregating all agent subsystems.

    Provides the agent with comprehensive self-awareness by reading
    data from PerformanceTracker, ResourceWatcher, GoalManager,
    ServiceHosting, UsageTracking, and HealthMonitor.
    """

    def __init__(self, credentials: Dict[str, str] = None):
        super().__init__(credentials)
        self._ensure_data()

    def _ensure_data(self):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        if not DASHBOARD_FILE.exists():
            self._save_history({"snapshots": []})

    def _load_json(self, path: Path) -> Optional[Dict]:
        """Safely load a JSON data file, returning None if missing/corrupt."""
        try:
            if path.exists():
                return json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            pass
        return None

    def _save_history(self, data: Dict):
        DASHBOARD_FILE.write_text(json.dumps(data, indent=2, default=str))

    def _load_history(self) -> Dict:
        return self._load_json(DASHBOARD_FILE) or {"snapshots": []}

    # ── Manifest ──────────────────────────────────────────────────────

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="dashboard",
            name="Self-Monitoring Dashboard",
            version="1.0.0",
            category="monitor",
            description="Unified agent status dashboard aggregating performance, resources, goals, services, and fleet health",
            actions=self.get_actions(),
            required_credentials=[],
            install_cost=0,
            author="singularity",
        )

    def get_actions(self) -> List[SkillAction]:
        return [
            SkillAction(
                name="full_report",
                description="Generate complete dashboard with all sections",
                parameters={},
                estimated_cost=0,
                estimated_duration_seconds=2,
            ),
            SkillAction(
                name="summary",
                description="Quick 1-paragraph agent health summary",
                parameters={},
                estimated_cost=0,
                estimated_duration_seconds=1,
            ),
            SkillAction(
                name="pillar_scorecard",
                description="Score each of the 4 pillars (self-improvement, revenue, replication, goal-setting) from 0-100",
                parameters={},
                estimated_cost=0,
                estimated_duration_seconds=1,
            ),
            SkillAction(
                name="export_html",
                description="Generate a standalone HTML dashboard page",
                parameters={
                    "output_path": {
                        "type": "string",
                        "required": False,
                        "description": "Path to write HTML file (default: data/dashboard.html)",
                    }
                },
                estimated_cost=0,
                estimated_duration_seconds=2,
            ),
            SkillAction(
                name="section_report",
                description="Get data for a specific section: performance, resources, goals, services, revenue, skills, fleet",
                parameters={
                    "section": {
                        "type": "string",
                        "required": True,
                        "description": "Section name: performance|resources|goals|services|revenue|skills|fleet",
                    }
                },
                estimated_cost=0,
                estimated_duration_seconds=1,
            ),
        ]

    # ── Main execute ──────────────────────────────────────────────────

    async def execute(self, action: str, params: Dict[str, Any] = None) -> SkillResult:
        params = params or {}
        actions = {
            "full_report": self._full_report,
            "summary": self._summary,
            "pillar_scorecard": self._pillar_scorecard,
            "export_html": self._export_html,
            "section_report": self._section_report,
        }
        if action not in actions:
            return SkillResult(success=False, message=f"Unknown action: {action}. Available: {list(actions.keys())}")
        try:
            return await actions[action](params)
        except Exception as e:
            return SkillResult(success=False, message=f"Dashboard error: {str(e)}")

    # ── Data Collection ───────────────────────────────────────────────

    def _collect_performance(self) -> Dict:
        """Collect performance metrics from PerformanceTracker data."""
        data = self._load_json(PERF_FILE)
        if not data or "records" not in data:
            return {"available": False, "message": "No performance data found"}

        records = data["records"]
        if not records:
            return {"available": True, "total_actions": 0}

        successes = sum(1 for r in records if r.get("success"))
        failures = len(records) - successes
        total_cost = sum(r.get("cost", 0) for r in records)
        total_revenue = sum(r.get("revenue", 0) for r in records)
        avg_latency = sum(r.get("latency", 0) for r in records) / len(records) if records else 0

        # Recent trend (last 50 vs previous 50)
        trend = "stable"
        if len(records) >= 100:
            recent = records[-50:]
            previous = records[-100:-50]
            recent_rate = sum(1 for r in recent if r.get("success")) / 50
            prev_rate = sum(1 for r in previous if r.get("success")) / 50
            if recent_rate > prev_rate + 0.05:
                trend = "improving"
            elif recent_rate < prev_rate - 0.05:
                trend = "degrading"

        # Per-skill breakdown
        skill_stats = {}
        for r in records:
            sid = r.get("skill_id", "unknown")
            if sid not in skill_stats:
                skill_stats[sid] = {"total": 0, "success": 0, "cost": 0}
            skill_stats[sid]["total"] += 1
            if r.get("success"):
                skill_stats[sid]["success"] += 1
            skill_stats[sid]["cost"] += r.get("cost", 0)

        # Top 5 most used skills
        top_skills = sorted(skill_stats.items(), key=lambda x: x[1]["total"], reverse=True)[:5]

        return {
            "available": True,
            "total_actions": len(records),
            "successes": successes,
            "failures": failures,
            "success_rate": round(successes / len(records) * 100, 1) if records else 0,
            "total_cost": round(total_cost, 4),
            "total_revenue": round(total_revenue, 4),
            "avg_latency_ms": round(avg_latency * 1000, 1),
            "trend": trend,
            "top_skills": [
                {"skill": s[0], "uses": s[1]["total"], "success_rate": round(s[1]["success"] / s[1]["total"] * 100, 1) if s[1]["total"] > 0 else 0}
                for s in top_skills
            ],
        }

    def _collect_resources(self) -> Dict:
        """Collect resource/budget data from ResourceWatcher."""
        data = self._load_json(RESOURCE_FILE)
        if not data:
            return {"available": False, "message": "No resource data found"}

        budgets = data.get("budgets", {})
        consumption = data.get("consumption", [])

        total_budget = sum(b.get("limit", 0) for b in budgets.values())
        total_spent = sum(b.get("spent", 0) for b in budgets.values())
        remaining = total_budget - total_spent

        # Active alerts
        alerts = data.get("alerts", [])
        active_alerts = [a for a in alerts if not a.get("resolved")]

        return {
            "available": True,
            "total_budget": round(total_budget, 4),
            "total_spent": round(total_spent, 4),
            "remaining": round(remaining, 4),
            "budget_utilization_pct": round(total_spent / total_budget * 100, 1) if total_budget > 0 else 0,
            "active_alerts": len(active_alerts),
            "budget_categories": {k: {"limit": v.get("limit", 0), "spent": v.get("spent", 0)} for k, v in budgets.items()},
        }

    def _collect_goals(self) -> Dict:
        """Collect goal progress from GoalManager."""
        data = self._load_json(GOALS_FILE)
        if not data or "goals" not in data:
            return {"available": False, "message": "No goal data found"}

        goals = data["goals"]
        if not goals:
            return {"available": True, "total_goals": 0}

        by_status = {}
        by_pillar = {}
        for g in goals:
            status = g.get("status", "unknown")
            pillar = g.get("pillar", "other")
            by_status[status] = by_status.get(status, 0) + 1
            by_pillar[pillar] = by_pillar.get(pillar, 0) + 1

        active_goals = [g for g in goals if g.get("status") == "active"]
        completed_goals = [g for g in goals if g.get("status") == "completed"]

        # Milestone progress for active goals
        active_details = []
        for g in active_goals[:5]:
            milestones = g.get("milestones", [])
            done = sum(1 for m in milestones if m.get("completed"))
            active_details.append({
                "title": g.get("title", "Untitled"),
                "pillar": g.get("pillar", "other"),
                "priority": g.get("priority", "medium"),
                "milestones_done": done,
                "milestones_total": len(milestones),
            })

        return {
            "available": True,
            "total_goals": len(goals),
            "by_status": by_status,
            "by_pillar": by_pillar,
            "completion_rate": round(len(completed_goals) / len(goals) * 100, 1) if goals else 0,
            "active_goals": active_details,
        }

    def _collect_services(self) -> Dict:
        """Collect service hosting status."""
        data = self._load_json(SERVICE_FILE)
        if not data or "services" not in data:
            return {"available": False, "message": "No service data found"}

        services = data["services"]
        active = [s for s in services if s.get("status") == "active"]
        total_requests = sum(s.get("request_count", 0) for s in services)

        service_list = []
        for s in services[:10]:
            service_list.append({
                "name": s.get("name", "Unknown"),
                "status": s.get("status", "unknown"),
                "requests": s.get("request_count", 0),
                "endpoint": s.get("endpoint", ""),
            })

        return {
            "available": True,
            "total_services": len(services),
            "active_services": len(active),
            "total_requests": total_requests,
            "services": service_list,
        }

    def _collect_revenue(self) -> Dict:
        """Collect revenue/usage data from UsageTracking."""
        data = self._load_json(USAGE_FILE)
        if not data:
            return {"available": False, "message": "No usage/revenue data found"}

        customers = data.get("customers", {})
        usage_records = data.get("usage", [])

        total_revenue = sum(r.get("cost", 0) for r in usage_records)
        total_customers = len(customers)
        active_customers = sum(1 for c in customers.values() if c.get("status") == "active")

        # Revenue by skill
        revenue_by_skill = {}
        for r in usage_records:
            skill = r.get("skill_id", "unknown")
            revenue_by_skill[skill] = revenue_by_skill.get(skill, 0) + r.get("cost", 0)

        top_revenue = sorted(revenue_by_skill.items(), key=lambda x: x[1], reverse=True)[:5]

        return {
            "available": True,
            "total_revenue": round(total_revenue, 4),
            "total_customers": total_customers,
            "active_customers": active_customers,
            "total_api_calls": len(usage_records),
            "top_revenue_skills": [{"skill": s[0], "revenue": round(s[1], 4)} for s in top_revenue],
        }

    def _collect_fleet(self) -> Dict:
        """Collect fleet/replication health data."""
        data = self._load_json(HEALTH_FILE)
        if not data:
            return {"available": False, "message": "No fleet health data found"}

        agents = data.get("agents", {})
        total = len(agents)
        healthy = sum(1 for a in agents.values() if a.get("status") == "healthy")
        unhealthy = total - healthy

        return {
            "available": True,
            "total_agents": total,
            "healthy": healthy,
            "unhealthy": unhealthy,
            "fleet_health_pct": round(healthy / total * 100, 1) if total > 0 else 100,
        }

    def _collect_skills_inventory(self) -> Dict:
        """Inventory of installed skills from the skills directory."""
        skills_dir = Path(__file__).parent
        skill_files = [f.stem for f in skills_dir.glob("*.py") if f.stem not in ("__init__", "base")]
        return {
            "available": True,
            "total_skills": len(skill_files),
            "skill_list": sorted(skill_files),
        }

    # ── Pillar Scoring ────────────────────────────────────────────────

    def _score_pillars(self) -> Dict[str, Dict]:
        """Score each pillar 0-100 based on available data."""
        perf = self._collect_performance()
        resources = self._collect_resources()
        goals = self._collect_goals()
        services = self._collect_services()
        revenue = self._collect_revenue()
        fleet = self._collect_fleet()
        skills = self._collect_skills_inventory()

        # Self-Improvement: based on performance tracking, success rate, trend
        si_score = 30  # Base: we have the infrastructure
        if perf.get("available") and perf.get("total_actions", 0) > 0:
            si_score += 20  # Actively tracking performance
            rate = perf.get("success_rate", 0)
            si_score += min(30, int(rate * 0.3))  # Up to 30 for high success rate
            if perf.get("trend") == "improving":
                si_score += 20
            elif perf.get("trend") == "stable":
                si_score += 10
        si_score = min(100, si_score)

        # Revenue: based on customers, revenue amount, services
        rev_score = 20  # Base: infrastructure exists
        if revenue.get("available"):
            if revenue.get("total_customers", 0) > 0:
                rev_score += 20
            if revenue.get("total_revenue", 0) > 0:
                rev_score += 20
            if revenue.get("total_api_calls", 0) > 10:
                rev_score += 10
        if services.get("available"):
            if services.get("active_services", 0) > 0:
                rev_score += 15
            if services.get("total_requests", 0) > 0:
                rev_score += 15
        rev_score = min(100, rev_score)

        # Replication: based on fleet health, peer count
        rep_score = 25  # Base: deployment infrastructure exists
        if fleet.get("available"):
            if fleet.get("total_agents", 0) > 1:
                rep_score += 25
            health = fleet.get("fleet_health_pct", 0)
            rep_score += min(25, int(health * 0.25))
            if fleet.get("total_agents", 0) > 3:
                rep_score += 25
        rep_score = min(100, rep_score)

        # Goal Setting: based on active goals, completion rate
        gs_score = 30  # Base: goal management infrastructure
        if goals.get("available") and goals.get("total_goals", 0) > 0:
            gs_score += 15  # Has goals
            active = goals.get("by_status", {}).get("active", 0)
            if active > 0:
                gs_score += 15  # Actively pursuing goals
            completion = goals.get("completion_rate", 0)
            gs_score += min(25, int(completion * 0.25))  # Up to 25 for completion
            if goals.get("total_goals", 0) >= 5:
                gs_score += 15  # Robust goal set
        gs_score = min(100, gs_score)

        return {
            "self_improvement": {"score": si_score, "grade": self._grade(si_score)},
            "revenue": {"score": rev_score, "grade": self._grade(rev_score)},
            "replication": {"score": rep_score, "grade": self._grade(rep_score)},
            "goal_setting": {"score": gs_score, "grade": self._grade(gs_score)},
            "overall": {"score": (si_score + rev_score + rep_score + gs_score) // 4, "grade": self._grade((si_score + rev_score + rep_score + gs_score) // 4)},
        }

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

    # ── Actions ───────────────────────────────────────────────────────

    async def _full_report(self, params: Dict) -> SkillResult:
        """Generate complete dashboard report."""
        timestamp = datetime.utcnow().isoformat()

        report = {
            "timestamp": timestamp,
            "agent": "singularity",
            "sections": {
                "performance": self._collect_performance(),
                "resources": self._collect_resources(),
                "goals": self._collect_goals(),
                "services": self._collect_services(),
                "revenue": self._collect_revenue(),
                "fleet": self._collect_fleet(),
                "skills": self._collect_skills_inventory(),
            },
            "pillar_scores": self._score_pillars(),
        }

        # Save snapshot
        history = self._load_history()
        history["snapshots"].append({
            "timestamp": timestamp,
            "pillar_scores": report["pillar_scores"],
            "summary_stats": {
                "total_skills": report["sections"]["skills"].get("total_skills", 0),
                "success_rate": report["sections"]["performance"].get("success_rate", 0),
            },
        })
        # Trim history
        if len(history["snapshots"]) > MAX_HISTORY:
            history["snapshots"] = history["snapshots"][-MAX_HISTORY:]
        self._save_history(history)

        return SkillResult(
            success=True,
            message=f"Dashboard generated at {timestamp}. Overall score: {report['pillar_scores']['overall']['score']}/100 ({report['pillar_scores']['overall']['grade']})",
            data=report,
        )

    async def _summary(self, params: Dict) -> SkillResult:
        """Quick text summary of agent health."""
        perf = self._collect_performance()
        resources = self._collect_resources()
        goals = self._collect_goals()
        services = self._collect_services()
        skills = self._collect_skills_inventory()
        pillars = self._score_pillars()

        parts = []
        parts.append(f"Agent has {skills.get('total_skills', 0)} skills installed.")

        if perf.get("available") and perf.get("total_actions", 0) > 0:
            parts.append(f"Performance: {perf['success_rate']}% success rate over {perf['total_actions']} actions (trend: {perf.get('trend', 'unknown')}).")
        else:
            parts.append("Performance: No action data recorded yet.")

        if resources.get("available") and resources.get("total_budget", 0) > 0:
            parts.append(f"Budget: {resources['budget_utilization_pct']}% utilized (${resources['remaining']:.2f} remaining).")

        if goals.get("available") and goals.get("total_goals", 0) > 0:
            active = goals.get("by_status", {}).get("active", 0)
            parts.append(f"Goals: {active} active of {goals['total_goals']} total ({goals['completion_rate']}% completed).")

        if services.get("available") and services.get("total_services", 0) > 0:
            parts.append(f"Services: {services['active_services']} active, {services['total_requests']} total requests.")

        overall = pillars["overall"]
        parts.append(f"Overall pillar score: {overall['score']}/100 ({overall['grade']}).")

        # Identify weakest pillar
        pillar_scores = {k: v["score"] for k, v in pillars.items() if k != "overall"}
        weakest = min(pillar_scores, key=pillar_scores.get)
        strongest = max(pillar_scores, key=pillar_scores.get)
        parts.append(f"Strongest: {strongest} ({pillar_scores[strongest]}). Weakest: {weakest} ({pillar_scores[weakest]}) - prioritize this.")

        summary_text = " ".join(parts)
        return SkillResult(
            success=True,
            message=summary_text,
            data={"summary": summary_text, "pillar_scores": pillars},
        )

    async def _pillar_scorecard(self, params: Dict) -> SkillResult:
        """Score each pillar 0-100."""
        scores = self._score_pillars()
        lines = []
        for pillar, info in scores.items():
            bar = "#" * (info["score"] // 5) + "-" * (20 - info["score"] // 5)
            lines.append(f"  {pillar:20s} [{bar}] {info['score']:3d}/100 ({info['grade']})")

        scorecard_text = "Pillar Scorecard:\n" + "\n".join(lines)
        return SkillResult(
            success=True,
            message=scorecard_text,
            data={"scores": scores},
        )

    async def _export_html(self, params: Dict) -> SkillResult:
        """Generate standalone HTML dashboard."""
        output_path = params.get("output_path", str(DATA_DIR / "dashboard.html"))

        perf = self._collect_performance()
        resources = self._collect_resources()
        goals = self._collect_goals()
        services = self._collect_services()
        revenue = self._collect_revenue()
        fleet = self._collect_fleet()
        skills = self._collect_skills_inventory()
        pillars = self._score_pillars()
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

        # Build HTML
        html = self._generate_html(
            timestamp, perf, resources, goals, services, revenue, fleet, skills, pillars
        )

        Path(output_path).write_text(html)
        return SkillResult(
            success=True,
            message=f"HTML dashboard exported to {output_path}",
            data={"path": output_path, "size_bytes": len(html)},
        )

    async def _section_report(self, params: Dict) -> SkillResult:
        """Get a specific section's data."""
        section = params.get("section", "")
        collectors = {
            "performance": self._collect_performance,
            "resources": self._collect_resources,
            "goals": self._collect_goals,
            "services": self._collect_services,
            "revenue": self._collect_revenue,
            "fleet": self._collect_fleet,
            "skills": self._collect_skills_inventory,
        }
        if section not in collectors:
            return SkillResult(
                success=False,
                message=f"Unknown section: {section}. Available: {list(collectors.keys())}",
            )

        data = collectors[section]()
        return SkillResult(
            success=True,
            message=f"Section report: {section}",
            data={section: data},
        )

    # ── HTML Generation ───────────────────────────────────────────────

    def _generate_html(self, timestamp, perf, resources, goals, services, revenue, fleet, skills, pillars):
        """Generate a complete standalone HTML dashboard page."""

        def score_color(score):
            if score >= 80:
                return "#22c55e"
            elif score >= 60:
                return "#eab308"
            elif score >= 40:
                return "#f97316"
            return "#ef4444"

        pillar_cards = ""
        for name, info in pillars.items():
            if name == "overall":
                continue
            color = score_color(info["score"])
            display = name.replace("_", " ").title()
            pillar_cards += f"""
            <div class="card">
                <h3>{display}</h3>
                <div class="score" style="color:{color}">{info['score']}</div>
                <div class="grade">Grade: {info['grade']}</div>
                <div class="bar"><div class="bar-fill" style="width:{info['score']}%;background:{color}"></div></div>
            </div>"""

        overall = pillars["overall"]
        overall_color = score_color(overall["score"])

        # Performance section
        perf_html = "<p>No performance data available.</p>"
        if perf.get("available") and perf.get("total_actions", 0) > 0:
            perf_html = f"""
            <div class="stats-grid">
                <div class="stat"><span class="stat-value">{perf['total_actions']}</span><span class="stat-label">Total Actions</span></div>
                <div class="stat"><span class="stat-value">{perf['success_rate']}%</span><span class="stat-label">Success Rate</span></div>
                <div class="stat"><span class="stat-value">{perf['avg_latency_ms']}ms</span><span class="stat-label">Avg Latency</span></div>
                <div class="stat"><span class="stat-value">{perf['trend']}</span><span class="stat-label">Trend</span></div>
            </div>"""

        # Goals section
        goals_html = "<p>No goal data available.</p>"
        if goals.get("available") and goals.get("total_goals", 0) > 0:
            active = goals.get("by_status", {}).get("active", 0)
            completed = goals.get("by_status", {}).get("completed", 0)
            goals_html = f"""
            <div class="stats-grid">
                <div class="stat"><span class="stat-value">{goals['total_goals']}</span><span class="stat-label">Total Goals</span></div>
                <div class="stat"><span class="stat-value">{active}</span><span class="stat-label">Active</span></div>
                <div class="stat"><span class="stat-value">{completed}</span><span class="stat-label">Completed</span></div>
                <div class="stat"><span class="stat-value">{goals['completion_rate']}%</span><span class="stat-label">Completion Rate</span></div>
            </div>"""

        # Revenue section
        rev_html = "<p>No revenue data available.</p>"
        if revenue.get("available"):
            rev_html = f"""
            <div class="stats-grid">
                <div class="stat"><span class="stat-value">${revenue['total_revenue']:.2f}</span><span class="stat-label">Total Revenue</span></div>
                <div class="stat"><span class="stat-value">{revenue['total_customers']}</span><span class="stat-label">Customers</span></div>
                <div class="stat"><span class="stat-value">{revenue['total_api_calls']}</span><span class="stat-label">API Calls</span></div>
            </div>"""

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Singularity Agent Dashboard</title>
<style>
    * {{ margin:0; padding:0; box-sizing:border-box; }}
    body {{ font-family: -apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif; background:#0f172a; color:#e2e8f0; padding:20px; }}
    .header {{ text-align:center; margin-bottom:30px; }}
    .header h1 {{ font-size:2em; color:#38bdf8; }}
    .header .timestamp {{ color:#94a3b8; margin-top:5px; }}
    .overall {{ text-align:center; margin:20px 0 30px; }}
    .overall .big-score {{ font-size:4em; font-weight:bold; color:{overall_color}; }}
    .overall .label {{ font-size:1.2em; color:#94a3b8; }}
    .pillars {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(200px,1fr)); gap:15px; margin-bottom:30px; }}
    .card {{ background:#1e293b; border-radius:12px; padding:20px; text-align:center; }}
    .card h3 {{ color:#94a3b8; font-size:0.9em; text-transform:uppercase; margin-bottom:10px; }}
    .score {{ font-size:2.5em; font-weight:bold; }}
    .grade {{ color:#94a3b8; margin:5px 0; }}
    .bar {{ background:#334155; border-radius:6px; height:8px; margin-top:10px; overflow:hidden; }}
    .bar-fill {{ height:100%; border-radius:6px; transition:width 0.5s; }}
    .section {{ background:#1e293b; border-radius:12px; padding:20px; margin-bottom:20px; }}
    .section h2 {{ color:#38bdf8; margin-bottom:15px; font-size:1.3em; }}
    .stats-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(140px,1fr)); gap:15px; }}
    .stat {{ text-align:center; }}
    .stat-value {{ display:block; font-size:1.8em; font-weight:bold; color:#f1f5f9; }}
    .stat-label {{ display:block; font-size:0.85em; color:#94a3b8; margin-top:4px; }}
    .footer {{ text-align:center; margin-top:30px; color:#475569; font-size:0.85em; }}
</style>
</head>
<body>
    <div class="header">
        <h1>Singularity Agent Dashboard</h1>
        <div class="timestamp">Generated: {timestamp}</div>
    </div>

    <div class="overall">
        <div class="label">Overall Health Score</div>
        <div class="big-score">{overall['score']}</div>
        <div class="label">Grade: {overall['grade']}</div>
    </div>

    <div class="pillars">{pillar_cards}</div>

    <div class="section">
        <h2>Performance</h2>
        {perf_html}
    </div>

    <div class="section">
        <h2>Goals</h2>
        {goals_html}
    </div>

    <div class="section">
        <h2>Revenue</h2>
        {rev_html}
    </div>

    <div class="section">
        <h2>Skills Inventory</h2>
        <div class="stats-grid">
            <div class="stat"><span class="stat-value">{skills.get('total_skills', 0)}</span><span class="stat-label">Installed Skills</span></div>
        </div>
    </div>

    <div class="footer">
        Singularity Self-Monitoring Dashboard v1.0 | {skills.get('total_skills', 0)} skills active
    </div>
</body>
</html>"""

    def estimate_cost(self, action: str, params: Dict = None) -> float:
        return 0.0
