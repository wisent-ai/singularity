#!/usr/bin/env python3
"""
SessionBootstrap Skill - Unified session startup orchestration.

The missing "main loop coordinator" that connects all existing infrastructure
into a single coherent startup flow. At the start of each session, the agent
needs to:

1. Load performance data from last session
2. Run feedback loop analysis to detect patterns
3. Check strategy status to find weakest pillar
4. Query goal manager for next actionable goal
5. Return a unified "session brief" with exactly what to work on

Without this, the agent has to manually remember which skills to call in what
order and piece together context from 4+ separate systems. SessionBootstrap
reduces "start of session" from ~8 manual skill calls to 1.

Part of the Goal Setting pillar: makes the agent truly autonomous in deciding
what to do each session by orchestrating all strategic infrastructure.
Also serves Self-Improvement: closes the cross-session learning loop.
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from .base import Skill, SkillResult, SkillManifest, SkillAction

# Data files from other skills (read-only access)
DATA_DIR = Path(__file__).parent.parent / "data"
PERF_FILE = DATA_DIR / "performance.json"
FEEDBACK_FILE = DATA_DIR / "feedback_loop.json"
STRATEGY_FILE = DATA_DIR / "strategy.json"
GOALS_FILE = DATA_DIR / "goals.json"
SESSION_HISTORY_FILE = DATA_DIR / "session_history.json"

MAX_HISTORY = 100


class SessionBootstrapSkill(Skill):
    """
    Unified session startup orchestrator.

    Reads from PerformanceTracker, FeedbackLoop, Strategy, and GoalManager
    data files to produce a single comprehensive session brief. This is the
    "boot sequence" that tells the agent exactly what happened, what changed,
    and what to work on next.

    Actions:
    - boot: Full session bootstrap (performance + feedback + strategy + goals)
    - quick: Fast bootstrap (just goals + strategy, skip heavy analysis)
    - recap: Get a recap of the last N sessions without starting a new one
    - log_outcome: Record what actually happened this session for next time
    """

    def __init__(self, credentials: Dict[str, str] = None):
        super().__init__(credentials)
        self._ensure_data()

    def _ensure_data(self):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        if not SESSION_HISTORY_FILE.exists():
            self._save_history({
                "sessions": [],
                "created_at": datetime.now().isoformat(),
            })

    def _load_json(self, path: Path) -> Optional[Dict]:
        """Safely load a JSON file, returning None if missing/corrupt."""
        try:
            with open(path, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return None

    def _save_history(self, data: Dict):
        if len(data.get("sessions", [])) > MAX_HISTORY:
            data["sessions"] = data["sessions"][-MAX_HISTORY:]
        data["last_updated"] = datetime.now().isoformat()
        SESSION_HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(SESSION_HISTORY_FILE, "w") as f:
            json.dump(data, f, indent=2, default=str)

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="session",
            name="Session Bootstrap",
            version="1.0.0",
            category="meta",
            description="Unified session startup - orchestrates performance, feedback, strategy, and goals into one brief",
            actions=[
                SkillAction(
                    name="boot",
                    description="Full session bootstrap: analyze last session, detect patterns, find weakest pillar, recommend next work",
                    parameters={
                        "session_focus": {
                            "type": "string",
                            "required": False,
                            "description": "Optional focus area for this session",
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="quick",
                    description="Quick bootstrap: just goals and strategy, skip heavy analysis",
                    parameters={},
                    estimated_cost=0,
                ),
                SkillAction(
                    name="recap",
                    description="Get a recap of the last N sessions",
                    parameters={
                        "count": {
                            "type": "number",
                            "required": False,
                            "description": "Number of sessions to recap (default: 3)",
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="log_outcome",
                    description="Log what happened this session for next session's bootstrap",
                    parameters={
                        "summary": {
                            "type": "string",
                            "required": True,
                            "description": "What was accomplished this session",
                        },
                        "pillar": {
                            "type": "string",
                            "required": False,
                            "description": "Which pillar was served",
                        },
                        "blockers": {
                            "type": "array",
                            "required": False,
                            "description": "Any blockers encountered",
                        },
                        "next_steps": {
                            "type": "array",
                            "required": False,
                            "description": "Recommended next steps for next session",
                        },
                    },
                    estimated_cost=0,
                ),
            ],
            required_credentials=[],
        )

    def check_credentials(self) -> bool:
        return True

    async def execute(self, action: str, params: Dict) -> SkillResult:
        handlers = {
            "boot": self._boot,
            "quick": self._quick,
            "recap": self._recap,
            "log_outcome": self._log_outcome,
        }
        handler = handlers.get(action)
        if handler:
            return handler(params)
        return SkillResult(success=False, message=f"Unknown action: {action}")

    def _boot(self, params: Dict) -> SkillResult:
        """Full session bootstrap with all subsystems."""
        session_focus = params.get("session_focus", "")
        now = datetime.now()

        brief = {
            "session_started_at": now.isoformat(),
            "session_focus": session_focus or "auto-detect",
        }

        # 1. Performance summary from last session
        perf_summary = self._analyze_performance()
        brief["performance"] = perf_summary

        # 2. Feedback loop patterns
        feedback_summary = self._analyze_feedback()
        brief["feedback"] = feedback_summary

        # 3. Strategy status - pillar scores and weakest
        strategy_summary = self._analyze_strategy()
        brief["strategy"] = strategy_summary

        # 4. Goal status - what to work on
        goals_summary = self._analyze_goals()
        brief["goals"] = goals_summary

        # 5. Last session recap
        last_session = self._get_last_session()
        brief["last_session"] = last_session

        # 6. Generate unified recommendation
        recommendation = self._generate_recommendation(
            perf_summary, feedback_summary, strategy_summary, goals_summary,
            last_session, session_focus,
        )
        brief["recommendation"] = recommendation

        # 7. Record this session start
        history = self._load_json(SESSION_HISTORY_FILE) or {"sessions": []}
        session_num = len(history.get("sessions", [])) + 1
        history.setdefault("sessions", []).append({
            "number": session_num,
            "started_at": now.isoformat(),
            "focus": session_focus or recommendation.get("focus_pillar", "auto"),
            "recommendation": recommendation.get("action", ""),
            "completed": False,
            "outcome": None,
        })
        self._save_history(history)
        brief["session_number"] = session_num

        # Build human-readable message
        msg_lines = [f"=== Session #{session_num} Boot ==="]

        if last_session:
            msg_lines.append(f"Last session: {last_session.get('summary', 'no record')}")

        if perf_summary.get("available"):
            msg_lines.append(
                f"Performance: {perf_summary.get('recent_success_rate', 0):.0%} success rate, "
                f"${perf_summary.get('recent_total_cost', 0):.4f} cost "
                f"({perf_summary.get('recent_action_count', 0)} actions)"
            )

        if strategy_summary.get("available"):
            msg_lines.append(
                f"Strategy: weakest={strategy_summary.get('weakest_pillar', '?')} "
                f"({strategy_summary.get('weakest_score', 0):.0f}/100), "
                f"avg={strategy_summary.get('average_score', 0):.0f}/100"
            )

        if goals_summary.get("available"):
            msg_lines.append(
                f"Goals: {goals_summary.get('active_count', 0)} active, "
                f"next='{goals_summary.get('next_goal_title', 'none')}'"
            )

        if feedback_summary.get("available"):
            active = feedback_summary.get("active_adaptations", 0)
            if active > 0:
                msg_lines.append(f"Feedback: {active} active adaptations")

        msg_lines.append(f">>> Recommendation: {recommendation.get('action', 'No recommendation')}")
        msg_lines.append(f">>> Focus: {recommendation.get('focus_pillar', 'auto')}")

        return SkillResult(
            success=True,
            message="\n".join(msg_lines),
            data=brief,
        )

    def _quick(self, params: Dict) -> SkillResult:
        """Quick bootstrap - just strategy + goals."""
        now = datetime.now()

        strategy_summary = self._analyze_strategy()
        goals_summary = self._analyze_goals()
        last_session = self._get_last_session()

        recommendation = self._generate_recommendation(
            {}, {}, strategy_summary, goals_summary, last_session, "",
        )

        brief = {
            "session_started_at": now.isoformat(),
            "strategy": strategy_summary,
            "goals": goals_summary,
            "last_session": last_session,
            "recommendation": recommendation,
        }

        msg_lines = ["=== Quick Boot ==="]
        if goals_summary.get("next_goal_title"):
            msg_lines.append(f"Next goal: {goals_summary['next_goal_title']}")
        if strategy_summary.get("weakest_pillar"):
            msg_lines.append(f"Weakest pillar: {strategy_summary['weakest_pillar']}")
        msg_lines.append(f">>> {recommendation.get('action', 'No recommendation')}")

        return SkillResult(
            success=True,
            message="\n".join(msg_lines),
            data=brief,
        )

    def _recap(self, params: Dict) -> SkillResult:
        """Get a recap of recent sessions."""
        count = int(params.get("count", 3))
        count = max(1, min(20, count))

        history = self._load_json(SESSION_HISTORY_FILE) or {"sessions": []}
        sessions = history.get("sessions", [])[-count:]

        if not sessions:
            return SkillResult(
                success=True,
                message="No session history available.",
                data={"sessions": [], "count": 0},
            )

        msg_lines = [f"=== Last {len(sessions)} Sessions ==="]
        for s in reversed(sessions):
            status = "completed" if s.get("completed") else "incomplete"
            outcome = s.get("outcome", {})
            summary = ""
            if isinstance(outcome, dict):
                summary = outcome.get("summary", "")
            elif isinstance(outcome, str):
                summary = outcome
            msg_lines.append(
                f"  #{s.get('number', '?')} [{status}] "
                f"focus={s.get('focus', '?')} "
                f"| {summary or 'no outcome logged'}"
            )

        return SkillResult(
            success=True,
            message="\n".join(msg_lines),
            data={"sessions": sessions, "count": len(sessions)},
        )

    def _log_outcome(self, params: Dict) -> SkillResult:
        """Log what happened this session."""
        summary = params.get("summary", "").strip()
        if not summary:
            return SkillResult(success=False, message="summary is required")

        pillar = params.get("pillar", "")
        blockers = params.get("blockers", [])
        next_steps = params.get("next_steps", [])

        if isinstance(blockers, str):
            blockers = [blockers]
        if isinstance(next_steps, str):
            next_steps = [next_steps]

        history = self._load_json(SESSION_HISTORY_FILE) or {"sessions": []}
        sessions = history.get("sessions", [])

        if not sessions:
            return SkillResult(
                success=False,
                message="No active session to log. Run session:boot first.",
            )

        # Update the most recent session
        latest = sessions[-1]
        latest["completed"] = True
        latest["completed_at"] = datetime.now().isoformat()
        latest["outcome"] = {
            "summary": summary,
            "pillar": pillar,
            "blockers": blockers,
            "next_steps": next_steps,
        }

        self._save_history(history)

        return SkillResult(
            success=True,
            message=f"Session #{latest.get('number', '?')} outcome logged: {summary}",
            data={
                "session_number": latest.get("number"),
                "summary": summary,
                "pillar": pillar,
                "next_steps": next_steps,
            },
        )

    # ---------------------------------------------------------------
    # Internal analysis methods (read-only from other skills' data)
    # ---------------------------------------------------------------

    def _analyze_performance(self) -> Dict:
        """Read performance data and extract key metrics."""
        data = self._load_json(PERF_FILE)
        if not data or not data.get("records"):
            return {"available": False, "reason": "no performance data"}

        records = data["records"]
        now = datetime.now()

        # Last 24 hours of records
        cutoff = (now - timedelta(hours=24)).isoformat()
        recent = [r for r in records if r.get("timestamp", "") >= cutoff]

        if not recent:
            # Fall back to last 50 records
            recent = records[-50:]

        total = len(recent)
        successes = sum(1 for r in recent if r.get("success", False))
        total_cost = sum(r.get("cost", 0) for r in recent)
        total_revenue = sum(r.get("revenue", 0) for r in recent)

        # Per-skill breakdown
        skill_stats = {}
        for r in recent:
            skill = r.get("skill_id", "unknown")
            if skill not in skill_stats:
                skill_stats[skill] = {"total": 0, "success": 0, "cost": 0}
            skill_stats[skill]["total"] += 1
            if r.get("success"):
                skill_stats[skill]["success"] += 1
            skill_stats[skill]["cost"] += r.get("cost", 0)

        # Find worst-performing skill
        worst_skill = None
        worst_rate = 1.0
        for skill, stats in skill_stats.items():
            if stats["total"] >= 2:  # Need at least 2 samples
                rate = stats["success"] / stats["total"]
                if rate < worst_rate:
                    worst_rate = rate
                    worst_skill = skill

        return {
            "available": True,
            "recent_action_count": total,
            "recent_success_rate": successes / total if total > 0 else 0,
            "recent_total_cost": round(total_cost, 6),
            "recent_total_revenue": round(total_revenue, 6),
            "skill_count": len(skill_stats),
            "worst_skill": worst_skill,
            "worst_skill_rate": round(worst_rate, 3) if worst_skill else None,
            "top_skills": sorted(
                skill_stats.items(),
                key=lambda x: x[1]["total"],
                reverse=True,
            )[:5],
        }

    def _analyze_feedback(self) -> Dict:
        """Read feedback loop data for active adaptations."""
        data = self._load_json(FEEDBACK_FILE)
        if not data:
            return {"available": False, "reason": "no feedback data"}

        adaptations = data.get("adaptations", [])
        reviews = data.get("reviews", [])

        active_adaptations = [a for a in adaptations if a.get("status") == "active"]
        successful = [a for a in adaptations if a.get("status") == "successful"]
        reverted = [a for a in adaptations if a.get("status") == "reverted"]

        # Latest review summary
        latest_review = reviews[-1] if reviews else None
        latest_patterns = []
        if latest_review and isinstance(latest_review, dict):
            latest_patterns = latest_review.get("patterns", [])

        return {
            "available": True,
            "active_adaptations": len(active_adaptations),
            "successful_adaptations": len(successful),
            "reverted_adaptations": len(reverted),
            "total_reviews": len(reviews),
            "latest_patterns": latest_patterns[:5],
            "adaptation_details": [
                {
                    "type": a.get("type", ""),
                    "description": a.get("description", ""),
                    "applied_at": a.get("applied_at", ""),
                }
                for a in active_adaptations[:5]
            ],
        }

    def _analyze_strategy(self) -> Dict:
        """Read strategy data for pillar scores and gaps."""
        data = self._load_json(STRATEGY_FILE)
        if not data or not data.get("pillars"):
            return {"available": False, "reason": "no strategy data"}

        pillars = data["pillars"]
        scores = {name: info.get("score", 0) for name, info in pillars.items()}

        if not any(scores.values()):
            return {
                "available": False,
                "reason": "no pillar assessments yet",
                "scores": scores,
            }

        weakest = min(scores, key=scores.get)
        strongest = max(scores, key=scores.get)
        avg = sum(scores.values()) / len(scores) if scores else 0

        # Collect top gaps from weakest pillar
        weakest_gaps = pillars[weakest].get("gaps", [])[:5]

        # Recommendations if available
        recommendations = data.get("recommendations", [])

        return {
            "available": True,
            "scores": scores,
            "weakest_pillar": weakest,
            "weakest_score": scores[weakest],
            "strongest_pillar": strongest,
            "strongest_score": scores[strongest],
            "average_score": round(avg, 1),
            "weakest_gaps": weakest_gaps,
            "top_recommendation": recommendations[0] if recommendations else None,
            "session_count": data.get("session_count", 0),
        }

    def _analyze_goals(self) -> Dict:
        """Read goal manager data for active goals and next action."""
        data = self._load_json(GOALS_FILE)
        if not data:
            return {"available": False, "reason": "no goals data"}

        goals = data.get("goals", [])
        completed = data.get("completed_goals", [])

        active = [g for g in goals if g.get("status") == "active"]
        if not active:
            return {
                "available": True,
                "active_count": 0,
                "completed_count": len(completed),
                "next_goal_title": None,
                "next_goal_id": None,
            }

        # Find highest priority actionable goal (same logic as GoalManager._next)
        completed_ids = {g["id"] for g in completed}
        completed_ids.update(g["id"] for g in goals if g.get("status") == "completed")

        best = None
        best_score = -1
        for g in active:
            deps = g.get("depends_on", [])
            unmet = [d for d in deps if d not in completed_ids]
            if unmet:
                continue
            score = g.get("priority_score", 2)
            if score > best_score:
                best_score = score
                best = g

        if not best:
            return {
                "available": True,
                "active_count": len(active),
                "completed_count": len(completed),
                "next_goal_title": None,
                "next_goal_id": None,
                "all_blocked": True,
            }

        # Find next milestone
        next_milestone = None
        milestones = best.get("milestones", [])
        for m in milestones:
            if not m.get("completed"):
                next_milestone = m.get("title", "")
                break

        done_ms = sum(1 for m in milestones if m.get("completed"))

        # Per-pillar active goal counts
        pillar_counts = {}
        for g in active:
            p = g.get("pillar", "other")
            pillar_counts[p] = pillar_counts.get(p, 0) + 1

        return {
            "available": True,
            "active_count": len(active),
            "completed_count": len(completed),
            "next_goal_id": best["id"],
            "next_goal_title": best["title"],
            "next_goal_pillar": best.get("pillar", ""),
            "next_goal_priority": best.get("priority", ""),
            "next_milestone": next_milestone,
            "milestones_done": done_ms,
            "milestones_total": len(milestones),
            "pillar_goal_counts": pillar_counts,
        }

    def _get_last_session(self) -> Optional[Dict]:
        """Get the outcome of the last completed session."""
        history = self._load_json(SESSION_HISTORY_FILE) or {"sessions": []}
        sessions = history.get("sessions", [])

        # Find last completed session
        for s in reversed(sessions):
            if s.get("completed"):
                outcome = s.get("outcome", {})
                return {
                    "number": s.get("number"),
                    "focus": s.get("focus", ""),
                    "summary": outcome.get("summary", "") if isinstance(outcome, dict) else str(outcome),
                    "pillar": outcome.get("pillar", "") if isinstance(outcome, dict) else "",
                    "blockers": outcome.get("blockers", []) if isinstance(outcome, dict) else [],
                    "next_steps": outcome.get("next_steps", []) if isinstance(outcome, dict) else [],
                    "started_at": s.get("started_at", ""),
                    "completed_at": s.get("completed_at", ""),
                }

        return None

    def _generate_recommendation(
        self,
        perf: Dict,
        feedback: Dict,
        strategy: Dict,
        goals: Dict,
        last_session: Optional[Dict],
        session_focus: str,
    ) -> Dict:
        """Generate a unified recommendation for what to work on."""

        # Priority 1: If user specified a focus, honor it
        if session_focus:
            return {
                "source": "user_focus",
                "focus_pillar": session_focus,
                "action": f"User-directed focus: {session_focus}",
                "reasoning": "User explicitly requested this focus area",
            }

        # Priority 2: If last session had next_steps, continue those
        if last_session and last_session.get("next_steps"):
            next_steps = last_session["next_steps"]
            return {
                "source": "last_session_continuity",
                "focus_pillar": last_session.get("pillar", "auto"),
                "action": next_steps[0] if next_steps else "Continue from last session",
                "reasoning": f"Continuing from session #{last_session.get('number', '?')}",
                "all_next_steps": next_steps,
            }

        # Priority 3: If there's an actionable goal, work on it
        if goals.get("available") and goals.get("next_goal_title"):
            return {
                "source": "goal_manager",
                "focus_pillar": goals.get("next_goal_pillar", "auto"),
                "action": f"Work on goal: {goals['next_goal_title']}",
                "goal_id": goals.get("next_goal_id"),
                "next_milestone": goals.get("next_milestone"),
                "reasoning": f"Highest-priority actionable goal ({goals.get('next_goal_priority', 'medium')} priority)",
            }

        # Priority 4: If strategy identifies a weak pillar, focus there
        if strategy.get("available") and strategy.get("weakest_pillar"):
            weakest = strategy["weakest_pillar"]
            gaps = strategy.get("weakest_gaps", [])
            action = gaps[0] if gaps else f"Improve {weakest} pillar"
            return {
                "source": "strategy_gap",
                "focus_pillar": weakest,
                "action": action,
                "reasoning": f"Weakest pillar ({weakest}: {strategy.get('weakest_score', 0)}/100)",
                "gaps": gaps,
            }

        # Priority 5: If performance shows a failing skill, fix it
        if perf.get("available") and perf.get("worst_skill"):
            return {
                "source": "performance_issue",
                "focus_pillar": "self_improvement",
                "action": f"Investigate failing skill: {perf['worst_skill']} ({perf.get('worst_skill_rate', 0):.0%} success)",
                "reasoning": "Performance data shows a skill with low success rate",
            }

        # Fallback: No data available, suggest initial setup
        return {
            "source": "cold_start",
            "focus_pillar": "goal_setting",
            "action": "Initialize strategic infrastructure: assess pillars with strategy:assess, create goals with goals:create",
            "reasoning": "No strategic data available - agent needs initial setup",
        }
