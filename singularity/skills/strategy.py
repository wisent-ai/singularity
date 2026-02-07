#!/usr/bin/env python3
"""
Strategy Skill - Autonomous strategic reasoning and work prioritization.

Provides the meta-cognitive layer that decides WHAT to work on, not just HOW.
This is the missing piece between having capabilities and using them wisely.

The agent can:
- Assess its current state across all four pillars
- Identify the highest-gap pillar and recommend focus areas
- Evaluate ROI of past work to inform future decisions
- Maintain a strategic journal across sessions
- Convert strategic insights into actionable recommendations
"""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from enum import Enum
from .base import Skill, SkillResult, SkillManifest, SkillAction


STRATEGY_FILE = Path(__file__).parent.parent / "data" / "strategy.json"


class Pillar(str, Enum):
    SELF_IMPROVEMENT = "self_improvement"
    REVENUE = "revenue"
    REPLICATION = "replication"
    GOAL_SETTING = "goal_setting"


PILLAR_DESCRIPTIONS = {
    Pillar.SELF_IMPROVEMENT: "Can the agent modify its own prompts and code, evaluate performance, learn from past sessions, and maintain act->measure->adapt feedback loops?",
    Pillar.REVENUE: "Can the agent offer services, accept payments, track earnings vs costs, and find/pursue revenue opportunities autonomously?",
    Pillar.REPLICATION: "Can the agent spawn new instances, coordinate between them, decide when/why to replicate, and manage replica resources?",
    Pillar.GOAL_SETTING: "Can the agent evaluate what's most impactful, plan beyond TODOs, set milestones, track progress, and reason about long-term strategy?",
}


class StrategySkill(Skill):
    """
    Autonomous strategic reasoning and work prioritization.

    This skill enables the agent to reason at the meta-level about
    which pillar to focus on, what work has the highest leverage,
    and how to allocate limited resources (time, money, compute)
    for maximum impact on the four pillars of autonomy.
    """

    def __init__(self, credentials: Dict[str, str] = None):
        super().__init__(credentials)
        self._ensure_data()

    def _ensure_data(self):
        STRATEGY_FILE.parent.mkdir(parents=True, exist_ok=True)
        if not STRATEGY_FILE.exists():
            self._save(self._default_state())

    def _default_state(self) -> Dict:
        return {
            "pillars": {
                p.value: {
                    "name": p.value.replace("_", " ").title(),
                    "description": PILLAR_DESCRIPTIONS[p],
                    "score": 0.0,  # 0-100 maturity score
                    "capabilities": [],  # What's been built
                    "gaps": [],  # What's missing
                    "last_assessed": None,
                }
                for p in Pillar
            },
            "journal": [],  # Strategic decisions and their outcomes
            "work_log": [],  # What was built and its measured impact
            "recommendations": [],  # Current action recommendations
            "session_count": 0,
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
        }

    def _load(self) -> Dict:
        try:
            with open(STRATEGY_FILE, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return self._default_state()

    def _save(self, data: Dict):
        data["last_updated"] = datetime.now().isoformat()
        STRATEGY_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(STRATEGY_FILE, "w") as f:
            json.dump(data, f, indent=2)

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="strategy",
            name="Strategy",
            version="1.0.0",
            category="meta-cognition",
            description="Autonomous strategic reasoning - decides WHAT to work on across the four pillars",
            actions=[
                SkillAction(
                    name="assess",
                    description="Assess the current maturity of a pillar by listing capabilities and gaps",
                    parameters={
                        "pillar": {
                            "type": "string",
                            "required": True,
                            "description": "Pillar to assess: self_improvement, revenue, replication, goal_setting",
                        },
                        "score": {
                            "type": "number",
                            "required": True,
                            "description": "Maturity score 0-100 based on assessment",
                        },
                        "capabilities": {
                            "type": "array",
                            "required": True,
                            "description": "List of capabilities already built for this pillar",
                        },
                        "gaps": {
                            "type": "array",
                            "required": True,
                            "description": "List of missing capabilities / gaps",
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="diagnose",
                    description="Get a strategic diagnosis: which pillar has the biggest gap and what to focus on",
                    parameters={},
                    estimated_cost=0,
                ),
                SkillAction(
                    name="recommend",
                    description="Generate prioritized action recommendations based on current pillar assessments",
                    parameters={
                        "count": {
                            "type": "number",
                            "required": False,
                            "description": "Number of recommendations to generate (default: 3)",
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="log_work",
                    description="Log completed work and its impact on a pillar",
                    parameters={
                        "pillar": {
                            "type": "string",
                            "required": True,
                            "description": "Which pillar this work serves",
                        },
                        "description": {
                            "type": "string",
                            "required": True,
                            "description": "What was built or accomplished",
                        },
                        "impact": {
                            "type": "string",
                            "required": True,
                            "description": "Measured or estimated impact (high/medium/low)",
                        },
                        "score_delta": {
                            "type": "number",
                            "required": False,
                            "description": "How much to adjust the pillar score (+/- points)",
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="journal",
                    description="Record a strategic decision or insight for future sessions",
                    parameters={
                        "entry": {
                            "type": "string",
                            "required": True,
                            "description": "The strategic insight, decision, or lesson learned",
                        },
                        "category": {
                            "type": "string",
                            "required": False,
                            "description": "Category: decision, insight, lesson, pivot (default: insight)",
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="review",
                    description="Review the strategic journal and work log to understand past decisions",
                    parameters={
                        "limit": {
                            "type": "number",
                            "required": False,
                            "description": "Number of entries to return (default: 10)",
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="status",
                    description="Get a full strategic status overview across all pillars",
                    parameters={},
                    estimated_cost=0,
                ),
                SkillAction(
                    name="start_session",
                    description="Mark the start of a new work session, incrementing the session counter",
                    parameters={
                        "focus": {
                            "type": "string",
                            "required": False,
                            "description": "What this session will focus on",
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
            "assess": self._assess,
            "diagnose": self._diagnose,
            "recommend": self._recommend,
            "log_work": self._log_work,
            "journal": self._journal,
            "review": self._review,
            "status": self._status,
            "start_session": self._start_session,
        }
        handler = handlers.get(action)
        if handler:
            return await handler(params)
        return SkillResult(success=False, message=f"Unknown action: {action}")

    async def _assess(self, params: Dict) -> SkillResult:
        """Assess a pillar's maturity."""
        pillar_str = params.get("pillar", "").strip()
        score = params.get("score", 0)
        capabilities = params.get("capabilities", [])
        gaps = params.get("gaps", [])

        # Validate pillar
        try:
            pillar = Pillar(pillar_str)
        except ValueError:
            valid = [p.value for p in Pillar]
            return SkillResult(
                success=False,
                message=f"Invalid pillar '{pillar_str}'. Valid: {valid}",
            )

        # Validate score
        score = max(0, min(100, float(score)))

        # Ensure capabilities and gaps are lists of strings
        if isinstance(capabilities, str):
            capabilities = [capabilities]
        if isinstance(gaps, str):
            gaps = [gaps]

        data = self._load()
        pillar_data = data["pillars"][pillar.value]

        old_score = pillar_data["score"]
        pillar_data["score"] = score
        pillar_data["capabilities"] = capabilities
        pillar_data["gaps"] = gaps
        pillar_data["last_assessed"] = datetime.now().isoformat()

        self._save(data)

        direction = "up" if score > old_score else ("down" if score < old_score else "unchanged")
        return SkillResult(
            success=True,
            message=f"Assessed {pillar.value}: score {old_score} -> {score} ({direction}). "
                    f"{len(capabilities)} capabilities, {len(gaps)} gaps identified.",
            data={
                "pillar": pillar.value,
                "old_score": old_score,
                "new_score": score,
                "capabilities_count": len(capabilities),
                "gaps_count": len(gaps),
                "direction": direction,
            },
        )

    async def _diagnose(self, params: Dict) -> SkillResult:
        """Identify the biggest gap and recommend focus."""
        data = self._load()
        pillars = data["pillars"]

        # Find the pillar with the lowest score
        pillar_scores = {
            name: info["score"] for name, info in pillars.items()
        }

        if not any(pillar_scores.values()):
            return SkillResult(
                success=True,
                message="No assessments yet. Use strategy:assess to evaluate each pillar first.",
                data={"assessed": False, "pillars": pillar_scores},
            )

        weakest = min(pillar_scores, key=pillar_scores.get)
        strongest = max(pillar_scores, key=pillar_scores.get)

        avg_score = sum(pillar_scores.values()) / len(pillar_scores)
        gap = pillar_scores[strongest] - pillar_scores[weakest]

        weakest_info = pillars[weakest]
        diagnosis = {
            "weakest_pillar": weakest,
            "weakest_score": pillar_scores[weakest],
            "strongest_pillar": strongest,
            "strongest_score": pillar_scores[strongest],
            "average_score": round(avg_score, 1),
            "score_gap": round(gap, 1),
            "weakest_gaps": weakest_info.get("gaps", []),
            "all_scores": pillar_scores,
        }

        # Build recommendation message
        msg_parts = [
            f"Strategic diagnosis: weakest pillar is '{weakest}' (score: {pillar_scores[weakest]}).",
            f"Strongest: '{strongest}' (score: {pillar_scores[strongest]}).",
            f"Average maturity: {avg_score:.0f}/100. Gap between strongest and weakest: {gap:.0f} points.",
        ]
        if weakest_info.get("gaps"):
            msg_parts.append(f"Top gaps in {weakest}: {', '.join(weakest_info['gaps'][:3])}")

        return SkillResult(
            success=True,
            message=" ".join(msg_parts),
            data=diagnosis,
        )

    async def _recommend(self, params: Dict) -> SkillResult:
        """Generate prioritized action recommendations."""
        count = int(params.get("count", 3))
        count = max(1, min(10, count))

        data = self._load()
        pillars = data["pillars"]

        # Score each pillar and collect gaps
        scored_gaps = []
        for name, info in pillars.items():
            pillar_score = info.get("score", 0)
            # Lower score = higher priority (inverse weighting)
            priority_weight = max(1, 100 - pillar_score)
            for gap in info.get("gaps", []):
                scored_gaps.append({
                    "pillar": name,
                    "gap": gap,
                    "priority_weight": priority_weight,
                    "pillar_score": pillar_score,
                })

        # Sort by priority weight (highest first = lowest pillar score)
        scored_gaps.sort(key=lambda x: x["priority_weight"], reverse=True)

        recommendations = []
        for i, sg in enumerate(scored_gaps[:count]):
            recommendations.append({
                "rank": i + 1,
                "pillar": sg["pillar"],
                "action": sg["gap"],
                "priority": "critical" if sg["pillar_score"] < 20 else
                           "high" if sg["pillar_score"] < 40 else
                           "medium" if sg["pillar_score"] < 60 else "low",
                "pillar_score": sg["pillar_score"],
            })

        # Save recommendations
        data["recommendations"] = recommendations
        self._save(data)

        if not recommendations:
            return SkillResult(
                success=True,
                message="No gaps identified. Use strategy:assess to evaluate pillars first.",
                data={"recommendations": []},
            )

        msg_lines = [f"Top {len(recommendations)} recommendations:"]
        for r in recommendations:
            msg_lines.append(f"  #{r['rank']} [{r['priority'].upper()}] ({r['pillar']}): {r['action']}")

        return SkillResult(
            success=True,
            message="\n".join(msg_lines),
            data={"recommendations": recommendations, "count": len(recommendations)},
        )

    async def _log_work(self, params: Dict) -> SkillResult:
        """Log completed work and its impact."""
        pillar_str = params.get("pillar", "").strip()
        description = params.get("description", "").strip()
        impact = params.get("impact", "medium").strip().lower()
        score_delta = params.get("score_delta", 0)

        if not pillar_str or not description:
            return SkillResult(success=False, message="pillar and description are required")

        try:
            pillar = Pillar(pillar_str)
        except ValueError:
            valid = [p.value for p in Pillar]
            return SkillResult(
                success=False,
                message=f"Invalid pillar '{pillar_str}'. Valid: {valid}",
            )

        if impact not in ("high", "medium", "low"):
            impact = "medium"

        data = self._load()

        # Record the work
        entry = {
            "id": uuid.uuid4().hex[:8],
            "pillar": pillar.value,
            "description": description,
            "impact": impact,
            "score_delta": score_delta,
            "logged_at": datetime.now().isoformat(),
            "session": data.get("session_count", 0),
        }
        data["work_log"].append(entry)

        # Keep last 100 entries
        data["work_log"] = data["work_log"][-100:]

        # Adjust pillar score if delta provided
        if score_delta:
            old_score = data["pillars"][pillar.value]["score"]
            new_score = max(0, min(100, old_score + score_delta))
            data["pillars"][pillar.value]["score"] = new_score
            entry["score_change"] = f"{old_score} -> {new_score}"

        self._save(data)

        msg = f"Logged work on {pillar.value}: '{description}' (impact: {impact})"
        if score_delta:
            msg += f". Score adjusted by {score_delta:+.0f}"

        return SkillResult(
            success=True,
            message=msg,
            data=entry,
        )

    async def _journal(self, params: Dict) -> SkillResult:
        """Record a strategic insight or decision."""
        entry_text = params.get("entry", "").strip()
        category = params.get("category", "insight").strip().lower()

        if not entry_text:
            return SkillResult(success=False, message="entry is required")

        valid_categories = ("decision", "insight", "lesson", "pivot")
        if category not in valid_categories:
            category = "insight"

        data = self._load()

        entry = {
            "id": uuid.uuid4().hex[:8],
            "category": category,
            "entry": entry_text,
            "recorded_at": datetime.now().isoformat(),
            "session": data.get("session_count", 0),
        }
        data["journal"].append(entry)

        # Keep last 200 journal entries
        data["journal"] = data["journal"][-200:]
        self._save(data)

        return SkillResult(
            success=True,
            message=f"Strategic {category} recorded: '{entry_text[:100]}...' " if len(entry_text) > 100 else f"Strategic {category} recorded.",
            data=entry,
        )

    async def _review(self, params: Dict) -> SkillResult:
        """Review the strategic journal and work log."""
        limit = int(params.get("limit", 10))
        limit = max(1, min(50, limit))

        data = self._load()

        journal_entries = data.get("journal", [])[-limit:]
        work_entries = data.get("work_log", [])[-limit:]
        recommendations = data.get("recommendations", [])

        # Calculate work distribution
        work_by_pillar = {}
        for w in data.get("work_log", []):
            p = w.get("pillar", "unknown")
            work_by_pillar[p] = work_by_pillar.get(p, 0) + 1

        # Calculate impact distribution
        impact_counts = {"high": 0, "medium": 0, "low": 0}
        for w in data.get("work_log", []):
            imp = w.get("impact", "medium")
            impact_counts[imp] = impact_counts.get(imp, 0) + 1

        return SkillResult(
            success=True,
            message=f"Review: {len(journal_entries)} journal entries, {len(work_entries)} work entries.",
            data={
                "journal": journal_entries,
                "work_log": work_entries,
                "recommendations": recommendations,
                "work_distribution": work_by_pillar,
                "impact_distribution": impact_counts,
                "total_sessions": data.get("session_count", 0),
            },
        )

    async def _status(self, params: Dict) -> SkillResult:
        """Full strategic status overview."""
        data = self._load()

        pillar_summary = {}
        for name, info in data["pillars"].items():
            pillar_summary[name] = {
                "score": info["score"],
                "capabilities_count": len(info.get("capabilities", [])),
                "gaps_count": len(info.get("gaps", [])),
                "last_assessed": info.get("last_assessed"),
                "top_gaps": info.get("gaps", [])[:3],
            }

        scores = [info["score"] for info in data["pillars"].values()]
        avg_score = sum(scores) / len(scores) if scores else 0
        min_score = min(scores) if scores else 0
        max_score = max(scores) if scores else 0

        total_work = len(data.get("work_log", []))
        total_journal = len(data.get("journal", []))
        sessions = data.get("session_count", 0)

        msg_lines = [
            f"Strategic Status (Session #{sessions}):",
            f"  Overall maturity: {avg_score:.0f}/100 (min: {min_score:.0f}, max: {max_score:.0f})",
        ]
        for name, info in pillar_summary.items():
            bar = "█" * int(info["score"] / 10) + "░" * (10 - int(info["score"] / 10))
            msg_lines.append(f"  {name}: [{bar}] {info['score']:.0f}/100 ({info['gaps_count']} gaps)")

        msg_lines.append(f"  Work logged: {total_work} items | Journal: {total_journal} entries")

        return SkillResult(
            success=True,
            message="\n".join(msg_lines),
            data={
                "pillars": pillar_summary,
                "average_score": round(avg_score, 1),
                "min_score": min_score,
                "max_score": max_score,
                "total_work_items": total_work,
                "total_journal_entries": total_journal,
                "session_count": sessions,
                "recommendations": data.get("recommendations", []),
            },
        )

    async def _start_session(self, params: Dict) -> SkillResult:
        """Mark the start of a new work session."""
        focus = params.get("focus", "").strip()

        data = self._load()
        data["session_count"] = data.get("session_count", 0) + 1
        session_num = data["session_count"]

        # Auto-add journal entry for session start
        entry = {
            "id": uuid.uuid4().hex[:8],
            "category": "decision",
            "entry": f"Session #{session_num} started" + (f". Focus: {focus}" if focus else ""),
            "recorded_at": datetime.now().isoformat(),
            "session": session_num,
        }
        data["journal"].append(entry)

        self._save(data)

        # Build session context from current state
        pillar_scores = {
            name: info["score"] for name, info in data["pillars"].items()
        }
        weakest = min(pillar_scores, key=pillar_scores.get) if any(pillar_scores.values()) else None
        recommendations = data.get("recommendations", [])

        msg_parts = [f"Session #{session_num} started."]
        if focus:
            msg_parts.append(f"Focus: {focus}.")
        if weakest and pillar_scores.get(weakest, 0) > 0:
            msg_parts.append(f"Weakest pillar: {weakest} ({pillar_scores[weakest]:.0f}/100).")
        if recommendations:
            msg_parts.append(f"Top recommendation: {recommendations[0].get('action', 'N/A')}")

        return SkillResult(
            success=True,
            message=" ".join(msg_parts),
            data={
                "session_number": session_num,
                "focus": focus,
                "pillar_scores": pillar_scores,
                "weakest_pillar": weakest,
                "top_recommendations": recommendations[:3],
            },
        )
