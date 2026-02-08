#!/usr/bin/env python3
"""
DecisionLogSkill - Structured decision logging and analysis for autonomous agents.

Every significant agent decision is logged with full context: what was decided,
what alternatives were considered, the reasoning, confidence level, and later
the actual outcome. This creates a queryable "institutional memory" that
enables the agent to:

1. Learn from past decisions (Self-Improvement pillar)
   - Query: "What happened last time I chose X over Y?"
   - Identify systematic errors or biases

2. Explain reasoning to users/customers (Revenue/trust)
   - Full audit trail of why actions were taken
   - Transparency builds customer confidence

3. Build decision playbooks (Goal Setting pillar)
   - Common situations get pre-computed best responses
   - New agents can inherit decision wisdom from experienced ones

4. Track decision quality over time (Self-Improvement)
   - Decision accuracy metrics, calibration analysis
   - Confidence vs actual outcome correlation

Architecture:
  Decision = {context, choice, alternatives, reasoning, confidence, outcome}
  Playbook = pattern -> recommended decision (auto-generated from history)
  Analysis = aggregate stats, calibration, bias detection

Part of the Self-Improvement pillar: the "wisdom layer" on top of raw learning.
"""

import json
import uuid
import time
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

from .base import Skill, SkillResult, SkillManifest, SkillAction


DECISION_LOG_FILE = Path(__file__).parent.parent / "data" / "decision_log.json"
MAX_DECISIONS = 5000
MAX_PLAYBOOK_ENTRIES = 200
MAX_TAGS = 20


# Decision categories
CATEGORIES = [
    "skill_selection",    # Which skill to use
    "prioritization",     # What to work on next
    "error_handling",     # How to handle failures
    "resource_allocation", # Budget/compute decisions
    "strategy",           # High-level strategic choices
    "customer",           # Customer-facing decisions
    "replication",        # Spawn/clone decisions
    "other",
]

# Confidence levels
CONFIDENCE_LEVELS = {
    "very_low": 0.1,
    "low": 0.3,
    "medium": 0.5,
    "high": 0.7,
    "very_high": 0.9,
}


class DecisionLogSkill(Skill):
    """
    Structured decision logging with outcome tracking and pattern analysis.

    Enables agents to:
    - Log decisions with context, alternatives, reasoning, and confidence
    - Record outcomes for past decisions (success/failure + details)
    - Query decision history by category, tags, time range
    - Analyze decision quality (calibration, accuracy, patterns)
    - Auto-generate playbooks from successful decision patterns
    - Get recommendations based on similar past decisions
    """

    def __init__(self, credentials: Dict[str, str] = None):
        super().__init__(credentials)
        self._ensure_data()

    def _ensure_data(self):
        DECISION_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        if not DECISION_LOG_FILE.exists():
            self._save(self._default_state())

    def _default_state(self) -> Dict:
        return {
            "decisions": [],
            "playbook": [],
            "stats": {
                "total_decisions": 0,
                "total_with_outcomes": 0,
                "accuracy_by_category": {},
                "calibration_data": [],
            },
        }

    def _load(self) -> Dict:
        try:
            with open(DECISION_LOG_FILE, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return self._default_state()

    def _save(self, state: Dict):
        with open(DECISION_LOG_FILE, "w") as f:
            json.dump(state, f, indent=2)

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="decision_log",
            name="Decision Log",
            version="1.0.0",
            category="meta",
            description="Structured decision logging, outcome tracking, and pattern analysis",
            actions=[
                SkillAction(
                    name="log_decision",
                    description="Log a significant decision with context and reasoning",
                    parameters={
                        "category": {
                            "type": "string",
                            "required": True,
                            "description": f"Decision category: {', '.join(CATEGORIES)}",
                        },
                        "context": {
                            "type": "string",
                            "required": True,
                            "description": "What situation required a decision",
                        },
                        "choice": {
                            "type": "string",
                            "required": True,
                            "description": "What was decided",
                        },
                        "alternatives": {
                            "type": "array",
                            "required": False,
                            "description": "List of alternatives that were considered",
                        },
                        "reasoning": {
                            "type": "string",
                            "required": True,
                            "description": "Why this choice was made over alternatives",
                        },
                        "confidence": {
                            "type": "string",
                            "required": False,
                            "description": "Confidence level: very_low, low, medium, high, very_high (default: medium)",
                        },
                        "tags": {
                            "type": "array",
                            "required": False,
                            "description": "Tags for categorization (max 20)",
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="record_outcome",
                    description="Record the outcome of a previous decision",
                    parameters={
                        "decision_id": {
                            "type": "string",
                            "required": True,
                            "description": "ID of the decision to update",
                        },
                        "success": {
                            "type": "boolean",
                            "required": True,
                            "description": "Whether the decision led to a good outcome",
                        },
                        "details": {
                            "type": "string",
                            "required": False,
                            "description": "Details about what happened",
                        },
                        "impact_score": {
                            "type": "number",
                            "required": False,
                            "description": "Impact score from -1.0 (very bad) to 1.0 (very good)",
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="query",
                    description="Query decision history with filters",
                    parameters={
                        "category": {
                            "type": "string",
                            "required": False,
                            "description": "Filter by category",
                        },
                        "tag": {
                            "type": "string",
                            "required": False,
                            "description": "Filter by tag",
                        },
                        "has_outcome": {
                            "type": "boolean",
                            "required": False,
                            "description": "Filter to only decisions with recorded outcomes",
                        },
                        "limit": {
                            "type": "integer",
                            "required": False,
                            "description": "Max results to return (default: 20)",
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="analyze",
                    description="Analyze decision quality: accuracy, calibration, patterns",
                    parameters={
                        "category": {
                            "type": "string",
                            "required": False,
                            "description": "Analyze a specific category (default: all)",
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="recommend",
                    description="Get recommendations based on similar past decisions",
                    parameters={
                        "context": {
                            "type": "string",
                            "required": True,
                            "description": "Current situation description",
                        },
                        "category": {
                            "type": "string",
                            "required": False,
                            "description": "Decision category to search within",
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="playbook",
                    description="View or manage the auto-generated decision playbook",
                    parameters={
                        "action": {
                            "type": "string",
                            "required": False,
                            "description": "'view' (default), 'generate' (rebuild from history), or 'clear'",
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="stats",
                    description="Get summary statistics about decision-making quality",
                    parameters={},
                    estimated_cost=0,
                ),
            ],
            required_credentials=[],
        )

    def check_credentials(self) -> bool:
        return True

    async def execute(self, action: str, params: Dict) -> SkillResult:
        handlers = {
            "log_decision": self._log_decision,
            "record_outcome": self._record_outcome,
            "query": self._query,
            "analyze": self._analyze,
            "recommend": self._recommend,
            "playbook": self._playbook,
            "stats": self._stats,
        }
        handler = handlers.get(action)
        if not handler:
            return SkillResult(success=False, message=f"Unknown action: {action}")
        return handler(params)

    # --- Core actions ---

    def _log_decision(self, params: Dict) -> SkillResult:
        category = params.get("category", "").strip()
        if category not in CATEGORIES:
            return SkillResult(
                success=False,
                message=f"Invalid category '{category}'. Must be one of: {', '.join(CATEGORIES)}",
            )

        context = params.get("context", "").strip()
        choice = params.get("choice", "").strip()
        reasoning = params.get("reasoning", "").strip()

        if not context or not choice or not reasoning:
            return SkillResult(
                success=False,
                message="context, choice, and reasoning are all required",
            )

        alternatives = params.get("alternatives", [])
        if isinstance(alternatives, str):
            alternatives = [a.strip() for a in alternatives.split(",") if a.strip()]

        confidence_str = params.get("confidence", "medium").strip().lower()
        confidence_val = CONFIDENCE_LEVELS.get(confidence_str, 0.5)

        tags = params.get("tags", [])
        if isinstance(tags, str):
            tags = [t.strip() for t in tags.split(",") if t.strip()]
        tags = tags[:MAX_TAGS]

        state = self._load()

        decision = {
            "id": str(uuid.uuid4())[:12],
            "category": category,
            "context": context,
            "choice": choice,
            "alternatives": alternatives,
            "reasoning": reasoning,
            "confidence": confidence_val,
            "confidence_label": confidence_str,
            "tags": tags,
            "timestamp": time.time(),
            "created_at": datetime.now().isoformat(),
            "outcome": None,
        }

        state["decisions"].append(decision)
        state["stats"]["total_decisions"] += 1

        # Trim if over max
        if len(state["decisions"]) > MAX_DECISIONS:
            state["decisions"] = state["decisions"][-MAX_DECISIONS:]

        self._save(state)

        return SkillResult(
            success=True,
            message=f"Decision logged: '{choice}' (category: {category}, confidence: {confidence_str})",
            data={
                "decision_id": decision["id"],
                "category": category,
                "confidence": confidence_val,
                "total_decisions": state["stats"]["total_decisions"],
            },
        )

    def _record_outcome(self, params: Dict) -> SkillResult:
        decision_id = params.get("decision_id", "").strip()
        if not decision_id:
            return SkillResult(success=False, message="decision_id is required")

        success = params.get("success")
        if success is None:
            return SkillResult(success=False, message="success (boolean) is required")
        if isinstance(success, str):
            success = success.lower() in ("true", "1", "yes")

        details = params.get("details", "")
        impact_score = params.get("impact_score")
        if impact_score is not None:
            try:
                impact_score = float(impact_score)
                impact_score = max(-1.0, min(1.0, impact_score))
            except (ValueError, TypeError):
                impact_score = None

        state = self._load()

        # Find the decision
        decision = None
        for d in state["decisions"]:
            if d["id"] == decision_id:
                decision = d
                break

        if not decision:
            return SkillResult(
                success=False,
                message=f"Decision '{decision_id}' not found",
            )

        if decision["outcome"] is not None:
            return SkillResult(
                success=False,
                message=f"Decision '{decision_id}' already has an outcome recorded",
            )

        decision["outcome"] = {
            "success": success,
            "details": details,
            "impact_score": impact_score,
            "recorded_at": time.time(),
        }

        state["stats"]["total_with_outcomes"] += 1

        # Update category accuracy
        cat = decision["category"]
        if cat not in state["stats"]["accuracy_by_category"]:
            state["stats"]["accuracy_by_category"][cat] = {
                "total": 0, "successes": 0, "failures": 0,
            }
        cat_stats = state["stats"]["accuracy_by_category"][cat]
        cat_stats["total"] += 1
        if success:
            cat_stats["successes"] += 1
        else:
            cat_stats["failures"] += 1

        # Track calibration data (confidence vs outcome)
        state["stats"]["calibration_data"].append({
            "confidence": decision["confidence"],
            "success": success,
        })
        # Keep calibration data bounded
        if len(state["stats"]["calibration_data"]) > 1000:
            state["stats"]["calibration_data"] = state["stats"]["calibration_data"][-1000:]

        self._save(state)

        return SkillResult(
            success=True,
            message=f"Outcome recorded for decision '{decision_id}': {'success' if success else 'failure'}",
            data={
                "decision_id": decision_id,
                "success": success,
                "impact_score": impact_score,
                "category": cat,
            },
        )

    def _query(self, params: Dict) -> SkillResult:
        state = self._load()
        decisions = state["decisions"]

        # Apply filters
        category = params.get("category", "").strip()
        if category:
            decisions = [d for d in decisions if d["category"] == category]

        tag = params.get("tag", "").strip()
        if tag:
            decisions = [d for d in decisions if tag in d.get("tags", [])]

        has_outcome = params.get("has_outcome")
        if has_outcome is not None:
            if isinstance(has_outcome, str):
                has_outcome = has_outcome.lower() in ("true", "1", "yes")
            if has_outcome:
                decisions = [d for d in decisions if d.get("outcome") is not None]
            else:
                decisions = [d for d in decisions if d.get("outcome") is None]

        limit = int(params.get("limit", 20))
        limit = max(1, min(limit, 100))

        # Return most recent first
        results = decisions[-limit:][::-1]

        # Summarize for readability
        summaries = []
        for d in results:
            summary = {
                "id": d["id"],
                "category": d["category"],
                "choice": d["choice"],
                "confidence": d["confidence_label"],
                "created_at": d["created_at"],
                "has_outcome": d.get("outcome") is not None,
            }
            if d.get("outcome"):
                summary["outcome_success"] = d["outcome"]["success"]
                summary["impact_score"] = d["outcome"].get("impact_score")
            summaries.append(summary)

        return SkillResult(
            success=True,
            message=f"Found {len(decisions)} decisions, showing {len(summaries)}",
            data={
                "total_matching": len(decisions),
                "decisions": summaries,
            },
        )

    def _analyze(self, params: Dict) -> SkillResult:
        state = self._load()
        decisions = state["decisions"]

        category = params.get("category", "").strip()
        if category:
            decisions = [d for d in decisions if d["category"] == category]

        # Basic stats
        total = len(decisions)
        with_outcomes = [d for d in decisions if d.get("outcome") is not None]
        successes = [d for d in with_outcomes if d["outcome"]["success"]]
        failures = [d for d in with_outcomes if not d["outcome"]["success"]]

        accuracy = len(successes) / len(with_outcomes) if with_outcomes else None

        # Calibration analysis: group by confidence bucket and compare predicted vs actual
        calibration = self._compute_calibration(with_outcomes)

        # Category breakdown
        category_breakdown = defaultdict(lambda: {"total": 0, "successes": 0, "failures": 0})
        for d in with_outcomes:
            cat = d["category"]
            category_breakdown[cat]["total"] += 1
            if d["outcome"]["success"]:
                category_breakdown[cat]["successes"] += 1
            else:
                category_breakdown[cat]["failures"] += 1

        # Top performing tags
        tag_performance = defaultdict(lambda: {"total": 0, "successes": 0})
        for d in with_outcomes:
            for tag in d.get("tags", []):
                tag_performance[tag]["total"] += 1
                if d["outcome"]["success"]:
                    tag_performance[tag]["successes"] += 1

        top_tags = sorted(
            tag_performance.items(),
            key=lambda x: x[1]["successes"] / max(x[1]["total"], 1),
            reverse=True,
        )[:10]

        # Impact distribution
        impacts = [
            d["outcome"]["impact_score"]
            for d in with_outcomes
            if d["outcome"].get("impact_score") is not None
        ]
        avg_impact = sum(impacts) / len(impacts) if impacts else None

        analysis = {
            "total_decisions": total,
            "with_outcomes": len(with_outcomes),
            "pending_outcomes": total - len(with_outcomes),
            "accuracy": round(accuracy, 3) if accuracy is not None else None,
            "success_count": len(successes),
            "failure_count": len(failures),
            "calibration": calibration,
            "category_breakdown": dict(category_breakdown),
            "top_tags": [{"tag": t, **s} for t, s in top_tags],
            "avg_impact_score": round(avg_impact, 3) if avg_impact is not None else None,
        }

        # Generate insights
        insights = self._generate_insights(analysis, with_outcomes)

        return SkillResult(
            success=True,
            message=f"Analysis: {total} decisions, accuracy {accuracy:.1%}" if accuracy else f"Analysis: {total} decisions (no outcomes yet)",
            data={**analysis, "insights": insights},
        )

    def _compute_calibration(self, decisions_with_outcomes: List[Dict]) -> List[Dict]:
        """Compute calibration: does confidence predict accuracy?"""
        buckets = {
            "very_low (0.1)": (0.0, 0.2),
            "low (0.3)": (0.2, 0.4),
            "medium (0.5)": (0.4, 0.6),
            "high (0.7)": (0.6, 0.8),
            "very_high (0.9)": (0.8, 1.0),
        }

        calibration = []
        for label, (lo, hi) in buckets.items():
            in_bucket = [
                d for d in decisions_with_outcomes
                if lo <= d["confidence"] < hi or (hi == 1.0 and d["confidence"] == 1.0)
            ]
            if not in_bucket:
                continue
            actual_success = sum(1 for d in in_bucket if d["outcome"]["success"]) / len(in_bucket)
            expected_mid = (lo + hi) / 2
            calibration.append({
                "bucket": label,
                "count": len(in_bucket),
                "expected_success_rate": round(expected_mid, 2),
                "actual_success_rate": round(actual_success, 3),
                "calibration_error": round(abs(actual_success - expected_mid), 3),
            })

        return calibration

    def _generate_insights(self, analysis: Dict, with_outcomes: List[Dict]) -> List[str]:
        """Generate human-readable insights from the analysis."""
        insights = []

        accuracy = analysis.get("accuracy")
        if accuracy is not None:
            if accuracy >= 0.8:
                insights.append(f"Strong decision quality: {accuracy:.0%} accuracy")
            elif accuracy >= 0.6:
                insights.append(f"Moderate decision quality: {accuracy:.0%} accuracy - room for improvement")
            else:
                insights.append(f"Low decision quality: {accuracy:.0%} accuracy - review decision-making process")

        # Calibration insight
        cal = analysis.get("calibration", [])
        if cal:
            overconfident = [c for c in cal if c["actual_success_rate"] < c["expected_success_rate"] - 0.15]
            underconfident = [c for c in cal if c["actual_success_rate"] > c["expected_success_rate"] + 0.15]
            if overconfident:
                buckets = ", ".join(c["bucket"] for c in overconfident)
                insights.append(f"Overconfident in: {buckets} - lower confidence or improve execution")
            if underconfident:
                buckets = ", ".join(c["bucket"] for c in underconfident)
                insights.append(f"Underconfident in: {buckets} - could be more decisive")

        # Category insight
        cat_breakdown = analysis.get("category_breakdown", {})
        worst_cat = None
        worst_rate = 1.0
        for cat, stats in cat_breakdown.items():
            if stats["total"] >= 3:
                rate = stats["successes"] / stats["total"]
                if rate < worst_rate:
                    worst_rate = rate
                    worst_cat = cat
        if worst_cat and worst_rate < 0.6:
            insights.append(f"Weakest category: '{worst_cat}' ({worst_rate:.0%} success) - needs attention")

        # Pending outcomes
        pending = analysis.get("pending_outcomes", 0)
        if pending > 10:
            insights.append(f"{pending} decisions awaiting outcome recording - close the feedback loop")

        return insights

    def _recommend(self, params: Dict) -> SkillResult:
        context = params.get("context", "").strip()
        if not context:
            return SkillResult(success=False, message="context is required")

        category = params.get("category", "").strip()
        state = self._load()
        decisions = state["decisions"]

        # Filter to decisions with outcomes
        with_outcomes = [d for d in decisions if d.get("outcome") is not None]
        if category:
            with_outcomes = [d for d in with_outcomes if d["category"] == category]

        if not with_outcomes:
            return SkillResult(
                success=True,
                message="No past decisions with outcomes to base recommendations on",
                data={"recommendations": [], "based_on": 0},
            )

        # Simple keyword matching for finding similar decisions
        context_words = set(context.lower().split())
        scored = []
        for d in with_outcomes:
            d_words = set(d["context"].lower().split()) | set(d["choice"].lower().split())
            overlap = len(context_words & d_words)
            if overlap > 0:
                scored.append((overlap, d))

        scored.sort(key=lambda x: x[0], reverse=True)
        similar = scored[:10]

        if not similar:
            return SkillResult(
                success=True,
                message="No similar past decisions found",
                data={"recommendations": [], "based_on": 0},
            )

        # Group by choice and compute success rates
        choice_stats = defaultdict(lambda: {"count": 0, "successes": 0, "avg_impact": [], "example_reasoning": ""})
        for _, d in similar:
            ch = d["choice"]
            choice_stats[ch]["count"] += 1
            if d["outcome"]["success"]:
                choice_stats[ch]["successes"] += 1
            if d["outcome"].get("impact_score") is not None:
                choice_stats[ch]["avg_impact"].append(d["outcome"]["impact_score"])
            if not choice_stats[ch]["example_reasoning"]:
                choice_stats[ch]["example_reasoning"] = d["reasoning"]

        recommendations = []
        for choice, stats in choice_stats.items():
            rate = stats["successes"] / stats["count"]
            avg_imp = sum(stats["avg_impact"]) / len(stats["avg_impact"]) if stats["avg_impact"] else None
            recommendations.append({
                "choice": choice,
                "success_rate": round(rate, 3),
                "based_on_count": stats["count"],
                "avg_impact": round(avg_imp, 3) if avg_imp is not None else None,
                "example_reasoning": stats["example_reasoning"],
            })

        recommendations.sort(key=lambda r: r["success_rate"], reverse=True)

        return SkillResult(
            success=True,
            message=f"Found {len(recommendations)} recommendation(s) from {len(similar)} similar decisions",
            data={
                "recommendations": recommendations,
                "based_on": len(similar),
            },
        )

    def _playbook(self, params: Dict) -> SkillResult:
        action = params.get("action", "view").strip().lower()
        state = self._load()

        if action == "clear":
            state["playbook"] = []
            self._save(state)
            return SkillResult(success=True, message="Playbook cleared", data={"entries": 0})

        if action == "generate":
            return self._generate_playbook(state)

        # Default: view
        playbook = state.get("playbook", [])
        return SkillResult(
            success=True,
            message=f"Playbook has {len(playbook)} entries",
            data={"playbook": playbook},
        )

    def _generate_playbook(self, state: Dict) -> SkillResult:
        """Auto-generate playbook entries from high-confidence successful patterns."""
        decisions = state["decisions"]
        with_outcomes = [d for d in decisions if d.get("outcome") is not None]

        # Group by (category, choice) and find patterns with high success rates
        patterns = defaultdict(lambda: {"total": 0, "successes": 0, "contexts": [], "reasoning": []})
        for d in with_outcomes:
            key = f"{d['category']}::{d['choice']}"
            patterns[key]["total"] += 1
            if d["outcome"]["success"]:
                patterns[key]["successes"] += 1
            patterns[key]["contexts"].append(d["context"][:100])
            patterns[key]["reasoning"].append(d["reasoning"][:100])

        playbook = []
        for key, stats in patterns.items():
            if stats["total"] < 2:
                continue
            success_rate = stats["successes"] / stats["total"]
            if success_rate < 0.6:
                continue

            category, choice = key.split("::", 1)
            entry = {
                "category": category,
                "recommended_choice": choice,
                "success_rate": round(success_rate, 3),
                "based_on": stats["total"],
                "typical_contexts": stats["contexts"][:3],
                "typical_reasoning": stats["reasoning"][0] if stats["reasoning"] else "",
            }
            playbook.append(entry)

        # Sort by success rate * count (reward both quality and quantity)
        playbook.sort(key=lambda e: e["success_rate"] * math.log1p(e["based_on"]), reverse=True)
        playbook = playbook[:MAX_PLAYBOOK_ENTRIES]

        state["playbook"] = playbook
        self._save(state)

        return SkillResult(
            success=True,
            message=f"Generated playbook with {len(playbook)} entries from {len(with_outcomes)} decisions",
            data={"playbook": playbook, "source_decisions": len(with_outcomes)},
        )

    def _stats(self, params: Dict) -> SkillResult:
        state = self._load()
        decisions = state["decisions"]

        total = len(decisions)
        with_outcomes = [d for d in decisions if d.get("outcome") is not None]
        successes = [d for d in with_outcomes if d["outcome"]["success"]]

        # Category distribution
        cat_dist = defaultdict(int)
        for d in decisions:
            cat_dist[d["category"]] += 1

        # Confidence distribution
        conf_dist = defaultdict(int)
        for d in decisions:
            conf_dist[d.get("confidence_label", "unknown")] += 1

        # Recent trend (last 20 decisions with outcomes)
        recent = with_outcomes[-20:]
        recent_accuracy = (
            sum(1 for d in recent if d["outcome"]["success"]) / len(recent)
            if recent
            else None
        )

        # Time span
        if decisions:
            first_ts = decisions[0].get("timestamp", 0)
            last_ts = decisions[-1].get("timestamp", 0)
            span_hours = (last_ts - first_ts) / 3600 if first_ts and last_ts else 0
        else:
            span_hours = 0

        summary = {
            "total_decisions": total,
            "with_outcomes": len(with_outcomes),
            "pending_outcomes": total - len(with_outcomes),
            "overall_accuracy": round(len(successes) / len(with_outcomes), 3) if with_outcomes else None,
            "recent_accuracy": round(recent_accuracy, 3) if recent_accuracy is not None else None,
            "category_distribution": dict(cat_dist),
            "confidence_distribution": dict(conf_dist),
            "playbook_entries": len(state.get("playbook", [])),
            "history_span_hours": round(span_hours, 1),
        }

        return SkillResult(
            success=True,
            message=f"Decision log: {total} decisions, {len(with_outcomes)} with outcomes",
            data=summary,
        )
