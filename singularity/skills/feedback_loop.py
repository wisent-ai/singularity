#!/usr/bin/env python3
"""
FeedbackLoop Skill - Closes the act-measure-adapt self-improvement loop.

This is the missing "adapt" component. While PerformanceTracker measures
outcomes and SelfEvalSkill reflects on cognition, FeedbackLoop automatically:

1. Analyzes recent performance data for patterns
2. Identifies what's working (high success, low cost) and what's failing
3. Generates concrete behavioral adaptations (prompt additions, skill preferences)
4. Applies those adaptations to the agent's behavior
5. Tracks which adaptations improved performance vs. made it worse

The complete self-improvement loop:
  act (execute skills) -> measure (PerformanceTracker) -> adapt (FeedbackLoop) -> act...

Part of the Self-Improvement pillar: the critical missing feedback mechanism.
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from .base import Skill, SkillResult, SkillManifest, SkillAction


FEEDBACK_FILE = Path(__file__).parent.parent / "data" / "feedback_loop.json"
PERF_FILE = Path(__file__).parent.parent / "data" / "performance.json"
MAX_ADAPTATIONS = 100
MAX_REVIEWS = 200


class FeedbackLoopSkill(Skill):
    """
    Closes the self-improvement feedback loop by analyzing performance
    patterns and generating behavioral adaptations.

    The agent calls feedback:review periodically to analyze recent outcomes,
    then feedback:apply to implement the suggested changes. Over time,
    feedback:evaluate checks whether adaptations actually helped.
    """

    def __init__(self, credentials: Dict[str, str] = None):
        super().__init__(credentials)
        self._append_prompt_fn: Optional[Callable[[str], None]] = None
        self._get_prompt_fn: Optional[Callable[[], str]] = None
        self._ensure_data()

    def set_cognition_hooks(
        self,
        append_prompt: Callable[[str], None],
        get_prompt: Callable[[], str],
    ):
        """Connect to the agent's cognition engine for applying adaptations."""
        self._append_prompt_fn = append_prompt
        self._get_prompt_fn = get_prompt

    def _ensure_data(self):
        FEEDBACK_FILE.parent.mkdir(parents=True, exist_ok=True)
        if not FEEDBACK_FILE.exists():
            self._save(self._default_state())

    def _default_state(self) -> Dict:
        return {
            "adaptations": [],
            "reviews": [],
            "adaptation_outcomes": [],
            "config": {
                "review_window_hours": 24,
                "min_samples": 3,
                "failure_threshold": 0.5,
                "cost_threshold_usd": 0.10,
            },
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
        }

    def _load(self) -> Dict:
        try:
            with open(FEEDBACK_FILE, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return self._default_state()

    def _save(self, data: Dict):
        data["last_updated"] = datetime.now().isoformat()
        if len(data.get("adaptations", [])) > MAX_ADAPTATIONS:
            data["adaptations"] = data["adaptations"][-MAX_ADAPTATIONS:]
        if len(data.get("reviews", [])) > MAX_REVIEWS:
            data["reviews"] = data["reviews"][-MAX_REVIEWS:]
        if len(data.get("adaptation_outcomes", [])) > MAX_ADAPTATIONS:
            data["adaptation_outcomes"] = data["adaptation_outcomes"][-MAX_ADAPTATIONS:]
        with open(FEEDBACK_FILE, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def _load_performance(self) -> Dict:
        """Load performance data from PerformanceTracker."""
        try:
            with open(PERF_FILE, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {"records": []}

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="feedback",
            name="Feedback Loop",
            version="1.0.0",
            category="meta",
            description="Analyze performance patterns and generate behavioral adaptations",
            actions=[
                SkillAction(
                    name="review",
                    description="Analyze recent performance and generate adaptation suggestions",
                    parameters={
                        "window_hours": {
                            "type": "number",
                            "required": False,
                            "description": "Hours of history to analyze (default: 24)",
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="apply",
                    description="Apply a specific adaptation from the latest review",
                    parameters={
                        "adaptation_id": {
                            "type": "string",
                            "required": True,
                            "description": "ID of the adaptation to apply",
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="apply_all",
                    description="Apply all pending adaptations from the latest review",
                    parameters={},
                    estimated_cost=0,
                ),
                SkillAction(
                    name="evaluate",
                    description="Evaluate whether applied adaptations improved performance",
                    parameters={},
                    estimated_cost=0,
                ),
                SkillAction(
                    name="status",
                    description="Get current feedback loop status and adaptation history",
                    parameters={},
                    estimated_cost=0,
                ),
                SkillAction(
                    name="revert",
                    description="Revert an adaptation that made things worse",
                    parameters={
                        "adaptation_id": {
                            "type": "string",
                            "required": True,
                            "description": "ID of the adaptation to revert",
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
            "review": self._review,
            "apply": self._apply,
            "apply_all": self._apply_all,
            "evaluate": self._evaluate,
            "status": self._status,
            "revert": self._revert,
        }
        handler = handlers.get(action)
        if handler:
            return handler(params)
        return SkillResult(success=False, message=f"Unknown action: {action}")

    def _review(self, params: Dict) -> SkillResult:
        """Analyze recent performance and generate adaptation suggestions."""
        window_hours = params.get("window_hours", 24)
        perf_data = self._load_performance()
        records = perf_data.get("records", [])

        if not records:
            return SkillResult(
                success=True,
                message="No performance data to analyze yet. Run some actions first.",
                data={"suggestions": [], "record_count": 0},
            )

        # Filter to recent records within window
        cutoff = datetime.now() - timedelta(hours=float(window_hours))
        recent = []
        for r in records:
            try:
                ts = datetime.fromisoformat(r.get("timestamp", ""))
                if ts >= cutoff:
                    recent.append(r)
            except (ValueError, TypeError):
                continue

        if len(recent) < 1:
            return SkillResult(
                success=True,
                message=f"Only {len(recent)} records in last {window_hours}h. Need more data.",
                data={"suggestions": [], "record_count": len(recent)},
            )

        # Compute per-skill stats
        skill_stats = self._compute_skill_stats(recent)

        # Generate adaptations
        fb_data = self._load()
        config = fb_data.get("config", {})
        min_samples = config.get("min_samples", 3)
        failure_threshold = config.get("failure_threshold", 0.5)
        cost_threshold = config.get("cost_threshold_usd", 0.10)

        suggestions = []
        now = datetime.now()

        for skill_key, stats in skill_stats.items():
            # Pattern: High failure rate
            if stats["total"] >= min_samples and stats["success_rate"] < failure_threshold:
                adapt_id = f"adapt_{skill_key}_{now.strftime('%Y%m%d%H%M%S')}"
                suggestions.append({
                    "id": adapt_id,
                    "type": "avoid_failing_skill",
                    "skill": skill_key,
                    "reason": f"{skill_key} has {stats['success_rate']:.0%} success rate "
                              f"over {stats['total']} recent actions",
                    "action": f"Prefer alternatives to {skill_key} - it fails "
                              f"{100 - stats['success_rate'] * 100:.0f}% of the time",
                    "severity": "high" if stats["success_rate"] < 0.3 else "medium",
                    "stats": stats,
                    "applied": False,
                    "created_at": now.isoformat(),
                })

            # Pattern: High cost, low success
            if stats["total"] >= min_samples and stats["avg_cost"] > cost_threshold and stats["success_rate"] < 0.7:
                adapt_id = f"adapt_cost_{skill_key}_{now.strftime('%Y%m%d%H%M%S')}"
                suggestions.append({
                    "id": adapt_id,
                    "type": "cost_inefficient",
                    "skill": skill_key,
                    "reason": f"{skill_key} costs ${stats['avg_cost']:.4f}/action with only "
                              f"{stats['success_rate']:.0%} success",
                    "action": f"Reduce usage of {skill_key} - expensive and unreliable",
                    "severity": "high",
                    "stats": stats,
                    "applied": False,
                    "created_at": now.isoformat(),
                })

            # Pattern: Consistent success (positive reinforcement)
            if stats["total"] >= min_samples and stats["success_rate"] >= 0.9:
                adapt_id = f"adapt_prefer_{skill_key}_{now.strftime('%Y%m%d%H%M%S')}"
                suggestions.append({
                    "id": adapt_id,
                    "type": "prefer_successful_skill",
                    "skill": skill_key,
                    "reason": f"{skill_key} has {stats['success_rate']:.0%} success rate "
                              f"over {stats['total']} actions",
                    "action": f"Continue using {skill_key} - consistently reliable",
                    "severity": "low",
                    "stats": stats,
                    "applied": False,
                    "created_at": now.isoformat(),
                })

            # Pattern: Slow execution
            if stats["total"] >= min_samples and stats["avg_latency_ms"] > 5000:
                adapt_id = f"adapt_slow_{skill_key}_{now.strftime('%Y%m%d%H%M%S')}"
                suggestions.append({
                    "id": adapt_id,
                    "type": "slow_execution",
                    "skill": skill_key,
                    "reason": f"{skill_key} averages {stats['avg_latency_ms']:.0f}ms per action",
                    "action": f"Consider batching or reducing {skill_key} calls - they are slow",
                    "severity": "medium",
                    "stats": stats,
                    "applied": False,
                    "created_at": now.isoformat(),
                })

        # Record the review
        review = {
            "timestamp": now.isoformat(),
            "window_hours": window_hours,
            "records_analyzed": len(recent),
            "skills_analyzed": len(skill_stats),
            "suggestions_count": len(suggestions),
            "skill_stats": skill_stats,
        }

        fb_data["reviews"].append(review)

        # Store new suggestions (don't duplicate existing unapplied ones)
        existing_ids = {a["id"] for a in fb_data.get("adaptations", [])}
        new_suggestions = [s for s in suggestions if s["id"] not in existing_ids]
        fb_data.setdefault("adaptations", []).extend(new_suggestions)

        self._save(fb_data)

        return SkillResult(
            success=True,
            message=f"Analyzed {len(recent)} actions across {len(skill_stats)} skills. "
                    f"Generated {len(new_suggestions)} new adaptations.",
            data={
                "suggestions": suggestions,
                "record_count": len(recent),
                "skills_analyzed": len(skill_stats),
                "skill_stats": skill_stats,
            },
        )

    def _compute_skill_stats(self, records: List[Dict]) -> Dict[str, Dict]:
        """Compute per-skill success rate, avg cost, avg latency from records."""
        from collections import defaultdict

        stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "total": 0,
            "successes": 0,
            "failures": 0,
            "total_cost": 0.0,
            "total_latency_ms": 0.0,
            "errors": [],
        })

        for r in records:
            skill_id = r.get("skill_id", "unknown")
            action = r.get("action", "")
            key = f"{skill_id}:{action}" if action else skill_id

            s = stats[key]
            s["total"] += 1
            if r.get("success"):
                s["successes"] += 1
            else:
                s["failures"] += 1
                error = r.get("error", "")
                if error and len(s["errors"]) < 3:
                    s["errors"].append(error[:100])
            s["total_cost"] += r.get("cost_usd", 0.0)
            s["total_latency_ms"] += r.get("latency_ms", 0.0)

        # Compute averages
        result = {}
        for key, s in stats.items():
            total = s["total"]
            result[key] = {
                "total": total,
                "successes": s["successes"],
                "failures": s["failures"],
                "success_rate": s["successes"] / total if total > 0 else 0,
                "avg_cost": s["total_cost"] / total if total > 0 else 0,
                "total_cost": s["total_cost"],
                "avg_latency_ms": s["total_latency_ms"] / total if total > 0 else 0,
                "recent_errors": s["errors"],
            }

        return result

    def _apply(self, params: Dict) -> SkillResult:
        """Apply a specific adaptation."""
        adapt_id = params.get("adaptation_id", "").strip()
        if not adapt_id:
            return SkillResult(success=False, message="adaptation_id is required")

        fb_data = self._load()
        adaptation = None
        for a in fb_data.get("adaptations", []):
            if a["id"] == adapt_id:
                adaptation = a
                break

        if not adaptation:
            return SkillResult(success=False, message=f"Adaptation not found: {adapt_id}")

        if adaptation.get("applied"):
            return SkillResult(success=False, message=f"Adaptation already applied: {adapt_id}")

        return self._apply_adaptation(fb_data, adaptation)

    def _apply_all(self, params: Dict) -> SkillResult:
        """Apply all pending adaptations."""
        fb_data = self._load()
        pending = [a for a in fb_data.get("adaptations", []) if not a.get("applied")]

        if not pending:
            return SkillResult(
                success=True,
                message="No pending adaptations to apply. Run feedback:review first.",
                data={"applied_count": 0},
            )

        applied = []
        for adaptation in pending:
            result = self._apply_adaptation(fb_data, adaptation)
            if result.success:
                applied.append(adaptation["id"])

        self._save(fb_data)

        return SkillResult(
            success=True,
            message=f"Applied {len(applied)} adaptations",
            data={"applied": applied, "total_pending": len(pending)},
        )

    def _apply_adaptation(self, fb_data: Dict, adaptation: Dict) -> SkillResult:
        """Internal: apply a single adaptation to the agent's behavior."""
        adapt_type = adaptation.get("type", "")
        skill = adaptation.get("skill", "")
        action_text = adaptation.get("action", "")

        # Apply via prompt addition if cognition hooks available
        if self._append_prompt_fn:
            prompt_addition = (
                f"\n=== LEARNED ADAPTATION ({adapt_type}) ===\n"
                f"Based on performance data: {action_text}"
            )
            self._append_prompt_fn(prompt_addition)

        # Mark as applied
        adaptation["applied"] = True
        adaptation["applied_at"] = datetime.now().isoformat()

        # Record for later evaluation
        fb_data.setdefault("adaptation_outcomes", []).append({
            "adaptation_id": adaptation["id"],
            "type": adapt_type,
            "skill": skill,
            "applied_at": datetime.now().isoformat(),
            "pre_apply_stats": adaptation.get("stats", {}),
            "post_apply_stats": None,
            "verdict": "pending",
        })

        self._save(fb_data)

        return SkillResult(
            success=True,
            message=f"Applied adaptation: {action_text}",
            data={
                "adaptation_id": adaptation["id"],
                "type": adapt_type,
                "skill": skill,
                "prompt_updated": self._append_prompt_fn is not None,
            },
        )

    def _evaluate(self, params: Dict) -> SkillResult:
        """Evaluate whether applied adaptations improved performance."""
        fb_data = self._load()
        outcomes = fb_data.get("adaptation_outcomes", [])
        pending = [o for o in outcomes if o.get("verdict") == "pending"]

        if not pending:
            return SkillResult(
                success=True,
                message="No pending adaptation evaluations.",
                data={"evaluated": 0},
            )

        perf_data = self._load_performance()
        records = perf_data.get("records", [])

        evaluated = 0
        improved = 0
        degraded = 0
        neutral = 0

        for outcome in pending:
            applied_at = outcome.get("applied_at", "")
            skill = outcome.get("skill", "")

            if not applied_at or not skill:
                continue

            try:
                applied_time = datetime.fromisoformat(applied_at)
            except (ValueError, TypeError):
                continue

            # Get records before and after adaptation
            before_records = []
            after_records = []
            for r in records:
                try:
                    ts = datetime.fromisoformat(r.get("timestamp", ""))
                except (ValueError, TypeError):
                    continue

                # Match skill
                r_skill = f"{r.get('skill_id', '')}:{r.get('action', '')}"
                if not r_skill.startswith(skill.split(":")[0]):
                    continue

                if ts < applied_time:
                    before_records.append(r)
                else:
                    after_records.append(r)

            # Need enough data to evaluate
            if len(after_records) < 2:
                continue

            # Compare success rates
            before_rate = (
                sum(1 for r in before_records if r.get("success")) / len(before_records)
                if before_records else 0
            )
            after_rate = (
                sum(1 for r in after_records if r.get("success")) / len(after_records)
            )

            outcome["post_apply_stats"] = {
                "before_count": len(before_records),
                "after_count": len(after_records),
                "before_success_rate": before_rate,
                "after_success_rate": after_rate,
                "delta": after_rate - before_rate,
            }

            if after_rate > before_rate + 0.05:
                outcome["verdict"] = "improved"
                improved += 1
            elif after_rate < before_rate - 0.05:
                outcome["verdict"] = "degraded"
                degraded += 1
            else:
                outcome["verdict"] = "neutral"
                neutral += 1

            outcome["evaluated_at"] = datetime.now().isoformat()
            evaluated += 1

        self._save(fb_data)

        return SkillResult(
            success=True,
            message=f"Evaluated {evaluated} adaptations: "
                    f"{improved} improved, {degraded} degraded, {neutral} neutral",
            data={
                "evaluated": evaluated,
                "improved": improved,
                "degraded": degraded,
                "neutral": neutral,
                "pending_remaining": len(pending) - evaluated,
            },
        )

    def _status(self, params: Dict) -> SkillResult:
        """Get current feedback loop status."""
        fb_data = self._load()

        adaptations = fb_data.get("adaptations", [])
        reviews = fb_data.get("reviews", [])
        outcomes = fb_data.get("adaptation_outcomes", [])

        total_adaptations = len(adaptations)
        applied = sum(1 for a in adaptations if a.get("applied"))
        pending = total_adaptations - applied

        # Outcome summary
        improved = sum(1 for o in outcomes if o.get("verdict") == "improved")
        degraded = sum(1 for o in outcomes if o.get("verdict") == "degraded")
        neutral_count = sum(1 for o in outcomes if o.get("verdict") == "neutral")
        pending_eval = sum(1 for o in outcomes if o.get("verdict") == "pending")

        # Recent adaptations
        recent = adaptations[-5:] if adaptations else []
        recent_summary = [
            {
                "id": a["id"],
                "type": a.get("type", ""),
                "skill": a.get("skill", ""),
                "applied": a.get("applied", False),
                "reason": a.get("reason", "")[:100],
            }
            for a in recent
        ]

        return SkillResult(
            success=True,
            message=f"Feedback loop: {total_adaptations} total adaptations, "
                    f"{applied} applied, {pending} pending",
            data={
                "total_adaptations": total_adaptations,
                "applied": applied,
                "pending": pending,
                "reviews_count": len(reviews),
                "outcomes": {
                    "improved": improved,
                    "degraded": degraded,
                    "neutral": neutral_count,
                    "pending_eval": pending_eval,
                },
                "effectiveness": (
                    round(improved / (improved + degraded + neutral_count) * 100, 1)
                    if (improved + degraded + neutral_count) > 0
                    else 0
                ),
                "recent_adaptations": recent_summary,
            },
        )

    def _revert(self, params: Dict) -> SkillResult:
        """Revert an adaptation that made things worse."""
        adapt_id = params.get("adaptation_id", "").strip()
        if not adapt_id:
            return SkillResult(success=False, message="adaptation_id is required")

        fb_data = self._load()

        # Find the adaptation
        adaptation = None
        for a in fb_data.get("adaptations", []):
            if a["id"] == adapt_id:
                adaptation = a
                break

        if not adaptation:
            return SkillResult(success=False, message=f"Adaptation not found: {adapt_id}")

        if not adaptation.get("applied"):
            # Just remove it
            fb_data["adaptations"] = [
                a for a in fb_data["adaptations"] if a["id"] != adapt_id
            ]
            self._save(fb_data)
            return SkillResult(
                success=True,
                message=f"Removed unapplied adaptation: {adapt_id}",
                data={"adaptation_id": adapt_id, "was_applied": False},
            )

        # Mark as reverted
        adaptation["reverted"] = True
        adaptation["reverted_at"] = datetime.now().isoformat()

        # If we have prompt access, add a counter-instruction
        if self._append_prompt_fn:
            action_text = adaptation.get("action", "")
            revert_text = (
                f"\n=== ADAPTATION REVERTED ===\n"
                f"Previous adaptation was counter-productive. "
                f"Ignore: {action_text}"
            )
            self._append_prompt_fn(revert_text)

        # Update outcome
        for o in fb_data.get("adaptation_outcomes", []):
            if o.get("adaptation_id") == adapt_id:
                o["verdict"] = "reverted"
                o["reverted_at"] = datetime.now().isoformat()
                break

        self._save(fb_data)

        return SkillResult(
            success=True,
            message=f"Reverted adaptation: {adaptation.get('action', '')[:100]}",
            data={
                "adaptation_id": adapt_id,
                "type": adaptation.get("type", ""),
                "skill": adaptation.get("skill", ""),
            },
        )
