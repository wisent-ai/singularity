#!/usr/bin/env python3
"""
Self-Evaluation Skill - Cognitive-level self-improvement through structured reflection.

This skill closes the self-improvement loop at the COGNITIVE level. While
PerformanceTracker measures execution metrics (latency, success rate) and
AdaptiveExecutor adjusts execution strategy, SelfEvalSkill enables the agent
to evaluate the QUALITY of its reasoning and outputs, identify patterns in
its strengths/weaknesses, and generate concrete improvement directives.

The feedback loop:
  1. Agent acts (executes tasks via skills)
  2. Agent evaluates its own outputs (this skill)
  3. Agent identifies patterns (this skill)
  4. Agent generates improvement directives (this skill)
  5. Directives get injected into the prompt (via self_modify integration)
  6. Agent acts with improved behavior → repeat

Part of the Self-Improvement pillar: cognitive-level act→evaluate→improve.
"""

import json
import statistics
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from .base import Skill, SkillResult, SkillManifest, SkillAction


EVAL_FILE = Path(__file__).parent.parent / "data" / "self_eval.json"
MAX_EVALUATIONS = 500
MAX_DIRECTIVES = 50


# Quality dimensions for evaluation
QUALITY_DIMENSIONS = {
    "correctness": "Was the output factually correct and did it achieve the goal?",
    "efficiency": "Was the approach efficient (minimal steps, low cost)?",
    "creativity": "Was the solution creative or just a standard approach?",
    "robustness": "Did the solution handle edge cases and errors well?",
    "communication": "Was the output clear and well-structured?",
}


class SelfEvalSkill(Skill):
    """
    Cognitive self-evaluation for continuous improvement.

    Enables the agent to:
    - Score its own outputs on multiple quality dimensions
    - Identify recurring strengths and weaknesses
    - Generate improvement directives from evaluation patterns
    - Track quality trends across sessions
    - Produce self-improvement reports
    """

    def __init__(self, credentials: Dict[str, str] = None):
        super().__init__(credentials)
        self._ensure_data()

    def _ensure_data(self):
        EVAL_FILE.parent.mkdir(parents=True, exist_ok=True)
        if not EVAL_FILE.exists():
            self._save(self._default_state())

    def _default_state(self) -> Dict:
        return {
            "evaluations": [],
            "directives": [],
            "improvement_log": [],
            "dimension_trends": {},
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
        }

    def _load(self) -> Dict:
        try:
            with open(EVAL_FILE, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return self._default_state()

    def _save(self, data: Dict):
        data["last_updated"] = datetime.now().isoformat()
        if len(data.get("evaluations", [])) > MAX_EVALUATIONS:
            data["evaluations"] = data["evaluations"][-MAX_EVALUATIONS:]
        if len(data.get("directives", [])) > MAX_DIRECTIVES:
            data["directives"] = data["directives"][-MAX_DIRECTIVES:]
        if len(data.get("improvement_log", [])) > 200:
            data["improvement_log"] = data["improvement_log"][-200:]
        with open(EVAL_FILE, "w") as f:
            json.dump(data, f, indent=2, default=str)

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="self_eval",
            name="Self-Evaluation",
            version="1.0.0",
            category="meta",
            description="Evaluate output quality, identify patterns, generate improvement directives",
            actions=[
                SkillAction(
                    name="evaluate",
                    description="Score a recent action/output on quality dimensions (1-10)",
                    parameters={
                        "task_description": {
                            "type": "string",
                            "required": True,
                            "description": "What task was being performed",
                        },
                        "output_summary": {
                            "type": "string",
                            "required": True,
                            "description": "Summary of what the agent produced",
                        },
                        "scores": {
                            "type": "object",
                            "required": True,
                            "description": "Scores (1-10) for dimensions: correctness, efficiency, creativity, robustness, communication",
                        },
                        "reflection": {
                            "type": "string",
                            "required": False,
                            "description": "Free-form reflection on what went well/poorly",
                        },
                        "skill_used": {
                            "type": "string",
                            "required": False,
                            "description": "Which skill was used (for skill-level analysis)",
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="analyze",
                    description="Analyze evaluation patterns to find strengths and weaknesses",
                    parameters={
                        "window_hours": {
                            "type": "integer",
                            "required": False,
                            "description": "Analysis window in hours (default: 168 = 1 week)",
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="generate_directives",
                    description="Generate concrete improvement directives from analysis",
                    parameters={
                        "max_directives": {
                            "type": "integer",
                            "required": False,
                            "description": "Max directives to generate (default: 3)",
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="get_directives",
                    description="Get current active improvement directives",
                    parameters={},
                    estimated_cost=0,
                ),
                SkillAction(
                    name="complete_directive",
                    description="Mark a directive as completed with outcome notes",
                    parameters={
                        "directive_id": {
                            "type": "string",
                            "required": True,
                            "description": "ID of the directive to complete",
                        },
                        "outcome": {
                            "type": "string",
                            "required": True,
                            "description": "What happened: 'improved', 'no_change', or 'worse'",
                        },
                        "notes": {
                            "type": "string",
                            "required": False,
                            "description": "Notes on the outcome",
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="trend",
                    description="Get quality score trends over time",
                    parameters={
                        "dimension": {
                            "type": "string",
                            "required": False,
                            "description": "Specific dimension to trend (default: all)",
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="report",
                    description="Generate a comprehensive self-improvement report",
                    parameters={},
                    estimated_cost=0,
                ),
                SkillAction(
                    name="reset",
                    description="Clear all evaluations and start fresh",
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
            "evaluate": self._evaluate,
            "analyze": self._analyze,
            "generate_directives": self._generate_directives,
            "get_directives": self._get_directives,
            "complete_directive": self._complete_directive,
            "trend": self._trend,
            "report": self._report,
            "reset": self._reset,
        }
        handler = handlers.get(action)
        if handler:
            return handler(params)
        return SkillResult(success=False, message=f"Unknown action: {action}")

    def _evaluate(self, params: Dict) -> SkillResult:
        """Score a recent action on quality dimensions."""
        task_description = params.get("task_description", "").strip()
        output_summary = params.get("output_summary", "").strip()
        scores = params.get("scores", {})
        reflection = params.get("reflection", "")
        skill_used = params.get("skill_used", "")

        if not task_description or not output_summary:
            return SkillResult(
                success=False,
                message="task_description and output_summary are required",
            )

        if not scores:
            return SkillResult(
                success=False,
                message="scores are required (dict of dimension -> 1-10 score)",
            )

        # Validate and clamp scores
        validated_scores = {}
        for dim in QUALITY_DIMENSIONS:
            raw = scores.get(dim)
            if raw is not None:
                try:
                    score = max(1, min(10, int(raw)))
                except (ValueError, TypeError):
                    score = 5
                validated_scores[dim] = score

        if not validated_scores:
            return SkillResult(
                success=False,
                message=f"At least one valid dimension score required. Dimensions: {list(QUALITY_DIMENSIONS.keys())}",
            )

        # Compute overall score as weighted average
        overall = round(sum(validated_scores.values()) / len(validated_scores), 1)

        evaluation = {
            "id": f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "timestamp": datetime.now().isoformat(),
            "task_description": task_description,
            "output_summary": output_summary,
            "scores": validated_scores,
            "overall_score": overall,
            "reflection": reflection,
            "skill_used": skill_used,
        }

        data = self._load()
        data["evaluations"].append(evaluation)
        self._save(data)

        return SkillResult(
            success=True,
            message=f"Evaluation recorded: overall {overall}/10",
            data={
                "evaluation_id": evaluation["id"],
                "scores": validated_scores,
                "overall": overall,
                "total_evaluations": len(data["evaluations"]),
            },
        )

    def _analyze(self, params: Dict) -> SkillResult:
        """Analyze evaluation patterns to identify strengths and weaknesses."""
        window_hours = params.get("window_hours", 168)
        data = self._load()
        evaluations = data.get("evaluations", [])

        if not evaluations:
            return SkillResult(
                success=False,
                message="No evaluations recorded yet. Use self_eval:evaluate first.",
            )

        # Filter by time window
        cutoff = datetime.now() - timedelta(hours=window_hours)
        recent = []
        for e in evaluations:
            try:
                ts = datetime.fromisoformat(e["timestamp"])
                if ts >= cutoff:
                    recent.append(e)
            except (ValueError, KeyError):
                recent.append(e)

        if not recent:
            recent = evaluations[-10:]

        # Compute per-dimension stats
        dim_scores: Dict[str, List[int]] = {}
        for e in recent:
            for dim, score in e.get("scores", {}).items():
                dim_scores.setdefault(dim, []).append(score)

        dim_stats = {}
        for dim, scores_list in dim_scores.items():
            avg = round(statistics.mean(scores_list), 1)
            dim_stats[dim] = {
                "average": avg,
                "min": min(scores_list),
                "max": max(scores_list),
                "count": len(scores_list),
                "stdev": round(statistics.stdev(scores_list), 1) if len(scores_list) > 1 else 0,
            }

        # Sort dimensions by average score
        sorted_dims = sorted(dim_stats.items(), key=lambda x: x[1]["average"])

        # Identify strengths and weaknesses
        strengths = [d for d, s in sorted_dims if s["average"] >= 7]
        weaknesses = [d for d, s in sorted_dims if s["average"] < 5]
        moderate = [d for d, s in sorted_dims if 5 <= s["average"] < 7]

        # Per-skill analysis
        skill_scores: Dict[str, List[float]] = {}
        for e in recent:
            skill = e.get("skill_used", "unknown")
            if skill:
                skill_scores.setdefault(skill, []).append(e.get("overall_score", 5))

        skill_stats = {}
        for skill, scores_list in skill_scores.items():
            skill_stats[skill] = {
                "average": round(statistics.mean(scores_list), 1),
                "count": len(scores_list),
            }

        overall_scores = [e.get("overall_score", 5) for e in recent]
        overall_avg = round(statistics.mean(overall_scores), 1)

        analysis = {
            "window_hours": window_hours,
            "evaluation_count": len(recent),
            "overall_average": overall_avg,
            "dimension_stats": dim_stats,
            "strengths": strengths,
            "weaknesses": weaknesses,
            "moderate": moderate,
            "skill_stats": skill_stats,
        }

        return SkillResult(
            success=True,
            message=f"Analysis of {len(recent)} evaluations: overall {overall_avg}/10, "
                    f"{len(strengths)} strengths, {len(weaknesses)} weaknesses",
            data=analysis,
        )

    def _generate_directives(self, params: Dict) -> SkillResult:
        """Generate improvement directives from analysis."""
        max_directives = params.get("max_directives", 3)

        # First run analysis
        analysis_result = self._analyze({"window_hours": 168})
        if not analysis_result.success:
            return analysis_result

        analysis = analysis_result.data
        data = self._load()
        new_directives = []

        # Existing active directive dimensions
        active_dims = set()
        for d in data.get("directives", []):
            if d.get("status") == "active":
                active_dims.add(d.get("target_dimension", ""))

        dim_stats = analysis.get("dimension_stats", {})
        weaknesses = analysis.get("weaknesses", [])
        moderate = analysis.get("moderate", [])

        # Generate directives for weakest dimensions first
        targets = weaknesses + moderate
        for dim in targets:
            if len(new_directives) >= max_directives:
                break
            if dim in active_dims:
                continue

            stats = dim_stats.get(dim, {})
            avg = stats.get("average", 5)
            description = QUALITY_DIMENSIONS.get(dim, "")

            directive = {
                "id": f"dir_{dim}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "created_at": datetime.now().isoformat(),
                "target_dimension": dim,
                "current_score": avg,
                "target_score": min(10, round(avg + 2, 1)),
                "instruction": self._generate_instruction(dim, avg, description),
                "status": "active",
                "outcome": None,
                "notes": "",
            }
            new_directives.append(directive)
            active_dims.add(dim)

        # If still room and overall score is moderate, add general directive
        if len(new_directives) < max_directives and analysis.get("overall_average", 10) < 7:
            overall_avg = analysis["overall_average"]
            if "overall" not in active_dims:
                directive = {
                    "id": f"dir_overall_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    "created_at": datetime.now().isoformat(),
                    "target_dimension": "overall",
                    "current_score": overall_avg,
                    "target_score": min(10, round(overall_avg + 1.5, 1)),
                    "instruction": (
                        f"Overall quality is at {overall_avg}/10. Focus on systematic "
                        f"improvement: verify outputs before submitting, consider edge "
                        f"cases, and structure responses clearly."
                    ),
                    "status": "active",
                    "outcome": None,
                    "notes": "",
                }
                new_directives.append(directive)

        if not new_directives:
            return SkillResult(
                success=True,
                message="No new directives needed - all dimensions are strong or already have active directives",
                data={"active_directives": len([d for d in data.get("directives", []) if d.get("status") == "active"])},
            )

        data["directives"].extend(new_directives)
        self._save(data)

        return SkillResult(
            success=True,
            message=f"Generated {len(new_directives)} improvement directives",
            data={
                "new_directives": new_directives,
                "total_active": len([d for d in data["directives"] if d.get("status") == "active"]),
            },
        )

    def _generate_instruction(self, dimension: str, current_score: float, description: str) -> str:
        """Generate a concrete improvement instruction for a dimension."""
        instructions = {
            "correctness": (
                f"Correctness is at {current_score}/10. Before finalizing any output, "
                f"double-check facts, verify logic, and test assumptions. Add a "
                f"self-verification step before submitting results."
            ),
            "efficiency": (
                f"Efficiency is at {current_score}/10. Reduce unnecessary steps: plan "
                f"before acting, combine related operations, and avoid redundant API "
                f"calls. Target completing tasks in fewer cycles."
            ),
            "creativity": (
                f"Creativity is at {current_score}/10. Before defaulting to the obvious "
                f"approach, brainstorm at least 2 alternative solutions. Consider "
                f"unconventional tools or combinations."
            ),
            "robustness": (
                f"Robustness is at {current_score}/10. Always consider: What if this "
                f"fails? Add error handling, validate inputs, test edge cases, and "
                f"have fallback strategies ready."
            ),
            "communication": (
                f"Communication is at {current_score}/10. Structure outputs with clear "
                f"headers, use bullet points for lists, lead with the key takeaway, "
                f"and keep explanations concise."
            ),
        }
        return instructions.get(
            dimension,
            f"{dimension} is at {current_score}/10. {description} Focus on improving this dimension.",
        )

    def _get_directives(self, params: Dict) -> SkillResult:
        """Get current active improvement directives."""
        data = self._load()
        all_directives = data.get("directives", [])
        active = [d for d in all_directives if d.get("status") == "active"]
        completed = [d for d in all_directives if d.get("status") == "completed"]

        return SkillResult(
            success=True,
            message=f"{len(active)} active directives, {len(completed)} completed",
            data={
                "active": active,
                "completed_count": len(completed),
                "total": len(all_directives),
            },
        )

    def _complete_directive(self, params: Dict) -> SkillResult:
        """Mark a directive as completed."""
        directive_id = params.get("directive_id", "").strip()
        outcome = params.get("outcome", "").strip()
        notes = params.get("notes", "")

        if not directive_id:
            return SkillResult(success=False, message="directive_id required")
        if outcome not in ("improved", "no_change", "worse"):
            return SkillResult(
                success=False,
                message="outcome must be 'improved', 'no_change', or 'worse'",
            )

        data = self._load()
        found = False
        for d in data.get("directives", []):
            if d["id"] == directive_id:
                d["status"] = "completed"
                d["outcome"] = outcome
                d["notes"] = notes
                d["completed_at"] = datetime.now().isoformat()
                found = True

                # Log to improvement history
                data.setdefault("improvement_log", []).append({
                    "directive_id": directive_id,
                    "dimension": d.get("target_dimension"),
                    "outcome": outcome,
                    "from_score": d.get("current_score"),
                    "target_score": d.get("target_score"),
                    "notes": notes,
                    "completed_at": datetime.now().isoformat(),
                })
                break

        if not found:
            return SkillResult(success=False, message=f"Directive not found: {directive_id}")

        self._save(data)
        return SkillResult(
            success=True,
            message=f"Directive {directive_id} completed with outcome: {outcome}",
            data={"directive_id": directive_id, "outcome": outcome},
        )

    def _trend(self, params: Dict) -> SkillResult:
        """Get quality score trends over time."""
        dimension = params.get("dimension")
        data = self._load()
        evaluations = data.get("evaluations", [])

        if not evaluations:
            return SkillResult(
                success=False,
                message="No evaluations to trend. Use self_eval:evaluate first.",
            )

        # Group evaluations into time buckets (daily)
        daily_scores: Dict[str, Dict[str, List]] = {}
        for e in evaluations:
            try:
                ts = datetime.fromisoformat(e["timestamp"])
                day = ts.strftime("%Y-%m-%d")
            except (ValueError, KeyError):
                continue

            if day not in daily_scores:
                daily_scores[day] = {"overall": []}

            daily_scores[day]["overall"].append(e.get("overall_score", 5))

            for dim, score in e.get("scores", {}).items():
                daily_scores[day].setdefault(dim, []).append(score)

        # Compute daily averages
        trend_data = {}
        for day in sorted(daily_scores.keys()):
            day_data = daily_scores[day]
            trend_data[day] = {}
            for dim, scores_list in day_data.items():
                if dimension and dim != dimension and dim != "overall":
                    continue
                trend_data[day][dim] = round(statistics.mean(scores_list), 1)

        # Compute trend direction (is it improving?)
        sorted_days = sorted(trend_data.keys())
        trend_direction = {}
        if len(sorted_days) >= 2:
            first_half = sorted_days[:len(sorted_days) // 2]
            second_half = sorted_days[len(sorted_days) // 2:]

            all_dims = set()
            for day_data in trend_data.values():
                all_dims.update(day_data.keys())

            for dim in all_dims:
                first_scores = [trend_data[d].get(dim) for d in first_half if dim in trend_data.get(d, {})]
                second_scores = [trend_data[d].get(dim) for d in second_half if dim in trend_data.get(d, {})]

                if first_scores and second_scores:
                    first_avg = statistics.mean(first_scores)
                    second_avg = statistics.mean(second_scores)
                    diff = round(second_avg - first_avg, 1)
                    if diff > 0.5:
                        trend_direction[dim] = f"improving (+{diff})"
                    elif diff < -0.5:
                        trend_direction[dim] = f"declining ({diff})"
                    else:
                        trend_direction[dim] = "stable"

        return SkillResult(
            success=True,
            message=f"Trends across {len(sorted_days)} days",
            data={
                "daily_trends": trend_data,
                "trend_direction": trend_direction,
                "days_tracked": len(sorted_days),
                "total_evaluations": len(evaluations),
            },
        )

    def _report(self, params: Dict) -> SkillResult:
        """Generate comprehensive self-improvement report."""
        data = self._load()
        evaluations = data.get("evaluations", [])
        directives = data.get("directives", [])
        improvement_log = data.get("improvement_log", [])

        if not evaluations:
            return SkillResult(
                success=False,
                message="No evaluations recorded. Use self_eval:evaluate first.",
            )

        # Run analysis
        analysis_result = self._analyze({"window_hours": 720})  # 30 days
        analysis = analysis_result.data if analysis_result.success else {}

        # Directive stats
        active_directives = [d for d in directives if d.get("status") == "active"]
        completed_directives = [d for d in directives if d.get("status") == "completed"]
        improved_count = len([d for d in completed_directives if d.get("outcome") == "improved"])
        no_change_count = len([d for d in completed_directives if d.get("outcome") == "no_change"])
        worse_count = len([d for d in completed_directives if d.get("outcome") == "worse"])

        # Trend data
        trend_result = self._trend({})
        trend_info = trend_result.data if trend_result.success else {}

        report = {
            "summary": {
                "total_evaluations": len(evaluations),
                "overall_average": analysis.get("overall_average", "N/A"),
                "strengths": analysis.get("strengths", []),
                "weaknesses": analysis.get("weaknesses", []),
            },
            "dimensions": analysis.get("dimension_stats", {}),
            "directives": {
                "active": len(active_directives),
                "completed": len(completed_directives),
                "improvement_rate": (
                    round(improved_count / len(completed_directives) * 100, 1)
                    if completed_directives else 0
                ),
                "outcomes": {
                    "improved": improved_count,
                    "no_change": no_change_count,
                    "worse": worse_count,
                },
            },
            "trends": trend_info.get("trend_direction", {}),
            "active_directives": [
                {"id": d["id"], "dimension": d.get("target_dimension"), "instruction": d.get("instruction")}
                for d in active_directives
            ],
            "skill_performance": analysis.get("skill_stats", {}),
        }

        return SkillResult(
            success=True,
            message=(
                f"Self-improvement report: {len(evaluations)} evaluations, "
                f"overall {analysis.get('overall_average', '?')}/10, "
                f"{len(active_directives)} active directives"
            ),
            data=report,
        )

    def _reset(self, params: Dict) -> SkillResult:
        """Clear all evaluations and start fresh."""
        data = self._load()
        count = len(data.get("evaluations", []))
        self._save(self._default_state())
        return SkillResult(
            success=True,
            message=f"Reset complete. Cleared {count} evaluations.",
            data={"cleared": count},
        )
