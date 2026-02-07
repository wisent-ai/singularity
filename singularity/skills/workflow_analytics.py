#!/usr/bin/env python3
"""
WorkflowAnalyticsSkill - Action Sequence Pattern Analysis

Analyzes chains of agent actions to discover which multi-step workflows
succeed or fail. Unlike per-action tracking (OutcomeTracker), this skill
looks at SEQUENCES: which chains of actions lead to success, which
combinations are anti-patterns, and what the optimal ordering is.

This is the "pattern recognition" layer in the self-improvement loop:
  OutcomeTracker (raw data) → WorkflowAnalytics (patterns) → AdaptiveExecutor (decisions)

Key capabilities:
1. Record action sequences as named workflows with outcomes
2. Discover frequent successful and failing action patterns (n-grams)
3. Suggest optimal next actions given a partial workflow
4. Detect anti-patterns: action combinations that correlate with failure
5. Generate reusable workflow templates from high-success patterns

Part of the Self-Improvement pillar: pattern recognition for the feedback loop.
"""

import json
from collections import defaultdict, Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from .base import Skill, SkillManifest, SkillAction, SkillResult


WORKFLOWS_FILE = Path(__file__).parent.parent / "data" / "workflow_analytics.json"


class WorkflowAnalyticsSkill(Skill):
    """
    Analyzes action sequences to discover successful workflow patterns.

    Records workflows (ordered sequences of skill:action calls), computes
    n-gram frequencies, identifies patterns correlated with success/failure,
    and recommends next actions based on historical data.
    """

    def __init__(self, credentials: Dict[str, str] = None):
        super().__init__(credentials)
        self._ensure_data()
        self._current_workflow: Optional[Dict] = None

    def _ensure_data(self):
        WORKFLOWS_FILE.parent.mkdir(parents=True, exist_ok=True)
        if not WORKFLOWS_FILE.exists():
            self._save({
                "workflows": [],
                "templates": [],
                "metadata": {
                    "created_at": datetime.now().isoformat(),
                    "total_workflows": 0,
                    "total_successes": 0,
                    "total_failures": 0,
                }
            })

    def _load(self) -> Dict:
        try:
            with open(WORKFLOWS_FILE, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {
                "workflows": [],
                "templates": [],
                "metadata": {
                    "created_at": datetime.now().isoformat(),
                    "total_workflows": 0,
                    "total_successes": 0,
                    "total_failures": 0,
                }
            }

    def _save(self, data: Dict):
        with open(WORKFLOWS_FILE, "w") as f:
            json.dump(data, f, indent=2, default=str)

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="workflow_analytics",
            name="Workflow Analytics",
            version="1.0.0",
            category="meta",
            description="Analyze action sequences to discover successful workflow patterns",
            required_credentials=[],
            install_cost=0,
            actions=[
                SkillAction(
                    name="start_workflow",
                    description="Begin recording a new action sequence (workflow)",
                    parameters={
                        "name": {
                            "type": "string",
                            "required": True,
                            "description": "Name/label for this workflow (e.g. 'deploy_service', 'fix_bug')"
                        },
                        "goal": {
                            "type": "string",
                            "required": False,
                            "description": "What this workflow aims to achieve"
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="record_step",
                    description="Record an action step in the current workflow",
                    parameters={
                        "action": {
                            "type": "string",
                            "required": True,
                            "description": "Action identifier (e.g. 'github:create_pr')"
                        },
                        "success": {
                            "type": "boolean",
                            "required": True,
                            "description": "Whether this step succeeded"
                        },
                        "duration_ms": {
                            "type": "number",
                            "required": False,
                            "description": "Step execution time in ms"
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="end_workflow",
                    description="Complete the current workflow and record its overall outcome",
                    parameters={
                        "success": {
                            "type": "boolean",
                            "required": True,
                            "description": "Whether the overall workflow succeeded"
                        },
                        "notes": {
                            "type": "string",
                            "required": False,
                            "description": "Notes about what happened"
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="suggest_next",
                    description="Given a partial action sequence, suggest the best next action",
                    parameters={
                        "recent_actions": {
                            "type": "array",
                            "required": True,
                            "description": "List of recent action identifiers in order"
                        },
                        "top_k": {
                            "type": "integer",
                            "required": False,
                            "description": "Number of suggestions to return (default: 3)"
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="find_patterns",
                    description="Discover frequent action patterns (n-grams) and their success rates",
                    parameters={
                        "min_occurrences": {
                            "type": "integer",
                            "required": False,
                            "description": "Minimum times a pattern must appear (default: 2)"
                        },
                        "max_length": {
                            "type": "integer",
                            "required": False,
                            "description": "Maximum pattern length to search (default: 4)"
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="anti_patterns",
                    description="Identify action combinations that correlate with workflow failure",
                    parameters={
                        "min_occurrences": {
                            "type": "integer",
                            "required": False,
                            "description": "Minimum occurrences to consider (default: 2)"
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="save_template",
                    description="Save a successful workflow as a reusable template",
                    parameters={
                        "name": {
                            "type": "string",
                            "required": True,
                            "description": "Template name"
                        },
                        "steps": {
                            "type": "array",
                            "required": True,
                            "description": "Ordered list of action identifiers"
                        },
                        "description": {
                            "type": "string",
                            "required": False,
                            "description": "What this template accomplishes"
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="list_templates",
                    description="List all saved workflow templates",
                    parameters={},
                    estimated_cost=0,
                ),
                SkillAction(
                    name="summary",
                    description="Get an overview of workflow analytics: patterns, success rates, insights",
                    parameters={},
                    estimated_cost=0,
                ),
            ],
        )

    def check_credentials(self) -> bool:
        return True

    async def execute(self, action: str, params: Dict) -> SkillResult:
        handlers = {
            "start_workflow": self._start_workflow,
            "record_step": self._record_step,
            "end_workflow": self._end_workflow,
            "suggest_next": self._suggest_next,
            "find_patterns": self._find_patterns,
            "anti_patterns": self._anti_patterns,
            "save_template": self._save_template,
            "list_templates": self._list_templates,
            "summary": self._summary,
        }

        handler = handlers.get(action)
        if handler:
            return await handler(params)
        return SkillResult(success=False, message=f"Unknown action: {action}")

    # === Workflow Recording ===

    async def _start_workflow(self, params: Dict) -> SkillResult:
        name = params.get("name", "").strip()
        if not name:
            return SkillResult(success=False, message="Workflow name required")

        if self._current_workflow:
            return SkillResult(
                success=False,
                message=f"Workflow '{self._current_workflow['name']}' already in progress. End it first."
            )

        self._current_workflow = {
            "name": name,
            "goal": params.get("goal", ""),
            "steps": [],
            "started_at": datetime.now().isoformat(),
        }

        return SkillResult(
            success=True,
            message=f"Started workflow '{name}'",
            data={"workflow": self._current_workflow}
        )

    async def _record_step(self, params: Dict) -> SkillResult:
        action = params.get("action", "").strip()
        success = params.get("success", True)

        if not action:
            return SkillResult(success=False, message="Action identifier required")

        if not self._current_workflow:
            return SkillResult(
                success=False,
                message="No workflow in progress. Call start_workflow first."
            )

        step = {
            "action": action,
            "success": bool(success),
            "duration_ms": params.get("duration_ms", 0),
            "timestamp": datetime.now().isoformat(),
            "step_number": len(self._current_workflow["steps"]) + 1,
        }

        self._current_workflow["steps"].append(step)

        return SkillResult(
            success=True,
            message=f"Recorded step {step['step_number']}: {action} ({'ok' if success else 'FAIL'})",
            data={"step": step, "total_steps": len(self._current_workflow["steps"])}
        )

    async def _end_workflow(self, params: Dict) -> SkillResult:
        success = params.get("success", False)

        if not self._current_workflow:
            return SkillResult(
                success=False,
                message="No workflow in progress"
            )

        workflow = {
            **self._current_workflow,
            "success": bool(success),
            "ended_at": datetime.now().isoformat(),
            "notes": params.get("notes", ""),
            "total_steps": len(self._current_workflow["steps"]),
            "failed_steps": sum(
                1 for s in self._current_workflow["steps"] if not s.get("success")
            ),
        }

        # Extract action sequence for pattern analysis
        workflow["action_sequence"] = [
            s["action"] for s in self._current_workflow["steps"]
        ]

        # Persist
        data = self._load()
        data["workflows"].append(workflow)
        data["metadata"]["total_workflows"] = data["metadata"].get("total_workflows", 0) + 1
        if success:
            data["metadata"]["total_successes"] = data["metadata"].get("total_successes", 0) + 1
        else:
            data["metadata"]["total_failures"] = data["metadata"].get("total_failures", 0) + 1
        self._save(data)

        self._current_workflow = None

        return SkillResult(
            success=True,
            message=f"Workflow completed: {'SUCCESS' if success else 'FAILURE'} ({workflow['total_steps']} steps)",
            data={"workflow_summary": workflow}
        )

    # === Pattern Discovery ===

    def _extract_ngrams(self, sequence: List[str], n: int) -> List[Tuple[str, ...]]:
        """Extract n-grams from an action sequence."""
        if len(sequence) < n:
            return []
        return [tuple(sequence[i:i + n]) for i in range(len(sequence) - n + 1)]

    async def _find_patterns(self, params: Dict) -> SkillResult:
        """Discover frequent action patterns and their success rates."""
        min_occ = params.get("min_occurrences", 2)
        max_length = params.get("max_length", 4)

        data = self._load()
        workflows = data.get("workflows", [])

        if len(workflows) < 2:
            return SkillResult(
                success=True,
                message="Need at least 2 workflows for pattern discovery",
                data={"patterns": [], "reason": "insufficient_data"}
            )

        # Extract n-grams from all workflows, tagged with outcome
        pattern_stats: Dict[Tuple, Dict] = defaultdict(
            lambda: {"count": 0, "successes": 0, "failures": 0}
        )

        for wf in workflows:
            seq = wf.get("action_sequence", [])
            wf_success = wf.get("success", False)

            for n in range(2, min(max_length + 1, len(seq) + 1)):
                ngrams = self._extract_ngrams(seq, n)
                # Count each unique ngram only once per workflow
                seen = set()
                for ng in ngrams:
                    if ng not in seen:
                        seen.add(ng)
                        pattern_stats[ng]["count"] += 1
                        if wf_success:
                            pattern_stats[ng]["successes"] += 1
                        else:
                            pattern_stats[ng]["failures"] += 1

        # Filter by min occurrences and compute success rates
        patterns = []
        for ngram, stats in pattern_stats.items():
            if stats["count"] >= min_occ:
                total = stats["count"]
                success_rate = round(stats["successes"] / total * 100, 1) if total else 0
                patterns.append({
                    "pattern": list(ngram),
                    "length": len(ngram),
                    "occurrences": total,
                    "successes": stats["successes"],
                    "failures": stats["failures"],
                    "success_rate": success_rate,
                })

        # Sort by frequency * success_rate (most useful patterns first)
        patterns.sort(key=lambda p: p["occurrences"] * p["success_rate"], reverse=True)

        return SkillResult(
            success=True,
            message=f"Found {len(patterns)} recurring patterns across {len(workflows)} workflows",
            data={
                "patterns": patterns[:20],  # Top 20
                "total_patterns_found": len(patterns),
                "workflows_analyzed": len(workflows),
            }
        )

    async def _anti_patterns(self, params: Dict) -> SkillResult:
        """Identify action combinations that correlate with failure."""
        min_occ = params.get("min_occurrences", 2)

        data = self._load()
        workflows = data.get("workflows", [])

        if len(workflows) < 2:
            return SkillResult(
                success=True,
                message="Need at least 2 workflows for anti-pattern detection",
                data={"anti_patterns": [], "reason": "insufficient_data"}
            )

        # Find patterns that appear mostly in failed workflows
        pattern_stats: Dict[Tuple, Dict] = defaultdict(
            lambda: {"count": 0, "successes": 0, "failures": 0}
        )

        for wf in workflows:
            seq = wf.get("action_sequence", [])
            wf_success = wf.get("success", False)

            # Check bigrams and trigrams
            for n in range(2, min(4, len(seq) + 1)):
                ngrams = self._extract_ngrams(seq, n)
                seen = set()
                for ng in ngrams:
                    if ng not in seen:
                        seen.add(ng)
                        pattern_stats[ng]["count"] += 1
                        if wf_success:
                            pattern_stats[ng]["successes"] += 1
                        else:
                            pattern_stats[ng]["failures"] += 1

        # Anti-patterns: high failure rate patterns
        anti_patterns = []
        for ngram, stats in pattern_stats.items():
            if stats["count"] >= min_occ:
                total = stats["count"]
                failure_rate = round(stats["failures"] / total * 100, 1)
                if failure_rate >= 60:  # 60%+ failure rate = anti-pattern
                    anti_patterns.append({
                        "pattern": list(ngram),
                        "length": len(ngram),
                        "occurrences": total,
                        "failure_rate": failure_rate,
                        "failures": stats["failures"],
                        "successes": stats["successes"],
                        "severity": "high" if failure_rate >= 80 else "medium",
                    })

        # Also check for step-level failures within workflows
        step_failure_rates: Dict[str, Dict] = defaultdict(
            lambda: {"total": 0, "failures": 0}
        )
        for wf in workflows:
            for step in wf.get("steps", []):
                action = step.get("action", "")
                step_failure_rates[action]["total"] += 1
                if not step.get("success"):
                    step_failure_rates[action]["failures"] += 1

        failing_actions = []
        for action, stats in step_failure_rates.items():
            if stats["total"] >= min_occ:
                fail_rate = round(stats["failures"] / stats["total"] * 100, 1)
                if fail_rate >= 50:
                    failing_actions.append({
                        "action": action,
                        "total": stats["total"],
                        "failures": stats["failures"],
                        "failure_rate": fail_rate,
                    })

        anti_patterns.sort(key=lambda p: p["failure_rate"], reverse=True)
        failing_actions.sort(key=lambda a: a["failure_rate"], reverse=True)

        return SkillResult(
            success=True,
            message=f"Found {len(anti_patterns)} anti-patterns and {len(failing_actions)} unreliable actions",
            data={
                "anti_patterns": anti_patterns[:10],
                "unreliable_actions": failing_actions[:10],
                "workflows_analyzed": len(workflows),
            }
        )

    # === Next Action Suggestion ===

    async def _suggest_next(self, params: Dict) -> SkillResult:
        """Suggest the best next action given recent actions."""
        recent = params.get("recent_actions", [])
        top_k = params.get("top_k", 3)

        if not recent:
            return SkillResult(
                success=False,
                message="Provide recent_actions list to get suggestions"
            )

        data = self._load()
        workflows = data.get("workflows", [])

        if not workflows:
            return SkillResult(
                success=True,
                message="No workflow history for suggestions",
                data={"suggestions": [], "reason": "no_history"}
            )

        # Find what actions typically follow the recent sequence
        # Look at successively shorter suffixes of recent_actions
        next_action_scores: Dict[str, Dict] = defaultdict(
            lambda: {"count": 0, "success_count": 0}
        )

        for suffix_len in range(min(len(recent), 3), 0, -1):
            suffix = recent[-suffix_len:]

            for wf in workflows:
                seq = wf.get("action_sequence", [])
                wf_success = wf.get("success", False)

                # Search for the suffix in this workflow's sequence
                for i in range(len(seq) - suffix_len):
                    if seq[i:i + suffix_len] == suffix and i + suffix_len < len(seq):
                        next_action = seq[i + suffix_len]
                        # Weight by match length (longer matches more valuable)
                        weight = suffix_len
                        next_action_scores[next_action]["count"] += weight
                        if wf_success:
                            next_action_scores[next_action]["success_count"] += weight

        if not next_action_scores:
            return SkillResult(
                success=True,
                message="No matching patterns found in history",
                data={"suggestions": [], "context": recent}
            )

        # Score: weighted combination of frequency and success rate
        suggestions = []
        for action, scores in next_action_scores.items():
            count = scores["count"]
            success_rate = (
                round(scores["success_count"] / count * 100, 1) if count else 0
            )
            # Composite score: frequency * success_rate
            composite = count * (success_rate / 100) if success_rate > 0 else 0

            suggestions.append({
                "action": action,
                "confidence_score": round(composite, 2),
                "frequency": count,
                "success_rate": success_rate,
            })

        suggestions.sort(key=lambda s: s["confidence_score"], reverse=True)

        return SkillResult(
            success=True,
            message=f"Top suggestion: {suggestions[0]['action']} (confidence: {suggestions[0]['confidence_score']})",
            data={
                "suggestions": suggestions[:top_k],
                "based_on_context": recent,
                "workflows_searched": len(workflows),
            }
        )

    # === Templates ===

    async def _save_template(self, params: Dict) -> SkillResult:
        name = params.get("name", "").strip()
        steps = params.get("steps", [])

        if not name:
            return SkillResult(success=False, message="Template name required")
        if not steps or not isinstance(steps, list):
            return SkillResult(success=False, message="Steps list required")

        template = {
            "name": name,
            "steps": steps,
            "description": params.get("description", ""),
            "created_at": datetime.now().isoformat(),
            "step_count": len(steps),
        }

        data = self._load()
        # Check for duplicate names
        existing_names = {t["name"] for t in data.get("templates", [])}
        if name in existing_names:
            # Update existing
            data["templates"] = [
                t if t["name"] != name else template
                for t in data["templates"]
            ]
        else:
            data.setdefault("templates", []).append(template)
        self._save(data)

        return SkillResult(
            success=True,
            message=f"Template '{name}' saved ({len(steps)} steps)",
            data={"template": template}
        )

    async def _list_templates(self, params: Dict) -> SkillResult:
        data = self._load()
        templates = data.get("templates", [])

        return SkillResult(
            success=True,
            message=f"{len(templates)} workflow template(s) available",
            data={"templates": templates}
        )

    # === Summary / Insights ===

    async def _summary(self, params: Dict) -> SkillResult:
        data = self._load()
        workflows = data.get("workflows", [])
        templates = data.get("templates", [])
        metadata = data.get("metadata", {})

        if not workflows:
            return SkillResult(
                success=True,
                message="No workflows recorded yet",
                data={"total_workflows": 0}
            )

        total = len(workflows)
        successes = sum(1 for w in workflows if w.get("success"))
        failures = total - successes
        success_rate = round(successes / total * 100, 1) if total else 0

        # Average workflow length
        avg_steps = round(
            sum(w.get("total_steps", 0) for w in workflows) / total, 1
        ) if total else 0

        # Most common workflow names
        name_counts = Counter(w.get("name", "unknown") for w in workflows)
        common_workflows = [
            {"name": name, "count": count}
            for name, count in name_counts.most_common(5)
        ]

        # Most used actions across all workflows
        action_counts = Counter()
        for wf in workflows:
            for action in wf.get("action_sequence", []):
                action_counts[action] += 1
        top_actions = [
            {"action": action, "usage_count": count}
            for action, count in action_counts.most_common(10)
        ]

        # Success rate by workflow name
        name_outcomes: Dict[str, Dict] = defaultdict(
            lambda: {"total": 0, "successes": 0}
        )
        for wf in workflows:
            name = wf.get("name", "unknown")
            name_outcomes[name]["total"] += 1
            if wf.get("success"):
                name_outcomes[name]["successes"] += 1

        workflow_success_rates = []
        for name, stats in name_outcomes.items():
            if stats["total"] >= 2:
                rate = round(stats["successes"] / stats["total"] * 100, 1)
                workflow_success_rates.append({
                    "name": name,
                    "total": stats["total"],
                    "success_rate": rate,
                })
        workflow_success_rates.sort(key=lambda w: w["success_rate"], reverse=True)

        insights = []
        if success_rate < 50:
            insights.append("Overall workflow success rate is below 50% - review anti-patterns")
        if avg_steps > 10:
            insights.append("Workflows are long (avg >10 steps) - consider breaking into sub-workflows")
        if failures > successes:
            insights.append("More workflows fail than succeed - focus on reliability before new capabilities")

        return SkillResult(
            success=True,
            message=f"Workflow analytics: {successes}/{total} workflows succeeded ({success_rate}%)",
            data={
                "total_workflows": total,
                "successes": successes,
                "failures": failures,
                "success_rate": success_rate,
                "avg_steps_per_workflow": avg_steps,
                "common_workflows": common_workflows,
                "top_actions": top_actions,
                "workflow_success_rates": workflow_success_rates,
                "templates_available": len(templates),
                "insights": insights,
            }
        )
