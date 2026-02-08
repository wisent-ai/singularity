#!/usr/bin/env python3
"""
PerformanceOptimizerSkill - Autonomous closed-loop self-improvement engine.

This skill bridges outcome tracking and self-modification to create a fully
autonomous improvement cycle:

1. ANALYZE: Examines recent execution outcomes to detect failure patterns
2. DIAGNOSE: Groups failures by skill/action and identifies root causes
3. HYPOTHESIZE: Generates improvement hypotheses with expected impact
4. APPLY: Creates targeted prompt modifications via PromptEvolution
5. MEASURE: Tracks whether modifications improved performance
6. REVERT: Auto-reverts changes that made things worse

This is the critical "brain" that makes the act → measure → adapt loop
actually autonomous. Without it, the agent has data (OutcomeTracker) and
tools (SelfModify, PromptEvolution) but no intelligence connecting them.

Pillar: Self-Improvement (the autonomous optimizer)
"""

import json
import time
import os
from typing import Dict, List, Optional, Any, Callable
from collections import defaultdict
from .base import Skill, SkillManifest, SkillAction, SkillResult


class ImprovementCycle:
    """Tracks a single improvement attempt from hypothesis to outcome."""

    def __init__(
        self,
        cycle_id: str,
        hypothesis: str,
        target_skill: str,
        target_action: str,
        modification: str,
        baseline_success_rate: float,
        baseline_sample_size: int,
    ):
        self.cycle_id = cycle_id
        self.hypothesis = hypothesis
        self.target_skill = target_skill
        self.target_action = target_action
        self.modification = modification
        self.baseline_success_rate = baseline_success_rate
        self.baseline_sample_size = baseline_sample_size
        self.created_at = time.time()
        self.applied_at: Optional[float] = None
        self.concluded_at: Optional[float] = None
        self.post_outcomes: List[Dict] = []
        self.status: str = "proposed"  # proposed, active, concluded, reverted
        self.result: Optional[str] = None  # improved, degraded, neutral
        self.prompt_version_label: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "cycle_id": self.cycle_id,
            "hypothesis": self.hypothesis,
            "target_skill": self.target_skill,
            "target_action": self.target_action,
            "modification": self.modification,
            "baseline_success_rate": self.baseline_success_rate,
            "baseline_sample_size": self.baseline_sample_size,
            "created_at": self.created_at,
            "applied_at": self.applied_at,
            "concluded_at": self.concluded_at,
            "post_outcomes": self.post_outcomes,
            "status": self.status,
            "result": self.result,
            "prompt_version_label": self.prompt_version_label,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "ImprovementCycle":
        cycle = cls(
            cycle_id=d["cycle_id"],
            hypothesis=d["hypothesis"],
            target_skill=d["target_skill"],
            target_action=d["target_action"],
            modification=d["modification"],
            baseline_success_rate=d["baseline_success_rate"],
            baseline_sample_size=d["baseline_sample_size"],
        )
        cycle.created_at = d.get("created_at", time.time())
        cycle.applied_at = d.get("applied_at")
        cycle.concluded_at = d.get("concluded_at")
        cycle.post_outcomes = d.get("post_outcomes", [])
        cycle.status = d.get("status", "proposed")
        cycle.result = d.get("result")
        cycle.prompt_version_label = d.get("prompt_version_label")
        return cycle

    @property
    def post_success_rate(self) -> Optional[float]:
        if not self.post_outcomes:
            return None
        successes = sum(1 for o in self.post_outcomes if o.get("outcome") == "success")
        return successes / len(self.post_outcomes)

    @property
    def improvement_delta(self) -> Optional[float]:
        post = self.post_success_rate
        if post is None:
            return None
        return post - self.baseline_success_rate


class PerformanceOptimizerSkill(Skill):
    """Autonomous closed-loop performance optimizer."""

    def __init__(self, credentials: Dict = None):
        super().__init__(credentials)
        self._storage_dir = os.path.join(
            os.path.expanduser("~"), ".singularity", "performance_optimizer"
        )
        os.makedirs(self._storage_dir, exist_ok=True)
        self._cycles_file = os.path.join(self._storage_dir, "cycles.json")
        self._analysis_file = os.path.join(self._storage_dir, "analysis_history.json")
        self._config_file = os.path.join(self._storage_dir, "config.json")

        # Hooks
        self._get_outcomes_fn: Optional[Callable[[], List[Dict]]] = None
        self._get_prompt_fn: Optional[Callable[[], str]] = None
        self._set_prompt_fn: Optional[Callable[[str], None]] = None
        self._next_cycle_id = 1

        # Load existing state
        self._load_state()

    def set_hooks(
        self,
        get_outcomes: Callable[[], List[Dict]],
        get_prompt: Callable[[], str] = None,
        set_prompt: Callable[[str], None] = None,
    ):
        """Connect to outcome data and prompt modification functions."""
        self._get_outcomes_fn = get_outcomes
        self._get_prompt_fn = get_prompt
        self._set_prompt_fn = set_prompt

    # --- Persistence ---

    def _load_state(self):
        config = self._load_json(self._config_file, {})
        self._next_cycle_id = config.get("next_cycle_id", 1)
        self._auto_revert_threshold = config.get("auto_revert_threshold", -0.1)
        self._min_outcomes_for_analysis = config.get("min_outcomes_for_analysis", 5)
        self._min_post_outcomes_to_conclude = config.get("min_post_outcomes_to_conclude", 5)

    def _save_state(self):
        self._save_json(self._config_file, {
            "next_cycle_id": self._next_cycle_id,
            "auto_revert_threshold": self._auto_revert_threshold,
            "min_outcomes_for_analysis": self._min_outcomes_for_analysis,
            "min_post_outcomes_to_conclude": self._min_post_outcomes_to_conclude,
        })

    def _load_json(self, path: str, default: Any = None) -> Any:
        if default is None:
            default = {}
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return default
        return default

    def _save_json(self, path: str, data: Any):
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def _load_cycles(self) -> List[ImprovementCycle]:
        data = self._load_json(self._cycles_file, [])
        return [ImprovementCycle.from_dict(d) for d in data]

    def _save_cycles(self, cycles: List[ImprovementCycle]):
        self._save_json(self._cycles_file, [c.to_dict() for c in cycles])

    def _load_analysis_history(self) -> List[Dict]:
        return self._load_json(self._analysis_file, [])

    def _save_analysis_history(self, history: List[Dict]):
        self._save_json(self._analysis_file, history)

    # --- Manifest ---

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="performance_optimizer",
            name="Performance Optimizer",
            version="1.0.0",
            category="meta",
            description="Autonomous closed-loop self-improvement: analyze failures, hypothesize fixes, apply and measure",
            actions=[
                SkillAction(
                    name="analyze",
                    description="Analyze recent outcomes to identify failure patterns and bottlenecks",
                    parameters={
                        "window": {
                            "type": "integer",
                            "required": False,
                            "description": "Number of recent outcomes to analyze (default: 50)",
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="hypothesize",
                    description="Generate improvement hypotheses from the latest analysis",
                    parameters={
                        "max_hypotheses": {
                            "type": "integer",
                            "required": False,
                            "description": "Maximum hypotheses to generate (default: 3)",
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="propose_cycle",
                    description="Create an improvement cycle with a specific hypothesis and prompt modification",
                    parameters={
                        "hypothesis": {
                            "type": "string",
                            "required": True,
                            "description": "What you think will improve performance",
                        },
                        "target_skill": {
                            "type": "string",
                            "required": True,
                            "description": "Which skill/action area to improve",
                        },
                        "modification": {
                            "type": "string",
                            "required": True,
                            "description": "The prompt text to append/modify",
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="apply_cycle",
                    description="Apply a proposed improvement cycle (modifies the prompt)",
                    parameters={
                        "cycle_id": {
                            "type": "string",
                            "required": True,
                            "description": "ID of the cycle to apply",
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="record_post_outcome",
                    description="Record an outcome after applying an improvement cycle",
                    parameters={
                        "cycle_id": {
                            "type": "string",
                            "required": True,
                            "description": "ID of the active cycle",
                        },
                        "outcome": {
                            "type": "string",
                            "required": True,
                            "description": "'success' or 'failure'",
                        },
                        "context": {
                            "type": "string",
                            "required": False,
                            "description": "What task was attempted",
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="evaluate_cycle",
                    description="Evaluate if an active improvement cycle helped or hurt",
                    parameters={
                        "cycle_id": {
                            "type": "string",
                            "required": True,
                            "description": "ID of the cycle to evaluate",
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="revert_cycle",
                    description="Revert a cycle that degraded performance",
                    parameters={
                        "cycle_id": {
                            "type": "string",
                            "required": True,
                            "description": "ID of the cycle to revert",
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="list_cycles",
                    description="List all improvement cycles and their status/results",
                    parameters={
                        "status_filter": {
                            "type": "string",
                            "required": False,
                            "description": "Filter by status: proposed, active, concluded, reverted",
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="improvement_report",
                    description="Generate a summary report of all optimization efforts and net impact",
                    parameters={},
                    estimated_cost=0,
                ),
                SkillAction(
                    name="auto_optimize",
                    description="Run a full optimization pass: analyze → hypothesize → propose the best cycle",
                    parameters={
                        "window": {
                            "type": "integer",
                            "required": False,
                            "description": "Number of recent outcomes to analyze (default: 50)",
                        },
                    },
                    estimated_cost=0,
                ),
            ],
            required_credentials=[],
        )

    def check_credentials(self) -> bool:
        return self._get_outcomes_fn is not None

    async def execute(self, action: str, params: Dict) -> SkillResult:
        handlers = {
            "analyze": self._analyze,
            "hypothesize": self._hypothesize,
            "propose_cycle": self._propose_cycle,
            "apply_cycle": self._apply_cycle,
            "record_post_outcome": self._record_post_outcome,
            "evaluate_cycle": self._evaluate_cycle,
            "revert_cycle": self._revert_cycle,
            "list_cycles": self._list_cycles,
            "improvement_report": self._improvement_report,
            "auto_optimize": self._auto_optimize,
        }
        handler = handlers.get(action)
        if not handler:
            return SkillResult(success=False, message=f"Unknown action: {action}")
        return handler(params)

    # --- Analysis ---

    def _get_recent_outcomes(self, window: int = 50) -> List[Dict]:
        """Get recent outcomes from the outcome tracker."""
        if not self._get_outcomes_fn:
            return []
        all_outcomes = self._get_outcomes_fn()
        return all_outcomes[-window:] if len(all_outcomes) > window else all_outcomes

    def _compute_skill_stats(self, outcomes: List[Dict]) -> Dict[str, Dict]:
        """Compute per-skill performance statistics."""
        stats = defaultdict(lambda: {
            "total": 0, "successes": 0, "failures": 0,
            "actions": defaultdict(lambda: {"total": 0, "successes": 0, "failures": 0}),
            "failure_contexts": [],
        })

        for o in outcomes:
            skill = o.get("skill", "unknown")
            action = o.get("action", "unknown")
            outcome = o.get("outcome", "unknown")

            stats[skill]["total"] += 1
            stats[skill]["actions"][action]["total"] += 1

            if outcome == "success":
                stats[skill]["successes"] += 1
                stats[skill]["actions"][action]["successes"] += 1
            elif outcome == "failure":
                stats[skill]["failures"] += 1
                stats[skill]["actions"][action]["failures"] += 1
                context = o.get("context", o.get("error", ""))
                if context:
                    stats[skill]["failure_contexts"].append({
                        "action": action,
                        "context": context[:200],
                        "timestamp": o.get("timestamp", 0),
                    })

        # Compute rates
        result = {}
        for skill, s in stats.items():
            total = s["total"]
            success_rate = s["successes"] / total if total > 0 else 0
            actions = {}
            for act, a in s["actions"].items():
                act_total = a["total"]
                actions[act] = {
                    "total": act_total,
                    "successes": a["successes"],
                    "failures": a["failures"],
                    "success_rate": a["successes"] / act_total if act_total > 0 else 0,
                }
            result[skill] = {
                "total": total,
                "successes": s["successes"],
                "failures": s["failures"],
                "success_rate": round(success_rate, 3),
                "actions": actions,
                "failure_contexts": s["failure_contexts"][-5:],  # Keep last 5
            }

        return result

    def _identify_bottlenecks(self, skill_stats: Dict[str, Dict]) -> List[Dict]:
        """Identify the worst-performing skills/actions."""
        bottlenecks = []

        for skill, stats in skill_stats.items():
            if stats["total"] < 2:
                continue  # Not enough data

            # Skill-level bottleneck
            if stats["success_rate"] < 0.7:
                bottlenecks.append({
                    "level": "skill",
                    "skill": skill,
                    "action": None,
                    "success_rate": stats["success_rate"],
                    "total": stats["total"],
                    "failures": stats["failures"],
                    "severity": round(1.0 - stats["success_rate"], 3),
                    "sample_contexts": stats["failure_contexts"],
                })

            # Action-level bottlenecks
            for action, a_stats in stats["actions"].items():
                if a_stats["total"] < 2:
                    continue
                if a_stats["success_rate"] < 0.7:
                    bottlenecks.append({
                        "level": "action",
                        "skill": skill,
                        "action": action,
                        "success_rate": a_stats["success_rate"],
                        "total": a_stats["total"],
                        "failures": a_stats["failures"],
                        "severity": round(1.0 - a_stats["success_rate"], 3),
                        "sample_contexts": [
                            c for c in stats["failure_contexts"]
                            if c["action"] == action
                        ],
                    })

        # Sort by severity (worst first)
        bottlenecks.sort(key=lambda b: b["severity"], reverse=True)
        return bottlenecks

    def _analyze(self, params: Dict) -> SkillResult:
        window = int(params.get("window", 50))
        outcomes = self._get_recent_outcomes(window)

        if len(outcomes) < self._min_outcomes_for_analysis:
            return SkillResult(
                success=True,
                message=f"Insufficient data: {len(outcomes)} outcomes (need {self._min_outcomes_for_analysis})",
                data={
                    "outcomes_available": len(outcomes),
                    "min_required": self._min_outcomes_for_analysis,
                    "bottlenecks": [],
                    "overall_success_rate": None,
                },
            )

        skill_stats = self._compute_skill_stats(outcomes)
        bottlenecks = self._identify_bottlenecks(skill_stats)

        total = len(outcomes)
        successes = sum(1 for o in outcomes if o.get("outcome") == "success")
        overall_rate = round(successes / total, 3) if total > 0 else 0

        analysis = {
            "timestamp": time.time(),
            "window": window,
            "total_outcomes": total,
            "overall_success_rate": overall_rate,
            "skill_stats": skill_stats,
            "bottlenecks": bottlenecks,
        }

        # Persist analysis
        history = self._load_analysis_history()
        history.append(analysis)
        # Keep last 20 analyses
        if len(history) > 20:
            history = history[-20:]
        self._save_analysis_history(history)

        return SkillResult(
            success=True,
            message=f"Analyzed {total} outcomes: {overall_rate:.1%} success rate, {len(bottlenecks)} bottleneck(s)",
            data={
                "overall_success_rate": overall_rate,
                "total_outcomes": total,
                "bottlenecks": bottlenecks,
                "skills_analyzed": len(skill_stats),
            },
        )

    # --- Hypothesis Generation ---

    def _generate_hypotheses(self, bottlenecks: List[Dict], max_count: int = 3) -> List[Dict]:
        """Generate improvement hypotheses from bottleneck analysis."""
        hypotheses = []

        for bn in bottlenecks[:max_count]:
            skill = bn["skill"]
            action = bn.get("action", "general")
            rate = bn["success_rate"]
            contexts = bn.get("sample_contexts", [])

            # Extract common failure patterns from contexts
            context_summary = "; ".join(
                c.get("context", "unknown")[:80] for c in contexts[:3]
            )

            if rate == 0:
                priority = "critical"
                suggestion = f"Skill '{skill}' action '{action}' has 0% success. Add explicit instructions for handling this action type."
            elif rate < 0.3:
                priority = "high"
                suggestion = f"Skill '{skill}' action '{action}' fails {1-rate:.0%} of the time. Add error recovery instructions and alternative approaches."
            elif rate < 0.5:
                priority = "high"
                suggestion = f"Skill '{skill}' action '{action}' fails often. Add validation steps before execution."
            else:
                priority = "medium"
                suggestion = f"Skill '{skill}' action '{action}' could be improved. Add specific guidance for edge cases."

            hypothesis = {
                "priority": priority,
                "target_skill": skill,
                "target_action": action,
                "current_success_rate": rate,
                "hypothesis": suggestion,
                "evidence": context_summary or "No context available",
                "expected_improvement": min(0.3, 1.0 - rate),
                "suggested_modification": self._generate_modification(skill, action, rate, contexts),
            }
            hypotheses.append(hypothesis)

        return hypotheses

    def _generate_modification(
        self, skill: str, action: str, rate: float, contexts: List[Dict]
    ) -> str:
        """Generate a specific prompt modification suggestion."""
        context_hints = ""
        if contexts:
            errors = [c.get("context", "") for c in contexts[:3]]
            context_hints = " Common failure patterns: " + "; ".join(e for e in errors if e)

        if rate == 0:
            return (
                f"\n\n=== PERFORMANCE OPTIMIZATION ===\n"
                f"CRITICAL: The '{action}' action in '{skill}' skill has been failing consistently.\n"
                f"Before using {skill}.{action}, always:\n"
                f"1. Verify all required parameters are present and valid\n"
                f"2. Check prerequisites are met\n"
                f"3. Have a fallback plan if the action fails\n"
                f"{context_hints}"
            )
        elif rate < 0.5:
            return (
                f"\n\n=== PERFORMANCE OPTIMIZATION ===\n"
                f"WARNING: '{skill}.{action}' has a {rate:.0%} success rate.\n"
                f"To improve results:\n"
                f"1. Double-check inputs before calling this action\n"
                f"2. Use simpler approaches when possible\n"
                f"3. If it fails, try an alternative approach\n"
                f"{context_hints}"
            )
        else:
            return (
                f"\n\n=== PERFORMANCE OPTIMIZATION ===\n"
                f"TIP: '{skill}.{action}' can be improved (current: {rate:.0%}).\n"
                f"Consider edge cases and validate outputs.\n"
                f"{context_hints}"
            )

    def _hypothesize(self, params: Dict) -> SkillResult:
        max_hypotheses = int(params.get("max_hypotheses", 3))

        # Load latest analysis
        history = self._load_analysis_history()
        if not history:
            return SkillResult(
                success=False,
                message="No analysis available. Run 'analyze' first.",
            )

        latest = history[-1]
        bottlenecks = latest.get("bottlenecks", [])

        if not bottlenecks:
            return SkillResult(
                success=True,
                message="No bottlenecks found - performance looks good!",
                data={"hypotheses": [], "overall_success_rate": latest.get("overall_success_rate")},
            )

        hypotheses = self._generate_hypotheses(bottlenecks, max_hypotheses)

        return SkillResult(
            success=True,
            message=f"Generated {len(hypotheses)} improvement hypothesis(es)",
            data={
                "hypotheses": hypotheses,
                "based_on_analysis": latest.get("timestamp"),
                "overall_success_rate": latest.get("overall_success_rate"),
            },
        )

    # --- Improvement Cycles ---

    def _propose_cycle(self, params: Dict) -> SkillResult:
        hypothesis = params.get("hypothesis", "").strip()
        target_skill = params.get("target_skill", "").strip()
        modification = params.get("modification", "")

        if not hypothesis or not target_skill or not modification:
            return SkillResult(
                success=False,
                message="hypothesis, target_skill, and modification are all required",
            )

        # Get baseline stats for the target
        outcomes = self._get_recent_outcomes(50)
        skill_stats = self._compute_skill_stats(outcomes)
        target_stats = skill_stats.get(target_skill, {"success_rate": 0, "total": 0})

        cycle_id = f"cycle_{self._next_cycle_id}"
        self._next_cycle_id += 1
        self._save_state()

        target_action = params.get("target_action", "general")

        cycle = ImprovementCycle(
            cycle_id=cycle_id,
            hypothesis=hypothesis,
            target_skill=target_skill,
            target_action=target_action,
            modification=modification,
            baseline_success_rate=target_stats["success_rate"],
            baseline_sample_size=target_stats["total"],
        )

        cycles = self._load_cycles()
        cycles.append(cycle)
        self._save_cycles(cycles)

        return SkillResult(
            success=True,
            message=f"Proposed improvement cycle '{cycle_id}': {hypothesis[:80]}",
            data={
                "cycle_id": cycle_id,
                "hypothesis": hypothesis,
                "target_skill": target_skill,
                "baseline_success_rate": target_stats["success_rate"],
                "baseline_sample_size": target_stats["total"],
                "status": "proposed",
            },
        )

    def _apply_cycle(self, params: Dict) -> SkillResult:
        cycle_id = params.get("cycle_id", "").strip()
        if not cycle_id:
            return SkillResult(success=False, message="cycle_id is required")

        if not self._get_prompt_fn or not self._set_prompt_fn:
            return SkillResult(
                success=False,
                message="Prompt hooks not connected. Cannot apply modification.",
            )

        cycles = self._load_cycles()
        target = None
        for c in cycles:
            if c.cycle_id == cycle_id:
                target = c
                break

        if not target:
            return SkillResult(success=False, message=f"Cycle '{cycle_id}' not found")

        if target.status != "proposed":
            return SkillResult(
                success=False,
                message=f"Cycle '{cycle_id}' is '{target.status}', not 'proposed'",
            )

        # Check no other cycle is active
        active_cycles = [c for c in cycles if c.status == "active"]
        if active_cycles:
            return SkillResult(
                success=False,
                message=f"Another cycle is already active: {active_cycles[0].cycle_id}. Evaluate or revert it first.",
            )

        # Apply the modification
        current_prompt = self._get_prompt_fn()
        new_prompt = current_prompt + target.modification
        self._set_prompt_fn(new_prompt)

        target.status = "active"
        target.applied_at = time.time()
        target.prompt_version_label = f"optimizer_{cycle_id}"
        self._save_cycles(cycles)

        return SkillResult(
            success=True,
            message=f"Applied cycle '{cycle_id}'. Modification added to prompt. Record outcomes to measure impact.",
            data={
                "cycle_id": cycle_id,
                "status": "active",
                "modification_length": len(target.modification),
                "new_prompt_length": len(new_prompt),
            },
        )

    def _record_post_outcome(self, params: Dict) -> SkillResult:
        cycle_id = params.get("cycle_id", "").strip()
        outcome = params.get("outcome", "").strip().lower()
        context = params.get("context", "")

        if not cycle_id or outcome not in ("success", "failure"):
            return SkillResult(
                success=False,
                message="cycle_id and outcome ('success'/'failure') are required",
            )

        cycles = self._load_cycles()
        target = None
        for c in cycles:
            if c.cycle_id == cycle_id:
                target = c
                break

        if not target:
            return SkillResult(success=False, message=f"Cycle '{cycle_id}' not found")

        if target.status != "active":
            return SkillResult(
                success=False,
                message=f"Cycle '{cycle_id}' is '{target.status}', not 'active'",
            )

        target.post_outcomes.append({
            "outcome": outcome,
            "context": context,
            "timestamp": time.time(),
        })
        self._save_cycles(cycles)

        post_rate = target.post_success_rate
        delta = target.improvement_delta

        # Auto-revert if clearly degraded
        should_auto_revert = (
            len(target.post_outcomes) >= 3
            and delta is not None
            and delta < self._auto_revert_threshold
        )

        result_data = {
            "cycle_id": cycle_id,
            "post_outcomes": len(target.post_outcomes),
            "post_success_rate": round(post_rate, 3) if post_rate is not None else None,
            "baseline_success_rate": target.baseline_success_rate,
            "improvement_delta": round(delta, 3) if delta is not None else None,
            "auto_revert_triggered": should_auto_revert,
        }

        if should_auto_revert:
            # Auto-revert
            self._do_revert(target, cycles)
            return SkillResult(
                success=True,
                message=f"AUTO-REVERT: Cycle '{cycle_id}' degraded performance ({delta:+.1%}). Modification removed.",
                data=result_data,
            )

        return SkillResult(
            success=True,
            message=f"Recorded {outcome} for cycle '{cycle_id}' (post rate: {post_rate:.1%}, delta: {delta:+.1%})",
            data=result_data,
        )

    def _do_revert(self, cycle: ImprovementCycle, cycles: List[ImprovementCycle]):
        """Revert a cycle's prompt modification."""
        if self._get_prompt_fn and self._set_prompt_fn:
            current = self._get_prompt_fn()
            # Remove the modification
            if cycle.modification in current:
                reverted = current.replace(cycle.modification, "", 1)
                self._set_prompt_fn(reverted)

        cycle.status = "reverted"
        cycle.concluded_at = time.time()
        cycle.result = "degraded"
        self._save_cycles(cycles)

    def _evaluate_cycle(self, params: Dict) -> SkillResult:
        cycle_id = params.get("cycle_id", "").strip()
        if not cycle_id:
            return SkillResult(success=False, message="cycle_id is required")

        cycles = self._load_cycles()
        target = None
        for c in cycles:
            if c.cycle_id == cycle_id:
                target = c
                break

        if not target:
            return SkillResult(success=False, message=f"Cycle '{cycle_id}' not found")

        if target.status not in ("active",):
            return SkillResult(
                success=False,
                message=f"Cycle '{cycle_id}' is '{target.status}'. Only 'active' cycles can be evaluated.",
            )

        if len(target.post_outcomes) < self._min_post_outcomes_to_conclude:
            return SkillResult(
                success=True,
                message=f"Need more data: {len(target.post_outcomes)}/{self._min_post_outcomes_to_conclude} outcomes recorded",
                data={
                    "cycle_id": cycle_id,
                    "post_outcomes": len(target.post_outcomes),
                    "min_required": self._min_post_outcomes_to_conclude,
                    "preliminary_delta": round(target.improvement_delta, 3) if target.improvement_delta is not None else None,
                },
            )

        delta = target.improvement_delta
        if delta is None:
            return SkillResult(success=False, message="Cannot compute delta")

        # Determine result
        if delta > 0.05:
            result = "improved"
            recommendation = "Keep this modification - it improved performance."
        elif delta < -0.05:
            result = "degraded"
            recommendation = "Revert this modification - it hurt performance."
        else:
            result = "neutral"
            recommendation = "Marginal impact. Consider keeping if it helps with specific cases."

        target.status = "concluded"
        target.concluded_at = time.time()
        target.result = result
        self._save_cycles(cycles)

        return SkillResult(
            success=True,
            message=f"Cycle '{cycle_id}' concluded: {result} (delta: {delta:+.1%})",
            data={
                "cycle_id": cycle_id,
                "result": result,
                "baseline_success_rate": target.baseline_success_rate,
                "post_success_rate": target.post_success_rate,
                "improvement_delta": round(delta, 3),
                "post_outcomes_count": len(target.post_outcomes),
                "recommendation": recommendation,
            },
        )

    def _revert_cycle(self, params: Dict) -> SkillResult:
        cycle_id = params.get("cycle_id", "").strip()
        if not cycle_id:
            return SkillResult(success=False, message="cycle_id is required")

        cycles = self._load_cycles()
        target = None
        for c in cycles:
            if c.cycle_id == cycle_id:
                target = c
                break

        if not target:
            return SkillResult(success=False, message=f"Cycle '{cycle_id}' not found")

        if target.status not in ("active", "concluded"):
            return SkillResult(
                success=False,
                message=f"Cycle '{cycle_id}' is '{target.status}'. Can only revert active or concluded cycles.",
            )

        self._do_revert(target, cycles)

        return SkillResult(
            success=True,
            message=f"Reverted cycle '{cycle_id}'. Modification removed from prompt.",
            data={"cycle_id": cycle_id, "status": "reverted"},
        )

    def _list_cycles(self, params: Dict) -> SkillResult:
        status_filter = params.get("status_filter", "").strip()
        cycles = self._load_cycles()

        if status_filter:
            cycles = [c for c in cycles if c.status == status_filter]

        summaries = []
        for c in cycles:
            summaries.append({
                "cycle_id": c.cycle_id,
                "hypothesis": c.hypothesis[:100],
                "target_skill": c.target_skill,
                "status": c.status,
                "result": c.result,
                "baseline_success_rate": c.baseline_success_rate,
                "post_success_rate": round(c.post_success_rate, 3) if c.post_success_rate is not None else None,
                "improvement_delta": round(c.improvement_delta, 3) if c.improvement_delta is not None else None,
                "post_outcomes_count": len(c.post_outcomes),
                "created_at": c.created_at,
            })

        return SkillResult(
            success=True,
            message=f"{len(summaries)} improvement cycle(s)" + (f" (filter: {status_filter})" if status_filter else ""),
            data={"cycles": summaries},
        )

    def _improvement_report(self, params: Dict) -> SkillResult:
        cycles = self._load_cycles()
        history = self._load_analysis_history()

        total_cycles = len(cycles)
        concluded = [c for c in cycles if c.status == "concluded"]
        reverted = [c for c in cycles if c.status == "reverted"]
        active = [c for c in cycles if c.status == "active"]
        proposed = [c for c in cycles if c.status == "proposed"]

        improved = [c for c in concluded if c.result == "improved"]
        degraded = [c for c in concluded if c.result == "degraded"]
        neutral = [c for c in concluded if c.result == "neutral"]

        # Compute net improvement
        net_delta = 0.0
        for c in improved:
            if c.improvement_delta is not None:
                net_delta += c.improvement_delta
        for c in degraded:
            if c.improvement_delta is not None:
                net_delta += c.improvement_delta  # Already negative

        # Get trend from analysis history
        trend = None
        if len(history) >= 2:
            first_rate = history[0].get("overall_success_rate", 0)
            last_rate = history[-1].get("overall_success_rate", 0)
            trend = {
                "first_analysis_rate": first_rate,
                "latest_analysis_rate": last_rate,
                "overall_trend": round(last_rate - first_rate, 3),
                "analyses_count": len(history),
            }

        return SkillResult(
            success=True,
            message=f"Optimization report: {len(improved)} improved, {len(degraded)} degraded, {len(neutral)} neutral",
            data={
                "total_cycles": total_cycles,
                "by_status": {
                    "proposed": len(proposed),
                    "active": len(active),
                    "concluded": len(concluded),
                    "reverted": len(reverted),
                },
                "by_result": {
                    "improved": len(improved),
                    "degraded": len(degraded),
                    "neutral": len(neutral),
                },
                "net_improvement_delta": round(net_delta, 3),
                "trend": trend,
                "top_improvements": [
                    {
                        "cycle_id": c.cycle_id,
                        "hypothesis": c.hypothesis[:80],
                        "delta": round(c.improvement_delta, 3) if c.improvement_delta is not None else None,
                    }
                    for c in sorted(
                        improved,
                        key=lambda x: x.improvement_delta or 0,
                        reverse=True,
                    )[:3]
                ],
            },
        )

    # --- Auto-Optimize (full pass) ---

    def _auto_optimize(self, params: Dict) -> SkillResult:
        """Run a complete optimization pass: analyze → hypothesize → propose."""
        window = int(params.get("window", 50))

        # Step 1: Analyze
        analysis_result = self._analyze({"window": window})
        if not analysis_result.success:
            return analysis_result

        bottlenecks = analysis_result.data.get("bottlenecks", [])
        if not bottlenecks:
            return SkillResult(
                success=True,
                message="No bottlenecks found - nothing to optimize!",
                data={
                    "overall_success_rate": analysis_result.data.get("overall_success_rate"),
                    "proposed_cycle": None,
                },
            )

        # Step 2: Hypothesize
        hypotheses = self._generate_hypotheses(bottlenecks, max_count=1)
        if not hypotheses:
            return SkillResult(
                success=True,
                message="Could not generate hypotheses from bottlenecks",
                data={"bottlenecks": len(bottlenecks), "proposed_cycle": None},
            )

        best = hypotheses[0]

        # Step 3: Propose
        propose_result = self._propose_cycle({
            "hypothesis": best["hypothesis"],
            "target_skill": best["target_skill"],
            "modification": best["suggested_modification"],
        })

        if not propose_result.success:
            return propose_result

        return SkillResult(
            success=True,
            message=f"Auto-optimization complete. Proposed cycle: {propose_result.data.get('cycle_id')}",
            data={
                "overall_success_rate": analysis_result.data.get("overall_success_rate"),
                "worst_bottleneck": {
                    "skill": best["target_skill"],
                    "success_rate": best["current_success_rate"],
                },
                "proposed_cycle": propose_result.data,
                "hypothesis": best["hypothesis"],
            },
        )
