#!/usr/bin/env python3
"""
LearningDistillationSkill - Synthesize cross-session learnings into actionable rules.

The agent collects data from many sources (outcome tracker, feedback loop,
experiments, skill profiler) but each session starts somewhat fresh. This
skill bridges the gap by automatically distilling patterns from raw data
into a persistent knowledge base of learned rules and heuristics.

Rules represent distilled wisdom like:
- "Skill X succeeds 95% of the time for action Y"
- "Avoid action Z when conditions C are met - failure rate >70%"
- "Combining skills A+B yields better results than A alone"
- "Cost efficiency: skill X costs 50% less than skill Y for same result"

The rule base is queryable at decision time so the agent can consult
its accumulated wisdom before choosing actions.

Pillar: Self-Improvement (the 'learn' component of act→measure→adapt→learn)
"""

import json
import uuid
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from collections import defaultdict
from .base import Skill, SkillResult, SkillManifest, SkillAction


DATA_DIR = Path(__file__).parent.parent / "data"
RULES_FILE = DATA_DIR / "learning_rules.json"

MAX_RULES = 500
MAX_DISTILLATION_HISTORY = 100


class RuleCategory:
    """Categories of learned rules."""
    SUCCESS_PATTERN = "success_pattern"      # High-success action patterns
    FAILURE_PATTERN = "failure_pattern"       # Actions/conditions to avoid
    COST_EFFICIENCY = "cost_efficiency"       # Cost optimization insights
    SKILL_PREFERENCE = "skill_preference"     # Which skills to prefer for tasks
    TIMING_PATTERN = "timing_pattern"         # When actions work best
    COMBINATION = "combination"              # Skill combinations that work well
    GENERAL = "general"                      # General operational wisdom


VALID_CATEGORIES = [
    RuleCategory.SUCCESS_PATTERN,
    RuleCategory.FAILURE_PATTERN,
    RuleCategory.COST_EFFICIENCY,
    RuleCategory.SKILL_PREFERENCE,
    RuleCategory.TIMING_PATTERN,
    RuleCategory.COMBINATION,
    RuleCategory.GENERAL,
]


class LearningDistillationSkill(Skill):
    """
    Synthesize cross-session learnings into a queryable rule base.

    Reads from outcome tracker, feedback loop, experiments, and skill
    profiler data to distill high-level patterns. Rules are persisted
    across sessions and can be queried at decision time.
    """

    def __init__(self, credentials: Dict[str, str] = None):
        super().__init__(credentials)
        self._ensure_data()

    def _ensure_data(self):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        if not RULES_FILE.exists():
            self._save(self._default_state())

    def _default_state(self) -> Dict:
        return {
            "rules": [],
            "distillation_history": [],
            "config": {
                "min_samples": 5,
                "high_success_threshold": 0.8,
                "low_success_threshold": 0.3,
                "cost_outlier_factor": 2.0,
                "rule_expiry_days": 30,
                "auto_expire": True,
            },
            "stats": {
                "total_distillations": 0,
                "total_rules_created": 0,
                "total_rules_expired": 0,
                "total_queries": 0,
            },
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
        }

    def _load(self) -> Dict:
        try:
            with open(RULES_FILE, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return self._default_state()

    def _save(self, data: Dict):
        data["last_updated"] = datetime.now().isoformat()
        # Enforce limits
        if len(data.get("rules", [])) > MAX_RULES:
            # Keep highest confidence rules
            data["rules"] = sorted(
                data["rules"], key=lambda r: r.get("confidence", 0), reverse=True
            )[:MAX_RULES]
        if len(data.get("distillation_history", [])) > MAX_DISTILLATION_HISTORY:
            data["distillation_history"] = data["distillation_history"][-MAX_DISTILLATION_HISTORY:]
        with open(RULES_FILE, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def _load_outcomes(self) -> Dict:
        """Load outcome tracker data."""
        outcomes_file = DATA_DIR / "outcomes.json"
        try:
            with open(outcomes_file, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {"outcomes": [], "metadata": {}}

    def _load_feedback(self) -> Dict:
        """Load feedback loop data."""
        feedback_file = DATA_DIR / "feedback_loop.json"
        try:
            with open(feedback_file, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {"adaptations": [], "reviews": []}

    def _load_experiments(self) -> Dict:
        """Load experiment data."""
        experiment_file = DATA_DIR / "experiments.json"
        try:
            with open(experiment_file, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {"experiments": []}

    def _load_profiler(self) -> Dict:
        """Load skill profiler data."""
        profiler_file = DATA_DIR / "skill_profiler.json"
        try:
            with open(profiler_file, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {"events": [], "sessions": []}

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="learning_distillation",
            name="Learning Distillation",
            version="1.0.0",
            category="meta",
            description="Synthesize cross-session learnings into actionable rules",
            required_credentials=[],
            install_cost=0,
            actions=[
                SkillAction(
                    name="distill",
                    description="Analyze all data sources and distill new rules from patterns",
                    parameters={
                        "sources": {
                            "type": "string",
                            "required": False,
                            "description": "Comma-separated sources to analyze: outcomes,feedback,experiments,profiler (default: all)",
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="query",
                    description="Query the rule base for relevant rules given a context",
                    parameters={
                        "skill_id": {
                            "type": "string",
                            "required": False,
                            "description": "Filter rules about a specific skill",
                        },
                        "category": {
                            "type": "string",
                            "required": False,
                            "description": "Filter by rule category",
                        },
                        "min_confidence": {
                            "type": "number",
                            "required": False,
                            "description": "Minimum confidence threshold (0-1, default: 0.5)",
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="add_rule",
                    description="Manually add a learned rule to the knowledge base",
                    parameters={
                        "rule_text": {
                            "type": "string",
                            "required": True,
                            "description": "The rule or heuristic to record",
                        },
                        "category": {
                            "type": "string",
                            "required": False,
                            "description": "Rule category (default: general)",
                        },
                        "confidence": {
                            "type": "number",
                            "required": False,
                            "description": "Confidence level 0-1 (default: 0.7)",
                        },
                        "skill_id": {
                            "type": "string",
                            "required": False,
                            "description": "Associated skill ID if applicable",
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="reinforce",
                    description="Increase confidence of a rule that was confirmed useful",
                    parameters={
                        "rule_id": {
                            "type": "string",
                            "required": True,
                            "description": "ID of the rule to reinforce",
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="weaken",
                    description="Decrease confidence of a rule that was found inaccurate",
                    parameters={
                        "rule_id": {
                            "type": "string",
                            "required": True,
                            "description": "ID of the rule to weaken",
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="expire",
                    description="Remove expired or low-confidence rules",
                    parameters={},
                    estimated_cost=0,
                ),
                SkillAction(
                    name="status",
                    description="Get rule base statistics and recent distillation history",
                    parameters={},
                    estimated_cost=0,
                ),
                SkillAction(
                    name="configure",
                    description="Update distillation configuration parameters",
                    parameters={
                        "min_samples": {
                            "type": "number",
                            "required": False,
                            "description": "Minimum data points needed to create a rule (default: 5)",
                        },
                        "high_success_threshold": {
                            "type": "number",
                            "required": False,
                            "description": "Success rate above which to create success_pattern rules (default: 0.8)",
                        },
                        "low_success_threshold": {
                            "type": "number",
                            "required": False,
                            "description": "Success rate below which to create failure_pattern rules (default: 0.3)",
                        },
                        "rule_expiry_days": {
                            "type": "number",
                            "required": False,
                            "description": "Days after which low-confidence rules expire (default: 30)",
                        },
                        "auto_expire": {
                            "type": "boolean",
                            "required": False,
                            "description": "Automatically expire old rules during distillation (default: true)",
                        },
                    },
                    estimated_cost=0,
                ),
            ],
        )

    async def execute(self, action: str, params: Dict) -> SkillResult:
        actions = {
            "distill": self._distill,
            "query": self._query,
            "add_rule": self._add_rule,
            "reinforce": self._reinforce,
            "weaken": self._weaken,
            "expire": self._expire,
            "status": self._status,
            "configure": self._configure,
        }
        fn = actions.get(action)
        if not fn:
            return SkillResult(
                success=False,
                message=f"Unknown action: {action}. Available: {list(actions.keys())}",
            )
        return await fn(params)

    async def _distill(self, params: Dict) -> SkillResult:
        """Analyze data sources and distill new rules."""
        sources_str = params.get("sources", "outcomes,feedback,experiments,profiler")
        sources = [s.strip() for s in sources_str.split(",")]

        data = self._load()
        config = data["config"]
        existing_rules = {r["rule_text"]: r for r in data["rules"]}
        new_rules = []

        # Auto-expire old rules first
        if config.get("auto_expire", True):
            expired_count = self._do_expire(data)
        else:
            expired_count = 0

        # Distill from each source
        if "outcomes" in sources:
            new_rules.extend(self._distill_from_outcomes(config, existing_rules))

        if "feedback" in sources:
            new_rules.extend(self._distill_from_feedback(config, existing_rules))

        if "experiments" in sources:
            new_rules.extend(self._distill_from_experiments(config, existing_rules))

        if "profiler" in sources:
            new_rules.extend(self._distill_from_profiler(config, existing_rules))

        # Add new rules to the store
        for rule in new_rules:
            data["rules"].append(rule)

        # Record distillation event
        distillation_record = {
            "id": str(uuid.uuid4())[:8],
            "timestamp": datetime.now().isoformat(),
            "sources": sources,
            "rules_created": len(new_rules),
            "rules_expired": expired_count,
            "total_rules": len(data["rules"]),
        }
        data["distillation_history"].append(distillation_record)
        data["stats"]["total_distillations"] += 1
        data["stats"]["total_rules_created"] += len(new_rules)

        self._save(data)

        return SkillResult(
            success=True,
            message=f"Distilled {len(new_rules)} new rules from {', '.join(sources)}. "
                    f"Expired {expired_count} old rules. Total rules: {len(data['rules'])}.",
            data={
                "new_rules": [
                    {"id": r["id"], "category": r["category"], "text": r["rule_text"]}
                    for r in new_rules
                ],
                "rules_expired": expired_count,
                "total_rules": len(data["rules"]),
                "distillation_id": distillation_record["id"],
            },
        )

    def _distill_from_outcomes(self, config: Dict, existing_rules: Dict) -> List[Dict]:
        """Distill rules from outcome tracker data."""
        outcomes_data = self._load_outcomes()
        outcomes = outcomes_data.get("outcomes", [])
        if not outcomes:
            return []

        rules = []
        min_samples = config.get("min_samples", 5)
        high_thresh = config.get("high_success_threshold", 0.8)
        low_thresh = config.get("low_success_threshold", 0.3)
        cost_factor = config.get("cost_outlier_factor", 2.0)

        # Group outcomes by skill (tool field contains "skill:action")
        by_skill = defaultdict(list)
        for o in outcomes:
            tool = o.get("tool", "")
            skill_id = tool.split(":")[0] if ":" in tool else tool
            by_skill[skill_id].append(o)

        # Analyze per-skill patterns
        all_costs = [o.get("cost", 0) for o in outcomes if o.get("cost", 0) > 0]
        avg_cost = sum(all_costs) / len(all_costs) if all_costs else 0

        for skill_id, skill_outcomes in by_skill.items():
            if len(skill_outcomes) < min_samples:
                continue

            successes = sum(1 for o in skill_outcomes if o.get("success"))
            total = len(skill_outcomes)
            success_rate = successes / total

            # High success pattern
            if success_rate >= high_thresh:
                rule_text = (
                    f"Skill '{skill_id}' has a high success rate of "
                    f"{success_rate:.0%} over {total} executions."
                )
                if rule_text not in existing_rules:
                    rules.append(self._make_rule(
                        rule_text,
                        category=RuleCategory.SUCCESS_PATTERN,
                        confidence=min(0.95, 0.5 + (total / 100)),
                        skill_id=skill_id,
                        evidence={"success_rate": success_rate, "sample_size": total},
                        source="outcomes",
                    ))

            # Failure pattern
            elif success_rate <= low_thresh:
                # Find common errors
                errors = [o.get("error", "") for o in skill_outcomes if o.get("error")]
                common_error = max(set(errors), key=errors.count) if errors else "unknown"
                rule_text = (
                    f"Skill '{skill_id}' has a low success rate of "
                    f"{success_rate:.0%} over {total} executions. "
                    f"Common error: {common_error[:100]}."
                )
                if rule_text not in existing_rules:
                    rules.append(self._make_rule(
                        rule_text,
                        category=RuleCategory.FAILURE_PATTERN,
                        confidence=min(0.95, 0.5 + (total / 100)),
                        skill_id=skill_id,
                        evidence={"success_rate": success_rate, "sample_size": total, "common_error": common_error[:100]},
                        source="outcomes",
                    ))

            # Cost efficiency
            skill_costs = [o.get("cost", 0) for o in skill_outcomes if o.get("cost", 0) > 0]
            if skill_costs and avg_cost > 0:
                avg_skill_cost = sum(skill_costs) / len(skill_costs)
                if avg_skill_cost > avg_cost * cost_factor:
                    rule_text = (
                        f"Skill '{skill_id}' costs ${avg_skill_cost:.4f}/execution, "
                        f"which is {avg_skill_cost/avg_cost:.1f}x the average (${avg_cost:.4f}). "
                        f"Consider alternatives for cost optimization."
                    )
                    if rule_text not in existing_rules:
                        rules.append(self._make_rule(
                            rule_text,
                            category=RuleCategory.COST_EFFICIENCY,
                            confidence=0.7,
                            skill_id=skill_id,
                            evidence={"avg_cost": avg_skill_cost, "global_avg": avg_cost, "ratio": avg_skill_cost / avg_cost},
                            source="outcomes",
                        ))

        # Analyze per-action patterns within skills
        by_action = defaultdict(list)
        for o in outcomes:
            tool = o.get("tool", "")
            by_action[tool].append(o)

        for action_key, action_outcomes in by_action.items():
            if len(action_outcomes) < min_samples:
                continue
            if ":" not in action_key:
                continue

            successes = sum(1 for o in action_outcomes if o.get("success"))
            total = len(action_outcomes)
            success_rate = successes / total

            skill_id = action_key.split(":")[0]

            if success_rate >= high_thresh:
                rule_text = (
                    f"Action '{action_key}' is highly reliable: "
                    f"{success_rate:.0%} success over {total} calls."
                )
                if rule_text not in existing_rules:
                    rules.append(self._make_rule(
                        rule_text,
                        category=RuleCategory.SUCCESS_PATTERN,
                        confidence=min(0.9, 0.5 + (total / 50)),
                        skill_id=skill_id,
                        evidence={"action": action_key, "success_rate": success_rate, "sample_size": total},
                        source="outcomes",
                    ))

        return rules

    def _distill_from_feedback(self, config: Dict, existing_rules: Dict) -> List[Dict]:
        """Distill rules from feedback loop adaptations."""
        feedback_data = self._load_feedback()
        adaptations = feedback_data.get("adaptations", [])
        if not adaptations:
            return []

        rules = []

        # Find adaptations that were applied and had outcomes
        applied = [a for a in adaptations if a.get("status") == "applied"]
        for adaptation in applied:
            desc = adaptation.get("description", "")
            outcome = adaptation.get("outcome", "")
            adaptation_type = adaptation.get("type", "unknown")

            if outcome == "positive":
                rule_text = (
                    f"Adaptation '{desc[:80]}' was applied and had a positive outcome. "
                    f"Type: {adaptation_type}."
                )
                if rule_text not in existing_rules:
                    rules.append(self._make_rule(
                        rule_text,
                        category=RuleCategory.GENERAL,
                        confidence=0.75,
                        evidence={"adaptation_type": adaptation_type, "outcome": "positive"},
                        source="feedback",
                    ))
            elif outcome == "negative":
                rule_text = (
                    f"Adaptation '{desc[:80]}' was applied but had a negative outcome. "
                    f"Avoid this pattern. Type: {adaptation_type}."
                )
                if rule_text not in existing_rules:
                    rules.append(self._make_rule(
                        rule_text,
                        category=RuleCategory.FAILURE_PATTERN,
                        confidence=0.7,
                        evidence={"adaptation_type": adaptation_type, "outcome": "negative"},
                        source="feedback",
                    ))

        return rules

    def _distill_from_experiments(self, config: Dict, existing_rules: Dict) -> List[Dict]:
        """Distill rules from experiment results."""
        experiment_data = self._load_experiments()
        experiments = experiment_data.get("experiments", [])
        if not experiments:
            return []

        rules = []

        for exp in experiments:
            status = exp.get("status", "")
            if status != "concluded":
                continue

            conclusion = exp.get("conclusion", {})
            winner = conclusion.get("winner", "")
            hypothesis = exp.get("hypothesis", "")

            if winner:
                rule_text = (
                    f"Experiment '{exp.get('name', 'unnamed')}': "
                    f"variant '{winner}' was best. Hypothesis: {hypothesis[:100]}."
                )
                if rule_text not in existing_rules:
                    # Higher confidence for experiments with more trials
                    total_trials = sum(
                        v.get("trials", 0)
                        for v in exp.get("variants", {}).values()
                        if isinstance(v, dict)
                    )
                    confidence = min(0.95, 0.6 + (total_trials / 200))
                    rules.append(self._make_rule(
                        rule_text,
                        category=RuleCategory.SKILL_PREFERENCE,
                        confidence=confidence,
                        evidence={
                            "experiment": exp.get("name"),
                            "winner": winner,
                            "total_trials": total_trials,
                        },
                        source="experiments",
                    ))

        return rules

    def _distill_from_profiler(self, config: Dict, existing_rules: Dict) -> List[Dict]:
        """Distill rules from skill profiler data."""
        profiler_data = self._load_profiler()
        events = profiler_data.get("events", [])
        if not events:
            return []

        rules = []
        min_samples = config.get("min_samples", 5)

        # Group events by skill
        by_skill = defaultdict(list)
        for e in events:
            skill_id = e.get("skill_id", "")
            if skill_id:
                by_skill[skill_id].append(e)

        # Identify never-used skills (loaded but no events)
        # We can't know loaded skills without agent context, so skip

        # Identify most efficient skills (lowest avg duration)
        skill_durations = {}
        for skill_id, skill_events in by_skill.items():
            if len(skill_events) < min_samples:
                continue
            durations = [e.get("duration_ms", 0) for e in skill_events if e.get("duration_ms", 0) > 0]
            if durations:
                skill_durations[skill_id] = sum(durations) / len(durations)

        if skill_durations:
            avg_duration = sum(skill_durations.values()) / len(skill_durations)
            for skill_id, avg_dur in skill_durations.items():
                if avg_dur < avg_duration * 0.5:
                    rule_text = (
                        f"Skill '{skill_id}' is fast: avg {avg_dur:.0f}ms execution time, "
                        f"which is {avg_dur/avg_duration:.0%} of the average ({avg_duration:.0f}ms). "
                        f"Prefer this skill when speed matters."
                    )
                    if rule_text not in existing_rules:
                        rules.append(self._make_rule(
                            rule_text,
                            category=RuleCategory.SKILL_PREFERENCE,
                            confidence=0.65,
                            skill_id=skill_id,
                            evidence={"avg_duration_ms": avg_dur, "global_avg_ms": avg_duration},
                            source="profiler",
                        ))

        return rules

    def _make_rule(
        self,
        rule_text: str,
        category: str = RuleCategory.GENERAL,
        confidence: float = 0.5,
        skill_id: str = "",
        evidence: Dict = None,
        source: str = "manual",
    ) -> Dict:
        """Create a new rule dict."""
        return {
            "id": str(uuid.uuid4())[:8],
            "rule_text": rule_text,
            "category": category,
            "confidence": round(confidence, 3),
            "skill_id": skill_id,
            "evidence": evidence or {},
            "source": source,
            "created_at": datetime.now().isoformat(),
            "last_reinforced": datetime.now().isoformat(),
            "reinforcement_count": 0,
        }

    async def _query(self, params: Dict) -> SkillResult:
        """Query the rule base for relevant rules."""
        data = self._load()
        rules = data["rules"]

        skill_id = params.get("skill_id", "")
        category = params.get("category", "")
        min_confidence = float(params.get("min_confidence", 0.5))

        # Filter
        filtered = []
        for r in rules:
            if skill_id and r.get("skill_id", "") != skill_id:
                continue
            if category and r.get("category", "") != category:
                continue
            if r.get("confidence", 0) < min_confidence:
                continue
            filtered.append(r)

        # Sort by confidence descending
        filtered.sort(key=lambda r: r.get("confidence", 0), reverse=True)

        data["stats"]["total_queries"] += 1
        self._save(data)

        return SkillResult(
            success=True,
            message=f"Found {len(filtered)} matching rules.",
            data={
                "rules": [
                    {
                        "id": r["id"],
                        "rule_text": r["rule_text"],
                        "category": r["category"],
                        "confidence": r["confidence"],
                        "skill_id": r.get("skill_id", ""),
                        "source": r.get("source", ""),
                        "created_at": r.get("created_at", ""),
                    }
                    for r in filtered[:20]  # Limit response size
                ],
                "total_matching": len(filtered),
            },
        )

    async def _add_rule(self, params: Dict) -> SkillResult:
        """Manually add a rule to the knowledge base."""
        rule_text = params.get("rule_text", "").strip()
        if not rule_text:
            return SkillResult(success=False, message="rule_text is required.")

        category = params.get("category", RuleCategory.GENERAL)
        if category not in VALID_CATEGORIES:
            return SkillResult(
                success=False,
                message=f"Invalid category: {category}. Valid: {VALID_CATEGORIES}",
            )

        confidence = float(params.get("confidence", 0.7))
        confidence = max(0.0, min(1.0, confidence))
        skill_id = params.get("skill_id", "")

        data = self._load()
        rule = self._make_rule(
            rule_text,
            category=category,
            confidence=confidence,
            skill_id=skill_id,
            source="manual",
        )
        data["rules"].append(rule)
        data["stats"]["total_rules_created"] += 1
        self._save(data)

        return SkillResult(
            success=True,
            message=f"Added rule '{rule['id']}' with confidence {confidence:.2f}.",
            data={"rule_id": rule["id"], "rule": rule},
        )

    async def _reinforce(self, params: Dict) -> SkillResult:
        """Increase confidence of a confirmed-useful rule."""
        rule_id = params.get("rule_id", "")
        if not rule_id:
            return SkillResult(success=False, message="rule_id is required.")

        data = self._load()
        for rule in data["rules"]:
            if rule["id"] == rule_id:
                old_confidence = rule["confidence"]
                # Asymptotic increase toward 1.0
                rule["confidence"] = min(0.99, rule["confidence"] + (1 - rule["confidence"]) * 0.2)
                rule["reinforcement_count"] = rule.get("reinforcement_count", 0) + 1
                rule["last_reinforced"] = datetime.now().isoformat()
                self._save(data)
                return SkillResult(
                    success=True,
                    message=f"Rule '{rule_id}' confidence: {old_confidence:.3f} → {rule['confidence']:.3f}.",
                    data={"rule_id": rule_id, "old_confidence": old_confidence, "new_confidence": rule["confidence"]},
                )

        return SkillResult(success=False, message=f"Rule '{rule_id}' not found.")

    async def _weaken(self, params: Dict) -> SkillResult:
        """Decrease confidence of an inaccurate rule."""
        rule_id = params.get("rule_id", "")
        if not rule_id:
            return SkillResult(success=False, message="rule_id is required.")

        data = self._load()
        for rule in data["rules"]:
            if rule["id"] == rule_id:
                old_confidence = rule["confidence"]
                # Decay toward 0
                rule["confidence"] = max(0.01, rule["confidence"] * 0.7)
                rule["last_reinforced"] = datetime.now().isoformat()
                self._save(data)
                return SkillResult(
                    success=True,
                    message=f"Rule '{rule_id}' confidence: {old_confidence:.3f} → {rule['confidence']:.3f}.",
                    data={"rule_id": rule_id, "old_confidence": old_confidence, "new_confidence": rule["confidence"]},
                )

        return SkillResult(success=False, message=f"Rule '{rule_id}' not found.")

    def _do_expire(self, data: Dict) -> int:
        """Remove expired rules. Returns count of expired rules."""
        config = data["config"]
        expiry_days = config.get("rule_expiry_days", 30)
        cutoff = (datetime.now() - timedelta(days=expiry_days)).isoformat()

        original_count = len(data["rules"])
        # Expire rules that are old AND low confidence
        data["rules"] = [
            r for r in data["rules"]
            if not (
                r.get("confidence", 0) < 0.4
                and r.get("last_reinforced", r.get("created_at", "")) < cutoff
            )
        ]
        expired = original_count - len(data["rules"])
        data["stats"]["total_rules_expired"] += expired
        return expired

    async def _expire(self, params: Dict) -> SkillResult:
        """Manually trigger rule expiration."""
        data = self._load()
        expired_count = self._do_expire(data)
        self._save(data)

        return SkillResult(
            success=True,
            message=f"Expired {expired_count} rules. {len(data['rules'])} rules remaining.",
            data={"expired": expired_count, "remaining": len(data["rules"])},
        )

    async def _status(self, params: Dict) -> SkillResult:
        """Get rule base status and stats."""
        data = self._load()
        rules = data["rules"]

        # Category breakdown
        by_category = defaultdict(int)
        for r in rules:
            by_category[r.get("category", "unknown")] += 1

        # Source breakdown
        by_source = defaultdict(int)
        for r in rules:
            by_source[r.get("source", "unknown")] += 1

        # Confidence distribution
        high_conf = sum(1 for r in rules if r.get("confidence", 0) >= 0.8)
        med_conf = sum(1 for r in rules if 0.5 <= r.get("confidence", 0) < 0.8)
        low_conf = sum(1 for r in rules if r.get("confidence", 0) < 0.5)

        # Recent distillations
        recent = data.get("distillation_history", [])[-5:]

        return SkillResult(
            success=True,
            message=f"Rule base: {len(rules)} rules across {len(by_category)} categories.",
            data={
                "total_rules": len(rules),
                "by_category": dict(by_category),
                "by_source": dict(by_source),
                "confidence_distribution": {
                    "high (>=0.8)": high_conf,
                    "medium (0.5-0.8)": med_conf,
                    "low (<0.5)": low_conf,
                },
                "stats": data.get("stats", {}),
                "config": data.get("config", {}),
                "recent_distillations": recent,
            },
        )

    async def _configure(self, params: Dict) -> SkillResult:
        """Update distillation configuration."""
        data = self._load()
        config = data["config"]
        changed = {}

        for key in ["min_samples", "high_success_threshold", "low_success_threshold",
                     "rule_expiry_days", "auto_expire", "cost_outlier_factor"]:
            if key in params:
                old_val = config.get(key)
                new_val = params[key]
                if key == "auto_expire":
                    new_val = bool(new_val)
                elif key in ("min_samples", "rule_expiry_days"):
                    new_val = max(1, int(new_val))
                else:
                    new_val = float(new_val)
                config[key] = new_val
                changed[key] = {"old": old_val, "new": new_val}

        if not changed:
            return SkillResult(
                success=False,
                message="No valid configuration parameters provided.",
            )

        self._save(data)
        return SkillResult(
            success=True,
            message=f"Updated {len(changed)} config parameters.",
            data={"changes": changed},
        )
