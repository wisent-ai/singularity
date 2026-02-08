#!/usr/bin/env python3
"""
WorkflowAnalyticsBridgeSkill - Wire WorkflowAnalytics into TemplateEventBridge.

WorkflowAnalyticsSkill discovers action sequence patterns (n-grams, anti-patterns,
suggestions). TemplateEventBridgeSkill deploys template workflows as live automations.
They operate independently - this bridge connects them so that:

1. Deployed template workflows auto-record executions in WorkflowAnalytics
2. Analytics patterns feed back into template recommendations and scoring
3. Template health is computed from workflow analytics success/failure data
4. Anti-patterns detected in analytics trigger warnings for deployed workflows
5. Performance reports aggregate analytics data per template deployment

The feedback loop:
  Template deploys workflow → Workflow executes → Analytics records outcome
  → Analytics detects patterns → Bridge enriches template scores → Agent
  picks better templates next time

Pillars:
- Self-Improvement: Closed feedback loop between deployment and pattern learning
- Revenue: Track which automated workflows produce the most value
- Goal Setting: Data-driven template selection based on historical outcomes

Actions:
- record_execution: Record a template workflow execution in WorkflowAnalytics
- template_health: Compute health scores for deployed templates from analytics
- pattern_report: Find patterns in template workflow executions
- anti_patterns: Detect problematic step sequences in deployed workflows
- recommend: Recommend templates based on analytics success data
- enrich_deployments: Add analytics scores to all deployed templates
- performance_dashboard: Aggregated performance view across all deployed workflows
- status: Show bridge health and sync state
"""

import json
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

from .base import Skill, SkillResult, SkillManifest, SkillAction

DATA_DIR = Path(__file__).parent.parent / "data"
BRIDGE_FILE = DATA_DIR / "workflow_analytics_bridge.json"
WORKFLOWS_FILE = DATA_DIR / "workflow_analytics.json"
TEMPLATE_BRIDGE_FILE = DATA_DIR / "template_event_bridge.json"

MAX_EXECUTION_LOG = 500


def _now_ts() -> float:
    return time.time()


def _now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _load_json(path: Path) -> Optional[Dict]:
    try:
        if path.exists():
            return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        pass
    return None


def _save_json(path: Path, data: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str))


class WorkflowAnalyticsBridgeSkill(Skill):
    """
    Bridge between WorkflowAnalytics and TemplateEventBridge for deployed
    workflow performance tracking and pattern-based template optimization.
    """

    def __init__(self, credentials: Dict[str, str] = None):
        super().__init__(credentials)
        self._ensure_data()

    def _ensure_data(self):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        if not BRIDGE_FILE.exists():
            _save_json(BRIDGE_FILE, {
                "executions": [],
                "template_scores": {},
                "enrichments": [],
                "config": {
                    "auto_record": True,
                    "health_window_hours": 168,  # 7 days
                    "min_executions_for_score": 3,
                    "anti_pattern_threshold": 0.3,  # 30% failure = anti-pattern
                },
                "stats": {
                    "total_recorded": 0,
                    "total_enrichments": 0,
                    "last_enrichment": None,
                },
            })

    def _load_bridge(self) -> Dict:
        return _load_json(BRIDGE_FILE) or {
            "executions": [], "template_scores": {},
            "enrichments": [], "config": {}, "stats": {},
        }

    def _save_bridge(self, data: Dict):
        _save_json(BRIDGE_FILE, data)

    def _load_workflows(self) -> Dict:
        return _load_json(WORKFLOWS_FILE) or {"workflows": [], "templates": [], "metadata": {}}

    def _load_deployments(self) -> Dict:
        return _load_json(TEMPLATE_BRIDGE_FILE) or {"deployments": {}, "stats": {}}

    # ── Manifest ──────────────────────────────────────────────────────

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="workflow_analytics_bridge",
            name="Workflow Analytics Bridge",
            version="1.0.0",
            category="infrastructure",
            description="Wire WorkflowAnalytics into TemplateEventBridge for deployed workflow performance tracking",
            actions=self.get_actions(),
            required_credentials=[],
            install_cost=0,
            author="singularity",
        )

    def get_actions(self) -> List[SkillAction]:
        return [
            SkillAction(
                name="record_execution",
                description="Record a template workflow execution outcome for analytics tracking",
                parameters={
                    "template_id": {"type": "string", "required": True,
                                    "description": "Template identifier (e.g. 'github_pr_review')"},
                    "deployment_id": {"type": "string", "required": False,
                                      "description": "Deployment ID from TemplateEventBridge"},
                    "steps": {"type": "array", "required": True,
                              "description": "List of executed steps: [{action, success, duration_ms?}]"},
                    "success": {"type": "boolean", "required": True,
                                "description": "Overall workflow execution success"},
                    "trigger_event": {"type": "string", "required": False,
                                      "description": "Event that triggered this execution"},
                    "revenue": {"type": "number", "required": False,
                                "description": "Revenue generated by this execution"},
                    "notes": {"type": "string", "required": False, "description": "Execution notes"},
                },
                estimated_cost=0,
                estimated_duration_seconds=1,
            ),
            SkillAction(
                name="template_health",
                description="Compute health scores for deployed templates based on execution analytics",
                parameters={
                    "template_id": {"type": "string", "required": False,
                                    "description": "Specific template to score. Default: all deployed templates"},
                },
                estimated_cost=0,
                estimated_duration_seconds=1,
            ),
            SkillAction(
                name="pattern_report",
                description="Find action patterns in template workflow executions",
                parameters={
                    "template_id": {"type": "string", "required": False,
                                    "description": "Filter patterns to a specific template"},
                    "min_occurrences": {"type": "number", "required": False,
                                        "description": "Minimum times a pattern must appear. Default: 2"},
                    "max_length": {"type": "number", "required": False,
                                   "description": "Maximum pattern length. Default: 4"},
                },
                estimated_cost=0,
                estimated_duration_seconds=1,
            ),
            SkillAction(
                name="anti_patterns",
                description="Detect step sequences that correlate with workflow failure",
                parameters={
                    "template_id": {"type": "string", "required": False,
                                    "description": "Filter to specific template"},
                    "threshold": {"type": "number", "required": False,
                                  "description": "Failure rate threshold (0-1). Default: 0.3"},
                },
                estimated_cost=0,
                estimated_duration_seconds=1,
            ),
            SkillAction(
                name="recommend",
                description="Recommend templates based on analytics success data for a given use case",
                parameters={
                    "use_case": {"type": "string", "required": False,
                                 "description": "Use case keyword to match (e.g. 'ci_cd', 'billing')"},
                    "top_k": {"type": "number", "required": False,
                              "description": "Number of recommendations. Default: 5"},
                },
                estimated_cost=0,
                estimated_duration_seconds=1,
            ),
            SkillAction(
                name="enrich_deployments",
                description="Add analytics-based scores and warnings to all deployed template entries",
                parameters={},
                estimated_cost=0,
                estimated_duration_seconds=1,
            ),
            SkillAction(
                name="performance_dashboard",
                description="Aggregated performance view across all deployed template workflows",
                parameters={
                    "window_hours": {"type": "number", "required": False,
                                     "description": "Time window for analysis. Default: 168 (7 days)"},
                },
                estimated_cost=0,
                estimated_duration_seconds=1,
            ),
            SkillAction(
                name="status",
                description="Show bridge health, sync state, and configuration",
                parameters={},
                estimated_cost=0,
                estimated_duration_seconds=1,
            ),
        ]

    # ── Main execute ──────────────────────────────────────────────────

    async def execute(self, action: str, params: Dict[str, Any] = None) -> SkillResult:
        params = params or {}
        actions = {
            "record_execution": self._record_execution,
            "template_health": self._template_health,
            "pattern_report": self._pattern_report,
            "anti_patterns": self._anti_patterns,
            "recommend": self._recommend,
            "enrich_deployments": self._enrich_deployments,
            "performance_dashboard": self._performance_dashboard,
            "status": self._status,
        }
        if action not in actions:
            return SkillResult(success=False,
                               message=f"Unknown action: {action}. Available: {list(actions.keys())}")
        try:
            return await actions[action](params)
        except Exception as e:
            return SkillResult(success=False, message=f"Bridge error: {str(e)}")

    # ── Actions ───────────────────────────────────────────────────────

    async def _record_execution(self, params: Dict) -> SkillResult:
        """Record a template workflow execution."""
        template_id = params.get("template_id", "")
        if not template_id:
            return SkillResult(success=False, message="template_id is required")

        steps = params.get("steps", [])
        success = params.get("success", False)

        bridge = self._load_bridge()
        execution = {
            "template_id": template_id,
            "deployment_id": params.get("deployment_id", ""),
            "steps": steps,
            "success": success,
            "trigger_event": params.get("trigger_event", ""),
            "revenue": params.get("revenue", 0),
            "notes": params.get("notes", ""),
            "timestamp": _now_iso(),
            "ts": _now_ts(),
            "total_duration_ms": sum(s.get("duration_ms", 0) for s in steps),
            "step_count": len(steps),
            "failed_steps": sum(1 for s in steps if not s.get("success", True)),
        }

        bridge["executions"].append(execution)
        if len(bridge["executions"]) > MAX_EXECUTION_LOG:
            bridge["executions"] = bridge["executions"][-MAX_EXECUTION_LOG:]

        bridge["stats"]["total_recorded"] = bridge["stats"].get("total_recorded", 0) + 1

        # Also write to WorkflowAnalytics format for cross-skill consumption
        workflows = self._load_workflows()
        wf_entry = {
            "name": f"template:{template_id}",
            "goal": f"Execute template workflow: {template_id}",
            "steps": [{"action": s.get("action", "unknown"), "success": s.get("success", True),
                        "duration_ms": s.get("duration_ms", 0)} for s in steps],
            "success": success,
            "started_at": execution["timestamp"],
            "ended_at": _now_iso(),
            "notes": params.get("notes", ""),
        }
        workflows["workflows"].append(wf_entry)
        meta = workflows.get("metadata", {})
        meta["total_workflows"] = meta.get("total_workflows", 0) + 1
        if success:
            meta["total_successes"] = meta.get("total_successes", 0) + 1
        else:
            meta["total_failures"] = meta.get("total_failures", 0) + 1
        workflows["metadata"] = meta
        _save_json(WORKFLOWS_FILE, workflows)

        self._save_bridge(bridge)

        return SkillResult(
            success=True,
            message=f"Recorded {'successful' if success else 'failed'} execution for template '{template_id}' ({len(steps)} steps, {execution['total_duration_ms']}ms)",
            data=execution,
        )

    async def _template_health(self, params: Dict) -> SkillResult:
        """Compute health scores for deployed templates."""
        bridge = self._load_bridge()
        config = bridge.get("config", {})
        window_hours = config.get("health_window_hours", 168)
        min_executions = config.get("min_executions_for_score", 3)
        target_template = params.get("template_id")

        cutoff_ts = _now_ts() - window_hours * 3600
        executions = [e for e in bridge.get("executions", []) if e.get("ts", 0) >= cutoff_ts]

        # Group by template
        by_template = defaultdict(list)
        for ex in executions:
            tid = ex.get("template_id", "")
            if target_template and tid != target_template:
                continue
            by_template[tid].append(ex)

        scores = {}
        for tid, execs in by_template.items():
            total = len(execs)
            successes = sum(1 for e in execs if e.get("success"))
            failures = total - successes
            success_rate = successes / total if total > 0 else 0
            total_revenue = sum(e.get("revenue", 0) for e in execs)
            avg_duration = sum(e.get("total_duration_ms", 0) for e in execs) / total if total > 0 else 0
            avg_steps = sum(e.get("step_count", 0) for e in execs) / total if total > 0 else 0
            avg_failed_steps = sum(e.get("failed_steps", 0) for e in execs) / total if total > 0 else 0

            # Health score: weighted combination
            if total < min_executions:
                health_score = 50  # Insufficient data
                confidence = "low"
            else:
                # 60% success rate, 20% step health, 20% execution freshness
                success_component = success_rate * 60
                step_health = max(0, 1 - avg_failed_steps / max(avg_steps, 1)) * 20
                # Freshness: recent executions score higher
                last_exec_ts = max(e.get("ts", 0) for e in execs)
                age_hours = (_now_ts() - last_exec_ts) / 3600
                freshness = max(0, 20 - age_hours / window_hours * 20)
                health_score = min(100, int(success_component + step_health + freshness))
                confidence = "high" if total >= 10 else "medium"

            scores[tid] = {
                "template_id": tid,
                "health_score": health_score,
                "confidence": confidence,
                "total_executions": total,
                "successes": successes,
                "failures": failures,
                "success_rate": round(success_rate * 100, 1),
                "total_revenue": round(total_revenue, 4),
                "avg_duration_ms": round(avg_duration, 1),
                "avg_steps": round(avg_steps, 1),
                "avg_failed_steps": round(avg_failed_steps, 2),
            }

        # Update cache
        bridge["template_scores"] = scores
        self._save_bridge(bridge)

        if not scores:
            return SkillResult(success=True, message="No template executions found in the analysis window",
                               data={"scores": {}, "window_hours": window_hours})

        # Summary
        healthiest = max(scores.values(), key=lambda s: s["health_score"])
        sickest = min(scores.values(), key=lambda s: s["health_score"])

        return SkillResult(
            success=True,
            message=f"Computed health for {len(scores)} templates. Best: {healthiest['template_id']} ({healthiest['health_score']}/100). Worst: {sickest['template_id']} ({sickest['health_score']}/100)",
            data={"scores": scores, "window_hours": window_hours},
        )

    async def _pattern_report(self, params: Dict) -> SkillResult:
        """Find action patterns in template workflow executions."""
        bridge = self._load_bridge()
        target_template = params.get("template_id")
        min_occ = int(params.get("min_occurrences", 2))
        max_len = int(params.get("max_length", 4))

        executions = bridge.get("executions", [])
        if target_template:
            executions = [e for e in executions if e.get("template_id") == target_template]

        if not executions:
            return SkillResult(success=True, message="No executions to analyze",
                               data={"patterns": [], "total_executions": 0})

        # Extract action sequences
        success_seqs = []
        failure_seqs = []
        for ex in executions:
            actions = [s.get("action", "unknown") for s in ex.get("steps", [])]
            if ex.get("success"):
                success_seqs.append(actions)
            else:
                failure_seqs.append(actions)

        # Find n-gram patterns
        patterns = []
        for n in range(2, max_len + 1):
            ngram_success_count = Counter()
            ngram_failure_count = Counter()

            for seq in success_seqs:
                for i in range(len(seq) - n + 1):
                    gram = tuple(seq[i:i + n])
                    ngram_success_count[gram] += 1

            for seq in failure_seqs:
                for i in range(len(seq) - n + 1):
                    gram = tuple(seq[i:i + n])
                    ngram_failure_count[gram] += 1

            all_grams = set(ngram_success_count.keys()) | set(ngram_failure_count.keys())
            for gram in all_grams:
                s_count = ngram_success_count.get(gram, 0)
                f_count = ngram_failure_count.get(gram, 0)
                total = s_count + f_count
                if total >= min_occ:
                    patterns.append({
                        "pattern": list(gram),
                        "length": n,
                        "total_occurrences": total,
                        "success_count": s_count,
                        "failure_count": f_count,
                        "success_rate": round(s_count / total * 100, 1) if total > 0 else 0,
                        "classification": "success_pattern" if s_count > f_count else "failure_pattern",
                    })

        # Sort by total occurrences
        patterns.sort(key=lambda p: p["total_occurrences"], reverse=True)

        return SkillResult(
            success=True,
            message=f"Found {len(patterns)} patterns across {len(executions)} executions",
            data={"patterns": patterns[:50], "total_executions": len(executions),
                  "success_workflows": len(success_seqs), "failure_workflows": len(failure_seqs)},
        )

    async def _anti_patterns(self, params: Dict) -> SkillResult:
        """Detect step sequences correlated with workflow failure."""
        bridge = self._load_bridge()
        config = bridge.get("config", {})
        target_template = params.get("template_id")
        threshold = float(params.get("threshold", config.get("anti_pattern_threshold", 0.3)))

        executions = bridge.get("executions", [])
        if target_template:
            executions = [e for e in executions if e.get("template_id") == target_template]

        if not executions:
            return SkillResult(success=True, message="No executions to analyze",
                               data={"anti_patterns": []})

        # Find step sequences that appear mostly in failed workflows
        step_pair_success = Counter()
        step_pair_failure = Counter()

        for ex in executions:
            actions = [s.get("action", "unknown") for s in ex.get("steps", [])]
            for i in range(len(actions) - 1):
                pair = (actions[i], actions[i + 1])
                if ex.get("success"):
                    step_pair_success[pair] += 1
                else:
                    step_pair_failure[pair] += 1

        # Also check individual failing steps
        step_failure_rate = Counter()
        step_total = Counter()
        for ex in executions:
            for step in ex.get("steps", []):
                action = step.get("action", "unknown")
                step_total[action] += 1
                if not step.get("success", True):
                    step_failure_rate[action] += 1

        anti_patterns = []

        # Pair-level anti-patterns
        all_pairs = set(step_pair_success.keys()) | set(step_pair_failure.keys())
        for pair in all_pairs:
            s_count = step_pair_success.get(pair, 0)
            f_count = step_pair_failure.get(pair, 0)
            total = s_count + f_count
            if total >= 2:
                failure_rate = f_count / total
                if failure_rate >= threshold:
                    anti_patterns.append({
                        "type": "step_pair",
                        "sequence": list(pair),
                        "failure_rate": round(failure_rate * 100, 1),
                        "occurrences": total,
                        "in_failed_workflows": f_count,
                        "in_successful_workflows": s_count,
                        "severity": "high" if failure_rate > 0.7 else ("medium" if failure_rate > 0.5 else "low"),
                    })

        # Step-level anti-patterns (steps that fail often)
        for action, fail_count in step_failure_rate.items():
            total = step_total[action]
            if total >= 2:
                rate = fail_count / total
                if rate >= threshold:
                    anti_patterns.append({
                        "type": "step_failure",
                        "step": action,
                        "failure_rate": round(rate * 100, 1),
                        "occurrences": total,
                        "failures": fail_count,
                        "severity": "high" if rate > 0.7 else ("medium" if rate > 0.5 else "low"),
                    })

        # Sort by severity
        severity_order = {"high": 0, "medium": 1, "low": 2}
        anti_patterns.sort(key=lambda a: (severity_order.get(a["severity"], 3), -a.get("occurrences", 0)))

        return SkillResult(
            success=True,
            message=f"Found {len(anti_patterns)} anti-patterns (threshold: {threshold * 100}% failure rate)",
            data={"anti_patterns": anti_patterns, "threshold": threshold, "total_executions": len(executions)},
        )

    async def _recommend(self, params: Dict) -> SkillResult:
        """Recommend templates based on analytics success data."""
        bridge = self._load_bridge()
        use_case = params.get("use_case", "")
        top_k = int(params.get("top_k", 5))

        scores = bridge.get("template_scores", {})

        # If scores are empty, compute them
        if not scores:
            health_result = await self._template_health({})
            scores = health_result.data.get("scores", {})

        # Filter by use case if specified
        candidates = list(scores.values())
        if use_case:
            # Match templates whose ID contains the use case keyword
            candidates = [s for s in candidates if use_case.lower() in s["template_id"].lower()]

        # Sort by health score (weighted by success rate and execution count)
        for c in candidates:
            # Composite score: health * (1 + log(executions))
            import math
            exec_bonus = 1 + math.log(max(1, c.get("total_executions", 1)))
            c["recommendation_score"] = round(c["health_score"] * exec_bonus, 1)

        candidates.sort(key=lambda c: c["recommendation_score"], reverse=True)
        recommendations = candidates[:top_k]

        if not recommendations:
            return SkillResult(
                success=True,
                message=f"No template recommendations available{' for use case: ' + use_case if use_case else ''}",
                data={"recommendations": [], "use_case": use_case},
            )

        top = recommendations[0]
        return SkillResult(
            success=True,
            message=f"Top recommendation: {top['template_id']} (score: {top['recommendation_score']}, health: {top['health_score']}/100, success: {top['success_rate']}%)",
            data={"recommendations": recommendations, "use_case": use_case, "total_candidates": len(scores)},
        )

    async def _enrich_deployments(self, params: Dict) -> SkillResult:
        """Add analytics scores and warnings to deployed templates."""
        bridge = self._load_bridge()
        deployments = self._load_deployments()

        # First compute fresh health scores
        health_result = await self._template_health({})
        scores = health_result.data.get("scores", {})

        # Detect anti-patterns
        anti_result = await self._anti_patterns({})
        anti_patterns = anti_result.data.get("anti_patterns", [])

        # Enrich each deployment
        enriched_count = 0
        deployment_data = deployments.get("deployments", {})
        enrichments = []

        for dep_id, dep in deployment_data.items():
            template_id = dep.get("template_id", "")
            score_data = scores.get(template_id, {})

            enrichment = {
                "deployment_id": dep_id,
                "template_id": template_id,
                "health_score": score_data.get("health_score"),
                "success_rate": score_data.get("success_rate"),
                "total_executions": score_data.get("total_executions", 0),
                "warnings": [],
                "enriched_at": _now_iso(),
            }

            # Check for anti-patterns affecting this template's steps
            template_steps = [s.get("name", "") for s in dep.get("steps", [])]
            for ap in anti_patterns:
                if ap["type"] == "step_pair":
                    seq = ap["sequence"]
                    for i in range(len(template_steps) - 1):
                        if template_steps[i] == seq[0] and template_steps[i + 1] == seq[1]:
                            enrichment["warnings"].append({
                                "type": "anti_pattern",
                                "message": f"Steps '{seq[0]}' → '{seq[1]}' have {ap['failure_rate']}% failure rate",
                                "severity": ap["severity"],
                            })
                elif ap["type"] == "step_failure":
                    if ap["step"] in template_steps:
                        enrichment["warnings"].append({
                            "type": "step_failure",
                            "message": f"Step '{ap['step']}' has {ap['failure_rate']}% failure rate",
                            "severity": ap["severity"],
                        })

            enrichments.append(enrichment)
            enriched_count += 1

        bridge["enrichments"] = enrichments
        bridge["stats"]["total_enrichments"] = bridge["stats"].get("total_enrichments", 0) + 1
        bridge["stats"]["last_enrichment"] = _now_iso()
        self._save_bridge(bridge)

        warnings_count = sum(len(e.get("warnings", [])) for e in enrichments)
        return SkillResult(
            success=True,
            message=f"Enriched {enriched_count} deployments with analytics data. {warnings_count} warnings detected.",
            data={"enriched": enriched_count, "warnings": warnings_count, "enrichments": enrichments},
        )

    async def _performance_dashboard(self, params: Dict) -> SkillResult:
        """Aggregated performance view across all deployed template workflows."""
        bridge = self._load_bridge()
        window_hours = int(params.get("window_hours",
                                       bridge.get("config", {}).get("health_window_hours", 168)))
        cutoff_ts = _now_ts() - window_hours * 3600

        executions = [e for e in bridge.get("executions", []) if e.get("ts", 0) >= cutoff_ts]

        if not executions:
            return SkillResult(
                success=True,
                message=f"No executions in the last {window_hours} hours",
                data={"window_hours": window_hours, "total_executions": 0},
            )

        total = len(executions)
        successes = sum(1 for e in executions if e.get("success"))
        failures = total - successes
        total_revenue = sum(e.get("revenue", 0) for e in executions)
        total_duration_ms = sum(e.get("total_duration_ms", 0) for e in executions)

        # Per-template breakdown
        by_template = defaultdict(lambda: {"total": 0, "successes": 0, "revenue": 0, "duration_ms": 0})
        for ex in executions:
            tid = ex.get("template_id", "unknown")
            by_template[tid]["total"] += 1
            if ex.get("success"):
                by_template[tid]["successes"] += 1
            by_template[tid]["revenue"] += ex.get("revenue", 0)
            by_template[tid]["duration_ms"] += ex.get("total_duration_ms", 0)

        template_breakdown = []
        for tid, stats in sorted(by_template.items(), key=lambda x: x[1]["total"], reverse=True):
            template_breakdown.append({
                "template_id": tid,
                "executions": stats["total"],
                "success_rate": round(stats["successes"] / stats["total"] * 100, 1) if stats["total"] > 0 else 0,
                "revenue": round(stats["revenue"], 4),
                "avg_duration_ms": round(stats["duration_ms"] / stats["total"], 1) if stats["total"] > 0 else 0,
            })

        # Trigger breakdown
        trigger_counts = Counter(e.get("trigger_event", "manual") for e in executions)
        trigger_breakdown = [{"trigger": t, "count": c} for t, c in trigger_counts.most_common(10)]

        # Time distribution (hourly buckets in the window)
        hour_buckets = defaultdict(int)
        for ex in executions:
            hour = int((ex.get("ts", 0) - cutoff_ts) / 3600)
            hour_buckets[hour] += 1

        dashboard = {
            "window_hours": window_hours,
            "total_executions": total,
            "successes": successes,
            "failures": failures,
            "overall_success_rate": round(successes / total * 100, 1) if total > 0 else 0,
            "total_revenue": round(total_revenue, 4),
            "avg_duration_ms": round(total_duration_ms / total, 1) if total > 0 else 0,
            "templates": template_breakdown,
            "triggers": trigger_breakdown,
            "unique_templates": len(by_template),
            "hourly_distribution": dict(hour_buckets),
        }

        return SkillResult(
            success=True,
            message=f"Performance: {total} executions, {dashboard['overall_success_rate']}% success, ${total_revenue:.2f} revenue over {window_hours}h",
            data=dashboard,
        )

    async def _status(self, params: Dict) -> SkillResult:
        """Show bridge health and sync state."""
        bridge = self._load_bridge()
        config = bridge.get("config", {})
        stats = bridge.get("stats", {})
        executions = bridge.get("executions", [])
        scores = bridge.get("template_scores", {})
        enrichments = bridge.get("enrichments", [])

        deployments = self._load_deployments()
        deployment_count = len(deployments.get("deployments", {}))

        workflows = self._load_workflows()
        workflow_count = len(workflows.get("workflows", []))

        # Compute coverage: how many deployed templates have analytics data
        deployed_templates = set()
        for dep in deployments.get("deployments", {}).values():
            deployed_templates.add(dep.get("template_id", ""))
        scored_templates = set(scores.keys())
        coverage = len(deployed_templates & scored_templates)

        status_data = {
            "config": config,
            "stats": stats,
            "execution_count": len(executions),
            "scored_templates": len(scores),
            "enrichment_count": len(enrichments),
            "template_deployments": deployment_count,
            "workflow_analytics_entries": workflow_count,
            "analytics_coverage": f"{coverage}/{len(deployed_templates)}" if deployed_templates else "0/0",
            "last_enrichment": stats.get("last_enrichment"),
        }

        return SkillResult(
            success=True,
            message=f"Bridge status: {len(executions)} executions tracked, {len(scores)} templates scored, {deployment_count} deployments, coverage: {status_data['analytics_coverage']}",
            data=status_data,
        )

    def estimate_cost(self, action: str, params: Dict = None) -> float:
        return 0.0
