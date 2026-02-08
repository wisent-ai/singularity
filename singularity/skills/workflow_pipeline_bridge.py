#!/usr/bin/env python3
"""
WorkflowPipelineBridgeSkill - Bidirectional bridge between WorkflowSkill and PipelineExecutor.

Currently, WorkflowSkill and PipelineExecutor are two independent execution
engines that don't talk to each other:
- WorkflowSkill: persistent DAGs of skill actions, with conditions and retries
- PipelineExecutor: fast single-cycle batch execution of tool steps

This bridge enables:
1. Convert a workflow into a pipeline for fast batch execution
2. Convert a pipeline plan into a reusable workflow definition
3. Execute a workflow via PipelineExecutor (faster, single-cycle)
4. Compare execution stats between the two engines
5. Auto-suggest which engine to use based on workflow characteristics

This is the #1 priority from session 173. It unifies the two execution engines
and lets the agent choose the most efficient execution strategy.

Pillar: Self-Improvement (unified execution = smarter resource usage)
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from .base import Skill, SkillAction, SkillManifest, SkillResult


BRIDGE_DATA_FILE = Path(__file__).parent.parent / "data" / "workflow_pipeline_bridge.json"

# Maps workflow skill_id:action patterns to PipelineExecutor tool format
SKILL_TO_TOOL_MAP = {
    "shell:run": "shell:run",
    "github:create_pr": "github:create_pr",
    "github:list_issues": "github:list_issues",
    "filesystem:write": "filesystem:write",
    "filesystem:read": "filesystem:read",
    "code_review:review": "code_review:review",
    "deployment:deploy": "deployment:deploy",
    "content:generate": "content:generate",
    "email:send": "email:send",
    "browser:fetch": "browser:fetch",
    "planner:update_task": "planner:update_task",
}

# Heuristics for recommending which engine to use
ENGINE_THRESHOLDS = {
    "max_pipeline_steps": 15,      # Pipelines work best with fewer steps
    "needs_persistence": False,     # Workflows persist, pipelines don't
    "needs_conditions": True,       # Both support conditions
    "max_pipeline_duration_s": 120, # Pipeline timeout limit
}


class WorkflowPipelineBridgeSkill(Skill):
    """
    Bidirectional bridge between WorkflowSkill and PipelineExecutor.

    Converts between formats, recommends optimal execution engine,
    and tracks cross-engine execution comparisons for learning.
    """

    def __init__(self, credentials: Dict[str, str] = None):
        super().__init__(credentials)
        self._ensure_data()

    def _ensure_data(self):
        BRIDGE_DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
        if not BRIDGE_DATA_FILE.exists():
            self._save({"conversions": [], "comparisons": [], "recommendations": []})

    def _load(self) -> Dict:
        try:
            with open(BRIDGE_DATA_FILE, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {"conversions": [], "comparisons": [], "recommendations": []}

    def _save(self, data: Dict):
        with open(BRIDGE_DATA_FILE, "w") as f:
            json.dump(data, f, indent=2, default=str)

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="workflow_pipeline_bridge",
            name="Workflow-Pipeline Bridge",
            version="1.0.0",
            category="automation",
            description="Convert between WorkflowSkill and PipelineExecutor formats, recommend optimal engine",
            required_credentials=[],
            install_cost=0,
            actions=[
                SkillAction(
                    name="workflow_to_pipeline",
                    description="Convert a workflow definition into PipelineExecutor step dicts",
                    parameters={
                        "workflow_name": {"type": "string", "required": True, "description": "Name of workflow to convert"},
                        "max_cost": {"type": "float", "required": False, "description": "Total cost budget (default 0.50)"},
                    },
                ),
                SkillAction(
                    name="pipeline_to_workflow",
                    description="Convert pipeline steps into a WorkflowSkill workflow definition",
                    parameters={
                        "pipeline": {"type": "array", "required": True, "description": "Pipeline step dicts to convert"},
                        "workflow_name": {"type": "string", "required": True, "description": "Name for the new workflow"},
                        "description": {"type": "string", "required": False, "description": "Workflow description"},
                    },
                ),
                SkillAction(
                    name="recommend_engine",
                    description="Analyze a workflow/pipeline and recommend the optimal execution engine",
                    parameters={
                        "workflow_name": {"type": "string", "required": False, "description": "Workflow name to analyze"},
                        "pipeline": {"type": "array", "required": False, "description": "Pipeline steps to analyze"},
                    },
                ),
                SkillAction(
                    name="record_comparison",
                    description="Record execution results from both engines for the same workflow",
                    parameters={
                        "workflow_name": {"type": "string", "required": True, "description": "Workflow/pipeline name"},
                        "engine": {"type": "string", "required": True, "description": "'workflow' or 'pipeline'"},
                        "success": {"type": "boolean", "required": True, "description": "Whether execution succeeded"},
                        "duration_ms": {"type": "float", "required": True, "description": "Total duration in ms"},
                        "cost": {"type": "float", "required": False, "description": "Total cost"},
                        "steps_succeeded": {"type": "integer", "required": False, "description": "Steps that succeeded"},
                        "total_steps": {"type": "integer", "required": False, "description": "Total steps"},
                    },
                ),
                SkillAction(
                    name="compare_engines",
                    description="Compare execution performance between workflow and pipeline engines",
                    parameters={
                        "workflow_name": {"type": "string", "required": False, "description": "Specific workflow (omit for all)"},
                    },
                ),
                SkillAction(
                    name="status",
                    description="Show bridge status: conversions, comparisons, recommendations",
                    parameters={},
                ),
            ],
        )

    def check_credentials(self) -> bool:
        return True

    def execute(self, action: str, params: Dict[str, Any]) -> SkillResult:
        actions = {
            "workflow_to_pipeline": self._workflow_to_pipeline,
            "pipeline_to_workflow": self._pipeline_to_workflow,
            "recommend_engine": self._recommend_engine,
            "record_comparison": self._record_comparison,
            "compare_engines": self._compare_engines,
            "status": self._status,
        }
        handler = actions.get(action)
        if not handler:
            return SkillResult(success=False, message=f"Unknown action: {action}")
        return handler(params)

    def _load_workflow(self, name: str) -> Optional[Dict]:
        """Load a workflow definition from WorkflowSkill's data file."""
        workflow_file = Path(__file__).parent.parent / "data" / "workflows.json"
        try:
            with open(workflow_file, "r") as f:
                data = json.load(f)
            return data.get("workflows", {}).get(name)
        except (FileNotFoundError, json.JSONDecodeError):
            return None

    def _workflow_step_to_pipeline_step(self, step: Dict) -> Dict:
        """Convert a single WorkflowSkill step to a PipelineExecutor step dict."""
        skill_id = step.get("skill_id", "")
        action = step.get("action", "")
        tool_key = f"{skill_id}:{action}"
        tool = SKILL_TO_TOOL_MAP.get(tool_key, tool_key)

        pipeline_step = {
            "tool": tool,
            "params": dict(step.get("params", {})),
            "label": step.get("label", tool_key),
            "timeout_seconds": step.get("timeout_seconds", 30.0),
            "max_cost": step.get("max_cost", 0.05),
            "required": step.get("on_failure", "stop") == "stop",
            "retry_count": step.get("max_retries", 0),
        }

        # Convert workflow condition to pipeline condition
        wf_condition = step.get("condition")
        if wf_condition:
            pipeline_step["condition"] = self._convert_condition_to_pipeline(wf_condition)

        # Convert on_failure to pipeline on_failure
        on_failure = step.get("on_failure", "stop")
        if isinstance(on_failure, dict):
            pipeline_step["on_failure"] = {
                "tool": f"{on_failure.get('skill_id', '')}:{on_failure.get('action', '')}",
                "params": on_failure.get("params", {}),
            }

        return pipeline_step

    def _convert_condition_to_pipeline(self, wf_condition: Dict) -> Dict:
        """Convert a WorkflowSkill condition to PipelineExecutor condition format."""
        if wf_condition.get("always"):
            return {}

        pipeline_condition = {}

        ref = wf_condition.get("ref", "")
        if "equals" in wf_condition and "success" in str(wf_condition["equals"]):
            pipeline_condition["prev_success"] = True
        elif "equals" in wf_condition and "fail" in str(wf_condition["equals"]):
            pipeline_condition["prev_success"] = False
        elif ref:
            # Generic: check if previous step succeeded
            pipeline_condition["prev_success"] = True

        return pipeline_condition

    def _pipeline_step_to_workflow_step(self, step: Dict, index: int) -> Dict:
        """Convert a PipelineExecutor step dict to a WorkflowSkill step."""
        tool = step.get("tool", "")
        parts = tool.split(":", 1)
        skill_id = parts[0] if parts else tool
        action = parts[1] if len(parts) > 1 else "execute"

        wf_step = {
            "index": index,
            "skill_id": skill_id,
            "action": action,
            "params": dict(step.get("params", {})),
            "label": step.get("label", f"{skill_id}:{action}"),
            "on_failure": "stop" if step.get("required", True) else "skip",
            "max_retries": step.get("retry_count", 0),
            "condition": None,
        }

        # Convert pipeline condition to workflow condition
        pl_condition = step.get("condition")
        if pl_condition:
            wf_step["condition"] = self._convert_condition_to_workflow(pl_condition, index)

        return wf_step

    def _convert_condition_to_workflow(self, pl_condition: Dict, step_index: int) -> Optional[Dict]:
        """Convert a PipelineExecutor condition to WorkflowSkill condition format."""
        if not pl_condition:
            return None

        if "prev_success" in pl_condition:
            prev_idx = step_index - 1
            if prev_idx >= 0:
                return {
                    "ref": f"$steps.{prev_idx}.status",
                    "equals": "success" if pl_condition["prev_success"] else "failed",
                }

        if "prev_contains" in pl_condition:
            prev_idx = step_index - 1
            if prev_idx >= 0:
                return {
                    "ref": f"$steps.{prev_idx}._result",
                    "equals": pl_condition["prev_contains"],
                }

        return None

    def _workflow_to_pipeline(self, params: Dict) -> SkillResult:
        """Convert a workflow definition into PipelineExecutor step dicts."""
        workflow_name = params.get("workflow_name", "")
        if not workflow_name:
            return SkillResult(success=False, message="workflow_name is required")

        max_cost = params.get("max_cost", 0.50)

        workflow = self._load_workflow(workflow_name)
        if not workflow:
            return SkillResult(
                success=False,
                message=f"Workflow '{workflow_name}' not found",
            )

        steps = workflow.get("steps", [])
        if not steps:
            return SkillResult(
                success=True,
                message=f"Workflow '{workflow_name}' has no steps",
                data={"pipeline": [], "workflow_name": workflow_name},
            )

        # Convert each step
        pipeline_steps = []
        for step in steps:
            pipeline_step = self._workflow_step_to_pipeline_step(step)
            pipeline_steps.append(pipeline_step)

        # Apply cost budget
        total_cost = sum(s.get("max_cost", 0.05) for s in pipeline_steps)
        if total_cost > max_cost and total_cost > 0:
            scale = max_cost / total_cost
            for s in pipeline_steps:
                s["max_cost"] = round(s.get("max_cost", 0.05) * scale, 4)

        # Record conversion
        data = self._load()
        data["conversions"].append({
            "direction": "workflow_to_pipeline",
            "workflow_name": workflow_name,
            "step_count": len(pipeline_steps),
            "max_cost": max_cost,
            "converted_at": datetime.now().isoformat(),
        })
        data["conversions"] = data["conversions"][-100:]
        self._save(data)

        return SkillResult(
            success=True,
            message=f"Converted workflow '{workflow_name}' to pipeline with {len(pipeline_steps)} steps",
            data={
                "pipeline": pipeline_steps,
                "workflow_name": workflow_name,
                "step_count": len(pipeline_steps),
                "estimated_cost": sum(s.get("max_cost", 0.05) for s in pipeline_steps),
                "estimated_duration_s": sum(s.get("timeout_seconds", 30) for s in pipeline_steps),
            },
        )

    def _pipeline_to_workflow(self, params: Dict) -> SkillResult:
        """Convert pipeline steps into a WorkflowSkill workflow definition."""
        pipeline = params.get("pipeline", [])
        workflow_name = params.get("workflow_name", "")
        description = params.get("description", "")

        if not pipeline:
            return SkillResult(success=False, message="pipeline is required")
        if not workflow_name:
            return SkillResult(success=False, message="workflow_name is required")

        # Convert each step
        wf_steps = []
        for i, step in enumerate(pipeline):
            wf_step = self._pipeline_step_to_workflow_step(step, i)
            wf_steps.append(wf_step)

        workflow_def = {
            "name": workflow_name,
            "description": description or f"Auto-converted from pipeline ({len(wf_steps)} steps)",
            "steps": wf_steps,
        }

        # Record conversion
        data = self._load()
        data["conversions"].append({
            "direction": "pipeline_to_workflow",
            "workflow_name": workflow_name,
            "step_count": len(wf_steps),
            "converted_at": datetime.now().isoformat(),
        })
        data["conversions"] = data["conversions"][-100:]
        self._save(data)

        return SkillResult(
            success=True,
            message=f"Converted pipeline to workflow '{workflow_name}' with {len(wf_steps)} steps",
            data={
                "workflow_definition": workflow_def,
                "workflow_name": workflow_name,
                "step_count": len(wf_steps),
            },
        )

    def _analyze_characteristics(self, steps: List[Dict], is_pipeline: bool = False) -> Dict:
        """Analyze workflow/pipeline characteristics for engine recommendation."""
        step_count = len(steps)
        has_conditions = any(s.get("condition") for s in steps)
        has_retries = any(
            s.get("retry_count" if is_pipeline else "max_retries", 0) > 0
            for s in steps
        )
        has_fallbacks = any(s.get("on_failure") for s in steps)

        total_timeout = sum(s.get("timeout_seconds", 30) for s in steps)
        total_cost = sum(s.get("max_cost", 0.05) for s in steps)

        # Check for data dependencies (references between steps)
        has_data_deps = False
        for s in steps:
            params = s.get("params", {})
            for v in params.values():
                if isinstance(v, str) and v.startswith("$"):
                    has_data_deps = True
                    break

        # Unique tools/skills used
        if is_pipeline:
            tools = set(s.get("tool", "") for s in steps)
        else:
            tools = set(f"{s.get('skill_id', '')}:{s.get('action', '')}" for s in steps)

        return {
            "step_count": step_count,
            "has_conditions": has_conditions,
            "has_retries": has_retries,
            "has_fallbacks": has_fallbacks,
            "has_data_deps": has_data_deps,
            "total_timeout_s": total_timeout,
            "total_cost": total_cost,
            "unique_tools": len(tools),
            "tools": list(tools),
        }

    def _recommend_engine(self, params: Dict) -> SkillResult:
        """Recommend the optimal execution engine for a workflow/pipeline."""
        workflow_name = params.get("workflow_name", "")
        pipeline = params.get("pipeline", [])

        if not workflow_name and not pipeline:
            return SkillResult(
                success=False,
                message="Provide either workflow_name or pipeline steps",
            )

        # Get steps to analyze
        if workflow_name:
            workflow = self._load_workflow(workflow_name)
            if not workflow:
                return SkillResult(success=False, message=f"Workflow '{workflow_name}' not found")
            steps = workflow.get("steps", [])
            is_pipeline = False
        else:
            steps = pipeline
            is_pipeline = True

        chars = self._analyze_characteristics(steps, is_pipeline)

        # Score each engine
        pipeline_score = 0
        workflow_score = 0
        reasons = []

        # Pipeline is better for: small step count, no persistence needed, fast execution
        if chars["step_count"] <= ENGINE_THRESHOLDS["max_pipeline_steps"]:
            pipeline_score += 2
            reasons.append("Pipeline: step count within pipeline limit")
        else:
            workflow_score += 3
            reasons.append("Workflow: too many steps for pipeline")

        if chars["total_timeout_s"] <= ENGINE_THRESHOLDS["max_pipeline_duration_s"]:
            pipeline_score += 2
            reasons.append("Pipeline: total duration within pipeline timeout")
        else:
            workflow_score += 2
            reasons.append("Workflow: long duration benefits from persistence")

        # Pipeline is faster for simple sequential execution
        if not chars["has_data_deps"]:
            pipeline_score += 1
            reasons.append("Pipeline: no data dependencies, simple execution")
        else:
            workflow_score += 1
            reasons.append("Workflow: data dependencies benefit from reference resolution")

        # Workflows are better for persistence and reuse
        if chars["step_count"] > 5:
            workflow_score += 1
            reasons.append("Workflow: complex enough to benefit from persistence")

        # Both support conditions equally
        if chars["has_conditions"]:
            pipeline_score += 1
            workflow_score += 1

        # Both support retries
        if chars["has_retries"]:
            pipeline_score += 1
            workflow_score += 1

        # Check historical comparison data
        data = self._load()
        historical = self._get_historical_comparison(data, workflow_name or "pipeline")
        if historical:
            if historical["pipeline_better"]:
                pipeline_score += 2
                reasons.append(f"Pipeline: historically {historical['pipeline_speedup']:.1f}x faster")
            else:
                workflow_score += 2
                reasons.append(f"Workflow: historically {historical['workflow_speedup']:.1f}x faster")

        recommended = "pipeline" if pipeline_score > workflow_score else "workflow"
        confidence = abs(pipeline_score - workflow_score) / max(pipeline_score + workflow_score, 1)

        # Record recommendation
        data["recommendations"].append({
            "workflow_name": workflow_name or "ad-hoc",
            "recommended": recommended,
            "pipeline_score": pipeline_score,
            "workflow_score": workflow_score,
            "confidence": round(confidence, 3),
            "recommended_at": datetime.now().isoformat(),
        })
        data["recommendations"] = data["recommendations"][-50:]
        self._save(data)

        return SkillResult(
            success=True,
            message=f"Recommended engine: {recommended} (score: {pipeline_score} vs {workflow_score})",
            data={
                "recommended_engine": recommended,
                "pipeline_score": pipeline_score,
                "workflow_score": workflow_score,
                "confidence": round(confidence, 3),
                "reasons": reasons,
                "characteristics": chars,
                "historical_data": historical,
            },
        )

    def _get_historical_comparison(self, data: Dict, name: str) -> Optional[Dict]:
        """Get historical comparison data for a workflow/pipeline name."""
        comparisons = [c for c in data.get("comparisons", []) if c.get("workflow_name") == name]
        if len(comparisons) < 2:
            return None

        pipeline_runs = [c for c in comparisons if c["engine"] == "pipeline"]
        workflow_runs = [c for c in comparisons if c["engine"] == "workflow"]

        if not pipeline_runs or not workflow_runs:
            return None

        avg_pipeline_ms = sum(r["duration_ms"] for r in pipeline_runs) / len(pipeline_runs)
        avg_workflow_ms = sum(r["duration_ms"] for r in workflow_runs) / len(workflow_runs)

        pipeline_success = sum(1 for r in pipeline_runs if r["success"]) / len(pipeline_runs)
        workflow_success = sum(1 for r in workflow_runs if r["success"]) / len(workflow_runs)

        pipeline_better = avg_pipeline_ms < avg_workflow_ms

        return {
            "pipeline_avg_ms": round(avg_pipeline_ms, 1),
            "workflow_avg_ms": round(avg_workflow_ms, 1),
            "pipeline_success_rate": round(pipeline_success, 3),
            "workflow_success_rate": round(workflow_success, 3),
            "pipeline_better": pipeline_better,
            "pipeline_speedup": round(avg_workflow_ms / avg_pipeline_ms, 2) if avg_pipeline_ms > 0 else 1.0,
            "workflow_speedup": round(avg_pipeline_ms / avg_workflow_ms, 2) if avg_workflow_ms > 0 else 1.0,
            "pipeline_runs": len(pipeline_runs),
            "workflow_runs": len(workflow_runs),
        }

    def _record_comparison(self, params: Dict) -> SkillResult:
        """Record execution results for engine comparison."""
        workflow_name = params.get("workflow_name", "")
        engine = params.get("engine", "")
        success = params.get("success", False)
        duration_ms = params.get("duration_ms", 0)

        if not workflow_name:
            return SkillResult(success=False, message="workflow_name is required")
        if engine not in ("workflow", "pipeline"):
            return SkillResult(success=False, message="engine must be 'workflow' or 'pipeline'")

        data = self._load()
        comparison = {
            "workflow_name": workflow_name,
            "engine": engine,
            "success": success,
            "duration_ms": duration_ms,
            "cost": params.get("cost", 0),
            "steps_succeeded": params.get("steps_succeeded", 0),
            "total_steps": params.get("total_steps", 0),
            "recorded_at": datetime.now().isoformat(),
        }
        data["comparisons"].append(comparison)
        data["comparisons"] = data["comparisons"][-200:]
        self._save(data)

        return SkillResult(
            success=True,
            message=f"Recorded {engine} execution for '{workflow_name}': {'success' if success else 'failure'} in {duration_ms:.0f}ms",
            data=comparison,
        )

    def _compare_engines(self, params: Dict) -> SkillResult:
        """Compare execution performance between workflow and pipeline engines."""
        workflow_name = params.get("workflow_name")
        data = self._load()

        comparisons = data.get("comparisons", [])
        if workflow_name:
            comparisons = [c for c in comparisons if c.get("workflow_name") == workflow_name]

        if not comparisons:
            return SkillResult(
                success=True,
                message="No comparison data available yet. Record executions with record_comparison first.",
                data={"comparisons": []},
            )

        # Group by workflow name
        by_workflow = {}
        for c in comparisons:
            name = c["workflow_name"]
            if name not in by_workflow:
                by_workflow[name] = {"pipeline": [], "workflow": []}
            by_workflow[name][c["engine"]].append(c)

        results = []
        for name, engines in by_workflow.items():
            comparison = self._get_historical_comparison(data, name)
            if comparison:
                results.append({
                    "workflow_name": name,
                    **comparison,
                })

        overall_pipeline_wins = sum(1 for r in results if r.get("pipeline_better", False))
        overall_workflow_wins = len(results) - overall_pipeline_wins

        return SkillResult(
            success=True,
            message=f"Engine comparison: pipeline wins {overall_pipeline_wins}, workflow wins {overall_workflow_wins} across {len(results)} workflows",
            data={
                "comparisons": results,
                "pipeline_wins": overall_pipeline_wins,
                "workflow_wins": overall_workflow_wins,
                "total_workflows_compared": len(results),
                "total_data_points": len(comparisons),
            },
        )

    def _status(self, params: Dict) -> SkillResult:
        """Show bridge status."""
        data = self._load()

        conversions = data.get("conversions", [])
        comparisons = data.get("comparisons", [])
        recommendations = data.get("recommendations", [])

        w2p = sum(1 for c in conversions if c["direction"] == "workflow_to_pipeline")
        p2w = sum(1 for c in conversions if c["direction"] == "pipeline_to_workflow")

        rec_pipeline = sum(1 for r in recommendations if r["recommended"] == "pipeline")
        rec_workflow = sum(1 for r in recommendations if r["recommended"] == "workflow")

        return SkillResult(
            success=True,
            message=f"Bridge: {len(conversions)} conversions ({w2p} W→P, {p2w} P→W), {len(comparisons)} comparison points, {len(recommendations)} recommendations",
            data={
                "total_conversions": len(conversions),
                "workflow_to_pipeline": w2p,
                "pipeline_to_workflow": p2w,
                "total_comparisons": len(comparisons),
                "total_recommendations": len(recommendations),
                "recommended_pipeline": rec_pipeline,
                "recommended_workflow": rec_workflow,
                "recent_conversions": conversions[-5:] if conversions else [],
                "recent_recommendations": recommendations[-5:] if recommendations else [],
            },
        )
