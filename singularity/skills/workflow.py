#!/usr/bin/env python3
"""
Workflow Skill - Multi-step automated workflow execution engine.

Enables agents to define, persist, and execute DAGs of skill actions
as automated pipelines. This is the critical bridge between planning
and execution - turning strategic plans into repeatable automated workflows.

The agent can:
- Define workflows as sequences of skill action steps
- Execute workflows with automatic step-by-step progression
- Pass data between steps using output references
- Handle step failures with configurable retry and fallback
- Persist workflow definitions for reuse across sessions
- Track workflow execution history with timing and results
- Pause and resume long-running workflows
- Define conditional branching based on step outcomes

Architecture:
  Workflow = ordered list of Steps
  Step = skill_id + action + params + optional conditions
  Execution = runtime instance of a workflow with state tracking
"""

import json
import uuid
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from enum import Enum
from .base import Skill, SkillResult, SkillManifest, SkillAction


WORKFLOW_FILE = Path(__file__).parent.parent / "data" / "workflows.json"


class StepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class WorkflowStatus(str, Enum):
    DRAFT = "draft"
    READY = "ready"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


class WorkflowSkill(Skill):
    """
    Multi-step automated workflow execution engine.

    Workflows are DAGs of skill actions that execute sequentially,
    passing data between steps and handling failures gracefully.
    This enables the agent to automate complex multi-skill tasks
    as repeatable, persistent pipelines.
    """

    def __init__(self, credentials: Dict[str, str] = None):
        super().__init__(credentials)
        self._ensure_data()

    def _ensure_data(self):
        WORKFLOW_FILE.parent.mkdir(parents=True, exist_ok=True)
        if not WORKFLOW_FILE.exists():
            self._save(self._default_state())

    def _default_state(self) -> Dict:
        return {
            "workflows": {},
            "executions": {},
            "templates": {},
            "stats": {
                "total_created": 0,
                "total_executed": 0,
                "total_succeeded": 0,
                "total_failed": 0,
            },
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
        }

    def _load(self) -> Dict:
        try:
            with open(WORKFLOW_FILE, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return self._default_state()

    def _save(self, data: Dict):
        data["last_updated"] = datetime.now().isoformat()
        WORKFLOW_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(WORKFLOW_FILE, "w") as f:
            json.dump(data, f, indent=2)

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="workflow",
            name="Workflow Engine",
            version="1.0.0",
            category="automation",
            description="Define and execute multi-step automated workflows as DAGs of skill actions",
            actions=[
                SkillAction(
                    name="create",
                    description="Create a new workflow definition with ordered steps",
                    parameters={
                        "name": {
                            "type": "string",
                            "required": True,
                            "description": "Unique workflow name",
                        },
                        "description": {
                            "type": "string",
                            "required": False,
                            "description": "What this workflow does",
                        },
                        "steps": {
                            "type": "array",
                            "required": True,
                            "description": "Ordered list of steps. Each step: {skill_id, action, params, on_failure?, condition?}",
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="execute",
                    description="Execute a workflow by name, running all steps in sequence",
                    parameters={
                        "name": {
                            "type": "string",
                            "required": True,
                            "description": "Name of the workflow to execute",
                        },
                        "inputs": {
                            "type": "object",
                            "required": False,
                            "description": "Input variables to pass to the workflow",
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="list",
                    description="List all defined workflows",
                    parameters={},
                    estimated_cost=0,
                ),
                SkillAction(
                    name="get",
                    description="Get detailed info about a specific workflow",
                    parameters={
                        "name": {
                            "type": "string",
                            "required": True,
                            "description": "Workflow name",
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="delete",
                    description="Delete a workflow definition",
                    parameters={
                        "name": {
                            "type": "string",
                            "required": True,
                            "description": "Workflow name to delete",
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="history",
                    description="View execution history for a workflow or all workflows",
                    parameters={
                        "name": {
                            "type": "string",
                            "required": False,
                            "description": "Workflow name (omit for all)",
                        },
                        "limit": {
                            "type": "number",
                            "required": False,
                            "description": "Max entries to return (default: 10)",
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="save_template",
                    description="Save a workflow as a reusable template with parameter placeholders",
                    parameters={
                        "name": {
                            "type": "string",
                            "required": True,
                            "description": "Source workflow name",
                        },
                        "template_name": {
                            "type": "string",
                            "required": True,
                            "description": "Template name for reuse",
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="from_template",
                    description="Create a workflow from a saved template",
                    parameters={
                        "template_name": {
                            "type": "string",
                            "required": True,
                            "description": "Template to instantiate",
                        },
                        "name": {
                            "type": "string",
                            "required": True,
                            "description": "Name for the new workflow",
                        },
                        "overrides": {
                            "type": "object",
                            "required": False,
                            "description": "Parameter overrides for the template",
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="stats",
                    description="Get workflow execution statistics",
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
            "create": self._create,
            "execute": self._execute,
            "list": self._list,
            "get": self._get,
            "delete": self._delete,
            "history": self._history,
            "save_template": self._save_template,
            "from_template": self._from_template,
            "stats": self._stats,
        }
        handler = handlers.get(action)
        if handler:
            return await handler(params)
        return SkillResult(success=False, message=f"Unknown action: {action}")

    def _validate_steps(self, steps: List[Dict]) -> Optional[str]:
        """Validate workflow step definitions. Returns error message or None."""
        if not steps:
            return "Workflow must have at least one step"
        if not isinstance(steps, list):
            return "Steps must be a list"

        for i, step in enumerate(steps):
            if not isinstance(step, dict):
                return f"Step {i} must be a dictionary"
            if "skill_id" not in step:
                return f"Step {i} missing 'skill_id'"
            if "action" not in step:
                return f"Step {i} missing 'action'"

        return None

    def _resolve_references(self, params: Dict, step_outputs: Dict, inputs: Dict) -> Dict:
        """
        Resolve parameter references to previous step outputs or workflow inputs.

        References use the format:
        - $steps.{step_index}.{key} - reference output from step N
        - $inputs.{key} - reference workflow input variable
        - $steps.{step_index}._result - reference the full result data

        Example: {"query": "$steps.0.url"} -> resolves to output of step 0's "url" field
        """
        resolved = {}
        for key, value in params.items():
            if isinstance(value, str) and value.startswith("$"):
                resolved[key] = self._resolve_single(value, step_outputs, inputs)
            elif isinstance(value, dict):
                resolved[key] = self._resolve_references(value, step_outputs, inputs)
            elif isinstance(value, list):
                resolved[key] = [
                    self._resolve_single(v, step_outputs, inputs)
                    if isinstance(v, str) and v.startswith("$")
                    else v
                    for v in value
                ]
            else:
                resolved[key] = value
        return resolved

    def _resolve_single(self, ref: str, step_outputs: Dict, inputs: Dict) -> Any:
        """Resolve a single $reference to its value."""
        parts = ref.split(".")

        if len(parts) >= 2 and parts[0] == "$inputs":
            key = ".".join(parts[1:])
            return inputs.get(key, ref)

        if len(parts) >= 3 and parts[0] == "$steps":
            try:
                step_idx = int(parts[1])
            except ValueError:
                return ref
            step_data = step_outputs.get(step_idx, {})
            if len(parts) == 3 and parts[2] == "_result":
                return step_data
            key = ".".join(parts[2:])
            return step_data.get(key, ref)

        return ref

    def _evaluate_condition(self, condition: Dict, step_outputs: Dict, inputs: Dict) -> bool:
        """
        Evaluate a step condition to decide whether to run or skip.

        Condition format:
        - {"ref": "$steps.0.status", "equals": "success"} - check equality
        - {"ref": "$steps.1.count", "greater_than": 0} - numeric comparison
        - {"ref": "$inputs.mode", "in": ["fast", "normal"]} - membership check
        - {"always": True} - always run (default)
        """
        if not condition:
            return True

        if condition.get("always"):
            return True

        ref = condition.get("ref", "")
        resolved = self._resolve_single(ref, step_outputs, inputs)

        if "equals" in condition:
            return resolved == condition["equals"]
        if "not_equals" in condition:
            return resolved != condition["not_equals"]
        if "greater_than" in condition:
            try:
                return float(resolved) > float(condition["greater_than"])
            except (ValueError, TypeError):
                return False
        if "less_than" in condition:
            try:
                return float(resolved) < float(condition["less_than"])
            except (ValueError, TypeError):
                return False
        if "in" in condition:
            return resolved in condition["in"]
        if "not_in" in condition:
            return resolved not in condition["not_in"]

        return True

    async def _create(self, params: Dict) -> SkillResult:
        """Create a new workflow definition."""
        name = params.get("name", "").strip()
        description = params.get("description", "")
        steps = params.get("steps", [])

        if not name:
            return SkillResult(success=False, message="Workflow name is required")

        # Parse steps if provided as string
        if isinstance(steps, str):
            try:
                steps = json.loads(steps)
            except json.JSONDecodeError:
                return SkillResult(success=False, message="Invalid steps JSON")

        error = self._validate_steps(steps)
        if error:
            return SkillResult(success=False, message=error)

        data = self._load()

        # Normalize steps
        normalized_steps = []
        for i, step in enumerate(steps):
            normalized_steps.append({
                "index": i,
                "skill_id": step["skill_id"],
                "action": step["action"],
                "params": step.get("params", {}),
                "on_failure": step.get("on_failure", "stop"),  # stop, skip, retry
                "max_retries": step.get("max_retries", 0),
                "condition": step.get("condition", None),
                "label": step.get("label", f"{step['skill_id']}:{step['action']}"),
            })

        workflow = {
            "name": name,
            "description": description,
            "steps": normalized_steps,
            "status": WorkflowStatus.READY.value,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "run_count": 0,
        }

        data["workflows"][name] = workflow
        data["stats"]["total_created"] += 1
        self._save(data)

        return SkillResult(
            success=True,
            message=f"Workflow '{name}' created with {len(normalized_steps)} steps.",
            data={"name": name, "step_count": len(normalized_steps)},
        )

    async def _execute(self, params: Dict) -> SkillResult:
        """Execute a workflow, running all steps in sequence."""
        name = params.get("name", "").strip()
        inputs = params.get("inputs", {})

        if not name:
            return SkillResult(success=False, message="Workflow name is required")

        data = self._load()
        workflow = data["workflows"].get(name)
        if not workflow:
            available = list(data["workflows"].keys())
            return SkillResult(
                success=False,
                message=f"Workflow '{name}' not found. Available: {available}",
            )

        # Check that we have a context to call skills through
        if not self.context:
            return SkillResult(
                success=False,
                message="No skill context available - cannot execute workflow steps",
            )

        execution_id = uuid.uuid4().hex[:12]
        steps = workflow["steps"]
        step_outputs: Dict[int, Dict] = {}
        step_results: List[Dict] = []
        total_cost = 0.0
        start_time = datetime.now()

        workflow_succeeded = True
        final_step_idx = -1

        for step in steps:
            idx = step["index"]
            final_step_idx = idx

            # Check condition
            condition = step.get("condition")
            if condition and not self._evaluate_condition(condition, step_outputs, inputs):
                step_result = {
                    "index": idx,
                    "label": step["label"],
                    "status": StepStatus.SKIPPED.value,
                    "message": "Condition not met, skipped",
                    "duration_ms": 0,
                }
                step_results.append(step_result)
                continue

            # Resolve parameter references
            resolved_params = self._resolve_references(
                step.get("params", {}), step_outputs, inputs
            )

            # Execute the step with retry support
            retries = 0
            max_retries = step.get("max_retries", 0)
            step_succeeded = False
            result = None

            while retries <= max_retries:
                step_start = datetime.now()
                try:
                    result = await self.context.call_skill(
                        step["skill_id"], step["action"], resolved_params
                    )
                except Exception as e:
                    result = SkillResult(success=False, message=f"Exception: {e}")

                step_duration = (datetime.now() - step_start).total_seconds() * 1000

                if result.success:
                    step_succeeded = True
                    break

                retries += 1
                if retries <= max_retries:
                    await asyncio.sleep(0.1 * retries)  # Brief backoff

            # Record step result
            step_result = {
                "index": idx,
                "label": step["label"],
                "skill_id": step["skill_id"],
                "action": step["action"],
                "status": StepStatus.COMPLETED.value if step_succeeded else StepStatus.FAILED.value,
                "message": result.message if result else "No result",
                "data": result.data if result else {},
                "cost": result.cost if result else 0,
                "retries": retries,
                "duration_ms": round(step_duration, 1),
            }
            step_results.append(step_result)

            if step_succeeded:
                step_outputs[idx] = result.data if result else {}
                total_cost += result.cost if result else 0
            else:
                on_failure = step.get("on_failure", "stop")
                if on_failure == "stop":
                    workflow_succeeded = False
                    break
                elif on_failure == "skip":
                    step_outputs[idx] = {}
                    continue

        total_duration = (datetime.now() - start_time).total_seconds() * 1000

        # Record execution
        execution = {
            "id": execution_id,
            "workflow": name,
            "status": WorkflowStatus.COMPLETED.value if workflow_succeeded else WorkflowStatus.FAILED.value,
            "inputs": inputs,
            "step_results": step_results,
            "total_cost": total_cost,
            "total_duration_ms": round(total_duration, 1),
            "steps_completed": sum(1 for s in step_results if s["status"] == StepStatus.COMPLETED.value),
            "steps_failed": sum(1 for s in step_results if s["status"] == StepStatus.FAILED.value),
            "steps_skipped": sum(1 for s in step_results if s["status"] == StepStatus.SKIPPED.value),
            "executed_at": start_time.isoformat(),
            "finished_at": datetime.now().isoformat(),
        }

        # Update persistent data
        data["executions"][execution_id] = execution
        data["workflows"][name]["run_count"] = workflow.get("run_count", 0) + 1
        data["workflows"][name]["last_run"] = execution_id
        data["stats"]["total_executed"] += 1
        if workflow_succeeded:
            data["stats"]["total_succeeded"] += 1
        else:
            data["stats"]["total_failed"] += 1

        # Keep last 100 executions
        if len(data["executions"]) > 100:
            sorted_execs = sorted(
                data["executions"].items(),
                key=lambda x: x[1].get("executed_at", ""),
            )
            data["executions"] = dict(sorted_execs[-100:])

        self._save(data)

        steps_summary = ", ".join(
            f"{s['label']}:{s['status']}" for s in step_results
        )
        msg = (
            f"Workflow '{name}' {'completed' if workflow_succeeded else 'failed'}. "
            f"{execution['steps_completed']}/{len(steps)} steps succeeded. "
            f"Duration: {total_duration:.0f}ms. Cost: ${total_cost:.4f}."
        )

        return SkillResult(
            success=workflow_succeeded,
            message=msg,
            data={
                "execution_id": execution_id,
                "status": execution["status"],
                "steps_completed": execution["steps_completed"],
                "steps_failed": execution["steps_failed"],
                "steps_skipped": execution["steps_skipped"],
                "total_steps": len(steps),
                "total_cost": total_cost,
                "total_duration_ms": round(total_duration, 1),
                "step_results": step_results,
            },
            cost=total_cost,
        )

    async def _list(self, params: Dict) -> SkillResult:
        """List all defined workflows."""
        data = self._load()
        workflows = []
        for name, wf in data["workflows"].items():
            workflows.append({
                "name": name,
                "description": wf.get("description", ""),
                "step_count": len(wf.get("steps", [])),
                "run_count": wf.get("run_count", 0),
                "status": wf.get("status", "unknown"),
                "created_at": wf.get("created_at"),
            })

        if not workflows:
            return SkillResult(
                success=True,
                message="No workflows defined yet. Use workflow:create to define one.",
                data={"workflows": [], "count": 0},
            )

        msg_lines = [f"Found {len(workflows)} workflow(s):"]
        for wf in workflows:
            msg_lines.append(
                f"  - {wf['name']} ({wf['step_count']} steps, run {wf['run_count']}x)"
            )

        return SkillResult(
            success=True,
            message="\n".join(msg_lines),
            data={"workflows": workflows, "count": len(workflows)},
        )

    async def _get(self, params: Dict) -> SkillResult:
        """Get detailed info about a specific workflow."""
        name = params.get("name", "").strip()
        if not name:
            return SkillResult(success=False, message="Workflow name is required")

        data = self._load()
        workflow = data["workflows"].get(name)
        if not workflow:
            return SkillResult(
                success=False,
                message=f"Workflow '{name}' not found",
            )

        return SkillResult(
            success=True,
            message=f"Workflow '{name}': {len(workflow['steps'])} steps, run {workflow.get('run_count', 0)} times.",
            data=workflow,
        )

    async def _delete(self, params: Dict) -> SkillResult:
        """Delete a workflow definition."""
        name = params.get("name", "").strip()
        if not name:
            return SkillResult(success=False, message="Workflow name is required")

        data = self._load()
        if name not in data["workflows"]:
            return SkillResult(
                success=False,
                message=f"Workflow '{name}' not found",
            )

        del data["workflows"][name]
        self._save(data)

        return SkillResult(
            success=True,
            message=f"Workflow '{name}' deleted.",
            data={"deleted": name},
        )

    async def _history(self, params: Dict) -> SkillResult:
        """View execution history."""
        name = params.get("name", "").strip()
        limit = int(params.get("limit", 10))
        limit = max(1, min(50, limit))

        data = self._load()
        executions = list(data.get("executions", {}).values())

        if name:
            executions = [e for e in executions if e.get("workflow") == name]

        # Sort by execution time, newest first
        executions.sort(key=lambda e: e.get("executed_at", ""), reverse=True)
        executions = executions[:limit]

        if not executions:
            return SkillResult(
                success=True,
                message=f"No execution history{' for ' + name if name else ''}.",
                data={"executions": [], "count": 0},
            )

        summaries = []
        for ex in executions:
            summaries.append({
                "id": ex["id"],
                "workflow": ex["workflow"],
                "status": ex["status"],
                "steps_completed": ex["steps_completed"],
                "total_steps": ex["steps_completed"] + ex["steps_failed"] + ex["steps_skipped"],
                "duration_ms": ex["total_duration_ms"],
                "cost": ex["total_cost"],
                "executed_at": ex["executed_at"],
            })

        msg_lines = [f"Execution history ({len(summaries)} entries):"]
        for s in summaries:
            msg_lines.append(
                f"  [{s['id']}] {s['workflow']}: {s['status']} "
                f"({s['steps_completed']} steps, {s['duration_ms']:.0f}ms)"
            )

        return SkillResult(
            success=True,
            message="\n".join(msg_lines),
            data={"executions": summaries, "count": len(summaries)},
        )

    async def _save_template(self, params: Dict) -> SkillResult:
        """Save a workflow as a reusable template."""
        name = params.get("name", "").strip()
        template_name = params.get("template_name", "").strip()

        if not name or not template_name:
            return SkillResult(
                success=False,
                message="Both 'name' (source) and 'template_name' are required",
            )

        data = self._load()
        workflow = data["workflows"].get(name)
        if not workflow:
            return SkillResult(
                success=False,
                message=f"Workflow '{name}' not found",
            )

        template = {
            "template_name": template_name,
            "source_workflow": name,
            "description": workflow.get("description", ""),
            "steps": workflow["steps"],
            "created_at": datetime.now().isoformat(),
        }

        data["templates"][template_name] = template
        self._save(data)

        return SkillResult(
            success=True,
            message=f"Template '{template_name}' saved from workflow '{name}'.",
            data={"template_name": template_name, "step_count": len(workflow["steps"])},
        )

    async def _from_template(self, params: Dict) -> SkillResult:
        """Create a workflow from a saved template."""
        template_name = params.get("template_name", "").strip()
        name = params.get("name", "").strip()
        overrides = params.get("overrides", {})

        if not template_name or not name:
            return SkillResult(
                success=False,
                message="Both 'template_name' and 'name' are required",
            )

        data = self._load()
        template = data["templates"].get(template_name)
        if not template:
            available = list(data["templates"].keys())
            return SkillResult(
                success=False,
                message=f"Template '{template_name}' not found. Available: {available}",
            )

        # Deep copy steps and apply overrides
        import copy
        steps = copy.deepcopy(template["steps"])

        # Apply parameter overrides: overrides = {step_index: {param: value}}
        if overrides:
            for step_idx_str, param_overrides in overrides.items():
                try:
                    step_idx = int(step_idx_str)
                except (ValueError, TypeError):
                    continue
                if 0 <= step_idx < len(steps):
                    steps[step_idx].setdefault("params", {}).update(param_overrides)

        workflow = {
            "name": name,
            "description": f"Created from template '{template_name}'. {template.get('description', '')}",
            "steps": steps,
            "status": WorkflowStatus.READY.value,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "run_count": 0,
            "source_template": template_name,
        }

        data["workflows"][name] = workflow
        data["stats"]["total_created"] += 1
        self._save(data)

        return SkillResult(
            success=True,
            message=f"Workflow '{name}' created from template '{template_name}' with {len(steps)} steps.",
            data={"name": name, "template": template_name, "step_count": len(steps)},
        )

    async def _stats(self, params: Dict) -> SkillResult:
        """Get workflow execution statistics."""
        data = self._load()
        stats = data.get("stats", {})
        workflow_count = len(data.get("workflows", {}))
        template_count = len(data.get("templates", {}))
        execution_count = len(data.get("executions", {}))

        # Compute per-workflow stats
        workflow_stats = {}
        for ex in data.get("executions", {}).values():
            wf_name = ex.get("workflow", "unknown")
            if wf_name not in workflow_stats:
                workflow_stats[wf_name] = {
                    "runs": 0, "successes": 0, "failures": 0,
                    "total_duration_ms": 0, "total_cost": 0,
                }
            ws = workflow_stats[wf_name]
            ws["runs"] += 1
            if ex.get("status") == WorkflowStatus.COMPLETED.value:
                ws["successes"] += 1
            else:
                ws["failures"] += 1
            ws["total_duration_ms"] += ex.get("total_duration_ms", 0)
            ws["total_cost"] += ex.get("total_cost", 0)

        success_rate = 0
        if stats.get("total_executed", 0) > 0:
            success_rate = stats.get("total_succeeded", 0) / stats["total_executed"] * 100

        msg_lines = [
            f"Workflow Stats:",
            f"  Workflows defined: {workflow_count}",
            f"  Templates: {template_count}",
            f"  Total executions: {stats.get('total_executed', 0)}",
            f"  Success rate: {success_rate:.0f}%",
            f"  Total created: {stats.get('total_created', 0)}",
        ]

        return SkillResult(
            success=True,
            message="\n".join(msg_lines),
            data={
                "workflow_count": workflow_count,
                "template_count": template_count,
                "execution_count": execution_count,
                "total_created": stats.get("total_created", 0),
                "total_executed": stats.get("total_executed", 0),
                "total_succeeded": stats.get("total_succeeded", 0),
                "total_failed": stats.get("total_failed", 0),
                "success_rate": round(success_rate, 1),
                "per_workflow": workflow_stats,
            },
        )
