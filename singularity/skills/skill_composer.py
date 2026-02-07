#!/usr/bin/env python3
"""
SkillComposerSkill - Dynamically compose new skills from existing ones.

This is a core Self-Improvement capability: the agent can create new composite
skills by chaining existing skill actions together, effectively extending its
own capability set at runtime without human intervention.

The agent can:
- Define composite skills as pipelines of existing skill actions
- Wire outputs from one step as inputs to the next (data flow)
- Register composed skills so they appear as first-class skills
- Persist compositions across sessions for reuse
- Track which compositions are useful and prune unused ones
- Generate skill code from compositions (self-modification)

This bridges the gap between WorkflowSkill (predefined DAGs) and true
self-improvement: the agent can observe what action sequences it repeats,
compose them into reusable skills, and evolve its own toolkit.

Architecture:
  Composition = ordered pipeline of SkillSteps
  SkillStep = skill_id + action + param_mapping
  ParamMapping = static values + references to previous step outputs
  ComposedSkill = registered Skill wrapping a Composition

Part of the Self-Improvement pillar: autonomous capability expansion.
"""

import json
import uuid
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from .base import Skill, SkillResult, SkillManifest, SkillAction


COMPOSER_FILE = Path(__file__).parent.parent / "data" / "compositions.json"
MAX_COMPOSITIONS = 100
MAX_STEPS_PER_COMPOSITION = 20


class SkillComposerSkill(Skill):
    """
    Dynamically compose new skills from existing ones.

    Enables agents to create reusable composite skills by chaining
    together existing skill actions with data flow between steps.
    Composed skills are registered as first-class skills and persist
    across sessions.
    """

    def __init__(self, credentials: Dict[str, str] = None):
        super().__init__(credentials)
        self._ensure_data()
        self._context = None  # Set when registered with agent

    def set_context(self, context):
        """Receive skill context for inter-skill execution."""
        self._context = context

    def _ensure_data(self):
        COMPOSER_FILE.parent.mkdir(parents=True, exist_ok=True)
        if not COMPOSER_FILE.exists():
            self._save(self._default_state())

    def _default_state(self) -> Dict:
        return {
            "compositions": {},
            "execution_history": [],
            "stats": {
                "total_created": 0,
                "total_executed": 0,
                "total_successful": 0,
                "total_failed": 0,
            },
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
        }

    def _load(self) -> Dict:
        try:
            with open(COMPOSER_FILE, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return self._default_state()

    def _save(self, data: Dict):
        data["last_updated"] = datetime.now().isoformat()
        COMPOSER_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(COMPOSER_FILE, "w") as f:
            json.dump(data, f, indent=2)

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="skill_composer",
            name="Skill Composer",
            version="1.0.0",
            category="self_improvement",
            description=(
                "Dynamically compose new skills from existing ones. "
                "Create reusable pipelines of skill actions with data flow between steps. "
                "This enables autonomous capability expansion."
            ),
            actions=[
                SkillAction(
                    name="compose",
                    description="Create a new composite skill from a pipeline of existing actions",
                    parameters={
                        "name": {"type": "str", "required": True, "description": "Name for the composed skill"},
                        "description": {"type": "str", "required": True, "description": "What the composed skill does"},
                        "steps": {"type": "list", "required": True, "description": "List of steps: [{skill_id, action, params, output_map}]"},
                        "input_params": {"type": "dict", "required": False, "description": "External parameters the composition accepts"},
                    },
                ),
                SkillAction(
                    name="execute_composition",
                    description="Execute a previously composed skill by ID",
                    parameters={
                        "composition_id": {"type": "str", "required": True, "description": "ID of the composition to execute"},
                        "params": {"type": "dict", "required": False, "description": "Input parameters for the composition"},
                    },
                ),
                SkillAction(
                    name="list_compositions",
                    description="List all saved compositions with usage stats",
                    parameters={},
                ),
                SkillAction(
                    name="get_composition",
                    description="Get details of a specific composition",
                    parameters={
                        "composition_id": {"type": "str", "required": True, "description": "ID of the composition"},
                    },
                ),
                SkillAction(
                    name="delete_composition",
                    description="Delete a composition that is no longer useful",
                    parameters={
                        "composition_id": {"type": "str", "required": True, "description": "ID to delete"},
                    },
                ),
                SkillAction(
                    name="suggest_compositions",
                    description="Analyze execution history to suggest useful compositions",
                    parameters={},
                ),
                SkillAction(
                    name="generate_code",
                    description="Generate standalone Python skill code from a composition",
                    parameters={
                        "composition_id": {"type": "str", "required": True, "description": "ID of the composition"},
                    },
                ),
            ],
            required_credentials=[],
        )

    async def execute(self, action: str, params: Dict[str, Any] = None) -> SkillResult:
        params = params or {}
        handlers = {
            "compose": self._compose,
            "execute_composition": self._execute_composition,
            "list_compositions": self._list_compositions,
            "get_composition": self._get_composition,
            "delete_composition": self._delete_composition,
            "suggest_compositions": self._suggest_compositions,
            "generate_code": self._generate_code,
        }
        handler = handlers.get(action)
        if not handler:
            return SkillResult(success=False, message=f"Unknown action: {action}")
        return await handler(params)

    async def _compose(self, params: Dict) -> SkillResult:
        """Create a new composite skill from a pipeline of steps."""
        name = params.get("name")
        description = params.get("description")
        steps = params.get("steps", [])
        input_params = params.get("input_params", {})

        if not name:
            return SkillResult(success=False, message="Name is required")
        if not description:
            return SkillResult(success=False, message="Description is required")
        if not steps:
            return SkillResult(success=False, message="At least one step is required")
        if len(steps) > MAX_STEPS_PER_COMPOSITION:
            return SkillResult(success=False, message=f"Max {MAX_STEPS_PER_COMPOSITION} steps per composition")

        data = self._load()

        if len(data["compositions"]) >= MAX_COMPOSITIONS:
            return SkillResult(success=False, message=f"Max {MAX_COMPOSITIONS} compositions reached. Delete unused ones first.")

        # Validate steps
        validated_steps = []
        for i, step in enumerate(steps):
            if not isinstance(step, dict):
                return SkillResult(success=False, message=f"Step {i} must be a dict")
            skill_id = step.get("skill_id")
            action = step.get("action")
            if not skill_id or not action:
                return SkillResult(success=False, message=f"Step {i} requires 'skill_id' and 'action'")

            validated_steps.append({
                "step_index": i,
                "skill_id": skill_id,
                "action": action,
                "params": step.get("params", {}),
                "output_map": step.get("output_map", {}),
                "continue_on_failure": step.get("continue_on_failure", False),
            })

        comp_id = f"comp_{uuid.uuid4().hex[:10]}"
        composition = {
            "id": comp_id,
            "name": name,
            "description": description,
            "steps": validated_steps,
            "input_params": input_params,
            "created_at": datetime.now().isoformat(),
            "execution_count": 0,
            "success_count": 0,
            "failure_count": 0,
            "last_executed": None,
            "avg_duration_ms": 0,
            "enabled": True,
        }

        data["compositions"][comp_id] = composition
        data["stats"]["total_created"] += 1
        self._save(data)

        return SkillResult(
            success=True,
            message=f"Composed skill '{name}' with {len(validated_steps)} steps",
            data={
                "composition_id": comp_id,
                "name": name,
                "steps_count": len(validated_steps),
                "step_summary": [
                    f"{s['skill_id']}.{s['action']}" for s in validated_steps
                ],
            },
        )

    async def _execute_composition(self, params: Dict) -> SkillResult:
        """Execute a composed skill pipeline."""
        comp_id = params.get("composition_id")
        input_params = params.get("params", {})

        if not comp_id:
            return SkillResult(success=False, message="composition_id is required")

        data = self._load()
        comp = data["compositions"].get(comp_id)
        if not comp:
            return SkillResult(success=False, message=f"Composition '{comp_id}' not found")

        if not comp.get("enabled", True):
            return SkillResult(success=False, message=f"Composition '{comp['name']}' is disabled")

        start_time = datetime.now()
        step_results = []
        accumulated_outputs = {"_input": input_params}
        overall_success = True
        total_cost = 0.0
        total_revenue = 0.0

        for step in comp["steps"]:
            step_idx = step["step_index"]
            skill_id = step["skill_id"]
            action = step["action"]
            step_params = dict(step.get("params", {}))

            # Resolve parameter references from previous step outputs
            output_map = step.get("output_map", {})
            for param_name, ref in output_map.items():
                value = self._resolve_ref(ref, accumulated_outputs)
                if value is not None:
                    step_params[param_name] = value

            # Also resolve $ref syntax in params directly
            for key, val in list(step_params.items()):
                if isinstance(val, str) and val.startswith("$ref:"):
                    resolved = self._resolve_ref(val[5:], accumulated_outputs)
                    if resolved is not None:
                        step_params[key] = resolved

            # Execute the step
            step_result = await self._execute_step(skill_id, action, step_params)
            step_record = {
                "step_index": step_idx,
                "skill_id": skill_id,
                "action": action,
                "success": step_result.success,
                "message": step_result.message,
                "data_keys": list(step_result.data.keys()) if step_result.data else [],
            }
            step_results.append(step_record)

            total_cost += step_result.cost
            total_revenue += step_result.revenue

            # Store outputs for downstream steps
            accumulated_outputs[f"step_{step_idx}"] = step_result.data
            accumulated_outputs[f"{skill_id}.{action}"] = step_result.data

            if not step_result.success:
                if not step.get("continue_on_failure", False):
                    overall_success = False
                    break

        duration_ms = (datetime.now() - start_time).total_seconds() * 1000

        # Update stats
        comp["execution_count"] = comp.get("execution_count", 0) + 1
        if overall_success:
            comp["success_count"] = comp.get("success_count", 0) + 1
        else:
            comp["failure_count"] = comp.get("failure_count", 0) + 1
        comp["last_executed"] = datetime.now().isoformat()

        # Running average of duration
        prev_avg = comp.get("avg_duration_ms", 0)
        prev_count = comp["execution_count"] - 1
        if prev_count > 0:
            comp["avg_duration_ms"] = (prev_avg * prev_count + duration_ms) / comp["execution_count"]
        else:
            comp["avg_duration_ms"] = duration_ms

        data["stats"]["total_executed"] += 1
        if overall_success:
            data["stats"]["total_successful"] += 1
        else:
            data["stats"]["total_failed"] += 1

        # Record execution history (keep last 50)
        history_entry = {
            "composition_id": comp_id,
            "name": comp["name"],
            "timestamp": datetime.now().isoformat(),
            "success": overall_success,
            "duration_ms": duration_ms,
            "steps_completed": len(step_results),
            "total_steps": len(comp["steps"]),
        }
        data["execution_history"].append(history_entry)
        if len(data["execution_history"]) > 50:
            data["execution_history"] = data["execution_history"][-50:]

        self._save(data)

        return SkillResult(
            success=overall_success,
            message=(
                f"Composition '{comp['name']}' completed: "
                f"{len(step_results)}/{len(comp['steps'])} steps"
            ),
            data={
                "composition_id": comp_id,
                "steps_completed": len(step_results),
                "total_steps": len(comp["steps"]),
                "step_results": step_results,
                "duration_ms": round(duration_ms, 1),
                "final_outputs": accumulated_outputs.get(
                    f"step_{len(comp['steps'])-1}", {}
                ),
            },
            cost=total_cost,
            revenue=total_revenue,
        )

    def _resolve_ref(self, ref: str, outputs: Dict) -> Any:
        """Resolve a reference like 'step_0.api_key' from accumulated outputs."""
        parts = ref.split(".", 1)
        if len(parts) == 1:
            return outputs.get(parts[0])
        container = outputs.get(parts[0])
        if isinstance(container, dict):
            return container.get(parts[1])
        return None

    async def _execute_step(self, skill_id: str, action: str, params: Dict) -> SkillResult:
        """Execute a single step by delegating to the target skill."""
        if self._context is None:
            return SkillResult(
                success=False,
                message="No skill context available - cannot execute sub-skills",
            )

        try:
            result = await self._context.execute_skill(skill_id, action, params)
            if isinstance(result, SkillResult):
                return result
            # If context returns raw dict, wrap it
            return SkillResult(
                success=result.get("success", True) if isinstance(result, dict) else True,
                message=str(result) if not isinstance(result, dict) else result.get("message", ""),
                data=result if isinstance(result, dict) else {"result": result},
            )
        except Exception as e:
            return SkillResult(
                success=False,
                message=f"Step execution failed: {str(e)}",
            )

    async def _list_compositions(self, params: Dict) -> SkillResult:
        """List all compositions with usage stats."""
        data = self._load()
        compositions = data["compositions"]

        if not compositions:
            return SkillResult(
                success=True,
                message="No compositions defined yet",
                data={"compositions": [], "stats": data["stats"]},
            )

        comp_list = []
        for comp_id, comp in compositions.items():
            success_rate = 0
            if comp.get("execution_count", 0) > 0:
                success_rate = round(
                    comp.get("success_count", 0) / comp["execution_count"] * 100, 1
                )

            comp_list.append({
                "id": comp_id,
                "name": comp["name"],
                "description": comp["description"],
                "steps_count": len(comp["steps"]),
                "execution_count": comp.get("execution_count", 0),
                "success_rate": success_rate,
                "enabled": comp.get("enabled", True),
                "last_executed": comp.get("last_executed"),
                "avg_duration_ms": round(comp.get("avg_duration_ms", 0), 1),
            })

        # Sort by execution count descending (most used first)
        comp_list.sort(key=lambda c: c["execution_count"], reverse=True)

        return SkillResult(
            success=True,
            message=f"Found {len(comp_list)} compositions",
            data={"compositions": comp_list, "stats": data["stats"]},
        )

    async def _get_composition(self, params: Dict) -> SkillResult:
        """Get detailed info about a specific composition."""
        comp_id = params.get("composition_id")
        if not comp_id:
            return SkillResult(success=False, message="composition_id is required")

        data = self._load()
        comp = data["compositions"].get(comp_id)
        if not comp:
            return SkillResult(success=False, message=f"Composition '{comp_id}' not found")

        # Include recent execution history for this composition
        recent_executions = [
            h for h in data["execution_history"]
            if h.get("composition_id") == comp_id
        ][-10:]

        return SkillResult(
            success=True,
            message=f"Composition '{comp['name']}'",
            data={
                "composition": comp,
                "recent_executions": recent_executions,
            },
        )

    async def _delete_composition(self, params: Dict) -> SkillResult:
        """Delete a composition."""
        comp_id = params.get("composition_id")
        if not comp_id:
            return SkillResult(success=False, message="composition_id is required")

        data = self._load()
        if comp_id not in data["compositions"]:
            return SkillResult(success=False, message=f"Composition '{comp_id}' not found")

        name = data["compositions"][comp_id]["name"]
        del data["compositions"][comp_id]
        self._save(data)

        return SkillResult(
            success=True,
            message=f"Deleted composition '{name}' ({comp_id})",
            data={"deleted_id": comp_id, "name": name},
        )

    async def _suggest_compositions(self, params: Dict) -> SkillResult:
        """Analyze execution history to suggest useful compositions."""
        data = self._load()
        history = data.get("execution_history", [])

        if len(history) < 2:
            return SkillResult(
                success=True,
                message="Not enough execution history to suggest compositions. Execute more skills first.",
                data={"suggestions": []},
            )

        # Find frequently used action pairs
        action_sequences = {}
        for i in range(len(history) - 1):
            curr = history[i]
            next_h = history[i + 1]
            pair_key = f"{curr.get('name', 'unknown')}->{next_h.get('name', 'unknown')}"
            action_sequences[pair_key] = action_sequences.get(pair_key, 0) + 1

        # Suggest compositions for frequent sequences
        suggestions = []
        for sequence, count in sorted(action_sequences.items(), key=lambda x: -x[1]):
            if count >= 2:
                suggestions.append({
                    "pattern": sequence,
                    "frequency": count,
                    "suggestion": f"Consider composing a skill for the '{sequence}' pattern (used {count} times)",
                })

        # Also check for compositions with low success rates
        underperformers = []
        for comp_id, comp in data["compositions"].items():
            exec_count = comp.get("execution_count", 0)
            if exec_count >= 3:
                success_rate = comp.get("success_count", 0) / exec_count * 100
                if success_rate < 50:
                    underperformers.append({
                        "id": comp_id,
                        "name": comp["name"],
                        "success_rate": round(success_rate, 1),
                        "suggestion": "Consider revising or deleting this underperforming composition",
                    })

        return SkillResult(
            success=True,
            message=f"Found {len(suggestions)} sequence patterns, {len(underperformers)} underperformers",
            data={
                "frequent_sequences": suggestions[:10],
                "underperforming_compositions": underperformers,
            },
        )

    async def _generate_code(self, params: Dict) -> SkillResult:
        """Generate standalone Python skill code from a composition."""
        comp_id = params.get("composition_id")
        if not comp_id:
            return SkillResult(success=False, message="composition_id is required")

        data = self._load()
        comp = data["compositions"].get(comp_id)
        if not comp:
            return SkillResult(success=False, message=f"Composition '{comp_id}' not found")

        # Generate a class name from the composition name
        class_name = "".join(
            word.capitalize() for word in comp["name"].replace("-", " ").replace("_", " ").split()
        ) + "Skill"

        skill_id = comp["name"].lower().replace(" ", "_").replace("-", "_")

        # Build step execution code
        step_code_blocks = []
        for step in comp["steps"]:
            step_code = f"""
        # Step {step['step_index']}: {step['skill_id']}.{step['action']}
        step_{step['step_index']}_result = await self._context.execute_skill(
            "{step['skill_id']}", "{step['action']}", {json.dumps(step.get('params', {}))}
        )
        if not step_{step['step_index']}_result.success and not {step.get('continue_on_failure', False)}:
            return SkillResult(success=False, message=f"Step {step['step_index']} failed: {{step_{step['step_index']}_result.message}}")
        results.append(step_{step['step_index']}_result)"""
            step_code_blocks.append(step_code)

        steps_code = "\n".join(step_code_blocks)

        code = f'''#!/usr/bin/env python3
"""
{comp['name']} - Auto-generated from composition {comp_id}.

{comp['description']}

Generated by SkillComposerSkill on {datetime.now().isoformat()}.
"""

from typing import Dict, Any
from .base import Skill, SkillResult, SkillManifest, SkillAction


class {class_name}(Skill):
    """{comp['description']}"""

    def __init__(self, credentials: Dict[str, str] = None):
        super().__init__(credentials)
        self._context = None

    def set_context(self, context):
        self._context = context

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="{skill_id}",
            name="{comp['name']}",
            version="1.0.0",
            category="composed",
            description="{comp['description']}",
            actions=[
                SkillAction(
                    name="run",
                    description="Execute the composed skill pipeline",
                    parameters={json.dumps(comp.get('input_params', {}))},
                ),
            ],
            required_credentials=[],
        )

    async def execute(self, action: str, params: Dict[str, Any] = None) -> SkillResult:
        if action != "run":
            return SkillResult(success=False, message=f"Unknown action: {{action}}")
        params = params or {{}}
        results = []
{steps_code}

        return SkillResult(
            success=True,
            message=f"Pipeline completed: {{len(results)}} steps executed",
            data={{"step_count": len(results)}},
        )
'''

        return SkillResult(
            success=True,
            message=f"Generated code for '{comp['name']}' ({len(comp['steps'])} steps)",
            data={
                "code": code,
                "class_name": class_name,
                "skill_id": skill_id,
                "file_name": f"{skill_id}.py",
            },
        )
