#!/usr/bin/env python3
"""
EventDrivenWorkflowSkill - Connect external events to autonomous agent workflows.

This is the critical bridge between WebhookSkill (external events) and
AutonomousLoopSkill (autonomous execution). It enables the agent to define
automation rules that trigger full workflows when specific events occur.

Instead of webhooks mapping 1:1 to a single skill action, event workflows can:
1. Match events by type, source, and payload conditions
2. Execute multi-step skill pipelines with data flow between steps
3. Escalate to the autonomous loop for complex decisions
4. Apply conditional branching based on event payload
5. Track workflow execution history and success rates

Example flows:
- GitHub PR opened -> code_review -> post comment -> log outcome
- Stripe payment received -> register customer -> provision access -> send welcome
- Cron tick -> assess pillars -> if revenue low, run marketing -> log result
- Error alert -> diagnose -> if critical, escalate to autonomous_loop

Pillars served:
- Revenue Generation: Automate service delivery on external triggers
- Goal Setting: Events can trigger goal reassessment
- Self-Improvement: Track which workflows succeed/fail and adapt
"""

import json
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field, asdict

from .base import Skill, SkillManifest, SkillAction, SkillResult


@dataclass
class WorkflowStep:
    """A single step in an event-driven workflow."""
    step_id: str
    skill_id: str
    action: str
    # Param mapping: {"target_param": "source"} where source can be:
    #   - "event.field.path" -> extract from event payload
    #   - "step.<step_id>.field" -> extract from previous step result
    #   - literal string/number for static values
    param_mapping: Dict[str, str] = field(default_factory=dict)
    static_params: Dict[str, Any] = field(default_factory=dict)
    # Conditional: only run this step if condition is met
    # {"field": "path.to.check", "op": "eq|ne|gt|lt|contains|exists", "value": ...}
    condition: Optional[Dict[str, Any]] = None
    # If true, workflow continues even if this step fails
    continue_on_failure: bool = False
    # Timeout in seconds
    timeout: float = 30.0

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class WorkflowRule:
    """An event-driven workflow automation rule."""
    rule_id: str
    name: str
    description: str
    # Event matching
    event_type: str  # e.g., "webhook", "cron", "internal", "error"
    event_source: str = ""  # e.g., "github", "stripe", specific webhook name
    # Payload conditions: all must match for the rule to fire
    # {"path.to.field": {"op": "eq", "value": "expected"}}
    conditions: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # Workflow steps
    steps: List[WorkflowStep] = field(default_factory=list)
    # Options
    enabled: bool = True
    priority: int = 0  # Higher = evaluated first
    # Whether to escalate to autonomous_loop if all steps fail
    escalate_on_failure: bool = False
    # Stats
    total_triggers: int = 0
    total_successes: int = 0
    total_failures: int = 0
    avg_duration_ms: float = 0.0
    created_at: str = ""
    last_triggered_at: Optional[str] = None

    def to_dict(self) -> Dict:
        d = {
            "rule_id": self.rule_id,
            "name": self.name,
            "description": self.description,
            "event_type": self.event_type,
            "event_source": self.event_source,
            "conditions": self.conditions,
            "steps": [s.to_dict() for s in self.steps],
            "enabled": self.enabled,
            "priority": self.priority,
            "escalate_on_failure": self.escalate_on_failure,
            "total_triggers": self.total_triggers,
            "total_successes": self.total_successes,
            "total_failures": self.total_failures,
            "avg_duration_ms": self.avg_duration_ms,
            "created_at": self.created_at,
            "last_triggered_at": self.last_triggered_at,
        }
        return d


@dataclass
class WorkflowExecution:
    """Record of a workflow execution."""
    execution_id: str
    rule_id: str
    rule_name: str
    event_type: str
    event_source: str
    started_at: str
    # Results
    completed_at: Optional[str] = None
    success: bool = False
    step_results: List[Dict[str, Any]] = field(default_factory=list)
    escalated: bool = False
    escalation_result: Optional[str] = None
    duration_ms: float = 0.0
    error: Optional[str] = None

    def to_dict(self) -> Dict:
        return asdict(self)


class EventDrivenWorkflowSkill(Skill):
    """
    Connects external events to autonomous agent workflows.

    Bridges WebhookSkill events to multi-step skill pipelines with
    conditional logic, data flow, and autonomous loop escalation.
    """

    def __init__(self, credentials: Dict[str, str] = None, data_dir: str = None):
        super().__init__(credentials)
        self._data_dir = Path(data_dir) if data_dir else Path("singularity/data")
        self._rules: Dict[str, WorkflowRule] = {}
        self._executions: List[WorkflowExecution] = []
        self._max_executions = 500
        self._load_data()

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="event_workflow",
            name="Event-Driven Workflows",
            version="1.0.0",
            category="autonomy",
            description=(
                "Connect external events to autonomous agent workflows. "
                "Define automation rules that trigger multi-step skill pipelines "
                "when specific events occur (webhooks, cron, errors). "
                "Supports conditional logic, data flow between steps, and "
                "escalation to the autonomous loop for complex decisions."
            ),
            actions=[
                SkillAction(
                    name="create_rule",
                    description=(
                        "Create an event-driven workflow rule. Define which events "
                        "trigger the workflow and what steps to execute."
                    ),
                    parameters={
                        "name": {"type": "string", "required": True,
                                 "description": "Unique name for the workflow rule"},
                        "description": {"type": "string", "required": False,
                                        "description": "Human-readable description"},
                        "event_type": {"type": "string", "required": True,
                                       "description": "Event type: webhook, cron, internal, error"},
                        "event_source": {"type": "string", "required": False,
                                         "description": "Specific source (e.g., 'github', 'stripe')"},
                        "conditions": {"type": "object", "required": False,
                                       "description": "Payload conditions that must match"},
                        "steps": {"type": "array", "required": True,
                                  "description": "Workflow steps to execute"},
                        "priority": {"type": "integer", "required": False,
                                     "description": "Priority (higher = evaluated first)"},
                        "escalate_on_failure": {"type": "boolean", "required": False,
                                                "description": "Escalate to autonomous loop on failure"},
                    },
                ),
                SkillAction(
                    name="trigger",
                    description=(
                        "Trigger event-driven workflows by dispatching an event. "
                        "All matching rules will execute their workflows."
                    ),
                    parameters={
                        "event_type": {"type": "string", "required": True,
                                       "description": "Event type (webhook, cron, internal, error)"},
                        "event_source": {"type": "string", "required": False,
                                         "description": "Event source identifier"},
                        "payload": {"type": "object", "required": False,
                                    "description": "Event payload data"},
                    },
                ),
                SkillAction(
                    name="list_rules",
                    description="List all workflow rules with their stats.",
                    parameters={
                        "event_type": {"type": "string", "required": False,
                                       "description": "Filter by event type"},
                        "enabled_only": {"type": "boolean", "required": False,
                                         "description": "Only show enabled rules"},
                    },
                ),
                SkillAction(
                    name="get_rule",
                    description="Get details of a specific workflow rule.",
                    parameters={
                        "name": {"type": "string", "required": True,
                                 "description": "Rule name"},
                    },
                ),
                SkillAction(
                    name="update_rule",
                    description="Update an existing workflow rule.",
                    parameters={
                        "name": {"type": "string", "required": True,
                                 "description": "Rule name to update"},
                        "enabled": {"type": "boolean", "required": False},
                        "priority": {"type": "integer", "required": False},
                        "escalate_on_failure": {"type": "boolean", "required": False},
                        "conditions": {"type": "object", "required": False},
                        "steps": {"type": "array", "required": False},
                    },
                ),
                SkillAction(
                    name="delete_rule",
                    description="Delete a workflow rule.",
                    parameters={
                        "name": {"type": "string", "required": True,
                                 "description": "Rule name to delete"},
                    },
                ),
                SkillAction(
                    name="get_executions",
                    description="View workflow execution history.",
                    parameters={
                        "rule_name": {"type": "string", "required": False,
                                      "description": "Filter by rule name"},
                        "success_only": {"type": "boolean", "required": False},
                        "limit": {"type": "integer", "required": False,
                                  "description": "Max results (default 20)"},
                    },
                ),
                SkillAction(
                    name="get_stats",
                    description="Get workflow performance statistics and analytics.",
                    parameters={},
                ),
            ],
            required_credentials=[],
        )

    async def execute(self, action: str, params: Dict) -> SkillResult:
        actions = {
            "create_rule": self._create_rule,
            "trigger": self._trigger,
            "list_rules": self._list_rules,
            "get_rule": self._get_rule,
            "update_rule": self._update_rule,
            "delete_rule": self._delete_rule,
            "get_executions": self._get_executions,
            "get_stats": self._get_stats,
        }
        handler = actions.get(action)
        if not handler:
            return SkillResult(
                success=False,
                message=f"Unknown action: {action}. Available: {list(actions.keys())}"
            )
        return await handler(params)

    # --- Core Actions ---

    async def _create_rule(self, params: Dict) -> SkillResult:
        """Create a new event-driven workflow rule."""
        name = params.get("name", "").strip()
        if not name:
            return SkillResult(success=False, message="Rule name is required")

        # Check for duplicate names
        for rule in self._rules.values():
            if rule.name == name:
                return SkillResult(
                    success=False,
                    message=f"Rule '{name}' already exists. Use update_rule to modify."
                )

        event_type = params.get("event_type", "").strip()
        if not event_type:
            return SkillResult(success=False, message="event_type is required")

        raw_steps = params.get("steps", [])
        if not raw_steps:
            return SkillResult(success=False, message="At least one workflow step is required")

        # Parse steps
        steps = []
        for i, raw_step in enumerate(raw_steps):
            skill_id = raw_step.get("skill_id", "").strip()
            action = raw_step.get("action", "").strip()
            if not skill_id or not action:
                return SkillResult(
                    success=False,
                    message=f"Step {i+1}: skill_id and action are required"
                )
            step = WorkflowStep(
                step_id=raw_step.get("step_id", f"step_{i+1}"),
                skill_id=skill_id,
                action=action,
                param_mapping=raw_step.get("param_mapping", {}),
                static_params=raw_step.get("static_params", {}),
                condition=raw_step.get("condition"),
                continue_on_failure=raw_step.get("continue_on_failure", False),
                timeout=raw_step.get("timeout", 30.0),
            )
            steps.append(step)

        # Parse conditions
        conditions = params.get("conditions", {})

        rule = WorkflowRule(
            rule_id=str(uuid.uuid4()),
            name=name,
            description=params.get("description", ""),
            event_type=event_type,
            event_source=params.get("event_source", ""),
            conditions=conditions,
            steps=steps,
            enabled=True,
            priority=params.get("priority", 0),
            escalate_on_failure=params.get("escalate_on_failure", False),
            created_at=datetime.utcnow().isoformat(),
        )

        self._rules[rule.rule_id] = rule
        self._save_data()

        return SkillResult(
            success=True,
            message=f"Workflow rule '{name}' created with {len(steps)} step(s). "
                    f"Triggers on {event_type}" +
                    (f" from {rule.event_source}" if rule.event_source else ""),
            data={
                "rule_id": rule.rule_id,
                "name": name,
                "event_type": event_type,
                "event_source": rule.event_source,
                "steps_count": len(steps),
                "escalate_on_failure": rule.escalate_on_failure,
            }
        )

    async def _trigger(self, params: Dict) -> SkillResult:
        """Trigger event-driven workflows by dispatching an event."""
        event_type = params.get("event_type", "").strip()
        if not event_type:
            return SkillResult(success=False, message="event_type is required")

        event_source = params.get("event_source", "")
        payload = params.get("payload", {})

        # Find matching rules (sorted by priority desc)
        matching_rules = self._find_matching_rules(event_type, event_source, payload)

        if not matching_rules:
            return SkillResult(
                success=True,
                message=f"No workflow rules matched event {event_type}"
                        + (f" from {event_source}" if event_source else ""),
                data={"matched_rules": 0, "executions": []}
            )

        # Execute all matching workflows
        executions = []
        for rule in matching_rules:
            execution = await self._execute_workflow(rule, event_type, event_source, payload)
            executions.append(execution)

        total = len(executions)
        successful = sum(1 for e in executions if e.success)
        escalated = sum(1 for e in executions if e.escalated)

        return SkillResult(
            success=successful > 0 or total == 0,
            message=f"Triggered {total} workflow(s): {successful} succeeded, "
                    f"{total - successful} failed"
                    + (f", {escalated} escalated" if escalated else ""),
            data={
                "matched_rules": total,
                "successful": successful,
                "failed": total - successful,
                "escalated": escalated,
                "executions": [e.to_dict() for e in executions],
            }
        )

    async def _list_rules(self, params: Dict) -> SkillResult:
        """List all workflow rules."""
        event_type = params.get("event_type")
        enabled_only = params.get("enabled_only", False)

        rules = list(self._rules.values())
        if event_type:
            rules = [r for r in rules if r.event_type == event_type]
        if enabled_only:
            rules = [r for r in rules if r.enabled]

        # Sort by priority desc
        rules.sort(key=lambda r: r.priority, reverse=True)

        return SkillResult(
            success=True,
            message=f"Found {len(rules)} workflow rule(s)",
            data={
                "rules": [r.to_dict() for r in rules],
                "total": len(rules),
            }
        )

    async def _get_rule(self, params: Dict) -> SkillResult:
        """Get details of a specific rule."""
        name = params.get("name", "").strip()
        rule = self._find_rule_by_name(name)
        if not rule:
            return SkillResult(success=False, message=f"Rule '{name}' not found")

        # Include recent executions
        recent = [
            e.to_dict() for e in self._executions
            if e.rule_name == name
        ][-10:]

        data = rule.to_dict()
        data["recent_executions"] = recent
        data["success_rate"] = (
            rule.total_successes / rule.total_triggers
            if rule.total_triggers > 0 else 0.0
        )

        return SkillResult(success=True, message=f"Rule '{name}' details", data=data)

    async def _update_rule(self, params: Dict) -> SkillResult:
        """Update an existing rule."""
        name = params.get("name", "").strip()
        rule = self._find_rule_by_name(name)
        if not rule:
            return SkillResult(success=False, message=f"Rule '{name}' not found")

        updated_fields = []

        for field_name in ["enabled", "priority", "escalate_on_failure", "conditions"]:
            if field_name in params:
                setattr(rule, field_name, params[field_name])
                updated_fields.append(field_name)

        # Handle steps update
        if "steps" in params:
            raw_steps = params["steps"]
            steps = []
            for i, raw_step in enumerate(raw_steps):
                step = WorkflowStep(
                    step_id=raw_step.get("step_id", f"step_{i+1}"),
                    skill_id=raw_step.get("skill_id", ""),
                    action=raw_step.get("action", ""),
                    param_mapping=raw_step.get("param_mapping", {}),
                    static_params=raw_step.get("static_params", {}),
                    condition=raw_step.get("condition"),
                    continue_on_failure=raw_step.get("continue_on_failure", False),
                    timeout=raw_step.get("timeout", 30.0),
                )
                steps.append(step)
            rule.steps = steps
            updated_fields.append("steps")

        self._save_data()

        return SkillResult(
            success=True,
            message=f"Updated rule '{name}': {', '.join(updated_fields)}",
            data={"updated_fields": updated_fields, "rule": rule.to_dict()}
        )

    async def _delete_rule(self, params: Dict) -> SkillResult:
        """Delete a workflow rule."""
        name = params.get("name", "").strip()
        rule = self._find_rule_by_name(name)
        if not rule:
            return SkillResult(success=False, message=f"Rule '{name}' not found")

        del self._rules[rule.rule_id]
        self._save_data()

        return SkillResult(
            success=True,
            message=f"Deleted workflow rule '{name}'",
            data={"deleted_rule_id": rule.rule_id, "name": name}
        )

    async def _get_executions(self, params: Dict) -> SkillResult:
        """View workflow execution history."""
        rule_name = params.get("rule_name")
        success_only = params.get("success_only", False)
        limit = params.get("limit", 20)

        executions = list(self._executions)
        if rule_name:
            executions = [e for e in executions if e.rule_name == rule_name]
        if success_only:
            executions = [e for e in executions if e.success]

        # Most recent first
        executions = executions[-limit:]
        executions.reverse()

        return SkillResult(
            success=True,
            message=f"Found {len(executions)} execution(s)",
            data={
                "executions": [e.to_dict() for e in executions],
                "total": len(executions),
            }
        )

    async def _get_stats(self, params: Dict) -> SkillResult:
        """Get workflow performance statistics."""
        total_rules = len(self._rules)
        active_rules = sum(1 for r in self._rules.values() if r.enabled)
        total_executions = len(self._executions)
        successful_executions = sum(1 for e in self._executions if e.success)
        escalated_count = sum(1 for e in self._executions if e.escalated)

        # Per-event-type stats
        by_event_type: Dict[str, Dict] = {}
        for rule in self._rules.values():
            et = rule.event_type
            if et not in by_event_type:
                by_event_type[et] = {"rules": 0, "triggers": 0, "successes": 0}
            by_event_type[et]["rules"] += 1
            by_event_type[et]["triggers"] += rule.total_triggers
            by_event_type[et]["successes"] += rule.total_successes

        # Top rules by trigger count
        top_rules = sorted(
            self._rules.values(),
            key=lambda r: r.total_triggers,
            reverse=True
        )[:5]

        # Average execution time
        durations = [e.duration_ms for e in self._executions if e.duration_ms > 0]
        avg_duration = sum(durations) / len(durations) if durations else 0.0

        overall_success_rate = (
            successful_executions / total_executions
            if total_executions > 0 else 0.0
        )

        return SkillResult(
            success=True,
            message=f"Workflow stats: {total_rules} rules ({active_rules} active), "
                    f"{total_executions} executions ({overall_success_rate:.0%} success rate)",
            data={
                "total_rules": total_rules,
                "active_rules": active_rules,
                "total_executions": total_executions,
                "successful_executions": successful_executions,
                "failed_executions": total_executions - successful_executions,
                "escalated_count": escalated_count,
                "success_rate": overall_success_rate,
                "avg_duration_ms": round(avg_duration, 2),
                "by_event_type": by_event_type,
                "top_rules": [
                    {"name": r.name, "triggers": r.total_triggers,
                     "success_rate": r.total_successes / r.total_triggers if r.total_triggers > 0 else 0}
                    for r in top_rules
                ],
            }
        )

    # --- Workflow Execution Engine ---

    async def _execute_workflow(
        self, rule: WorkflowRule, event_type: str,
        event_source: str, payload: Dict
    ) -> WorkflowExecution:
        """Execute a workflow rule's steps against an event payload."""
        start_time = time.monotonic()

        execution = WorkflowExecution(
            execution_id=str(uuid.uuid4()),
            rule_id=rule.rule_id,
            rule_name=rule.name,
            event_type=event_type,
            event_source=event_source,
            started_at=datetime.utcnow().isoformat(),
        )

        # Context for data flow between steps
        step_outputs: Dict[str, Dict] = {}
        all_succeeded = True
        steps_executed = 0

        for step in rule.steps:
            # Check condition
            if step.condition and not self._evaluate_condition(
                step.condition, payload, step_outputs
            ):
                execution.step_results.append({
                    "step_id": step.step_id,
                    "skill_id": step.skill_id,
                    "action": step.action,
                    "skipped": True,
                    "reason": "condition not met",
                })
                continue

            # Resolve parameters
            resolved_params = self._resolve_params(
                step.param_mapping, step.static_params, payload, step_outputs
            )

            # Execute step
            step_result = await self._execute_step(step, resolved_params)
            steps_executed += 1

            step_record = {
                "step_id": step.step_id,
                "skill_id": step.skill_id,
                "action": step.action,
                "success": step_result.success,
                "message": step_result.message[:200],
                "skipped": False,
            }
            execution.step_results.append(step_record)

            # Store output for data flow
            step_outputs[step.step_id] = step_result.data or {}

            if not step_result.success:
                all_succeeded = False
                if not step.continue_on_failure:
                    step_record["aborted_workflow"] = True
                    break

        execution.success = all_succeeded and steps_executed > 0
        execution.duration_ms = (time.monotonic() - start_time) * 1000
        execution.completed_at = datetime.utcnow().isoformat()

        # Escalate to autonomous loop if all steps failed and escalation is enabled
        if not execution.success and rule.escalate_on_failure:
            escalation_result = await self._escalate_to_autonomous_loop(
                rule, event_type, event_source, payload, execution
            )
            execution.escalated = True
            execution.escalation_result = escalation_result

        # Update rule stats
        rule.total_triggers += 1
        if execution.success:
            rule.total_successes += 1
        else:
            rule.total_failures += 1
        rule.last_triggered_at = datetime.utcnow().isoformat()
        # Rolling average of duration
        n = rule.total_triggers
        rule.avg_duration_ms = (
            (rule.avg_duration_ms * (n - 1) + execution.duration_ms) / n
        )

        # Record execution
        self._executions.append(execution)
        if len(self._executions) > self._max_executions:
            self._executions = self._executions[-self._max_executions:]

        self._save_data()
        return execution

    async def _execute_step(self, step: WorkflowStep, params: Dict) -> SkillResult:
        """Execute a single workflow step via skill context."""
        if self.context:
            try:
                return await self.context.call_skill(
                    step.skill_id, step.action, params
                )
            except Exception as e:
                return SkillResult(
                    success=False,
                    message=f"Step execution error: {str(e)[:200]}"
                )
        else:
            # No context - dry run mode
            return SkillResult(
                success=True,
                message=f"Dry run: {step.skill_id}:{step.action}",
                data={"params": params, "dry_run": True}
            )

    async def _escalate_to_autonomous_loop(
        self, rule: WorkflowRule, event_type: str,
        event_source: str, payload: Dict,
        execution: WorkflowExecution
    ) -> str:
        """Escalate failed workflow to the autonomous loop for decision-making."""
        if not self.context:
            return "No context available for escalation (dry run)"

        try:
            # Tell the autonomous loop to assess and decide what to do
            result = await self.context.call_skill(
                "autonomous_loop", "step", {
                    "force_assess": True,
                }
            )
            if result.success:
                return f"Escalated to autonomous loop: {result.message[:150]}"
            else:
                return f"Escalation attempted but loop returned: {result.message[:150]}"
        except Exception as e:
            return f"Escalation failed: {str(e)[:150]}"

    # --- Matching & Resolution ---

    def _find_matching_rules(
        self, event_type: str, event_source: str, payload: Dict
    ) -> List[WorkflowRule]:
        """Find all rules that match the given event."""
        matching = []

        for rule in sorted(
            self._rules.values(),
            key=lambda r: r.priority,
            reverse=True
        ):
            if not rule.enabled:
                continue

            # Match event type
            if rule.event_type != event_type:
                continue

            # Match event source (if specified in rule)
            if rule.event_source and rule.event_source != event_source:
                continue

            # Check payload conditions
            if rule.conditions and not self._check_conditions(payload, rule.conditions):
                continue

            matching.append(rule)

        return matching

    def _check_conditions(self, payload: Dict, conditions: Dict) -> bool:
        """Check if payload matches all conditions."""
        for path, condition in conditions.items():
            actual = self._extract_field(payload, path)
            op = condition.get("op", "eq")
            expected = condition.get("value")

            if not self._evaluate_op(actual, op, expected):
                return False
        return True

    def _evaluate_condition(
        self, condition: Dict, payload: Dict, step_outputs: Dict
    ) -> bool:
        """Evaluate a step condition."""
        field_path = condition.get("field", "")
        op = condition.get("op", "eq")
        expected = condition.get("value")

        # Resolve field from either payload or step outputs
        if field_path.startswith("step."):
            parts = field_path.split(".", 2)
            if len(parts) >= 3:
                step_id = parts[1]
                sub_path = parts[2]
                source = step_outputs.get(step_id, {})
                actual = self._extract_field(source, sub_path)
            else:
                actual = None
        else:
            actual = self._extract_field(payload, field_path)

        return self._evaluate_op(actual, op, expected)

    def _evaluate_op(self, actual: Any, op: str, expected: Any) -> bool:
        """Evaluate a comparison operation."""
        if op == "eq":
            return actual == expected
        elif op == "ne":
            return actual != expected
        elif op == "gt":
            try:
                return float(actual) > float(expected)
            except (TypeError, ValueError):
                return False
        elif op == "lt":
            try:
                return float(actual) < float(expected)
            except (TypeError, ValueError):
                return False
        elif op == "contains":
            if isinstance(actual, str) and isinstance(expected, str):
                return expected in actual
            if isinstance(actual, list):
                return expected in actual
            return False
        elif op == "exists":
            return actual is not None
        elif op == "not_exists":
            return actual is None
        return False

    def _resolve_params(
        self, param_mapping: Dict[str, str], static_params: Dict[str, Any],
        payload: Dict, step_outputs: Dict
    ) -> Dict[str, Any]:
        """Resolve step parameters from event payload and previous step outputs."""
        result = dict(static_params)

        for target_param, source_ref in param_mapping.items():
            if source_ref.startswith("event."):
                # Extract from event payload
                path = source_ref[6:]  # Remove "event." prefix
                value = self._extract_field(payload, path)
            elif source_ref.startswith("step."):
                # Extract from previous step output
                parts = source_ref.split(".", 2)
                if len(parts) >= 3:
                    step_id = parts[1]
                    sub_path = parts[2]
                    source = step_outputs.get(step_id, {})
                    value = self._extract_field(source, sub_path)
                else:
                    value = None
            else:
                # Literal value
                value = source_ref

            if value is not None:
                result[target_param] = value

        return result

    def _extract_field(self, data: Any, path: str) -> Any:
        """Extract a nested field using dot notation."""
        if not path:
            return data
        parts = path.split(".")
        current = data
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None
        return current

    # --- Helpers ---

    def _find_rule_by_name(self, name: str) -> Optional[WorkflowRule]:
        """Find a rule by name."""
        for rule in self._rules.values():
            if rule.name == name:
                return rule
        return None

    # --- Persistence ---

    def _get_data_path(self) -> Path:
        return self._data_dir / "event_workflows.json"

    def _save_data(self):
        """Save rules and executions to disk."""
        try:
            self._data_dir.mkdir(parents=True, exist_ok=True)
            data = {
                "rules": {
                    rid: r.to_dict() for rid, r in self._rules.items()
                },
                "executions": [e.to_dict() for e in self._executions[-100:]],
            }
            with open(self._get_data_path(), "w") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass

    def _load_data(self):
        """Load rules from disk."""
        path = self._get_data_path()
        if not path.exists():
            return
        try:
            with open(path) as f:
                data = json.load(f)

            for rid, rule_data in data.get("rules", {}).items():
                # Reconstruct steps
                raw_steps = rule_data.pop("steps", [])
                steps = []
                for s in raw_steps:
                    steps.append(WorkflowStep(**s))
                rule_data["steps"] = steps
                self._rules[rid] = WorkflowRule(**rule_data)

            for exec_data in data.get("executions", []):
                self._executions.append(WorkflowExecution(**exec_data))
        except Exception:
            pass
