#!/usr/bin/env python3
"""
PipelineExecutor - Multi-step action chains within a single agent cycle.

The fundamental limitation of the current agent loop is: one action per cycle.
This means tasks like "deploy a service" require many expensive LLM calls,
one per step, when the plan could be determined upfront and executed as a batch.

PipelineExecutor enables the agent to define and execute action pipelines:
- Sequential step execution with result passing between steps
- Conditional branching (on_success / on_failure paths)
- Cost guards and timeout limits per pipeline and per step
- Result aggregation and summary generation
- Step references: later steps can reference results from earlier steps

This is a force multiplier: the agent thinks once, executes many.

Example pipeline:
    [
        {"tool": "shell:run", "params": {"command": "git status"}},
        {"tool": "github:create_pr", "params": {"title": "Auto fix"},
         "condition": {"prev_contains": "modified"}},
        {"tool": "shell:run", "params": {"command": "pytest"},
         "on_failure": {"tool": "shell:run", "params": {"command": "git stash"}}}
    ]

Pillar: Self-Improvement (do more per cycle = more efficient = lower cost)
"""

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional


@dataclass
class PipelineStep:
    """A single step in an execution pipeline."""
    tool: str
    params: Dict[str, Any] = field(default_factory=dict)
    label: str = ""  # Human-readable label
    condition: Optional[Dict[str, Any]] = None  # When to execute
    on_failure: Optional[Dict[str, Any]] = None  # Fallback step
    timeout_seconds: float = 30.0
    max_cost: float = 0.05  # Max cost for this step
    required: bool = True  # If False, failure won't stop pipeline
    retry_count: int = 0  # Number of retries on failure


@dataclass
class StepResult:
    """Result of executing a single pipeline step."""
    step_index: int
    tool: str
    label: str
    success: bool
    result: Dict[str, Any] = field(default_factory=dict)
    error: str = ""
    duration_ms: float = 0.0
    cost: float = 0.0
    skipped: bool = False
    skip_reason: str = ""
    retries: int = 0


@dataclass
class PipelineResult:
    """Aggregate result of a complete pipeline execution."""
    success: bool
    steps: List[StepResult] = field(default_factory=list)
    total_duration_ms: float = 0.0
    total_cost: float = 0.0
    steps_executed: int = 0
    steps_succeeded: int = 0
    steps_failed: int = 0
    steps_skipped: int = 0
    aborted: bool = False
    abort_reason: str = ""

    def summary(self) -> str:
        """Generate a human-readable summary."""
        status = "SUCCESS" if self.success else "FAILED"
        parts = [
            f"Pipeline {status}: {self.steps_executed} steps executed",
            f"({self.steps_succeeded} ok, {self.steps_failed} failed, {self.steps_skipped} skipped)",
            f"in {self.total_duration_ms:.0f}ms, cost ${self.total_cost:.4f}",
        ]
        if self.aborted:
            parts.append(f"[ABORTED: {self.abort_reason}]")
        return " ".join(parts)


class PipelineExecutor:
    """
    Executes multi-step action pipelines with conditional logic and cost guards.

    Usage:
        executor = PipelineExecutor(execute_fn=agent._execute)
        result = await executor.run(steps, max_cost=0.10, timeout=60.0)
    """

    def __init__(
        self,
        execute_fn: Callable,
        max_pipeline_cost: float = 0.50,
        max_pipeline_timeout: float = 120.0,
        log_fn: Optional[Callable] = None,
    ):
        """
        Initialize the pipeline executor.

        Args:
            execute_fn: Async function that executes a tool action.
                       Signature: async (Action) -> Dict[str, Any]
            max_pipeline_cost: Maximum total cost for any pipeline
            max_pipeline_timeout: Maximum total time for any pipeline (seconds)
            log_fn: Optional logging function
        """
        self._execute_fn = execute_fn
        self._max_cost = max_pipeline_cost
        self._max_timeout = max_pipeline_timeout
        self._log = log_fn or (lambda *a: None)
        self._history: List[PipelineResult] = []

    def parse_steps(self, raw_steps: List[Dict]) -> List[PipelineStep]:
        """Parse raw step dicts into PipelineStep objects."""
        steps = []
        for i, raw in enumerate(raw_steps):
            step = PipelineStep(
                tool=raw.get("tool", ""),
                params=raw.get("params", {}),
                label=raw.get("label", f"step-{i}"),
                condition=raw.get("condition"),
                on_failure=raw.get("on_failure"),
                timeout_seconds=raw.get("timeout_seconds", 30.0),
                max_cost=raw.get("max_cost", 0.05),
                required=raw.get("required", True),
                retry_count=raw.get("retry_count", 0),
            )
            steps.append(step)
        return steps

    def _substitute_refs(self, params: Dict, prev_results: List[StepResult]) -> Dict:
        """
        Substitute references to previous step results in params.

        Supports patterns like:
            {"key": "$step.0.result.data.value"} -> result of step 0
            {"key": "$prev.result.status"} -> result of previous step
        """
        substituted = {}
        for key, value in params.items():
            if isinstance(value, str) and value.startswith("$"):
                substituted[key] = self._resolve_ref(value, prev_results)
            elif isinstance(value, dict):
                substituted[key] = self._substitute_refs(value, prev_results)
            else:
                substituted[key] = value
        return substituted

    def _resolve_ref(self, ref: str, prev_results: List[StepResult]) -> Any:
        """Resolve a reference like $step.0.result.status or $prev.result.data."""
        if not prev_results:
            return ref

        parts = ref.lstrip("$").split(".")
        if not parts:
            return ref

        # Determine which step to reference
        if parts[0] == "prev":
            target = prev_results[-1]
            parts = parts[1:]
        elif parts[0] == "step" and len(parts) > 1:
            try:
                idx = int(parts[1])
                if 0 <= idx < len(prev_results):
                    target = prev_results[idx]
                    parts = parts[2:]
                else:
                    return ref
            except (ValueError, IndexError):
                return ref
        else:
            return ref

        # Navigate into the target
        current: Any = target
        for part in parts:
            if isinstance(current, dict):
                current = current.get(part, ref)
            elif isinstance(current, StepResult):
                current = getattr(current, part, ref)
            else:
                return ref

        return current

    def _check_condition(self, condition: Dict, prev_results: List[StepResult]) -> bool:
        """
        Evaluate a step condition against previous results.

        Supported conditions:
            prev_success: True/False - previous step succeeded
            prev_contains: str - previous step result contains string
            step_success: {index: bool} - specific step succeeded
            any_failed: True - any previous step failed
        """
        if not condition:
            return True

        if not prev_results:
            # No previous results; skip condition-dependent steps
            return condition.get("always", False)

        prev = prev_results[-1]

        # Check previous step success
        if "prev_success" in condition:
            if prev.success != condition["prev_success"]:
                return False

        # Check if previous result contains a string
        if "prev_contains" in condition:
            search_str = condition["prev_contains"]
            result_str = str(prev.result)
            if search_str.lower() not in result_str.lower():
                return False

        # Check specific step success
        if "step_success" in condition:
            for idx_str, expected in condition["step_success"].items():
                idx = int(idx_str)
                if 0 <= idx < len(prev_results):
                    if prev_results[idx].success != expected:
                        return False

        # Check if any step failed
        if condition.get("any_failed"):
            if not any(r.success is False and not r.skipped for r in prev_results):
                return False

        return True

    async def _execute_step(
        self,
        step: PipelineStep,
        step_index: int,
        prev_results: List[StepResult],
    ) -> StepResult:
        """Execute a single pipeline step."""
        # Substitute references in params
        params = self._substitute_refs(step.params, prev_results)

        # Build action-like object for the execute function
        action = _PipelineAction(tool=step.tool, params=params)

        start_time = time.time()
        retries = 0
        last_error = ""

        while True:
            try:
                result = await asyncio.wait_for(
                    self._execute_fn(action),
                    timeout=step.timeout_seconds,
                )
                duration_ms = (time.time() - start_time) * 1000
                success = result.get("status") == "success"

                if not success and retries < step.retry_count:
                    retries += 1
                    last_error = str(result.get("message", ""))
                    self._log("PIPELINE", f"Step {step_index} failed, retry {retries}/{step.retry_count}")
                    await asyncio.sleep(0.5 * retries)  # Brief backoff
                    continue

                return StepResult(
                    step_index=step_index,
                    tool=step.tool,
                    label=step.label,
                    success=success,
                    result=result,
                    error=str(result.get("message", "")) if not success else "",
                    duration_ms=duration_ms,
                    retries=retries,
                )

            except asyncio.TimeoutError:
                duration_ms = (time.time() - start_time) * 1000
                if retries < step.retry_count:
                    retries += 1
                    self._log("PIPELINE", f"Step {step_index} timed out, retry {retries}/{step.retry_count}")
                    continue

                return StepResult(
                    step_index=step_index,
                    tool=step.tool,
                    label=step.label,
                    success=False,
                    error=f"Timeout after {step.timeout_seconds}s",
                    duration_ms=duration_ms,
                    retries=retries,
                )

            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                if retries < step.retry_count:
                    retries += 1
                    last_error = str(e)
                    self._log("PIPELINE", f"Step {step_index} error: {e}, retry {retries}/{step.retry_count}")
                    await asyncio.sleep(0.5 * retries)
                    continue

                return StepResult(
                    step_index=step_index,
                    tool=step.tool,
                    label=step.label,
                    success=False,
                    error=str(e),
                    duration_ms=duration_ms,
                    retries=retries,
                )

    async def run(
        self,
        steps: List[PipelineStep],
        max_cost: Optional[float] = None,
        timeout: Optional[float] = None,
    ) -> PipelineResult:
        """
        Execute a pipeline of steps.

        Args:
            steps: Ordered list of PipelineStep objects
            max_cost: Override max pipeline cost (default: self._max_cost)
            timeout: Override max pipeline timeout (default: self._max_timeout)

        Returns:
            PipelineResult with aggregate metrics
        """
        max_cost = max_cost or self._max_cost
        timeout = timeout or self._max_timeout

        pipeline_start = time.time()
        result = PipelineResult(success=True)
        step_results: List[StepResult] = []

        self._log("PIPELINE", f"Starting pipeline with {len(steps)} steps (max ${max_cost:.2f}, timeout {timeout}s)")

        for i, step in enumerate(steps):
            # Check pipeline timeout
            elapsed = time.time() - pipeline_start
            if elapsed > timeout:
                result.aborted = True
                result.abort_reason = f"Pipeline timeout ({timeout}s) exceeded"
                result.success = False
                break

            # Check pipeline cost
            if result.total_cost >= max_cost:
                result.aborted = True
                result.abort_reason = f"Pipeline cost limit (${max_cost:.2f}) exceeded"
                result.success = False
                break

            # Check condition
            if step.condition and not self._check_condition(step.condition, step_results):
                skip_result = StepResult(
                    step_index=i,
                    tool=step.tool,
                    label=step.label,
                    success=True,
                    skipped=True,
                    skip_reason="Condition not met",
                )
                step_results.append(skip_result)
                result.steps_skipped += 1
                self._log("PIPELINE", f"Step {i} ({step.label}): SKIPPED (condition not met)")
                continue

            # Execute step
            self._log("PIPELINE", f"Step {i} ({step.label}): {step.tool}")
            step_result = await self._execute_step(step, i, step_results)
            step_results.append(step_result)
            result.steps_executed += 1
            result.total_cost += step_result.cost

            if step_result.success:
                result.steps_succeeded += 1
                self._log("PIPELINE", f"Step {i} ({step.label}): OK ({step_result.duration_ms:.0f}ms)")
            else:
                result.steps_failed += 1
                self._log("PIPELINE", f"Step {i} ({step.label}): FAILED - {step_result.error}")

                # Try on_failure fallback
                if step.on_failure:
                    fallback_step = PipelineStep(
                        tool=step.on_failure.get("tool", ""),
                        params=step.on_failure.get("params", {}),
                        label=f"fallback-{i}",
                        timeout_seconds=step.timeout_seconds,
                    )
                    self._log("PIPELINE", f"Step {i}: trying fallback → {fallback_step.tool}")
                    fallback_result = await self._execute_step(fallback_step, i, step_results)
                    step_results.append(fallback_result)
                    result.steps_executed += 1
                    if fallback_result.success:
                        result.steps_succeeded += 1
                        self._log("PIPELINE", f"Step {i} fallback: OK")
                        continue  # Don't abort, fallback worked
                    else:
                        result.steps_failed += 1
                        self._log("PIPELINE", f"Step {i} fallback: FAILED - {fallback_result.error}")

                # If required step failed (and no successful fallback), abort
                if step.required:
                    result.success = False
                    result.aborted = True
                    result.abort_reason = f"Required step {i} ({step.label}) failed: {step_result.error}"
                    break

        result.steps = step_results
        result.total_duration_ms = (time.time() - pipeline_start) * 1000

        self._log("PIPELINE", result.summary())
        self._history.append(result)

        return result

    async def run_from_dicts(
        self,
        raw_steps: List[Dict],
        max_cost: Optional[float] = None,
        timeout: Optional[float] = None,
    ) -> PipelineResult:
        """Convenience: parse and run a pipeline from raw dicts."""
        steps = self.parse_steps(raw_steps)
        return await self.run(steps, max_cost=max_cost, timeout=timeout)

    def get_history(self, limit: int = 10) -> List[Dict]:
        """Get recent pipeline execution history."""
        results = []
        for pr in self._history[-limit:]:
            results.append({
                "success": pr.success,
                "steps_executed": pr.steps_executed,
                "steps_succeeded": pr.steps_succeeded,
                "steps_failed": pr.steps_failed,
                "steps_skipped": pr.steps_skipped,
                "total_duration_ms": pr.total_duration_ms,
                "total_cost": pr.total_cost,
                "aborted": pr.aborted,
                "abort_reason": pr.abort_reason,
                "summary": pr.summary(),
            })
        return results

    def get_stats(self) -> Dict:
        """Get aggregate pipeline execution statistics."""
        if not self._history:
            return {
                "total_pipelines": 0,
                "success_rate": 0.0,
                "avg_steps": 0.0,
                "avg_duration_ms": 0.0,
                "total_cost": 0.0,
            }
        total = len(self._history)
        successes = sum(1 for r in self._history if r.success)
        return {
            "total_pipelines": total,
            "success_rate": successes / total if total else 0.0,
            "avg_steps": sum(r.steps_executed for r in self._history) / total,
            "avg_duration_ms": sum(r.total_duration_ms for r in self._history) / total,
            "total_cost": sum(r.total_cost for r in self._history),
        }


class _PipelineAction:
    """Minimal action object for the execute function interface."""

    def __init__(self, tool: str, params: Dict):
        self.tool = tool
        self.params = params
        self.reasoning = "Pipeline step"
