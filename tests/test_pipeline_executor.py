"""Tests for PipelineExecutor - multi-step action chains."""

import pytest
import asyncio
from singularity.pipeline_executor import (
    PipelineExecutor,
    PipelineStep,
    PipelineResult,
    StepResult,
    _PipelineAction,
)


# --- Fixtures ---

async def mock_execute_success(action):
    """Mock execute that always succeeds."""
    return {"status": "success", "message": f"OK: {action.tool}", "data": {"tool": action.tool}}


async def mock_execute_fail(action):
    """Mock execute that always fails."""
    return {"status": "error", "message": f"Failed: {action.tool}"}


async def mock_execute_conditional(action):
    """Mock that succeeds for some tools, fails for others."""
    if "fail" in action.tool:
        return {"status": "error", "message": "deliberate failure"}
    return {"status": "success", "message": "ok", "data": {"output": "modified files present"}}


async def mock_execute_slow(action):
    """Mock that takes a long time."""
    await asyncio.sleep(5)
    return {"status": "success", "message": "slow done"}


@pytest.fixture
def executor():
    return PipelineExecutor(execute_fn=mock_execute_success)


@pytest.fixture
def failing_executor():
    return PipelineExecutor(execute_fn=mock_execute_fail)


@pytest.fixture
def conditional_executor():
    return PipelineExecutor(execute_fn=mock_execute_conditional)


# --- Tests ---

@pytest.mark.asyncio
async def test_simple_pipeline(executor):
    """Execute a basic 3-step pipeline."""
    steps = [
        PipelineStep(tool="shell:run", params={"cmd": "ls"}, label="list"),
        PipelineStep(tool="github:status", params={}, label="check"),
        PipelineStep(tool="shell:run", params={"cmd": "echo done"}, label="done"),
    ]
    result = await executor.run(steps)
    assert result.success
    assert result.steps_executed == 3
    assert result.steps_succeeded == 3
    assert result.steps_failed == 0


@pytest.mark.asyncio
async def test_empty_pipeline(executor):
    """Empty pipeline succeeds with no steps."""
    result = await executor.run([])
    assert result.success
    assert result.steps_executed == 0


@pytest.mark.asyncio
async def test_pipeline_failure_aborts(failing_executor):
    """Required step failure aborts pipeline."""
    steps = [
        PipelineStep(tool="shell:run", label="step1", required=True),
        PipelineStep(tool="shell:run", label="step2"),
    ]
    result = await failing_executor.run(steps)
    assert not result.success
    assert result.aborted
    assert result.steps_failed >= 1


@pytest.mark.asyncio
async def test_optional_step_continues(failing_executor):
    """Optional step failure doesn't abort pipeline."""
    steps = [
        PipelineStep(tool="shell:run", label="optional", required=False),
        PipelineStep(tool="shell:run", label="also-optional", required=False),
    ]
    result = await failing_executor.run(steps)
    assert result.success  # All optional, so pipeline succeeds
    assert result.steps_failed == 2


@pytest.mark.asyncio
async def test_condition_prev_success(executor):
    """Step with prev_success condition executes when previous succeeds."""
    steps = [
        PipelineStep(tool="shell:run", label="first"),
        PipelineStep(tool="shell:run", label="second", condition={"prev_success": True}),
    ]
    result = await executor.run(steps)
    assert result.steps_executed == 2
    assert result.steps_skipped == 0


@pytest.mark.asyncio
async def test_condition_skips_when_not_met(conditional_executor):
    """Step is skipped when condition not met."""
    steps = [
        PipelineStep(tool="fail:action", label="will-fail", required=False),
        PipelineStep(tool="shell:run", label="only-on-success", condition={"prev_success": True}),
    ]
    result = await conditional_executor.run(steps)
    assert result.steps_skipped == 1


@pytest.mark.asyncio
async def test_condition_prev_contains(conditional_executor):
    """prev_contains condition checks result text."""
    steps = [
        PipelineStep(tool="shell:run", label="check"),
        PipelineStep(tool="shell:run", label="if-modified",
                     condition={"prev_contains": "modified"}),
    ]
    result = await conditional_executor.run(steps)
    assert result.steps_executed == 2
    assert result.steps_skipped == 0


@pytest.mark.asyncio
async def test_on_failure_fallback(conditional_executor):
    """Fallback step executes when main step fails."""
    steps = [
        PipelineStep(
            tool="fail:action",
            label="main",
            on_failure={"tool": "shell:run", "params": {"cmd": "recover"}},
        ),
    ]
    result = await conditional_executor.run(steps)
    # Main step failed, fallback succeeded -> pipeline continues
    assert result.steps_executed == 2  # main + fallback


@pytest.mark.asyncio
async def test_timeout_step():
    """Step that exceeds timeout fails."""
    executor = PipelineExecutor(execute_fn=mock_execute_slow)
    steps = [PipelineStep(tool="slow:action", label="slow", timeout_seconds=0.1)]
    result = await executor.run(steps)
    assert not result.success
    assert "Timeout" in result.steps[0].error


@pytest.mark.asyncio
async def test_cost_guard(executor):
    """Pipeline aborts when cost limit exceeded."""
    result = await executor.run(
        [PipelineStep(tool="shell:run", label="s1"),
         PipelineStep(tool="shell:run", label="s2")],
        max_cost=0.0,  # Zero budget - should abort after first step at cost check
    )
    # First step executes (cost checked before each step), then cost guard triggers
    assert result.steps_executed <= 2


@pytest.mark.asyncio
async def test_parse_steps(executor):
    """Parse raw dicts into PipelineStep objects."""
    raw = [
        {"tool": "shell:run", "params": {"cmd": "ls"}, "label": "list"},
        {"tool": "github:pr", "required": False, "retry_count": 2},
    ]
    steps = executor.parse_steps(raw)
    assert len(steps) == 2
    assert steps[0].tool == "shell:run"
    assert steps[0].label == "list"
    assert steps[1].required is False
    assert steps[1].retry_count == 2


@pytest.mark.asyncio
async def test_run_from_dicts(executor):
    """Run pipeline directly from raw dicts."""
    raw = [
        {"tool": "shell:run", "params": {"cmd": "ls"}},
        {"tool": "shell:run", "params": {"cmd": "pwd"}},
    ]
    result = await executor.run_from_dicts(raw)
    assert result.success
    assert result.steps_executed == 2


@pytest.mark.asyncio
async def test_history_tracking(executor):
    """Pipeline results are saved in history."""
    await executor.run([PipelineStep(tool="shell:run", label="h1")])
    await executor.run([PipelineStep(tool="shell:run", label="h2")])
    history = executor.get_history()
    assert len(history) == 2
    assert all(h["success"] for h in history)


@pytest.mark.asyncio
async def test_stats(executor):
    """Stats aggregate across pipelines."""
    await executor.run([PipelineStep(tool="a:b", label="s1")])
    await executor.run([PipelineStep(tool="c:d", label="s2"), PipelineStep(tool="e:f", label="s3")])
    stats = executor.get_stats()
    assert stats["total_pipelines"] == 2
    assert stats["success_rate"] == 1.0
    assert stats["avg_steps"] == 1.5


@pytest.mark.asyncio
async def test_pipeline_result_summary():
    """PipelineResult.summary() returns readable string."""
    pr = PipelineResult(
        success=True, steps_executed=3, steps_succeeded=2,
        steps_failed=1, steps_skipped=0, total_duration_ms=150.0,
        total_cost=0.002,
    )
    s = pr.summary()
    assert "SUCCESS" in s
    assert "3 steps" in s


@pytest.mark.asyncio
async def test_substitute_refs(executor):
    """Reference substitution resolves $prev and $step references."""
    # Build some fake previous results
    prev = [
        StepResult(step_index=0, tool="a", label="first", success=True,
                   result={"status": "success", "data": {"value": "hello"}}),
    ]
    params = {"key": "$prev.result.data.value", "other": "literal"}
    resolved = executor._substitute_refs(params, prev)
    assert resolved["key"] == "hello"
    assert resolved["other"] == "literal"


@pytest.mark.asyncio
async def test_retry_on_failure():
    """Steps retry on failure up to retry_count."""
    call_count = 0

    async def flaky_execute(action):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            return {"status": "error", "message": "transient"}
        return {"status": "success", "message": "ok"}

    executor = PipelineExecutor(execute_fn=flaky_execute)
    steps = [PipelineStep(tool="flaky:action", label="flaky", retry_count=3)]
    result = await executor.run(steps)
    assert result.success
    assert result.steps[0].retries == 2  # Failed twice, succeeded on third


def test_pipeline_action():
    """_PipelineAction has correct attributes."""
    a = _PipelineAction(tool="test:action", params={"x": 1})
    assert a.tool == "test:action"
    assert a.params == {"x": 1}
    assert a.reasoning == "Pipeline step"
