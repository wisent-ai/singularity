"""Tests for RuntimeMetrics."""
import time
from singularity.runtime_metrics import RuntimeMetrics


def test_initial_state():
    m = RuntimeMetrics()
    assert m.total_cycles == 0
    assert m.success_rate == 1.0
    assert m.avg_decision_latency == 0.0
    assert m.avg_execution_latency == 0.0


def test_record_decision():
    m = RuntimeMetrics()
    m.record_decision(latency=0.5, tokens=100, api_cost=0.001)
    assert m.total_tokens == 100
    assert m.total_api_cost == 0.001
    assert m.avg_decision_latency == 0.5
    assert m.avg_tokens_per_cycle == 100
    assert m.avg_cost_per_cycle == 0.001


def test_record_execution():
    m = RuntimeMetrics()
    m.record_execution("shell:bash", 0.2, True)
    m.record_execution("filesystem:view", 0.1, True)
    m.record_execution("shell:bash", 0.3, False)
    assert m.total_cycles == 3
    assert m.total_successes == 2
    assert m.total_failures == 1
    assert abs(m.success_rate - 2/3) < 0.01


def test_top_tools():
    m = RuntimeMetrics()
    for _ in range(5):
        m.record_execution("shell:bash", 0.1, True)
    for _ in range(3):
        m.record_execution("fs:view", 0.1, True)
    top = m.top_tools(2)
    assert len(top) == 2
    assert top[0]["tool"] == "shell:bash"
    assert top[0]["count"] == 5


def test_failing_tools():
    m = RuntimeMetrics()
    m.record_execution("bad:tool", 0.1, False)
    m.record_execution("bad:tool", 0.1, False)
    m.record_execution("good:tool", 0.1, True)
    failing = m.failing_tools()
    assert len(failing) == 1
    assert failing[0]["tool"] == "bad:tool"
    assert failing[0]["failure_rate"] == 1.0


def test_timer():
    m = RuntimeMetrics()
    m.start_timer("decision")
    time.sleep(0.01)
    elapsed = m.stop_timer("decision")
    assert elapsed is not None
    assert elapsed >= 0.01
    assert m.avg_decision_latency >= 0.01


def test_summary():
    m = RuntimeMetrics()
    m.record_decision(0.5, 200, 0.002)
    m.record_execution("shell:bash", 0.1, True)
    s = m.summary()
    assert "total_cycles" in s
    assert s["total_cycles"] == 1
    assert s["total_tokens"] == 200


def test_context_summary_empty():
    m = RuntimeMetrics()
    assert m.context_summary() == ""


def test_context_summary_with_data():
    m = RuntimeMetrics()
    m.record_decision(0.5, 100, 0.001)
    m.record_execution("shell:bash", 0.1, True)
    ctx = m.context_summary()
    assert "Performance" in ctx
    assert "Success rate" in ctx


def test_window_size():
    m = RuntimeMetrics(window_size=3)
    for i in range(5):
        m.record_decision(float(i), 10, 0.001)
    # Only last 3 should be in the window
    assert len(m._decision_latencies) == 3
    assert m.avg_decision_latency == (2.0 + 3.0 + 4.0) / 3


def test_record_error():
    m = RuntimeMetrics()
    m.record_error("bad:tool")
    assert m.total_errors == 1
    assert m._tool_failures["bad:tool"] == 1
