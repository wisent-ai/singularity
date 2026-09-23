"""Tests for RetryExecutor."""
import asyncio
import pytest
import time
from unittest.mock import AsyncMock, MagicMock
from dataclasses import dataclass
from singularity.retry_executor import RetryExecutor, SkillHealth, TRANSIENT_PATTERNS


@dataclass
class FakeResult:
    success: bool
    data: dict = None
    message: str = ""
    def __post_init__(self):
        self.data = self.data or {}


class FakeSkill:
    def __init__(self, results=None):
        self._results = results or []
        self._call_count = 0

    async def execute(self, action, params):
        idx = min(self._call_count, len(self._results) - 1)
        self._call_count += 1
        r = self._results[idx]
        if isinstance(r, Exception):
            raise r
        return r


class FakeRegistry:
    def __init__(self, skills=None):
        self._skills = skills or {}
    def get(self, skill_id):
        return self._skills.get(skill_id)


@pytest.fixture
def registry():
    return FakeRegistry()


def test_wait_action(registry):
    executor = RetryExecutor(registry)
    result = asyncio.get_event_loop().run_until_complete(executor.execute("wait", {}))
    assert result["status"] == "waited"


def test_unknown_tool_format(registry):
    executor = RetryExecutor(registry)
    result = asyncio.get_event_loop().run_until_complete(executor.execute("badtool", {}))
    assert result["status"] == "error"


def test_skill_not_found(registry):
    executor = RetryExecutor(registry)
    result = asyncio.get_event_loop().run_until_complete(executor.execute("missing:action", {}))
    assert result["status"] == "error"
    assert "not found" in result["message"]


def test_successful_execution():
    skill = FakeSkill([FakeResult(success=True, data={"key": "val"}, message="ok")])
    reg = FakeRegistry({"myskill": skill})
    executor = RetryExecutor(reg)
    result = asyncio.get_event_loop().run_until_complete(executor.execute("myskill:do", {}))
    assert result["status"] == "success"
    assert result["data"]["key"] == "val"


def test_non_transient_failure_no_retry():
    skill = FakeSkill([FakeResult(success=False, message="invalid parameter")])
    reg = FakeRegistry({"s": skill})
    executor = RetryExecutor(reg, max_retries=2, base_delay=0.01)
    result = asyncio.get_event_loop().run_until_complete(executor.execute("s:act", {}))
    assert result["status"] == "failed"
    assert skill._call_count == 1  # No retry for non-transient


def test_transient_failure_retries():
    skill = FakeSkill([
        FakeResult(success=False, message="rate limit exceeded"),
        FakeResult(success=True, data={"ok": True}),
    ])
    reg = FakeRegistry({"s": skill})
    executor = RetryExecutor(reg, max_retries=2, base_delay=0.01)
    result = asyncio.get_event_loop().run_until_complete(executor.execute("s:act", {}))
    assert result["status"] == "success"
    assert skill._call_count == 2


def test_transient_exception_retries():
    skill = FakeSkill([
        ConnectionError("timeout connecting"),
        FakeResult(success=True, data={"recovered": True}),
    ])
    reg = FakeRegistry({"s": skill})
    executor = RetryExecutor(reg, max_retries=2, base_delay=0.01)
    result = asyncio.get_event_loop().run_until_complete(executor.execute("s:act", {}))
    assert result["status"] == "success"
    assert skill._call_count == 2


def test_non_transient_exception_no_retry():
    skill = FakeSkill([ValueError("bad value")])
    reg = FakeRegistry({"s": skill})
    executor = RetryExecutor(reg, max_retries=2, base_delay=0.01)
    result = asyncio.get_event_loop().run_until_complete(executor.execute("s:act", {}))
    assert result["status"] == "error"
    assert skill._call_count == 1


def test_circuit_breaker_opens():
    skill = FakeSkill([FakeResult(success=False, message="server error 503")] * 10)
    reg = FakeRegistry({"s": skill})
    executor = RetryExecutor(reg, max_retries=0, circuit_break_threshold=3, circuit_break_duration=60)
    # 3 failures to trigger circuit
    for _ in range(3):
        asyncio.get_event_loop().run_until_complete(executor.execute("s:act", {}))
    # Next call should be blocked by circuit
    result = asyncio.get_event_loop().run_until_complete(executor.execute("s:act", {}))
    assert result["status"] == "error"
    assert "disabled" in result["message"]


def test_circuit_breaker_resets_on_success():
    health = SkillHealth(skill_id="test")
    for _ in range(5):
        health.record_failure("err")
    health.open_circuit(60)
    assert health.is_circuit_open
    # Simulate success after circuit expires
    health.circuit_open_until = 0
    health.record_success()
    assert health.consecutive_failures == 0
    assert not health.is_circuit_open


def test_health_summary_empty():
    executor = RetryExecutor(FakeRegistry())
    assert executor.get_health_summary() == ""


def test_health_summary_with_warnings():
    executor = RetryExecutor(FakeRegistry())
    h = executor._get_health("badskill")
    for _ in range(5):
        h.record_failure("err")
    h.open_circuit(60)
    summary = executor.get_health_summary()
    assert "badskill" in summary
    assert "DISABLED" in summary


def test_get_all_health():
    executor = RetryExecutor(FakeRegistry())
    h = executor._get_health("s1")
    h.record_success()
    h.record_failure("err")
    info = executor.get_all_health()
    assert info["s1"]["total_calls"] == 2
    assert info["s1"]["successes"] == 1


def test_reset_health():
    executor = RetryExecutor(FakeRegistry())
    executor._get_health("s1").record_success()
    executor._get_health("s2").record_success()
    executor.reset_health("s1")
    assert "s1" not in executor._health
    assert "s2" in executor._health
    executor.reset_health()
    assert len(executor._health) == 0


def test_success_rate():
    h = SkillHealth(skill_id="test")
    assert h.success_rate == 1.0
    h.record_success()
    h.record_failure("err")
    assert h.success_rate == 0.5
