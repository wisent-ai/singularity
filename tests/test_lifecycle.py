"""Tests for lifecycle hooks system."""
import pytest
from singularity.lifecycle import (
    LifecycleHook, HookManager, StartupInfo, CycleInfo, CycleResult, ShutdownInfo,
    OutcomeTrackingHook, CycleMetricsHook, AdaptiveIntervalHook,
)


def _startup():
    return StartupInfo(agent_name="Test", agent_ticker="TST", agent_type="general",
                       balance=100.0, instance_type="local", skill_count=3, skill_ids=["a","b","c"])

def _cycle(n=1):
    return CycleInfo(cycle=n, balance=90.0, runway_cycles=100, runway_hours=10)

def _result(cycle=1, success=True, tool="test:action", cost=0.01):
    return CycleResult(cycle=cycle, tool=tool, params={}, result={"status": "success" if success else "error"},
                       success=success, api_cost=cost, tokens_used=100, duration_seconds=1.0, balance_after=89.0)

def _shutdown():
    return ShutdownInfo(total_cycles=10, total_api_cost=0.1, total_tokens=1000,
                        balance_remaining=90.0, runtime_seconds=60.0)


class TestHookManager:
    def test_add_remove(self):
        mgr = HookManager()
        hook = OutcomeTrackingHook()
        mgr.add(hook)
        assert len(mgr.hooks) == 1
        assert mgr.get("OutcomeTrackingHook") is hook
        assert mgr.remove("OutcomeTrackingHook")
        assert len(mgr.hooks) == 0

    def test_dispatch_startup(self):
        mgr = HookManager()
        hook = CycleMetricsHook()
        mgr.add(hook)
        mgr.dispatch_startup(_startup())
        assert hook._start_time is not None

    def test_dispatch_post_cycle(self):
        mgr = HookManager()
        hook = OutcomeTrackingHook()
        mgr.add(hook)
        mgr.dispatch_post_cycle(_result(success=True))
        assert hook._total_success == 1

    def test_pre_cycle_merges_hints(self):
        class HintHook(LifecycleHook):
            def pre_cycle(self, info):
                return {"context_hint": "do something"}
        mgr = HookManager()
        mgr.add(HintHook())
        hints = mgr.dispatch_pre_cycle(_cycle())
        assert "do something" in hints["context_hints"]

    def test_error_in_hook_doesnt_crash(self):
        class BadHook(LifecycleHook):
            def on_startup(self, info):
                raise RuntimeError("boom")
        mgr = HookManager()
        mgr.add(BadHook())
        mgr.dispatch_startup(_startup())  # Should not raise

    def test_get_all_status(self):
        mgr = HookManager()
        mgr.add(OutcomeTrackingHook())
        mgr.add(CycleMetricsHook())
        statuses = mgr.get_all_status()
        assert len(statuses) == 2


class TestOutcomeTracking:
    def test_success_tracking(self):
        hook = OutcomeTrackingHook()
        for i in range(5):
            hook.post_cycle(_result(cycle=i, success=True))
        assert hook.success_rate == 1.0
        assert hook._current_streak == 5
        assert hook._best_streak == 5

    def test_failure_tracking(self):
        hook = OutcomeTrackingHook()
        for i in range(3):
            hook.post_cycle(_result(cycle=i, success=False))
        assert hook.success_rate == 0.0
        assert hook._current_streak == -3
        assert hook._worst_streak == -3

    def test_mixed_outcomes(self):
        hook = OutcomeTrackingHook()
        hook.post_cycle(_result(cycle=1, success=True))
        hook.post_cycle(_result(cycle=2, success=False))
        hook.post_cycle(_result(cycle=3, success=True))
        assert hook.success_rate == pytest.approx(2/3)
        assert hook._current_streak == 1

    def test_per_skill_tracking(self):
        hook = OutcomeTrackingHook()
        hook.post_cycle(_result(tool="fs:read", success=True))
        hook.post_cycle(_result(tool="fs:write", success=False))
        hook.post_cycle(_result(tool="shell:bash", success=True))
        assert hook.skill_success_rate("fs") == 0.5
        assert hook.skill_success_rate("shell") == 1.0

    def test_failure_warning_hint(self):
        hook = OutcomeTrackingHook()
        for i in range(5):
            hook.post_cycle(_result(cycle=i, success=False))
        hint = hook.pre_cycle(_cycle())
        assert hint is not None
        assert "WARNING" in hint["context_hint"]


class TestCycleMetrics:
    def test_tracks_metrics(self):
        hook = CycleMetricsHook()
        hook.on_startup(_startup())
        hook.post_cycle(_result(cost=0.01))
        hook.post_cycle(_result(cost=0.02))
        assert hook.avg_cost == pytest.approx(0.015)
        assert hook._total_cycles == 2

    def test_status(self):
        hook = CycleMetricsHook()
        hook.on_startup(_startup())
        hook.post_cycle(_result())
        status = hook.get_status()
        assert status["total_cycles"] == 1
        assert "duration" in status
        assert "cost" in status


class TestAdaptiveInterval:
    def test_suggests_speedup(self):
        hook = AdaptiveIntervalHook()
        for i in range(10):
            hook.post_cycle(_result(cycle=i, success=True))
        hint = hook.pre_cycle(_cycle())
        if hint:
            assert hint.get("suggested_interval", 999) < 5.0

    def test_suggests_slowdown(self):
        hook = AdaptiveIntervalHook()
        for i in range(10):
            hook.post_cycle(_result(cycle=i, success=False))
        assert hook._suggested_interval is not None
        assert hook._suggested_interval > 1.0
