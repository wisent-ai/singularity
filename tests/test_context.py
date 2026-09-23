"""Tests for ContextManager - unified context merging."""

from singularity.context import (
    ContextManager,
    ContextProvider,
    StaticProvider,
    CallableProvider,
)


class TestStaticProvider:
    def test_basic(self):
        p = StaticProvider("goals", "Build revenue", priority=10)
        assert p.name == "goals"
        assert p.priority == 10
        assert p.enabled is True
        assert p.get_context() == "Build revenue"

    def test_set_content(self):
        p = StaticProvider("x", "old")
        p.set_content("new")
        assert p.get_context() == "new"

    def test_disable(self):
        p = StaticProvider("x", "data")
        p.set_enabled(False)
        assert p.enabled is False


class TestCallableProvider:
    def test_basic(self):
        p = CallableProvider("stats", lambda: "ok", priority=20)
        assert p.name == "stats"
        assert p.get_context() == "ok"

    def test_exception_returns_empty(self):
        def bad():
            raise ValueError("boom")
        p = CallableProvider("bad", bad)
        assert p.get_context() == ""

    def test_disable(self):
        p = CallableProvider("x", lambda: "y")
        p.set_enabled(False)
        assert p.enabled is False


class TestContextManager:
    def test_empty(self):
        ctx = ContextManager()
        assert ctx.get_merged_context() == ""
        assert ctx.provider_count == 0

    def test_single_provider(self):
        ctx = ContextManager()
        ctx.register(StaticProvider("goals", "Ship feature X"))
        merged = ctx.get_merged_context()
        assert "=== goals ===" in merged
        assert "Ship feature X" in merged

    def test_priority_ordering(self):
        ctx = ContextManager()
        ctx.register(StaticProvider("low", "LOW", priority=90))
        ctx.register(StaticProvider("high", "HIGH", priority=10))
        ctx.register(StaticProvider("mid", "MID", priority=50))
        merged = ctx.get_merged_context()
        high_pos = merged.index("HIGH")
        mid_pos = merged.index("MID")
        low_pos = merged.index("LOW")
        assert high_pos < mid_pos < low_pos

    def test_disabled_provider_skipped(self):
        ctx = ContextManager()
        p = StaticProvider("skip", "HIDDEN")
        p.set_enabled(False)
        ctx.register(p)
        ctx.register(StaticProvider("show", "VISIBLE"))
        merged = ctx.get_merged_context()
        assert "HIDDEN" not in merged
        assert "VISIBLE" in merged

    def test_empty_content_skipped(self):
        ctx = ContextManager()
        ctx.register(StaticProvider("empty", ""))
        ctx.register(StaticProvider("full", "data"))
        merged = ctx.get_merged_context()
        assert "=== empty ===" not in merged
        assert "data" in merged

    def test_max_chars_truncation(self):
        ctx = ContextManager(max_chars=50)
        ctx.register(StaticProvider("big", "x" * 200))
        merged = ctx.get_merged_context()
        assert len(merged) == 50
        assert merged.endswith("...")

    def test_no_truncation_when_under_limit(self):
        ctx = ContextManager(max_chars=5000)
        ctx.register(StaticProvider("small", "hello"))
        merged = ctx.get_merged_context()
        assert "..." not in merged

    def test_register_replaces_existing(self):
        ctx = ContextManager()
        ctx.register(StaticProvider("x", "old"))
        ctx.register(StaticProvider("x", "new"))
        assert ctx.provider_count == 1
        assert "new" in ctx.get_merged_context()

    def test_unregister(self):
        ctx = ContextManager()
        ctx.register(StaticProvider("x", "data"))
        assert ctx.unregister("x") is True
        assert ctx.unregister("x") is False
        assert ctx.provider_count == 0

    def test_get_provider(self):
        ctx = ContextManager()
        p = StaticProvider("x", "data")
        ctx.register(p)
        assert ctx.get_provider("x") is p
        assert ctx.get_provider("nope") is None

    def test_list_providers(self):
        ctx = ContextManager()
        ctx.register(StaticProvider("b", "B", priority=20))
        ctx.register(StaticProvider("a", "A", priority=10))
        providers = ctx.list_providers()
        assert len(providers) == 2
        assert providers[0]["name"] == "a"
        assert providers[1]["name"] == "b"

    def test_active_provider_count(self):
        ctx = ContextManager()
        p1 = StaticProvider("a", "A")
        p2 = StaticProvider("b", "B")
        p2.set_enabled(False)
        ctx.register(p1)
        ctx.register(p2)
        assert ctx.provider_count == 2
        assert ctx.active_provider_count == 1

    def test_clear(self):
        ctx = ContextManager()
        ctx.register(StaticProvider("x", "data"))
        ctx.clear()
        assert ctx.provider_count == 0
        assert ctx.get_merged_context() == ""

    def test_callable_provider_integration(self):
        counter = {"n": 0}
        def dynamic():
            counter["n"] += 1
            return f"call #{counter['n']}"
        ctx = ContextManager()
        ctx.register(CallableProvider("dynamic", dynamic))
        assert "call #1" in ctx.get_merged_context()
        assert "call #2" in ctx.get_merged_context()

    def test_error_in_provider_skipped(self):
        def boom():
            raise RuntimeError("fail")
        ctx = ContextManager()
        ctx.register(CallableProvider("bad", boom))
        ctx.register(StaticProvider("good", "OK"))
        merged = ctx.get_merged_context()
        assert "OK" in merged
        assert "bad" not in merged
