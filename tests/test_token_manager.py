"""Tests for TokenManager — context window management."""

from singularity.token_manager import (
    TokenManager,
    estimate_tokens,
    get_context_limit,
    SessionTokenStats,
)


def test_estimate_tokens_empty():
    assert estimate_tokens("") == 0
    assert estimate_tokens(None) == 0


def test_estimate_tokens_basic():
    # ~4 chars per token
    assert estimate_tokens("hello world") == 2  # 11 chars / 4 = 2
    text_400 = "a" * 400
    assert estimate_tokens(text_400) == 100


def test_get_context_limit_known_models():
    assert get_context_limit("claude-sonnet-4-20250514") == 200_000
    assert get_context_limit("gpt-4o") == 128_000
    assert get_context_limit("gemini-2.0-flash-001") == 1_048_576


def test_get_context_limit_unknown():
    assert get_context_limit("unknown-model-v1") == 8_192


def test_token_manager_init():
    tm = TokenManager(model="gpt-4o")
    assert tm.context_limit == 128_000
    assert tm.effective_limit == int(128_000 * 0.95)


def test_compute_budget():
    tm = TokenManager(model="gpt-4o", output_reserve=1000)
    budget = tm.compute_budget("system prompt here", "user prompt base")
    assert budget.system_prompt > 0
    assert budget.user_prompt_fixed > 0
    assert budget.available > 0
    assert budget.input_limit == tm.effective_limit - 1000


def test_fit_recent_actions_empty():
    tm = TokenManager()
    assert tm.fit_recent_actions([], 1000) == []


def test_fit_recent_actions_fits():
    tm = TokenManager()
    actions = [
        {"tool": "fs:view", "result": {"status": "success"}},
        {"tool": "shell:bash", "result": {"status": "success"}},
    ]
    result = tm.fit_recent_actions(actions, 10000)
    assert len(result) == 2


def test_fit_recent_actions_truncates():
    tm = TokenManager()
    actions = [{"tool": f"skill:action{i}", "result": {"status": "ok"}} for i in range(100)]
    result = tm.fit_recent_actions(actions, 50)  # very small budget
    assert len(result) < 100
    assert len(result) >= 2  # min_actions default


def test_truncate_to_budget_fits():
    tm = TokenManager()
    text = "short text"
    assert tm.truncate_to_budget(text, 1000) == text


def test_truncate_to_budget_truncates():
    tm = TokenManager()
    text = "x" * 10000  # ~2500 tokens
    result = tm.truncate_to_budget(text, 100)
    assert len(result) < len(text)
    assert result.endswith("[truncated to fit context window]")


def test_truncate_to_budget_zero():
    tm = TokenManager()
    assert tm.truncate_to_budget("some text", 0) == ""


def test_will_fit():
    tm = TokenManager(model="gpt-4o")
    assert tm.will_fit("short system", "short user") is True
    huge = "x" * 1_000_000
    assert tm.will_fit(huge, huge) is False


def test_utilization():
    tm = TokenManager(model="gpt-4o")
    u = tm.utilization("short", "short")
    assert 0 < u < 0.1  # small prompt, big window


def test_record_usage():
    tm = TokenManager()
    tm.record_usage(1000, 200, 0.01)
    tm.record_usage(1500, 300, 0.02)
    stats = tm.get_stats()
    assert stats["calls"] == 2
    assert stats["total_input_tokens"] == 2500
    assert stats["total_output_tokens"] == 500
    assert stats["peak_input_tokens"] == 1500
    assert stats["total_cost_usd"] == 0.03


def test_session_stats_avg():
    s = SessionTokenStats()
    s.record(100, 50)
    s.record(200, 100)
    assert s.avg_input_tokens == 150
    assert s.total_tokens == 450


def test_update_model():
    tm = TokenManager(model="gpt-4o")
    assert tm.context_limit == 128_000
    tm.update_model("claude-sonnet-4-20250514")
    assert tm.context_limit == 200_000


def test_cognition_engine_has_token_manager():
    from singularity.cognition import CognitionEngine
    engine = CognitionEngine(llm_provider="none")
    assert hasattr(engine, "token_manager")
    assert engine.token_manager.model == "claude-sonnet-4-20250514"
