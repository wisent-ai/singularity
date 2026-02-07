"""Tests for LLM retry logic with exponential backoff."""

import asyncio
import pytest
from singularity.llm_retry import (
    RetryConfig, RetryStats, LLMRetryWrapper,
    is_transient_error, calculate_delay, retry_llm_call,
)


class TestIsTransientError:
    def test_rate_limit_string(self):
        assert is_transient_error(Exception("rate limit exceeded"))

    def test_429_string(self):
        assert is_transient_error(Exception("429 Too Many Requests"))

    def test_server_error(self):
        assert is_transient_error(Exception("500 Internal Server Error"))

    def test_timeout(self):
        assert is_transient_error(Exception("Request timed out"))

    def test_overloaded(self):
        assert is_transient_error(Exception("API is overloaded"))

    def test_connection_error(self):
        assert is_transient_error(Exception("connection refused"))

    def test_not_transient(self):
        assert not is_transient_error(Exception("invalid api key"))

    def test_not_transient_auth(self):
        assert not is_transient_error(Exception("authentication failed"))

    def test_status_code_429(self):
        e = Exception("error")
        e.status_code = 429
        assert is_transient_error(e)

    def test_status_code_200(self):
        e = Exception("error")
        e.status_code = 200
        assert not is_transient_error(e)


class TestCalculateDelay:
    def test_first_attempt(self):
        config = RetryConfig(base_delay=1.0, jitter=False)
        assert calculate_delay(0, config) == 1.0

    def test_exponential_growth(self):
        config = RetryConfig(base_delay=1.0, exponential_base=2.0, jitter=False)
        assert calculate_delay(0, config) == 1.0
        assert calculate_delay(1, config) == 2.0
        assert calculate_delay(2, config) == 4.0

    def test_max_delay_cap(self):
        config = RetryConfig(base_delay=1.0, max_delay=5.0, jitter=False)
        assert calculate_delay(10, config) == 5.0

    def test_jitter_within_bounds(self):
        config = RetryConfig(base_delay=2.0, jitter=True)
        for _ in range(20):
            delay = calculate_delay(0, config)
            assert 0 <= delay <= 2.0


class TestRetryLlmCall:
    @pytest.mark.asyncio
    async def test_success_first_try(self):
        async def ok():
            return "result"
        result, stats = await retry_llm_call(ok)
        assert result == "result"
        assert stats.attempts == 1
        assert stats.succeeded

    @pytest.mark.asyncio
    async def test_retry_on_transient(self):
        call_count = 0
        async def fail_then_ok():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("rate limit exceeded")
            return "ok"
        config = RetryConfig(max_retries=3, base_delay=0.01)
        result, stats = await retry_llm_call(fail_then_ok, config)
        assert result == "ok"
        assert stats.attempts == 3
        assert stats.succeeded

    @pytest.mark.asyncio
    async def test_no_retry_on_non_transient(self):
        async def auth_fail():
            raise Exception("invalid api key")
        config = RetryConfig(max_retries=3, base_delay=0.01)
        with pytest.raises(Exception, match="invalid api key"):
            await retry_llm_call(auth_fail, config)

    @pytest.mark.asyncio
    async def test_exhaust_retries(self):
        async def always_fail():
            raise Exception("rate limit exceeded")
        config = RetryConfig(max_retries=2, base_delay=0.01)
        with pytest.raises(Exception, match="rate limit"):
            await retry_llm_call(always_fail, config)

    @pytest.mark.asyncio
    async def test_callback_called(self):
        callbacks = []
        def on_retry(attempt, error, delay):
            callbacks.append(attempt)
        call_count = 0
        async def fail_once():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("503 service unavailable")
            return "ok"
        config = RetryConfig(max_retries=2, base_delay=0.01, on_retry=on_retry)
        await retry_llm_call(fail_once, config)
        assert len(callbacks) == 1
        assert callbacks[0] == 1


class TestLLMRetryWrapper:
    def test_get_retry_summary_empty(self):
        wrapper = LLMRetryWrapper(None)
        summary = wrapper.get_retry_summary()
        assert summary["total_calls"] == 0

    def test_retry_config(self):
        config = RetryConfig(max_retries=5, base_delay=2.0)
        wrapper = LLMRetryWrapper(None, config=config)
        assert wrapper.config.max_retries == 5
        assert wrapper.config.base_delay == 2.0
