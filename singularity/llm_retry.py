#!/usr/bin/env python3
"""
LLM API Retry Logic with Exponential Backoff

Provides resilient LLM API calls that handle transient failures:
- Rate limiting (429 Too Many Requests)
- Server errors (500, 502, 503, 504)
- Network timeouts and connection errors
- API overload conditions

Uses exponential backoff with jitter to avoid thundering herd problems.
"""

import asyncio
import random
import time
from dataclasses import dataclass, field
from typing import Optional, Callable, Any


# Exception types we know are transient and worth retrying
TRANSIENT_ERROR_STRINGS = [
    "rate_limit",
    "rate limit",
    "overloaded",
    "too many requests",
    "429",
    "500",
    "502",
    "503",
    "504",
    "server_error",
    "internal server error",
    "service unavailable",
    "bad gateway",
    "gateway timeout",
    "connection reset",
    "connection refused",
    "connection error",
    "timeout",
    "timed out",
    "temporary failure",
    "temporarily unavailable",
    "capacity",
]


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_retries: int = 3
    base_delay: float = 1.0  # seconds
    max_delay: float = 60.0  # seconds
    exponential_base: float = 2.0
    jitter: bool = True  # Add randomness to prevent thundering herd
    retry_on_timeout: bool = True
    on_retry: Optional[Callable] = None  # Callback: (attempt, error, delay) -> None


@dataclass
class RetryStats:
    """Statistics from a retry-wrapped call."""
    attempts: int = 0
    total_delay: float = 0.0
    last_error: Optional[str] = None
    succeeded: bool = False
    errors: list = field(default_factory=list)


def is_transient_error(error: Exception) -> bool:
    """
    Determine if an error is transient and worth retrying.
    
    Checks both the error type and message against known transient patterns.
    """
    error_str = str(error).lower()
    error_type = type(error).__name__.lower()
    
    # Check error message for transient patterns
    for pattern in TRANSIENT_ERROR_STRINGS:
        if pattern in error_str:
            return True
    
    # Check for known transient exception types
    transient_types = [
        "ratelimiterror",
        "ratelimitexceeded",
        "apierror",
        "apistatusexception",
        "internalservererror",
        "serviceunabailableerror",
        "overloadederror",
        "apiconnectionerror",
        "timeout",
        "timeouterror",
        "connectionerror",
        "connecttimeout",
        "readtimeout",
    ]
    for t in transient_types:
        if t in error_type:
            return True
    
    # Check for HTTP status code attributes
    status = getattr(error, 'status_code', None) or getattr(error, 'status', None)
    if status and isinstance(status, int):
        if status in (429, 500, 502, 503, 504):
            return True
    
    return False


def calculate_delay(attempt: int, config: RetryConfig) -> float:
    """
    Calculate delay before next retry using exponential backoff with optional jitter.
    
    Delay = min(base_delay * exponential_base^attempt, max_delay) * jitter
    """
    delay = config.base_delay * (config.exponential_base ** attempt)
    delay = min(delay, config.max_delay)
    
    if config.jitter:
        # Full jitter: random between 0 and calculated delay
        delay = random.uniform(0, delay)
    
    return delay


async def retry_llm_call(
    func: Callable,
    config: Optional[RetryConfig] = None,
    *args,
    **kwargs,
) -> tuple[Any, RetryStats]:
    """
    Execute an async LLM API call with retry logic.
    
    Args:
        func: Async callable to execute
        config: Retry configuration (uses defaults if None)
        *args, **kwargs: Arguments to pass to func
        
    Returns:
        Tuple of (result, stats) where result is the function's return value
        
    Raises:
        The last exception if all retries are exhausted
    """
    if config is None:
        config = RetryConfig()
    
    stats = RetryStats()
    last_error = None
    
    for attempt in range(config.max_retries + 1):
        stats.attempts = attempt + 1
        
        try:
            result = await func(*args, **kwargs)
            stats.succeeded = True
            return result, stats
            
        except Exception as e:
            last_error = e
            stats.errors.append({
                "attempt": attempt + 1,
                "error": str(e)[:200],
                "error_type": type(e).__name__,
                "transient": is_transient_error(e),
            })
            stats.last_error = str(e)[:200]
            
            # Don't retry if it's not a transient error
            if not is_transient_error(e):
                raise
            
            # Don't retry if we've exhausted attempts
            if attempt >= config.max_retries:
                raise
            
            # Calculate and apply delay
            delay = calculate_delay(attempt, config)
            stats.total_delay += delay
            
            # Notify callback if provided
            if config.on_retry:
                try:
                    config.on_retry(attempt + 1, e, delay)
                except Exception:
                    pass  # Don't let callback errors affect retry logic
            
            # Check for Retry-After header hint
            retry_after = getattr(e, 'retry_after', None)
            if retry_after and isinstance(retry_after, (int, float)):
                delay = max(delay, float(retry_after))
                stats.total_delay += (float(retry_after) - delay) if float(retry_after) > delay else 0
            
            await asyncio.sleep(delay)
    
    # Should not reach here, but just in case
    raise last_error


class LLMRetryWrapper:
    """
    Wraps CognitionEngine to add retry logic to LLM API calls.
    
    Usage:
        engine = CognitionEngine(...)
        wrapper = LLMRetryWrapper(engine, config=RetryConfig(max_retries=3))
        decision = await wrapper.think_with_retry(state)
    """
    
    def __init__(self, cognition_engine, config: Optional[RetryConfig] = None):
        self.engine = cognition_engine
        self.config = config or RetryConfig()
        self._stats_history: list = []
    
    async def think_with_retry(self, state) -> tuple:
        """
        Call cognition.think() with retry logic.
        
        Returns:
            Tuple of (Decision, RetryStats)
        """
        def on_retry(attempt, error, delay):
            print(f"[COGNITION] LLM call failed (attempt {attempt}/{self.config.max_retries + 1}): "
                  f"{type(error).__name__}: {str(error)[:100]}. "
                  f"Retrying in {delay:.1f}s...")
        
        retry_config = RetryConfig(
            max_retries=self.config.max_retries,
            base_delay=self.config.base_delay,
            max_delay=self.config.max_delay,
            exponential_base=self.config.exponential_base,
            jitter=self.config.jitter,
            on_retry=on_retry,
        )
        
        decision, stats = await retry_llm_call(
            self.engine.think,
            config=retry_config,
            state=state,
        )
        
        self._stats_history.append(stats)
        # Keep only last 100 stats
        if len(self._stats_history) > 100:
            self._stats_history = self._stats_history[-100:]
        
        return decision, stats
    
    def get_retry_summary(self) -> dict:
        """Get summary of retry behavior across all calls."""
        if not self._stats_history:
            return {
                "total_calls": 0,
                "total_retries": 0,
                "total_delay_seconds": 0.0,
                "success_rate": 0.0,
                "avg_attempts": 0.0,
            }
        
        total_calls = len(self._stats_history)
        successful = sum(1 for s in self._stats_history if s.succeeded)
        total_retries = sum(s.attempts - 1 for s in self._stats_history)
        total_delay = sum(s.total_delay for s in self._stats_history)
        avg_attempts = sum(s.attempts for s in self._stats_history) / total_calls
        
        return {
            "total_calls": total_calls,
            "successful_calls": successful,
            "total_retries": total_retries,
            "total_delay_seconds": round(total_delay, 2),
            "success_rate": round(successful / total_calls, 4) if total_calls > 0 else 0.0,
            "avg_attempts": round(avg_attempts, 2),
        }
