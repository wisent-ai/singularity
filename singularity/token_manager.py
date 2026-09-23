"""
Token Manager — Context window management for LLM interactions.

Provides token counting, context budget allocation, and intelligent
truncation to prevent context window overflow. Works without external
tokenizer dependencies using character-based estimation.

Pillar: Self-Improvement — agents need context-window awareness to
operate reliably without overflow errors.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


# Context window limits per model (in tokens)
MODEL_CONTEXT_LIMITS = {
    # Anthropic
    "claude-sonnet-4-20250514": 200_000,
    "claude-3-5-sonnet-20241022": 200_000,
    "claude-3-haiku-20240307": 200_000,
    "claude-3-5-haiku-20241022": 200_000,
    # OpenAI
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
    "gpt-4-turbo": 128_000,
    "gpt-4": 8_192,
    "gpt-3.5-turbo": 16_385,
    # Vertex / Gemini
    "gemini-2.0-flash-001": 1_048_576,
    "gemini-1.5-pro-002": 2_097_152,
    "claude-3-5-sonnet-v2@20241022": 200_000,
    # Local defaults
    "_default": 8_192,
}

# Reserve tokens for output generation
DEFAULT_OUTPUT_RESERVE = 2_048


def estimate_tokens(text: str) -> int:
    """Estimate token count from text using character-based heuristic.

    Uses ~4 characters per token which is a good average for English text
    with code mixed in. This avoids requiring tiktoken or other tokenizer deps.
    """
    if not text:
        return 0
    return max(1, len(text) // 4)


def get_context_limit(model: str) -> int:
    """Get the context window limit for a model."""
    if model in MODEL_CONTEXT_LIMITS:
        return MODEL_CONTEXT_LIMITS[model]
    # Try prefix matching
    for key, limit in MODEL_CONTEXT_LIMITS.items():
        if model.startswith(key.split("-")[0]):
            return limit
    return MODEL_CONTEXT_LIMITS["_default"]


@dataclass
class TokenBudget:
    """Allocation of token budget across prompt components."""
    total_limit: int
    output_reserve: int
    system_prompt: int = 0
    user_prompt_fixed: int = 0  # balance, tools, etc.
    recent_actions: int = 0
    project_context: int = 0
    available: int = 0

    @property
    def input_limit(self) -> int:
        return self.total_limit - self.output_reserve

    def compute_available(self) -> int:
        """Compute remaining available tokens after fixed allocations."""
        used = self.system_prompt + self.user_prompt_fixed
        self.available = self.input_limit - used
        return self.available


@dataclass
class SessionTokenStats:
    """Cumulative token usage tracking across a session."""
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    call_count: int = 0
    peak_input_tokens: int = 0  # largest single prompt

    def record(self, input_tokens: int, output_tokens: int, cost_usd: float = 0.0):
        """Record a single LLM call's token usage."""
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cost_usd += cost_usd
        self.call_count += 1
        if input_tokens > self.peak_input_tokens:
            self.peak_input_tokens = input_tokens

    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens

    @property
    def avg_input_tokens(self) -> int:
        if self.call_count == 0:
            return 0
        return self.total_input_tokens // self.call_count

    def summary(self) -> Dict[str, Any]:
        return {
            "calls": self.call_count,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_tokens,
            "peak_input_tokens": self.peak_input_tokens,
            "avg_input_tokens": self.avg_input_tokens,
            "total_cost_usd": round(self.total_cost_usd, 6),
        }


class TokenManager:
    """Manages token budgets and context window allocation.

    Usage:
        tm = TokenManager(model="claude-sonnet-4-20250514")

        # Check if content fits
        budget = tm.compute_budget(system_prompt, user_prompt_base)
        actions_text = tm.fit_recent_actions(actions, budget.available // 2)
        context_text = tm.truncate_to_budget(project_context, budget.available // 2)

        # Track usage
        tm.record_usage(input_tokens=1500, output_tokens=300, cost=0.01)
        print(tm.session_stats.summary())
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        output_reserve: int = DEFAULT_OUTPUT_RESERVE,
        safety_margin: float = 0.95,  # use at most 95% of context window
    ):
        self.model = model
        self.context_limit = get_context_limit(model)
        self.output_reserve = output_reserve
        self.safety_margin = safety_margin
        self.session_stats = SessionTokenStats()

    @property
    def effective_limit(self) -> int:
        """Context limit with safety margin applied."""
        return int(self.context_limit * self.safety_margin)

    def update_model(self, model: str):
        """Update the model (e.g., after model switching)."""
        self.model = model
        self.context_limit = get_context_limit(model)

    def compute_budget(
        self,
        system_prompt: str,
        user_prompt_base: str,
    ) -> TokenBudget:
        """Compute token budget allocation for a think() call.

        Args:
            system_prompt: The full system prompt text
            user_prompt_base: The fixed part of the user prompt (balance, tools)

        Returns:
            TokenBudget with allocations and available space
        """
        budget = TokenBudget(
            total_limit=self.effective_limit,
            output_reserve=self.output_reserve,
            system_prompt=estimate_tokens(system_prompt),
            user_prompt_fixed=estimate_tokens(user_prompt_base),
        )
        budget.compute_available()
        return budget

    def fit_recent_actions(
        self,
        actions: List[Dict],
        token_budget: int,
        min_actions: int = 2,
    ) -> List[Dict]:
        """Select recent actions that fit within the token budget.

        Keeps the most recent actions, dropping older ones if needed.
        Always keeps at least min_actions if any exist.

        Args:
            actions: List of action dicts (most recent last)
            token_budget: Maximum tokens for actions text
            min_actions: Minimum actions to keep even if over budget

        Returns:
            Subset of actions that fit within budget
        """
        if not actions:
            return []

        if token_budget <= 0:
            return actions[-min_actions:] if len(actions) >= min_actions else actions

        # Start from most recent, add until budget exhausted
        result = []
        tokens_used = 0

        for action in reversed(actions):
            action_text = f"- {action.get('tool', 'unknown')}: {action.get('result', {}).get('status', 'unknown')}"
            action_tokens = estimate_tokens(action_text)

            if tokens_used + action_tokens > token_budget and len(result) >= min_actions:
                break

            result.append(action)
            tokens_used += action_tokens

        result.reverse()
        return result

    def truncate_to_budget(
        self,
        text: str,
        token_budget: int,
        truncation_suffix: str = "\n... [truncated to fit context window]",
    ) -> str:
        """Truncate text to fit within a token budget.

        Args:
            text: Text to potentially truncate
            token_budget: Maximum tokens allowed
            truncation_suffix: Appended when text is truncated

        Returns:
            Original text or truncated version
        """
        if not text:
            return text

        current_tokens = estimate_tokens(text)
        if current_tokens <= token_budget:
            return text

        if token_budget <= 0:
            return ""

        # Calculate how many characters we can keep
        suffix_tokens = estimate_tokens(truncation_suffix)
        available_tokens = max(0, token_budget - suffix_tokens)
        char_limit = available_tokens * 4  # inverse of estimate_tokens

        if char_limit <= 0:
            return ""

        return text[:char_limit] + truncation_suffix

    def will_fit(self, system_prompt: str, user_prompt: str) -> bool:
        """Check if a prompt will fit in the context window."""
        total = estimate_tokens(system_prompt) + estimate_tokens(user_prompt) + self.output_reserve
        return total <= self.effective_limit

    def utilization(self, system_prompt: str, user_prompt: str) -> float:
        """Calculate what percentage of the context window a prompt uses.

        Returns a value between 0.0 and 1.0+. Values > 1.0 mean overflow.
        """
        total = estimate_tokens(system_prompt) + estimate_tokens(user_prompt) + self.output_reserve
        return total / self.effective_limit if self.effective_limit > 0 else 1.0

    def record_usage(self, input_tokens: int, output_tokens: int, cost_usd: float = 0.0):
        """Record token usage from an LLM call."""
        self.session_stats.record(input_tokens, output_tokens, cost_usd)

    def get_stats(self) -> Dict[str, Any]:
        """Get session token usage stats."""
        return {
            "model": self.model,
            "context_limit": self.context_limit,
            "effective_limit": self.effective_limit,
            "output_reserve": self.output_reserve,
            **self.session_stats.summary(),
        }
