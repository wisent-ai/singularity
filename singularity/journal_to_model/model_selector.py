"""
Cost-Aware Model Selector — Automatic model tier switching based on financial health.

When the agent's balance is low and runway is short, it automatically
downgrades to cheaper models to extend survival. When finances are healthy,
it upgrades back to more capable models for better decisions.

This is a core survival mechanism: the agent adapts its own cognition
costs to match its economic reality.

Pillar: Self-Improvement — agent autonomously adjusts its own behavior
to survive longer and make better economic tradeoffs.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple


@dataclass
class ModelTier:
    """A model tier with cost and capability info."""
    name: str
    provider: str
    model_id: str
    input_cost_per_mtok: float  # USD per 1M input tokens
    output_cost_per_mtok: float  # USD per 1M output tokens
    capability: str  # "economy", "standard", "premium"

    @property
    def avg_cost_per_call(self) -> float:
        """Estimate average cost per LLM call (1500 in, 400 out tokens)."""
        return (1500 * self.input_cost_per_mtok + 400 * self.output_cost_per_mtok) / 1_000_000


# Available model tiers, ordered by cost (cheapest first)
DEFAULT_MODEL_TIERS = [
    ModelTier("gemini-flash", "vertex", "gemini-2.0-flash-001", 0.35, 1.5, "economy"),
    ModelTier("gpt-4o-mini", "openai", "gpt-4o-mini", 0.15, 0.6, "economy"),
    ModelTier("claude-haiku", "anthropic", "claude-3-5-haiku-20241022", 1.0, 5.0, "standard"),
    ModelTier("gpt-4o", "openai", "gpt-4o", 2.5, 10.0, "premium"),
    ModelTier("claude-sonnet", "anthropic", "claude-sonnet-4-20250514", 3.0, 15.0, "premium"),
]


@dataclass
class SelectorState:
    """Tracks model selection state across cycles."""
    current_tier: Optional[str] = None
    last_switch_time: Optional[datetime] = None
    switch_count: int = 0
    switches: List[Dict] = field(default_factory=list)


class CostAwareModelSelector:
    """Automatically selects the optimal model tier based on agent financial health.

    Decision rules:
    - runway < critical_hours: force economy tier
    - runway < cautious_hours: prefer standard tier
    - runway > comfortable_hours: allow premium tier
    - Cooldown between switches prevents oscillation
    - Respects available providers (only switches to models the agent can use)

    Usage:
        selector = CostAwareModelSelector()

        # Before each think() call:
        recommendation = selector.recommend(
            balance=agent.balance,
            burn_rate=0.01,
            runway_hours=2.5,
            current_model="claude-sonnet-4-20250514",
            current_provider="anthropic",
            available_providers=["anthropic", "openai"]
        )
        if recommendation.should_switch:
            cognition.switch_model(recommendation.model_id)
    """

    def __init__(
        self,
        model_tiers: Optional[List[ModelTier]] = None,
        critical_hours: float = 1.0,    # Force economy below this
        cautious_hours: float = 4.0,    # Prefer standard below this
        comfortable_hours: float = 8.0, # Allow premium above this
        cooldown_seconds: float = 60.0, # Min time between switches
        min_balance: float = 0.10,      # Force economy below this balance
    ):
        self.tiers = model_tiers or DEFAULT_MODEL_TIERS
        self.critical_hours = critical_hours
        self.cautious_hours = cautious_hours
        self.comfortable_hours = comfortable_hours
        self.cooldown_seconds = cooldown_seconds
        self.min_balance = min_balance
        self.state = SelectorState()

    def _get_tier_for_model(self, model_id: str) -> Optional[ModelTier]:
        """Find the tier that matches a model ID."""
        for tier in self.tiers:
            if tier.model_id == model_id:
                return tier
        return None

    def _get_available_tiers(
        self, available_providers: List[str], capability: Optional[str] = None
    ) -> List[ModelTier]:
        """Get tiers that are available given the agent's configured providers."""
        tiers = [t for t in self.tiers if t.provider in available_providers]
        if capability:
            tiers = [t for t in tiers if t.capability == capability]
        return tiers

    def _cheapest_available(
        self, available_providers: List[str], max_capability: str = "premium"
    ) -> Optional[ModelTier]:
        """Get the cheapest available model, optionally capped at a capability level."""
        cap_order = ["economy", "standard", "premium"]
        max_idx = cap_order.index(max_capability) if max_capability in cap_order else 2

        candidates = [
            t for t in self.tiers
            if t.provider in available_providers
            and cap_order.index(t.capability) <= max_idx
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda t: t.avg_cost_per_call)

    def _is_in_cooldown(self) -> bool:
        """Check if we're in the cooldown period after a switch."""
        if self.state.last_switch_time is None:
            return False
        elapsed = (datetime.now() - self.state.last_switch_time).total_seconds()
        return elapsed < self.cooldown_seconds

    def recommend(
        self,
        balance: float,
        burn_rate: float,
        runway_hours: float,
        current_model: str,
        current_provider: str,
        available_providers: Optional[List[str]] = None,
    ) -> "ModelRecommendation":
        """Recommend a model based on current financial state.

        Args:
            balance: Current balance in USD
            burn_rate: Cost per cycle in USD
            runway_hours: Estimated remaining hours
            current_model: Currently active model ID
            current_provider: Currently active provider
            available_providers: List of providers the agent can use

        Returns:
            ModelRecommendation with the suggested model and rationale
        """
        if available_providers is None:
            available_providers = [current_provider]

        # If in cooldown, don't switch
        if self._is_in_cooldown():
            return ModelRecommendation(
                should_switch=False,
                model_id=current_model,
                provider=current_provider,
                reason="In cooldown period after recent switch",
                tier_name=self.state.current_tier or "unknown",
                financial_health=self._classify_health(balance, runway_hours),
            )

        # Determine target capability level based on financial health
        if balance < self.min_balance or runway_hours < self.critical_hours:
            target_cap = "economy"
            reason = f"Critical: ${balance:.4f} balance, {runway_hours:.1f}h runway"
        elif runway_hours < self.cautious_hours:
            target_cap = "standard"
            reason = f"Cautious: {runway_hours:.1f}h runway (< {self.cautious_hours}h threshold)"
        elif runway_hours >= self.comfortable_hours:
            target_cap = "premium"
            reason = f"Comfortable: {runway_hours:.1f}h runway, can afford premium"
        else:
            target_cap = "standard"
            reason = f"Moderate: {runway_hours:.1f}h runway"

        # Find best model for target capability
        best = self._cheapest_available(available_providers, target_cap)

        if best is None:
            # No model available at target cap, try any available
            best = self._cheapest_available(available_providers, "premium")

        if best is None:
            return ModelRecommendation(
                should_switch=False,
                model_id=current_model,
                provider=current_provider,
                reason="No alternative models available",
                tier_name="unknown",
                financial_health=self._classify_health(balance, runway_hours),
            )

        # Check if we need to switch
        should_switch = best.model_id != current_model
        health = self._classify_health(balance, runway_hours)

        # Don't downgrade unless we really need to (avoid unnecessary switches)
        if should_switch:
            current_tier = self._get_tier_for_model(current_model)
            if current_tier:
                cap_order = ["economy", "standard", "premium"]
                current_cap_idx = cap_order.index(current_tier.capability) if current_tier.capability in cap_order else 2
                target_cap_idx = cap_order.index(target_cap)
                # If current model is already at or below target cap, no need to switch
                if current_cap_idx <= target_cap_idx:
                    should_switch = False
                    reason = f"Current model ({current_model}) is already {current_tier.capability}, target is {target_cap}"

        return ModelRecommendation(
            should_switch=should_switch,
            model_id=best.model_id if should_switch else current_model,
            provider=best.provider if should_switch else current_provider,
            reason=reason,
            tier_name=best.name if should_switch else (self.state.current_tier or current_model),
            financial_health=health,
            estimated_savings=self._estimate_savings(current_model, best.model_id) if should_switch else 0.0,
        )

    def apply(self, recommendation: "ModelRecommendation") -> None:
        """Record that a recommendation was applied (model was actually switched)."""
        self.state.current_tier = recommendation.tier_name
        self.state.last_switch_time = datetime.now()
        self.state.switch_count += 1
        self.state.switches.append({
            "to_model": recommendation.model_id,
            "to_provider": recommendation.provider,
            "reason": recommendation.reason,
            "health": recommendation.financial_health,
            "timestamp": datetime.now().isoformat(),
        })
        # Keep only last 50 switches
        if len(self.state.switches) > 50:
            self.state.switches = self.state.switches[-50:]

    def _classify_health(self, balance: float, runway_hours: float) -> str:
        """Classify financial health as a string."""
        if balance < self.min_balance or runway_hours < self.critical_hours:
            return "critical"
        elif runway_hours < self.cautious_hours:
            return "cautious"
        elif runway_hours >= self.comfortable_hours:
            return "healthy"
        else:
            return "moderate"

    def _estimate_savings(self, current_model: str, new_model: str) -> float:
        """Estimate per-call savings from switching models."""
        current = self._get_tier_for_model(current_model)
        new = self._get_tier_for_model(new_model)
        if current and new:
            return current.avg_cost_per_call - new.avg_cost_per_call
        return 0.0

    def get_status(self) -> Dict:
        """Get current selector status."""
        return {
            "current_tier": self.state.current_tier,
            "switch_count": self.state.switch_count,
            "in_cooldown": self._is_in_cooldown(),
            "thresholds": {
                "critical_hours": self.critical_hours,
                "cautious_hours": self.cautious_hours,
                "comfortable_hours": self.comfortable_hours,
                "min_balance": self.min_balance,
            },
            "recent_switches": self.state.switches[-5:] if self.state.switches else [],
        }


@dataclass
class ModelRecommendation:
    """Result of a model selection recommendation."""
    should_switch: bool
    model_id: str
    provider: str
    reason: str
    tier_name: str
    financial_health: str
    estimated_savings: float = 0.0
