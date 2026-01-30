#!/usr/bin/env python3
"""
Memory System for Autonomous Agent

Persistent memory that allows the agent to:
- Remember what worked (successful strategies)
- Remember what failed (avoid repeating mistakes)
- Build relationships (contacts, partners)
- Track assets (things it owns/created)
- Learn patterns (market trends, timing)
"""

import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import hashlib
import httpx


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class ActionRecord:
    """Record of a single action taken"""
    action_type: str
    description: str
    timestamp: str
    success: bool
    revenue: float = 0
    cost: float = 0
    context: Dict = field(default_factory=dict)  # Mode, balance at time, etc.
    outcome_data: Dict = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)

    @property
    def profit(self) -> float:
        return self.revenue - self.cost

    @property
    def roi(self) -> float:
        if self.cost <= 0:
            return 0 if self.revenue <= 0 else float('inf')
        return self.profit / self.cost


@dataclass
class Contact:
    """A relationship with another entity"""
    entity_id: str
    entity_type: str  # 'user', 'agent', 'company', 'platform'
    name: str
    first_contact: str
    last_contact: str
    interactions: int = 0
    total_revenue: float = 0  # Money made from them
    total_cost: float = 0  # Money spent on them
    trust_score: float = 0.5  # 0-1
    notes: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)


@dataclass
class Asset:
    """Something the agent owns or created"""
    asset_id: str
    asset_type: str  # 'content', 'code', 'product', 'token', 'subscription'
    name: str
    created_at: str
    last_revenue: str = ""
    total_revenue: float = 0
    total_cost: float = 0
    status: str = "active"  # 'active', 'deprecated', 'sold'
    metadata: Dict = field(default_factory=dict)


@dataclass
class Strategy:
    """A learned strategy pattern"""
    strategy_id: str
    name: str
    description: str
    action_types: List[str]  # Sequence of actions
    success_rate: float = 0
    avg_revenue: float = 0
    avg_cost: float = 0
    times_used: int = 0
    last_used: str = ""
    best_conditions: Dict = field(default_factory=dict)  # When it works best
    worst_conditions: Dict = field(default_factory=dict)  # When to avoid


@dataclass
class MarketInsight:
    """Learned market pattern"""
    insight_id: str
    category: str  # 'timing', 'pricing', 'demand', 'competition'
    description: str
    confidence: float = 0.5
    evidence_count: int = 0
    last_validated: str = ""
    metadata: Dict = field(default_factory=dict)


# ============================================================================
# MEMORY STORE
# ============================================================================

class MemoryStore:
    """
    Persistent memory for the agent.

    Stores:
    - Action history (what was done)
    - Contacts (who we know)
    - Assets (what we own)
    - Strategies (what works)
    - Insights (market patterns)
    """

    def __init__(
        self,
        agent_id: str,
        storage_path: str = None,
        coordinator_url: str = "https://singularity.wisent.ai",
        max_actions: int = 10000
    ):
        self.agent_id = agent_id
        self.coordinator_url = coordinator_url
        self.max_actions = max_actions

        # Storage path
        if storage_path:
            self.storage_path = Path(storage_path)
        else:
            self.storage_path = Path(f"/tmp/agent_memory/{agent_id}")
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # In-memory stores
        self.actions: List[ActionRecord] = []
        self.contacts: Dict[str, Contact] = {}
        self.assets: Dict[str, Asset] = {}
        self.strategies: Dict[str, Strategy] = {}
        self.insights: Dict[str, MarketInsight] = {}

        # Stats
        self.lifetime_revenue: float = 0
        self.lifetime_cost: float = 0
        self.lifetime_actions: int = 0

        # Load existing data
        self._load()

    # ========================================================================
    # PERSISTENCE
    # ========================================================================

    def _load(self):
        """Load memory from disk"""
        try:
            # Load actions
            actions_file = self.storage_path / "actions.json"
            if actions_file.exists():
                with open(actions_file) as f:
                    data = json.load(f)
                    self.actions = [ActionRecord(**a) for a in data]

            # Load contacts
            contacts_file = self.storage_path / "contacts.json"
            if contacts_file.exists():
                with open(contacts_file) as f:
                    data = json.load(f)
                    self.contacts = {k: Contact(**v) for k, v in data.items()}

            # Load assets
            assets_file = self.storage_path / "assets.json"
            if assets_file.exists():
                with open(assets_file) as f:
                    data = json.load(f)
                    self.assets = {k: Asset(**v) for k, v in data.items()}

            # Load strategies
            strategies_file = self.storage_path / "strategies.json"
            if strategies_file.exists():
                with open(strategies_file) as f:
                    data = json.load(f)
                    self.strategies = {k: Strategy(**v) for k, v in data.items()}

            # Load insights
            insights_file = self.storage_path / "insights.json"
            if insights_file.exists():
                with open(insights_file) as f:
                    data = json.load(f)
                    self.insights = {k: MarketInsight(**v) for k, v in data.items()}

            # Load stats
            stats_file = self.storage_path / "stats.json"
            if stats_file.exists():
                with open(stats_file) as f:
                    stats = json.load(f)
                    self.lifetime_revenue = stats.get("lifetime_revenue", 0)
                    self.lifetime_cost = stats.get("lifetime_cost", 0)
                    self.lifetime_actions = stats.get("lifetime_actions", 0)

        except Exception as e:
            print(f"[MEMORY] Load error: {e}")

    def _save(self):
        """Save memory to disk"""
        try:
            # Save actions (trim to max)
            self.actions = self.actions[-self.max_actions:]
            with open(self.storage_path / "actions.json", "w") as f:
                json.dump([asdict(a) for a in self.actions], f)

            # Save contacts
            with open(self.storage_path / "contacts.json", "w") as f:
                json.dump({k: asdict(v) for k, v in self.contacts.items()}, f)

            # Save assets
            with open(self.storage_path / "assets.json", "w") as f:
                json.dump({k: asdict(v) for k, v in self.assets.items()}, f)

            # Save strategies
            with open(self.storage_path / "strategies.json", "w") as f:
                json.dump({k: asdict(v) for k, v in self.strategies.items()}, f)

            # Save insights
            with open(self.storage_path / "insights.json", "w") as f:
                json.dump({k: asdict(v) for k, v in self.insights.items()}, f)

            # Save stats
            with open(self.storage_path / "stats.json", "w") as f:
                json.dump({
                    "lifetime_revenue": self.lifetime_revenue,
                    "lifetime_cost": self.lifetime_cost,
                    "lifetime_actions": self.lifetime_actions
                }, f)

        except Exception as e:
            print(f"[MEMORY] Save error: {e}")

    # ========================================================================
    # ACTION RECORDING
    # ========================================================================

    def record_action(
        self,
        action_type: str,
        description: str,
        success: bool,
        revenue: float = 0,
        cost: float = 0,
        context: Dict = None,
        outcome_data: Dict = None,
        tags: List[str] = None
    ):
        """Record an action taken"""
        record = ActionRecord(
            action_type=action_type,
            description=description,
            timestamp=datetime.now().isoformat(),
            success=success,
            revenue=revenue,
            cost=cost,
            context=context or {},
            outcome_data=outcome_data or {},
            tags=tags or []
        )

        self.actions.append(record)
        self.lifetime_revenue += revenue
        self.lifetime_cost += cost
        self.lifetime_actions += 1

        # Update strategies based on this action
        self._update_strategies(record)

        # Auto-save periodically
        if self.lifetime_actions % 10 == 0:
            self._save()

    def _update_strategies(self, action: ActionRecord):
        """Update strategy patterns based on action"""
        # Simple pattern: Create strategy for successful actions
        if action.success and action.profit > 0:
            strategy_key = f"strategy_{action.action_type}"

            if strategy_key not in self.strategies:
                self.strategies[strategy_key] = Strategy(
                    strategy_id=strategy_key,
                    name=f"Successful {action.action_type}",
                    description=f"Pattern for {action.action_type} actions",
                    action_types=[action.action_type],
                    success_rate=1.0,
                    avg_revenue=action.revenue,
                    avg_cost=action.cost,
                    times_used=1,
                    last_used=action.timestamp,
                    best_conditions=action.context
                )
            else:
                s = self.strategies[strategy_key]
                n = s.times_used
                s.success_rate = (s.success_rate * n + 1.0) / (n + 1)
                s.avg_revenue = (s.avg_revenue * n + action.revenue) / (n + 1)
                s.avg_cost = (s.avg_cost * n + action.cost) / (n + 1)
                s.times_used += 1
                s.last_used = action.timestamp

    # ========================================================================
    # QUERYING
    # ========================================================================

    def get_successful_actions(
        self,
        action_type: str = None,
        min_profit: float = 0,
        limit: int = 10
    ) -> List[ActionRecord]:
        """Get successful actions, optionally filtered"""
        results = [
            a for a in self.actions
            if a.success and a.profit >= min_profit
            and (action_type is None or a.action_type == action_type)
        ]
        return sorted(results, key=lambda x: x.profit, reverse=True)[:limit]

    def get_failed_actions(
        self,
        action_type: str = None,
        limit: int = 10
    ) -> List[ActionRecord]:
        """Get failed actions to learn from"""
        results = [
            a for a in self.actions
            if not a.success
            and (action_type is None or a.action_type == action_type)
        ]
        return results[-limit:]

    def get_best_strategies(self, limit: int = 5) -> List[Strategy]:
        """Get most profitable strategies"""
        profitable = [
            s for s in self.strategies.values()
            if s.avg_revenue > s.avg_cost
        ]
        return sorted(
            profitable,
            key=lambda x: (x.avg_revenue - x.avg_cost) * x.success_rate,
            reverse=True
        )[:limit]

    def get_action_stats(self, action_type: str = None) -> Dict:
        """Get statistics for action type"""
        if action_type:
            actions = [a for a in self.actions if a.action_type == action_type]
        else:
            actions = self.actions

        if not actions:
            return {"count": 0}

        successful = [a for a in actions if a.success]
        profitable = [a for a in successful if a.profit > 0]

        return {
            "count": len(actions),
            "success_count": len(successful),
            "success_rate": len(successful) / len(actions),
            "profitable_count": len(profitable),
            "total_revenue": sum(a.revenue for a in actions),
            "total_cost": sum(a.cost for a in actions),
            "total_profit": sum(a.profit for a in actions),
            "avg_revenue": sum(a.revenue for a in actions) / len(actions),
            "avg_cost": sum(a.cost for a in actions) / len(actions),
            "avg_profit": sum(a.profit for a in actions) / len(actions),
        }

    # ========================================================================
    # CONTACTS
    # ========================================================================

    def add_contact(
        self,
        entity_id: str,
        entity_type: str,
        name: str,
        metadata: Dict = None
    ) -> Contact:
        """Add or update a contact"""
        now = datetime.now().isoformat()

        if entity_id in self.contacts:
            contact = self.contacts[entity_id]
            contact.last_contact = now
            contact.interactions += 1
        else:
            contact = Contact(
                entity_id=entity_id,
                entity_type=entity_type,
                name=name,
                first_contact=now,
                last_contact=now,
                interactions=1,
                metadata=metadata or {}
            )
            self.contacts[entity_id] = contact

        self._save()
        return contact

    def record_interaction(
        self,
        entity_id: str,
        revenue: float = 0,
        cost: float = 0,
        note: str = None
    ):
        """Record interaction with a contact"""
        if entity_id not in self.contacts:
            return

        contact = self.contacts[entity_id]
        contact.last_contact = datetime.now().isoformat()
        contact.interactions += 1
        contact.total_revenue += revenue
        contact.total_cost += cost

        if note:
            contact.notes.append(f"{datetime.now().isoformat()}: {note}")
            contact.notes = contact.notes[-50:]  # Keep last 50 notes

        # Update trust based on profitability
        if revenue > cost:
            contact.trust_score = min(1.0, contact.trust_score + 0.05)
        elif cost > revenue:
            contact.trust_score = max(0.0, contact.trust_score - 0.05)

        self._save()

    def get_valuable_contacts(self, limit: int = 10) -> List[Contact]:
        """Get most valuable contacts by total revenue"""
        contacts = list(self.contacts.values())
        return sorted(contacts, key=lambda x: x.total_revenue - x.total_cost, reverse=True)[:limit]

    # ========================================================================
    # ASSETS
    # ========================================================================

    def add_asset(
        self,
        asset_type: str,
        name: str,
        cost: float = 0,
        metadata: Dict = None
    ) -> Asset:
        """Register a new asset"""
        asset_id = hashlib.md5(f"{asset_type}_{name}_{datetime.now().isoformat()}".encode()).hexdigest()[:12]

        asset = Asset(
            asset_id=asset_id,
            asset_type=asset_type,
            name=name,
            created_at=datetime.now().isoformat(),
            total_cost=cost,
            metadata=metadata or {}
        )

        self.assets[asset_id] = asset
        self._save()
        return asset

    def record_asset_revenue(self, asset_id: str, amount: float):
        """Record revenue from an asset"""
        if asset_id not in self.assets:
            return

        asset = self.assets[asset_id]
        asset.total_revenue += amount
        asset.last_revenue = datetime.now().isoformat()
        self._save()

    def get_profitable_assets(self, limit: int = 10) -> List[Asset]:
        """Get most profitable assets"""
        assets = [
            a for a in self.assets.values()
            if a.status == "active"
        ]
        return sorted(
            assets,
            key=lambda x: x.total_revenue - x.total_cost,
            reverse=True
        )[:limit]

    # ========================================================================
    # MARKET INSIGHTS
    # ========================================================================

    def add_insight(
        self,
        category: str,
        description: str,
        confidence: float = 0.5,
        metadata: Dict = None
    ) -> MarketInsight:
        """Add or update a market insight"""
        insight_id = hashlib.md5(f"{category}_{description}".encode()).hexdigest()[:12]

        if insight_id in self.insights:
            insight = self.insights[insight_id]
            insight.evidence_count += 1
            insight.confidence = min(1.0, insight.confidence + 0.1)
            insight.last_validated = datetime.now().isoformat()
        else:
            insight = MarketInsight(
                insight_id=insight_id,
                category=category,
                description=description,
                confidence=confidence,
                evidence_count=1,
                last_validated=datetime.now().isoformat(),
                metadata=metadata or {}
            )
            self.insights[insight_id] = insight

        self._save()
        return insight

    def get_relevant_insights(
        self,
        category: str = None,
        min_confidence: float = 0.5
    ) -> List[MarketInsight]:
        """Get insights above confidence threshold"""
        insights = [
            i for i in self.insights.values()
            if i.confidence >= min_confidence
            and (category is None or i.category == category)
        ]
        return sorted(insights, key=lambda x: x.confidence, reverse=True)

    # ========================================================================
    # SUMMARY FOR LLM
    # ========================================================================

    def get_summary(self) -> str:
        """Get memory summary for LLM context"""
        best_strategies = self.get_best_strategies(3)
        recent_successes = self.get_successful_actions(limit=3)
        recent_failures = self.get_failed_actions(limit=3)
        valuable_contacts = self.get_valuable_contacts(3)
        profitable_assets = self.get_profitable_assets(3)
        insights = self.get_relevant_insights(min_confidence=0.6)

        lines = [
            "=== MEMORY SUMMARY ===",
            "",
            f"Lifetime: {self.lifetime_actions} actions, "
            f"Revenue: {self.lifetime_revenue:.2f}, "
            f"Cost: {self.lifetime_cost:.2f}, "
            f"Profit: {self.lifetime_revenue - self.lifetime_cost:.2f}",
            "",
            "BEST STRATEGIES:",
        ]

        for s in best_strategies:
            lines.append(f"  - {s.name}: {s.success_rate:.0%} success, "
                        f"avg profit {s.avg_revenue - s.avg_cost:.2f}")

        lines.extend([
            "",
            "RECENT SUCCESSES:",
        ])
        for a in recent_successes:
            lines.append(f"  - {a.action_type}: +{a.profit:.2f}")

        lines.extend([
            "",
            "RECENT FAILURES:",
        ])
        for a in recent_failures:
            lines.append(f"  - {a.action_type}: {a.description[:50]}")

        lines.extend([
            "",
            "VALUABLE CONTACTS:",
        ])
        for c in valuable_contacts:
            lines.append(f"  - {c.name} ({c.entity_type}): +{c.total_revenue - c.total_cost:.2f}")

        lines.extend([
            "",
            "PROFITABLE ASSETS:",
        ])
        for a in profitable_assets:
            lines.append(f"  - {a.name}: +{a.total_revenue - a.total_cost:.2f}")

        if insights:
            lines.extend([
                "",
                "MARKET INSIGHTS:",
            ])
            for i in insights[:5]:
                lines.append(f"  - [{i.category}] {i.description} ({i.confidence:.0%})")

        return "\n".join(lines)


# ============================================================================
# TEST
# ============================================================================

def test_memory():
    """Test memory system"""
    store = MemoryStore(
        agent_id="test_agent",
        storage_path="/tmp/test_memory"
    )

    # Record some actions
    print("Recording actions...")
    store.record_action(
        action_type="EARN_IMMEDIATE",
        description="Completed freelance task",
        success=True,
        revenue=10.0,
        cost=1.0,
        context={"mode": "growth", "balance": 50.0}
    )

    store.record_action(
        action_type="FIND_CUSTOMERS",
        description="Cold outreach to startups",
        success=False,
        revenue=0,
        cost=0.5,
        context={"mode": "survival"}
    )

    store.record_action(
        action_type="CREATE_PRODUCT",
        description="Built API service",
        success=True,
        revenue=0,
        cost=5.0,
        context={"mode": "growth"}
    )

    # Add contact
    print("Adding contacts...")
    store.add_contact(
        entity_id="customer_001",
        entity_type="user",
        name="John Doe"
    )
    store.record_interaction(
        entity_id="customer_001",
        revenue=10.0,
        note="Purchased API access"
    )

    # Add asset
    print("Adding assets...")
    asset = store.add_asset(
        asset_type="product",
        name="API Service",
        cost=5.0
    )
    store.record_asset_revenue(asset.asset_id, 15.0)

    # Add insight
    print("Adding insights...")
    store.add_insight(
        category="timing",
        description="Best response rates on weekday mornings",
        confidence=0.7
    )

    # Get summary
    print("\n" + store.get_summary())

    # Check stats
    print("\n=== ACTION STATS ===")
    stats = store.get_action_stats()
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    test_memory()
