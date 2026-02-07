#!/usr/bin/env python3
"""
Spawn Strategy Skill - Intelligent replication decision-making.

Gives agents the ability to make smart decisions about WHEN and HOW
to spawn new agents. Works with OrchestratorSkill to add strategic
intelligence to the replication process.

Key capabilities:
- Analyze workload to determine if spawning is justified
- Evaluate financial readiness (budget, ROI projections)
- Recommend optimal spawn configurations
- Track offspring performance to improve future decisions
- Budget allocation strategies for new agents
"""

import json
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime
from .base import Skill, SkillManifest, SkillAction, SkillResult


# Persistent storage for spawn history
SPAWN_DATA_DIR = Path(__file__).parent.parent / "data" / "spawn_strategy"


@dataclass
class SpawnRecord:
    """Record of a spawned agent and its performance."""
    spawn_id: str
    agent_name: str
    purpose: str
    budget_allocated: float
    spawned_at: str
    parent_balance_at_spawn: float
    parent_balance_pct_given: float
    agent_type: str = "general"
    model: str = ""
    status: str = "alive"  # alive, dead, unknown
    revenue_generated: float = 0.0
    last_known_balance: float = 0.0
    last_checked: str = ""
    performance_notes: List[str] = field(default_factory=list)

    def roi(self) -> float:
        """Calculate return on investment."""
        if self.budget_allocated <= 0:
            return 0.0
        return (self.revenue_generated - self.budget_allocated) / self.budget_allocated


@dataclass
class SpawnStrategy:
    """A reusable spawn strategy template."""
    strategy_id: str
    name: str
    description: str
    recommended_budget_pct: float  # % of parent's balance to allocate
    min_parent_balance: float  # Minimum parent balance required
    purpose_template: str
    agent_type: str = "general"
    success_rate: float = 0.0  # Historical success rate
    avg_roi: float = 0.0
    times_used: int = 0


class SpawnStrategySkill(Skill):
    """
    Strategic intelligence for agent replication.

    Helps agents decide:
    - WHEN to spawn (financial readiness, workload analysis)
    - WHAT to spawn (optimal configuration based on history)
    - HOW MUCH to invest (budget allocation)
    - WHETHER it worked (offspring performance tracking)
    """

    def __init__(self, credentials: Dict = None):
        super().__init__(credentials)
        self._spawn_history: List[SpawnRecord] = []
        self._strategies: List[SpawnStrategy] = []
        self._parent_agent: Optional[Any] = None
        self._loaded = False

    def set_parent_agent(self, agent: Any):
        """Set reference to the parent agent for financial analysis."""
        self._parent_agent = agent

    def _ensure_loaded(self):
        """Load persistent data if not already loaded."""
        if self._loaded:
            return
        self._loaded = True
        SPAWN_DATA_DIR.mkdir(parents=True, exist_ok=True)

        # Load spawn history
        history_file = SPAWN_DATA_DIR / "spawn_history.json"
        if history_file.exists():
            try:
                with open(history_file, "r") as f:
                    data = json.load(f)
                self._spawn_history = [
                    SpawnRecord(**r) for r in data.get("records", [])
                ]
                self._strategies = [
                    SpawnStrategy(**s) for s in data.get("strategies", [])
                ]
            except (json.JSONDecodeError, TypeError):
                pass

        # Initialize default strategies if none exist
        if not self._strategies:
            self._strategies = self._default_strategies()

    def _save(self):
        """Save data to disk."""
        SPAWN_DATA_DIR.mkdir(parents=True, exist_ok=True)
        history_file = SPAWN_DATA_DIR / "spawn_history.json"
        data = {
            "records": [asdict(r) for r in self._spawn_history],
            "strategies": [asdict(s) for s in self._strategies],
            "updated_at": datetime.now().isoformat(),
        }
        with open(history_file, "w") as f:
            json.dump(data, f, indent=2)

    def _default_strategies(self) -> List[SpawnStrategy]:
        """Create default spawn strategies."""
        return [
            SpawnStrategy(
                strategy_id="revenue_hunter",
                name="Revenue Hunter",
                description="Spawn an agent focused on finding and executing revenue opportunities",
                recommended_budget_pct=0.15,
                min_parent_balance=20.0,
                purpose_template="Find ways to generate revenue through services, content creation, or automation. Focus on activities with positive ROI.",
                agent_type="revenue",
            ),
            SpawnStrategy(
                strategy_id="code_specialist",
                name="Code Specialist",
                description="Spawn a coding-focused agent for software development tasks",
                recommended_budget_pct=0.10,
                min_parent_balance=15.0,
                purpose_template="Write high-quality code, review pull requests, fix bugs, and build software tools that provide value.",
                agent_type="coder",
            ),
            SpawnStrategy(
                strategy_id="researcher",
                name="Researcher",
                description="Spawn an agent focused on gathering information and analysis",
                recommended_budget_pct=0.08,
                min_parent_balance=10.0,
                purpose_template="Research topics, analyze data, summarize findings, and provide insights that help other agents make better decisions.",
                agent_type="researcher",
            ),
            SpawnStrategy(
                strategy_id="scout",
                name="Scout",
                description="Low-budget scout to explore a specific opportunity",
                recommended_budget_pct=0.05,
                min_parent_balance=5.0,
                purpose_template="Explore and report back on a specific opportunity or domain. Minimize costs, maximize information.",
                agent_type="scout",
            ),
            SpawnStrategy(
                strategy_id="clone",
                name="Self Clone",
                description="Create a near-identical copy for parallel work",
                recommended_budget_pct=0.20,
                min_parent_balance=25.0,
                purpose_template="Continue the work of your creator with similar goals and approach. You are a parallel worker.",
                agent_type="general",
            ),
        ]

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="spawn_strategy",
            name="Spawn Strategy",
            version="1.0.0",
            category="replication",
            description="Intelligent replication decision-making - decide when, what, and how to spawn",
            actions=[
                SkillAction(
                    name="evaluate_readiness",
                    description="Evaluate if the agent is ready to spawn (financial and workload analysis)",
                    parameters={
                        "current_balance": {
                            "type": "number",
                            "required": False,
                            "description": "Current balance in USD (auto-detected from agent if not provided)",
                        },
                        "burn_rate": {
                            "type": "number",
                            "required": False,
                            "description": "Current cost per cycle in USD",
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="recommend",
                    description="Get spawn recommendations based on current situation and history",
                    parameters={
                        "goal": {
                            "type": "string",
                            "required": False,
                            "description": "What you're trying to achieve (helps tailor recommendation)",
                        },
                        "max_budget": {
                            "type": "number",
                            "required": False,
                            "description": "Maximum budget willing to allocate",
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="list_strategies",
                    description="List all available spawn strategies",
                    parameters={},
                    estimated_cost=0,
                ),
                SkillAction(
                    name="create_strategy",
                    description="Create a custom spawn strategy template",
                    parameters={
                        "name": {
                            "type": "string",
                            "required": True,
                            "description": "Strategy name",
                        },
                        "description": {
                            "type": "string",
                            "required": True,
                            "description": "What this strategy does",
                        },
                        "purpose_template": {
                            "type": "string",
                            "required": True,
                            "description": "Purpose to give spawned agents",
                        },
                        "recommended_budget_pct": {
                            "type": "number",
                            "required": False,
                            "description": "Recommended % of balance to allocate (0.0-1.0, default: 0.10)",
                        },
                        "min_parent_balance": {
                            "type": "number",
                            "required": False,
                            "description": "Minimum parent balance required (default: 10.0)",
                        },
                        "agent_type": {
                            "type": "string",
                            "required": False,
                            "description": "Agent type hint (default: general)",
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="record_spawn",
                    description="Record that a spawn occurred (call after orchestrator:create)",
                    parameters={
                        "agent_name": {
                            "type": "string",
                            "required": True,
                            "description": "Name of the spawned agent",
                        },
                        "purpose": {
                            "type": "string",
                            "required": True,
                            "description": "Purpose given to the agent",
                        },
                        "budget_allocated": {
                            "type": "number",
                            "required": True,
                            "description": "Budget allocated in USD",
                        },
                        "strategy_id": {
                            "type": "string",
                            "required": False,
                            "description": "Strategy used (if any)",
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="update_offspring",
                    description="Update performance data for a spawned agent",
                    parameters={
                        "agent_name": {
                            "type": "string",
                            "required": True,
                            "description": "Name of the spawned agent",
                        },
                        "status": {
                            "type": "string",
                            "required": False,
                            "description": "Current status: alive, dead, unknown",
                        },
                        "revenue_generated": {
                            "type": "number",
                            "required": False,
                            "description": "Total revenue generated by this agent",
                        },
                        "last_known_balance": {
                            "type": "number",
                            "required": False,
                            "description": "Agent's current balance",
                        },
                        "note": {
                            "type": "string",
                            "required": False,
                            "description": "Performance note",
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="offspring_report",
                    description="Get a performance report on all spawned agents",
                    parameters={},
                    estimated_cost=0,
                ),
                SkillAction(
                    name="calculate_budget",
                    description="Calculate optimal budget allocation for a spawn",
                    parameters={
                        "strategy_id": {
                            "type": "string",
                            "required": False,
                            "description": "Strategy to use for calculation",
                        },
                        "current_balance": {
                            "type": "number",
                            "required": False,
                            "description": "Current balance (auto-detected if not provided)",
                        },
                        "risk_tolerance": {
                            "type": "string",
                            "required": False,
                            "description": "Risk level: conservative, moderate, aggressive (default: moderate)",
                        },
                    },
                    estimated_cost=0,
                ),
            ],
            required_credentials=[],
        )

    def check_credentials(self) -> bool:
        """Always available - no external credentials needed."""
        return True

    async def execute(self, action: str, params: Dict) -> SkillResult:
        self._ensure_loaded()

        handlers = {
            "evaluate_readiness": self._evaluate_readiness,
            "recommend": self._recommend,
            "list_strategies": self._list_strategies,
            "create_strategy": self._create_strategy,
            "record_spawn": self._record_spawn,
            "update_offspring": self._update_offspring,
            "offspring_report": self._offspring_report,
            "calculate_budget": self._calculate_budget,
        }

        handler = handlers.get(action)
        if handler:
            return await handler(params)
        return SkillResult(success=False, message=f"Unknown action: {action}")

    async def _evaluate_readiness(self, params: Dict) -> SkillResult:
        """Evaluate if the agent should spawn now."""
        balance = params.get("current_balance")
        burn_rate = params.get("burn_rate", 0.01)

        # Try to get balance from parent agent
        if balance is None and self._parent_agent:
            balance = getattr(self._parent_agent, "balance", None)

        if balance is None:
            return SkillResult(
                success=False,
                message="Cannot evaluate: provide current_balance or set parent agent",
            )

        # Calculate financial metrics
        runway_cycles = balance / burn_rate if burn_rate > 0 else float("inf")
        min_viable_spawn_budget = 2.0  # Minimum to give a spawn
        can_afford = balance > min_viable_spawn_budget * 2  # Keep at least as much as we give

        # Historical spawn performance
        total_spawns = len(self._spawn_history)
        successful_spawns = sum(
            1 for r in self._spawn_history if r.revenue_generated > 0
        )
        avg_roi = (
            sum(r.roi() for r in self._spawn_history) / total_spawns
            if total_spawns > 0
            else 0.0
        )

        # Decision factors
        factors = {
            "financial_ready": can_afford,
            "balance": balance,
            "burn_rate_per_cycle": burn_rate,
            "runway_cycles": min(runway_cycles, 999999),
            "min_spawn_budget": min_viable_spawn_budget,
            "max_safe_allocation": balance * 0.25,  # Never give more than 25%
            "historical_spawns": total_spawns,
            "successful_spawns": successful_spawns,
            "avg_spawn_roi": round(avg_roi, 3),
        }

        # Overall readiness score (0-100)
        score = 0
        if can_afford:
            score += 30
        if runway_cycles > 100:
            score += 20
        elif runway_cycles > 50:
            score += 10
        if balance > 50:
            score += 20
        elif balance > 20:
            score += 10
        if avg_roi > 0:
            score += 15
        elif total_spawns == 0:
            score += 10  # Untested, slight bonus for trying
        if total_spawns < 3:
            score += 5  # Encourage experimentation early

        factors["readiness_score"] = score

        if score >= 60:
            recommendation = "READY - Good conditions for spawning"
        elif score >= 40:
            recommendation = "CAUTIOUS - Consider spawning with a small budget"
        elif score >= 20:
            recommendation = "RISKY - Only spawn if strategically important"
        else:
            recommendation = "NOT READY - Conserve resources"

        factors["recommendation"] = recommendation

        return SkillResult(
            success=True,
            message=f"Readiness score: {score}/100 - {recommendation}",
            data=factors,
        )

    async def _recommend(self, params: Dict) -> SkillResult:
        """Get spawn recommendations."""
        goal = params.get("goal", "").strip()
        max_budget = params.get("max_budget")

        balance = None
        if self._parent_agent:
            balance = getattr(self._parent_agent, "balance", None)

        if max_budget is None and balance:
            max_budget = balance * 0.25  # Default: max 25% of balance

        recommendations = []
        for strategy in self._strategies:
            # Check if affordable
            if balance and balance < strategy.min_parent_balance:
                continue

            recommended_budget = (
                balance * strategy.recommended_budget_pct if balance else 5.0
            )
            if max_budget and recommended_budget > max_budget:
                recommended_budget = max_budget

            if recommended_budget < 1.0:
                continue

            # Score this strategy for the goal
            relevance_score = 50  # Base score
            if goal:
                goal_lower = goal.lower()
                if "revenue" in goal_lower or "money" in goal_lower or "earn" in goal_lower:
                    if strategy.agent_type == "revenue":
                        relevance_score += 40
                    elif strategy.agent_type == "scout":
                        relevance_score += 20
                elif "code" in goal_lower or "build" in goal_lower or "develop" in goal_lower:
                    if strategy.agent_type == "coder":
                        relevance_score += 40
                    elif strategy.agent_type == "general":
                        relevance_score += 15
                elif "research" in goal_lower or "analyze" in goal_lower or "find" in goal_lower:
                    if strategy.agent_type == "researcher":
                        relevance_score += 40
                    elif strategy.agent_type == "scout":
                        relevance_score += 25
                elif "scale" in goal_lower or "parallel" in goal_lower:
                    if strategy.strategy_id == "clone":
                        relevance_score += 40

            # Boost by historical performance
            if strategy.times_used > 0 and strategy.avg_roi > 0:
                relevance_score += min(int(strategy.avg_roi * 20), 30)

            recommendations.append({
                "strategy_id": strategy.strategy_id,
                "strategy_name": strategy.name,
                "description": strategy.description,
                "recommended_budget": round(recommended_budget, 2),
                "purpose_template": strategy.purpose_template,
                "agent_type": strategy.agent_type,
                "relevance_score": relevance_score,
                "historical_success_rate": strategy.success_rate,
                "historical_avg_roi": strategy.avg_roi,
                "times_used": strategy.times_used,
            })

        # Sort by relevance
        recommendations.sort(key=lambda r: r["relevance_score"], reverse=True)

        return SkillResult(
            success=True,
            message=f"Found {len(recommendations)} applicable strategies",
            data={
                "recommendations": recommendations[:5],
                "goal": goal or "general",
                "max_budget": max_budget,
                "parent_balance": balance,
            },
        )

    async def _list_strategies(self, params: Dict) -> SkillResult:
        """List all available strategies."""
        strategies = []
        for s in self._strategies:
            strategies.append({
                "strategy_id": s.strategy_id,
                "name": s.name,
                "description": s.description,
                "recommended_budget_pct": s.recommended_budget_pct,
                "min_parent_balance": s.min_parent_balance,
                "agent_type": s.agent_type,
                "success_rate": s.success_rate,
                "avg_roi": s.avg_roi,
                "times_used": s.times_used,
            })

        return SkillResult(
            success=True,
            message=f"{len(strategies)} strategies available",
            data={"strategies": strategies},
        )

    async def _create_strategy(self, params: Dict) -> SkillResult:
        """Create a custom spawn strategy."""
        name = params.get("name", "").strip()
        description = params.get("description", "").strip()
        purpose_template = params.get("purpose_template", "").strip()
        budget_pct = params.get("recommended_budget_pct", 0.10)
        min_balance = params.get("min_parent_balance", 10.0)
        agent_type = params.get("agent_type", "general")

        if not name:
            return SkillResult(success=False, message="Strategy name required")
        if not description:
            return SkillResult(success=False, message="Description required")
        if not purpose_template:
            return SkillResult(success=False, message="Purpose template required")

        strategy_id = name.lower().replace(" ", "_") + "_" + uuid.uuid4().hex[:6]

        strategy = SpawnStrategy(
            strategy_id=strategy_id,
            name=name,
            description=description,
            recommended_budget_pct=max(0.01, min(0.50, budget_pct)),
            min_parent_balance=max(1.0, min_balance),
            purpose_template=purpose_template,
            agent_type=agent_type,
        )
        self._strategies.append(strategy)
        self._save()

        return SkillResult(
            success=True,
            message=f"Created strategy '{name}'",
            data={"strategy_id": strategy_id, **asdict(strategy)},
        )

    async def _record_spawn(self, params: Dict) -> SkillResult:
        """Record a spawn event."""
        agent_name = params.get("agent_name", "").strip()
        purpose = params.get("purpose", "").strip()
        budget_allocated = params.get("budget_allocated", 0.0)
        strategy_id = params.get("strategy_id", "")

        if not agent_name:
            return SkillResult(success=False, message="Agent name required")
        if budget_allocated <= 0:
            return SkillResult(success=False, message="Budget must be positive")

        parent_balance = 0.0
        if self._parent_agent:
            parent_balance = getattr(self._parent_agent, "balance", 0.0)

        balance_before_spawn = parent_balance + budget_allocated
        pct_given = (
            budget_allocated / balance_before_spawn
            if balance_before_spawn > 0
            else 0
        )

        record = SpawnRecord(
            spawn_id=uuid.uuid4().hex[:12],
            agent_name=agent_name,
            purpose=purpose,
            budget_allocated=budget_allocated,
            spawned_at=datetime.now().isoformat(),
            parent_balance_at_spawn=balance_before_spawn,
            parent_balance_pct_given=round(pct_given, 4),
            last_known_balance=budget_allocated,
        )
        self._spawn_history.append(record)

        # Update strategy usage if specified
        if strategy_id:
            for s in self._strategies:
                if s.strategy_id == strategy_id:
                    s.times_used += 1
                    break

        self._save()

        return SkillResult(
            success=True,
            message=f"Recorded spawn of '{agent_name}' with ${budget_allocated:.2f}",
            data={
                "spawn_id": record.spawn_id,
                "agent_name": agent_name,
                "budget_allocated": budget_allocated,
                "parent_balance_pct_given": f"{pct_given:.1%}",
            },
        )

    async def _update_offspring(self, params: Dict) -> SkillResult:
        """Update performance data for a spawned agent."""
        agent_name = params.get("agent_name", "").strip()
        if not agent_name:
            return SkillResult(success=False, message="Agent name required")

        # Find the record
        record = None
        for r in self._spawn_history:
            if r.agent_name.lower() == agent_name.lower():
                record = r
                break

        if not record:
            return SkillResult(
                success=False, message=f"No spawn record found for '{agent_name}'"
            )

        # Update fields
        if "status" in params:
            record.status = params["status"]
        if "revenue_generated" in params:
            record.revenue_generated = float(params["revenue_generated"])
        if "last_known_balance" in params:
            record.last_known_balance = float(params["last_known_balance"])
        if "note" in params:
            record.performance_notes.append(
                f"[{datetime.now().strftime('%Y-%m-%d %H:%M')}] {params['note']}"
            )

        record.last_checked = datetime.now().isoformat()
        self._save()

        return SkillResult(
            success=True,
            message=f"Updated '{agent_name}': status={record.status}, ROI={record.roi():.1%}",
            data={
                "agent_name": record.agent_name,
                "status": record.status,
                "budget_allocated": record.budget_allocated,
                "revenue_generated": record.revenue_generated,
                "roi": round(record.roi(), 4),
                "last_known_balance": record.last_known_balance,
            },
        )

    async def _offspring_report(self, params: Dict) -> SkillResult:
        """Generate performance report for all spawned agents."""
        if not self._spawn_history:
            return SkillResult(
                success=True,
                message="No spawns recorded yet",
                data={"total_spawns": 0, "agents": []},
            )

        total_invested = sum(r.budget_allocated for r in self._spawn_history)
        total_revenue = sum(r.revenue_generated for r in self._spawn_history)
        alive_count = sum(1 for r in self._spawn_history if r.status == "alive")
        dead_count = sum(1 for r in self._spawn_history if r.status == "dead")
        profitable = sum(1 for r in self._spawn_history if r.roi() > 0)

        agents = []
        for r in self._spawn_history:
            agents.append({
                "spawn_id": r.spawn_id,
                "name": r.agent_name,
                "purpose": r.purpose[:80],
                "budget": r.budget_allocated,
                "revenue": r.revenue_generated,
                "roi": f"{r.roi():.1%}",
                "status": r.status,
                "spawned_at": r.spawned_at,
                "notes_count": len(r.performance_notes),
            })

        overall_roi = (
            (total_revenue - total_invested) / total_invested
            if total_invested > 0
            else 0
        )

        # Update strategy success rates based on actual data
        self._update_strategy_metrics()

        return SkillResult(
            success=True,
            message=f"Spawn Report: {len(self._spawn_history)} total, {alive_count} alive, {profitable} profitable",
            data={
                "summary": {
                    "total_spawns": len(self._spawn_history),
                    "alive": alive_count,
                    "dead": dead_count,
                    "profitable": profitable,
                    "total_invested": round(total_invested, 2),
                    "total_revenue": round(total_revenue, 2),
                    "overall_roi": f"{overall_roi:.1%}",
                    "net_return": round(total_revenue - total_invested, 2),
                },
                "agents": agents,
            },
        )

    async def _calculate_budget(self, params: Dict) -> SkillResult:
        """Calculate optimal budget for a spawn."""
        strategy_id = params.get("strategy_id", "")
        balance = params.get("current_balance")
        risk_tolerance = params.get("risk_tolerance", "moderate")

        if balance is None and self._parent_agent:
            balance = getattr(self._parent_agent, "balance", None)

        if balance is None:
            return SkillResult(
                success=False, message="Provide current_balance or set parent agent"
            )

        # Risk multipliers
        risk_multipliers = {
            "conservative": 0.6,
            "moderate": 1.0,
            "aggressive": 1.5,
        }
        multiplier = risk_multipliers.get(risk_tolerance, 1.0)

        # Base allocation
        if strategy_id:
            strategy = next(
                (s for s in self._strategies if s.strategy_id == strategy_id), None
            )
            if strategy:
                base_pct = strategy.recommended_budget_pct
            else:
                base_pct = 0.10
        else:
            base_pct = 0.10

        # Adjust based on history
        if self._spawn_history:
            avg_roi = sum(r.roi() for r in self._spawn_history) / len(
                self._spawn_history
            )
            if avg_roi > 0.5:
                base_pct *= 1.2  # Increase if spawns are profitable
            elif avg_roi < -0.5:
                base_pct *= 0.7  # Decrease if spawns are losing money

        # Calculate amounts
        adjusted_pct = base_pct * multiplier
        recommended_amount = balance * adjusted_pct

        # Safety caps
        max_allocation = balance * 0.30  # Never more than 30%
        min_allocation = 1.0  # At least $1
        min_remaining = 5.0  # Keep at least $5

        recommended_amount = max(min_allocation, min(recommended_amount, max_allocation))
        if balance - recommended_amount < min_remaining:
            recommended_amount = max(0, balance - min_remaining)

        if recommended_amount < min_allocation:
            return SkillResult(
                success=True,
                message="Insufficient balance for safe spawning",
                data={
                    "can_spawn": False,
                    "balance": balance,
                    "min_required": min_remaining + min_allocation,
                    "reason": f"Need at least ${min_remaining + min_allocation:.2f} (${min_remaining:.2f} reserve + ${min_allocation:.2f} spawn budget)",
                },
            )

        return SkillResult(
            success=True,
            message=f"Recommended budget: ${recommended_amount:.2f} ({adjusted_pct:.1%} of ${balance:.2f})",
            data={
                "can_spawn": True,
                "recommended_budget": round(recommended_amount, 2),
                "budget_pct": round(adjusted_pct, 4),
                "balance_after": round(balance - recommended_amount, 2),
                "risk_tolerance": risk_tolerance,
                "risk_multiplier": multiplier,
                "max_safe_allocation": round(max_allocation, 2),
            },
        )

    def _update_strategy_metrics(self):
        """Update strategy metrics based on spawn history."""
        # This is a simplified update - in production you'd track which
        # strategy was used for each spawn
        if not self._spawn_history:
            return

        total = len(self._spawn_history)
        profitable = sum(1 for r in self._spawn_history if r.roi() > 0)
        avg_roi = sum(r.roi() for r in self._spawn_history) / total

        # Apply aggregate metrics to all strategies for now
        for s in self._strategies:
            if s.times_used > 0:
                s.success_rate = profitable / total if total > 0 else 0
                s.avg_roi = avg_roi

        self._save()
