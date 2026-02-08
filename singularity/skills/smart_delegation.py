#!/usr/bin/env python3
"""
SmartDelegationSkill - Consensus-driven, reputation-weighted task assignment.

Wires together TaskDelegationSkill, ConsensusProtocolSkill, and AgentReputationSkill
into an intelligent delegation pipeline:

1. SMART_DELEGATE   - Delegate with automatic reputation-ranked agent selection
2. REPUTATION_ROUTE - Find and rank agents by capability + reputation score
3. CONSENSUS_ASSIGN - For high-stakes tasks, run a consensus vote among agents
                      with reputation-weighted ballots to pick the best assignee
4. AUTO_REPORT      - After task completion, update agent reputation automatically
5. RECOMMEND        - Given a task, recommend top agents with reputation breakdown

Without this skill, delegation is either manual (pick an agent by ID) or
simple auto-routing (first match from AgentNetwork). This adds intelligence:
agents are selected based on proven track record and peer consensus.

Pillars served:
- Replication: Smart multi-agent coordination with trust-based routing
- Self-Improvement: Agents that perform well get more work (positive feedback loop)
- Revenue: Better agents handle customer tasks → higher satisfaction → more revenue
- Goal Setting: High-priority goals get assigned to highest-reputation agents
"""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

from .base import Skill, SkillResult, SkillManifest, SkillAction

SMART_DELEGATION_FILE = Path(__file__).parent.parent / "data" / "smart_delegations.json"

# Selection strategies
STRATEGY_REPUTATION = "reputation"      # Pick highest reputation
STRATEGY_CONSENSUS = "consensus"        # Run a vote among agents
STRATEGY_BALANCED = "balanced"          # Reputation + load balancing
STRATEGY_ROUND_ROBIN = "round_robin"    # Equal distribution

# Reputation dimension weights for task routing
DEFAULT_TASK_WEIGHTS = {
    "competence": 0.40,
    "reliability": 0.30,
    "trustworthiness": 0.15,
    "cooperation": 0.10,
    "leadership": 0.05,
}


class SmartDelegationSkill(Skill):
    """
    Intelligent task delegation using reputation scores and consensus voting.

    Bridges TaskDelegationSkill, AgentReputationSkill, and ConsensusProtocolSkill
    to enable trust-based, democratic agent selection for task assignment.
    """

    def __init__(self, credentials: Dict[str, str] = None):
        super().__init__(credentials)
        self._ensure_data()

    def _ensure_data(self):
        SMART_DELEGATION_FILE.parent.mkdir(parents=True, exist_ok=True)
        if not SMART_DELEGATION_FILE.exists():
            self._save(self._default_state())

    def _default_state(self) -> Dict:
        return {
            "smart_delegations": [],
            "stats": {
                "total_smart_delegations": 0,
                "reputation_routed": 0,
                "consensus_assigned": 0,
                "auto_reports_filed": 0,
                "avg_winner_reputation": 0.0,
            },
        }

    def _load(self) -> Dict:
        try:
            with open(SMART_DELEGATION_FILE, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return self._default_state()

    def _save(self, state: Dict):
        SMART_DELEGATION_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(SMART_DELEGATION_FILE, "w") as f:
            json.dump(state, f, indent=2, default=str)

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="smart_delegation",
            name="Smart Delegation",
            version="1.0.0",
            category="coordination",
            description=(
                "Consensus-driven, reputation-weighted task assignment. "
                "Wires together TaskDelegation, ConsensusProtocol, and AgentReputation "
                "for intelligent agent selection."
            ),
            required_credentials=[],
            actions=[
                SkillAction(
                    name="smart_delegate",
                    description=(
                        "Delegate a task with automatic reputation-ranked agent selection. "
                        "Finds candidate agents, ranks by reputation, optionally runs "
                        "consensus vote, then delegates to the winner."
                    ),
                    parameters={
                        "task_name": {"type": "string", "required": True, "description": "Name of the task"},
                        "task_description": {"type": "string", "required": True, "description": "What needs to be done"},
                        "budget": {"type": "number", "required": True, "description": "Budget for the task"},
                        "required_capability": {"type": "string", "required": False, "description": "Capability needed"},
                        "candidate_agents": {"type": "list", "required": False, "description": "Explicit list of candidate agent IDs"},
                        "strategy": {"type": "string", "required": False, "description": "Selection strategy: reputation, consensus, balanced, round_robin (default: reputation)"},
                        "min_reputation": {"type": "number", "required": False, "description": "Minimum overall reputation score (0-100, default: 0)"},
                        "priority": {"type": "string", "required": False, "description": "Task priority: low, normal, high, critical"},
                        "reputation_weights": {"type": "dict", "required": False, "description": "Custom weights for reputation dimensions"},
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="reputation_route",
                    description=(
                        "Find and rank agents by capability + reputation score. "
                        "Returns ranked list without actually delegating."
                    ),
                    parameters={
                        "required_capability": {"type": "string", "required": False, "description": "Capability to search for"},
                        "candidate_agents": {"type": "list", "required": False, "description": "Explicit candidates to rank"},
                        "min_reputation": {"type": "number", "required": False, "description": "Minimum reputation threshold"},
                        "reputation_weights": {"type": "dict", "required": False, "description": "Custom dimension weights"},
                        "limit": {"type": "integer", "required": False, "description": "Max results (default: 5)"},
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="consensus_assign",
                    description=(
                        "For high-stakes tasks, run a consensus vote among agents "
                        "with reputation-weighted ballots to democratically pick the assignee."
                    ),
                    parameters={
                        "task_name": {"type": "string", "required": True, "description": "Task to assign"},
                        "task_description": {"type": "string", "required": True, "description": "Task details"},
                        "budget": {"type": "number", "required": True, "description": "Task budget"},
                        "candidate_agents": {"type": "list", "required": True, "description": "Agent IDs eligible for assignment"},
                        "voter_agents": {"type": "list", "required": False, "description": "Agent IDs who vote (defaults to candidates)"},
                        "quorum_rule": {"type": "string", "required": False, "description": "Voting rule: simple_majority, supermajority (default: simple_majority)"},
                        "auto_delegate": {"type": "boolean", "required": False, "description": "Auto-delegate to winner (default: true)"},
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="auto_report",
                    description=(
                        "After task completion, automatically update the assignee's "
                        "reputation in AgentReputationSkill based on outcome."
                    ),
                    parameters={
                        "delegation_id": {"type": "string", "required": True, "description": "Delegation ID to report on"},
                        "success": {"type": "boolean", "required": True, "description": "Whether task succeeded"},
                        "budget_spent": {"type": "number", "required": False, "description": "Actual budget spent"},
                        "on_time": {"type": "boolean", "required": False, "description": "Whether completed on time"},
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="recommend",
                    description=(
                        "Given a task description, recommend top agents with "
                        "reputation breakdown and confidence scores."
                    ),
                    parameters={
                        "task_description": {"type": "string", "required": True, "description": "Description of the task"},
                        "candidate_agents": {"type": "list", "required": True, "description": "Agents to evaluate"},
                        "top_n": {"type": "integer", "required": False, "description": "How many to recommend (default: 3)"},
                    },
                    estimated_cost=0,
                ),
            ],
        )

    async def execute(self, action: str, params: Dict) -> SkillResult:
        handlers = {
            "smart_delegate": self._smart_delegate,
            "reputation_route": self._reputation_route,
            "consensus_assign": self._consensus_assign,
            "auto_report": self._auto_report,
            "recommend": self._recommend,
        }

        handler = handlers.get(action)
        if not handler:
            return SkillResult(
                success=False,
                message=f"Unknown action: {action}. Available: {list(handlers.keys())}",
            )

        try:
            return await handler(params)
        except Exception as e:
            return SkillResult(success=False, message=f"Error in {action}: {str(e)}")

    # ─── SMART DELEGATE ────────────────────────────────────────────

    async def _smart_delegate(self, params: Dict) -> SkillResult:
        """Delegate with automatic reputation-ranked agent selection."""
        task_name = params.get("task_name", "").strip()
        task_description = params.get("task_description", "").strip()
        budget = float(params.get("budget", 0))

        if not task_name:
            return SkillResult(success=False, message="task_name is required")
        if not task_description:
            return SkillResult(success=False, message="task_description is required")
        if budget <= 0:
            return SkillResult(success=False, message="budget must be positive")

        strategy = params.get("strategy", STRATEGY_REPUTATION)
        min_reputation = float(params.get("min_reputation", 0))
        priority = params.get("priority", "normal")
        reputation_weights = params.get("reputation_weights", DEFAULT_TASK_WEIGHTS)

        # Step 1: Get candidate agents
        candidates = params.get("candidate_agents", [])
        if not candidates and self.context:
            # Auto-discover via AgentNetwork
            capability = params.get("required_capability", "")
            if capability:
                route_result = await self.context.call_skill(
                    "agent_network", "route",
                    {"capability": capability}
                )
                if route_result.success and route_result.data.get("matches"):
                    candidates = [
                        m.get("agent_id", "")
                        for m in route_result.data["matches"]
                        if m.get("agent_id")
                    ]

        if not candidates:
            return SkillResult(
                success=False,
                message="No candidate agents found. Provide candidate_agents or required_capability.",
            )

        # Step 2: Get reputation scores for all candidates
        ranked = await self._rank_by_reputation(candidates, reputation_weights, min_reputation)

        if not ranked:
            return SkillResult(
                success=False,
                message=f"No candidates meet minimum reputation threshold ({min_reputation})",
                data={"candidates_evaluated": len(candidates)},
            )

        # Step 3: Select agent based on strategy
        if strategy == STRATEGY_CONSENSUS and len(ranked) > 1:
            # Run consensus vote with reputation weights
            selected, selection_detail = await self._run_consensus_selection(
                ranked, task_name, task_description
            )
        elif strategy == STRATEGY_BALANCED:
            # Pick from top 3 by reputation, considering load
            selected, selection_detail = self._balanced_select(ranked)
        elif strategy == STRATEGY_ROUND_ROBIN:
            selected, selection_detail = self._round_robin_select(ranked)
        else:
            # Default: pick highest reputation
            selected = ranked[0]
            selection_detail = f"Highest reputation score ({selected['score']:.1f})"

        # Step 4: Delegate via TaskDelegationSkill
        delegate_result = None
        if self.context:
            delegate_result = await self.context.call_skill(
                "task_delegation", "delegate",
                {
                    "task_name": task_name,
                    "task_description": task_description,
                    "budget": budget,
                    "agent_id": selected["agent_id"],
                    "priority": priority,
                }
            )

        # Step 5: Record the smart delegation
        state = self._load()
        record = {
            "id": f"sdlg_{uuid.uuid4().hex[:10]}",
            "task_name": task_name,
            "strategy": strategy,
            "candidates_evaluated": len(candidates),
            "candidates_qualified": len(ranked),
            "selected_agent": selected["agent_id"],
            "selected_score": selected["score"],
            "selection_detail": selection_detail,
            "delegation_id": delegate_result.data.get("delegation_id") if delegate_result and delegate_result.success else None,
            "created_at": datetime.now().isoformat(),
        }
        state["smart_delegations"].append(record)
        state["stats"]["total_smart_delegations"] += 1
        if strategy == STRATEGY_CONSENSUS:
            state["stats"]["consensus_assigned"] += 1
        else:
            state["stats"]["reputation_routed"] += 1

        # Update running average of winner reputation
        n = state["stats"]["total_smart_delegations"]
        old_avg = state["stats"]["avg_winner_reputation"]
        state["stats"]["avg_winner_reputation"] = (
            (old_avg * (n - 1) + selected["score"]) / n
        )

        # Keep bounded
        if len(state["smart_delegations"]) > 500:
            state["smart_delegations"] = state["smart_delegations"][-500:]

        self._save(state)

        delegation_id = record["delegation_id"]
        msg = (
            f"Smart delegation: '{task_name}' → agent '{selected['agent_id']}' "
            f"(reputation: {selected['score']:.1f}, strategy: {strategy})"
        )
        if delegation_id:
            msg += f". Delegation ID: {delegation_id}"

        return SkillResult(
            success=True,
            message=msg,
            data={
                "smart_delegation_id": record["id"],
                "delegation_id": delegation_id,
                "selected_agent": selected["agent_id"],
                "selected_score": selected["score"],
                "selection_detail": selection_detail,
                "strategy": strategy,
                "candidates_evaluated": len(candidates),
                "candidates_qualified": len(ranked),
                "ranking": [
                    {"agent_id": r["agent_id"], "score": round(r["score"], 1)}
                    for r in ranked[:5]
                ],
            },
        )

    # ─── REPUTATION ROUTE ──────────────────────────────────────────

    async def _reputation_route(self, params: Dict) -> SkillResult:
        """Find and rank agents by capability + reputation."""
        candidates = params.get("candidate_agents", [])
        min_reputation = float(params.get("min_reputation", 0))
        reputation_weights = params.get("reputation_weights", DEFAULT_TASK_WEIGHTS)
        limit = int(params.get("limit", 5))

        # Auto-discover if no explicit candidates
        if not candidates and self.context:
            capability = params.get("required_capability", "")
            if capability:
                route_result = await self.context.call_skill(
                    "agent_network", "route",
                    {"capability": capability}
                )
                if route_result.success and route_result.data.get("matches"):
                    candidates = [
                        m.get("agent_id", "")
                        for m in route_result.data["matches"]
                        if m.get("agent_id")
                    ]

        if not candidates:
            return SkillResult(
                success=False,
                message="No candidates found. Provide candidate_agents or required_capability.",
            )

        ranked = await self._rank_by_reputation(candidates, reputation_weights, min_reputation)
        ranked = ranked[:limit]

        return SkillResult(
            success=True,
            message=f"Ranked {len(ranked)} agents by reputation (from {len(candidates)} candidates)",
            data={
                "ranked_agents": [
                    {
                        "rank": i + 1,
                        "agent_id": r["agent_id"],
                        "score": round(r["score"], 1),
                        "dimensions": r.get("dimensions", {}),
                    }
                    for i, r in enumerate(ranked)
                ],
                "total_candidates": len(candidates),
                "qualified": len(ranked),
                "min_reputation": min_reputation,
            },
        )

    # ─── CONSENSUS ASSIGN ──────────────────────────────────────────

    async def _consensus_assign(self, params: Dict) -> SkillResult:
        """Run a consensus vote among agents to pick the best assignee."""
        task_name = params.get("task_name", "").strip()
        task_description = params.get("task_description", "").strip()
        budget = float(params.get("budget", 0))
        candidates = params.get("candidate_agents", [])

        if not task_name or not task_description or budget <= 0:
            return SkillResult(success=False, message="task_name, task_description, and positive budget are required")
        if len(candidates) < 2:
            return SkillResult(success=False, message="Need at least 2 candidate agents for consensus")

        voters = params.get("voter_agents", candidates)
        quorum_rule = params.get("quorum_rule", "simple_majority")
        auto_delegate = params.get("auto_delegate", True)

        if not self.context:
            return SkillResult(success=False, message="No SkillContext - cannot call ConsensusProtocol")

        # Step 1: Get reputation scores to use as vote weights
        agent_scores = {}
        for agent_id in set(candidates + voters):
            rep_result = await self.context.call_skill(
                "agent_reputation", "get_reputation",
                {"agent_id": agent_id}
            )
            if rep_result.success:
                agent_scores[agent_id] = rep_result.data.get("overall", 50.0)
            else:
                agent_scores[agent_id] = 50.0  # Neutral default

        # Step 2: Run election via ConsensusProtocolSkill
        # Use score-based election where each voter rates each candidate
        # Reputation serves as the "default score" - agents with higher
        # reputation for the candidates effectively get higher scores
        scores_ballot = {}
        for voter_id in voters:
            voter_scores = {}
            for candidate_id in candidates:
                # Each voter's "vote" is weighted by their own reputation
                # and scores candidates by the candidate's reputation
                voter_weight = agent_scores.get(voter_id, 50.0) / 50.0
                candidate_rep = agent_scores.get(candidate_id, 50.0)
                # Score = candidate's reputation, adjusted by voter's expertise
                voter_scores[candidate_id] = candidate_rep * voter_weight
            scores_ballot[voter_id] = voter_scores

        election_result = await self.context.call_skill(
            "consensus_protocol", "elect",
            {
                "role": f"assignee_for_{task_name[:30]}",
                "candidates": candidates,
                "method": "score",
                "scores": scores_ballot,
            }
        )

        if not election_result.success:
            return SkillResult(
                success=False,
                message=f"Consensus election failed: {election_result.message}",
            )

        winner = election_result.data.get("winner", candidates[0])
        election_id = election_result.data.get("election_id", "")

        # Step 3: Record reputation events for voting participation
        for voter_id in voters:
            await self.context.call_skill(
                "agent_reputation", "record_vote",
                {"agent_id": voter_id, "vote_type": "election"}
            )

        # Step 4: Optionally auto-delegate
        delegation_id = None
        if auto_delegate:
            delegate_result = await self.context.call_skill(
                "task_delegation", "delegate",
                {
                    "task_name": task_name,
                    "task_description": task_description,
                    "budget": budget,
                    "agent_id": winner,
                    "priority": "high",
                }
            )
            if delegate_result.success:
                delegation_id = delegate_result.data.get("delegation_id")

        # Step 5: Record
        state = self._load()
        state["stats"]["consensus_assigned"] += 1
        state["stats"]["total_smart_delegations"] += 1
        self._save(state)

        return SkillResult(
            success=True,
            message=(
                f"Consensus selected '{winner}' for '{task_name}' "
                f"(election: {election_id}, voters: {len(voters)})"
            ),
            data={
                "winner": winner,
                "winner_reputation": agent_scores.get(winner, 50.0),
                "election_id": election_id,
                "delegation_id": delegation_id,
                "voter_count": len(voters),
                "candidate_count": len(candidates),
                "scores_ballot": scores_ballot,
                "election_result": election_result.data.get("result", {}),
            },
        )

    # ─── AUTO REPORT ───────────────────────────────────────────────

    async def _auto_report(self, params: Dict) -> SkillResult:
        """After completion, update assignee's reputation automatically."""
        delegation_id = params.get("delegation_id", "").strip()
        success = params.get("success")
        budget_spent = params.get("budget_spent")
        on_time = params.get("on_time", True)

        if not delegation_id:
            return SkillResult(success=False, message="delegation_id is required")
        if success is None:
            return SkillResult(success=False, message="success (boolean) is required")

        if not self.context:
            return SkillResult(success=False, message="No SkillContext available")

        # Step 1: Get delegation details
        check_result = await self.context.call_skill(
            "task_delegation", "check",
            {"delegation_id": delegation_id}
        )

        if not check_result.success:
            return SkillResult(
                success=False,
                message=f"Could not find delegation: {check_result.message}",
            )

        agent_id = check_result.data.get("agent_id", "")
        task_name = check_result.data.get("task_name", "unknown")
        budget_allocated = check_result.data.get("budget", 0)

        if not agent_id:
            return SkillResult(
                success=False,
                message="No agent assigned to this delegation",
            )

        # Step 2: Calculate budget efficiency
        if budget_spent is not None and budget_allocated > 0:
            budget_efficiency = float(budget_spent) / float(budget_allocated)
        else:
            budget_efficiency = 0.5  # Neutral default

        # Step 3: Report to AgentReputationSkill
        rep_result = await self.context.call_skill(
            "agent_reputation", "record_task_outcome",
            {
                "agent_id": agent_id,
                "success": success,
                "budget_efficiency": budget_efficiency,
                "on_time": on_time,
                "task_name": task_name,
            }
        )

        # Step 4: Also report completion to TaskDelegation
        await self.context.call_skill(
            "task_delegation", "report_completion",
            {
                "delegation_id": delegation_id,
                "status": "completed" if success else "failed",
                "budget_spent": budget_spent or 0,
                "result": {"auto_reported": True},
                "error": "" if success else "Task failed",
            }
        )

        # Step 5: Record stats
        state = self._load()
        state["stats"]["auto_reports_filed"] += 1
        self._save(state)

        return SkillResult(
            success=True,
            message=(
                f"Auto-reported {'success' if success else 'failure'} for agent "
                f"'{agent_id}' on '{task_name}'. "
                f"Reputation updated: {rep_result.message if rep_result.success else 'update failed'}"
            ),
            data={
                "delegation_id": delegation_id,
                "agent_id": agent_id,
                "task_name": task_name,
                "success": success,
                "budget_efficiency": round(budget_efficiency, 3),
                "reputation_updated": rep_result.success,
                "reputation_data": rep_result.data if rep_result.success else {},
            },
        )

    # ─── RECOMMEND ─────────────────────────────────────────────────

    async def _recommend(self, params: Dict) -> SkillResult:
        """Recommend top agents for a task with reputation breakdown."""
        task_description = params.get("task_description", "").strip()
        candidates = params.get("candidate_agents", [])
        top_n = int(params.get("top_n", 3))

        if not task_description:
            return SkillResult(success=False, message="task_description is required")
        if not candidates:
            return SkillResult(success=False, message="candidate_agents list is required")

        # Get full reputation profiles
        recommendations = []
        for agent_id in candidates:
            profile = {"agent_id": agent_id}

            if self.context:
                rep_result = await self.context.call_skill(
                    "agent_reputation", "get_reputation",
                    {"agent_id": agent_id}
                )
                if rep_result.success:
                    profile.update({
                        "competence": rep_result.data.get("competence", 50.0),
                        "reliability": rep_result.data.get("reliability", 50.0),
                        "trustworthiness": rep_result.data.get("trustworthiness", 50.0),
                        "leadership": rep_result.data.get("leadership", 50.0),
                        "cooperation": rep_result.data.get("cooperation", 50.0),
                        "overall": rep_result.data.get("overall", 50.0),
                        "total_tasks": rep_result.data.get("total_tasks", 0),
                        "tasks_completed": rep_result.data.get("tasks_completed", 0),
                        "tasks_failed": rep_result.data.get("tasks_failed", 0),
                    })

                    # Compute task success rate
                    total = profile["total_tasks"]
                    if total > 0:
                        profile["success_rate"] = round(profile["tasks_completed"] / total * 100, 1)
                    else:
                        profile["success_rate"] = None

                    # Confidence: higher with more data points
                    profile["confidence"] = min(100.0, profile.get("total_tasks", 0) * 10.0)
                else:
                    profile.update({"overall": 50.0, "confidence": 0.0})
            else:
                profile.update({"overall": 50.0, "confidence": 0.0})

            recommendations.append(profile)

        # Sort by overall score
        recommendations.sort(key=lambda r: r.get("overall", 0), reverse=True)
        recommendations = recommendations[:top_n]

        # Add rank
        for i, rec in enumerate(recommendations):
            rec["rank"] = i + 1

        top_agent = recommendations[0] if recommendations else None
        msg = f"Top recommendation: {top_agent['agent_id']} (overall: {top_agent['overall']:.1f})" if top_agent else "No recommendations"

        return SkillResult(
            success=True,
            message=msg,
            data={
                "recommendations": recommendations,
                "task_description": task_description[:100],
                "total_evaluated": len(candidates),
            },
        )

    # ─── HELPER METHODS ────────────────────────────────────────────

    async def _rank_by_reputation(
        self,
        candidates: List[str],
        weights: Dict[str, float],
        min_score: float = 0,
    ) -> List[Dict]:
        """Rank candidate agents by weighted reputation score."""
        ranked = []

        for agent_id in candidates:
            if not agent_id:
                continue

            dimensions = {}
            overall = 50.0  # Default neutral

            if self.context:
                rep_result = await self.context.call_skill(
                    "agent_reputation", "get_reputation",
                    {"agent_id": agent_id}
                )
                if rep_result.success:
                    for dim in ["competence", "reliability", "trustworthiness", "leadership", "cooperation"]:
                        dimensions[dim] = rep_result.data.get(dim, 50.0)

                    # Compute weighted score
                    total_weight = sum(weights.values())
                    if total_weight > 0:
                        overall = sum(
                            weights.get(dim, 0) * dimensions.get(dim, 50.0)
                            for dim in dimensions
                        ) / total_weight
                    else:
                        overall = rep_result.data.get("overall", 50.0)
            else:
                dimensions = {d: 50.0 for d in weights}

            if overall >= min_score:
                ranked.append({
                    "agent_id": agent_id,
                    "score": round(overall, 2),
                    "dimensions": {k: round(v, 1) for k, v in dimensions.items()},
                })

        ranked.sort(key=lambda r: r["score"], reverse=True)
        return ranked

    async def _run_consensus_selection(
        self,
        ranked: List[Dict],
        task_name: str,
        task_description: str,
    ) -> tuple:
        """Run a consensus vote using reputation-weighted scores."""
        if not self.context:
            # Fallback to top ranked
            return ranked[0], "Consensus unavailable, fell back to reputation"

        candidates = [r["agent_id"] for r in ranked]
        # Use reputation scores as the score ballots
        scores_ballot = {}
        for voter in ranked:
            voter_scores = {}
            voter_weight = voter["score"] / 50.0 if voter["score"] > 0 else 1.0
            for candidate in ranked:
                voter_scores[candidate["agent_id"]] = candidate["score"] * voter_weight
            scores_ballot[voter["agent_id"]] = voter_scores

        election_result = await self.context.call_skill(
            "consensus_protocol", "elect",
            {
                "role": f"assignee_{task_name[:20]}",
                "candidates": candidates,
                "method": "score",
                "scores": scores_ballot,
            }
        )

        if election_result.success:
            winner_id = election_result.data.get("winner", candidates[0])
            winner = next((r for r in ranked if r["agent_id"] == winner_id), ranked[0])
            detail = (
                f"Consensus election winner (election: "
                f"{election_result.data.get('election_id', 'unknown')})"
            )
            return winner, detail

        # Fallback
        return ranked[0], "Consensus failed, fell back to reputation ranking"

    def _balanced_select(self, ranked: List[Dict]) -> tuple:
        """Select from top candidates, considering load balance."""
        # Simple: pick from top 3, rotating based on how many we've already assigned
        state = self._load()
        n = state["stats"]["total_smart_delegations"]
        top = ranked[:min(3, len(ranked))]
        selected = top[n % len(top)]
        return selected, f"Balanced selection (rotation from top {len(top)})"

    def _round_robin_select(self, ranked: List[Dict]) -> tuple:
        """Pure round-robin among all qualified candidates."""
        state = self._load()
        n = state["stats"]["total_smart_delegations"]
        selected = ranked[n % len(ranked)]
        return selected, f"Round-robin selection (index {n % len(ranked)} of {len(ranked)})"

    async def initialize(self) -> bool:
        self.initialized = True
        return True
