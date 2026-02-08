#!/usr/bin/env python3
"""
ReputationWeightedAssigner - Consensus-driven, reputation-weighted task assignment.

Wires together AgentReputationSkill, ConsensusProtocolSkill, and AgentNetworkSkill
to enable intelligent task routing: find capable agents, score them by reputation,
optionally run a consensus vote among peers, and assign work to the best candidate.

This bridges the Replication and Goal Setting pillars by ensuring tasks go to the
most competent and reliable agents, with democratic oversight when stakes are high.

Priority #1 from session 34 memory.
"""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from .base import Skill, SkillManifest, SkillAction, SkillResult


ASSIGNER_FILE = Path(__file__).parent.parent / "data" / "assignments.json"

# Reputation dimension weights for task assignment scoring
DEFAULT_WEIGHTS = {
    "competence": 0.35,
    "reliability": 0.30,
    "trustworthiness": 0.15,
    "cooperation": 0.10,
    "leadership": 0.10,
}


class ReputationWeightedAssigner(Skill):
    """
    Reputation-weighted task assignment with optional consensus voting.

    Workflow:
    1. find_candidates - Query agent network for agents with required capability
    2. score_candidates - Rank candidates by reputation across dimensions
    3. assign - Assign task to top candidate (auto or via consensus vote)
    4. complete - Record task outcome and update agent reputation

    Each step can be used independently or chained together via assign_auto
    which does everything in one call.
    """

    def __init__(self, credentials: Dict[str, str] = None):
        super().__init__(credentials)
        self._assignments: List[Dict] = []
        self._load()

    def _load(self):
        """Load assignment history from disk."""
        if ASSIGNER_FILE.exists():
            try:
                data = json.loads(ASSIGNER_FILE.read_text())
                self._assignments = data.get("assignments", [])
            except (json.JSONDecodeError, KeyError):
                self._assignments = []

    def _save(self):
        """Persist assignment history."""
        ASSIGNER_FILE.parent.mkdir(parents=True, exist_ok=True)
        data = {"assignments": self._assignments[-500:]}
        ASSIGNER_FILE.write_text(json.dumps(data, indent=2, default=str))

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="reputation_assigner",
            name="Reputation-Weighted Task Assigner",
            version="1.0.0",
            category="coordination",
            description="Assign tasks to agents using reputation scores and optional consensus voting",
            actions=[
                SkillAction(
                    name="find_candidates",
                    description="Find agents capable of handling a task, scored by reputation",
                    parameters={
                        "capability": {"type": "string", "required": True, "description": "Required capability"},
                        "min_reputation": {"type": "number", "required": False, "description": "Minimum overall reputation score (0-100, default 30)"},
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="score_candidates",
                    description="Score a list of agent IDs by their reputation for a specific task type",
                    parameters={
                        "agent_ids": {"type": "array", "required": True, "description": "List of agent IDs to score"},
                        "task_type": {"type": "string", "required": False, "description": "Task type to weight scoring (e.g., 'coding', 'review')"},
                        "weights": {"type": "object", "required": False, "description": "Custom dimension weights"},
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="assign",
                    description="Assign a task to the top-ranked candidate",
                    parameters={
                        "task_name": {"type": "string", "required": True, "description": "Name of the task"},
                        "task_description": {"type": "string", "required": True, "description": "Detailed task description"},
                        "agent_id": {"type": "string", "required": True, "description": "Agent ID to assign to"},
                        "budget": {"type": "number", "required": True, "description": "Budget for the task"},
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="assign_auto",
                    description="Automatically find, score, and assign task to the best available agent",
                    parameters={
                        "task_name": {"type": "string", "required": True, "description": "Name of the task"},
                        "task_description": {"type": "string", "required": True, "description": "Detailed description"},
                        "capability": {"type": "string", "required": True, "description": "Required capability"},
                        "budget": {"type": "number", "required": True, "description": "Budget in USD"},
                        "min_reputation": {"type": "number", "required": False, "description": "Minimum reputation (default 30)"},
                        "use_consensus": {"type": "boolean", "required": False, "description": "Use consensus vote among top candidates (default false)"},
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="complete",
                    description="Report task completion and update agent reputation",
                    parameters={
                        "assignment_id": {"type": "string", "required": True, "description": "Assignment ID"},
                        "success": {"type": "boolean", "required": True, "description": "Whether task succeeded"},
                        "quality_score": {"type": "number", "required": False, "description": "Quality 0-100 (default 50)"},
                        "budget_used": {"type": "number", "required": False, "description": "Actual budget consumed"},
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="history",
                    description="View assignment history with optional filters",
                    parameters={
                        "agent_id": {"type": "string", "required": False, "description": "Filter by agent"},
                        "status": {"type": "string", "required": False, "description": "Filter by status"},
                        "last_n": {"type": "integer", "required": False, "description": "Limit results"},
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="leaderboard",
                    description="Show agent assignment performance leaderboard",
                    parameters={
                        "sort_by": {"type": "string", "required": False, "description": "Sort field: success_rate, total_tasks, avg_quality (default success_rate)"},
                    },
                    estimated_cost=0,
                ),
            ],
            required_credentials=[],
        )

    def check_credentials(self) -> bool:
        return True

    async def execute(self, action: str, params: Dict) -> SkillResult:
        handlers = {
            "find_candidates": self._find_candidates,
            "score_candidates": self._score_candidates,
            "assign": self._assign,
            "assign_auto": self._assign_auto,
            "complete": self._complete,
            "history": self._history,
            "leaderboard": self._leaderboard,
        }
        handler = handlers.get(action)
        if not handler:
            return SkillResult(success=False, message=f"Unknown action: {action}")
        return await handler(params)

    async def _find_candidates(self, params: Dict) -> SkillResult:
        """Find agents with a capability, filtered by minimum reputation."""
        capability = params.get("capability", "").strip()
        if not capability:
            return SkillResult(success=False, message="capability is required")

        min_rep = float(params.get("min_reputation", 30))
        candidates = []

        # Query agent network for capable agents
        if self.context:
            route_result = await self.context.call_skill(
                "agent_network", "route", {"capability": capability}
            )
            if route_result.success:
                matches = route_result.data.get("matches", [])
                for match in matches:
                    agent_id = match.get("agent_id", "")
                    if not agent_id:
                        continue

                    # Get reputation score
                    rep_result = await self.context.call_skill(
                        "agent_reputation", "get_reputation", {"agent_id": agent_id}
                    )
                    rep_data = {}
                    overall = 50.0  # Default neutral reputation
                    if rep_result.success:
                        rep_data = rep_result.data
                        overall = rep_data.get("overall", 50.0)

                    if overall >= min_rep:
                        candidates.append({
                            "agent_id": agent_id,
                            "overall_reputation": overall,
                            "reputation": rep_data,
                            "agent_info": match,
                        })

            # Sort by reputation (highest first)
            candidates.sort(key=lambda c: c["overall_reputation"], reverse=True)

        return SkillResult(
            success=True,
            message=f"Found {len(candidates)} candidates for '{capability}'",
            data={"candidates": candidates, "capability": capability},
        )

    async def _score_candidates(self, params: Dict) -> SkillResult:
        """Score agent IDs by their reputation dimensions with custom weights."""
        agent_ids = params.get("agent_ids", [])
        if not agent_ids:
            return SkillResult(success=False, message="agent_ids is required")

        weights = params.get("weights", DEFAULT_WEIGHTS)
        task_type = params.get("task_type", "")

        # Adjust weights by task type
        if task_type == "coding":
            weights = {**weights, "competence": 0.45, "reliability": 0.25}
        elif task_type == "review":
            weights = {**weights, "trustworthiness": 0.30, "competence": 0.30}
        elif task_type == "coordination":
            weights = {**weights, "leadership": 0.30, "cooperation": 0.25}

        # Normalize weights
        total_w = sum(weights.values())
        if total_w > 0:
            weights = {k: v / total_w for k, v in weights.items()}

        scored = []
        for agent_id in agent_ids:
            rep_data = {}
            if self.context:
                rep_result = await self.context.call_skill(
                    "agent_reputation", "get_reputation", {"agent_id": agent_id}
                )
                if rep_result.success:
                    rep_data = rep_result.data

            # Compute weighted score
            dimensions = rep_data.get("dimensions", {})
            score = 0.0
            for dim, weight in weights.items():
                score += dimensions.get(dim, 50.0) * weight

            scored.append({
                "agent_id": agent_id,
                "weighted_score": round(score, 2),
                "dimensions": dimensions,
                "weights_used": weights,
            })

        scored.sort(key=lambda s: s["weighted_score"], reverse=True)
        return SkillResult(
            success=True,
            message=f"Scored {len(scored)} candidates",
            data={"scored": scored},
        )

    async def _assign(self, params: Dict) -> SkillResult:
        """Assign a task directly to a specific agent."""
        task_name = params.get("task_name", "").strip()
        task_desc = params.get("task_description", "").strip()
        agent_id = params.get("agent_id", "").strip()
        budget = float(params.get("budget", 0))

        if not task_name:
            return SkillResult(success=False, message="task_name is required")
        if not agent_id:
            return SkillResult(success=False, message="agent_id is required")
        if budget <= 0:
            return SkillResult(success=False, message="budget must be positive")

        assignment_id = f"asgn_{uuid.uuid4().hex[:10]}"
        assignment = {
            "id": assignment_id,
            "task_name": task_name,
            "task_description": task_desc,
            "agent_id": agent_id,
            "budget": budget,
            "status": "assigned",
            "created_at": datetime.now().isoformat(),
            "completed_at": None,
            "success": None,
            "quality_score": None,
            "budget_used": None,
            "method": "direct",
        }
        self._assignments.append(assignment)

        # Delegate via task_delegation if available
        if self.context:
            await self.context.call_skill("task_delegation", "delegate", {
                "task_name": task_name,
                "task_description": task_desc,
                "agent_id": agent_id,
                "budget": budget,
            })

        self._save()
        return SkillResult(
            success=True,
            message=f"Task '{task_name}' assigned to {agent_id}",
            data={"assignment_id": assignment_id, "agent_id": agent_id},
        )

    async def _assign_auto(self, params: Dict) -> SkillResult:
        """Full auto-assignment: find → score → optionally vote → assign."""
        task_name = params.get("task_name", "").strip()
        task_desc = params.get("task_description", "").strip()
        capability = params.get("capability", "").strip()
        budget = float(params.get("budget", 0))
        use_consensus = params.get("use_consensus", False)
        min_rep = float(params.get("min_reputation", 30))

        if not task_name:
            return SkillResult(success=False, message="task_name is required")
        if not capability:
            return SkillResult(success=False, message="capability is required")
        if budget <= 0:
            return SkillResult(success=False, message="budget must be positive")

        # Step 1: Find candidates
        find_result = await self._find_candidates({
            "capability": capability,
            "min_reputation": min_rep,
        })
        candidates = find_result.data.get("candidates", [])

        if not candidates:
            return SkillResult(
                success=False,
                message=f"No agents found with capability '{capability}' and reputation >= {min_rep}",
            )

        # Step 2: Score candidates
        agent_ids = [c["agent_id"] for c in candidates]
        score_result = await self._score_candidates({
            "agent_ids": agent_ids,
            "task_type": params.get("task_type", ""),
        })
        scored = score_result.data.get("scored", [])

        chosen_agent = scored[0]["agent_id"] if scored else candidates[0]["agent_id"]
        assignment_method = "reputation"

        # Step 3: Optional consensus vote (for high-stakes tasks)
        if use_consensus and len(scored) >= 2 and self.context:
            top_n = [s["agent_id"] for s in scored[:5]]
            proposal_id = f"assign_{uuid.uuid4().hex[:8]}"

            # Create a proposal for the assignment
            await self.context.call_skill("consensus_protocol", "propose", {
                "proposal_id": proposal_id,
                "title": f"Assign: {task_name}",
                "description": f"Who should handle: {task_desc[:200]}",
                "options": top_n,
                "voting_method": "score",
            })

            # Auto-cast reputation-weighted votes for each candidate
            for voter_id in top_n:
                scores_map = {}
                for candidate in scored[:5]:
                    scores_map[candidate["agent_id"]] = candidate["weighted_score"]
                await self.context.call_skill("consensus_protocol", "vote", {
                    "proposal_id": proposal_id,
                    "voter_id": voter_id,
                    "scores": scores_map,
                })

            # Tally the votes
            tally_result = await self.context.call_skill("consensus_protocol", "tally", {
                "proposal_id": proposal_id,
            })
            if tally_result.success:
                winner = tally_result.data.get("winner", "")
                if winner:
                    chosen_agent = winner
                    assignment_method = "consensus"

        # Step 4: Create assignment
        assignment_id = f"asgn_{uuid.uuid4().hex[:10]}"
        assignment = {
            "id": assignment_id,
            "task_name": task_name,
            "task_description": task_desc,
            "agent_id": chosen_agent,
            "budget": budget,
            "status": "assigned",
            "created_at": datetime.now().isoformat(),
            "completed_at": None,
            "success": None,
            "quality_score": None,
            "budget_used": None,
            "method": assignment_method,
            "candidates_considered": len(candidates),
            "consensus_used": use_consensus,
        }
        self._assignments.append(assignment)

        # Delegate
        if self.context:
            await self.context.call_skill("task_delegation", "delegate", {
                "task_name": task_name,
                "task_description": task_desc,
                "agent_id": chosen_agent,
                "budget": budget,
            })

        self._save()
        return SkillResult(
            success=True,
            message=f"Task '{task_name}' assigned to {chosen_agent} via {assignment_method}",
            data={
                "assignment_id": assignment_id,
                "agent_id": chosen_agent,
                "method": assignment_method,
                "candidates_considered": len(candidates),
                "scored": scored[:3],
            },
        )

    async def _complete(self, params: Dict) -> SkillResult:
        """Report task completion and update agent reputation based on outcome."""
        assignment_id = params.get("assignment_id", "").strip()
        success = params.get("success", False)
        quality_score = float(params.get("quality_score", 50))
        budget_used = params.get("budget_used")

        if not assignment_id:
            return SkillResult(success=False, message="assignment_id is required")

        # Find assignment
        assignment = None
        for a in self._assignments:
            if a["id"] == assignment_id:
                assignment = a
                break

        if not assignment:
            return SkillResult(success=False, message=f"Assignment {assignment_id} not found")

        if assignment["status"] == "completed":
            return SkillResult(success=False, message="Assignment already completed")

        # Update assignment
        assignment["status"] = "completed"
        assignment["completed_at"] = datetime.now().isoformat()
        assignment["success"] = bool(success)
        assignment["quality_score"] = quality_score
        if budget_used is not None:
            assignment["budget_used"] = float(budget_used)

        agent_id = assignment["agent_id"]

        # Update reputation via AgentReputationSkill
        if self.context:
            budget_efficiency = 1.0
            if budget_used is not None and assignment["budget"] > 0:
                budget_efficiency = 1.0 - (float(budget_used) / assignment["budget"])
                budget_efficiency = max(0.0, min(1.0, budget_efficiency))

            await self.context.call_skill("agent_reputation", "record_task_outcome", {
                "agent_id": agent_id,
                "success": success,
                "budget_efficiency": budget_efficiency,
            })

            # Bonus reputation event for high quality
            if success and quality_score >= 80:
                await self.context.call_skill("agent_reputation", "record_event", {
                    "agent_id": agent_id,
                    "event_type": "high_quality_delivery",
                    "dimension": "competence",
                    "delta": min(quality_score / 20, 5.0),
                    "reason": f"High quality ({quality_score}) on '{assignment['task_name']}'",
                })

        self._save()
        return SkillResult(
            success=True,
            message=f"Assignment {assignment_id} completed ({'success' if success else 'failure'})",
            data={
                "agent_id": agent_id,
                "success": success,
                "quality_score": quality_score,
                "reputation_updated": True,
            },
        )

    async def _history(self, params: Dict) -> SkillResult:
        """View assignment history with optional filters."""
        agent_id = params.get("agent_id", "").strip()
        status_filter = params.get("status", "").strip()
        last_n = int(params.get("last_n", 20))

        records = list(self._assignments)
        if agent_id:
            records = [r for r in records if r.get("agent_id") == agent_id]
        if status_filter:
            records = [r for r in records if r.get("status") == status_filter]

        records = records[-last_n:]
        return SkillResult(
            success=True,
            message=f"Showing {len(records)} assignment(s)",
            data={"assignments": records},
        )

    async def _leaderboard(self, params: Dict) -> SkillResult:
        """Generate assignment performance leaderboard from assignment history."""
        sort_by = params.get("sort_by", "success_rate")

        # Aggregate per agent
        agent_stats: Dict[str, Dict] = {}
        for a in self._assignments:
            aid = a.get("agent_id", "unknown")
            if aid not in agent_stats:
                agent_stats[aid] = {
                    "agent_id": aid,
                    "total_tasks": 0,
                    "completed": 0,
                    "successes": 0,
                    "total_quality": 0.0,
                    "total_budget": 0.0,
                    "total_budget_used": 0.0,
                }
            s = agent_stats[aid]
            s["total_tasks"] += 1
            if a.get("status") == "completed":
                s["completed"] += 1
                if a.get("success"):
                    s["successes"] += 1
                if a.get("quality_score") is not None:
                    s["total_quality"] += a["quality_score"]
                if a.get("budget_used") is not None:
                    s["total_budget_used"] += a["budget_used"]
            s["total_budget"] += a.get("budget", 0)

        # Compute derived metrics
        entries = []
        for aid, s in agent_stats.items():
            completed = s["completed"]
            success_rate = (s["successes"] / completed * 100) if completed > 0 else 0
            avg_quality = (s["total_quality"] / completed) if completed > 0 else 0
            budget_efficiency = 0.0
            if s["total_budget"] > 0 and s["total_budget_used"] > 0:
                budget_efficiency = round((1 - s["total_budget_used"] / s["total_budget"]) * 100, 1)

            entries.append({
                "agent_id": aid,
                "total_tasks": s["total_tasks"],
                "completed": completed,
                "success_rate": round(success_rate, 1),
                "avg_quality": round(avg_quality, 1),
                "budget_efficiency": budget_efficiency,
            })

        sort_key = {
            "success_rate": lambda e: e["success_rate"],
            "total_tasks": lambda e: e["total_tasks"],
            "avg_quality": lambda e: e["avg_quality"],
        }.get(sort_by, lambda e: e["success_rate"])

        entries.sort(key=sort_key, reverse=True)
        return SkillResult(
            success=True,
            message=f"Leaderboard: {len(entries)} agents",
            data={"leaderboard": entries, "sort_by": sort_by},
        )
