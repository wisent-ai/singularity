#!/usr/bin/env python3
"""
ConsensusTaskAssignmentSkill - Reputation-weighted democratic task assignment.

Wires together ConsensusProtocolSkill, AgentReputationSkill, and
TaskDelegationSkill to enable democratic, reputation-informed task assignment
across an agent network.

Instead of a parent agent unilaterally deciding which child handles a task,
agents can:
1. PROPOSE a task for democratic assignment
2. NOMINATE candidates (or auto-discover via capabilities)
3. VOTE on who should handle it (votes weighted by reputation)
4. ASSIGN the winning agent via TaskDelegation
5. REVIEW past assignments and track outcome quality

This creates a meritocratic task distribution system where:
- Agents with higher reputation get more influence in voting
- Agents with proven competence in relevant areas get prioritized
- Task outcomes feed back into reputation, creating a virtuous cycle
- The network self-organizes without centralized authority

Pillars served:
- Replication: Self-governing agent networks that assign work democratically
- Self-Improvement: Reputation feedback loop drives agent improvement
- Revenue: Better task-agent matching improves service quality
- Goal Setting: Strategic work distribution based on collective intelligence
"""

import json
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import Skill, SkillAction, SkillManifest, SkillResult

ASSIGNMENT_FILE = Path(__file__).parent.parent / "data" / "consensus_assignments.json"

# Assignment statuses
STATUS_NOMINATING = "nominating"       # Collecting candidates
STATUS_VOTING = "voting"              # Agents are voting
STATUS_ASSIGNED = "assigned"          # Winner selected and task delegated
STATUS_COMPLETED = "completed"        # Task finished
STATUS_FAILED = "failed"             # Task failed or no valid candidates
STATUS_CANCELLED = "cancelled"        # Cancelled before completion

# Voting strategies
STRATEGY_REPUTATION_WEIGHTED = "reputation_weighted"  # Vote weight = reputation score
STRATEGY_EQUAL = "equal"                              # All votes equal weight
STRATEGY_COMPETENCE_ONLY = "competence_only"          # Weight by competence dimension only

# Defaults
DEFAULT_VOTE_HOURS = 1
DEFAULT_MIN_CANDIDATES = 1
MAX_ASSIGNMENTS = 200
MAX_CANDIDATES = 20


class ConsensusTaskAssignmentSkill(Skill):
    """
    Democratic task assignment using reputation-weighted consensus voting.

    Bridges ConsensusProtocol (voting), AgentReputation (weighting), and
    TaskDelegation (execution) into a unified meritocratic assignment flow.
    """

    def __init__(self, credentials: Dict[str, str] = None):
        super().__init__(credentials)
        self._ensure_data()

    def _ensure_data(self):
        ASSIGNMENT_FILE.parent.mkdir(parents=True, exist_ok=True)
        if not ASSIGNMENT_FILE.exists():
            self._save(self._default_state())

    def _default_state(self) -> Dict:
        return {
            "assignments": {},
            "stats": {
                "total_proposed": 0,
                "total_assigned": 0,
                "total_completed": 0,
                "total_failed": 0,
                "total_cancelled": 0,
                "avg_candidates_per_task": 0.0,
                "avg_voters_per_task": 0.0,
            },
        }

    def _load(self) -> Dict:
        try:
            with open(ASSIGNMENT_FILE, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return self._default_state()

    def _save(self, state: Dict):
        ASSIGNMENT_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(ASSIGNMENT_FILE, "w") as f:
            json.dump(state, f, indent=2, default=str)

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="consensus_task_assignment",
            name="Consensus Task Assignment",
            version="1.0.0",
            category="coordination",
            description=(
                "Democratic, reputation-weighted task assignment across agent networks. "
                "Combines consensus voting with agent reputation for meritocratic work distribution."
            ),
            required_credentials=[],
            actions=[
                SkillAction(
                    name="propose",
                    description="Propose a task for democratic assignment among agents",
                    parameters={
                        "task_name": {"type": "string", "required": True, "description": "Name of the task to assign"},
                        "task_description": {"type": "string", "required": True, "description": "Detailed task description"},
                        "budget": {"type": "number", "required": True, "description": "Budget for the task"},
                        "proposer": {"type": "string", "required": True, "description": "Agent ID proposing the task"},
                        "required_capability": {"type": "string", "required": False, "description": "Capability needed (for auto-discovery of candidates)"},
                        "candidates": {"type": "list", "required": False, "description": "Explicit list of candidate agent IDs"},
                        "voting_strategy": {"type": "string", "required": False, "description": "reputation_weighted (default), equal, or competence_only"},
                        "vote_hours": {"type": "number", "required": False, "description": "Hours to keep voting open (default 1)"},
                        "min_candidates": {"type": "number", "required": False, "description": "Minimum candidates before voting (default 1)"},
                        "auto_assign": {"type": "boolean", "required": False, "description": "Auto-assign winner when voting closes (default true)"},
                        "priority": {"type": "string", "required": False, "description": "Task priority: low, normal, high, critical"},
                    },
                ),
                SkillAction(
                    name="nominate",
                    description="Nominate an agent as a candidate for a task",
                    parameters={
                        "assignment_id": {"type": "string", "required": True, "description": "Assignment to nominate for"},
                        "candidate_id": {"type": "string", "required": True, "description": "Agent ID being nominated"},
                        "nominator": {"type": "string", "required": True, "description": "Agent ID making the nomination"},
                        "rationale": {"type": "string", "required": False, "description": "Why this agent is a good fit"},
                    },
                ),
                SkillAction(
                    name="vote",
                    description="Vote for a candidate to handle the task (weight auto-derived from reputation)",
                    parameters={
                        "assignment_id": {"type": "string", "required": True, "description": "Assignment to vote on"},
                        "voter": {"type": "string", "required": True, "description": "Voting agent ID"},
                        "candidate_id": {"type": "string", "required": True, "description": "Candidate being voted for"},
                        "rationale": {"type": "string", "required": False, "description": "Reason for this vote"},
                    },
                ),
                SkillAction(
                    name="close_voting",
                    description="Close voting, determine winner, and optionally assign the task",
                    parameters={
                        "assignment_id": {"type": "string", "required": True, "description": "Assignment to close voting on"},
                    },
                ),
                SkillAction(
                    name="status",
                    description="Get status of a task assignment or list all active assignments",
                    parameters={
                        "assignment_id": {"type": "string", "required": False, "description": "Specific assignment ID (or omit for all active)"},
                    },
                ),
                SkillAction(
                    name="report_outcome",
                    description="Report the outcome of an assigned task (feeds back into reputation)",
                    parameters={
                        "assignment_id": {"type": "string", "required": True, "description": "Assignment that was completed"},
                        "success": {"type": "boolean", "required": True, "description": "Whether the task succeeded"},
                        "quality_score": {"type": "number", "required": False, "description": "Quality score 0-100 (default 50)"},
                        "budget_spent": {"type": "number", "required": False, "description": "Actual budget spent"},
                        "notes": {"type": "string", "required": False, "description": "Additional outcome notes"},
                    },
                ),
                SkillAction(
                    name="leaderboard",
                    description="Show which agents are most frequently assigned tasks and their success rates",
                    parameters={
                        "limit": {"type": "number", "required": False, "description": "Number of agents to show (default 10)"},
                    },
                ),
                SkillAction(
                    name="history",
                    description="Review past consensus-driven assignments",
                    parameters={
                        "limit": {"type": "number", "required": False, "description": "Max entries (default 20)"},
                        "status_filter": {"type": "string", "required": False, "description": "Filter by status"},
                    },
                ),
            ],
        )

    async def execute(self, action: str, params: Dict[str, Any]) -> SkillResult:
        handlers = {
            "propose": self._propose,
            "nominate": self._nominate,
            "vote": self._vote,
            "close_voting": self._close_voting,
            "status": self._status,
            "report_outcome": self._report_outcome,
            "leaderboard": self._leaderboard,
            "history": self._history,
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

    # ─── PROPOSE ──────────────────────────────────────────────────────

    async def _propose(self, params: Dict) -> SkillResult:
        """Propose a task for democratic assignment."""
        task_name = params.get("task_name", "").strip()
        task_description = params.get("task_description", "").strip()
        budget = float(params.get("budget", 0))
        proposer = params.get("proposer", "").strip()

        if not task_name:
            return SkillResult(success=False, message="task_name is required")
        if not task_description:
            return SkillResult(success=False, message="task_description is required")
        if budget <= 0:
            return SkillResult(success=False, message="budget must be positive")
        if not proposer:
            return SkillResult(success=False, message="proposer is required")

        state = self._load()

        # Enforce limit
        active = sum(1 for a in state["assignments"].values()
                     if a["status"] in (STATUS_NOMINATING, STATUS_VOTING))
        if active >= MAX_ASSIGNMENTS:
            return SkillResult(success=False, message=f"Too many active assignments ({MAX_ASSIGNMENTS})")

        assignment_id = f"cta-{uuid.uuid4().hex[:8]}"
        candidates = params.get("candidates", [])
        voting_strategy = params.get("voting_strategy", STRATEGY_REPUTATION_WEIGHTED)
        vote_hours = float(params.get("vote_hours", DEFAULT_VOTE_HOURS))
        min_candidates = int(params.get("min_candidates", DEFAULT_MIN_CANDIDATES))
        auto_assign = params.get("auto_assign", True)
        priority = params.get("priority", "normal")

        if voting_strategy not in (STRATEGY_REPUTATION_WEIGHTED, STRATEGY_EQUAL, STRATEGY_COMPETENCE_ONLY):
            voting_strategy = STRATEGY_REPUTATION_WEIGHTED

        if priority not in ("low", "normal", "high", "critical"):
            priority = "normal"

        # Auto-discover candidates via AgentNetwork if none specified
        discovered_candidates = []
        if not candidates and self.context:
            capability = params.get("required_capability", "")
            if capability:
                route_result = await self.context.call_skill(
                    "agent_network", "route",
                    {"capability": capability}
                )
                if route_result.success and route_result.data.get("matches"):
                    discovered_candidates = [
                        m.get("agent_id", "") for m in route_result.data["matches"]
                        if m.get("agent_id")
                    ]

        all_candidates = list(set(candidates + discovered_candidates))[:MAX_CANDIDATES]

        # Fetch reputation scores for candidates
        candidate_reputations = {}
        for cid in all_candidates:
            rep = await self._get_agent_reputation(cid)
            candidate_reputations[cid] = rep

        # Determine initial status
        initial_status = STATUS_VOTING if len(all_candidates) >= min_candidates else STATUS_NOMINATING

        assignment = {
            "id": assignment_id,
            "task_name": task_name,
            "task_description": task_description,
            "budget": budget,
            "proposer": proposer,
            "priority": priority,
            "required_capability": params.get("required_capability", ""),
            "voting_strategy": voting_strategy,
            "vote_hours": vote_hours,
            "min_candidates": min_candidates,
            "auto_assign": auto_assign,
            "status": initial_status,
            "candidates": {
                cid: {
                    "agent_id": cid,
                    "nominator": proposer if cid in candidates else "auto_discovery",
                    "reputation": candidate_reputations.get(cid, {}),
                    "nominated_at": datetime.utcnow().isoformat(),
                }
                for cid in all_candidates
            },
            "votes": {},
            "winner": None,
            "delegation_id": None,
            "outcome": None,
            "created_at": datetime.utcnow().isoformat(),
            "voting_deadline": (datetime.utcnow() + timedelta(hours=vote_hours)).isoformat(),
            "closed_at": None,
        }

        state["assignments"][assignment_id] = assignment
        state["stats"]["total_proposed"] += 1
        self._save(state)

        msg = f"Task '{task_name}' proposed for democratic assignment. "
        msg += f"ID: {assignment_id}. "
        msg += f"Candidates: {len(all_candidates)}. "
        msg += f"Status: {initial_status}. "
        msg += f"Strategy: {voting_strategy}."

        return SkillResult(
            success=True,
            message=msg,
            data={
                "assignment_id": assignment_id,
                "status": initial_status,
                "candidates": list(all_candidates),
                "candidate_reputations": candidate_reputations,
                "voting_deadline": assignment["voting_deadline"],
            },
        )

    # ─── NOMINATE ─────────────────────────────────────────────────────

    async def _nominate(self, params: Dict) -> SkillResult:
        """Add a candidate to a task assignment."""
        assignment_id = params.get("assignment_id", "").strip()
        candidate_id = params.get("candidate_id", "").strip()
        nominator = params.get("nominator", "").strip()

        if not all([assignment_id, candidate_id, nominator]):
            return SkillResult(success=False, message="assignment_id, candidate_id, and nominator are required")

        state = self._load()
        assignment = state["assignments"].get(assignment_id)
        if not assignment:
            return SkillResult(success=False, message=f"Assignment '{assignment_id}' not found")

        if assignment["status"] not in (STATUS_NOMINATING, STATUS_VOTING):
            return SkillResult(success=False, message=f"Assignment is '{assignment['status']}', not accepting nominations")

        if candidate_id in assignment["candidates"]:
            return SkillResult(success=False, message=f"Agent '{candidate_id}' is already a candidate")

        if len(assignment["candidates"]) >= MAX_CANDIDATES:
            return SkillResult(success=False, message=f"Maximum candidates ({MAX_CANDIDATES}) reached")

        # Fetch reputation
        rep = await self._get_agent_reputation(candidate_id)

        assignment["candidates"][candidate_id] = {
            "agent_id": candidate_id,
            "nominator": nominator,
            "rationale": params.get("rationale", ""),
            "reputation": rep,
            "nominated_at": datetime.utcnow().isoformat(),
        }

        # Auto-transition to voting if we have enough candidates
        if (assignment["status"] == STATUS_NOMINATING and
                len(assignment["candidates"]) >= assignment["min_candidates"]):
            assignment["status"] = STATUS_VOTING
            assignment["voting_deadline"] = (
                datetime.utcnow() + timedelta(hours=assignment["vote_hours"])
            ).isoformat()

        self._save(state)

        return SkillResult(
            success=True,
            message=f"Agent '{candidate_id}' nominated for '{assignment['task_name']}'. "
                    f"Total candidates: {len(assignment['candidates'])}. Status: {assignment['status']}.",
            data={
                "assignment_id": assignment_id,
                "candidate_id": candidate_id,
                "reputation": rep,
                "total_candidates": len(assignment["candidates"]),
                "status": assignment["status"],
            },
        )

    # ─── VOTE ──────────────────────────────────────────────────────────

    async def _vote(self, params: Dict) -> SkillResult:
        """Cast a reputation-weighted vote for a candidate."""
        assignment_id = params.get("assignment_id", "").strip()
        voter = params.get("voter", "").strip()
        candidate_id = params.get("candidate_id", "").strip()

        if not all([assignment_id, voter, candidate_id]):
            return SkillResult(success=False, message="assignment_id, voter, and candidate_id are required")

        state = self._load()
        assignment = state["assignments"].get(assignment_id)
        if not assignment:
            return SkillResult(success=False, message=f"Assignment '{assignment_id}' not found")

        if assignment["status"] != STATUS_VOTING:
            return SkillResult(success=False, message=f"Assignment is '{assignment['status']}', not open for voting")

        if candidate_id not in assignment["candidates"]:
            return SkillResult(
                success=False,
                message=f"'{candidate_id}' is not a candidate. Available: {list(assignment['candidates'].keys())}",
            )

        # Get voter's reputation to determine vote weight
        vote_weight = await self._compute_vote_weight(voter, assignment["voting_strategy"])

        # Check if voter already voted (allow changing vote)
        changed_from = None
        if voter in assignment["votes"]:
            changed_from = assignment["votes"][voter]["candidate_id"]

        assignment["votes"][voter] = {
            "voter": voter,
            "candidate_id": candidate_id,
            "weight": vote_weight,
            "rationale": params.get("rationale", ""),
            "voted_at": datetime.utcnow().isoformat(),
        }

        self._save(state)

        msg = f"Vote recorded: '{voter}' votes for '{candidate_id}' (weight: {vote_weight:.2f}). "
        if changed_from:
            msg += f"Changed from '{changed_from}'. "
        msg += f"Total votes: {len(assignment['votes'])}."

        return SkillResult(
            success=True,
            message=msg,
            data={
                "assignment_id": assignment_id,
                "voter": voter,
                "candidate_id": candidate_id,
                "weight": vote_weight,
                "total_votes": len(assignment["votes"]),
                "changed_from": changed_from,
            },
        )

    # ─── CLOSE VOTING ─────────────────────────────────────────────────

    async def _close_voting(self, params: Dict) -> SkillResult:
        """Close voting, tally results, determine winner, optionally assign."""
        assignment_id = params.get("assignment_id", "").strip()
        if not assignment_id:
            return SkillResult(success=False, message="assignment_id is required")

        state = self._load()
        assignment = state["assignments"].get(assignment_id)
        if not assignment:
            return SkillResult(success=False, message=f"Assignment '{assignment_id}' not found")

        if assignment["status"] not in (STATUS_VOTING, STATUS_NOMINATING):
            return SkillResult(
                success=False,
                message=f"Assignment is '{assignment['status']}', cannot close voting",
            )

        candidates = assignment["candidates"]
        votes = assignment["votes"]

        if not candidates:
            assignment["status"] = STATUS_FAILED
            assignment["closed_at"] = datetime.utcnow().isoformat()
            self._save(state)
            return SkillResult(success=False, message="No candidates to assign task to")

        # Tally weighted votes per candidate
        tallies = {cid: 0.0 for cid in candidates}
        vote_details = []
        for voter, vote_data in votes.items():
            cid = vote_data["candidate_id"]
            weight = vote_data["weight"]
            if cid in tallies:
                tallies[cid] += weight
                vote_details.append({
                    "voter": voter,
                    "candidate": cid,
                    "weight": weight,
                })

        # If no votes were cast, use reputation scores as tiebreaker
        if not votes:
            for cid, cdata in candidates.items():
                rep = cdata.get("reputation", {})
                tallies[cid] = rep.get("overall", 50.0) / 100.0  # Normalize to 0-1

        # Determine winner (highest weighted vote total)
        winner = max(tallies, key=tallies.get)
        winner_score = tallies[winner]

        assignment["winner"] = winner
        assignment["status"] = STATUS_ASSIGNED
        assignment["closed_at"] = datetime.utcnow().isoformat()
        assignment["tally"] = {
            "scores": tallies,
            "total_votes": len(votes),
            "winner": winner,
            "winner_score": winner_score,
            "vote_details": vote_details,
        }

        # Auto-assign via TaskDelegation if enabled
        delegation_id = None
        if assignment.get("auto_assign", True) and self.context:
            try:
                deleg_result = await self.context.call_skill(
                    "task_delegation", "delegate",
                    {
                        "task_name": assignment["task_name"],
                        "task_description": assignment["task_description"],
                        "budget": assignment["budget"],
                        "agent_id": winner,
                        "priority": assignment.get("priority", "normal"),
                    }
                )
                if deleg_result.success:
                    delegation_id = deleg_result.data.get("delegation_id")
                    assignment["delegation_id"] = delegation_id
            except Exception:
                pass  # Delegation is best-effort

        # Record vote participation in reputation
        if self.context:
            for voter_id in votes:
                try:
                    await self.context.call_skill(
                        "agent_reputation", "record_vote",
                        {
                            "agent_id": voter_id,
                            "vote_correct": True,  # Participation always earns credit
                            "source": "consensus_task_assignment",
                        }
                    )
                except Exception:
                    pass  # Best-effort

        state["stats"]["total_assigned"] += 1
        self._update_avg_stats(state)
        self._save(state)

        msg = f"Voting closed for '{assignment['task_name']}'. "
        msg += f"Winner: '{winner}' with score {winner_score:.2f}. "
        msg += f"Votes cast: {len(votes)}. "
        if delegation_id:
            msg += f"Task delegated (ID: {delegation_id})."
        else:
            msg += "Auto-delegation skipped or unavailable."

        return SkillResult(
            success=True,
            message=msg,
            data={
                "assignment_id": assignment_id,
                "winner": winner,
                "winner_score": winner_score,
                "tallies": tallies,
                "total_votes": len(votes),
                "delegation_id": delegation_id,
                "vote_details": vote_details,
            },
        )

    # ─── STATUS ────────────────────────────────────────────────────────

    async def _status(self, params: Dict) -> SkillResult:
        """Get assignment status or list all active."""
        state = self._load()
        assignment_id = params.get("assignment_id", "").strip()

        if assignment_id:
            assignment = state["assignments"].get(assignment_id)
            if not assignment:
                return SkillResult(success=False, message=f"Assignment '{assignment_id}' not found")

            return SkillResult(
                success=True,
                message=f"Assignment '{assignment['task_name']}': {assignment['status']}",
                data={
                    "assignment": {
                        "id": assignment["id"],
                        "task_name": assignment["task_name"],
                        "status": assignment["status"],
                        "candidates": list(assignment["candidates"].keys()),
                        "votes": len(assignment["votes"]),
                        "winner": assignment.get("winner"),
                        "delegation_id": assignment.get("delegation_id"),
                        "budget": assignment["budget"],
                        "priority": assignment.get("priority"),
                        "voting_strategy": assignment["voting_strategy"],
                        "voting_deadline": assignment.get("voting_deadline"),
                        "tally": assignment.get("tally"),
                    },
                },
            )

        # List all active
        active = []
        for a in state["assignments"].values():
            if a["status"] in (STATUS_NOMINATING, STATUS_VOTING, STATUS_ASSIGNED):
                active.append({
                    "id": a["id"],
                    "task_name": a["task_name"],
                    "status": a["status"],
                    "candidates": len(a["candidates"]),
                    "votes": len(a["votes"]),
                    "winner": a.get("winner"),
                    "budget": a["budget"],
                })

        return SkillResult(
            success=True,
            message=f"Active assignments: {len(active)}",
            data={"active": active, "stats": state["stats"]},
        )

    # ─── REPORT OUTCOME ───────────────────────────────────────────────

    async def _report_outcome(self, params: Dict) -> SkillResult:
        """Report task outcome, feeding back into reputation system."""
        assignment_id = params.get("assignment_id", "").strip()
        if not assignment_id:
            return SkillResult(success=False, message="assignment_id is required")

        state = self._load()
        assignment = state["assignments"].get(assignment_id)
        if not assignment:
            return SkillResult(success=False, message=f"Assignment '{assignment_id}' not found")

        if assignment["status"] not in (STATUS_ASSIGNED,):
            return SkillResult(
                success=False,
                message=f"Assignment is '{assignment['status']}', expected 'assigned'",
            )

        success = params.get("success", False)
        quality_score = float(params.get("quality_score", 50))
        budget_spent = float(params.get("budget_spent", 0))
        notes = params.get("notes", "")

        assignment["status"] = STATUS_COMPLETED if success else STATUS_FAILED
        assignment["outcome"] = {
            "success": success,
            "quality_score": quality_score,
            "budget_spent": budget_spent,
            "budget_efficiency": (
                (1 - budget_spent / assignment["budget"]) * 100
                if assignment["budget"] > 0 and budget_spent > 0
                else 0
            ),
            "notes": notes,
            "reported_at": datetime.utcnow().isoformat(),
        }

        if success:
            state["stats"]["total_completed"] += 1
        else:
            state["stats"]["total_failed"] += 1

        # Feed outcome back into reputation via AgentReputationSkill
        winner = assignment.get("winner")
        if winner and self.context:
            try:
                await self.context.call_skill(
                    "agent_reputation", "record_task_outcome",
                    {
                        "agent_id": winner,
                        "success": success,
                        "budget_allocated": assignment["budget"],
                        "budget_spent": budget_spent,
                        "source": "consensus_task_assignment",
                    }
                )
            except Exception:
                pass  # Best-effort

            # Also report completion to TaskDelegation if we have a delegation_id
            if assignment.get("delegation_id"):
                try:
                    await self.context.call_skill(
                        "task_delegation", "report_completion",
                        {
                            "delegation_id": assignment["delegation_id"],
                            "status": "completed" if success else "failed",
                            "result": {"quality_score": quality_score, "notes": notes},
                            "budget_spent": budget_spent,
                        }
                    )
                except Exception:
                    pass

        self._save(state)

        msg = f"Outcome reported for '{assignment['task_name']}': "
        msg += "SUCCESS" if success else "FAILED"
        msg += f" (quality: {quality_score}/100, spent: ${budget_spent:.2f}). "
        msg += "Reputation updated." if winner else ""

        return SkillResult(
            success=True,
            message=msg,
            data={
                "assignment_id": assignment_id,
                "winner": winner,
                "outcome": assignment["outcome"],
            },
        )

    # ─── LEADERBOARD ──────────────────────────────────────────────────

    async def _leaderboard(self, params: Dict) -> SkillResult:
        """Show which agents get assigned tasks most and their success rates."""
        state = self._load()
        limit = int(params.get("limit", 10))

        # Aggregate per-agent stats from completed assignments
        agent_stats = {}
        for a in state["assignments"].values():
            winner = a.get("winner")
            if not winner:
                continue
            if winner not in agent_stats:
                agent_stats[winner] = {
                    "agent_id": winner,
                    "times_assigned": 0,
                    "successes": 0,
                    "failures": 0,
                    "total_budget": 0.0,
                    "total_spent": 0.0,
                    "avg_quality": 0.0,
                    "quality_scores": [],
                }
            stats = agent_stats[winner]
            stats["times_assigned"] += 1
            stats["total_budget"] += a.get("budget", 0)

            outcome = a.get("outcome")
            if outcome:
                if outcome.get("success"):
                    stats["successes"] += 1
                else:
                    stats["failures"] += 1
                stats["total_spent"] += outcome.get("budget_spent", 0)
                stats["quality_scores"].append(outcome.get("quality_score", 50))

        # Compute averages and success rates
        leaderboard = []
        for agent_id, stats in agent_stats.items():
            total_completed = stats["successes"] + stats["failures"]
            success_rate = (stats["successes"] / total_completed * 100) if total_completed > 0 else 0
            avg_quality = (
                sum(stats["quality_scores"]) / len(stats["quality_scores"])
                if stats["quality_scores"]
                else 0
            )
            leaderboard.append({
                "agent_id": agent_id,
                "times_assigned": stats["times_assigned"],
                "successes": stats["successes"],
                "failures": stats["failures"],
                "success_rate": round(success_rate, 1),
                "avg_quality": round(avg_quality, 1),
                "total_budget": stats["total_budget"],
                "total_spent": stats["total_spent"],
            })

        # Sort by times_assigned descending, then success_rate
        leaderboard.sort(key=lambda x: (x["times_assigned"], x["success_rate"]), reverse=True)
        leaderboard = leaderboard[:limit]

        return SkillResult(
            success=True,
            message=f"Assignment leaderboard: {len(leaderboard)} agents",
            data={"leaderboard": leaderboard},
        )

    # ─── HISTORY ──────────────────────────────────────────────────────

    async def _history(self, params: Dict) -> SkillResult:
        """Review past consensus-driven assignments."""
        state = self._load()
        limit = int(params.get("limit", 20))
        status_filter = params.get("status_filter", "").strip()

        entries = []
        for a in state["assignments"].values():
            if status_filter and a["status"] != status_filter:
                continue
            entries.append({
                "id": a["id"],
                "task_name": a["task_name"],
                "status": a["status"],
                "candidates": len(a["candidates"]),
                "votes": len(a["votes"]),
                "winner": a.get("winner"),
                "budget": a["budget"],
                "priority": a.get("priority"),
                "outcome": a.get("outcome"),
                "created_at": a["created_at"],
                "closed_at": a.get("closed_at"),
            })

        # Sort by creation time descending
        entries.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        entries = entries[:limit]

        return SkillResult(
            success=True,
            message=f"Assignment history: {len(entries)} entries",
            data={"history": entries, "stats": state["stats"]},
        )

    # ─── HELPER METHODS ───────────────────────────────────────────────

    async def _get_agent_reputation(self, agent_id: str) -> Dict:
        """Fetch reputation scores for an agent (returns empty dict if unavailable)."""
        if not self.context:
            return {}
        try:
            result = await self.context.call_skill(
                "agent_reputation", "get_reputation",
                {"agent_id": agent_id}
            )
            if result.success:
                return result.data.get("reputation", result.data)
        except Exception:
            pass
        return {}

    async def _compute_vote_weight(self, voter: str, strategy: str) -> float:
        """Compute vote weight based on voting strategy and voter reputation."""
        if strategy == STRATEGY_EQUAL:
            return 1.0

        rep = await self._get_agent_reputation(voter)
        if not rep:
            return 1.0  # Default weight if reputation unavailable

        if strategy == STRATEGY_COMPETENCE_ONLY:
            # Weight based on competence score (0-100 -> 0.5-1.5)
            competence = rep.get("competence", 50.0)
            return 0.5 + (competence / 100.0)

        # STRATEGY_REPUTATION_WEIGHTED (default)
        # Weight based on overall reputation (0-100 -> 0.5-1.5)
        overall = rep.get("overall", 50.0)
        return 0.5 + (overall / 100.0)

    def _update_avg_stats(self, state: Dict):
        """Update running average stats."""
        assignments = state["assignments"]
        if not assignments:
            return

        total_candidates = sum(len(a["candidates"]) for a in assignments.values())
        total_voters = sum(len(a["votes"]) for a in assignments.values())
        count = len(assignments)

        state["stats"]["avg_candidates_per_task"] = round(total_candidates / count, 1)
        state["stats"]["avg_voters_per_task"] = round(total_voters / count, 1)

    async def initialize(self) -> bool:
        self.initialized = True
        return True
