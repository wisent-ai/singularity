#!/usr/bin/env python3
"""
ConsensusProtocolSkill - Multi-agent decision-making for shared resources.

When multiple agents operate in a network, they need to make collective
decisions about shared concerns:
- Which agent handles a customer request (leader election for tasks)
- Whether to scale up/down the agent pool (quorum-based decisions)
- Which strategy to adopt network-wide (proposal voting)
- Resource allocation when demand exceeds capacity (fair allocation)
- Conflict resolution when agents disagree on approach

Without consensus, agents either duplicate work, conflict on resources, or
need a human arbiter. With it, the agent network self-governs.

Supported consensus mechanisms:
1. PROPOSE   - Submit a proposal for the network to vote on
2. VOTE      - Cast a vote on an active proposal (approve/reject/abstain)
3. TALLY     - Count votes and determine outcome based on quorum rules
4. ELECT     - Run a leader election among candidates for a role/task
5. ALLOCATE  - Fair resource allocation using weighted priority scoring
6. RESOLVE   - Conflict resolution via structured negotiation protocol
7. STATUS    - View active proposals, elections, and their current state
8. HISTORY   - Review past decisions and their outcomes

Consensus rules are configurable:
- Simple majority (>50%)
- Supermajority (>66%)
- Unanimous
- Weighted voting (agents get votes proportional to expertise/stake)

Part of the Replication pillar: enables self-governing agent networks
that make collective decisions without human intervention.
"""

import json
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from .base import Skill, SkillResult, SkillManifest, SkillAction


CONSENSUS_FILE = Path(__file__).parent.parent / "data" / "consensus_store.json"

# Proposal statuses
PROPOSAL_OPEN = "open"
PROPOSAL_PASSED = "passed"
PROPOSAL_REJECTED = "rejected"
PROPOSAL_EXPIRED = "expired"
PROPOSAL_CANCELLED = "cancelled"

# Vote options
VOTE_APPROVE = "approve"
VOTE_REJECT = "reject"
VOTE_ABSTAIN = "abstain"

# Quorum rules
QUORUM_SIMPLE = "simple_majority"     # >50%
QUORUM_SUPER = "supermajority"        # >66%
QUORUM_UNANIMOUS = "unanimous"        # 100%
QUORUM_WEIGHTED = "weighted_majority" # weighted >50%

# Election methods
ELECTION_PLURALITY = "plurality"       # Most votes wins
ELECTION_RANKED = "ranked_choice"      # Instant runoff
ELECTION_SCORE = "score"               # Average score wins

# Limits
MAX_PROPOSALS = 200
MAX_ELECTIONS = 100
PROPOSAL_TTL_HOURS = 48
ELECTION_TTL_HOURS = 24


class ConsensusProtocolSkill(Skill):
    """
    Enables multi-agent collective decision-making through structured
    proposals, voting, leader election, and resource allocation.

    This transforms a collection of independent agents into a
    self-governing network that can coordinate on shared concerns.
    """

    def __init__(self):
        super().__init__()
        self._store = None

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="consensus_protocol",
            name="Consensus Protocol",
            version="1.0.0",
            category="replication",
            description="Multi-agent decision-making through proposals, voting, elections, and resource allocation",
            actions=[
                SkillAction(
                    name="propose",
                    description="Submit a proposal for agents to vote on",
                    parameters={
                        "title": {"type": "string", "required": True, "description": "Short title for the proposal"},
                        "description": {"type": "string", "required": True, "description": "Detailed description of what is being proposed"},
                        "proposer": {"type": "string", "required": True, "description": "Agent ID of the proposer"},
                        "category": {"type": "string", "required": False, "description": "Category: strategy, resource, policy, task, scaling"},
                        "quorum_rule": {"type": "string", "required": False, "description": "Voting rule: simple_majority, supermajority, unanimous, weighted_majority"},
                        "min_voters": {"type": "integer", "required": False, "description": "Minimum number of votes required for validity"},
                        "ttl_hours": {"type": "integer", "required": False, "description": "Hours before proposal expires (default 48)"},
                        "options": {"type": "list", "required": False, "description": "Custom vote options beyond approve/reject/abstain"},
                    },
                ),
                SkillAction(
                    name="vote",
                    description="Cast a vote on an active proposal",
                    parameters={
                        "proposal_id": {"type": "string", "required": True, "description": "ID of the proposal to vote on"},
                        "voter": {"type": "string", "required": True, "description": "Agent ID casting the vote"},
                        "choice": {"type": "string", "required": True, "description": "Vote choice: approve, reject, abstain, or custom option"},
                        "weight": {"type": "float", "required": False, "description": "Vote weight (for weighted voting, default 1.0)"},
                        "rationale": {"type": "string", "required": False, "description": "Reason for this vote"},
                    },
                ),
                SkillAction(
                    name="tally",
                    description="Count votes and determine proposal outcome",
                    parameters={
                        "proposal_id": {"type": "string", "required": True, "description": "ID of the proposal to tally"},
                        "force_close": {"type": "boolean", "required": False, "description": "Close voting even if TTL hasn't expired"},
                    },
                ),
                SkillAction(
                    name="elect",
                    description="Run a leader election for a role or task",
                    parameters={
                        "role": {"type": "string", "required": True, "description": "The role or task to elect a leader for"},
                        "candidates": {"type": "list", "required": True, "description": "List of agent IDs as candidates"},
                        "voters": {"type": "list", "required": False, "description": "List of agent IDs who can vote (all candidates if not set)"},
                        "method": {"type": "string", "required": False, "description": "Election method: plurality, ranked_choice, score"},
                        "scores": {"type": "dict", "required": False, "description": "Voter scores for score-based election: {voter: {candidate: score}}"},
                        "rankings": {"type": "dict", "required": False, "description": "Voter rankings for ranked choice: {voter: [ranked_candidates]}"},
                        "votes": {"type": "dict", "required": False, "description": "Voter votes for plurality: {voter: candidate}"},
                    },
                ),
                SkillAction(
                    name="allocate",
                    description="Fair resource allocation among competing agents",
                    parameters={
                        "resource": {"type": "string", "required": True, "description": "What resource to allocate (e.g., 'compute_budget', 'customer_queue')"},
                        "total_amount": {"type": "float", "required": True, "description": "Total amount of resource available"},
                        "requests": {"type": "list", "required": True, "description": "List of {agent_id, requested_amount, priority, justification}"},
                        "method": {"type": "string", "required": False, "description": "Allocation method: proportional, priority_weighted, equal, need_based"},
                    },
                ),
                SkillAction(
                    name="resolve",
                    description="Structured conflict resolution between agents",
                    parameters={
                        "conflict_id": {"type": "string", "required": False, "description": "Existing conflict to continue resolving"},
                        "parties": {"type": "list", "required": False, "description": "Agent IDs involved in the conflict"},
                        "issue": {"type": "string", "required": False, "description": "Description of the conflict"},
                        "positions": {"type": "dict", "required": False, "description": "Each party's position: {agent_id: position_description}"},
                        "resolution": {"type": "string", "required": False, "description": "Proposed resolution to finalize"},
                    },
                ),
                SkillAction(
                    name="status",
                    description="View active proposals, elections, and conflicts",
                    parameters={
                        "filter_status": {"type": "string", "required": False, "description": "Filter by status: open, passed, rejected, expired"},
                        "filter_category": {"type": "string", "required": False, "description": "Filter by category"},
                    },
                ),
                SkillAction(
                    name="history",
                    description="Review past consensus decisions and outcomes",
                    parameters={
                        "limit": {"type": "integer", "required": False, "description": "Max results to return (default 20)"},
                        "include_votes": {"type": "boolean", "required": False, "description": "Include individual vote details"},
                    },
                ),
            ],
            required_credentials=[],
        )

    def _load_store(self) -> Dict:
        """Load consensus store from disk."""
        if self._store is not None:
            return self._store
        if CONSENSUS_FILE.exists():
            try:
                self._store = json.loads(CONSENSUS_FILE.read_text())
            except (json.JSONDecodeError, IOError):
                self._store = self._empty_store()
        else:
            self._store = self._empty_store()
        return self._store

    def _empty_store(self) -> Dict:
        return {
            "proposals": {},
            "elections": {},
            "conflicts": {},
            "allocations": [],
            "stats": {
                "total_proposals": 0,
                "total_elections": 0,
                "total_conflicts": 0,
                "total_allocations": 0,
            },
        }

    def _save_store(self):
        """Persist consensus store to disk."""
        CONSENSUS_FILE.parent.mkdir(parents=True, exist_ok=True)
        CONSENSUS_FILE.write_text(json.dumps(self._store, indent=2, default=str))

    async def execute(self, action: str, params: Dict[str, Any]) -> SkillResult:
        store = self._load_store()
        try:
            if action == "propose":
                return self._propose(store, params)
            elif action == "vote":
                return self._vote(store, params)
            elif action == "tally":
                return self._tally(store, params)
            elif action == "elect":
                return self._elect(store, params)
            elif action == "allocate":
                return self._allocate(store, params)
            elif action == "resolve":
                return self._resolve(store, params)
            elif action == "status":
                return self._status(store, params)
            elif action == "history":
                return self._history(store, params)
            else:
                return SkillResult(success=False, message=f"Unknown action: {action}")
        except Exception as e:
            return SkillResult(success=False, message=f"Error in {action}: {str(e)}")

    # ─── PROPOSE ──────────────────────────────────────────────────────

    def _propose(self, store: Dict, params: Dict) -> SkillResult:
        title = params.get("title")
        description = params.get("description")
        proposer = params.get("proposer")
        if not all([title, description, proposer]):
            return SkillResult(success=False, message="title, description, and proposer are required")

        # Enforce limits
        open_count = sum(1 for p in store["proposals"].values() if p["status"] == PROPOSAL_OPEN)
        if open_count >= MAX_PROPOSALS:
            return SkillResult(success=False, message=f"Too many open proposals ({MAX_PROPOSALS}). Close some first.")

        proposal_id = f"prop-{uuid.uuid4().hex[:8]}"
        ttl = params.get("ttl_hours", PROPOSAL_TTL_HOURS)
        quorum = params.get("quorum_rule", QUORUM_SIMPLE)
        custom_options = params.get("options", [])

        proposal = {
            "id": proposal_id,
            "title": title,
            "description": description,
            "proposer": proposer,
            "category": params.get("category", "general"),
            "quorum_rule": quorum,
            "min_voters": params.get("min_voters", 1),
            "status": PROPOSAL_OPEN,
            "votes": {},
            "custom_options": custom_options,
            "created_at": datetime.utcnow().isoformat(),
            "expires_at": (datetime.utcnow() + timedelta(hours=ttl)).isoformat(),
        }

        store["proposals"][proposal_id] = proposal
        store["stats"]["total_proposals"] += 1
        self._save_store()

        return SkillResult(
            success=True,
            message=f"Proposal '{title}' created. ID: {proposal_id}. "
                    f"Quorum: {quorum}, expires in {ttl}h.",
            data={"proposal_id": proposal_id, "proposal": proposal},
        )

    # ─── VOTE ──────────────────────────────────────────────────────────

    def _vote(self, store: Dict, params: Dict) -> SkillResult:
        proposal_id = params.get("proposal_id")
        voter = params.get("voter")
        choice = params.get("choice")
        if not all([proposal_id, voter, choice]):
            return SkillResult(success=False, message="proposal_id, voter, and choice are required")

        proposal = store["proposals"].get(proposal_id)
        if not proposal:
            return SkillResult(success=False, message=f"Proposal {proposal_id} not found")

        if proposal["status"] != PROPOSAL_OPEN:
            return SkillResult(success=False, message=f"Proposal is {proposal['status']}, not open for voting")

        # Check expiration
        expires = datetime.fromisoformat(proposal["expires_at"])
        if datetime.utcnow() > expires:
            proposal["status"] = PROPOSAL_EXPIRED
            self._save_store()
            return SkillResult(success=False, message="Proposal has expired")

        # Validate choice
        valid_choices = [VOTE_APPROVE, VOTE_REJECT, VOTE_ABSTAIN] + proposal.get("custom_options", [])
        if choice not in valid_choices:
            return SkillResult(success=False, message=f"Invalid choice. Options: {valid_choices}")

        # Check for duplicate vote
        if voter in proposal["votes"]:
            old_choice = proposal["votes"][voter]["choice"]
            proposal["votes"][voter] = {
                "choice": choice,
                "weight": params.get("weight", 1.0),
                "rationale": params.get("rationale", ""),
                "voted_at": datetime.utcnow().isoformat(),
            }
            self._save_store()
            return SkillResult(
                success=True,
                message=f"Vote updated from '{old_choice}' to '{choice}' on '{proposal['title']}'",
                data={"proposal_id": proposal_id, "changed_from": old_choice},
            )

        proposal["votes"][voter] = {
            "choice": choice,
            "weight": params.get("weight", 1.0),
            "rationale": params.get("rationale", ""),
            "voted_at": datetime.utcnow().isoformat(),
        }
        self._save_store()

        vote_count = len(proposal["votes"])
        return SkillResult(
            success=True,
            message=f"Vote '{choice}' recorded on '{proposal['title']}'. "
                    f"Total votes: {vote_count}.",
            data={"proposal_id": proposal_id, "total_votes": vote_count},
        )

    # ─── TALLY ──────────────────────────────────────────────────────────

    def _tally(self, store: Dict, params: Dict) -> SkillResult:
        proposal_id = params.get("proposal_id")
        if not proposal_id:
            return SkillResult(success=False, message="proposal_id is required")

        proposal = store["proposals"].get(proposal_id)
        if not proposal:
            return SkillResult(success=False, message=f"Proposal {proposal_id} not found")

        if proposal["status"] != PROPOSAL_OPEN:
            return SkillResult(
                success=True,
                message=f"Proposal already {proposal['status']}",
                data={"proposal": proposal},
            )

        votes = proposal["votes"]
        force_close = params.get("force_close", False)

        # Check min voters
        non_abstain = {v: d for v, d in votes.items() if d["choice"] != VOTE_ABSTAIN}
        if len(non_abstain) < proposal["min_voters"] and not force_close:
            return SkillResult(
                success=False,
                message=f"Not enough voters. Need {proposal['min_voters']}, have {len(non_abstain)} non-abstain votes.",
                data={"votes_so_far": len(votes), "non_abstain": len(non_abstain)},
            )

        # Calculate result based on quorum rule
        result = self._calculate_result(proposal)
        proposal["status"] = result["status"]
        proposal["result"] = result
        proposal["closed_at"] = datetime.utcnow().isoformat()
        self._save_store()

        return SkillResult(
            success=True,
            message=f"Proposal '{proposal['title']}' {result['status']}. "
                    f"{result['summary']}",
            data={"proposal": proposal, "result": result},
        )

    def _calculate_result(self, proposal: Dict) -> Dict:
        """Calculate voting result based on quorum rule."""
        votes = proposal["votes"]
        quorum = proposal["quorum_rule"]

        # Tally votes by choice (with weights)
        tallies = {}
        total_weight = 0.0
        for voter, vote_data in votes.items():
            choice = vote_data["choice"]
            weight = vote_data.get("weight", 1.0)
            if choice == VOTE_ABSTAIN:
                continue
            tallies[choice] = tallies.get(choice, 0.0) + weight
            total_weight += weight

        approve_weight = tallies.get(VOTE_APPROVE, 0.0)
        reject_weight = tallies.get(VOTE_REJECT, 0.0)

        if total_weight == 0:
            return {
                "status": PROPOSAL_REJECTED,
                "summary": "No non-abstain votes cast",
                "tallies": tallies,
                "approve_pct": 0,
            }

        approve_pct = (approve_weight / total_weight) * 100

        # Determine pass threshold
        if quorum == QUORUM_SIMPLE:
            threshold = 50.0
        elif quorum == QUORUM_SUPER:
            threshold = 66.67
        elif quorum == QUORUM_UNANIMOUS:
            threshold = 100.0
        elif quorum == QUORUM_WEIGHTED:
            threshold = 50.0  # Same as simple but weights matter more
        else:
            threshold = 50.0

        passed = approve_pct > threshold
        status = PROPOSAL_PASSED if passed else PROPOSAL_REJECTED

        return {
            "status": status,
            "summary": f"Approve: {approve_pct:.1f}% (threshold: {threshold}%), "
                       f"votes: {len(votes)}, weighted total: {total_weight:.1f}",
            "tallies": tallies,
            "approve_pct": approve_pct,
            "threshold": threshold,
            "total_weight": total_weight,
        }

    # ─── ELECT ──────────────────────────────────────────────────────────

    def _elect(self, store: Dict, params: Dict) -> SkillResult:
        role = params.get("role")
        candidates = params.get("candidates")
        if not role or not candidates:
            return SkillResult(success=False, message="role and candidates are required")

        method = params.get("method", ELECTION_PLURALITY)
        election_id = f"elect-{uuid.uuid4().hex[:8]}"

        if method == ELECTION_PLURALITY:
            result = self._plurality_election(candidates, params.get("votes", {}))
        elif method == ELECTION_RANKED:
            result = self._ranked_choice_election(candidates, params.get("rankings", {}))
        elif method == ELECTION_SCORE:
            result = self._score_election(candidates, params.get("scores", {}))
        else:
            return SkillResult(success=False, message=f"Unknown election method: {method}")

        election = {
            "id": election_id,
            "role": role,
            "candidates": candidates,
            "method": method,
            "result": result,
            "created_at": datetime.utcnow().isoformat(),
        }

        store["elections"][election_id] = election
        store["stats"]["total_elections"] += 1
        self._save_store()

        winner = result.get("winner")
        return SkillResult(
            success=True,
            message=f"Election for '{role}': winner is '{winner}' via {method}. "
                    f"{result.get('summary', '')}",
            data={"election_id": election_id, "winner": winner, "result": result},
        )

    def _plurality_election(self, candidates: List[str], votes: Dict[str, str]) -> Dict:
        """Simple plurality: most votes wins."""
        tallies = {c: 0 for c in candidates}
        for voter, choice in votes.items():
            if choice in tallies:
                tallies[choice] += 1

        if not votes:
            # No votes - first candidate wins by default
            winner = candidates[0]
            return {"winner": winner, "tallies": tallies, "summary": "No votes cast, first candidate selected by default"}

        winner = max(tallies, key=tallies.get)
        return {
            "winner": winner,
            "tallies": tallies,
            "summary": f"Tallies: {tallies}",
        }

    def _ranked_choice_election(self, candidates: List[str], rankings: Dict[str, List[str]]) -> Dict:
        """Instant-runoff ranked choice voting."""
        if not rankings:
            winner = candidates[0]
            return {"winner": winner, "rounds": [], "summary": "No rankings provided, first candidate selected"}

        remaining = list(candidates)
        rounds = []

        for round_num in range(len(candidates)):
            # Count first-choice votes among remaining candidates
            tallies = {c: 0 for c in remaining}
            for voter, ranking in rankings.items():
                # Find voter's top remaining candidate
                for choice in ranking:
                    if choice in remaining:
                        tallies[choice] += 1
                        break

            total_votes = sum(tallies.values())
            rounds.append({"round": round_num + 1, "tallies": dict(tallies)})

            # Check for majority
            for candidate, count in tallies.items():
                if total_votes > 0 and count > total_votes / 2:
                    return {
                        "winner": candidate,
                        "rounds": rounds,
                        "summary": f"Won in round {round_num + 1} with {count}/{total_votes} votes",
                    }

            # Eliminate lowest
            if remaining:
                lowest = min(remaining, key=lambda c: tallies.get(c, 0))
                remaining.remove(lowest)

            if len(remaining) <= 1:
                break

        winner = remaining[0] if remaining else candidates[0]
        return {
            "winner": winner,
            "rounds": rounds,
            "summary": f"Won after {len(rounds)} rounds of elimination",
        }

    def _score_election(self, candidates: List[str], scores: Dict[str, Dict[str, float]]) -> Dict:
        """Score voting: highest average score wins."""
        if not scores:
            winner = candidates[0]
            return {"winner": winner, "averages": {}, "summary": "No scores provided, first candidate selected"}

        totals = {c: [] for c in candidates}
        for voter, voter_scores in scores.items():
            for candidate, score in voter_scores.items():
                if candidate in totals:
                    totals[candidate].append(score)

        averages = {}
        for candidate, score_list in totals.items():
            averages[candidate] = sum(score_list) / len(score_list) if score_list else 0.0

        winner = max(averages, key=averages.get)
        return {
            "winner": winner,
            "averages": averages,
            "summary": f"Averages: {', '.join(f'{c}: {s:.1f}' for c, s in averages.items())}",
        }

    # ─── ALLOCATE ──────────────────────────────────────────────────────

    def _allocate(self, store: Dict, params: Dict) -> SkillResult:
        resource = params.get("resource")
        total = params.get("total_amount")
        requests = params.get("requests")
        if not resource or total is None or not requests:
            return SkillResult(success=False, message="resource, total_amount, and requests are required")

        method = params.get("method", "proportional")

        if method == "equal":
            allocation = self._allocate_equal(total, requests)
        elif method == "proportional":
            allocation = self._allocate_proportional(total, requests)
        elif method == "priority_weighted":
            allocation = self._allocate_priority(total, requests)
        elif method == "need_based":
            allocation = self._allocate_need_based(total, requests)
        else:
            return SkillResult(success=False, message=f"Unknown allocation method: {method}")

        record = {
            "resource": resource,
            "total_amount": total,
            "method": method,
            "requests": requests,
            "allocation": allocation,
            "created_at": datetime.utcnow().isoformat(),
        }

        store["allocations"].append(record)
        # Keep allocations bounded
        if len(store["allocations"]) > 100:
            store["allocations"] = store["allocations"][-100:]
        store["stats"]["total_allocations"] += 1
        self._save_store()

        return SkillResult(
            success=True,
            message=f"Allocated {total} units of '{resource}' among {len(requests)} agents via {method}.",
            data={"allocation": allocation, "method": method},
        )

    def _allocate_equal(self, total: float, requests: List[Dict]) -> Dict[str, float]:
        """Equal share for each requester."""
        per_agent = total / len(requests) if requests else 0
        return {r["agent_id"]: min(per_agent, r.get("requested_amount", per_agent)) for r in requests}

    def _allocate_proportional(self, total: float, requests: List[Dict]) -> Dict[str, float]:
        """Proportional to requested amount."""
        total_requested = sum(r.get("requested_amount", 0) for r in requests)
        if total_requested == 0:
            return self._allocate_equal(total, requests)
        ratio = min(1.0, total / total_requested)
        return {r["agent_id"]: r.get("requested_amount", 0) * ratio for r in requests}

    def _allocate_priority(self, total: float, requests: List[Dict]) -> Dict[str, float]:
        """Priority-weighted: higher priority agents get filled first."""
        sorted_reqs = sorted(requests, key=lambda r: r.get("priority", 0), reverse=True)
        allocation = {}
        remaining = total
        for req in sorted_reqs:
            amount = min(req.get("requested_amount", 0), remaining)
            allocation[req["agent_id"]] = amount
            remaining -= amount
            if remaining <= 0:
                break
        # Ensure all agents are in the result
        for req in requests:
            if req["agent_id"] not in allocation:
                allocation[req["agent_id"]] = 0
        return allocation

    def _allocate_need_based(self, total: float, requests: List[Dict]) -> Dict[str, float]:
        """Need-based: allocate minimum needs first, then distribute surplus."""
        allocation = {}
        remaining = total

        # Phase 1: Give everyone their minimum (50% of request or available)
        for req in requests:
            minimum = req.get("requested_amount", 0) * 0.5
            amount = min(minimum, remaining)
            allocation[req["agent_id"]] = amount
            remaining -= amount

        # Phase 2: Distribute remaining proportionally to unfilled needs
        if remaining > 0:
            unfilled = {}
            for req in requests:
                needed = req.get("requested_amount", 0) - allocation.get(req["agent_id"], 0)
                if needed > 0:
                    unfilled[req["agent_id"]] = needed
            total_unfilled = sum(unfilled.values())
            if total_unfilled > 0:
                ratio = min(1.0, remaining / total_unfilled)
                for agent_id, needed in unfilled.items():
                    allocation[agent_id] = allocation.get(agent_id, 0) + needed * ratio

        return allocation

    # ─── RESOLVE ──────────────────────────────────────────────────────

    def _resolve(self, store: Dict, params: Dict) -> SkillResult:
        conflict_id = params.get("conflict_id")

        # Continue existing conflict
        if conflict_id:
            conflict = store["conflicts"].get(conflict_id)
            if not conflict:
                return SkillResult(success=False, message=f"Conflict {conflict_id} not found")

            resolution = params.get("resolution")
            if resolution:
                conflict["resolution"] = resolution
                conflict["status"] = "resolved"
                conflict["resolved_at"] = datetime.utcnow().isoformat()
                self._save_store()
                return SkillResult(
                    success=True,
                    message=f"Conflict '{conflict['issue']}' resolved: {resolution}",
                    data={"conflict": conflict},
                )

            # Add new positions
            new_positions = params.get("positions", {})
            if new_positions:
                conflict["positions"].update(new_positions)
                conflict["rounds"] += 1
                self._save_store()
                return SkillResult(
                    success=True,
                    message=f"Round {conflict['rounds']}: positions updated. "
                            f"Parties: {list(conflict['positions'].keys())}",
                    data={"conflict": conflict},
                )

            return SkillResult(success=True, message="Conflict status", data={"conflict": conflict})

        # Create new conflict
        parties = params.get("parties")
        issue = params.get("issue")
        if not parties or not issue:
            return SkillResult(success=False, message="parties and issue required to create a conflict")

        conflict_id = f"conflict-{uuid.uuid4().hex[:8]}"
        conflict = {
            "id": conflict_id,
            "parties": parties,
            "issue": issue,
            "positions": params.get("positions", {}),
            "status": "open",
            "rounds": 1,
            "resolution": None,
            "created_at": datetime.utcnow().isoformat(),
        }

        store["conflicts"][conflict_id] = conflict
        store["stats"]["total_conflicts"] += 1
        self._save_store()

        return SkillResult(
            success=True,
            message=f"Conflict registered: '{issue}' between {parties}. ID: {conflict_id}",
            data={"conflict_id": conflict_id, "conflict": conflict},
        )

    # ─── STATUS ──────────────────────────────────────────────────────

    def _status(self, store: Dict, params: Dict) -> SkillResult:
        filter_status = params.get("filter_status")
        filter_category = params.get("filter_category")

        # Expire old proposals
        now = datetime.utcnow()
        for p in store["proposals"].values():
            if p["status"] == PROPOSAL_OPEN:
                expires = datetime.fromisoformat(p["expires_at"])
                if now > expires:
                    p["status"] = PROPOSAL_EXPIRED

        proposals = list(store["proposals"].values())
        if filter_status:
            proposals = [p for p in proposals if p["status"] == filter_status]
        if filter_category:
            proposals = [p for p in proposals if p.get("category") == filter_category]

        open_conflicts = [c for c in store["conflicts"].values() if c["status"] == "open"]

        summary = {
            "proposals": [
                {
                    "id": p["id"],
                    "title": p["title"],
                    "status": p["status"],
                    "votes": len(p["votes"]),
                    "category": p.get("category"),
                    "proposer": p["proposer"],
                }
                for p in proposals[-20:]
            ],
            "recent_elections": [
                {
                    "id": e["id"],
                    "role": e["role"],
                    "winner": e["result"].get("winner"),
                    "method": e["method"],
                }
                for e in list(store["elections"].values())[-10:]
            ],
            "open_conflicts": [
                {
                    "id": c["id"],
                    "issue": c["issue"],
                    "parties": c["parties"],
                    "rounds": c["rounds"],
                }
                for c in open_conflicts
            ],
            "stats": store["stats"],
        }

        open_proposals = sum(1 for p in store["proposals"].values() if p["status"] == PROPOSAL_OPEN)
        return SkillResult(
            success=True,
            message=f"Active: {open_proposals} open proposals, "
                    f"{len(open_conflicts)} open conflicts, "
                    f"{store['stats']['total_elections']} elections total.",
            data=summary,
        )

    # ─── HISTORY ──────────────────────────────────────────────────────

    def _history(self, store: Dict, params: Dict) -> SkillResult:
        limit = params.get("limit", 20)
        include_votes = params.get("include_votes", False)

        # Collect all decisions with timestamps
        decisions = []

        for p in store["proposals"].values():
            if p["status"] in [PROPOSAL_PASSED, PROPOSAL_REJECTED, PROPOSAL_EXPIRED]:
                entry = {
                    "type": "proposal",
                    "id": p["id"],
                    "title": p["title"],
                    "status": p["status"],
                    "category": p.get("category"),
                    "vote_count": len(p["votes"]),
                    "timestamp": p.get("closed_at", p["created_at"]),
                }
                if include_votes:
                    entry["votes"] = p["votes"]
                if "result" in p:
                    entry["approve_pct"] = p["result"].get("approve_pct", 0)
                decisions.append(entry)

        for e in store["elections"].values():
            decisions.append({
                "type": "election",
                "id": e["id"],
                "role": e["role"],
                "winner": e["result"].get("winner"),
                "method": e["method"],
                "timestamp": e["created_at"],
            })

        for c in store["conflicts"].values():
            if c["status"] == "resolved":
                decisions.append({
                    "type": "conflict_resolution",
                    "id": c["id"],
                    "issue": c["issue"],
                    "resolution": c["resolution"],
                    "rounds": c["rounds"],
                    "timestamp": c.get("resolved_at", c["created_at"]),
                })

        # Sort by timestamp descending
        decisions.sort(key=lambda d: d.get("timestamp", ""), reverse=True)
        decisions = decisions[:limit]

        return SkillResult(
            success=True,
            message=f"Showing {len(decisions)} past decisions.",
            data={"decisions": decisions, "total_in_store": store["stats"]},
        )
