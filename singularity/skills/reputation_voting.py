#!/usr/bin/env python3
"""
ReputationWeightedVotingSkill - Reputation-aware consensus voting.

Integrates AgentReputationSkill into ConsensusProtocolSkill so that vote
weights are automatically computed from agent reputation scores. This means
agents with higher trust/competence have proportionally more influence in
collective decisions.

Key behaviors:
1. CREATE_PROPOSAL  - Create a proposal that uses reputation-weighted voting
2. CAST_VOTE        - Vote on a proposal; weight is auto-computed from reputation
3. TALLY            - Tally votes with reputation-derived weights
4. RUN_ELECTION     - Run election where scores are reputation-weighted
5. GET_VOTER_WEIGHT - Preview what weight a voter would receive
6. CONFIGURE        - Set which reputation dimensions influence vote weight

Without this, all votes are equal regardless of an agent's track record.
With it, agents that consistently deliver good results and behave honestly
carry more influence — creating a meritocratic governance model.

Pillars served:
- Replication: Meritocratic self-governance for agent networks
- Self-Improvement: Good performance → more influence → incentive to improve
"""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

from .base import Skill, SkillResult, SkillManifest, SkillAction

DATA_FILE = Path(__file__).parent.parent / "data" / "reputation_voting.json"

# Default dimension weights for computing vote weight from reputation
DEFAULT_VOTE_DIMENSION_WEIGHTS = {
    "trustworthiness": 0.35,   # Honest voters matter most
    "competence": 0.25,        # Knowledgeable voters
    "reliability": 0.20,       # Consistent participants
    "cooperation": 0.15,       # Team players
    "leadership": 0.05,        # Leadership track record
}

# Vote weight range: minimum and maximum multipliers
MIN_VOTE_WEIGHT = 0.1   # Even lowest-rep agents get some voice
MAX_VOTE_WEIGHT = 3.0   # Highest-rep agents get up to 3x weight

# Category-specific dimension overrides
CATEGORY_DIMENSION_OVERRIDES = {
    "strategy": {
        "competence": 0.35,
        "trustworthiness": 0.30,
        "leadership": 0.20,
        "reliability": 0.10,
        "cooperation": 0.05,
    },
    "resource": {
        "trustworthiness": 0.40,
        "reliability": 0.25,
        "cooperation": 0.20,
        "competence": 0.10,
        "leadership": 0.05,
    },
    "task": {
        "competence": 0.40,
        "reliability": 0.30,
        "trustworthiness": 0.15,
        "cooperation": 0.10,
        "leadership": 0.05,
    },
    "scaling": {
        "leadership": 0.30,
        "competence": 0.25,
        "reliability": 0.20,
        "trustworthiness": 0.15,
        "cooperation": 0.10,
    },
}


def compute_vote_weight(
    reputation: Dict[str, float],
    dimension_weights: Dict[str, float] = None,
    min_weight: float = MIN_VOTE_WEIGHT,
    max_weight: float = MAX_VOTE_WEIGHT,
) -> float:
    """
    Compute a voter's weight from their reputation profile.

    Reputation scores are 0-100, with 50 being neutral.
    Weight is scaled so that:
    - reputation 0  → min_weight
    - reputation 50 → 1.0 (neutral)
    - reputation 100 → max_weight

    Uses linear interpolation in two segments:
    - [0, 50]  → [min_weight, 1.0]
    - [50, 100] → [1.0, max_weight]
    """
    weights = dimension_weights or DEFAULT_VOTE_DIMENSION_WEIGHTS
    total_w = sum(weights.values()) or 1.0

    # Compute weighted reputation score
    weighted_score = 0.0
    for dim, w in weights.items():
        score = reputation.get(dim, 50.0)  # Default to neutral
        weighted_score += (w / total_w) * score

    # Linear interpolation to vote weight
    if weighted_score <= 50.0:
        # [0, 50] → [min_weight, 1.0]
        t = weighted_score / 50.0
        vote_weight = min_weight + t * (1.0 - min_weight)
    else:
        # [50, 100] → [1.0, max_weight]
        t = (weighted_score - 50.0) / 50.0
        vote_weight = 1.0 + t * (max_weight - 1.0)

    return round(vote_weight, 3)


class ReputationWeightedVotingSkill(Skill):
    """
    Wraps ConsensusProtocolSkill with automatic reputation-based vote weighting.

    When an agent casts a vote through this skill, their reputation is looked
    up from AgentReputationSkill and converted into a vote weight. This creates
    meritocratic governance where trusted, competent agents have more influence.
    """

    def __init__(self, credentials: Dict[str, str] = None):
        super().__init__(credentials)
        self._store = None

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="reputation_voting",
            name="Reputation Weighted Voting",
            version="1.0.0",
            category="replication",
            description=(
                "Reputation-aware consensus voting: vote weights are automatically "
                "derived from agent reputation scores for meritocratic governance"
            ),
            actions=[
                SkillAction(
                    name="create_proposal",
                    description="Create a reputation-weighted proposal for agents to vote on",
                    parameters={
                        "title": {"type": "string", "required": True, "description": "Short title for the proposal"},
                        "description": {"type": "string", "required": True, "description": "Detailed description"},
                        "proposer": {"type": "string", "required": True, "description": "Agent ID of the proposer"},
                        "category": {"type": "string", "required": False, "description": "Category: strategy, resource, task, scaling, general"},
                        "quorum_rule": {"type": "string", "required": False, "description": "simple_majority, supermajority, unanimous, weighted_majority"},
                        "min_voters": {"type": "integer", "required": False, "description": "Minimum number of votes required"},
                        "ttl_hours": {"type": "integer", "required": False, "description": "Hours before proposal expires"},
                    },
                ),
                SkillAction(
                    name="cast_vote",
                    description="Vote on a proposal with auto-computed reputation weight",
                    parameters={
                        "proposal_id": {"type": "string", "required": True, "description": "ID of the proposal"},
                        "voter": {"type": "string", "required": True, "description": "Agent ID casting the vote"},
                        "choice": {"type": "string", "required": True, "description": "approve, reject, abstain"},
                        "rationale": {"type": "string", "required": False, "description": "Reason for this vote"},
                    },
                ),
                SkillAction(
                    name="tally",
                    description="Tally votes with reputation-derived weights",
                    parameters={
                        "proposal_id": {"type": "string", "required": True, "description": "ID of the proposal to tally"},
                        "force_close": {"type": "boolean", "required": False, "description": "Close even if TTL hasn't expired"},
                    },
                ),
                SkillAction(
                    name="run_election",
                    description="Run election with reputation-weighted scoring",
                    parameters={
                        "role": {"type": "string", "required": True, "description": "Role or task to elect for"},
                        "candidates": {"type": "list", "required": True, "description": "List of candidate agent IDs"},
                        "voters": {"type": "dict", "required": True, "description": "Voter preferences: {voter_id: candidate_id}"},
                    },
                ),
                SkillAction(
                    name="get_voter_weight",
                    description="Preview what vote weight an agent would receive",
                    parameters={
                        "agent_id": {"type": "string", "required": True, "description": "Agent ID to check"},
                        "category": {"type": "string", "required": False, "description": "Proposal category for context-specific weights"},
                    },
                ),
                SkillAction(
                    name="configure",
                    description="Configure reputation dimension weights for voting",
                    parameters={
                        "dimension_weights": {"type": "dict", "required": False, "description": "Override dimension weights: {dimension: weight}"},
                        "min_weight": {"type": "float", "required": False, "description": "Minimum vote weight (default 0.1)"},
                        "max_weight": {"type": "float", "required": False, "description": "Maximum vote weight (default 3.0)"},
                    },
                ),
            ],
            required_credentials=[],
        )

    # ─── State Management ──────────────────────────────────────────

    def _load_store(self) -> Dict:
        if self._store is not None:
            return self._store
        if DATA_FILE.exists():
            try:
                self._store = json.loads(DATA_FILE.read_text())
            except (json.JSONDecodeError, IOError):
                self._store = self._default_store()
        else:
            self._store = self._default_store()
        return self._store

    def _default_store(self) -> Dict:
        return {
            "proposals": {},
            "elections": [],
            "config": {
                "dimension_weights": dict(DEFAULT_VOTE_DIMENSION_WEIGHTS),
                "min_weight": MIN_VOTE_WEIGHT,
                "max_weight": MAX_VOTE_WEIGHT,
            },
            "stats": {
                "total_proposals": 0,
                "total_votes": 0,
                "total_elections": 0,
                "avg_vote_weight": 0.0,
                "weight_sum": 0.0,
            },
        }

    def _save_store(self):
        DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
        DATA_FILE.write_text(json.dumps(self._store, indent=2, default=str))

    # ─── Reputation Lookup ─────────────────────────────────────────

    def _get_agent_reputation(self, agent_id: str) -> Dict[str, float]:
        """
        Look up an agent's reputation via AgentReputationSkill.

        Uses SkillContext for cross-skill communication if available.
        Falls back to neutral reputation (50.0 for all dimensions) if
        the reputation skill is not available.
        """
        # Try to load AgentReputationSkill directly from data file
        rep_file = Path(__file__).parent.parent / "data" / "agent_reputation.json"
        if rep_file.exists():
            try:
                data = json.loads(rep_file.read_text())
                agents = data.get("agents", {})
                if agent_id in agents:
                    agent = agents[agent_id]
                    return {
                        "competence": agent.get("competence", 50.0),
                        "reliability": agent.get("reliability", 50.0),
                        "trustworthiness": agent.get("trustworthiness", 50.0),
                        "leadership": agent.get("leadership", 50.0),
                        "cooperation": agent.get("cooperation", 50.0),
                        "overall": agent.get("overall", 50.0),
                    }
            except (json.JSONDecodeError, IOError):
                pass

        # No reputation data found — return neutral
        return {
            "competence": 50.0,
            "reliability": 50.0,
            "trustworthiness": 50.0,
            "leadership": 50.0,
            "cooperation": 50.0,
            "overall": 50.0,
        }

    def _compute_weight(self, agent_id: str, category: str = "general") -> float:
        """Compute vote weight for an agent based on their reputation."""
        reputation = self._get_agent_reputation(agent_id)

        # Use category-specific dimension weights if available
        dim_weights = CATEGORY_DIMENSION_OVERRIDES.get(
            category, self._load_store()["config"]["dimension_weights"]
        )
        config = self._load_store()["config"]

        return compute_vote_weight(
            reputation,
            dimension_weights=dim_weights,
            min_weight=config["min_weight"],
            max_weight=config["max_weight"],
        )

    # ─── Execute ───────────────────────────────────────────────────

    async def execute(self, action: str, params: Dict[str, Any]) -> SkillResult:
        self._load_store()
        try:
            if action == "create_proposal":
                return self._create_proposal(params)
            elif action == "cast_vote":
                return self._cast_vote(params)
            elif action == "tally":
                return self._tally(params)
            elif action == "run_election":
                return self._run_election(params)
            elif action == "get_voter_weight":
                return self._get_voter_weight(params)
            elif action == "configure":
                return self._configure(params)
            else:
                return SkillResult(success=False, message=f"Unknown action: {action}")
        except Exception as e:
            return SkillResult(success=False, message=f"Error in {action}: {str(e)}")

    # ─── CREATE_PROPOSAL ───────────────────────────────────────────

    def _create_proposal(self, params: Dict) -> SkillResult:
        title = params.get("title")
        description = params.get("description")
        proposer = params.get("proposer")
        if not all([title, description, proposer]):
            return SkillResult(success=False, message="title, description, and proposer are required")

        proposal_id = f"rwv-{uuid.uuid4().hex[:8]}"
        category = params.get("category", "general")
        quorum_rule = params.get("quorum_rule", "weighted_majority")
        min_voters = params.get("min_voters", 1)
        ttl_hours = params.get("ttl_hours", 48)

        # Look up proposer's reputation weight
        proposer_weight = self._compute_weight(proposer, category)

        from datetime import timedelta
        expires_at = (datetime.utcnow() + timedelta(hours=ttl_hours)).isoformat()

        proposal = {
            "id": proposal_id,
            "title": title,
            "description": description,
            "proposer": proposer,
            "proposer_weight": proposer_weight,
            "category": category,
            "quorum_rule": quorum_rule,
            "min_voters": min_voters,
            "status": "open",
            "votes": {},
            "reputation_weighted": True,
            "created_at": datetime.utcnow().isoformat(),
            "expires_at": expires_at,
        }

        store = self._load_store()
        store["proposals"][proposal_id] = proposal
        store["stats"]["total_proposals"] += 1
        self._save_store()

        return SkillResult(
            success=True,
            message=(
                f"Reputation-weighted proposal '{title}' created. "
                f"ID: {proposal_id}. Category: {category}. "
                f"Proposer weight: {proposer_weight:.2f}x."
            ),
            data={"proposal_id": proposal_id, "proposal": proposal},
        )

    # ─── CAST_VOTE ─────────────────────────────────────────────────

    def _cast_vote(self, params: Dict) -> SkillResult:
        proposal_id = params.get("proposal_id")
        voter = params.get("voter")
        choice = params.get("choice")
        if not all([proposal_id, voter, choice]):
            return SkillResult(success=False, message="proposal_id, voter, and choice are required")

        store = self._load_store()
        proposal = store["proposals"].get(proposal_id)
        if not proposal:
            return SkillResult(success=False, message=f"Proposal {proposal_id} not found")
        if proposal["status"] != "open":
            return SkillResult(success=False, message=f"Proposal is {proposal['status']}, not open")

        # Check expiration
        expires = datetime.fromisoformat(proposal["expires_at"])
        if datetime.utcnow() > expires:
            proposal["status"] = "expired"
            self._save_store()
            return SkillResult(success=False, message="Proposal has expired")

        valid_choices = ["approve", "reject", "abstain"]
        if choice not in valid_choices:
            return SkillResult(success=False, message=f"Invalid choice. Options: {valid_choices}")

        # Auto-compute reputation weight
        category = proposal.get("category", "general")
        weight = self._compute_weight(voter, category)

        old_choice = None
        if voter in proposal["votes"]:
            old_choice = proposal["votes"][voter]["choice"]

        proposal["votes"][voter] = {
            "choice": choice,
            "weight": weight,
            "reputation_derived": True,
            "rationale": params.get("rationale", ""),
            "voted_at": datetime.utcnow().isoformat(),
        }

        # Update stats
        store["stats"]["total_votes"] += 1
        store["stats"]["weight_sum"] += weight
        store["stats"]["avg_vote_weight"] = (
            store["stats"]["weight_sum"] / store["stats"]["total_votes"]
        )
        self._save_store()

        msg = (
            f"Vote '{choice}' recorded with reputation weight {weight:.2f}x "
            f"on '{proposal['title']}'. Total votes: {len(proposal['votes'])}."
        )
        if old_choice:
            msg = f"Vote changed from '{old_choice}' to '{choice}' (weight {weight:.2f}x)."

        return SkillResult(
            success=True,
            message=msg,
            data={
                "proposal_id": proposal_id,
                "voter": voter,
                "weight": weight,
                "total_votes": len(proposal["votes"]),
            },
        )

    # ─── TALLY ─────────────────────────────────────────────────────

    def _tally(self, params: Dict) -> SkillResult:
        proposal_id = params.get("proposal_id")
        if not proposal_id:
            return SkillResult(success=False, message="proposal_id is required")

        store = self._load_store()
        proposal = store["proposals"].get(proposal_id)
        if not proposal:
            return SkillResult(success=False, message=f"Proposal {proposal_id} not found")

        if proposal["status"] != "open":
            return SkillResult(
                success=True,
                message=f"Proposal already {proposal['status']}",
                data={"proposal": proposal},
            )

        force_close = params.get("force_close", False)
        votes = proposal["votes"]

        # Check min voters
        non_abstain = {v: d for v, d in votes.items() if d["choice"] != "abstain"}
        if len(non_abstain) < proposal["min_voters"] and not force_close:
            return SkillResult(
                success=False,
                message=f"Not enough voters. Need {proposal['min_voters']}, have {len(non_abstain)}.",
            )

        # Tally with reputation weights
        tallies = {}  # choice -> total weighted score
        total_weight = 0.0
        voter_details = []

        for voter_id, vote_data in votes.items():
            choice = vote_data["choice"]
            weight = vote_data.get("weight", 1.0)
            if choice == "abstain":
                continue
            tallies[choice] = tallies.get(choice, 0.0) + weight
            total_weight += weight
            voter_details.append({
                "voter": voter_id,
                "choice": choice,
                "weight": weight,
            })

        approve_weight = tallies.get("approve", 0.0)

        if total_weight == 0:
            result = {
                "status": "rejected",
                "summary": "No non-abstain votes cast",
                "tallies": tallies,
                "approve_pct": 0,
                "total_weight": 0,
            }
        else:
            approve_pct = (approve_weight / total_weight) * 100.0

            quorum = proposal.get("quorum_rule", "weighted_majority")
            if quorum == "simple_majority":
                threshold = 50.0
            elif quorum == "supermajority":
                threshold = 66.67
            elif quorum == "unanimous":
                threshold = 100.0
            else:  # weighted_majority
                threshold = 50.0

            passed = approve_pct > threshold
            status = "passed" if passed else "rejected"

            result = {
                "status": status,
                "summary": (
                    f"Approve: {approve_pct:.1f}% (threshold: {threshold}%), "
                    f"total weighted votes: {total_weight:.2f}, "
                    f"voter count: {len(votes)}"
                ),
                "tallies": tallies,
                "approve_pct": approve_pct,
                "threshold": threshold,
                "total_weight": total_weight,
                "voter_details": voter_details,
            }

        proposal["status"] = result["status"]
        proposal["result"] = result
        proposal["closed_at"] = datetime.utcnow().isoformat()
        self._save_store()

        return SkillResult(
            success=True,
            message=f"Proposal '{proposal['title']}' {result['status']}. {result['summary']}",
            data={"proposal": proposal, "result": result},
        )

    # ─── RUN_ELECTION ──────────────────────────────────────────────

    def _run_election(self, params: Dict) -> SkillResult:
        role = params.get("role")
        candidates = params.get("candidates")
        voters = params.get("voters")  # {voter_id: candidate_id}
        if not role or not candidates or not voters:
            return SkillResult(success=False, message="role, candidates, and voters are required")

        election_id = f"rwelect-{uuid.uuid4().hex[:8]}"

        # Compute reputation-weighted tallies
        candidate_scores = {c: 0.0 for c in candidates}
        voter_weights = {}

        for voter_id, preferred_candidate in voters.items():
            weight = self._compute_weight(voter_id)
            voter_weights[voter_id] = weight

            if preferred_candidate in candidate_scores:
                candidate_scores[preferred_candidate] += weight

        if not candidate_scores:
            return SkillResult(success=False, message="No valid candidates")

        winner = max(candidate_scores, key=candidate_scores.get)
        total_weight = sum(candidate_scores.values())

        election = {
            "id": election_id,
            "role": role,
            "candidates": candidates,
            "voter_count": len(voters),
            "voter_weights": voter_weights,
            "candidate_scores": candidate_scores,
            "winner": winner,
            "total_weight": total_weight,
            "reputation_weighted": True,
            "created_at": datetime.utcnow().isoformat(),
        }

        store = self._load_store()
        store["elections"].append(election)
        store["stats"]["total_elections"] += 1
        self._save_store()

        scores_str = ", ".join(
            f"{c}: {s:.2f}" for c, s in sorted(
                candidate_scores.items(), key=lambda x: -x[1]
            )
        )

        return SkillResult(
            success=True,
            message=(
                f"Reputation-weighted election for '{role}': "
                f"winner is '{winner}' with weighted score {candidate_scores[winner]:.2f}. "
                f"Scores: {scores_str}"
            ),
            data={
                "election_id": election_id,
                "winner": winner,
                "candidate_scores": candidate_scores,
                "voter_weights": voter_weights,
            },
        )

    # ─── GET_VOTER_WEIGHT ──────────────────────────────────────────

    def _get_voter_weight(self, params: Dict) -> SkillResult:
        agent_id = params.get("agent_id")
        if not agent_id:
            return SkillResult(success=False, message="agent_id is required")

        category = params.get("category", "general")
        reputation = self._get_agent_reputation(agent_id)
        weight = self._compute_weight(agent_id, category)

        # Get dimension weights used
        dim_weights = CATEGORY_DIMENSION_OVERRIDES.get(
            category, self._load_store()["config"]["dimension_weights"]
        )

        return SkillResult(
            success=True,
            message=f"Agent '{agent_id}' vote weight: {weight:.3f}x (category: {category})",
            data={
                "agent_id": agent_id,
                "weight": weight,
                "reputation": reputation,
                "dimension_weights_used": dim_weights,
                "category": category,
            },
        )

    # ─── CONFIGURE ─────────────────────────────────────────────────

    def _configure(self, params: Dict) -> SkillResult:
        store = self._load_store()
        config = store["config"]

        changes = []
        if "dimension_weights" in params:
            new_weights = params["dimension_weights"]
            # Validate all dimensions exist
            valid_dims = {"competence", "reliability", "trustworthiness", "leadership", "cooperation"}
            for dim in new_weights:
                if dim not in valid_dims:
                    return SkillResult(success=False, message=f"Invalid dimension: {dim}. Valid: {valid_dims}")
            config["dimension_weights"] = new_weights
            changes.append(f"dimension_weights={new_weights}")

        if "min_weight" in params:
            val = float(params["min_weight"])
            if val < 0 or val > 1.0:
                return SkillResult(success=False, message="min_weight must be between 0 and 1.0")
            config["min_weight"] = val
            changes.append(f"min_weight={val}")

        if "max_weight" in params:
            val = float(params["max_weight"])
            if val < 1.0 or val > 10.0:
                return SkillResult(success=False, message="max_weight must be between 1.0 and 10.0")
            config["max_weight"] = val
            changes.append(f"max_weight={val}")

        if not changes:
            return SkillResult(
                success=True,
                message="No configuration changes specified.",
                data={"config": config},
            )

        self._save_store()

        return SkillResult(
            success=True,
            message=f"Configuration updated: {', '.join(changes)}",
            data={"config": config},
        )
