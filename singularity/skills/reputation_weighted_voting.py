#!/usr/bin/env python3
"""
ReputationWeightedVotingSkill - Automatically weight consensus votes by agent reputation.

Currently, ConsensusProtocolSkill accepts a manual `weight` parameter on votes,
and AgentReputationSkill tracks agent trustworthiness/competence scores separately.
This skill bridges the gap: it automatically looks up an agent's reputation when
they vote, computes a reputation-based weight, and casts the vote with that weight.

It also closes the feedback loop: when proposals are tallied, it records vote
participation and outcome correctness back into AgentReputationSkill.

This transforms equal-weight "one agent, one vote" into a meritocratic system
where more trusted/competent agents have proportionally greater influence.

Actions:
1. VOTE         - Cast a reputation-weighted vote on a proposal
2. ELECT        - Run a reputation-weighted election
3. TALLY        - Tally a proposal AND record reputation feedback
4. CONFIGURE    - Set weight formula, dimension preferences, min/max weights
5. SIMULATE     - Preview how reputation would affect vote weights without casting
6. AUDIT        - View how reputation affected a past proposal's outcome

Pillar: Replication (self-governing agent networks with trust-based voting)
"""

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from singularity.skills.base import Skill, SkillAction, SkillManifest, SkillResult

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
CONFIG_FILE = os.path.join(DATA_DIR, "reputation_voting_config.json")
AUDIT_FILE = os.path.join(DATA_DIR, "reputation_voting_audit.json")

# Default configuration
DEFAULT_CONFIG = {
    # Which reputation dimensions factor into vote weight, and how much
    "dimension_weights": {
        "trustworthiness": 0.40,
        "competence": 0.30,
        "cooperation": 0.20,
        "leadership": 0.10,
    },
    # Weight bounds: prevent any agent from having outsized influence
    "min_weight": 0.3,
    "max_weight": 3.0,
    # Neutral reputation score (agents at this level get weight 1.0)
    "neutral_score": 50.0,
    # How aggressively reputation affects weight (higher = more spread)
    "sensitivity": 2.0,
    # Whether to auto-record reputation feedback on tally
    "auto_feedback": True,
    # Category-specific dimension overrides (e.g., strategy proposals weight leadership more)
    "category_overrides": {
        "strategy": {"leadership": 0.35, "competence": 0.30, "trustworthiness": 0.25, "cooperation": 0.10},
        "resource": {"competence": 0.40, "trustworthiness": 0.30, "cooperation": 0.20, "leadership": 0.10},
        "policy": {"trustworthiness": 0.45, "cooperation": 0.25, "competence": 0.20, "leadership": 0.10},
    },
}


class ReputationWeightedVotingSkill(Skill):
    """
    Bridges AgentReputationSkill and ConsensusProtocolSkill to enable
    automatic reputation-weighted voting.
    """

    def __init__(self, credentials: Dict[str, str] = None):
        super().__init__(credentials)
        self._config = None
        self._audit_log = None

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="reputation_weighted_voting",
            name="Reputation-Weighted Voting",
            version="1.0.0",
            category="replication",
            description="Automatically weight consensus votes by agent reputation scores, enabling meritocratic self-governance",
            actions=[
                SkillAction(
                    name="vote",
                    description="Cast a reputation-weighted vote on a consensus proposal",
                    parameters={
                        "proposal_id": {"type": "string", "required": True, "description": "ID of the proposal to vote on"},
                        "voter": {"type": "string", "required": True, "description": "Agent ID casting the vote"},
                        "choice": {"type": "string", "required": True, "description": "Vote choice: approve, reject, abstain"},
                        "rationale": {"type": "string", "required": False, "description": "Reason for this vote"},
                        "override_weight": {"type": "float", "required": False, "description": "Manual weight override (skips reputation lookup)"},
                    },
                ),
                SkillAction(
                    name="elect",
                    description="Run a reputation-weighted election where candidate scores include reputation",
                    parameters={
                        "role": {"type": "string", "required": True, "description": "Role or task to elect a leader for"},
                        "candidates": {"type": "list", "required": True, "description": "List of candidate agent IDs"},
                        "voters": {"type": "dict", "required": False, "description": "Manual votes {voter: candidate} - reputation weights added automatically"},
                        "method": {"type": "string", "required": False, "description": "Election method: plurality, score (default: score)"},
                    },
                ),
                SkillAction(
                    name="tally",
                    description="Tally a proposal and auto-record reputation feedback for voters",
                    parameters={
                        "proposal_id": {"type": "string", "required": True, "description": "ID of the proposal to tally"},
                        "force_close": {"type": "boolean", "required": False, "description": "Close voting even if TTL hasn't expired"},
                    },
                ),
                SkillAction(
                    name="configure",
                    description="Set reputation-weight configuration (dimension weights, sensitivity, bounds)",
                    parameters={
                        "dimension_weights": {"type": "dict", "required": False, "description": "Dimension weight map, e.g. {trustworthiness: 0.4, competence: 0.3}"},
                        "min_weight": {"type": "float", "required": False, "description": "Minimum vote weight (default 0.3)"},
                        "max_weight": {"type": "float", "required": False, "description": "Maximum vote weight (default 3.0)"},
                        "sensitivity": {"type": "float", "required": False, "description": "How aggressively reputation affects weight (default 2.0)"},
                        "auto_feedback": {"type": "boolean", "required": False, "description": "Auto-record reputation feedback on tally"},
                        "category_overrides": {"type": "dict", "required": False, "description": "Category-specific dimension weight overrides"},
                    },
                ),
                SkillAction(
                    name="simulate",
                    description="Preview how reputation would weight votes for a set of agents",
                    parameters={
                        "agent_ids": {"type": "list", "required": True, "description": "List of agent IDs to simulate weights for"},
                        "category": {"type": "string", "required": False, "description": "Proposal category for dimension overrides"},
                    },
                ),
                SkillAction(
                    name="audit",
                    description="View how reputation affected a past proposal's outcome",
                    parameters={
                        "proposal_id": {"type": "string", "required": False, "description": "Specific proposal to audit"},
                        "limit": {"type": "integer", "required": False, "description": "Number of recent audits to return (default 10)"},
                    },
                ),
            ],
            required_credentials=[],
        )

    def _load_config(self) -> Dict:
        if self._config is not None:
            return self._config
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE) as f:
                    self._config = json.load(f)
            except (json.JSONDecodeError, IOError):
                self._config = dict(DEFAULT_CONFIG)
        else:
            self._config = dict(DEFAULT_CONFIG)
        return self._config

    def _save_config(self):
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(CONFIG_FILE, "w") as f:
            json.dump(self._config, f, indent=2)

    def _load_audit(self) -> List[Dict]:
        if self._audit_log is not None:
            return self._audit_log
        if os.path.exists(AUDIT_FILE):
            try:
                with open(AUDIT_FILE) as f:
                    self._audit_log = json.load(f)
            except (json.JSONDecodeError, IOError):
                self._audit_log = []
        else:
            self._audit_log = []
        return self._audit_log

    def _save_audit(self):
        os.makedirs(DATA_DIR, exist_ok=True)
        # Keep last 500 entries
        self._audit_log = self._audit_log[-500:]
        with open(AUDIT_FILE, "w") as f:
            json.dump(self._audit_log, f, indent=2)

    def _get_reputation_skill(self):
        """Get AgentReputationSkill from context or create standalone."""
        if self.context:
            rep_skill = self.context.get_skill("agent_reputation")
            if rep_skill:
                return rep_skill

        # Standalone fallback
        from singularity.skills.agent_reputation import AgentReputationSkill
        return AgentReputationSkill()

    def _get_consensus_skill(self):
        """Get ConsensusProtocolSkill from context or create standalone."""
        if self.context:
            cons_skill = self.context.get_skill("consensus_protocol")
            if cons_skill:
                return cons_skill

        from singularity.skills.consensus import ConsensusProtocolSkill
        return ConsensusProtocolSkill()

    def _compute_vote_weight(self, reputation_data: Dict, category: str = None) -> float:
        """
        Compute vote weight from reputation scores.

        Formula: weight = (weighted_score / neutral_score) * sensitivity
        Clamped to [min_weight, max_weight]
        """
        config = self._load_config()
        neutral = config["neutral_score"]
        sensitivity = config["sensitivity"]
        min_w = config["min_weight"]
        max_w = config["max_weight"]

        # Use category-specific dimension weights if available
        dim_weights = config.get("category_overrides", {}).get(category, config["dimension_weights"])

        # Compute weighted reputation score
        weighted_score = 0.0
        total_dim_weight = 0.0
        for dim, w in dim_weights.items():
            score = reputation_data.get(dim, neutral)
            weighted_score += score * w
            total_dim_weight += w

        if total_dim_weight > 0:
            weighted_score /= total_dim_weight

        # Convert to vote weight: agents at neutral get 1.0
        # sensitivity controls how much deviation from neutral matters
        if neutral > 0:
            raw_weight = (weighted_score / neutral) ** (sensitivity / 2.0)
        else:
            raw_weight = 1.0

        return max(min_w, min(max_w, raw_weight))

    async def execute(self, action: str, params: Dict[str, Any]) -> SkillResult:
        handlers = {
            "vote": self._vote,
            "elect": self._elect,
            "tally": self._tally,
            "configure": self._configure,
            "simulate": self._simulate,
            "audit": self._audit,
        }
        handler = handlers.get(action)
        if not handler:
            return SkillResult(success=False, message=f"Unknown action: {action}")
        try:
            return await handler(params)
        except Exception as e:
            return SkillResult(success=False, message=f"Error in {action}: {e}")

    async def _vote(self, params: Dict) -> SkillResult:
        """Cast a reputation-weighted vote on a consensus proposal."""
        proposal_id = params.get("proposal_id")
        voter = params.get("voter")
        choice = params.get("choice")

        if not all([proposal_id, voter, choice]):
            return SkillResult(success=False, message="proposal_id, voter, and choice are required")

        override_weight = params.get("override_weight")

        if override_weight is not None:
            # Manual override
            weight = float(override_weight)
            reputation_data = {}
            weight_source = "manual_override"
        else:
            # Look up reputation
            rep_skill = self._get_reputation_skill()
            rep_result = await rep_skill.execute("get_reputation", {"agent_id": voter})

            if rep_result.success:
                reputation_data = rep_result.data or {}
            else:
                reputation_data = {}

            # Determine proposal category for dimension override
            cons_skill = self._get_consensus_skill()
            store = cons_skill._load_store()
            proposal = store.get("proposals", {}).get(proposal_id, {})
            category = proposal.get("category", "general")

            weight = self._compute_vote_weight(reputation_data, category)
            weight_source = "reputation"

        # Cast the vote via ConsensusProtocolSkill
        cons_skill = self._get_consensus_skill()
        vote_result = await cons_skill.execute("vote", {
            "proposal_id": proposal_id,
            "voter": voter,
            "choice": choice,
            "weight": weight,
            "rationale": params.get("rationale", ""),
        })

        if not vote_result.success:
            return vote_result

        # Record in audit log
        audit_entry = {
            "type": "vote",
            "proposal_id": proposal_id,
            "voter": voter,
            "choice": choice,
            "weight": round(weight, 3),
            "weight_source": weight_source,
            "reputation_snapshot": {
                k: round(v, 1) if isinstance(v, (int, float)) else v
                for k, v in reputation_data.items()
                if k in ["competence", "reliability", "trustworthiness", "leadership", "cooperation", "overall"]
            },
            "timestamp": datetime.utcnow().isoformat(),
        }
        audit_log = self._load_audit()
        audit_log.append(audit_entry)
        self._save_audit()

        return SkillResult(
            success=True,
            message=f"Reputation-weighted vote: {voter} voted '{choice}' on {proposal_id} "
                    f"with weight {weight:.2f} (source: {weight_source})",
            data={
                "proposal_id": proposal_id,
                "voter": voter,
                "choice": choice,
                "weight": round(weight, 3),
                "weight_source": weight_source,
                "reputation_snapshot": audit_entry["reputation_snapshot"],
            },
        )

    async def _elect(self, params: Dict) -> SkillResult:
        """Run a reputation-weighted election."""
        role = params.get("role")
        candidates = params.get("candidates")
        if not role or not candidates:
            return SkillResult(success=False, message="role and candidates are required")

        method = params.get("method", "score")
        manual_votes = params.get("voters", {})

        rep_skill = self._get_reputation_skill()
        cons_skill = self._get_consensus_skill()

        if method == "score":
            # Build reputation-based scores for each candidate
            # Each "voter" is a reputation dimension acting as an evaluator
            scores = {}
            candidate_reps = {}

            for candidate in candidates:
                rep_result = await rep_skill.execute("get_reputation", {"agent_id": candidate})
                if rep_result.success:
                    candidate_reps[candidate] = rep_result.data or {}
                else:
                    candidate_reps[candidate] = {}

            # Create a synthetic "reputation_evaluator" voter whose scores are the reputation
            config = self._load_config()
            dim_weights = config["dimension_weights"]

            scores["reputation_evaluator"] = {}
            for candidate in candidates:
                rep_data = candidate_reps.get(candidate, {})
                weighted = 0.0
                total_w = 0.0
                for dim, w in dim_weights.items():
                    val = rep_data.get(dim, 50.0)
                    weighted += val * w
                    total_w += w
                score = (weighted / total_w) if total_w > 0 else 50.0
                scores["reputation_evaluator"][candidate] = round(score, 1)

            # Add manual voter scores with reputation weighting
            for voter, candidate in manual_votes.items():
                # Give voted candidate a high score, others low
                voter_rep = await rep_skill.execute("get_reputation", {"agent_id": voter})
                voter_data = voter_rep.data if voter_rep.success else {}
                vote_weight = self._compute_vote_weight(voter_data)

                scores[voter] = {}
                for c in candidates:
                    if c == candidate:
                        scores[voter][c] = 100.0 * vote_weight
                    else:
                        scores[voter][c] = 0.0

            # Run score election via consensus
            result = await cons_skill.execute("elect", {
                "role": role,
                "candidates": candidates,
                "method": "score",
                "scores": scores,
            })

            # Audit
            audit_entry = {
                "type": "election",
                "role": role,
                "candidates": candidates,
                "method": method,
                "candidate_reputations": {
                    c: {k: round(v, 1) if isinstance(v, (int, float)) else v
                        for k, v in candidate_reps.get(c, {}).items()
                        if k in ["competence", "trustworthiness", "cooperation", "leadership", "overall"]}
                    for c in candidates
                },
                "winner": result.data.get("winner") if result.data else None,
                "timestamp": datetime.utcnow().isoformat(),
            }
            audit_log = self._load_audit()
            audit_log.append(audit_entry)
            self._save_audit()

            return result

        elif method == "plurality":
            # Weight each voter's vote by their reputation
            # For plurality, we convert to score: voted candidate gets reputation-weight, others get 0
            if not manual_votes:
                return SkillResult(success=False, message="voters dict required for plurality election")

            scores = {}
            for voter, candidate in manual_votes.items():
                voter_rep = await rep_skill.execute("get_reputation", {"agent_id": voter})
                voter_data = voter_rep.data if voter_rep.success else {}
                vote_weight = self._compute_vote_weight(voter_data)

                scores[voter] = {}
                for c in candidates:
                    if c == candidate:
                        scores[voter][c] = vote_weight * 100.0
                    else:
                        scores[voter][c] = 0.0

            result = await cons_skill.execute("elect", {
                "role": role,
                "candidates": candidates,
                "method": "score",
                "scores": scores,
            })

            return result

        else:
            return SkillResult(success=False, message=f"Unsupported method: {method}. Use 'score' or 'plurality'.")

    async def _tally(self, params: Dict) -> SkillResult:
        """Tally a proposal and auto-record reputation feedback."""
        proposal_id = params.get("proposal_id")
        if not proposal_id:
            return SkillResult(success=False, message="proposal_id is required")

        cons_skill = self._get_consensus_skill()
        config = self._load_config()

        # First, get the proposal to see who voted
        store = cons_skill._load_store()
        proposal = store.get("proposals", {}).get(proposal_id)
        if not proposal:
            return SkillResult(success=False, message=f"Proposal {proposal_id} not found")

        voters_before = dict(proposal.get("votes", {}))

        # Tally via ConsensusProtocolSkill
        tally_result = await cons_skill.execute("tally", {
            "proposal_id": proposal_id,
            "force_close": params.get("force_close", False),
        })

        if not tally_result.success:
            return tally_result

        # Auto-record reputation feedback if enabled
        feedback_recorded = []
        if config.get("auto_feedback", True):
            result_data = tally_result.data or {}
            result_info = result_data.get("result", {})
            outcome = result_info.get("status", "")

            rep_skill = self._get_reputation_skill()

            for voter_id, vote_data in voters_before.items():
                choice = vote_data.get("choice", "")
                if choice == "abstain":
                    # Still record participation
                    await rep_skill.execute("record_vote", {
                        "agent_id": voter_id,
                        "vote_type": "proposal",
                    })
                    feedback_recorded.append({"voter": voter_id, "feedback": "participation_only"})
                    continue

                # Determine if vote was "correct" (aligned with outcome)
                if outcome == "passed":
                    outcome_correct = (choice == "approve")
                elif outcome == "rejected":
                    outcome_correct = (choice == "reject")
                else:
                    outcome_correct = None

                await rep_skill.execute("record_vote", {
                    "agent_id": voter_id,
                    "vote_type": "proposal",
                    "outcome_correct": outcome_correct,
                })
                feedback_recorded.append({
                    "voter": voter_id,
                    "choice": choice,
                    "outcome_correct": outcome_correct,
                })

        # Record in audit log
        audit_entry = {
            "type": "tally",
            "proposal_id": proposal_id,
            "outcome": tally_result.data.get("result", {}).get("status") if tally_result.data else "unknown",
            "vote_count": len(voters_before),
            "feedback_recorded": len(feedback_recorded),
            "feedback_details": feedback_recorded,
            "timestamp": datetime.utcnow().isoformat(),
        }
        audit_log = self._load_audit()
        audit_log.append(audit_entry)
        self._save_audit()

        # Enrich the tally result message
        feedback_msg = ""
        if feedback_recorded:
            correct = sum(1 for f in feedback_recorded if f.get("outcome_correct") is True)
            incorrect = sum(1 for f in feedback_recorded if f.get("outcome_correct") is False)
            feedback_msg = f" Reputation feedback: {correct} correct, {incorrect} incorrect votes recorded."

        return SkillResult(
            success=True,
            message=tally_result.message + feedback_msg,
            data={
                **(tally_result.data or {}),
                "reputation_feedback": feedback_recorded,
            },
        )

    async def _configure(self, params: Dict) -> SkillResult:
        """Update reputation-weight configuration."""
        config = self._load_config()

        updated = []
        if "dimension_weights" in params:
            config["dimension_weights"] = params["dimension_weights"]
            updated.append("dimension_weights")
        if "min_weight" in params:
            config["min_weight"] = float(params["min_weight"])
            updated.append("min_weight")
        if "max_weight" in params:
            config["max_weight"] = float(params["max_weight"])
            updated.append("max_weight")
        if "sensitivity" in params:
            config["sensitivity"] = float(params["sensitivity"])
            updated.append("sensitivity")
        if "auto_feedback" in params:
            config["auto_feedback"] = bool(params["auto_feedback"])
            updated.append("auto_feedback")
        if "category_overrides" in params:
            config["category_overrides"] = params["category_overrides"]
            updated.append("category_overrides")

        self._config = config
        self._save_config()

        return SkillResult(
            success=True,
            message=f"Configuration updated: {', '.join(updated) if updated else 'no changes'}",
            data={"config": config},
        )

    async def _simulate(self, params: Dict) -> SkillResult:
        """Preview reputation-based vote weights for a set of agents."""
        agent_ids = params.get("agent_ids", [])
        category = params.get("category")

        if not agent_ids:
            return SkillResult(success=False, message="agent_ids list is required")

        rep_skill = self._get_reputation_skill()
        simulations = []

        for agent_id in agent_ids:
            rep_result = await rep_skill.execute("get_reputation", {"agent_id": agent_id})
            rep_data = rep_result.data if rep_result.success else {}

            weight = self._compute_vote_weight(rep_data, category)

            simulations.append({
                "agent_id": agent_id,
                "vote_weight": round(weight, 3),
                "reputation": {
                    k: round(v, 1) if isinstance(v, (int, float)) else v
                    for k, v in rep_data.items()
                    if k in ["competence", "trustworthiness", "cooperation", "leadership", "overall"]
                },
            })

        # Sort by weight descending
        simulations.sort(key=lambda s: s["vote_weight"], reverse=True)

        return SkillResult(
            success=True,
            message=f"Simulated weights for {len(simulations)} agents. "
                    f"Range: {simulations[-1]['vote_weight']:.2f} to {simulations[0]['vote_weight']:.2f}" if simulations else "No agents",
            data={
                "simulations": simulations,
                "category": category or "default",
                "config": {
                    "min_weight": self._load_config()["min_weight"],
                    "max_weight": self._load_config()["max_weight"],
                    "sensitivity": self._load_config()["sensitivity"],
                },
            },
        )

    async def _audit(self, params: Dict) -> SkillResult:
        """View audit log of reputation-weighted voting decisions."""
        proposal_id = params.get("proposal_id")
        limit = int(params.get("limit", 10))

        audit_log = self._load_audit()

        if proposal_id:
            entries = [e for e in audit_log if e.get("proposal_id") == proposal_id]
        else:
            entries = audit_log[-limit:]

        return SkillResult(
            success=True,
            message=f"Audit log: {len(entries)} entries",
            data={"entries": entries, "total_entries": len(audit_log)},
        )
