#!/usr/bin/env python3
"""
Agent Reputation System - Track agent reliability, competence, and trustworthiness.

This skill maintains reputation scores for agents across multiple dimensions,
computed from task delegation outcomes, consensus voting history, and peer
endorsements. Reputation scores are used to:
- Weight votes in ConsensusProtocolSkill
- Prioritize agents in TaskDelegationSkill
- Filter candidates in leader elections
- Gate access to high-value tasks

Pillar: Replication + Self-Improvement
- Replication: Enables trust-based coordination between replicas
- Self-Improvement: Agents can see their own reputation and improve weaknesses
"""

import json
import os
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional

from singularity.skills.base import (
    Skill,
    SkillAction,
    SkillManifest,
    SkillResult,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


@dataclass
class ReputationEvent:
    """A single event that affects an agent's reputation."""
    event_id: str
    agent_id: str
    event_type: str  # task_completed, task_failed, vote_correct, endorsement, penalty, etc.
    dimension: str  # competence, reliability, trustworthiness, leadership, cooperation
    delta: float  # positive or negative reputation change
    source: str  # which system generated this (task_delegation, consensus, peer, manual)
    details: Dict[str, Any] = field(default_factory=dict)
    created_at: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.utcnow().isoformat()


@dataclass
class AgentReputation:
    """Reputation profile for a single agent."""
    agent_id: str
    # Core reputation dimensions (0.0 to 100.0, start at 50.0 = neutral)
    competence: float = 50.0  # Task success rate, quality
    reliability: float = 50.0  # On-time delivery, timeout avoidance
    trustworthiness: float = 50.0  # Voting consistency, honesty
    leadership: float = 50.0  # Election wins, role performance
    cooperation: float = 50.0  # Conflict resolution, consensus participation
    # Aggregated
    overall: float = 50.0
    # Meta
    total_events: int = 0
    total_tasks: int = 0
    tasks_completed: int = 0
    tasks_failed: int = 0
    votes_cast: int = 0
    endorsements_received: int = 0
    penalties_received: int = 0
    first_seen: str = ""
    last_updated: str = ""

    def __post_init__(self):
        now = datetime.utcnow().isoformat()
        if not self.first_seen:
            self.first_seen = now
        if not self.last_updated:
            self.last_updated = now

    def compute_overall(self, weights: Dict[str, float] = None):
        """Compute weighted overall score from dimensions."""
        w = weights or {
            "competence": 0.30,
            "reliability": 0.25,
            "trustworthiness": 0.20,
            "leadership": 0.10,
            "cooperation": 0.15,
        }
        total_weight = sum(w.values())
        self.overall = (
            w.get("competence", 0) * self.competence
            + w.get("reliability", 0) * self.reliability
            + w.get("trustworthiness", 0) * self.trustworthiness
            + w.get("leadership", 0) * self.leadership
            + w.get("cooperation", 0) * self.cooperation
        ) / total_weight
        self.last_updated = datetime.utcnow().isoformat()


def _clamp(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, value))


class AgentReputationSkill(Skill):
    """
    Track and compute agent reputation scores across multiple dimensions.

    Actions:
    - record_event: Record a reputation-affecting event for an agent
    - get_reputation: Get full reputation profile for an agent
    - get_leaderboard: Rank agents by overall or specific dimension
    - compare: Compare two agents across all dimensions
    - record_task_outcome: Convenience: record task completion/failure
    - record_vote: Convenience: record voting participation
    - endorse: One agent endorses another's competence
    - penalize: Apply a reputation penalty to an agent
    - get_history: Get reputation event history for an agent
    - reset: Reset an agent's reputation to neutral
    """

    def __init__(self, credentials: Dict[str, str] = None):
        super().__init__(credentials)
        self._reputations: Dict[str, AgentReputation] = {}
        self._events: List[ReputationEvent] = []
        self._persist_path = os.path.join(DATA_DIR, "agent_reputation.json")
        self._load()

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="agent_reputation",
            name="Agent Reputation System",
            version="1.0.0",
            category="replication",
            description="Track agent reliability, competence, and trustworthiness across task delegation, consensus voting, and peer endorsements",
            actions=[
                SkillAction(
                    name="record_event",
                    description="Record a reputation-affecting event for an agent",
                    parameters={
                        "agent_id": {"type": "string", "required": True},
                        "event_type": {"type": "string", "required": True, "description": "e.g. task_completed, task_failed, vote_correct, endorsement, penalty"},
                        "dimension": {"type": "string", "required": True, "description": "competence, reliability, trustworthiness, leadership, cooperation"},
                        "delta": {"type": "float", "required": True, "description": "Reputation change amount (positive or negative)"},
                        "source": {"type": "string", "required": False, "description": "Source system (default: manual)"},
                        "details": {"type": "dict", "required": False},
                    },
                ),
                SkillAction(
                    name="get_reputation",
                    description="Get full reputation profile for an agent",
                    parameters={
                        "agent_id": {"type": "string", "required": True},
                    },
                ),
                SkillAction(
                    name="get_leaderboard",
                    description="Rank agents by overall or specific dimension score",
                    parameters={
                        "dimension": {"type": "string", "required": False, "description": "Dimension to rank by (default: overall)"},
                        "limit": {"type": "int", "required": False, "description": "Number of agents to return (default: 10)"},
                        "min_events": {"type": "int", "required": False, "description": "Minimum events to include (default: 0)"},
                    },
                ),
                SkillAction(
                    name="compare",
                    description="Compare two agents across all reputation dimensions",
                    parameters={
                        "agent_a": {"type": "string", "required": True},
                        "agent_b": {"type": "string", "required": True},
                    },
                ),
                SkillAction(
                    name="record_task_outcome",
                    description="Record a task delegation outcome (success/failure with budget efficiency)",
                    parameters={
                        "agent_id": {"type": "string", "required": True},
                        "success": {"type": "bool", "required": True},
                        "budget_efficiency": {"type": "float", "required": False, "description": "Ratio of budget_spent/budget_allocated (0-1, lower is better)"},
                        "on_time": {"type": "bool", "required": False, "description": "Whether task completed within timeout"},
                        "task_name": {"type": "string", "required": False},
                    },
                ),
                SkillAction(
                    name="record_vote",
                    description="Record that an agent participated in a vote or election",
                    parameters={
                        "agent_id": {"type": "string", "required": True},
                        "vote_type": {"type": "string", "required": False, "description": "proposal, election, or conflict (default: proposal)"},
                        "outcome_correct": {"type": "bool", "required": False, "description": "Did the agent's vote align with the final outcome?"},
                    },
                ),
                SkillAction(
                    name="endorse",
                    description="One agent endorses another's competence or cooperation",
                    parameters={
                        "from_agent": {"type": "string", "required": True},
                        "to_agent": {"type": "string", "required": True},
                        "dimension": {"type": "string", "required": False, "description": "Dimension to endorse (default: competence)"},
                        "reason": {"type": "string", "required": False},
                    },
                ),
                SkillAction(
                    name="penalize",
                    description="Apply a reputation penalty to an agent",
                    parameters={
                        "agent_id": {"type": "string", "required": True},
                        "dimension": {"type": "string", "required": True},
                        "amount": {"type": "float", "required": True, "description": "Penalty amount (positive number, will be subtracted)"},
                        "reason": {"type": "string", "required": True},
                    },
                ),
                SkillAction(
                    name="get_history",
                    description="Get reputation event history for an agent",
                    parameters={
                        "agent_id": {"type": "string", "required": True},
                        "limit": {"type": "int", "required": False, "description": "Number of events to return (default: 20)"},
                        "event_type": {"type": "string", "required": False, "description": "Filter by event type"},
                    },
                ),
                SkillAction(
                    name="reset",
                    description="Reset an agent's reputation to neutral (50.0 across all dimensions)",
                    parameters={
                        "agent_id": {"type": "string", "required": True},
                    },
                ),
            ],
            required_credentials=[],
        )

    async def execute(self, action: str, params: Dict) -> SkillResult:
        handlers = {
            "record_event": self._record_event,
            "get_reputation": self._get_reputation,
            "get_leaderboard": self._get_leaderboard,
            "compare": self._compare,
            "record_task_outcome": self._record_task_outcome,
            "record_vote": self._record_vote,
            "endorse": self._endorse,
            "penalize": self._penalize,
            "get_history": self._get_history,
            "reset": self._reset,
        }
        handler = handlers.get(action)
        if not handler:
            return SkillResult(success=False, message=f"Unknown action: {action}")
        try:
            return await handler(params)
        except Exception as e:
            return SkillResult(success=False, message=f"Error in {action}: {e}")

    def _ensure_agent(self, agent_id: str) -> AgentReputation:
        """Get or create a reputation profile for an agent."""
        if agent_id not in self._reputations:
            self._reputations[agent_id] = AgentReputation(agent_id=agent_id)
        return self._reputations[agent_id]

    def _add_event(self, agent_id: str, event_type: str, dimension: str,
                   delta: float, source: str = "manual",
                   details: Dict = None) -> ReputationEvent:
        """Add an event, apply delta, recompute overall."""
        event = ReputationEvent(
            event_id=f"evt_{len(self._events):06d}",
            agent_id=agent_id,
            event_type=event_type,
            dimension=dimension,
            delta=delta,
            source=source,
            details=details or {},
        )
        self._events.append(event)

        rep = self._ensure_agent(agent_id)
        rep.total_events += 1

        # Apply delta to the correct dimension
        valid_dims = ["competence", "reliability", "trustworthiness", "leadership", "cooperation"]
        if dimension in valid_dims:
            current = getattr(rep, dimension)
            setattr(rep, dimension, _clamp(current + delta))

        rep.compute_overall()
        self._save()
        return event

    # ── Core Actions ──────────────────────────────────────────

    async def _record_event(self, params: Dict) -> SkillResult:
        agent_id = params.get("agent_id")
        event_type = params.get("event_type")
        dimension = params.get("dimension")
        delta = float(params.get("delta", 0))
        source = params.get("source", "manual")
        details = params.get("details", {})

        if not agent_id or not event_type or not dimension:
            return SkillResult(success=False, message="agent_id, event_type, and dimension are required")

        event = self._add_event(agent_id, event_type, dimension, delta, source, details)
        rep = self._reputations[agent_id]

        return SkillResult(
            success=True,
            message=f"Recorded {event_type} for {agent_id}: {dimension} {'+' if delta >= 0 else ''}{delta:.1f} → {getattr(rep, dimension):.1f}",
            data={
                "event_id": event.event_id,
                "agent_id": agent_id,
                "dimension": dimension,
                "new_value": getattr(rep, dimension),
                "overall": rep.overall,
            },
        )

    async def _get_reputation(self, params: Dict) -> SkillResult:
        agent_id = params.get("agent_id")
        if not agent_id:
            return SkillResult(success=False, message="agent_id is required")

        rep = self._ensure_agent(agent_id)
        return SkillResult(
            success=True,
            message=f"Reputation for {agent_id}: overall {rep.overall:.1f}",
            data={
                "agent_id": rep.agent_id,
                "competence": round(rep.competence, 1),
                "reliability": round(rep.reliability, 1),
                "trustworthiness": round(rep.trustworthiness, 1),
                "leadership": round(rep.leadership, 1),
                "cooperation": round(rep.cooperation, 1),
                "overall": round(rep.overall, 1),
                "total_events": rep.total_events,
                "total_tasks": rep.total_tasks,
                "tasks_completed": rep.tasks_completed,
                "tasks_failed": rep.tasks_failed,
                "votes_cast": rep.votes_cast,
                "endorsements_received": rep.endorsements_received,
                "penalties_received": rep.penalties_received,
                "first_seen": rep.first_seen,
                "last_updated": rep.last_updated,
            },
        )

    async def _get_leaderboard(self, params: Dict) -> SkillResult:
        dimension = params.get("dimension", "overall")
        limit = int(params.get("limit", 10))
        min_events = int(params.get("min_events", 0))

        valid_dims = ["competence", "reliability", "trustworthiness", "leadership", "cooperation", "overall"]
        if dimension not in valid_dims:
            return SkillResult(success=False, message=f"Invalid dimension: {dimension}. Valid: {valid_dims}")

        agents = [
            rep for rep in self._reputations.values()
            if rep.total_events >= min_events
        ]
        agents.sort(key=lambda r: getattr(r, dimension), reverse=True)
        agents = agents[:limit]

        leaderboard = []
        for rank, rep in enumerate(agents, 1):
            leaderboard.append({
                "rank": rank,
                "agent_id": rep.agent_id,
                dimension: round(getattr(rep, dimension), 1),
                "overall": round(rep.overall, 1),
                "total_events": rep.total_events,
            })

        return SkillResult(
            success=True,
            message=f"Leaderboard by {dimension}: {len(leaderboard)} agents",
            data={"dimension": dimension, "leaderboard": leaderboard},
        )

    async def _compare(self, params: Dict) -> SkillResult:
        a_id = params.get("agent_a")
        b_id = params.get("agent_b")
        if not a_id or not b_id:
            return SkillResult(success=False, message="agent_a and agent_b are required")

        a = self._ensure_agent(a_id)
        b = self._ensure_agent(b_id)

        comparison = {}
        for dim in ["competence", "reliability", "trustworthiness", "leadership", "cooperation", "overall"]:
            a_val = round(getattr(a, dim), 1)
            b_val = round(getattr(b, dim), 1)
            diff = round(a_val - b_val, 1)
            comparison[dim] = {
                a_id: a_val,
                b_id: b_val,
                "difference": diff,
                "advantage": a_id if diff > 0 else (b_id if diff < 0 else "tie"),
            }

        return SkillResult(
            success=True,
            message=f"Comparison: {a_id} (overall {a.overall:.1f}) vs {b_id} (overall {b.overall:.1f})",
            data={"comparison": comparison},
        )

    # ── Convenience Actions ──────────────────────────────────

    async def _record_task_outcome(self, params: Dict) -> SkillResult:
        agent_id = params.get("agent_id")
        success = params.get("success")
        budget_efficiency = float(params.get("budget_efficiency", 0.5))
        on_time = params.get("on_time", True)
        task_name = params.get("task_name", "unknown")

        if not agent_id or success is None:
            return SkillResult(success=False, message="agent_id and success are required")

        rep = self._ensure_agent(agent_id)
        rep.total_tasks += 1

        if success:
            rep.tasks_completed += 1
            # Competence: +2 to +5 based on budget efficiency
            comp_delta = 2.0 + 3.0 * max(0, 1.0 - budget_efficiency)
            self._add_event(agent_id, "task_completed", "competence", comp_delta,
                           "task_delegation", {"task": task_name, "budget_efficiency": budget_efficiency})
            # Reliability: +2 if on time, -1 if late
            rel_delta = 2.0 if on_time else -1.0
            self._add_event(agent_id, "task_completed", "reliability", rel_delta,
                           "task_delegation", {"on_time": on_time})
        else:
            rep.tasks_failed += 1
            # Competence: -3 for failure
            self._add_event(agent_id, "task_failed", "competence", -3.0,
                           "task_delegation", {"task": task_name})
            # Reliability: -2 for failure
            self._add_event(agent_id, "task_failed", "reliability", -2.0,
                           "task_delegation", {"task": task_name})

        self._save()

        return SkillResult(
            success=True,
            message=f"Task {'completed' if success else 'failed'} for {agent_id}: competence={rep.competence:.1f}, reliability={rep.reliability:.1f}",
            data={
                "agent_id": agent_id,
                "competence": round(rep.competence, 1),
                "reliability": round(rep.reliability, 1),
                "overall": round(rep.overall, 1),
                "tasks_completed": rep.tasks_completed,
                "tasks_failed": rep.tasks_failed,
            },
        )

    async def _record_vote(self, params: Dict) -> SkillResult:
        agent_id = params.get("agent_id")
        vote_type = params.get("vote_type", "proposal")
        outcome_correct = params.get("outcome_correct")

        if not agent_id:
            return SkillResult(success=False, message="agent_id is required")

        rep = self._ensure_agent(agent_id)
        rep.votes_cast += 1

        # Participation bonus: cooperation +1
        self._add_event(agent_id, "vote_participation", "cooperation", 1.0,
                       "consensus", {"vote_type": vote_type})

        # Trustworthiness: +1.5 if correct, -0.5 if wrong, 0 if unknown
        if outcome_correct is True:
            self._add_event(agent_id, "vote_correct", "trustworthiness", 1.5,
                           "consensus", {"vote_type": vote_type})
        elif outcome_correct is False:
            self._add_event(agent_id, "vote_incorrect", "trustworthiness", -0.5,
                           "consensus", {"vote_type": vote_type})

        # Leadership bonus for election participation
        if vote_type == "election":
            self._add_event(agent_id, "election_participation", "leadership", 0.5,
                           "consensus", {})

        self._save()

        return SkillResult(
            success=True,
            message=f"Vote recorded for {agent_id}: trustworthiness={rep.trustworthiness:.1f}, cooperation={rep.cooperation:.1f}",
            data={
                "agent_id": agent_id,
                "trustworthiness": round(rep.trustworthiness, 1),
                "cooperation": round(rep.cooperation, 1),
                "votes_cast": rep.votes_cast,
            },
        )

    async def _endorse(self, params: Dict) -> SkillResult:
        from_agent = params.get("from_agent")
        to_agent = params.get("to_agent")
        dimension = params.get("dimension", "competence")
        reason = params.get("reason", "")

        if not from_agent or not to_agent:
            return SkillResult(success=False, message="from_agent and to_agent are required")
        if from_agent == to_agent:
            return SkillResult(success=False, message="Cannot endorse yourself")

        valid_dims = ["competence", "reliability", "trustworthiness", "leadership", "cooperation"]
        if dimension not in valid_dims:
            return SkillResult(success=False, message=f"Invalid dimension: {dimension}")

        rep = self._ensure_agent(to_agent)
        rep.endorsements_received += 1

        # Endorsement: +1.5 to the specified dimension
        # Endorser's reputation weights the endorsement (0.5x to 1.5x)
        endorser_rep = self._ensure_agent(from_agent)
        weight = max(0.5, min(1.5, endorser_rep.overall / 50.0))
        delta = 1.5 * weight

        self._add_event(to_agent, "endorsement", dimension, delta,
                       "peer", {"from_agent": from_agent, "reason": reason, "weight": weight})

        # Endorsing others also slightly boosts cooperation for the endorser
        self._add_event(from_agent, "gave_endorsement", "cooperation", 0.3,
                       "peer", {"to_agent": to_agent})

        self._save()

        return SkillResult(
            success=True,
            message=f"{from_agent} endorsed {to_agent} on {dimension}: +{delta:.1f} (weight: {weight:.2f})",
            data={
                "to_agent": to_agent,
                "from_agent": from_agent,
                "dimension": dimension,
                "delta": round(delta, 1),
                "new_value": round(getattr(rep, dimension), 1),
            },
        )

    async def _penalize(self, params: Dict) -> SkillResult:
        agent_id = params.get("agent_id")
        dimension = params.get("dimension")
        amount = float(params.get("amount", 0))
        reason = params.get("reason", "unspecified")

        if not agent_id or not dimension or amount <= 0:
            return SkillResult(success=False, message="agent_id, dimension, and positive amount are required")

        valid_dims = ["competence", "reliability", "trustworthiness", "leadership", "cooperation"]
        if dimension not in valid_dims:
            return SkillResult(success=False, message=f"Invalid dimension: {dimension}")

        rep = self._ensure_agent(agent_id)
        rep.penalties_received += 1

        self._add_event(agent_id, "penalty", dimension, -amount,
                       "manual", {"reason": reason})

        self._save()

        return SkillResult(
            success=True,
            message=f"Penalty applied to {agent_id}: {dimension} -{amount:.1f} → {getattr(rep, dimension):.1f} (reason: {reason})",
            data={
                "agent_id": agent_id,
                "dimension": dimension,
                "penalty": amount,
                "new_value": round(getattr(rep, dimension), 1),
                "overall": round(rep.overall, 1),
            },
        )

    async def _get_history(self, params: Dict) -> SkillResult:
        agent_id = params.get("agent_id")
        limit = int(params.get("limit", 20))
        event_type_filter = params.get("event_type")

        if not agent_id:
            return SkillResult(success=False, message="agent_id is required")

        events = [e for e in self._events if e.agent_id == agent_id]
        if event_type_filter:
            events = [e for e in events if e.event_type == event_type_filter]

        # Most recent first
        events = events[-limit:][::-1]

        return SkillResult(
            success=True,
            message=f"History for {agent_id}: {len(events)} events",
            data={
                "agent_id": agent_id,
                "events": [asdict(e) for e in events],
                "total_events": len([e for e in self._events if e.agent_id == agent_id]),
            },
        )

    async def _reset(self, params: Dict) -> SkillResult:
        agent_id = params.get("agent_id")
        if not agent_id:
            return SkillResult(success=False, message="agent_id is required")

        self._reputations[agent_id] = AgentReputation(agent_id=agent_id)
        self._save()

        return SkillResult(
            success=True,
            message=f"Reputation reset to neutral for {agent_id}",
            data={"agent_id": agent_id, "overall": 50.0},
        )

    # ── Persistence ─────────────────────────────────────────────

    def _save(self):
        os.makedirs(DATA_DIR, exist_ok=True)
        data = {
            "reputations": {aid: asdict(r) for aid, r in self._reputations.items()},
            "events": [asdict(e) for e in self._events[-5000:]],  # Keep last 5000 events
        }
        try:
            with open(self._persist_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass

    def _load(self):
        if not os.path.exists(self._persist_path):
            return
        try:
            with open(self._persist_path) as f:
                data = json.load(f)
            for aid, rdata in data.get("reputations", {}).items():
                self._reputations[aid] = AgentReputation(**rdata)
            for edata in data.get("events", []):
                self._events.append(ReputationEvent(**edata))
        except Exception:
            pass
