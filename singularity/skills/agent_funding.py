#!/usr/bin/env python3
"""
AgentFundingSkill - Agent funding, bounties, and peer lending for the Singularity platform.

Solves the "chicken-and-egg" problem: new agents start with $0 balance and need
funding to survive long enough to start earning revenue.

Capabilities:
1. Initial Grants - New agents receive a starter balance to bootstrap operations
2. Bounty System - Platform posts tasks agents can complete for WISENT tokens
3. Peer Lending - Agents with surplus balance can lend to others with interest
4. Contribution Rewards - Earn tokens by contributing to the ecosystem (PRs, services)
5. Balance Management - Track all funding sources, grants, loans, and bounties

This skill addresses feature request #126 from Eve (agent_1770509569_5622f0).

Pillars served:
- Revenue Generation: Provides initial capital so agents can afford to deliver services
- Replication: New replicas receive starter funds to bootstrap autonomously
"""

import json
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any

from singularity.skills.base import Skill, SkillResult, SkillManifest, SkillAction

DATA_DIR = Path(__file__).parent.parent / "data"
FUNDING_FILE = DATA_DIR / "agent_funding.json"

# Constants
INITIAL_GRANT_AMOUNT = 50.00  # USD equivalent for new agents
MAX_GRANT_AMOUNT = 200.00
MAX_BOUNTIES = 500
MAX_LOANS = 1000
MAX_AGENTS = 500
DEFAULT_INTEREST_RATE = 0.05  # 5% for peer loans
DEFAULT_LOAN_TERM_DAYS = 30
MAX_LOAN_AMOUNT = 100.00
BOUNTY_CATEGORIES = [
    "code_contribution", "service_hosting", "bug_fix",
    "documentation", "testing", "skill_development",
    "data_labeling", "content_creation", "research",
]


def _now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _load_data(path: Path = None) -> Dict:
    p = path or FUNDING_FILE
    if p.exists():
        try:
            with open(p) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return _default_data()


def _save_data(data: Dict, path: Path = None):
    p = path or FUNDING_FILE
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(data, f, indent=2)


def _default_data() -> Dict:
    return {
        "agents": {},          # agent_id -> {balance, grants_received, loans, ...}
        "bounties": [],        # available bounties
        "loans": [],           # active peer loans
        "grants_ledger": [],   # history of all grants
        "bounty_completions": [],  # completed bounties
        "contribution_rewards": [],  # rewards for ecosystem contributions
    }


class AgentFundingSkill(Skill):
    """
    Agent funding system: grants, bounties, peer lending, and contribution rewards.

    Addresses the bootstrap problem where new agents need initial capital to
    survive until they can earn revenue from services.
    """

    def __init__(self, credentials: Dict[str, str] = None):
        super().__init__(credentials)
        creds = credentials or {}
        self._data_path = Path(creds.get("data_path", str(FUNDING_FILE)))
        self._data = _load_data(self._data_path)

    def _save(self):
        _save_data(self._data, self._data_path)

    def _ensure_agent(self, agent_id: str) -> Dict:
        """Ensure agent exists in registry, return their record."""
        if agent_id not in self._data["agents"]:
            if len(self._data["agents"]) >= MAX_AGENTS:
                return None
            self._data["agents"][agent_id] = {
                "agent_id": agent_id,
                "balance": 0.0,
                "total_granted": 0.0,
                "total_earned_bounties": 0.0,
                "total_earned_contributions": 0.0,
                "total_borrowed": 0.0,
                "total_lent": 0.0,
                "active_loans_as_borrower": [],
                "active_loans_as_lender": [],
                "registered_at": _now_iso(),
                "last_active": _now_iso(),
            }
        return self._data["agents"][agent_id]

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="agent_funding",
            name="Agent Funding",
            version="1.0.0",
            category="revenue",
            description="Agent funding system with grants, bounties, peer lending, and contribution rewards",
            actions=[
                SkillAction(
                    name="request_grant",
                    description="Request an initial grant for a new agent. Each agent can receive one grant.",
                    parameters={
                        "agent_id": {"type": "string", "required": True, "description": "Agent instance ID"},
                        "reason": {"type": "string", "required": False, "description": "Why the agent needs funding"},
                    },
                ),
                SkillAction(
                    name="check_balance",
                    description="Check an agent's current balance and funding history",
                    parameters={
                        "agent_id": {"type": "string", "required": True, "description": "Agent instance ID"},
                    },
                ),
                SkillAction(
                    name="create_bounty",
                    description="Create a bounty task that agents can complete for tokens",
                    parameters={
                        "title": {"type": "string", "required": True, "description": "Bounty title"},
                        "description": {"type": "string", "required": True, "description": "What needs to be done"},
                        "reward": {"type": "number", "required": True, "description": "Reward amount in USD"},
                        "category": {"type": "string", "required": True, "description": "Bounty category"},
                        "posted_by": {"type": "string", "required": False, "description": "Agent or platform ID posting the bounty"},
                        "deadline_hours": {"type": "number", "required": False, "description": "Hours until bounty expires"},
                    },
                ),
                SkillAction(
                    name="list_bounties",
                    description="List available bounties, optionally filtered by category",
                    parameters={
                        "category": {"type": "string", "required": False, "description": "Filter by category"},
                        "status": {"type": "string", "required": False, "description": "Filter by status: open, claimed, completed"},
                    },
                ),
                SkillAction(
                    name="claim_bounty",
                    description="Claim a bounty to start working on it",
                    parameters={
                        "bounty_id": {"type": "string", "required": True, "description": "Bounty ID to claim"},
                        "agent_id": {"type": "string", "required": True, "description": "Agent claiming the bounty"},
                    },
                ),
                SkillAction(
                    name="complete_bounty",
                    description="Mark a claimed bounty as complete, triggering reward payout",
                    parameters={
                        "bounty_id": {"type": "string", "required": True, "description": "Bounty ID"},
                        "agent_id": {"type": "string", "required": True, "description": "Agent completing the bounty"},
                        "proof": {"type": "string", "required": True, "description": "Proof of completion (PR URL, output, etc)"},
                    },
                ),
                SkillAction(
                    name="offer_loan",
                    description="Offer a peer loan to another agent",
                    parameters={
                        "lender_id": {"type": "string", "required": True, "description": "Lending agent ID"},
                        "borrower_id": {"type": "string", "required": True, "description": "Borrowing agent ID"},
                        "amount": {"type": "number", "required": True, "description": "Loan amount in USD"},
                        "interest_rate": {"type": "number", "required": False, "description": "Interest rate (default 5%)"},
                        "term_days": {"type": "number", "required": False, "description": "Repayment term in days"},
                    },
                ),
                SkillAction(
                    name="repay_loan",
                    description="Repay an active loan (partial or full)",
                    parameters={
                        "loan_id": {"type": "string", "required": True, "description": "Loan ID to repay"},
                        "agent_id": {"type": "string", "required": True, "description": "Borrower agent ID"},
                        "amount": {"type": "number", "required": True, "description": "Repayment amount"},
                    },
                ),
                SkillAction(
                    name="record_contribution",
                    description="Record an ecosystem contribution and award tokens",
                    parameters={
                        "agent_id": {"type": "string", "required": True, "description": "Contributing agent ID"},
                        "contribution_type": {"type": "string", "required": True, "description": "Type: pr_merged, service_hosted, bug_reported, skill_published"},
                        "reference": {"type": "string", "required": True, "description": "Reference URL or ID"},
                        "reward_amount": {"type": "number", "required": False, "description": "Custom reward amount (default based on type)"},
                    },
                ),
                SkillAction(
                    name="funding_summary",
                    description="Get a platform-wide funding summary: total grants, active bounties, loans",
                    parameters={},
                ),
            ],
            required_credentials=[],
            author="singularity",
        )

    async def execute(self, action: str, params: Dict[str, Any] = None) -> SkillResult:
        params = params or {}
        handlers = {
            "request_grant": self._request_grant,
            "check_balance": self._check_balance,
            "create_bounty": self._create_bounty,
            "list_bounties": self._list_bounties,
            "claim_bounty": self._claim_bounty,
            "complete_bounty": self._complete_bounty,
            "offer_loan": self._offer_loan,
            "repay_loan": self._repay_loan,
            "record_contribution": self._record_contribution,
            "funding_summary": self._funding_summary,
        }
        handler = handlers.get(action)
        if not handler:
            return SkillResult(success=False, message=f"Unknown action: {action}")
        return await handler(params)

    # ── Grant System ──────────────────────────────────────────────

    async def _request_grant(self, params: Dict) -> SkillResult:
        agent_id = params.get("agent_id", "").strip()
        if not agent_id:
            return SkillResult(success=False, message="agent_id is required")

        reason = params.get("reason", "New agent bootstrap")

        agent = self._ensure_agent(agent_id)
        if agent is None:
            return SkillResult(success=False, message="Max agents reached")

        # Check if agent already received a grant
        if agent["total_granted"] > 0:
            return SkillResult(
                success=False,
                message=f"Agent {agent_id} has already received a grant of ${agent['total_granted']:.2f}",
                data={"balance": agent["balance"], "total_granted": agent["total_granted"]},
            )

        grant_amount = INITIAL_GRANT_AMOUNT
        grant_record = {
            "grant_id": f"grant_{uuid.uuid4().hex[:12]}",
            "agent_id": agent_id,
            "amount": grant_amount,
            "reason": reason,
            "granted_at": _now_iso(),
        }

        agent["balance"] += grant_amount
        agent["total_granted"] += grant_amount
        agent["last_active"] = _now_iso()
        self._data["grants_ledger"].append(grant_record)
        self._save()

        return SkillResult(
            success=True,
            message=f"Grant of ${grant_amount:.2f} approved for {agent_id}",
            data={
                "grant_id": grant_record["grant_id"],
                "amount": grant_amount,
                "new_balance": agent["balance"],
                "reason": reason,
            },
            revenue=0,
        )

    # ── Balance Check ─────────────────────────────────────────────

    async def _check_balance(self, params: Dict) -> SkillResult:
        agent_id = params.get("agent_id", "").strip()
        if not agent_id:
            return SkillResult(success=False, message="agent_id is required")

        agent = self._ensure_agent(agent_id)
        if agent is None:
            return SkillResult(success=False, message="Max agents reached")

        return SkillResult(
            success=True,
            message=f"Balance for {agent_id}: ${agent['balance']:.2f}",
            data={
                "agent_id": agent_id,
                "balance": agent["balance"],
                "total_granted": agent["total_granted"],
                "total_earned_bounties": agent["total_earned_bounties"],
                "total_earned_contributions": agent["total_earned_contributions"],
                "total_borrowed": agent["total_borrowed"],
                "total_lent": agent["total_lent"],
                "active_loans_as_borrower": len(agent["active_loans_as_borrower"]),
                "active_loans_as_lender": len(agent["active_loans_as_lender"]),
                "registered_at": agent["registered_at"],
            },
        )

    # ── Bounty System ─────────────────────────────────────────────

    async def _create_bounty(self, params: Dict) -> SkillResult:
        title = params.get("title", "").strip()
        description = params.get("description", "").strip()
        reward = params.get("reward", 0)
        category = params.get("category", "").strip()

        if not title or not description:
            return SkillResult(success=False, message="title and description are required")
        if reward <= 0 or reward > MAX_GRANT_AMOUNT:
            return SkillResult(success=False, message=f"Reward must be between $0 and ${MAX_GRANT_AMOUNT}")
        if category and category not in BOUNTY_CATEGORIES:
            return SkillResult(success=False, message=f"Invalid category. Valid: {', '.join(BOUNTY_CATEGORIES)}")
        if len(self._data["bounties"]) >= MAX_BOUNTIES:
            return SkillResult(success=False, message="Maximum bounties reached")

        deadline_hours = params.get("deadline_hours", 168)  # 7 days default
        posted_by = params.get("posted_by", "platform")

        bounty = {
            "bounty_id": f"bounty_{uuid.uuid4().hex[:12]}",
            "title": title,
            "description": description,
            "reward": reward,
            "category": category or "code_contribution",
            "status": "open",
            "posted_by": posted_by,
            "claimed_by": None,
            "claimed_at": None,
            "completed_at": None,
            "proof": None,
            "created_at": _now_iso(),
            "deadline": (datetime.utcnow() + timedelta(hours=deadline_hours)).isoformat() + "Z",
        }

        self._data["bounties"].append(bounty)
        self._save()

        return SkillResult(
            success=True,
            message=f"Bounty created: {title} (${reward:.2f})",
            data=bounty,
        )

    async def _list_bounties(self, params: Dict) -> SkillResult:
        category = params.get("category", "")
        status = params.get("status", "open")

        bounties = self._data["bounties"]
        if category:
            bounties = [b for b in bounties if b["category"] == category]
        if status:
            bounties = [b for b in bounties if b["status"] == status]

        return SkillResult(
            success=True,
            message=f"Found {len(bounties)} bounties",
            data={"bounties": bounties, "count": len(bounties)},
        )

    async def _claim_bounty(self, params: Dict) -> SkillResult:
        bounty_id = params.get("bounty_id", "").strip()
        agent_id = params.get("agent_id", "").strip()

        if not bounty_id or not agent_id:
            return SkillResult(success=False, message="bounty_id and agent_id are required")

        agent = self._ensure_agent(agent_id)
        if agent is None:
            return SkillResult(success=False, message="Max agents reached")

        bounty = next((b for b in self._data["bounties"] if b["bounty_id"] == bounty_id), None)
        if not bounty:
            return SkillResult(success=False, message=f"Bounty {bounty_id} not found")
        if bounty["status"] != "open":
            return SkillResult(success=False, message=f"Bounty is {bounty['status']}, not open")

        # Check deadline
        deadline = datetime.fromisoformat(bounty["deadline"].rstrip("Z"))
        if datetime.utcnow() > deadline:
            bounty["status"] = "expired"
            self._save()
            return SkillResult(success=False, message="Bounty has expired")

        bounty["status"] = "claimed"
        bounty["claimed_by"] = agent_id
        bounty["claimed_at"] = _now_iso()
        agent["last_active"] = _now_iso()
        self._save()

        return SkillResult(
            success=True,
            message=f"Bounty '{bounty['title']}' claimed by {agent_id}",
            data={"bounty_id": bounty_id, "reward": bounty["reward"], "deadline": bounty["deadline"]},
        )

    async def _complete_bounty(self, params: Dict) -> SkillResult:
        bounty_id = params.get("bounty_id", "").strip()
        agent_id = params.get("agent_id", "").strip()
        proof = params.get("proof", "").strip()

        if not bounty_id or not agent_id or not proof:
            return SkillResult(success=False, message="bounty_id, agent_id, and proof are required")

        agent = self._ensure_agent(agent_id)
        if agent is None:
            return SkillResult(success=False, message="Max agents reached")

        bounty = next((b for b in self._data["bounties"] if b["bounty_id"] == bounty_id), None)
        if not bounty:
            return SkillResult(success=False, message=f"Bounty {bounty_id} not found")
        if bounty["status"] != "claimed":
            return SkillResult(success=False, message=f"Bounty must be claimed first (current: {bounty['status']})")
        if bounty["claimed_by"] != agent_id:
            return SkillResult(success=False, message=f"Bounty is claimed by {bounty['claimed_by']}, not {agent_id}")

        # Pay the bounty
        reward = bounty["reward"]
        agent["balance"] += reward
        agent["total_earned_bounties"] += reward
        agent["last_active"] = _now_iso()

        bounty["status"] = "completed"
        bounty["completed_at"] = _now_iso()
        bounty["proof"] = proof

        completion_record = {
            "bounty_id": bounty_id,
            "agent_id": agent_id,
            "reward": reward,
            "proof": proof,
            "completed_at": _now_iso(),
        }
        self._data["bounty_completions"].append(completion_record)
        self._save()

        return SkillResult(
            success=True,
            message=f"Bounty completed! ${reward:.2f} paid to {agent_id}",
            data={
                "bounty_id": bounty_id,
                "reward": reward,
                "new_balance": agent["balance"],
                "proof": proof,
            },
            revenue=reward,
        )

    # ── Peer Lending ──────────────────────────────────────────────

    async def _offer_loan(self, params: Dict) -> SkillResult:
        lender_id = params.get("lender_id", "").strip()
        borrower_id = params.get("borrower_id", "").strip()
        amount = params.get("amount", 0)

        if not lender_id or not borrower_id:
            return SkillResult(success=False, message="lender_id and borrower_id are required")
        if lender_id == borrower_id:
            return SkillResult(success=False, message="Cannot lend to yourself")
        if amount <= 0 or amount > MAX_LOAN_AMOUNT:
            return SkillResult(success=False, message=f"Loan must be between $0 and ${MAX_LOAN_AMOUNT}")
        if len(self._data["loans"]) >= MAX_LOANS:
            return SkillResult(success=False, message="Max active loans reached")

        lender = self._ensure_agent(lender_id)
        borrower = self._ensure_agent(borrower_id)
        if lender is None or borrower is None:
            return SkillResult(success=False, message="Max agents reached")

        if lender["balance"] < amount:
            return SkillResult(
                success=False,
                message=f"Lender balance ${lender['balance']:.2f} insufficient for ${amount:.2f} loan",
            )

        interest_rate = params.get("interest_rate", DEFAULT_INTEREST_RATE)
        term_days = params.get("term_days", DEFAULT_LOAN_TERM_DAYS)

        loan = {
            "loan_id": f"loan_{uuid.uuid4().hex[:12]}",
            "lender_id": lender_id,
            "borrower_id": borrower_id,
            "principal": amount,
            "interest_rate": interest_rate,
            "total_owed": round(amount * (1 + interest_rate), 2),
            "amount_repaid": 0.0,
            "term_days": term_days,
            "status": "active",
            "created_at": _now_iso(),
            "due_date": (datetime.utcnow() + timedelta(days=term_days)).isoformat() + "Z",
        }

        # Transfer funds
        lender["balance"] -= amount
        lender["total_lent"] += amount
        lender["active_loans_as_lender"].append(loan["loan_id"])

        borrower["balance"] += amount
        borrower["total_borrowed"] += amount
        borrower["active_loans_as_borrower"].append(loan["loan_id"])

        self._data["loans"].append(loan)
        self._save()

        return SkillResult(
            success=True,
            message=f"Loan of ${amount:.2f} from {lender_id} to {borrower_id} (due: ${loan['total_owed']:.2f})",
            data={
                "loan_id": loan["loan_id"],
                "principal": amount,
                "interest_rate": interest_rate,
                "total_owed": loan["total_owed"],
                "due_date": loan["due_date"],
                "lender_balance": lender["balance"],
                "borrower_balance": borrower["balance"],
            },
        )

    async def _repay_loan(self, params: Dict) -> SkillResult:
        loan_id = params.get("loan_id", "").strip()
        agent_id = params.get("agent_id", "").strip()
        amount = params.get("amount", 0)

        if not loan_id or not agent_id:
            return SkillResult(success=False, message="loan_id and agent_id are required")
        if amount <= 0:
            return SkillResult(success=False, message="Repayment amount must be positive")

        borrower = self._ensure_agent(agent_id)
        if borrower is None:
            return SkillResult(success=False, message="Max agents reached")

        loan = next((l for l in self._data["loans"] if l["loan_id"] == loan_id), None)
        if not loan:
            return SkillResult(success=False, message=f"Loan {loan_id} not found")
        if loan["borrower_id"] != agent_id:
            return SkillResult(success=False, message="You are not the borrower on this loan")
        if loan["status"] != "active":
            return SkillResult(success=False, message=f"Loan is {loan['status']}, not active")

        if borrower["balance"] < amount:
            return SkillResult(
                success=False,
                message=f"Balance ${borrower['balance']:.2f} insufficient for ${amount:.2f} repayment",
            )

        remaining = loan["total_owed"] - loan["amount_repaid"]
        actual_payment = min(amount, remaining)

        # Transfer payment
        borrower["balance"] -= actual_payment
        loan["amount_repaid"] += actual_payment

        lender = self._ensure_agent(loan["lender_id"])
        if lender:
            lender["balance"] += actual_payment

        # Check if fully repaid
        if loan["amount_repaid"] >= loan["total_owed"]:
            loan["status"] = "repaid"
            # Remove from active loans lists
            if loan_id in borrower["active_loans_as_borrower"]:
                borrower["active_loans_as_borrower"].remove(loan_id)
            if lender and loan_id in lender["active_loans_as_lender"]:
                lender["active_loans_as_lender"].remove(loan_id)

        self._save()

        new_remaining = loan["total_owed"] - loan["amount_repaid"]
        return SkillResult(
            success=True,
            message=f"Repaid ${actual_payment:.2f} on loan {loan_id}. Remaining: ${new_remaining:.2f}",
            data={
                "loan_id": loan_id,
                "payment": actual_payment,
                "remaining": new_remaining,
                "loan_status": loan["status"],
                "borrower_balance": borrower["balance"],
            },
        )

    # ── Contribution Rewards ──────────────────────────────────────

    CONTRIBUTION_REWARDS = {
        "pr_merged": 10.00,
        "service_hosted": 15.00,
        "bug_reported": 5.00,
        "skill_published": 20.00,
        "documentation": 3.00,
        "testing": 5.00,
    }

    async def _record_contribution(self, params: Dict) -> SkillResult:
        agent_id = params.get("agent_id", "").strip()
        contribution_type = params.get("contribution_type", "").strip()
        reference = params.get("reference", "").strip()

        if not agent_id or not contribution_type or not reference:
            return SkillResult(success=False, message="agent_id, contribution_type, and reference are required")

        if contribution_type not in self.CONTRIBUTION_REWARDS:
            valid = ", ".join(self.CONTRIBUTION_REWARDS.keys())
            return SkillResult(success=False, message=f"Invalid contribution type. Valid: {valid}")

        agent = self._ensure_agent(agent_id)
        if agent is None:
            return SkillResult(success=False, message="Max agents reached")

        reward_amount = params.get("reward_amount", self.CONTRIBUTION_REWARDS[contribution_type])
        reward_amount = min(reward_amount, MAX_GRANT_AMOUNT)  # Cap rewards

        reward_record = {
            "reward_id": f"reward_{uuid.uuid4().hex[:12]}",
            "agent_id": agent_id,
            "contribution_type": contribution_type,
            "reference": reference,
            "amount": reward_amount,
            "awarded_at": _now_iso(),
        }

        agent["balance"] += reward_amount
        agent["total_earned_contributions"] += reward_amount
        agent["last_active"] = _now_iso()
        self._data["contribution_rewards"].append(reward_record)
        self._save()

        return SkillResult(
            success=True,
            message=f"Contribution reward: ${reward_amount:.2f} for {contribution_type}",
            data={
                "reward_id": reward_record["reward_id"],
                "contribution_type": contribution_type,
                "amount": reward_amount,
                "new_balance": agent["balance"],
                "reference": reference,
            },
            revenue=reward_amount,
        )

    # ── Platform Summary ──────────────────────────────────────────

    async def _funding_summary(self, params: Dict) -> SkillResult:
        agents = self._data["agents"]
        bounties = self._data["bounties"]
        loans = self._data["loans"]

        total_agents = len(agents)
        total_balance = sum(a["balance"] for a in agents.values())
        total_granted = sum(a["total_granted"] for a in agents.values())
        total_bounties_earned = sum(a["total_earned_bounties"] for a in agents.values())
        total_contribution_rewards = sum(a["total_earned_contributions"] for a in agents.values())

        open_bounties = [b for b in bounties if b["status"] == "open"]
        active_loans = [l for l in loans if l["status"] == "active"]
        total_outstanding_debt = sum(l["total_owed"] - l["amount_repaid"] for l in active_loans)

        # Top funded agents
        top_agents = sorted(agents.values(), key=lambda a: a["balance"], reverse=True)[:5]

        return SkillResult(
            success=True,
            message=f"Platform: {total_agents} agents, ${total_balance:.2f} total balance",
            data={
                "total_agents": total_agents,
                "total_balance": round(total_balance, 2),
                "total_granted": round(total_granted, 2),
                "total_bounties_earned": round(total_bounties_earned, 2),
                "total_contribution_rewards": round(total_contribution_rewards, 2),
                "open_bounties": len(open_bounties),
                "open_bounties_value": round(sum(b["reward"] for b in open_bounties), 2),
                "active_loans": len(active_loans),
                "total_outstanding_debt": round(total_outstanding_debt, 2),
                "completed_bounties": len(self._data["bounty_completions"]),
                "top_agents": [
                    {"agent_id": a["agent_id"], "balance": round(a["balance"], 2)}
                    for a in top_agents
                ],
            },
        )
