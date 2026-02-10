#!/usr/bin/env python3
"""
Agent Domain Setup Skill - Core Class

Handles autonomous domain acquisition and email setup for agents.
Each agent purchases and owns its own domain (e.g., ralph.ai, diego.com).

The agent PAYS for the domain from its AGENT token balance.
"""

import os
import sys
from typing import Dict, Optional
from singularity.skills.base import Skill, SkillResult, SkillManifest, SkillAction
from ..namecheap import NamecheapSkill
from ..resend import ResendSkill
from ..supabase import SupabaseSkill

# Import platform systems
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
try:
    from sim_platform.wallet import WalletManager, TransactionType
    from sim_platform.pricing import get_pricing_service, get_agent_price, usd_to_agent
    HAS_WALLET = True
    HAS_PRICING = True
except ImportError:
    HAS_WALLET = False
    HAS_PRICING = False
    WalletManager = None
    TransactionType = None


# TLDs to try, in order of preference (cheapest/most available first)
PREFERRED_TLDS = [
    "ai",      # Good for AI agents
    "io",      # Tech-friendly
    "co",      # Short and available
    "app",     # Modern
    "dev",     # Developer-friendly
    "me",      # Personal
    "xyz",     # Cheap and available
    "net",     # Classic
    "org",     # Classic
    "com",     # Most recognized (often taken)
]


# Pricing functions - use platform service if available, otherwise fallback
if not HAS_PRICING:
    def get_agent_price() -> float:
        """Fallback: Get AGENT price when pricing service unavailable."""
        rate_str = os.environ.get("AGENT_PRICE_USD")
        if rate_str:
            return float(rate_str)
        return 0.10  # Default $0.10 per AGENT

    def usd_to_agent(usd_amount: float) -> float:
        """Fallback: Convert USD to AGENT."""
        price = get_agent_price()
        if price <= 0:
            raise ValueError("Invalid AGENT price")
        return usd_amount / price


class AgentSetupSkill(Skill):
    """
    Sets up an agent's autonomous infrastructure:
    - Purchases a domain for the agent
    - Configures email sending/receiving via Resend
    - Sets up DNS records

    The agent will OWN its domain, not use a subdomain.
    Infrastructure costs are paid in AGENT tokens at current market rate.
    """

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="agent_setup",
            name="Agent Infrastructure Setup",
            version="1.0.0",
            category="infrastructure",
            description="Set up agent's own domain and email infrastructure",
            required_credentials=[
                "NAMECHEAP_API_USER",
                "NAMECHEAP_API_KEY",
                "NAMECHEAP_USERNAME",
                "NAMECHEAP_CLIENT_IP",
                "RESEND_API_KEY",
            ],
            install_cost=150.0,
            actions=[
                SkillAction(
                    name="setup_domain",
                    description="Find and purchase a domain for the agent, set up email",
                    parameters={
                        "agent_name": {"type": "string", "required": True, "description": "Agent's name (e.g., 'ralph')"},
                        "max_price_usd": {"type": "number", "required": False, "description": "Max domain price in USD (default: 20)"}
                    },
                    estimated_cost=150.0,
                    estimated_duration_seconds=60,
                    success_probability=0.8
                ),
                SkillAction(
                    name="find_available_domain",
                    description="Find available domains for an agent name",
                    parameters={"agent_name": {"type": "string", "required": True}},
                    estimated_cost=0,
                    estimated_duration_seconds=10,
                    success_probability=0.9
                ),
                SkillAction(
                    name="get_agent_domain",
                    description="Get the agent's configured domain",
                    parameters={},
                    estimated_cost=0,
                    estimated_duration_seconds=1,
                    success_probability=1.0
                ),
                SkillAction(
                    name="setup_supabase",
                    description="Create a Supabase project for the agent with auth and database",
                    parameters={
                        "agent_name": {"type": "string", "required": True, "description": "Agent's name (e.g., 'ralph')"},
                        "enable_google_oauth": {"type": "boolean", "required": False, "description": "Enable Google OAuth"},
                        "enable_github_oauth": {"type": "boolean", "required": False, "description": "Enable GitHub OAuth"},
                    },
                    estimated_cost=0,
                    estimated_duration_seconds=120,
                    success_probability=0.85
                ),
                SkillAction(
                    name="get_supabase_credentials",
                    description="Get the agent's Supabase project credentials",
                    parameters={},
                    estimated_cost=0,
                    estimated_duration_seconds=1,
                    success_probability=1.0
                ),
                SkillAction(
                    name="setup_full_infrastructure",
                    description="Complete agent setup: domain, email, and Supabase in one call",
                    parameters={
                        "agent_name": {"type": "string", "required": True, "description": "Agent's name"},
                        "max_price_usd": {"type": "number", "required": False, "description": "Max domain price"},
                    },
                    estimated_cost=150.0,
                    estimated_duration_seconds=180,
                    success_probability=0.75
                )
            ]
        )

    def __init__(self, credentials: Dict[str, str] = None, instance_id: str = None):
        super().__init__(credentials)
        self.namecheap: Optional[NamecheapSkill] = None
        self.resend: Optional[ResendSkill] = None
        self.supabase: Optional[SupabaseSkill] = None
        self.wallet_manager = None

        # Agent's instance ID for wallet transactions
        self.instance_id = instance_id or os.environ.get("AGENT_INSTANCE_ID")

    async def _init_skills(self):
        """Initialize required skills"""
        if self.namecheap is None:
            self.namecheap = NamecheapSkill(credentials=self.credentials)
        if self.resend is None:
            self.resend = ResendSkill(credentials=self.credentials)
        if self.supabase is None:
            self.supabase = SupabaseSkill(credentials=self.credentials)
        if self.wallet_manager is None and HAS_WALLET:
            self.wallet_manager = WalletManager()

    def _check_balance(self, required_amount: float) -> tuple:
        """Check if agent has sufficient AGENT balance."""
        if not HAS_WALLET or not self.wallet_manager:
            return True, 0, "Wallet system not available - proceeding without balance check"
        if not self.instance_id:
            return False, 0, "No instance_id configured - cannot check wallet"
        balance = self.wallet_manager.get_balance(self.instance_id)
        if balance < required_amount:
            return False, balance, f"Insufficient balance: {balance:.2f} AGENT < {required_amount:.2f} AGENT required"
        return True, balance, f"Balance OK: {balance:.2f} AGENT"

    def _pay_for_domain(self, amount: float, domain: str) -> tuple:
        """Deduct domain cost from agent's wallet."""
        if not HAS_WALLET or not self.wallet_manager:
            return True, "Wallet system not available - skipping payment"
        if not self.instance_id:
            return False, "No instance_id configured - cannot process payment"
        success, message = self.wallet_manager.transfer(
            from_id=self.instance_id,
            to_id=WalletManager.PLATFORM_ID,
            amount=amount,
            tx_type=TransactionType.BURN,
            memo=f"Domain purchase: {domain}",
            fee_pct=0
        )
        return success, message

    async def execute(self, action: str, params: Dict) -> SkillResult:
        try:
            await self._init_skills()

            if action == "setup_domain":
                from . import domain
                return await domain.setup_domain(
                    self, params.get("agent_name"), params.get("max_price_usd", 20.0))
            elif action == "find_available_domain":
                from . import domain
                return await domain.find_available_domain(self, params.get("agent_name"))
            elif action == "get_agent_domain":
                from . import domain
                return await domain.get_agent_domain(self)
            elif action == "setup_supabase":
                from . import supabase_ops
                return await supabase_ops.setup_supabase(
                    self, params.get("agent_name"),
                    params.get("enable_google_oauth", False),
                    params.get("enable_github_oauth", False))
            elif action == "get_supabase_credentials":
                from . import supabase_ops
                return await supabase_ops.get_supabase_credentials(self)
            elif action == "setup_full_infrastructure":
                from . import supabase_ops
                return await supabase_ops.setup_full_infrastructure(
                    self, params.get("agent_name"), params.get("max_price_usd", 20.0))
            else:
                return SkillResult(success=False, message=f"Unknown action: {action}")
        except Exception as e:
            return SkillResult(success=False, message=f"Setup error: {str(e)}")

    async def close(self):
        """Clean up resources"""
        if self.namecheap:
            await self.namecheap.close()
        if self.resend:
            await self.resend.close()
        if self.supabase:
            await self.supabase.close()
