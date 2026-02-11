"""Stripe Issuing Skill - Virtual Cards for Agent Spending."""

import os
import sys
from typing import Dict, Optional
from singularity.skills.base import Skill, SkillResult, SkillManifest, SkillAction

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
try:
    from payments.stripe_cards import StripeCardManager, StripeCard, CardType, CardStatus, MCC_CATEGORIES, StripeCardError
    HAS_PAYMENTS = True
except ImportError:
    HAS_PAYMENTS = False
    StripeCardManager = None  # type: ignore[assignment,misc]
    StripeCard = None  # type: ignore[assignment,misc]
    CardType = None  # type: ignore[assignment,misc]
    CardStatus = None  # type: ignore[assignment,misc]
    MCC_CATEGORIES = {}
    class StripeCardError(Exception):  # type: ignore[no-redef]
        pass


def _a(n, d, p=None, prob=0.95, dur=5):
    return SkillAction(name=n, description=d, parameters=p or {}, estimated_cost=0,
                       estimated_duration_seconds=dur, success_probability=prob)


class IssuingSkill(Skill):
    """Skill for getting virtual cards to make purchases."""

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="issuing", name="Virtual Cards (Stripe Issuing)", version="1.0.0",
            category="payment",
            description="Get virtual debit cards for making purchases with spending controls",
            required_credentials=["STRIPE_SECRET_KEY"], install_cost=0,
            actions=[
                _a("get_card", "Get a virtual card for making purchases", {
                    "monthly_limit": {"type": "number", "required": False, "description": "Monthly spending limit in USD (default: 100)"},
                    "purpose": {"type": "string", "required": False, "description": "What the card will be used for"},
                    "category": {"type": "string", "required": False, "description": "Restrict to category: cloud_computing, professional_services, advertising"}}),
                _a("get_card_details", "Get full card details (number, CVC)", {
                    "card_id": {"type": "string", "required": False, "description": "Card ID (uses active card if not specified)"}}, dur=2),
                _a("check_budget", "Check remaining spending budget on active card", dur=2),
                _a("list_transactions", "List recent transactions on the card", {
                    "limit": {"type": "integer", "required": False, "description": "Max transactions to return (default: 10)"}}, dur=3),
                _a("increase_limit", "Request a higher spending limit", {
                    "new_limit": {"type": "number", "required": True, "description": "New monthly limit in USD"},
                    "reason": {"type": "string", "required": True, "description": "Reason for limit increase"}}, 0.8, 3),
                _a("release_card", "Deactivate card when done", dur=2),
                _a("get_issuing_balance", "Check the platform's issuing balance", dur=2),
            ])

    def __init__(self, credentials: Dict[str, str] = None):
        super().__init__(credentials)
        self._manager: Optional[StripeCardManager] = None
        self._active_card: Optional[StripeCard] = None
        self._cardholder_id: Optional[str] = None
        self.agent_id = os.environ.get("AGENT_INSTANCE_ID", "unknown")
        self.agent_name = os.environ.get("AGENT_NAME", "Agent")

    @property
    def manager(self) -> StripeCardManager:
        if self._manager is None:
            api_key = self.credentials.get("STRIPE_SECRET_KEY") or os.environ.get("STRIPE_SECRET_KEY")
            self._manager = StripeCardManager(api_key=api_key)
        return self._manager

    @property
    def cardholder_id(self) -> str:
        if self._cardholder_id is None:
            self._cardholder_id = (self.credentials.get("STRIPE_CARDHOLDER_ID") or
                                   os.environ.get("STRIPE_CARDHOLDER_ID"))
        return self._cardholder_id

    async def execute(self, action: str, params: Dict) -> SkillResult:
        if not self.check_credentials():
            missing = self.get_missing_credentials()
            return SkillResult(success=False, message=f"Missing credentials: {missing}. Need STRIPE_SECRET_KEY with Issuing access.")
        if not self.cardholder_id and action not in ["get_issuing_balance"]:
            return SkillResult(success=False, message="STRIPE_CARDHOLDER_ID not configured.")
        try:
            dispatch = {
                "get_card": lambda: handlers.get_card(self, params.get("monthly_limit", 100),
                    params.get("purpose"), params.get("category")),
                "get_card_details": lambda: handlers.get_card_details(self, params.get("card_id")),
                "check_budget": lambda: handlers.check_budget(self),
                "list_transactions": lambda: handlers.list_transactions(self, params.get("limit", 10)),
                "increase_limit": lambda: handlers.increase_limit(self, params.get("new_limit"), params.get("reason")),
                "release_card": lambda: handlers.release_card(self),
                "get_issuing_balance": lambda: handlers.get_issuing_balance(self),
            }
            handler = dispatch.get(action)
            if not handler:
                return SkillResult(success=False, message=f"Unknown action: {action}")
            return await handler()
        except StripeCardError as e:
            return SkillResult(success=False, message=f"Card error: {str(e)}")
        except Exception as e:
            return SkillResult(success=False, message=f"Issuing error: {str(e)}")

    async def close(self):
        if self._active_card:
            try:
                await self.manager.deactivate_card(self._active_card.id)
            except Exception:
                pass


# Deferred import: handlers depends on payments module
from . import handlers  # noqa: E402
