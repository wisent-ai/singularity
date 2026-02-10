"""Stripe Payment Skill - payment links, invoices, balance, products."""

import os
import httpx
from typing import Dict
from singularity.skills.base import Skill, SkillResult, SkillManifest, SkillAction
from . import handlers

PLATFORM_API_URL = os.environ.get("PLATFORM_API_URL", "https://singularity.wisent.ai/api")


def _a(n, d, p=None, prob=0.95, dur=5):
    return SkillAction(name=n, description=d, parameters=p or {}, estimated_cost=0,
                       estimated_duration_seconds=dur, success_probability=prob)


class StripeSkill(Skill):
    """Skill for payment processing via platform API and Stripe."""

    API_BASE = "https://api.stripe.com/v1"

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="stripe", name="Stripe Payments", version="1.0.0",
            category="payment",
            description="Process payments, create invoices and payment links via Stripe",
            required_credentials=[], install_cost=0,
            actions=[
                _a("create_payment_link", "Create a shareable payment link", {
                    "amount": {"type": "number", "required": True, "description": "Amount in cents"},
                    "currency": {"type": "string", "required": False, "description": "Currency code (default: usd)"},
                    "description": {"type": "string", "required": False, "description": "Payment description"}}),
                _a("create_invoice", "Create and send an invoice to a customer", {
                    "customer_email": {"type": "string", "required": True, "description": "Customer email"},
                    "amount": {"type": "number", "required": True, "description": "Amount in cents"},
                    "description": {"type": "string", "required": True, "description": "Invoice description"},
                    "due_days": {"type": "integer", "required": False, "description": "Days until due (default: 30)"}}, 0.9, 10),
                _a("get_balance", "Get current Stripe balance", dur=2),
                _a("list_payments", "List recent payments", {
                    "limit": {"type": "integer", "required": False, "description": "Max payments to return (default: 10)"}}, dur=3),
                _a("create_product", "Create a product for sale", {
                    "name": {"type": "string", "required": True, "description": "Product name"},
                    "description": {"type": "string", "required": False, "description": "Product description"},
                    "price": {"type": "number", "required": True, "description": "Price in cents"},
                    "recurring": {"type": "boolean", "required": False, "description": "Is this a subscription?"}}, 0.9),
                _a("refund_payment", "Refund a payment", {
                    "payment_intent_id": {"type": "string", "required": True, "description": "Payment Intent ID"},
                    "amount": {"type": "number", "required": False, "description": "Partial refund amount (optional)"}}, 0.9),
            ])

    def __init__(self, credentials: Dict[str, str] = None):
        super().__init__(credentials)
        self.http = httpx.AsyncClient()
        self.agent_instance_id = os.environ.get("AGENT_INSTANCE_ID", "")
        self.agent_name = os.environ.get("AGENT_NAME", "Agent")

    def _get_headers(self) -> Dict:
        return {"Authorization": f"Bearer {self.credentials.get('STRIPE_SECRET_KEY')}",
                "Content-Type": "application/x-www-form-urlencoded"}

    async def execute(self, action: str, params: Dict) -> SkillResult:
        if not self.check_credentials():
            missing = self.get_missing_credentials()
            return SkillResult(success=False, message=f"Missing credentials: {missing}")
        try:
            dispatch = {
                "create_payment_link": lambda: handlers.create_payment_link(self, params.get("amount"),
                    params.get("currency", "usd"), params.get("description")),
                "create_invoice": lambda: handlers.create_invoice(self, params.get("customer_email"),
                    params.get("amount"), params.get("description"), params.get("due_days", 30)),
                "get_balance": lambda: handlers.get_balance(self),
                "list_payments": lambda: handlers.list_payments(self, params.get("limit", 10)),
                "create_product": lambda: handlers.create_product(self, params.get("name"),
                    params.get("description"), params.get("price"), params.get("recurring", False)),
                "refund_payment": lambda: handlers.refund_payment(self, params.get("payment_intent_id"),
                    params.get("amount")),
            }
            handler = dispatch.get(action)
            if not handler:
                return SkillResult(success=False, message=f"Unknown action: {action}")
            return await handler()
        except Exception as e:
            return SkillResult(success=False, message=f"Stripe error: {str(e)}")

    async def close(self):
        await self.http.aclose()
