"""Handler functions for IssuingSkill actions."""

from typing import Optional
from singularity.skills.base import SkillResult
from payments.stripe_cards import CardType, CardStatus, MCC_CATEGORIES


async def get_card(skill, monthly_limit: float = 100,
                   purpose: str = None, category: str = None) -> SkillResult:
    if skill._active_card and skill._active_card.status == CardStatus.ACTIVE.value:
        return SkillResult(success=True,
            message=f"You already have an active card ending in {skill._active_card.last4}",
            data={"card_id": skill._active_card.id, "last4": skill._active_card.last4,
                  "status": skill._active_card.status, "has_full_details": bool(skill._active_card.number)})
    limit_cents = int(monthly_limit * 100)
    allowed_categories = None
    if category and category in MCC_CATEGORIES:
        allowed_categories = MCC_CATEGORIES[category]
    card = await skill.manager.create_agent_card(
        agent_id=skill.agent_id, cardholder_id=skill.cardholder_id,
        monthly_limit_cents=limit_cents, allowed_categories=allowed_categories)
    skill._active_card = card
    return SkillResult(success=True, message=f"Card created with ${monthly_limit}/month limit",
        data={"card_id": card.id, "last4": card.last4, "brand": card.brand,
              "card_number": card.number, "cvc": card.cvc, "exp_month": card.exp_month,
              "exp_year": card.exp_year, "expiry": card.expiry,
              "monthly_limit_usd": monthly_limit, "status": card.status, "purpose": purpose},
        asset_created={"type": "virtual_card", "card_id": card.id,
                       "last4": card.last4, "limit_usd": monthly_limit})


async def get_card_details(skill, card_id: str = None) -> SkillResult:
    if card_id:
        card = await skill.manager.get_card_details(card_id)
    elif skill._active_card:
        card = await skill.manager.get_card_details(skill._active_card.id)
    else:
        return SkillResult(success=False, message="No active card. Use get_card first.")
    skill._active_card = card
    return SkillResult(success=True, message=f"Card details for ...{card.last4}",
        data={"card_id": card.id, "card_number": card.number, "cvc": card.cvc,
              "exp_month": card.exp_month, "exp_year": card.exp_year, "expiry": card.expiry,
              "brand": card.brand, "last4": card.last4, "status": card.status})


async def check_budget(skill) -> SkillResult:
    if not skill._active_card:
        return SkillResult(success=False, message="No active card. Use get_card first.")
    transactions = await skill.manager.list_transactions(card_id=skill._active_card.id, limit=100)
    total_spent_cents = sum(abs(t.get("amount", 0)) for t in transactions)
    total_spent = total_spent_cents / 100
    limit_cents = 10000
    if skill._active_card.spending_controls:
        limits = skill._active_card.spending_controls.get("spending_limits", [])
        for lim in limits:
            if lim.get("interval") == "monthly":
                limit_cents = lim.get("amount", 10000)
                break
    limit_usd = limit_cents / 100
    remaining = max(0, limit_usd - total_spent)
    return SkillResult(success=True,
        message=f"${remaining:.2f} remaining of ${limit_usd:.2f} monthly limit",
        data={"monthly_limit_usd": limit_usd, "spent_usd": total_spent,
              "remaining_usd": remaining, "transaction_count": len(transactions),
              "card_id": skill._active_card.id, "card_last4": skill._active_card.last4})


async def list_transactions(skill, limit: int = 10) -> SkillResult:
    if not skill._active_card:
        return SkillResult(success=False, message="No active card. Use get_card first.")
    transactions = await skill.manager.list_transactions(card_id=skill._active_card.id, limit=limit)
    formatted = [{"id": t.get("id"), "amount_usd": abs(t.get("amount", 0)) / 100,
                  "merchant": t.get("merchant_name", "Unknown"),
                  "category": t.get("merchant_category"), "date": t.get("created")}
                 for t in transactions]
    total_spent = sum(t["amount_usd"] for t in formatted)
    return SkillResult(success=True, message=f"{len(formatted)} transactions, ${total_spent:.2f} total",
        data={"transactions": formatted, "count": len(formatted), "total_spent_usd": total_spent})


async def increase_limit(skill, new_limit: float, reason: str) -> SkillResult:
    if not skill._active_card:
        return SkillResult(success=False, message="No active card. Use get_card first.")
    if not new_limit or not reason:
        return SkillResult(success=False, message="Both new_limit and reason are required")
    card = await skill.manager.update_spending_limit(skill._active_card.id, int(new_limit * 100))
    skill._active_card = card
    return SkillResult(success=True, message=f"Limit increased to ${new_limit}/month",
        data={"card_id": card.id, "new_limit_usd": new_limit, "reason": reason})


async def release_card(skill) -> SkillResult:
    if not skill._active_card:
        return SkillResult(success=False, message="No active card to release.")
    card = await skill.manager.deactivate_card(skill._active_card.id)
    card_id = skill._active_card.id
    last4 = skill._active_card.last4
    skill._active_card = None
    return SkillResult(success=True, message=f"Card ...{last4} deactivated",
        data={"card_id": card_id, "status": "inactive"})


async def get_issuing_balance(skill) -> SkillResult:
    balance = await skill.manager.get_issuing_balance()
    return SkillResult(success=True,
        message=f"Issuing balance: ${balance['amount']/100:.2f} {balance['currency'].upper()}",
        data={"balance_cents": balance["amount"], "balance_usd": balance["amount"] / 100,
              "currency": balance["currency"]})
