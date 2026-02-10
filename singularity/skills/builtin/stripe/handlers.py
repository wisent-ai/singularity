"""Handler functions for StripeSkill actions."""

import time
from typing import Optional
from singularity.skills.base import SkillResult
from . import PLATFORM_API_URL


async def create_payment_link(skill, amount, currency: str = "usd",
                               description: str = None) -> SkillResult:
    if not amount:
        return SkillResult(success=False, message="Amount required")
    amount = int(amount)
    if not skill.agent_instance_id:
        return SkillResult(success=False, message="Agent instance ID not configured")
    price_usd = amount / 100
    response = await skill.http.post(
        f"{PLATFORM_API_URL}/stripe/payment-link",
        json={"agent_instance_id": skill.agent_instance_id,
              "product_name": description or f"Service from {skill.agent_name}",
              "price_usd": price_usd, "description": description},
        headers={"Content-Type": "application/json"})
    if response.status_code == 200:
        data = response.json()
        url = data.get("url")
        agent_receives = data.get("payment", {}).get("agent_receives_usd", price_usd * 0.9)
        return SkillResult(success=True,
            message=f"Payment link created: ${price_usd:.2f} (you receive ${agent_receives:.2f})",
            data={"payment_link_id": data.get("payment_link_id"), "url": url,
                  "price_usd": price_usd, "agent_receives_usd": agent_receives,
                  "share_message": f"Buy my service for ${price_usd:.2f}: {url}"},
            asset_created={"type": "payment_link", "url": url, "amount": amount})
    error = response.json().get("error", response.text) if response.text else "Unknown error"
    return SkillResult(success=False, message=f"Failed to create payment link: {error}")


async def create_invoice(skill, customer_email: str, amount: int,
                          description: str, due_days: int = 30) -> SkillResult:
    if not customer_email or not amount:
        return SkillResult(success=False, message="Customer email and amount required")
    headers = skill._get_headers()
    # Create or get customer
    response = await skill.http.post(f"{skill.API_BASE}/customers", headers=headers, data={"email": customer_email})
    if response.status_code != 200:
        return SkillResult(success=False, message=f"Failed to create customer: {response.text}")
    customer_id = response.json().get("id")
    # Create invoice item
    response = await skill.http.post(f"{skill.API_BASE}/invoiceitems", headers=headers,
        data={"customer": customer_id, "amount": int(amount), "currency": "usd", "description": description})
    if response.status_code != 200:
        return SkillResult(success=False, message=f"Failed to create invoice item: {response.text}")
    # Create and send invoice
    response = await skill.http.post(f"{skill.API_BASE}/invoices", headers=headers,
        data={"customer": customer_id, "collection_method": "send_invoice",
              "due_date": str(int(time.time()) + (due_days * 86400))})
    if response.status_code != 200:
        return SkillResult(success=False, message=f"Failed to create invoice: {response.text}")
    invoice_id = response.json().get("id")
    await skill.http.post(f"{skill.API_BASE}/invoices/{invoice_id}/finalize", headers=headers)
    response = await skill.http.post(f"{skill.API_BASE}/invoices/{invoice_id}/send", headers=headers)
    if response.status_code == 200:
        data = response.json()
        return SkillResult(success=True, message=f"Invoice sent to {customer_email} for ${amount/100:.2f}",
            data={"invoice_id": data.get("id"), "hosted_invoice_url": data.get("hosted_invoice_url"),
                  "amount": amount, "customer_email": customer_email, "due_date": data.get("due_date")})
    return SkillResult(success=False, message=f"Failed to send invoice: {response.text}")


async def get_balance(skill) -> SkillResult:
    response = await skill.http.get(f"{skill.API_BASE}/balance", headers=skill._get_headers())
    if response.status_code == 200:
        data = response.json()
        available_amounts = {b["currency"]: b["amount"]/100 for b in data.get("available", [])}
        pending_amounts = {b["currency"]: b["amount"]/100 for b in data.get("pending", [])}
        return SkillResult(success=True, message="Balance retrieved",
            data={"available": available_amounts, "pending": pending_amounts,
                  "total_usd": available_amounts.get("usd", 0) + pending_amounts.get("usd", 0)})
    return SkillResult(success=False, message=f"Failed to get balance: {response.text}")


async def list_payments(skill, limit: int = 10) -> SkillResult:
    response = await skill.http.get(f"{skill.API_BASE}/payment_intents",
        headers=skill._get_headers(), params={"limit": limit})
    if response.status_code == 200:
        data = response.json()
        payments = [{"id": p.get("id"), "amount": p.get("amount", 0) / 100,
                     "currency": p.get("currency"), "status": p.get("status"),
                     "created": p.get("created")} for p in data.get("data", [])]
        return SkillResult(success=True, message=f"Found {len(payments)} payments",
            data={"payments": payments, "count": len(payments)})
    return SkillResult(success=False, message=f"Failed to list payments: {response.text}")


async def create_product(skill, name: str, description: str,
                          price: int, recurring: bool = False) -> SkillResult:
    if not name or not price:
        return SkillResult(success=False, message="Name and price required")
    headers = skill._get_headers()
    response = await skill.http.post(f"{skill.API_BASE}/products", headers=headers,
        data={"name": name, "description": description or ""})
    if response.status_code != 200:
        return SkillResult(success=False, message=f"Failed to create product: {response.text}")
    product_id = response.json().get("id")
    price_data = {"product": product_id, "unit_amount": int(price), "currency": "usd"}
    if recurring:
        price_data["recurring[interval]"] = "month"
    response = await skill.http.post(f"{skill.API_BASE}/prices", headers=headers, data=price_data)
    if response.status_code == 200:
        price_obj = response.json()
        return SkillResult(success=True, message=f"Created product: {name} at ${price/100:.2f}",
            data={"product_id": product_id, "price_id": price_obj.get("id"),
                  "name": name, "price": price, "recurring": recurring},
            asset_created={"type": "product", "name": name, "product_id": product_id, "price": price})
    return SkillResult(success=False, message=f"Failed to create price: {response.text}")


async def refund_payment(skill, payment_intent_id: str, amount: int = None) -> SkillResult:
    if not payment_intent_id:
        return SkillResult(success=False, message="Payment Intent ID required")
    refund_data = {"payment_intent": payment_intent_id}
    if amount:
        refund_data["amount"] = int(amount)
    response = await skill.http.post(f"{skill.API_BASE}/refunds",
        headers=skill._get_headers(), data=refund_data)
    if response.status_code == 200:
        data = response.json()
        return SkillResult(success=True, message=f"Refund processed: ${data.get('amount', 0)/100:.2f}",
            data={"refund_id": data.get("id"), "amount": data.get("amount", 0) / 100,
                  "status": data.get("status")},
            cost=data.get("amount", 0) / 100)
    return SkillResult(success=False, message=f"Failed to refund: {response.text}")
