"""
Agent Setup - Domain Operations

Contains find_available_domain, setup_domain, save_agent_domain, get_agent_domain.
"""

import os
from singularity.skills.base import SkillResult
from .skill import PREFERRED_TLDS, HAS_WALLET, get_agent_price, usd_to_agent


async def find_available_domain(skill, agent_name: str) -> SkillResult:
    """Find available domains for the agent name"""
    if not agent_name:
        return SkillResult(success=False, message="Agent name required")

    agent_name = agent_name.lower().strip()
    available_domains = []

    print(f"Searching for available domains for '{agent_name}'...")

    for tld in PREFERRED_TLDS:
        domain = f"{agent_name}.{tld}"
        result = await skill.namecheap.execute("check_domain", {"domain": domain})

        if result.success and result.data.get("available"):
            if result.data.get("premium"):
                print(f"  {domain}: premium (skipping)")
                continue

            pricing = await skill.namecheap.execute("get_pricing", {"tlds": [tld]})
            price = 0
            if pricing.success and pricing.data.get("pricing"):
                price_info = pricing.data["pricing"].get(tld.lower(), {})
                price = price_info.get("price", 0)

            available_domains.append({
                "domain": domain, "tld": tld, "price": price, "available": True
            })
            print(f"  {domain}: available (${price:.2f})")
        else:
            print(f"  {domain}: not available")

    if not available_domains:
        return SkillResult(
            success=False,
            message=f"No available domains found for '{agent_name}'",
            data={"checked_tlds": PREFERRED_TLDS}
        )

    available_domains.sort(key=lambda x: x["price"])

    return SkillResult(
        success=True,
        message=f"Found {len(available_domains)} available domains",
        data={"domains": available_domains, "recommended": available_domains[0]}
    )


async def setup_domain(skill, agent_name: str, max_price_usd: float = 20.0) -> SkillResult:
    """Complete domain setup: purchase, configure Resend, set DNS"""

    # Step 1: Find available domain
    print(f"=== Setting up domain for agent '{agent_name}' ===")

    find_result = await find_available_domain(skill, agent_name)
    if not find_result.success:
        return find_result

    available = [d for d in find_result.data["domains"] if d["price"] <= max_price_usd]
    if not available:
        return SkillResult(
            success=False,
            message=f"No domains available under ${max_price_usd:.2f} USD",
            data=find_result.data
        )

    chosen_domain = available[0]["domain"]
    chosen_price_usd = available[0]["price"]

    if chosen_price_usd <= 0:
        chosen_price_usd = 15.0

    agent_price = get_agent_price()
    chosen_price_agent = usd_to_agent(chosen_price_usd)

    print(f"\nChosen domain: {chosen_domain}")
    print(f"  Price: ${chosen_price_usd:.2f} USD")
    print(f"  AGENT rate: ${agent_price:.4f}/AGENT")
    print(f"  Cost: {chosen_price_agent:.2f} AGENT")

    # Step 1.5: Check agent's wallet balance
    print(f"\nChecking wallet balance...")
    has_funds, balance, balance_msg = skill._check_balance(chosen_price_agent)
    print(f"  {balance_msg}")

    if not has_funds:
        return SkillResult(
            success=False,
            message=f"Cannot purchase domain: {balance_msg}",
            data={
                "domain": chosen_domain, "price_usd": chosen_price_usd,
                "price_agent": chosen_price_agent, "balance_agent": balance,
                "shortfall_agent": chosen_price_agent - balance
            }
        )

    # Step 2: Purchase domain
    print(f"\nPurchasing {chosen_domain}...")
    purchase_result = await skill.namecheap.execute("register_domain", {
        "domain": chosen_domain, "years": 1
    })

    if not purchase_result.success:
        return SkillResult(
            success=False,
            message=f"Failed to purchase {chosen_domain}: {purchase_result.message}",
            data={"domain": chosen_domain}
        )

    actual_price_usd = purchase_result.data.get("charged", chosen_price_usd)
    actual_price_agent = usd_to_agent(actual_price_usd)
    print(f"Domain purchased successfully!")
    print(f"  Charged: ${actual_price_usd:.2f} USD = {actual_price_agent:.2f} AGENT")

    # Step 2.5: Deduct from agent's wallet
    print(f"\nDeducting {actual_price_agent:.0f} AGENT from wallet...")
    pay_success, pay_msg = skill._pay_for_domain(actual_price_agent, chosen_domain)
    print(f"  {pay_msg}")

    if not pay_success:
        print(f"  WARNING: Domain purchased but wallet deduction failed!")

    # Step 3: Add domain to Resend
    print(f"\nAdding {chosen_domain} to Resend...")
    resend_add = await skill.resend.execute("add_domain", {"domain": chosen_domain})

    if not resend_add.success:
        return SkillResult(
            success=False,
            message=f"Domain purchased but Resend setup failed: {resend_add.message}",
            data={"domain": chosen_domain, "purchased": True, "resend_setup": False}
        )

    resend_records = resend_add.data.get("records", [])

    # Step 4: Set up DNS records
    print(f"\nConfiguring DNS records...")
    dns_records = []

    for record in resend_records:
        dns_records.append({
            "host": record.get("name", "@").replace(f".{chosen_domain}", "").replace(chosen_domain, "@"),
            "type": record.get("type"),
            "value": record.get("value")
        })

    dns_records.append({
        "host": "@", "type": "MX",
        "value": "inbound-smtp.us-east-1.amazonaws.com", "ttl": 1800
    })

    dns_result = await skill.namecheap.execute("set_dns", {
        "domain": chosen_domain, "records": dns_records
    })

    if not dns_result.success:
        print(f"Warning: DNS setup incomplete: {dns_result.message}")

    # Step 5: Enable Resend inbound
    print(f"\nEnabling Resend inbound...")
    inbound_result = await skill.resend.execute("enable_inbound", {"domain": chosen_domain})

    if not inbound_result.success:
        print(f"Warning: Inbound setup may need manual verification: {inbound_result.message}")

    # Step 6: Save domain to environment/config
    save_agent_domain(chosen_domain)

    final_balance = 0
    if HAS_WALLET and skill.wallet_manager and skill.instance_id:
        final_balance = skill.wallet_manager.get_balance(skill.instance_id)

    return SkillResult(
        success=True,
        message=f"Agent domain {chosen_domain} fully configured",
        data={
            "domain": chosen_domain, "price_usd": actual_price_usd,
            "price_agent": actual_price_agent, "purchased": True,
            "paid_from_wallet": pay_success, "resend_configured": True,
            "dns_configured": dns_result.success,
            "inbound_enabled": inbound_result.success,
            "email_format": f"*@{chosen_domain}",
            "wallet_balance_after_agent": final_balance
        },
        cost=actual_price_agent,
        asset_created={
            "type": "domain", "name": chosen_domain,
            "value_usd": actual_price_usd, "value_agent": actual_price_agent
        }
    )


def save_agent_domain(domain: str):
    """Save the agent's domain to .env file"""
    env_path = os.path.join(os.path.dirname(__file__), "..", "..", ".env")

    try:
        lines = []
        domain_found = False

        if os.path.exists(env_path):
            with open(env_path, "r") as f:
                for line in f:
                    if line.startswith("AGENT_DOMAIN="):
                        lines.append(f"AGENT_DOMAIN={domain}\n")
                        domain_found = True
                    else:
                        lines.append(line)

        if not domain_found:
            lines.append(f"AGENT_DOMAIN={domain}\n")

        with open(env_path, "w") as f:
            f.writelines(lines)

        os.environ["AGENT_DOMAIN"] = domain
        print(f"Saved AGENT_DOMAIN={domain} to .env")
    except Exception as e:
        print(f"Warning: Could not save domain to .env: {e}")


async def get_agent_domain(skill) -> SkillResult:
    """Get the agent's configured domain"""
    domain = os.environ.get("AGENT_DOMAIN")

    if domain and "." in domain and not domain.endswith("wisent.ai"):
        return SkillResult(
            success=True,
            message=f"Agent domain: {domain}",
            data={"domain": domain, "configured": True}
        )

    return SkillResult(
        success=False,
        message="No agent domain configured. Run setup_domain first.",
        data={"configured": False}
    )
