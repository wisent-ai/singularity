"""
Account Creator - Platform-Specific Implementations

Contains create_account (router), create_reddit_account, create_tiktok_account,
create_generic_account, create_bulk, verify_email.
"""

import asyncio
import random
from datetime import datetime
from typing import Dict, Optional
from singularity.skills.base import SkillResult
from ..captcha import get_site_config, SITE_CONFIGS
from .skill import generate_username, generate_email, generate_password


async def create_account(skill, site: str, username: str = None,
                         email: str = None, password: str = None) -> SkillResult:
    """Create a single account on a platform"""
    config = get_site_config(site)
    if not config:
        return SkillResult(
            success=False,
            message=f"No configuration for {site}. Supported: {', '.join(SITE_CONFIGS.keys())}"
        )

    username = username or generate_username()
    email = email or generate_email()
    password = password or generate_password()

    await skill._init_browser()
    await skill._init_captcha()

    signup_url = config["signup_url"]
    selectors = config["selectors"]

    try:
        nav_result = await skill.browser.execute("goto", {"url": signup_url})
        if not nav_result.success:
            return SkillResult(success=False, message=f"Failed to load {signup_url}")

        await asyncio.sleep(2)

        if site == "reddit.com":
            return await create_reddit_account(skill, username, email, password, selectors)
        elif site == "tiktok.com":
            return await create_tiktok_account(skill, username, email, password, selectors)
        elif site == "linkedin.com":
            from . import linkedin
            return await linkedin.create_linkedin_account_with_retries(
                skill, username, password, selectors, max_attempts=3)
        elif site == "twitter.com":
            from . import twitter
            return await twitter.create_twitter_account(skill, username, email, password, selectors)
        else:
            return await create_generic_account(skill, site, username, email, password, config)
    except Exception as e:
        return SkillResult(success=False, message=f"Account creation failed: {str(e)}")


async def create_reddit_account(skill, username: str, email: str,
                                password: str, selectors: Dict) -> SkillResult:
    """Create Reddit account - multi-step flow with email verification via Resend"""
    print(f"Entering email: {email}")
    await skill.browser.execute("fill_role", {"role": "textbox", "text": email})
    await asyncio.sleep(1)
    await skill.browser.execute("click_button", {"text": "Continue"})
    await asyncio.sleep(5)

    print(f"Waiting for verification code email to {email}...")
    verification_code = await skill._wait_for_verification_code(
        email=email, sender_contains="reddit")

    if not verification_code:
        account = {
            "site": "reddit.com", "username": username, "email": email,
            "password": password, "created_at": datetime.now().isoformat(),
            "status": "max_wait_waiting_for_code",
            "message": "Verification code not received within 120 seconds"
        }
        skill.created_accounts.append(account)
        return SkillResult(success=False,
            message=f"Reddit signup failed - verification code not received for {email}",
            data=account)

    print(f"Got verification code: {verification_code}")
    await skill.browser.execute("fill_role", {"role": "textbox", "text": verification_code})
    await asyncio.sleep(1)
    await skill.browser.execute("click_button", {"text": "Continue"})
    await asyncio.sleep(5)
    status = "code_entered"

    page_content = await skill.browser.execute("get_page_content", {"format": "text"})
    page_text = page_content.data.get("content", "") if page_content.success else ""

    if "username" in page_text.lower():
        print("Filling username...")
        await skill.browser.execute("fill_role", {"role": "textbox", "text": username, "index": 0})
        await asyncio.sleep(2)
        print("Filling password...")
        await skill.browser.execute("fill_role", {"role": "textbox", "text": password, "index": 1})
        await asyncio.sleep(2)
        print("Submitting account...")
        await skill.browser.execute("click_button", {"text": "Continue"})
        await asyncio.sleep(8)
        status = "created"

    page_content = await skill.browser.execute("get_page_content", {"format": "text"})
    page_text = page_content.data.get("content", "") if page_content.success else ""

    if any(x in page_text.lower() for x in ["about you", "interests", "topics", "welcome", "home"]):
        status = "created"
    elif "blocked" in page_text.lower() or "error" in page_text.lower():
        status = "failed"

    account = {
        "site": "reddit.com", "username": username, "email": email,
        "password": password, "created_at": datetime.now().isoformat(), "status": status
    }
    skill.created_accounts.append(account)
    return SkillResult(success=status == "created",
        message=f"Reddit account u/{username} - {status}", data=account)


async def create_tiktok_account(skill, username: str, email: str,
                                password: str, selectors: Dict) -> SkillResult:
    """Create TikTok account"""
    birth_year = random.randint(1980, 2005)
    birth_month = random.randint(1, 12)
    birth_day = random.randint(1, 28)

    selects = await skill.browser.execute("evaluate", {
        "script": "Array.from(document.querySelectorAll('select')).length"})

    if selects.success and selects.data.get("result", 0) >= 3:
        await skill.browser.execute("evaluate", {
            "script": f"document.querySelectorAll('select')[0].value = '{birth_month}'; document.querySelectorAll('select')[0].dispatchEvent(new Event('change'))"})
        await asyncio.sleep(0.3)
        await skill.browser.execute("evaluate", {
            "script": f"document.querySelectorAll('select')[1].value = '{birth_day}'; document.querySelectorAll('select')[1].dispatchEvent(new Event('change'))"})
        await asyncio.sleep(0.3)
        await skill.browser.execute("evaluate", {
            "script": f"document.querySelectorAll('select')[2].value = '{birth_year}'; document.querySelectorAll('select')[2].dispatchEvent(new Event('change'))"})
        await asyncio.sleep(0.5)

    await skill.browser.execute("type", {"selector": selectors.get("email", "input[name='email']"), "text": email})
    await asyncio.sleep(0.5)
    await skill.browser.execute("type", {"selector": selectors.get("password", "input[type='password']"), "text": password})
    await asyncio.sleep(0.5)
    await skill.browser.execute("click", {"selector": selectors.get("send_code", "button[data-e2e='send-code-button']")})
    await asyncio.sleep(2)

    captcha_present = await skill.browser.execute("evaluate", {
        "script": "!!document.querySelector('[class*=\"captcha\"], [class*=\"Captcha\"], iframe[src*=\"captcha\"]')"})

    status = "pending_email_code"
    if captcha_present.success and captcha_present.data.get("result"):
        status = "blocked_by_captcha"

    page_content = await skill.browser.execute("get_page_content", {"format": "text"})
    page_text = page_content.data.get("content", "") if page_content.success else ""

    if "blocked" in page_text.lower() or "suspicious" in page_text.lower():
        status = "blocked"
    elif "code sent" in page_text.lower() or "check your email" in page_text.lower():
        status = "pending_email_code"

    account = {
        "site": "tiktok.com", "username": username, "email": email,
        "password": password, "birthday": f"{birth_year}-{birth_month:02d}-{birth_day:02d}",
        "created_at": datetime.now().isoformat(), "status": status
    }
    skill.created_accounts.append(account)
    return SkillResult(success=status not in ["blocked", "blocked_by_captcha"],
        message=f"TikTok signup for @{username} - {status}", data=account)


async def create_generic_account(skill, site: str, username: str,
                                 email: str, password: str, config: Dict) -> SkillResult:
    """Generic account creation for unconfigured sites"""
    selectors = config.get("selectors", {})
    if selectors.get("username"):
        await skill.browser.execute("type", {"selector": selectors["username"], "text": username})
    if selectors.get("email"):
        await skill.browser.execute("type", {"selector": selectors["email"], "text": email})
    if selectors.get("password"):
        await skill.browser.execute("type", {"selector": selectors["password"], "text": password})
    if selectors.get("submit"):
        await skill.browser.execute("click", {"selector": selectors["submit"]})
    await asyncio.sleep(3)

    account = {
        "site": site, "username": username, "email": email, "password": password,
        "created_at": datetime.now().isoformat(), "status": "unknown"
    }
    skill.created_accounts.append(account)
    return SkillResult(success=True, message=f"Account creation attempted on {site}", data=account)


async def create_bulk(skill, site: str, count: int, username_prefix: str = "") -> SkillResult:
    """Create multiple accounts"""
    results = []
    success_count = 0

    for i in range(count):
        username = generate_username(username_prefix)
        email = generate_email()
        password = generate_password()
        result = await create_account(skill, site, username, email, password)
        results.append({"index": i + 1, "success": result.success, "username": username, "message": result.message})
        if result.success:
            success_count += 1
        if i < count - 1:
            await asyncio.sleep(random.uniform(5, 15))

    return SkillResult(
        success=success_count > 0,
        message=f"Created {success_count}/{count} accounts on {site}",
        data={"site": site, "total": count, "success": success_count,
              "failed": count - success_count, "results": results})


async def verify_email(skill, email: str, site: str) -> SkillResult:
    """Check for and complete email verification"""
    return SkillResult(success=False, message="Email verification not yet implemented - need TEMP_MAIL_API_KEY")
