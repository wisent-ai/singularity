"""
Account Creator - Twitter/X Implementation

Contains create_twitter_account, test_site.
"""

import asyncio
import random
from datetime import datetime
from typing import Dict
from singularity.skills.base import SkillResult
from ..captcha import get_site_config


async def create_twitter_account(skill, username: str, email: str,
                                 password: str, selectors: Dict) -> SkillResult:
    """Create Twitter/X account with FunCaptcha solving."""
    page = skill.browser._page
    status = "started"

    try:
        await asyncio.sleep(5)

        # Step 1: Click "Create account"
        try:
            await page.get_by_role("button", name="Create account").click()
            await asyncio.sleep(3)
            status = "create_clicked"
        except Exception:
            pass

        # Step 2: Click "Use email instead"
        try:
            email_link = page.get_by_text("Use email instead")
            if await email_link.count() > 0:
                await email_link.click()
                await asyncio.sleep(2)
        except Exception:
            pass

        # Step 3: Fill form
        display_name = f"{username.replace('_', ' ').title()}"
        try:
            name_field = page.get_by_label("Name")
            if await name_field.count() > 0:
                await name_field.fill(display_name)
                status = "name_filled"
        except Exception:
            pass

        try:
            email_field = page.get_by_label("Email")
            if await email_field.count() > 0:
                await email_field.fill(email)
                status = "email_filled"
        except Exception:
            pass

        await asyncio.sleep(1)

        # Step 4: Set DOB
        birth_year = random.randint(1985, 2000)
        try:
            month_combo = page.get_by_label("Month")
            if await month_combo.count() > 0:
                await month_combo.select_option("June")
            day_combo = page.get_by_label("Day")
            if await day_combo.count() > 0:
                await day_combo.select_option("15")
            year_combo = page.get_by_label("Year")
            if await year_combo.count() > 0:
                await year_combo.select_option(str(birth_year))
            status = "dob_filled"
        except Exception as e:
            print(f"  DOB selection issue: {e}")

        await asyncio.sleep(1)

        # Step 5: Click Next buttons
        for _ in range(3):
            try:
                next_btn = page.get_by_role("button", name="Next")
                if await next_btn.count() > 0:
                    await next_btn.click()
                    await asyncio.sleep(2)
                    status = "next_clicked"
            except Exception:
                break

        # Step 6: Click "Sign up" to trigger CAPTCHA
        try:
            signup_btn = page.get_by_role("button", name="Sign up")
            if await signup_btn.count() > 0:
                await signup_btn.click()
                await asyncio.sleep(5)
                status = "signup_clicked"
        except Exception:
            pass

        # Step 7: Extract and solve FunCaptcha
        funcaptcha_params = {}
        for attempt in range(10):
            await asyncio.sleep(2)
            funcaptcha_params = await skill.captcha.extract_funcaptcha_params(page)
            if funcaptcha_params.get("public_key"):
                break

        if funcaptcha_params.get("public_key"):
            status = "captcha_detected"
            print(f"  FunCaptcha detected: {funcaptcha_params['public_key']}")

            solve_result = await skill.captcha.execute("solve_funcaptcha", {
                "public_key": funcaptcha_params["public_key"],
                "url": "https://twitter.com/i/flow/signup",
                "subdomain": "client-api.arkoselabs.com",
                "blob": funcaptcha_params.get("blob")
            })

            if solve_result.success:
                token = solve_result.data.get("token")
                status = "captcha_solved"
                print(f"  CAPTCHA solved!")

                await page.evaluate('''(token) => {
                    const inputs = document.querySelectorAll('input[type="hidden"]');
                    inputs.forEach(inp => {
                        if (inp.name && inp.name.toLowerCase().includes('token')) {
                            inp.value = token;
                        }
                    });
                    window.postMessage({
                        eventId: "challenge-complete",
                        payload: { sessionToken: token }
                    }, "*");
                }''', token)
                await asyncio.sleep(5)
            else:
                status = "captcha_failed"
                print(f"  CAPTCHA failed: {solve_result.message}")

        # Step 8: Check for email verification
        page_content = await page.content()
        if "verification" in page_content.lower() or "code" in page_content.lower():
            status = "pending_email_verification"
            print(f"  Waiting for verification code to {email}...")

            code = await skill._wait_for_verification_code(email, "twitter", 120)
            if code:
                code_inputs = await page.query_selector_all('input[type="text"]')
                for inp in code_inputs:
                    try:
                        await inp.fill(code)
                        break
                    except Exception:
                        continue
                await asyncio.sleep(1)
                try:
                    next_btn = page.get_by_role("button", name="Next")
                    if await next_btn.count() > 0:
                        await next_btn.click()
                        await asyncio.sleep(3)
                        status = "email_verified"
                except Exception:
                    pass

        # Step 9: Set password if prompted
        try:
            password_field = page.get_by_label("Password")
            if await password_field.count() > 0:
                await password_field.fill(password)
                await asyncio.sleep(1)
                next_btn = page.get_by_role("button", name="Next")
                if await next_btn.count() > 0:
                    await next_btn.click()
                    await asyncio.sleep(3)
                    status = "password_set"
        except Exception:
            pass

        # Check final status
        final_content = await page.content()
        if any(x in final_content.lower() for x in [
            "profile picture", "pick a username", "home",
            "what's happening", "for you", "skip for now"
        ]):
            status = "created"

    except Exception as e:
        print(f"  Twitter signup error: {e}")
        status = f"error: {str(e)[:50]}"

    account = {
        "site": "twitter.com", "username": username, "email": email,
        "password": password, "created_at": datetime.now().isoformat(), "status": status
    }
    skill.created_accounts.append(account)

    return SkillResult(
        success=status in ["created", "email_verified", "password_set"],
        message=f"Twitter signup for @{username} - {status}", data=account)


async def test_site(skill, site: str) -> SkillResult:
    """Test if account creation works on a site"""
    config = get_site_config(site)
    if not config:
        return SkillResult(success=False, message=f"No configuration for {site}")

    await skill._init_browser()

    signup_url = config["signup_url"]
    result = await skill.browser.execute("goto", {"url": signup_url})
    if not result.success:
        return SkillResult(success=False, message=f"Cannot load signup page: {signup_url}")

    screenshot_result = await skill.browser.execute("screenshot", {"path": f"/tmp/{site.replace('.', '_')}_signup.png"})

    selectors = config.get("selectors", {})
    found_elements = []
    missing_elements = []

    for name, selector in selectors.items():
        check_result = await skill.browser.execute("evaluate", {
            "script": f"!!document.querySelector('{selector}')"})
        if check_result.success and check_result.data.get("result"):
            found_elements.append(name)
        else:
            missing_elements.append(name)

    return SkillResult(
        success=len(missing_elements) == 0 or len(found_elements) > 0,
        message=f"Site test for {config['name']}: Found {len(found_elements)}/{len(selectors)} elements",
        data={
            "site": site, "signup_url": signup_url,
            "captcha_type": config["captcha_type"],
            "found_elements": found_elements, "missing_elements": missing_elements,
            "screenshot": screenshot_result.data.get("path") if screenshot_result.success else None
        })
