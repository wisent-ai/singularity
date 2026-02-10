"""
Account Creator - LinkedIn Implementation

Contains create_linkedin_account_with_retries, create_linkedin_account.
"""

import asyncio
import random
import re
from datetime import datetime
from typing import Dict
from singularity.skills.base import SkillResult
from .skill import generate_email


async def create_linkedin_account_with_retries(skill, username: str, password: str,
                                                selectors: Dict, max_attempts: int = 3) -> SkillResult:
    """Create LinkedIn account with retries until we get email verification."""
    last_result = None

    for attempt in range(max_attempts):
        print(f"\n{'='*60}")
        print(f"LinkedIn Attempt {attempt + 1}/{max_attempts}")
        print(f"{'='*60}")

        email = generate_email(skill._email_domain)
        print(f"Using email: {email}")

        if skill.browser:
            await skill.browser.close()
            skill.browser = None
        await skill._init_browser()

        signup_url = "https://www.linkedin.com/signup"
        nav_result = await skill.browser.execute("goto", {"url": signup_url})
        if not nav_result.success:
            print(f"  Failed to load signup page")
            continue

        await asyncio.sleep(2)
        result = await create_linkedin_account(skill, username, email, password, selectors)
        last_result = result

        if result.data.get("got_email_verification"):
            print(f"\n SUCCESS! Got email verification on attempt {attempt + 1}")
            return result
        if result.data.get("status") in ["created", "email_verified"]:
            print(f"\n SUCCESS! Account created on attempt {attempt + 1}")
            return result

        if attempt < max_attempts - 1:
            wait_time = random.randint(30, 60)
            print(f"\n Waiting {wait_time}s before next attempt...")
            await asyncio.sleep(wait_time)

    print(f"\n All {max_attempts} attempts used CAPTCHA checkpoint")
    return last_result or SkillResult(
        success=False, message=f"LinkedIn signup failed after {max_attempts} attempts")


async def create_linkedin_account(skill, username: str, email: str,
                                  password: str, selectors: Dict) -> SkillResult:
    """Create LinkedIn account - multi-step wizard with checkpoint handling."""
    page = skill.browser._page

    first_names = ["Michael", "James", "Robert", "David", "William", "Richard", "Joseph",
                   "Thomas", "Christopher", "Daniel", "Matthew", "Anthony", "Mark", "Steven"]
    last_names = ["Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
                  "Wilson", "Anderson", "Taylor", "Thomas", "Jackson", "White", "Harris"]
    first_name = random.choice(first_names)
    last_name = random.choice(last_names)

    status = "started"
    got_email_verification = False

    try:
        # Step 1: Enter email
        print(f"  Step 1: Entering email...")
        email_input = await page.wait_for_selector('input#email-address', state='visible')
        await email_input.click()
        await asyncio.sleep(random.uniform(0.3, 0.6))
        await email_input.type(email, delay=random.randint(50, 100))
        await asyncio.sleep(random.uniform(0.5, 1.0))

        continue_btn = await page.query_selector('button#join-form-submit')
        if continue_btn:
            await continue_btn.click()
        await asyncio.sleep(random.uniform(2.0, 3.0))
        status = "email_submitted"

        # Step 2: Enter password
        print(f"  Step 2: Entering password...")
        try:
            await page.wait_for_selector('input#password', state='visible')
            password_input = await page.query_selector('input#password')
            if password_input:
                await password_input.click()
                await asyncio.sleep(random.uniform(0.3, 0.6))
                await password_input.type(password, delay=random.randint(40, 80))
                await asyncio.sleep(random.uniform(0.5, 1.0))
                continue_btn = await page.query_selector('button#join-form-submit')
                if continue_btn:
                    await continue_btn.click()
                await asyncio.sleep(random.uniform(2.0, 3.0))
                status = "password_submitted"
        except Exception as e:
            print(f"  Password step error: {e}")

        # Step 3: Enter name
        print(f"  Step 3: Entering name ({first_name} {last_name})...")
        try:
            await page.wait_for_selector('input#first-name', state='visible')
            first_input = await page.query_selector('input#first-name')
            if first_input:
                await first_input.click()
                await asyncio.sleep(random.uniform(0.2, 0.4))
                await first_input.type(first_name, delay=random.randint(60, 100))
            await asyncio.sleep(random.uniform(0.3, 0.6))
            last_input = await page.query_selector('input#last-name')
            if last_input:
                await last_input.click()
                await asyncio.sleep(random.uniform(0.2, 0.4))
                await last_input.type(last_name, delay=random.randint(60, 100))
            await asyncio.sleep(random.uniform(1.0, 2.0))
            status = "name_filled"
        except Exception as e:
            print(f"  Name step error: {e}")

        # Step 4: Skip form CAPTCHA
        print(f"  Step 4: Skipping form CAPTCHA - will handle at checkpoint...")

        # Step 5: Submit
        print(f"  Step 5: Submitting form...")
        submit_btn = await page.query_selector('button#join-form-submit')
        if submit_btn:
            await submit_btn.click()
        await asyncio.sleep(random.uniform(4.0, 6.0))

        # Step 6: Handle checkpoint
        print(f"  Step 6: Checking for checkpoint...")
        page_content = await page.content()

        if 'challengeUrl' in page_content or 'checkpoint' in page_content.lower():
            status = "checkpoint_detected"
            await asyncio.sleep(2)

            checkpoint_iframe = None
            for selector in ['iframe[src*="checkpoint"]', 'iframe[src*="challenge"]']:
                checkpoint_iframe = await page.query_selector(selector)
                if checkpoint_iframe:
                    break

            if checkpoint_iframe:
                print(f"  Found checkpoint iframe!")
                frame = await checkpoint_iframe.content_frame()

                if frame:
                    frame_content = await frame.content()

                    if 'resendUrl' in frame_content or 'verification' in frame_content.lower():
                        print(f"  -> Email verification checkpoint detected!")
                        got_email_verification = True
                        status = "pending_email_verification"

                        print(f"  Waiting for verification code to {email}...")
                        code = await skill._wait_for_verification_code(email, "linkedin", 120)
                        if code:
                            print(f"  Got verification code: {code}")
                            code_input = await frame.query_selector('input[type="text"], input[name*="pin"], input[name*="code"]')
                            if code_input:
                                await code_input.fill(code)
                                await asyncio.sleep(1)
                                submit_btn = await frame.query_selector('button[type="submit"], button[data-tracking*="submit"]')
                                if submit_btn:
                                    await submit_btn.click()
                                    await asyncio.sleep(5)
                                    status = "email_verified"
                        else:
                            print(f"  No verification code received")

                    elif 'captchaSiteKey' in frame_content or 'recaptcha' in frame_content.lower():
                        print(f"  -> CAPTCHA checkpoint detected!")
                        status = "captcha_checkpoint"

                        sitekey_match = re.search(r'captchaSiteKey["\s:]+([^"&\s]+)', frame_content)
                        checkpoint_sitekey = sitekey_match.group(1) if sitekey_match else None

                        if checkpoint_sitekey:
                            is_enterprise = checkpoint_sitekey.startswith("6Lfm") or checkpoint_sitekey.startswith("6Le")
                            print(f"  Checkpoint sitekey: {checkpoint_sitekey[:30]}...")

                            result = await skill.captcha.execute("solve_recaptcha_v2", {
                                "sitekey": checkpoint_sitekey,
                                "url": "https://www.linkedin.com/checkpoint/challenge/verify",
                                "invisible": False, "enterprise": is_enterprise})

                            if result.success:
                                token = result.data.get("token")
                                print(f"  Checkpoint CAPTCHA solved!")
                                await frame.evaluate('''(token) => {
                                    const textareas = document.querySelectorAll('textarea[name="g-recaptcha-response"]');
                                    textareas.forEach(ta => { ta.value = token; ta.innerHTML = token; });
                                }''', token)
                                submit_btn = await frame.query_selector('button[type="submit"]')
                                if submit_btn:
                                    await submit_btn.click()
                                    await asyncio.sleep(5)
                                    status = "captcha_solved"
                            else:
                                print(f"  Checkpoint CAPTCHA failed: {result.message}")
                                status = "captcha_failed"

        final_url = page.url
        if 'feed' in final_url or 'onboarding' in final_url or 'mynetwork' in final_url:
            status = "created"
        elif 'signup' in final_url and not got_email_verification:
            status = "blocked_still_on_signup"

    except Exception as e:
        print(f"  LinkedIn signup error: {e}")
        status = f"error: {str(e)[:50]}"

    account = {
        "site": "linkedin.com", "username": email, "email": email,
        "password": password, "name": f"{first_name} {last_name}",
        "created_at": datetime.now().isoformat(), "status": status,
        "got_email_verification": got_email_verification
    }
    skill.created_accounts.append(account)

    return SkillResult(
        success=status in ["created", "email_verified", "pending_email_verification"],
        message=f"LinkedIn account for {first_name} {last_name} - {status}",
        data=account)
