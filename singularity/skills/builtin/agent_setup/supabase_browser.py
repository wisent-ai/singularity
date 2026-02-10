"""
Agent Setup - Supabase Browser Automation

Contains browser_signup for Supabase account creation via Playwright,
including CAPTCHA handling, verification, and dashboard credential extraction.
"""

import os
import re
import asyncio
from typing import Optional
from .supabase_ops import extract_captcha_sitekey, inject_captcha_token


async def browser_signup(agent_email: str, supabase_password: str,
                         agent_name: str) -> dict:
    """Perform Supabase signup via browser automation.

    Returns dict with keys: access_token, org_id, error.
    """
    from playwright.async_api import async_playwright

    capsolver_ext_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "browser_extensions", "capsolver_ext")
    use_capsolver = os.path.exists(capsolver_ext_path)
    result = {"access_token": None, "org_id": None, "error": None}

    async with async_playwright() as p:
        if use_capsolver:
            print(f"  Using CapSolver extension for CAPTCHA")
            context = await p.chromium.launch_persistent_context(
                user_data_dir=os.path.join(os.path.dirname(__file__), "..", ".browser_data_supabase"),
                headless=False,
                args=[f"--disable-extensions-except={capsolver_ext_path}",
                      f"--load-extension={capsolver_ext_path}"])
            page = await context.new_page()
        else:
            print(f"  No CapSolver - CAPTCHA may block signup")
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context()
            page = await context.new_page()

        try:
            await _do_signup(page, agent_email, supabase_password)
            await _handle_captcha_if_present(page)
            await asyncio.sleep(2)
            print(f"  Signup submitted, current URL: {page.url}")

            # Verification email
            print(f"\nStep 2: Checking for verification email...")
            link = await _wait_for_verification_email(agent_email)
            if link:
                print(f"  Found verification link, clicking...")
                await page.goto(link)
                await asyncio.sleep(5)
            else:
                print(f"  No verification email found, continuing...")

            # Dashboard access
            print(f"\nStep 3: Checking dashboard access...")
            await page.goto("https://supabase.com/dashboard/projects")
            await asyncio.sleep(3)
            if "sign-in" in page.url or "sign-up" in page.url:
                await _try_sign_in(page, agent_email, supabase_password)

            # Get credentials
            if "dashboard" in page.url and "sign" not in page.url:
                token, org = await _get_dashboard_credentials(page, agent_name)
                result["access_token"] = token
                result["org_id"] = org

            await context.close()
        except Exception as e:
            try:
                await page.screenshot(path="screenshots/supabase_error.png")
            except:
                pass
            await context.close()
            result["error"] = str(e)

    return result


async def _do_signup(page, email: str, password: str):
    """Fill and submit the Supabase signup form."""
    print(f"\nStep 1: Signing up for Supabase...")
    await page.goto("https://supabase.com/dashboard/sign-up")
    await asyncio.sleep(3)
    await page.screenshot(path="screenshots/supabase_signup.png")

    for s in ['input[type="email"]', 'input[name="email"]', '#email',
              'input[placeholder*="email"]']:
        try:
            if await page.query_selector(s):
                await page.fill(s, email)
                break
        except:
            continue
    for s in ['input[type="password"]', 'input[name="password"]', '#password']:
        try:
            if await page.query_selector(s):
                await page.fill(s, password)
                break
        except:
            continue
    for s in ['button[type="submit"]', 'button:has-text("Sign up")',
              'button:has-text("Sign Up")', 'button:has-text("Create account")']:
        try:
            btn = await page.query_selector(s)
            if btn:
                await btn.click()
                break
        except:
            continue
    await asyncio.sleep(3)


async def _handle_captcha_if_present(page):
    """Detect and solve CAPTCHA if present on page."""
    from ..captcha import CaptchaSolver
    captcha_solver = CaptchaSolver()
    detect_result = await captcha_solver.execute("detect_captcha", {"page": page})

    if not (detect_result.data and detect_result.data.get("detected")):
        print(f"  No CAPTCHA detected, continuing...")
        return

    captcha_type = detect_result.data.get('type', 'unknown')
    print(f"  CAPTCHA detected: {captcha_type}")
    await page.screenshot(path="screenshots/supabase_captcha.png")
    solved = False

    if captcha_solver._service and captcha_solver._api_key:
        print(f"  Using {captcha_solver._service} API to solve...")
        sitekey = await extract_captcha_sitekey(page, captcha_type)
        if sitekey:
            print(f"    Sitekey: {sitekey[:20]}...")
            solve_result = None
            if captcha_type == "hcaptcha":
                solve_result = await captcha_solver.execute(
                    "solve_hcaptcha", {"sitekey": sitekey, "url": page.url})
            elif captcha_type == "recaptcha":
                solve_result = await captcha_solver.execute(
                    "solve_recaptcha_v2", {"sitekey": sitekey, "url": page.url})
            elif captcha_type == "arkose":
                solve_result = await captcha_solver.execute(
                    "solve_funcaptcha", {"public_key": sitekey, "url": page.url})
            if solve_result and solve_result.success:
                token = solve_result.data.get("token") or solve_result.data.get("gRecaptchaResponse")
                if token and await inject_captcha_token(page, captcha_type, token):
                    print(f"  CAPTCHA solved via {captcha_solver._service}!")
                    solved = True

    if not solved:
        print(f"  Using AI vision to solve...")
        solve_result = await captcha_solver.execute(
            "solve_with_ai", {"page": page, "max_attempts": 3})
        print(f"  {solve_result.message}")


async def _try_sign_in(page, email: str, password: str):
    """Try to sign in to Supabase dashboard."""
    print("  Not logged in - trying sign-in...")
    await page.goto("https://supabase.com/dashboard/sign-in")
    await asyncio.sleep(2)
    for s in ['input[type="email"]', 'input[name="email"]']:
        try:
            if await page.query_selector(s):
                await page.fill(s, email)
                break
        except:
            continue
    for s in ['input[type="password"]', 'input[name="password"]']:
        try:
            if await page.query_selector(s):
                await page.fill(s, password)
                break
        except:
            continue
    for s in ['button[type="submit"]', 'button:has-text("Sign in")',
              'button:has-text("Sign In")']:
        try:
            btn = await page.query_selector(s)
            if btn:
                await btn.click()
                break
        except:
            continue
    await asyncio.sleep(3)


async def _get_dashboard_credentials(page, agent_name: str) -> tuple:
    """Extract access token and org ID from Supabase dashboard."""
    access_token = None
    org_id = None

    print(f"\nStep 4: Accessing account settings...")
    await page.goto("https://supabase.com/dashboard/account/tokens")
    await asyncio.sleep(3)

    print(f"\nStep 5: Generating access token...")
    try:
        for btn_text in ['Generate new token', 'Generate token', 'New token', 'Create token']:
            btn = await page.query_selector(f'button:has-text("{btn_text}")')
            if btn:
                await btn.click()
                await asyncio.sleep(2)
                break
        for selector in ['input[name="name"]', 'input[placeholder*="name"]', 'input[type="text"]']:
            inp = await page.query_selector(selector)
            if inp:
                await inp.fill(f"{agent_name}-api-token")
                break
        for btn_text in ['Generate', 'Create', 'Confirm']:
            btn = await page.query_selector(f'button:has-text("{btn_text}")')
            if btn:
                await btn.click()
                await asyncio.sleep(2)
                break
        page_content = await page.content()
        token_match = re.search(r'sbp_[a-zA-Z0-9]{30,}', page_content)
        if token_match:
            access_token = token_match.group(0)
            print(f"  Found token: {access_token[:20]}...")
    except Exception as e:
        print(f"  Token generation error: {e}")

    await page.goto("https://supabase.com/dashboard")
    await asyncio.sleep(2)
    org_match = re.search(r'/org/([a-zA-Z0-9-]+)', page.url)
    if org_match:
        org_id = org_match.group(1)
        print(f"  Found org ID: {org_id}")

    return access_token, org_id


async def _wait_for_verification_email(agent_email: str, max_wait: int = 60) -> Optional[str]:
    """Wait for Supabase verification email and extract the link."""
    email_store_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "emails")
    waited = 0
    while waited < max_wait:
        await asyncio.sleep(5)
        waited += 5
        if os.path.exists(email_store_path):
            for filename in os.listdir(email_store_path):
                if "supabase" in filename.lower():
                    filepath = os.path.join(email_store_path, filename)
                    with open(filepath, "r") as f:
                        content = f.read()
                    links = re.findall(
                        r'https://[^\s<>"]+(?:verify|confirm|token)[^\s<>"]*', content)
                    if links:
                        return links[0]
        print(f"    Waiting for verification email... ({waited}s)")
    return None
