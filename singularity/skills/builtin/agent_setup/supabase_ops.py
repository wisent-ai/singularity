"""
Agent Setup - Supabase Operations & CAPTCHA Helpers

Contains setup_supabase orchestrator, get_supabase_credentials,
setup_full_infrastructure, credential persistence, and CAPTCHA
sitekey extraction / token injection helpers.
"""

import os
import re
import asyncio
import secrets
from typing import Optional
from singularity.skills.base import SkillResult
from ..supabase import SupabaseSkill


async def setup_supabase(skill, agent_name: str, enable_google_oauth: bool = False,
                         enable_github_oauth: bool = False) -> SkillResult:
    """Create agent's own Supabase account and project."""
    if not agent_name:
        return SkillResult(success=False, message="Agent name required")
    agent_domain = os.environ.get("AGENT_DOMAIN")
    if not agent_domain:
        return SkillResult(success=False,
            message="Agent domain not configured. Run setup_domain first to get email.")

    agent_email = f"{agent_name.lower()}@{agent_domain}"
    supabase_password = f"{secrets.token_urlsafe(12)}!A1"

    print(f"=== Setting up Supabase account for agent '{agent_name}' ===")
    print(f"Email: {agent_email}")

    from . import supabase_browser
    browser_result = await supabase_browser.browser_signup(
        agent_email, supabase_password, agent_name)

    if browser_result.get("error"):
        return SkillResult(success=False,
            message=f"Browser automation failed: {browser_result['error']}",
            data={"account_email": agent_email, "account_password": supabase_password,
                  "note": "Check screenshots/ for debugging."})

    access_token = browser_result.get("access_token")
    org_id = browser_result.get("org_id")
    if access_token and org_id:
        return await _create_project(skill, agent_name, agent_email,
                                     supabase_password, access_token, org_id)
    return SkillResult(success=False,
        message="Could not complete Supabase setup - manual intervention may be required",
        data={"account_email": agent_email, "account_password": supabase_password,
              "note": "Account may be created. Log in manually to complete setup."})


async def _create_project(skill, agent_name, agent_email, supabase_password,
                          access_token, org_id) -> SkillResult:
    """Create Supabase project via API after browser signup."""
    print(f"\nStep 6: Creating project via API...")
    skill.supabase = SupabaseSkill(credentials={"SUPABASE_ACCESS_TOKEN": access_token})
    project_name = f"{agent_name.lower()}-{secrets.token_hex(4)}"
    db_password = secrets.token_urlsafe(24)
    create_result = await skill.supabase.execute("create_project", {
        "name": project_name, "organization_id": org_id,
        "db_password": db_password, "region": "us-east-1", "plan": "free"})
    if not create_result.success:
        return SkillResult(success=False,
            message=f"Account created but project failed: {create_result.message}",
            data={"access_token": access_token, "org_id": org_id})
    project_id = create_result.data.get("project_id")
    print(f"  Waiting for project to be ready...")
    for _ in range(18):
        await asyncio.sleep(10)
        status_result = await skill.supabase.execute("get_project", {"project_id": project_id})
        if status_result.success:
            status = status_result.data.get("project", {}).get("status")
            print(f"    Status: {status}")
            if status == "ACTIVE_HEALTHY":
                break
    keys_result = await skill.supabase.execute("get_api_keys", {"project_id": project_id})
    if keys_result.success:
        _save_credentials(project_id, keys_result.data.get("api_url"),
            keys_result.data.get("anon_key"), keys_result.data.get("service_role_key"),
            db_password, access_token=access_token, org_id=org_id,
            email=agent_email, password=supabase_password)
        return SkillResult(success=True,
            message=f"Supabase account and project created for {agent_name}",
            data={"account_email": agent_email, "org_id": org_id,
                  "project_id": project_id, "project_name": project_name,
                  "api_url": keys_result.data.get("api_url"),
                  "anon_key": keys_result.data.get("anon_key"),
                  "service_role_key": keys_result.data.get("service_role_key"),
                  "access_token": access_token[:20] + "...",
                  "db_host": f"db.{project_id}.supabase.co"})
    return SkillResult(success=False, message="Project created but could not retrieve API keys")


def _save_credentials(project_id, api_url, anon_key, service_role_key, db_password,
                      access_token=None, org_id=None, email=None, password=None):
    """Save Supabase credentials to .env file"""
    env_path = os.path.join(os.path.dirname(__file__), "..", "..", ".env")
    new_vars = {"SUPABASE_PROJECT_ID": project_id, "SUPABASE_URL": api_url,
                "SUPABASE_ANON_KEY": anon_key, "SUPABASE_SERVICE_ROLE_KEY": service_role_key,
                "SUPABASE_DB_PASSWORD": db_password}
    if access_token: new_vars["SUPABASE_ACCESS_TOKEN"] = access_token
    if org_id: new_vars["SUPABASE_ORG_ID"] = org_id
    if email: new_vars["SUPABASE_ACCOUNT_EMAIL"] = email
    if password: new_vars["SUPABASE_ACCOUNT_PASSWORD"] = password
    try:
        lines, existing_keys = [], set()
        if os.path.exists(env_path):
            with open(env_path, "r") as f:
                for line in f:
                    key = line.split("=")[0].strip() if "=" in line else ""
                    if key in new_vars:
                        lines.append(f"{key}={new_vars[key]}\n")
                        existing_keys.add(key)
                    else:
                        lines.append(line)
        for key, value in new_vars.items():
            if key not in existing_keys:
                lines.append(f"{key}={value}\n")
        with open(env_path, "w") as f:
            f.writelines(lines)
        for key, value in new_vars.items():
            os.environ[key] = value
        print(f"Saved Supabase credentials to .env")
    except Exception as e:
        print(f"Warning: Could not save Supabase credentials to .env: {e}")


async def get_supabase_credentials(skill) -> SkillResult:
    """Get the agent's Supabase credentials"""
    project_id = os.environ.get("SUPABASE_PROJECT_ID")
    api_url = os.environ.get("SUPABASE_URL")
    anon_key = os.environ.get("SUPABASE_ANON_KEY")
    service_role_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    if project_id and api_url and anon_key:
        return SkillResult(success=True, message=f"Supabase configured: {project_id}",
            data={"project_id": project_id, "api_url": api_url, "anon_key": anon_key,
                  "service_role_key": service_role_key, "configured": True})
    return SkillResult(success=False,
        message="No Supabase project configured. Run setup_supabase first.",
        data={"configured": False})


async def setup_full_infrastructure(skill, agent_name: str,
                                    max_price_usd: float = 20.0) -> SkillResult:
    """Complete agent infrastructure setup: domain, email, and Supabase"""
    if not agent_name:
        return SkillResult(success=False, message="Agent name required")
    print(f"=== Full Infrastructure Setup for '{agent_name}' ===\n")
    results = {"agent_name": agent_name, "domain": None, "supabase": None, "errors": []}
    print("--- Step 1: Domain & Email ---")
    from . import domain
    domain_result = await domain.setup_domain(skill, agent_name, max_price_usd)
    if domain_result.success:
        results["domain"] = domain_result.data
        print(f"Domain setup complete: {domain_result.data.get('domain')}\n")
    else:
        results["errors"].append(f"Domain: {domain_result.message}")
        print(f"Domain setup failed: {domain_result.message}\n")
    print("--- Step 2: Supabase ---")
    supabase_result = await setup_supabase(skill, agent_name)
    if supabase_result.success:
        results["supabase"] = supabase_result.data
    else:
        results["errors"].append(f"Supabase: {supabase_result.message}")
    success = results["domain"] is not None and results["supabase"] is not None
    return SkillResult(success=success,
        message=f"Infrastructure {'complete' if success else 'partially complete'} for {agent_name}",
        data=results, cost=domain_result.cost if domain_result.success else 0)


# ==================== CAPTCHA HELPERS ====================

async def extract_captcha_sitekey(page, captcha_type: str) -> Optional[str]:
    """Extract the sitekey from a CAPTCHA element on the page."""
    try:
        selectors_map = {
            "hcaptcha": ['[data-sitekey]', '.h-captcha[data-sitekey]',
                         'div[data-sitekey]', 'iframe[src*="hcaptcha"]'],
            "recaptcha": ['.g-recaptcha[data-sitekey]', '[data-sitekey]',
                          'iframe[src*="recaptcha"]'],
            "arkose": ['[data-pkey]', 'iframe[src*="arkoselabs"]', 'iframe[src*="funcaptcha"]'],
            "funcaptcha": ['[data-pkey]', 'iframe[src*="arkoselabs"]'],
            "turnstile": ['.cf-turnstile[data-sitekey]', '[data-sitekey]'],
        }
        for selector in selectors_map.get(captcha_type, ['[data-sitekey]']):
            el = await page.query_selector(selector)
            if el:
                for attr in ['data-sitekey', 'data-pkey']:
                    val = await el.get_attribute(attr)
                    if val: return val
                src = await el.get_attribute('src')
                if src:
                    for pat in [r'sitekey=([a-f0-9-]+)', r'[?&]k=([^&]+)', r'pk=([^&]+)',
                                r'/([A-F0-9]{8}-[A-F0-9]{4}-[A-F0-9]{4}-[A-F0-9]{4}-[A-F0-9]{12})/']:
                        m = re.search(pat, src, re.I)
                        if m: return m.group(1)
        content = await page.content()
        for pat in [r'sitekey["\']?\s*[:=]\s*["\']([a-f0-9-]{20,})["\']',
                    r'captcha[_-]?key["\']?\s*[:=]\s*["\']([a-f0-9-]{20,})["\']']:
            m = re.search(pat, content, re.I)
            if m: return m.group(1)
        if captcha_type == "hcaptcha":
            for uuid in re.findall(r'[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}', content):
                if not uuid.startswith('0000') and not uuid.endswith('0000'):
                    return uuid
        return None
    except Exception as e:
        print(f"    Error extracting sitekey: {e}")
        return None


async def inject_captcha_token(page, captcha_type: str, token: str) -> bool:
    """Inject a solved CAPTCHA token into the page."""
    try:
        if captcha_type == "hcaptcha":
            await page.evaluate(f'''() => {{
                let ta = document.querySelector('textarea[name="h-captcha-response"]');
                if (ta) {{ ta.value = "{token}"; ta.innerHTML = "{token}"; }}
                let gr = document.querySelector('textarea[name="g-recaptcha-response"]');
                if (gr) {{ gr.value = "{token}"; gr.innerHTML = "{token}"; }}
                if (typeof window.hcaptcha !== 'undefined') {{ try {{
                    let wid = Object.keys(window.hcaptcha._widgets || {{}})[0];
                    if (wid && window.hcaptcha._widgets[wid].callback)
                        window.hcaptcha._widgets[wid].callback("{token}");
                }} catch(e) {{}} }}
            }}''')
            return True
        elif captcha_type == "recaptcha":
            await page.evaluate(f'''() => {{
                let ta = document.querySelector('textarea[name="g-recaptcha-response"]')
                    || document.querySelector('#g-recaptcha-response');
                if (ta) {{ ta.value = "{token}"; ta.innerHTML = "{token}"; }}
                if (typeof grecaptcha !== 'undefined') {{ try {{
                    let c = document.querySelector('.g-recaptcha');
                    if (c) {{ let cb = c.getAttribute('data-callback');
                        if (cb && typeof window[cb]==='function') window[cb]("{token}"); }}
                }} catch(e) {{}} }}
            }}''')
            return True
        elif captcha_type in ["arkose", "funcaptcha"]:
            return await page.evaluate(f'''() => {{
                for (let cb of ['ArkoseEnforcement','funcaptcha','fc'])
                    if (typeof window[cb]!=='undefined' && window[cb].setSessionToken) {{
                        window[cb].setSessionToken("{token}"); return true; }}
                let inp = document.querySelector('input[name="fc-token"]')
                    || document.querySelector('input[name="arkose_token"]');
                if (inp) {{ inp.value = "{token}"; return true; }}
                let form = document.querySelector('form');
                if (form) {{ let h = document.createElement('input');
                    h.type='hidden'; h.name='fc-token'; h.value="{token}";
                    form.appendChild(h); return true; }}
                return false;
            }}''')
        elif captcha_type == "turnstile":
            return await page.evaluate(f'''() => {{
                let inp = document.querySelector('input[name="cf-turnstile-response"]');
                if (inp) {{ inp.value = "{token}"; return true; }}
                return false;
            }}''')
        return False
    except Exception as e:
        print(f"    Error injecting token: {e}")
        return False
