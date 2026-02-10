"""
Captcha Solver - Solve Methods

Contains solve_recaptcha_v2, solve_recaptcha_v3, solve_hcaptcha,
solve_turnstile, solve_image, solve_funcaptcha, extract_funcaptcha_params,
get_balance, solve_for_site.
"""

import asyncio
import base64
import urllib.parse
import re
from typing import Dict
from singularity.skills.base import SkillResult
from . import services as svc
from .skill import get_site_config


def _add_proxy(solver, task: dict):
    """Add proxy and user agent to task dict if configured."""
    if solver._user_agent: task["userAgent"] = solver._user_agent
    if solver._proxy:
        task.update({"proxyType": solver._proxy.get("type", "http"),
                     "proxyAddress": solver._proxy.get("address"),
                     "proxyPort": solver._proxy.get("port")})
        if solver._proxy.get("login"): task["proxyLogin"] = solver._proxy["login"]
        if solver._proxy.get("password"): task["proxyPassword"] = solver._proxy["password"]


def _try_next_service(solver, tried):
    """Switch to next untried service. Returns True if switched."""
    for s, k in solver._available_services.items():
        if s not in tried:
            solver._service, solver._api_key = s, k
            return True
    return False


async def solve_recaptcha_v2(solver, sitekey, url, invisible=False,
                              enterprise=False, enterprise_payload=None) -> SkillResult:
    """Solve reCAPTCHA v2 or Enterprise with automatic fallback."""
    tried, last_error = [], None
    action = (enterprise_payload or {}).get("action")
    s_param = (enterprise_payload or {}).get("s")
    use_proxy = solver._proxy is not None
    while True:
        sn = solver._service
        tried.append(sn)
        try:
            if sn == "nopecha" and not enterprise:
                sol = await svc.nopecha_solve_recaptcha(solver, sitekey, url, "v2")
                return SkillResult(success=True, message=f"reCAPTCHA v2 solved via nopecha",
                    data={"token": sol["token"], "task_id": "nopecha", "service": "nopecha"})
            elif sn == "capsolver":
                pfx = "ReCaptchaV2Enterprise" if enterprise else "ReCaptchaV2"
                task = {"type": f"{pfx}Task" if use_proxy else f"{pfx}TaskProxyLess",
                        "websiteURL": url, "websiteKey": sitekey, "isInvisible": invisible}
                if use_proxy: _add_proxy(solver, task)
                elif solver._user_agent: task["userAgent"] = solver._user_agent
                if action: task.setdefault("enterprisePayload", {})["action"] = action
                if s_param: task.setdefault("enterprisePayload", {})["s"] = s_param
                tid = await svc.capsolver_submit(solver, task)
                token = (await svc.capsolver_result(solver, tid)).get("gRecaptchaResponse")
            elif sn == "2captcha":
                p = {"method": "userrecaptcha", "googlekey": sitekey, "pageurl": url,
                     "invisible": 1 if invisible else 0}
                if solver._user_agent: p["userAgent"] = solver._user_agent
                if enterprise: p["enterprise"] = 1
                if action: p["action"] = action
                if s_param: p["data-s"] = s_param
                if use_proxy:
                    ps = f"{solver._proxy['address']}:{solver._proxy['port']}"
                    if solver._proxy.get("login"): ps = f"{solver._proxy['login']}:{solver._proxy['password']}@{ps}"
                    p["proxy"] = ps; p["proxytype"] = solver._proxy.get("type", "HTTP").upper()
                tid = await svc.twocaptcha_submit(solver, p)
                token = await svc.twocaptcha_result(solver, tid)
            elif sn == "anticaptcha":
                pfx = "RecaptchaV2Enterprise" if enterprise else "RecaptchaV2"
                task = {"type": f"{pfx}Task" if use_proxy else f"{pfx}TaskProxyless",
                        "websiteURL": url, "websiteKey": sitekey, "isInvisible": invisible}
                if use_proxy: _add_proxy(solver, task)
                elif solver._user_agent: task["userAgent"] = solver._user_agent
                if action or s_param:
                    ep = {}
                    if action: ep["action"] = action
                    if s_param: ep["s"] = s_param
                    task["enterprisePayload"] = ep
                tid = await svc.anticaptcha_submit(solver, task)
                token = (await svc.anticaptcha_result(solver, tid)).get("gRecaptchaResponse")
            else:
                raise Exception(f"Service {sn} doesn't support reCAPTCHA v2")
            return SkillResult(success=True, message=f"reCAPTCHA v2 solved via {sn}",
                data={"token": token, "task_id": tid, "service": sn, "used_proxy": use_proxy})
        except Exception as e:
            last_error = str(e); print(f"  {sn} failed: {last_error}")
            if not _try_next_service(solver, tried): break
    return SkillResult(success=False, message=f"All services failed: {last_error}", data={"tried_services": tried})


async def solve_recaptcha_v3(solver, sitekey, url, action="verify", min_score=0.3) -> SkillResult:
    if solver._service == "capsolver":
        tid = await svc.capsolver_submit(solver, {"type": "ReCaptchaV3TaskProxyLess",
            "websiteURL": url, "websiteKey": sitekey, "pageAction": action, "minScore": min_score})
        token = (await svc.capsolver_result(solver, tid)).get("gRecaptchaResponse")
    elif solver._service == "2captcha":
        tid = await svc.twocaptcha_submit(solver, {"method": "userrecaptcha", "googlekey": sitekey,
            "pageurl": url, "version": "v3", "action": action, "min_score": min_score})
        token = await svc.twocaptcha_result(solver, tid)
    else:
        tid = await svc.anticaptcha_submit(solver, {"type": "RecaptchaV3TaskProxyless",
            "websiteURL": url, "websiteKey": sitekey, "minScore": min_score, "pageAction": action})
        token = (await svc.anticaptcha_result(solver, tid)).get("gRecaptchaResponse")
    return SkillResult(success=True, message="reCAPTCHA v3 solved", data={"token": token, "task_id": tid})


async def solve_hcaptcha(solver, sitekey, url) -> SkillResult:
    tried, last_error = [], None
    while True:
        sn = solver._service; tried.append(sn)
        try:
            if sn == "nopecha":
                sol = await svc.nopecha_solve_hcaptcha(solver, sitekey, url)
                token, tid = sol["token"], "nopecha"
            elif sn == "2captcha":
                tid = await svc.twocaptcha_submit(solver, {"method": "hcaptcha", "sitekey": sitekey, "pageurl": url})
                token = await svc.twocaptcha_result(solver, tid)
            elif sn == "anticaptcha":
                tid = await svc.anticaptcha_submit(solver, {"type": "HCaptchaTaskProxyless", "websiteURL": url, "websiteKey": sitekey})
                token = (await svc.anticaptcha_result(solver, tid)).get("gRecaptchaResponse")
            elif sn == "nextcaptcha":
                tid = await svc.nextcaptcha_submit(solver, {"type": "HCaptchaTaskProxyless", "websiteURL": url, "websiteKey": sitekey})
                token = (await svc.nextcaptcha_result(solver, tid)).get("gRecaptchaResponse")
            elif sn == "nocaptchaai":
                sol = await svc.nocaptchaai_solve_hcaptcha(solver, sitekey, url)
                token, tid = sol["token"], "nocaptchaai"
            elif sn == "capsolver":
                tid = await svc.capsolver_submit(solver, {"type": "HCaptchaTaskProxyLess", "websiteURL": url, "websiteKey": sitekey})
                token = (await svc.capsolver_result(solver, tid)).get("gRecaptchaResponse")
            else: raise Exception(f"Unknown service: {sn}")
            return SkillResult(success=True, message=f"hCaptcha solved via {sn}",
                data={"token": token, "task_id": tid, "service": sn})
        except Exception as e:
            last_error = str(e); print(f"  {sn} failed: {last_error}")
            if not _try_next_service(solver, tried): break
    return SkillResult(success=False, message=f"All services failed: {last_error}", data={"tried_services": tried})


async def solve_turnstile(solver, sitekey, url) -> SkillResult:
    if solver._service == "capsolver":
        tid = await svc.capsolver_submit(solver, {"type": "AntiTurnstileTaskProxyLess", "websiteURL": url, "websiteKey": sitekey})
        token = (await svc.capsolver_result(solver, tid)).get("token")
    elif solver._service == "2captcha":
        tid = await svc.twocaptcha_submit(solver, {"method": "turnstile", "sitekey": sitekey, "pageurl": url})
        token = await svc.twocaptcha_result(solver, tid)
    else:
        tid = await svc.anticaptcha_submit(solver, {"type": "TurnstileTaskProxyless", "websiteURL": url, "websiteKey": sitekey})
        token = (await svc.anticaptcha_result(solver, tid)).get("token")
    return SkillResult(success=True, message="Turnstile solved", data={"token": token, "task_id": tid})


async def solve_image(solver, image, case_sensitive=False, numeric=False) -> SkillResult:
    if image.startswith("http"):
        image_data = base64.b64encode((await solver.http.get(image)).content).decode()
    elif image.startswith("data:"): image_data = image.split(",")[1]
    else: image_data = image
    if solver._service == "capsolver":
        tid = await svc.capsolver_submit(solver, {"type": "ImageToTextTask", "body": image_data})
        text = (await svc.capsolver_result(solver, tid)).get("text")
    elif solver._service == "2captcha":
        p = {"method": "base64", "body": image_data}
        if case_sensitive: p["regsense"] = 1
        if numeric: p["numeric"] = 1
        tid = await svc.twocaptcha_submit(solver, p)
        text = await svc.twocaptcha_result(solver, tid)
    else:
        tid = await svc.anticaptcha_submit(solver, {"type": "ImageToTextTask", "body": image_data, "case": case_sensitive, "numeric": numeric})
        text = (await svc.anticaptcha_result(solver, tid)).get("text")
    return SkillResult(success=True, message="Image captcha solved", data={"text": text, "task_id": tid})


async def solve_funcaptcha(solver, public_key, url, subdomain=None, blob=None) -> SkillResult:
    """Solve FunCaptcha/Arkose Labs with blob support and fallback."""
    tried, last_error = [], None
    norm_sub = subdomain.replace("https://", "").replace("http://", "").replace(
        "iframe.arkoselabs.com", "client-api.arkoselabs.com") if subdomain else None
    for svc_name in ["anticaptcha", "2captcha", "capsolver"]:
        if svc_name not in solver._available_services or svc_name in tried: continue
        tried.append(svc_name)
        api_key = solver._available_services[svc_name]
        try:
            if svc_name == "anticaptcha":
                task = {"type": "FunCaptchaTaskProxyless", "websiteURL": url, "websitePublicKey": public_key}
                if norm_sub: task["funcaptchaApiJSSubdomain"] = norm_sub
                if blob: task["data"] = f'{{"blob":"{blob}"}}'
                r = await solver.http.post(f"{solver.ANTICAPTCHA_API}/createTask", json={"clientKey": api_key, "task": task})
                d = r.json()
                if d.get("errorId", 0) != 0: raise Exception(d.get("errorDescription"))
                tid = str(d["taskId"])
                token = await _poll_anticaptcha(solver, api_key, tid)
            elif svc_name == "2captcha":
                p = {"key": api_key, "method": "funcaptcha", "publickey": public_key, "pageurl": url, "json": 1}
                if norm_sub: p["surl"] = f"https://{norm_sub}" if not norm_sub.startswith("http") else norm_sub
                if blob: p["data[blob]"] = blob
                r = await solver.http.post(f"{solver.TWOCAPTCHA_API}/in.php", data=p)
                d = r.json()
                if d.get("status") != 1: raise Exception(d.get("request"))
                tid = d["request"]
                token = await _poll_2captcha(solver, api_key, tid)
            elif svc_name == "capsolver":
                task = {"type": "FunCaptchaTaskProxyLess", "websiteURL": url, "websitePublicKey": public_key}
                if subdomain: task["funcaptchaApiJSSubdomain"] = subdomain
                if blob: task["data"] = f'{{"blob":"{blob}"}}'
                r = await solver.http.post(f"{solver.CAPSOLVER_API}/createTask", json={"clientKey": api_key, "task": task})
                d = r.json()
                if d.get("errorId", 0) != 0: raise Exception(d.get("errorDescription"))
                tid = d["taskId"]
                token = await _poll_capsolver(solver, api_key, tid)
            return SkillResult(success=True, message=f"FunCaptcha solved via {svc_name}",
                data={"token": token, "task_id": tid, "service": svc_name})
        except Exception as e:
            last_error = str(e); print(f"  {svc_name} failed: {last_error}")
    return SkillResult(success=False, message=f"All FunCaptcha services failed: {last_error}", data={"tried_services": tried})


async def _poll_anticaptcha(solver, api_key, tid, max_iter=30):
    for _ in range(max_iter):
        await asyncio.sleep(5)
        r = await solver.http.post(f"{solver.ANTICAPTCHA_API}/getTaskResult", json={"clientKey": api_key, "taskId": int(tid)})
        d = r.json()
        if d.get("errorId", 0) != 0: raise Exception(d.get("errorDescription"))
        if d.get("status") == "ready": return d["solution"]["token"]
    raise Exception("polling exceeded")

async def _poll_2captcha(solver, api_key, tid, max_iter=30):
    for _ in range(max_iter):
        await asyncio.sleep(5)
        r = await solver.http.get(f"{solver.TWOCAPTCHA_API}/res.php", params={"key": api_key, "action": "get", "id": tid, "json": 1})
        d = r.json()
        if d.get("status") == 1: return d["request"]
        elif d.get("request") != "CAPCHA_NOT_READY": raise Exception(d.get("request"))
    raise Exception("polling exceeded")

async def _poll_capsolver(solver, api_key, tid, max_iter=40):
    for _ in range(max_iter):
        await asyncio.sleep(3)
        r = await solver.http.post(f"{solver.CAPSOLVER_API}/getTaskResult", json={"clientKey": api_key, "taskId": tid})
        d = r.json()
        if d.get("errorId", 0) != 0: raise Exception(d.get("errorDescription"))
        if d.get("status") == "ready": return d["solution"]["token"]
    raise Exception("polling exceeded")


async def extract_funcaptcha_params(page) -> Dict:
    params = {"public_key": None, "subdomain": None, "blob": None}
    for iframe in await page.query_selector_all('iframe'):
        src = await iframe.get_attribute('src') or ''
        if 'arkoselabs' not in src.lower() and 'funcaptcha' not in src.lower(): continue
        parsed = urllib.parse.urlparse(src)
        query = urllib.parse.parse_qs(parsed.query)
        params["public_key"] = query.get('pkey', [None])[0]
        if not params["public_key"]:
            m = re.search(r'/([A-F0-9-]{36})/', src)
            if m: params["public_key"] = m.group(1)
        params["subdomain"] = query.get('surl', [f"{parsed.scheme}://{parsed.netloc}"])[0]
        for key in ['data[blob]', 'data']:
            if key in query: params["blob"] = query[key][0]; break
        if not params["blob"]:
            m = re.search(r'data%5Bblob%5D=([^&]+)', src) or re.search(r'[?&]data=([^&]+)', src)
            if m: params["blob"] = urllib.parse.unquote(m.group(1))
        if params["public_key"]: break
    return params


async def get_balance(solver) -> SkillResult:
    if solver._service == "capsolver":
        r = await solver.http.post(f"{solver.CAPSOLVER_API}/getBalance", json={"clientKey": solver._api_key})
        bal = r.json().get("balance", 0)
    elif solver._service == "2captcha":
        r = await solver.http.get(f"{solver.TWOCAPTCHA_API}/res.php", params={"key": solver._api_key, "action": "getbalance", "json": 1})
        bal = float(r.json().get("request", 0))
    else:
        r = await solver.http.post(f"{solver.ANTICAPTCHA_API}/getBalance", json={"clientKey": solver._api_key})
        bal = r.json().get("balance", 0)
    return SkillResult(success=True, message=f"Balance: ${bal:.2f}", data={"balance": bal, "service": solver._service})


async def solve_for_site(solver, site: str, sitekey: str) -> SkillResult:
    config = get_site_config(site)
    if not config: return SkillResult(success=False, message=f"No config for {site}")
    ct = config["captcha_type"]
    url = config.get("signup_url") or config.get("login_url") or f"https://{site}"
    if ct == "hcaptcha": return await solve_hcaptcha(solver, sitekey, url)
    elif ct == "recaptcha_v2": return await solve_recaptcha_v2(solver, sitekey, url)
    elif ct == "recaptcha_v3": return await solve_recaptcha_v3(solver, sitekey, url)
    elif ct == "turnstile": return await solve_turnstile(solver, sitekey, url)
    elif ct == "arkose": return await solve_funcaptcha(solver, sitekey, url)
    return SkillResult(success=False, message=f"Unsupported captcha type: {ct}")
