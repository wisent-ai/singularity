"""
Captcha Solver - Service Implementations

Submit/result methods for all supported captcha solving services:
2captcha, anticaptcha, capsolver, nopecha, nextcaptcha, nocaptchaai.
All functions take `solver` (CaptchaSolver instance) as first parameter.
"""

import asyncio
from typing import Dict


# ==================== 2CAPTCHA ====================

async def twocaptcha_submit(solver, params: Dict) -> str:
    """Submit task to 2captcha, return task ID"""
    params["key"] = solver._api_key
    params["json"] = 1
    resp = await solver.http.post(f"{solver.TWOCAPTCHA_API}/in.php", data=params)
    data = resp.json()
    if data.get("status") != 1:
        raise Exception(f"2captcha error: {data.get('request', 'Unknown error')}")
    return data["request"]


async def twocaptcha_result(solver, task_id: str, max_wait: int = 120) -> str:
    """Poll for result from 2captcha"""
    for _ in range(max_wait // 5):
        await asyncio.sleep(5)
        resp = await solver.http.get(f"{solver.TWOCAPTCHA_API}/res.php",
            params={"key": solver._api_key, "action": "get", "id": task_id, "json": 1})
        data = resp.json()
        if data.get("status") == 1:
            return data["request"]
        elif data.get("request") != "CAPCHA_NOT_READY":
            raise Exception(f"2captcha error: {data.get('request')}")
    raise Exception("Captcha solving timeout")


# ==================== ANTI-CAPTCHA ====================

async def anticaptcha_submit(solver, task: Dict) -> str:
    """Submit task to anti-captcha, return task ID"""
    resp = await solver.http.post(f"{solver.ANTICAPTCHA_API}/createTask",
        json={"clientKey": solver._api_key, "task": task})
    data = resp.json()
    if data.get("errorId", 0) != 0:
        raise Exception(f"Anti-captcha error: {data.get('errorDescription', 'Unknown')}")
    return str(data["taskId"])


async def anticaptcha_result(solver, task_id: str, max_wait: int = 120) -> Dict:
    """Poll for result from anti-captcha"""
    for _ in range(max_wait // 5):
        await asyncio.sleep(5)
        resp = await solver.http.post(f"{solver.ANTICAPTCHA_API}/getTaskResult",
            json={"clientKey": solver._api_key, "taskId": int(task_id)})
        data = resp.json()
        if data.get("errorId", 0) != 0:
            raise Exception(f"Anti-captcha error: {data.get('errorDescription')}")
        if data.get("status") == "ready":
            return data.get("solution", {})
    raise Exception("Captcha solving timeout")


# ==================== CAPSOLVER ====================

async def capsolver_submit(solver, task: Dict) -> str:
    """Submit task to CapSolver, return task ID"""
    resp = await solver.http.post(f"{solver.CAPSOLVER_API}/createTask",
        json={"clientKey": solver._api_key, "task": task})
    data = resp.json()
    if data.get("errorId", 0) != 0:
        raise Exception(f"CapSolver error: {data.get('errorDescription', 'Unknown')}")
    return data["taskId"]


async def capsolver_result(solver, task_id: str, max_wait: int = 120) -> Dict:
    """Poll for result from CapSolver"""
    for _ in range(max_wait // 3):
        await asyncio.sleep(3)
        resp = await solver.http.post(f"{solver.CAPSOLVER_API}/getTaskResult",
            json={"clientKey": solver._api_key, "taskId": task_id})
        data = resp.json()
        if data.get("errorId", 0) != 0:
            raise Exception(f"CapSolver error: {data.get('errorDescription')}")
        if data.get("status") == "ready":
            return data.get("solution", {})
    raise Exception("Captcha solving timeout")


# ==================== NOPECHA ====================

async def nopecha_solve_hcaptcha(solver, sitekey: str, url: str) -> Dict:
    """Solve hCaptcha using NopeCHA Token API"""
    resp = await solver.http.post(f"{solver.NOPECHA_API}/v1/token/hcaptcha",
        headers={"Authorization": f"Bearer {solver._api_key}"},
        json={"sitekey": sitekey, "url": url})
    data = resp.json()
    if "error" in data:
        raise Exception(f"NopeCHA error: {data.get('message', data.get('error'))}")
    job_id = data.get("data")
    if not job_id:
        raise Exception(f"NopeCHA: No job ID returned: {data}")
    for _ in range(60):
        await asyncio.sleep(5)
        resp = await solver.http.get(f"{solver.NOPECHA_API}/v1/token/hcaptcha",
            headers={"Authorization": f"Bearer {solver._api_key}"}, params={"id": job_id})
        data = resp.json()
        if "error" in data:
            if data.get("error") == 14: continue
            raise Exception(f"NopeCHA error: {data.get('message', data.get('error'))}")
        if "data" in data and isinstance(data["data"], str) and len(data["data"]) > 50:
            return {"token": data["data"]}
    raise Exception("NopeCHA: Timeout waiting for solution")


async def nopecha_solve_recaptcha(solver, sitekey: str, url: str, version: str = "v2") -> Dict:
    """Solve reCAPTCHA using NopeCHA Token API"""
    endpoint = "recaptcha2" if version == "v2" else "recaptcha3"
    resp = await solver.http.post(f"{solver.NOPECHA_API}/v1/token/{endpoint}",
        headers={"Authorization": f"Bearer {solver._api_key}"},
        json={"sitekey": sitekey, "url": url})
    data = resp.json()
    if "error" in data:
        raise Exception(f"NopeCHA error: {data.get('message', data.get('error'))}")
    job_id = data.get("data")
    if not job_id:
        raise Exception(f"NopeCHA: No job ID returned")
    for _ in range(60):
        await asyncio.sleep(5)
        resp = await solver.http.get(f"{solver.NOPECHA_API}/v1/token/{endpoint}",
            headers={"Authorization": f"Bearer {solver._api_key}"}, params={"id": job_id})
        data = resp.json()
        if "error" in data:
            if data.get("error") == 14: continue
            raise Exception(f"NopeCHA error: {data.get('message')}")
        if "data" in data and isinstance(data["data"], str) and len(data["data"]) > 50:
            return {"token": data["data"]}
    raise Exception("NopeCHA: Timeout waiting for solution")


# ==================== NEXTCAPTCHA ====================

async def nextcaptcha_submit(solver, task: Dict) -> str:
    """Submit task to NextCaptcha, return task ID"""
    resp = await solver.http.post(f"{solver.NEXTCAPTCHA_API}/createTask",
        json={"clientKey": solver._api_key, "task": task})
    data = resp.json()
    if data.get("errorId", 0) != 0:
        raise Exception(f"NextCaptcha error: {data.get('errorDescription', 'Unknown')}")
    return data["taskId"]


async def nextcaptcha_result(solver, task_id: str, max_wait: int = 120) -> Dict:
    """Poll for result from NextCaptcha"""
    for _ in range(max_wait // 5):
        await asyncio.sleep(5)
        resp = await solver.http.post(f"{solver.NEXTCAPTCHA_API}/getTaskResult",
            json={"clientKey": solver._api_key, "taskId": task_id})
        data = resp.json()
        if data.get("errorId", 0) != 0:
            raise Exception(f"NextCaptcha error: {data.get('errorDescription')}")
        if data.get("status") == "ready":
            return data.get("solution", {})
    raise Exception("Captcha solving timeout")


# ==================== NOCAPTCHAAI ====================

async def nocaptchaai_solve_hcaptcha(solver, sitekey: str, url: str) -> Dict:
    """Solve hCaptcha using NoCaptchaAI"""
    resp = await solver.http.post(f"{solver.NOCAPTCHAAI_API}/hcaptcha",
        headers={"apikey": solver._api_key}, json={"sitekey": sitekey, "url": url})
    data = resp.json()
    if data.get("status") == "solved":
        return {"token": data.get("token")}
    elif data.get("status") == "processing":
        task_id = data.get("id")
        for _ in range(24):
            await asyncio.sleep(5)
            resp = await solver.http.get(f"{solver.NOCAPTCHAAI_API}/status/{task_id}",
                headers={"apikey": solver._api_key})
            data = resp.json()
            if data.get("status") == "solved": return {"token": data.get("token")}
            elif data.get("status") == "failed":
                raise Exception(f"NoCaptchaAI failed: {data.get('message')}")
    raise Exception(f"NoCaptchaAI error: {data.get('message', 'Unknown')}")


async def nocaptchaai_solve_recaptcha(solver, sitekey: str, url: str, version: str = "v2") -> Dict:
    """Solve reCAPTCHA using NoCaptchaAI"""
    endpoint = "recaptcha" if version == "v2" else "recaptchav3"
    resp = await solver.http.post(f"{solver.NOCAPTCHAAI_API}/{endpoint}",
        headers={"apikey": solver._api_key}, json={"sitekey": sitekey, "url": url})
    data = resp.json()
    if data.get("status") == "solved":
        return {"token": data.get("token")}
    elif data.get("status") == "processing":
        task_id = data.get("id")
        for _ in range(24):
            await asyncio.sleep(5)
            resp = await solver.http.get(f"{solver.NOCAPTCHAAI_API}/status/{task_id}",
                headers={"apikey": solver._api_key})
            data = resp.json()
            if data.get("status") == "solved": return {"token": data.get("token")}
            elif data.get("status") == "failed":
                raise Exception(f"NoCaptchaAI failed: {data.get('message')}")
    raise Exception(f"NoCaptchaAI error: {data.get('message', 'Unknown')}")
