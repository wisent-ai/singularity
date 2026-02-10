"""
Captcha Solver - AI Vision Methods

Contains detect_captcha, solve_with_ai, solve_recaptcha_challenge,
solve_recaptcha_in_iframe, and vision analysis helpers.
"""

import os
import re
import json
import asyncio
import base64
import random
import httpx
from typing import Dict, Optional, List
from singularity.skills.base import SkillResult
from .skill import human_like_mouse_move


async def detect_captcha(solver, page) -> SkillResult:
    info = await _find_captcha_on_page(page)
    if info:
        return SkillResult(success=True, message=f"CAPTCHA detected: {info['type']}", data=info)
    return SkillResult(success=True, message="No CAPTCHA detected", data={"detected": False})


async def _find_captcha_on_page(page) -> Optional[Dict]:
    types = {'hcaptcha': ['hcaptcha.com'], 'recaptcha': ['recaptcha', 'google.com/recaptcha'],
             'arkose': ['arkoselabs', 'funcaptcha']}
    try:
        for iframe in await page.query_selector_all('iframe'):
            src = (await iframe.get_attribute('src') or '').lower()
            for ctype, indicators in types.items():
                if any(ind in src for ind in indicators):
                    box = await iframe.bounding_box()
                    if box and box['width'] > 200 and box['height'] > 200:
                        return {'detected': True, 'type': ctype, 'iframe_src': src[:100], 'bounds': box}
        content = (await page.content()).lower()
        for p in ['pick the', 'click on the', 'select all', 'verify you are human']:
            if p in content:
                return {'detected': True, 'type': 'unknown', 'iframe_src': None, 'bounds': None}
    except Exception as e:
        print(f"CAPTCHA detection error: {e}")
    return None


async def solve_with_ai(solver, page, max_attempts: int = 3) -> SkillResult:
    for attempt in range(max_attempts):
        if not await _find_captcha_on_page(page):
            return SkillResult(success=True, message="No CAPTCHA detected", data={"attempts": attempt})
        ss = base64.standard_b64encode(await page.screenshot()).decode('utf-8')
        positions = await _analyze_vision(solver, ss, _default_prompt())
        if not positions: continue
        vp = page.viewport_size or {'width': 1280, 'height': 720}
        gx, gy, gw, gh = (vp['width']-500)/2+15, 220, 460, 320
        gs = 5 if max(positions) > 9 else 3
        cw, ch = gw/gs, gh/gs
        for pos in positions:
            r, c = (pos-1)//gs, (pos-1)%gs
            await page.mouse.click(gx+c*cw+cw/2, gy+r*ch+ch/2); await asyncio.sleep(0.3)
        for txt in ['Next', 'Submit', 'Verify', 'Check', 'Done', 'Skip']:
            btn = await page.query_selector(f'button:has-text("{txt}")')
            if btn and await btn.is_visible(): await btn.click(); break
        await asyncio.sleep(2)
        if not await _find_captcha_on_page(page):
            return SkillResult(success=True, message=f"Solved in {attempt+1} attempt(s)",
                data={"attempts": attempt+1, "positions": positions})
    return SkillResult(success=False, message=f"Failed after {max_attempts} attempts")


async def solve_recaptcha_challenge(solver, bframe_handle, bframe, anchor_frame,
                                     max_attempts=5, parent_frame=None, page=None) -> SkillResult:
    """Solve reCAPTCHA image challenge using AI vision with direct bframe access."""
    for attempt in range(max_attempts):
        if attempt > 0 and parent_frame:
            bframe_handle, bframe = await _relocate_bframe(bframe_handle, bframe, parent_frame)
            if not bframe: continue
        instruction = await _ev(bframe, "document.querySelector('.rc-imageselect-desc, .rc-imageselect-desc-no-canonical, .rc-imageselect-desc-wrapper')?.innerText||''")
        ss = await _bframe_screenshot(bframe_handle, page)
        if not ss: continue
        gs = await _grid_size(bframe)
        positions = await _analyze_vision(solver, ss, _rc_prompt(instruction, gs))
        if positions is None: continue
        if positions:
            mx, my = random.uniform(200, 400), random.uniform(200, 400)
            for pos in positions:
                mx, my = await _click_tile(bframe, bframe_handle, page, pos, gs, mx, my)
                await asyncio.sleep(random.uniform(0.3, 0.7))
        await asyncio.sleep(random.uniform(0.3, 0.6))
        await _click_verify(bframe, bframe_handle, page)
        await asyncio.sleep(2)
        if await _ev(anchor_frame, "document.querySelector('.recaptcha-checkbox')?.getAttribute('aria-checked')==='true'"):
            return SkillResult(success=True, message=f"reCAPTCHA solved in {attempt+1} attempts",
                data={"service": "ai_vision", "attempts": attempt+1})
        err = await _ev(bframe, "(() => { let e = document.querySelector('.rc-imageselect-error-select-more, .rc-imageselect-error-dynamic-more, .rc-imageselect-incorrect-response'); return e && e.offsetParent !== null ? e.textContent : null; })()")
        if err: print(f"  [AI VISION] Error: {err}")
    return SkillResult(success=False, message=f"Failed after {max_attempts} attempts")


async def solve_recaptcha_in_iframe(solver, page, max_attempts=5) -> SkillResult:
    """Find reCAPTCHA bframe in page and solve it."""
    bframe, bframe_handle = await _find_bframe(page)
    if not bframe:
        token = await _check_solved(page)
        if token: return SkillResult(success=True, message="Already solved", data={"token": token})
        return SkillResult(success=False, message="Could not find challenge iframe")
    anchor_frame = None
    for iframe in await page.query_selector_all('iframe'):
        src = await iframe.get_attribute('src') or ''
        if 'anchor' in src:
            anchor_frame = await iframe.content_frame()
            break
    return await solve_recaptcha_challenge(solver, bframe_handle, bframe,
        anchor_frame, max_attempts, page=page)


# ==================== HELPERS ====================

async def _ev(frame, script):
    try: return await frame.evaluate(script)
    except: return None

async def _relocate_bframe(handle, frame, parent):
    try: await frame.evaluate('()=>1'); return handle, frame
    except: pass
    h = await parent.query_selector('iframe[src*="bframe"]')
    if h:
        f = await h.content_frame()
        if f: return h, f
    return handle, None

async def _bframe_screenshot(handle, page) -> Optional[str]:
    try:
        try: await handle.scroll_into_view_if_needed()
        except: pass
        await asyncio.sleep(0.5)
        return base64.standard_b64encode(await handle.screenshot()).decode('utf-8')
    except:
        try:
            box = await handle.bounding_box()
            if box and page:
                return base64.standard_b64encode(await page.screenshot(clip=box)).decode('utf-8')
        except: pass
    return None

async def _grid_size(bframe) -> int:
    info = await _ev(bframe, """(() => {
        let t = document.querySelector('table.rc-imageselect-table, table.rc-imageselect-table-33, table.rc-imageselect-table-44');
        return t ? { cols: t.querySelectorAll('tr')[0]?.querySelectorAll('td').length || 3 } : null;
    })()""")
    return (info or {}).get('cols', 3)

async def _find_bframe(page):
    for iframe in await page.query_selector_all('iframe'):
        src = await iframe.get_attribute('src') or ''
        if 'bframe' in src or ('recaptcha' in src and 'anchor' not in src):
            return await iframe.content_frame(), iframe
    for outer in await page.query_selector_all('iframe'):
        try:
            of = await outer.content_frame()
            if of:
                for inner in await of.query_selector_all('iframe'):
                    src = await inner.get_attribute('src') or ''
                    if 'bframe' in src: return await inner.content_frame(), inner
        except: continue
    return None, None

async def _check_solved(page) -> Optional[str]:
    for iframe in await page.query_selector_all('iframe'):
        if 'anchor' in (await iframe.get_attribute('src') or ''):
            af = await iframe.content_frame()
            if af:
                cb = await af.query_selector('.recaptcha-checkbox')
                if cb and await cb.get_attribute('aria-checked') == 'true':
                    return await _ev(af, "document.querySelector('textarea[name=\"g-recaptcha-response\"]')?.value")
    return None

async def _click_tile(bframe, handle, page, pos, gs, mx, my):
    row, col = (pos-1)//gs, (pos-1)%gs
    if page:
        ti = await _ev(bframe, f"(() => {{ let t = document.querySelector('table.rc-imageselect-table, table.rc-imageselect-table-33, table.rc-imageselect-table-44'); if(!t) return null; let rows=t.querySelectorAll('tr'); if(rows.length<={row}) return null; let cells=rows[{row}].querySelectorAll('td'); if(cells.length<={col}) return null; let r=cells[{col}].getBoundingClientRect(); return {{x:r.left+r.width/2,y:r.top+r.height/2}}; }})()")
        if ti:
            bb = await handle.bounding_box()
            if bb:
                ax, ay = bb['x']+ti['x']+random.uniform(-10,10), bb['y']+ti['y']+random.uniform(-10,10)
                await human_like_mouse_move(page, mx, my, ax, ay)
                await asyncio.sleep(random.uniform(0.05, 0.15))
                await page.mouse.click(ax, ay); return ax, ay
    await _ev(bframe, f"(() => {{ let t = document.querySelector('table.rc-imageselect-table, table.rc-imageselect-table-33, table.rc-imageselect-table-44'); if(!t) return; let rows=t.querySelectorAll('tr'); if(rows.length>{row}) {{ let cells=rows[{row}].querySelectorAll('td'); if(cells.length>{col}) cells[{col}].click(); }} }})()")
    return mx, my

async def _click_verify(bframe, handle, page):
    if page:
        vi = await _ev(bframe, "(() => { let b=document.querySelector('#recaptcha-verify-button'); if(b){let r=b.getBoundingClientRect(); return {x:r.left+r.width/2,y:r.top+r.height/2};} return null; })()")
        if vi:
            bb = await handle.bounding_box()
            if bb: await page.mouse.click(bb['x']+vi['x']+random.uniform(-5,5), bb['y']+vi['y']+random.uniform(-3,3)); return
    await _ev(bframe, "document.querySelector('#recaptcha-verify-button')?.click()")

def _default_prompt():
    return "This is a CAPTCHA. For 3x3: 1-9, for 5x5: 1-25, left-to-right top-to-bottom. RESPOND WITH ONLY A JSON ARRAY OF POSITION NUMBERS. Examples: [5] or [2,6]\nYour answer:"

def _rc_prompt(instruction, gs):
    gp = "1 2 3\n4 5 6\n7 8 9" if gs == 3 else "1  2  3  4\n5  6  7  8\n9  10 11 12\n13 14 15 16"
    return f'Solving reCAPTCHA. TARGET: "{instruction}"\nGrid: {gp}\nInclude ANY square with ANY part of target. If none match, return []. RESPOND WITH ONLY A JSON ARRAY.\nYour answer:'

def _parse_positions(answer: str) -> Optional[List]:
    try:
        p = json.loads(answer)
        if isinstance(p, list) and all(isinstance(x, int) for x in p): return p
    except: pass
    nums = [int(x) for x in re.findall(r'\d+', answer)]
    return nums if nums else None

async def _analyze_vision(solver, ss_b64: str, prompt: str) -> Optional[List]:
    try:
        async with httpx.AsyncClient() as c:
            if solver._model_provider == "anthropic":
                key = os.environ.get("ANTHROPIC_API_KEY")
                if not key: return None
                r = await c.post("https://api.anthropic.com/v1/messages",
                    headers={"x-api-key": key, "content-type": "application/json", "anthropic-version": "2023-06-01"},
                    json={"model": solver._model_name, "max_tokens": 100, "messages": [{"role": "user", "content": [
                        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": ss_b64}},
                        {"type": "text", "text": prompt}]}]})
                if r.status_code == 200: return _parse_positions(r.json()["content"][0]["text"].strip())
            elif solver._model_provider == "openai":
                key = os.environ.get("OPENAI_API_KEY")
                if not key: return None
                r = await c.post("https://api.openai.com/v1/chat/completions",
                    headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                    json={"model": solver._model_name or "gpt-4o", "max_tokens": 100, "messages": [{"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{ss_b64}"}},
                        {"type": "text", "text": prompt}]}]})
                if r.status_code == 200: return _parse_positions(r.json()["choices"][0]["message"]["content"].strip())
            elif solver._model_provider == "local":
                base = solver._api_base.replace("/v1", "")
                r = await c.post(f"{base}/api/generate", json={"model": solver._model_name or "llama3.2-vision:latest",
                    "prompt": prompt, "images": [ss_b64], "stream": False})
                if r.status_code == 200: return _parse_positions(r.json().get("response", "").strip())
            elif solver._model_provider == "vertex":
                try:
                    import google.auth, google.auth.transport.requests
                    creds, proj = google.auth.default()
                    pid = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GCP_PROJECT") or proj
                    if not pid: return None
                    loc = os.environ.get("VERTEX_LOCATION", "us-central1")
                    model = solver._model_name or "gemini-2.0-flash-001"
                    if creds.expired or not creds.token: creds.refresh(google.auth.transport.requests.Request())
                    r = await c.post(f"https://{loc}-aiplatform.googleapis.com/v1/projects/{pid}/locations/{loc}/publishers/google/models/{model}:generateContent",
                        headers={"Authorization": f"Bearer {creds.token}", "Content-Type": "application/json"},
                        json={"contents": [{"role": "user", "parts": [{"inlineData": {"mimeType": "image/png", "data": ss_b64}}, {"text": prompt}]}],
                              "generationConfig": {"maxOutputTokens": 100, "temperature": 0.1}})
                    if r.status_code == 200:
                        cands = r.json().get("candidates", [])
                        if cands:
                            parts = cands[0].get("content", {}).get("parts", [])
                            if parts: return _parse_positions(parts[0].get("text", "").strip())
                except Exception as e: print(f"  [AI VISION] Vertex: {e}")
    except Exception as e: print(f"  [AI VISION] Error: {e}")
    return None
