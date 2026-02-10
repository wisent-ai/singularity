"""
Clone SaaS — Analysis & Status Actions

analyze_saas: Browser-use or httpx fallback to extract SaaS features, pages, design, and pricing.
get_status:   Pipeline progress and next-step suggestion.
"""

import json
import os
from typing import Dict

import httpx

from singularity.skills.base import SkillResult

try:
    from browser_use import Agent as BrowserAgent, Browser, BrowserConfig
    from langchain_anthropic import ChatAnthropic
    HAS_BROWSER_USE = True
except ImportError:
    HAS_BROWSER_USE = False


# ── analyze_saas ──────────────────────────────────────────────────────

async def analyze_saas(skill, params: Dict) -> SkillResult:
    """Analyze a SaaS website to extract features, pages, design, and pricing."""
    url = params.get("url")
    if not url:
        return SkillResult(success=False, message="url parameter is required")

    depth = params.get("depth", "standard")
    project_name = url.split("//")[-1].split("/")[0].replace(".", "-").replace("www-", "")

    if HAS_BROWSER_USE:
        analysis = await _analyze_with_browser(skill, url, depth)
    else:
        analysis = await _analyze_with_httpx(skill, url, depth)

    if not analysis:
        return SkillResult(success=False, message="Analysis failed — could not extract data from target")

    # Persist to state
    state = skill._load_state(project_name)
    state["analysis"] = analysis
    state["analysis"]["url"] = url
    state["analysis"]["depth"] = depth
    skill._save_state(project_name, state)

    return SkillResult(
        success=True,
        message=f"Analyzed {url} ({depth}): {len(analysis.get('pages', []))} pages, "
                f"{len(analysis.get('features', []))} features found",
        data={"project_name": project_name, "analysis": analysis}
    )


async def _analyze_with_browser(skill, url: str, depth: str) -> Dict:
    """Use browser-use agent to navigate and extract SaaS info."""
    api_key = skill.credentials.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return await _analyze_with_httpx(skill, url, depth)

    max_steps = {"quick": 8, "standard": 15, "deep": 25}.get(depth, 15)

    task = (
        f"Navigate to {url} and analyze this SaaS application. Extract:\n"
        "1. All visible pages/routes (homepage, pricing, features, about, login, dashboard if visible)\n"
        "2. Key features and their descriptions\n"
        "3. Design system: colors (primary, secondary, accent), fonts, layout patterns\n"
        "4. Pricing tiers (names, prices, features per tier)\n"
        "5. Navigation structure\n"
        "6. Any API integrations mentioned\n"
        "7. Authentication methods visible (email, OAuth, etc.)\n\n"
        "Return the results as a JSON object with keys: pages, features, design, pricing, "
        "navigation, integrations, auth_methods, summary"
    )

    try:
        config = BrowserConfig(headless=True, disable_security=True)
        browser = Browser(config=config)
        llm = ChatAnthropic(model="claude-sonnet-4-20250514", api_key=api_key,  max_tokens=8000)
        agent = BrowserAgent(task=task, llm=llm, browser=browser, max_actions_per_step=3)
        result = await agent.run(max_steps=max_steps)
        await browser.close()

        extracted = result.extracted_content() if hasattr(result, "extracted_content") else None
        if extracted:
            # Try to parse JSON from extracted content
            return _parse_analysis_json(extracted)
    except Exception:
        pass

    # Fallback
    return await _analyze_with_httpx(skill, url, depth)


async def _analyze_with_httpx(skill, url: str, depth: str) -> Dict:
    """Fallback: fetch HTML and use LLM to parse content."""
    try:
        resp = await skill.http.get(url, follow_redirects=True)
        html = resp.text[:50000]  # Cap at 50k chars
    except Exception as e:
        return {"error": f"Could not fetch {url}: {str(e)}", "pages": [], "features": []}

    if skill.llm_type == "none":
        # No LLM available — return raw structure
        return {
            "pages": ["/"],
            "features": ["Unknown — no LLM available for analysis"],
            "design": {},
            "pricing": [],
            "navigation": [],
            "integrations": [],
            "auth_methods": [],
            "summary": f"HTML fetched from {url} ({len(html)} chars), but no LLM to analyze it.",
            "raw_html_length": len(html),
        }

    prompt = (
        f"Analyze this HTML from {url} and extract a JSON object with these keys:\n"
        "- pages: array of page routes found (links in nav, footer, etc.)\n"
        "- features: array of {{name, description}} objects for key product features\n"
        "- design: {{primary_color, secondary_color, accent_color, font_family, layout_style}}\n"
        "- pricing: array of {{tier_name, price, period, features[]}} objects\n"
        "- navigation: array of nav items\n"
        "- integrations: array of third-party services mentioned\n"
        "- auth_methods: array of auth methods visible\n"
        "- summary: one-paragraph summary of what this SaaS does\n\n"
        "Return ONLY valid JSON, no markdown fences.\n\n"
        f"HTML:\n{html}"
    )

    try:
        raw = await skill._generate(prompt, system="You are a SaaS analyst. Return only valid JSON.")
        return _parse_analysis_json(raw)
    except Exception as e:
        return {
            "pages": ["/"],
            "features": [],
            "design": {},
            "pricing": [],
            "summary": f"LLM analysis failed: {str(e)}",
        }


def _parse_analysis_json(raw) -> Dict:
    """Best-effort parse JSON from LLM output."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, list):
        # Might be a list of extracted strings
        raw = " ".join(str(x) for x in raw)

    text = str(raw).strip()
    # Strip markdown fences
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
    if text.endswith("```"):
        text = text.rsplit("```", 1)[0]
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON object in text
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass
    return {"summary": text[:500], "pages": [], "features": []}


# ── get_status ────────────────────────────────────────────────────────

async def get_status(skill, params: Dict) -> SkillResult:
    """Get pipeline progress and next step suggestion."""
    project_name = params.get("project_name")
    if not project_name:
        return SkillResult(success=False, message="project_name parameter is required")

    state = skill._load_state(project_name)

    steps = []
    next_action = None

    # 1. Analysis
    if state.get("analysis"):
        steps.append({"step": "analyze_saas", "status": "done",
                       "detail": f"{len(state['analysis'].get('pages', []))} pages found"})
    else:
        steps.append({"step": "analyze_saas", "status": "pending"})
        if not next_action:
            next_action = "analyze_saas"

    # 2. Plan
    if state.get("plan"):
        steps.append({"step": "generate_plan", "status": "done",
                       "detail": f"{len(state['plan'].get('pages', []))} pages planned"})
    else:
        steps.append({"step": "generate_plan", "status": "pending"})
        if not next_action:
            next_action = "generate_plan"

    # 3. Code generation
    plan = state.get("plan") or {}
    planned_pages = [p.get("name") or p for p in plan.get("pages", [])]
    planned_apis = [r.get("name") or r for r in plan.get("api_routes", [])]
    gen_status = state.get("generation_status", {})

    pages_done = sum(1 for p in planned_pages if gen_status.get(f"page:{p}") == "done")
    apis_done = sum(1 for r in planned_apis if gen_status.get(f"api:{r}") == "done")

    if planned_pages or planned_apis:
        total = len(planned_pages) + len(planned_apis)
        done = pages_done + apis_done
        if done == total:
            steps.append({"step": "generate_code", "status": "done", "detail": f"{done}/{total}"})
        else:
            steps.append({"step": "generate_code", "status": "in_progress", "detail": f"{done}/{total}"})
            if not next_action:
                # Find next ungenerated page or API
                for p in planned_pages:
                    if gen_status.get(f"page:{p}") != "done":
                        next_action = f"generate_page(page_name={p})"
                        break
                if not next_action:
                    for r in planned_apis:
                        if gen_status.get(f"api:{r}") != "done":
                            next_action = f"generate_api(route_name={r})"
                            break
    else:
        steps.append({"step": "generate_code", "status": "pending"})

    # 4. Repo
    if state.get("repo"):
        steps.append({"step": "create_repo", "status": "done",
                       "detail": state["repo"].get("full_name")})
    else:
        steps.append({"step": "create_repo", "status": "pending"})
        if not next_action:
            next_action = "create_repo"

    # 5. Push
    pushed = state.get("repo", {}).get("pushed", False)
    if pushed:
        steps.append({"step": "push_files", "status": "done"})
    else:
        steps.append({"step": "push_files", "status": "pending"})
        if not next_action:
            next_action = "push_files"

    # 6. Deploy
    if state.get("deployment"):
        steps.append({"step": "deploy_to_vercel", "status": "done",
                       "detail": state["deployment"].get("url")})
    else:
        steps.append({"step": "deploy_to_vercel", "status": "pending"})
        if not next_action:
            next_action = "deploy_to_vercel"

    done_count = sum(1 for s in steps if s["status"] == "done")
    total_steps = len(steps)

    return SkillResult(
        success=True,
        message=f"Pipeline progress: {done_count}/{total_steps} steps complete"
                + (f" — next: {next_action}" if next_action else " — all done!"),
        data={
            "project_name": project_name,
            "steps": steps,
            "next_action": next_action,
            "progress": f"{done_count}/{total_steps}",
            "generated_files_count": len(state.get("generated_files", {})),
        }
    )
