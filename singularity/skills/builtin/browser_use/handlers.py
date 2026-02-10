"""Handler functions for BrowserUseSkill actions."""

from singularity.skills.base import SkillResult


async def run_task(skill, params) -> SkillResult:
    task = params.get("task", "")
    max_steps = int(params.get("max_steps", 10))
    start_url = params.get("start_url")
    if not task:
        return SkillResult(success=False, message="Task description required")
    result = await skill._run_browser_task(task, max_steps, start_url)
    return SkillResult(success=result.get("success", False),
                       message="Browser task completed", data=result, cost=0.05)


async def extract_data(skill, params) -> SkillResult:
    url = params.get("url", "")
    data_description = params.get("data_description", "")
    if not url or not data_description:
        return SkillResult(success=False, message="URL and data_description required")
    task = f"Go to {url} and extract the following data: {data_description}. Return the data in a structured format."
    result = await skill._run_browser_task(task, max_steps=8)
    return SkillResult(success=result.get("success", False),
                       message="Data extraction completed", data=result, cost=0.03)


async def fill_form(skill, params) -> SkillResult:
    url = params.get("url", "")
    form_data = params.get("form_data", {})
    submit = params.get("submit", True)
    if not url or not form_data:
        return SkillResult(success=False, message="URL and form_data required")
    fields = ", ".join([f"{k}: {v}" for k, v in form_data.items()])
    task = f"Go to {url} and fill out the form with: {fields}."
    if submit:
        task += " Then submit the form."
    result = await skill._run_browser_task(task, max_steps=15)
    return SkillResult(success=result.get("success", False),
                       message="Form filled" + (" and submitted" if submit else ""),
                       data=result, cost=0.04)


async def screenshot(skill, params) -> SkillResult:
    url = params.get("url", "")
    filename = params.get("filename", "screenshot.png")
    if not url:
        return SkillResult(success=False, message="URL required")
    task = f"Go to {url} and take a screenshot."
    result = await skill._run_browser_task(task, max_steps=3)
    browser = await skill._get_browser()
    if browser.page:
        await browser.page.screenshot(path=filename)
    return SkillResult(success=True, message=f"Screenshot saved to {filename}",
                       data={"path": filename, **result}, cost=0.01)


async def search_and_click(skill, params) -> SkillResult:
    search_engine = params.get("search_engine", "google")
    query = params.get("query", "")
    click_result = params.get("click_result", "first result")
    if not query:
        return SkillResult(success=False, message="Search query required")
    engine_urls = {"google": "https://google.com", "bing": "https://bing.com",
                   "duckduckgo": "https://duckduckgo.com"}
    start_url = engine_urls.get(search_engine, "https://google.com")
    task = f"Search for '{query}' and click on the {click_result}."
    result = await skill._run_browser_task(task, max_steps=10, start_url=start_url)
    return SkillResult(success=result.get("success", False),
                       message=f"Searched '{query}' and clicked result", data=result, cost=0.05)


async def monitor_page(skill, params) -> SkillResult:
    url = params.get("url", "")
    condition = params.get("condition", "")
    max_wait = int(params.get("max_wait_seconds", 60))
    if not url or not condition:
        return SkillResult(success=False, message="URL and condition required")
    task = f"Go to {url} and check if: {condition}. If the condition is met, report it. Check the page periodically."
    max_steps = min(max_wait // 5, 20)
    result = await skill._run_browser_task(task, max_steps=max_steps)
    return SkillResult(success=result.get("success", False),
                       message="Page monitoring completed", data=result, cost=0.1)
