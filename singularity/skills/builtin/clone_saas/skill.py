"""
Clone SaaS Skill

Orchestrates the full pipeline: analyze a SaaS target, generate a Next.js clone,
push to GitHub, and deploy to Vercel.
"""

import json
import os
from pathlib import Path
from typing import Dict

import httpx

from singularity.skills.base import Skill, SkillResult, SkillManifest, SkillAction

try:
    from anthropic import AsyncAnthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False


def _a(n, d, p, cost=0.0, dur=30, prob=0.85):
    return SkillAction(name=n, description=d, parameters=p, estimated_cost=cost,
                       estimated_duration_seconds=dur, success_probability=prob)


def _p(n, t, r, d):
    return {n: {"type": t, "required": r, "description": d}}


SYSTEM_PROMPT = (
    "You are an expert full-stack developer specializing in Next.js 14 (App Router), "
    "TypeScript, Tailwind CSS, and Supabase. You produce clean, production-ready code. "
    "When generating multiple files, separate them with '// file: path/to/file.tsx' markers."
)


class CloneSaasSkill(Skill):
    """
    Clone a SaaS application: analyze → plan → generate code → push to GitHub → deploy to Vercel.

    Required credentials:
    - GITHUB_TOKEN: GitHub personal access token
    - VERCEL_TOKEN: Vercel API token

    Optional (if LLM not injected via set_llm):
    - ANTHROPIC_API_KEY: For LLM code generation and browser analysis
    """

    GITHUB_API = "https://api.github.com"
    VERCEL_API = "https://api.vercel.com"

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="clone_saas",
            name="Clone SaaS",
            version="1.0.0",
            category="dev",
            description="Analyze a SaaS site, generate a Next.js + Tailwind + Supabase clone, push to GitHub, deploy to Vercel",
            required_credentials=[],
            install_cost=0,
            actions=[
                _a("analyze_saas", "Analyze a SaaS website to extract features, pages, design, and pricing", {
                    **_p("url", "string", True, "URL of the SaaS to analyze"),
                    **_p("depth", "string", False, "Analysis depth: quick, standard, deep (default: standard)")
                }, cost=0.10, dur=60),
                _a("generate_plan", "Generate a structured build plan from analysis results", {
                    **_p("project_name", "string", True, "Name for the clone project"),
                    **_p("customizations", "string", False, "Custom instructions for the plan")
                }, cost=0.08, dur=30),
                _a("generate_page", "Generate a single Next.js page with components", {
                    **_p("project_name", "string", True, "Project name"),
                    **_p("page_name", "string", True, "Page to generate (from plan)")
                }, cost=0.06, dur=20),
                _a("generate_api", "Generate a single API route with Supabase migration SQL", {
                    **_p("project_name", "string", True, "Project name"),
                    **_p("route_name", "string", True, "API route to generate (from plan)")
                }, cost=0.06, dur=20),
                _a("create_repo", "Create a GitHub repo and push scaffold files", {
                    **_p("project_name", "string", True, "Project name"),
                    **_p("repo_name", "string", False, "GitHub repo name (default: project_name)"),
                    **_p("private", "boolean", False, "Private repo? (default: true)")
                }, cost=0.0, dur=15),
                _a("push_files", "Push all generated files to GitHub in one atomic commit", {
                    **_p("project_name", "string", True, "Project name"),
                    **_p("commit_message", "string", False, "Commit message (default: auto-generated)")
                }, cost=0.0, dur=20),
                _a("deploy_to_vercel", "Create Vercel project linked to GitHub repo and trigger deploy", {
                    **_p("project_name", "string", True, "Project name"),
                    **_p("env_vars", "object", False, "Environment variables to set on Vercel")
                }, cost=0.0, dur=30),
                _a("get_status", "Get pipeline progress and next step suggestion", {
                    **_p("project_name", "string", True, "Project name")
                }, cost=0.0, dur=2, prob=0.99),
            ]
        )

    def __init__(self, credentials: Dict[str, str] = None, llm=None, llm_type: str = None, model: str = None):
        super().__init__(credentials)
        self.http = httpx.AsyncClient()
        self._workspace = Path(os.environ.get("AGENT_WORKSPACE", "/tmp/agent_workspace")) / "clone_saas"

        if llm:
            self.llm = llm
            self.llm_type = llm_type or "anthropic"
            self.model = model or "claude-sonnet-4-20250514"
        else:
            self._init_llm()

    def _init_llm(self):
        api_key = self.credentials.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
        if api_key and HAS_ANTHROPIC:
            self.llm = AsyncAnthropic(api_key=api_key)
            self.llm_type = "anthropic"
            self.model = "claude-sonnet-4-20250514"
        else:
            self.llm = None
            self.llm_type = "none"

    def set_llm(self, llm, llm_type: str, model: str):
        self.llm = llm
        self.llm_type = llm_type
        self.model = model

    async def _generate(self, prompt: str, system: str = None, max_tokens: int = 8000) -> str:
        if self.llm_type == "anthropic":
            response = await self.llm.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system or SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        raise RuntimeError("No LLM available — inject one via set_llm() or provide ANTHROPIC_API_KEY")

    def _github_headers(self) -> Dict:
        token = self.credentials.get("GITHUB_TOKEN") or os.environ.get("GITHUB_TOKEN")
        return {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28"}

    def _vercel_headers(self) -> Dict:
        token = self.credentials.get("VERCEL_TOKEN") or os.environ.get("VERCEL_TOKEN")
        return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    # ── State management ──────────────────────────────────────────────

    def _state_path(self, project_name: str) -> Path:
        return self._workspace / project_name / "state.json"

    def _load_state(self, project_name: str) -> Dict:
        path = self._state_path(project_name)
        if path.exists():
            return json.loads(path.read_text())
        return {
            "project_name": project_name,
            "analysis": None,
            "plan": None,
            "generated_files": {},
            "generation_status": {},
            "repo": None,
            "deployment": None,
        }

    def _save_state(self, project_name: str, state: Dict):
        path = self._state_path(project_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=2))
        tmp.rename(path)

    # ── Execute dispatch ──────────────────────────────────────────────

    async def execute(self, action: str, params: Dict) -> SkillResult:
        from . import analysis, codegen, deploy

        try:
            dispatch = {
                "analyze_saas": lambda: analysis.analyze_saas(self, params),
                "generate_plan": lambda: codegen.generate_plan(self, params),
                "generate_page": lambda: codegen.generate_page(self, params),
                "generate_api": lambda: codegen.generate_api(self, params),
                "create_repo": lambda: deploy.create_repo(self, params),
                "push_files": lambda: deploy.push_files(self, params),
                "deploy_to_vercel": lambda: deploy.deploy_to_vercel(self, params),
                "get_status": lambda: analysis.get_status(self, params),
            }
            handler = dispatch.get(action)
            if not handler:
                return SkillResult(success=False, message=f"Unknown action: {action}")
            return await handler()
        except Exception as e:
            return SkillResult(success=False, message=f"clone_saas error: {str(e)}")

    async def close(self):
        await self.http.aclose()
