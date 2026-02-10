"""GitHub Skill - repos, issues, gists, search."""

import httpx
from typing import Dict, List
from singularity.skills.base import Skill, SkillResult, SkillManifest, SkillAction
from . import handlers


def _a(n, d, p=None, prob=0.95, dur=5):
    return SkillAction(name=n, description=d, parameters=p or {}, estimated_cost=0,
                       estimated_duration_seconds=dur, success_probability=prob)

def _p(n, t, r, d):
    return {n: {"type": t, "required": r, "description": d}}


class GitHubSkill(Skill):
    """Skill for GitHub API interactions."""

    API_BASE = "https://api.github.com"

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="github", name="GitHub Management", version="1.0.0", category="dev",
            description="Create repos, manage issues, and interact with GitHub",
            required_credentials=["GITHUB_TOKEN"], install_cost=0,
            actions=[
                _a("create_repo", "Create a new GitHub repository", {
                    **_p("name", "string", True, "Repository name"),
                    **_p("description", "string", False, "Repo description"),
                    **_p("private", "boolean", False, "Private repo? (default: false)")}),
                _a("create_issue", "Create an issue in a repository", {
                    **_p("repo", "string", True, "Repo (owner/name)"),
                    **_p("title", "string", True, "Issue title"),
                    **_p("body", "string", False, "Issue body"),
                    **_p("labels", "array", False, "Labels to add")}, dur=3),
                _a("search_repos", "Search GitHub repositories", {
                    **_p("query", "string", True, "Search query"),
                    **_p("sort", "string", False, "Sort by (stars, forks, updated)"),
                    **_p("limit", "integer", False, "Max results")}, 0.9),
                _a("search_issues", "Search GitHub issues", {
                    **_p("query", "string", True, "Search query"),
                    **_p("state", "string", False, "State (open, closed, all)"),
                    **_p("labels", "string", False, "Label filter")}, 0.9),
                _a("fork_repo", "Fork a repository",
                   _p("repo", "string", True, "Repo to fork (owner/name)"), 0.9, 10),
                _a("star_repo", "Star a repository",
                   _p("repo", "string", True, "Repo to star (owner/name)"), dur=2),
                _a("get_user", "Get GitHub user info",
                   _p("username", "string", False, "Username (default: authenticated user)"), dur=2),
                _a("create_gist", "Create a GitHub Gist", {
                    **_p("description", "string", False, "Gist description"),
                    **_p("files", "object", True, "Files {filename: content}"),
                    **_p("public", "boolean", False, "Public gist? (default: true)")}, dur=3),
            ])

    def __init__(self, credentials: Dict[str, str] = None):
        super().__init__(credentials)
        self.http = httpx.AsyncClient()

    def _get_headers(self) -> Dict:
        return {"Authorization": f"Bearer {self.credentials.get('GITHUB_TOKEN')}",
                "Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}

    async def execute(self, action: str, params: Dict) -> SkillResult:
        if not self.check_credentials():
            missing = self.get_missing_credentials()
            return SkillResult(success=False, message=f"Missing credentials: {missing}")
        try:
            dispatch = {
                "create_repo": lambda: handlers.create_repo(self, params.get("name"),
                    params.get("description"), params.get("private", False)),
                "create_issue": lambda: handlers.create_issue(self, params.get("repo"),
                    params.get("title"), params.get("body"), params.get("labels", [])),
                "search_repos": lambda: handlers.search_repos(self, params.get("query"),
                    params.get("sort"), params.get("limit", 10)),
                "search_issues": lambda: handlers.search_issues(self, params.get("query"),
                    params.get("state", "open"), params.get("labels")),
                "fork_repo": lambda: handlers.fork_repo(self, params.get("repo")),
                "star_repo": lambda: handlers.star_repo(self, params.get("repo")),
                "get_user": lambda: handlers.get_user(self, params.get("username")),
                "create_gist": lambda: handlers.create_gist(self, params.get("description"),
                    params.get("files"), params.get("public", True)),
            }
            handler = dispatch.get(action)
            if not handler:
                return SkillResult(success=False, message=f"Unknown action: {action}")
            return await handler()
        except Exception as e:
            return SkillResult(success=False, message=f"GitHub error: {str(e)}")

    async def close(self):
        await self.http.aclose()
