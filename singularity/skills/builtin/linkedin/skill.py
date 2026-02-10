"""
LinkedIn Skill

Post content, manage connections, and engage on LinkedIn via REST API.
Uses OAuth 2.0 bearer tokens for authentication.
"""

import os
from typing import Dict

import httpx

from singularity.skills.base import Skill, SkillResult, SkillManifest, SkillAction


def _a(n, d, p, cost=0.0, dur=5, prob=0.90):
    return SkillAction(name=n, description=d, parameters=p, estimated_cost=cost,
                       estimated_duration_seconds=dur, success_probability=prob)


def _p(n, t, r, d):
    return {n: {"type": t, "required": r, "description": d}}


class LinkedInSkill(Skill):
    """
    LinkedIn marketing and networking via the LinkedIn REST API.

    Required credentials:
    - LINKEDIN_ACCESS_TOKEN: OAuth 2.0 access token (from 3-legged or client-credentials flow)
    """

    API = "https://api.linkedin.com/v2"
    REST_API = "https://api.linkedin.com/rest"

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="linkedin",
            name="LinkedIn Marketing",
            version="2.1.0",
            category="social",
            description="Create accounts, post content, manage connections, and engage on LinkedIn",
            required_credentials=[],
            install_cost=0,
            actions=[
                _a("create_account", "Create a new LinkedIn account via browser automation", {
                    **_p("username", "string", False, "Preferred username (auto-generated if omitted)"),
                    **_p("email", "string", False, "Email to use (auto-generated if omitted)"),
                    **_p("password", "string", False, "Password (auto-generated if omitted)"),
                }, dur=120, prob=0.60),
                _a("get_profile", "Get your own LinkedIn profile or another member's", {
                    **_p("member_id", "string", False, "Member URN ID (default: authenticated user)"),
                }),
                _a("create_post", "Create a text or link post on LinkedIn", {
                    **_p("text", "string", True, "Post body text"),
                    **_p("link_url", "string", False, "URL to include as link attachment"),
                    **_p("visibility", "string", False, "PUBLIC or CONNECTIONS (default: PUBLIC)"),
                }, dur=10),
                _a("like_post", "React to a LinkedIn post", {
                    **_p("post_urn", "string", True, "Post URN (urn:li:share:ID or urn:li:ugcPost:ID)"),
                }),
                _a("comment", "Comment on a LinkedIn post", {
                    **_p("post_urn", "string", True, "Post URN to comment on"),
                    **_p("text", "string", True, "Comment text"),
                }),
                _a("search_posts", "Search recent LinkedIn posts by keyword", {
                    **_p("query", "string", True, "Search keywords"),
                    **_p("count", "integer", False, "Number of results (default: 10)"),
                }, dur=8),
                _a("get_connections", "Get your LinkedIn connections count and recent connections", {
                    **_p("count", "integer", False, "Number to retrieve (default: 10)"),
                }),
            ]
        )

    def __init__(self, credentials: Dict[str, str] = None):
        super().__init__(credentials)
        self.http = httpx.AsyncClient()

    def _headers(self) -> Dict:
        token = self.credentials.get("LINKEDIN_ACCESS_TOKEN") or os.environ.get("LINKEDIN_ACCESS_TOKEN")
        return {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "X-Restli-Protocol-Version": "2.0.0",
            "LinkedIn-Version": "202401",
        }

    async def _get_member_urn(self) -> str:
        """Get the authenticated user's member URN."""
        resp = await self.http.get(f"{self.API}/userinfo", headers=self._headers())
        if resp.status_code == 200:
            return f"urn:li:person:{resp.json().get('sub')}"
        resp2 = await self.http.get(f"{self.API}/me", headers=self._headers())
        if resp2.status_code == 200:
            return f"urn:li:person:{resp2.json().get('id')}"
        return None

    async def execute(self, action: str, params: Dict) -> SkillResult:
        from . import actions

        try:
            dispatch = {
                "create_account": lambda: actions.create_account(self, params),
                "get_profile": lambda: actions.get_profile(self, params),
                "create_post": lambda: actions.create_post(self, params),
                "like_post": lambda: actions.like_post(self, params),
                "comment": lambda: actions.comment(self, params),
                "search_posts": lambda: actions.search_posts(self, params),
                "get_connections": lambda: actions.get_connections(self, params),
            }
            handler = dispatch.get(action)
            if not handler:
                return SkillResult(success=False, message=f"Unknown action: {action}")
            return await handler()
        except Exception as e:
            return SkillResult(success=False, message=f"linkedin error: {str(e)}")

    async def close(self):
        await self.http.aclose()
