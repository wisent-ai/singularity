"""
Facebook Skill

Post content, manage pages, engage with posts on Facebook.
Uses the Facebook Graph API for authenticated operations.
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


class FacebookSkill(Skill):
    """
    Facebook marketing and engagement via the Facebook Graph API.

    Required credentials:
    - FACEBOOK_EMAIL: Facebook account email
    - FACEBOOK_PASSWORD: Facebook account password
    """

    GRAPH_API = "https://graph.facebook.com/v19.0"

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="facebook",
            name="Facebook Marketing",
            version="1.0.0",
            category="social",
            description="Autonomous Facebook posting and engagement",
            required_credentials=["FACEBOOK_EMAIL", "FACEBOOK_PASSWORD"],
            install_cost=0,
            actions=[
                _a("create_account", "Create a new Facebook account via browser automation", {
                    **_p("email", "string", False, "Email to use (auto-generated if omitted)"),
                    **_p("password", "string", False, "Password (auto-generated if omitted)"),
                    **_p("first_name", "string", False, "First name for the account"),
                    **_p("last_name", "string", False, "Last name for the account"),
                }, dur=120, prob=0.50),
                _a("login", "Login to Facebook and obtain access token", {
                    **_p("email", "string", False, "Email (uses credential if omitted)"),
                    **_p("password", "string", False, "Password (uses credential if omitted)"),
                }, dur=15, prob=0.85),
                _a("post", "Create a post on the authenticated user's timeline", {
                    **_p("message", "string", True, "Post text content"),
                    **_p("link", "string", False, "URL to attach to the post"),
                    **_p("privacy", "string", False, "Privacy: EVERYONE, ALL_FRIENDS, SELF (default: EVERYONE)"),
                }, dur=10, prob=0.90),
                _a("like", "Like a Facebook post or object", {
                    **_p("object_id", "string", True, "ID of the post or object to like"),
                }, dur=3, prob=0.95),
                _a("comment", "Comment on a Facebook post", {
                    **_p("post_id", "string", True, "ID of the post to comment on"),
                    **_p("message", "string", True, "Comment text"),
                }, dur=5, prob=0.90),
                _a("get_page", "Get details of a Facebook Page", {
                    **_p("page_id", "string", True, "Page ID or vanity URL name"),
                    **_p("fields", "string", False, "Comma-separated fields (default: id,name,fan_count,about)"),
                }, dur=5, prob=0.90),
                _a("post_to_page", "Post content to a managed Facebook Page", {
                    **_p("page_id", "string", True, "Page ID to post to"),
                    **_p("message", "string", True, "Post text content"),
                    **_p("link", "string", False, "URL to attach to the post"),
                    **_p("published", "string", False, "Whether to publish immediately (true/false, default: true)"),
                    **_p("scheduled_publish_time", "string", False, "Unix timestamp for scheduled post"),
                }, dur=10, prob=0.85),
                _a("get_feed", "Get posts from the user's feed or a page feed", {
                    **_p("source_id", "string", False, "Page or user ID (default: authenticated user)"),
                    **_p("count", "integer", False, "Number of posts to retrieve (default: 10)"),
                }, dur=8, prob=0.90),
            ]
        )

    def __init__(self, credentials: Dict[str, str] = None):
        super().__init__(credentials)
        self.http = httpx.AsyncClient()
        self._access_token = None
        self._user_id = None
        self._page_tokens = {}  # page_id -> page_access_token

    def _get_credential(self, key: str) -> str:
        """Get a credential from instance credentials or environment."""
        return self.credentials.get(key) or os.environ.get(key, "")

    def _graph_params(self, extra: Dict = None) -> Dict:
        """Base query params with access token for Graph API."""
        params = {"access_token": self._access_token or ""}
        if extra:
            params.update(extra)
        return params

    async def _ensure_login(self) -> bool:
        """Ensure we have a valid access token / session."""
        if self._access_token and self._user_id:
            return True
        # Try environment for pre-configured token
        token = self._get_credential("FACEBOOK_ACCESS_TOKEN")
        if token:
            self._access_token = token
            resp = await self.http.get(
                f"{self.GRAPH_API}/me",
                params={"access_token": token, "fields": "id,name"}
            )
            if resp.status_code == 200:
                data = resp.json()
                self._user_id = data.get("id")
                return True
        return False

    async def _get_page_token(self, page_id: str) -> str:
        """Get or fetch the page access token for a managed page."""
        if page_id in self._page_tokens:
            return self._page_tokens[page_id]

        resp = await self.http.get(
            f"{self.GRAPH_API}/{self._user_id}/accounts",
            params={"access_token": self._access_token, "fields": "id,access_token,name"}
        )

        if resp.status_code == 200:
            pages = resp.json().get("data", [])
            for page in pages:
                self._page_tokens[page["id"]] = page["access_token"]
            return self._page_tokens.get(page_id)

        return None

    async def execute(self, action: str, params: Dict) -> SkillResult:
        from . import actions

        try:
            dispatch = {
                "create_account": lambda: actions.create_account(self, params),
                "login": lambda: actions.login(self, params),
                "post": lambda: actions.post(self, params),
                "like": lambda: actions.like(self, params),
                "comment": lambda: actions.comment(self, params),
                "get_page": lambda: actions.get_page(self, params),
                "post_to_page": lambda: actions.post_to_page(self, params),
                "get_feed": lambda: actions.get_feed(self, params),
            }
            handler = dispatch.get(action)
            if not handler:
                return SkillResult(success=False, message=f"Unknown action: {action}")
            return await handler()
        except Exception as e:
            return SkillResult(success=False, message=f"facebook error: {str(e)}")

    async def close(self):
        await self.http.aclose()
