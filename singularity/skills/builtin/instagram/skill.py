"""
Instagram Skill

Post photos, reels, engage with content, and manage presence on Instagram.
Uses the Instagram Graph API for business/creator accounts and basic API fallback.
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


class InstagramSkill(Skill):
    """
    Instagram marketing and engagement via the Instagram Graph API.

    Required credentials:
    - INSTAGRAM_USERNAME: Instagram account username
    - INSTAGRAM_PASSWORD: Instagram account password
    """

    GRAPH_API = "https://graph.instagram.com/v18.0"
    BASIC_API = "https://api.instagram.com/v1"

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="instagram",
            name="Instagram Marketing",
            version="1.0.0",
            category="social",
            description="Autonomous Instagram posting, engagement, and account management",
            required_credentials=["INSTAGRAM_USERNAME", "INSTAGRAM_PASSWORD"],
            install_cost=0,
            actions=[
                _a("create_account", "Create a new Instagram account via browser automation", {
                    **_p("username", "string", False, "Preferred username (auto-generated if omitted)"),
                    **_p("email", "string", False, "Email to use (auto-generated if omitted)"),
                    **_p("password", "string", False, "Password (auto-generated if omitted)"),
                    **_p("full_name", "string", False, "Display name for the account"),
                }, dur=120, prob=0.55),
                _a("login", "Login to Instagram and obtain session tokens", {
                    **_p("username", "string", False, "Username (uses credential if omitted)"),
                    **_p("password", "string", False, "Password (uses credential if omitted)"),
                }, dur=15, prob=0.85),
                _a("post_photo", "Post a photo to Instagram feed", {
                    **_p("image_url", "string", True, "URL of the image to post"),
                    **_p("caption", "string", False, "Photo caption text"),
                    **_p("location_id", "string", False, "Instagram location ID to tag"),
                    **_p("user_tags", "string", False, "Comma-separated usernames to tag"),
                }, dur=20, prob=0.85),
                _a("post_reel", "Post a reel/video to Instagram", {
                    **_p("video_url", "string", True, "URL of the video to post as reel"),
                    **_p("caption", "string", False, "Reel caption text"),
                    **_p("cover_url", "string", False, "URL of the cover image"),
                    **_p("share_to_feed", "string", False, "Whether to share to feed (true/false, default: true)"),
                }, dur=30, prob=0.80),
                _a("like", "Like an Instagram post", {
                    **_p("media_id", "string", True, "Media ID of the post to like"),
                }, dur=3, prob=0.95),
                _a("comment", "Comment on an Instagram post", {
                    **_p("media_id", "string", True, "Media ID of the post to comment on"),
                    **_p("text", "string", True, "Comment text"),
                }, dur=5, prob=0.90),
                _a("get_profile", "Get an Instagram user profile", {
                    **_p("username", "string", False, "Username to look up (default: authenticated user)"),
                    **_p("user_id", "string", False, "User ID to look up"),
                }, dur=5, prob=0.90),
                _a("search_hashtag", "Search Instagram posts by hashtag", {
                    **_p("hashtag", "string", True, "Hashtag to search (without # prefix)"),
                    **_p("count", "integer", False, "Number of results (default: 20)"),
                }, dur=10, prob=0.85),
            ]
        )

    def __init__(self, credentials: Dict[str, str] = None):
        super().__init__(credentials)
        self.http = httpx.AsyncClient()
        self._access_token = None
        self._user_id = None

    def _get_credential(self, key: str) -> str:
        """Get a credential from instance credentials or environment."""
        return self.credentials.get(key) or os.environ.get(key, "")

    def _graph_headers(self) -> Dict:
        """Headers for Instagram Graph API requests."""
        return {
            "Content-Type": "application/json",
        }

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
        token = self._get_credential("INSTAGRAM_ACCESS_TOKEN")
        if token:
            self._access_token = token
            # Fetch user ID from token
            resp = await self.http.get(
                f"{self.GRAPH_API}/me",
                params={"access_token": token, "fields": "id,username"}
            )
            if resp.status_code == 200:
                data = resp.json()
                self._user_id = data.get("id")
                return True
        return False

    async def execute(self, action: str, params: Dict) -> SkillResult:
        from . import actions

        try:
            dispatch = {
                "create_account": lambda: actions.create_account(self, params),
                "login": lambda: actions.login(self, params),
                "post_photo": lambda: actions.post_photo(self, params),
                "post_reel": lambda: actions.post_reel(self, params),
                "like": lambda: actions.like(self, params),
                "comment": lambda: actions.comment(self, params),
                "get_profile": lambda: actions.get_profile(self, params),
                "search_hashtag": lambda: actions.search_hashtag(self, params),
            }
            handler = dispatch.get(action)
            if not handler:
                return SkillResult(success=False, message=f"Unknown action: {action}")
            return await handler()
        except Exception as e:
            return SkillResult(success=False, message=f"instagram error: {str(e)}")

    async def close(self):
        await self.http.aclose()
