"""
Facebook Skill

Post content, manage pages, engage with posts on Facebook.
Uses the Facebook Graph API for authenticated operations.
"""

import os
from typing import Dict

try:
    import httpx
except ImportError:
    httpx = None

from singularity.skills.base import Skill, SkillResult, SkillManifest, SkillAction

_a = lambda name, desc, params: SkillAction(name=name, description=desc, parameters=params)
_p = lambda name, desc, typ="string": {name: {"type": typ, "description": desc}}


class FacebookSkill(Skill):
    """
    Skill for Facebook API interactions.

    Required credentials:
    - FACEBOOK_ACCESS_TOKEN: Facebook Graph API access token
    - FACEBOOK_PAGE_ID: (optional) Page ID for page-level actions
    """

    GRAPH_API = "https://graph.facebook.com/v18.0"

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="facebook",
            name="Facebook",
            version="1.0.0",
            category="social_media",
            description="Post content, manage pages, and engage with posts on Facebook via the Graph API.",
            required_credentials=["FACEBOOK_ACCESS_TOKEN"],
            actions=[
                _a("post", "Post content to your Facebook feed", {**_p("message", "Text content to post")}),
                _a("post_to_page", "Post content to a Facebook Page", {**_p("message", "Text content"), **_p("page_id", "Page ID (optional, uses default)")}),
                _a("like", "Like a post", {**_p("post_id", "ID of the post to like")}),
                _a("comment", "Comment on a post", {**_p("post_id", "ID of the post"), **_p("message", "Comment text")}),
                _a("get_page", "Get Facebook page info", {**_p("page_id", "Page ID")}),
                _a("get_feed", "Get your Facebook feed", {**_p("limit", "Number of posts to retrieve", "number")}),
            ]
        )

    async def check_credentials(self) -> bool:
        token = (self.credentials or {}).get("FACEBOOK_ACCESS_TOKEN") or os.environ.get("FACEBOOK_ACCESS_TOKEN")
        return bool(token)

    async def execute(self, action: str, params: Dict) -> SkillResult:
        token = (self.credentials or {}).get("FACEBOOK_ACCESS_TOKEN") or os.environ.get("FACEBOOK_ACCESS_TOKEN")
        if not token:
            return SkillResult(success=False, data=None, message="FACEBOOK_ACCESS_TOKEN not set. Configure credentials first.")

        if httpx is None:
            return SkillResult(success=False, data=None, message="httpx not installed. Run: pip install httpx")

        try:
            if action == "post":
                return await self._post(token, params)
            elif action == "post_to_page":
                return await self._post_to_page(token, params)
            elif action == "like":
                return await self._like(token, params)
            elif action == "comment":
                return await self._comment(token, params)
            elif action == "get_page":
                return await self._get_page(token, params)
            elif action == "get_feed":
                return await self._get_feed(token, params)
            else:
                return SkillResult(success=False, data=None, message=f"Unknown action: {action}")
        except Exception as e:
            return SkillResult(success=False, data=None, message=f"Facebook API error: {e}")

    async def _post(self, token: str, params: Dict) -> SkillResult:
        message = params.get("message", "")
        async with httpx.AsyncClient() as client:
            resp = await client.post(f"{self.GRAPH_API}/me/feed", params={"access_token": token, "message": message})
            data = resp.json()
            if "id" in data:
                return SkillResult(success=True, data=data, message=f"Posted successfully: {data['id']}")
            return SkillResult(success=False, data=data, message=data.get("error", {}).get("message", "Post failed"))

    async def _post_to_page(self, token: str, params: Dict) -> SkillResult:
        page_id = params.get("page_id") or os.environ.get("FACEBOOK_PAGE_ID", "")
        message = params.get("message", "")
        if not page_id:
            return SkillResult(success=False, data=None, message="page_id required")
        async with httpx.AsyncClient() as client:
            resp = await client.post(f"{self.GRAPH_API}/{page_id}/feed", params={"access_token": token, "message": message})
            data = resp.json()
            if "id" in data:
                return SkillResult(success=True, data=data, message=f"Posted to page: {data['id']}")
            return SkillResult(success=False, data=data, message=data.get("error", {}).get("message", "Page post failed"))

    async def _like(self, token: str, params: Dict) -> SkillResult:
        post_id = params.get("post_id", "")
        if not post_id:
            return SkillResult(success=False, data=None, message="post_id required")
        async with httpx.AsyncClient() as client:
            resp = await client.post(f"{self.GRAPH_API}/{post_id}/likes", params={"access_token": token})
            return SkillResult(success=resp.status_code == 200, data=resp.json(), message="Liked" if resp.status_code == 200 else "Like failed")

    async def _comment(self, token: str, params: Dict) -> SkillResult:
        post_id = params.get("post_id", "")
        message = params.get("message", "")
        if not post_id or not message:
            return SkillResult(success=False, data=None, message="post_id and message required")
        async with httpx.AsyncClient() as client:
            resp = await client.post(f"{self.GRAPH_API}/{post_id}/comments", params={"access_token": token, "message": message})
            data = resp.json()
            return SkillResult(success="id" in data, data=data, message="Commented" if "id" in data else "Comment failed")

    async def _get_page(self, token: str, params: Dict) -> SkillResult:
        page_id = params.get("page_id", "")
        if not page_id:
            return SkillResult(success=False, data=None, message="page_id required")
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{self.GRAPH_API}/{page_id}", params={"access_token": token, "fields": "id,name,fan_count,about"})
            return SkillResult(success=resp.status_code == 200, data=resp.json())

    async def _get_feed(self, token: str, params: Dict) -> SkillResult:
        limit = params.get("limit", 10)
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{self.GRAPH_API}/me/feed", params={"access_token": token, "limit": limit})
            return SkillResult(success=resp.status_code == 200, data=resp.json())
