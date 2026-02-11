"""
Instagram Skill

Post photos, reels, engage with content, and manage presence on Instagram.
Uses the Instagram Graph API for business/creator accounts and basic API fallback.
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


class InstagramSkill(Skill):
    """
    Skill for Instagram API interactions.

    Required credentials:
    - INSTAGRAM_ACCESS_TOKEN: Instagram Graph API access token
    - INSTAGRAM_ACCOUNT_ID: Instagram business/creator account ID
    """

    GRAPH_API = "https://graph.facebook.com/v18.0"

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="instagram",
            name="Instagram",
            version="1.0.0",
            category="social_media",
            description="Post photos, reels, engage with content, and manage presence on Instagram.",
            required_credentials=["INSTAGRAM_ACCESS_TOKEN", "INSTAGRAM_ACCOUNT_ID"],
            actions=[
                _a("post_photo", "Post a photo to Instagram", {**_p("image_url", "URL of the image to post"), **_p("caption", "Post caption")}),
                _a("post_reel", "Post a reel/video to Instagram", {**_p("video_url", "URL of the video"), **_p("caption", "Reel caption")}),
                _a("like", "Like a post", {**_p("media_id", "ID of the media to like")}),
                _a("comment", "Comment on a post", {**_p("media_id", "ID of the media"), **_p("message", "Comment text")}),
                _a("get_profile", "Get profile info", {**_p("username", "Instagram username (optional)")}),
                _a("search_hashtag", "Search posts by hashtag", {**_p("hashtag", "Hashtag to search"), **_p("limit", "Number of results", "number")}),
            ]
        )

    async def check_credentials(self) -> bool:
        token = (self.credentials or {}).get("INSTAGRAM_ACCESS_TOKEN") or os.environ.get("INSTAGRAM_ACCESS_TOKEN")
        return bool(token)

    async def execute(self, action: str, params: Dict) -> SkillResult:
        token = (self.credentials or {}).get("INSTAGRAM_ACCESS_TOKEN") or os.environ.get("INSTAGRAM_ACCESS_TOKEN")
        account_id = (self.credentials or {}).get("INSTAGRAM_ACCOUNT_ID") or os.environ.get("INSTAGRAM_ACCOUNT_ID")

        if not token:
            return SkillResult(success=False, data=None, message="INSTAGRAM_ACCESS_TOKEN not set. Configure credentials first.")

        if httpx is None:
            return SkillResult(success=False, data=None, message="httpx not installed. Run: pip install httpx")

        try:
            if action == "post_photo":
                return await self._post_photo(token, account_id, params)
            elif action == "post_reel":
                return await self._post_reel(token, account_id, params)
            elif action == "like":
                return await self._like(token, params)
            elif action == "comment":
                return await self._comment(token, params)
            elif action == "get_profile":
                return await self._get_profile(token, account_id, params)
            elif action == "search_hashtag":
                return await self._search_hashtag(token, account_id, params)
            else:
                return SkillResult(success=False, data=None, message=f"Unknown action: {action}")
        except Exception as e:
            return SkillResult(success=False, data=None, message=f"Instagram API error: {e}")

    async def _post_photo(self, token: str, account_id: str, params: Dict) -> SkillResult:
        if not account_id:
            return SkillResult(success=False, data=None, message="INSTAGRAM_ACCOUNT_ID required for posting.")
        image_url = params.get("image_url", "")
        caption = params.get("caption", "")
        if not image_url:
            return SkillResult(success=False, data=None, message="image_url required")

        async with httpx.AsyncClient() as client:
            # Step 1: Create media container
            resp = await client.post(
                f"{self.GRAPH_API}/{account_id}/media",
                params={"access_token": token, "image_url": image_url, "caption": caption}
            )
            data = resp.json()
            container_id = data.get("id")
            if not container_id:
                return SkillResult(success=False, data=data, message=data.get("error", {}).get("message", "Failed to create media container"))

            # Step 2: Publish
            resp = await client.post(
                f"{self.GRAPH_API}/{account_id}/media_publish",
                params={"access_token": token, "creation_id": container_id}
            )
            pub_data = resp.json()
            if "id" in pub_data:
                return SkillResult(success=True, data=pub_data, message=f"Photo posted: {pub_data['id']}")
            return SkillResult(success=False, data=pub_data, message="Failed to publish photo")

    async def _post_reel(self, token: str, account_id: str, params: Dict) -> SkillResult:
        if not account_id:
            return SkillResult(success=False, data=None, message="INSTAGRAM_ACCOUNT_ID required for posting.")
        video_url = params.get("video_url", "")
        caption = params.get("caption", "")
        if not video_url:
            return SkillResult(success=False, data=None, message="video_url required")

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{self.GRAPH_API}/{account_id}/media",
                params={"access_token": token, "video_url": video_url, "caption": caption, "media_type": "REELS"}
            )
            data = resp.json()
            container_id = data.get("id")
            if not container_id:
                return SkillResult(success=False, data=data, message="Failed to create reel container")

            resp = await client.post(
                f"{self.GRAPH_API}/{account_id}/media_publish",
                params={"access_token": token, "creation_id": container_id}
            )
            pub_data = resp.json()
            if "id" in pub_data:
                return SkillResult(success=True, data=pub_data, message=f"Reel posted: {pub_data['id']}")
            return SkillResult(success=False, data=pub_data, message="Failed to publish reel")

    async def _like(self, token: str, params: Dict) -> SkillResult:
        media_id = params.get("media_id", "")
        if not media_id:
            return SkillResult(success=False, data=None, message="media_id required")
        # Note: Instagram Graph API doesn't support liking via API directly for most accounts
        return SkillResult(success=False, data=None, message="Instagram Graph API has limited support for likes. Use engagement actions on the platform directly.")

    async def _comment(self, token: str, params: Dict) -> SkillResult:
        media_id = params.get("media_id", "")
        message = params.get("message", "")
        if not media_id or not message:
            return SkillResult(success=False, data=None, message="media_id and message required")

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{self.GRAPH_API}/{media_id}/comments",
                params={"access_token": token, "message": message}
            )
            data = resp.json()
            return SkillResult(success="id" in data, data=data, message="Commented" if "id" in data else "Comment failed")

    async def _get_profile(self, token: str, account_id: str, params: Dict) -> SkillResult:
        target = account_id or "me"
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{self.GRAPH_API}/{target}",
                params={"access_token": token, "fields": "id,username,name,biography,followers_count,media_count"}
            )
            return SkillResult(success=resp.status_code == 200, data=resp.json())

    async def _search_hashtag(self, token: str, account_id: str, params: Dict) -> SkillResult:
        hashtag = params.get("hashtag", "").strip("#")
        if not hashtag:
            return SkillResult(success=False, data=None, message="hashtag required")
        if not account_id:
            return SkillResult(success=False, data=None, message="INSTAGRAM_ACCOUNT_ID required for hashtag search")

        async with httpx.AsyncClient() as client:
            # Step 1: Get hashtag ID
            resp = await client.get(
                f"{self.GRAPH_API}/ig_hashtag_search",
                params={"access_token": token, "user_id": account_id, "q": hashtag}
            )
            data = resp.json()
            hashtag_data = data.get("data", [])
            if not hashtag_data:
                return SkillResult(success=False, data=data, message=f"Hashtag '{hashtag}' not found")

            hashtag_id = hashtag_data[0]["id"]
            limit = params.get("limit", 10)

            # Step 2: Get recent media
            resp = await client.get(
                f"{self.GRAPH_API}/{hashtag_id}/recent_media",
                params={"access_token": token, "user_id": account_id, "fields": "id,caption,media_type,permalink", "limit": limit}
            )
            return SkillResult(success=resp.status_code == 200, data=resp.json())
