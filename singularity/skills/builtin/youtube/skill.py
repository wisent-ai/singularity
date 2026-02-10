"""
YouTube Skill

Search videos, upload content, comment, and manage channels via YouTube Data API v3.
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


class YouTubeSkill(Skill):
    """
    YouTube management via YouTube Data API v3.

    Required credentials:
    - YOUTUBE_API_KEY: API key for read-only operations (search, get info)
    - YOUTUBE_ACCESS_TOKEN: OAuth 2.0 token for write operations (upload, comment, subscribe)
    """

    API = "https://www.googleapis.com/youtube/v3"
    UPLOAD_API = "https://www.googleapis.com/upload/youtube/v3/videos"

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="youtube",
            name="YouTube Management",
            version="1.0.0",
            category="social",
            description="Search, upload, comment, and manage YouTube content via Data API v3",
            required_credentials=["YOUTUBE_API_KEY"],
            install_cost=0,
            actions=[
                _a("search", "Search YouTube videos by query", {
                    **_p("query", "string", True, "Search query"),
                    **_p("max_results", "integer", False, "Number of results 1-50 (default: 10)"),
                    **_p("order", "string", False, "date, rating, relevance, viewCount (default: relevance)"),
                    **_p("type", "string", False, "video, channel, playlist (default: video)"),
                }, dur=5),
                _a("get_video", "Get detailed info and stats for a video", {
                    **_p("video_id", "string", True, "YouTube video ID"),
                }),
                _a("get_channel", "Get channel info and stats", {
                    **_p("channel_id", "string", False, "Channel ID (default: authenticated user's channel)"),
                    **_p("username", "string", False, "Channel username (alternative to channel_id)"),
                }),
                _a("upload_video", "Upload a video to YouTube", {
                    **_p("file_path", "string", True, "Local path to video file"),
                    **_p("title", "string", True, "Video title"),
                    **_p("description", "string", False, "Video description"),
                    **_p("tags", "string", False, "Comma-separated tags"),
                    **_p("privacy", "string", False, "public, unlisted, private (default: private)"),
                    **_p("category_id", "string", False, "YouTube category ID (default: 22=People & Blogs)"),
                }, dur=120, prob=0.80),
                _a("comment", "Post a comment on a video", {
                    **_p("video_id", "string", True, "Video ID to comment on"),
                    **_p("text", "string", True, "Comment text"),
                }),
                _a("get_comments", "Get comments on a video", {
                    **_p("video_id", "string", True, "Video ID"),
                    **_p("max_results", "integer", False, "Number of comments 1-100 (default: 20)"),
                    **_p("order", "string", False, "time or relevance (default: relevance)"),
                }),
                _a("subscribe", "Subscribe to a YouTube channel", {
                    **_p("channel_id", "string", True, "Channel ID to subscribe to"),
                }),
            ]
        )

    def __init__(self, credentials: Dict[str, str] = None):
        super().__init__(credentials)
        self.http = httpx.AsyncClient()

    def _api_key(self) -> str:
        return self.credentials.get("YOUTUBE_API_KEY") or os.environ.get("YOUTUBE_API_KEY", "")

    def _oauth_token(self) -> str:
        return self.credentials.get("YOUTUBE_ACCESS_TOKEN") or os.environ.get("YOUTUBE_ACCESS_TOKEN", "")

    def _read_headers(self) -> Dict:
        """Headers for read-only API key requests."""
        return {"Accept": "application/json"}

    def _write_headers(self) -> Dict:
        """Headers for OAuth-authenticated write requests."""
        token = self._oauth_token()
        if not token:
            raise RuntimeError("YOUTUBE_ACCESS_TOKEN required for write operations")
        return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    async def execute(self, action: str, params: Dict) -> SkillResult:
        from . import actions

        try:
            dispatch = {
                "search": lambda: actions.search(self, params),
                "get_video": lambda: actions.get_video(self, params),
                "get_channel": lambda: actions.get_channel(self, params),
                "upload_video": lambda: actions.upload_video(self, params),
                "comment": lambda: actions.comment(self, params),
                "get_comments": lambda: actions.get_comments(self, params),
                "subscribe": lambda: actions.subscribe(self, params),
            }
            handler = dispatch.get(action)
            if not handler:
                return SkillResult(success=False, message=f"Unknown action: {action}")
            return await handler()
        except Exception as e:
            return SkillResult(success=False, message=f"youtube error: {str(e)}")

    async def close(self):
        await self.http.aclose()
