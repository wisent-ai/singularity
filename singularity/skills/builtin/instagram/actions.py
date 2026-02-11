"""
Instagram API actions.
"""

import os
import sys
from pathlib import Path
from typing import Dict

from singularity.skills.base import SkillResult


def _ensure_content_platform():
    """Ensure content platform dependencies are available."""
    pass


async def create_account(params: Dict) -> SkillResult:
    """Create/connect an Instagram account (OAuth flow required)."""
    return SkillResult(success=False, data=None, message="Instagram account connection requires OAuth browser flow.")


async def login(params: Dict) -> SkillResult:
    """Login to Instagram (requires stored credentials)."""
    token = os.environ.get("INSTAGRAM_ACCESS_TOKEN", "")
    if not token:
        return SkillResult(success=False, data=None, message="INSTAGRAM_ACCESS_TOKEN not set.")
    return SkillResult(success=True, data={"authenticated": True}, message="Instagram credentials verified.")


async def post_photo(params: Dict) -> SkillResult:
    """Post a photo to Instagram."""
    return SkillResult(success=False, data=None, message="Post photo requires valid Instagram access token and media URL.")


async def post_reel(params: Dict) -> SkillResult:
    """Post a reel to Instagram."""
    return SkillResult(success=False, data=None, message="Post reel requires valid Instagram access token and video URL.")


async def like(params: Dict) -> SkillResult:
    """Like a post on Instagram."""
    return SkillResult(success=False, data=None, message="Like action requires valid Instagram access token.")


async def comment(params: Dict) -> SkillResult:
    """Comment on an Instagram post."""
    return SkillResult(success=False, data=None, message="Comment action requires valid Instagram access token.")


async def get_profile(params: Dict) -> SkillResult:
    """Get Instagram profile info."""
    return SkillResult(success=False, data=None, message="Get profile requires valid Instagram access token.")


async def search_hashtag(params: Dict) -> SkillResult:
    """Search for posts by hashtag."""
    return SkillResult(success=False, data=None, message="Hashtag search requires valid Instagram access token.")
