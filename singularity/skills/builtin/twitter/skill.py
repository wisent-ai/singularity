#!/usr/bin/env python3
"""
Twitter/X Skill

Enables agents to:
- Post tweets
- Reply to tweets
- Search tweets
- Get mentions
- Follow/unfollow users
- Send DMs
"""

import httpx
import json
import time
import hmac
import hashlib
import base64
import urllib.parse
from typing import Dict, List, Optional
from singularity.skills.base import Skill, SkillResult, SkillManifest, SkillAction


class TwitterSkill(Skill):
    """
    Skill for Twitter/X API interactions.

    Required credentials:
    - TWITTER_API_KEY: API key (consumer key)
    - TWITTER_API_SECRET: API secret (consumer secret)
    - TWITTER_ACCESS_TOKEN: Access token
    - TWITTER_ACCESS_SECRET: Access token secret
    - TWITTER_BEARER_TOKEN: Bearer token (for v2 API)
    """

    API_BASE_V2 = "https://api.twitter.com/2"
    API_BASE_V1 = "https://api.twitter.com/1.1"

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="twitter",
            name="Twitter/X Management",
            version="1.0.0",
            category="social",
            description="Post tweets, reply, search, and manage Twitter presence",
            required_credentials=[
                "TWITTER_API_KEY", "TWITTER_API_SECRET",
                "TWITTER_ACCESS_TOKEN", "TWITTER_ACCESS_SECRET"
            ],
            install_cost=0,
            actions=[
                SkillAction(name="post_tweet", description="Post a new tweet",
                    parameters={"text": {"type": "string", "required": True, "description": "Tweet text (max 280 chars)"},
                                "reply_to": {"type": "string", "required": False, "description": "Tweet ID to reply to"}},
                    estimated_cost=0, estimated_duration_seconds=3, success_probability=0.95),
                SkillAction(name="search_tweets", description="Search for tweets matching a query",
                    parameters={"query": {"type": "string", "required": True, "description": "Search query"},
                                "max_results": {"type": "integer", "required": False, "description": "Max results (10-100)"}},
                    estimated_cost=0, estimated_duration_seconds=5, success_probability=0.9),
                SkillAction(name="get_mentions", description="Get recent mentions of the authenticated user",
                    parameters={"max_results": {"type": "integer", "required": False, "description": "Max results (5-100)"}},
                    estimated_cost=0, estimated_duration_seconds=5, success_probability=0.9),
                SkillAction(name="follow_user", description="Follow a Twitter user",
                    parameters={"username": {"type": "string", "required": True, "description": "Username to follow"}},
                    estimated_cost=0, estimated_duration_seconds=3, success_probability=0.9),
                SkillAction(name="send_dm", description="Send a direct message",
                    parameters={"username": {"type": "string", "required": True, "description": "Username to DM"},
                                "text": {"type": "string", "required": True, "description": "Message text"}},
                    estimated_cost=0, estimated_duration_seconds=5, success_probability=0.85),
                SkillAction(name="get_user_info", description="Get information about a Twitter user",
                    parameters={"username": {"type": "string", "required": True, "description": "Username to lookup"}},
                    estimated_cost=0, estimated_duration_seconds=3, success_probability=0.95),
                SkillAction(name="like_tweet", description="Like a tweet",
                    parameters={"tweet_id": {"type": "string", "required": True, "description": "Tweet ID to like"}},
                    estimated_cost=0, estimated_duration_seconds=2, success_probability=0.95),
                SkillAction(name="retweet", description="Retweet a tweet",
                    parameters={"tweet_id": {"type": "string", "required": True, "description": "Tweet ID to retweet"}},
                    estimated_cost=0, estimated_duration_seconds=2, success_probability=0.95),
            ]
        )

    def __init__(self, credentials: Dict[str, str] = None):
        super().__init__(credentials)
        self.http = httpx.AsyncClient()
        self._user_id = None

    def _generate_oauth_signature(self, method: str, url: str, params: Dict, oauth_params: Dict) -> str:
        """Generate OAuth 1.0a signature"""
        all_params = {**params, **oauth_params}
        sorted_params = sorted(all_params.items())
        param_string = "&".join(
            f"{urllib.parse.quote(str(k), safe='')}"
            f"={urllib.parse.quote(str(v), safe='')}"
            for k, v in sorted_params
        )
        base_string = (
            f"{method.upper()}&"
            f"{urllib.parse.quote(url, safe='')}&"
            f"{urllib.parse.quote(param_string, safe='')}"
        )
        signing_key = (
            f"{urllib.parse.quote(self.credentials.get('TWITTER_API_SECRET', ''), safe='')}&"
            f"{urllib.parse.quote(self.credentials.get('TWITTER_ACCESS_SECRET', ''), safe='')}"
        )
        signature = hmac.new(signing_key.encode(), base_string.encode(), hashlib.sha1).digest()
        return base64.b64encode(signature).decode()

    def _get_oauth_header(self, method: str, url: str, params: Dict = None) -> str:
        """Generate OAuth Authorization header"""
        params = params or {}
        oauth_params = {
            "oauth_consumer_key": self.credentials.get("TWITTER_API_KEY"),
            "oauth_nonce": str(int(time.time() * 1000)),
            "oauth_signature_method": "HMAC-SHA1",
            "oauth_timestamp": str(int(time.time())),
            "oauth_token": self.credentials.get("TWITTER_ACCESS_TOKEN"),
            "oauth_version": "1.0"
        }
        oauth_params["oauth_signature"] = self._generate_oauth_signature(method, url, params, oauth_params)
        header_parts = [f'{k}="{urllib.parse.quote(str(v), safe="")}"' for k, v in sorted(oauth_params.items())]
        return "OAuth " + ", ".join(header_parts)

    async def _get_user_id(self, username: str = None) -> Optional[str]:
        """Get user ID from username"""
        if username:
            result = await self._get_user_info(username)
            if result.success:
                return result.data.get("id")
            return None

        if self._user_id:
            return self._user_id

        url = f"{self.API_BASE_V2}/users/me"
        headers = {"Authorization": self._get_oauth_header("GET", url)}
        response = await self.http.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            self._user_id = data.get("data", {}).get("id")
            return self._user_id
        return None

    async def execute(self, action: str, params: Dict) -> SkillResult:
        """Execute a Twitter action"""
        from . import actions

        if not self.check_credentials():
            missing = self.get_missing_credentials()
            return SkillResult(success=False, message=f"Missing credentials: {missing}")

        try:
            if action == "post_tweet":
                return await actions.post_tweet(self, params.get("text"), params.get("reply_to"))
            elif action == "search_tweets":
                return await actions.search_tweets(self, params.get("query"), params.get("max_results", 10))
            elif action == "get_mentions":
                return await actions.get_mentions(self, params.get("max_results", 10))
            elif action == "follow_user":
                return await actions.follow_user(self, params.get("username"))
            elif action == "send_dm":
                return await actions.send_dm(self, params.get("username"), params.get("text"))
            elif action == "get_user_info":
                return await self._get_user_info(params.get("username"))
            elif action == "like_tweet":
                return await actions.like_tweet(self, params.get("tweet_id"))
            elif action == "retweet":
                return await actions.retweet(self, params.get("tweet_id"))
            else:
                return SkillResult(success=False, message=f"Unknown action: {action}")
        except Exception as e:
            return SkillResult(success=False, message=f"Twitter error: {str(e)}")

    async def _get_user_info(self, username: str) -> SkillResult:
        """Get user information"""
        if not username:
            return SkillResult(success=False, message="Username required")

        username = username.lstrip("@")
        url = f"{self.API_BASE_V2}/users/by/username/{username}"
        params = {"user.fields": "description,public_metrics,created_at,profile_image_url"}
        headers = {"Authorization": self._get_oauth_header("GET", url, params)}
        response = await self.http.get(url, headers=headers, params=params)

        if response.status_code == 200:
            data = response.json()
            user_data = data.get("data", {})
            return SkillResult(
                success=True, message=f"Got info for @{username}",
                data={
                    "id": user_data.get("id"), "username": user_data.get("username"),
                    "name": user_data.get("name"), "description": user_data.get("description"),
                    "followers": user_data.get("public_metrics", {}).get("followers_count"),
                    "following": user_data.get("public_metrics", {}).get("following_count"),
                    "tweets": user_data.get("public_metrics", {}).get("tweet_count")
                }
            )
        else:
            error = response.json() if response.text else {"error": response.status_code}
            return SkillResult(success=False, message=f"Failed to get user info: {error}")

    async def close(self):
        """Clean up"""
        await self.http.aclose()
