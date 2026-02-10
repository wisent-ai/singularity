"""
Reddit Skill

Post, comment, search, vote, and manage subreddits via the Reddit API.
Uses OAuth2 password grant for script-type apps.
"""

import os
from typing import Dict, Optional

import httpx

from singularity.skills.base import Skill, SkillResult, SkillManifest, SkillAction


def _a(n, d, p, cost=0.0, dur=5, prob=0.90):
    return SkillAction(name=n, description=d, parameters=p, estimated_cost=cost,
                       estimated_duration_seconds=dur, success_probability=prob)


def _p(n, t, r, d):
    return {n: {"type": t, "required": r, "description": d}}


class RedditSkill(Skill):
    """
    Reddit posting and engagement via the Reddit OAuth API.

    Required credentials:
    - REDDIT_CLIENT_ID: Reddit app client ID (script type)
    - REDDIT_CLIENT_SECRET: Reddit app client secret
    - REDDIT_USERNAME: Reddit account username
    - REDDIT_PASSWORD: Reddit account password
    """

    AUTH_URL = "https://www.reddit.com/api/v1/access_token"
    API = "https://oauth.reddit.com"
    USER_AGENT = "WisentAgent/1.0"

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="reddit",
            name="Reddit Marketing",
            version="2.1.0",
            category="social",
            description="Create accounts, post, comment, search, vote, and manage subreddits via Reddit API",
            required_credentials=[],
            install_cost=0,
            actions=[
                _a("create_account", "Create a new Reddit account via browser automation", {
                    **_p("username", "string", False, "Preferred username (auto-generated if omitted)"),
                    **_p("email", "string", False, "Email to use (auto-generated if omitted)"),
                    **_p("password", "string", False, "Password (auto-generated if omitted)"),
                }, dur=120, prob=0.60),
                _a("create_post", "Submit a text or link post to a subreddit", {
                    **_p("subreddit", "string", True, "Subreddit name (without r/)"),
                    **_p("title", "string", True, "Post title"),
                    **_p("text", "string", False, "Post body (for self/text posts)"),
                    **_p("url", "string", False, "URL (for link posts)"),
                    **_p("flair_text", "string", False, "Flair text"),
                }, dur=8),
                _a("comment", "Add a comment to a post or reply to a comment", {
                    **_p("thing_id", "string", True, "Fullname of post (t3_xxx) or comment (t1_xxx)"),
                    **_p("text", "string", True, "Comment body (markdown)"),
                }),
                _a("search", "Search posts across Reddit or within a subreddit", {
                    **_p("query", "string", True, "Search query"),
                    **_p("subreddit", "string", False, "Limit to subreddit (optional)"),
                    **_p("sort", "string", False, "relevance, hot, top, new, comments (default: relevance)"),
                    **_p("limit", "integer", False, "Number of results 1-100 (default: 10)"),
                }, dur=8),
                _a("get_posts", "Get posts from a subreddit", {
                    **_p("subreddit", "string", True, "Subreddit name"),
                    **_p("sort", "string", False, "hot, new, top, rising (default: hot)"),
                    **_p("limit", "integer", False, "Number of posts 1-100 (default: 10)"),
                    **_p("time", "string", False, "For top: hour, day, week, month, year, all"),
                }),
                _a("vote", "Upvote or downvote a post or comment", {
                    **_p("thing_id", "string", True, "Fullname of post (t3_xxx) or comment (t1_xxx)"),
                    **_p("direction", "integer", False, "1=upvote, 0=unvote, -1=downvote (default: 1)"),
                }),
                _a("subscribe", "Join or leave a subreddit", {
                    **_p("subreddit", "string", True, "Subreddit name"),
                    **_p("action", "string", False, "sub or unsub (default: sub)"),
                }),
                _a("get_user_info", "Get info about a Reddit user", {
                    **_p("username", "string", True, "Reddit username"),
                }),
                _a("get_subreddit_info", "Get info about a subreddit", {
                    **_p("subreddit", "string", True, "Subreddit name"),
                }),
            ]
        )

    def __init__(self, credentials: Dict[str, str] = None):
        super().__init__(credentials)
        self.http = httpx.AsyncClient(headers={"User-Agent": self.USER_AGENT})
        self._access_token: Optional[str] = None

    def _cred(self, key: str) -> str:
        return self.credentials.get(key) or os.environ.get(key, "")

    async def _ensure_token(self):
        """Authenticate via OAuth2 password grant if needed."""
        if self._access_token:
            return
        resp = await self.http.post(
            self.AUTH_URL,
            auth=(self._cred("REDDIT_CLIENT_ID"), self._cred("REDDIT_CLIENT_SECRET")),
            data={
                "grant_type": "password",
                "username": self._cred("REDDIT_USERNAME"),
                "password": self._cred("REDDIT_PASSWORD"),
            },
        )
        if resp.status_code != 200:
            raise RuntimeError(f"Reddit auth failed: {resp.text[:200]}")
        self._access_token = resp.json()["access_token"]

    def _headers(self) -> Dict:
        return {"Authorization": f"Bearer {self._access_token}", "User-Agent": self.USER_AGENT}

    async def execute(self, action: str, params: Dict) -> SkillResult:
        from . import actions

        try:
            if action != "create_account":
                await self._ensure_token()
            dispatch = {
                "create_account": lambda: actions.create_account(self, params),
                "create_post": lambda: actions.create_post(self, params),
                "comment": lambda: actions.comment(self, params),
                "search": lambda: actions.search(self, params),
                "get_posts": lambda: actions.get_posts(self, params),
                "vote": lambda: actions.vote(self, params),
                "subscribe": lambda: actions.subscribe(self, params),
                "get_user_info": lambda: actions.get_user_info(self, params),
                "get_subreddit_info": lambda: actions.get_subreddit_info(self, params),
            }
            handler = dispatch.get(action)
            if not handler:
                return SkillResult(success=False, message=f"Unknown action: {action}")
            return await handler()
        except Exception as e:
            return SkillResult(success=False, message=f"reddit error: {str(e)}")

    async def close(self):
        await self.http.aclose()
