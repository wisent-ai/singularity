"""
Reddit API actions.
"""

import os
import sys
from pathlib import Path
from typing import Dict

from singularity.skills.base import SkillResult


def _ensure_content_platform():
    """Add the content-platform account-api/skills to sys.path."""
    cp_path = os.environ.get("CONTENT_PLATFORM_PATH")
    if not cp_path:
        cp_path = str(Path(__file__).resolve().parents[6] / "content-platform" / "account-api")
    if cp_path not in sys.path:
        sys.path.insert(0, cp_path)


async def create_account(skill, params: Dict) -> SkillResult:
    """Create a new Reddit account via Playwright browser automation."""
    _ensure_content_platform()

    from skills.account_creator import AccountCreator

    credentials = {
        k: skill.credentials.get(k) or os.environ.get(k, "")
        for k in ("CAPSOLVER_API_KEY", "TWOCAPTCHA_API_KEY", "ANTICAPTCHA_API_KEY",
                   "RESEND_API_KEY", "AGENT_DOMAIN",
                   "PACKETSTREAM_USERNAME", "PACKETSTREAM_PASSWORD")
    }

    creator = AccountCreator(credentials=credentials, use_proxy=True)
    try:
        result = await creator.execute("create_account", {
            "site": "reddit.com",
            "username": params.get("username"),
            "email": params.get("email"),
            "password": params.get("password"),
        })
        if result.success:
            return SkillResult(
                success=True,
                message=f"Reddit account created: {result.data.get('username', 'unknown')}",
                data=result.data
            )
        return SkillResult(success=False, message=f"Account creation failed: {result.message}",
                           data=result.data)
    finally:
        await creator.close()


async def create_post(skill, params: Dict) -> SkillResult:
    """Submit a text or link post to a subreddit."""
    subreddit = params.get("subreddit")
    title = params.get("title")
    if not subreddit or not title:
        return SkillResult(success=False, message="Missing required: subreddit, title")

    data = {
        "sr": subreddit,
        "title": title,
        "kind": "link" if params.get("url") else "self",
        "resubmit": True,
    }
    if params.get("url"):
        data["url"] = params["url"]
    if params.get("text"):
        data["text"] = params["text"]
    if params.get("flair_text"):
        data["flair_text"] = params["flair_text"]

    resp = await skill.http.post(
        f"{skill.API}/api/submit", headers=skill._headers(), data=data
    )
    if resp.status_code == 200:
        result = resp.json()
        success = result.get("success", False)
        if success or not result.get("json", {}).get("errors"):
            post_data = result.get("json", {}).get("data", {})
            return SkillResult(
                success=True, message=f"Posted to r/{subreddit}",
                data={"subreddit": subreddit, "title": title,
                      "url": post_data.get("url"), "id": post_data.get("id"),
                      "name": post_data.get("name")}
            )
        errors = result.get("json", {}).get("errors", [])
        return SkillResult(success=False, message=f"Post rejected: {errors}")
    return SkillResult(success=False, message=f"Post failed: {resp.status_code} {resp.text[:200]}")


async def comment(skill, params: Dict) -> SkillResult:
    """Add a comment to a post or reply to a comment."""
    thing_id = params.get("thing_id")
    text = params.get("text")
    if not thing_id or not text:
        return SkillResult(success=False, message="Missing required: thing_id, text")

    resp = await skill.http.post(
        f"{skill.API}/api/comment", headers=skill._headers(),
        data={"thing_id": thing_id, "text": text}
    )
    if resp.status_code == 200:
        result = resp.json()
        errors = result.get("json", {}).get("errors", [])
        if not errors:
            return SkillResult(success=True, message="Comment posted",
                               data={"thing_id": thing_id, "text": text[:100]})
        return SkillResult(success=False, message=f"Comment rejected: {errors}")
    return SkillResult(success=False, message=f"Comment failed: {resp.status_code}")


async def search(skill, params: Dict) -> SkillResult:
    """Search posts across Reddit or within a subreddit."""
    query = params.get("query")
    if not query:
        return SkillResult(success=False, message="Missing required: query")

    subreddit = params.get("subreddit")
    base = f"{skill.API}/r/{subreddit}/search" if subreddit else f"{skill.API}/search"

    resp = await skill.http.get(
        base, headers=skill._headers(),
        params={
            "q": query,
            "sort": params.get("sort", "relevance"),
            "limit": min(int(params.get("limit", 10)), 100),
            "restrict_sr": "true" if subreddit else "false",
            "type": "link",
        }
    )
    if resp.status_code == 200:
        posts = _extract_posts(resp.json())
        return SkillResult(
            success=True, message=f"Found {len(posts)} results for '{query}'",
            data={"query": query, "posts": posts}
        )
    return SkillResult(success=False, message=f"Search failed: {resp.status_code}")


async def get_posts(skill, params: Dict) -> SkillResult:
    """Get posts from a subreddit."""
    subreddit = params.get("subreddit")
    if not subreddit:
        return SkillResult(success=False, message="Missing required: subreddit")

    sort = params.get("sort", "hot")
    url_params = {"limit": min(int(params.get("limit", 10)), 100)}
    if sort == "top" and params.get("time"):
        url_params["t"] = params["time"]

    resp = await skill.http.get(
        f"{skill.API}/r/{subreddit}/{sort}", headers=skill._headers(), params=url_params
    )
    if resp.status_code == 200:
        posts = _extract_posts(resp.json())
        return SkillResult(
            success=True, message=f"Got {len(posts)} posts from r/{subreddit}/{sort}",
            data={"subreddit": subreddit, "sort": sort, "posts": posts}
        )
    return SkillResult(success=False, message=f"Get posts failed: {resp.status_code}")


async def vote(skill, params: Dict) -> SkillResult:
    """Upvote or downvote a post or comment."""
    thing_id = params.get("thing_id")
    if not thing_id:
        return SkillResult(success=False, message="Missing required: thing_id")

    direction = int(params.get("direction", 1))
    resp = await skill.http.post(
        f"{skill.API}/api/vote", headers=skill._headers(),
        data={"id": thing_id, "dir": direction}
    )
    if resp.status_code == 200:
        labels = {1: "Upvoted", 0: "Unvoted", -1: "Downvoted"}
        return SkillResult(success=True, message=f"{labels.get(direction, 'Voted')} {thing_id}",
                           data={"thing_id": thing_id, "direction": direction})
    return SkillResult(success=False, message=f"Vote failed: {resp.status_code}")


async def subscribe(skill, params: Dict) -> SkillResult:
    """Join or leave a subreddit."""
    subreddit = params.get("subreddit")
    if not subreddit:
        return SkillResult(success=False, message="Missing required: subreddit")

    action = params.get("action", "sub")
    resp = await skill.http.post(
        f"{skill.API}/api/subscribe", headers=skill._headers(),
        data={"sr_name": subreddit, "action": action}
    )
    if resp.status_code == 200:
        verb = "Joined" if action == "sub" else "Left"
        return SkillResult(success=True, message=f"{verb} r/{subreddit}",
                           data={"subreddit": subreddit, "action": action})
    return SkillResult(success=False, message=f"Subscribe failed: {resp.status_code}")


async def get_user_info(skill, params: Dict) -> SkillResult:
    """Get info about a Reddit user."""
    username = params.get("username")
    if not username:
        return SkillResult(success=False, message="Missing required: username")

    resp = await skill.http.get(f"{skill.API}/user/{username}/about", headers=skill._headers())
    if resp.status_code == 200:
        data = resp.json().get("data", {})
        return SkillResult(
            success=True, message=f"u/{username}",
            data={"username": data.get("name"), "karma": data.get("total_karma"),
                  "link_karma": data.get("link_karma"), "comment_karma": data.get("comment_karma"),
                  "created_utc": data.get("created_utc"), "is_gold": data.get("is_gold")}
        )
    return SkillResult(success=False, message=f"User lookup failed: {resp.status_code}")


async def get_subreddit_info(skill, params: Dict) -> SkillResult:
    """Get info about a subreddit."""
    subreddit = params.get("subreddit")
    if not subreddit:
        return SkillResult(success=False, message="Missing required: subreddit")

    resp = await skill.http.get(f"{skill.API}/r/{subreddit}/about", headers=skill._headers())
    if resp.status_code == 200:
        data = resp.json().get("data", {})
        return SkillResult(
            success=True, message=f"r/{subreddit}: {data.get('subscribers', 0)} subscribers",
            data={"name": data.get("display_name"), "title": data.get("title"),
                  "description": data.get("public_description", "")[:200],
                  "subscribers": data.get("subscribers"), "active_users": data.get("active_user_count"),
                  "created_utc": data.get("created_utc"), "nsfw": data.get("over18")}
        )
    return SkillResult(success=False, message=f"Subreddit lookup failed: {resp.status_code}")


def _extract_posts(listing: Dict) -> list:
    """Extract posts from a Reddit listing response."""
    children = listing.get("data", {}).get("children", [])
    return [{
        "id": c["data"].get("id"), "name": c["data"].get("name"),
        "title": c["data"].get("title"), "author": c["data"].get("author"),
        "subreddit": c["data"].get("subreddit"),
        "score": c["data"].get("score"), "num_comments": c["data"].get("num_comments"),
        "url": c["data"].get("url"), "permalink": c["data"].get("permalink"),
        "created_utc": c["data"].get("created_utc"),
    } for c in children if c.get("kind") == "t3"]
