"""
Facebook Graph API actions.
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
    """Create a new Facebook account via Playwright browser automation."""
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
            "site": "facebook.com",
            "email": params.get("email"),
            "password": params.get("password"),
            "first_name": params.get("first_name"),
            "last_name": params.get("last_name"),
        })
        if result.success:
            return SkillResult(
                success=True,
                message=f"Facebook account created: {result.data.get('email', 'unknown')}",
                data=result.data
            )
        return SkillResult(success=False, message=f"Account creation failed: {result.message}",
                           data=result.data)
    finally:
        await creator.close()


async def login(skill, params: Dict) -> SkillResult:
    """Login to Facebook and obtain an access token."""
    email = params.get("email") or skill._get_credential("FACEBOOK_EMAIL")
    password = params.get("password") or skill._get_credential("FACEBOOK_PASSWORD")

    if not email or not password:
        return SkillResult(success=False, message="Missing credentials: FACEBOOK_EMAIL and FACEBOOK_PASSWORD required")

    # Use Facebook's OAuth device login flow or direct graph API auth
    # For server-side, use the access token grant
    auth_url = "https://graph.facebook.com/oauth/access_token"

    # Try client credentials approach if app credentials exist
    app_id = skill._get_credential("FACEBOOK_APP_ID")
    app_secret = skill._get_credential("FACEBOOK_APP_SECRET")

    if app_id and app_secret:
        resp = await skill.http.get(auth_url, params={
            "client_id": app_id,
            "client_secret": app_secret,
            "grant_type": "client_credentials",
        })

        if resp.status_code == 200:
            data = resp.json()
            skill._access_token = data.get("access_token")
            # Verify the token and get user info
            me_resp = await skill.http.get(
                f"{skill.GRAPH_API}/me",
                params={"access_token": skill._access_token, "fields": "id,name,email"}
            )
            if me_resp.status_code == 200:
                me_data = me_resp.json()
                skill._user_id = me_data.get("id")
                return SkillResult(
                    success=True,
                    message=f"Logged in as {me_data.get('name', 'unknown')}",
                    data={
                        "user_id": me_data.get("id"),
                        "name": me_data.get("name"),
                        "email": me_data.get("email"),
                    }
                )

    # Fallback: browser automation login for obtaining tokens
    # This is a placeholder for Playwright-based login flow
    _ensure_content_platform()
    try:
        from skills.account_creator import AccountCreator

        credentials = {
            k: skill.credentials.get(k) or os.environ.get(k, "")
            for k in ("CAPSOLVER_API_KEY", "TWOCAPTCHA_API_KEY",
                       "PACKETSTREAM_USERNAME", "PACKETSTREAM_PASSWORD")
        }

        creator = AccountCreator(credentials=credentials, use_proxy=True)
        try:
            result = await creator.execute("login", {
                "site": "facebook.com",
                "email": email,
                "password": password,
            })
            if result.success:
                skill._access_token = result.data.get("access_token")
                skill._user_id = result.data.get("user_id")
                return SkillResult(
                    success=True,
                    message=f"Logged in to Facebook via browser",
                    data={
                        "user_id": result.data.get("user_id"),
                        "name": result.data.get("name"),
                    }
                )
            return SkillResult(success=False, message=f"Browser login failed: {result.message}")
        finally:
            await creator.close()
    except ImportError:
        return SkillResult(
            success=False,
            message="Login requires either FACEBOOK_APP_ID/FACEBOOK_APP_SECRET or the content-platform browser module"
        )


async def post(skill, params: Dict) -> SkillResult:
    """Create a post on the authenticated user's timeline."""
    message = params.get("message")
    if not message:
        return SkillResult(success=False, message="Missing required parameter: message")

    if not await skill._ensure_login():
        return SkillResult(success=False, message="Not authenticated. Call login first or provide FACEBOOK_ACCESS_TOKEN.")

    post_data = {
        "message": message,
        "access_token": skill._access_token,
    }

    link = params.get("link")
    if link:
        post_data["link"] = link

    privacy = params.get("privacy", "EVERYONE")
    privacy_map = {
        "EVERYONE": {"value": "EVERYONE"},
        "ALL_FRIENDS": {"value": "ALL_FRIENDS"},
        "SELF": {"value": "SELF"},
    }
    if privacy in privacy_map:
        import json
        post_data["privacy"] = json.dumps(privacy_map[privacy])

    resp = await skill.http.post(
        f"{skill.GRAPH_API}/{skill._user_id}/feed",
        data=post_data
    )

    if resp.status_code == 200:
        post_id = resp.json().get("id", "")
        return SkillResult(
            success=True,
            message=f"Posted to Facebook timeline",
            data={"post_id": post_id, "message": message[:100]}
        )

    return SkillResult(success=False, message=f"Post failed: {resp.status_code} {resp.text[:200]}")


async def like(skill, params: Dict) -> SkillResult:
    """Like a Facebook post or object."""
    object_id = params.get("object_id")
    if not object_id:
        return SkillResult(success=False, message="Missing required parameter: object_id")

    if not await skill._ensure_login():
        return SkillResult(success=False, message="Not authenticated. Call login first or provide FACEBOOK_ACCESS_TOKEN.")

    resp = await skill.http.post(
        f"{skill.GRAPH_API}/{object_id}/likes",
        data={"access_token": skill._access_token}
    )

    if resp.status_code == 200:
        success = resp.json().get("success", False)
        if success:
            return SkillResult(success=True, message=f"Liked object {object_id}", data={"object_id": object_id})
        return SkillResult(success=False, message="Like request returned success=false")

    return SkillResult(success=False, message=f"Like failed: {resp.status_code} {resp.text[:200]}")


async def comment(skill, params: Dict) -> SkillResult:
    """Comment on a Facebook post."""
    post_id = params.get("post_id")
    message = params.get("message")
    if not post_id or not message:
        return SkillResult(success=False, message="Missing required parameters: post_id, message")

    if not await skill._ensure_login():
        return SkillResult(success=False, message="Not authenticated. Call login first or provide FACEBOOK_ACCESS_TOKEN.")

    resp = await skill.http.post(
        f"{skill.GRAPH_API}/{post_id}/comments",
        data={
            "message": message,
            "access_token": skill._access_token,
        }
    )

    if resp.status_code == 200:
        comment_id = resp.json().get("id", "")
        return SkillResult(
            success=True,
            message="Commented on Facebook post",
            data={"post_id": post_id, "comment_id": comment_id, "message": message[:100]}
        )

    return SkillResult(success=False, message=f"Comment failed: {resp.status_code} {resp.text[:200]}")


async def get_page(skill, params: Dict) -> SkillResult:
    """Get details of a Facebook Page."""
    page_id = params.get("page_id")
    if not page_id:
        return SkillResult(success=False, message="Missing required parameter: page_id")

    if not await skill._ensure_login():
        return SkillResult(success=False, message="Not authenticated. Call login first or provide FACEBOOK_ACCESS_TOKEN.")

    fields = params.get("fields", "id,name,fan_count,about,category,website,link,description,phone,location")

    resp = await skill.http.get(
        f"{skill.GRAPH_API}/{page_id}",
        params={
            "fields": fields,
            "access_token": skill._access_token,
        }
    )

    if resp.status_code == 200:
        data = resp.json()
        return SkillResult(
            success=True,
            message=f"Page: {data.get('name', 'unknown')}",
            data={
                "id": data.get("id"),
                "name": data.get("name"),
                "fan_count": data.get("fan_count"),
                "about": data.get("about"),
                "category": data.get("category"),
                "website": data.get("website"),
                "link": data.get("link"),
                "description": data.get("description"),
                "phone": data.get("phone"),
                "location": data.get("location"),
            }
        )

    return SkillResult(success=False, message=f"Failed to get page: {resp.status_code} {resp.text[:200]}")


async def post_to_page(skill, params: Dict) -> SkillResult:
    """Post content to a managed Facebook Page."""
    page_id = params.get("page_id")
    message = params.get("message")
    if not page_id or not message:
        return SkillResult(success=False, message="Missing required parameters: page_id, message")

    if not await skill._ensure_login():
        return SkillResult(success=False, message="Not authenticated. Call login first or provide FACEBOOK_ACCESS_TOKEN.")

    # Get page-specific access token
    page_token = await skill._get_page_token(page_id)
    if not page_token:
        return SkillResult(
            success=False,
            message=f"No page access token for page {page_id}. Ensure you manage this page."
        )

    post_data = {
        "message": message,
        "access_token": page_token,
    }

    link = params.get("link")
    if link:
        post_data["link"] = link

    published = params.get("published", "true").lower() == "true"
    post_data["published"] = str(published).lower()

    scheduled_time = params.get("scheduled_publish_time")
    if scheduled_time and not published:
        post_data["scheduled_publish_time"] = scheduled_time

    resp = await skill.http.post(
        f"{skill.GRAPH_API}/{page_id}/feed",
        data=post_data
    )

    if resp.status_code == 200:
        post_id = resp.json().get("id", "")
        status = "published" if published else "scheduled"
        return SkillResult(
            success=True,
            message=f"Posted to page ({status})",
            data={"post_id": post_id, "page_id": page_id, "message": message[:100], "status": status}
        )

    return SkillResult(success=False, message=f"Page post failed: {resp.status_code} {resp.text[:200]}")


async def get_feed(skill, params: Dict) -> SkillResult:
    """Get posts from the user's feed or a page feed."""
    if not await skill._ensure_login():
        return SkillResult(success=False, message="Not authenticated. Call login first or provide FACEBOOK_ACCESS_TOKEN.")

    source_id = params.get("source_id") or skill._user_id
    count = min(int(params.get("count", 10)), 100)

    resp = await skill.http.get(
        f"{skill.GRAPH_API}/{source_id}/feed",
        params={
            "fields": "id,message,created_time,from,permalink_url,type,likes.summary(true),comments.summary(true),shares",
            "limit": count,
            "access_token": skill._access_token,
        }
    )

    if resp.status_code == 200:
        data = resp.json()
        posts = []
        for item in data.get("data", []):
            posts.append({
                "id": item.get("id"),
                "message": (item.get("message") or "")[:200],
                "created_time": item.get("created_time"),
                "from": item.get("from", {}).get("name"),
                "permalink_url": item.get("permalink_url"),
                "type": item.get("type"),
                "likes_count": item.get("likes", {}).get("summary", {}).get("total_count", 0),
                "comments_count": item.get("comments", {}).get("summary", {}).get("total_count", 0),
                "shares_count": item.get("shares", {}).get("count", 0),
            })
        return SkillResult(
            success=True,
            message=f"Retrieved {len(posts)} posts from feed",
            data={"source_id": source_id, "count": len(posts), "posts": posts}
        )

    return SkillResult(success=False, message=f"Failed to get feed: {resp.status_code} {resp.text[:200]}")
