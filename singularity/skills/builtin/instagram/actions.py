"""
Instagram API actions.
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
    """Create a new Instagram account via Playwright browser automation."""
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
            "site": "instagram.com",
            "username": params.get("username"),
            "email": params.get("email"),
            "password": params.get("password"),
            "full_name": params.get("full_name"),
        })
        if result.success:
            return SkillResult(
                success=True,
                message=f"Instagram account created: {result.data.get('username', 'unknown')}",
                data=result.data
            )
        return SkillResult(success=False, message=f"Account creation failed: {result.message}",
                           data=result.data)
    finally:
        await creator.close()


async def login(skill, params: Dict) -> SkillResult:
    """Login to Instagram and obtain session tokens."""
    username = params.get("username") or skill._get_credential("INSTAGRAM_USERNAME")
    password = params.get("password") or skill._get_credential("INSTAGRAM_PASSWORD")

    if not username or not password:
        return SkillResult(success=False, message="Missing credentials: INSTAGRAM_USERNAME and INSTAGRAM_PASSWORD required")

    # Attempt login via Instagram's web API
    login_url = "https://www.instagram.com/accounts/login/ajax/"
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "X-CSRFToken": "",
        "X-Requested-With": "XMLHttpRequest",
        "Referer": "https://www.instagram.com/accounts/login/",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    }

    # First get CSRF token
    csrf_resp = await skill.http.get("https://www.instagram.com/accounts/login/", headers={
        "User-Agent": headers["User-Agent"],
    })

    csrf_token = ""
    if csrf_resp.status_code == 200:
        for cookie_name, cookie_value in csrf_resp.cookies.items():
            if cookie_name == "csrftoken":
                csrf_token = cookie_value
                break

    headers["X-CSRFToken"] = csrf_token

    resp = await skill.http.post(login_url, headers=headers, data={
        "username": username,
        "enc_password": f"#PWD_INSTAGRAM_BROWSER:0:0:{password}",
        "queryParams": "{}",
        "optIntoOneTap": "false",
    })

    if resp.status_code == 200:
        data = resp.json()
        if data.get("authenticated"):
            skill._user_id = data.get("userId")
            # Extract session cookies
            session_id = ""
            for cookie_name, cookie_value in resp.cookies.items():
                if cookie_name == "sessionid":
                    session_id = cookie_value
            return SkillResult(
                success=True,
                message=f"Logged in as {username}",
                data={
                    "user_id": data.get("userId"),
                    "username": username,
                    "session_id": session_id[:8] + "..." if session_id else "",
                }
            )
        elif data.get("two_factor_required"):
            return SkillResult(
                success=False,
                message="Two-factor authentication required",
                data={"two_factor_info": data.get("two_factor_info", {})}
            )
        else:
            return SkillResult(success=False, message=f"Login failed: {data.get('message', 'invalid credentials')}")

    return SkillResult(success=False, message=f"Login request failed: {resp.status_code} {resp.text[:200]}")


async def post_photo(skill, params: Dict) -> SkillResult:
    """Post a photo to Instagram feed via Graph API."""
    image_url = params.get("image_url")
    if not image_url:
        return SkillResult(success=False, message="Missing required parameter: image_url")

    if not await skill._ensure_login():
        return SkillResult(success=False, message="Not authenticated. Call login first or provide INSTAGRAM_ACCESS_TOKEN.")

    caption = params.get("caption", "")
    location_id = params.get("location_id")
    user_tags = params.get("user_tags")

    # Step 1: Create media container
    container_data = {
        "image_url": image_url,
        "caption": caption,
        "access_token": skill._access_token,
    }

    if location_id:
        container_data["location_id"] = location_id

    if user_tags:
        # Format user tags as Instagram expects
        tags = [u.strip().lstrip("@") for u in user_tags.split(",")]
        tag_list = [{"username": tag, "x": 0.5, "y": 0.5} for tag in tags]
        import json
        container_data["user_tags"] = json.dumps(tag_list)

    resp = await skill.http.post(
        f"{skill.GRAPH_API}/{skill._user_id}/media",
        headers=skill._graph_headers(),
        data=container_data
    )

    if resp.status_code != 200:
        return SkillResult(success=False, message=f"Failed to create media container: {resp.status_code} {resp.text[:200]}")

    container_id = resp.json().get("id")
    if not container_id:
        return SkillResult(success=False, message="No container ID returned from Instagram")

    # Step 2: Publish the container
    publish_resp = await skill.http.post(
        f"{skill.GRAPH_API}/{skill._user_id}/media_publish",
        data={
            "creation_id": container_id,
            "access_token": skill._access_token,
        }
    )

    if publish_resp.status_code == 200:
        media_id = publish_resp.json().get("id", "")
        return SkillResult(
            success=True,
            message="Photo posted to Instagram",
            data={"media_id": media_id, "caption": caption[:100]}
        )

    return SkillResult(success=False, message=f"Publish failed: {publish_resp.status_code} {publish_resp.text[:200]}")


async def post_reel(skill, params: Dict) -> SkillResult:
    """Post a reel/video to Instagram via Graph API."""
    video_url = params.get("video_url")
    if not video_url:
        return SkillResult(success=False, message="Missing required parameter: video_url")

    if not await skill._ensure_login():
        return SkillResult(success=False, message="Not authenticated. Call login first or provide INSTAGRAM_ACCESS_TOKEN.")

    caption = params.get("caption", "")
    cover_url = params.get("cover_url")
    share_to_feed = params.get("share_to_feed", "true").lower() == "true"

    # Step 1: Create reel container
    container_data = {
        "video_url": video_url,
        "caption": caption,
        "media_type": "REELS",
        "share_to_feed": str(share_to_feed).lower(),
        "access_token": skill._access_token,
    }

    if cover_url:
        container_data["cover_url"] = cover_url

    resp = await skill.http.post(
        f"{skill.GRAPH_API}/{skill._user_id}/media",
        headers=skill._graph_headers(),
        data=container_data
    )

    if resp.status_code != 200:
        return SkillResult(success=False, message=f"Failed to create reel container: {resp.status_code} {resp.text[:200]}")

    container_id = resp.json().get("id")
    if not container_id:
        return SkillResult(success=False, message="No container ID returned from Instagram")

    # Step 2: Wait for video processing and check status
    import asyncio
    max_retries = 30
    for _ in range(max_retries):
        status_resp = await skill.http.get(
            f"{skill.GRAPH_API}/{container_id}",
            params={"fields": "status_code", "access_token": skill._access_token}
        )
        if status_resp.status_code == 200:
            status = status_resp.json().get("status_code")
            if status == "FINISHED":
                break
            elif status == "ERROR":
                return SkillResult(success=False, message="Instagram video processing failed")
        await asyncio.sleep(2)

    # Step 3: Publish the reel
    publish_resp = await skill.http.post(
        f"{skill.GRAPH_API}/{skill._user_id}/media_publish",
        data={
            "creation_id": container_id,
            "access_token": skill._access_token,
        }
    )

    if publish_resp.status_code == 200:
        media_id = publish_resp.json().get("id", "")
        return SkillResult(
            success=True,
            message="Reel posted to Instagram",
            data={"media_id": media_id, "caption": caption[:100], "type": "REELS"}
        )

    return SkillResult(success=False, message=f"Reel publish failed: {publish_resp.status_code} {publish_resp.text[:200]}")


async def like(skill, params: Dict) -> SkillResult:
    """Like an Instagram post via the web API."""
    media_id = params.get("media_id")
    if not media_id:
        return SkillResult(success=False, message="Missing required parameter: media_id")

    if not await skill._ensure_login():
        return SkillResult(success=False, message="Not authenticated. Call login first or provide INSTAGRAM_ACCESS_TOKEN.")

    # Use web API for liking
    resp = await skill.http.post(
        f"https://www.instagram.com/api/v1/web/likes/{media_id}/like/",
        headers={
            "X-CSRFToken": skill.credentials.get("_csrf_token", ""),
            "X-Requested-With": "XMLHttpRequest",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Referer": "https://www.instagram.com/",
        }
    )

    if resp.status_code == 200:
        return SkillResult(success=True, message=f"Liked post {media_id}", data={"media_id": media_id})

    # Fallback: try Graph API comment like endpoint
    graph_resp = await skill.http.post(
        f"{skill.GRAPH_API}/{media_id}/likes",
        data={"access_token": skill._access_token}
    )

    if graph_resp.status_code == 200:
        return SkillResult(success=True, message=f"Liked post {media_id}", data={"media_id": media_id})

    return SkillResult(success=False, message=f"Like failed: {resp.status_code} {resp.text[:200]}")


async def comment(skill, params: Dict) -> SkillResult:
    """Comment on an Instagram post."""
    media_id = params.get("media_id")
    text = params.get("text")
    if not media_id or not text:
        return SkillResult(success=False, message="Missing required parameters: media_id, text")

    if not await skill._ensure_login():
        return SkillResult(success=False, message="Not authenticated. Call login first or provide INSTAGRAM_ACCESS_TOKEN.")

    resp = await skill.http.post(
        f"{skill.GRAPH_API}/{media_id}/comments",
        data={
            "message": text,
            "access_token": skill._access_token,
        }
    )

    if resp.status_code == 200:
        comment_id = resp.json().get("id", "")
        return SkillResult(
            success=True,
            message="Commented on Instagram post",
            data={"media_id": media_id, "comment_id": comment_id, "text": text[:100]}
        )

    return SkillResult(success=False, message=f"Comment failed: {resp.status_code} {resp.text[:200]}")


async def get_profile(skill, params: Dict) -> SkillResult:
    """Get an Instagram user profile."""
    username = params.get("username")
    user_id = params.get("user_id")

    if not await skill._ensure_login():
        return SkillResult(success=False, message="Not authenticated. Call login first or provide INSTAGRAM_ACCESS_TOKEN.")

    if user_id:
        # Direct Graph API lookup by ID
        resp = await skill.http.get(
            f"{skill.GRAPH_API}/{user_id}",
            params={
                "fields": "id,username,name,biography,followers_count,follows_count,media_count,profile_picture_url,website",
                "access_token": skill._access_token,
            }
        )
    elif username:
        # Search by username via business discovery
        resp = await skill.http.get(
            f"{skill.GRAPH_API}/{skill._user_id}",
            params={
                "fields": f"business_discovery.fields(id,username,name,biography,followers_count,follows_count,media_count,profile_picture_url,website).username({username})",
                "access_token": skill._access_token,
            }
        )
    else:
        # Get own profile
        resp = await skill.http.get(
            f"{skill.GRAPH_API}/me",
            params={
                "fields": "id,username,name,biography,followers_count,follows_count,media_count,profile_picture_url,website",
                "access_token": skill._access_token,
            }
        )

    if resp.status_code == 200:
        data = resp.json()
        # Handle business_discovery wrapper
        if "business_discovery" in data:
            data = data["business_discovery"]
        return SkillResult(
            success=True,
            message=f"Profile: @{data.get('username', 'unknown')}",
            data={
                "id": data.get("id"),
                "username": data.get("username"),
                "name": data.get("name"),
                "biography": data.get("biography"),
                "followers_count": data.get("followers_count"),
                "follows_count": data.get("follows_count"),
                "media_count": data.get("media_count"),
                "profile_picture_url": data.get("profile_picture_url"),
                "website": data.get("website"),
            }
        )

    return SkillResult(success=False, message=f"Failed to get profile: {resp.status_code} {resp.text[:200]}")


async def search_hashtag(skill, params: Dict) -> SkillResult:
    """Search Instagram posts by hashtag via Graph API."""
    hashtag = params.get("hashtag")
    if not hashtag:
        return SkillResult(success=False, message="Missing required parameter: hashtag")

    if not await skill._ensure_login():
        return SkillResult(success=False, message="Not authenticated. Call login first or provide INSTAGRAM_ACCESS_TOKEN.")

    # Remove # prefix if present
    hashtag = hashtag.lstrip("#")
    count = min(int(params.get("count", 20)), 50)

    # Step 1: Get hashtag ID
    search_resp = await skill.http.get(
        f"{skill.GRAPH_API}/ig_hashtag_search",
        params={
            "q": hashtag,
            "user_id": skill._user_id,
            "access_token": skill._access_token,
        }
    )

    if search_resp.status_code != 200:
        return SkillResult(success=False, message=f"Hashtag search failed: {search_resp.status_code} {search_resp.text[:200]}")

    hashtag_data = search_resp.json().get("data", [])
    if not hashtag_data:
        return SkillResult(success=False, message=f"Hashtag '#{hashtag}' not found")

    hashtag_id = hashtag_data[0].get("id")

    # Step 2: Get recent media for the hashtag
    media_resp = await skill.http.get(
        f"{skill.GRAPH_API}/{hashtag_id}/recent_media",
        params={
            "user_id": skill._user_id,
            "fields": "id,caption,media_type,media_url,permalink,timestamp,like_count,comments_count",
            "limit": count,
            "access_token": skill._access_token,
        }
    )

    if media_resp.status_code == 200:
        media_data = media_resp.json().get("data", [])
        results = []
        for item in media_data:
            results.append({
                "id": item.get("id"),
                "caption": (item.get("caption") or "")[:200],
                "media_type": item.get("media_type"),
                "permalink": item.get("permalink"),
                "timestamp": item.get("timestamp"),
                "like_count": item.get("like_count"),
                "comments_count": item.get("comments_count"),
            })
        return SkillResult(
            success=True,
            message=f"Found {len(results)} posts for #{hashtag}",
            data={"hashtag": hashtag, "hashtag_id": hashtag_id, "posts": results}
        )

    return SkillResult(success=False, message=f"Failed to get hashtag media: {media_resp.status_code} {media_resp.text[:200]}")
