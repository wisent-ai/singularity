"""
LinkedIn API actions.
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
        # Resolve relative to the Wisent project root
        cp_path = str(Path(__file__).resolve().parents[6] / "content-platform" / "account-api")
    if cp_path not in sys.path:
        sys.path.insert(0, cp_path)


async def create_account(skill, params: Dict) -> SkillResult:
    """Create a new LinkedIn account via Playwright browser automation."""
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
            "site": "linkedin.com",
            "username": params.get("username"),
            "email": params.get("email"),
            "password": params.get("password"),
        })
        if result.success:
            return SkillResult(
                success=True,
                message=f"LinkedIn account created: {result.data.get('username', 'unknown')}",
                data=result.data
            )
        return SkillResult(success=False, message=f"Account creation failed: {result.message}",
                           data=result.data)
    finally:
        await creator.close()


async def get_profile(skill, params: Dict) -> SkillResult:
    """Get the authenticated user's profile or another member's."""
    member_id = params.get("member_id")

    if member_id:
        url = f"{skill.API}/people/(id:{member_id})"
    else:
        url = f"{skill.API}/me"

    resp = await skill.http.get(
        url,
        headers=skill._headers(),
        params={"projection": "(id,firstName,lastName,headline,vanityName)"}
    )

    if resp.status_code == 200:
        data = resp.json()
        # LinkedIn localizes names
        first = _localized(data.get("firstName", {}))
        last = _localized(data.get("lastName", {}))
        return SkillResult(
            success=True,
            message=f"Profile: {first} {last}",
            data={
                "id": data.get("id"),
                "first_name": first,
                "last_name": last,
                "headline": _localized(data.get("headline", {})),
                "vanity_name": data.get("vanityName"),
            }
        )
    return SkillResult(success=False, message=f"Failed to get profile: {resp.status_code} {resp.text[:200]}")


async def create_post(skill, params: Dict) -> SkillResult:
    """Create a text or link post."""
    text = params.get("text")
    if not text:
        return SkillResult(success=False, message="Missing required parameter: text")

    author_urn = await skill._get_member_urn()
    if not author_urn:
        return SkillResult(success=False, message="Could not determine member URN. Check access token.")

    visibility = params.get("visibility", "PUBLIC")
    link_url = params.get("link_url")

    body = {
        "author": author_urn,
        "lifecycleState": "PUBLISHED",
        "specificContent": {
            "com.linkedin.ugc.ShareContent": {
                "shareCommentary": {"text": text},
                "shareMediaCategory": "NONE"
            }
        },
        "visibility": {"com.linkedin.ugc.MemberNetworkVisibility": visibility}
    }

    if link_url:
        body["specificContent"]["com.linkedin.ugc.ShareContent"]["shareMediaCategory"] = "ARTICLE"
        body["specificContent"]["com.linkedin.ugc.ShareContent"]["media"] = [{
            "status": "READY",
            "originalUrl": link_url,
        }]

    resp = await skill.http.post(f"{skill.API}/ugcPosts", headers=skill._headers(), json=body)

    if resp.status_code in (200, 201):
        post_id = resp.json().get("id", resp.headers.get("x-restli-id", ""))
        return SkillResult(
            success=True,
            message=f"Posted to LinkedIn ({visibility})",
            data={"post_id": post_id, "text": text[:100]}
        )
    return SkillResult(success=False, message=f"Post failed: {resp.status_code} {resp.text[:200]}")


async def like_post(skill, params: Dict) -> SkillResult:
    """React to a LinkedIn post."""
    post_urn = params.get("post_urn")
    if not post_urn:
        return SkillResult(success=False, message="Missing required parameter: post_urn")

    actor_urn = await skill._get_member_urn()
    if not actor_urn:
        return SkillResult(success=False, message="Could not determine member URN.")

    body = {
        "actor": actor_urn,
        "object": post_urn,
    }

    resp = await skill.http.post(
        f"{skill.REST_API}/reactions?action=create",
        headers=skill._headers(),
        json=body
    )

    if resp.status_code in (200, 201):
        return SkillResult(success=True, message=f"Liked {post_urn}", data={"post_urn": post_urn})
    return SkillResult(success=False, message=f"Like failed: {resp.status_code} {resp.text[:200]}")


async def comment(skill, params: Dict) -> SkillResult:
    """Comment on a LinkedIn post."""
    post_urn = params.get("post_urn")
    text = params.get("text")
    if not post_urn or not text:
        return SkillResult(success=False, message="Missing required parameters: post_urn, text")

    actor_urn = await skill._get_member_urn()
    if not actor_urn:
        return SkillResult(success=False, message="Could not determine member URN.")

    body = {
        "actor": actor_urn,
        "object": post_urn,
        "message": {"text": text},
    }

    resp = await skill.http.post(
        f"{skill.REST_API}/socialActions/{post_urn}/comments",
        headers=skill._headers(),
        json=body
    )

    if resp.status_code in (200, 201):
        return SkillResult(
            success=True, message="Commented on LinkedIn post",
            data={"post_urn": post_urn, "text": text[:100]}
        )
    return SkillResult(success=False, message=f"Comment failed: {resp.status_code} {resp.text[:200]}")


async def search_posts(skill, params: Dict) -> SkillResult:
    """Search LinkedIn posts by keyword."""
    query = params.get("query")
    if not query:
        return SkillResult(success=False, message="Missing required parameter: query")

    count = min(int(params.get("count", 10)), 50)

    resp = await skill.http.get(
        f"{skill.API}/search/blended",
        headers=skill._headers(),
        params={"q": "all", "keywords": query, "count": count}
    )

    if resp.status_code == 200:
        data = resp.json()
        elements = data.get("elements", [])
        results = []
        for elem in elements:
            for item in elem.get("elements", []):
                results.append({
                    "title": item.get("title", {}).get("text", ""),
                    "url": item.get("navigationUrl", ""),
                })
        return SkillResult(
            success=True, message=f"Found {len(results)} results for '{query}'",
            data={"query": query, "results": results[:count]}
        )
    return SkillResult(success=False, message=f"Search failed: {resp.status_code} {resp.text[:200]}")


async def get_connections(skill, params: Dict) -> SkillResult:
    """Get connection count and recent connections."""
    resp = await skill.http.get(
        f"{skill.API}/connections",
        headers=skill._headers(),
        params={"q": "viewer", "count": min(int(params.get("count", 10)), 50)}
    )

    if resp.status_code == 200:
        data = resp.json()
        total = data.get("paging", {}).get("total", 0)
        elements = data.get("elements", [])
        return SkillResult(
            success=True, message=f"{total} connections",
            data={"total": total, "recent_count": len(elements)}
        )
    return SkillResult(success=False, message=f"Failed: {resp.status_code} {resp.text[:200]}")


def _localized(field) -> str:
    """Extract localized LinkedIn field value."""
    if isinstance(field, str):
        return field
    if isinstance(field, dict):
        localized = field.get("localized", {})
        if localized:
            return next(iter(localized.values()), "")
        return field.get("preferredLocale", {}).get("language", "")
    return ""
