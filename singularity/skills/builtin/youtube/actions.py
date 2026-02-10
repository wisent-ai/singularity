"""
YouTube Data API v3 actions.
"""

import json
from pathlib import Path
from typing import Dict

from singularity.skills.base import SkillResult


async def search(skill, params: Dict) -> SkillResult:
    """Search YouTube videos by query."""
    query = params.get("query")
    if not query:
        return SkillResult(success=False, message="Missing required: query")

    resp = await skill.http.get(
        f"{skill.API}/search",
        headers=skill._read_headers(),
        params={
            "key": skill._api_key(),
            "q": query,
            "part": "snippet",
            "maxResults": min(int(params.get("max_results", 10)), 50),
            "order": params.get("order", "relevance"),
            "type": params.get("type", "video"),
        }
    )
    if resp.status_code == 200:
        items = resp.json().get("items", [])
        results = [{
            "video_id": item.get("id", {}).get("videoId"),
            "channel_id": item.get("id", {}).get("channelId"),
            "title": item["snippet"]["title"],
            "channel": item["snippet"]["channelTitle"],
            "description": item["snippet"]["description"][:200],
            "published_at": item["snippet"]["publishedAt"],
        } for item in items]
        return SkillResult(
            success=True, message=f"Found {len(results)} results for '{query}'",
            data={"query": query, "results": results}
        )
    return SkillResult(success=False, message=f"Search failed: {resp.status_code} {resp.text[:200]}")


async def get_video(skill, params: Dict) -> SkillResult:
    """Get detailed info and stats for a video."""
    video_id = params.get("video_id")
    if not video_id:
        return SkillResult(success=False, message="Missing required: video_id")

    resp = await skill.http.get(
        f"{skill.API}/videos",
        headers=skill._read_headers(),
        params={"key": skill._api_key(), "id": video_id, "part": "snippet,statistics,contentDetails"}
    )
    if resp.status_code == 200:
        items = resp.json().get("items", [])
        if not items:
            return SkillResult(success=False, message=f"Video {video_id} not found")
        item = items[0]
        stats = item.get("statistics", {})
        return SkillResult(
            success=True, message=f"Video: {item['snippet']['title']}",
            data={
                "video_id": video_id, "title": item["snippet"]["title"],
                "channel": item["snippet"]["channelTitle"],
                "description": item["snippet"]["description"][:500],
                "published_at": item["snippet"]["publishedAt"],
                "duration": item.get("contentDetails", {}).get("duration"),
                "views": int(stats.get("viewCount", 0)),
                "likes": int(stats.get("likeCount", 0)),
                "comments": int(stats.get("commentCount", 0)),
            }
        )
    return SkillResult(success=False, message=f"Get video failed: {resp.status_code}")


async def get_channel(skill, params: Dict) -> SkillResult:
    """Get channel info and stats."""
    channel_id = params.get("channel_id")
    username = params.get("username")

    url_params = {"key": skill._api_key(), "part": "snippet,statistics"}
    if channel_id:
        url_params["id"] = channel_id
    elif username:
        url_params["forUsername"] = username
    else:
        # Get authenticated user's channel
        url_params = {"part": "snippet,statistics", "mine": "true"}
        resp = await skill.http.get(
            f"{skill.API}/channels", headers=skill._write_headers(), params=url_params
        )
        return _parse_channel_response(resp)

    resp = await skill.http.get(f"{skill.API}/channels", headers=skill._read_headers(), params=url_params)
    return _parse_channel_response(resp)


async def upload_video(skill, params: Dict) -> SkillResult:
    """Upload a video to YouTube via resumable upload."""
    file_path = params.get("file_path")
    title = params.get("title")
    if not file_path or not title:
        return SkillResult(success=False, message="Missing required: file_path, title")

    path = Path(file_path)
    if not path.exists():
        return SkillResult(success=False, message=f"File not found: {file_path}")

    metadata = {
        "snippet": {
            "title": title,
            "description": params.get("description", ""),
            "tags": [t.strip() for t in params.get("tags", "").split(",") if t.strip()],
            "categoryId": params.get("category_id", "22"),
        },
        "status": {"privacyStatus": params.get("privacy", "private")}
    }

    # Step 1: Initiate resumable upload
    headers = skill._write_headers()
    headers["X-Upload-Content-Type"] = "video/*"
    headers["X-Upload-Content-Length"] = str(path.stat().st_size)

    init_resp = await skill.http.post(
        f"{skill.UPLOAD_API}?uploadType=resumable&part=snippet,status",
        headers=headers, json=metadata
    )
    if init_resp.status_code not in (200, 308):
        return SkillResult(success=False, message=f"Upload init failed: {init_resp.status_code} {init_resp.text[:200]}")

    upload_url = init_resp.headers.get("Location")
    if not upload_url:
        return SkillResult(success=False, message="No upload URL in response")

    # Step 2: Upload the video file
    video_bytes = path.read_bytes()
    upload_resp = await skill.http.put(
        upload_url,
        headers={"Content-Type": "video/*"},
        content=video_bytes,
        
    )

    if upload_resp.status_code in (200, 201):
        data = upload_resp.json()
        video_id = data.get("id")
        return SkillResult(
            success=True,
            message=f"Uploaded: {title}",
            data={"video_id": video_id, "title": title,
                  "url": f"https://www.youtube.com/watch?v={video_id}"},
            asset_created=f"https://www.youtube.com/watch?v={video_id}"
        )
    return SkillResult(success=False, message=f"Upload failed: {upload_resp.status_code} {upload_resp.text[:200]}")


async def comment(skill, params: Dict) -> SkillResult:
    """Post a comment on a video."""
    video_id = params.get("video_id")
    text = params.get("text")
    if not video_id or not text:
        return SkillResult(success=False, message="Missing required: video_id, text")

    resp = await skill.http.post(
        f"{skill.API}/commentThreads",
        headers=skill._write_headers(),
        params={"part": "snippet"},
        json={
            "snippet": {
                "videoId": video_id,
                "topLevelComment": {"snippet": {"textOriginal": text}}
            }
        }
    )
    if resp.status_code in (200, 201):
        return SkillResult(success=True, message="Comment posted",
                           data={"video_id": video_id, "text": text[:100]})
    return SkillResult(success=False, message=f"Comment failed: {resp.status_code} {resp.text[:200]}")


async def get_comments(skill, params: Dict) -> SkillResult:
    """Get comments on a video."""
    video_id = params.get("video_id")
    if not video_id:
        return SkillResult(success=False, message="Missing required: video_id")

    resp = await skill.http.get(
        f"{skill.API}/commentThreads",
        headers=skill._read_headers(),
        params={
            "key": skill._api_key(), "videoId": video_id, "part": "snippet",
            "maxResults": min(int(params.get("max_results", 20)), 100),
            "order": params.get("order", "relevance"),
        }
    )
    if resp.status_code == 200:
        items = resp.json().get("items", [])
        comments = [{
            "author": c["snippet"]["topLevelComment"]["snippet"]["authorDisplayName"],
            "text": c["snippet"]["topLevelComment"]["snippet"]["textDisplay"][:300],
            "likes": c["snippet"]["topLevelComment"]["snippet"]["likeCount"],
            "published_at": c["snippet"]["topLevelComment"]["snippet"]["publishedAt"],
        } for c in items]
        return SkillResult(
            success=True, message=f"Got {len(comments)} comments",
            data={"video_id": video_id, "comments": comments}
        )
    return SkillResult(success=False, message=f"Get comments failed: {resp.status_code}")


async def subscribe(skill, params: Dict) -> SkillResult:
    """Subscribe to a YouTube channel."""
    channel_id = params.get("channel_id")
    if not channel_id:
        return SkillResult(success=False, message="Missing required: channel_id")

    resp = await skill.http.post(
        f"{skill.API}/subscriptions",
        headers=skill._write_headers(),
        params={"part": "snippet"},
        json={"snippet": {"resourceId": {"kind": "youtube#channel", "channelId": channel_id}}}
    )
    if resp.status_code in (200, 201):
        return SkillResult(success=True, message=f"Subscribed to {channel_id}",
                           data={"channel_id": channel_id})
    return SkillResult(success=False, message=f"Subscribe failed: {resp.status_code} {resp.text[:200]}")


def _parse_channel_response(resp) -> SkillResult:
    """Parse channel API response."""
    if resp.status_code == 200:
        items = resp.json().get("items", [])
        if not items:
            return SkillResult(success=False, message="Channel not found")
        ch = items[0]
        stats = ch.get("statistics", {})
        return SkillResult(
            success=True, message=f"Channel: {ch['snippet']['title']}",
            data={
                "channel_id": ch["id"], "title": ch["snippet"]["title"],
                "description": ch["snippet"]["description"][:300],
                "subscribers": int(stats.get("subscriberCount", 0)),
                "videos": int(stats.get("videoCount", 0)),
                "views": int(stats.get("viewCount", 0)),
            }
        )
    return SkillResult(success=False, message=f"Channel lookup failed: {resp.status_code}")
