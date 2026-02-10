"""
Twitter/X Skill - API Action Methods

Contains the Twitter API action implementations.
"""

from typing import Optional
from singularity.skills.base import SkillResult


async def post_tweet(skill, text: str, reply_to: str = None) -> SkillResult:
    """Post a tweet"""
    if not text:
        return SkillResult(success=False, message="Tweet text required")
    if len(text) > 280:
        return SkillResult(success=False, message="Tweet exceeds 280 characters")

    url = f"{skill.API_BASE_V2}/tweets"
    payload = {"text": text}
    if reply_to:
        payload["reply"] = {"in_reply_to_tweet_id": reply_to}

    headers = {"Authorization": skill._get_oauth_header("POST", url), "Content-Type": "application/json"}
    response = await skill.http.post(url, headers=headers, json=payload)

    if response.status_code in [200, 201]:
        data = response.json()
        tweet_data = data.get("data", {})
        return SkillResult(
            success=True, message=f"Tweet posted: {text[:50]}...",
            data={
                "tweet_id": tweet_data.get("id"), "text": tweet_data.get("text"),
                "url": f"https://twitter.com/i/web/status/{tweet_data.get('id')}"
            },
            asset_created={"type": "tweet", "id": tweet_data.get("id"), "text": text}
        )
    else:
        error = response.json() if response.text else {"error": response.status_code}
        return SkillResult(success=False, message=f"Failed to post tweet: {error}")


async def search_tweets(skill, query: str, max_results: int = 10) -> SkillResult:
    """Search for tweets"""
    if not query:
        return SkillResult(success=False, message="Search query required")

    max_results = min(max(max_results, 10), 100)
    url = f"{skill.API_BASE_V2}/tweets/search/recent"
    params = {"query": query, "max_results": max_results, "tweet.fields": "created_at,author_id,public_metrics"}
    headers = {"Authorization": skill._get_oauth_header("GET", url, params)}
    response = await skill.http.get(url, headers=headers, params=params)

    if response.status_code == 200:
        data = response.json()
        tweets = data.get("data", [])
        return SkillResult(success=True, message=f"Found {len(tweets)} tweets",
                          data={"tweets": tweets, "count": len(tweets), "query": query})
    else:
        error = response.json() if response.text else {"error": response.status_code}
        return SkillResult(success=False, message=f"Search failed: {error}")


async def get_mentions(skill, max_results: int = 10) -> SkillResult:
    """Get mentions of authenticated user"""
    user_id = await skill._get_user_id()
    if not user_id:
        return SkillResult(success=False, message="Could not get user ID")

    url = f"{skill.API_BASE_V2}/users/{user_id}/mentions"
    params = {"max_results": min(max(max_results, 5), 100), "tweet.fields": "created_at,author_id,public_metrics"}
    headers = {"Authorization": skill._get_oauth_header("GET", url, params)}
    response = await skill.http.get(url, headers=headers, params=params)

    if response.status_code == 200:
        data = response.json()
        mentions = data.get("data", [])
        return SkillResult(success=True, message=f"Found {len(mentions)} mentions",
                          data={"mentions": mentions, "count": len(mentions)})
    else:
        error = response.json() if response.text else {"error": response.status_code}
        return SkillResult(success=False, message=f"Failed to get mentions: {error}")


async def follow_user(skill, username: str) -> SkillResult:
    """Follow a user"""
    if not username:
        return SkillResult(success=False, message="Username required")

    target_id = await skill._get_user_id(username)
    if not target_id:
        return SkillResult(success=False, message=f"User not found: {username}")

    user_id = await skill._get_user_id()
    if not user_id:
        return SkillResult(success=False, message="Could not get authenticated user ID")

    url = f"{skill.API_BASE_V2}/users/{user_id}/following"
    headers = {"Authorization": skill._get_oauth_header("POST", url), "Content-Type": "application/json"}
    response = await skill.http.post(url, headers=headers, json={"target_user_id": target_id})

    if response.status_code in [200, 201]:
        data = response.json()
        following = data.get("data", {}).get("following", False)
        return SkillResult(success=True,
                          message=f"{'Now following' if following else 'Follow request sent to'} @{username}",
                          data={"username": username, "following": following})
    else:
        error = response.json() if response.text else {"error": response.status_code}
        return SkillResult(success=False, message=f"Failed to follow: {error}")


async def send_dm(skill, username: str, text: str) -> SkillResult:
    """Send a direct message"""
    if not username or not text:
        return SkillResult(success=False, message="Username and text required")

    target_id = await skill._get_user_id(username)
    if not target_id:
        return SkillResult(success=False, message=f"User not found: {username}")

    url = f"{skill.API_BASE_V2}/dm_conversations/with/{target_id}/messages"
    headers = {"Authorization": skill._get_oauth_header("POST", url), "Content-Type": "application/json"}
    response = await skill.http.post(url, headers=headers, json={"text": text})

    if response.status_code in [200, 201]:
        data = response.json()
        dm_data = data.get("data", {})
        return SkillResult(success=True, message=f"DM sent to @{username}",
                          data={"dm_id": dm_data.get("dm_event_id"), "recipient": username, "text": text})
    else:
        error = response.json() if response.text else {"error": response.status_code}
        return SkillResult(success=False, message=f"Failed to send DM: {error}")


async def like_tweet(skill, tweet_id: str) -> SkillResult:
    """Like a tweet"""
    if not tweet_id:
        return SkillResult(success=False, message="Tweet ID required")

    user_id = await skill._get_user_id()
    if not user_id:
        return SkillResult(success=False, message="Could not get user ID")

    url = f"{skill.API_BASE_V2}/users/{user_id}/likes"
    headers = {"Authorization": skill._get_oauth_header("POST", url), "Content-Type": "application/json"}
    response = await skill.http.post(url, headers=headers, json={"tweet_id": tweet_id})

    if response.status_code in [200, 201]:
        return SkillResult(success=True, message=f"Liked tweet {tweet_id}",
                          data={"tweet_id": tweet_id, "liked": True})
    else:
        error = response.json() if response.text else {"error": response.status_code}
        return SkillResult(success=False, message=f"Failed to like tweet: {error}")


async def retweet(skill, tweet_id: str) -> SkillResult:
    """Retweet a tweet"""
    if not tweet_id:
        return SkillResult(success=False, message="Tweet ID required")

    user_id = await skill._get_user_id()
    if not user_id:
        return SkillResult(success=False, message="Could not get user ID")

    url = f"{skill.API_BASE_V2}/users/{user_id}/retweets"
    headers = {"Authorization": skill._get_oauth_header("POST", url), "Content-Type": "application/json"}
    response = await skill.http.post(url, headers=headers, json={"tweet_id": tweet_id})

    if response.status_code in [200, 201]:
        return SkillResult(success=True, message=f"Retweeted {tweet_id}",
                          data={"tweet_id": tweet_id, "retweeted": True})
    else:
        error = response.json() if response.text else {"error": response.status_code}
        return SkillResult(success=False, message=f"Failed to retweet: {error}")
