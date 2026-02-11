"""Handler functions for TikTokSkill actions."""

import asyncio
from typing import Dict, Any
from singularity.skills.base import SkillResult


def _video_summary(v, fields=("video_id", "url", "author", "views", "caption", "hashtags")):
    """Build a compact video summary dict."""
    out = {}
    if "video_id" in fields: out["video_id"] = v.video_id
    if "url" in fields: out["url"] = v.video_url
    if "author" in fields: out["author"] = v.author_username
    if "views" in fields: out["views"] = v.views
    if "likes" in fields: out["likes"] = v.likes
    if "caption" in fields: out["caption"] = v.caption[:100] if v.caption else None
    if "hashtags" in fields: out["hashtags"] = v.hashtags
    return out


async def scrape_fyp(skill, params: Dict[str, Any]) -> SkillResult:
    scraper = await skill._get_scraper(params.get("headless", False))
    videos = await scraper.scrape_for_you_page(scroll_count=params.get("scroll_count", 10))
    return SkillResult(success=True, data={
        "videos_found": len(videos),
        "videos": [_video_summary(v) for v in videos[:20]]})


async def scrape_hashtag(skill, params: Dict[str, Any]) -> SkillResult:
    hashtag = params["hashtag"]
    scraper = await skill._get_scraper(params.get("headless", False))
    videos = await scraper.scrape_hashtag(hashtag, scroll_count=params.get("scroll_count", 10))
    return SkillResult(success=True, data={
        "hashtag": hashtag, "videos_found": len(videos),
        "videos": [_video_summary(v, ("video_id", "url", "author", "views", "caption")) for v in videos[:20]]})


async def scrape_search(skill, params: Dict[str, Any]) -> SkillResult:
    query = params["query"]
    scraper = await skill._get_scraper(params.get("headless", False))
    videos = await scraper.scrape_search(query, scroll_count=params.get("scroll_count", 10))
    return SkillResult(success=True, data={
        "query": query, "videos_found": len(videos),
        "videos": [_video_summary(v, ("video_id", "url", "author", "views", "caption")) for v in videos[:20]]})


async def scrape_user(skill, params: Dict[str, Any]) -> SkillResult:
    username = params["username"]
    scraper = await skill._get_scraper(headless=True)
    videos = await scraper.scrape_user_videos(username, limit=params.get("limit", 50))
    return SkillResult(success=True, data={
        "username": username, "videos_found": len(videos),
        "videos": [_video_summary(v, ("video_id", "url", "views", "likes", "caption")) for v in videos[:20]]})


async def download_videos(skill, params: Dict[str, Any]) -> SkillResult:
    downloader = skill._get_downloader()
    loop = asyncio.get_event_loop()
    stats = await loop.run_in_executor(None,
        lambda: downloader.download_pending(limit=params.get("limit", 50), scail_only=params.get("scail_only", False)))
    return SkillResult(success=True, data={
        "downloaded": stats.get("downloaded", 0), "failed": stats.get("failed", 0),
        "skipped": stats.get("skipped", 0)})


async def analyze_scail(skill, params: Dict[str, Any]) -> SkillResult:
    f = skill._get_filter()
    loop = asyncio.get_event_loop()
    stats = await loop.run_in_executor(None, lambda: f.analyze_pending(limit=params.get("limit", 50)))
    return SkillResult(success=True, data={
        "suitable": stats.get("suitable", 0), "unsuitable": stats.get("unsuitable", 0),
        "by_movement": stats.get("by_movement", {})})


async def get_scail_videos(skill, params: Dict[str, Any]) -> SkillResult:
    try:
        from scraping.tiktok.database import TikTokDatabase
        db = TikTokDatabase()
        videos = db.get_scail_ready(movement_type=params.get("movement_type"), limit=params.get("limit", 20))
        return SkillResult(success=True, data={"count": len(videos), "videos": [
            {"video_id": v["video_id"], "url": v["video_url"], "movement_type": v.get("movement_type"),
             "views": v.get("views", 0), "duration": v.get("duration_seconds"),
             "local_path": v.get("local_video_path")} for v in videos]})
    except Exception as e:
        return SkillResult(success=False, message=str(e))


async def get_trending_hashtags(skill, params: Dict[str, Any]) -> SkillResult:
    analyzer = skill._get_trend_analyzer()
    hashtags = analyzer.get_hashtag_suggestions(movement_type=params.get("movement_type"))
    return SkillResult(success=True, data={"hashtags": hashtags, "movement_type": params.get("movement_type")})


async def get_viral_sounds(skill, params: Dict[str, Any]) -> SkillResult:
    analyzer = skill._get_trend_analyzer()
    sounds = analyzer.get_sound_suggestions(min_videos=params.get("min_videos", 3))
    return SkillResult(success=True, data={"sounds": [
        {"name": s["name"], "author": s.get("author"), "video_count": s["count"]} for s in sounds]})


async def generate_trend_report(skill, params: Dict[str, Any]) -> SkillResult:
    analyzer = skill._get_trend_analyzer()
    report = analyzer.generate_report()
    return SkillResult(success=True, data={
        "report": report.to_markdown() if hasattr(report, 'to_markdown') else str(report)})


async def analyze_hooks(skill, params: Dict[str, Any]) -> SkillResult:
    try:
        from tiktok.research.hook_extractor import HookExtractor
        extractor = HookExtractor()
        report = extractor.generate_hook_report()
        return SkillResult(success=True, data={"report": report})
    except Exception as e:
        return SkillResult(success=False, message=str(e))


async def get_stats(skill, params: Dict[str, Any]) -> SkillResult:
    try:
        from scraping.tiktok.database import TikTokDatabase
        db = TikTokDatabase()
        downloader = skill._get_downloader()
        db_stats = db.get_stats()
        file_stats = downloader.get_stats()
        return SkillResult(success=True, data={
            "database": {"total_videos": db_stats.get("total_videos", 0),
                         "pending_download": db_stats.get("pending_download", 0),
                         "downloaded": db_stats.get("downloaded", 0),
                         "scail_ready": db_stats.get("scail_ready", 0),
                         "by_movement_type": db_stats.get("by_movement_type", {})},
            "files": {"pending": file_stats.get("pending", 0), "rejected": file_stats.get("rejected", 0),
                      "scail_ready_total": file_stats.get("scail_ready_total", 0),
                      "by_type": file_stats.get("scail_ready_by_type", {})}})
    except Exception as e:
        return SkillResult(success=False, message=str(e))


async def close(skill, params: Dict[str, Any]) -> SkillResult:
    if skill._scraper:
        try:
            await skill._scraper.close()
        except Exception:
            pass
        skill._scraper = None
    return SkillResult(success=True, data={"closed": True})
