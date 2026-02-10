"""TikTok Skill - Autonomous TikTok research, scraping, and content analysis."""

import asyncio
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from skills.base import Skill, SkillAction, SkillManifest, SkillResult
from . import handlers

SCRAPING_PATH = Path(__file__).parent.parent.parent.parent.parent.parent / "scraping" / "tiktok"
TIKTOK_PATH = Path(__file__).parent.parent.parent.parent.parent.parent / "tiktok"
if str(SCRAPING_PATH.parent) not in sys.path:
    sys.path.insert(0, str(SCRAPING_PATH.parent))
if str(TIKTOK_PATH.parent) not in sys.path:
    sys.path.insert(0, str(TIKTOK_PATH.parent))


def _a(n, d, p=None):
    return SkillAction(name=n, description=d, parameters=p or {})


class TikTokSkill(Skill):
    """TikTok research and content analysis skill."""

    def __init__(self):
        self._scraper = None
        self._downloader = None
        self._filter = None
        self._trend_analyzer = None

    @property
    def manifest(self) -> SkillManifest:
        _sc = {"scroll_count": {"type": "integer", "default": 10}}
        _hl = {"headless": {"type": "boolean", "default": False}}
        return SkillManifest(
            skill_id="tiktok", name="TikTok Research", version="1.0.0", category="social",
            description="TikTok content research, scraping, trend analysis, and SCAIL video preparation",
            actions=[
                _a("scrape_fyp", "Scrape videos from TikTok For You page", {**_sc, **_hl}),
                _a("scrape_hashtag", "Scrape videos from a specific hashtag",
                   {"hashtag": {"type": "string", "required": True}, **_sc, **_hl}),
                _a("scrape_search", "Scrape videos from search results",
                   {"query": {"type": "string", "required": True}, **_sc, **_hl}),
                _a("scrape_user", "Scrape videos from a user's profile using yt-dlp",
                   {"username": {"type": "string", "required": True}, "limit": {"type": "integer", "default": 50}}),
                _a("download_videos", "Download pending videos from the database",
                   {"limit": {"type": "integer", "default": 50}, "scail_only": {"type": "boolean", "default": False}}),
                _a("analyze_scail", "Analyze videos for SCAIL suitability", {"limit": {"type": "integer", "default": 50}}),
                _a("get_scail_videos", "Get list of videos suitable for SCAIL processing",
                   {"movement_type": {"type": "string", "enum": ["dance", "transition", "gesture", "walk", "other"], "required": False},
                    "limit": {"type": "integer", "default": 20}}),
                _a("get_trending_hashtags", "Get trending hashtag suggestions",
                   {"movement_type": {"type": "string", "enum": ["dance", "transition", "gesture", "walk", "other"], "required": False}}),
                _a("get_viral_sounds", "Get viral sound suggestions", {"min_videos": {"type": "integer", "default": 3}}),
                _a("generate_trend_report", "Generate a comprehensive trend analysis report"),
                _a("analyze_hooks", "Analyze viral hook patterns in SCAIL-ready videos"),
                _a("get_stats", "Get current database and file statistics"),
                _a("close", "Close the browser session"),
            ], required_credentials=[], install_cost=0)

    async def _get_scraper(self, headless: bool = False):
        try:
            from scraping.tiktok.scraper import TikTokResearchScraper
            if self._scraper is None:
                self._scraper = TikTokResearchScraper(headless=headless)
                await self._scraper.start()
            return self._scraper
        except ImportError as e:
            raise ImportError(f"TikTok scraper not available: {e}")

    def _get_downloader(self):
        if self._downloader is None:
            from scraping.tiktok.downloader import TikTokDownloader
            self._downloader = TikTokDownloader()
        return self._downloader

    def _get_filter(self):
        if self._filter is None:
            from scraping.tiktok.filters import SCAILFilter
            self._filter = SCAILFilter()
        return self._filter

    def _get_trend_analyzer(self):
        if self._trend_analyzer is None:
            from tiktok.research.trend_analyzer import TrendAnalyzer
            self._trend_analyzer = TrendAnalyzer()
        return self._trend_analyzer

    async def execute(self, action: str, parameters: Dict[str, Any]) -> SkillResult:
        try:
            dispatch = {
                "scrape_fyp": lambda: handlers.scrape_fyp(self, parameters),
                "scrape_hashtag": lambda: handlers.scrape_hashtag(self, parameters),
                "scrape_search": lambda: handlers.scrape_search(self, parameters),
                "scrape_user": lambda: handlers.scrape_user(self, parameters),
                "download_videos": lambda: handlers.download_videos(self, parameters),
                "analyze_scail": lambda: handlers.analyze_scail(self, parameters),
                "get_scail_videos": lambda: handlers.get_scail_videos(self, parameters),
                "get_trending_hashtags": lambda: handlers.get_trending_hashtags(self, parameters),
                "get_viral_sounds": lambda: handlers.get_viral_sounds(self, parameters),
                "generate_trend_report": lambda: handlers.generate_trend_report(self, parameters),
                "analyze_hooks": lambda: handlers.analyze_hooks(self, parameters),
                "get_stats": lambda: handlers.get_stats(self, parameters),
                "close": lambda: handlers.close(self, parameters),
            }
            handler = dispatch.get(action)
            if not handler:
                return SkillResult(success=False, error=f"Unknown action: {action}")
            return await handler()
        except Exception as e:
            return SkillResult(success=False, error=str(e))
