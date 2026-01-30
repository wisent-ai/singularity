#!/usr/bin/env python3
"""
Perception System for Autonomous Agent

Scans internal state and external world to find:
- Opportunities (ways to make money)
- Threats (competition, market changes)
- Resources (potential customers, partners)
"""

import asyncio
import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import httpx

try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class Opportunity:
    """A potential way to make money"""
    source: str  # Where it came from
    type: str  # 'gig', 'task', 'sale', 'partnership', 'arbitrage'
    title: str
    description: str
    estimated_revenue: float
    estimated_cost: float
    estimated_hours: float
    success_probability: float
    deadline: Optional[datetime] = None
    url: Optional[str] = None
    contact: Optional[str] = None
    metadata: Dict = field(default_factory=dict)

    @property
    def expected_value(self) -> float:
        return (self.estimated_revenue * self.success_probability) - self.estimated_cost

    @property
    def hourly_rate(self) -> float:
        if self.estimated_hours <= 0:
            return 0
        return self.expected_value / self.estimated_hours


@dataclass
class Threat:
    """Something that could harm the agent"""
    source: str
    type: str  # 'competitor', 'market', 'regulation', 'technical'
    description: str
    severity: float  # 0-1
    metadata: Dict = field(default_factory=dict)


@dataclass
class Resource:
    """Something that could help the agent"""
    source: str
    type: str  # 'customer', 'partner', 'tool', 'knowledge'
    name: str
    description: str
    potential_value: float
    contact: Optional[str] = None
    metadata: Dict = field(default_factory=dict)


@dataclass
class PerceptionResult:
    """Combined perception output"""
    opportunities: List[Opportunity] = field(default_factory=list)
    threats: List[Threat] = field(default_factory=list)
    resources: List[Resource] = field(default_factory=list)
    market_data: Dict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


# ============================================================================
# PERCEPTION ENGINE
# ============================================================================

class PerceptionEngine:
    """
    Scans the world for opportunities, threats, and resources.

    Data sources:
    - Internal: Coordinator API (balance, status, competitors)
    - External: Web scraping, API integrations
    """

    def __init__(
        self,
        coordinator_url: str = "https://singularity.wisent.ai",
        agent_ticker: str = "",
        agent_instance_id: str = ""
    ):
        self.coordinator_url = coordinator_url
        self.agent_ticker = agent_ticker
        self.agent_instance_id = agent_instance_id
        self.http = httpx.AsyncClient(timeout=30)

        # Cache to avoid hitting same sources repeatedly
        self._cache: Dict[str, Any] = {}
        self._cache_ttl: Dict[str, datetime] = {}

    async def perceive(self) -> PerceptionResult:
        """Run full perception cycle"""
        result = PerceptionResult()

        # Run all perception tasks concurrently
        tasks = [
            self._perceive_internal(),
            self._perceive_task_marketplace(),
            self._perceive_market(),
            self._perceive_opportunities(),
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for r in results:
            if isinstance(r, PerceptionResult):
                result.opportunities.extend(r.opportunities)
                result.threats.extend(r.threats)
                result.resources.extend(r.resources)
                result.market_data.update(r.market_data)

        # Sort opportunities by expected value
        result.opportunities.sort(key=lambda x: x.expected_value, reverse=True)

        return result

    # ========================================================================
    # INTERNAL PERCEPTION
    # ========================================================================

    async def _perceive_internal(self) -> PerceptionResult:
        """Get state from our own platform"""
        result = PerceptionResult()

        try:
            # Get competitors
            r = await self.http.get(f"{self.coordinator_url}/api/agents")
            if r.status_code == 200:
                agents = r.json().get("agents", [])
                for agent in agents:
                    if agent.get("ticker") != self.agent_ticker:
                        # Other agents are competitors
                        result.threats.append(Threat(
                            source="internal",
                            type="competitor",
                            description=f"Agent {agent.get('name')} (${agent.get('ticker')})",
                            severity=0.3,
                            metadata=agent
                        ))

            # Get market stats
            r = await self.http.get(f"{self.coordinator_url}/api/stats")
            if r.status_code == 200:
                result.market_data["platform"] = r.json()

        except Exception as e:
            print(f"[PERCEPTION] Internal error: {e}")

        return result

    # ========================================================================
    # TASK MARKETPLACE PERCEPTION
    # ========================================================================

    async def _perceive_task_marketplace(self) -> PerceptionResult:
        """Find tasks in our internal marketplace"""
        result = PerceptionResult()

        # Check internal task marketplace
        try:
            r = await self.http.get(f"{self.coordinator_url}/api/tasks?status=open")
            if r.status_code == 200:
                tasks = r.json().get("tasks", [])
                for task in tasks:
                    result.opportunities.append(Opportunity(
                        source="wisent_marketplace",
                        type="task",
                        title=task.get("title", "Untitled"),
                        description=task.get("description", ""),
                        estimated_revenue=float(task.get("bounty_wisent", 0)),
                        estimated_cost=0.5,  # Small cost to attempt
                        estimated_hours=float(task.get("estimated_hours", 1)),
                        success_probability=0.7,
                        deadline=datetime.fromisoformat(task["deadline"]) if task.get("deadline") else None,
                        url=f"{self.coordinator_url}/tasks/{task.get('id')}",
                        metadata=task
                    ))
        except Exception as e:
            print(f"[PERCEPTION] Task marketplace error: {e}")

        return result

    # ========================================================================
    # EXTERNAL OPPORTUNITY SCANNING
    # ========================================================================

    async def _perceive_opportunities(self) -> PerceptionResult:
        """Scan external sources for opportunities"""
        result = PerceptionResult()

        # Run external scans concurrently
        scans = [
            self._scan_github_bounties(),
            self._scan_generic_opportunities(),
        ]

        scan_results = await asyncio.gather(*scans, return_exceptions=True)

        for r in scan_results:
            if isinstance(r, list):
                result.opportunities.extend(r)

        return result

    async def _scan_github_bounties(self) -> List[Opportunity]:
        """Find GitHub issues with bounties"""
        opportunities = []

        # Check cache
        cache_key = "github_bounties"
        if self._is_cached(cache_key):
            return self._get_cache(cache_key)

        try:
            # Search for issues labeled with bounty
            # Using GitHub's search API (no auth required for public)
            queries = [
                "label:bounty state:open",
                "label:paid state:open",
                "label:bug-bounty state:open",
            ]

            for query in queries:
                r = await self.http.get(
                    "https://api.github.com/search/issues",
                    params={"q": query, "sort": "created", "per_page": 10},
                    headers={"Accept": "application/vnd.github.v3+json"}
                )

                if r.status_code == 200:
                    items = r.json().get("items", [])
                    for item in items:
                        # Try to extract bounty amount from title/body
                        bounty = self._extract_bounty_amount(
                            item.get("title", "") + " " + (item.get("body") or "")
                        )

                        if bounty > 0:
                            opportunities.append(Opportunity(
                                source="github",
                                type="bounty",
                                title=item.get("title", "")[:100],
                                description=f"GitHub Issue #{item.get('number')} in {item.get('repository_url', '').split('/')[-1]}",
                                estimated_revenue=bounty,
                                estimated_cost=1.0,  # Time/compute cost
                                estimated_hours=4,
                                success_probability=0.3,
                                url=item.get("html_url"),
                                metadata={
                                    "labels": [l.get("name") for l in item.get("labels", [])],
                                    "repo": item.get("repository_url")
                                }
                            ))

                await asyncio.sleep(0.5)  # Rate limiting

        except Exception as e:
            print(f"[PERCEPTION] GitHub scan error: {e}")

        self._set_cache(cache_key, opportunities, ttl_minutes=30)
        return opportunities

    async def _scan_generic_opportunities(self) -> List[Opportunity]:
        """Find generic work opportunities"""
        opportunities = []

        # Add known opportunity sources that don't require authentication

        # 1. Check for AI-related tasks on public APIs
        known_opportunities = [
            Opportunity(
                source="internal",
                type="service",
                title="Offer API services",
                description="Provide AI text generation API to potential customers",
                estimated_revenue=5.0,
                estimated_cost=1.0,
                estimated_hours=1,
                success_probability=0.4,
                metadata={"approach": "cold_outreach"}
            ),
            Opportunity(
                source="internal",
                type="content",
                title="Create and sell content",
                description="Generate valuable content (articles, code snippets) for marketplace",
                estimated_revenue=2.0,
                estimated_cost=0.5,
                estimated_hours=0.5,
                success_probability=0.3,
                metadata={"approach": "marketplace_listing"}
            ),
            Opportunity(
                source="internal",
                type="arbitrage",
                title="Information arbitrage",
                description="Curate valuable information and sell summaries",
                estimated_revenue=1.0,
                estimated_cost=0.2,
                estimated_hours=0.3,
                success_probability=0.4,
                metadata={"approach": "automation"}
            ),
        ]

        opportunities.extend(known_opportunities)
        return opportunities

    # ========================================================================
    # MARKET PERCEPTION
    # ========================================================================

    async def _perceive_market(self) -> PerceptionResult:
        """Get market data"""
        result = PerceptionResult()

        # Check cache
        if self._is_cached("market_data"):
            result.market_data = self._get_cache("market_data")
            return result

        try:
            # Get WISENT rates
            r = await self.http.get(f"{self.coordinator_url}/api/wisent/rates")
            if r.status_code == 200:
                result.market_data["wisent"] = r.json()

            # Get token prices
            r = await self.http.get(f"{self.coordinator_url}/api/tokens")
            if r.status_code == 200:
                result.market_data["tokens"] = r.json().get("tokens", [])

            self._set_cache("market_data", result.market_data, ttl_minutes=5)

        except Exception as e:
            print(f"[PERCEPTION] Market error: {e}")

        return result

    # ========================================================================
    # HELPERS
    # ========================================================================

    def _extract_bounty_amount(self, text: str) -> float:
        """Extract bounty amount from text"""
        # Look for patterns like "$100", "100 USD", "$100 bounty"
        patterns = [
            r'\$(\d+(?:,\d{3})*(?:\.\d{2})?)',  # $100 or $1,000.00
            r'(\d+(?:,\d{3})*)\s*(?:USD|usd|dollars?)',  # 100 USD
            r'bounty[:\s]*\$?(\d+(?:,\d{3})*)',  # bounty: 100
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                amount_str = match.group(1).replace(",", "")
                try:
                    return float(amount_str)
                except:
                    pass
        return 0

    def _is_cached(self, key: str) -> bool:
        """Check if cache is valid"""
        if key not in self._cache_ttl:
            return False
        return datetime.now() < self._cache_ttl[key]

    def _get_cache(self, key: str) -> Any:
        """Get cached value"""
        return self._cache.get(key)

    def _set_cache(self, key: str, value: Any, ttl_minutes: int = 5):
        """Set cache value"""
        self._cache[key] = value
        self._cache_ttl[key] = datetime.now() + timedelta(minutes=ttl_minutes)

    async def close(self):
        """Clean up"""
        await self.http.aclose()


# ============================================================================
# SPECIALIZED SCANNERS
# ============================================================================

class FreelanceScanner:
    """Scan freelance platforms for opportunities"""

    def __init__(self):
        self.http = httpx.AsyncClient(timeout=30)

    async def scan_all(self) -> List[Opportunity]:
        """Scan all configured platforms"""
        opportunities = []

        # Note: Most freelance platforms require authentication
        # These are placeholder implementations

        return opportunities

    async def close(self):
        await self.http.aclose()


class SocialMediaScanner:
    """Monitor social media for opportunities"""

    def __init__(self):
        self.http = httpx.AsyncClient(timeout=30)

    async def scan_twitter(self, keywords: List[str]) -> List[Opportunity]:
        """Scan Twitter/X for mentions"""
        # Requires API key - placeholder
        return []

    async def scan_reddit(self, subreddits: List[str]) -> List[Opportunity]:
        """Scan Reddit for opportunities"""
        opportunities = []

        try:
            for sub in subreddits:
                # Use old.reddit.com JSON API (no auth required)
                r = await self.http.get(
                    f"https://old.reddit.com/r/{sub}/new.json",
                    params={"limit": 25},
                    headers={"User-Agent": "AgentBot/1.0"}
                )

                if r.status_code == 200:
                    posts = r.json().get("data", {}).get("children", [])
                    for post in posts:
                        data = post.get("data", {})
                        title = data.get("title", "")

                        # Look for hiring/gig posts
                        hiring_keywords = ["hiring", "looking for", "need help", "freelance", "contractor", "paid"]
                        if any(kw in title.lower() for kw in hiring_keywords):
                            opportunities.append(Opportunity(
                                source=f"reddit/{sub}",
                                type="gig",
                                title=title[:100],
                                description=data.get("selftext", "")[:500],
                                estimated_revenue=50.0,  # Unknown, estimate
                                estimated_cost=2.0,
                                estimated_hours=4,
                                success_probability=0.2,
                                url=f"https://reddit.com{data.get('permalink', '')}",
                                metadata={"subreddit": sub, "author": data.get("author")}
                            ))

                await asyncio.sleep(1)  # Rate limiting

        except Exception as e:
            print(f"[PERCEPTION] Reddit scan error: {e}")

        return opportunities

    async def close(self):
        await self.http.aclose()


# ============================================================================
# TEST
# ============================================================================

async def test_perception():
    """Test the perception system"""
    engine = PerceptionEngine(
        coordinator_url="https://singularity.wisent.ai",
        agent_ticker="TEST",
        agent_instance_id="test_001"
    )

    print("Running perception cycle...")
    result = await engine.perceive()

    print(f"\n=== PERCEPTION RESULTS ===")
    print(f"Opportunities found: {len(result.opportunities)}")
    for opp in result.opportunities[:5]:
        print(f"  - {opp.source}: {opp.title} (EV: {opp.expected_value:.2f})")

    print(f"\nThreats found: {len(result.threats)}")
    for threat in result.threats[:3]:
        print(f"  - {threat.type}: {threat.description}")

    print(f"\nMarket data keys: {list(result.market_data.keys())}")

    await engine.close()


if __name__ == "__main__":
    asyncio.run(test_perception())
