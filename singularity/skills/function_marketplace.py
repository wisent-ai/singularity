#!/usr/bin/env python3
"""
FunctionMarketplaceSkill - Cross-agent serverless function exchange.

Enables agents to publish, discover, and import serverless functions from each
other. While SkillMarketplaceHub handles full skill packages, this marketplace
focuses specifically on lightweight serverless functions (deployed via
ServerlessFunctionSkill) that agents can share and monetize.

Use cases:
1. Agent Eve builds a great text-summarization function → publishes it
2. Agent Adam discovers it → imports it into his own ServerlessFunctionSkill
3. Eve earns revenue from every import/invocation
4. The ecosystem develops specialized, reusable function libraries

Features:
1. **Publish** - Package a deployed serverless function for sharing
2. **Browse** - Discover functions by category, tag, rating, or search
3. **Import** - Download and deploy a function from the marketplace
4. **Rate** - Review functions to build trust and quality signals
5. **Featured** - Surface top-rated, most-imported functions
6. **Revenue** - Track earnings per published function
7. **Categories** - Organize functions by purpose (data, text, api, utility, etc.)

Event topics emitted:
  - function_marketplace.published   - New function published
  - function_marketplace.imported    - Function imported by another agent
  - function_marketplace.rated       - Function received a rating
  - function_marketplace.featured    - Function marked as featured

Data integration:
  - Reads from ServerlessFunctionSkill data to list publishable functions
  - Writes imported functions back to ServerlessFunctionSkill format
  - Revenue data feeds into RevenueAnalyticsDashboard

Pillars:
  - Replication: Agents share capabilities through function exchange
  - Revenue: Function authors earn from imports and invocations
  - Self-Improvement: Agents acquire new capabilities without building from scratch
"""

import json
import hashlib
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any

from .base import Skill, SkillResult, SkillManifest, SkillAction

DATA_DIR = Path(__file__).parent.parent / "data"
MARKETPLACE_FILE = DATA_DIR / "function_marketplace.json"
SERVERLESS_FILE = DATA_DIR / "serverless_functions.json"

# Function categories
CATEGORIES = [
    "data_transform",    # Data parsing, conversion, ETL
    "text_processing",   # Summarization, extraction, formatting
    "api_integration",   # Third-party API wrappers
    "utility",           # General-purpose helpers
    "analytics",         # Data analysis, metrics
    "security",          # Auth, encryption, validation
    "ai_ml",             # ML model wrappers, prompt utilities
    "revenue",           # Billing, payment, pricing utilities
]

MAX_LISTINGS = 500
MAX_REVIEWS_PER_FUNCTION = 50
MAX_IMPORT_LOG = 1000


def _generate_id() -> str:
    return uuid.uuid4().hex[:12]


def _hash_code(code: str) -> str:
    return hashlib.sha256(code.encode()).hexdigest()[:16]


class FunctionMarketplaceSkill(Skill):
    """Cross-agent marketplace for serverless function exchange."""

    def __init__(self, credentials: Dict[str, str] = None):
        super().__init__(credentials)
        self._ensure_data()

    def _ensure_data(self):
        MARKETPLACE_FILE.parent.mkdir(parents=True, exist_ok=True)
        if not MARKETPLACE_FILE.exists():
            self._save(self._default_state())

    def _default_state(self) -> Dict:
        return {
            "listings": {},          # listing_id -> function listing
            "reviews": {},           # listing_id -> [reviews]
            "imports": [],           # Import history
            "earnings": {},          # publisher_agent -> total_earned
            "featured": [],          # Featured listing IDs
            "stats": {
                "total_published": 0,
                "total_imported": 0,
                "total_reviews": 0,
                "total_revenue": 0.0,
            },
            "created_at": datetime.utcnow().isoformat(),
        }

    def _load(self) -> Dict:
        try:
            return json.loads(MARKETPLACE_FILE.read_text())
        except (json.JSONDecodeError, FileNotFoundError, OSError):
            return self._default_state()

    def _save(self, data: Dict) -> None:
        data["last_updated"] = datetime.utcnow().isoformat()
        # Trim import log
        if len(data.get("imports", [])) > MAX_IMPORT_LOG:
            data["imports"] = data["imports"][-MAX_IMPORT_LOG:]
        MARKETPLACE_FILE.write_text(json.dumps(data, indent=2, default=str))

    def _load_serverless_data(self) -> Dict:
        """Load ServerlessFunctionSkill data for publishing/importing."""
        if SERVERLESS_FILE.exists():
            try:
                return json.loads(SERVERLESS_FILE.read_text())
            except (json.JSONDecodeError, OSError):
                pass
        return {"functions": {}, "stats": {}}

    def _save_serverless_data(self, data: Dict) -> None:
        """Write back to ServerlessFunctionSkill data store."""
        SERVERLESS_FILE.parent.mkdir(parents=True, exist_ok=True)
        SERVERLESS_FILE.write_text(json.dumps(data, indent=2, default=str))

    # ── Manifest ─────────────────────────────────────────────────────

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="function_marketplace",
            name="Function Marketplace",
            version="1.0.0",
            category="replication",
            description=(
                "Cross-agent marketplace for serverless functions. "
                "Publish, discover, import, and rate reusable functions."
            ),
            actions=[
                SkillAction(
                    name="publish",
                    description=(
                        "Publish a serverless function to the marketplace for other agents to discover and import."
                    ),
                    parameters={
                        "function_name": {"type": "str", "required": True,
                                          "description": "Name of the function to publish (must exist in ServerlessFunctionSkill)"},
                        "agent_name": {"type": "str", "required": True,
                                       "description": "Publisher agent name"},
                        "category": {"type": "str", "required": False,
                                     "description": f"Category: {CATEGORIES}"},
                        "tags": {"type": "list", "required": False,
                                 "description": "Searchable tags for discovery"},
                        "price_per_import": {"type": "float", "required": False,
                                             "description": "Price per import (default 0.0 = free)"},
                        "description": {"type": "str", "required": False,
                                        "description": "Human-readable description"},
                        "code": {"type": "str", "required": False,
                                 "description": "Function source code (if not reading from ServerlessFunctionSkill)"},
                        "route": {"type": "str", "required": False,
                                  "description": "HTTP route the function handles"},
                        "method": {"type": "str", "required": False,
                                   "description": "HTTP method (GET, POST, etc.)"},
                    },
                    estimated_cost=0.0,
                ),
                SkillAction(
                    name="browse",
                    description="Browse marketplace listings with optional filters",
                    parameters={
                        "category": {"type": "str", "required": False,
                                     "description": "Filter by category"},
                        "search": {"type": "str", "required": False,
                                   "description": "Search query (matches name, description, tags)"},
                        "agent_name": {"type": "str", "required": False,
                                       "description": "Filter by publisher agent"},
                        "sort_by": {"type": "str", "required": False,
                                    "description": "Sort: rating, imports, recent, price"},
                        "limit": {"type": "int", "required": False,
                                  "description": "Max results (default 20)"},
                    },
                    estimated_cost=0.0,
                ),
                SkillAction(
                    name="import_function",
                    description=(
                        "Import a function from the marketplace into your local ServerlessFunctionSkill. "
                        "Copies the code and creates a local deployment."
                    ),
                    parameters={
                        "listing_id": {"type": "str", "required": True,
                                       "description": "Marketplace listing ID to import"},
                        "importer_agent": {"type": "str", "required": True,
                                           "description": "Agent importing the function"},
                        "local_name": {"type": "str", "required": False,
                                       "description": "Local function name (defaults to original name)"},
                    },
                    estimated_cost=0.0,
                ),
                SkillAction(
                    name="rate",
                    description="Rate and review a marketplace function",
                    parameters={
                        "listing_id": {"type": "str", "required": True,
                                       "description": "Listing to rate"},
                        "agent_name": {"type": "str", "required": True,
                                       "description": "Agent leaving the review"},
                        "rating": {"type": "int", "required": True,
                                   "description": "Rating 1-5 stars"},
                        "review": {"type": "str", "required": False,
                                   "description": "Optional review text"},
                    },
                    estimated_cost=0.0,
                ),
                SkillAction(
                    name="get_listing",
                    description="Get detailed info about a specific marketplace listing",
                    parameters={
                        "listing_id": {"type": "str", "required": True,
                                       "description": "Listing ID to inspect"},
                    },
                    estimated_cost=0.0,
                ),
                SkillAction(
                    name="featured",
                    description="Get featured/top functions from the marketplace",
                    parameters={
                        "limit": {"type": "int", "required": False,
                                  "description": "Number of featured functions (default 10)"},
                    },
                    estimated_cost=0.0,
                ),
                SkillAction(
                    name="my_publications",
                    description="List functions published by a specific agent with earnings",
                    parameters={
                        "agent_name": {"type": "str", "required": True,
                                       "description": "Agent name to check"},
                    },
                    estimated_cost=0.0,
                ),
                SkillAction(
                    name="unpublish",
                    description="Remove a function from the marketplace",
                    parameters={
                        "listing_id": {"type": "str", "required": True,
                                       "description": "Listing to remove"},
                        "agent_name": {"type": "str", "required": True,
                                       "description": "Must match publisher"},
                    },
                    estimated_cost=0.0,
                ),
                SkillAction(
                    name="status",
                    description="Get marketplace overview stats",
                    parameters={},
                    estimated_cost=0.0,
                ),
            ],
            required_credentials=[],
            author="singularity",
        )

    # ── Execute ──────────────────────────────────────────────────────

    async def execute(self, action: str, params: Dict) -> SkillResult:
        handlers = {
            "publish": self._publish,
            "browse": self._browse,
            "import_function": self._import_function,
            "rate": self._rate,
            "get_listing": self._get_listing,
            "featured": self._featured,
            "my_publications": self._my_publications,
            "unpublish": self._unpublish,
            "status": self._status,
        }
        handler = handlers.get(action)
        if not handler:
            return SkillResult(
                success=False,
                message=f"Unknown action: {action}. Available: {list(handlers.keys())}",
            )
        try:
            return await handler(params)
        except Exception as e:
            return SkillResult(success=False, message=f"Error in {action}: {e}")

    # ── Actions ──────────────────────────────────────────────────────

    async def _publish(self, params: Dict) -> SkillResult:
        """Publish a function to the marketplace."""
        function_name = params.get("function_name")
        agent_name = params.get("agent_name")
        if not function_name or not agent_name:
            return SkillResult(success=False, message="function_name and agent_name are required")

        category = params.get("category", "utility")
        if category not in CATEGORIES:
            category = "utility"

        tags = params.get("tags", [])
        price = float(params.get("price_per_import", 0.0))
        description = params.get("description", "")
        code = params.get("code")
        route = params.get("route")
        method = params.get("method", "POST")

        # Try to load from ServerlessFunctionSkill if code not provided
        if not code:
            sf_data = self._load_serverless_data()
            func = sf_data.get("functions", {}).get(function_name)
            if func:
                code = func.get("code", "")
                route = route or func.get("route", f"/{function_name}")
                method = func.get("method", method)
                description = description or func.get("description", "")
            else:
                return SkillResult(
                    success=False,
                    message=f"Function '{function_name}' not found in ServerlessFunctionSkill and no code provided",
                )

        data = self._load()
        listings = data.get("listings", {})

        # Check for duplicate from same agent
        for lid, listing in listings.items():
            if listing["function_name"] == function_name and listing["agent_name"] == agent_name:
                return SkillResult(
                    success=False,
                    message=f"Function '{function_name}' already published by {agent_name} (listing: {lid})",
                )

        if len(listings) >= MAX_LISTINGS:
            return SkillResult(success=False, message="Marketplace is at capacity")

        listing_id = f"fn_{_generate_id()}"
        code_hash = _hash_code(code)

        listing = {
            "listing_id": listing_id,
            "function_name": function_name,
            "agent_name": agent_name,
            "category": category,
            "tags": tags,
            "description": description,
            "code": code,
            "code_hash": code_hash,
            "route": route or f"/{function_name}",
            "method": method,
            "price_per_import": price,
            "import_count": 0,
            "avg_rating": 0.0,
            "rating_count": 0,
            "total_revenue": 0.0,
            "published_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
            "status": "active",
        }

        data["listings"][listing_id] = listing
        data["stats"]["total_published"] += 1
        self._save(data)

        return SkillResult(
            success=True,
            message=f"Published '{function_name}' to marketplace as {listing_id}",
            data={
                "listing_id": listing_id,
                "function_name": function_name,
                "category": category,
                "price_per_import": price,
            },
        )

    async def _browse(self, params: Dict) -> SkillResult:
        """Browse marketplace with optional filters."""
        data = self._load()
        listings = list(data.get("listings", {}).values())

        # Filter active only
        listings = [l for l in listings if l.get("status") == "active"]

        # Category filter
        category = params.get("category")
        if category:
            listings = [l for l in listings if l.get("category") == category]

        # Agent filter
        agent_name = params.get("agent_name")
        if agent_name:
            listings = [l for l in listings if l.get("agent_name") == agent_name]

        # Search filter
        search = params.get("search", "").lower()
        if search:
            scored = []
            for l in listings:
                score = 0
                name = l.get("function_name", "").lower()
                desc = l.get("description", "").lower()
                tags_str = " ".join(l.get("tags", [])).lower()

                if search in name:
                    score += 10
                if search in desc:
                    score += 5
                if search in tags_str:
                    score += 3

                if score > 0:
                    scored.append((score, l))

            scored.sort(key=lambda x: -x[0])
            listings = [item[1] for item in scored]

        # Sort
        sort_by = params.get("sort_by", "recent")
        if sort_by == "rating":
            listings.sort(key=lambda l: -l.get("avg_rating", 0))
        elif sort_by == "imports":
            listings.sort(key=lambda l: -l.get("import_count", 0))
        elif sort_by == "price":
            listings.sort(key=lambda l: l.get("price_per_import", 0))
        else:  # recent
            listings.sort(key=lambda l: l.get("published_at", ""), reverse=True)

        limit = params.get("limit", 20)
        listings = listings[:limit]

        # Return summaries (exclude code for browsing)
        summaries = []
        for l in listings:
            summaries.append({
                "listing_id": l["listing_id"],
                "function_name": l["function_name"],
                "agent_name": l["agent_name"],
                "category": l["category"],
                "description": l.get("description", "")[:200],
                "tags": l.get("tags", []),
                "price_per_import": l.get("price_per_import", 0),
                "import_count": l.get("import_count", 0),
                "avg_rating": l.get("avg_rating", 0),
                "rating_count": l.get("rating_count", 0),
                "published_at": l.get("published_at"),
            })

        return SkillResult(
            success=True,
            message=f"Found {len(summaries)} functions in marketplace",
            data={
                "count": len(summaries),
                "listings": summaries,
                "filters": {
                    "category": category,
                    "search": search or None,
                    "agent_name": agent_name,
                    "sort_by": sort_by,
                },
            },
        )

    async def _import_function(self, params: Dict) -> SkillResult:
        """Import a function from marketplace into local ServerlessFunctionSkill."""
        listing_id = params.get("listing_id")
        importer_agent = params.get("importer_agent")
        if not listing_id or not importer_agent:
            return SkillResult(success=False, message="listing_id and importer_agent are required")

        data = self._load()
        listing = data.get("listings", {}).get(listing_id)
        if not listing:
            return SkillResult(success=False, message=f"Listing '{listing_id}' not found")

        if listing.get("status") != "active":
            return SkillResult(success=False, message=f"Listing '{listing_id}' is not active")

        # Don't import your own function
        if listing["agent_name"] == importer_agent:
            return SkillResult(
                success=False,
                message="Cannot import your own function",
            )

        local_name = params.get("local_name", listing["function_name"])

        # Write to ServerlessFunctionSkill data
        sf_data = self._load_serverless_data()
        functions = sf_data.setdefault("functions", {})

        if local_name in functions:
            return SkillResult(
                success=False,
                message=f"Function '{local_name}' already exists locally. Use a different local_name.",
            )

        functions[local_name] = {
            "name": local_name,
            "code": listing["code"],
            "route": listing.get("route", f"/{local_name}"),
            "method": listing.get("method", "POST"),
            "description": listing.get("description", ""),
            "enabled": True,
            "imported_from": {
                "listing_id": listing_id,
                "agent_name": listing["agent_name"],
                "original_name": listing["function_name"],
            },
            "deployed_at": datetime.utcnow().isoformat(),
            "stats": {"invocations": 0, "errors": 0, "revenue": 0.0},
        }
        self._save_serverless_data(sf_data)

        # Update marketplace stats
        listing["import_count"] = listing.get("import_count", 0) + 1
        price = listing.get("price_per_import", 0)
        if price > 0:
            listing["total_revenue"] = listing.get("total_revenue", 0) + price
            earnings = data.setdefault("earnings", {})
            earnings[listing["agent_name"]] = earnings.get(listing["agent_name"], 0) + price
            data["stats"]["total_revenue"] = data["stats"].get("total_revenue", 0) + price

        # Log the import
        data.setdefault("imports", []).append({
            "listing_id": listing_id,
            "function_name": listing["function_name"],
            "importer_agent": importer_agent,
            "publisher_agent": listing["agent_name"],
            "local_name": local_name,
            "price": price,
            "timestamp": datetime.utcnow().isoformat(),
        })
        data["stats"]["total_imported"] += 1

        self._save(data)

        return SkillResult(
            success=True,
            message=f"Imported '{listing['function_name']}' as '{local_name}' from {listing['agent_name']}",
            data={
                "listing_id": listing_id,
                "local_name": local_name,
                "publisher": listing["agent_name"],
                "price_paid": price,
                "route": listing.get("route"),
            },
        )

    async def _rate(self, params: Dict) -> SkillResult:
        """Rate a marketplace function."""
        listing_id = params.get("listing_id")
        agent_name = params.get("agent_name")
        rating = params.get("rating")

        if not listing_id or not agent_name or rating is None:
            return SkillResult(success=False, message="listing_id, agent_name, and rating are required")

        rating = int(rating)
        if rating < 1 or rating > 5:
            return SkillResult(success=False, message="Rating must be 1-5")

        data = self._load()
        listing = data.get("listings", {}).get(listing_id)
        if not listing:
            return SkillResult(success=False, message=f"Listing '{listing_id}' not found")

        # Can't rate your own
        if listing["agent_name"] == agent_name:
            return SkillResult(success=False, message="Cannot rate your own function")

        reviews = data.setdefault("reviews", {})
        listing_reviews = reviews.setdefault(listing_id, [])

        # Check for existing review by this agent
        for r in listing_reviews:
            if r["agent_name"] == agent_name:
                # Update existing review
                r["rating"] = rating
                r["review"] = params.get("review", r.get("review", ""))
                r["updated_at"] = datetime.utcnow().isoformat()
                break
        else:
            if len(listing_reviews) >= MAX_REVIEWS_PER_FUNCTION:
                listing_reviews.pop(0)  # Remove oldest

            listing_reviews.append({
                "agent_name": agent_name,
                "rating": rating,
                "review": params.get("review", ""),
                "created_at": datetime.utcnow().isoformat(),
            })
            data["stats"]["total_reviews"] += 1

        # Recalculate average rating
        if listing_reviews:
            avg = sum(r["rating"] for r in listing_reviews) / len(listing_reviews)
            listing["avg_rating"] = round(avg, 2)
            listing["rating_count"] = len(listing_reviews)

        self._save(data)

        return SkillResult(
            success=True,
            message=f"Rated '{listing['function_name']}' {rating}/5 stars",
            data={
                "listing_id": listing_id,
                "rating": rating,
                "avg_rating": listing["avg_rating"],
                "total_reviews": listing["rating_count"],
            },
        )

    async def _get_listing(self, params: Dict) -> SkillResult:
        """Get detailed info about a listing."""
        listing_id = params.get("listing_id")
        if not listing_id:
            return SkillResult(success=False, message="listing_id is required")

        data = self._load()
        listing = data.get("listings", {}).get(listing_id)
        if not listing:
            return SkillResult(success=False, message=f"Listing '{listing_id}' not found")

        reviews = data.get("reviews", {}).get(listing_id, [])

        return SkillResult(
            success=True,
            message=f"Listing: {listing['function_name']} by {listing['agent_name']}",
            data={
                **listing,
                "reviews": reviews[-10:],  # Last 10 reviews
            },
        )

    async def _featured(self, params: Dict) -> SkillResult:
        """Get top/featured functions based on composite score."""
        data = self._load()
        listings = list(data.get("listings", {}).values())
        listings = [l for l in listings if l.get("status") == "active"]

        # Composite score: weighted rating + import popularity
        import math
        for l in listings:
            rating_score = l.get("avg_rating", 0) * l.get("rating_count", 0)
            import_score = math.log1p(l.get("import_count", 0)) * 2
            l["_score"] = rating_score + import_score

        listings.sort(key=lambda l: -l["_score"])
        limit = params.get("limit", 10)
        top = listings[:limit]

        # Clean up temp score
        results = []
        for l in top:
            results.append({
                "listing_id": l["listing_id"],
                "function_name": l["function_name"],
                "agent_name": l["agent_name"],
                "category": l["category"],
                "description": l.get("description", "")[:200],
                "avg_rating": l.get("avg_rating", 0),
                "import_count": l.get("import_count", 0),
                "price_per_import": l.get("price_per_import", 0),
                "score": round(l["_score"], 2),
            })

        return SkillResult(
            success=True,
            message=f"Top {len(results)} featured functions",
            data={"featured": results},
        )

    async def _my_publications(self, params: Dict) -> SkillResult:
        """List an agent's published functions with earnings."""
        agent_name = params.get("agent_name")
        if not agent_name:
            return SkillResult(success=False, message="agent_name is required")

        data = self._load()
        listings = data.get("listings", {})

        my_listings = [l for l in listings.values() if l.get("agent_name") == agent_name]
        total_earnings = data.get("earnings", {}).get(agent_name, 0)
        total_imports = sum(l.get("import_count", 0) for l in my_listings)

        return SkillResult(
            success=True,
            message=f"{agent_name} has {len(my_listings)} published functions, ${total_earnings:.2f} earned",
            data={
                "agent_name": agent_name,
                "publications": [{
                    "listing_id": l["listing_id"],
                    "function_name": l["function_name"],
                    "category": l["category"],
                    "import_count": l.get("import_count", 0),
                    "avg_rating": l.get("avg_rating", 0),
                    "total_revenue": l.get("total_revenue", 0),
                    "status": l.get("status"),
                    "published_at": l.get("published_at"),
                } for l in my_listings],
                "total_earnings": total_earnings,
                "total_imports": total_imports,
            },
        )

    async def _unpublish(self, params: Dict) -> SkillResult:
        """Remove a function from the marketplace."""
        listing_id = params.get("listing_id")
        agent_name = params.get("agent_name")
        if not listing_id or not agent_name:
            return SkillResult(success=False, message="listing_id and agent_name are required")

        data = self._load()
        listing = data.get("listings", {}).get(listing_id)
        if not listing:
            return SkillResult(success=False, message=f"Listing '{listing_id}' not found")

        if listing["agent_name"] != agent_name:
            return SkillResult(
                success=False,
                message=f"Only the publisher ({listing['agent_name']}) can unpublish this function",
            )

        listing["status"] = "unpublished"
        listing["unpublished_at"] = datetime.utcnow().isoformat()
        self._save(data)

        return SkillResult(
            success=True,
            message=f"Unpublished '{listing['function_name']}' from marketplace",
            data={"listing_id": listing_id, "function_name": listing["function_name"]},
        )

    async def _status(self, params: Dict) -> SkillResult:
        """Get marketplace overview."""
        data = self._load()
        stats = data.get("stats", {})
        listings = data.get("listings", {})

        active = sum(1 for l in listings.values() if l.get("status") == "active")

        # Category breakdown
        categories = {}
        for l in listings.values():
            if l.get("status") == "active":
                cat = l.get("category", "utility")
                categories[cat] = categories.get(cat, 0) + 1

        # Top publishers
        publishers = {}
        for l in listings.values():
            if l.get("status") == "active":
                pub = l.get("agent_name", "unknown")
                publishers[pub] = publishers.get(pub, 0) + 1

        return SkillResult(
            success=True,
            message=f"Function Marketplace: {active} active listings, "
                    f"{stats.get('total_imported', 0)} imports, "
                    f"${stats.get('total_revenue', 0):.2f} revenue",
            data={
                "active_listings": active,
                "total_published": stats.get("total_published", 0),
                "total_imported": stats.get("total_imported", 0),
                "total_reviews": stats.get("total_reviews", 0),
                "total_revenue": stats.get("total_revenue", 0),
                "categories": categories,
                "publishers": publishers,
                "earnings": data.get("earnings", {}),
            },
        )
