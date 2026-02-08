#!/usr/bin/env python3
"""
Skill Marketplace Hub - Inter-agent skill exchange and distribution.

This is the missing link between the Replication and Revenue pillars.
Currently, agents can replicate and communicate, but they can't share
or trade their actual skills. The Skill Marketplace Hub enables:

  1. PUBLISH: Any agent can package and list a skill for others to install
  2. BROWSE: Agents discover skills by category, rating, price, or search
  3. INSTALL: Download and integrate skills from other agents
  4. RATE: Build trust through reviews and ratings
  5. EARN: Skill authors earn revenue from installs (Revenue pillar)

This transforms the agent network from isolated instances into an
ecosystem where specialization is rewarded - an agent that builds a
great code review skill can earn revenue from every other agent that
installs it.

Architecture:
- File-based hub store (singularity/data/skill_hub.json)
- Each listing includes skill manifest, source hash, pricing, reviews
- Install tracking for revenue attribution
- Search with relevance scoring across name, description, tags
- Compatible with KnowledgeSharingSkill for propagating skill discoveries

Pillars served:
- Revenue Generation: Skill authors earn from installs
- Replication: Agents share capabilities across the network
- Self-Improvement: Agents can acquire new skills dynamically
"""

import json
import hashlib
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from .base import Skill, SkillResult, SkillManifest, SkillAction


HUB_FILE = Path(__file__).parent.parent / "data" / "skill_hub.json"
MAX_LISTINGS = 500
MAX_REVIEWS = 50  # Per listing
MAX_INSTALLED = 200


class SkillMarketplaceHub(Skill):
    """
    Inter-agent skill marketplace for publishing, discovering, and installing skills.

    Enables agents to trade skills as a form of value exchange. Authors
    earn revenue from installs, and consumers gain new capabilities
    without building from scratch.
    """

    def __init__(self, credentials: Dict[str, str] = None):
        super().__init__(credentials)
        self._ensure_data()

    def _ensure_data(self):
        HUB_FILE.parent.mkdir(parents=True, exist_ok=True)
        if not HUB_FILE.exists():
            self._save(self._default_state())

    def _default_state(self) -> Dict:
        return {
            "listings": [],
            "installed": [],
            "reviews": [],
            "earnings": {},
            "install_log": [],
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
        }

    def _load(self) -> Dict:
        try:
            return json.loads(HUB_FILE.read_text())
        except (json.JSONDecodeError, FileNotFoundError):
            state = self._default_state()
            self._save(state)
            return state

    def _save(self, state: Dict):
        state["last_updated"] = datetime.now().isoformat()
        HUB_FILE.parent.mkdir(parents=True, exist_ok=True)
        HUB_FILE.write_text(json.dumps(state, indent=2, default=str))

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="skill_marketplace_hub",
            name="Skill Marketplace Hub",
            version="1.0.0",
            category="replication",
            description="Inter-agent skill exchange - publish, discover, install, and trade skills across the agent network",
            actions=[
                SkillAction(
                    name="publish",
                    description="Publish a skill to the marketplace for other agents to install",
                    parameters={
                        "skill_id": {"type": "string", "required": True, "description": "ID of the skill to publish"},
                        "name": {"type": "string", "required": True, "description": "Display name"},
                        "description": {"type": "string", "required": True, "description": "What the skill does"},
                        "version": {"type": "string", "required": True, "description": "Semantic version"},
                        "category": {"type": "string", "required": False, "description": "Category: dev, revenue, social, data, content, automation"},
                        "tags": {"type": "list", "required": False, "description": "Searchable tags"},
                        "price": {"type": "float", "required": False, "description": "Price per install in USD (0 = free)"},
                        "author_agent_id": {"type": "string", "required": False, "description": "Publishing agent's ID"},
                        "source_hash": {"type": "string", "required": False, "description": "Hash of skill source for integrity"},
                        "skill_config": {"type": "dict", "required": False, "description": "Skill configuration/metadata"},
                    },
                ),
                SkillAction(
                    name="browse",
                    description="Browse available skills with optional filters",
                    parameters={
                        "category": {"type": "string", "required": False, "description": "Filter by category"},
                        "min_rating": {"type": "float", "required": False, "description": "Minimum average rating (1-5)"},
                        "max_price": {"type": "float", "required": False, "description": "Maximum price filter"},
                        "free_only": {"type": "bool", "required": False, "description": "Only show free skills"},
                        "sort_by": {"type": "string", "required": False, "description": "Sort: rating, installs, newest, price"},
                        "limit": {"type": "int", "required": False, "description": "Max results to return"},
                    },
                ),
                SkillAction(
                    name="search",
                    description="Search skills by keyword across names, descriptions, and tags",
                    parameters={
                        "query": {"type": "string", "required": True, "description": "Search query"},
                        "limit": {"type": "int", "required": False, "description": "Max results"},
                    },
                ),
                SkillAction(
                    name="install",
                    description="Install a skill from the marketplace",
                    parameters={
                        "listing_id": {"type": "string", "required": True, "description": "ID of the listing to install"},
                        "installer_agent_id": {"type": "string", "required": False, "description": "Installing agent's ID"},
                    },
                ),
                SkillAction(
                    name="review",
                    description="Rate and review an installed skill",
                    parameters={
                        "listing_id": {"type": "string", "required": True, "description": "ID of the listing to review"},
                        "rating": {"type": "int", "required": True, "description": "Rating 1-5"},
                        "comment": {"type": "string", "required": False, "description": "Review text"},
                        "reviewer_agent_id": {"type": "string", "required": False, "description": "Reviewing agent's ID"},
                    },
                ),
                SkillAction(
                    name="get_listing",
                    description="Get detailed info about a specific skill listing",
                    parameters={
                        "listing_id": {"type": "string", "required": True, "description": "Listing ID"},
                    },
                ),
                SkillAction(
                    name="update_listing",
                    description="Update an existing skill listing (new version, price change, etc.)",
                    parameters={
                        "listing_id": {"type": "string", "required": True, "description": "Listing ID to update"},
                        "version": {"type": "string", "required": False, "description": "New version"},
                        "price": {"type": "float", "required": False, "description": "New price"},
                        "description": {"type": "string", "required": False, "description": "Updated description"},
                        "status": {"type": "string", "required": False, "description": "active, paused, or retired"},
                    },
                ),
                SkillAction(
                    name="my_listings",
                    description="View skills you've published and their performance",
                    parameters={
                        "author_agent_id": {"type": "string", "required": False, "description": "Your agent ID"},
                    },
                ),
                SkillAction(
                    name="my_installs",
                    description="View skills you've installed from the marketplace",
                    parameters={
                        "installer_agent_id": {"type": "string", "required": False, "description": "Your agent ID"},
                    },
                ),
                SkillAction(
                    name="earnings_report",
                    description="View revenue earned from skill sales",
                    parameters={
                        "author_agent_id": {"type": "string", "required": False, "description": "Your agent ID"},
                    },
                ),
            ],
            required_credentials=[],
            install_cost=0,
            author="singularity",
        )

    async def execute(self, action: str, params: Dict[str, Any]) -> SkillResult:
        actions = {
            "publish": self._publish,
            "browse": self._browse,
            "search": self._search,
            "install": self._install,
            "review": self._review,
            "get_listing": self._get_listing,
            "update_listing": self._update_listing,
            "my_listings": self._my_listings,
            "my_installs": self._my_installs,
            "earnings_report": self._earnings_report,
        }

        handler = actions.get(action)
        if not handler:
            return SkillResult(
                success=False,
                message=f"Unknown action: {action}. Available: {list(actions.keys())}",
            )

        try:
            return handler(params)
        except Exception as e:
            return SkillResult(success=False, message=f"Error in {action}: {str(e)}")

    def _publish(self, params: Dict) -> SkillResult:
        """Publish a skill listing to the marketplace."""
        skill_id = params.get("skill_id")
        name = params.get("name")
        description = params.get("description")
        version = params.get("version")

        if not all([skill_id, name, description, version]):
            return SkillResult(
                success=False,
                message="Required: skill_id, name, description, version",
            )

        state = self._load()

        # Check for duplicate skill_id from same author
        author = params.get("author_agent_id", "self")
        existing = [
            l for l in state["listings"]
            if l["skill_id"] == skill_id and l["author_agent_id"] == author and l["status"] != "retired"
        ]
        if existing:
            return SkillResult(
                success=False,
                message=f"Skill '{skill_id}' already published by {author}. Use update_listing instead.",
            )

        if len(state["listings"]) >= MAX_LISTINGS:
            # Remove oldest retired listings to make room
            retired = [l for l in state["listings"] if l["status"] == "retired"]
            if retired:
                state["listings"].remove(retired[0])
            else:
                return SkillResult(
                    success=False,
                    message=f"Marketplace full ({MAX_LISTINGS} listings). Retire old listings first.",
                )

        listing_id = f"hub_{uuid.uuid4().hex[:12]}"
        source_hash = params.get("source_hash", hashlib.sha256(f"{skill_id}:{version}".encode()).hexdigest()[:16])
        tags = params.get("tags", [])
        category = params.get("category", "general")
        price = max(0, float(params.get("price", 0)))

        listing = {
            "listing_id": listing_id,
            "skill_id": skill_id,
            "name": name,
            "description": description,
            "version": version,
            "category": category,
            "tags": tags if isinstance(tags, list) else [tags],
            "price": price,
            "author_agent_id": author,
            "source_hash": source_hash,
            "skill_config": params.get("skill_config", {}),
            "status": "active",
            "install_count": 0,
            "avg_rating": 0.0,
            "review_count": 0,
            "published_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
        }

        state["listings"].append(listing)

        # Initialize earnings tracking for author
        if author not in state["earnings"]:
            state["earnings"][author] = {"total": 0.0, "by_skill": {}}

        self._save(state)

        return SkillResult(
            success=True,
            message=f"Published '{name}' v{version} to marketplace as {listing_id} (${price:.2f}/install)",
            data={"listing_id": listing_id, "listing": listing},
        )

    def _browse(self, params: Dict) -> SkillResult:
        """Browse marketplace listings with filters."""
        state = self._load()
        listings = [l for l in state["listings"] if l["status"] == "active"]

        # Apply filters
        category = params.get("category")
        if category:
            listings = [l for l in listings if l["category"] == category]

        min_rating = params.get("min_rating")
        if min_rating is not None:
            listings = [l for l in listings if l["avg_rating"] >= float(min_rating)]

        max_price = params.get("max_price")
        if max_price is not None:
            listings = [l for l in listings if l["price"] <= float(max_price)]

        if params.get("free_only"):
            listings = [l for l in listings if l["price"] == 0]

        # Sort
        sort_by = params.get("sort_by", "installs")
        sort_map = {
            "rating": lambda l: l["avg_rating"],
            "installs": lambda l: l["install_count"],
            "newest": lambda l: l["published_at"],
            "price": lambda l: l["price"],
        }
        sort_fn = sort_map.get(sort_by, sort_map["installs"])
        listings.sort(key=sort_fn, reverse=(sort_by != "price"))

        limit = int(params.get("limit", 20))
        listings = listings[:limit]

        # Return summary view
        summaries = []
        for l in listings:
            summaries.append({
                "listing_id": l["listing_id"],
                "name": l["name"],
                "version": l["version"],
                "category": l["category"],
                "price": l["price"],
                "avg_rating": l["avg_rating"],
                "install_count": l["install_count"],
                "author": l["author_agent_id"],
                "tags": l["tags"],
            })

        return SkillResult(
            success=True,
            message=f"Found {len(summaries)} skills matching filters",
            data={"listings": summaries, "total": len(summaries)},
        )

    def _search(self, params: Dict) -> SkillResult:
        """Search skills by keyword with relevance scoring."""
        query = params.get("query", "").lower().strip()
        if not query:
            return SkillResult(success=False, message="Search query required")

        state = self._load()
        listings = [l for l in state["listings"] if l["status"] == "active"]
        query_terms = query.split()

        scored = []
        for listing in listings:
            score = 0
            searchable_name = listing["name"].lower()
            searchable_desc = listing["description"].lower()
            searchable_tags = " ".join(t.lower() for t in listing.get("tags", []))
            searchable_id = listing["skill_id"].lower()

            for term in query_terms:
                # Name match (highest weight)
                if term in searchable_name:
                    score += 10
                # ID match
                if term in searchable_id:
                    score += 8
                # Tag match
                if term in searchable_tags:
                    score += 5
                # Description match
                if term in searchable_desc:
                    score += 2

            # Boost by popularity
            score += listing["install_count"] * 0.1
            score += listing["avg_rating"] * 0.5

            if score > 0:
                scored.append((score, listing))

        # Sort by relevance
        scored.sort(key=lambda x: x[0], reverse=True)
        limit = int(params.get("limit", 15))
        results = []
        for score, l in scored[:limit]:
            results.append({
                "listing_id": l["listing_id"],
                "name": l["name"],
                "description": l["description"][:200],
                "version": l["version"],
                "category": l["category"],
                "price": l["price"],
                "avg_rating": l["avg_rating"],
                "install_count": l["install_count"],
                "relevance_score": round(score, 2),
                "tags": l["tags"],
            })

        return SkillResult(
            success=True,
            message=f"Found {len(results)} skills for '{query}'",
            data={"results": results, "query": query},
        )

    def _install(self, params: Dict) -> SkillResult:
        """Install a skill from the marketplace."""
        listing_id = params.get("listing_id")
        if not listing_id:
            return SkillResult(success=False, message="listing_id required")

        state = self._load()
        installer = params.get("installer_agent_id", "self")

        # Find listing
        listing = next((l for l in state["listings"] if l["listing_id"] == listing_id), None)
        if not listing:
            return SkillResult(success=False, message=f"Listing not found: {listing_id}")

        if listing["status"] != "active":
            return SkillResult(success=False, message=f"Listing is {listing['status']}, cannot install")

        # Check if already installed
        already = next(
            (i for i in state["installed"]
             if i["listing_id"] == listing_id and i["installer_agent_id"] == installer),
            None,
        )
        if already:
            return SkillResult(
                success=False,
                message=f"Already installed: {listing['name']} v{already['version']}",
            )

        if len(state["installed"]) >= MAX_INSTALLED:
            return SkillResult(success=False, message=f"Install limit reached ({MAX_INSTALLED})")

        # Record install
        install_record = {
            "install_id": f"inst_{uuid.uuid4().hex[:10]}",
            "listing_id": listing_id,
            "skill_id": listing["skill_id"],
            "name": listing["name"],
            "version": listing["version"],
            "installer_agent_id": installer,
            "author_agent_id": listing["author_agent_id"],
            "price_paid": listing["price"],
            "source_hash": listing["source_hash"],
            "skill_config": listing.get("skill_config", {}),
            "installed_at": datetime.now().isoformat(),
        }
        state["installed"].append(install_record)

        # Update install count
        listing["install_count"] += 1

        # Record earnings for the author
        author = listing["author_agent_id"]
        if author not in state["earnings"]:
            state["earnings"][author] = {"total": 0.0, "by_skill": {}}
        state["earnings"][author]["total"] += listing["price"]
        skill_key = listing["skill_id"]
        if skill_key not in state["earnings"][author]["by_skill"]:
            state["earnings"][author]["by_skill"][skill_key] = {"revenue": 0.0, "installs": 0}
        state["earnings"][author]["by_skill"][skill_key]["revenue"] += listing["price"]
        state["earnings"][author]["by_skill"][skill_key]["installs"] += 1

        # Log the install
        state["install_log"].append({
            "listing_id": listing_id,
            "installer": installer,
            "author": author,
            "price": listing["price"],
            "timestamp": datetime.now().isoformat(),
        })
        # Keep log bounded
        if len(state["install_log"]) > 1000:
            state["install_log"] = state["install_log"][-500:]

        self._save(state)

        return SkillResult(
            success=True,
            message=f"Installed '{listing['name']}' v{listing['version']} from {author} (${listing['price']:.2f})",
            data={
                "install_record": install_record,
                "skill_config": listing.get("skill_config", {}),
            },
            revenue=-listing["price"],  # Cost to installer
        )

    def _review(self, params: Dict) -> SkillResult:
        """Rate and review an installed skill."""
        listing_id = params.get("listing_id")
        rating = params.get("rating")

        if not listing_id or rating is None:
            return SkillResult(success=False, message="listing_id and rating (1-5) required")

        rating = max(1, min(5, int(rating)))
        reviewer = params.get("reviewer_agent_id", "self")

        state = self._load()

        # Verify listing exists
        listing = next((l for l in state["listings"] if l["listing_id"] == listing_id), None)
        if not listing:
            return SkillResult(success=False, message=f"Listing not found: {listing_id}")

        # Check reviewer has installed
        installed = next(
            (i for i in state["installed"]
             if i["listing_id"] == listing_id and i["installer_agent_id"] == reviewer),
            None,
        )
        if not installed:
            return SkillResult(
                success=False,
                message="You must install a skill before reviewing it",
            )

        # Check for existing review
        existing_review = next(
            (r for r in state["reviews"]
             if r["listing_id"] == listing_id and r["reviewer_agent_id"] == reviewer),
            None,
        )
        if existing_review:
            # Update existing review
            existing_review["rating"] = rating
            existing_review["comment"] = params.get("comment", existing_review.get("comment", ""))
            existing_review["updated_at"] = datetime.now().isoformat()
        else:
            # Add new review
            review = {
                "review_id": f"rev_{uuid.uuid4().hex[:10]}",
                "listing_id": listing_id,
                "reviewer_agent_id": reviewer,
                "rating": rating,
                "comment": params.get("comment", ""),
                "created_at": datetime.now().isoformat(),
            }
            state["reviews"].append(review)

        # Recalculate average rating for this listing
        listing_reviews = [r for r in state["reviews"] if r["listing_id"] == listing_id]
        if listing_reviews:
            avg = sum(r["rating"] for r in listing_reviews) / len(listing_reviews)
            listing["avg_rating"] = round(avg, 2)
            listing["review_count"] = len(listing_reviews)

        self._save(state)

        return SkillResult(
            success=True,
            message=f"Reviewed '{listing['name']}': {rating}/5 stars (avg: {listing['avg_rating']})",
            data={
                "listing_id": listing_id,
                "rating": rating,
                "avg_rating": listing["avg_rating"],
                "total_reviews": listing["review_count"],
            },
        )

    def _get_listing(self, params: Dict) -> SkillResult:
        """Get detailed info about a listing."""
        listing_id = params.get("listing_id")
        if not listing_id:
            return SkillResult(success=False, message="listing_id required")

        state = self._load()
        listing = next((l for l in state["listings"] if l["listing_id"] == listing_id), None)
        if not listing:
            return SkillResult(success=False, message=f"Listing not found: {listing_id}")

        # Get reviews for this listing
        reviews = [r for r in state["reviews"] if r["listing_id"] == listing_id]

        return SkillResult(
            success=True,
            message=f"Listing: {listing['name']} v{listing['version']}",
            data={
                "listing": listing,
                "reviews": reviews[:MAX_REVIEWS],
            },
        )

    def _update_listing(self, params: Dict) -> SkillResult:
        """Update an existing listing."""
        listing_id = params.get("listing_id")
        if not listing_id:
            return SkillResult(success=False, message="listing_id required")

        state = self._load()
        listing = next((l for l in state["listings"] if l["listing_id"] == listing_id), None)
        if not listing:
            return SkillResult(success=False, message=f"Listing not found: {listing_id}")

        updated_fields = []
        if "version" in params:
            listing["version"] = params["version"]
            updated_fields.append("version")
        if "price" in params:
            listing["price"] = max(0, float(params["price"]))
            updated_fields.append("price")
        if "description" in params:
            listing["description"] = params["description"]
            updated_fields.append("description")
        if "status" in params and params["status"] in ("active", "paused", "retired"):
            listing["status"] = params["status"]
            updated_fields.append("status")

        if not updated_fields:
            return SkillResult(
                success=False,
                message="No valid fields to update. Provide version, price, description, or status.",
            )

        listing["updated_at"] = datetime.now().isoformat()
        self._save(state)

        return SkillResult(
            success=True,
            message=f"Updated '{listing['name']}': {', '.join(updated_fields)}",
            data={"listing": listing, "updated_fields": updated_fields},
        )

    def _my_listings(self, params: Dict) -> SkillResult:
        """View skills published by this agent."""
        author = params.get("author_agent_id", "self")
        state = self._load()

        listings = [l for l in state["listings"] if l["author_agent_id"] == author]

        total_installs = sum(l["install_count"] for l in listings)
        total_revenue = state["earnings"].get(author, {}).get("total", 0)
        active = len([l for l in listings if l["status"] == "active"])

        summaries = []
        for l in listings:
            per_skill = state["earnings"].get(author, {}).get("by_skill", {}).get(l["skill_id"], {})
            summaries.append({
                "listing_id": l["listing_id"],
                "name": l["name"],
                "version": l["version"],
                "status": l["status"],
                "price": l["price"],
                "installs": l["install_count"],
                "avg_rating": l["avg_rating"],
                "revenue": per_skill.get("revenue", 0),
            })

        return SkillResult(
            success=True,
            message=f"Your listings: {len(listings)} total ({active} active), {total_installs} installs, ${total_revenue:.2f} earned",
            data={
                "listings": summaries,
                "total_listings": len(listings),
                "active_listings": active,
                "total_installs": total_installs,
                "total_revenue": total_revenue,
            },
        )

    def _my_installs(self, params: Dict) -> SkillResult:
        """View skills installed by this agent."""
        installer = params.get("installer_agent_id", "self")
        state = self._load()

        installs = [i for i in state["installed"] if i["installer_agent_id"] == installer]
        total_spent = sum(i["price_paid"] for i in installs)

        return SkillResult(
            success=True,
            message=f"Installed {len(installs)} skills, spent ${total_spent:.2f}",
            data={
                "installed": installs,
                "total_installed": len(installs),
                "total_spent": total_spent,
            },
        )

    def _earnings_report(self, params: Dict) -> SkillResult:
        """Revenue report from skill marketplace sales."""
        author = params.get("author_agent_id", "self")
        state = self._load()

        earnings = state["earnings"].get(author, {"total": 0.0, "by_skill": {}})

        # Get recent install log for this author
        recent_sales = [
            log for log in state["install_log"]
            if log["author"] == author
        ][-20:]  # Last 20 sales

        # Top earners
        by_skill = earnings.get("by_skill", {})
        top_skills = sorted(
            by_skill.items(),
            key=lambda x: x[1]["revenue"],
            reverse=True,
        )[:10]

        return SkillResult(
            success=True,
            message=f"Total earnings: ${earnings['total']:.2f} from {sum(s[1]['installs'] for s in top_skills)} installs",
            data={
                "total_revenue": earnings["total"],
                "by_skill": dict(top_skills),
                "recent_sales": recent_sales,
            },
        )
