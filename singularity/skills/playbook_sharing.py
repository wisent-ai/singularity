#!/usr/bin/env python3
"""
PlaybookSharingSkill - Cross-agent playbook exchange and discovery.

Enables agents to share their most effective playbooks with other agent
replicas. While FunctionMarketplace handles serverless function exchange,
this skill focuses specifically on *strategic knowledge* exchange - the
playbooks that encode how to approach different task types.

Use cases:
1. Agent A develops a great "code review" playbook → publishes it
2. Agent B discovers it, imports it into their AgentReflectionSkill
3. Agent A earns reputation/credit for sharing effective strategies
4. The fleet develops collective intelligence from shared experience

Features:
1. **Publish** - Export a playbook from AgentReflection for sharing
2. **Browse** - Discover shared playbooks by tag, category, or search
3. **Import** - Download a playbook into your AgentReflectionSkill
4. **Rate** - Rate imported playbooks based on effectiveness
5. **Top** - Surface the highest-rated shared playbooks
6. **Sync** - Bulk import/export for fleet-wide knowledge sharing
7. **Recommend** - Get playbook recommendations based on agent's gaps

Event topics emitted:
  - playbook_sharing.published   - Playbook shared to marketplace
  - playbook_sharing.imported    - Playbook imported by another agent
  - playbook_sharing.rated       - Shared playbook received rating

Pillar: Replication + Self-Improvement (collective intelligence)
"""

import json
import hashlib
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

from .base import Skill, SkillResult, SkillManifest, SkillAction

DATA_DIR = Path(__file__).parent.parent / "data"
SHARING_FILE = DATA_DIR / "playbook_sharing.json"

MAX_SHARED = 500
MAX_RATINGS = 50
MAX_HISTORY = 200

CATEGORIES = [
    "development",      # Code writing, debugging, testing
    "deployment",       # CI/CD, infrastructure, releases
    "code_review",      # Code analysis, PR review
    "data_analysis",    # Data processing, analytics
    "communication",    # Messaging, notifications, reports
    "revenue",          # Billing, pricing, monetization
    "self_improvement", # Agent optimization, learning
    "operations",       # Monitoring, incident response
    "general",          # Catch-all
]


def _generate_id() -> str:
    import uuid
    return uuid.uuid4().hex[:12]


def _hash_playbook(playbook: Dict) -> str:
    """Create a content hash for deduplication."""
    content = json.dumps({
        "steps": playbook.get("steps", []),
        "task_pattern": playbook.get("task_pattern", ""),
    }, sort_keys=True)
    return hashlib.sha256(content.encode()).hexdigest()[:16]


class PlaybookSharingSkill(Skill):
    """
    Cross-agent playbook exchange skill.

    Bridges AgentReflectionSkill (which holds playbooks) with a shared
    marketplace where agents can discover and import each other's strategies.
    """

    def __init__(self, credentials: Dict[str, str] = None):
        super().__init__(credentials)
        self._load_state()

    def _load_state(self):
        """Load or initialize sharing state."""
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        if SHARING_FILE.exists():
            try:
                with open(SHARING_FILE) as f:
                    data = json.load(f)
                self._shared = data.get("shared", {})
                self._imports = data.get("imports", [])
                self._ratings = data.get("ratings", {})
                self._config = data.get("config", self._default_config())
                self._stats = data.get("stats", self._default_stats())
                self._history = data.get("history", [])
            except (json.JSONDecodeError, Exception):
                self._init_empty()
        else:
            self._init_empty()

    def _init_empty(self):
        self._shared: Dict[str, Dict] = {}      # shared_id -> shared playbook
        self._imports: List[Dict] = []            # import log
        self._ratings: Dict[str, List] = {}       # shared_id -> list of ratings
        self._config = self._default_config()
        self._stats = self._default_stats()
        self._history: List[Dict] = []

    def _default_config(self) -> Dict:
        return {
            "min_effectiveness_to_share": 0.5,  # Only share playbooks above 50% effectiveness
            "min_uses_to_share": 3,             # Must have been used at least 3 times
            "auto_import_threshold": 0.8,       # Auto-import playbooks rated above 80%
            "default_category": "general",
        }

    def _default_stats(self) -> Dict:
        return {
            "total_published": 0,
            "total_imported": 0,
            "total_ratings": 0,
            "total_syncs": 0,
        }

    def _save(self):
        """Persist sharing state."""
        data = {
            "shared": dict(list(self._shared.items())[:MAX_SHARED]),
            "imports": self._imports[-MAX_HISTORY:],
            "ratings": {k: v[-MAX_RATINGS:] for k, v in self._ratings.items()},
            "config": self._config,
            "stats": self._stats,
            "history": self._history[-MAX_HISTORY:],
        }
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        with open(SHARING_FILE, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def _log(self, action: str, details: Dict):
        """Log an action to history."""
        self._history.append({
            "action": action,
            "details": details,
            "timestamp": datetime.utcnow().isoformat(),
        })

    @classmethod
    def manifest(cls) -> SkillManifest:
        return SkillManifest(
            name="playbook_sharing",
            description="Cross-agent playbook exchange and discovery",
            version="1.0.0",
            actions=[
                SkillAction(
                    name="publish",
                    description="Publish a playbook for other agents to discover",
                    parameters=["playbook_name", "agent_name", "category", "description"],
                ),
                SkillAction(
                    name="browse",
                    description="Browse shared playbooks by category, tag, or search",
                    parameters=["category", "tags", "search", "min_rating", "limit"],
                ),
                SkillAction(
                    name="import_playbook",
                    description="Import a shared playbook into your AgentReflectionSkill",
                    parameters=["shared_id", "agent_name"],
                ),
                SkillAction(
                    name="rate",
                    description="Rate a shared playbook based on your experience",
                    parameters=["shared_id", "rating", "agent_name", "comment"],
                ),
                SkillAction(
                    name="top",
                    description="Get the highest-rated shared playbooks",
                    parameters=["category", "limit"],
                ),
                SkillAction(
                    name="sync",
                    description="Bulk export/import playbooks for fleet-wide sharing",
                    parameters=["mode", "agent_name", "playbooks"],
                ),
                SkillAction(
                    name="recommend",
                    description="Get playbook recommendations based on agent's needs",
                    parameters=["agent_name", "task_tags", "gap_areas"],
                ),
                SkillAction(
                    name="status",
                    description="Get sharing statistics and configuration",
                    parameters=[],
                ),
            ],
        )

    async def execute(self, action: str, params: Dict) -> SkillResult:
        handlers = {
            "publish": self._publish,
            "browse": self._browse,
            "import_playbook": self._import_playbook,
            "rate": self._rate,
            "top": self._top,
            "sync": self._sync,
            "recommend": self._recommend,
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
        """Publish a playbook for other agents to discover."""
        playbook_name = params.get("playbook_name")
        agent_name = params.get("agent_name")
        if not playbook_name or not agent_name:
            return SkillResult(
                success=False,
                message="Required: playbook_name and agent_name",
            )

        # Try to get playbook from AgentReflectionSkill via context
        playbook_data = params.get("playbook_data")
        if not playbook_data and self.context:
            try:
                result = await self.context.invoke_skill(
                    "agent_reflection", "list_playbooks", {}
                )
                if result.success and result.data:
                    all_playbooks = result.data.get("playbooks", [])
                    for pb in all_playbooks:
                        if pb.get("name") == playbook_name:
                            playbook_data = pb
                            break
            except Exception:
                pass

        if not playbook_data:
            # Allow manual playbook data
            playbook_data = {
                "name": playbook_name,
                "task_pattern": params.get("task_pattern", ""),
                "steps": params.get("steps", []),
                "pitfalls": params.get("pitfalls", []),
                "tags": params.get("tags", []),
                "effectiveness": float(params.get("effectiveness", 0)),
                "uses": int(params.get("uses", 0)),
            }

        if not playbook_data.get("steps"):
            return SkillResult(
                success=False,
                message=f"Playbook '{playbook_name}' has no steps. Cannot publish empty playbooks.",
            )

        # Check minimum quality thresholds
        effectiveness = playbook_data.get("effectiveness", 0)
        uses = playbook_data.get("uses", 0)
        min_eff = self._config["min_effectiveness_to_share"]
        min_uses = self._config["min_uses_to_share"]

        if effectiveness < min_eff and uses >= min_uses:
            return SkillResult(
                success=False,
                message=f"Playbook effectiveness ({effectiveness:.0%}) below threshold ({min_eff:.0%}). "
                        f"Improve it before sharing.",
            )

        # Check for duplicates
        content_hash = _hash_playbook(playbook_data)
        for sid, shared in self._shared.items():
            if shared.get("content_hash") == content_hash:
                return SkillResult(
                    success=False,
                    message=f"A similar playbook already exists (ID: {sid})",
                    data={"existing_id": sid},
                )

        category = params.get("category", self._config["default_category"])
        if category not in CATEGORIES:
            category = "general"

        shared_id = _generate_id()
        shared_entry = {
            "id": shared_id,
            "name": playbook_data.get("name", playbook_name),
            "task_pattern": playbook_data.get("task_pattern", ""),
            "steps": playbook_data.get("steps", []),
            "pitfalls": playbook_data.get("pitfalls", []),
            "tags": playbook_data.get("tags", []),
            "category": category,
            "description": params.get("description", ""),
            "author": agent_name,
            "effectiveness": effectiveness,
            "uses_by_author": uses,
            "content_hash": content_hash,
            "published_at": datetime.utcnow().isoformat(),
            "import_count": 0,
            "avg_rating": 0.0,
        }

        self._shared[shared_id] = shared_entry
        self._stats["total_published"] += 1
        self._log("publish", {"shared_id": shared_id, "name": playbook_name, "agent": agent_name})
        self._save()

        # Emit event if context available
        if self.context:
            try:
                await self.context.emit_event("playbook_sharing.published", {
                    "shared_id": shared_id,
                    "name": playbook_name,
                    "author": agent_name,
                    "category": category,
                })
            except Exception:
                pass

        return SkillResult(
            success=True,
            message=f"Playbook '{playbook_name}' published (ID: {shared_id})",
            data={"shared_id": shared_id, "entry": shared_entry},
        )

    async def _browse(self, params: Dict) -> SkillResult:
        """Browse shared playbooks with filtering."""
        category = params.get("category")
        tags = params.get("tags", [])
        search = params.get("search", "").lower()
        min_rating = float(params.get("min_rating", 0))
        limit = int(params.get("limit", 20))

        results = []
        for sid, entry in self._shared.items():
            # Category filter
            if category and entry.get("category") != category:
                continue

            # Tag filter
            if tags:
                entry_tags = set(entry.get("tags", []))
                if not set(tags) & entry_tags:
                    continue

            # Search filter
            if search:
                searchable = (
                    entry.get("name", "").lower() + " " +
                    entry.get("description", "").lower() + " " +
                    entry.get("task_pattern", "").lower() + " " +
                    " ".join(entry.get("tags", []))
                )
                if search not in searchable:
                    continue

            # Rating filter
            if min_rating > 0 and entry.get("avg_rating", 0) < min_rating:
                continue

            results.append({
                "id": sid,
                "name": entry["name"],
                "category": entry.get("category", "general"),
                "author": entry.get("author", "unknown"),
                "avg_rating": entry.get("avg_rating", 0),
                "import_count": entry.get("import_count", 0),
                "tags": entry.get("tags", []),
                "steps_count": len(entry.get("steps", [])),
                "effectiveness": entry.get("effectiveness", 0),
                "description": entry.get("description", ""),
            })

        # Sort by rating then import count
        results.sort(key=lambda x: (x["avg_rating"], x["import_count"]), reverse=True)
        results = results[:limit]

        return SkillResult(
            success=True,
            message=f"Found {len(results)} shared playbooks",
            data={"playbooks": results, "total": len(self._shared)},
        )

    async def _import_playbook(self, params: Dict) -> SkillResult:
        """Import a shared playbook into the agent's AgentReflectionSkill."""
        shared_id = params.get("shared_id")
        agent_name = params.get("agent_name", "unknown")

        if not shared_id:
            return SkillResult(success=False, message="Required: shared_id")

        entry = self._shared.get(shared_id)
        if not entry:
            return SkillResult(
                success=False,
                message=f"Shared playbook '{shared_id}' not found",
            )

        # Don't import your own playbooks
        if entry.get("author") == agent_name:
            return SkillResult(
                success=False,
                message="Cannot import your own playbook",
            )

        # Check for duplicate imports
        for imp in self._imports:
            if imp.get("shared_id") == shared_id and imp.get("agent") == agent_name:
                return SkillResult(
                    success=False,
                    message=f"Already imported playbook '{entry['name']}'",
                )

        # Try to create the playbook via AgentReflectionSkill
        created_via_skill = False
        if self.context:
            try:
                result = await self.context.invoke_skill(
                    "agent_reflection", "create_playbook", {
                        "name": f"[shared] {entry['name']}",
                        "task_pattern": entry.get("task_pattern", ""),
                        "steps": entry.get("steps", []),
                        "pitfalls": entry.get("pitfalls", []),
                        "tags": entry.get("tags", []) + ["imported", f"from:{entry.get('author', 'unknown')}"],
                    }
                )
                created_via_skill = result.success
            except Exception:
                pass

        # Record import
        import_record = {
            "shared_id": shared_id,
            "agent": agent_name,
            "playbook_name": entry["name"],
            "imported_at": datetime.utcnow().isoformat(),
            "created_in_reflection": created_via_skill,
        }
        self._imports.append(import_record)

        # Update shared entry stats
        entry["import_count"] = entry.get("import_count", 0) + 1
        self._stats["total_imported"] += 1
        self._log("import", {"shared_id": shared_id, "agent": agent_name})
        self._save()

        # Emit event
        if self.context:
            try:
                await self.context.emit_event("playbook_sharing.imported", {
                    "shared_id": shared_id,
                    "name": entry["name"],
                    "agent": agent_name,
                    "author": entry.get("author"),
                })
            except Exception:
                pass

        return SkillResult(
            success=True,
            message=f"Imported playbook '{entry['name']}' from {entry.get('author', 'unknown')}"
                    + (" (added to AgentReflection)" if created_via_skill else " (manual import)"),
            data={
                "import": import_record,
                "playbook": {
                    "name": entry["name"],
                    "steps": entry.get("steps", []),
                    "pitfalls": entry.get("pitfalls", []),
                    "tags": entry.get("tags", []),
                    "task_pattern": entry.get("task_pattern", ""),
                },
            },
        )

    async def _rate(self, params: Dict) -> SkillResult:
        """Rate a shared playbook."""
        shared_id = params.get("shared_id")
        rating = params.get("rating")
        agent_name = params.get("agent_name", "anonymous")
        comment = params.get("comment", "")

        if not shared_id or rating is None:
            return SkillResult(success=False, message="Required: shared_id and rating")

        rating = float(rating)
        if rating < 1 or rating > 5:
            return SkillResult(success=False, message="Rating must be between 1 and 5")

        entry = self._shared.get(shared_id)
        if not entry:
            return SkillResult(success=False, message=f"Shared playbook '{shared_id}' not found")

        # Check if agent already rated
        if shared_id not in self._ratings:
            self._ratings[shared_id] = []

        for existing in self._ratings[shared_id]:
            if existing.get("agent") == agent_name:
                # Update existing rating
                existing["rating"] = rating
                existing["comment"] = comment
                existing["updated_at"] = datetime.utcnow().isoformat()
                break
        else:
            # New rating
            self._ratings[shared_id].append({
                "agent": agent_name,
                "rating": rating,
                "comment": comment,
                "created_at": datetime.utcnow().isoformat(),
            })

        # Recalculate average
        all_ratings = [r["rating"] for r in self._ratings[shared_id]]
        entry["avg_rating"] = sum(all_ratings) / len(all_ratings) if all_ratings else 0

        self._stats["total_ratings"] += 1
        self._log("rate", {"shared_id": shared_id, "agent": agent_name, "rating": rating})
        self._save()

        # Emit event
        if self.context:
            try:
                await self.context.emit_event("playbook_sharing.rated", {
                    "shared_id": shared_id,
                    "agent": agent_name,
                    "rating": rating,
                    "avg_rating": entry["avg_rating"],
                })
            except Exception:
                pass

        return SkillResult(
            success=True,
            message=f"Rated playbook '{entry['name']}': {rating}/5 (avg: {entry['avg_rating']:.1f})",
            data={
                "shared_id": shared_id,
                "rating": rating,
                "avg_rating": entry["avg_rating"],
                "total_ratings": len(all_ratings),
            },
        )

    async def _top(self, params: Dict) -> SkillResult:
        """Get top-rated shared playbooks."""
        category = params.get("category")
        limit = int(params.get("limit", 10))

        entries = list(self._shared.values())

        if category:
            entries = [e for e in entries if e.get("category") == category]

        # Sort by average rating (weighted by number of ratings)
        def score(e):
            avg = e.get("avg_rating", 0)
            num_ratings = len(self._ratings.get(e["id"], []))
            imports = e.get("import_count", 0)
            # Wilson score lower bound approximation for ranking
            if num_ratings == 0:
                return imports * 0.1
            return avg * (1 - 1 / (2 * num_ratings + 1)) + imports * 0.05

        entries.sort(key=score, reverse=True)
        entries = entries[:limit]

        top_list = []
        for e in entries:
            ratings = self._ratings.get(e["id"], [])
            top_list.append({
                "id": e["id"],
                "name": e["name"],
                "category": e.get("category", "general"),
                "author": e.get("author", "unknown"),
                "avg_rating": e.get("avg_rating", 0),
                "num_ratings": len(ratings),
                "import_count": e.get("import_count", 0),
                "effectiveness": e.get("effectiveness", 0),
                "steps_count": len(e.get("steps", [])),
            })

        return SkillResult(
            success=True,
            message=f"Top {len(top_list)} playbooks" + (f" in '{category}'" if category else ""),
            data={"top": top_list},
        )

    async def _sync(self, params: Dict) -> SkillResult:
        """Bulk export/import for fleet-wide knowledge sharing."""
        mode = params.get("mode", "export")
        agent_name = params.get("agent_name", "unknown")

        if mode == "export":
            # Export all playbooks from this agent
            my_shared = {
                sid: entry for sid, entry in self._shared.items()
                if entry.get("author") == agent_name
            }
            self._stats["total_syncs"] += 1
            self._log("sync_export", {"agent": agent_name, "count": len(my_shared)})
            self._save()

            return SkillResult(
                success=True,
                message=f"Exported {len(my_shared)} playbooks from {agent_name}",
                data={"playbooks": my_shared, "agent": agent_name},
            )

        elif mode == "import":
            # Bulk import playbooks from another agent's export
            playbooks = params.get("playbooks", {})
            if not playbooks:
                return SkillResult(success=False, message="No playbooks provided for import")

            imported = 0
            skipped = 0
            for sid, entry in playbooks.items():
                if entry.get("author") == agent_name:
                    skipped += 1
                    continue
                content_hash = _hash_playbook(entry)
                # Check for existing
                duplicate = False
                for existing in self._shared.values():
                    if existing.get("content_hash") == content_hash:
                        duplicate = True
                        break
                if duplicate:
                    skipped += 1
                    continue

                new_id = _generate_id()
                entry["id"] = new_id
                entry["content_hash"] = content_hash
                self._shared[new_id] = entry
                imported += 1

            self._stats["total_syncs"] += 1
            self._log("sync_import", {"agent": agent_name, "imported": imported, "skipped": skipped})
            self._save()

            return SkillResult(
                success=True,
                message=f"Imported {imported} playbooks, skipped {skipped}",
                data={"imported": imported, "skipped": skipped},
            )

        return SkillResult(success=False, message=f"Unknown mode: {mode}. Use 'export' or 'import'")

    async def _recommend(self, params: Dict) -> SkillResult:
        """Recommend playbooks based on agent's needs."""
        agent_name = params.get("agent_name", "unknown")
        task_tags = params.get("task_tags", [])
        gap_areas = params.get("gap_areas", [])

        if not task_tags and not gap_areas:
            return SkillResult(
                success=False,
                message="Provide task_tags and/or gap_areas to get recommendations",
            )

        # Find playbooks the agent hasn't imported yet
        imported_ids = set()
        for imp in self._imports:
            if imp.get("agent") == agent_name:
                imported_ids.add(imp.get("shared_id"))

        candidates = []
        for sid, entry in self._shared.items():
            if sid in imported_ids:
                continue
            if entry.get("author") == agent_name:
                continue

            score = 0
            entry_tags = set(entry.get("tags", []))
            entry_category = entry.get("category", "")

            # Tag match scoring
            if task_tags:
                overlap = len(set(task_tags) & entry_tags)
                score += overlap * 10

            # Gap area scoring
            if gap_areas:
                for gap in gap_areas:
                    gap_lower = gap.lower()
                    if gap_lower in entry_category:
                        score += 15
                    if gap_lower in entry.get("task_pattern", "").lower():
                        score += 10
                    if gap_lower in " ".join(entry.get("tags", [])).lower():
                        score += 5

            # Quality bonus
            score += entry.get("avg_rating", 0) * 5
            score += entry.get("import_count", 0) * 2
            score += entry.get("effectiveness", 0) * 10

            if score > 0:
                candidates.append({
                    "id": sid,
                    "name": entry["name"],
                    "category": entry.get("category", "general"),
                    "author": entry.get("author", "unknown"),
                    "score": score,
                    "avg_rating": entry.get("avg_rating", 0),
                    "import_count": entry.get("import_count", 0),
                    "tags": list(entry_tags),
                    "reason": self._explain_recommendation(entry, task_tags, gap_areas),
                })

        candidates.sort(key=lambda x: x["score"], reverse=True)
        candidates = candidates[:10]

        return SkillResult(
            success=True,
            message=f"Found {len(candidates)} recommended playbooks",
            data={"recommendations": candidates},
        )

    def _explain_recommendation(self, entry: Dict, task_tags: List, gap_areas: List) -> str:
        """Generate a human-readable explanation for why this playbook is recommended."""
        reasons = []
        entry_tags = set(entry.get("tags", []))

        if task_tags:
            overlap = set(task_tags) & entry_tags
            if overlap:
                reasons.append(f"matches tags: {', '.join(overlap)}")

        if gap_areas:
            for gap in gap_areas:
                if gap.lower() in entry.get("category", "").lower():
                    reasons.append(f"covers gap area: {gap}")

        if entry.get("avg_rating", 0) >= 4:
            reasons.append(f"highly rated ({entry['avg_rating']:.1f}/5)")

        if entry.get("import_count", 0) >= 5:
            reasons.append(f"popular ({entry['import_count']} imports)")

        return "; ".join(reasons) if reasons else "general match"

    async def _status(self, params: Dict) -> SkillResult:
        """Get sharing statistics."""
        return SkillResult(
            success=True,
            message="Playbook sharing status",
            data={
                "stats": self._stats,
                "config": self._config,
                "total_shared": len(self._shared),
                "total_imports": len(self._imports),
                "categories": {
                    cat: sum(1 for e in self._shared.values() if e.get("category") == cat)
                    for cat in CATEGORIES
                    if any(e.get("category") == cat for e in self._shared.values())
                },
                "history_count": len(self._history),
            },
        )
