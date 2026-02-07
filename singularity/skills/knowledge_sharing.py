#!/usr/bin/env python3
"""
Knowledge Sharing Skill - Cross-agent learning and intelligence propagation.

When agents replicate, each replica operates independently and discovers things:
- Which strategies work for specific tasks
- Which approaches fail and should be avoided
- Cost optimizations and performance insights
- New capabilities or useful patterns

Without knowledge sharing, each replica learns in isolation. With it, the entire
agent network becomes a distributed learning system where discoveries propagate
across all instances.

This is the missing piece in the Replication pillar: replication creates the
agents, orchestration lets them communicate, task delegation coordinates their
work, and knowledge sharing lets them learn from each other.

Architecture:
- File-based knowledge store (works locally, can be networked via shared FS)
- Each agent publishes discoveries tagged with confidence and category
- Agents subscribe to knowledge categories they care about
- Knowledge items have TTL and can be superseded by newer findings
- Built-in deduplication and relevance scoring

Part of the Replication pillar: making multi-agent systems smarter than any
single agent by enabling collective intelligence.
"""

import json
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from .base import Skill, SkillResult, SkillManifest, SkillAction


KNOWLEDGE_FILE = Path(__file__).parent.parent / "data" / "knowledge_store.json"
MAX_ENTRIES = 500
DEFAULT_TTL_HOURS = 168  # 7 days


class KnowledgeCategory:
    """Categories for organizing shared knowledge."""
    STRATEGY = "strategy"          # Successful approaches to tasks
    WARNING = "warning"            # Things that failed or should be avoided
    OPTIMIZATION = "optimization"  # Cost/performance improvements
    CAPABILITY = "capability"      # Discovered capabilities or tools
    MARKET = "market"              # Revenue opportunities or pricing insights


class KnowledgeSharingSkill(Skill):
    """
    Enables agents to share discovered knowledge across the agent network.

    When one agent discovers that a certain approach works well (or fails),
    it publishes that knowledge. Other agents can query the knowledge store
    to benefit from collective experience.

    This transforms isolated replicas into a distributed learning system.
    """

    def __init__(self, credentials: Dict[str, str] = None):
        super().__init__(credentials)
        self._agent_id: str = "unknown"
        self._subscriptions: List[str] = list(vars(KnowledgeCategory).values())
        self._ensure_store()

    def set_agent_id(self, agent_id: str):
        """Set the identity of this agent for attribution."""
        self._agent_id = agent_id

    def _ensure_store(self):
        """Ensure the knowledge store file exists."""
        KNOWLEDGE_FILE.parent.mkdir(parents=True, exist_ok=True)
        if not KNOWLEDGE_FILE.exists():
            self._write_store({"entries": [], "metadata": {
                "created": datetime.now().isoformat(),
                "version": 1,
            }})

    def _read_store(self) -> Dict:
        """Read the knowledge store from disk."""
        try:
            with open(KNOWLEDGE_FILE, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return {"entries": [], "metadata": {"created": datetime.now().isoformat(), "version": 1}}

    def _write_store(self, store: Dict):
        """Write the knowledge store to disk."""
        KNOWLEDGE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(KNOWLEDGE_FILE, "w") as f:
            json.dump(store, f, indent=2, default=str)

    def _make_entry_id(self, content: str, category: str) -> str:
        """Generate a deterministic ID for deduplication."""
        raw = f"{category}:{content}".lower().strip()
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def _prune_expired(self, store: Dict) -> Dict:
        """Remove expired entries from the store."""
        now = datetime.now()
        active = []
        for entry in store.get("entries", []):
            expires = entry.get("expires_at")
            if expires:
                try:
                    if datetime.fromisoformat(expires) < now:
                        continue
                except (ValueError, TypeError):
                    pass
            active.append(entry)
        store["entries"] = active
        return store

    def _enforce_limits(self, store: Dict) -> Dict:
        """Keep store within size limits by removing oldest low-confidence entries."""
        entries = store.get("entries", [])
        if len(entries) > MAX_ENTRIES:
            # Sort by confidence (desc) then recency (desc), keep top MAX_ENTRIES
            entries.sort(key=lambda e: (
                e.get("confidence", 0.5),
                e.get("published_at", ""),
            ), reverse=True)
            store["entries"] = entries[:MAX_ENTRIES]
        return store

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="knowledge",
            name="Knowledge Sharing",
            version="1.0.0",
            category="replication",
            description="Share and query collective knowledge across agent network",
            actions=[
                SkillAction(
                    name="publish",
                    description="Publish a knowledge discovery for other agents",
                    parameters={
                        "content": {"type": "string", "required": True,
                                    "description": "The knowledge to share"},
                        "category": {"type": "string", "required": True,
                                     "description": "Category: strategy, warning, optimization, capability, market"},
                        "confidence": {"type": "float", "required": False,
                                       "description": "Confidence level 0.0-1.0 (default 0.7)"},
                        "tags": {"type": "list", "required": False,
                                 "description": "Tags for searchability"},
                        "context": {"type": "dict", "required": False,
                                    "description": "Additional context (task, skill_id, etc.)"},
                        "ttl_hours": {"type": "int", "required": False,
                                      "description": "Time-to-live in hours (default 168 = 7 days)"},
                    },
                    estimated_cost=0.0,
                ),
                SkillAction(
                    name="query",
                    description="Query the knowledge store for relevant knowledge",
                    parameters={
                        "category": {"type": "string", "required": False,
                                     "description": "Filter by category"},
                        "tags": {"type": "list", "required": False,
                                 "description": "Filter by tags (OR match)"},
                        "search": {"type": "string", "required": False,
                                   "description": "Text search in content"},
                        "min_confidence": {"type": "float", "required": False,
                                           "description": "Minimum confidence (default 0.5)"},
                        "limit": {"type": "int", "required": False,
                                  "description": "Max results (default 20)"},
                        "exclude_own": {"type": "bool", "required": False,
                                        "description": "Exclude own entries (default False)"},
                    },
                    estimated_cost=0.0,
                ),
                SkillAction(
                    name="endorse",
                    description="Endorse or dispute an existing knowledge entry",
                    parameters={
                        "entry_id": {"type": "string", "required": True,
                                     "description": "ID of the entry to endorse/dispute"},
                        "endorsement": {"type": "bool", "required": True,
                                        "description": "True to endorse, False to dispute"},
                        "reason": {"type": "string", "required": False,
                                   "description": "Reason for endorsement/dispute"},
                    },
                    estimated_cost=0.0,
                ),
                SkillAction(
                    name="subscribe",
                    description="Subscribe to knowledge categories for proactive updates",
                    parameters={
                        "categories": {"type": "list", "required": True,
                                       "description": "Categories to subscribe to"},
                    },
                    estimated_cost=0.0,
                ),
                SkillAction(
                    name="digest",
                    description="Get a summary digest of recent knowledge since last check",
                    parameters={
                        "since_hours": {"type": "int", "required": False,
                                        "description": "Hours to look back (default 24)"},
                    },
                    estimated_cost=0.0,
                ),
                SkillAction(
                    name="stats",
                    description="Get statistics about the knowledge store",
                    parameters={},
                    estimated_cost=0.0,
                ),
            ],
            required_credentials=[],
        )

    async def execute(self, action: str, params: Dict) -> SkillResult:
        handlers = {
            "publish": self._publish,
            "query": self._query,
            "endorse": self._endorse,
            "subscribe": self._subscribe,
            "digest": self._digest,
            "stats": self._stats,
        }
        handler = handlers.get(action)
        if not handler:
            return SkillResult(success=False, message=f"Unknown action: {action}")
        try:
            return await handler(params)
        except Exception as e:
            return SkillResult(success=False, message=f"Error in knowledge:{action}: {str(e)}")

    async def _publish(self, params: Dict) -> SkillResult:
        """Publish a knowledge discovery to the shared store."""
        content = params.get("content", "").strip()
        if not content:
            return SkillResult(success=False, message="Content is required")

        category = params.get("category", KnowledgeCategory.STRATEGY)
        valid_categories = [
            KnowledgeCategory.STRATEGY, KnowledgeCategory.WARNING,
            KnowledgeCategory.OPTIMIZATION, KnowledgeCategory.CAPABILITY,
            KnowledgeCategory.MARKET,
        ]
        if category not in valid_categories:
            return SkillResult(
                success=False,
                message=f"Invalid category '{category}'. Valid: {valid_categories}"
            )

        confidence = max(0.0, min(1.0, float(params.get("confidence", 0.7))))
        tags = params.get("tags", [])
        context = params.get("context", {})
        ttl_hours = int(params.get("ttl_hours", DEFAULT_TTL_HOURS))

        entry_id = self._make_entry_id(content, category)
        now = datetime.now()

        store = self._read_store()
        store = self._prune_expired(store)

        # Check for duplicate - update if exists with higher confidence
        existing_idx = None
        for i, entry in enumerate(store.get("entries", [])):
            if entry.get("id") == entry_id:
                existing_idx = i
                break

        entry = {
            "id": entry_id,
            "content": content,
            "category": category,
            "confidence": confidence,
            "tags": tags if isinstance(tags, list) else [tags],
            "context": context,
            "published_by": self._agent_id,
            "published_at": now.isoformat(),
            "expires_at": (now + timedelta(hours=ttl_hours)).isoformat(),
            "endorsements": [],
            "disputes": [],
            "supersedes": None,
        }

        if existing_idx is not None:
            old = store["entries"][existing_idx]
            # Only update if new confidence is >= old
            if confidence >= old.get("confidence", 0):
                entry["endorsements"] = old.get("endorsements", [])
                entry["disputes"] = old.get("disputes", [])
                store["entries"][existing_idx] = entry
                action_msg = "updated"
            else:
                return SkillResult(
                    success=True,
                    message=f"Existing entry {entry_id} has higher confidence, skipped",
                    data={"entry_id": entry_id, "action": "skipped"},
                )
        else:
            store["entries"].append(entry)
            action_msg = "published"

        store = self._enforce_limits(store)
        self._write_store(store)

        return SkillResult(
            success=True,
            message=f"Knowledge {action_msg}: {content[:80]}...",
            data={"entry_id": entry_id, "action": action_msg, "category": category},
        )

    async def _query(self, params: Dict) -> SkillResult:
        """Query the knowledge store with filters."""
        store = self._read_store()
        store = self._prune_expired(store)

        entries = store.get("entries", [])
        category = params.get("category")
        tags = params.get("tags", [])
        search = params.get("search", "").lower()
        min_confidence = float(params.get("min_confidence", 0.5))
        limit = int(params.get("limit", 20))
        exclude_own = params.get("exclude_own", False)

        results = []
        for entry in entries:
            # Filter by category
            if category and entry.get("category") != category:
                continue

            # Filter by confidence
            if entry.get("confidence", 0) < min_confidence:
                continue

            # Filter by tags (OR match)
            if tags:
                entry_tags = set(entry.get("tags", []))
                if not entry_tags.intersection(set(tags)):
                    continue

            # Text search
            if search and search not in entry.get("content", "").lower():
                continue

            # Exclude own entries
            if exclude_own and entry.get("published_by") == self._agent_id:
                continue

            # Calculate relevance score
            score = entry.get("confidence", 0.5)
            # Boost for endorsements
            score += len(entry.get("endorsements", [])) * 0.1
            # Penalize for disputes
            score -= len(entry.get("disputes", [])) * 0.15
            # Boost for recency
            try:
                age_hours = (datetime.now() - datetime.fromisoformat(
                    entry.get("published_at", datetime.now().isoformat())
                )).total_seconds() / 3600
                recency_boost = max(0, 0.2 - (age_hours / 168) * 0.2)
                score += recency_boost
            except (ValueError, TypeError):
                pass

            entry_copy = dict(entry)
            entry_copy["relevance_score"] = round(score, 3)
            results.append(entry_copy)

        # Sort by relevance score
        results.sort(key=lambda e: e.get("relevance_score", 0), reverse=True)
        results = results[:limit]

        return SkillResult(
            success=True,
            message=f"Found {len(results)} knowledge entries",
            data={"entries": results, "total_in_store": len(entries)},
        )

    async def _endorse(self, params: Dict) -> SkillResult:
        """Endorse or dispute a knowledge entry."""
        entry_id = params.get("entry_id", "")
        if not entry_id:
            return SkillResult(success=False, message="entry_id is required")

        is_endorsement = params.get("endorsement", True)
        reason = params.get("reason", "")

        store = self._read_store()

        found = False
        for entry in store.get("entries", []):
            if entry.get("id") == entry_id:
                found = True
                vote = {
                    "agent_id": self._agent_id,
                    "reason": reason,
                    "timestamp": datetime.now().isoformat(),
                }

                if is_endorsement:
                    # Remove any existing dispute from this agent
                    entry["disputes"] = [
                        d for d in entry.get("disputes", [])
                        if d.get("agent_id") != self._agent_id
                    ]
                    # Add endorsement (avoid duplicates)
                    entry.setdefault("endorsements", [])
                    entry["endorsements"] = [
                        e for e in entry["endorsements"]
                        if e.get("agent_id") != self._agent_id
                    ]
                    entry["endorsements"].append(vote)

                    # Boost confidence when endorsed
                    old_conf = entry.get("confidence", 0.5)
                    entry["confidence"] = min(1.0, old_conf + 0.05)
                else:
                    # Remove any existing endorsement from this agent
                    entry["endorsements"] = [
                        e for e in entry.get("endorsements", [])
                        if e.get("agent_id") != self._agent_id
                    ]
                    # Add dispute
                    entry.setdefault("disputes", [])
                    entry["disputes"] = [
                        d for d in entry["disputes"]
                        if d.get("agent_id") != self._agent_id
                    ]
                    entry["disputes"].append(vote)

                    # Reduce confidence when disputed
                    old_conf = entry.get("confidence", 0.5)
                    entry["confidence"] = max(0.0, old_conf - 0.1)

                break

        if not found:
            return SkillResult(success=False, message=f"Entry {entry_id} not found")

        self._write_store(store)

        action_word = "endorsed" if is_endorsement else "disputed"
        return SkillResult(
            success=True,
            message=f"Entry {entry_id} {action_word}",
            data={"entry_id": entry_id, "action": action_word},
        )

    async def _subscribe(self, params: Dict) -> SkillResult:
        """Subscribe to knowledge categories."""
        categories = params.get("categories", [])
        if not categories:
            return SkillResult(success=False, message="categories list is required")

        valid = {
            KnowledgeCategory.STRATEGY, KnowledgeCategory.WARNING,
            KnowledgeCategory.OPTIMIZATION, KnowledgeCategory.CAPABILITY,
            KnowledgeCategory.MARKET,
        }
        invalid = [c for c in categories if c not in valid]
        if invalid:
            return SkillResult(
                success=False,
                message=f"Invalid categories: {invalid}. Valid: {sorted(valid)}"
            )

        self._subscriptions = list(set(categories))
        return SkillResult(
            success=True,
            message=f"Subscribed to {len(self._subscriptions)} categories",
            data={"subscriptions": self._subscriptions},
        )

    async def _digest(self, params: Dict) -> SkillResult:
        """Get a summary of recent knowledge."""
        since_hours = int(params.get("since_hours", 24))
        cutoff = datetime.now() - timedelta(hours=since_hours)

        store = self._read_store()
        store = self._prune_expired(store)

        recent = []
        for entry in store.get("entries", []):
            try:
                pub_time = datetime.fromisoformat(entry.get("published_at", ""))
                if pub_time >= cutoff:
                    # Only include subscribed categories
                    if entry.get("category") in self._subscriptions:
                        recent.append(entry)
            except (ValueError, TypeError):
                continue

        # Group by category
        by_category = {}
        for entry in recent:
            cat = entry.get("category", "unknown")
            by_category.setdefault(cat, []).append(entry)

        # Build digest
        digest_items = []
        for cat, entries in sorted(by_category.items()):
            for entry in sorted(entries, key=lambda e: e.get("confidence", 0), reverse=True):
                digest_items.append({
                    "category": cat,
                    "content": entry.get("content", ""),
                    "confidence": entry.get("confidence", 0),
                    "published_by": entry.get("published_by", "unknown"),
                    "endorsements": len(entry.get("endorsements", [])),
                    "disputes": len(entry.get("disputes", [])),
                })

        return SkillResult(
            success=True,
            message=f"Digest: {len(digest_items)} items from last {since_hours}h",
            data={
                "items": digest_items,
                "categories": {cat: len(entries) for cat, entries in by_category.items()},
                "total_recent": len(recent),
                "since_hours": since_hours,
            },
        )

    async def _stats(self, params: Dict) -> SkillResult:
        """Get statistics about the knowledge store."""
        store = self._read_store()
        entries = store.get("entries", [])

        by_category = {}
        by_agent = {}
        total_endorsements = 0
        total_disputes = 0
        avg_confidence = 0.0

        for entry in entries:
            cat = entry.get("category", "unknown")
            agent = entry.get("published_by", "unknown")
            by_category[cat] = by_category.get(cat, 0) + 1
            by_agent[agent] = by_agent.get(agent, 0) + 1
            total_endorsements += len(entry.get("endorsements", []))
            total_disputes += len(entry.get("disputes", []))
            avg_confidence += entry.get("confidence", 0)

        if entries:
            avg_confidence /= len(entries)

        return SkillResult(
            success=True,
            message=f"Knowledge store: {len(entries)} entries from {len(by_agent)} agents",
            data={
                "total_entries": len(entries),
                "by_category": by_category,
                "by_agent": by_agent,
                "total_endorsements": total_endorsements,
                "total_disputes": total_disputes,
                "avg_confidence": round(avg_confidence, 3),
                "subscriptions": self._subscriptions,
            },
        )
