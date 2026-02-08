#!/usr/bin/env python3
"""
NaturalLanguageRouter - Intelligent task routing from natural language to skills.

This is the critical bridge between external users (via ServiceAPI) and the
agent's 50+ internal skills. Without this, callers must know exact skill IDs
and action names. With it, a user can say "review this code for security issues"
and the router figures out to call code_review:security_scan.

Architecture:
1. BUILD CATALOG: On initialization, scans all registered skills and builds
   a searchable catalog of (skill_id, action, description, keywords).
2. MATCH QUERY: Given a natural language task description, scores each
   skill+action pair using keyword overlap, TF-IDF-like weighting, and
   category matching to find the best matches.
3. EXECUTE ROUTED: Optionally executes the top match directly, passing
   through the extracted parameters.
4. LEARN FROM OUTCOMES: Tracks which routings succeeded/failed and adjusts
   match weights over time (persistent across sessions).

Serves THREE pillars:
- Revenue Generation: External users can submit tasks without knowing internals
- Goal Setting: High-level goals get decomposed into skill actions
- Self-Improvement: Router learns better matchings from outcomes

No external dependencies - pure Python keyword matching and scoring.
"""

import json
import re
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from .base import Skill, SkillManifest, SkillAction, SkillResult


ROUTER_DATA = Path(__file__).parent.parent / "data" / "nl_router.json"
MAX_HISTORY = 500

# Category keywords help boost matches when the task mentions a domain
CATEGORY_KEYWORDS = {
    "dev": ["code", "program", "debug", "fix", "build", "test", "develop", "software", "bug", "error", "compile", "refactor"],
    "content": ["write", "article", "blog", "copy", "text", "content", "draft", "edit", "proofread"],
    "social": ["tweet", "post", "social", "share", "publish", "twitter", "x.com"],
    "payment": ["pay", "invoice", "bill", "charge", "revenue", "money", "price", "cost", "earning"],
    "domain": ["domain", "dns", "website", "register", "namecheap", "url"],
    "email": ["email", "mail", "send", "message", "notify", "notification"],
    "deploy": ["deploy", "host", "server", "cloud", "docker", "container", "kubernetes", "vercel"],
    "data": ["data", "transform", "convert", "csv", "json", "xml", "parse", "analyze", "analysis"],
    "security": ["security", "scan", "vulnerability", "secret", "credential", "password", "encrypt", "auth"],
    "monitor": ["monitor", "health", "status", "metrics", "alert", "watch", "track"],
    "workflow": ["workflow", "pipeline", "automate", "schedule", "chain", "sequence", "orchestrate"],
    "review": ["review", "audit", "inspect", "quality", "check", "lint", "analyze"],
    "file": ["file", "read", "write", "directory", "folder", "path", "filesystem"],
    "shell": ["shell", "command", "terminal", "bash", "execute", "run", "script"],
    "git": ["git", "github", "commit", "pull", "push", "branch", "merge", "pr", "issue", "repository"],
    "crypto": ["crypto", "blockchain", "wallet", "token", "ethereum", "web3", "transaction"],
    "memory": ["memory", "remember", "recall", "store", "retrieve", "knowledge", "learn"],
    "strategy": ["strategy", "plan", "goal", "prioritize", "assess", "recommend", "diagnose"],
}

# Stop words to ignore in matching
STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "through", "during",
    "before", "after", "above", "below", "between", "out", "off", "over",
    "under", "again", "further", "then", "once", "here", "there", "when",
    "where", "why", "how", "all", "each", "every", "both", "few", "more",
    "most", "other", "some", "such", "no", "nor", "not", "only", "own",
    "same", "so", "than", "too", "very", "just", "because", "but", "and",
    "or", "if", "while", "about", "up", "it", "its", "this", "that",
    "these", "those", "i", "me", "my", "we", "our", "you", "your", "he",
    "him", "his", "she", "her", "they", "them", "their", "what", "which",
    "who", "whom", "please", "want", "need", "help", "like",
}


def _tokenize(text: str) -> List[str]:
    """Tokenize text into lowercase words, filtering stop words."""
    words = re.findall(r'[a-z0-9_]+', text.lower())
    return [w for w in words if w not in STOP_WORDS and len(w) > 1]


def _extract_keywords_from_desc(description: str) -> List[str]:
    """Extract meaningful keywords from a skill/action description."""
    return _tokenize(description)


class CatalogEntry:
    """A single routable skill+action with matching metadata."""

    __slots__ = ['skill_id', 'action', 'description', 'keywords',
                 'category', 'success_count', 'fail_count', 'weight_boost']

    def __init__(self, skill_id: str, action: str, description: str,
                 category: str, keywords: Optional[List[str]] = None):
        self.skill_id = skill_id
        self.action = action
        self.description = description
        self.category = category
        self.keywords = keywords or _extract_keywords_from_desc(description)
        self.success_count = 0
        self.fail_count = 0
        self.weight_boost = 0.0  # Learned adjustment

    @property
    def success_rate(self) -> float:
        total = self.success_count + self.fail_count
        if total == 0:
            return 0.5  # No data = neutral
        return self.success_count / total

    def to_dict(self) -> Dict:
        return {
            "skill_id": self.skill_id,
            "action": self.action,
            "description": self.description,
            "category": self.category,
            "keywords": self.keywords,
            "success_count": self.success_count,
            "fail_count": self.fail_count,
            "weight_boost": self.weight_boost,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "CatalogEntry":
        entry = cls(
            skill_id=d["skill_id"],
            action=d["action"],
            description=d["description"],
            category=d.get("category", ""),
            keywords=d.get("keywords"),
        )
        entry.success_count = d.get("success_count", 0)
        entry.fail_count = d.get("fail_count", 0)
        entry.weight_boost = d.get("weight_boost", 0.0)
        return entry


class NaturalLanguageRouter(Skill):
    """
    Routes natural language task descriptions to the best matching skill+action.

    Uses keyword matching with TF-IDF-like scoring, category boosting,
    and learned weight adjustments from past routing outcomes.
    """

    def __init__(self, credentials: Dict[str, str] = None):
        super().__init__(credentials)
        self._catalog: List[CatalogEntry] = []
        self._idf: Dict[str, float] = {}  # Inverse doc frequency for keywords
        self._history: List[Dict] = []
        self._ensure_data()

    def _ensure_data(self):
        ROUTER_DATA.parent.mkdir(parents=True, exist_ok=True)
        if not ROUTER_DATA.exists():
            self._save_state({"catalog": [], "history": [], "idf": {}})

    def _load_state(self) -> Dict:
        try:
            with open(ROUTER_DATA, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return {"catalog": [], "history": [], "idf": {}}

    def _save_state(self, state: Dict):
        with open(ROUTER_DATA, "w") as f:
            json.dump(state, f, indent=2)

    def _persist(self):
        """Save current state to disk."""
        state = {
            "catalog": [e.to_dict() for e in self._catalog],
            "history": self._history[-MAX_HISTORY:],
            "idf": self._idf,
        }
        self._save_state(state)

    def _restore(self):
        """Restore learned weights from disk."""
        state = self._load_state()
        stored_catalog = {
            f"{e['skill_id']}:{e['action']}": e
            for e in state.get("catalog", [])
        }
        # Merge stored weights into current catalog
        for entry in self._catalog:
            key = f"{entry.skill_id}:{entry.action}"
            if key in stored_catalog:
                stored = stored_catalog[key]
                entry.success_count = stored.get("success_count", 0)
                entry.fail_count = stored.get("fail_count", 0)
                entry.weight_boost = stored.get("weight_boost", 0.0)
        self._history = state.get("history", [])[-MAX_HISTORY:]
        self._idf = state.get("idf", {})

    def build_catalog(self):
        """
        Build the routing catalog from all registered skills.

        Call this after the skill context is set (i.e., after all skills
        are registered with the agent).
        """
        if not self.context:
            return

        self._catalog.clear()
        doc_freq: Dict[str, int] = defaultdict(int)
        total_entries = 0

        for skill_id in self.context.list_skills():
            skill = self.context.get_skill(skill_id)
            if skill is None or skill_id == "nl_router":
                continue  # Don't route to self

            manifest = skill.manifest
            category = manifest.category

            for action_def in skill.get_actions():
                desc = f"{manifest.name} {action_def.name}: {action_def.description}"
                keywords = _extract_keywords_from_desc(desc)

                # Add category-specific boost keywords
                if category in CATEGORY_KEYWORDS:
                    keywords.extend(CATEGORY_KEYWORDS[category])

                # Deduplicate
                keywords = list(set(keywords))

                entry = CatalogEntry(
                    skill_id=skill_id,
                    action=action_def.name,
                    description=action_def.description,
                    category=category,
                    keywords=keywords,
                )
                self._catalog.append(entry)
                total_entries += 1

                # Track document frequency for IDF
                for kw in set(keywords):
                    doc_freq[kw] += 1

        # Compute IDF scores
        if total_entries > 0:
            self._idf = {
                kw: math.log(total_entries / (1 + freq))
                for kw, freq in doc_freq.items()
            }

        # Restore learned weights from previous sessions
        self._restore()

    def _score_entry(self, query_tokens: List[str], entry: CatalogEntry) -> float:
        """
        Score a catalog entry against query tokens.

        Scoring components:
        1. Keyword overlap (TF-IDF weighted)
        2. Category boost (if query mentions category terms)
        3. Success rate boost (learned from outcomes)
        4. Weight boost (manually adjusted or learned)
        """
        if not entry.keywords or not query_tokens:
            return 0.0

        entry_kw_set = set(entry.keywords)
        query_set = set(query_tokens)

        # 1. TF-IDF weighted keyword overlap
        overlap = query_set & entry_kw_set
        if not overlap:
            return 0.0

        tfidf_score = sum(self._idf.get(kw, 1.0) for kw in overlap)

        # Normalize by query length to avoid bias toward long queries
        tfidf_score /= max(len(query_tokens), 1)

        # 2. Category boost: +0.5 if the query matches category keywords
        category_boost = 0.0
        if entry.category in CATEGORY_KEYWORDS:
            cat_kws = set(CATEGORY_KEYWORDS[entry.category])
            cat_overlap = query_set & cat_kws
            if cat_overlap:
                category_boost = 0.5 * len(cat_overlap) / max(len(cat_kws), 1)

        # 3. Success rate boost: up to +0.3 for consistently successful routes
        total_attempts = entry.success_count + entry.fail_count
        success_boost = 0.0
        if total_attempts >= 3:
            success_boost = 0.3 * (entry.success_rate - 0.5)  # Range: -0.15 to +0.15

        # 4. Learned weight boost
        weight_boost = entry.weight_boost

        return tfidf_score + category_boost + success_boost + weight_boost

    def route(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Route a natural language query to the best matching skill+action pairs.

        Args:
            query: Natural language task description
            top_k: Number of top matches to return

        Returns:
            List of matches sorted by score, each with:
            - skill_id, action, description, score, category
        """
        if not self._catalog:
            self.build_catalog()

        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        scored = []
        for entry in self._catalog:
            score = self._score_entry(query_tokens, entry)
            if score > 0:
                scored.append((score, entry))

        scored.sort(key=lambda x: -x[0])

        return [
            {
                "skill_id": entry.skill_id,
                "action": entry.action,
                "description": entry.description,
                "category": entry.category,
                "score": round(score, 4),
                "success_rate": round(entry.success_rate, 2),
            }
            for score, entry in scored[:top_k]
        ]

    def record_outcome(self, skill_id: str, action: str, success: bool, query: str = ""):
        """Record whether a routed execution succeeded or failed."""
        key = f"{skill_id}:{action}"
        for entry in self._catalog:
            if f"{entry.skill_id}:{entry.action}" == key:
                if success:
                    entry.success_count += 1
                    # Small positive weight boost for success
                    entry.weight_boost = min(entry.weight_boost + 0.02, 1.0)
                else:
                    entry.fail_count += 1
                    # Small negative weight for failure
                    entry.weight_boost = max(entry.weight_boost - 0.03, -1.0)
                break

        self._history.append({
            "query": query,
            "skill_id": skill_id,
            "action": action,
            "success": success,
            "timestamp": datetime.now().isoformat(),
        })
        self._persist()

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="nl_router",
            name="Natural Language Router",
            version="1.0.0",
            category="meta",
            description="Routes natural language task descriptions to the best matching skill and action",
            actions=[
                SkillAction(
                    name="route",
                    description="Find the best skill+action matches for a natural language task description",
                    parameters={
                        "query": {"type": "string", "required": True, "description": "Natural language task description"},
                        "top_k": {"type": "integer", "required": False, "description": "Number of top matches (default 5)"},
                    },
                    estimated_cost=0,
                    estimated_duration_seconds=1,
                    success_probability=0.9,
                ),
                SkillAction(
                    name="route_and_execute",
                    description="Route a task to the best skill and execute it immediately",
                    parameters={
                        "query": {"type": "string", "required": True, "description": "Natural language task description"},
                        "params": {"type": "object", "required": False, "description": "Additional parameters to pass to the matched skill"},
                    },
                    estimated_cost=0,
                    estimated_duration_seconds=10,
                    success_probability=0.7,
                ),
                SkillAction(
                    name="record_outcome",
                    description="Record whether a routed execution succeeded or failed (for learning)",
                    parameters={
                        "skill_id": {"type": "string", "required": True, "description": "Skill that was executed"},
                        "action": {"type": "string", "required": True, "description": "Action that was executed"},
                        "success": {"type": "boolean", "required": True, "description": "Whether execution succeeded"},
                        "query": {"type": "string", "required": False, "description": "Original query (for history)"},
                    },
                    estimated_cost=0,
                    estimated_duration_seconds=1,
                    success_probability=1.0,
                ),
                SkillAction(
                    name="catalog",
                    description="List the full routing catalog (all routable skill+action pairs)",
                    parameters={
                        "category": {"type": "string", "required": False, "description": "Filter by category"},
                    },
                    estimated_cost=0,
                    estimated_duration_seconds=1,
                    success_probability=1.0,
                ),
                SkillAction(
                    name="stats",
                    description="Show routing statistics: total routes, success rates, top performers",
                    parameters={},
                    estimated_cost=0,
                    estimated_duration_seconds=1,
                    success_probability=1.0,
                ),
                SkillAction(
                    name="rebuild",
                    description="Rebuild the routing catalog from current skills",
                    parameters={},
                    estimated_cost=0,
                    estimated_duration_seconds=2,
                    success_probability=1.0,
                ),
            ],
            required_credentials=[],
        )

    async def execute(self, action: str, params: Dict) -> SkillResult:
        """Execute a router action."""
        actions = {
            "route": self._route,
            "route_and_execute": self._route_and_execute,
            "record_outcome": self._record_outcome,
            "catalog": self._catalog_action,
            "stats": self._stats,
            "rebuild": self._rebuild,
        }
        handler = actions.get(action)
        if not handler:
            return SkillResult(
                success=False,
                message=f"Unknown action '{action}'. Available: {list(actions.keys())}",
            )
        return await handler(params)

    async def _route(self, params: Dict) -> SkillResult:
        """Route a natural language query to matching skills."""
        query = params.get("query", "")
        if not query:
            return SkillResult(success=False, message="Parameter 'query' is required")

        top_k = params.get("top_k", 5)
        matches = self.route(query, top_k=top_k)

        if not matches:
            return SkillResult(
                success=True,
                message="No matching skills found for the query",
                data={"matches": [], "query": query},
            )

        top = matches[0]
        return SkillResult(
            success=True,
            message=f"Best match: {top['skill_id']}:{top['action']} (score={top['score']})",
            data={"matches": matches, "query": query, "best": top},
        )

    async def _route_and_execute(self, params: Dict) -> SkillResult:
        """Route a query and execute the best match."""
        query = params.get("query", "")
        if not query:
            return SkillResult(success=False, message="Parameter 'query' is required")

        extra_params = params.get("params", {})

        matches = self.route(query, top_k=3)
        if not matches:
            return SkillResult(
                success=False,
                message="No matching skill found for the query",
                data={"query": query},
            )

        best = matches[0]
        skill_id = best["skill_id"]
        action = best["action"]

        if not self.context:
            return SkillResult(
                success=False,
                message="No skill context available - cannot execute routed skill",
                data={"best_match": best},
            )

        # Execute via context
        result = await self.context.call_skill(skill_id, action, extra_params)

        # Record outcome for learning
        self.record_outcome(skill_id, action, result.success, query)

        return SkillResult(
            success=result.success,
            message=f"Routed to {skill_id}:{action} -> {result.message}",
            data={
                "routed_to": {"skill_id": skill_id, "action": action},
                "routing_score": best["score"],
                "alternatives": matches[1:],
                "result": result.data,
            },
            cost=result.cost,
            revenue=result.revenue,
        )

    async def _record_outcome(self, params: Dict) -> SkillResult:
        """Record routing outcome for learning."""
        skill_id = params.get("skill_id", "")
        action = params.get("action", "")
        success = params.get("success", False)
        query = params.get("query", "")

        if not skill_id or not action:
            return SkillResult(
                success=False,
                message="Parameters 'skill_id' and 'action' are required",
            )

        self.record_outcome(skill_id, action, success, query)
        return SkillResult(
            success=True,
            message=f"Recorded {'success' if success else 'failure'} for {skill_id}:{action}",
        )

    async def _catalog_action(self, params: Dict) -> SkillResult:
        """List the routing catalog."""
        if not self._catalog:
            self.build_catalog()

        category = params.get("category", "")
        entries = self._catalog
        if category:
            entries = [e for e in entries if e.category == category]

        catalog_data = [e.to_dict() for e in entries]
        categories = list(set(e.category for e in self._catalog))

        return SkillResult(
            success=True,
            message=f"Catalog: {len(catalog_data)} entries across {len(categories)} categories",
            data={
                "entries": catalog_data,
                "total": len(catalog_data),
                "categories": sorted(categories),
            },
        )

    async def _stats(self, params: Dict) -> SkillResult:
        """Show routing statistics."""
        total_routes = len(self._history)
        if total_routes == 0:
            return SkillResult(
                success=True,
                message="No routing history yet",
                data={"total_routes": 0},
            )

        successes = sum(1 for h in self._history if h.get("success"))
        failures = total_routes - successes

        # Top performers (by success count)
        by_route: Dict[str, Dict] = {}
        for h in self._history:
            key = f"{h['skill_id']}:{h['action']}"
            if key not in by_route:
                by_route[key] = {"successes": 0, "failures": 0, "total": 0}
            by_route[key]["total"] += 1
            if h.get("success"):
                by_route[key]["successes"] += 1
            else:
                by_route[key]["failures"] += 1

        top_routes = sorted(
            by_route.items(),
            key=lambda x: x[1]["successes"],
            reverse=True,
        )[:10]

        return SkillResult(
            success=True,
            message=f"Total routes: {total_routes}, Success rate: {successes/total_routes:.1%}",
            data={
                "total_routes": total_routes,
                "successes": successes,
                "failures": failures,
                "success_rate": round(successes / total_routes, 3),
                "top_routes": [
                    {"route": k, **v} for k, v in top_routes
                ],
                "catalog_size": len(self._catalog),
            },
        )

    async def _rebuild(self, params: Dict) -> SkillResult:
        """Rebuild the catalog from current skills."""
        old_size = len(self._catalog)
        self.build_catalog()
        new_size = len(self._catalog)

        return SkillResult(
            success=True,
            message=f"Catalog rebuilt: {old_size} -> {new_size} entries",
            data={"old_size": old_size, "new_size": new_size},
        )

    async def initialize(self) -> bool:
        """Initialize and build catalog."""
        self.initialized = True
        # Catalog will be built on first route or explicit rebuild
        return True
