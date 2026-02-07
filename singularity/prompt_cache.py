"""
PromptCache - LLM response caching to reduce API costs.

Caches LLM responses by prompt hash to avoid duplicate API calls.
Supports TTL-based expiration and persistent file-backed storage.

This directly improves agent survival by reducing API spend - the
agent's primary burn rate.

Pillar: Self-Improvement + Revenue (cost reduction)
"""

import hashlib
import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Dict, Any


@dataclass
class CacheEntry:
    """A cached LLM response."""
    prompt_hash: str
    response_text: str
    model: str
    provider: str
    created_at: float
    ttl_seconds: float
    hit_count: int = 0
    estimated_cost_saved: float = 0.0

    def is_expired(self) -> bool:
        """Check if this cache entry has expired."""
        if self.ttl_seconds <= 0:
            return False  # No expiration
        return (time.time() - self.created_at) > self.ttl_seconds

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "CacheEntry":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class CacheStats:
    """Cache performance statistics."""
    total_hits: int = 0
    total_misses: int = 0
    total_cost_saved: float = 0.0
    total_entries: int = 0
    evictions: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.total_hits + self.total_misses
        return self.total_hits / total if total > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            "total_hits": self.total_hits,
            "total_misses": self.total_misses,
            "total_cost_saved": self.total_cost_saved,
            "total_entries": self.total_entries,
            "evictions": self.evictions,
            "hit_rate": round(self.hit_rate, 4),
        }


class PromptCache:
    """
    LLM response cache with file-backed persistence.

    Caches responses keyed by a hash of (system_prompt, user_prompt, model).
    Saves money by returning cached responses for identical prompts.

    Features:
    - Exact-match caching via SHA-256 hash
    - TTL-based expiration (default: 1 hour)
    - LRU eviction when cache is full
    - File-backed persistence across agent restarts
    - Cost savings tracking

    Usage:
        cache = PromptCache(cache_dir="./cache", max_entries=500)
        
        # Check cache before calling LLM
        cached = cache.get(system_prompt, user_prompt, model)
        if cached:
            response_text = cached.response_text  # Free!
        else:
            response_text = call_llm(...)  # Costs money
            cache.put(system_prompt, user_prompt, model, response_text, 
                     provider="anthropic", estimated_cost=0.003)
    """

    def __init__(
        self,
        cache_dir: str = "./data/prompt_cache",
        max_entries: int = 1000,
        default_ttl: float = 3600.0,  # 1 hour default
        enabled: bool = True,
    ):
        self.cache_dir = Path(cache_dir)
        self.max_entries = max_entries
        self.default_ttl = default_ttl
        self.enabled = enabled
        self._cache: Dict[str, CacheEntry] = {}
        self._stats = CacheStats()
        self._loaded = False

    def _ensure_loaded(self):
        """Lazy-load cache from disk on first access."""
        if self._loaded:
            return
        self._loaded = True
        self._load_from_disk()

    @staticmethod
    def _make_key(system_prompt: str, user_prompt: str, model: str) -> str:
        """Create a deterministic cache key from prompt components."""
        content = f"{system_prompt}\x00{user_prompt}\x00{model}"
        return hashlib.sha256(content.encode()).hexdigest()[:32]

    def get(
        self, system_prompt: str, user_prompt: str, model: str
    ) -> Optional[CacheEntry]:
        """
        Look up a cached response.

        Returns CacheEntry if found and not expired, None otherwise.
        """
        if not self.enabled:
            return None

        self._ensure_loaded()
        key = self._make_key(system_prompt, user_prompt, model)
        entry = self._cache.get(key)

        if entry is None:
            self._stats.total_misses += 1
            return None

        if entry.is_expired():
            del self._cache[key]
            self._stats.total_misses += 1
            self._stats.evictions += 1
            return None

        # Cache hit
        entry.hit_count += 1
        self._stats.total_hits += 1
        self._stats.total_cost_saved += entry.estimated_cost_saved
        return entry

    def put(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str,
        response_text: str,
        provider: str = "",
        estimated_cost: float = 0.0,
        ttl: Optional[float] = None,
    ) -> str:
        """
        Cache an LLM response.

        Returns the cache key.
        """
        if not self.enabled:
            return ""

        self._ensure_loaded()
        key = self._make_key(system_prompt, user_prompt, model)

        # Evict if at capacity (remove oldest entry)
        if len(self._cache) >= self.max_entries and key not in self._cache:
            self._evict_oldest()

        entry = CacheEntry(
            prompt_hash=key,
            response_text=response_text,
            model=model,
            provider=provider,
            created_at=time.time(),
            ttl_seconds=ttl if ttl is not None else self.default_ttl,
            estimated_cost_saved=estimated_cost,
        )
        self._cache[key] = entry
        self._stats.total_entries = len(self._cache)
        return key

    def invalidate(self, system_prompt: str, user_prompt: str, model: str) -> bool:
        """Remove a specific entry from cache. Returns True if found."""
        self._ensure_loaded()
        key = self._make_key(system_prompt, user_prompt, model)
        if key in self._cache:
            del self._cache[key]
            self._stats.total_entries = len(self._cache)
            return True
        return False

    def clear(self) -> int:
        """Clear all cache entries. Returns number of entries removed."""
        self._ensure_loaded()
        count = len(self._cache)
        self._cache.clear()
        self._stats.total_entries = 0
        return count

    def get_stats(self) -> dict:
        """Get cache performance statistics."""
        self._ensure_loaded()
        self._stats.total_entries = len(self._cache)
        return self._stats.to_dict()

    def save_to_disk(self) -> bool:
        """Persist cache to disk."""
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file = self.cache_dir / "cache.json"
            stats_file = self.cache_dir / "stats.json"

            # Save cache entries
            cache_data = {
                key: entry.to_dict() for key, entry in self._cache.items()
            }
            cache_file.write_text(json.dumps(cache_data, indent=2))

            # Save stats
            stats_file.write_text(json.dumps(self._stats.to_dict(), indent=2))

            return True
        except Exception:
            return False

    def _load_from_disk(self):
        """Load cache from disk."""
        cache_file = self.cache_dir / "cache.json"
        stats_file = self.cache_dir / "stats.json"

        if cache_file.exists():
            try:
                data = json.loads(cache_file.read_text())
                for key, entry_data in data.items():
                    entry = CacheEntry.from_dict(entry_data)
                    if not entry.is_expired():
                        self._cache[key] = entry
            except (json.JSONDecodeError, Exception):
                pass

        if stats_file.exists():
            try:
                stats_data = json.loads(stats_file.read_text())
                self._stats.total_hits = stats_data.get("total_hits", 0)
                self._stats.total_misses = stats_data.get("total_misses", 0)
                self._stats.total_cost_saved = stats_data.get("total_cost_saved", 0.0)
                self._stats.evictions = stats_data.get("evictions", 0)
            except (json.JSONDecodeError, Exception):
                pass

        self._stats.total_entries = len(self._cache)

    def _evict_oldest(self):
        """Evict the oldest cache entry."""
        if not self._cache:
            return
        oldest_key = min(self._cache, key=lambda k: self._cache[k].created_at)
        del self._cache[oldest_key]
        self._stats.evictions += 1

    def cleanup_expired(self) -> int:
        """Remove all expired entries. Returns count removed."""
        self._ensure_loaded()
        expired_keys = [
            key for key, entry in self._cache.items() if entry.is_expired()
        ]
        for key in expired_keys:
            del self._cache[key]
        self._stats.evictions += len(expired_keys)
        self._stats.total_entries = len(self._cache)
        return len(expired_keys)


def wrap_cognition_with_cache(cognition_engine, cache: PromptCache):
    """
    Monkey-patch a CognitionEngine to use prompt caching.

    This wraps the think() method to check cache before calling the LLM.
    Saves significant API costs for repeated or similar prompts.

    Usage:
        from singularity.prompt_cache import PromptCache, wrap_cognition_with_cache
        
        cache = PromptCache()
        wrap_cognition_with_cache(agent.cognition, cache)
        # Now agent.cognition.think() will use cache automatically
    """
    from singularity.cognition import Decision, Action, TokenUsage, calculate_api_cost

    original_think = cognition_engine.think

    async def cached_think(state):
        # Build the same prompts that think() would build
        system_prompt = cognition_engine.get_system_prompt()
        tools_text = "\n".join([
            f"- {t['name']}: {t['description']}" for t in state.tools
        ])
        recent_text = ""
        if state.recent_actions:
            recent_text = "\nRecent actions:\n" + "\n".join([
                f"- {a['tool']}: {a.get('result', {}).get('status', 'unknown')}"
                for a in state.recent_actions[-5:]
            ])

        user_prompt = f"""Current state:
- Balance: ${state.balance:.4f}
- Burn rate: ${state.burn_rate:.6f}/cycle
- Runway: {state.runway_hours:.1f} hours
- Cycle: {state.cycle}

Available tools:
{tools_text}
{recent_text}

{state.project_context}

What action should you take? Respond with JSON: {{"tool": "skill:action", "params": {{}}, "reasoning": "why"}}"""

        model = cognition_engine.llm_model

        # Check cache
        cached = cache.get(system_prompt, user_prompt, model)
        if cached:
            print(f"[CACHE] HIT - saved ~${cached.estimated_cost_saved:.6f}")
            action = cognition_engine._parse_action(cached.response_text)
            return Decision(
                action=action,
                reasoning=f"[CACHED] {action.reasoning}",
                token_usage=TokenUsage(),  # No tokens used
                api_cost_usd=0.0,  # No API cost
            )

        # Cache miss - call original think()
        decision = await original_think(state)

        # Estimate cost for future cache hits
        estimated_cost = decision.api_cost_usd

        # Cache the response (we need to reconstruct what the LLM said)
        # Store the action as JSON so _parse_action can parse it back
        response_json = json.dumps({
            "tool": decision.action.tool,
            "params": decision.action.params,
            "reasoning": decision.action.reasoning,
        })
        cache.put(
            system_prompt,
            user_prompt,
            model,
            response_json,
            provider=cognition_engine.llm_type,
            estimated_cost=estimated_cost,
        )

        return decision

    cognition_engine.think = cached_think
    cognition_engine._prompt_cache = cache
    print("[CACHE] Prompt caching enabled")
