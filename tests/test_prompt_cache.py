"""Tests for PromptCache."""

import json
import time
import tempfile
import pytest
from singularity.prompt_cache import PromptCache, CacheEntry, CacheStats


@pytest.fixture
def cache(tmp_path):
    return PromptCache(cache_dir=str(tmp_path / "cache"), max_entries=10, default_ttl=60.0)


class TestCacheEntry:
    def test_not_expired(self):
        entry = CacheEntry("abc", "response", "model", "prov", time.time(), 60.0)
        assert not entry.is_expired()

    def test_expired(self):
        entry = CacheEntry("abc", "response", "model", "prov", time.time() - 120, 60.0)
        assert entry.is_expired()

    def test_no_ttl_never_expires(self):
        entry = CacheEntry("abc", "response", "model", "prov", time.time() - 999999, 0)
        assert not entry.is_expired()

    def test_roundtrip(self):
        entry = CacheEntry("abc", "response", "model", "prov", 1000.0, 60.0, 5, 0.01)
        d = entry.to_dict()
        restored = CacheEntry.from_dict(d)
        assert restored.prompt_hash == "abc"
        assert restored.hit_count == 5


class TestCacheStats:
    def test_hit_rate_empty(self):
        stats = CacheStats()
        assert stats.hit_rate == 0.0

    def test_hit_rate(self):
        stats = CacheStats(total_hits=3, total_misses=7)
        assert stats.hit_rate == 0.3

    def test_to_dict(self):
        stats = CacheStats(total_hits=10, total_misses=5)
        d = stats.to_dict()
        assert d["hit_rate"] == 0.6667


class TestPromptCache:
    def test_miss_returns_none(self, cache):
        assert cache.get("sys", "user", "model") is None

    def test_put_and_get(self, cache):
        cache.put("sys", "user", "model", "response", estimated_cost=0.01)
        entry = cache.get("sys", "user", "model")
        assert entry is not None
        assert entry.response_text == "response"

    def test_different_prompts_different_keys(self, cache):
        cache.put("sys", "user1", "model", "resp1")
        cache.put("sys", "user2", "model", "resp2")
        assert cache.get("sys", "user1", "model").response_text == "resp1"
        assert cache.get("sys", "user2", "model").response_text == "resp2"

    def test_different_models_different_keys(self, cache):
        cache.put("sys", "user", "gpt-4", "resp_gpt")
        cache.put("sys", "user", "claude", "resp_claude")
        assert cache.get("sys", "user", "gpt-4").response_text == "resp_gpt"
        assert cache.get("sys", "user", "claude").response_text == "resp_claude"

    def test_expired_entry_returns_none(self, cache):
        cache.put("sys", "user", "model", "response", ttl=0.01)
        time.sleep(0.02)
        assert cache.get("sys", "user", "model") is None

    def test_stats_tracking(self, cache):
        cache.put("sys", "user", "model", "response", estimated_cost=0.005)
        cache.get("sys", "user", "model")  # hit
        cache.get("sys", "other", "model")  # miss
        stats = cache.get_stats()
        assert stats["total_hits"] == 1
        assert stats["total_misses"] == 1
        assert stats["total_cost_saved"] == 0.005

    def test_invalidate(self, cache):
        cache.put("sys", "user", "model", "response")
        assert cache.invalidate("sys", "user", "model")
        assert cache.get("sys", "user", "model") is None
        assert not cache.invalidate("sys", "user", "model")

    def test_clear(self, cache):
        cache.put("sys", "u1", "m", "r1")
        cache.put("sys", "u2", "m", "r2")
        removed = cache.clear()
        assert removed == 2
        assert cache.get("sys", "u1", "m") is None

    def test_eviction_at_capacity(self, cache):
        # Fill cache (max_entries=10)
        for i in range(10):
            cache.put("sys", f"user_{i}", "model", f"resp_{i}")
        # Add one more - should evict oldest
        cache.put("sys", "user_new", "model", "resp_new")
        stats = cache.get_stats()
        assert stats["total_entries"] == 10
        assert stats["evictions"] >= 1

    def test_persistence(self, tmp_path):
        cache_dir = str(tmp_path / "persist")
        c1 = PromptCache(cache_dir=cache_dir, default_ttl=3600)
        c1.put("sys", "user", "model", "response", estimated_cost=0.01)
        c1.get("sys", "user", "model")  # record a hit
        c1.save_to_disk()

        c2 = PromptCache(cache_dir=cache_dir, default_ttl=3600)
        entry = c2.get("sys", "user", "model")
        assert entry is not None
        assert entry.response_text == "response"
        stats = c2.get_stats()
        assert stats["total_hits"] >= 1

    def test_disabled_cache(self, tmp_path):
        cache = PromptCache(cache_dir=str(tmp_path), enabled=False)
        result = cache.put("sys", "user", "model", "response")
        assert result == ""
        assert cache.get("sys", "user", "model") is None

    def test_cleanup_expired(self, cache):
        cache.put("sys", "u1", "model", "r1", ttl=0.01)
        cache.put("sys", "u2", "model", "r2", ttl=3600)
        time.sleep(0.02)
        removed = cache.cleanup_expired()
        assert removed == 1
        assert cache.get("sys", "u2", "model") is not None

    def test_hit_count_increments(self, cache):
        cache.put("sys", "user", "model", "response")
        cache.get("sys", "user", "model")
        cache.get("sys", "user", "model")
        entry = cache.get("sys", "user", "model")
        assert entry.hit_count == 3
