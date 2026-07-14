"""
tools/cache.py
Simple TTL-based in-memory cache for Cache-Augmented Generation (CAG).
Drop-in replacement works without Redis; swap backend if needed.
"""

from __future__ import annotations
import time
from typing import Any


class Cache:
    def __init__(self, ttl: int = 3600, max_size: int = 500):
        """
        ttl      : seconds before a cached entry expires
        max_size : evict oldest entry when cache exceeds this size
        """
        self._store: dict[str, dict] = {}  # key → {value, expires_at}
        self._ttl = ttl
        self._max_size = max_size

    def get(self, key: str) -> Any | None:
        entry = self._store.get(key)
        if entry is None:
            return None
        if time.time() > entry["expires_at"]:
            del self._store[key]
            return None
        return entry["value"]

    def set(self, key: str, value: Any):
        # Evict oldest if at capacity
        if len(self._store) >= self._max_size:
            oldest_key = min(self._store, key=lambda k: self._store[k]["expires_at"])
            del self._store[oldest_key]

        self._store[key] = {
            "value": value,
            "expires_at": time.time() + self._ttl,
        }

    def delete(self, key: str):
        self._store.pop(key, None)

    def clear(self):
        self._store.clear()

    def size(self) -> int:
        # Purge expired first
        now = time.time()
        expired = [k for k, v in self._store.items() if now > v["expires_at"]]
        for k in expired:
            del self._store[k]
        return len(self._store)

    def stats(self) -> dict:
        return {
            "size": self.size(),
            "max_size": self._max_size,
            "ttl_seconds": self._ttl,
        }


# ── Optional Redis backend ─────────────────────────────────────────────────────
class RedisCache:
    """
    Production-grade cache using Redis.
    Usage: from tools.cache import RedisCache as Cache
    Requires: pip install redis
    Set REDIS_URL in .env (default: redis://localhost:6379)
    """

    def __init__(self, ttl: int = 3600):
        import redis
        import os
        self._ttl = ttl
        self._client = redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379"))

    def get(self, key: str) -> Any | None:
        import json
        val = self._client.get(key)
        return json.loads(val) if val else None

    def set(self, key: str, value: Any):
        import json
        self._client.setex(key, self._ttl, json.dumps(value))

    def delete(self, key: str):
        self._client.delete(key)

    def clear(self):
        self._client.flushdb()

    def size(self) -> int:
        return self._client.dbsize()