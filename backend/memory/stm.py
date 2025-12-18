"""
Redis-backed Short-Term Memory (STM) Manager

This implementation stores a small, ephemeral per-user context in Redis with TTL.
It intentionally stores only distilled context (short snippets, active task, intent)
and uses the Redis key TTL to implement the ephemeral behavior requested.

Requirements:
- `REDIS_URL` environment variable or explicit `redis_url` passed to constructor.

Behavior:
- `add_memory` appends a JSON-encoded memory to a Redis list and resets the TTL.
- `get_relevant_memories` reads the list, scores items locally and returns top-N.
- `get_all_memories` returns all items (chronological) and resets the TTL on access.
- `clear_memories` deletes the Redis key for the user.
"""

import os
import time
import json
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict, field

logger = logging.getLogger(__name__)

try:
    from redis import Redis
except Exception:
    Redis = None


@dataclass
class MemoryItem:
    id: str
    content: str
    timestamp: float
    importance: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryItem':
        return cls(**data)


class STMManager:
    """Redis-backed STM Manager. Requires `redis_url` or `REDIS_URL` env var."""

    def __init__(self, redis_url: Optional[str] = None, ttl_seconds: int = 1800, max_size: int = 50):
        if Redis is None:
            raise RuntimeError("redis package not installed. Install `redis` to use Redis-backed STM.")

        self.redis_url = redis_url or os.getenv("REDIS_URL")
        if not self.redis_url:
            raise RuntimeError("REDIS_URL not provided — STM must use Redis as per security policy")

        self.client = Redis.from_url(self.redis_url, decode_responses=True)
        self.ttl_seconds = int(ttl_seconds)
        self.max_size = max_size

    def _key(self, user_id: str) -> str:
        return f"stm:{user_id}:memories"

    def add_memory(self, user_id: str, content: str, importance: float = 1.0,
                   metadata: Optional[Dict[str, Any]] = None) -> str:
        if not user_id:
            raise ValueError("user_id is required")

        item = MemoryItem(
            id=f"stm_{user_id}_{int(time.time() * 1000)}",
            content=content,
            timestamp=time.time(),
            importance=float(importance),
            metadata=metadata or {}
        )

        data = json.dumps(item.to_dict())
        key = self._key(user_id)

        
        self.client.lpush(key, data)
        self.client.ltrim(key, 0, self.max_size - 1)
        
        self.client.expire(key, self.ttl_seconds)

        return item.id

    def get_relevant_memories(self, user_id: str, query: str = "", limit: int = 10) -> List[MemoryItem]:
        if not user_id:
            raise ValueError("user_id is required")

        key = self._key(user_id)
        raw = self.client.lrange(key, 0, -1)
        if not raw:
            return []

        
        self.client.expire(key, self.ttl_seconds)

        memories: List[MemoryItem] = []
        for r in raw:
            try:
                obj = json.loads(r)
                memories.append(MemoryItem.from_dict(obj))
            except Exception:
                continue

        
        current = time.time()
        scored = []
        query_words = set(query.lower().split()) if query else set()

        for mem in memories:
            score = mem.importance
            
            time_diff = current - mem.timestamp
            decay = 0.95 ** (time_diff / 3600)
            score *= decay

            if query_words:
                overlap = len(query_words.intersection(set(mem.content.lower().split())))
                if overlap:
                    score *= (1 + overlap * 0.1)

            scored.append((score, mem))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in scored[:limit]]

    def get_all_memories(self, user_id: str) -> List[MemoryItem]:
        if not user_id:
            raise ValueError("user_id is required")

        key = self._key(user_id)
        raw = self.client.lrange(key, 0, -1)
        if not raw:
            return []

        
        self.client.expire(key, self.ttl_seconds)

        memories = []
        for r in reversed(raw):  
            try:
                obj = json.loads(r)
                memories.append(MemoryItem.from_dict(obj))
            except Exception:
                continue

        return memories

    def clear_memories(self, user_id: str):
        if not user_id:
            raise ValueError("user_id is required")

        key = self._key(user_id)
        self.client.delete(key)
        logger.info(f"Cleared STM for user {user_id}")


