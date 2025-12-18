"""Simple Redis client helper for the backend.

Provides `get_redis()` to obtain a Redis client configured from `REDIS_URL`.
"""
import os
from typing import Optional

try:
    from redis import Redis
except Exception:
    Redis = None


_CLIENT: Optional[Redis] = None


def get_redis() -> Redis:
    global _CLIENT
    if _CLIENT is not None:
        return _CLIENT

    if Redis is None:
        raise RuntimeError("redis package not installed. Install `redis` to use Redis features.")

    redis_url = os.getenv("REDIS_URL")
    if not redis_url:
        raise RuntimeError("REDIS_URL not set — Redis is required for STM/conv/pdf storage")

    _CLIENT = Redis.from_url(redis_url, decode_responses=True)
    return _CLIENT
