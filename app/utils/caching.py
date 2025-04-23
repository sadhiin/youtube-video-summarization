"""
Caching utility module for the YouTube Video Summarizer.
"""

import json
import time
import functools
from typing import Any, Dict, Optional, Callable, Union
from functools import lru_cache
from app.utils.logger import logging

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logging.info("Redis not available, using in-memory caching instead")

# Global Redis client
_redis_client = None

# In-memory cache as fallback
_memory_cache = {}


def setup_redis_cache(redis_url: str) -> bool:
    """
    Set up Redis caching.

    Args:
        redis_url: Redis connection URL

    Returns:
        True if successful, False otherwise
    """
    global _redis_client

    if not REDIS_AVAILABLE:
        return False

    try:
        _redis_client = redis.from_url(redis_url)
        _redis_client.ping()
        logging.info("Redis cache configured successfully")
        return True
    except Exception as e:
        logging.error(f"Error configuring Redis: {e}")
        _redis_client = None
        return False


def is_redis_available() -> bool:
    """Check if Redis is available for caching."""
    return _redis_client is not None


def cache_set(key: str, value: Any, expires: int = 3600) -> bool:
    """
    Set a value in the cache.

    Args:
        key: Cache key
        value: Value to cache
        expires: Expiration time in seconds (default: 1 hour)

    Returns:
        True if successful, False otherwise
    """
    # Convert value to JSON
    try:
        serialized = json.dumps(value)
    except (TypeError, ValueError):
        logging.error(f"Error serializing value for key {key}")
        return False

    # Store in appropriate cache
    if is_redis_available():
        try:
            return _redis_client.setex(key, expires, serialized)
        except Exception as e:
            logging.error(f"Redis error in cache_set: {e}")
            # Fall back to memory cache

    # Memory cache fallback
    expiry_time = time.time() + expires
    _memory_cache[key] = {
        "value": serialized,
        "expires": expiry_time
    }
    return True


def cache_get(key: str) -> Optional[Any]:
    """
    Get a value from the cache.

    Args:
        key: Cache key

    Returns:
        Cached value or None if not found or expired
    """
    # Try Redis first
    if is_redis_available():
        try:
            value = _redis_client.get(key)
            if value:
                return json.loads(value)
        except Exception as e:
            print(f"Redis error in cache_get: {e}")

    # Memory cache fallback
    if key in _memory_cache:
        cache_entry = _memory_cache[key]

        # Check expiration
        if cache_entry["expires"] > time.time():
            return json.loads(cache_entry["value"])
        else:
            # Remove expired entry
            del _memory_cache[key]

    return None


def cache_delete(key: str) -> bool:
    """
    Delete a value from the cache.

    Args:
        key: Cache key

    Returns:
        True if successful, False otherwise
    """
    # Delete from Redis
    redis_result = False
    if is_redis_available():
        try:
            redis_result = bool(_redis_client.delete(key))
        except Exception as e:
            logging.error(f"Redis error in cache_delete: {e}")

    # Delete from memory cache
    memory_result = False
    if key in _memory_cache:
        del _memory_cache[key]
        memory_result = True

    return redis_result or memory_result


def cached(expires: int = 3600, prefix: str = "cache"):
    """
    Decorator for caching function results.

    Args:
        expires: Cache expiration time in seconds
        prefix: Prefix for cache keys

    Returns:
        Decorated function
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create a cache key from function name, args and kwargs
            key_parts = [prefix, func.__name__]

            # Add args and kwargs to key
            for arg in args:
                if isinstance(arg, (str, int, float, bool)):
                    key_parts.append(str(arg))

            for k, v in sorted(kwargs.items()):
                if isinstance(v, (str, int, float, bool)):
                    key_parts.append(f"{k}={v}")

            cache_key = ":".join(key_parts)

            # Check cache
            cached_result = cache_get(cache_key)
            if cached_result is not None:
                return cached_result

            # Call function and cache result
            result = func(*args, **kwargs)
            if result is not None:
                cache_set(cache_key, result, expires)

            return result
        return wrapper
    return decorator


# Simple in-memory cache for frequently used functions
@lru_cache(maxsize=128)
def memoized_function(arg1, arg2, **kwargs):
    """Example of using Python's built-in lru_cache for simple memoization."""
    return arg1 + arg2


def clear_memory_cache():
    """Clear the in-memory cache."""
    global _memory_cache
    _memory_cache = {}


def clear_redis_cache(pattern: str = "*"):
    """
    Clear Redis cache with a pattern.

    Args:
        pattern: Redis key pattern to match
    """
    if is_redis_available():
        try:
            keys = _redis_client.keys(pattern)
            if keys:
                _redis_client.delete(*keys)
            return True
        except Exception as e:
            logging.error(f"Redis error in clear_redis_cache: {e}")

    return False