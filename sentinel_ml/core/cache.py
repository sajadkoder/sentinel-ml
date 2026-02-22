"""
Redis cache management for SentinelML
"""
import json
import hashlib
from typing import Optional, Any, Dict
from datetime import timedelta

import redis.asyncio as aioredis

from sentinel_ml.config import config
from sentinel_ml.core.logging import get_logger

logger = get_logger(__name__)


class CacheKeyBuilder:
    """Build consistent cache keys for different data types"""
    
    PREFIX = "sentinel_ml"
    
    @staticmethod
    def prediction(transaction_id: str) -> str:
        return f"{CacheKeyBuilder.PREFIX}:pred:{transaction_id}"
    
    @staticmethod
    def user_stats(user_id: str) -> str:
        return f"{CacheKeyBuilder.PREFIX}:user_stats:{user_id}"
    
    @staticmethod
    def merchant_stats(merchant_id: str) -> str:
        return f"{CacheKeyBuilder.PREFIX}:merchant_stats:{merchant_id}"
    
    @staticmethod
    def feature_hash(feature_vector: Dict) -> str:
        feature_str = json.dumps(feature_vector, sort_keys=True)
        hash_value = hashlib.sha256(feature_str.encode()).hexdigest()[:16]
        return f"{CacheKeyBuilder.PREFIX}:features:{hash_value}"
    
    @staticmethod
    def model_metadata(model_version: str) -> str:
        return f"{CacheKeyBuilder.PREFIX}:model:{model_version}"
    
    @staticmethod
    def rate_limit(identifier: str) -> str:
        return f"{CacheKeyBuilder.PREFIX}:rate_limit:{identifier}"


class RedisCache:
    """Async Redis cache manager with connection pooling"""
    
    def __init__(self):
        self._client: Optional[aioredis.Redis] = None
        self._connected = False
    
    async def connect(self):
        """Establish Redis connection"""
        if self._connected:
            return
        
        try:
            self._client = aioredis.Redis(
                host=config.redis.host,
                port=config.redis.port,
                db=config.redis.db,
                password=config.redis.password,
                max_connections=config.redis.max_connections,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5
            )
            
            await self._client.ping()
            self._connected = True
            logger.info(
                "Redis connection established",
                extra={"host": config.redis.host, "port": config.redis.port}
            )
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise
    
    async def disconnect(self):
        """Close Redis connection"""
        if self._client:
            await self._client.aclose()
            self._connected = False
            logger.info("Redis connection closed")
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if not self._connected:
            await self.connect()
        
        try:
            value = await self._client.get(key)
            if value:
                logger.debug(f"Cache hit for key: {key}")
                return json.loads(value)
            return None
        except Exception as e:
            logger.warning(f"Cache get error for key {key}: {e}")
            return None
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> bool:
        """Set value in cache with optional TTL"""
        if not self._connected:
            await self.connect()
        
        try:
            ttl = ttl or config.redis.ttl
            serialized = json.dumps(value)
            await self._client.setex(key, ttl, serialized)
            logger.debug(f"Cache set for key: {key}, ttl: {ttl}s")
            return True
        except Exception as e:
            logger.warning(f"Cache set error for key {key}: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete key from cache"""
        if not self._connected:
            await self.connect()
        
        try:
            await self._client.delete(key)
            logger.debug(f"Cache deleted for key: {key}")
            return True
        except Exception as e:
            logger.warning(f"Cache delete error for key {key}: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        if not self._connected:
            await self.connect()
        
        try:
            return await self._client.exists(key) > 0
        except Exception as e:
            logger.warning(f"Cache exists error for key {key}: {e}")
            return False
    
    async def increment(self, key: str, amount: int = 1) -> int:
        """Increment counter in cache"""
        if not self._connected:
            await self.connect()
        
        try:
            return await self._client.incrby(key, amount)
        except Exception as e:
            logger.warning(f"Cache increment error for key {key}: {e}")
            return 0
    
    async def set_with_expiry(
        self,
        key: str,
        value: Any,
        expire_seconds: int
    ) -> bool:
        """Set value with custom expiry"""
        return await self.set(key, value, ttl=expire_seconds)
    
    async def get_ttl(self, key: str) -> int:
        """Get remaining TTL for a key"""
        if not self._connected:
            await self.connect()
        
        try:
            return await self._client.ttl(key)
        except Exception as e:
            logger.warning(f"Cache TTL error for key {key}: {e}")
            return -1
    
    async def health_check(self) -> Dict[str, Any]:
        """Check Redis health status"""
        try:
            if not self._connected:
                await self.connect()
            
            start_time = asyncio.get_event_loop().time()
            await self._client.ping()
            latency = (asyncio.get_event_loop().time() - start_time) * 1000
            
            info = await self._client.info()
            
            return {
                "status": "healthy",
                "latency_ms": round(latency, 2),
                "connected_clients": info.get("connected_clients"),
                "used_memory_human": info.get("used_memory_human"),
                "keyspace_hits": info.get("keyspace_hits"),
                "keyspace_misses": info.get("keyspace_misses"),
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }


import asyncio

cache = RedisCache()
