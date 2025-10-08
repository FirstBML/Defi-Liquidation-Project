"""
Cache manager with Redis support but falls back to in-memory cache
"""
import logging
import json
from typing import Optional, Any
import os
#from dotenv import load_dotenv
#load_dotenv()

logger = logging.getLogger(__name__)

class InMemoryCache:
    """Fallback in-memory cache when Redis is not available"""
    def __init__(self):
        self._cache = {}
        logger.info("✅ Using in-memory cache (Redis not available)")
    
    def set(self, key: str, value: Any, expire: int = 3600) -> bool:
        """Set value in memory cache (expire is ignored for simplicity)"""
        try:
            self._cache[key] = {
                'value': value,
                'timestamp': os.times().elapsed  # Simple time reference
            }
            return True
        except Exception as e:
            logger.error(f"In-memory cache set error: {e}")
            return False
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from memory cache"""
        try:
            if key in self._cache:
                return self._cache[key]['value']
            return None
        except Exception as e:
            logger.error(f"In-memory cache get error: {e}")
            return None
    
    def delete(self, key: str) -> bool:
        """Delete key from memory cache"""
        try:
            if key in self._cache:
                del self._cache[key]
                return True
            return False
        except Exception as e:
            logger.error(f"In-memory cache delete error: {e}")
            return False
    
    def health_check(self) -> dict:
        """In-memory cache health check"""
        return {
            "connected": True,
            "info": "Using in-memory cache (Redis not available)",
            "type": "memory",
            "keys_stored": len(self._cache)
        }

class CacheManager:
    def __init__(self):
        self.redis_client = None
        self.fallback_cache = InMemoryCache()
        self._initialize_redis()
    
    def _initialize_redis(self):
        """Initialize Redis connection with simple parameters"""
        try:
            import redis
            
            # Get credentials from environment variables
            redis_host = os.getenv("REDIS_HOST")
            redis_port = int(os.getenv("REDIS_PORT", 6379))
            redis_username = os.getenv("REDIS_USERNAME", "default")
            redis_password = os.getenv("REDIS_PASSWORD")
            
            # Validate required credentials
            if not redis_host or not redis_password:
                logger.warning("❌ Redis credentials missing in environment variables")
                self.redis_client = None
                return
            
            # Simple connection without SSL parameters
            self.redis_client = redis.Redis(
                host=redis_host,
                port=redis_port,
                username=redis_username,
                password=redis_password,
                decode_responses=True,
                socket_connect_timeout=10,
                socket_timeout=10
                # Remove ssl and ssl_cert_reqs parameters
            )
            
            # Test connection
            self.redis_client.ping()
            logger.info("✅ Redis Cloud connected successfully")
            
        except Exception as e:
            logger.warning(f"❌ Redis connection failed: {e}. Using in-memory cache.")
            self.redis_client = None
    
    def set(self, key: str, value: Any, expire: int = 3600) -> bool:
        """Set value in cache"""
        if self.redis_client:
            try:
                serialized_value = json.dumps(value)
                return bool(self.redis_client.setex(key, expire, serialized_value))
            except Exception as e:
                logger.error(f"Redis set failed, using memory cache: {e}")
                return self.fallback_cache.set(key, value, expire)
        else:
            return self.fallback_cache.set(key, value, expire)
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if self.redis_client:
            try:
                value = self.redis_client.get(key)
                if value:
                    return json.loads(value)
                return None
            except Exception as e:
                logger.error(f"Redis get failed, trying memory cache: {e}")
                return self.fallback_cache.get(key)
        else:
            return self.fallback_cache.get(key)
    
    def delete(self, key: str) -> bool:
        """Delete key from cache"""
        if self.redis_client:
            try:
                return bool(self.redis_client.delete(key))
            except Exception as e:
                logger.error(f"Redis delete failed, trying memory cache: {e}")
                return self.fallback_cache.delete(key)
        else:
            return self.fallback_cache.delete(key)
    
    def health_check(self) -> dict:
        """Check cache health"""
        if self.redis_client:
            try:
                if self.redis_client.ping():
                    return {
                        "connected": True,
                        "info": "Redis is connected and responsive",
                        "type": "redis"
                    }
            except Exception as e:
                return {
                    "connected": False,
                    "info": f"Redis health check failed: {e}",
                    "type": "redis"
                }
        
        # Fall back to memory cache health
        return self.fallback_cache.health_check()

# Global cache instance
cache = CacheManager()