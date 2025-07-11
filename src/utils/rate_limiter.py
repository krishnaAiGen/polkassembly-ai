#!/usr/bin/env python3
"""
Rate limiting utility for the Polkadot AI Chatbot using Redis.
"""

import os
import logging
import redis
from typing import Optional, Tuple
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

try:
    from redis_rate_limit import RateLimiter, TooManyRequests
    RATE_LIMIT_AVAILABLE = True
except ImportError:
    logger.warning("python-redis-rate-limit package not installed. Rate limiting will be disabled.")
    RATE_LIMIT_AVAILABLE = False

class ChatbotRateLimiter:
    """Rate limiter for chatbot API requests"""
    
    def __init__(self, 
                 redis_host: str = "localhost",
                 redis_port: int = 6379,
                 redis_db: int = 0,
                 redis_password: Optional[str] = None,
                 max_requests: int = 20,
                 expire_seconds: int = 3600):  # 1 hour
        """
        Initialize the rate limiter
        
        Args:
            redis_host: Redis server host
            redis_port: Redis server port
            redis_db: Redis database number
            redis_password: Redis password (if required)
            max_requests: Maximum requests allowed per time window
            expire_seconds: Time window in seconds (default: 1 hour)
        """
        self.enabled = False
        self.limiter = None
        self.max_requests = max_requests
        self.expire_seconds = expire_seconds
        
        if not RATE_LIMIT_AVAILABLE:
            logger.warning("Rate limiting not available. Install: pip install python-redis-rate-limit")
            return
        
        try:
            # Create Redis connection pool
            redis_pool = redis.ConnectionPool(
                host=redis_host,
                port=redis_port,
                db=redis_db,
                password=redis_password,
                decode_responses=True
            )
            
            # Test Redis connection
            redis_client = redis.Redis(connection_pool=redis_pool)
            redis_client.ping()
            
            # Initialize rate limiter
            self.limiter = RateLimiter(
                resource='polkadot_chatbot_api',
                max_requests=max_requests,
                expire=expire_seconds,
                redis_pool=redis_pool
            )
            
            self.enabled = True
            logger.info(f"Rate limiter initialized: {max_requests} requests per {expire_seconds} seconds")
            
        except redis.ConnectionError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            logger.warning("Rate limiting disabled due to Redis connection failure")
        except Exception as e:
            logger.error(f"Failed to initialize rate limiter: {e}")
    
    def check_rate_limit(self, client_id: str) -> Tuple[bool, int]:
        """
        Check if client is within rate limits and consume one request if allowed
        
        Args:
            client_id: Unique identifier for the client (user_id or IP)
            
        Returns:
            Tuple of (is_allowed: bool, remaining_requests: int)
        """
        if not self.enabled:
            # If rate limiting is disabled, always allow with max requests
            return True, self.max_requests
        
        try:
            # Get current remaining requests before consuming
            current_remaining = self._get_remaining_requests(client_id)
            
            # Try to consume one request from the limit
            with self.limiter.limit(client=client_id):
                # If successful, return the remaining count after consumption
                remaining_after = max(0, current_remaining - 1)
                return True, remaining_after
                
        except TooManyRequests:
            # Rate limit exceeded
            return False, 0
        except Exception as e:
            logger.error(f"Error checking rate limit: {e}")
            # On error, default to allowing request
            return True, self.max_requests
    
    def _get_remaining_requests(self, client_id: str) -> int:
        """
        Get remaining requests for a client
        
        Args:
            client_id: Client identifier
            
        Returns:
            Number of remaining requests
        """
        if not self.enabled:
            return self.max_requests
        
        try:
            # Get current request count from Redis
            redis_client = redis.Redis(connection_pool=self.limiter.redis_pool)
            key = f"rate_limit:polkadot_chatbot_api_{client_id}"
            current_count = redis_client.get(key)
            
            if current_count is None:
                return self.max_requests
            
            current_count = int(current_count)
            remaining = max(0, self.max_requests - current_count)
            return remaining
            
        except Exception as e:
            logger.error(f"Error getting remaining requests: {e}")
            return self.max_requests
    
    def get_client_stats(self, client_id: str) -> dict:
        """
        Get detailed stats for a client
        
        Args:
            client_id: Client identifier
            
        Returns:
            Dictionary with client stats
        """
        try:
            remaining = self._get_remaining_requests(client_id)
            used = self.max_requests - remaining
            
            return {
                'client_id': client_id,
                'max_requests': self.max_requests,
                'used_requests': used,
                'remaining_requests': remaining,
                'time_window_seconds': self.expire_seconds,
                'rate_limiting_enabled': self.enabled
            }
        except Exception as e:
            logger.error(f"Error getting client stats: {e}")
            return {
                'client_id': client_id,
                'max_requests': self.max_requests,
                'used_requests': 0,
                'remaining_requests': self.max_requests,
                'time_window_seconds': self.expire_seconds,
                'rate_limiting_enabled': False,
                'error': str(e)
            }


# Global rate limiter instance
_rate_limiter_instance = None

def get_rate_limiter() -> ChatbotRateLimiter:
    """Get the global rate limiter instance"""
    global _rate_limiter_instance
    if _rate_limiter_instance is None:
        # Get configuration from environment variables
        redis_host = os.getenv("REDIS_HOST", "localhost")
        redis_port = int(os.getenv("REDIS_PORT", "6379"))
        redis_db = int(os.getenv("REDIS_DB", "0"))
        redis_password = os.getenv("REDIS_PASSWORD")
        max_requests = int(os.getenv("RATE_LIMIT_MAX_REQUESTS", "20"))
        expire_seconds = int(os.getenv("RATE_LIMIT_EXPIRE_SECONDS", "3600"))  # 1 hour
        
        _rate_limiter_instance = ChatbotRateLimiter(
            redis_host=redis_host,
            redis_port=redis_port,
            redis_db=redis_db,
            redis_password=redis_password,
            max_requests=max_requests,
            expire_seconds=expire_seconds
        )
    return _rate_limiter_instance

def check_rate_limit(client_id: str) -> Tuple[bool, int]:
    """
    Convenience function to check rate limits
    
    Args:
        client_id: Client identifier (user_id or IP)
        
    Returns:
        Tuple of (is_allowed: bool, remaining_requests: int)
    """
    rate_limiter = get_rate_limiter()
    return rate_limiter.check_rate_limit(client_id)

def get_client_stats(client_id: str) -> dict:
    """
    Convenience function to get client stats
    
    Args:
        client_id: Client identifier
        
    Returns:
        Dictionary with client stats
    """
    rate_limiter = get_rate_limiter()
    return rate_limiter.get_client_stats(client_id) 