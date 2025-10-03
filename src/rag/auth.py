#!/usr/bin/env python3
"""
Authentication module for Polkassembly AI API.
Provides token-based authentication for all API endpoints.
"""

import logging
from typing import Optional
from fastapi import HTTPException, Security, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from .config import Config

logger = logging.getLogger(__name__)

# Initialize HTTP Bearer security scheme
security = HTTPBearer(auto_error=False)

class AuthenticationError(Exception):
    """Custom exception for authentication errors"""
    pass

def verify_token(token: str) -> bool:
    """
    Verify if the provided token matches the expected POLKASSEMBLY_AI_TOKEN.
    
    Args:
        token: The token to verify
        
    Returns:
        bool: True if token is valid, False otherwise
    """
    if not Config.ENABLE_AUTHENTICATION:
        # If authentication is disabled, allow all requests
        return True
    
    if not Config.POLKASSEMBLY_AI_TOKEN:
        logger.error("POLKASSEMBLY_AI_TOKEN not configured but authentication is enabled")
        return False
    
    # Simple token comparison - in production, you might want more sophisticated validation
    return token == Config.POLKASSEMBLY_AI_TOKEN

async def authenticate_request(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(security)
) -> bool:
    """
    FastAPI dependency to authenticate requests using Bearer token.
    
    Args:
        credentials: HTTP Authorization credentials from request header
        
    Returns:
        bool: True if authenticated successfully
        
    Raises:
        HTTPException: If authentication fails
    """
    # Skip authentication if disabled
    if not Config.ENABLE_AUTHENTICATION:
        logger.debug("Authentication disabled, allowing request")
        return True
    
    # Check if credentials are provided
    if not credentials:
        logger.warning("No authorization credentials provided")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header required. Please provide 'Authorization: Bearer <POLKASSEMBLY-AI-TOKEN>'",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Verify the token
    if not verify_token(credentials.credentials):
        logger.warning(f"Invalid token provided: {credentials.credentials[:10]}...")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token. Please check your POLKASSEMBLY-AI-TOKEN.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    logger.debug("Request authenticated successfully")
    return True

def get_auth_header_example() -> str:
    """
    Get an example of the expected Authorization header format.
    
    Returns:
        str: Example authorization header
    """
    return "Authorization: Bearer YOUR_POLKASSEMBLY_AI_TOKEN"

def get_auth_status() -> dict:
    """
    Get the current authentication configuration status.
    
    Returns:
        dict: Authentication status information
    """
    return {
        "authentication_enabled": Config.ENABLE_AUTHENTICATION,
        "token_configured": bool(Config.POLKASSEMBLY_AI_TOKEN),
        "auth_header_format": get_auth_header_example()
    }
