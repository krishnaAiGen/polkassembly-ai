#!/usr/bin/env python3
"""
Mem0 memory management for the Polkadot AI Chatbot system.
Provides persistent memory functionality for maintaining conversation context.
"""

import os
import logging
import warnings
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

# Filter out the specific Mem0 deprecation warning
warnings.filterwarnings("ignore", message=".*output_format.*deprecated.*", category=DeprecationWarning)

try:
    from mem0 import MemoryClient
    MEM0_AVAILABLE = True
except ImportError:
    logger.warning("mem0 package not installed. Memory functionality will be disabled.")
    MEM0_AVAILABLE = False

class Mem0Memory:
    """Memory management using Mem0 for conversation context retention"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Mem0 memory client
        
        Args:
            api_key: Mem0 API key, if not provided will use MEM0_API_KEY env var
        """
        self.enabled = False
        self.client = None
        
        # Check if Mem0 is enabled via environment variable
        use_mem0 = os.getenv("USE_MEM0", "false").lower() in ("true", "1", "yes", "on")
        if not use_mem0:
            logger.info("Mem0 disabled via USE_MEM0 environment variable")
            return
        
        if not MEM0_AVAILABLE:
            logger.warning("Mem0 package not available. Memory features disabled.")
            return
        
        api_key = api_key or os.getenv("MEM0_API_KEY")
        if not api_key:
            logger.warning("MEM0_API_KEY not found. Memory features disabled.")
            return
            
        try:
            self.client = MemoryClient(api_key=api_key)
            self.enabled = True
            logger.info("Mem0 memory client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Mem0 client: {e}")
    
    def get_memory_context(self, query: str, user_id: str = "polkassembly_bot") -> str:
        """
        Get relevant memory context for a query without adding it to memory.
        Useful for getting context before LLM calls.
        
        Args:
            query: The query to search for relevant memories
            user_id: Unique identifier for the user/bot
            
        Returns:
            Formatted string of relevant memories or empty string
        """
        if not self.enabled:
            return ""
            
        try:
            search_result = self.client.search(query=query, user_id=user_id, limit=5)
            
            # Handle different response structures
            if isinstance(search_result, list):
                relevant_memories = [entry.get("memory", "") for entry in search_result if isinstance(entry, dict)]
            elif isinstance(search_result, dict) and "results" in search_result:
                relevant_memories = [entry.get("memory", "") for entry in search_result["results"]]
            else:
                relevant_memories = []
            
            if relevant_memories:
                # Filter out empty memories
                valid_memories = [m for m in relevant_memories if m and m.strip()]
                if valid_memories:
                    memories_str = "\n".join(f"- {m}" for m in valid_memories)
                    return f"Previous relevant interactions:\n{memories_str}\n\n"
            
            return ""
            
        except Exception as e:
            logger.error(f"Error getting memory context: {e}")
            return ""
    
    def add_user_query(self, query: str, user_id: str = "polkassembly_bot") -> None:
        """
        Add user query to memory.
        
        Args:
            query: The user's query
            user_id: Unique identifier for the user/bot
        """
        if not self.enabled:
            return
            
        try:
            self.client.add([{"role": "user", "content": query}], user_id=user_id)
            logger.debug(f"Added user query to memory for user {user_id}")
        except Exception as e:
            logger.error(f"Error adding user query to memory: {e}")
    
    def add_assistant_response(self, response: str, user_id: str = "polkassembly_bot") -> None:
        """
        Add assistant response to memory.
        Only stores the text response, not sources or follow-up questions.
        
        Args:
            response: The assistant's response text
            user_id: Unique identifier for the user/bot
        """
        if not self.enabled:
            return
            
        try:
            self.client.add([{"role": "assistant", "content": response}], user_id=user_id)
            logger.debug(f"Added assistant response to memory for user {user_id}")
        except Exception as e:
            logger.error(f"Error adding assistant response to memory: {e}")
    
    def handle_memory(self, query: str, type: str, user_id: str = "polkassembly_bot") -> str:
        """
        Handle memory operations for the chatbot.
        
        Args:
            query: The text content to process
            type: Either "query" or "response" 
            user_id: Unique identifier for the user/bot
            
        Returns:
            String of relevant memories (empty if type is "response")
        """
        if not self.enabled:
            return ""
            
        try:
            if type == "query":
                # Search for relevant memories first
                context = self.get_memory_context(query, user_id)
                
                # Add the query to memory
                self.add_user_query(query, user_id)
                
                return context
                
            elif type == "response":
                # Just add the response to memory, don't search
                self.add_assistant_response(query, user_id)
                return ""
            else:
                logger.warning(f"Invalid type '{type}'. Must be 'query' or 'response'")
                return ""
                
        except Exception as e:
            logger.error(f"Error handling memory operation: {e}")
            return ""


# Global memory instance
_memory_instance = None

def get_memory_manager() -> Mem0Memory:
    """Get the global memory manager instance"""
    global _memory_instance
    if _memory_instance is None:
        _memory_instance = Mem0Memory()
    return _memory_instance

def get_memory_context(query: str, user_id: str = "polkassembly_bot") -> str:
    """
    Convenience function to get memory context.
    
    Args:
        query: The query to search for relevant memories
        user_id: Unique identifier for the user/bot
        
    Returns:
        Formatted string of relevant memories
    """
    memory_manager = get_memory_manager()
    return memory_manager.get_memory_context(query, user_id)

def add_user_query(query: str, user_id: str = "polkassembly_bot") -> None:
    """
    Convenience function to add user query to memory.
    
    Args:
        query: The user's query
        user_id: Unique identifier for the user/bot
    """
    memory_manager = get_memory_manager()
    memory_manager.add_user_query(query, user_id)

def add_assistant_response(response: str, user_id: str = "polkassembly_bot") -> None:
    """
    Convenience function to add assistant response to memory.
    
    Args:
        response: The assistant's response text
        user_id: Unique identifier for the user/bot
    """
    memory_manager = get_memory_manager()
    memory_manager.add_assistant_response(response, user_id) 