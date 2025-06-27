#!/usr/bin/env python3
"""
Entry point script for running the API server.
"""

import os
import sys

# Add src to path
src_path = os.path.join(os.path.dirname(__file__), 'src')
sys.path.insert(0, src_path)

if __name__ == "__main__":
    import uvicorn
    from src.rag.config import Config
    
    print("Starting Polkadot AI Chatbot API server...")
    uvicorn.run(
        "src.rag.api_server:app",
        host=Config.API_HOST,
        port=Config.API_PORT,
        log_level="info",
        reload=False
    ) 