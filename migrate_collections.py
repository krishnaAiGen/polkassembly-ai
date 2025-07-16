#!/usr/bin/env python3
"""
Wrapper script to migrate from single to multi-collection setup
"""

import sys
import os

# Add src to path
src_path = os.path.join(os.path.dirname(__file__), 'src')
sys.path.insert(0, src_path)

if __name__ == "__main__":
    from src.utils.migrate_to_multi_collection import migrate_to_multi_collection
    
    migrate_to_multi_collection() 