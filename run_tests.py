#!/usr/bin/env python3
"""
Entry point script for running API tests.
"""

import os
import sys

# Add src to path
src_path = os.path.join(os.path.dirname(__file__), 'src')
sys.path.insert(0, src_path)

if __name__ == "__main__":
    from src.test.test_api import main
    main() 