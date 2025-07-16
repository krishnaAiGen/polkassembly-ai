#!/usr/bin/env python3
"""
Wrapper script to check onchain data status
"""

import sys
import os

# Add src to path
src_path = os.path.join(os.path.dirname(__file__), 'src')
sys.path.insert(0, src_path)

if __name__ == "__main__":
    from src.utils.check_onchain_status import main
    main() 