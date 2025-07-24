#!/usr/bin/env python3
"""
Entry point script for running all tests.
"""

import os
import sys
import argparse

# Add project root to path (go up two levels from src/test/)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, project_root)

def run_api_tests():
    """Run basic API functionality tests"""
    print("🚀 Running API Tests...")
    from src.test.test_api import main as api_main
    api_main()

def run_memory_tests():
    """Run memory integration tests"""
    print("\n🧠 Running Memory Tests...")
    from src.test.test_memory import main as memory_main
    memory_main()

def run_formatting_tests():
    """Run formatting validation tests"""
    print("\n🎨 Running Formatting Tests...")
    from src.test.test_formatting import main as formatting_main
    formatting_main()

def run_guardrails_tests():
    """Run enhanced guardrails tests"""
    print("\n🛡️ Running Enhanced Guardrails Tests...")
    from src.test.test_enhanced_guardrails import unittest
    import sys
    
    # Discover and run guardrails tests
    loader = unittest.TestLoader()
    suite = loader.discover('src/test', pattern='test_enhanced_guardrails.py')
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    if result.wasSuccessful():
        print("✅ All guardrails tests passed!")
    else:
        print(f"❌ {len(result.failures)} failures, {len(result.errors)} errors")
        sys.exit(1)

def run_rate_limit_tests():
    """Run rate limiting tests"""
    print("\n🔐 Running Rate Limit Tests...")
    from src.test.test_rate_limit import main as rate_limit_main
    rate_limit_main()

def main():
    """Main test runner with options"""
    parser = argparse.ArgumentParser(description='Run Polkadot AI Chatbot tests')
    parser.add_argument('--api', action='store_true', help='Run API tests only')
    parser.add_argument('--memory', action='store_true', help='Run memory tests only')
    parser.add_argument('--formatting', action='store_true', help='Run formatting tests only')
    parser.add_argument('--guardrails', action='store_true', help='Run enhanced guardrails tests only')
    parser.add_argument('--rate-limit', action='store_true', help='Run rate limiting tests only')
    parser.add_argument('--all', action='store_true', default=True, help='Run all tests (default)')
    
    args = parser.parse_args()
    
    print("🔬 Polkadot AI Chatbot Test Suite")
    print("=" * 50)
    
    # Determine which tests to run
    run_all = args.all and not any([args.api, args.memory, args.formatting, args.guardrails, getattr(args, 'rate_limit', False)])
    
    if args.api or run_all:
        run_api_tests()
    
    if args.memory or run_all:
        run_memory_tests()
    
    if args.formatting or run_all:
        run_formatting_tests()
    
    if args.guardrails or run_all:
        run_guardrails_tests()
    
    if getattr(args, 'rate_limit', False) or run_all:
        run_rate_limit_tests()
    
    print("\n" + "=" * 50)
    print("✅ Test suite completed!")
    print("\nNote: Make sure the API server is running on localhost:8000")
    print("Run with: python run_server.py")
    print("\nFor rate limiting tests, also ensure Redis is running:")
    print("Run with: redis-server")

if __name__ == "__main__":
    main() 