#!/usr/bin/env python3
"""
Test script for rate limiting functionality.
This script makes multiple requests to test the rate limiting system.
"""

import os
import sys
import time
import requests
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

API_BASE_URL = "http://localhost:8000"

def make_query_request(user_id: str, request_number: int, question: str = None):
    """Make a single query request"""
    if question is None:
        question = f"What is Polkadot? (Request {request_number})"
    
    payload = {
        "question": question,
        "user_id": user_id,
        "client_ip": "192.168.1.100",
        "max_chunks": 3,
        "include_sources": True
    }
    
    try:
        start_time = time.time()
        response = requests.post(f"{API_BASE_URL}/query", json=payload)
        elapsed_time = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            return {
                'success': True,
                'request_number': request_number,
                'remaining_requests': data.get('remaining_requests', 'unknown'),
                'answer_preview': data['answer'][:50] + '...',
                'response_time': round(elapsed_time, 2),
                'status_code': 200
            }
        elif response.status_code == 429:
            return {
                'success': False,
                'request_number': request_number,
                'remaining_requests': 0,
                'error': 'Rate limited',
                'response_time': round(elapsed_time, 2),
                'status_code': 429
            }
        else:
            return {
                'success': False,
                'request_number': request_number,
                'error': f"HTTP {response.status_code}: {response.text}",
                'response_time': round(elapsed_time, 2),
                'status_code': response.status_code
            }
            
    except Exception as e:
        return {
            'success': False,
            'request_number': request_number,
            'error': str(e),
            'response_time': 0,
            'status_code': None
        }

def test_sequential_requests():
    """Test rate limiting with sequential requests"""
    print("🔄 Testing Sequential Requests")
    print("-" * 40)
    
    user_id = "test_user_sequential"
    max_requests = 16  # Stay under 20 to avoid IP blocking
    
    results = []
    
    for i in range(1, max_requests + 1):
        print(f"Making request {i}/{max_requests}...")
        result = make_query_request(user_id, i, f"Tell me about staking in Polkadot (Request {i})")
        results.append(result)
        
        if result['success']:
            print(f"  ✅ Success - Remaining: {result['remaining_requests']} - Time: {result['response_time']}s")
        else:
            print(f"  ❌ Failed - {result['error']} - Time: {result['response_time']}s")
        
        # Small delay to avoid overwhelming the server
        time.sleep(0.5)
    
    return results

def test_concurrent_requests():
    """Test rate limiting with concurrent requests"""
    print("\n🚀 Testing Concurrent Requests")
    print("-" * 40)
    
    user_id = "test_user_concurrent"
    max_requests = 10  # Smaller number for concurrent testing
    
    questions = [
        "What is Polkadot governance?",
        "How do parachains work?",
        "What are validators?",
        "How to stake DOT tokens?",
        "What is the relay chain?",
        "How does nominating work?",
        "What are parachain auctions?",
        "How does treasury work?",
        "What is XCM?",
        "How to participate in governance?"
    ]
    
    results = []
    
    with ThreadPoolExecutor(max_workers=5) as executor:
        # Submit all requests
        futures = [
            executor.submit(make_query_request, user_id, i+1, questions[i % len(questions)])
            for i in range(max_requests)
        ]
        
        # Collect results as they complete
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            
            if result['success']:
                print(f"  ✅ Request {result['request_number']} - Remaining: {result['remaining_requests']} - Time: {result['response_time']}s")
            else:
                print(f"  ❌ Request {result['request_number']} - {result['error']} - Time: {result['response_time']}s")
    
    # Sort results by request number for better display
    results.sort(key=lambda x: x['request_number'])
    return results

def test_different_users():
    """Test rate limiting with different users"""
    print("\n👥 Testing Different Users")
    print("-" * 40)
    
    users = ["alice", "bob", "charlie"]
    requests_per_user = 3
    
    all_results = {}
    
    for user in users:
        print(f"\nTesting user: {user}")
        user_results = []
        
        for i in range(1, requests_per_user + 1):
            result = make_query_request(user, i, f"What is Polkadot? (User {user}, Request {i})")
            user_results.append(result)
            
            if result['success']:
                print(f"  ✅ Request {i} - Remaining: {result['remaining_requests']}")
            else:
                print(f"  ❌ Request {i} - {result['error']}")
            
            time.sleep(0.3)
        
        all_results[user] = user_results
    
    return all_results

def check_rate_limit_status(user_id: str):
    """Check rate limit status for a user"""
    try:
        response = requests.get(f"{API_BASE_URL}/rate-limit/{user_id}")
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"HTTP {response.status_code}"}
    except Exception as e:
        return {"error": str(e)}

def test_health_check():
    """Test if the API is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        response.raise_for_status()
        data = response.json()
        return data['status'] == 'healthy'
    except:
        return False

def analyze_results(results, test_name):
    """Analyze and display test results"""
    print(f"\n📊 {test_name} Results Analysis:")
    print("-" * 50)
    
    if isinstance(results, dict):
        # Multiple users
        for user, user_results in results.items():
            successful = [r for r in user_results if r['success']]
            failed = [r for r in user_results if not r['success']]
            
            print(f"User {user}:")
            print(f"  ✅ Successful requests: {len(successful)}")
            print(f"  ❌ Failed requests: {len(failed)}")
            
            if successful:
                avg_time = sum(r['response_time'] for r in successful) / len(successful)
                print(f"  ⏱️  Average response time: {avg_time:.2f}s")
    else:
        # Single test
        successful = [r for r in results if r['success']]
        failed = [r for r in results if not r['success']]
        rate_limited = [r for r in failed if r.get('status_code') == 429]
        
        print(f"Total requests: {len(results)}")
        print(f"✅ Successful: {len(successful)}")
        print(f"❌ Failed: {len(failed)}")
        print(f"🚫 Rate limited: {len(rate_limited)}")
        
        if successful:
            avg_time = sum(r['response_time'] for r in successful) / len(successful)
            print(f"⏱️  Average response time: {avg_time:.2f}s")
        
        if rate_limited:
            first_rate_limit = min(r['request_number'] for r in rate_limited)
            print(f"🔥 First rate limit at request: {first_rate_limit}")

def main():
    """Run all rate limiting tests"""
    print("🔐 Rate Limiting Test Suite")
    print("=" * 50)
    
    # Check if API is running
    if not test_health_check():
        print("❌ API is not ready for testing")
        print("Please start the server with: python run_server.py")
        return
    
    print("✅ API is healthy and ready for testing")
    print("\nNote: Making maximum 16 requests to avoid IP blocking")
    
    # Test 1: Sequential requests
    try:
        sequential_results = test_sequential_requests()
        analyze_results(sequential_results, "Sequential Requests")
    except Exception as e:
        print(f"❌ Sequential test failed: {e}")
    
    # Wait between tests
    print("\n⏳ Waiting 5 seconds between tests...")
    time.sleep(5)
    
    # Test 2: Concurrent requests
    try:
        concurrent_results = test_concurrent_requests()
        analyze_results(concurrent_results, "Concurrent Requests")
    except Exception as e:
        print(f"❌ Concurrent test failed: {e}")
    
    # Wait between tests
    print("\n⏳ Waiting 3 seconds...")
    time.sleep(3)
    
    # Test 3: Different users
    try:
        multi_user_results = test_different_users()
        analyze_results(multi_user_results, "Multiple Users")
    except Exception as e:
        print(f"❌ Multi-user test failed: {e}")
    
    # Check final rate limit status
    print("\n🔍 Final Rate Limit Status:")
    print("-" * 30)
    test_users = ["test_user_sequential", "test_user_concurrent", "alice", "bob", "charlie"]
    
    for user in test_users:
        status = check_rate_limit_status(user)
        if 'error' not in status:
            stats = status.get('rate_limit_stats', {})
            remaining = stats.get('remaining_requests', 'unknown')
            used = stats.get('used_requests', 'unknown')
            print(f"{user}: {used} used, {remaining} remaining")
        else:
            print(f"{user}: Error getting status")
    
    print("\n" + "=" * 50)
    print("✅ Rate limiting tests completed!")
    print("\nTroubleshooting:")
    print("1. Ensure Redis is running: redis-server")
    print("2. Install rate limiting: pip install python-redis-rate-limit redis")
    print("3. Check Redis connection in logs")

if __name__ == "__main__":
    main() 