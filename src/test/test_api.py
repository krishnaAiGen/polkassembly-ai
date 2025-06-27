#!/usr/bin/env python3
"""
Simple test script for the Polkadot AI Chatbot API.
"""

import requests
import json
import time
from typing import Dict, Any

API_BASE_URL = "http://localhost:8000"

def test_health_endpoint():
    """Test the health endpoint"""
    print("Testing health endpoint...")
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        response.raise_for_status()
        data = response.json()
        print(f"✅ Health check passed: {data['status']}")
        print(f"   Collection stats: {data['collection_stats']}")
        return True
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

def test_query_endpoint():
    """Test the main query endpoint"""
    print("\nTesting query endpoint...")
    
    test_questions = [
        "What is Polkadot?",
        "How does staking work in Polkadot?",
        "What are parachains?",
        "How do I become a validator?",
        "What is the DOT token used for?"
    ]
    
    for question in test_questions:
        print(f"\nAsking: '{question}'")
        try:
            payload = {
                "question": question,
                "max_chunks": 3,
                "include_sources": True
            }
            
            response = requests.post(f"{API_BASE_URL}/query", json=payload)
            response.raise_for_status()
            data = response.json()
            
            print(f"✅ Answer received (confidence: {data['confidence']:.2f})")
            print(f"   Processing time: {data['processing_time_ms']:.2f}ms")
            print(f"   Answer: {data['answer'][:100]}...")
            print(f"   Sources: {len(data['sources'])}")
            
            time.sleep(1)  # Be nice to the API
            
        except Exception as e:
            print(f"❌ Query failed: {e}")

def test_search_endpoint():
    """Test the search endpoint"""
    print("\nTesting search endpoint...")
    
    test_searches = [
        "staking rewards",
        "governance voting",
        "parachain slots"
    ]
    
    for search_query in test_searches:
        print(f"\nSearching: '{search_query}'")
        try:
            payload = {
                "query": search_query,
                "n_results": 3
            }
            
            response = requests.post(f"{API_BASE_URL}/search", json=payload)
            response.raise_for_status()
            data = response.json()
            
            print(f"✅ Search completed")
            print(f"   Processing time: {data['processing_time_ms']:.2f}ms")
            print(f"   Results found: {data['total_results']}")
            
            for i, result in enumerate(data['results']):
                print(f"   Result {i+1}: {result['content'][:80]}... (score: {result['similarity_score']:.3f})")
            
            time.sleep(1)  # Be nice to the API
            
        except Exception as e:
            print(f"❌ Search failed: {e}")

def test_stats_endpoint():
    """Test the stats endpoint"""
    print("\nTesting stats endpoint...")
    try:
        response = requests.get(f"{API_BASE_URL}/stats")
        response.raise_for_status()
        data = response.json()
        
        print("✅ Stats retrieved:")
        print(f"   Collection stats: {json.dumps(data['collection_stats'], indent=2)}")
        
    except Exception as e:
        print(f"❌ Stats request failed: {e}")

def main():
    """Run all tests"""
    print("🚀 Starting API tests...")
    print(f"Testing API at: {API_BASE_URL}")
    
    # Test health first
    if not test_health_endpoint():
        print("❌ Health check failed - API may not be running or ready")
        return
    
    # Test other endpoints
    test_stats_endpoint()
    test_search_endpoint()
    test_query_endpoint()
    
    print("\n✅ All tests completed!")

if __name__ == "__main__":
    main() 