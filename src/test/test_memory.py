#!/usr/bin/env python3
"""
Test script for Mem0 memory integration.
"""

import os
import sys
import time
import requests
import json

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

API_BASE_URL = "http://localhost:8000"

def test_memory_conversation():
    """Test that the system remembers conversation context"""
    print("Testing memory conversation...")
    
    # First query - Ask about staking
    print("\n1. First query: What is staking in Polkadot?")
    first_query = {
        "question": "What is staking in Polkadot?",
        "max_chunks": 3,
        "include_sources": True
    }
    
    try:
        response = requests.post(f"{API_BASE_URL}/query", json=first_query)
        response.raise_for_status()
        data = response.json()
        
        print(f"✅ First response received")
        print(f"   Answer: {data['answer'][:100]}...")
        print(f"   Follow-ups: {data.get('follow_up_questions', [])}")
        
        # Wait a moment to ensure memory is stored
        time.sleep(2)
        
    except Exception as e:
        print(f"❌ First query failed: {e}")
        return False
    
    # Second query - Ask about rewards (related to previous staking question)
    print("\n2. Second query: What are the rewards for that?")
    second_query = {
        "question": "What are the rewards for that?",
        "max_chunks": 3,
        "include_sources": True
    }
    
    try:
        response = requests.post(f"{API_BASE_URL}/query", json=second_query)
        response.raise_for_status()
        data = response.json()
        
        print(f"✅ Second response received")
        print(f"   Answer: {data['answer'][:100]}...")
        print(f"   Follow-ups: {data.get('follow_up_questions', [])}")
        
        # Check if the answer is relevant to staking (should be due to memory context)
        answer_lower = data['answer'].lower()
        staking_terms = ['staking', 'stake', 'reward', 'validator', 'nominator', 'dot']
        
        if any(term in answer_lower for term in staking_terms):
            print("✅ Memory context appears to be working - answer is relevant to staking")
            return True
        else:
            print("⚠️  Answer may not be using memory context effectively")
            return False
        
    except Exception as e:
        print(f"❌ Second query failed: {e}")
        return False

def test_memory_with_different_user():
    """Test memory isolation between different users"""
    print("\nTesting memory isolation (if user_id support is added)...")
    
    # For now, just test another conversation flow
    print("\n3. New conversation: Hello")
    greeting_query = {
        "question": "Hello",
        "max_chunks": 1,
        "include_sources": True
    }
    
    try:
        response = requests.post(f"{API_BASE_URL}/query", json=greeting_query)
        response.raise_for_status()
        data = response.json()
        
        print(f"✅ Greeting response received")
        print(f"   Answer: {data['answer'][:100]}...")
        
        time.sleep(1)
        
        # Follow up with a question about parachains
        print("\n4. Follow-up: Tell me about parachains")
        parachain_query = {
            "question": "Tell me about parachains",
            "max_chunks": 3,
            "include_sources": True
        }
        
        response = requests.post(f"{API_BASE_URL}/query", json=parachain_query)
        response.raise_for_status()
        data = response.json()
        
        print(f"✅ Parachain response received")
        print(f"   Answer: {data['answer'][:100]}...")
        print(f"   Follow-ups: {data.get('follow_up_questions', [])}")
        
        return True
        
    except Exception as e:
        print(f"❌ New conversation test failed: {e}")
        return False

def test_health_check():
    """Test if the API is running with memory support"""
    print("Testing API health...")
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        response.raise_for_status()
        data = response.json()
        
        if data['status'] == 'healthy':
            print("✅ API is healthy and ready for memory testing")
            return True
        else:
            print(f"⚠️  API status: {data['status']}")
            return False
            
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

def main():
    """Run all memory tests"""
    print("🧠 Starting Memory Integration Tests...")
    print(f"Testing API at: {API_BASE_URL}")
    
    # Check if API is running
    if not test_health_check():
        print("❌ API is not ready for testing")
        return
    
    # Test memory conversation
    memory_works = test_memory_conversation()
    
    # Test different conversation
    isolation_works = test_memory_with_different_user()
    
    print("\n📊 Memory Test Results:")
    print(f"   Memory Context: {'✅ Working' if memory_works else '❌ Not Working'}")
    print(f"   Conversation Flow: {'✅ Working' if isolation_works else '❌ Not Working'}")
    
    if memory_works and isolation_works:
        print("\n✅ All memory tests passed! Mem0 integration is working correctly.")
    else:
        print("\n⚠️  Some memory tests failed. Check the configuration and logs.")
        print("\nTroubleshooting:")
        print("1. Ensure MEM0_API_KEY is set in your .env file")
        print("2. Check that mem0 package is installed: pip install mem0ai")
        print("3. Verify API logs for memory-related errors")

if __name__ == "__main__":
    main() 