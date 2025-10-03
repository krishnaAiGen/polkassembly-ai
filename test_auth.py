#!/usr/bin/env python3
"""
Test script for Polkassembly AI API authentication.
Tests both authenticated and unauthenticated requests.
"""

import requests
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
API_BASE_URL = "http://localhost:8000"
POLKASSEMBLY_AI_TOKEN = os.getenv("POLKASSEMBLY_AI_TOKEN", "test-token-123")

def test_public_endpoints():
    """Test public endpoints that don't require authentication"""
    print("🔓 Testing public endpoints...")
    
    # Test root endpoint
    try:
        response = requests.get(f"{API_BASE_URL}/")
        print(f"✅ GET / - Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   Authentication enabled: {data.get('authentication', {}).get('authentication_enabled', 'Unknown')}")
    except Exception as e:
        print(f"❌ GET / - Error: {e}")
    
    # Test auth status endpoint
    try:
        response = requests.get(f"{API_BASE_URL}/auth-status")
        print(f"✅ GET /auth-status - Status: {response.status_code}")
        if response.status_code == 200:
            print(f"   Response: {response.json()}")
    except Exception as e:
        print(f"❌ GET /auth-status - Error: {e}")

def test_protected_endpoints_without_auth():
    """Test protected endpoints without authentication (should fail)"""
    print("\n🔒 Testing protected endpoints WITHOUT authentication...")
    
    endpoints = [
        ("GET", "/health"),
        ("GET", "/stats"),
        ("POST", "/query", {
            "question": "What is Polkadot?",
            "user_id": "test-user",
            "client_ip": "127.0.0.1"
        }),
        ("POST", "/search", {
            "query": "governance"
        })
    ]
    
    for method, endpoint, *data in endpoints:
        try:
            if method == "GET":
                response = requests.get(f"{API_BASE_URL}{endpoint}")
            else:
                response = requests.post(
                    f"{API_BASE_URL}{endpoint}",
                    json=data[0] if data else {},
                    headers={"Content-Type": "application/json"}
                )
            
            if response.status_code == 401:
                print(f"✅ {method} {endpoint} - Correctly blocked (401)")
            else:
                print(f"⚠️  {method} {endpoint} - Unexpected status: {response.status_code}")
                
        except Exception as e:
            print(f"❌ {method} {endpoint} - Error: {e}")

def test_protected_endpoints_with_auth():
    """Test protected endpoints with authentication (should succeed)"""
    print(f"\n🔑 Testing protected endpoints WITH authentication...")
    print(f"   Using token: {POLKASSEMBLY_AI_TOKEN[:10]}...")
    
    headers = {
        "Authorization": f"Bearer {POLKASSEMBLY_AI_TOKEN}",
        "Content-Type": "application/json"
    }
    
    endpoints = [
        ("GET", "/health"),
        ("GET", "/stats"),
        ("POST", "/query", {
            "question": "What is Polkadot?",
            "user_id": "test-user",
            "client_ip": "127.0.0.1"
        }),
        ("POST", "/search", {
            "query": "governance"
        })
    ]
    
    for method, endpoint, *data in endpoints:
        try:
            if method == "GET":
                response = requests.get(f"{API_BASE_URL}{endpoint}", headers=headers)
            else:
                response = requests.post(
                    f"{API_BASE_URL}{endpoint}",
                    json=data[0] if data else {},
                    headers=headers
                )
            
            if response.status_code in [200, 503]:  # 503 is OK if service not fully initialized
                print(f"✅ {method} {endpoint} - Status: {response.status_code}")
            else:
                print(f"❌ {method} {endpoint} - Status: {response.status_code}")
                print(f"   Response: {response.text[:200]}...")
                
        except Exception as e:
            print(f"❌ {method} {endpoint} - Error: {e}")

def test_invalid_token():
    """Test with invalid token (should fail)"""
    print(f"\n🚫 Testing with invalid token...")
    
    headers = {
        "Authorization": "Bearer invalid-token-123",
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.get(f"{API_BASE_URL}/health", headers=headers)
        if response.status_code == 401:
            print(f"✅ Invalid token correctly rejected (401)")
        else:
            print(f"⚠️  Invalid token - Unexpected status: {response.status_code}")
    except Exception as e:
        print(f"❌ Invalid token test - Error: {e}")

def main():
    """Run all authentication tests"""
    print("🧪 Polkassembly AI API Authentication Tests")
    print("=" * 50)
    
    # Test public endpoints
    test_public_endpoints()
    
    # Test protected endpoints without auth
    test_protected_endpoints_without_auth()
    
    # Test protected endpoints with auth
    test_protected_endpoints_with_auth()
    
    # Test invalid token
    test_invalid_token()
    
    print("\n" + "=" * 50)
    print("🏁 Authentication tests completed!")
    print("\n📝 Frontend Integration Example:")
    print(f"   Headers: {{'Authorization': 'Bearer {POLKASSEMBLY_AI_TOKEN}'}}")
    print(f"   Example: fetch('{API_BASE_URL}/query', {{")
    print(f"     method: 'POST',")
    print(f"     headers: {{")
    print(f"       'Authorization': 'Bearer {POLKASSEMBLY_AI_TOKEN}',")
    print(f"       'Content-Type': 'application/json'")
    print(f"     }},")
    print(f"     body: JSON.stringify({{")
    print(f"       question: 'What is Polkadot?',")
    print(f"       user_id: 'user123',")
    print(f"       client_ip: '192.168.1.1'")
    print(f"     }})")
    print(f"   }})")

if __name__ == "__main__":
    main()
