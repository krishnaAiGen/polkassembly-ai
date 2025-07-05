#!/usr/bin/env python3
"""
Test script to verify the API response format matches the expected structure.
"""

import requests
import json
from typing import Dict, Any

def test_api_format():
    """Test the API response format"""
    
    # Test request
    request_data = {
        "question": "What is Polkadot governance?",
        "user_id": "krishna",
        "client_ip": "192.168.1.1"
    }
    
    print("🧪 Testing API Response Format")
    print("=" * 50)
    
    try:
        # Make request
        response = requests.post("http://localhost:8000/query", json=request_data)
        
        if response.status_code == 200:
            data = response.json()
            
            print("✅ Request successful!")
            print(f"📊 Response status: {response.status_code}")
            
            # Check required fields
            required_fields = [
                "answer", "sources", "follow_up_questions", "remaining_requests",
                "confidence", "context_used", "model_used", "chunks_used",
                "processing_time_ms", "timestamp", "search_method"
            ]
            
            print("\n🔍 Checking required fields:")
            missing_fields = []
            for field in required_fields:
                if field in data:
                    value = data[field]
                    if value is not None:
                        print(f"  ✅ {field}: {type(value).__name__} = {value}")
                    else:
                        print(f"  ⚠️  {field}: None (should have a value)")
                        missing_fields.append(field)
                else:
                    print(f"  ❌ {field}: Missing")
                    missing_fields.append(field)
            
            # Check sources structure
            print("\n📚 Checking sources structure:")
            if "sources" in data and isinstance(data["sources"], list):
                for i, source in enumerate(data["sources"]):
                    print(f"  Source {i+1}:")
                    source_fields = ["title", "url", "source_type", "similarity_score"]
                    for field in source_fields:
                        if field in source:
                            print(f"    ✅ {field}: {source[field]}")
                        else:
                            print(f"    ❌ {field}: Missing")
            else:
                print("  ❌ sources: Not a list or missing")
            
            # Check follow-up questions
            print("\n❓ Checking follow-up questions:")
            if "follow_up_questions" in data and isinstance(data["follow_up_questions"], list):
                print(f"  ✅ Found {len(data['follow_up_questions'])} follow-up questions")
                for i, question in enumerate(data["follow_up_questions"]):
                    print(f"    {i+1}. {question}")
            else:
                print("  ❌ follow_up_questions: Not a list or missing")
            
            # Summary
            print("\n" + "=" * 50)
            if missing_fields:
                print(f"❌ Missing or null fields: {missing_fields}")
                return False
            else:
                print("✅ All required fields present and have values!")
                print("✅ API response format matches expected structure!")
                return True
                
        else:
            print(f"❌ Request failed with status {response.status_code}")
            print(f"Error: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to API server")
        print("Make sure the server is running: python run_server.py")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def show_expected_format():
    """Show the expected request and response formats"""
    
    print("\n📋 Expected Request Format:")
    print("-" * 30)
    expected_request = {
        "question": "What is Polkadot governance?",
        "user_id": "krishna",
        "client_ip": "192.168.1.1"
    }
    print(json.dumps(expected_request, indent=2))
    
    print("\n📋 Expected Response Format:")
    print("-" * 30)
    expected_response = {
        "answer": "Polkadot governance is a sophisticated democracy system...",
        "sources": [
            {
                "title": "Polkadot Governance Overview",
                "url": "https://wiki.polkadot.network/docs/learn-governance",
                "source_type": "polkadot_wiki",
                "similarity_score": 0.85
            }
        ],
        "follow_up_questions": [
            "How do I participate in Polkadot governance?",
            "What are the different types of proposals in Polkadot?",
            "How does the voting mechanism work?"
        ],
        "confidence": 0.85,
        "context_used": True,
        "model_used": "gpt-4",
        "chunks_used": 5,
        "processing_time_ms": 1250,
        "timestamp": "2024-01-15T10:30:45.123Z",
        "search_method": "hybrid_search",
        "remaining_requests": 7
    }
    print(json.dumps(expected_response, indent=2))

if __name__ == "__main__":
    show_expected_format()
    
    print("\n" + "=" * 50)
    success = test_api_format()
    
    if success:
        print("\n🎉 API format verification passed!")
    else:
        print("\n⚠️  API format verification failed!")
        print("Check the server logs for more details.") 