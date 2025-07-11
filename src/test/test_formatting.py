#!/usr/bin/env python3
"""
Test script to verify clean formatting without markdown symbols.
"""

import os
import sys
import requests
import json
import re

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

API_BASE_URL = "http://localhost:8000"

def check_markdown_symbols(text: str) -> list:
    """Check for forbidden markdown symbols in text"""
    forbidden_patterns = [
        (r'\*\*.*?\*\*', 'Bold markdown (**text**)'),
        (r'\*.*?\*(?!\*)', 'Italic markdown (*text*)'),
        (r'__.*?__', 'Bold markdown (__text__)'),
        (r'_.*?_(?!_)', 'Italic markdown (_text_)'),
        (r'^#{1,6}\s', 'Headers (# ## ###)'),
        (r'```.*?```', 'Code blocks (```code```)'),
        (r'`.*?`', 'Inline code (`code`)'),
        (r'\[.*?\]\(.*?\)', 'Links ([text](url))'),
        (r'->', 'Arrow symbols (->)'),
        (r'•', 'Bullet symbols (•)'),
        (r'~~.*?~~', 'Strikethrough (~~text~~)'),
    ]
    
    found_issues = []
    for pattern, description in forbidden_patterns:
        matches = re.findall(pattern, text, re.MULTILINE | re.DOTALL)
        if matches:
            found_issues.append(f"{description}: {matches[:3]}")  # Show first 3 matches
    
    return found_issues

def test_staking_formatting():
    """Test formatting for a staking question"""
    print("Testing formatting for staking question...")
    
    query = {
        "question": "How do I stake DOT tokens on Polkassembly?",
        "max_chunks": 3,
        "include_sources": True
    }
    
    try:
        response = requests.post(f"{API_BASE_URL}/query", json=query)
        response.raise_for_status()
        data = response.json()
        
        answer = data['answer']
        print(f"✅ Received answer ({len(answer)} characters)")
        print(f"First 200 chars: {answer[:200]}...")
        
        # Check for markdown symbols
        issues = check_markdown_symbols(answer)
        
        if not issues:
            print("✅ Clean formatting - no markdown symbols found!")
            
            # Check for proper formatting
            has_numbered_lists = bool(re.search(r'^\d+\.', answer, re.MULTILINE))
            has_bullet_points = bool(re.search(r'^- ', answer, re.MULTILINE))
            
            print(f"   Numbered lists: {'✅' if has_numbered_lists else '❌'}")
            print(f"   Bullet points: {'✅' if has_bullet_points else '❌'}")
            
            return True
        else:
            print("❌ Formatting issues found:")
            for issue in issues:
                print(f"   - {issue}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def test_governance_formatting():
    """Test formatting for a governance question"""
    print("\nTesting formatting for governance question...")
    
    query = {
        "question": "What are the steps to participate in Polkadot governance?",
        "max_chunks": 3,
        "include_sources": True
    }
    
    try:
        response = requests.post(f"{API_BASE_URL}/query", json=query)
        response.raise_for_status()
        data = response.json()
        
        answer = data['answer']
        print(f"✅ Received answer ({len(answer)} characters)")
        print(f"First 200 chars: {answer[:200]}...")
        
        # Check for markdown symbols
        issues = check_markdown_symbols(answer)
        
        if not issues:
            print("✅ Clean formatting - no markdown symbols found!")
            return True
        else:
            print("❌ Formatting issues found:")
            for issue in issues:
                print(f"   - {issue}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def test_greeting_formatting():
    """Test formatting for greeting response"""
    print("\nTesting formatting for greeting...")
    
    query = {
        "question": "Hello",
        "max_chunks": 1,
        "include_sources": True
    }
    
    try:
        response = requests.post(f"{API_BASE_URL}/query", json=query)
        response.raise_for_status()
        data = response.json()
        
        answer = data['answer']
        print(f"✅ Received greeting ({len(answer)} characters)")
        print(f"First 200 chars: {answer[:200]}...")
        
        # Check for markdown symbols
        issues = check_markdown_symbols(answer)
        
        if not issues:
            print("✅ Clean formatting - no markdown symbols found!")
            return True
        else:
            print("❌ Formatting issues found:")
            for issue in issues:
                print(f"   - {issue}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def test_health_check():
    """Test if the API is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        response.raise_for_status()
        data = response.json()
        return data['status'] == 'healthy'
    except:
        return False

def main():
    """Run formatting tests"""
    print("🎨 Testing Clean Formatting (No Markdown)")
    print("=" * 50)
    
    # Check if API is running
    if not test_health_check():
        print("❌ API is not ready for testing")
        return
    
    # Run formatting tests
    staking_clean = test_staking_formatting()
    governance_clean = test_governance_formatting() 
    greeting_clean = test_greeting_formatting()
    
    print("\n" + "=" * 50)
    print("📊 Formatting Test Results:")
    print(f"   Staking Question: {'✅ Clean' if staking_clean else '❌ Has Markdown'}")
    print(f"   Governance Question: {'✅ Clean' if governance_clean else '❌ Has Markdown'}")
    print(f"   Greeting Response: {'✅ Clean' if greeting_clean else '❌ Has Markdown'}")
    
    if staking_clean and governance_clean and greeting_clean:
        print("\n✅ All formatting tests passed! Responses are clean and professional.")
    else:
        print("\n⚠️  Some responses contain markdown formatting. Check the system prompts.")
        print("\nExpected format example:")
        print("To stake DOT tokens:")
        print("")
        print("1. Create and fund your wallet")
        print("2. Access Polkassembly website") 
        print("3. Select reliable validators")
        print("4. Nominate your chosen validators")
        print("5. Monitor your staking rewards")
        print("")
        print("Key benefits:")
        print("- Earn passive income through rewards")
        print("- Support network security")

if __name__ == "__main__":
    main() 