"""
Test script to verify LLM-based table selection
"""
import os
from dotenv import load_dotenv
from src.utils.qa_generator import QAGenerator

load_dotenv()

def test_table_selection():
    """Test the LLM-based table selection"""
    print("=" * 70)
    print("TEST: LLM-Based Table Selection (governance_data vs voting_data)")
    print("=" * 70)
    
    # Initialize QAGenerator
    qa_gen = QAGenerator(
        openai_api_key=os.getenv('OPENAI_API_KEY'),
        model="gpt-3.5-turbo",
        temperature=0.1,
        max_tokens=1000,
        enable_web_search=False,
        enable_memory=False
    )
    
    # Test queries
    test_cases = [
        # Should select governance_data
        ("Show me recent treasury proposals", "governance_data"),
        ("What's the status of proposal 123?", "governance_data"),
        ("Find proposals requesting more than 10000 DOT", "governance_data"),
        ("List all executed referendums", "governance_data"),
        ("Who proposed referendum 456?", "governance_data"),
        ("Show me proposals about DeFi", "governance_data"),
        
        # Should select voting_data
        ("How many people voted on proposal 123?", "voting_data"),
        ("Show me votes with 6x conviction", "voting_data"),
        ("Who voted Aye on referendum 456?", "voting_data"),
        ("List voters with more than 1000 DOT voting power", "voting_data"),
        ("Show delegated votes for proposal 100", "voting_data"),
        ("Count unique voters in the last 30 days", "voting_data"),
        ("What did address 0x123 vote on proposal 50?", "voting_data"),
        
        # Edge cases
        ("Show me treasury proposals and how many people voted", "governance_data"),  # Focus: proposals
        ("List voters who participated in treasury proposals", "voting_data"),  # Focus: voters
    ]
    
    print("\n")
    passed = 0
    failed = 0
    
    for query, expected in test_cases:
        result = qa_gen._determine_table_from_query(query)
        status = "✅" if result == expected else "❌"
        
        if result == expected:
            passed += 1
        else:
            failed += 1
        
        print(f"{status} Query: {query}")
        print(f"   Expected: {expected}, Got: {result}")
        print()
    
    print("=" * 70)
    print(f"RESULTS: {passed} passed, {failed} failed out of {len(test_cases)} tests")
    print("=" * 70)
    
    if failed == 0:
        print("✅ All tests passed!")
    else:
        print(f"⚠️  {failed} test(s) failed")

if __name__ == "__main__":
    try:
        test_table_selection()
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

