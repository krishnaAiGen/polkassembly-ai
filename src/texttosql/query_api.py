"""
Simple API wrapper for the Query2SQL system
"""
import json
import sys
import os
from typing import Optional, List, Dict, Any
from datetime import datetime

try:
    # Try relative import first (when imported as module)
    from .query2sql import Query2SQL
except ImportError:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, current_dir)
    from query2sql import Query2SQL


# Add SlackBot for error notifications
try:
    # Try relative import first (when imported as module)
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
    from slack_bot import SlackBot
except ImportError:
    SlackBot = None

def send_error_to_slack(query: str, error: str, error_source: str = "Query2SQL") -> None:
    """Send error notification to Slack channel"""
    if not SlackBot:
        return
    
    try:
        slack_bot = SlackBot()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
        error_context = {
            "query": query,
            "timestamp": timestamp,
            "error_source": error_source,
            "error_details": str(error)
        }
        
        slack_bot.post_error_to_slack(
            "Query processing failed", 
            context=error_context
        )
        print("Error notification sent to Slack")
    except Exception as slack_error:
        print(f"Failed to send error notification to Slack: {slack_error}")

def ask_question(question: str, conversation_history: Optional[List[Dict[str, Any]]] = None) -> dict:
    """
    Simple function to ask a natural language question and get results
    
    Args:
        question (str): Natural language question about governance data
        
    Returns:
        dict: Response containing SQL query, results, and natural language answer
    """
    try:
        # Initialize the query processor
        processor = Query2SQL()
        
        # Process the question with conversation history
        result = processor.process_query(question, conversation_history)
        
        return result
        
    except Exception as e:
        # Send error notification to Slack
        send_error_to_slack(question, str(e), "Query2SQL")
        
        return {
            "original_query": question,
            "sql_query": None,
            "result_count": 0,
            "results": [],
            "columns": [],
            "natural_response": f"I'm having trouble processing your query. Please try again or rephrase your question in the next prompt",
            "success": False,
            "error": str(e)
        }

def main():
   
    question = "How much they ask in clarys resubmission proposal"
    # question = "did polkassembly submitted any proposal in 2025?"
    # question = "give me proosals which were submitted last week"
    # question = "is there any proposal who asked for more than 1 million usd"
    # question = "how many proposal were submitted in the month of August 2025. can you name few?"
    # question = "unique accounts list who have voted in last 60 days"
    result = ask_question(question)
    print(f"\nQuery: {question}")
    print(f"SQL Queries: {result.get('sql_queries', [])}")
    print(f"Result Count: {result.get('result_count', 0)}")
    print(f"Natural Response: {result.get('natural_response', 'No response')}")
    print(f"Success: {result.get('success', False)}")
    
if __name__ == "__main__":
    main()





#The query returned a total of 10 results, however, it seems that the specific amounts requested in these proposals are not available (represented as \'nan\' in the data). \n\nLet\'s look at the three most relevant proposals:\n\n1. The proposal titled "SmallTipper" with ID 502 was executed on March 6, 2025. Unfortunately, the requested amount is not specified in the data.\n2. Another proposal, also titled "SmallTipper" with ID 496, was executed on February 19, 2025. Again, the requested amount is not specified.\n3. A proposal titled "Test" with ID 467 timed out and was not executed. This proposal was submitted on November 5, 2024, and the requested amount is also not specified.\n\nUnfortunately, based on the available data, we cannot determine if any of these proposals requested more than 1 million USD. To get a more accurate answer, we would need more detailed data about the requested amounts in these proposals. \n\nRemember, in Polkadot/Kusama governance, proposals are a crucial part of the decision-making process. They allow network participants to suggest changes and improvements, which are then voted on by the community.