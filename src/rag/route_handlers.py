"""
Route-specific handlers for different query types.
"""

from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)


async def handle_dynamic_query_sql(
    query: str,
    conversation_history: Optional[List[Dict[str, Any]]],
    qa_generator,
    log_step
) -> Dict[str, Any]:
    """
    Handle dynamic queries (on-chain data) using SQL generation.
    
    Returns:
        Dictionary with answer, sources, and metadata for SQL-based responses
    """
    log_step("dynamic_handler_start", {"query_preview": query[:100]})
    
    try:
        table = qa_generator._determine_table_from_query(query)
        log_step("dynamic_table_determined", {"table": table})
        
        from ..texttosql.query_api import ask_question
        
        sql_result = ask_question(query, conversation_history, table)
        
        log_step("dynamic_sql_complete", {
            "success": sql_result.get('success', False),
            "result_count": sql_result.get('result_count', 0)
        })
        
        return {
            'answer': sql_result.get('natural_response', 'No response available'),
            'sources': [],
            'chunks_used': 0,
            'search_method': 'sql_query',
            'sql_query': sql_result.get('sql_queries', []),
            'result_count': sql_result.get('result_count', 0),
            'success': sql_result.get('success', False),
            'confidence': 0.9 if sql_result.get('success', False) else 0.5
        }
        
    except Exception as e:
        log_step("dynamic_handler_error", {"error": str(e)}, "error")
        return {
            'answer': "I'm sorry, I encountered an error processing your database query. Please try rephrasing your question or try again later.",
            'sources': [],
            'chunks_used': 0,
            'search_method': 'sql_error_fallback',
            'sql_query': [],
            'result_count': 0,
            'success': False,
            'confidence': 0.0,
            'follow_up_questions': [
                "How does Polkadot's governance system work?",
                "What are the benefits of staking DOT tokens?",
                "How do parachains connect to Polkadot?"
            ]
        }


