"""
Internet search fallback for low-confidence queries with no data.
"""

from typing import Dict, Any, Optional, List
import logging
import os

logger = logging.getLogger(__name__)


async def generate_internet_search_response(
    query: str,
    qa_generator,
    log_step
) -> Dict[str, Any]:
    """
    Generate response using internet search when no data is available.
    
    Args:
        query: The user's query
        qa_generator: QA generator instance
        log_step: Logging function
    
    Returns:
        Dictionary with answer from internet search and metadata
    """
    log_step("internet_fallback_start", {"query_preview": query[:100]})
    
    try:
        from ..utils.web_search import search_tavily
        
        internet_prompt = f"""
You are Klara, an AI-powered governance assistant for Polkadot and Kusama on Polkassembly.

A user has asked: "{query}"

I have no context or data about this query in my system. Please search the internet and provide the best answer you can find.

Important guidelines:
- Provide a helpful, accurate answer based on internet search results
- Focus on Polkadot/Kusama/blockchain governance topics if relevant
- Keep the response concise and informative
- If the query is not related to Polkadot/Kusama, still provide a helpful answer
- Use the search results to inform your response

Search the internet and provide the best answer you can:
"""
        
        web_search_enabled = os.getenv("WEB_SEARCH", "true").lower() == "true"
        
        if web_search_enabled:
            try:
                log_step("internet_search_tavily", {})
                answer, sources = await search_tavily(query, min_score=0.5)
                
                if answer:
                    log_step("internet_search_success", {
                        "answer_length": len(answer),
                        "sources_count": len(sources)
                    })
                    
                    formatted_answer = f"We do not have direct data for this in our system. Here is the best answer I found on the internet:\n\n{answer}"
                    
                    follow_up_questions = [
                        "How does Polkadot's governance system work?",
                        "What are the benefits of staking DOT tokens?",
                        "How do parachains connect to Polkadot?"
                    ]
                    
                    return {
                        'answer': formatted_answer,
                        'sources': sources,
                        'confidence': 0.6,
                        'follow_up_questions': follow_up_questions,
                        'context_used': False,
                        'model_used': 'tavily_web_search',
                        'chunks_used': 0,
                        'search_method': 'internet_search',
                        'internet_fallback': True
                    }
                else:
                    log_step("internet_search_no_results", {})
            except Exception as e:
                log_step("internet_search_error", {"error": str(e)}, "error")
                logger.error(f"Tavily search failed: {e}")
        
        # Fallback to LLM without web search
        log_step("internet_fallback_llm", {})
        
        llm_prompt = f"""
You are Klara, an AI-powered governance assistant for Polkadot and Kusama on Polkassembly.

A user has asked: "{query}"

I have no context or data about this query in my system. Based on your general knowledge, provide the best answer you can.

Important guidelines:
- Provide a helpful, accurate answer based on your knowledge
- Focus on Polkadot/Kusama/blockchain governance topics if relevant
- Keep the response concise and informative
- If you don't know the answer, politely explain that you don't have specific information

Provide the best answer you can:
"""
        
        if qa_generator.gemini_client:
            model_name = getattr(qa_generator.gemini_client, 'model_name', 'Gemini')
            log_step("internet_fallback_llm_call", {"model": model_name})
            
            response = qa_generator.gemini_client.get_response(llm_prompt)
            
            formatted_answer = f"We do not have direct data for this in our system. Here is the best answer I found on the internet:\n\n{response.strip()}"
            
            log_step("internet_fallback_complete", {
                "response_length": len(formatted_answer)
            })
            
            return {
                'answer': formatted_answer,
                'sources': [],
                'confidence': 0.5,
                'follow_up_questions': [
                    "How does Polkadot's governance system work?",
                    "What are the benefits of staking DOT tokens?",
                    "How do parachains connect to Polkadot?"
                ],
                'context_used': False,
                'model_used': model_name,
                'chunks_used': 0,
                'search_method': 'internet_fallback_llm',
                'internet_fallback': True
            }
        else:
            if hasattr(qa_generator, 'client'):
                system_prompt = """You are Klara, an AI-powered governance assistant for Polkadot and Kusama on Polkassembly. 
You help users with questions even when you don't have specific data in your system."""
                
                response = qa_generator.client.chat.completions.create(
                    model=qa_generator.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": llm_prompt}
                    ],
                    temperature=0.7,
                    max_tokens=500
                )
                answer = response.choices[0].message.content
                
                formatted_answer = f"We do not have direct data for this in our system. Here is the best answer I found on the internet:\n\n{answer.strip()}"
                
                return {
                    'answer': formatted_answer,
                    'sources': [],
                    'confidence': 0.5,
                    'follow_up_questions': [
                        "How does Polkadot's governance system work?",
                        "What are the benefits of staking DOT tokens?",
                        "How do parachains connect to Polkadot?"
                    ],
                    'context_used': False,
                    'model_used': qa_generator.model,
                    'chunks_used': 0,
                    'search_method': 'internet_fallback_openai',
                    'internet_fallback': True
                }
        
        fallback_answer = "We do not have direct data for this in our system. I apologize, but I'm unable to provide an answer at this time. Please try rephrasing your question or ask about Polkadot/Kusama governance topics."
        return {
            'answer': fallback_answer,
            'sources': [],
            'confidence': 0.3,
            'follow_up_questions': [
                "How does Polkadot's governance system work?",
                "What are the benefits of staking DOT tokens?",
                "How do parachains connect to Polkadot?"
            ],
            'context_used': False,
            'model_used': 'fallback',
            'chunks_used': 0,
            'search_method': 'internet_fallback_error',
            'internet_fallback': True
        }
        
    except Exception as e:
        log_step("internet_fallback_error", {"error": str(e)}, "error")
        logger.error(f"Internet fallback error: {e}")
        
        fallback_answer = "We do not have direct data for this in our system. I apologize, but I encountered an error while searching for information. Please try again later."
        return {
            'answer': fallback_answer,
            'sources': [],
            'confidence': 0.3,
            'follow_up_questions': [
                "How does Polkadot's governance system work?",
                "What are the benefits of staking DOT tokens?",
                "How do parachains connect to Polkadot?"
            ],
            'context_used': False,
            'model_used': 'error_fallback',
            'chunks_used': 0,
            'search_method': 'internet_fallback_error',
            'internet_fallback': True
        }

