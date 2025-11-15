"""
Clarification handler for low-confidence queries.
"""

from typing import Dict, Any, Optional, List
import logging
import base64
import re

logger = logging.getLogger(__name__)


async def generate_clarification_question(
    query: str,
    route: str,
    router_confidence: float,
    qa_generator,
    log_step
) -> Dict[str, Any]:
    """
    Generate a clarifying question when retrieval confidence is low.
    
    Args:
        query: The original user query
        route: The route that was selected
        router_confidence: The router's confidence level
        qa_generator: QA generator instance
        log_step: Logging function
    
    Returns:
        Dictionary with clarification question and metadata
    """
    log_step("clarification_start", {
        "query_preview": query[:100],
        "route": route,
        "router_confidence": router_confidence
    })
    
    # Build context-aware clarification prompt based on route
    route_context = ""
    if route == "dynamic":
        route_context = """
This is a dynamic/on-chain data query. Common ambiguities include:
- Network selection (Polkadot vs Kusama)
- Proposal/referendum type or status
- Time period or date range
- Specific filters (active, passed, rejected, etc.)

For queries about proposals, referenda, votes, or treasury data, the most common ambiguity is which network (Polkadot or Kusama).
"""
    elif route == "static":
        route_context = """
This is a static/educational query. Common ambiguities include:
- Specific topic or concept within the broader subject
- Level of detail needed (overview vs deep dive)
- Specific use case or scenario
"""
    elif route == "hybrid":
        route_context = """
This is a hybrid query needing both explanation and data. Common ambiguities include:
- Network selection (Polkadot vs Kusama) for the data portion
- Scope of explanation vs data requested
"""
    
    # Use LLM to dynamically generate context-aware clarification
    clarification_prompt = f"""
You are Klara, an AI-powered governance assistant for Polkadot and Kusama on Polkassembly.

The user asked: "{query}"

This query was routed to the "{route}" category. The query is ambiguous and needs clarification.

{route_context}

CRITICAL INSTRUCTIONS:
- You MUST ask ONE specific clarifying question based on the EXACT terms used in the user's query
- DO NOT use generic phrases like "Could you please provide more details" or "What specific information"
- Be DIRECT and SPECIFIC - use the SAME terminology the user used (e.g., if they said "referenda", say "referenda", not "proposals")
- Match the query's language and terminology exactly
- For queries about governance data (proposals, referenda, votes, treasury, bounties) without network specification: Ask which network (Polkadot or Kusama)
- For queries about votes without proposal/referendum ID: Ask which specific proposal/referendum
- Be natural and conversational

EXAMPLES:
- Query: "show me proposals" → Response: "Are you looking for proposals on Polkadot or Kusama network?"
- Query: "show me active referenda" → Response: "Are you looking for referenda on Polkadot or Kusama network?"
- Query: "list referenda" → Response: "Are you looking for referenda on Polkadot or Kusama network?"
- Query: "what are the votes" → Response: "Which proposal or referendum are you asking about? Please provide the ID or title."
- Query: "treasury data" → Response: "Are you asking about Polkadot or Kusama treasury proposals?"
- Query: "show me bounties" → Response: "Are you looking for bounties on Polkadot or Kusama network?"

Now, for the query "{query}", respond with ONLY the clarifying question (no explanations, no extra text). Use the exact same terminology the user used:
"""
    
    try:
        if qa_generator.gemini_client:
            model_name = getattr(qa_generator.gemini_client, 'model_name', 'Gemini')
            log_step("clarification_llm_call", {"model": model_name})
            
            response = qa_generator.gemini_client.get_response(clarification_prompt)
            clarification_question = response.strip()
            
            # Clean up the response - remove quotes if LLM wrapped it
            if clarification_question.startswith('"') and clarification_question.endswith('"'):
                clarification_question = clarification_question[1:-1]
            elif clarification_question.startswith("'") and clarification_question.endswith("'"):
                clarification_question = clarification_question[1:-1]
        elif hasattr(qa_generator, 'client'):
            system_prompt = """You are Klara, an AI-powered governance assistant for Polkadot and Kusama on Polkassembly. 
You help clarify ambiguous user queries by asking one specific question. Always use the exact same terminology the user used in their query."""
            
            response = qa_generator.client.chat.completions.create(
                model=qa_generator.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": clarification_prompt}
                ],
                temperature=0.7,
                max_tokens=150
            )
            clarification_question = response.choices[0].message.content.strip()
        else:
            raise Exception("No LLM client available")
    except Exception as e:
        log_step("clarification_llm_error", {"error": str(e)}, "error")
        # Minimal fallback - still try to be specific
        if route == "dynamic":
            clarification_question = "Are you looking for this information on Polkadot or Kusama network?"
        else:
            clarification_question = "Could you please provide more details about what you're looking for?"
    
    # Embed hidden marker with original query, route, and router confidence for later detection
    # Format: <!--CLARIFICATION_MARKER:base64_encoded_query|route|router_confidence-->
    encoded_query = base64.b64encode(query.encode('utf-8')).decode('utf-8')
    marker = f"<!--CLARIFICATION_MARKER:{encoded_query}|{route}|{router_confidence:.3f}-->"
    clarification_question_with_marker = f"{clarification_question}\n{marker}"
    
    log_step("clarification_complete", {
        "question": clarification_question[:100],
        "method": "llm_generated"
    })
    
    model_name = getattr(qa_generator.gemini_client, 'model_name', 'Gemini') if qa_generator.gemini_client else (qa_generator.model if hasattr(qa_generator, 'model') else 'fallback')
    
    return {
        'answer': clarification_question_with_marker,
        'sources': [],
        'confidence': router_confidence,
        'follow_up_questions': [],
        'context_used': False,
        'model_used': model_name,
        'chunks_used': 0,
        'search_method': 'clarification',
        'requires_clarification': True,
        'clarification_question': clarification_question
    }

