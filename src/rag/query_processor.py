"""
Query processing pipeline for the Polkadot AI Chatbot.
Implements the new routing-first architecture with structured logging.
"""

import logging
import json
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)

def log_step(step_name: str, data: Dict[str, Any], level: str = "info"):
    """Log a pipeline step with structured data"""
    log_data = {
        "step": step_name,
        "timestamp": datetime.now().isoformat(),
        **data
    }
    if level == "info":
        logger.info(f"[{step_name}] {json.dumps(log_data, default=str)}")
    elif level == "warning":
        logger.warning(f"[{step_name}] {json.dumps(log_data, default=str)}")
    elif level == "error":
        logger.error(f"[{step_name}] {json.dumps(log_data, default=str)}")
    elif level == "debug":
        logger.debug(f"[{step_name}] {json.dumps(log_data, default=str)}")


async def route_query_llm(
    query: str,
    conversation_history: Optional[List[Dict[str, Any]]],
    qa_generator
) -> Dict[str, Any]:
    """
    Route query using LLM to determine the appropriate route.
    
    Returns:
        {
            "route": "static" | "dynamic" | "hybrid" | "generic",
            "confidence": float (0.0-1.0)
        }
    """
    log_step("router_llm_start", {"query_preview": query[:100]})
    
    try:
        routing_prompt = f"""
You are a query router. Analyze this user query and determine the best route for answering it.

Query: "{query}"

Available Routes:
1. "static" - For general educational/informational topics:
   - Governance/OpenGov concepts and explanations
   - Ambassador Programme information
   - Parachains & AnV explanations
   - Hyperbridge, JAM definitions
   - Dashboard information
   - Wiki pages, how-to guides, tutorials
   - General "what is" or "how does" questions without specific data requests
   - "How to" questions about using Polkassembly features

2. "dynamic" - For blockchain data queries requiring on-chain data:
   - Proposal information (title, content, status, dates, network)
   - Proposal metadata (type, proposer, beneficiary, amounts)
   - Voting data (voter information, voting power, decisions)
   - Proposal filtering by ID, dates, network, type, status
   - Aggregations, counts, or summaries of on-chain data
   - Keywords: "proposal", "referendum", "bounty", "treasury", "voter", "vote"
   - Questions asking for specific data from the blockchain

3. "hybrid" - For queries that need both static context and dynamic data:
   - Questions that require explaining concepts AND showing specific data
   - Example: "What is OpenGov and show me recent proposals"

4. "generic" - For queries that don't fit the above categories:
   - Greetings (hi, hello, hey, greetings, etc.)
   - Casual conversation and small talk
   - Questions completely outside Polkadot/blockchain domain
   - Requests for general help or introduction
   - Ambiguous or unclear queries that can't be categorized

CRITICAL: You MUST respond with ONLY valid JSON. No explanations, no text before or after. Just the JSON object.

Example responses:
{{"route": "static", "confidence": 0.9}}
{{"route": "dynamic", "confidence": 0.85}}
{{"route": "generic", "confidence": 0.7}}

Now respond with ONLY the JSON for this query:
"""
        
        if qa_generator.gemini_client:
            model_name = getattr(qa_generator.gemini_client, 'model_name', 'Gemini')
            log_step("router_llm_call", {"model": model_name})
            
            response = qa_generator.gemini_client.get_response(routing_prompt)
            
            result = None
            
            try:
                result = json.loads(response.strip())
            except json.JSONDecodeError:
                import re
                json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
                if json_match:
                    try:
                        result = json.loads(json_match.group(1))
                    except json.JSONDecodeError:
                        pass
                
                if not result:
                    json_match = re.search(r'\{[^{}]*"route"[^{}]*"confidence"[^{}]*\}', response)
                    if json_match:
                        try:
                            result = json.loads(json_match.group(0))
                        except json.JSONDecodeError:
                            pass
                
                if not result:
                    route_match = re.search(r'"route"\s*:\s*"([^"]+)"', response, re.IGNORECASE)
                    confidence_match = re.search(r'"confidence"\s*:\s*([0-9.]+)', response, re.IGNORECASE)
                    if route_match:
                        route_str = route_match.group(1).lower()
                        confidence_val = float(confidence_match.group(1)) if confidence_match else 0.5
                        if route_str in ['static', 'dynamic', 'hybrid', 'generic']:
                            result = {"route": route_str, "confidence": confidence_val}
            
            if result:
                route = result.get('route', 'static').lower()
                confidence = float(result.get('confidence', 0.5))
                
                if route not in ['static', 'dynamic', 'hybrid', 'generic']:
                    log_step("router_llm_invalid_route", {"route": route}, "warning")
                    route = 'static'
                    confidence = 0.5
                
                confidence = max(0.0, min(1.0, confidence))
                
                log_step("router_llm_complete", {
                    "route": route,
                    "confidence": confidence
                })
                
                return {
                    "route": route,
                    "confidence": confidence
                }
            else:
                log_step("router_llm_parse_error", {
                    "error": "Could not extract JSON from response",
                    "response_preview": response[:200]
                }, "error")
                
                query_lower = query.lower()
                
                dynamic_keywords = ['proposal', 'referendum', 'bounty', 'treasury', 'voter', 'vote', 'show me', 'list', 'count', 'how many']
                if any(keyword in query_lower for keyword in dynamic_keywords):
                    route = "dynamic"
                elif any(phrase in query_lower for phrase in ['how to', 'how can i', 'what is', 'how does', 'explain', 'tutorial', 'guide']):
                    route = "static"
                elif any(word in query_lower for word in ['hi', 'hello', 'hey', 'greetings']):
                    route = "generic"
                else:
                    route = "static"
                
                log_step("router_llm_fallback_inference", {
                    "route": route,
                    "reason": "json_parse_failed_using_query_analysis"
                }, "warning")
                
                return {
                    "route": route,
                    "confidence": 0.5
                }
        else:
            log_step("router_llm_fallback", {"reason": "no_gemini_client"}, "warning")
            return {
                "route": "static",
                "confidence": 0.5
            }
            
    except Exception as e:
        log_step("router_llm_error", {"error": str(e)}, "error")
        return {
            "route": "static",
            "confidence": 0.3
        }


async def handle_dynamic_query_sql(
    query: str,
    conversation_history: Optional[List[Dict[str, Any]]],
    qa_generator
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


def _is_greeting_query(query: str) -> bool:
    greeting_keywords = [
        'hi', 'hello', 'hey', 'greetings', 'good morning', 'good afternoon', 
        'good evening', 'what\'s up', 'whats up', 'wassup', 'howdy',
        'intro', 'introduction', 'who are you', 'what are you',
        'what do you do', 'what is this', 'help', 'start'
    ]
    
    query_lower = query.lower().strip()
    
    for keyword in greeting_keywords:
        if query_lower == keyword:
            return True
    
    if len(query_lower.split()) <= 3:
        for keyword in greeting_keywords:
            if keyword in query_lower:
                data_indicators = ['number', 'count', 'total', 'sum', 'average', 'highest', 'lowest', 
                                   'vote', 'votes', 'voter', 'proposal', 'referendum', 'treasury',
                                   'month', 'year', 'date', 'time', 'day', 'week']
                has_data_indicator = any(indicator in query_lower for indicator in data_indicators)
                if not has_data_indicator:
                    return True
    
    return False


def _get_polkassembly_introduction() -> Dict[str, Any]:
    introduction = """Hello! I'm **Klara** 👋 – your AI-powered governance assistant for **Polkadot** and **Kusama**!

I'm here to help you explore the governance ecosystem through **Polkassembly**, making it easy to query on-chain data, analyze proposals, and understand the voting process—all in natural language.

**What I can help you with:**

🗳 **Governance Data** - Query proposals, referenda, bounties, and treasury activities

📊 **Voting Analysis** - Track voter behavior, delegation, and voting power

💰 **Treasury Insights** - Explore funding proposals and beneficiary data

🧭 **Platform Guidance** - Learn how to use Polkassembly features and OpenGov

**Who I'm built for:**

- Community members exploring proposals
- Delegates analyzing voting patterns  
- Builders tracking treasury activities
- Researchers studying governance trends

**Try asking me things like:**

- "Show all active referenda on Polkadot"
- "Who voted on referendum 472?"
- "List treasury proposals above 100k DOT"
- "How does conviction voting work?"

**Useful Links:**

- **Klara Guide**: [klara.polkassembly.io/guide](https://klara.polkassembly.io/guide) - If you want detailed guidance on how to use Klara, follow this doc
- **Polkadot Governance**: [polkadot.polkassembly.io](https://polkadot.polkassembly.io)
- **Kusama Governance**: [kusama.polkassembly.io](https://kusama.polkassembly.io)
- **Documentation**: [docs.polkassembly.io](https://docs.polkassembly.io)"""

    sources = [
        {
            'title': 'Polkassembly Main Platform',
            'url': 'https://polkassembly.io',
            'source_type': 'platform',
            'similarity_score': 1.0
        },
        {
            'title': 'Polkadot Governance on Polkassembly',
            'url': 'https://polkadot.polkassembly.io',
            'source_type': 'platform',
            'similarity_score': 1.0
        },
        {
            'title': 'Kusama Governance on Polkassembly',
            'url': 'https://kusama.polkassembly.io',
            'source_type': 'platform',
            'similarity_score': 1.0
        },
        {
            'title': 'Polkassembly Documentation',
            'url': 'https://docs.polkassembly.io',
            'source_type': 'documentation',
            'similarity_score': 1.0
        }
    ]

    greeting_follow_ups = [
        "How does Polkadot governance work?",
        "What are parachains and how do they work?",
        "How can I start staking DOT tokens?"
    ]
    
    return {
        'answer': introduction,
        'sources': sources,
        'confidence': 1.0,
        'follow_up_questions': greeting_follow_ups,
        'context_used': True,
        'model_used': 'polkassembly_intro',
        'chunks_used': 0,
        'search_method': 'greeting_response'
    }


async def handle_generic_query_llm(
    query: str,
    conversation_history: Optional[List[Dict[str, Any]]],
    qa_generator
) -> Dict[str, Any]:
    """
    Handle generic queries (greetings, non-Polkadot queries).
    For greetings, returns the exact same hardcoded response as before.
    For other generic queries, uses LLM.
    
    Returns:
        Dictionary with answer, sources, and metadata for generic responses
    """
    log_step("generic_handler_start", {"query_preview": query[:100]})
    
    is_greeting = _is_greeting_query(query)
    log_step("generic_handler_greeting_check", {"is_greeting": is_greeting})
    
    if is_greeting:
        log_step("generic_handler_greeting_detected", {})
        return _get_polkassembly_introduction()
    
    log_step("generic_handler_non_greeting", {"note": "Using LLM for non-greeting generic query"})
    try:
        generic_prompt = f"""
You are Klara, an AI-powered governance assistant for Polkadot and Kusama on Polkassembly.

The user has sent this query: "{query}"

This query has been identified as a generic query, which could be:
- A question outside the Polkadot/blockchain domain
- A request for general help or introduction
- An ambiguous query

Your task:
1. If it's a non-Polkadot question, politely redirect to Polkadot-related topics
2. If it's a request for help, provide helpful guidance about what you can do
3. Keep responses concise, friendly, and professional

Important guidelines:
- For non-Polkadot topics: Politely explain that you specialize in Polkadot/Kusama governance
- Always maintain a helpful and friendly tone
- If appropriate, suggest relevant Polkadot-related questions they could ask

Respond with a natural, conversational answer that addresses the user's query appropriately.
"""
        
        if qa_generator.gemini_client:
            model_name = getattr(qa_generator.gemini_client, 'model_name', 'Gemini')
            log_step("generic_handler_llm_call", {"model": model_name})
            
            response = qa_generator.gemini_client.get_response(generic_prompt)
            
            log_step("generic_handler_complete", {
                "response_length": len(response)
            })
            
            follow_up_questions = [
                "How does Polkadot's governance system work?",
                "What are the benefits of staking DOT tokens?",
                "How do parachains connect to Polkadot?"
            ]
            
            sources = [
                {
                    'title': 'Polkassembly Main Platform',
                    'url': 'https://polkassembly.io',
                    'source_type': 'platform',
                    'similarity_score': 1.0
                },
                {
                    'title': 'Polkadot Governance on Polkassembly',
                    'url': 'https://polkadot.polkassembly.io',
                    'source_type': 'platform',
                    'similarity_score': 1.0
                }
            ]
            
            return {
                'answer': response.strip(),
                'sources': sources,
                'confidence': 0.8,
                'follow_up_questions': follow_up_questions,
                'context_used': False,
                'model_used': model_name,
                'chunks_used': 0,
                'search_method': 'generic_llm_response'
            }
        else:
            log_step("generic_handler_fallback", {"reason": "no_gemini_client"}, "warning")
            
            if hasattr(qa_generator, 'client'):
                try:
                    system_prompt = """You are Klara, an AI-powered governance assistant for Polkadot and Kusama on Polkassembly. 
You help users with Polkadot governance questions, but you can also handle greetings and general queries in a friendly, helpful manner."""
                    
                    response = qa_generator.client.chat.completions.create(
                        model=qa_generator.model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": generic_prompt}
                        ],
                        temperature=0.7,
                        max_tokens=300
                    )
                    answer = response.choices[0].message.content
                    
                    return {
                        'answer': answer.strip(),
                        'sources': [],
                        'confidence': 0.7,
                        'follow_up_questions': [
                            "How does Polkadot's governance system work?",
                            "What are the benefits of staking DOT tokens?",
                            "How do parachains connect to Polkadot?"
                        ],
                        'context_used': False,
                        'model_used': qa_generator.model,
                        'chunks_used': 0,
                        'search_method': 'generic_llm_fallback'
                    }
                except Exception as e:
                    log_step("generic_handler_error", {"error": str(e)}, "error")
            
            return {
                'answer': "Hello! I'm Klara, your AI assistant for Polkadot and Kusama governance. I can help you with questions about proposals, voting, treasury, and more. What would you like to know?",
                'sources': [],
                'confidence': 0.5,
                'follow_up_questions': [
                    "How does Polkadot's governance system work?",
                    "What are the benefits of staking DOT tokens?",
                    "How do parachains connect to Polkadot?"
                ],
                'context_used': False,
                'model_used': 'fallback',
                'chunks_used': 0,
                'search_method': 'generic_fallback'
            }
            
    except Exception as e:
        log_step("generic_handler_error", {"error": str(e)}, "error")
        return {
            'answer': "Hello! I'm Klara, your AI assistant for Polkadot and Kusama governance. How can I help you today?",
            'sources': [],
            'confidence': 0.5,
            'follow_up_questions': [
                "How does Polkadot's governance system work?",
                "What are the benefits of staking DOT tokens?",
                "How do parachains connect to Polkadot?"
            ],
            'context_used': False,
            'model_used': 'error_fallback',
            'chunks_used': 0,
            'search_method': 'generic_error'
        }


async def processUserQuery(
    userMessage: str,
    conversationHistory: Optional[List[Dict[str, Any]]],
    static_embedding_manager,
    dynamic_embedding_manager,
    qa_generator,
    max_chunks: int = 5,
    custom_prompt: Optional[str] = None,
    user_id: str = "default_user"
) -> Dict[str, Any]:
    """
    Main entry point for processing user queries.
    Implements the new routing-first architecture.
    
    Args:
        userMessage: The user's query
        conversationHistory: Previous conversation messages
        static_embedding_manager: Manager for static embeddings
        dynamic_embedding_manager: Manager for dynamic embeddings
        qa_generator: QA generator instance
        max_chunks: Maximum number of chunks to retrieve
        custom_prompt: Optional custom system prompt
        user_id: User identifier
        
    Returns:
        Dictionary with answer, sources, and metadata
    """
    pipeline_start = datetime.now()
    log_step("pipeline_start", {
        "user_id": user_id,
        "query_preview": userMessage[:100],
        "has_history": bool(conversationHistory)
    })
    
    try:
        log_step("routing_start", {})
        route_result = await route_query_llm(
            userMessage,
            conversationHistory,
            qa_generator
        )
        route = route_result["route"]
        confidence = route_result["confidence"]
        log_step("routing_complete", {
            "route": route,
            "confidence": confidence
        })
        
        analyzed_query = userMessage
        if conversationHistory and qa_generator:
            log_step("query_analysis_start", {})
            try:
                analyzed_query = qa_generator.analyze_query_with_memory(
                    userMessage,
                    conversationHistory
                )
                log_step("query_analysis_complete", {
                    "original": userMessage[:100],
                    "analyzed": analyzed_query[:100]
                })
            except Exception as e:
                log_step("query_analysis_error", {"error": str(e)}, "error")
                analyzed_query = userMessage
        
        processing_time = (datetime.now() - pipeline_start).total_seconds() * 1000
        
        if route == "static":
            log_step("static_route_start", {})
            
            static_chunks = static_embedding_manager.search_similar_chunks(
                query=analyzed_query,
                n_results=max_chunks
            )
            from .chunks_reranker import rerank_static_chunks
            static_chunks = rerank_static_chunks(static_chunks)
            log_step("static_retrieval_complete", {"chunks_count": len(static_chunks)})
            
            qa_result = await qa_generator.generate_answer(
                query=analyzed_query,
                chunks=static_chunks,
                custom_prompt=custom_prompt,
                user_id=user_id,
                conversation_history=conversationHistory,
                route=route,
                route_confidence=confidence
            )
            
            log_step("static_route_complete", {
                "chunks_used": qa_result.get('chunks_used', 0),
                "search_method": qa_result.get('search_method', 'unknown')
            })
            
        elif route == "dynamic":
            log_step("dynamic_route_start", {})
            
            qa_result = await qa_generator.generate_answer(
                query=analyzed_query,
                chunks=[],
                custom_prompt=custom_prompt,
                user_id=user_id,
                conversation_history=conversationHistory,
                route=route,
                route_confidence=confidence
            )
            
            log_step("dynamic_route_complete", {
                "success": qa_result.get('success', False),
                "result_count": qa_result.get('result_count', 0),
                "search_method": qa_result.get('search_method', 'unknown')
            })
            
        elif route == "hybrid":
            log_step("hybrid_route_start", {})
            
            static_chunks = static_embedding_manager.search_similar_chunks(
                query=analyzed_query,
                n_results=max_chunks
            )
            from .chunks_reranker import rerank_static_chunks
            static_chunks = rerank_static_chunks(static_chunks)
            log_step("hybrid_static_retrieval_complete", {"chunks_count": len(static_chunks)})
            
            qa_result = await qa_generator.generate_answer(
                query=analyzed_query,
                chunks=static_chunks,
                custom_prompt=custom_prompt,
                user_id=user_id,
                conversation_history=conversationHistory,
                route=route,
                route_confidence=confidence
            )
            
            log_step("hybrid_route_complete", {
                "chunks_used": qa_result.get('chunks_used', 0),
                "search_method": qa_result.get('search_method', 'unknown')
            })
            
        elif route == "generic":
            log_step("generic_route_start", {})
            
            if qa_generator.memory_manager and qa_generator.memory_manager.enabled:
                try:
                    qa_generator.memory_manager.add_user_query(analyzed_query, user_id)
                except Exception as e:
                    log_step("memory_add_error", {"error": str(e)}, "warning")
            
            qa_result = await handle_generic_query_llm(
                analyzed_query,
                conversationHistory,
                qa_generator
            )
            
            if qa_generator.memory_manager and qa_generator.memory_manager.enabled:
                try:
                    qa_generator.memory_manager.add_assistant_response(
                        qa_result.get('answer', ''),
                        user_id
                    )
                except Exception as e:
                    log_step("memory_add_error", {"error": str(e)}, "warning")
            
            log_step("generic_route_complete", {
                "search_method": qa_result.get('search_method', 'unknown')
            })
        
        processing_time = (datetime.now() - pipeline_start).total_seconds() * 1000
        qa_result['route'] = route
        qa_result['route_confidence'] = confidence
        qa_result['processing_time_ms'] = processing_time
        
        log_step("pipeline_complete", {
            "route": route,
            "confidence": confidence,
            "processing_time_ms": processing_time,
            "success": True
        })
        
        return qa_result
        
    except Exception as e:
        processing_time = (datetime.now() - pipeline_start).total_seconds() * 1000
        log_step("pipeline_error", {
            "error": str(e),
            "processing_time_ms": processing_time
        }, "error")
        
        return {
            'answer': "I encountered an error processing your query. Please try again.",
            'sources': [],
            'chunks_used': 0,
            'search_method': 'error',
            'route': 'generic',
            'route_confidence': 0.0,
            'processing_time_ms': processing_time,
            'error': str(e)
        }

