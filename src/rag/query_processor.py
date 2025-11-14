"""
Query processing pipeline for the Polkadot AI Chatbot.
Implements the new routing-first architecture with structured logging.
"""

import logging
import json
import re
import base64
from typing import Dict, Any, Optional, List
from datetime import datetime

from .confidence import compute_retrieval_confidence
from .greeting_handler import handle_generic_query_llm
from .clarification import generate_clarification_question
from .internet_fallback import generate_internet_search_response

logger = logging.getLogger(__name__)


async def combine_query_with_clarification(
    original_query: str,
    clarification_response: str,
    qa_generator,
    log_step
) -> str:
    """
    Intelligently combine the original query with the user's clarification response
    using an LLM to create a better, more coherent combined query.
    
    Args:
        original_query: The original ambiguous query
        clarification_response: The user's response to the clarification question
        qa_generator: QA generator instance
        log_step: Logging function
    
    Returns:
        A combined query that intelligently merges the original query and clarification
    """
    log_step("query_combination_start", {
        "original_query": original_query[:100],
        "clarification_response": clarification_response[:100]
    })
    
    combination_prompt = f"""
You are helping to combine a user's original query with their clarification response.

Original query: "{original_query}"

User's clarification response: "{clarification_response}"

Your task:
- Create a single, clear, and coherent query that combines both the original intent and the clarification
- Make it natural and well-formed
- Preserve the original intent while incorporating the clarification details
- Do NOT add any explanations or meta-commentary - just output the combined query

Example:
Original: "show me proposals"
Clarification: "on Polkadot network"
Combined: "show me proposals on Polkadot network"

Another example:
Original: "what is governance"
Clarification: "I want to see recent proposals"
Combined: "what is governance and show me recent proposals"

Now create the combined query:
"""
    
    try:
        if qa_generator.gemini_client:
            model_name = getattr(qa_generator.gemini_client, 'model_name', 'Gemini')
            log_step("query_combination_llm_call", {"model": model_name})
            
            response = qa_generator.gemini_client.get_response(combination_prompt)
            combined_query = response.strip()
            
            # Remove any quotes if the LLM wrapped the response
            if combined_query.startswith('"') and combined_query.endswith('"'):
                combined_query = combined_query[1:-1]
            elif combined_query.startswith("'") and combined_query.endswith("'"):
                combined_query = combined_query[1:-1]
            
            log_step("query_combination_complete", {
                "combined_query": combined_query[:100]
            })
            
            return combined_query
        else:
            if hasattr(qa_generator, 'client'):
                system_prompt = """You are a query combination assistant. You combine user queries with clarification responses to create coherent, natural queries."""
                
                response = qa_generator.client.chat.completions.create(
                    model=qa_generator.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": combination_prompt}
                    ],
                    temperature=0.3,
                    max_tokens=200
                )
                combined_query = response.choices[0].message.content.strip()
                
                # Remove any quotes if the LLM wrapped the response
                if combined_query.startswith('"') and combined_query.endswith('"'):
                    combined_query = combined_query[1:-1]
                elif combined_query.startswith("'") and combined_query.endswith("'"):
                    combined_query = combined_query[1:-1]
                
                return combined_query
        
        # Fallback to simple concatenation if LLM is not available
        log_step("query_combination_fallback", {"note": "Using simple concatenation fallback"})
        return f"{original_query} {clarification_response}".strip()
        
    except Exception as e:
        log_step("query_combination_error", {"error": str(e)}, "error")
        # Fallback to simple concatenation on error
        return f"{original_query} {clarification_response}".strip()

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


async def detect_and_handle_clarification_response(
    userMessage: str,
    conversationHistory: Optional[List[Dict[str, Any]]],
    qa_generator,
    log_step
) -> Optional[Dict[str, Any]]:
    """
    Detect if the current user message is a response to a clarification question.
    Uses embedded marker in the clarification response for reliable detection.
    
    Args:
        userMessage: The current user message
        conversationHistory: Previous conversation messages
        qa_generator: QA generator instance (not used but kept for compatibility)
        log_step: Logging function
        
    Returns:
        None if not a clarification response, or dict with:
        - combined_query: Original query + clarification response
        - original_query: The original query that needed clarification
        - original_route: The original route that was used
        - clarification_response: The user's clarification response
    """
    if not conversationHistory or len(conversationHistory) < 1:
        return None
    
    # Get the last conversation entry
    last_entry = conversationHistory[-1]
    # Handle both Pydantic models and dicts
    if hasattr(last_entry, 'response'):
        last_response = last_entry.response.strip()
    elif isinstance(last_entry, dict):
        last_response = last_entry.get('response', '').strip()
    else:
        return None
    
    if not last_response:
        return None
    
    # Look for the embedded clarification marker
    # Try new format first: <!--CLARIFICATION_MARKER:base64_encoded_query|route|router_confidence-->
    marker_pattern_new = r'<!--CLARIFICATION_MARKER:([A-Za-z0-9+/=]+)\|([a-z]+)\|([0-9.]+)-->'
    marker_match = re.search(marker_pattern_new, last_response)
    
    if marker_match:
        try:
            # Decode the original query, route, and router confidence from the marker
            encoded_query = marker_match.group(1)
            original_route = marker_match.group(2)
            original_router_confidence = float(marker_match.group(3))
            original_query = base64.b64decode(encoded_query.encode('utf-8')).decode('utf-8')
            
            log_step("clarification_detected_marker", {
                "method": "embedded_marker_new_format",
                "original_query": original_query[:100],
                "original_route": original_route,
                "original_router_confidence": original_router_confidence,
                "clarification_response": userMessage[:100]
            })
            
            # Use LLM to intelligently combine original query with clarification response
            combined_query = await combine_query_with_clarification(
                original_query,
                userMessage,
                qa_generator,
                log_step
            )
            
            return {
                'combined_query': combined_query,
                'original_query': original_query,
                'original_route': original_route,
                'original_router_confidence': original_router_confidence,
                'clarification_response': userMessage
            }
        except Exception as e:
            log_step("clarification_marker_decode_error", {
                "error": str(e),
                "note": "Failed to decode new format marker, trying intermediate format"
            }, "warning")
            marker_match = None
    
    # Fallback to intermediate format (backward compatibility): <!--CLARIFICATION_MARKER:base64_encoded_query|route-->
    if not marker_match:
        marker_pattern_intermediate = r'<!--CLARIFICATION_MARKER:([A-Za-z0-9+/=]+)\|([a-z]+)-->'
        marker_match = re.search(marker_pattern_intermediate, last_response)
        
        if marker_match:
            try:
                # Decode the original query and route from the marker (intermediate format doesn't have router_confidence)
                encoded_query = marker_match.group(1)
                original_route = marker_match.group(2)
                original_query = base64.b64decode(encoded_query.encode('utf-8')).decode('utf-8')
                
                log_step("clarification_detected_marker", {
                    "method": "embedded_marker_intermediate_format",
                    "original_query": original_query[:100],
                    "original_route": original_route,
                    "note": "Intermediate format marker - re-routing to get router confidence",
                    "clarification_response": userMessage[:100]
                })
                
                # Re-route the original query to get router confidence
                route_result = await route_query_llm(
                    original_query,
                    conversationHistory[:-1] if conversationHistory else None,  # Exclude the clarification response
                    qa_generator
                )
                original_router_confidence = route_result.get("confidence", 0.7)
                
                log_step("clarification_intermediate_format_confidence_determined", {
                    "original_query": original_query[:100],
                    "determined_route": original_route,
                    "determined_router_confidence": original_router_confidence
                })
                
                # Use LLM to intelligently combine original query with clarification response
                combined_query = await combine_query_with_clarification(
                    original_query,
                    userMessage,
                    qa_generator,
                    log_step
                )
                
                return {
                    'combined_query': combined_query,
                    'original_query': original_query,
                    'original_route': original_route,
                    'original_router_confidence': original_router_confidence,
                    'clarification_response': userMessage
                }
            except Exception as e:
                log_step("clarification_marker_decode_error", {
                    "error": str(e),
                    "note": "Failed to decode intermediate format marker, trying old format"
                }, "warning")
                marker_match = None
    
    # Fallback to old format (backward compatibility): <!--CLARIFICATION_MARKER:base64_encoded_query-->
    if not marker_match:
        marker_pattern_old = r'<!--CLARIFICATION_MARKER:([A-Za-z0-9+/=]+)-->'
        marker_match = re.search(marker_pattern_old, last_response)
        
        if marker_match:
            try:
                # Decode the original query from the marker (old format doesn't have route or router_confidence)
                encoded_query = marker_match.group(1)
                original_query = base64.b64decode(encoded_query.encode('utf-8')).decode('utf-8')
                
                log_step("clarification_detected_marker", {
                    "method": "embedded_marker_old_format",
                    "original_query": original_query[:100],
                    "note": "Old format marker - re-routing original query to determine route and confidence",
                    "clarification_response": userMessage[:100]
                })
                
                # Re-route the original query to determine what route it should have been
                route_result = await route_query_llm(
                    original_query,
                    conversationHistory[:-1] if conversationHistory else None,  # Exclude the clarification response
                    qa_generator
                )
                original_route = route_result.get("route", "static")
                original_router_confidence = route_result.get("confidence", 0.7)
                
                log_step("clarification_old_format_route_determined", {
                    "original_query": original_query[:100],
                    "determined_route": original_route,
                    "determined_router_confidence": original_router_confidence
                })
                
                # Use LLM to intelligently combine original query with clarification response
                combined_query = await combine_query_with_clarification(
                    original_query,
                    userMessage,
                    qa_generator,
                    log_step
                )
                
                return {
                    'combined_query': combined_query,
                    'original_query': original_query,
                    'original_route': original_route,
                    'original_router_confidence': original_router_confidence,
                    'clarification_response': userMessage
                }
            except Exception as e:
                log_step("clarification_marker_decode_error", {
                    "error": str(e),
                    "note": "Failed to decode old format marker, treating as normal query"
                }, "warning")
                return None
    
    # No marker found - not a clarification response
    return None


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
        # Check if this is a clarification response
        clarification_info = await detect_and_handle_clarification_response(
            userMessage,
            conversationHistory,
            qa_generator,
            log_step
        )
        
        is_clarification_followup = clarification_info is not None
        original_query_for_route = userMessage
        combined_query = userMessage
        stored_route = None
        stored_router_confidence = None
        
        if is_clarification_followup:
            log_step("clarification_followup_detected", {
                "original_query": clarification_info['original_query'],
                "original_route": clarification_info.get('original_route', 'unknown'),
                "original_router_confidence": clarification_info.get('original_router_confidence', 'unknown'),
                "clarification_response": clarification_info['clarification_response'],
                "combined_query_preview": clarification_info['combined_query'][:100]
            })
            # Use the stored original route and router confidence (don't re-route)
            stored_route = clarification_info.get('original_route')
            stored_router_confidence = clarification_info.get('original_router_confidence', 0.7)
            original_query_for_route = clarification_info['original_query']
            combined_query = clarification_info['combined_query']
        
        # Use stored route if available (from clarification followup), otherwise route normally
        if stored_route and stored_route in ['static', 'dynamic', 'hybrid', 'generic']:
            log_step("routing_using_stored_route", {
                "is_clarification_followup": is_clarification_followup,
                "stored_route": stored_route,
                "stored_router_confidence": stored_router_confidence
            })
            route = stored_route
            # Re-route the combined query to get fresh router confidence for the clarified query
            # This gives a more accurate confidence assessment since the query is now clearer
            log_step("routing_combined_query_for_confidence", {
                "combined_query_preview": combined_query[:100],
                "enforced_route": stored_route
            })
            route_result = await route_query_llm(
                combined_query,
                conversationHistory,
                qa_generator
            )
            # Use the fresh router confidence from the clarified query, but enforce the stored route
            fresh_confidence = route_result.get("confidence", 0.7)
            suggested_route = route_result.get("route", stored_route)
            
            if suggested_route != stored_route:
                log_step("routing_route_mismatch", {
                    "stored_route": stored_route,
                    "suggested_route": suggested_route,
                    "note": "Using stored route but fresh confidence from clarified query"
                }, "warning")
            
            confidence = fresh_confidence
        else:
            log_step("routing_start", {
                "is_clarification_followup": is_clarification_followup,
                "query_for_routing": original_query_for_route[:100]
            })
            route_result = await route_query_llm(
                original_query_for_route,
                conversationHistory,
                qa_generator
            )
            route = route_result["route"]
            confidence = route_result["confidence"]
        log_step("routing_complete", {
            "route": route,
            "confidence": confidence,
            "is_clarification_followup": is_clarification_followup
        })
        
        # Use combined query if this is a clarification followup, otherwise use original
        analyzed_query = combined_query if is_clarification_followup else userMessage
        if conversationHistory and qa_generator:
            log_step("query_analysis_start", {})
            try:
                # Use analyzed_query (which may be combined) for memory analysis
                analyzed_query = qa_generator.analyze_query_with_memory(
                    analyzed_query,
                    conversationHistory
                )
                log_step("query_analysis_complete", {
                    "original": userMessage[:100],
                    "analyzed": analyzed_query[:100],
                    "is_clarification_followup": is_clarification_followup
                })
            except Exception as e:
                log_step("query_analysis_error", {"error": str(e)}, "error")
                # Keep the analyzed_query (combined or original) on error
                pass
        
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
            
            retrieval_confidence, semantic_completeness = await compute_retrieval_confidence(
                route=route,
                router_confidence=confidence,
                static_chunks=static_chunks,
                query=analyzed_query,
                qa_generator=qa_generator
            )
            
            # Use semantic_completeness from compute_retrieval_confidence (or default to 0.5)
            if semantic_completeness is None:
                semantic_completeness = 0.5
            
            static_similarity = 0.0
            if static_chunks and len(static_chunks) > 0:
                similarity_scores = [chunk.get('similarity_score', 0.0) for chunk in static_chunks]
                avg_similarity = sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0.0
                max_similarity = max(similarity_scores) if similarity_scores else 0.0
                static_similarity = (avg_similarity * 0.5 + max_similarity * 0.5)
            
            chunk_count_factor = min(len(static_chunks) / 5.0, 1.0) if static_chunks else 0.0
            final_static_confidence = retrieval_confidence
            
            # Decision logic for static route
            decision = None
            if final_static_confidence >= 0.65:
                decision = "ANSWER"
                qa_result = await qa_generator.generate_answer(
                    query=analyzed_query,
                    chunks=static_chunks,
                    custom_prompt=custom_prompt,
                    user_id=user_id,
                    conversation_history=conversationHistory,
                    route=route,
                    route_confidence=confidence
                )
                
                log_step("static_confidence_decision", {
                    "route": "static",
                    "routerConfidence": confidence,
                    "semanticCompleteness": semantic_completeness,
                    "staticSimilarity": static_similarity,
                    "chunkCountFactor": chunk_count_factor,
                    "finalStaticConfidence": final_static_confidence,
                    "decision": decision
                })
                
                log_step("static_route_complete", {
                    "chunks_used": qa_result.get('chunks_used', 0),
                    "search_method": qa_result.get('search_method', 'unknown'),
                    "retrieval_confidence": retrieval_confidence
                })
            elif 0.35 <= final_static_confidence < 0.65:
                decision = "AMBIGUITY_FLOW"
                if not is_clarification_followup:
                    log_step("static_confidence_decision", {
                        "route": "static",
                        "routerConfidence": confidence,
                        "semanticCompleteness": semantic_completeness,
                        "staticSimilarity": static_similarity,
                        "chunkCountFactor": chunk_count_factor,
                        "finalStaticConfidence": final_static_confidence,
                        "decision": decision
                    })
                    
                    clarification_result = await generate_clarification_question(
                        query=userMessage,
                        route=route,
                        router_confidence=confidence,
                        qa_generator=qa_generator,
                        log_step=log_step
                    )
                    
                    clarification_result['route'] = route
                    clarification_result['route_confidence'] = confidence
                    clarification_result['retrievalConfidence'] = retrieval_confidence
                    clarification_result['processing_time_ms'] = (datetime.now() - pipeline_start).total_seconds() * 1000
                    clarification_result['original_query'] = userMessage
                    
                    log_step("pipeline_complete", {
                        "route": route,
                        "confidence": confidence,
                        "retrieval_confidence": retrieval_confidence,
                        "processing_time_ms": clarification_result['processing_time_ms'],
                        "requires_clarification": True,
                        "success": True
                    })
                    
                    return clarification_result
                else:
                    # Already a clarification followup - proceed with answer
                    qa_result = await qa_generator.generate_answer(
                        query=analyzed_query,
                        chunks=static_chunks,
                        custom_prompt=custom_prompt,
                        user_id=user_id,
                        conversation_history=conversationHistory,
                        route=route,
                        route_confidence=confidence
                    )
                    
                    log_step("static_confidence_decision", {
                        "route": "static",
                        "routerConfidence": confidence,
                        "semanticCompleteness": semantic_completeness,
                        "staticSimilarity": static_similarity,
                        "chunkCountFactor": chunk_count_factor,
                        "finalStaticConfidence": final_static_confidence,
                        "decision": decision,
                        "note": "Proceeding despite medium confidence as this is already a clarification followup"
                    })
                    
                    log_step("static_route_complete", {
                        "chunks_used": qa_result.get('chunks_used', 0),
                        "search_method": qa_result.get('search_method', 'unknown'),
                        "retrieval_confidence": retrieval_confidence
                    })
            else:
                decision = "FALLBACK_FLOW"
                log_step("static_confidence_decision", {
                    "route": "static",
                    "routerConfidence": confidence,
                    "semanticCompleteness": semantic_completeness,
                    "staticSimilarity": static_similarity,
                    "chunkCountFactor": chunk_count_factor,
                    "finalStaticConfidence": final_static_confidence,
                    "decision": decision
                })
                
                internet_result = await generate_internet_search_response(
                    query=userMessage,
                    qa_generator=qa_generator,
                    log_step=log_step
                )
                
                internet_result['route'] = route
                internet_result['route_confidence'] = confidence
                internet_result['retrievalConfidence'] = retrieval_confidence
                internet_result['processing_time_ms'] = (datetime.now() - pipeline_start).total_seconds() * 1000
                
                log_step("pipeline_complete", {
                    "route": route,
                    "confidence": confidence,
                    "retrieval_confidence": retrieval_confidence,
                    "processing_time_ms": internet_result['processing_time_ms'],
                    "internet_fallback": True,
                    "success": True
                })
                
                return internet_result
            
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
            
            # Check if query is ambiguous (missing network specification)
            # Ambiguous if: user didn't specify network AND (SQL filtered by network OR SQL returned both networks)
            user_query_lower = userMessage.lower()
            has_network_in_user_query = any(network in user_query_lower for network in ['polkadot', 'kusama', 'dot', 'ksm'])
            
            # Check if SQL query contains network filter
            sql_queries = qa_result.get('sql_query', [])
            sql_has_network_filter = False
            if sql_queries:
                sql_query_str = ' '.join(sql_queries).lower()
                sql_has_network_filter = 'source_network' in sql_query_str and ('polkadot' in sql_query_str or 'kusama' in sql_query_str)
            
            # Check if results contain multiple networks (indicating no network filter was applied)
            # This is a proxy check - if SQL didn't filter by network, results likely contain both
            results_have_multiple_networks = False
            if qa_result.get('success', False) and qa_result.get('result_count', 0) > 0:
                # If SQL doesn't have network filter, assume it returned both networks
                results_have_multiple_networks = not sql_has_network_filter
            
            # Query is ambiguous if:
            # 1. User didn't specify network, AND
            # 2. (SQL filtered by a network user didn't specify OR SQL returned both networks), AND
            # 3. SQL got results
            is_ambiguous_query = (
                not has_network_in_user_query and 
                (sql_has_network_filter or results_have_multiple_networks) and
                qa_result.get('success', False) and 
                qa_result.get('result_count', 0) > 0
            )
            
            if is_ambiguous_query:
                log_step("ambiguous_query_detected", {
                    "query": analyzed_query,
                    "user_query": userMessage,
                    "sql_has_network_filter": sql_has_network_filter,
                    "note": "Query is ambiguous - SQL filtered by network user didn't specify"
                })
            
            retrieval_confidence, _ = await compute_retrieval_confidence(
                route=route,
                router_confidence=confidence,
                sql_result_count=qa_result.get('result_count', 0),
                sql_success=qa_result.get('success', False),
                is_ambiguous_query=is_ambiguous_query,
                query=analyzed_query,
                sql_query=qa_result.get('sql_query', []),
                qa_generator=qa_generator
            )
            
            log_step("dynamic_route_complete", {
                "success": qa_result.get('success', False),
                "result_count": qa_result.get('result_count', 0),
                "search_method": qa_result.get('search_method', 'unknown'),
                "retrieval_confidence": retrieval_confidence,
                "is_ambiguous_query": is_ambiguous_query
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
            
            hybrid_static_available = len(static_chunks) > 0 if static_chunks else False
            
            qa_result = await qa_generator.generate_answer(
                query=analyzed_query,
                chunks=static_chunks,
                custom_prompt=custom_prompt,
                user_id=user_id,
                conversation_history=conversationHistory,
                route=route,
                route_confidence=confidence
            )
            
            hybrid_dynamic_available = qa_result.get('success', False) and qa_result.get('result_count', 0) > 0
            
            retrieval_confidence, _ = await compute_retrieval_confidence(
                route=route,
                router_confidence=confidence,
                static_chunks=static_chunks,
                sql_result_count=qa_result.get('result_count', 0),
                sql_success=qa_result.get('success', False),
                hybrid_static_available=hybrid_static_available,
                hybrid_dynamic_available=hybrid_dynamic_available
            )
            
            log_step("hybrid_route_complete", {
                "chunks_used": qa_result.get('chunks_used', 0),
                "search_method": qa_result.get('search_method', 'unknown'),
                "retrieval_confidence": retrieval_confidence,
                "hybrid_static_available": hybrid_static_available,
                "hybrid_dynamic_available": hybrid_dynamic_available
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
                qa_generator,
                log_step
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
        
        # Static route handles its own decision logic, so skip unified threshold logic
        if route == "static":
            qa_result['route'] = route
            qa_result['route_confidence'] = confidence
            qa_result['retrievalConfidence'] = retrieval_confidence
            qa_result['processing_time_ms'] = processing_time
            if is_clarification_followup and clarification_info:
                qa_result['is_clarification_followup'] = True
                qa_result['original_query'] = clarification_info['original_query']
            
            log_step("pipeline_complete", {
                "route": route,
                "confidence": confidence,
                "retrieval_confidence": retrieval_confidence,
                "processing_time_ms": processing_time,
                "is_clarification_followup": is_clarification_followup,
                "success": True
            })
            
            return qa_result
        
        # retrieval_confidence may have been calculated in route-specific blocks above
        # Check if it exists and is not None, otherwise calculate it
        if route != "generic":
            try:
                # Check if retrieval_confidence was already calculated in route block
                if retrieval_confidence is None:
                    raise NameError("retrieval_confidence is None")
            except NameError:
                # retrieval_confidence doesn't exist or is None, calculate it
                # Static route already calculated it above, so this should only happen for dynamic/hybrid
                if route == "dynamic":
                    retrieval_confidence, _ = await compute_retrieval_confidence(
                        route=route,
                        router_confidence=confidence,
                        sql_result_count=qa_result.get('result_count', 0),
                        sql_success=qa_result.get('success', False),
                        query=analyzed_query,
                        sql_query=qa_result.get('sql_query', []),
                        qa_generator=qa_generator
                    )
                elif route == "hybrid":
                    hybrid_static_available = len(static_chunks) > 0 if 'static_chunks' in locals() and static_chunks else False
                    hybrid_dynamic_available = qa_result.get('success', False) and qa_result.get('result_count', 0) > 0
                    retrieval_confidence, _ = await compute_retrieval_confidence(
                        route=route,
                        router_confidence=confidence,
                        static_chunks=static_chunks if 'static_chunks' in locals() else None,
                        sql_result_count=qa_result.get('result_count', 0),
                        sql_success=qa_result.get('success', False),
                        hybrid_static_available=hybrid_static_available,
                        hybrid_dynamic_available=hybrid_dynamic_available
                    )
                log_step("retrieval_confidence_calculated", {
                    "retrieval_confidence": retrieval_confidence,
                    "route": route
                })
            else:
                # retrieval_confidence was already calculated in route block, use it
                log_step("using_existing_retrieval_confidence", {
                    "retrieval_confidence": retrieval_confidence,
                    "route": route
                })
            
            qa_result['retrievalConfidence'] = retrieval_confidence
            
            # Ensure retrieval_confidence is not None before comparison
            if retrieval_confidence is None:
                log_step("retrieval_confidence_none", {
                    "route": route,
                    "note": "Setting retrieval_confidence to 0.0 as fallback"
                }, "warning")
                retrieval_confidence = 0.0
                qa_result['retrievalConfidence'] = 0.0
            
            # New threshold logic: 0.65 for ANSWER, 0.35 for AMBIGUITY_FLOW, below 0.35 for FALLBACK_FLOW
            # BUT: If we have data but confidence is low, go to AMBIGUITY_FLOW instead of FALLBACK_FLOW
            final_confidence = retrieval_confidence
            
            # Check if we have data available (static route already handled its own logic)
            has_data = False
            if route == "dynamic":
                has_data = qa_result.get('success', False) and qa_result.get('result_count', 0) > 0
            elif route == "hybrid":
                has_data = (
                    (len(static_chunks) > 0 if 'static_chunks' in locals() and static_chunks else False) or
                    (qa_result.get('success', False) and qa_result.get('result_count', 0) > 0)
                )
            
            if final_confidence >= 0.65:
                # ANSWER: High confidence, proceed with answer
                log_step("confidence_high_answer", {
                    "final_confidence": final_confidence,
                    "threshold": 0.65,
                    "action": "ANSWER"
                })
            elif 0.35 <= final_confidence < 0.65:
                # AMBIGUITY_FLOW: Medium confidence, ask for clarification
                # Skip clarification if this is already a clarification followup (to avoid loops)
                if not is_clarification_followup:
                    log_step("confidence_medium_ambiguity", {
                        "final_confidence": final_confidence,
                        "threshold_low": 0.35,
                        "threshold_high": 0.65,
                        "action": "AMBIGUITY_FLOW"
                    })
                    
                    clarification_result = await generate_clarification_question(
                        query=userMessage,
                        route=route,
                        router_confidence=confidence,
                        qa_generator=qa_generator,
                        log_step=log_step
                    )
                    
                    clarification_result['route'] = route
                    clarification_result['route_confidence'] = confidence
                    clarification_result['retrievalConfidence'] = retrieval_confidence
                    clarification_result['processing_time_ms'] = processing_time
                    clarification_result['original_query'] = userMessage
                    
                    log_step("pipeline_complete", {
                        "route": route,
                        "confidence": confidence,
                        "retrieval_confidence": retrieval_confidence,
                        "processing_time_ms": processing_time,
                        "requires_clarification": True,
                        "success": True
                    })
                    
                    return clarification_result
                else:
                    # Already a clarification followup with medium confidence - log but proceed
                    log_step("clarification_followup_medium_confidence", {
                        "final_confidence": final_confidence,
                        "note": "Proceeding despite medium confidence as this is already a clarification followup"
                    }, "warning")
            else:
                # Low confidence (< 0.35)
                # If we have data but confidence is low (ambiguous query), go to AMBIGUITY_FLOW
                # Only use FALLBACK_FLOW if we truly have no data
                if has_data and not is_clarification_followup:
                    # We have data but query is ambiguous - ask for clarification
                    log_step("confidence_low_but_has_data_ambiguity", {
                        "final_confidence": final_confidence,
                        "has_data": has_data,
                        "threshold": 0.35,
                        "action": "AMBIGUITY_FLOW"
                    })
                    
                    clarification_result = await generate_clarification_question(
                        query=userMessage,
                        route=route,
                        router_confidence=confidence,
                        qa_generator=qa_generator,
                        log_step=log_step
                    )
                    
                    clarification_result['route'] = route
                    clarification_result['route_confidence'] = confidence
                    clarification_result['retrievalConfidence'] = retrieval_confidence
                    clarification_result['processing_time_ms'] = processing_time
                    clarification_result['original_query'] = userMessage
                    
                    log_step("pipeline_complete", {
                        "route": route,
                        "confidence": confidence,
                        "retrieval_confidence": retrieval_confidence,
                        "processing_time_ms": processing_time,
                        "requires_clarification": True,
                        "success": True
                    })
                    
                    return clarification_result
                else:
                    # FALLBACK_FLOW: Low confidence and no data, use internet search fallback
                    log_step("confidence_low_fallback", {
                        "final_confidence": final_confidence,
                        "has_data": has_data,
                        "threshold": 0.35,
                        "action": "FALLBACK_FLOW"
                    })
                    
                    internet_result = await generate_internet_search_response(
                        query=userMessage,
                        qa_generator=qa_generator,
                        log_step=log_step
                    )
                    
                    internet_result['route'] = route
                    internet_result['route_confidence'] = confidence
                    internet_result['retrievalConfidence'] = retrieval_confidence
                    internet_result['processing_time_ms'] = processing_time
                    
                    log_step("pipeline_complete", {
                        "route": route,
                        "confidence": confidence,
                        "retrieval_confidence": retrieval_confidence,
                        "processing_time_ms": processing_time,
                        "internet_fallback": True,
                        "success": True
                    })
                    
                    return internet_result
        
        qa_result['route'] = route
        qa_result['route_confidence'] = confidence
        qa_result['processing_time_ms'] = processing_time
        if is_clarification_followup and clarification_info:
            qa_result['is_clarification_followup'] = True
            qa_result['original_query'] = clarification_info['original_query']
        
        log_step("pipeline_complete", {
            "route": route,
            "confidence": confidence,
            "processing_time_ms": processing_time,
            "is_clarification_followup": is_clarification_followup,
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

