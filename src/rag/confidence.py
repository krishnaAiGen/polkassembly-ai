"""
Confidence computation utilities for retrieval quality assessment.
"""

import re
import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


async def getSemanticCompletenessScore(query: str, qa_generator) -> float:
    """
    Rate how specific a query is for SQL generation from 0.0 (very vague) to 1.0 (highly specific).
    
    Args:
        query: The user query to evaluate
        qa_generator: QA generator instance with LLM access
    
    Returns:
        Float score between 0.0 and 1.0
    """
    prompt = f"""Rate how specific this query is for SQL generation from 0.0 (very vague) to 1.0 (highly specific). Return only the number.

Query: {query}"""
    
    try:
        if qa_generator.gemini_client:
            response = qa_generator.gemini_client.get_response(prompt)
        elif hasattr(qa_generator, 'client'):
            response_obj = qa_generator.client.chat.completions.create(
                model=qa_generator.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=10
            )
            response = response_obj.choices[0].message.content.strip()
        else:
            logger.warning("No LLM client available for semantic completeness scoring")
            return 0.5
        
        # Extract number from response
        response = response.strip()
        # Remove any quotes
        response = response.strip('"').strip("'")
        # Extract first float number
        match = re.search(r'([0-9]*\.?[0-9]+)', response)
        if match:
            score = float(match.group(1))
            # Clamp to [0, 1]
            score = max(0.0, min(1.0, score))
            return score
        else:
            logger.warning(f"Could not parse semantic completeness score from response: {response}")
            return 0.5
    except Exception as e:
        logger.error(f"Error computing semantic completeness score: {e}")
        return 0.5


async def getSemanticCompletenessScoreForStatic(query: str, qa_generator) -> float:
    """
    Rate how specific a query is for static documentation lookup from 0.0 (very vague) to 1.0 (highly specific).
    
    Args:
        query: The user query to evaluate
        qa_generator: QA generator instance with LLM access
    
    Returns:
        Float score between 0.0 and 1.0
    """
    prompt = f"""Rate how specific this query is for static documentation lookup from 0.0 (very vague) to 1.0 (highly specific). Return ONLY the number.

Query: {query}"""
    
    try:
        if qa_generator.gemini_client:
            response = qa_generator.gemini_client.get_response(prompt)
        elif hasattr(qa_generator, 'client'):
            response_obj = qa_generator.client.chat.completions.create(
                model=qa_generator.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=10
            )
            response = response_obj.choices[0].message.content.strip()
        else:
            logger.warning("No LLM client available for static semantic completeness scoring")
            return 0.5
        
        # Extract number from response
        response = response.strip()
        # Remove any quotes
        response = response.strip('"').strip("'")
        # Extract first float number
        match = re.search(r'([0-9]*\.?[0-9]+)', response)
        if match:
            score = float(match.group(1))
            # Clamp to [0, 1]
            score = max(0.0, min(1.0, score))
            return score
        else:
            logger.warning(f"Could not parse static semantic completeness score from response: {response}")
            return 0.5
    except Exception as e:
        logger.error(f"Error computing static semantic completeness score: {e}")
        return 0.5


def getSQLPrecisionScore(sql: str) -> float:
    """
    Score SQL query precision based on filtering criteria.
    
    Args:
        sql: SQL query string
    
    Returns:
        Float score between 0.0 and 1.0
    """
    if not sql:
        return 0.0
    
    sql_lower = sql.lower()
    score = 0.0
    
    # +0.3 if SQL contains WHERE
    if 'where' in sql_lower:
        score += 0.3
    
    # +0.3 if SQL filters proposal_index
    if 'proposal_index' in sql_lower and 'where' in sql_lower:
        score += 0.3
    
    # +0.2 if SQL filters network (source_network)
    if 'source_network' in sql_lower and 'where' in sql_lower:
        score += 0.2
    
    # +0.1 if SQL filters onchaininfo_status
    if 'onchaininfo_status' in sql_lower and 'where' in sql_lower:
        score += 0.1
    
    # -0.3 if SQL uses SELECT * without WHERE
    if 'select *' in sql_lower and 'where' not in sql_lower:
        score -= 0.3
    
    # -0.3 if SQL uses LIMIT without WHERE
    if 'limit' in sql_lower and 'where' not in sql_lower:
        score -= 0.3
    
    # Clamp to [0, 1]
    return max(0.0, min(1.0, score))


async def compute_retrieval_confidence(
    route: str,
    router_confidence: float,
    static_chunks: Optional[List[Dict[str, Any]]] = None,
    sql_result_count: Optional[int] = None,
    sql_success: Optional[bool] = None,
    hybrid_static_available: Optional[bool] = None,
    hybrid_dynamic_available: Optional[bool] = None,
    file_fallback_available: bool = False,
    is_ambiguous_query: bool = False,
    query: Optional[str] = None,
    sql_query: Optional[List[str]] = None,
    qa_generator = None
) -> tuple[float, Optional[float]]:
    """
    Compute retrieval confidence based on multiple factors.
    
    Args:
        route: The route type (static, dynamic, hybrid, generic)
        router_confidence: Confidence from the router LLM (0.0-1.0)
        static_chunks: List of static chunks with similarity scores
        sql_result_count: Number of rows returned from SQL query
        sql_success: Whether SQL query was successful
        hybrid_static_available: Whether static data is available in hybrid route
        hybrid_dynamic_available: Whether dynamic data is available in hybrid route
        file_fallback_available: Whether file fallback API is available
        is_ambiguous_query: Whether the query is ambiguous
        query: The user query (required for dynamic route)
        sql_query: List of SQL query strings (required for dynamic route)
        qa_generator: QA generator instance (required for dynamic route)
    
    Returns:
        Tuple of (retrieval confidence score (0.0-1.0), semantic_completeness for static route or None)
    """
    base_confidence = router_confidence * 0.25
    
    file_fallback_bonus = 0.05 if file_fallback_available else 0.0
    
    if route == "static":
        semantic_completeness = 0.5
        if query and qa_generator:
            semantic_completeness = await getSemanticCompletenessScoreForStatic(query, qa_generator)
        
        static_similarity = 0.0
        if static_chunks and len(static_chunks) > 0:
            similarity_scores = [chunk.get('similarity_score', 0.0) for chunk in static_chunks]
            avg_similarity = sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0.0
            max_similarity = max(similarity_scores) if similarity_scores else 0.0
            static_similarity = (avg_similarity * 0.5 + max_similarity * 0.5)
        
        chunk_count_factor = min(len(static_chunks) / 5.0, 1.0) if static_chunks else 0.0
        
        static_confidence = (
            0.30 * router_confidence +
            0.40 * semantic_completeness +
            0.20 * static_similarity +
            0.10 * chunk_count_factor
        )
        
        ambiguity_penalty = 0.0
        if semantic_completeness < 0.45:
            ambiguity_penalty = -0.4
        
        final_static_confidence = static_confidence + ambiguity_penalty
        
        return (max(0.0, min(1.0, final_static_confidence)), semantic_completeness)
    
    elif route == "dynamic":
        semantic_completeness = 0.5
        if query and qa_generator:
            semantic_completeness = await getSemanticCompletenessScore(query, qa_generator)
        
        sql_precision = 0.0
        if sql_query and len(sql_query) > 0:
            combined_sql = ' '.join(sql_query)
            sql_precision = getSQLPrecisionScore(combined_sql)
        
        result_specificity = 0.0
        if sql_success and sql_result_count is not None:
            if sql_result_count > 0:
                result_specificity = min(sql_result_count / 10.0, 1.0)
            else:
                result_specificity = 0.1
        
        # If we have successful SQL results, trust the router confidence more
        # Successful results indicate the query was clear enough to execute
        has_results = sql_success and sql_result_count and sql_result_count > 0
        
        ambiguity_penalty = 0.0
        if is_ambiguous_query:
            ambiguity_penalty = -0.3
        
        # Adjust weights based on whether we have results
        # If we have results, trust router_confidence more (it was right about routing)
        if has_results:
            # High router confidence + results = query was clear enough
            final_confidence = (
                0.50 * router_confidence +
                0.20 * semantic_completeness +
                0.15 * sql_precision +
                0.15 * result_specificity +
                file_fallback_bonus +
                ambiguity_penalty
            )
        else:
            # No results - be more cautious, weight semantic completeness more
            final_confidence = (
                0.35 * router_confidence +
                0.35 * semantic_completeness +
                0.20 * sql_precision +
                0.10 * result_specificity +
                file_fallback_bonus +
                ambiguity_penalty
            )
        
        return (max(0.0, min(1.0, final_confidence)), None)
    
    elif route == "hybrid":
        static_confidence = 0.0
        if hybrid_static_available and static_chunks and len(static_chunks) > 0:
            similarity_scores = [chunk.get('similarity_score', 0.0) for chunk in static_chunks]
            avg_similarity = sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0.0
            static_confidence = avg_similarity * 0.25
        
        dynamic_confidence = 0.0
        if hybrid_dynamic_available:
            if sql_result_count and sql_result_count > 0:
                row_count_factor = min(sql_result_count / 10.0, 1.0)
                dynamic_confidence = 0.25 + (row_count_factor * 0.15)
            else:
                dynamic_confidence = 0.15
        else:
            dynamic_confidence = 0.05
        
        completeness_bonus = 0.0
        if hybrid_static_available and hybrid_dynamic_available:
            completeness_bonus = 0.1
        
        return (min(base_confidence + static_confidence + dynamic_confidence + completeness_bonus + file_fallback_bonus, 1.0), None)
    
    return (base_confidence, None)

