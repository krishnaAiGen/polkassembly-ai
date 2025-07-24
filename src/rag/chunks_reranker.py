#!/usr/bin/env python3
"""
Chunks Reranker - Logic to prioritize chunks based on content quality and image presence
"""

import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

def rerank_static_chunks(static_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Check if any chunk contains S3 image links and has good similarity scores.
    If S3 chunks are found and their similarity scores are within 5 points of the highest score,
    return only those chunks. Otherwise, return all chunks.
    
    Args:
        static_chunks: List of static chunk dictionaries
        
    Returns:
        List of reranked/filtered chunks with image priority based on similarity scores
    """
    if not static_chunks:
        return static_chunks
    
    s3_bucket_url = "https://polkassembly-ai.s3.us-east-1.amazonaws.com"
    
    # Find chunks containing S3 image links
    chunks_with_images = []
    
    for chunk in static_chunks:
        content = chunk.get('content', '')
        if s3_bucket_url in content:
            chunks_with_images.append(chunk)
            similarity_score = chunk.get('similarity_score', 0)
            title = chunk.get('metadata', {}).get('title', 'Unknown')
            logger.info(f"Found chunk with S3 image link: '{title}' (score: {similarity_score:.3f})")
    
    # If no image chunks found, return all chunks
    if not chunks_with_images:
        logger.info(f"No chunks with images found. Returning all {len(static_chunks)} chunks")
        return static_chunks
    
    # Get the highest similarity score from all chunks
    all_scores = [chunk.get('similarity_score', 0) for chunk in static_chunks]
    highest_score = max(all_scores) if all_scores else 0
    
    logger.info(f"Highest similarity score among all chunks: {highest_score:.3f}")
    
    # Check if any S3 chunk has similarity score within 5 points of the highest score
    qualifying_image_chunks = []
    
    for chunk in chunks_with_images:
        chunk_score = chunk.get('similarity_score', 0)
        score_difference = highest_score - chunk_score
        
        if score_difference <= 0.05:  # Within 5 points
            qualifying_image_chunks.append(chunk)
            title = chunk.get('metadata', {}).get('title', 'Unknown')
            logger.info(f"S3 chunk qualifies: '{title}' (score: {chunk_score:.3f}, diff: {score_difference:.3f})")
        else:
            title = chunk.get('metadata', {}).get('title', 'Unknown')
            logger.info(f"S3 chunk rejected: '{title}' (score: {chunk_score:.3f}, diff: {score_difference:.3f})")
    
    # If we have qualifying image chunks, return only those
    if qualifying_image_chunks:
        logger.info(f"Prioritizing {len(qualifying_image_chunks)} qualifying image chunks out of {len(static_chunks)} total chunks")
        return qualifying_image_chunks
    
    # If no qualifying image chunks, return all chunks
    logger.info(f"No qualifying image chunks found (score difference > 5). Returning all {len(static_chunks)} chunks")
    return static_chunks 
