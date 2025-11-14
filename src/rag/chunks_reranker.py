#!/usr/bin/env python3
"""
Chunks Reranker - Logic to prioritize chunks based on content quality and image presence
"""

import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

def rerank_static_chunks(static_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Rerank static chunks with priority order:
    1. Polkassembly chunks (highest priority)
    2. S3 image chunks (if no Polkassembly chunks)
    3. All chunks (if no qualifying chunks)
    
    Args:
        static_chunks: List of static chunk dictionaries
    
    Returns:
        List of reranked/filtered chunks with Polkassembly priority, then S3 image priority
    """
    if not static_chunks:
        return static_chunks
    
    s3_bucket_url = "https://polkassembly-ai.s3.us-east-1.amazonaws.com"
    
    # Get the highest similarity score from all chunks
    all_scores = [chunk.get('similarity_score', 0) for chunk in static_chunks]
    highest_score = max(all_scores) if all_scores else 0
    
    logger.info(f"Highest similarity score among all chunks: {highest_score:.3f}")
    
    # PRIORITY 1: Find chunks mentioning Polkassembly
    chunks_with_polkassembly = []
    
    for chunk in static_chunks:
        content = chunk.get('content', '').lower()
        metadata = chunk.get('metadata', {})
        title = metadata.get('title', '').lower()
        url = metadata.get('url', '').lower()
        source = metadata.get('source', '').lower()
        
        # Check if chunk mentions polkassembly in content, title, URL, or source
        if ('polkassembly' in content or 
            'polkassembly' in title or 
            'polkassembly' in url or 
            'polkassembly' in source):
            chunks_with_polkassembly.append(chunk)
            similarity_score = chunk.get('similarity_score', 0)
            chunk_title = metadata.get('title', 'Unknown')
            logger.info(f"Found chunk with Polkassembly mention: '{chunk_title}' (score: {similarity_score:.3f})")
    
    # If Polkassembly chunks found, check if any qualify (within 5 points of highest score)
    if chunks_with_polkassembly:
        qualifying_polkassembly_chunks = []
        
        for chunk in chunks_with_polkassembly:
            chunk_score = chunk.get('similarity_score', 0)
            score_difference = highest_score - chunk_score
            
            if score_difference <= 0.05:  # Within 5 points
                qualifying_polkassembly_chunks.append(chunk)
                title = chunk.get('metadata', {}).get('title', 'Unknown')
                logger.info(f"Polkassembly chunk qualifies: '{title}' (score: {chunk_score:.3f}, diff: {score_difference:.3f})")
            else:
                title = chunk.get('metadata', {}).get('title', 'Unknown')
                logger.info(f"Polkassembly chunk rejected: '{title}' (score: {chunk_score:.3f}, diff: {score_difference:.3f})")
        
        # If we have qualifying Polkassembly chunks, return only those (highest priority)
        if qualifying_polkassembly_chunks:
            logger.info(f"Prioritizing {len(qualifying_polkassembly_chunks)} qualifying Polkassembly chunks out of {len(static_chunks)} total chunks")
            return qualifying_polkassembly_chunks
    
    # PRIORITY 2: If no qualifying Polkassembly chunks, check for S3 image chunks
    chunks_with_images = []
    
    for chunk in static_chunks:
        content = chunk.get('content', '')
        if s3_bucket_url in content:
            chunks_with_images.append(chunk)
            similarity_score = chunk.get('similarity_score', 0)
            title = chunk.get('metadata', {}).get('title', 'Unknown')
            logger.info(f"Found chunk with S3 image link: '{title}' (score: {similarity_score:.3f})")
    
    # If S3 image chunks found, check if any qualify
    if chunks_with_images:
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
    
    # PRIORITY 3: If no qualifying chunks, return all chunks
    logger.info(f"No qualifying Polkassembly or S3 image chunks found. Returning all {len(static_chunks)} chunks")
    return static_chunks
