"""
Convenience helpers for dynamic embeddings indexing

Simple functions that can be imported and used from other modules
"""

import logging
from typing import Optional, Dict
from .index_dynamic_embeddings import DynamicEmbeddingsIndexer

logger = logging.getLogger(__name__)


def index_all(limit: Optional[int] = None) -> Dict[str, int]:
    """
    Index all governance and voting data into dynamic Chroma collection
    
    Args:
        limit: Optional limit on rows per table (for testing)
        
    Returns:
        Dict with counts: {"governance_indexed": int, "voting_indexed": int}
        
    Example:
        >>> from src.data.index_helpers import index_all
        >>> results = index_all(limit=100)
        >>> print(f"Indexed {results['governance_indexed']} governance rows")
    """
    try:
        indexer = DynamicEmbeddingsIndexer()
        return indexer.index_all_dynamic_embeddings(limit=limit)
    except Exception as e:
        logger.error(f"Failed to index dynamic embeddings: {e}")
        raise


def index_governance(limit: Optional[int] = None) -> int:
    """
    Index only governance data into dynamic Chroma collection
    
    Args:
        limit: Optional limit on rows
        
    Returns:
        Number of rows indexed
        
    Example:
        >>> from src.data.index_helpers import index_governance
        >>> count = index_governance(limit=50)
        >>> print(f"Indexed {count} governance rows")
    """
    try:
        indexer = DynamicEmbeddingsIndexer()
        return indexer.index_governance_dynamic_embeddings(limit=limit)
    except Exception as e:
        logger.error(f"Failed to index governance data: {e}")
        raise


def index_voting(limit: Optional[int] = None) -> int:
    """
    Index only voting data into dynamic Chroma collection
    
    Args:
        limit: Optional limit on rows
        
    Returns:
        Number of rows indexed
        
    Example:
        >>> from src.data.index_helpers import index_voting
        >>> count = index_voting(limit=50)
        >>> print(f"Indexed {count} voting rows")
    """
    try:
        indexer = DynamicEmbeddingsIndexer()
        return indexer.index_voting_dynamic_embeddings(limit=limit)
    except Exception as e:
        logger.error(f"Failed to index voting data: {e}")
        raise


def get_collection_stats() -> Dict[str, any]:
    """
    Get statistics about the dynamic collection
    
    Returns:
        Dict with collection stats
        
    Example:
        >>> from src.data.index_helpers import get_collection_stats
        >>> stats = get_collection_stats()
        >>> print(f"Collection has {stats['total_docs']} documents")
    """
    try:
        indexer = DynamicEmbeddingsIndexer()
        collection = indexer.get_dynamic_collection()
        
        total_docs = collection.count()
        
        # Get sample to count by doc_type
        sample = collection.get(
            limit=min(1000, total_docs),
            include=["metadatas"]
        )
        
        gov_count = 0
        vote_count = 0
        
        if sample and 'metadatas' in sample:
            for metadata in sample['metadatas']:
                doc_type = metadata.get('doc_type', '')
                if doc_type == 'governance':
                    gov_count += 1
                elif doc_type == 'vote':
                    vote_count += 1
        
        return {
            'total_docs': total_docs,
            'governance_docs': gov_count,
            'vote_docs': vote_count,
            'collection_name': indexer.embedding_manager.collection_name
        }
    except Exception as e:
        logger.error(f"Failed to get collection stats: {e}")
        raise


def verify_indexing(test_limit: int = 5) -> bool:
    """
    Verify that the dynamic collection is working properly
    
    Args:
        test_limit: Number of test queries to run
        
    Returns:
        True if verification passed, False otherwise
        
    Example:
        >>> from src.data.index_helpers import verify_indexing
        >>> if verify_indexing():
        ...     print("Collection is working!")
    """
    try:
        indexer = DynamicEmbeddingsIndexer()
        collection = indexer.get_dynamic_collection()
        
        # Check if collection has docs
        count = collection.count()
        if count == 0:
            logger.warning("Collection is empty")
            return False
        
        # Test query
        results = collection.query(
            query_texts=["recent proposals"],
            n_results=min(test_limit, count)
        )
        
        if not results or not results.get('documents'):
            logger.warning("Query returned no results")
            return False
        
        logger.info(f"✅ Verification passed: {count} docs, query works")
        return True
        
    except Exception as e:
        logger.error(f"Verification failed: {e}")
        return False


