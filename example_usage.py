#!/usr/bin/env python3
"""
Example usage of the external embeddings system
"""

import os
import sys

# Add src to path
src_path = os.path.join(os.path.dirname(__file__), 'src')
sys.path.insert(0, src_path)

from src.utils.embeddings import DualEmbeddingManager
from src.rag.config import Config

def main():
    print("=== EXTERNAL EMBEDDINGS SYSTEM EXAMPLE ===")
    
    # Initialize the embedding manager
    embedding_manager = DualEmbeddingManager(
        openai_api_key=Config.OPENAI_API_KEY,
        embedding_model=Config.OPENAI_EMBEDDING_MODEL,
        chroma_persist_directory=Config.CHROMA_PERSIST_DIRECTORY
    )
    
    # Get current collection stats
    stats = embedding_manager.get_collection_stats()
    
    print("\n=== CURRENT COLLECTIONS ===")
    print(f"Static collection: {stats['static_collection']['count']} documents")
    print(f"Dynamic collection: {stats['dynamic_collection']['count']} documents")
    
    if stats['external_collections']:
        print("\nExternal collections:")
        for ext_collection in stats['external_collections']:
            print(f"  - {ext_collection['name']}: {ext_collection['count']} documents")
    else:
        print("\nNo external collections found")
    
    print(f"\nTotal documents: {stats['total_documents']}")
    
    # Example search
    print("\n=== EXAMPLE SEARCH ===")
    query = "What is Polkadot?"
    results = embedding_manager.search_similar_chunks(query, n_results=3)
    
    print(f"Query: '{query}'")
    print(f"Found {len(results)} results:")
    
    for i, result in enumerate(results, 1):
        print(f"\n{i}. Collection: {result['collection']}")
        print(f"   Source: {result['source']}")
        print(f"   Score: {result['score']:.4f}")
        print(f"   Content: {result['content'][:200]}...")
        if result['metadata'].get('title'):
            print(f"   Title: {result['metadata']['title']}")

if __name__ == "__main__":
    main() 