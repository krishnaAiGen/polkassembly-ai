#!/usr/bin/env python3
"""
Migration script to convert from single collection to multi-collection setup
"""

import sys
import os

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .embeddings import EmbeddingManager, MultiCollectionEmbeddingManager
from .data_loader import DataLoader
from .text_chunker import TextChunker
from ..rag.config import Config

def migrate_to_multi_collection():
    """
    Migrate from single collection to multi-collection setup
    """
    print("=== MIGRATION TO MULTI-COLLECTION SETUP ===")
    
    # Initialize old single collection manager
    print("\n1. Checking existing single collection...")
    old_manager = EmbeddingManager(
        openai_api_key=Config.OPENAI_API_KEY,
        embedding_model=Config.OPENAI_EMBEDDING_MODEL,
        chroma_persist_directory=Config.CHROMA_PERSIST_DIRECTORY,
        collection_name=Config.CHROMA_COLLECTION_NAME
    )
    
    if not old_manager.collection_exists():
        print("❌ No existing single collection found. Nothing to migrate.")
        return
    
    # Get existing data
    print("2. Retrieving existing data from single collection...")
    old_stats = old_manager.get_collection_stats()
    print(f"   Found {old_stats['total_chunks']} chunks")
    print(f"   Sources: {old_stats['chunks_by_source']}")
    
    # Get all data from old collection
    all_data = old_manager.collection.get(include=["documents", "metadatas"])
    
    # Initialize new multi-collection manager
    print("\n3. Initializing new multi-collection setup...")
    new_manager = MultiCollectionEmbeddingManager(
        openai_api_key=Config.OPENAI_API_KEY,
        embedding_model=Config.OPENAI_EMBEDDING_MODEL,
        chroma_persist_directory=Config.CHROMA_PERSIST_DIRECTORY,
        static_collection_name="polkadot_static",
        onchain_collection_name="polkadot_onchain"
    )
    
    # Separate data by source
    print("\n4. Separating data by source...")
    static_chunks = []
    onchain_chunks = []
    
    for i, (document, metadata) in enumerate(zip(all_data['documents'], all_data['metadatas'])):
        chunk = {
            'content': document,
            'metadata': metadata
        }
        
        source = metadata.get('source', 'unknown')
        if source == 'polkassembly':
            onchain_chunks.append(chunk)
        else:
            static_chunks.append(chunk)
    
    print(f"   Static chunks: {len(static_chunks)}")
    print(f"   Onchain chunks: {len(onchain_chunks)}")
    
    # Add to new collections
    print("\n5. Adding data to new collections...")
    
    if static_chunks:
        print("   Adding static chunks...")
        success = new_manager.add_static_chunks_to_collection(static_chunks)
        if success:
            print(f"   ✅ Successfully added {len(static_chunks)} static chunks")
        else:
            print(f"   ❌ Failed to add static chunks")
    
    if onchain_chunks:
        print("   Adding onchain chunks...")
        success = new_manager.add_onchain_chunks_to_collection(onchain_chunks)
        if success:
            print(f"   ✅ Successfully added {len(onchain_chunks)} onchain chunks")
        else:
            print(f"   ❌ Failed to add onchain chunks")
    
    # Verify migration
    print("\n6. Verifying migration...")
    new_stats = new_manager.get_collection_stats()
    print(f"   New total chunks: {new_stats['total_chunks']}")
    print(f"   Static collection: {new_stats['static_collection']['chunks']} chunks")
    print(f"   Onchain collection: {new_stats['onchain_collection']['chunks']} chunks")
    
    if new_stats['total_chunks'] == old_stats['total_chunks']:
        print("   ✅ Migration successful - chunk counts match!")
        
        # Ask if user wants to remove old collection
        response = input("\n7. Remove old single collection? (y/N): ")
        if response.lower() == 'y':
            old_manager.clear_collection()
            print("   ✅ Old collection removed")
        else:
            print("   ⚠️  Old collection kept (you can remove it manually later)")
    else:
        print("   ❌ Migration issue - chunk counts don't match!")
        print(f"   Old: {old_stats['total_chunks']}, New: {new_stats['total_chunks']}")
    
    print("\n=== MIGRATION COMPLETE ===")
    print("You can now use the multi-collection setup!")

if __name__ == "__main__":
    migrate_to_multi_collection() 