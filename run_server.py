#!/usr/bin/env python3
"""
Entry point script for running the API server with dual embedding system.
"""

import os
import sys

# Add src to path
src_path = os.path.join(os.path.dirname(__file__), 'src')
sys.path.insert(0, src_path)

def initialize_embeddings():
    """Initialize embeddings for both static and dynamic data sources"""
    from src.rag.config import Config
    from src.utils.data_loader import DataLoader
    from src.utils.text_chunker import TextChunker
    from src.utils.embeddings import DualEmbeddingManager
    
    print("=== INITIALIZING DUAL EMBEDDING SYSTEM ===")
    
    # Validate configuration
    try:
        Config.validate_config()
        print("✓ Configuration validated")
    except ValueError as e:
        print(f"✗ Configuration error: {e}")
        return False
    
    # Initialize data loader
    data_loader = DataLoader(
        static_data_path=Config.STATIC_DATA_SOURCE,
        dynamic_data_path=Config.DYNAMIC_DATA_SOURCE
    )
    
    # Get data status
    data_status = data_loader.get_data_status()
    print(f"\nData Source Status:")
    print(f"  Static data: {data_status['static_data']['file_count']} .txt files in {data_status['static_data']['path']}")
    print(f"  Dynamic data: {data_status['dynamic_data']['file_count']} .json files in {data_status['dynamic_data']['path']}")
    
    # Initialize embedding manager
    embedding_manager = DualEmbeddingManager(
        openai_api_key=Config.OPENAI_API_KEY,
        embedding_model=Config.OPENAI_EMBEDDING_MODEL,
        chroma_persist_directory=Config.CHROMA_PERSIST_DIRECTORY
    )
    
    # Get current collection stats
    current_stats = embedding_manager.get_collection_stats()
    print(f"\nCurrent Embedding Status:")
    print(f"  Static collection: {current_stats['static_collection']['count']} documents")
    print(f"  Dynamic collection: {current_stats['dynamic_collection']['count']} documents")
    
    # Initialize text chunker
    text_chunker = TextChunker(
        chunk_size=Config.CHUNK_SIZE,
        chunk_overlap=Config.CHUNK_OVERLAP
    )
    
    # Process static data if needed
    if current_stats['static_collection']['count'] == 0 and data_status['static_data']['file_count'] > 0:
        print("\n=== PROCESSING STATIC DATA ===")
        
        # Load static data
        static_data = data_loader.load_static_data()
        
        if static_data:
            # Chunk the static data
            print(f"Chunking {len(static_data)} static documents...")
            static_chunks = []
            
            for doc in static_data:
                doc_chunks = text_chunker.chunk_document(doc)
                static_chunks.extend(doc_chunks)
            
            print(f"Created {len(static_chunks)} static chunks")
            
            # Add to static collection
            result = embedding_manager.add_static_documents(static_chunks)
            print(f"✓ {result['message']}")
        else:
            print("⚠ No static data found to process")
    
    # Process dynamic data if needed
    if current_stats['dynamic_collection']['count'] == 0 and data_status['dynamic_data']['file_count'] > 0:
        print("\n=== PROCESSING DYNAMIC DATA ===")
        
        # Load dynamic data
        dynamic_data = data_loader.load_dynamic_data()
        
        if dynamic_data:
            # Chunk the dynamic data
            print(f"Chunking {len(dynamic_data)} dynamic documents...")
            dynamic_chunks = []
            
            for doc in dynamic_data:
                doc_chunks = text_chunker.chunk_document(doc)
                dynamic_chunks.extend(doc_chunks)
            
            print(f"Created {len(dynamic_chunks)} dynamic chunks")
            
            # Add to dynamic collection
            result = embedding_manager.add_dynamic_documents(dynamic_chunks)
            print(f"✓ {result['message']}")
        else:
            print("⚠ No dynamic data found to process")
    
    # Final stats
    final_stats = embedding_manager.get_collection_stats()
    print(f"\n=== FINAL EMBEDDING STATUS ===")
    print(f"  Static collection: {final_stats['static_collection']['count']} documents")
    print(f"  Dynamic collection: {final_stats['dynamic_collection']['count']} documents")
    print(f"  Total documents: {final_stats['total_documents']}")
    
    return True

if __name__ == "__main__":
    import uvicorn
    from src.rag.config import Config
    
    print("=== POLKADOT AI CHATBOT SERVER STARTUP ===")
    
    # Initialize embeddings
    if not initialize_embeddings():
        print("✗ Failed to initialize embeddings")
        sys.exit(1)
    
    print("\n=== STARTING API SERVER ===")
    print("Starting Polkadot AI Chatbot API server...")
    
    try:
        uvicorn.run(
            "src.rag.api_server:app",
            host=Config.API_HOST,
            port=Config.API_PORT,
            log_level="info",
            reload=False
        )
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except Exception as e:
        print(f"Error starting server: {e}")
        sys.exit(1) 