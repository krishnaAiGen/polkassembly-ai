#!/usr/bin/env python3
"""
Script to create embeddings from Polkadot data sources and store them in ChromaDB.
This script should be run once to initialize the embeddings database.
"""

import os
import sys
import logging
import argparse
from datetime import datetime

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .config import Config
from ..utils.data_loader import DataLoader
from ..utils.text_chunker import TextChunker
from ..utils.embeddings import EmbeddingManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('embedding_creation.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Create embeddings from Polkadot data sources')
    parser.add_argument('--clear', action='store_true', help='Clear existing embeddings before creating new ones')
    parser.add_argument('--batch-size', type=int, default=50, help='Batch size for embedding generation')
    parser.add_argument('--min-tokens', type=int, default=50, help='Minimum tokens per chunk')
    parser.add_argument('--max-tokens', type=int, default=1000, help='Maximum tokens per chunk')
    
    args = parser.parse_args()
    
    logger.info("Starting embedding creation process...")
    logger.info(f"Arguments: {args}")
    
    try:
        # Validate configuration
        Config.validate_config()
        logger.info("Configuration validated successfully")
        
        # Initialize components
        logger.info("Initializing components...")
        
        # Data loader
        data_loader = DataLoader(
            polkadot_network_path=Config.POLKADOT_NETWORK_PATH,
            polkadot_wiki_path=Config.POLKADOT_WIKI_PATH
        )
        
        # Text chunker
        text_chunker = TextChunker(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP
        )
        
        # Embedding manager
        embedding_manager = EmbeddingManager(
            openai_api_key=Config.OPENAI_API_KEY,
            embedding_model=Config.OPENAI_EMBEDDING_MODEL,
            chroma_persist_directory=Config.CHROMA_PERSIST_DIRECTORY,
            collection_name=Config.CHROMA_COLLECTION_NAME
        )
        
        # Clear existing embeddings if requested
        if args.clear:
            logger.info("Clearing existing embeddings...")
            embedding_manager.clear_collection()
        
        # Check if collection already has data
        if embedding_manager.collection_exists() and not args.clear:
            logger.info("Collection already exists with data. Use --clear to recreate.")
            stats = embedding_manager.get_collection_stats()
            logger.info(f"Current collection stats: {stats}")
            
            response = input("Do you want to continue and add more data? (y/N): ")
            if response.lower() != 'y':
                logger.info("Exiting...")
                return
        
        # Load documents
        logger.info("Loading documents...")
        documents = data_loader.load_all_documents()
        
        if not documents:
            logger.error("No documents loaded. Please check your data paths.")
            return
        
        # Print document statistics
        doc_stats = data_loader.get_document_stats(documents)
        logger.info("Document statistics:")
        for key, value in doc_stats.items():
            logger.info(f"  {key}: {value}")
        
        # Chunk documents
        logger.info("Chunking documents...")
        chunks = text_chunker.chunk_documents(documents)
        
        if not chunks:
            logger.error("No chunks created from documents.")
            return
        
        # Print chunk statistics
        chunk_stats = text_chunker.get_chunk_stats(chunks)
        logger.info("Chunk statistics:")
        for key, value in chunk_stats.items():
            logger.info(f"  {key}: {value}")
        
        # Filter chunks by size
        filtered_chunks = text_chunker.filter_chunks_by_size(
            chunks, 
            min_tokens=args.min_tokens,
            max_tokens=args.max_tokens
        )
        
        if not filtered_chunks:
            logger.error("No chunks remaining after filtering.")
            return
        
        logger.info(f"Using {len(filtered_chunks)} chunks for embedding generation")
        
        # Create embeddings and add to ChromaDB
        logger.info("Creating embeddings and storing in ChromaDB...")
        start_time = datetime.now()
        
        # Process in smaller batches to manage memory and API limits
        batch_size = args.batch_size
        total_batches = (len(filtered_chunks) + batch_size - 1) // batch_size
        
        for i in range(0, len(filtered_chunks), batch_size):
            batch_num = i // batch_size + 1
            batch_chunks = filtered_chunks[i:i + batch_size]
            
            logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch_chunks)} chunks)")
            
            success = embedding_manager.add_chunks_to_collection(batch_chunks)
            
            if not success:
                logger.error(f"Failed to process batch {batch_num}")
                continue
            
            logger.info(f"Successfully processed batch {batch_num}")
        
        # Final statistics
        end_time = datetime.now()
        duration = end_time - start_time
        
        final_stats = embedding_manager.get_collection_stats()
        logger.info("Final collection statistics:")
        for key, value in final_stats.items():
            logger.info(f"  {key}: {value}")
        
        logger.info(f"Embedding creation completed in {duration}")
        logger.info(f"Total processing time: {duration.total_seconds():.2f} seconds")
        
        # Test the search functionality
        logger.info("Testing search functionality...")
        test_queries = [
            "What is Polkadot?",
            "How does staking work?",
            "What are parachains?"
        ]
        
        for query in test_queries:
            logger.info(f"Testing query: '{query}'")
            results = embedding_manager.search_similar_chunks(query, n_results=3)
            logger.info(f"Found {len(results)} results")
            
            if results:
                for i, result in enumerate(results):
                    logger.info(f"  Result {i+1}: {result['metadata'].get('title', 'No title')[:50]}... "
                              f"(score: {result['similarity_score']:.3f})")
        
        logger.info("Embedding creation process completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error during embedding creation: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 