#!/usr/bin/env python3
"""
Create embeddings for dynamic data (Polkadot and Kusama onchain data)
"""

import os
import sys
import logging
from pathlib import Path
from typing import List, Dict, Any
import json

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.utils.embeddings import EmbeddingManager
from src.utils.text_chunker import TextChunker
from src.utils.data_loader import DataLoader
from src.rag.config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_dynamic_data(data_dir: str) -> List[Dict[str, Any]]:
    """Load dynamic data from JSON files"""
    documents = []
    data_path = Path(data_dir)
    
    if not data_path.exists():
        logger.warning(f"Data directory not found: {data_path}")
        return documents
        
    # Process all JSON files in the directory
    for file_path in data_path.glob("*.json"):
        try:
            # Extract network from filename (polkadot_* or kusama_*)
            filename = file_path.name.lower()
            if filename.startswith("polkadot_"):
                network = "polkadot"
            elif filename.startswith("kusama_"):
                network = "kusama"
            else:
                logger.warning(f"Skipping file with unknown network prefix: {file_path}")
                continue
                
            if file_path.name == "fetch_summary.json":
                continue
                
            # Load JSON data
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Process each item in the file
            for item in data.get('items', []):
                # Extract content parts
                content_parts = []
                
                if item.get('title'):
                    content_parts.append(f"Title: {item['title']}")
                
                if item.get('content'):
                    content_parts.append(item['content'])
                
                if item.get('onChainInfo'):
                    info = item['onChainInfo']
                    content_parts.append(f"\nOn-chain Information:")
                    if info.get('proposer'):
                        content_parts.append(f"Proposer: {info['proposer']}")
                    if info.get('status'):
                        content_parts.append(f"Status: {info['status']}")
                    if info.get('hash'):
                        content_parts.append(f"Hash: {info['hash']}")
                
                content = '\n\n'.join(content_parts)
                
                # Create metadata
                metadata = {
                    'title': item.get('title', ''),
                    'network': network,
                    'proposalType': item.get('proposalType', ''),
                    'index': item.get('index', ''),
                    'createdAt': item.get('createdAt', ''),
                    'source': 'polkassembly',
                    'file_path': str(file_path)
                }
                
                if content.strip():
                    documents.append({
                        'content': content,
                        'metadata': metadata
                    })
                    
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {e}")
    
    return documents

def create_dynamic_embeddings(
    data_dir: str = None,
    chunk_size: int = None,
    chunk_overlap: int = None,
    collection_name: str = None
) -> bool:
    """
    Create embeddings for dynamic data
    
    Args:
        data_dir: Directory containing dynamic data
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks
        collection_name: Name for the Chroma collection
    """
    try:
        # Set default data directory if not provided
        if not data_dir:
            data_dir = os.path.join(project_root, Config.DYNAMIC_DATA_PATH)
        
        # Use config values if not provided
        chunk_size = chunk_size or Config.CHUNK_SIZE
        chunk_overlap = chunk_overlap or Config.CHUNK_OVERLAP
        collection_name = collection_name or Config.CHROMA_COLLECTION_NAME + "_dynamic"
        
        logger.info(f"Creating embeddings for dynamic data in {data_dir}")
        logger.info(f"Using collection name: {collection_name}")
        
        # Initialize components
        chunker = TextChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        embedding_manager = EmbeddingManager(
            openai_api_key=Config.OPENAI_API_KEY,
            collection_name=collection_name,
            chroma_persist_directory=Config.CHROMA_PERSIST_DIRECTORY + "_dynamic"
        )
        
        # Load documents
        documents = load_dynamic_data(data_dir)
        if not documents:
            logger.warning("No dynamic data found")
            return False
        
        logger.info(f"Loaded {len(documents)} documents")
        
        # Process documents into chunks
        all_chunks = []
        for doc in documents:
            chunks = chunker.chunk_document(doc)
            all_chunks.extend(chunks)
        
        logger.info(f"Created {len(all_chunks)} chunks")
        
        # Create embeddings
        embedding_manager.add_chunks_to_collection(all_chunks)
        logger.info("Successfully created embeddings for dynamic data")
        
        return True
        
    except Exception as e:
        logger.error(f"Error creating dynamic embeddings: {e}")
        return False

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Create embeddings for dynamic data")
    parser.add_argument("--data-dir", type=str, help="Directory containing dynamic data")
    parser.add_argument("--chunk-size", type=int, help="Size of text chunks")
    parser.add_argument("--chunk-overlap", type=int, help="Overlap between chunks")
    parser.add_argument("--collection-name", type=str, help="Name for the Chroma collection")
    
    args = parser.parse_args()
    
    success = create_dynamic_embeddings(
        data_dir=args.data_dir,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        collection_name=args.collection_name
    )
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main() 