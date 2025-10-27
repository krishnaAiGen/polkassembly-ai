#!/usr/bin/env python3
"""
Create embeddings for static data (documentation, wiki pages, etc.)
"""

import os
import sys
import logging
from pathlib import Path
from typing import List, Dict, Any

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.utils.embeddings import EmbeddingManager
from src.utils.text_chunker import TextChunker
from src.utils.data_loader import DataLoader
from src.rag.config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_static_data(data_dir: str) -> List[Dict[str, Any]]:
    """Load and combine static data from all subfolders into single documents"""
    documents = []
    data_path = Path(data_dir)
    
    if not data_path.exists():
        logger.warning(f"Data directory not found: {data_path}")
        return documents
    
    # Get all subdirectories in the static_sources folder
    subdirs = [d for d in data_path.iterdir() if d.is_dir()]
    logger.info(f"Found {len(subdirs)} subdirectories to process: {[d.name for d in subdirs]}")
    
    total_files = 0
    processed_count = 0
    skipped_count = 0
    
    # Process each subdirectory
    for subdir in subdirs:
        logger.info(f"\n📁 Processing subdirectory: {subdir.name}")
        
        # Get all .txt files in this subdirectory
        txt_files = list(subdir.glob("**/*.txt"))
        total_files += len(txt_files)
        
        # Combine all text files in this subdirectory into one document
        combined_content = []
        combined_metadata = {
            'source': 'static_documentation',
            'data_type': 'static',
            'subdirectory': subdir.name,
            'file_count': len(txt_files)
        }
        
        for file_path in txt_files:
            try:
                processed_count += 1
                logger.info(f"  [{processed_count}] Processing: {file_path.relative_to(data_path)}")
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                if not content.strip():
                    logger.warning(f"    └─ Skipping empty file: {file_path.name}")
                    skipped_count += 1
                    continue
                
                # Parse metadata from file header if present
                lines = content.split('\n')
                file_metadata = {}
                content_start = 0
                
                # Extract metadata from header lines
                for i, line in enumerate(lines[:10]):  # Check first 10 lines for metadata
                    if line.startswith('Title: '):
                        file_metadata['title'] = line.replace('Title: ', '').strip()
                        content_start = max(content_start, i + 1)
                    elif line.startswith('URL: '):
                        file_metadata['url'] = line.replace('URL: ', '').strip()
                        content_start = max(content_start, i + 1)
                    elif line.startswith('Description: '):
                        file_metadata['description'] = line.replace('Description: ', '').strip()
                        content_start = max(content_start, i + 1)
                    elif line.startswith('Type: '):
                        file_metadata['type'] = line.replace('Type: ', '').strip()
                        content_start = max(content_start, i + 1)
                    elif line.strip() == '---' or line.strip() == '':
                        content_start = i + 1
                        break
                
                # Use filename as title if no title found
                if 'title' not in file_metadata:
                    file_metadata['title'] = file_path.stem
                
                # Extract main content (skip metadata header)
                main_content = '\n'.join(lines[content_start:]).strip()
                
                if main_content:
                    # Add file separator and metadata
                    file_header = f"\n\n--- FILE: {file_metadata.get('title', file_path.stem)} ---\n"
                    combined_content.append(file_header + main_content)
                    logger.info(f"    └─ ✅ Added: {len(main_content)} characters, title: '{file_metadata.get('title', 'N/A')}'")
                else:
                    logger.warning(f"    └─ ⚠️  No content found after parsing headers in: {file_path.name}")
                    skipped_count += 1
                        
            except Exception as e:
                logger.error(f"    └─ ❌ Error loading file {file_path.name}: {e}")
                skipped_count += 1
        
        # Create combined document for this subdirectory
        if combined_content:
            final_content = '\n'.join(combined_content)
            combined_metadata['title'] = f"{subdir.name} Documentation"
            combined_metadata['description'] = f"Combined documentation from {subdir.name} subdirectory"
            combined_metadata['content_length'] = len(final_content)
            
            documents.append({
                'content': final_content,
                'metadata': combined_metadata
            })
            
            logger.info(f"  └─ ✅ Created combined document: {len(final_content)} characters from {len(txt_files)} files")
        else:
            logger.warning(f"  └─ ⚠️  No content found in subdirectory: {subdir.name}")
    
    # Log summary
    successful_count = len(documents)
    logger.info(f"\n📊 Processing Summary:")
    logger.info(f"  • Total subdirectories: {len(subdirs)}")
    logger.info(f"  • Total files found: {total_files}")
    logger.info(f"  • Successfully processed: {successful_count}")
    logger.info(f"  • Skipped/Failed: {skipped_count}")
    logger.info(f"  • Success rate: {(successful_count/len(subdirs)*100):.1f}%")
    
    return documents

def create_static_embeddings(
    data_dir: str = None,
    chunk_size: int = None,
    chunk_overlap: int = None,
    collection_name: str = None
) -> bool:
    """
    Create embeddings for static data
    
    Args:
        data_dir: Directory containing static data
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks
        collection_name: Name for the Chroma collection
    """
    try:
        # Set default data directory if not provided
        if not data_dir:
            # Use static_sources directory directly instead of joined_data/static
            data_dir = os.path.join(project_root, "data", "static_sources")
        
        # Use config values if not provided
        chunk_size = chunk_size or Config.CHUNK_SIZE
        chunk_overlap = chunk_overlap or Config.CHUNK_OVERLAP
        collection_name = collection_name or Config.CHROMA_COLLECTION_NAME
        
        logger.info("=" * 60)
        logger.info("🔧 STATIC EMBEDDINGS CREATION STARTED")
        logger.info("=" * 60)
        logger.info(f"📁 Data directory: {data_dir}")
        logger.info(f"🗄️  Collection name: {collection_name}")
        logger.info(f"📏 Chunk size: {chunk_size}")
        logger.info(f"🔄 Chunk overlap: {chunk_overlap}")
        logger.info("-" * 60)
        
        # Initialize components
        chunker = TextChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        embedding_manager = EmbeddingManager(
            openai_api_key=Config.OPENAI_API_KEY,
            collection_name=collection_name,
            chroma_persist_directory=Config.CHROMA_PERSIST_DIRECTORY
        )
        
        # Clear existing collection to start fresh
        logger.info("Clearing existing collection...")
        embedding_manager.clear_collection()
        
        # Load documents
        documents = load_static_data(data_dir)
        if not documents:
            logger.warning("No static data found")
            return False
        
        logger.info(f"Loaded {len(documents)} documents")
        
        # Process documents into chunks
        logger.info("🔄 Starting text chunking process...")
        all_chunks = []
        for i, doc in enumerate(documents, 1):
            chunks = chunker.chunk_document(doc)
            all_chunks.extend(chunks)
            title = doc['metadata'].get('title', 'Unknown')
            logger.info(f"  [{i}/{len(documents)}] '{title}' → {len(chunks)} chunks")
        
        logger.info(f"✅ Created {len(all_chunks)} total chunks from {len(documents)} documents")
        
        # Create embeddings
        logger.info(f"🚀 Creating embeddings for {len(all_chunks)} chunks...")
        embedding_manager.add_chunks_to_collection(all_chunks)
        logger.info("🎉 Successfully created embeddings for static data!")
        logger.info("=" * 60)
        logger.info("✅ STATIC EMBEDDINGS CREATION COMPLETED")
        logger.info("=" * 60)
        
        return True
        
    except Exception as e:
        logger.error(f"Error creating static embeddings: {e}")
        return False

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Create embeddings for static data")
    parser.add_argument("--data-dir", type=str, help="Directory containing static data")
    parser.add_argument("--chunk-size", type=int, help="Size of text chunks")
    parser.add_argument("--chunk-overlap", type=int, help="Overlap between chunks")
    parser.add_argument("--collection-name", type=str, help="Name for the Chroma collection")
    
    args = parser.parse_args()
    
    success = create_static_embeddings(
        data_dir=args.data_dir,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        collection_name=args.collection_name
    )
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main() 