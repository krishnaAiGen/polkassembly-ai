#!/usr/bin/env python3
"""
Create embeddings from external data sources and add them to the system.
This allows adding new embedding collections without rebuilding existing ones.
"""

import os
import sys
import json
from typing import List, Dict, Any, Optional
from pathlib import Path

# Add src to path
src_path = os.path.join(os.path.dirname(__file__), 'src')
sys.path.insert(0, src_path)

from src.rag.config import Config
from src.utils.embeddings import DualEmbeddingManager
from src.utils.text_chunker import TextChunker
import chromadb
from chromadb.config import Settings

class ExternalDataProcessor:
    """Process external data sources and create new embedding collections"""
    
    def __init__(self, external_data_source_path: str, collection_name: str):
        self.external_data_source_path = external_data_source_path
        self.collection_name = collection_name
        self.chroma_client = chromadb.PersistentClient(
            path=Config.CHROMA_PERSIST_DIRECTORY,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Initialize text chunker
        self.text_chunker = TextChunker(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP
        )
    
    def find_all_files(self, directory: str, extensions: List[str]) -> List[str]:
        """Recursively find all files with given extensions in directory and subdirectories"""
        file_paths = []
        
        for root, dirs, files in os.walk(directory):
            for file in files:
                if any(file.endswith(ext) for ext in extensions):
                    file_path = os.path.join(root, file)
                    file_paths.append(file_path)
        
        return file_paths
    
    def load_text_file(self, file_path: str) -> Dict[str, Any]:
        """Load a single text file and extract metadata"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            lines = content.split('\n')
            metadata = {}
            text_content = ""
            
            # Parse metadata from the beginning of text files
            if len(lines) > 0 and lines[0].startswith('Title: '):
                metadata['title'] = lines[0].replace('Title: ', '').strip()
            
            if len(lines) > 1 and lines[1].startswith('URL: '):
                metadata['url'] = lines[1].replace('URL: ', '').strip()
            
            if len(lines) > 2 and (lines[2].startswith('Description: ') or lines[2].startswith('Type: ')):
                if lines[2].startswith('Description: '):
                    metadata['description'] = lines[2].replace('Description: ', '').strip()
                elif lines[2].startswith('Type: '):
                    metadata['type'] = lines[2].replace('Type: ', '').strip()
                    if len(lines) > 3 and lines[3].startswith('Description: '):
                        metadata['description'] = lines[3].replace('Description: ', '').strip()
            
            # Find the separator line and extract main content
            separator_found = False
            for i, line in enumerate(lines):
                if '========' in line or '=======' in line:
                    text_content = '\n'.join(lines[i+1:]).strip()
                    separator_found = True
                    break
            
            if not separator_found:
                # If no separator found, use all content after metadata
                start_idx = 0
                for i, line in enumerate(lines):
                    if line.strip() == "":
                        start_idx = i + 1
                        break
                text_content = '\n'.join(lines[start_idx:]).strip()
            
            # Add source information
            metadata['source'] = 'external'
            metadata['collection'] = self.collection_name
            metadata['file_path'] = file_path
            metadata['relative_path'] = os.path.relpath(file_path, self.external_data_source_path)
            
            return {
                'content': text_content,
                'metadata': metadata
            }
            
        except Exception as e:
            print(f"Error loading text file {file_path}: {e}")
            return None
    
    def load_json_file(self, file_path: str) -> Dict[str, Any]:
        """Load a single JSON file and extract relevant content"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract content from JSON structure
            content_parts = []
            
            if 'title' in data:
                content_parts.append(f"Title: {data['title']}")
            
            if 'description' in data:
                content_parts.append(f"Description: {data['description']}")
            
            if 'content' in data and data['content']:
                content_parts.append(data['content'])
            
            # Add topics if available (for forum data)
            if 'topics' in data and data['topics']:
                topics_text = "Related Topics:\n"
                for topic in data['topics'][:10]:  # Limit to first 10 topics
                    topics_text += f"- {topic.get('title', '')}\n"
                content_parts.append(topics_text)
            
            content = '\n\n'.join(content_parts)
            
            metadata = {
                'title': data.get('title', ''),
                'url': data.get('url', ''),
                'description': data.get('description', ''),
                'source': 'external',
                'collection': self.collection_name,
                'file_path': file_path,
                'relative_path': os.path.relpath(file_path, self.external_data_source_path)
            }
            
            # Add any additional metadata from JSON
            for key, value in data.items():
                if key not in ['title', 'url', 'description', 'content', 'topics'] and isinstance(value, (str, int, float, bool)):
                    metadata[key] = value
            
            return {
                'content': content,
                'metadata': metadata
            }
            
        except Exception as e:
            print(f"Error loading JSON file {file_path}: {e}")
            return None
    
    def load_external_data(self) -> List[Dict[str, Any]]:
        """Load all files from external data source"""
        print(f"Loading external data from: {self.external_data_source_path}")
        
        if not os.path.exists(self.external_data_source_path):
            print(f"External data path does not exist: {self.external_data_source_path}")
            return []
        
        # Find all .txt and .json files recursively
        txt_files = self.find_all_files(self.external_data_source_path, ['.txt'])
        json_files = self.find_all_files(self.external_data_source_path, ['.json'])
        
        print(f"Found {len(txt_files)} .txt files and {len(json_files)} .json files")
        
        external_data = []
        
        # Process .txt files
        for file_path in txt_files:
            try:
                data = self.load_text_file(file_path)
                if data:
                    external_data.append(data)
            except Exception as e:
                print(f"Error loading text file {file_path}: {e}")
        
        # Process .json files
        for file_path in json_files:
            try:
                data = self.load_json_file(file_path)
                if data:
                    external_data.append(data)
            except Exception as e:
                print(f"Error loading JSON file {file_path}: {e}")
        
        print(f"Successfully loaded {len(external_data)} external data items")
        return external_data
    
    def create_external_collection(self, documents: List[Dict[str, Any]]) -> bool:
        """Create a new embedding collection for external data"""
        try:
            # Check if collection already exists
            try:
                existing_collection = self.chroma_client.get_collection(self.collection_name)
                print(f"Collection '{self.collection_name}' already exists with {existing_collection.count()} documents")
                
                # Ask user if they want to add to existing or recreate
                response = input(f"Do you want to (a)dd to existing collection or (r)ecreate it? [a/r]: ").lower()
                if response == 'r':
                    self.chroma_client.delete_collection(self.collection_name)
                    print(f"Deleted existing collection '{self.collection_name}'")
                elif response != 'a':
                    print("Operation cancelled")
                    return False
            except Exception:
                pass  # Collection doesn't exist, which is fine
            
            # Create or get collection
            try:
                collection = self.chroma_client.get_collection(self.collection_name)
                print(f"Adding to existing collection '{self.collection_name}'")
            except Exception:
                collection = self.chroma_client.create_collection(
                    name=self.collection_name,
                    metadata={"description": f"External data embeddings for {self.collection_name}"}
                )
                print(f"Created new collection '{self.collection_name}'")
            
            if not documents:
                print("No documents to process")
                return False
            
            # Chunk the documents
            print(f"Chunking {len(documents)} documents...")
            chunks = []
            
            for doc in documents:
                doc_chunks = self.text_chunker.chunk_document(doc)
                chunks.extend(doc_chunks)
            
            print(f"Created {len(chunks)} chunks")
            
            # Generate embeddings
            print("Generating embeddings...")
            import openai
            openai.api_key = Config.OPENAI_API_KEY
            
            texts = [chunk['content'] for chunk in chunks]
            metadatas = [chunk['metadata'] for chunk in chunks]
            
            # Generate embeddings in batches
            embeddings = []
            batch_size = 100
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                
                try:
                    response = openai.embeddings.create(
                        model=Config.OPENAI_EMBEDDING_MODEL,
                        input=batch
                    )
                    
                    batch_embeddings = [item.embedding for item in response.data]
                    embeddings.extend(batch_embeddings)
                    
                    print(f"Generated embeddings for batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
                    
                except Exception as e:
                    print(f"Error generating embeddings for batch {i//batch_size + 1}: {e}")
                    return False
            
            # Generate IDs
            import uuid
            ids = [f"external_{self.collection_name}_{uuid.uuid4().hex}" for _ in range(len(chunks))]
            
            # Add to collection
            print(f"Adding {len(chunks)} chunks to collection...")
            collection.add(
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            
            print(f"✓ Successfully created collection '{self.collection_name}' with {len(chunks)} chunks")
            return True
            
        except Exception as e:
            print(f"Error creating external collection: {e}")
            return False

def main():
    """Main function to create external embeddings"""
    print("=== EXTERNAL EMBEDDINGS CREATOR ===")
    
    # Get external data source path
    if len(sys.argv) < 2:
        external_data_source_path = input("Enter the path to external data source folder: ").strip()
    else:
        external_data_source_path = sys.argv[1]
    
    if not os.path.exists(external_data_source_path):
        print(f"Error: Path '{external_data_source_path}' does not exist")
        return
    
    # Get collection name
    if len(sys.argv) < 3:
        collection_name = input("Enter collection name (e.g., 'external_docs', 'new_data'): ").strip()
    else:
        collection_name = sys.argv[2]
    
    if not collection_name:
        print("Error: Collection name cannot be empty")
        return
    
    # Validate collection name (alphanumeric and underscores only)
    if not collection_name.replace('_', '').replace('-', '').isalnum():
        print("Error: Collection name must contain only letters, numbers, underscores, and hyphens")
        return
    
    try:
        # Validate configuration
        Config.validate_config()
        print("✓ Configuration validated")
    except ValueError as e:
        print(f"✗ Configuration error: {e}")
        return
    
    # Process external data
    processor = ExternalDataProcessor(external_data_source_path, collection_name)
    external_data = processor.load_external_data()
    
    if not external_data:
        print("No data found to process")
        return
    
    # Create embeddings
    if processor.create_external_collection(external_data):
        print(f"\n✓ Successfully created external collection '{collection_name}'")
        print(f"The system will now search across:")
        print(f"  - static_embeddings")
        print(f"  - dynamic_embeddings") 
        print(f"  - {collection_name}")
        print(f"\nTo use the new collection, you may need to update the search logic in the API server.")
    else:
        print("✗ Failed to create external collection")

if __name__ == "__main__":
    main() 