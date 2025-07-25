import openai
import chromadb
import logging
import uuid
import time
from typing import List, Dict, Any, Optional, Tuple
from chromadb.config import Settings
import numpy as np
from src.rag.config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingManager:
    """Manage OpenAI embeddings and ChromaDB operations"""
    
    def __init__(self, 
                 openai_api_key: str,
                 embedding_model: str = "text-embedding-ada-002",
                 chroma_persist_directory: str = "./chroma_db",
                 collection_name: str = "polkadot_embeddings"):
        
        self.openai_api_key = openai_api_key
        self.embedding_model = embedding_model
        self.chroma_persist_directory = chroma_persist_directory
        self.collection_name = collection_name
        
        # Initialize OpenAI client
        openai.api_key = self.openai_api_key
        
        # Initialize ChromaDB client
        self.chroma_client = chromadb.PersistentClient(
            path=chroma_persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Initialize or get collection
        self.collection = None
        self._init_collection()
    
    def _init_collection(self):
        """Initialize or get the ChromaDB collection"""
        try:
            # Try to get existing collection
            self.collection = self.chroma_client.get_collection(self.collection_name)
            logger.info(f"Loaded existing collection '{self.collection_name}' with {self.collection.count()} documents")
        except Exception:
            # Create new collection if it doesn't exist
            self.collection = self.chroma_client.create_collection(
                name=self.collection_name,
                metadata={"description": "Polkadot documentation embeddings"}
            )
            logger.info(f"Created new collection '{self.collection_name}'")
    
    def generate_embeddings(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        """
        Generate embeddings for a list of texts using OpenAI
        
        Args:
            texts: List of text strings to embed
            batch_size: Number of texts to process in each batch
            
        Returns:
            List of embedding vectors
        """
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            try:
                # Add retry logic for rate limiting
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        response = openai.embeddings.create(
                            model=self.embedding_model,
                            input=batch
                        )
                        
                        batch_embeddings = [item.embedding for item in response.data]
                        embeddings.extend(batch_embeddings)
                        
                        logger.info(f"Generated embeddings for batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
                        break
                        
                    except openai.RateLimitError as e:
                        if attempt < max_retries - 1:
                            wait_time = 2 ** attempt
                            logger.warning(f"Rate limit hit, waiting {wait_time} seconds...")
                            time.sleep(wait_time)
                        else:
                            raise e
                    except Exception as e:
                        logger.error(f"Error generating embeddings for batch {i//batch_size + 1}: {e}")
                        if attempt < max_retries - 1:
                            time.sleep(1)
                        else:
                            raise e
                
                # Small delay between batches to avoid rate limiting
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Failed to generate embeddings for batch {i//batch_size + 1}: {e}")
                # Add empty embeddings for failed batch
                embeddings.extend([[0.0] * 1536] * len(batch))  # ada-002 has 1536 dimensions
        
        return embeddings
    
    def add_chunks_to_collection(self, chunks: List[Dict[str, Any]]) -> bool:
        """
        Add chunks with embeddings to ChromaDB collection
        
        Args:
            chunks: List of chunk dictionaries with content and metadata
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Extract texts for embedding generation
            texts = [chunk['content'] for chunk in chunks]
            
            logger.info(f"Generating embeddings for {len(texts)} chunks...")
            embeddings = self.generate_embeddings(texts)
            
            if len(embeddings) != len(chunks):
                logger.error(f"Mismatch between chunks ({len(chunks)}) and embeddings ({len(embeddings)})")
                return False
            
            # Prepare data for ChromaDB
            ids = []
            documents = []
            metadatas = []
            
            for i, chunk in enumerate(chunks):
                # Generate unique ID for each chunk
                chunk_id = str(uuid.uuid4())
                ids.append(chunk_id)
                documents.append(chunk['content'])
                
                # Prepare metadata (ChromaDB doesn't support nested dicts)
                metadata = {}
                for key, value in chunk['metadata'].items():
                    if isinstance(value, (str, int, float, bool)):
                        metadata[key] = value
                    else:
                        metadata[key] = str(value)
                
                metadatas.append(metadata)
            
            # Add to collection
            logger.info(f"Adding {len(chunks)} chunks to ChromaDB collection...")
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )
            
            logger.info(f"Successfully added {len(chunks)} chunks to collection")
            return True
            
        except Exception as e:
            logger.error(f"Error adding chunks to collection: {e}")
            return False
    
    def search_similar_chunks(self, 
                            query: str, 
                            n_results: int = 5,
                            filter_metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Search for similar chunks using semantic similarity
        
        Args:
            query: Search query
            n_results: Number of results to return
            filter_metadata: Optional metadata filters
            
        Returns:
            List of similar chunks with content, metadata, and similarity scores
        """
        try:
            # If both search flags are false, return empty results
            if not Config.SEARCH_STATIC_DATA and not Config.SEARCH_DYNAMIC_DATA:
                logger.info("Both static and dynamic search are disabled")
                return []

            # Generate embedding for the query
            query_embedding = self.generate_embeddings([query])[0]
            
            results = []
            
            # Search in static collection if enabled
            if Config.SEARCH_STATIC_DATA and self.collection_name == Config.CHROMA_COLLECTION_NAME:
                static_results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=n_results,
                    where=filter_metadata,
                    include=["documents", "metadatas", "distances"]
                )
                if static_results['documents'] and len(static_results['documents']) > 0:
                    for i in range(len(static_results['documents'][0])):
                        result = {
                            'content': static_results['documents'][0][i],
                            'metadata': static_results['metadatas'][0][i],
                            'similarity_score': 1.0 - static_results['distances'][0][i],
                            'source': 'static'
                        }
                        results.append(result)
                    logger.info(f"Found {len(results)} chunks from static data")

            # Search in dynamic collection if enabled
            if Config.SEARCH_DYNAMIC_DATA and self.collection_name == Config.CHROMA_DYNAMIC_COLLECTION_NAME:
                dynamic_results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=n_results,
                    where=filter_metadata,
                    include=["documents", "metadatas", "distances"]
                )
                if dynamic_results['documents'] and len(dynamic_results['documents']) > 0:
                    for i in range(len(dynamic_results['documents'][0])):
                        result = {
                            'content': dynamic_results['documents'][0][i],
                            'metadata': dynamic_results['metadatas'][0][i],
                            'similarity_score': 1.0 - dynamic_results['distances'][0][i],
                            'source': 'dynamic'
                        }
                        results.append(result)
                    logger.info(f"Found {len(results)} chunks from dynamic data")

            # Sort all results by similarity score and take top n_results
            results.sort(key=lambda x: x['similarity_score'], reverse=True)
            results = results[:n_results]

            # Log the sources of returned results
            static_count = sum(1 for r in results if r['source'] == 'static')
            dynamic_count = sum(1 for r in results if r['source'] == 'dynamic')
            logger.info(f"Returning {static_count} static and {dynamic_count} dynamic results")
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching similar chunks: {e}")
            return []
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection"""
        try:
            count = self.collection.count()
            
            # Get sample of metadata to understand structure
            sample_results = self.collection.get(limit=10, include=["metadatas"])
            
            source_counts = {}
            if sample_results['metadatas']:
                # Get all metadata to count sources
                all_results = self.collection.get(include=["metadatas"])
                for metadata in all_results['metadatas']:
                    source = metadata.get('source', 'unknown')
                    source_counts[source] = source_counts.get(source, 0) + 1
            
            stats = {
                'total_chunks': count,
                'chunks_by_source': source_counts,
                'collection_name': self.collection_name,
                'embedding_model': self.embedding_model
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {}
    
    def clear_collection(self):
        """Clear all data from the collection"""
        try:
            # Delete the collection
            self.chroma_client.delete_collection(self.collection_name)
            logger.info(f"Deleted collection '{self.collection_name}'")
            
            # Recreate empty collection
            self._init_collection()
            logger.info(f"Recreated empty collection '{self.collection_name}'")
            
        except Exception as e:
            logger.error(f"Error clearing collection: {e}")
    
    def collection_exists(self) -> bool:
        """Check if collection exists and has data"""
        try:
            return self.collection is not None and self.collection.count() > 0
        except Exception:
            return False 