import openai
import chromadb
import logging
import uuid
import time
from typing import List, Dict, Any, Optional, Tuple
from chromadb.config import Settings
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DualEmbeddingManager:
    """Manage OpenAI embeddings and ChromaDB operations with support for multiple collections"""
    
    def __init__(self, 
                 openai_api_key: str,
                 embedding_model: str = "text-embedding-ada-002",
                 chroma_persist_directory: str = "./chroma_db"):
        
        self.openai_api_key = openai_api_key
        self.embedding_model = embedding_model
        self.chroma_persist_directory = chroma_persist_directory
        
        # Initialize OpenAI client
        openai.api_key = self.openai_api_key
        
        # Initialize ChromaDB client
        self.chroma_client = chromadb.PersistentClient(
            path=chroma_persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Initialize collections
        self.static_collection = None
        self.dynamic_collection = None
        self._init_collections()
    
    def _init_collections(self):
        """Initialize or get the ChromaDB collections"""
        # Static data collection
        try:
            self.static_collection = self.chroma_client.get_collection("static_embeddings")
            logger.info(f"Loaded existing static collection with {self.static_collection.count()} documents")
        except Exception:
            self.static_collection = self.chroma_client.create_collection(
                name="static_embeddings",
                metadata={"description": "Static data embeddings from .txt files"}
            )
            logger.info("Created new static collection")
        
        # Dynamic data collection
        try:
            self.dynamic_collection = self.chroma_client.get_collection("dynamic_embeddings")
            logger.info(f"Loaded existing dynamic collection with {self.dynamic_collection.count()} documents")
        except Exception:
            self.dynamic_collection = self.chroma_client.create_collection(
                name="dynamic_embeddings",
                metadata={"description": "Dynamic data embeddings from .json files"}
            )
            logger.info("Created new dynamic collection")
    
    def get_all_collections(self) -> List[Dict[str, Any]]:
        """Get all available collections in the database"""
        try:
            all_collections = self.chroma_client.list_collections()
            collection_info = []
            
            for collection in all_collections:
                collection_info.append({
                    'name': collection.name,
                    'count': collection.count(),
                    'metadata': collection.metadata
                })
            
            return collection_info
        except Exception as e:
            logger.error(f"Error getting collections: {e}")
            return []
    
    def get_external_collections(self) -> List[Dict[str, Any]]:
        """Get all external collections (not static or dynamic)"""
        all_collections = self.get_all_collections()
        external_collections = []
        
        for collection in all_collections:
            if collection['name'] not in ['static_embeddings', 'dynamic_embeddings']:
                external_collections.append(collection)
        
        return external_collections
    
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
                raise e
        
        return embeddings
    
    def add_static_documents(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Add static documents to the static collection
        
        Args:
            documents: List of document dictionaries with 'content' and 'metadata'
            
        Returns:
            Summary of the operation
        """
        if not documents:
            return {"total_added": 0, "message": "No documents to add"}
        
        logger.info(f"Adding {len(documents)} static documents...")
        
        # Extract texts and metadata
        texts = [doc['content'] for doc in documents]
        metadatas = [doc['metadata'] for doc in documents]
        
        # Generate embeddings
        embeddings = self.generate_embeddings(texts)
        
        # Generate IDs
        ids = [f"static_{uuid.uuid4().hex}" for _ in range(len(documents))]
        
        # Add to collection
        self.static_collection.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        
        logger.info(f"Successfully added {len(documents)} static documents")
        return {
            "total_added": len(documents),
            "message": f"Added {len(documents)} static documents to collection"
        }
    
    def add_dynamic_documents(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Add dynamic documents to the dynamic collection
        
        Args:
            documents: List of document dictionaries with 'content' and 'metadata'
            
        Returns:
            Summary of the operation
        """
        if not documents:
            return {"total_added": 0, "message": "No documents to add"}
        
        logger.info(f"Adding {len(documents)} dynamic documents...")
        
        # Extract texts and metadata
        texts = [doc['content'] for doc in documents]
        metadatas = [doc['metadata'] for doc in documents]
        
        # Generate embeddings
        embeddings = self.generate_embeddings(texts)
        
        # Generate IDs
        ids = [f"dynamic_{uuid.uuid4().hex}" for _ in range(len(documents))]
        
        # Add to collection
        self.dynamic_collection.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        
        logger.info(f"Successfully added {len(documents)} dynamic documents")
        return {
            "total_added": len(documents),
            "message": f"Added {len(documents)} dynamic documents to collection"
        }
    
    def search_similar_chunks(self, query: str, n_results: int = 5, search_onchain: Optional[bool] = None, 
                            search_external: bool = True) -> List[Dict[str, Any]]:
        """
        Search for similar chunks across all available collections
        
        Args:
            query: Search query
            n_results: Number of results to return
            search_onchain: Whether to search dynamic data (if None, uses config)
            search_external: Whether to search external collections
            
        Returns:
            List of similar chunks with metadata
        """
        # Import here to avoid circular imports
        from ..rag.config import Config
        
        # Determine if we should search dynamic data
        if search_onchain is None:
            search_onchain = Config.SEARCH_ONCHAIN_DATA
        
        # Generate query embedding
        query_embedding = self.generate_embeddings([query])[0]
        
        all_results = []
        
        # Search static collection (always search unless it's empty)
        if self.static_collection.count() > 0:
            try:
                static_results = self.static_collection.query(
                    query_embeddings=[query_embedding],
                    n_results=n_results
                )
                
                # Process static results
                for i in range(len(static_results['documents'][0])):
                    result = {
                        'content': static_results['documents'][0][i],
                        'metadata': static_results['metadatas'][0][i],
                        'score': 1 - static_results['distances'][0][i],  # Convert distance to similarity
                        'source': 'static',
                        'collection': 'static_embeddings'
                    }
                    all_results.append(result)
                
                logger.info(f"Found {len(static_results['documents'][0])} static results")
                
            except Exception as e:
                logger.error(f"Error searching static collection: {e}")
        
        # Search dynamic collection (only if enabled and has data)
        if search_onchain and self.dynamic_collection.count() > 0:
            try:
                dynamic_results = self.dynamic_collection.query(
                    query_embeddings=[query_embedding],
                    n_results=n_results
                )
                
                # Process dynamic results
                for i in range(len(dynamic_results['documents'][0])):
                    result = {
                        'content': dynamic_results['documents'][0][i],
                        'metadata': dynamic_results['metadatas'][0][i],
                        'score': 1 - dynamic_results['distances'][0][i],  # Convert distance to similarity
                        'source': 'dynamic',
                        'collection': 'dynamic_embeddings'
                    }
                    all_results.append(result)
                
                logger.info(f"Found {len(dynamic_results['documents'][0])} dynamic results")
                
            except Exception as e:
                logger.error(f"Error searching dynamic collection: {e}")
        
        # Search external collections
        if search_external:
            external_collections = self.get_external_collections()
            
            for collection_info in external_collections:
                try:
                    collection = self.chroma_client.get_collection(collection_info['name'])
                    
                    if collection.count() > 0:
                        external_results = collection.query(
                            query_embeddings=[query_embedding],
                            n_results=n_results
                        )
                        
                        # Process external results
                        for i in range(len(external_results['documents'][0])):
                            result = {
                                'content': external_results['documents'][0][i],
                                'metadata': external_results['metadatas'][0][i],
                                'score': 1 - external_results['distances'][0][i],  # Convert distance to similarity
                                'source': 'external',
                                'collection': collection_info['name']
                            }
                            all_results.append(result)
                        
                        logger.info(f"Found {len(external_results['documents'][0])} results from {collection_info['name']}")
                
                except Exception as e:
                    logger.error(f"Error searching external collection {collection_info['name']}: {e}")
        
        # Sort all results by score (highest first) and return top n_results
        all_results.sort(key=lambda x: x['score'], reverse=True)
        final_results = all_results[:n_results]
        
        logger.info(f"Returning {len(final_results)} total results from {len(set(r['collection'] for r in final_results))} collections")
        return final_results
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about all collections"""
        stats = {
            'static_collection': {
                'count': self.static_collection.count(),
                'name': 'static_embeddings'
            },
            'dynamic_collection': {
                'count': self.dynamic_collection.count(),
                'name': 'dynamic_embeddings'
            },
            'external_collections': [],
            'total_documents': self.static_collection.count() + self.dynamic_collection.count()
        }
        
        # Add external collections
        external_collections = self.get_external_collections()
        for collection_info in external_collections:
            stats['external_collections'].append({
                'name': collection_info['name'],
                'count': collection_info['count'],
                'metadata': collection_info.get('metadata', {})
            })
            stats['total_documents'] += collection_info['count']
        
        return stats
    
    def clear_static_collection(self):
        """Clear the static collection"""
        try:
            self.chroma_client.delete_collection("static_embeddings")
            self.static_collection = self.chroma_client.create_collection(
                name="static_embeddings",
                metadata={"description": "Static data embeddings from .txt files"}
            )
            logger.info("Cleared static collection")
        except Exception as e:
            logger.error(f"Error clearing static collection: {e}")
    
    def clear_dynamic_collection(self):
        """Clear the dynamic collection"""
        try:
            self.chroma_client.delete_collection("dynamic_embeddings")
            self.dynamic_collection = self.chroma_client.create_collection(
                name="dynamic_embeddings",
                metadata={"description": "Dynamic data embeddings from .json files"}
            )
            logger.info("Cleared dynamic collection")
        except Exception as e:
            logger.error(f"Error clearing dynamic collection: {e}")
    
    def clear_external_collection(self, collection_name: str):
        """Clear a specific external collection"""
        try:
            self.chroma_client.delete_collection(collection_name)
            logger.info(f"Cleared external collection: {collection_name}")
        except Exception as e:
            logger.error(f"Error clearing external collection {collection_name}: {e}")
    
    def clear_all_collections(self):
        """Clear all collections"""
        try:
            # Get all collections
            all_collections = self.chroma_client.list_collections()
            
            for collection in all_collections:
                self.chroma_client.delete_collection(collection.name)
                logger.info(f"Deleted collection: {collection.name}")
            
            # Reinitialize core collections
            self._init_collections()
            logger.info("Cleared all collections and reinitialized core collections")
        except Exception as e:
            logger.error(f"Error clearing all collections: {e}")


# Legacy compatibility - keep old class names but point to new implementation
class EmbeddingManager(DualEmbeddingManager):
    """Legacy compatibility class"""
    pass

class MultiCollectionEmbeddingManager(DualEmbeddingManager):
    """Legacy compatibility class"""
    pass 