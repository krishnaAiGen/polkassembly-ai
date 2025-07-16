import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    # OpenAI Configuration
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
    OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002")
    
    # Web Search Configuration
    WEB_SEARCH = os.getenv("WEB_SEARCH", "true").lower() == "true"
    ENABLE_WEB_SEARCH = WEB_SEARCH  # Alias for backward compatibility
    WEB_SEARCH_CONTEXT_SIZE = os.getenv("WEB_SEARCH_CONTEXT_SIZE", "high")  # low, medium, high
    SIMILARITY_THRESHOLD_FOR_WEB_SEARCH = float(os.getenv("SIMILARITY_THRESHOLD_FOR_WEB_SEARCH", 0.3))
    
    # ChromaDB Configuration
    CHROMA_PERSIST_DIRECTORY = os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db")
    CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "polkadot_embeddings")
    
    # API Configuration
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", 8000))
    
    # Chunk Configuration
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))
    
    # Retrieval Configuration
    TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", 5))
    SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", 0.7))
    SEARCH_ONCHAIN_DATA = os.getenv("SEARCH_ONCHAIN_DATA", "true").lower() == "true"
    
    # Data Sources Configuration
    STATIC_DATA_SOURCE = os.getenv("STATIC_DATA_SOURCE", "data/data_sources/static_data")
    DYNAMIC_DATA_SOURCE = os.getenv("DYNAMIC_DATA_SOURCE", "data/data_sources/onchain_data")
    
    @classmethod
    def validate_config(cls):
        """Validate that required configuration is present"""
        if not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY must be set in environment variables")
        
        if not os.path.exists(cls.STATIC_DATA_SOURCE):
            raise ValueError(f"Static data source path does not exist: {cls.STATIC_DATA_SOURCE}")
            
        if not os.path.exists(cls.DYNAMIC_DATA_SOURCE):
            raise ValueError(f"Dynamic data source path does not exist: {cls.DYNAMIC_DATA_SOURCE}")
            
        return True 