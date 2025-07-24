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
    CHROMA_DYNAMIC_COLLECTION_NAME = os.getenv("CHROMA_DYNAMIC_COLLECTION_NAME", "polkadot_embeddings_dynamic")
    
    # Search Configuration
    SEARCH_STATIC_DATA = os.getenv("SEARCH_STATIC_DATA", "true").lower() == "true"
    SEARCH_DYNAMIC_DATA = os.getenv("SEARCH_DYNAMIC_DATA", "true").lower() == "true"
    
    # Mem0 Memory Configuration
    USE_MEM0 = os.getenv("USE_MEM0", "false").lower() in ("true", "1", "yes", "on")
    MEM0_API_KEY = os.getenv("MEM0_API_KEY", "")
    
    # Redis Rate Limiting Configuration
    REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
    REDIS_DB = int(os.getenv("REDIS_DB", 0))
    REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)
    RATE_LIMIT_MAX_REQUESTS = int(os.getenv("RATE_LIMIT_MAX_REQUESTS", 20))
    RATE_LIMIT_EXPIRE_SECONDS = int(os.getenv("RATE_LIMIT_EXPIRE_SECONDS", 3600))
    
    # API Configuration
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", 8000))
    
    # Chunk Configuration
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))
    
    # Retrieval Configuration
    TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", 5))
    SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", 0.7))
    
    # Data paths
    STATIC_DATA_PATH = os.getenv("STATIC_DATA_PATH", "data/joined_data/static")  # Path for static documentation
    DYNAMIC_DATA_PATH = os.getenv("DYNAMIC_DATA_PATH", "data/dynamic_kusama_polka")  # Path for dynamic Polkadot/Kusama data
    DATA_SOURCES_PATH = DYNAMIC_DATA_PATH  # Backward compatibility
    POLKADOT_NETWORK_PATH = os.path.join(DATA_SOURCES_PATH, "polkadot_network")
    POLKADOT_WIKI_PATH = os.path.join(DATA_SOURCES_PATH, "polkadot_wiki")
    
    # Content Guardrails Configuration
    ENABLE_CONTENT_FILTERING = os.getenv("ENABLE_CONTENT_FILTERING", "true").lower() == "true"
    BLOCKED_DOMAINS = os.getenv("BLOCKED_DOMAINS", "subsquare.io,subsquare.com,subsquare.network").split(",")
    PREFERRED_DOMAINS = os.getenv("PREFERRED_DOMAINS", "polkadot.io,polkadot.network,polkassembly.io").split(",")
    
    # Safety Settings
    MAX_QUERY_LENGTH = int(os.getenv("MAX_QUERY_LENGTH", 500))
    ENABLE_OFFENSIVE_FILTER = os.getenv("ENABLE_OFFENSIVE_FILTER", "true").lower() == "true"
    
    @classmethod
    def validate_config(cls):
        """Validate that required configuration is present"""
        if not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY must be set in environment variables")
        
        if not os.path.exists(cls.DATA_SOURCES_PATH):
            raise ValueError(f"Data sources path does not exist: {cls.DATA_SOURCES_PATH}")
            
        return True 