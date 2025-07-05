#!/usr/bin/env python3
"""
FastAPI server for the Polkadot AI Chatbot system.
Provides endpoints for querying the knowledge base and getting AI-generated answers.
"""

import os
import sys
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .config import Config
from ..utils.embeddings import EmbeddingManager
from ..utils.qa_generator import QAGenerator
from ..utils.content_guardrails import get_guardrails

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Polkadot AI Chatbot API",
    description="AI-powered chatbot for Polkadot ecosystem questions",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global components
embedding_manager: Optional[EmbeddingManager] = None
qa_generator: Optional[QAGenerator] = None
content_guardrails = None

# Request/Response models
class QueryRequest(BaseModel):
    question: str = Field(..., description="The question to ask", min_length=1, max_length=500)
    max_chunks: int = Field(default=5, description="Maximum number of chunks to retrieve", ge=1, le=10)
    include_sources: bool = Field(default=True, description="Whether to include source information")
    custom_prompt: Optional[str] = Field(default=None, description="Custom system prompt for the AI")

class Source(BaseModel):
    title: str
    url: str
    source_type: str
    similarity_score: float

class QueryResponse(BaseModel):
    answer: str
    sources: List[Source]
    confidence: float
    follow_up_questions: List[str]
    context_used: bool
    model_used: str
    chunks_used: int
    processing_time_ms: float
    timestamp: str
    search_method: Optional[str] = Field(default=None, description="Method used for search (local_knowledge, web_search, etc.)")

class HealthResponse(BaseModel):
    status: str
    collection_stats: Dict[str, Any]
    timestamp: str

class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query", min_length=1, max_length=200)
    n_results: int = Field(default=5, description="Number of results to return", ge=1, le=20)
    source_filter: Optional[str] = Field(default=None, description="Filter by source type")

class SearchResult(BaseModel):
    content: str
    metadata: Dict[str, Any]
    similarity_score: float

class SearchResponse(BaseModel):
    results: List[SearchResult]
    total_results: int
    processing_time_ms: float
    timestamp: str

@app.on_event("startup")
async def startup_event():
    """Initialize the application components"""
    global embedding_manager, qa_generator, content_guardrails
    
    try:
        logger.info("Starting Polkadot AI Chatbot API...")
        
        # Validate configuration
        Config.validate_config()
        logger.info("Configuration validated")
        
        # Initialize enhanced content guardrails
        content_guardrails = get_guardrails(Config.OPENAI_API_KEY)
        logger.info("Enhanced AI-powered content guardrails initialized")
        
        # Initialize embedding manager
        logger.info("Initializing embedding manager...")
        embedding_manager = EmbeddingManager(
            openai_api_key=Config.OPENAI_API_KEY,
            embedding_model=Config.OPENAI_EMBEDDING_MODEL,
            chroma_persist_directory=Config.CHROMA_PERSIST_DIRECTORY,
            collection_name=Config.CHROMA_COLLECTION_NAME
        )
        
        # Check if collection has data
        if not embedding_manager.collection_exists():
            logger.warning("ChromaDB collection is empty. Please run create_embeddings.py first.")
        else:
            stats = embedding_manager.get_collection_stats()
            logger.info(f"Loaded collection with {stats.get('total_chunks', 0)} chunks")
        
        # Initialize QA generator
        logger.info("Initializing QA generator...")
        qa_generator = QAGenerator(
            openai_api_key=Config.OPENAI_API_KEY,
            model=Config.OPENAI_MODEL,
            temperature=0.1,
            enable_web_search=Config.ENABLE_WEB_SEARCH,
            web_search_context_size=Config.WEB_SEARCH_CONTEXT_SIZE,
            enable_memory=bool(Config.MEM0_API_KEY)  # Enable memory only if API key is provided
        )
        
        logger.info("API startup completed successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize API: {e}")
        raise e

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        if not embedding_manager:
            raise HTTPException(status_code=503, detail="Embedding manager not initialized")
        
        stats = embedding_manager.get_collection_stats()
        
        return HealthResponse(
            status="healthy" if stats.get('total_chunks', 0) > 0 else "no_data",
            collection_stats=stats,
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=str(e))

@app.post("/query", response_model=QueryResponse)
async def query_chatbot(request: QueryRequest):
    """Main chatbot query endpoint with enhanced guardrails"""
    start_time = datetime.now()
    
    try:
        if not embedding_manager or not qa_generator or not content_guardrails:
            raise HTTPException(status_code=503, detail="Service not initialized")
        
        if not embedding_manager.collection_exists():
            raise HTTPException(status_code=503, detail="No data available. Please create embeddings first.")
        
        # 🛡️ AI-powered content moderation
        is_safe, category, helpful_response = content_guardrails.moderate_content(request.question)
        
        if not is_safe:
            logger.warning(f"Unsafe query blocked - Category: {category}")
            return QueryResponse(
                answer=helpful_response,
                sources=[],
                confidence=0.0,
                follow_up_questions=[
                    "How does Polkadot's governance system work?",
                    "What are the benefits of staking DOT tokens?",
                    "How do parachains communicate with each other?"
                ],
                context_used=False,
                model_used=Config.OPENAI_MODEL,
                chunks_used=0,
                processing_time_ms=(datetime.now() - start_time).total_seconds() * 1000,
                timestamp=datetime.now().isoformat(),
                search_method=f"blocked_{category}"
            )
        
        logger.info(f"Processing query: '{request.question[:50]}...'")
        
        # Search for relevant chunks
        chunks = embedding_manager.search_similar_chunks(
            query=request.question,
            n_results=request.max_chunks
        )
        
        if not chunks:
            # Generate fallback follow-up questions when no results found
            fallback_questions = [
                "How does Polkadot differ from other blockchains?",
                "What are the main benefits of using Polkadot?", 
                "How can I get started with Polkadot?"
            ]
            
            return QueryResponse(
                answer="I couldn't find relevant information to answer your question. Please try rephrasing your query or ask about a different topic.",
                sources=[],
                confidence=0.0,
                follow_up_questions=fallback_questions,
                context_used=False,
                model_used=Config.OPENAI_MODEL,
                chunks_used=0,
                processing_time_ms=(datetime.now() - start_time).total_seconds() * 1000,
                timestamp=datetime.now().isoformat(),
                search_method="no_results"
            )
        
        # Generate answer using QA generator
        qa_result = qa_generator.generate_answer(
            query=request.question,
            chunks=chunks,
            custom_prompt=request.custom_prompt
        )
        
        # Format sources if requested
        sources = []
        if request.include_sources:
            sources = [
                Source(
                    title=src['title'],
                    url=src['url'],
                    source_type=src['source_type'],
                    similarity_score=src['similarity_score']
                )
                for src in qa_result.get('sources', [])
            ]
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        logger.info(f"Query processed in {processing_time:.2f}ms")
        
        return QueryResponse(
            answer=qa_result['answer'],
            sources=sources,
            confidence=qa_result.get('confidence', 0.0),
            follow_up_questions=qa_result.get('follow_up_questions', []),
            context_used=qa_result.get('context_used', False),
            model_used=qa_result.get('model_used', Config.OPENAI_MODEL),
            chunks_used=qa_result.get('chunks_used', 0),
            processing_time_ms=processing_time,
            timestamp=datetime.now().isoformat(),
            search_method=qa_result.get('search_method', 'unknown')
        )
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        # Return helpful response even on error
        return QueryResponse(
            answer="I'd be happy to help you with Polkadot questions! What would you like to know about governance, staking, or parachains?",
            sources=[],
            confidence=0.0,
            follow_up_questions=[
                "How does Polkadot's governance system work?",
                "What are the benefits of staking DOT tokens?",
                "How do parachains communicate with each other?"
            ],
            context_used=False,
            model_used=Config.OPENAI_MODEL,
            chunks_used=0,
            processing_time_ms=(datetime.now() - start_time).total_seconds() * 1000,
            timestamp=datetime.now().isoformat(),
            search_method="error"
        )

@app.post("/search", response_model=SearchResponse)
async def search_documents(request: SearchRequest):
    """Search for relevant document chunks"""
    start_time = datetime.now()
    
    try:
        if not embedding_manager:
            raise HTTPException(status_code=503, detail="Service not initialized")
        
        if not embedding_manager.collection_exists():
            raise HTTPException(status_code=503, detail="No data available. Please create embeddings first.")
        
        logger.info(f"Processing search: '{request.query[:50]}...'")
        
        # Prepare filter if source filter is specified
        filter_metadata = None
        if request.source_filter:
            filter_metadata = {"source": request.source_filter}
        
        # Search for chunks
        chunks = embedding_manager.search_similar_chunks(
            query=request.query,
            n_results=request.n_results,
            filter_metadata=filter_metadata
        )
        
        # Format results
        results = [
            SearchResult(
                content=chunk['content'],
                metadata=chunk['metadata'],
                similarity_score=chunk['similarity_score']
            )
            for chunk in chunks
        ]
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        logger.info(f"Search processed in {processing_time:.2f}ms, found {len(results)} results")
        
        return SearchResponse(
            results=results,
            total_results=len(results),
            processing_time_ms=processing_time,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error processing search: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/stats")
async def get_collection_stats():
    """Get collection statistics"""
    try:
        if not embedding_manager:
            raise HTTPException(status_code=503, detail="Service not initialized")
        
        stats = embedding_manager.get_collection_stats()
        return {
            "collection_stats": stats,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Polkadot AI Chatbot API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "query": "POST /query - Ask questions about Polkadot",
            "search": "POST /search - Search document chunks",
            "stats": "GET /stats - Get collection statistics",
            "health": "GET /health - Health check"
        }
    }

if __name__ == "__main__":
    import uvicorn
    
    logger.info("Starting Polkadot AI Chatbot API server...")
    uvicorn.run(
        app,
        host=Config.API_HOST,
        port=Config.API_PORT,
        log_level="info",
        reload=False
    ) 