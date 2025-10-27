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
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .config import Config
from .auth import authenticate_request, get_auth_status
from ..utils.embeddings import EmbeddingManager
from ..utils.qa_generator import QAGenerator
from ..guardrail.guardrail import check_with_guardrail_async
from ..utils.rate_limiter import check_rate_limit, get_client_stats
from .chunks_reranker import rerank_static_chunks
from ..utils.slack_bot import SlackBot

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global components
static_embedding_manager: Optional[EmbeddingManager] = None
dynamic_embedding_manager: Optional[EmbeddingManager] = None
qa_generator: Optional[QAGenerator] = None
slack_bot: Optional[SlackBot] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown"""
    # Startup
    global static_embedding_manager, dynamic_embedding_manager, qa_generator, slack_bot
    
    try:
        logger.info("Starting Polkadot AI Chatbot API...")
        
        # Validate configuration
        Config.validate_config()
        logger.info("Configuration validated")
        
        # Content guardrails now handled by Bedrock guardrails in the query endpoint
        logger.info("Bedrock guardrails will be used for content moderation")
        
        # Initialize static embedding manager
        logger.info("Initializing static embedding manager...")
        static_embedding_manager = EmbeddingManager(
            openai_api_key=Config.OPENAI_API_KEY,
            embedding_model=Config.OPENAI_EMBEDDING_MODEL,
            chroma_persist_directory=Config.CHROMA_PERSIST_DIRECTORY,
            collection_name=Config.CHROMA_COLLECTION_NAME
        )
        
        # Initialize dynamic embedding manager
        logger.info("Initializing dynamic embedding manager...")
        dynamic_embedding_manager = EmbeddingManager(
            openai_api_key=Config.OPENAI_API_KEY,
            embedding_model=Config.OPENAI_EMBEDDING_MODEL,
            chroma_persist_directory=Config.CHROMA_PERSIST_DIRECTORY,
            collection_name=Config.CHROMA_DYNAMIC_COLLECTION_NAME
        )
        
        # Check if collections have data
        if not static_embedding_manager.collection_exists():
            logger.warning("Static ChromaDB collection is empty. Please run create_embeddings.py first.")
        else:
            stats = static_embedding_manager.get_collection_stats()
            logger.info(f"Loaded static collection with {stats.get('total_chunks', 0)} chunks")
        
        if not dynamic_embedding_manager.collection_exists():
            logger.warning("Dynamic ChromaDB collection is empty. Please run create_dynamic_embeddings.py first.")
        else:
            stats = dynamic_embedding_manager.get_collection_stats()
            logger.info(f"Loaded dynamic collection with {stats.get('total_chunks', 0)} chunks")
        
        # Initialize QA generator
        logger.info("Initializing QA generator...")
        qa_generator = QAGenerator(
            openai_api_key=Config.OPENAI_API_KEY,
            model=Config.OPENAI_MODEL,
            temperature=0.1,
            enable_web_search=Config.ENABLE_WEB_SEARCH,
            web_search_context_size=Config.WEB_SEARCH_CONTEXT_SIZE,
            enable_memory=Config.USE_MEM0 and bool(Config.MEM0_API_KEY)  # Enable memory only if USE_MEM0 is true and API key is provided
        )
        
        # Initialize Slack bot for error reporting
        try:
            logger.info("Initializing Slack bot for error reporting...")
            slack_bot = SlackBot()
            logger.info("Slack bot initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize Slack bot: {e}. Error reporting to Slack will be disabled.")
            slack_bot = None
        
        logger.info("API startup completed successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize API: {e}")
        raise e
    
    # Application is now ready
    yield
    
    # Shutdown
    logger.info("Shutting down Polkadot AI Chatbot API...")

# Initialize FastAPI app (after lifespan function is defined)
app = FastAPI(
    title="Polkadot AI Chatbot API",
    description="AI-powered chatbot for Polkadot ecosystem questions",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class ConversationMessage(BaseModel):
    query: str = Field(..., description="Previous user query")
    response: str = Field(..., description="Previous AI response")
    timestamp: str = Field(..., description="ISO timestamp of the message")

class QueryRequest(BaseModel):
    question: str = Field(..., description="The question to ask", min_length=1, max_length=500)
    user_id: str = Field(..., description="Unique user identifier", min_length=1, max_length=100)
    client_ip: str = Field(..., description="Client IP address or client identifier")
    max_chunks: int = Field(default=5, description="Maximum number of chunks to retrieve", ge=1, le=10)
    include_sources: bool = Field(default=True, description="Whether to include source information")
    custom_prompt: Optional[str] = Field(default=None, description="Custom system prompt for the AI")
    conversation_history: Optional[List[ConversationMessage]] = Field(default=[], description="Previous conversation history")

class Source(BaseModel):
    title: str
    url: str
    source_type: str
    similarity_score: float

class QueryResponse(BaseModel):
    answer: str
    sources: List[Source]
    follow_up_questions: List[str]
    remaining_requests: int = Field(..., description="Number of remaining requests for this user")
    confidence: float = Field(..., description="Confidence score of the answer")
    context_used: bool = Field(..., description="Whether context was used")
    model_used: str = Field(..., description="AI model used")
    chunks_used: int = Field(..., description="Number of chunks used")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    timestamp: str = Field(..., description="Response timestamp")
    search_method: str = Field(..., description="Method used for search")

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



@app.get("/health", response_model=HealthResponse)
async def health_check(authenticated: bool = Depends(authenticate_request)):
    """Health check endpoint"""
    try:
        if not static_embedding_manager or not dynamic_embedding_manager:
            raise HTTPException(status_code=503, detail="Embedding managers not initialized")
        
        stats = static_embedding_manager.get_collection_stats()
        
        return HealthResponse(
            status="healthy" if stats.get('total_chunks', 0) > 0 else "no_data",
            collection_stats=stats,
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=str(e))

@app.post("/query", response_model=QueryResponse)
async def query_chatbot(request: QueryRequest, authenticated: bool = Depends(authenticate_request)):
    """Main chatbot query endpoint with enhanced guardrails and rate limiting"""
    start_time = datetime.now()
    
    try:
        if not static_embedding_manager or not dynamic_embedding_manager or not qa_generator:
            raise HTTPException(status_code=503, detail="Service not initialized")
        
        if not static_embedding_manager.collection_exists() and not dynamic_embedding_manager.collection_exists():
            raise HTTPException(status_code=503, detail="No data available. Please create embeddings first.")
        
        # Check rate limit using user_id as primary identifier
        is_allowed, remaining_requests = check_rate_limit(request.user_id)
        
        if not is_allowed:
            # Rate limit exceeded
            logger.warning(f"Rate limit exceeded for user {request.user_id} from IP {request.client_ip}")
            raise HTTPException(
                status_code=429, 
                detail="Rate limit exceeded. Please try again later."
            )
        
        # 🛡️ Bedrock Guardrail content moderation
        logger.info(f"Processing query from user {request.user_id}: '{request.question[:50]}...' (remaining: {remaining_requests})")
        
        guardrail_result = await check_with_guardrail_async(request.question)
        
        if guardrail_result["status"] == "blocked":
            logger.warning(f"Query blocked by guardrail for user {request.user_id} from IP {request.client_ip}: {guardrail_result['reason']}")
            return QueryResponse(
                answer="This query violates our terms of service. Continued violations may result in your IP being blocked.",
                sources=[],
                follow_up_questions=[
                    "How does Polkadot's governance system work?",
                    "What are the benefits of staking DOT tokens?",
                    "How do parachains communicate with each other?"
                ],
                remaining_requests=remaining_requests,
                confidence=0.0,
                context_used=False,
                model_used="guardrail",
                chunks_used=0,
                processing_time_ms=(datetime.now() - start_time).total_seconds() * 1000,
                timestamp=datetime.now().isoformat(),
                search_method="guardrail_blocked"
            )
        elif guardrail_result["status"] == "error":
            logger.error(f"Guardrail error for user {request.user_id}: {guardrail_result['reason']}")
            # Continue processing if guardrail fails - don't block legitimate queries due to technical issues
        
        # Search for relevant chunks in both collections
        static_chunks = static_embedding_manager.search_similar_chunks(
            query=request.question,
            n_results=request.max_chunks
        )
        # print("\033[92mAll the static chunks are\033[0m", static_chunks)

        # Apply reranking to prioritize chunks with images
        static_chunks = rerank_static_chunks(static_chunks)
        dynamic_chunks = []
       
        # Get max similarity score from each collection
        static_max_score = max([chunk['similarity_score'] for chunk in static_chunks]) if static_chunks else 0
        dynamic_max_score = max([chunk['similarity_score'] for chunk in dynamic_chunks]) if dynamic_chunks else 0

        # Use chunks from the collection with higher max similarity score
        chunks = static_chunks if static_max_score >= dynamic_max_score else dynamic_chunks
        max_score = max(static_max_score, dynamic_max_score)
        logger.info(f"for {request.question}, max score is, {max_score}")
        which_chunk = "static" if static_max_score >= dynamic_max_score else "dynamic"
        logger.info(f"Genrating response from {which_chunk}")


        if not chunks:
            # Generate fallback follow-up questions when no results found
            fallback_questions = [
                "How does Polkadot differ from other blockchains?",
                "What are the main benefits of using Polkadot?", 
                "How can I get started with Polkadot?"
            ]
            
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return QueryResponse(
                answer="I couldn't find relevant information to answer your question. Please try rephrasing your query or ask about a different topic.",
                sources=[],
                follow_up_questions=fallback_questions,
                remaining_requests=remaining_requests,
                confidence=0.0,
                context_used=False,
                model_used=Config.OPENAI_MODEL,
                chunks_used=0,
                processing_time_ms=processing_time,
                timestamp=datetime.now().isoformat(),
                search_method="no_results"
            )
        
        # Generate answer using QA generator
        try:
            qa_result = await qa_generator.generate_answer(
                query=request.question,
                chunks=chunks,
                custom_prompt=request.custom_prompt,
                user_id=request.user_id,
                conversation_history=request.conversation_history
            )
        except Exception as qa_error:
            # Log the error locally
            logger.error(f"Error in qa_generator.generate_answer: {qa_error}")
            
            # Send error details to Slack if available
            if slack_bot:
                try:
                    error_context = {
                        "query": request.question,
                        "user_id": request.user_id,
                        "timestamp": datetime.now().isoformat(),
                        "error_type": "qa_generator_error",
                        "chunks_count": len(chunks) if chunks else 0
                    }
                    slack_bot.post_error_to_slack(
                        error_message=f"QA Generator Error: {str(qa_error)}",
                        context=error_context
                    )
                except Exception as slack_error:
                    logger.error(f"Failed to send error to Slack: {slack_error}")
            
            # Return user-friendly error response
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            return QueryResponse(
                answer="I am currently having problems processing your prompt. Try it again in your next prompt.",
                sources=[],
                follow_up_questions=[
                    "How does Polkadot's governance system work?",
                    "What are the benefits of staking DOT tokens?",
                    "How do parachains communicate with each other?"
                ],
                remaining_requests=remaining_requests,
                confidence=0.0,
                context_used=False,
                model_used=Config.OPENAI_MODEL,
                chunks_used=len(chunks) if chunks else 0,
                processing_time_ms=processing_time,
                timestamp=datetime.now().isoformat(),
                search_method="qa_generator_error"
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
        
        logger.info(f"Query processed in {processing_time:.2f}ms for user {request.user_id} (remaining: {remaining_requests})")
        print(qa_result['answer'])

        return QueryResponse(
            answer=qa_result['answer'],
            sources=sources,
            follow_up_questions=qa_result.get('follow_up_questions', []),
            remaining_requests=remaining_requests,
            confidence=qa_result.get('confidence', 0.0),
            context_used=qa_result.get('context_used', False),
            model_used=qa_result.get('model_used', Config.OPENAI_MODEL),
            chunks_used=qa_result.get('chunks_used', 0),
            processing_time_ms=processing_time,
            timestamp=datetime.now().isoformat(),
            search_method=qa_result.get('search_method', 'unknown')
        )
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        # Try to get remaining requests if possible, otherwise default to 0
        try:
            _, remaining_requests = check_rate_limit(request.user_id)
        except:
            remaining_requests = 0
        
        # Return helpful response even on error
        return QueryResponse(
            answer="I'd be happy to help you with Polkadot questions! What would you like to know about governance, staking, or parachains?",
            sources=[],
            follow_up_questions=[
                "How does Polkadot's governance system work?",
                "What are the benefits of staking DOT tokens?",
                "How do parachains communicate with each other?"
            ],
            remaining_requests=remaining_requests,
            confidence=0.0,
            context_used=False,
            model_used=Config.OPENAI_MODEL,
            chunks_used=0,
            processing_time_ms=(datetime.now() - start_time).total_seconds() * 1000,
            timestamp=datetime.now().isoformat(),
            search_method="error"
        )

@app.post("/search", response_model=SearchResponse)
async def search_documents(request: SearchRequest, authenticated: bool = Depends(authenticate_request)):
    """Search for relevant document chunks"""
    start_time = datetime.now()
    
    try:
        if not static_embedding_manager or not dynamic_embedding_manager:
            raise HTTPException(status_code=503, detail="Service not initialized")
        
        if not static_embedding_manager.collection_exists() and not dynamic_embedding_manager.collection_exists():
            raise HTTPException(status_code=503, detail="No data available. Please create embeddings first.")
        
        logger.info(f"Processing search: '{request.query[:50]}...'")
        
        # Prepare filter if source filter is specified
        filter_metadata = None
        if request.source_filter:
            filter_metadata = {"source": request.source_filter}
        
        # Search for chunks
        static_chunks = static_embedding_manager.search_similar_chunks(
            query=request.query,
            n_results=request.n_results,
            filter_metadata=filter_metadata
        )
        
        dynamic_chunks = dynamic_embedding_manager.search_similar_chunks(
            query=request.query,
            n_results=request.n_results,
            filter_metadata=filter_metadata
        )

        # Combine and sort chunks by similarity score
        all_chunks = static_chunks + dynamic_chunks
        all_chunks.sort(key=lambda x: x['similarity_score'], reverse=True)
        results = [
            SearchResult(
                content=chunk['content'],
                metadata=chunk['metadata'],
                similarity_score=chunk['similarity_score']
            )
            for chunk in all_chunks[:request.n_results]
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
async def get_collection_stats(authenticated: bool = Depends(authenticate_request)):
    """Get collection statistics"""
    try:
        if not static_embedding_manager or not dynamic_embedding_manager:
            raise HTTPException(status_code=503, detail="Service not initialized")
        
        stats = static_embedding_manager.get_collection_stats()
        return {
            "collection_stats": stats,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/rate-limit/{user_id}")
async def get_rate_limit_status(user_id: str, authenticated: bool = Depends(authenticate_request)):
    """Get rate limit status for a user"""
    try:
        stats = get_client_stats(user_id)
        return {
            "user_id": user_id,
            "rate_limit_stats": stats,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting rate limit stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/auth-status")
async def get_authentication_status():
    """Get authentication configuration status (public endpoint)"""
    return get_auth_status()

@app.get("/")
async def root():
    """Root endpoint with API information (public endpoint)"""
    auth_info = get_auth_status()
    
    base_info = {
        "message": "Polkadot AI Chatbot API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "authentication": auth_info,
        "endpoints": {
            "query": "POST /query - Ask questions about Polkadot",
            "search": "POST /search - Search document chunks", 
            "stats": "GET /stats - Get collection statistics",
            "health": "GET /health - Health check",
            "rate-limit": "GET /rate-limit/{user_id} - Get rate limit status",
            "auth-status": "GET /auth-status - Get authentication status"
        }
    }
    
    if auth_info["authentication_enabled"]:
        base_info["authentication_required"] = True
        base_info["auth_header_example"] = auth_info["auth_header_format"]
        base_info["note"] = "All endpoints except / and /auth-status require authentication"
    
    return base_info

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