# Polkadot AI Chatbot

An AI-powered chatbot system that provides intelligent answers about the Polkadot ecosystem using RAG (Retrieval-Augmented Generation) with OpenAI embeddings and ChromaDB.

## Features

- 📚 **Knowledge Base**: Built from Polkadot Wiki and Forum data
- 🔍 **Semantic Search**: Uses OpenAI embeddings for accurate document retrieval
- 🤖 **AI-Powered Answers**: Generates contextual responses using GPT models
- 🚀 **Fast API**: RESTful API with automatic documentation
- 💾 **Persistent Storage**: ChromaDB for efficient vector storage
- 🔧 **Configurable**: Environment-based configuration
- 📊 **Analytics**: Built-in statistics and monitoring
- 🚀 **Production Ready**: Gunicorn with multiple workers for scalability
- 🔄 **Easy Management**: Startup/stop scripts for complete system management
- 📝 **Comprehensive Logging**: Separate log files for each service component

## Project Structure

```
polkassembly-ai/
├── README.md                  # Project documentation
├── env.example                # Environment variables template
├── requirements.txt           # Python dependencies
├── start_pa_ai.sh            # Startup script (Redis + API with Gunicorn)
├── stop_pa_ai.sh             # Stop script for all services
├── create_embeddings.py       # Entry point to generate embeddings (run once)
├── run_server.py              # Entry point for API server
├── run_tests.py               # Entry point for testing
├── data/                      # Data sources
│   └── data_sources/
│       ├── polkadot_network/  # Forum data (JSON/TXT files)
│       └── polkadot_wiki/     # Wiki data (TXT files)
└── src/                       # Source code directory
    ├── rag/                   # RAG application modules
    │   ├── config.py          # Configuration management
    │   ├── create_embeddings.py # Embedding generation logic
    │   └── api_server.py      # FastAPI server
    ├── utils/                 # Utility modules
    │   ├── data_loader.py     # Document loading and processing
    │   ├── text_chunker.py    # Text chunking for embeddings
    │   ├── embeddings.py      # OpenAI embeddings and ChromaDB
    │   └── qa_generator.py    # Q&A generation with OpenAI
    └── test/                  # Testing modules
        └── test_api.py        # API testing script
```

## Setup Instructions

### 1. Clone and Install

```bash
# Install dependencies
pip install fastapi uvicorn openai chromadb python-dotenv tiktoken mem0ai redis python-redis-rate-limit
```

### 2. Environment Configuration

```bash
# Copy the example environment file
cp env.example .env

# Edit .env with your OpenAI API key
# OPENAI_API_KEY=your_actual_openai_api_key_here
```

### 3. Create Embeddings (One-time Setup)

This step processes all documents and creates embeddings:

```bash
# Generate embeddings and store in ChromaDB
python create_embeddings.py

# Optional: Clear existing embeddings and recreate
python create_embeddings.py --clear

# Optional: Customize batch size and token limits
python create_embeddings.py --batch-size 25 --min-tokens 100 --max-tokens 800
```

### 4. Start the API Server

#### Option A: Using the Startup Script (Recommended)
```bash
# Make scripts executable (first time only)
chmod +x start_pa_ai.sh stop_pa_ai.sh

# Start the complete system (Redis + API with Gunicorn)
./start_pa_ai.sh

# Stop the system
./stop_pa_ai.sh

# Check system status
./stop_pa_ai.sh status
```

#### Option B: Manual Startup
```bash
# Start the FastAPI server
python run_server.py

# Or using uvicorn directly
uvicorn src.rag.api_server:app --host 0.0.0.0 --port 8000

# Or using gunicorn with multiple workers
gunicorn --bind 0.0.0.0:8000 --workers 3 --worker-class uvicorn.workers.UvicornWorker src.rag.api_server:app
```

### 5. (Optional) Configure Memory

To enable conversation memory with Mem0:

```bash
# Install mem0 package
pip install mem0ai

# Add your Mem0 API key to .env file
echo "MEM0_API_KEY=your_mem0_api_key_here" >> .env
```

### 6. (Optional) Configure Rate Limiting

To enable Redis-based rate limiting:

```bash
# Install and start Redis server
# On macOS:
brew install redis
brew services start redis

# On Ubuntu/Debian:
sudo apt-get install redis-server
sudo systemctl start redis-server

# On CentOS/RHEL:
sudo yum install redis
sudo systemctl start redis

# Test Redis connection
redis-cli ping  # Should return "PONG"
```

### 7. Test the System

```bash
# Run all tests
python src/test/run_tests.py

# Run specific tests
python src/test/run_tests.py --api         # API functionality only
python src/test/run_tests.py --memory      # Memory integration only
python src/test/run_tests.py --formatting  # Clean formatting validation
python src/test/run_tests.py --rate-limit  # Rate limiting functionality
```

## API Endpoints

### 🏥 Health Check
```http
GET /health
```

### 🤖 Ask Questions
```http
POST /query
Content-Type: application/json

{
  "question": "What is Polkadot?",
  "user_id": "krishna",
  "client_ip": "192.168.1.1",
  "max_chunks": 5,
  "include_sources": true,
  "custom_prompt": "optional custom system prompt"
}
```

**Response:**
```json
{
  "answer": "Polkadot is a blockchain protocol...",
  "sources": [...],
  "follow_up_questions": ["...", "...", "..."],
  "remaining_requests": 7
}
```

### 🔍 Search Documents
```http
POST /search
Content-Type: application/json

{
  "query": "staking rewards",
  "n_results": 10,
  "source_filter": "polkadot_wiki"
}
```

### 📊 Get Statistics
```http
GET /stats
```

### 🔐 Rate Limiting

The API implements Redis-based rate limiting to prevent abuse:

- **Default Limits**: 20 requests per hour per user_id
- **Rate Limit Response**: HTTP 429 when limit exceeded
- **Per-User Tracking**: Each user_id has separate rate limits
- **Automatic Reset**: Limits reset after time window expires

**Check Rate Limit Status:**
```http
GET /rate-limit/{user_id}
```

**Response:**
```json
{
  "user_id": "krishna",
  "rate_limit_stats": {
    "max_requests": 20,
    "used_requests": 13,
    "remaining_requests": 7,
    "time_window_seconds": 3600
  }
}
```

## 🧠 Memory Integration

The system includes **conversation memory** powered by Mem0, enabling context-aware conversations:

### Features
- **Context Retention**: Remembers previous questions and answers
- **Smart Follow-ups**: Uses conversation history for better responses
- **Automatic Memory**: No manual memory management required
- **Privacy**: Isolated memory per user session

### Memory Flow
1. **User Query**: System searches memory for relevant context
2. **Context Injection**: Memory context is added to prompt
3. **Response Generation**: AI considers both documents and memory
4. **Memory Storage**: Query and response are automatically stored

### Example Conversation
```
User: "What is staking in Polkadot?"
Bot: "Staking in Polkadot allows DOT holders to..."

User: "What are the rewards for that?"
Bot: "Staking rewards in Polkadot include..." # Uses memory context
```

## Usage Examples

### Python Client Example

```python
import requests

# Ask a question
response = requests.post("http://localhost:8000/query", json={
    "question": "How do I stake DOT tokens?",
    "user_id": "test_user",
    "client_ip": "192.168.1.1",
    "max_chunks": 3,
    "include_sources": True
})

data = response.json()
print(f"Answer: {data['answer']}")
print(f"Remaining requests: {data['remaining_requests']}")
print(f"Sources: {len(data['sources'])}")
print(f"Follow-up questions: {data['follow_up_questions']}")
```

### cURL Examples

```bash
# Health check
curl http://localhost:8000/health

# Ask a question
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are parachains?",
    "user_id": "curl_user",
    "client_ip": "192.168.1.1",
    "max_chunks": 3,
    "include_sources": true
  }'

# Search documents
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "governance voting",
    "n_results": 5
  }'
```

## 🚀 Startup Scripts

The project includes comprehensive startup and stop scripts for easy system management:

### `start_pa_ai.sh` - System Startup
Automatically starts both Redis and the API server with optimal production settings:

**Features:**
- ✅ **Redis Server**: Starts Redis with proper configuration
- ✅ **Gunicorn Workers**: Runs API with 3 worker processes for scalability
- ✅ **Process Management**: Tracks PIDs for proper shutdown
- ✅ **Health Checks**: Verifies services are running correctly
- ✅ **Comprehensive Logging**: Separate log files for each component
- ✅ **Error Handling**: Graceful error handling and recovery

**Configuration (via environment variables):**
```bash
export API_HOST="0.0.0.0"      # API server host
export API_PORT="8000"         # API server port
export WORKERS="3"             # Number of Gunicorn workers
export REDIS_HOST="localhost"  # Redis host
export REDIS_PORT="6379"       # Redis port
```

### `stop_pa_ai.sh` - System Shutdown
Gracefully stops all services with multiple options:

**Usage:**
```bash
./stop_pa_ai.sh              # Graceful shutdown
./stop_pa_ai.sh force        # Force stop all processes
./stop_pa_ai.sh cleanup      # Stop and clean up PID files
./stop_pa_ai.sh status       # Show current system status
./stop_pa_ai.sh help         # Show help information
```

### Log Files
The startup script creates organized log files:
- `logs/redis.log` - Redis server logs
- `logs/api_server.log` - API server logs
- `logs/api_server_error.log` - API error logs
- `logs/access.log` - HTTP access logs

### Process Management
- PID files stored in `pids/` directory
- Automatic process detection and cleanup
- Graceful shutdown with timeout handling

## Configuration Options

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | - | Your OpenAI API key (required) |
| `OPENAI_MODEL` | gpt-3.5-turbo | OpenAI model for answers |
| `OPENAI_EMBEDDING_MODEL` | text-embedding-ada-002 | Embedding model |
| `MEM0_API_KEY` | - | Mem0 API key for conversation memory (optional) |
| `REDIS_HOST` | localhost | Redis server host for rate limiting |
| `REDIS_PORT` | 6379 | Redis server port |
| `REDIS_PASSWORD` | - | Redis password (if required) |
| `RATE_LIMIT_MAX_REQUESTS` | 20 | Maximum requests per time window |
| `RATE_LIMIT_EXPIRE_SECONDS` | 3600 | Rate limit time window (1 hour) |
| `WEB_SEARCH` | true | Enable web search fallback |
| `WEB_SEARCH_CONTEXT_SIZE` | high | Web search context size (low/medium/high) |
| `CHROMA_PERSIST_DIRECTORY` | ./chroma_db | ChromaDB storage path |
| `CHROMA_COLLECTION_NAME` | polkadot_embeddings | Collection name |
| `API_HOST` | 0.0.0.0 | API server host |
| `API_PORT` | 8000 | API server port |
| `CHUNK_SIZE` | 1000 | Text chunk size for embeddings |
| `CHUNK_OVERLAP` | 200 | Overlap between chunks |

## Data Sources

The system processes two types of data:

1. **Polkadot Wiki** (`data/data_sources/polkadot_wiki/`)
   - Technical documentation
   - Guides and tutorials
   - Architecture information

2. **Polkadot Forum** (`data/data_sources/polkadot_network/`)
   - Community discussions
   - Governance proposals
   - Ambassador program information

## API Documentation

Once the server is running, visit:
- **Interactive API Docs**: http://localhost:8000/docs
- **ReDoc Documentation**: http://localhost:8000/redoc

## Monitoring and Logs

- Embedding creation logs: `embedding_creation.log`
- API server logs: Console output
- ChromaDB storage: `./chroma_db/` directory

## Troubleshooting

### Common Issues

1. **"OPENAI_API_KEY must be set"**
   - Ensure your `.env` file contains a valid OpenAI API key

2. **"No data available. Please create embeddings first."**
   - Run `python create_embeddings.py` to initialize the database

3. **"Collection already exists with data"**
   - Use `--clear` flag to recreate embeddings: `python create_embeddings.py --clear`

4. **Rate limiting from OpenAI**
   - The system includes automatic retry logic
   - Consider reducing batch size: `--batch-size 10`

5. **"Rate limiting disabled due to Redis connection failure"**
   - Ensure Redis is running: `redis-server`
   - Check Redis connection: `redis-cli ping`
   - Verify Redis configuration in `.env` file

6. **"HTTP 429 Too Many Requests"**
   - User has exceeded rate limits (default: 20 requests/hour)
   - Check remaining requests: `GET /rate-limit/{user_id}`
   - Wait for rate limit reset or increase limits in config

### Performance Tips

- **Faster embedding creation**: Increase `--batch-size` (but watch rate limits)
- **Memory optimization**: Reduce `CHUNK_SIZE` in configuration
- **Better accuracy**: Increase `max_chunks` in queries (but slower)

## Development

### Project Structure

The project uses a clean, modular structure:
- **Root level**: Contains entry point scripts and documentation
- **`src/rag/`**: Contains RAG application logic (config, embeddings, API)
- **`src/utils/`**: Contains utility modules for data processing
- **`src/test/`**: Contains testing modules

### Adding New Data Sources

1. Place new documents in `data/data_sources/polkadot_network/` or `data/data_sources/polkadot_wiki/`
2. Run `python create_embeddings.py` to process new documents
3. The system will automatically detect and process new files

### Customizing the AI

- Modify prompts in `src/utils/qa_generator.py`
- Adjust embedding parameters in `src/utils/embeddings.py`
- Change chunking strategy in `src/utils/text_chunker.py`
- Update configuration in `src/rag/config.py`

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test with `python run_tests.py`
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review the API documentation at `/docs`
3. Check logs for error details