# Polkadot AI Chatbot - Complete Setup Guide

This guide provides step-by-step instructions for setting up the Polkadot AI Chatbot system with all features including rate limiting, memory, web search, and performance timing.

## 📋 Prerequisites

- **Python 3.8+** installed
- **Git** for cloning the repository
- **OpenAI API key** (required)
- **Redis server** (optional, for rate limiting)
- **Mem0 API key** (optional, for conversation memory)

## 🚀 Step-by-Step Setup

### 1. Clone and Install Dependencies

```bash
# Clone the repository
git clone <repository-url>
cd polkassembly-ai

# Install Python dependencies
pip install fastapi uvicorn openai chromadb python-dotenv tiktoken mem0ai redis python-redis-rate-limit gunicorn
```

### 2. Environment Configuration

```bash
# Copy the example environment file
cp env.example .env

# Edit the .env file with your configuration
nano .env  # or use your preferred editor
```

**Required configuration in `.env`:**
```bash
# OpenAI Configuration (REQUIRED)
OPENAI_API_KEY=your_actual_openai_api_key_here
OPENAI_MODEL=gpt-3.5-turbo
OPENAI_EMBEDDING_MODEL=text-embedding-ada-002

# Web Search Configuration (optional)
WEB_SEARCH=true
WEB_SEARCH_CONTEXT_SIZE=high

# Memory Configuration (optional)
MEM0_API_KEY=your_mem0_api_key_here

# Redis Rate Limiting Configuration (optional)
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
REDIS_PASSWORD=
RATE_LIMIT_MAX_REQUESTS=20
RATE_LIMIT_EXPIRE_SECONDS=3600

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000

# ChromaDB Configuration
CHROMA_PERSIST_DIRECTORY=./chroma_db
CHROMA_COLLECTION_NAME=polkadot_embeddings

# Text Processing Configuration
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
```

### 3. Create Embeddings Database (One-time Setup)

```bash
# Generate embeddings and store in ChromaDB
python create_embeddings.py

# Optional: Clear existing embeddings and recreate
python create_embeddings.py --clear

# Optional: Customize processing parameters
python create_embeddings.py --batch-size 25 --min-tokens 100 --max-tokens 800
```

**Expected output:**
```
🚀 Starting embedding creation process...
📁 Loading documents from data sources...
✅ Loaded 150 documents from polkadot_wiki
✅ Loaded 200 documents from polkadot_network
📝 Processing documents into chunks...
🔄 Creating embeddings (batch 1/20)...
✅ Embedding creation completed!
📊 Total chunks created: 1,250
💾 Embeddings saved to: ./chroma_db
```

### 4. Optional: Configure Redis for Rate Limiting

If you want to enable rate limiting, install and start Redis:

#### macOS (using Homebrew):
```bash
# Install Redis
brew install redis

# Start Redis as a service (runs in background)
brew services start redis

# Or start Redis manually (runs in foreground)
redis-server
```

#### Ubuntu/Debian:
```bash
# Install Redis
sudo apt-get update
sudo apt-get install redis-server

# Start Redis service
sudo systemctl start redis-server

# Enable Redis to start on boot
sudo systemctl enable redis-server
```

#### CentOS/RHEL:
```bash
# Install Redis
sudo yum install redis

# Start Redis service
sudo systemctl start redis

# Enable Redis to start on boot
sudo systemctl enable redis
```

#### Verify Redis Installation:
```bash
# Test Redis connection
redis-cli ping
# Should return: PONG
```

### 5. Optional: Configure Mem0 Memory

To enable conversation memory:

```bash
# Sign up for Mem0 at https://mem0.ai/
# Get your API key from the dashboard

# Add to your .env file
echo "MEM0_API_KEY=your_mem0_api_key_here" >> .env
```

## 🔄 Running the System

### Option A: Using Startup Scripts (Recommended)

The project includes comprehensive startup and stop scripts for easy system management:

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

**Startup Script Features:**
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

### Option B: Manual Startup

#### Terminal 1: Redis Server (Optional but Recommended)
```bash
# Start Redis server
redis-server

# Keep this terminal open - Redis runs in foreground
# You'll see logs like:
# [1234] 01 Jan 12:00:00.000 * Ready to accept connections
```

#### Terminal 2: API Server
```bash
# Start the Polkadot AI API server
python run_server.py

# Or using uvicorn directly
uvicorn src.rag.api_server:app --host 0.0.0.0 --port 8000

# Or using gunicorn with multiple workers
gunicorn --bind 0.0.0.0:8000 --workers 3 --worker-class uvicorn.workers.UvicornWorker src.rag.api_server:app

# Expected output:
# 🚀 Starting Polkadot AI Chatbot API...
# ✅ Configuration validated
# 📊 Loaded collection with 1,250 chunks
# 🔐 Rate limiter initialized: 20 requests per 3600 seconds
# 🧠 Memory system enabled with Mem0
# 🌐 Web search enabled
# INFO: Uvicorn running on http://0.0.0.0:8000
```

#### Terminal 3: Testing (Optional)
```bash
# Test the complete system
python src/test/run_tests.py

# Test specific components
python src/test/run_tests.py --api         # API functionality
python src/test/run_tests.py --memory      # Memory integration
python src/test/run_tests.py --formatting  # Response formatting
python src/test/run_tests.py --rate-limit  # Rate limiting
```

### Alternative: Background Services

For production or development convenience, you can run services in the background:

```bash
# Start Redis as background service
# macOS:
brew services start redis

# Linux:
sudo systemctl start redis-server

# Start API server in background
nohup python run_server.py > api.log 2>&1 &

# Check if services are running
ps aux | grep redis
ps aux | grep python
```

## 🧪 Testing Your Setup

### 1. Health Check
```bash
curl http://localhost:8000/health
```

### 2. Basic Query Test
```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is Polkadot?",
    "user_id": "test_user",
    "client_ip": "192.168.1.1"
  }'
```

### 3. Rate Limit Status Check
```bash
curl http://localhost:8000/rate-limit/test_user
```

### 4. Run Comprehensive Tests
```bash
python src/test/run_tests.py --all
```

## ⏱️ Performance Timing

The API provides detailed timing information for each service component to help identify performance bottlenecks:

### Service Timing Breakdown
- **📚 Document Retrieval**: Time spent searching ChromaDB for relevant chunks
- **🧠 Memory Operations**: Time spent retrieving and storing conversation memory
- **🤖 OpenAI Generation**: Time spent calling OpenAI API for answer generation
- **⏱️ Total Processing**: Complete request processing time

### Timing Analysis
```json
{
  "service_timing": {
    "retrieval_time_ms": 150.2,    // Document search time
    "memory_time_ms": 25.8,        // Memory operations time
    "openai_time_ms": 1074.5,      // OpenAI API call time
    "total_time_ms": 1250.5        // Total processing time
  }
}
```

### Performance Insights
- **Typical Retrieval Time**: 50-300ms (depends on query complexity)
- **Memory Operations**: 10-100ms (depends on conversation history)
- **OpenAI Generation**: 500-2000ms (depends on model and response length)
- **Total Processing**: Usually 600-2500ms for complete requests

### Monitoring Performance
```bash
# Monitor timing in logs
tail -f logs/api_server.log | grep "Service timing"

# Check performance metrics
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "test", "user_id": "test", "client_ip": "127.0.0.1"}' | jq '.service_timing'
```

## 🔧 Configuration Options

### Rate Limiting Settings
- `RATE_LIMIT_MAX_REQUESTS`: Requests per time window (default: 20)
- `RATE_LIMIT_EXPIRE_SECONDS`: Time window in seconds (default: 3600 = 1 hour)

### Memory Settings
- `MEM0_API_KEY`: Required for conversation memory
- Memory automatically stores user queries and responses per user_id

### Web Search Settings
- `WEB_SEARCH`: Enable/disable web search fallback (default: true)
- `WEB_SEARCH_CONTEXT_SIZE`: Context size for web search (low/medium/high)

### API Performance Settings
- `CHUNK_SIZE`: Size of text chunks for embeddings (default: 1000)
- `CHUNK_OVERLAP`: Overlap between chunks (default: 200)

### Gunicorn Settings (via environment variables)
- `WORKERS`: Number of worker processes (default: 3)
- `API_HOST`: API server host (default: 0.0.0.0)
- `API_PORT`: API server port (default: 8000)

## 🚨 Troubleshooting

### Common Issues and Solutions

#### 1. "OPENAI_API_KEY must be set"
```bash
# Check your .env file
cat .env | grep OPENAI_API_KEY

# Ensure the key is valid and has sufficient credits
curl -H "Authorization: Bearer $OPENAI_API_KEY" https://api.openai.com/v1/models
```

#### 2. "No data available. Please create embeddings first"
```bash
# Create the embeddings database
python create_embeddings.py

# Check if embeddings were created
ls -la chroma_db/
```

#### 3. "Rate limiting disabled due to Redis connection failure"
```bash
# Check if Redis is running
redis-cli ping

# Start Redis if not running
redis-server

# Check Redis logs
tail -f /var/log/redis/redis-server.log
```

#### 4. "Memory features disabled"
```bash
# Check if Mem0 package is installed
pip list | grep mem0

# Install if missing
pip install mem0ai

# Verify API key in .env
cat .env | grep MEM0_API_KEY
```

#### 5. High Memory Usage
```bash
# Reduce chunk size
echo "CHUNK_SIZE=500" >> .env

# Clear and recreate embeddings with smaller chunks
python create_embeddings.py --clear
```

#### 6. Slow Response Times
```bash
# Reduce max_chunks in queries
# Use fewer chunks for faster responses but potentially lower accuracy

# Check API server logs for bottlenecks
tail -f logs/api_server.log
```

#### 7. Startup Script Issues
```bash
# Check if scripts are executable
ls -la start_pa_ai.sh stop_pa_ai.sh

# Make executable if needed
chmod +x start_pa_ai.sh stop_pa_ai.sh

# Check script logs
tail -f logs/api_server.log
tail -f logs/redis.log
```

### Performance Optimization

#### For Faster Embedding Creation:
- Increase `--batch-size` (but watch OpenAI rate limits)
- Use smaller `CHUNK_SIZE` values
- Filter out unnecessary documents

#### For Better Response Quality:
- Use larger `CHUNK_SIZE` values
- Increase `max_chunks` in API queries
- Enable web search fallback

#### For Production Deployment:
- Use Redis clustering for high availability
- Set up proper logging and monitoring
- Configure appropriate rate limits
- Use environment-specific configurations
- Use Gunicorn with multiple workers for scalability

## 📊 Monitoring and Logs

### Log Files (with Startup Scripts)
- **Redis**: `logs/redis.log`
- **API Server**: `logs/api_server.log`
- **API Errors**: `logs/api_server_error.log`
- **Access Logs**: `logs/access.log`
- **Embedding creation**: `embedding_creation.log`

### Health Monitoring
```bash
# API health
curl http://localhost:8000/health

# Redis health
redis-cli ping

# System resource usage
top -p $(pgrep -f "gunicorn")
```

### Usage Statistics
```bash
# Collection statistics
curl http://localhost:8000/stats

# Rate limit status for specific user
curl http://localhost:8000/rate-limit/{user_id}
```

### Process Management
```bash
# Check system status
./stop_pa_ai.sh status

# View running processes
ps aux | grep -E "(redis|gunicorn|python.*api_server)"
```

## 🌐 API Documentation

Once running, access interactive documentation:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 🔐 Security Considerations

1. **API Keys**: Never commit `.env` file to version control
2. **Rate Limiting**: Adjust limits based on your use case
3. **Network Security**: Configure firewall rules for production
4. **Redis Security**: Set Redis password in production
5. **HTTPS**: Use reverse proxy (nginx) with SSL for production
6. **Process Isolation**: Use startup scripts for proper process management

## 📱 Frontend Integration

Example request format for frontend applications:

```javascript
const response = await fetch('http://localhost:8000/query', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    question: "How do I stake DOT tokens?",
    user_id: "user123",
    client_ip: "192.168.1.1",
    max_chunks: 3,
    include_sources: true
  })
});

const data = await response.json();
console.log('Answer:', data.answer);
console.log('Remaining requests:', data.remaining_requests);
console.log('Follow-up questions:', data.follow_up_questions);

// Performance timing analysis
const timing = data.service_timing;
console.log('Retrieval time:', timing.retrieval_time_ms + 'ms');
console.log('Memory time:', timing.memory_time_ms + 'ms');
console.log('OpenAI time:', timing.openai_time_ms + 'ms');
console.log('Total time:', timing.total_time_ms + 'ms');
```

## 🎯 Next Steps

After completing the setup:

1. **Test all features** using the test suite
2. **Customize prompts** and responses as needed
3. **Add more data sources** by placing files in `data/data_sources/`
4. **Monitor usage** and adjust rate limits
5. **Set up production deployment** with proper infrastructure
6. **Monitor performance** using the timing features
7. **Scale horizontally** by adjusting worker count and Redis configuration

For questions and support, refer to the main README.md or project documentation. 