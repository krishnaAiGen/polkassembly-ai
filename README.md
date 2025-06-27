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

## Project Structure

```
polkassembly-ai/
├── README.md                  # Project documentation
├── env.example                # Environment variables template
├── requirements.txt           # Python dependencies
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
pip install -r requirements.txt
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

```bash
# Start the FastAPI server
python run_server.py

# Or using uvicorn directly
uvicorn src.rag.api_server:app --host 0.0.0.0 --port 8000
```

### 5. Test the System

```bash
# Run the test script
python run_tests.py
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
  "max_chunks": 5,
  "include_sources": true,
  "custom_prompt": "optional custom system prompt"
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

## Usage Examples

### Python Client Example

```python
import requests

# Ask a question
response = requests.post("http://localhost:8000/query", json={
    "question": "How do I stake DOT tokens?",
    "max_chunks": 3,
    "include_sources": True
})

data = response.json()
print(f"Answer: {data['answer']}")
print(f"Confidence: {data['confidence']}")
print(f"Sources: {len(data['sources'])}")
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

## Configuration Options

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | - | Your OpenAI API key (required) |
| `OPENAI_MODEL` | gpt-3.5-turbo | OpenAI model for answers |
| `OPENAI_EMBEDDING_MODEL` | text-embedding-ada-002 | Embedding model |
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