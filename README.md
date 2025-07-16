# Polkadot AI Chatbot

An AI-powered chatbot system that provides intelligent answers about the Polkadot ecosystem using RAG (Retrieval-Augmented Generation) with OpenAI embeddings and ChromaDB. The system features a **dual embedding architecture** with support for unlimited external data sources.

## 🚀 Features

- 📚 **Multi-Source Knowledge Base**: Static (.txt) and dynamic (.json) data processing
- 🔍 **Semantic Search**: Uses OpenAI embeddings for accurate document retrieval across multiple collections
- 🤖 **AI-Powered Answers**: Generates contextual responses using GPT models
- 🚀 **Fast API**: RESTful API with automatic documentation
- 💾 **Persistent Storage**: ChromaDB for efficient vector storage
- 🔧 **Configurable**: Environment-based configuration
- 📊 **Analytics**: Built-in statistics and monitoring
- 🎯 **Extensible**: Add unlimited external data sources without rebuilding existing embeddings
- 🔄 **Incremental Updates**: Add new collections without affecting existing ones

## 📁 Project Structure

```
polkassembly-ai/
├── README.md                      # Project documentation
├── env.example                    # Environment variables template
├── requirements.txt               # Python dependencies
├── run_server.py                  # Entry point for API server (auto-initializes embeddings)
├── create_external_embeddings.py # Create embeddings from external data sources
├── example_usage.py               # Example usage of the embedding system
├── data/                          # Data sources
│   └── data_sources/
│       ├── static_data/           # Static data (.txt files) - recursively processed
│       └── onchain_data/          # Dynamic data (.json files) - recursively processed
├── chroma_db/                     # ChromaDB storage
│   └── chroma.sqlite3             # Vector database with multiple collections
└── src/                           # Source code directory
    ├── rag/                       # RAG application modules
    │   ├── config.py              # Configuration management
    │   └── api_server.py          # FastAPI server
    ├── utils/                     # Utility modules
    │   ├── data_loader.py         # Document loading and processing
    │   ├── text_chunker.py        # Text chunking for embeddings
    │   ├── embeddings.py          # Multi-collection embedding manager
    │   ├── qa_generator.py        # Q&A generation with OpenAI
    │   └── onchain_data.py        # Onchain data processing utilities
    └── test/                      # Testing modules
        └── test_api.py            # API testing script
```

## 🏗️ Embedding Architecture

The system uses a **dual embedding architecture** with support for unlimited external collections:

```
ChromaDB Collections:
├── static_embeddings      # From STATIC_DATA_SOURCE (.txt files)
├── dynamic_embeddings     # From DYNAMIC_DATA_SOURCE (.json files)
├── external_docs          # Your external collection 1
├── api_documentation      # Your external collection 2
├── community_content      # Your external collection 3
└── ...                    # Unlimited external collections
```

### Search Behavior
- **Static embeddings**: Always searched (documentation, guides)
- **Dynamic embeddings**: Searched if `SEARCH_ONCHAIN_DATA=true` (governance, proposals)
- **External embeddings**: All external collections are searched automatically
- **Results**: Ranked by similarity score across ALL collections

## 🛠️ Setup Instructions

### 1. Clone and Install

```bash
# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Configuration

```bash
# Copy the example environment file
cp env.example .env

# Edit .env with your configuration
```

**Required Environment Variables:**
```env
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-3.5-turbo
OPENAI_EMBEDDING_MODEL=text-embedding-ada-002

# Data Sources Configuration
STATIC_DATA_SOURCE=data/data_sources/static_data
DYNAMIC_DATA_SOURCE=data/data_sources/onchain_data

# Search Configuration
SEARCH_ONCHAIN_DATA=true  # Set to false to search only static data

# ChromaDB Configuration
CHROMA_PERSIST_DIRECTORY=./chroma_db
```

### 3. Prepare Your Data

#### Core Data Sources (Auto-processed on first run)
```bash
# Place your static data (.txt files) in:
data/data_sources/static_data/
├── wiki/
├── docs/
└── guides/

# Place your dynamic data (.json files) in:
data/data_sources/onchain_data/
├── proposals/
├── governance/
└── discussions/
```

### 4. Start the System

```bash
# Start the API server (automatically initializes embeddings)
python run_server.py
```

**First Run Output:**
```
=== POLKADOT AI CHATBOT SERVER STARTUP ===
=== INITIALIZING DUAL EMBEDDING SYSTEM ===
✓ Configuration validated

Data Source Status:
  Static data: 202 .txt files in data/data_sources/static_data
  Dynamic data: 71 .json files in data/data_sources/onchain_data

=== PROCESSING STATIC DATA ===
Chunking 202 static documents...
Created 1,250 static chunks
✓ Added 1,250 static documents to collection

=== PROCESSING DYNAMIC DATA ===
Chunking 71 dynamic documents...
Created 890 dynamic chunks
✓ Added 890 dynamic documents to collection

=== FINAL EMBEDDING STATUS ===
  Static collection: 1,250 documents
  Dynamic collection: 890 documents
  Total documents: 2,140

=== STARTING API SERVER ===
```

## 🎯 Adding External Data Sources

### Method 1: Interactive Mode
```bash
python create_external_embeddings.py
# Follow the prompts to specify folder path and collection name
```

### Method 2: Command Line Arguments
```bash
python create_external_embeddings.py /path/to/external/data collection_name
```

### Examples

```bash
# Add API documentation
python create_external_embeddings.py ./data/api_docs api_documentation

# Add community content
python create_external_embeddings.py ./data/community community_content

# Add research papers
python create_external_embeddings.py ./data/research research_papers

# Add FAQ data
python create_external_embeddings.py ./data/faq faq_collection
```

### External Data Structure
Your external data can contain both .txt and .json files in any folder structure:
```
/path/to/external/data/
├── subfolder1/
│   ├── document1.txt
│   └── data1.json
├── subfolder2/
│   ├── document2.txt
│   └── data2.json
└── mixed_files/
    ├── readme.txt
    └── config.json
```

## 📊 Monitoring Your Collections

### View All Collections
```bash
python example_usage.py
```

**Output:**
```
=== CURRENT COLLECTIONS ===
Static collection: 1,250 documents
Dynamic collection: 890 documents

External collections:
  - api_documentation: 45 documents
  - community_content: 123 documents
  - research_papers: 67 documents

Total documents: 2,375

=== EXAMPLE SEARCH ===
Query: 'What is Polkadot?'
Found 3 results:
1. Collection: static_embeddings
   Score: 0.8945
   Content: Polkadot is a multi-chain blockchain platform...
```

## 🔌 API Endpoints

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
  "source_filter": "static"  # Options: "static", "dynamic", or null for all
}
```

### 📊 Get Statistics
```http
GET /stats
```

**Response includes all collections:**
```json
{
  "collection_stats": {
    "static_collection": {"count": 1250, "name": "static_embeddings"},
    "dynamic_collection": {"count": 890, "name": "dynamic_embeddings"},
    "external_collections": [
      {"name": "api_documentation", "count": 45},
      {"name": "community_content", "count": 123}
    ],
    "total_documents": 2308
  }
}
```

## 🎛️ Configuration Options

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | - | Your OpenAI API key (required) |
| `OPENAI_MODEL` | gpt-3.5-turbo | OpenAI model for answers |
| `OPENAI_EMBEDDING_MODEL` | text-embedding-ada-002 | Embedding model |
| `STATIC_DATA_SOURCE` | data/data_sources/static_data | Path to static data (.txt files) |
| `DYNAMIC_DATA_SOURCE` | data/data_sources/onchain_data | Path to dynamic data (.json files) |
| `SEARCH_ONCHAIN_DATA` | true | Whether to search dynamic data |
| `CHROMA_PERSIST_DIRECTORY` | ./chroma_db | ChromaDB storage path |
| `API_HOST` | 0.0.0.0 | API server host |
| `API_PORT` | 8000 | API server port |
| `CHUNK_SIZE` | 1000 | Text chunk size for embeddings |
| `CHUNK_OVERLAP` | 200 | Overlap between chunks |
| `TOP_K_RESULTS` | 5 | Default number of results to return |
| `SIMILARITY_THRESHOLD` | 0.7 | Minimum similarity threshold |

## 📝 Data Format Requirements

### Static Data (.txt files)
```
Title: Your Document Title
URL: https://example.com/document
Description: Document description
Type: guide
========================================
Your document content goes here...
```

### Dynamic Data (.json files)
```json
{
  "title": "Document Title",
  "description": "Document description",
  "content": "Main content text...",
  "url": "https://example.com",
  "topics": [
    {"title": "Topic 1"},
    {"title": "Topic 2"}
  ]
}
```

## 🔧 Usage Examples

### Python Client Example

```python
import requests

# Ask a question (searches all collections)
response = requests.post("http://localhost:8000/query", json={
    "question": "How do I stake DOT tokens?",
    "max_chunks": 3,
    "include_sources": True
})

data = response.json()
print(f"Answer: {data['answer']}")
print(f"Sources from {len(set(s['source_type'] for s in data['sources']))} collections")
```

### cURL Examples

```bash
# Search only static data
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "governance voting",
    "n_results": 5,
    "source_filter": "static"
  }'

# Search only dynamic data
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "recent proposals",
    "n_results": 5,
    "source_filter": "dynamic"
  }'
```

## 🎯 Advanced Usage

### Programmatic Collection Management

```python
from src.utils.embeddings import DualEmbeddingManager
from src.rag.config import Config

# Initialize manager
manager = DualEmbeddingManager(
    openai_api_key=Config.OPENAI_API_KEY,
    embedding_model=Config.OPENAI_EMBEDDING_MODEL,
    chroma_persist_directory=Config.CHROMA_PERSIST_DIRECTORY
)

# Get all collections
stats = manager.get_collection_stats()
print(f"Total collections: {len(stats['external_collections']) + 2}")

# Search with custom parameters
results = manager.search_similar_chunks(
    query="What is Polkadot?",
    n_results=10,
    search_onchain=True,
    search_external=True
)

# Clear specific external collection
manager.clear_external_collection('old_collection')
```

### Batch Processing External Data

```python
import os
from create_external_embeddings import ExternalDataProcessor

# Process multiple folders
external_folders = [
    ('/path/to/docs', 'documentation'),
    ('/path/to/blogs', 'blog_posts'),
    ('/path/to/research', 'research_papers')
]

for folder_path, collection_name in external_folders:
    if os.path.exists(folder_path):
        processor = ExternalDataProcessor(folder_path, collection_name)
        data = processor.load_external_data()
        processor.create_external_collection(data)
```

## 🚦 API Documentation

Once the server is running, visit:
- **Interactive API Docs**: http://localhost:8000/docs
- **ReDoc Documentation**: http://localhost:8000/redoc

## 📊 Monitoring and Logs

- **Server startup logs**: Console output with detailed initialization info
- **Search logs**: Real-time search performance and collection usage
- **ChromaDB storage**: `./chroma_db/` directory
- **Collection stats**: Available via `/stats` endpoint

## 🔍 Troubleshooting

### Common Issues

1. **"Configuration error: Static data source path does not exist"**
   ```bash
   # Create the required directories
   mkdir -p data/data_sources/static_data
   mkdir -p data/data_sources/onchain_data
   ```

2. **"AttributeError: 'TextChunker' object has no attribute 'chunk_text'"**
   - This is fixed in the current version - use `chunk_document()` method

3. **"No data available. Please run server initialization first."**
   - Run `python run_server.py` to initialize embeddings automatically

4. **Rate limiting from OpenAI**
   - The system includes automatic retry logic with exponential backoff
   - Monitor the logs for rate limit warnings

5. **Collection already exists**
   - External embedding creator will ask if you want to add or recreate
   - Choose 'a' to add to existing or 'r' to recreate

### Performance Tips

- **Faster searches**: Results are cached and ranked across all collections
- **Memory optimization**: Reduce `CHUNK_SIZE` for large datasets
- **Better accuracy**: Increase `max_chunks` in queries
- **Incremental updates**: Add external collections without rebuilding core collections

## 🔄 Migration from Old System

If you're upgrading from the old system:

1. **Backup existing data**: Your ChromaDB will be automatically migrated
2. **Update environment**: Add new variables to `.env`
3. **Run server**: `python run_server.py` will handle migration automatically
4. **Verify collections**: Use `python example_usage.py` to check all collections

## 🎨 Customization

### Adding New File Types

Modify `create_external_embeddings.py` to support new file types:

```python
# Add to find_all_files method
def find_all_files(self, directory: str, extensions: List[str]) -> List[str]:
    # Add new extensions like .md, .pdf, etc.
    extensions.extend(['.md', '.pdf', '.docx'])
```

### Custom Metadata Extraction

Modify the `load_text_file` or `load_json_file` methods in `create_external_embeddings.py` to extract custom metadata fields.

### Search Customization

Modify the `search_similar_chunks` method in `src/utils/embeddings.py` to:
- Add custom filtering logic
- Implement collection-specific weights
- Add custom ranking algorithms

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test with the API endpoints
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🆘 Support

For issues and questions:
1. Check the troubleshooting section
2. Review the API documentation at `/docs`
3. Check server logs for detailed error information
4. Use `python example_usage.py` to test your collections

## 🎯 Next Steps

After setup, you can:
1. **Add your own data**: Use `create_external_embeddings.py` to add domain-specific content
2. **Customize search**: Modify search parameters in the API calls
3. **Monitor usage**: Use the `/stats` endpoint to track collection usage
4. **Scale up**: Add more external collections as your knowledge base grows

The system is designed to scale with your needs - add unlimited external data sources without rebuilding existing embeddings! 🚀