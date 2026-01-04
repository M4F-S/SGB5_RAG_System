# SGB V RAG System - German Health Insurance Law

A production-ready Retrieval-Augmented Generation (RAG) system for the German Social Code Book V (Sozialgesetzbuch V) - Federal Health Insurance Law. Extract, process, embed, and retrieve information from German legal documents with semantic search and context-aware LLM responses.

## 📋 Table of Contents

- [Features](#features)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [API Reference](#api-reference)
- [Configuration](#configuration)
- [Development](#development)
- [Deployment](#deployment)
- [Legal & Compliance](#legal--compliance)

## ✨ Features

### 1. **Intelligent Web Scraper**
- Extracts all SGB V sections from gesetze-im-internet.de
- Handles complex nested paragraph structures (Absätze)
- Preserves section hierarchy and cross-references
- Intelligent rate limiting and robot.txt compliance
- Error handling with retry logic

### 2. **Advanced Data Processing**
- Legal text cleaning and normalization
- Clause-based chunking with context preservation
- Summary-Augmented Chunking (SAC) for document-level context
- Metadata enrichment (section_id, hierarchy_level, category)
- Duplicate detection and deduplication

### 3. **Multi-Modal Embeddings**
- German-optimized sentence-transformers (paraphrase-multilingual-mpnet-base-v2)
- Support for multiple embedding models
- Batch processing for efficiency
- Embedding caching

### 4. **Semantic Retrieval**
- FAISS vector database for fast similarity search
- BM25 lexical search for hybrid retrieval
- Reciprocal Rank Fusion (RRF) combining semantic + lexical
- Configurable top-k results
- Cross-reference aware retrieval

### 5. **LLM Integration**
- OpenAI GPT-4 / GPT-3.5 support
- Anthropic Claude integration ready
- Structured citation generation
- Temperature-controlled responses
- Context-aware prompt templates

### 6. **REST API & Web UI**
- FastAPI with async/await support
- OpenAPI documentation
- Streamlit web interface
- Example queries and result visualization
- Request/response logging

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- pip or conda
- API key (OpenAI or Anthropic)
- 2GB+ RAM for embeddings
- Internet connection for initial scraping

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/sgbv_rag_system.git
cd sgbv_rag_system

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API keys
nano .env
```

### Running the System

#### 1. Scrape and Process Data
```bash
# Scrape SGB V from source
python -m src.scraper

# Process and chunk data
python -m src.data_processor

# Generate embeddings
python -m src.embeddings
```

#### 2. Start the API Server
```bash
python -m src.api
# Server runs on http://localhost:8000
# OpenAPI docs at http://localhost:8000/docs
```

#### 3. Launch Web Interface
```bash
streamlit run ui/app.py
# Opens at http://localhost:8501
```

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│              USER INTERFACE LAYER                       │
│  ┌──────────────────────────────────────────────────┐  │
│  │  Streamlit Web UI  │  FastAPI REST Endpoints    │  │
│  └──────────────────────────────────────────────────┘  │
└────────┬─────────────────────────────────────┬──────────┘
         │                                     │
┌────────▼──────────────┐      ┌──────────────▼────────┐
│   RAG ENGINE LAYER    │      │   RETRIEVAL LAYER     │
│                       │      │                       │
│ • Query Rewriting     │      │ • FAISS Vector DB     │
│ • Prompt Engineering  │      │ • BM25 Lexical Search │
│ • Citation Generation │      │ • Hybrid RRF Fusion   │
│ • Response Streaming  │      │ • Context Ranking     │
└────────┬──────────────┘      └──────────────┬────────┘
         │                                     │
┌────────▼──────────────────────────────────────▼────────┐
│           DATA PROCESSING LAYER                        │
│                                                        │
│ • Scraper → Chunking → Embeddings → Vector Store     │
│ • Legal text normalization & hierarchy preservation   │
│ • Metadata enrichment & cross-reference tracking      │
└───────────────────────────────┬──────────────────────┘
                                │
┌───────────────────────────────▼──────────────────────┐
│         DATA SOURCES                                 │
│                                                      │
│ gesetze-im-internet.de/sgb_5/index.html             │
│ ↓                                                    │
│ Raw JSON & CSV Files → Processed Chunks → Vector DB │
└──────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
sgbv_rag_system/
├── src/
│   ├── __init__.py
│   ├── config.py              # Configuration management
│   ├── scraper.py             # Web scraper for SGB V
│   ├── data_processor.py       # Data cleaning & chunking
│   ├── embeddings.py           # Embedding generation
│   ├── rag_engine.py           # RAG orchestration
│   └── api.py                  # FastAPI server
│
├── ui/
│   ├── app.py                  # Streamlit application
│   └── styles.css              # UI styling
│
├── tests/
│   ├── __init__.py
│   ├── test_scraper.py         # Scraper unit tests
│   ├── test_processor.py        # Processor tests
│   └── test_retrieval.py        # RAG retrieval tests
│
├── data/
│   ├── raw/                    # Downloaded HTML
│   ├── processed/              # Cleaned JSON/CSV
│   └── embeddings/             # FAISS indices
│
├── docs/
│   ├── README.md               # This file
│   ├── ARCHITECTURE.md          # Detailed architecture
│   ├── API_DOCS.md              # API documentation
│   └── SETUP.md                 # Setup guide
│
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
│
├── requirements.txt             # Python dependencies
├── .env.example                 # Environment template
├── .gitignore
├── config.yaml                  # YAML configuration
└── setup.py                     # Package setup

```

## 🔌 API Reference

### REST Endpoints

#### Query SGB V
```http
POST /query
Content-Type: application/json

{
  "question": "What are the copayment requirements for prescriptions?",
  "top_k": 5,
  "temperature": 0.7
}

Response:
{
  "answer": "According to §31 SGB V...",
  "sources": [
    {
      "section_id": "31",
      "title": "Versorgung mit Arzneimitteln",
      "text": "...",
      "relevance_score": 0.92
    }
  ],
  "metadata": {
    "retrieval_time_ms": 145,
    "embedding_model": "paraphrase-multilingual-mpnet-base-v2"
  }
}
```

#### Search Sections
```http
POST /search
Content-Type: application/json

{
  "query": "Krankenversicherung Beitrag",
  "top_k": 10,
  "search_type": "hybrid"  # "semantic", "lexical", or "hybrid"
}

Response:
{
  "results": [
    {
      "section_id": "220",
      "title": "Grundsatz der Beitragsstabilität",
      "text": "...",
      "score": 0.88
    }
  ],
  "total_results": 45
}
```

#### Get Section Details
```http
GET /section/{section_id}

Response:
{
  "section_id": "31",
  "title": "Versorgung mit Arzneimitteln",
  "text": "...",
  "subsections": ["31a", "31b"],
  "cross_references": ["32", "33", "34"],
  "metadata": {
    "last_updated": "2024-01-04",
    "hierarchy_level": 1
  }
}
```

## ⚙️ Configuration

### Environment Variables (.env)

```bash
# API Keys
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=claude-...

# LLM Configuration
LLM_MODEL=gpt-4  # gpt-4, gpt-3.5-turbo, claude-3-sonnet
LLM_TEMPERATURE=0.3
LLM_MAX_TOKENS=2000

# Embedding Configuration
EMBEDDING_MODEL=paraphrase-multilingual-mpnet-base-v2
EMBEDDING_DEVICE=cpu  # cpu or cuda

# Database Configuration
FAISS_INDEX_PATH=./data/embeddings/sgbv.index
FAISS_METADATA_PATH=./data/embeddings/sgbv_metadata.json

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4

# Scraper Configuration
SCRAPER_DELAY_SECONDS=2
SCRAPER_TIMEOUT_SECONDS=30
SCRAPER_MAX_RETRIES=3

# Logging
LOG_LEVEL=INFO
LOG_FILE=./logs/sgbv_rag.log
```

### YAML Configuration (config.yaml)

```yaml
scraper:
  url: "https://www.gesetze-im-internet.de/sgb_5/index.html"
  delay_seconds: 2
  timeout_seconds: 30
  max_retries: 3
  respect_robots_txt: true

chunking:
  strategy: "clause-based"  # clause-based, semantic, hierarchical
  chunk_size: 1000  # characters
  overlap: 200
  preserve_cross_references: true

embedding:
  model: "paraphrase-multilingual-mpnet-base-v2"
  dimension: 768
  batch_size: 32
  cache_embeddings: true

retrieval:
  top_k: 5
  semantic_weight: 0.6  # For hybrid search
  lexical_weight: 0.4
  cross_reference_boost: 1.2

llm:
  model: "gpt-4"
  temperature: 0.3
  max_tokens: 2000
  timeout_seconds: 30

prompts:
  system: "You are a German legal expert specializing in SGB V (Sozialgesetzbuch V - health insurance law)..."
  user_template: "Based on the following German legal text: {context}\n\nAnswer the question: {question}"
```

## 🧪 Development

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_scraper.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

### Local Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt
pip install pytest pytest-cov black flake8

# Format code
black src/ ui/ tests/

# Lint code
flake8 src/ ui/ tests/

# Run linter and tests
pre-commit run --all-files
```

## 🐳 Deployment

### Docker

```bash
# Build image
docker build -f docker/Dockerfile -t sgbv-rag:latest .

# Run container
docker run -p 8000:8000 -p 8501:8501 \
  --env-file .env \
  --volume $(pwd)/data:/app/data \
  sgbv-rag:latest

# Or use Docker Compose
docker-compose -f docker/docker-compose.yml up -d
```

### Production Checklist

- [ ] Set `LLM_TEMPERATURE=0.3` for legal accuracy
- [ ] Use production API keys with rate limits
- [ ] Enable HTTPS/SSL for API
- [ ] Set up monitoring (logs, metrics, alerts)
- [ ] Configure CORS properly
- [ ] Enable request authentication/authorization
- [ ] Set up backup of FAISS indices daily
- [ ] Configure rate limiting per user
- [ ] Enable audit logging for all queries
- [ ] Set up error tracking (Sentry, etc.)

## ⚖️ Legal & Compliance

### GDPR Compliance

- ✅ No user data stored in embeddings
- ✅ API logs can be configured for retention
- ✅ Input/output filtering for PII detection
- ✅ Data minimization: only section IDs stored
- ✅ User can request data deletion (logs)

### Data Privacy

- All data from gesetze-im-internet.de (public domain)
- Legal text copyright: Bundesrepublik Deutschland
- System respects robots.txt and crawl delays
- No personal data processing

### Citation & Attribution

- All responses include source citations
- Full section text available for reference
- Cross-references tracked and returned
- Metadata preserves document lineage

## 📚 References & Resources

### Research Papers
- "Towards Reliable Retrieval in RAG Systems for Large Legal Datasets" (arXiv:2510.06999)
- "Chunk Twice, Retrieve Once: RAG Chunking Strategies" (Dell Technologies)
- "RAG Foundry: A Framework for Enhancing LLMs" (Intel Labs)

### Repositories
- [Sozialrecht_RAG](https://github.com/ma3u/Sozialrecht_RAG) - German legal documents
- [RAG-Anything](https://github.com/HKUDS/RAG-Anything) - Multimodal RAG
- [RAG Techniques](https://github.com/NirDiamant/RAG_Techniques)

### German Legal Sources
- [gesetze-im-internet.de](https://www.gesetze-im-internet.de/) - Official legal portal
- [SGB V (Health Insurance Law)](https://www.gesetze-im-internet.de/sgb_5/index.html)
- [Hamburg Data Protection Authority on AI](https://datenschutz-hamburg.de/)

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Add tests for new functionality
4. Submit a pull request

## 📄 License

MIT License - See LICENSE file for details

## ✉️ Support

For issues, questions, or suggestions:
- Open an issue on GitHub
- Email: Mohamedfathy7@hotmail.com
- Check discussions for Q&A

## 🔄 Changelog

### v1.0.0 (2026-01-04)
- Initial release
- Core RAG functionality
- Web scraper for SGB V
- FAISS vector database
- FastAPI & Streamlit interfaces
- Full test coverage
