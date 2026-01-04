# SGB V RAG System Setup Guide

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Data Pipeline](#data-pipeline)
4. [API Server](#api-server)
5. [Web Interface](#web-interface)
6. [Troubleshooting](#troubleshooting)

## Prerequisites

- **Python**: 3.10 or higher
- **Memory**: 2GB minimum (4GB recommended)
- **Disk Space**: 5GB for data and models
- **Internet**: For downloading models and scraping
- **API Key**: OpenAI API key (or Anthropic/local model)

Check Python version:
```bash
python --version
```

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/sgbv_rag_system.git
cd sgbv_rag_system
```

### 2. Create Virtual Environment

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

This will install:
- Web scraping: BeautifulSoup, requests
- Data processing: pandas, numpy
- Embeddings: sentence-transformers, torch
- Vector DB: FAISS
- API: FastAPI, uvicorn
- UI: Streamlit
- LLM: OpenAI SDK

### 4. Configure Environment

```bash
cp .env.example .env
nano .env  # Edit with your API keys
```

**Required settings:**
```env
OPENAI_API_KEY=sk-your-api-key-here
LLM_MODEL=gpt-4
LLM_TEMPERATURE=0.3
```

**Optional:**
```env
EMBEDDING_DEVICE=cpu  # Use 'cuda' if you have GPU
SCRAPER_DELAY_SECONDS=2  # Respectful crawl rate
```

### 5. Create Data Directories

```bash
mkdir -p data/raw data/processed data/embeddings logs
```

## Data Pipeline

### Step 1: Scrape SGB V

Download all SGB V sections from gesetze-im-internet.de:

```bash
python -m src.scraper
```

**Output:**
- `data/raw/sgbv_sections.json` - Raw sections
- `data/raw/sgbv_sections.csv` - Section index

**What it does:**
- Fetches all sections from the official German legal portal
- Respects robots.txt and crawl delays
- Extracts hierarchy and cross-references
- Handles timeouts and retries automatically
- Takes ~2-5 minutes (respects 2-second delay)

### Step 2: Process & Chunk Data

Clean, normalize, and split documents:

```bash
python -m src.data_processor
```

**Output:**
- `data/processed/sgbv_chunks.json` - Processed chunks with metadata

**What it does:**
- Normalizes legal text (removes artifacts, fixes encoding)
- Implements clause-based chunking strategy
- Preserves document hierarchy and cross-references
- Generates document summaries for context injection
- Handles complex paragraph structures

**Configuration:**
```env
CHUNK_SIZE=1000        # Characters per chunk
CHUNK_OVERLAP=200      # Overlap for context
```

### Step 3: Generate Embeddings

Create vector embeddings for semantic search:

```bash
python -m src.embeddings
```

**Output:**
- `data/embeddings/sgbv_embeddings.npy` - Binary embedding file
- `data/embeddings/sgbv_chunks_embedded.json` - Chunks with embeddings

**What it does:**
- Loads German-optimized sentence-transformer model
- Generates 768-dimensional embeddings
- Combines document summary + chunk text for better context
- Batch processing for efficiency
- Takes ~5-10 minutes depending on CPU

**Configuration:**
```env
EMBEDDING_MODEL=paraphrase-multilingual-mpnet-base-v2
EMBEDDING_DEVICE=cpu   # 'cuda' for GPU acceleration
EMBEDDING_BATCH_SIZE=32
```

**⚡ GPU Acceleration:**

If you have CUDA GPU:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# Then update .env: EMBEDDING_DEVICE=cuda
python -m src.embeddings  # 10x faster
```

## API Server

### Start API Server

```bash
python -m src.api
```

**Output:**
```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     OpenAPI docs at http://0.0.0.0:8000/docs
```

### API Endpoints

**Health Check:**
```bash
curl http://localhost:8000/health
```

**Query Example:**
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are copayment requirements for prescriptions?",
    "top_k": 5,
    "temperature": 0.3
  }'
```

**Search Example:**
```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Beitragssätze Krankenkasse",
    "top_k": 10,
    "search_type": "hybrid"
  }'
```

**Interactive Docs:**
Open http://localhost:8000/docs in your browser for Swagger UI

## Web Interface

### Start Streamlit UI

In a new terminal:

```bash
streamlit run ui/app.py
```

**Output:**
```
  You can now view your Streamlit app in your browser.
  Network URL: http://xxx.xxx.x.xxx:8501
  Local URL: http://localhost:8501
```

### Features

- 🔍 **Query Tab**: Ask legal questions, get answers with citations
- 📖 **Browse Tab**: Search and explore sections
- ℹ️ **About Tab**: System documentation

### Example Queries

1. "What are the copayment requirements for prescriptions?"
2. "Which sections cover dental treatment benefits?"
3. "What is the maximum deductible for insured persons?"
4. "How are chronic diseases defined in SGB V?"
5. "What are the rules for reimbursement of therapeutic aids?"

## Docker Deployment

### Using Docker Compose

```bash
# Build and start all services
docker-compose up -d

# View logs
docker-compose logs -f api

# Stop services
docker-compose down
```

**Services:**
- API: http://localhost:8000
- Streamlit: http://localhost:8501
- Redis: localhost:6379

### Using Docker Directly

```bash
# Build image
docker build -f docker/Dockerfile -t sgbv-rag:latest .

# Run API
docker run -p 8000:8000 \
  --env-file .env \
  --volume $(pwd)/data:/app/data \
  sgbv-rag:latest

# Run Streamlit
docker run -p 8501:8501 \
  --volume $(pwd)/data:/app/data \
  sgbv-rag:latest \
  streamlit run ui/app.py
```

## Development

### Run Tests

```bash
# Install test dependencies
pip install pytest pytest-cov pytest-asyncio

# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_scraper.py -v

# With coverage
pytest tests/ --cov=src --cov-report=html
```

### Code Quality

```bash
# Format code
black src/ ui/ tests/

# Lint code
flake8 src/ ui/ tests/

# Type checking
mypy src/
```

### Pre-commit Setup

```bash
pip install pre-commit

# Initialize
pre-commit install

# Run manually
pre-commit run --all-files
```

## Troubleshooting

### Issue: "OPENAI_API_KEY not found"

**Solution:**
```bash
# Copy template
cp .env.example .env

# Edit .env with your key
nano .env

# Make sure it's in the current shell
export OPENAI_API_KEY=sk-...
```

### Issue: "ModuleNotFoundError: No module named 'src'"

**Solution:**
```bash
# Make sure you're in the project root
cd sgbv_rag_system

# Install in editable mode
pip install -e .

# Or use absolute imports
python -c "import sys; sys.path.insert(0, '.'); from src.api import app"
```

### Issue: "FAISS index not found"

**Solution:**
```bash
# Run the full pipeline in order:
python -m src.scraper
python -m src.data_processor
python -m src.embeddings

# Verify files exist:
ls -lh data/embeddings/
```

### Issue: Out of Memory

**Solution:**
```bash
# Reduce batch size
export EMBEDDING_BATCH_SIZE=8

# Use CPU device
export EMBEDDING_DEVICE=cpu

# Or use GPU if available
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
export EMBEDDING_DEVICE=cuda
```

### Issue: "Connection refused" when API starts

**Solution:**
```bash
# Check if port 8000 is already in use
lsof -i :8000

# Kill existing process
kill -9 <PID>

# Or use different port
export API_PORT=8001
python -m src.api
```

### Issue: Slow embedding generation

**Solution:**
```bash
# Check device
python -c "import torch; print(torch.cuda.is_available())"

# Use GPU if available:
export EMBEDDING_DEVICE=cuda

# Or reduce batch size and use multi-processing
export EMBEDDING_BATCH_SIZE=16

# Monitor progress
tail -f logs/sgbv_rag.log
```

## Production Setup

### Recommended Configuration

```env
LLM_MODEL=gpt-4
LLM_TEMPERATURE=0.3
EMBEDDING_DEVICE=cuda
API_WORKERS=4
LOG_LEVEL=WARNING
SCRAPER_DELAY_SECONDS=3
```

### Security Checklist

- [ ] Use environment variables for all secrets
- [ ] Enable HTTPS/SSL for API
- [ ] Set up rate limiting
- [ ] Configure CORS properly
- [ ] Enable authentication/authorization
- [ ] Set up logging and monitoring
- [ ] Regular backups of FAISS indices
- [ ] Use production-grade WSGI server (Gunicorn)

### Monitoring

```bash
# View logs
tail -f logs/sgbv_rag.log

# Monitor API health
watch curl http://localhost:8000/health
```

## Next Steps

1. ✅ Install dependencies
2. ✅ Configure API key
3. ✅ Run scraper
4. ✅ Process data
5. ✅ Generate embeddings
6. ✅ Start API server
7. ✅ Launch Streamlit UI
8. 📚 Test with example queries
9. 🚀 Deploy to production

## Support & Resources

- **GitHub Issues**: https://github.com/yourusername/sgbv_rag_system/issues
- **Documentation**: See README.md
- **API Docs**: http://localhost:8000/docs
- **Official SGB V**: https://www.gesetze-im-internet.de/sgb_5/

