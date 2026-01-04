# SGB V RAG System - Quick Reference Guide

## 🚀 30-Second Start

```bash
# 1. Setup (one-time)
git clone <repo> && cd sgbv_rag_system
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt && cp .env.example .env

# 2. Configure
nano .env  # Add OPENAI_API_KEY=sk-...

# 3. Run pipeline (takes ~10-15 minutes)
python -m src.scraper && \
python -m src.data_processor && \
python -m src.embeddings

# 4. Start system (two terminals)
# Terminal 1:
python -m src.api

# Terminal 2:
streamlit run ui/app.py
```

**Access:**
- API: http://localhost:8000/docs
- UI: http://localhost:8501

---

## 📋 Module Reference

### Scraper (`src/scraper.py`)
```python
from src.scraper import SGBVScraper

scraper = SGBVScraper(
    delay_seconds=2,           # Rate limiting
    timeout_seconds=30,        # Request timeout
    max_retries=3,             # Retry attempts
    respect_robots_txt=True    # Follow robots.txt
)

sections = scraper.scrape_all()
scraper.save_json(sections, 'data/raw/sgbv_sections.json')
scraper.save_csv(sections, 'data/raw/sgbv_sections.csv')
```

### Data Processor (`src/data_processor.py`)
```python
from src.data_processor import LegalDocumentProcessor

processor = LegalDocumentProcessor(
    chunk_size=1000,           # Characters per chunk
    chunk_overlap=200,         # Context overlap
    preserve_context=True      # Keep hierarchy
)

sections = processor.load_sections('data/raw/sgbv_sections.json')
chunks = processor.process_sections(sections)
processor.save_chunks(chunks, 'data/processed/sgbv_chunks.json')
```

### Embeddings (`src/embeddings.py`)
```python
from src.embeddings import EmbeddingGenerator

generator = EmbeddingGenerator(
    model_name="paraphrase-multilingual-mpnet-base-v2",
    batch_size=32,
    device="cpu"  # or "cuda"
)

texts = [chunk['text'] for chunk in chunks]
embeddings = generator.embed_texts(texts)
generator.save_embeddings(embeddings, 'data/embeddings/sgbv_embeddings.npy')
```

### RAG Engine (`src/rag_engine.py`)
```python
from src.rag_engine import RAGEngine
import faiss
import numpy as np

# Load data
embeddings = np.load('data/embeddings/sgbv_embeddings.npy')
vector_db = faiss.IndexFlatL2(768)
vector_db.add(embeddings.astype('float32'))

# Initialize
rag = RAGEngine(
    vector_db=vector_db,
    chunks=chunks,
    llm_model="gpt-4",
    temperature=0.3,
    top_k=5
)

# Query
query_embedding = generator.embed_query("What are copayment rules?")
response = rag.query("What are copayment rules?", query_embedding)
```

---

## 🔌 API Endpoints

### Query Knowledge Base
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are copayment requirements?",
    "top_k": 5,
    "temperature": 0.3
  }'
```

### Search Sections
```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Beitragssätze",
    "top_k": 10,
    "search_type": "hybrid"
  }'
```

### Get Section
```bash
curl http://localhost:8000/section/31
```

### Health Check
```bash
curl http://localhost:8000/health
```

---

## ⚙️ Configuration

### Environment Variables (.env)
```env
# Required
OPENAI_API_KEY=sk-...

# Recommended
LLM_MODEL=gpt-4
LLM_TEMPERATURE=0.3
EMBEDDING_MODEL=paraphrase-multilingual-mpnet-base-v2
EMBEDDING_DEVICE=cpu

# Optional
SCRAPER_DELAY_SECONDS=2
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
API_PORT=8000
LOG_LEVEL=INFO
```

### YAML Config (config.yaml)
```yaml
scraper:
  delay_seconds: 2
  timeout_seconds: 30
  max_retries: 3

chunking:
  chunk_size: 1000
  overlap: 200
  strategy: "clause-based"

embedding:
  model: "paraphrase-multilingual-mpnet-base-v2"
  batch_size: 32

retrieval:
  top_k: 5
  semantic_weight: 0.6
  lexical_weight: 0.4

llm:
  model: "gpt-4"
  temperature: 0.3
  max_tokens: 2000
```

---

## 📊 Data Flow

### Ingestion
```
Source (gesetze-im-internet.de)
    ↓ scraper.py
raw/ (JSON, CSV)
    ↓ data_processor.py
processed/ (chunks with metadata)
    ↓ embeddings.py
embeddings/ (FAISS + vectors)
    ↓ api.py
RAG Engine (ready to query)
```

### Query
```
User Question
    ↓ validate
Query Embedding
    ↓ parallel
├─ FAISS search (semantic)
└─ BM25 search (lexical)
    ↓ RRF fusion
Top-5 chunks
    ↓ format
Prompt with context
    ↓ GPT-4
Response with citations
```

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Test specific module
pytest tests/test_scraper.py -v
pytest tests/test_processor.py -v
pytest tests/test_retrieval.py -v

# With coverage
pytest tests/ --cov=src --cov-report=html
```

---

## 🐳 Docker

```bash
# Build
docker build -f docker/Dockerfile -t sgbv-rag:latest .

# Run API
docker run -p 8000:8000 \
  --env-file .env \
  --volume $(pwd)/data:/app/data \
  sgbv-rag:latest

# Run with docker-compose
docker-compose up -d
docker-compose logs -f api
docker-compose down
```

---

## 🔍 Debugging

### Check Logs
```bash
tail -f logs/sgbv_rag.log
```

### Test Scraper
```python
python -c "
from src.scraper import SGBVScraper
scraper = SGBVScraper()
print(scraper.INDEX_URL)
"
```

### Test Embeddings
```python
python -c "
from src.embeddings import EmbeddingGenerator
gen = EmbeddingGenerator()
emb = gen.embed_query('test')
print(emb.shape)
"
```

### Test API Connection
```bash
curl http://localhost:8000/health
curl http://localhost:8000/docs
```

---

## 💾 Data Management

### Directory Sizes
```
data/raw/              ~15MB    (HTML/JSON from source)
data/processed/        ~30MB    (Cleaned chunks)
data/embeddings/       ~1.5GB   (FAISS vectors)
logs/                  varies    (Application logs)
```

### Backup & Restore
```bash
# Backup FAISS indices
cp -r data/embeddings/ backup_embeddings_$(date +%Y%m%d)/

# Restore
cp -r backup_embeddings_*/. data/embeddings/
```

### Cleanup
```bash
# Clear processed data (regenerate from raw)
rm -rf data/processed/* data/embeddings/*

# Keep raw data, reprocess
python -m src.data_processor
python -m src.embeddings
```

---

## 🚨 Common Issues

### Issue: "No module named 'src'"
```bash
# Solution
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
python -m src.api
```

### Issue: "FAISS index not found"
```bash
# Solution: Run full pipeline
python -m src.scraper
python -m src.data_processor
python -m src.embeddings
```

### Issue: "Connection refused"
```bash
# Solution: Check ports
lsof -i :8000
lsof -i :8501

# Kill process
kill -9 <PID>
```

### Issue: Out of memory
```bash
# Solution: Reduce batch size
export EMBEDDING_BATCH_SIZE=8
python -m src.embeddings
```

### Issue: Slow embeddings
```bash
# Solution: Use GPU
export EMBEDDING_DEVICE=cuda
python -m src.embeddings
```

---

## 📈 Performance Tips

### Speed Up Scraping
```env
SCRAPER_DELAY_SECONDS=1          # Faster crawl (if allowed)
SCRAPER_TIMEOUT_SECONDS=60       # Longer timeout
```

### Speed Up Embeddings
```env
EMBEDDING_DEVICE=cuda            # Use GPU (10x faster)
EMBEDDING_BATCH_SIZE=64          # Larger batches
```

### Faster Retrieval
```env
RETRIEVAL_TOP_K=3                # Fewer results
SEMANTIC_WEIGHT=0.8              # Faster lexical search
```

### Better Results
```env
RETRIEVAL_TOP_K=10               # More results to choose from
SEMANTIC_WEIGHT=0.6              # Balance both methods
LLM_MAX_TOKENS=3000              # More detailed responses
```

---

## 🔐 Security Checklist

- [ ] API key in .env file (not in code)
- [ ] .env file in .gitignore
- [ ] HTTPS enabled in production
- [ ] Rate limiting configured
- [ ] CORS properly set
- [ ] Input validation on endpoints
- [ ] Error messages don't leak info
- [ ] Logs don't contain secrets
- [ ] Database backups scheduled
- [ ] Access logs enabled

---

## 📚 Example Queries

```python
# Python client example
import requests

api_url = "http://localhost:8000"

# Query
response = requests.post(
    f"{api_url}/query",
    json={
        "question": "What are copayment requirements for prescriptions?",
        "top_k": 5,
        "temperature": 0.3
    }
)
print(response.json()['answer'])

# Search
response = requests.post(
    f"{api_url}/search",
    json={
        "query": "Beitragssätze",
        "top_k": 10,
        "search_type": "hybrid"
    }
)
for result in response.json()['results']:
    print(f"§{result['section_id']}: {result['title']}")
```

---

## 🎯 Feature Checklist

- [x] Web scraper with robots.txt respect
- [x] Legal document preprocessing
- [x] Clause-based chunking with SAC
- [x] German-optimized embeddings
- [x] FAISS vector database
- [x] BM25 lexical search
- [x] Hybrid retrieval (RRF)
- [x] GPT-4 integration
- [x] Citation generation
- [x] FastAPI server
- [x] Streamlit UI
- [x] Docker support
- [x] Comprehensive documentation
- [x] Unit tests
- [x] Configuration management
- [x] Error handling
- [x] Logging system
- [x] GDPR compliance

---

## 📞 Quick Links

- **GitHub**: https://github.com/yourusername/sgbv_rag_system
- **API Docs**: http://localhost:8000/docs (when running)
- **Official SGB V**: https://www.gesetze-im-internet.de/sgb_5/
- **OpenAI Docs**: https://platform.openai.com/docs
- **FAISS Docs**: https://github.com/facebookresearch/faiss

---

**Version:** 1.0.0  
**Status:** Production Ready ✅  
**Last Updated:** January 4, 2026
