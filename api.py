"""
SGB V RAG System - FastAPI Server
REST API endpoints for RAG system
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import logging
import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

# Local imports
from rag_engine import RAGEngine
from embeddings import EmbeddingGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize app
app = FastAPI(
    title="SGB V RAG API",
    description="Retrieval-Augmented Generation for German Health Insurance Law",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class QueryRequest(BaseModel):
    """Query request model"""
    question: str
    top_k: int = 5
    temperature: float = 0.3

class SearchRequest(BaseModel):
    """Search request model"""
    query: str
    top_k: int = 10
    search_type: str = "hybrid"  # "semantic", "lexical", or "hybrid"

class QueryResponse(BaseModel):
    """Query response model"""
    answer: str
    sources: List[dict]
    metadata: dict

class SearchResponse(BaseModel):
    """Search response model"""
    results: List[dict]
    total_results: int

# Global RAG system
rag_engine = None
embedding_generator = None
chunks = None


def initialize_rag_system():
    """Initialize RAG system on startup"""
    global rag_engine, embedding_generator, chunks
    
    logger.info("Initializing RAG system...")
    
    # Load chunks
    logger.info("Loading chunks...")
    with open('data/embeddings/sgbv_chunks_embedded.json', 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    logger.info(f"Loaded {len(chunks)} chunks")
    
    # Load embeddings
    logger.info("Loading embeddings...")
    embeddings = np.load('data/embeddings/sgbv_embeddings.npy')
    logger.info(f"Loaded embeddings shape: {embeddings.shape}")
    
    # Initialize FAISS
    dimension = embeddings.shape[1]
    vector_db = faiss.IndexFlatL2(dimension)
    vector_db.add(embeddings.astype('float32'))
    logger.info("FAISS index created")
    
    # Initialize embedding generator
    embedding_generator = EmbeddingGenerator(
        model_name="paraphrase-multilingual-mpnet-base-v2",
        device="cpu"
    )
    
    # Initialize RAG engine
    rag_engine = RAGEngine(
        vector_db=vector_db,
        chunks=chunks,
        llm_model="gpt-4",
        temperature=0.3,
        top_k=5
    )
    
    logger.info("RAG system initialized successfully")


@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    initialize_rag_system()


# Endpoints

@app.get("/", tags=["info"])
async def root():
    """API information"""
    return {
        "name": "SGB V RAG System",
        "description": "Retrieval-Augmented Generation for German Health Insurance Law",
        "version": "1.0.0",
        "docs_url": "/docs",
        "endpoints": {
            "query": "POST /query",
            "search": "POST /search",
            "section": "GET /section/{section_id}",
            "health": "GET /health"
        }
    }


@app.get("/health", tags=["system"])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "system": "RAG Engine",
        "chunks_loaded": len(chunks) if chunks else 0
    }


@app.post("/query", response_model=QueryResponse, tags=["rag"])
async def query(request: QueryRequest):
    """
    Query the SGB V knowledge base
    
    Args:
        request: QueryRequest with question
        
    Returns:
        QueryResponse with answer and sources
    """
    if not rag_engine:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    if not request.question or len(request.question.strip()) < 3:
        raise HTTPException(status_code=400, detail="Question must be at least 3 characters")
    
    try:
        # Generate query embedding
        query_embedding = embedding_generator.embed_query(request.question)
        
        # Get response
        response = rag_engine.query(request.question, query_embedding)
        
        return response
    
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search", response_model=SearchResponse, tags=["search"])
async def search(request: SearchRequest):
    """
    Search for relevant sections
    
    Args:
        request: SearchRequest with query
        
    Returns:
        SearchResponse with matching sections
    """
    if not rag_engine:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    if not request.query or len(request.query.strip()) < 2:
        raise HTTPException(status_code=400, detail="Query must be at least 2 characters")
    
    try:
        # Generate query embedding
        query_embedding = embedding_generator.embed_query(request.query)
        
        # Retrieve chunks based on search type
        if request.search_type == "semantic":
            results = rag_engine._retrieve_semantic(query_embedding, request.top_k)
            chunks_list = [rag_engine.chunk_lookup[cid] for cid, _ in results]
        
        elif request.search_type == "lexical":
            results = rag_engine._retrieve_lexical(request.query, request.top_k)
            chunks_list = [rag_engine.chunk_lookup[cid] for cid, _ in results]
        
        else:  # hybrid
            semantic_results = rag_engine._retrieve_semantic(query_embedding, request.top_k * 2)
            lexical_results = rag_engine._retrieve_lexical(request.query, request.top_k * 2)
            combined = rag_engine._reciprocal_rank_fusion(semantic_results, lexical_results)
            
            chunks_list = []
            for cid, score in combined[:request.top_k]:
                if cid in rag_engine.chunk_lookup:
                    chunk = rag_engine.chunk_lookup[cid].copy()
                    chunk['relevance_score'] = score
                    chunks_list.append(chunk)
        
        # Format results
        results = []
        for chunk in chunks_list:
            results.append({
                "section_id": chunk.get('section_id'),
                "title": chunk.get('section_title'),
                "text": chunk.get('text', '')[:500] + "...",
                "score": chunk.get('relevance_score', 0.0),
                "category": chunk.get('category')
            })
        
        return {
            "results": results,
            "total_results": len(results)
        }
    
    except Exception as e:
        logger.error(f"Error processing search: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/section/{section_id}", tags=["sections"])
async def get_section(section_id: str):
    """
    Get full section details
    
    Args:
        section_id: Section number (e.g., "31")
        
    Returns:
        Section details with metadata
    """
    if not chunks:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    # Find all chunks for this section
    section_chunks = [c for c in chunks if c.get('section_id') == section_id]
    
    if not section_chunks:
        raise HTTPException(status_code=404, detail=f"Section {section_id} not found")
    
    # Combine chunks
    first_chunk = section_chunks[0]
    full_text = '\n'.join([c.get('text', '') for c in section_chunks])
    
    return {
        "section_id": section_id,
        "title": first_chunk.get('section_title'),
        "text": full_text,
        "chunks_count": len(section_chunks),
        "category": first_chunk.get('category'),
        "hierarchy_level": first_chunk.get('hierarchy_level'),
        "cross_references": first_chunk.get('cross_references', []),
        "metadata": {
            "last_updated": first_chunk.get('last_updated'),
            "document_summary": first_chunk.get('document_summary')
        }
    }


@app.get("/sections", tags=["sections"])
async def list_sections():
    """List all available sections"""
    if not chunks:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    # Get unique sections
    sections = {}
    for chunk in chunks:
        section_id = chunk.get('section_id')
        if section_id not in sections:
            sections[section_id] = {
                "section_id": section_id,
                "title": chunk.get('section_title'),
                "category": chunk.get('category'),
                "chunks": 0
            }
        sections[section_id]["chunks"] += 1
    
    return {
        "total_sections": len(sections),
        "sections": list(sections.values())
    }


if __name__ == "__main__":
    import uvicorn
    
    # Get configuration from environment
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    workers = int(os.getenv("API_WORKERS", "1"))
    
    uvicorn.run(
        "api:app",
        host=host,
        port=port,
        workers=workers,
        reload=False
    )
