"""
SGB V RAG System - RAG Engine Module
Orchestrates retrieval and generation
"""

import json
import logging
from typing import List, Dict, Tuple, Optional
import numpy as np
from openai import OpenAI
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGEngine:
    """
    Orchestrates Retrieval-Augmented Generation
    
    Pipeline:
    1. Query embedding
    2. Semantic + lexical hybrid retrieval
    3. Reranking and context assembly
    4. LLM response generation with citations
    """
    
    def __init__(
        self,
        vector_db,
        chunks: List[Dict],
        llm_model: str = "gpt-4",
        temperature: float = 0.3,
        top_k: int = 5
    ):
        """
        Initialize RAG engine
        
        Args:
            vector_db: FAISS vector database
            chunks: List of chunk dictionaries
            llm_model: OpenAI model name
            temperature: LLM temperature for legal accuracy
            top_k: Number of chunks to retrieve
        """
        self.vector_db = vector_db
        self.chunks = chunks
        self.llm_model = llm_model
        self.temperature = temperature
        self.top_k = top_k
        
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        # Create chunk lookup
        self.chunk_lookup = {c['chunk_id']: c for c in chunks}
        
        logger.info(f"RAG engine initialized with {len(chunks)} chunks")
    
    def _retrieve_semantic(self, query_embedding: np.ndarray, k: int) -> List[Tuple[str, float]]:
        """Semantic retrieval using vector similarity"""
        distances, indices = self.vector_db.search(
            query_embedding.reshape(1, -1),
            k
        )
        
        # FAISS returns distances, convert to similarity scores
        scores = 1 / (1 + distances[0])
        
        results = []
        for idx, score in zip(indices[0], scores):
            if 0 <= idx < len(self.chunks):
                chunk_id = self.chunks[idx]['chunk_id']
                results.append((chunk_id, float(score)))
        
        return results
    
    def _retrieve_lexical(self, query: str, k: int) -> List[Tuple[str, float]]:
        """BM25 lexical retrieval"""
        from rank_bm25 import BM25Okapi
        
        # Tokenize chunks
        corpus = [c['text'].split() for c in self.chunks]
        bm25 = BM25Okapi(corpus)
        
        # Search
        query_tokens = query.split()
        scores = bm25.get_scores(query_tokens)
        
        # Get top k
        top_indices = np.argsort(scores)[-k:][::-1]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                chunk_id = self.chunks[idx]['chunk_id']
                results.append((chunk_id, float(scores[idx])))
        
        return results
    
    def _reciprocal_rank_fusion(
        self,
        semantic_results: List[Tuple[str, float]],
        lexical_results: List[Tuple[str, float]],
        semantic_weight: float = 0.6,
        lexical_weight: float = 0.4
    ) -> List[Tuple[str, float]]:
        """Combine semantic and lexical results using RRF"""
        combined = {}
        
        # Normalize and weight results
        for chunk_id, score in semantic_results:
            combined[chunk_id] = combined.get(chunk_id, 0) + semantic_weight * score
        
        for chunk_id, score in lexical_results:
            combined[chunk_id] = combined.get(chunk_id, 0) + lexical_weight * score
        
        # Sort by combined score
        sorted_results = sorted(combined.items(), key=lambda x: x[1], reverse=True)
        
        return sorted_results
    
    def retrieve(self, query: str, query_embedding: np.ndarray) -> List[Dict]:
        """
        Hybrid retrieval: semantic + lexical with RRF
        
        Args:
            query: User question
            query_embedding: Query embedding vector
            
        Returns:
            List of retrieved chunks with scores
        """
        # Semantic retrieval
        semantic_results = self._retrieve_semantic(query_embedding, self.top_k * 2)
        
        # Lexical retrieval
        lexical_results = self._retrieve_lexical(query, self.top_k * 2)
        
        # Combine with RRF
        combined = self._reciprocal_rank_fusion(
            semantic_results,
            lexical_results,
            semantic_weight=0.6,
            lexical_weight=0.4
        )
        
        # Get top k
        results = []
        for chunk_id, score in combined[:self.top_k]:
            if chunk_id in self.chunk_lookup:
                chunk = self.chunk_lookup[chunk_id].copy()
                chunk['relevance_score'] = score
                results.append(chunk)
        
        logger.info(f"Retrieved {len(results)} chunks for query")
        return results
    
    def _format_context(self, chunks: List[Dict]) -> str:
        """Format retrieved chunks as context"""
        context = ""
        for i, chunk in enumerate(chunks, 1):
            section_id = chunk.get('section_id', 'unknown')
            title = chunk.get('section_title', 'Untitled')
            text = chunk.get('text', '')
            
            context += f"\n[{i}] §{section_id}: {title}\n{text}\n"
        
        return context
    
    def generate_response(
        self,
        query: str,
        retrieved_chunks: List[Dict]
    ) -> Dict:
        """
        Generate response with citations
        
        Args:
            query: User question
            retrieved_chunks: Retrieved context chunks
            
        Returns:
            Dict with answer and citations
        """
        context = self._format_context(retrieved_chunks)
        
        # System prompt for legal expert
        system_prompt = """You are a German legal expert specializing in SGB V (Sozialgesetzbuch V - health insurance law).
Your task is to answer questions accurately based on the provided German legal text.
Always cite your sources by referencing the section numbers [1], [2], etc.
If the provided context doesn't contain relevant information, say "Based on the provided documents, I cannot find information about this."
Be precise and avoid generalizations. Use technical legal terminology appropriately."""
        
        # User prompt with context
        user_prompt = f"""Based on the following German legal text from SGB V:

{context}

Please answer the following question:

{query}

Requirements:
1. Answer strictly based on the provided text
2. Include citations [1], [2], etc. for each source
3. Use German legal terminology
4. If multiple sections are relevant, synthesize them
5. Be concise but complete"""
        
        # Call OpenAI
        logger.info(f"Calling {self.llm_model}...")
        response = self.client.chat.completions.create(
            model=self.llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=self.temperature,
            max_tokens=2000
        )
        
        answer = response.choices[0].message.content
        
        # Extract citations and map to sources
        sources = []
        for i, chunk in enumerate(retrieved_chunks, 1):
            sources.append({
                "citation": f"[{i}]",
                "section_id": chunk.get('section_id', 'unknown'),
                "title": chunk.get('section_title', 'Untitled'),
                "text": chunk.get('text', '')[:500] + "...",  # Preview
                "relevance_score": chunk.get('relevance_score', 0.0)
            })
        
        return {
            "answer": answer,
            "sources": sources,
            "metadata": {
                "model": self.llm_model,
                "temperature": self.temperature,
                "chunks_used": len(retrieved_chunks)
            }
        }
    
    def query(self, question: str, query_embedding: np.ndarray) -> Dict:
        """
        Complete RAG query pipeline
        
        Args:
            question: User question
            query_embedding: Question embedding
            
        Returns:
            Complete response with answer and citations
        """
        # Retrieve
        chunks = self.retrieve(question, query_embedding)
        
        # Generate response
        response = self.generate_response(question, chunks)
        
        return response


def main():
    """Main RAG engine execution"""
    print("RAG Engine module loaded successfully")


if __name__ == "__main__":
    main()
