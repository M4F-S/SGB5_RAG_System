"""
SGB V RAG System - Embeddings Module
Generates and manages embeddings for semantic search
"""

import json
import logging
import numpy as np
from typing import List, Dict, Tuple, Optional
import torch
from sentence_transformers import SentenceTransformer
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """
    Generates embeddings for legal document chunks
    
    Uses German-optimized sentence-transformers model
    """
    
    def __init__(
        self,
        model_name: str = "paraphrase-multilingual-mpnet-base-v2",
        batch_size: int = 32,
        device: str = "cpu"
    ):
        """
        Initialize embedding generator
        
        Args:
            model_name: Hugging Face model name
            batch_size: Batch size for embedding
            device: "cpu" or "cuda"
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name, device=device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Embedding dimension: {self.embedding_dim}")
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for multiple texts
        
        Args:
            texts: List of text strings
            
        Returns:
            Embeddings array of shape (n_texts, embedding_dim)
        """
        logger.info(f"Generating embeddings for {len(texts)} texts...")
        
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        logger.info(f"Generated embeddings shape: {embeddings.shape}")
        return embeddings
    
    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query"""
        return self.model.encode(query, convert_to_numpy=True)
    
    def embed_chunks(self, chunks: List[Dict]) -> Tuple[np.ndarray, List[Dict]]:
        """
        Embed document chunks with metadata
        
        Args:
            chunks: List of chunk dictionaries
            
        Returns:
            (embeddings array, enhanced chunks with embeddings)
        """
        # Extract texts (combine summary + text for better context)
        texts = []
        for chunk in chunks:
            summary = chunk.get('document_summary', '')
            text = chunk.get('text', '')
            combined = f"{summary}\n{text}" if summary else text
            texts.append(combined)
        
        # Generate embeddings
        embeddings = self.embed_texts(texts)
        
        # Add embeddings to chunks
        enhanced_chunks = []
        for i, chunk in enumerate(chunks):
            chunk['embedding'] = embeddings[i].tolist()
            enhanced_chunks.append(chunk)
        
        return embeddings, enhanced_chunks
    
    def save_embeddings(self, embeddings: np.ndarray, output_path: str):
        """Save embeddings to binary file"""
        np.save(output_path, embeddings)
        logger.info(f"Saved embeddings to {output_path}")
    
    def load_embeddings(self, input_path: str) -> np.ndarray:
        """Load embeddings from binary file"""
        embeddings = np.load(input_path)
        logger.info(f"Loaded embeddings shape: {embeddings.shape}")
        return embeddings


def main():
    """Main embedding generation execution"""
    import os
    
    os.makedirs('data/embeddings', exist_ok=True)
    
    # Load processed chunks
    logger.info("Loading processed chunks...")
    with open('data/processed/sgbv_chunks.json', 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    logger.info(f"Loaded {len(chunks)} chunks")
    
    # Initialize embedding generator
    generator = EmbeddingGenerator(
        model_name="paraphrase-multilingual-mpnet-base-v2",
        batch_size=32,
        device="cpu"  # Change to "cuda" if GPU available
    )
    
    # Generate embeddings
    embeddings, enhanced_chunks = generator.embed_chunks(chunks)
    
    # Save embeddings
    generator.save_embeddings(embeddings, 'data/embeddings/sgbv_embeddings.npy')
    
    # Save enhanced chunks with embeddings
    with open('data/embeddings/sgbv_chunks_embedded.json', 'w', encoding='utf-8') as f:
        json.dump(enhanced_chunks, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ Embedding generation complete")
    print(f"✓ Embeddings saved to: data/embeddings/sgbv_embeddings.npy")
    print(f"✓ Chunks with embeddings: data/embeddings/sgbv_chunks_embedded.json")


if __name__ == "__main__":
    main()
