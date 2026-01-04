"""
SGB V RAG System - Data Processing Module
Cleans, chunks, and preprocesses legal documents
"""

import json
import logging
from typing import List, Dict, Optional
from dataclasses import dataclass
import re
from pathlib import Path
import hashlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """Represents a text chunk with metadata"""
    chunk_id: str
    section_id: str
    section_title: str
    text: str
    hierarchy_level: int
    category: str
    cross_references: List[str]
    document_summary: str
    start_char: int
    end_char: int


class LegalDocumentProcessor:
    """
    Processes legal documents with clause-based chunking
    
    Implements Summary-Augmented Chunking (SAC) strategy:
    - Preserves legal document structure and hierarchy
    - Maintains cross-references
    - Injects document context for better retrieval
    """
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        preserve_context: bool = True
    ):
        """
        Initialize processor
        
        Args:
            chunk_size: Target chunk size in characters
            chunk_overlap: Overlap between chunks to maintain context
            preserve_context: Whether to preserve legal context
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.preserve_context = preserve_context
    
    def _normalize_text(self, text: str) -> str:
        """Clean and normalize legal text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove common HTML artifacts
        text = re.sub(r'<[^>]+>', '', text)
        
        # Normalize quotes
        text = text.replace('"', '"').replace('"', '"')
        
        # Fix common encoding issues
        text = text.replace('&nbsp;', ' ')
        text = text.replace('&quot;', '"')
        text = text.replace('&amp;', '&')
        
        return text.strip()
    
    def _extract_clauses(self, text: str) -> List[Tuple[int, int, str]]:
        """
        Extract natural clause boundaries
        
        Returns:
            List of (start, end, clause_text) tuples
        """
        clauses = []
        
        # Split by common legal delimiters
        # Pattern: numbered clauses (1), (2), etc.
        pattern = r'\(\d+\)|\n\d+\.|§\s*\d+'
        
        splits = list(re.finditer(pattern, text))
        
        if not splits:
            return [(0, len(text), text)]
        
        for i, match in enumerate(splits):
            start = match.start()
            end = splits[i + 1].start() if i + 1 < len(splits) else len(text)
            clause_text = text[start:end].strip()
            
            if clause_text:
                clauses.append((start, end, clause_text))
        
        return clauses
    
    def _create_chunks(self, text: str, section_id: str, section_title: str) -> List[str]:
        """
        Create chunks preserving legal structure
        
        Strategy:
        1. Try to break at clause boundaries
        2. If too large, break at paragraph boundaries
        3. Fallback to character-based splitting
        """
        chunks = []
        
        # First try clause-based splitting
        clauses = self._extract_clauses(text)
        current_chunk = ""
        
        for start, end, clause in clauses:
            clause = clause.strip()
            
            if len(current_chunk) + len(clause) + 1 <= self.chunk_size:
                current_chunk += (" " if current_chunk else "") + clause
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                
                # If single clause is too large, split it
                if len(clause) > self.chunk_size:
                    # Split by paragraphs
                    paragraphs = clause.split('\n')
                    para_chunk = ""
                    
                    for para in paragraphs:
                        if len(para_chunk) + len(para) + 1 <= self.chunk_size:
                            para_chunk += (" " if para_chunk else "") + para
                        else:
                            if para_chunk:
                                chunks.append(para_chunk)
                            para_chunk = para
                    
                    if para_chunk:
                        chunks.append(para_chunk)
                else:
                    current_chunk = clause
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def _generate_document_summary(self, text: str, section_title: str) -> str:
        """
        Generate a document-level summary for context injection
        
        This improves retrieval by providing document fingerprint
        """
        # Extract key sentences
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Keep first few sentences for context
        summary = '. '.join(sentences[:3])
        
        # Add section title
        summary = f"{section_title}: {summary}"
        
        # Limit to ~150 characters (SAC recommendation)
        if len(summary) > 150:
            summary = summary[:147] + "..."
        
        return summary
    
    def _generate_chunk_id(self, section_id: str, chunk_index: int) -> str:
        """Generate unique chunk ID"""
        return f"{section_id}_chunk_{chunk_index}"
    
    def process_sections(self, sections: List[Dict]) -> List[Chunk]:
        """
        Process raw sections into chunks
        
        Args:
            sections: List of section dictionaries from scraper
            
        Returns:
            List of Chunk objects
        """
        all_chunks = []
        
        for section in sections:
            section_id = section.get('section_id', 'unknown')
            section_title = section.get('title', 'Untitled')
            text = section.get('text', '')
            cross_refs = section.get('cross_references', [])
            hierarchy = section.get('hierarchy_level', 1)
            category = section.get('category', 'General')
            
            # Normalize text
            text = self._normalize_text(text)
            
            if not text or len(text) < 50:
                logger.warning(f"Section {section_id} too short, skipping")
                continue
            
            # Generate document summary
            doc_summary = self._generate_document_summary(text, section_title)
            
            # Create chunks
            chunk_texts = self._create_chunks(text, section_id, section_title)
            
            # Create chunk objects with metadata
            for i, chunk_text in enumerate(chunk_texts):
                chunk_id = self._generate_chunk_id(section_id, i)
                
                chunk = Chunk(
                    chunk_id=chunk_id,
                    section_id=section_id,
                    section_title=section_title,
                    text=chunk_text,
                    hierarchy_level=hierarchy,
                    category=category,
                    cross_references=cross_refs,
                    document_summary=doc_summary,
                    start_char=0,  # Could be calculated if needed
                    end_char=len(chunk_text)
                )
                
                all_chunks.append(chunk)
            
            logger.info(f"Processed section {section_id}: {len(chunk_texts)} chunks")
        
        return all_chunks
    
    def save_chunks(self, chunks: List[Chunk], output_path: str):
        """Save chunks to JSON"""
        data = []
        for chunk in chunks:
            data.append({
                'chunk_id': chunk.chunk_id,
                'section_id': chunk.section_id,
                'section_title': chunk.section_title,
                'text': chunk.text,
                'hierarchy_level': chunk.hierarchy_level,
                'category': chunk.category,
                'cross_references': chunk.cross_references,
                'document_summary': chunk.document_summary
            })
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved {len(chunks)} chunks to {output_path}")
    
    def load_sections(self, input_path: str) -> List[Dict]:
        """Load sections from JSON"""
        with open(input_path, 'r', encoding='utf-8') as f:
            return json.load(f)


def main():
    """Main processing execution"""
    import os
    
    os.makedirs('data/processed', exist_ok=True)
    
    # Load raw sections
    processor = LegalDocumentProcessor(
        chunk_size=1000,
        chunk_overlap=200
    )
    
    logger.info("Loading raw sections...")
    sections = processor.load_sections('data/raw/sgbv_sections.json')
    logger.info(f"Loaded {len(sections)} sections")
    
    # Process sections
    logger.info("Processing sections into chunks...")
    chunks = processor.process_sections(sections)
    logger.info(f"Created {len(chunks)} chunks")
    
    # Save chunks
    processor.save_chunks(chunks, 'data/processed/sgbv_chunks.json')
    
    print(f"\n✓ Processing complete: {len(chunks)} chunks")
    print(f"✓ Chunks saved to: data/processed/sgbv_chunks.json")


if __name__ == "__main__":
    main()
