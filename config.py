"""
Configuration management for SGB V RAG System
"""

import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Configuration class"""
    
    # API Configuration
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", "8000"))
    API_WORKERS = int(os.getenv("API_WORKERS", "1"))
    
    # LLM Configuration
    LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4")
    LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))
    LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "2000"))
    LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "30"))
    
    # Embedding Configuration
    EMBEDDING_MODEL = os.getenv(
        "EMBEDDING_MODEL",
        "paraphrase-multilingual-mpnet-base-v2"
    )
    EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu")
    EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))
    
    # Database Configuration
    FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "./data/embeddings/sgbv.index")
    FAISS_METADATA_PATH = os.getenv("FAISS_METADATA_PATH", "./data/embeddings/sgbv_metadata.json")
    
    # Data paths
    RAW_DATA_PATH = "./data/raw"
    PROCESSED_DATA_PATH = "./data/processed"
    EMBEDDINGS_PATH = "./data/embeddings"
    LOGS_PATH = "./logs"
    
    # Scraper Configuration
    SCRAPER_DELAY_SECONDS = int(os.getenv("SCRAPER_DELAY_SECONDS", "2"))
    SCRAPER_TIMEOUT_SECONDS = int(os.getenv("SCRAPER_TIMEOUT_SECONDS", "30"))
    SCRAPER_MAX_RETRIES = int(os.getenv("SCRAPER_MAX_RETRIES", "3"))
    SCRAPER_RESPECT_ROBOTS_TXT = os.getenv("SCRAPER_RESPECT_ROBOTS_TXT", "true").lower() == "true"
    
    # Processing Configuration
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
    
    # Retrieval Configuration
    RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "5"))
    SEMANTIC_WEIGHT = float(os.getenv("SEMANTIC_WEIGHT", "0.6"))
    LEXICAL_WEIGHT = float(os.getenv("LEXICAL_WEIGHT", "0.4"))
    
    # Logging Configuration
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE = os.getenv("LOG_FILE", "./logs/sgbv_rag.log")
    
    # API Keys
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")


def get_config() -> Config:
    """Get configuration instance"""
    return Config()


def validate_config() -> bool:
    """Validate required configuration"""
    required = ['OPENAI_API_KEY', 'LLM_MODEL']
    
    config = get_config()
    for key in required:
        if not getattr(config, key, None):
            print(f"ERROR: Missing required configuration: {key}")
            return False
    
    return True
