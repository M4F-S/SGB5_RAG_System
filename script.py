
import os
import json
from datetime import datetime

# Define the complete project structure
project_structure = {
    "sgbv_rag_system": {
        "src": {
            "scraper.py": "web_scraper",
            "data_processor.py": "data_processing",
            "embeddings.py": "embedding_generation",
            "rag_engine.py": "rag_core",
            "api.py": "fastapi_endpoints",
            "config.py": "configuration",
            "__init__.py": "init"
        },
        "ui": {
            "app.py": "streamlit_interface",
            "styles.css": "ui_styles"
        },
        "tests": {
            "test_scraper.py": "scraper_tests",
            "test_processor.py": "processor_tests",
            "test_retrieval.py": "retrieval_tests",
            "__init__.py": "init"
        },
        "data": {
            "raw": {},
            "processed": {},
            "embeddings": {}
        },
        "docs": {
            "README.md": "documentation",
            "ARCHITECTURE.md": "architecture_doc",
            "API_DOCS.md": "api_documentation",
            "SETUP.md": "setup_guide"
        },
        "docker": {
            "Dockerfile": "dockerfile",
            "docker-compose.yml": "compose"
        },
        "requirements.txt": "dependencies",
        ".env.example": "env_example",
        ".gitignore": "gitignore",
        "config.yaml": "config_yaml",
        "setup.py": "setup_py"
    }
}

# Count files
def count_files(d):
    count = 0
    for k, v in d.items():
        if isinstance(v, dict):
            count += count_files(v)
        else:
            count += 1
    return count

total_files = count_files(project_structure)
print(f"Total files to generate: {total_files}")
print(f"Project generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
