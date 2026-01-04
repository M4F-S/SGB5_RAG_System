FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY ui/ ./ui/
COPY . .

# Create necessary directories
RUN mkdir -p data/raw data/processed data/embeddings logs

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default: run API server
# Override with: docker run ... streamlit run ui/app.py
EXPOSE 8000 8501

# Start both API and Streamlit (optional)
# Or use: CMD ["python", "-m", "src.api"]
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
