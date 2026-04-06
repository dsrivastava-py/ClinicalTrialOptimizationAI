FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Set Python path so imports work correctly
ENV PYTHONPATH=/app

# HuggingFace Spaces requires port 7860
EXPOSE 7860

# Health check — uses /health endpoint provided by openenv create_fastapi_app
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Start the OpenEnv server
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]