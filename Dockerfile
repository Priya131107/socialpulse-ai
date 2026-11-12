# Production Dockerfile for SocialPulse AI
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy application source code and configurations
COPY config.yaml .env.example ./
COPY src/ ./src/
COPY dashboard/ ./dashboard/
COPY models_saved/ ./models_saved/
COPY data/ ./data/

# Expose API and Streamlit ports
EXPOSE 8000 8501

# Default command: launch FastAPI REST service
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
