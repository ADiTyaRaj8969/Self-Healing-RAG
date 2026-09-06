# ---- Stage 1: Build Frontend ----
FROM node:20-slim AS frontend-builder
WORKDIR /build/frontend
COPY frontend/package*.json ./
RUN npm ci || npm install
COPY frontend/ ./
RUN npm run build

# ---- Stage 2: Python Application ----
FROM python:3.11-slim

# Set environment variables for memory efficiency, persistent cache, and port binding
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    PORT=10000 \
    HOME=/home/user \
    HF_HOME=/home/user/.cache/huggingface

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 user
WORKDIR /home/user/app

# Install CPU-only PyTorch first to prevent downloading huge CUDA libraries (~3 GB)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install remaining Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Pre-download the embedding model into the container image cache at build time
RUN python -c "from langchain_huggingface import HuggingFaceEmbeddings; HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')"

# Copy backend code and sample files
COPY rag/ ./rag/
COPY server.py main.py sample_handbook.md ./

# Copy built frontend assets
COPY --from=frontend-builder /build/frontend/dist ./frontend/dist

# Ensure user 1000 has ownership of the app directory and cached model
RUN mkdir -p /home/user/app/chroma_db /home/user/.cache && \
    chown -R user:user /home/user

USER user

EXPOSE 10000

# Dynamically listen on $PORT provided by Render or any cloud environment
CMD ["sh", "-c", "exec uvicorn server:app --host 0.0.0.0 --port ${PORT:-10000}"]
