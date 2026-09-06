# ---- Stage 1: Build Frontend ----
FROM node:20-slim AS frontend-builder

# playwright sits in devDependencies (dev-only screenshot tooling) and npm ci
# installs devDependencies because tsc/vite live there too. Without this flag its
# postinstall downloads a full Chromium (~150 MB) that the build never uses.
ENV PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD=1

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

# Install Python dependencies. Note there is no torch install here on purpose:
# embeddings run through ONNX Runtime (bundled with chromadb), because torch alone
# costs ~190 MB RSS and the full sentence-transformers stack exceeded the 512 MB
# free-tier limit before serving a single request.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Pre-download the ONNX embedding model into the image so the first request
# doesn't pay for the download (and so a cold start isn't network-dependent).
RUN python -c "from chromadb.utils import embedding_functions; embedding_functions.ONNXMiniLM_L6_V2()(['warm up'])"

# Copy backend code and sample files
COPY rag/ ./rag/
COPY server.py main.py sample_handbook.md ./

# Copy built frontend assets
COPY --from=frontend-builder /build/frontend/dist ./frontend/dist

# Ensure user 1000 owns the app directory and every cache the pre-download wrote to
RUN mkdir -p /home/user/app/chroma_db /home/user/.cache && \
    chown -R user:user /home/user

USER user

EXPOSE 10000

# Dynamically listen on $PORT provided by Render or any cloud environment
CMD ["sh", "-c", "exec uvicorn server:app --host 0.0.0.0 --port ${PORT:-10000}"]
