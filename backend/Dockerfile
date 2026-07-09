# ── HomeIntel backend Dockerfile ──────────────────────────────────────────────
# Multi-stage build: `dev` for hot-reload, `prod` for the final image.
# ─────────────────────────────────────────────────────────────────────────────

# ── Base: shared Python environment ───────────────────────────────────────────
FROM python:3.11-slim AS base

# System deps needed by unstructured, opencv, and faster-whisper
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    ffmpeg \
    libmagic1 \
    poppler-utils \
    tesseract-ocr \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# This image is CPU-only by design: Ollama (the GPU-heavy component) always
# runs on the host, and docker-compose grants the container no GPU access.
# Install the CPU torch wheel explicitly — on Linux a bare `pip install torch`
# would pull the CUDA build and add multi-GB nvidia-* dependencies for nothing.
# Kept as its own layer, before requirements.txt is copied in, so editing
# requirements.txt never busts this cache and re-downloads torch.
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install Python deps first (layer caches unless requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Dev stage: source is mounted at runtime, not copied ───────────────────────
FROM base AS dev
# Nothing extra — docker-compose.dev.yml mounts ./backend:/app

# ── Prod stage: copy source into image ────────────────────────────────────────
FROM base AS prod
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]