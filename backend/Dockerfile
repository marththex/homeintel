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