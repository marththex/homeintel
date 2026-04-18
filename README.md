# 🏠 HomeIntel

> A fully local AI assistant for your NAS. Ask questions about your documents, photos, and audio recordings — everything stays on your hardware.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green)
![ChromaDB](https://img.shields.io/badge/ChromaDB-0.5-orange)
![Ollama](https://img.shields.io/badge/Ollama-qwen3.5%3A9b-purple)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

---

## What is HomeIntel?

HomeIntel indexes everything on your NAS — documents, config files, photos, and audio recordings — and lets you query it conversationally using a local LLM. Think of it as a private Google for your home server.

**Everything runs locally. No cloud APIs. No data leaving your machine.**

```
"Which of my Docker containers uses port 443?"
"Find my wedding contracts from 2025."
"What's in my Nextcloud database config?"
"Show me photos from Yosemite."
```

---

## Architecture

```
NAS File System (SMB mount)
        ↓
File Watcher (watchdog) — detects new/changed files
        ↓
Ingestion Pipeline
  ├── PDFs/Docs/Config → Unstructured → chunks → embeddings
  ├── Images           → CLIP embeddings + captions
  └── Audio            → Faster-Whisper → transcript → embeddings
        ↓
ChromaDB (persistent vector store → NAS)
        ↓
LangChain RAG + Ollama / Qwen3.5 9B (local LLM → your PC)
        ↓
FastAPI backend
        ↓
React chat interface
```

---

## Stack

| Layer | Technology |
|---|---|
| LLM | [Qwen3.5 9B](https://ollama.com/library/qwen3.5) via [Ollama](https://ollama.com) |
| Embeddings | [nomic-embed-text](https://ollama.com/library/nomic-embed-text) via Ollama |
| Vector Store | [ChromaDB](https://www.trychroma.com) (persistent, on NAS) |
| RAG Framework | [LangChain](https://langchain.com) |
| Document Parsing | [Unstructured](https://unstructured.io) + [pypdf](https://pypdf.readthedocs.io) (fallback) |
| Image Embeddings | [OpenCLIP](https://github.com/mlfoundations/open_clip) |
| Speech-to-Text | [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper) |
| Backend | [FastAPI](https://fastapi.tiangolo.com) |
| Frontend | [React](https://react.dev) + [Vite](https://vitejs.dev) |
| Deployment | [Docker Compose](https://docs.docker.com/compose/) |

---

## Hardware Setup

This project is designed for a split deployment:

| Component | Where it runs |
|---|---|
| Ollama (LLM + embeddings) | Your PC (GPU-accelerated) |
| FastAPI backend | Your PC |
| React frontend | Your PC |
| ChromaDB (vector store) | NAS (SMB mount) |
| Source files | NAS (SMB mount) |

The NAS doesn't need a GPU. All inference runs on your PC. The NAS just stores files and the vector database.

**Tested on:**
- PC: Windows 11, NVIDIA RTX 5080, 16GB+ RAM
- NAS: TrueNAS, SMB share mounted as `Z:`

---

## Prerequisites

- [Ollama](https://ollama.com/download/windows) installed and running
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Python 3.11+
- NAS mounted as a network drive (SMB)
- NVIDIA GPU recommended (CPU works but is slower)

---

## Quick Startssh

### 1. Clone the repo

```bash
git clone https://github.com/marththex/homeintel.git
cd homeintel
git checkout -b feature/your-feature
```

### 2. Set up the conda environment

```bash
conda create -n homeintel python=3.11 -y
conda activate homeintel

# For NVIDIA GPU (replace cu128 with your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r backend/requirements.txt
```

### 3. Configure environment

```bash
cp .env.example .env
```

Edit `.env` with your paths:

```dotenv
NAS_WATCH_PATH=/path/to/your/nas
CHROMA_PATH=/path/to/your/nas/chroma
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_LLM_MODEL=qwen3.5:9b
OLLAMA_EMBED_MODEL=nomic-embed-text
```

### 4. Pull Ollama models

```bash
ollama pull qwen3.5:9b
ollama pull nomic-embed-text
```

### 5. Create ChromaDB directory on NAS

```powershell
mkdir /path/to/your/nas/chroma
```

### 6. Verify the setup

```bash
cd backend
python ../scripts/test_vectorstore.py
```

All 7 tests should pass before proceeding.

### 7. Start the full stack

```bash
docker compose up -d
```

Then open `http://localhost:3000` in your browser.

---

## Development

For local development with hot-reload:

```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml up
```

The backend will reload on file changes. The frontend runs on Vite dev server at `http://localhost:5173`.

Running the backend directly (without Docker):

```bash
conda activate homeintel
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

---

## Project Structure

```
homeintel/
├── docker-compose.yml          # Production stack
├── docker-compose.dev.yml      # Dev overrides (hot-reload)
├── .env.example                # Config template
│
├── backend/
│   ├── main.py                 # FastAPI entrypoint
│   ├── config.py               # Settings (pydantic-settings)
│   ├── requirements.txt
│   ├── Dockerfile
│   │
│   ├── api/                    # Route handlers
│   │   ├── chat.py             # POST /chat
│   │   ├── search.py           # GET /search
│   │   └── status.py           # GET /health, /stats
│   │
│   ├── ingestion/              # File processing pipeline
│   │   ├── watcher.py          # watchdog file watcher
│   │   ├── pipeline.py         # orchestration
│   │   ├── chunker.py          # shared text chunking
│   │   └── processors/
│   │       ├── document.py     # PDF, DOCX, TXT, MD, YAML, JSON
│   │       ├── image.py        # CLIP embeddings + captions
│   │       └── audio.py        # Faster-Whisper transcription
│   │
│   ├── rag/                    # Retrieval and generation
│   │   ├── retriever.py        # ChromaDB queries
│   │   ├── chain.py            # LangChain RAG chain
│   │   └── prompts.py          # Prompt templates
│   │
│   ├── vectorstore/            # ChromaDB wrapper
│   │   └── chroma.py
│   │
│   └── models/                 # Pydantic schemas
│       ├── chat.py
│       └── file.py
│
├── frontend/                   # React + Vite
│   └── src/
│       ├── components/
│       │   ├── Chat/
│       │   ├── Sources/
│       │   └── Status/
│       ├── hooks/
│       └── api/
│
└── scripts/                    # Utilities
    ├── test_vectorstore.py     # Step 2 smoke test
    ├── test_ingestion.py       # Step 3 smoke test
    ├── cleanup_collection.py   # Wipe all chunks (with confirmation prompt)
    └── reindex.py              # Force full reindex
```

---

## Configuration Reference

| Variable | Default | Description |
|---|---|---|
| `NAS_WATCH_PATH` | `/data/files` | Root path to index |
| `SUPPORTED_EXTENSIONS` | `.pdf,.docx,...` | File types to index |
| `WATCHER_EXCLUDE_PATHS` | `` | Comma-separated paths to skip |
| `CHROMA_PATH` | `/data/chroma` | Vector store persistence path |
| `CHROMA_COLLECTION_NAME` | `homeintel` | ChromaDB collection name |
| `OLLAMA_BASE_URL` | `http://ollama:11434` | Ollama server URL |
| `OLLAMA_LLM_MODEL` | `qwen3.5:9b` | Model for chat/RAG |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text` | Model for embeddings |
| `CHUNK_SIZE` | `512` | Tokens per chunk |
| `CHUNK_OVERLAP` | `64` | Token overlap between chunks |
| `RETRIEVAL_TOP_K` | `6` | Chunks retrieved per query |

---

## What Gets Indexed

| File Type | How it's indexed |
|---|---|
| `.pdf`, `.docx`, `.txt`, `.md` | Text extracted → chunked → embedded |
| `.yml`, `.yaml`, `.json` | Parsed as text → chunked → embedded |
| `.png`, `.jpg`, `.jpeg` | CLIP visual embeddings + generated captions *(coming soon)* |
| `.mp3`, `.wav` | Whisper transcription → chunked → embedded *(coming soon)* |

**Excluded by default:** media files, ROM files, raw footage, vector store itself.

`docker/` on the NAS is also excluded — it contains Docker Engine overlay2 storage (binary layer data),
not user files. Actual service compose files live one level deeper, e.g. `<NAS_ROOT>/service-name/docker-compose.yml`.

---

## Privacy

- No data ever leaves your network
- No cloud API calls
- No telemetry (ChromaDB telemetry disabled)
- All models run locally via Ollama
- Source files are read-only (never modified)

---

## Roadmap

- [x] Step 1 — Scaffolding (Docker Compose, config, Dockerfile)
- [x] Step 2 — ChromaDB vector store with multi-modal metadata
- [x] Step 3 — Document ingestion processor (PDF, DOCX, TXT, MD, YAML, JSON)
- [x] Step 4 — FastAPI backend + RAG chain
- [ ] Step 5 — File watcher (automatic re-indexing on changes)
- [ ] Step 6 — React chat interface
- [ ] CI/CD pipeline
- [ ] Step 7 — Image processor (CLIP embeddings)
- [ ] Step 7 — Audio processor (Faster-Whisper)
- [ ] Re-index script for bulk operations

---

## License

MIT — do whatever you want with it.

---

*Built for personal use on real hardware. Not a demo. Actually useful.*