# HomeIntel

> A fully local AI assistant for your NAS. Ask questions about your documents, photos, and audio recordings — everything stays on your hardware.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green)
![Qdrant](https://img.shields.io/badge/Qdrant-hybrid_search-orange)
![Ollama](https://img.shields.io/badge/Ollama-qwen3%3A14b-purple)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

---

## What is HomeIntel?

HomeIntel indexes everything on your NAS — documents, config files, photos, and audio recordings — and lets you query it conversationally using a local LLM. Think of it as a private Google for your home server.

**Everything runs locally. No cloud APIs. No data leaving your machine.**

```
"Which of my Docker containers uses port 443?"
"Find my wedding contracts from 2025."
"What's in my Nextcloud database config?"
"Summarise the music in my library."
```

---

## Architecture

```
NAS File System (SMB mount at Z:)
        ↓
File Watcher (watchdog) — detects new/changed/deleted files
        ↓
Ingestion Pipeline
  ├── PDFs/Docs/Config → Docling → chunks → nomic-embed-text + BM25
  ├── Images           → Ollama vision caption → chunks → embeddings
  ├── Audio            → Faster-Whisper transcript → chunks → embeddings
  ├── PDFs (visual)    → ColPali page embeddings (batch, GPU)
  └── Photos (visual)  → CLIP image embeddings (batch, GPU)
        ↓
Qdrant (Docker on VM/NAS) — hybrid dense+sparse + ColPali + CLIP collections
        ↓
FastAPI backend — RAG chain + bge-reranker-v2-m3 + /visual-search + /file
        ↓
React chat interface (Vite, port 5173) — markdown answers, swipeable photo carousel,
mobile-first composer (camera/library upload, filter + result-count sheet)
```

---

## Stack

| Layer | Technology |
|---|---|
| LLM | [Qwen3 14B](https://ollama.com/library/qwen3) via [Ollama](https://ollama.com) |
| Embeddings | [nomic-embed-text](https://ollama.com/library/nomic-embed-text) via Ollama |
| Sparse index | BM25 via [fastembed](https://github.com/qdrant/fastembed) |
| Reranker | [bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3) (CrossEncoder) |
| Vector Store | [Qdrant](https://qdrant.tech) (Docker, hybrid dense+sparse+ColPali) |
| RAG Framework | [LangChain](https://langchain.com) |
| Document Parsing | [Docling](https://ds4sd.github.io/docling/) (PDF/DOCX → structured markdown) |
| Visual Retrieval (PDF) | [ColPali](https://github.com/illuin-tech/colpali) (vidore/colpali-v1.2, batch GPU) |
| Visual Similarity (photos) | [CLIP](https://huggingface.co/openai/clip-vit-large-patch14) (query-by-photo, `homeintel_visual`) |
| Image Captioning | Ollama vision model (opt-in via `OLLAMA_VISION_MODEL`) |
| Speech-to-Text | [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper) (GPU) |
| Backend | [FastAPI](https://fastapi.tiangolo.com) |
| Frontend | [React](https://react.dev) + [Vite](https://vitejs.dev) |
| Deployment | [Docker Compose](https://docs.docker.com/compose/) |

---

## Hardware Setup

| Component | Where it runs |
|---|---|
| Ollama (LLM + embeddings) | PC (GPU — RTX 5080) |
| FastAPI backend | PC |
| React frontend | PC |
| Qdrant (vector store) | VM or NAS (Docker) |
| Source files | NAS (SMB mount `Z:`) |

**VRAM budget on RTX 5080 (16 GB) during normal inference:**

| Model | VRAM |
|---|---|
| qwen3:14b (Q4_K_M) | ~9–10 GB |
| nomic-embed-text | ~270 MB |
| bge-reranker-v2-m3 | ~1.1 GB |
| CLIP large (lazy, first `/visual-search`) | ~500 MB |
| **Headroom** | **~4.5 GB** |

ColPali (~8 GB) is a separate batch process — not loaded during live inference; keep
`COLPALI_ENABLED=false` for normal use (it OOMs alongside qwen3:14b at query time).
Vision captioning (`qwen2.5vl:7b`, ~5 GB) runs only during file ingestion.
CLIP photo embeddings are indexed in batch by `scripts/index_visual.py`; the model
loads lazily (~500 MB) on the first `/visual-search` and stays resident for queries.

---

## Prerequisites

- [Ollama](https://ollama.com/download/windows) installed and running
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (or Docker on a VM/NAS)
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Python 3.11+
- NAS mounted as a network drive (SMB)
- NVIDIA GPU (CPU works but is much slower)

---

## Quick Start

### 1. Clone the repo

```bash
git clone https://github.com/marththex/homeintel.git
cd homeintel
```

### 2. Set up the conda environment

```bash
conda create -n homeintel python=3.11 -y
conda activate homeintel

# RTX 5080 / CUDA 12.8+ build
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r backend/requirements.txt
```

### 3. Deploy Qdrant

Copy `docker/qdrant.yml` to your VM or NAS and start it:

```bash
docker compose -f qdrant.yml up -d
# Confirm it's up:
curl http://<YOUR_QDRANT_IP>:6333/healthz
```

### 4. Configure environment

```bash
cp .env.example .env
```

Edit `.env` — at minimum set:

```dotenv
NAS_WATCH_PATH=Z:/
QDRANT_URL=http://<YOUR_QDRANT_IP>:6333
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_LLM_MODEL=qwen3:14b
OLLAMA_EMBED_MODEL=nomic-embed-text
```

### 5. Pull Ollama models

```bash
ollama pull qwen3:14b
ollama pull nomic-embed-text

# Optional — for image captioning:
ollama pull qwen2.5vl:7b
```

### 6. Smoke test

```bash
cd backend
python ../scripts/test_vectorstore.py   # Qdrant connectivity
python ../scripts/test_ingestion.py     # ingest a PDF + docker-compose + query
```

### 7. Start the backend

```bash
# Direct (dev)
conda activate homeintel
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Or via Docker Compose
docker compose up -d
```

### 8. Start the frontend

```bash
cd frontend
npm install
npm run dev
```

Open **http://localhost:5173**.

---

## Running the App

Both services start automatically at Windows login via Task Scheduler. Access the UI at **http://localhost:5173**.

**Manual start/stop:**
```powershell
Start-ScheduledTask "HomeIntel-API"
Stop-ScheduledTask "HomeIntel-API"
Start-ScheduledTask "HomeIntel-UI"
Stop-ScheduledTask "HomeIntel-UI"
```

**Freeing GPU VRAM (~11.4 GB held during normal use):**
```powershell
# Stop backend (frees reranker + embeddings ~1.4 GB)
Stop-ScheduledTask "HomeIntel-API"

# Stop Ollama (frees qwen3:14b + nomic-embed-text ~10 GB)
taskkill /IM "ollama app.exe" /F

# Restart when needed
Start-Process "$env:LOCALAPPDATA\Programs\Ollama\ollama app.exe"
Start-ScheduledTask "HomeIntel-API"
```

---

## Development

Run backend directly with hot-reload:

```bash
conda activate homeintel
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Run frontend dev server:

```bash
cd frontend
npm run dev   # http://localhost:5173
```

Run the full API verification suite (requires Qdrant + Ollama up and test chunks loaded):

```bash
cd backend
python ../scripts/test_ingestion.py --keep-chunks
python ../scripts/verify_api.py
```

---

## ColPali Visual Indexing (optional)

ColPali embeds PDF pages as patch-level multi-vectors for visual retrieval — useful for documents with charts, tables, or complex layouts.

Install extra deps first:

```bash
pip install colpali-engine>=0.3.0 pypdfium2>=4.0.0
```

Run the batch indexer (stop Ollama first to free VRAM — ColPali needs ~8 GB):

```bash
cd backend
python ../scripts/run_colpali.py                          # all PDFs under NAS_WATCH_PATH
python ../scripts/run_colpali.py --path Z:/Wedding_Contracts
python ../scripts/run_colpali.py --force                  # re-index
```

Then enable retrieval in `.env`:

```dotenv
COLPALI_ENABLED=true
```

> **Note:** ColPali OOMs alongside qwen3:14b at query time on a 16 GB GPU. Keep
> `COLPALI_ENABLED=false` for normal use — the indexed page vectors stay in Qdrant
> and cost no VRAM; you only need the model loaded to add new PDFs.

---

## CLIP Visual Photo Search (optional)

Query your photo library **by photo** — take or upload a picture of your dog and find
every other photo of that dog. CLIP encodes raw image pixels into a shared embedding
space, so visually similar photos cluster together (independent of any text caption).

No extra deps needed (`transformers` + `torch` are already installed). Build the index:

```bash
conda activate homeintel
cd backend
# Temporarily remove Z:/marcus_photoprism from WATCHER_EXCLUDE_PATHS in .env first
python ../scripts/index_visual.py --path Z:/marcus_photoprism/originals
python ../scripts/index_visual.py --clear --path Z:/photos     # wipe + reindex
python ../scripts/index_visual.py --skip-existing               # resume interrupted run
python ../scripts/index_visual.py --batch-size 32              # bigger GPU batches
# Restore WATCHER_EXCLUDE_PATHS after
```

The indexer batches the CLIP forward pass (`--batch-size`, default 16) and reads images
in parallel (`--read-workers`, default 4) — ~5–8× faster than one-at-a-time. Photos are
stored in the `homeintel_visual` Qdrant collection (one vector per photo, ~500 MB VRAM).
Once indexed, tap **+ → Take a photo / Choose from library** in the web UI. Matches above
0.70 cosine similarity are returned in a swipeable carousel with match-percentage labels.

### Faster ingestion (captions + documents)

`reindex.py` ingests one file at a time by default. For the photo-caption job (the slow
one — ~5–7 s/photo on the vision model), parallelize it:

```bash
# Set Ollama to serve concurrent requests, then restart the Ollama service:
setx OLLAMA_NUM_PARALLEL 3

python ../scripts/reindex.py --path Z:/photos --ext .jpg .jpeg .png --workers 3
python ../scripts/reindex.py --skip-existing --workers 3        # resume where you left off
```

`--workers N` issues N captions concurrently (network-bound, so the GPU stays busy);
pair it with `OLLAMA_NUM_PARALLEL=N`. Drop to 2 if `qwen2.5vl:7b` OOMs on 16 GB, or switch
`OLLAMA_VISION_MODEL=qwen2.5vl:3b` to roughly halve caption time. `--skip-existing` skips
files already in Qdrant, so a cancelled run (Ctrl+C) resumes cleanly.

---

## Web UI Features

- **Markdown answers** — bold, lists, and code render properly
- **Swipeable photo carousel** — image results show full-size, swipe left/right, "Show
  more" pagination; tap **+** to search by photo (camera or library)
- **Mobile-first composer** — single rounded input pill (`+` · text · send); the `+`
  sheet holds visual search, modality filter (All/Documents/Images/Audio), and a
  "Results to show" stepper (Auto/3/6/10/15/20)
- **Adaptive results** — image queries default to 20 results, doc/audio to 6; override
  per-query from the `+` sheet
- **Status sheet** — tap the health dot in the header for Ollama/Qdrant status + chunk counts
- **iOS PWA-ready** — add to home screen; safe-area insets handled

---

## Project Structure

```
homeintel/
├── docker-compose.yml              # Production stack
├── docker-compose.dev.yml          # Dev overrides (hot-reload)
├── docker/
│   └── qdrant.yml                  # Qdrant on VM/NAS
├── .env.example                    # Config template
│
├── backend/
│   ├── main.py                     # FastAPI entrypoint + watcher start
│   ├── config.py                   # Settings (pydantic-settings)
│   ├── requirements.txt
│   ├── Dockerfile
│   │
│   ├── api/
│   │   ├── chat.py                 # POST /chat
│   │   └── status.py               # GET /health, GET /stats
│   │
│   ├── ingestion/
│   │   ├── watcher.py              # watchdog NAS watcher (SMB reconnect)
│   │   ├── pipeline.py             # routes files to processors
│   │   ├── chunker.py              # shared LangChain text splitter
│   │   └── processors/
│   │       ├── document.py         # PDF/DOCX (Docling), TXT/MD/YAML/JSON
│   │       ├── image.py            # Ollama vision caption
│   │       ├── audio.py            # Faster-Whisper transcription
│   │       └── colpali.py          # ColPali page embeddings (batch)
│   │
│   ├── rag/
│   │   ├── retriever.py            # hybrid + ColPali merge + reranker + adaptive top-k
│   │   ├── chain.py                # LangChain RAG chain
│   │   └── prompts.py              # system prompt template
│   │
│   ├── vectorstore/
│   │   ├── qdrant.py               # Qdrant wrapper (dense+sparse+ColPali)
│   │   └── clip.py                 # CLIP visual store (homeintel_visual)
│   │
│   ├── api/
│   │   ├── chat.py                 # POST /chat
│   │   ├── status.py               # GET /health, /stats
│   │   └── files.py                # GET /file, POST /visual-search
│   │
│   └── models/
│       └── chat.py                 # ChatRequest / ChatResponse / SourceDoc
│
├── frontend/                       # React + Vite
│   └── src/
│       ├── App.tsx                 # layout, composer, action + status sheets
│       ├── api.ts                  # fetch wrappers (chat, visual-search, file)
│       ├── types.ts
│       ├── hooks/
│       │   └── useSystemStatus.ts  # health + stats polling
│       └── components/
│           ├── Header.tsx          # slim header + health dot
│           ├── ChatMessage.tsx     # markdown answers
│           ├── SourceList.tsx      # non-image sources + image carousel
│           ├── PhotoCarousel.tsx   # swipeable photo results
│           ├── BottomSheet.tsx     # reusable iOS-style sheet
│           └── icons.tsx           # inline SVG icons
│
└── scripts/
    ├── test_vectorstore.py         # Qdrant smoke test
    ├── test_ingestion.py           # ingest + query smoke test
    ├── verify_api.py               # 8-check API integration test
    ├── run_colpali.py              # batch ColPali PDF indexer
    └── index_visual.py             # batch CLIP photo indexer
```

---

## Configuration Reference

| Variable | Default | Description |
|---|---|---|
| `NAS_WATCH_PATH` | `/data/files` | Root path to watch and index |
| `SUPPORTED_EXTENSIONS` | `.pdf,.docx,.txt,.md,.yml,.yaml,.json,.png,.jpg,.jpeg` | Comma-separated extensions (audio excluded by default — too slow) |
| `WATCHER_EXCLUDE_PATHS` | `` | Comma-separated paths to skip |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant REST API URL |
| `QDRANT_COLLECTION_NAME` | `homeintel` | Collection name |
| `QDRANT_API_KEY` | `` | Leave empty for no-auth LAN setup |
| `EMBED_DIM` | `768` | nomic-embed-text output dim |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server (use `host.docker.internal` in Docker) |
| `OLLAMA_LLM_MODEL` | `qwen3:14b` | Chat/RAG model |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text` | Embedding model |
| `OLLAMA_VISION_MODEL` | `` | Vision model for image captioning (empty = skip images) |
| `WHISPER_MODEL_SIZE` | `base` | faster-whisper model size |
| `CHUNK_SIZE` | `512` | Tokens per chunk |
| `CHUNK_OVERLAP` | `64` | Token overlap between chunks |
| `RETRIEVAL_TOP_K` | `6` | Chunks returned per query (image queries auto-bump to 20; UI can override) |
| `RERANKER_ENABLED` | `true` | Cross-encoder reranking |
| `RERANKER_MODEL` | `BAAI/bge-reranker-v2-m3` | Reranker model |
| `COLPALI_ENABLED` | `false` | Enable ColPali retrieval (run batch indexer first; OOMs with qwen3:14b — keep false) |
| `COLPALI_MODEL` | `vidore/colpali-v1.2` | ColPali model |
| `CLIP_MODEL` | `openai/clip-vit-large-patch14` | CLIP model for visual photo search |
| `REDACT_SECRETS` | `true` | Redact passwords/keys/tokens at ingestion, context, and excerpts |
| `DOCLING_VLM_ENABLED` | `false` | Docling VLM picture descriptions |

---

## What Gets Indexed

| File Type | How it's indexed |
|---|---|
| `.pdf`, `.docx` | Docling → structured markdown → chunks → embeddings |
| `.txt`, `.md`, `.yml`, `.yaml`, `.json` | Plain text → chunks → embeddings |
| `.png`, `.jpg`, `.jpeg`, `.gif`, `.webp` | Ollama vision caption → chunks → embeddings (requires `OLLAMA_VISION_MODEL`) |
| `.mp3`, `.wav`, `.m4a`, `.flac`, `.ogg` | Faster-Whisper transcript → chunks → embeddings |
| PDFs (visual, batch) | ColPali page multi-vectors → `homeintel_colpali` collection |
| Photos (visual, batch) | CLIP image embeddings → `homeintel_visual` collection (query-by-photo) |

**Excluded by default:** media files (movies/TV), ROM files, raw footage, Qdrant storage, Docker overlay2, Music library (Whisper too slow), web project node_modules, service data dirs (Vaultwarden, MariaDB, Nextcloud app code).

**Photos (PhotoPrism):** indexed separately targeting only image files to skip YAML sidecars:
```bash
python ../scripts/reindex.py --path Z:/marcus_photoprism/originals --ext .jpg .jpeg .png
```

---

## Privacy

- No data ever leaves your network
- No cloud API calls
- All models run locally via Ollama or on-device (Whisper, reranker, ColPali, CLIP)
- Source files are read-only (never modified)
- **Secret redaction** (`REDACT_SECRETS=true`, default on): passwords, API keys,
  tokens, and private keys are detected and replaced with `<REDACTED>` at three
  layers — ingestion (kept out of Qdrant), LLM context (the model never sees the
  raw value), and source excerpts (UI never shows it). The system prompt also
  instructs the LLM to refuse to reveal credentials. See `backend/security/redact.py`.
  Existing secrets indexed before this was enabled are still scrubbed at response
  time; run `reindex.py` to also purge them from Qdrant storage.

---

## Roadmap

- [x] Step 1 — Scaffolding (Docker Compose, config, Dockerfile)
- [x] Step 2 — Qdrant vector store (hybrid dense+sparse, RRF fusion)
- [x] Step 3 — Document ingestion (Docling for PDF/DOCX, plain read for TXT/MD/YAML/JSON)
- [x] Step 4 — FastAPI backend + RAG chain (LangChain + Ollama)
- [x] Step 5 — File watcher (watchdog, SMB reconnection, auto re-index on change)
- [x] Step 6 — React chat interface (dark theme, source attribution, modality filter)
- [x] Step 7 — ColPali visual retrieval (batch PDF indexer, MaxSim Qdrant search)
- [x] Step 8 — Image & audio processors (vision caption + Whisper transcription)
- [x] Step 9 — CLIP visual photo search (query-by-photo, `homeintel_visual`)
- [x] Step 10 — Mobile-first UI redesign (markdown answers, swipeable carousel, pill composer + action sheet)
- [x] Full reindex script (bulk ingest existing NAS contents)
- [x] Windows auto-start (Task Scheduler — backend + frontend start at login)
- [ ] CI/CD pipeline

---

## License

MIT — do whatever you want with it.

---

*Built for personal use on real hardware. Not a demo. Actually useful.*
