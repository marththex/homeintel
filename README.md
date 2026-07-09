# HomeIntel

> Point HomeIntel at any local folder of documents, photos, and audio, and query
> it conversationally with a local LLM. Everything runs on your own machine —
> nothing leaves your network.

![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green)
![Qdrant](https://img.shields.io/badge/Qdrant-hybrid_search-orange)
![Ollama](https://img.shields.io/badge/Ollama-local_LLM-purple)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

---

## What is HomeIntel?

HomeIntel indexes a folder you choose — documents, config files, photos, audio
recordings — and lets you query it conversationally using a local LLM. Think of
it as a private, self-hosted "ask my files" assistant.

**Everything runs locally. No cloud APIs. No data leaving your machine.**

```
"What port does the reverse proxy listen on?"
"Find the photos from the trip to the coast."
"What's in my home-automation config?"
"Summarize the notes about the network setup."
```

A NAS mounted over SMB/NFS is a common source folder, but it's just one option —
point `NAS_WATCH_PATH` at any directory on any OS and it works the same way. See
[`docs/deployment-windows.md`](docs/deployment-windows.md) for the NAS/SMB +
Windows-auto-start recipe as an optional advanced setup.

---

## Features

- **Any folder, any OS** — documents (PDF/DOCX/TXT/MD/YAML/JSON), photos, and
  audio, indexed from wherever you point it
- **Hybrid retrieval** — dense (nomic-embed-text) + sparse (BM25) search with
  server-side RRF fusion in Qdrant, plus cross-encoder reranking
- **Live file watching** — new/changed/deleted files are re-indexed automatically
- **Streaming chat** — answers stream token-by-token over SSE with source
  attribution (file name, path, modality, excerpt)
- **Visual photo search** — CLIP-based query-by-photo and query-by-text over your
  image library, independent of caption text
- **Optional image captioning** — a vision-capable Ollama model describes photos
  so they become full-text searchable
- **Optional audio transcription** — Faster-Whisper turns recordings into
  searchable text
- **Optional ColPali visual PDF retrieval** — patch-level page embeddings for
  documents with charts/tables/complex layouts
- **Voice dictation** — record a question with the mic, transcribed locally
- **Secret redaction** — passwords/API keys/tokens/private keys are detected and
  redacted at ingestion, LLM context, and display time
- **Mobile-first, responsive UI** — swipeable photo carousel on phones, a
  multi-column grid + chat-history sidebar on desktop

---

## Architecture

```
Local folder (any OS — a plain directory, or a NAS mounted over SMB/NFS)
        ↓
File Watcher (watchdog) — detects new/changed/deleted files
        ↓
Ingestion Pipeline
  ├── PDFs/Docs/Config → Docling → chunks → nomic-embed-text + BM25
  ├── Images           → Ollama vision caption → chunks → embeddings (optional)
  ├── Audio            → Faster-Whisper transcript → chunks → embeddings (optional)
  ├── PDFs (visual)    → ColPali page embeddings (batch, GPU, optional)
  └── Photos (visual)  → CLIP image embeddings (batch or live, optional)
        ↓
Qdrant (Docker, anywhere reachable on the network) — hybrid dense+sparse +
  ColPali + CLIP collections
        ↓
FastAPI backend — RAG chain + bge-reranker-v2-m3 + /visual-search + /file
        ↓
React chat interface (Vite) — streaming markdown answers, swipeable/grid photo
results, mobile-first composer (camera/library upload, filter + result-count sheet)
```

---

## Stack

| Layer | Technology |
|---|---|
| LLM | [Qwen3](https://ollama.com/library/qwen3) (default: `qwen3:14b`) via [Ollama](https://ollama.com) |
| Embeddings | [nomic-embed-text](https://ollama.com/library/nomic-embed-text) via Ollama |
| Sparse index | BM25 via [fastembed](https://github.com/qdrant/fastembed) |
| Reranker | [bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3) (CrossEncoder) |
| Vector store | [Qdrant](https://qdrant.tech) (Docker, hybrid dense+sparse+ColPali) |
| RAG framework | [LangChain](https://langchain.com) |
| Document parsing | [Docling](https://ds4sd.github.io/docling/) (PDF/DOCX → structured markdown) |
| Visual retrieval (PDF, optional) | [ColPali](https://github.com/illuin-tech/colpali) (`vidore/colpali-v1.2`, batch GPU) |
| Visual similarity (photos, optional) | [CLIP](https://huggingface.co/openai/clip-vit-large-patch14) (query-by-photo/text, `homeintel_visual`) |
| Image captioning (optional) | Ollama vision model (opt-in via `OLLAMA_VISION_MODEL`) |
| Speech-to-text (optional) | [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper) |
| Backend | [FastAPI](https://fastapi.tiangolo.com) |
| Frontend | [React](https://react.dev) + [Vite](https://vitejs.dev) |
| Deployment | Native scripts (`setup.*`/`run.*`) or [Docker Compose](https://docs.docker.com/compose/) |

---

## Prerequisites

- **Python 3.11+**
- **Node.js 20+** (for the frontend)
- **[Ollama](https://ollama.com/download)** installed and running on the host machine
- **Qdrant**, reachable over the network — easiest via Docker
  (`docker compose -f docker/qdrant.yml up -d`), or any Qdrant instance you
  already run elsewhere
- **Docker** (optional — only needed if you run Qdrant, or the whole stack, via
  Docker Compose)
- **NVIDIA GPU** — optional. A GPU with roughly 12+ GB of VRAM is recommended for
  comfortable use of the default LLM; CPU-only works but generation is slower.
  macOS uses CPU/MPS (no CUDA support).

---

## Quick Start

### 1. Clone the repo

```bash
git clone https://github.com/marththex/homeintel.git
cd homeintel
```

### 2. Run the setup script

This creates a Python virtualenv (`.venv`), installs PyTorch (CPU by default),
installs backend + frontend dependencies, and copies `.env.example` → `.env`.

**macOS / Linux:**
```bash
./setup.sh          # CPU-only PyTorch (works everywhere)
./setup.sh --gpu    # NVIDIA CUDA (cu128) build — Linux only for --gpu; macOS has no CUDA
```

**Windows (PowerShell):**
```powershell
.\setup.ps1          # CPU-only PyTorch
.\setup.ps1 -Gpu     # NVIDIA CUDA (cu128) build
```

Safe to re-run — an existing `.venv`/`.env` are left alone.

### 3. Start Qdrant

```bash
docker compose -f docker/qdrant.yml up -d
curl http://localhost:6333/healthz     # expect {"title":"qdrant", ...}
```

Running Qdrant elsewhere (another machine, a NAS)? Point `.env`'s `QDRANT_URL` at
it instead — see `docker/qdrant.yml` for a standalone compose file you can deploy
anywhere reachable over the network.

### 4. Pull the Ollama models

```bash
ollama pull qwen3:14b
ollama pull nomic-embed-text

# Optional — enables image captioning:
ollama pull qwen2.5vl:7b
```

> **Smaller GPU or CPU-only?** `qwen3:14b` needs roughly 10 GB of VRAM. If that
> doesn't fit your machine, pull a smaller chat model instead (e.g. a 7–8B model
> such as `qwen2.5:7b` or `llama3.1:8b`) and set `OLLAMA_LLM_MODEL` in `.env` to
> match. CPU-only inference works too, just slower to generate.

### 5. Configure `.env`

`setup.*` already created `.env` from `.env.example` with sensible defaults —
`NAS_WATCH_PATH` points at the bundled `data/sample-docs/` out of the box, so you
can try the app immediately. When you're ready to index your own files, edit:

```dotenv
NAS_WATCH_PATH=/path/to/your/folder
QDRANT_URL=http://localhost:6333
```

(`NAS_WATCH_PATH` is just "the folder to index" — the name is kept for backward
compatibility; it does not need to be a NAS.)

### 6. Start the app

```bash
./run.sh             # macOS/Linux — starts backend (:8000) + frontend (:5173)
.\run.ps1            # Windows
```

Or start just one side: `./run.sh backend`, `./run.sh frontend` (same flags for
`run.ps1`).

Open **http://localhost:5173** and ask it something about the sample docs, e.g.
*"What companies has Jane Doe worked for?"*

---

## Running Tests

**Unit tests** (fast, pure-Python, no Qdrant/Ollama required):

```bash
pip install -r backend/requirements-dev.txt
python -m pytest backend/tests
```

**Lint:**

```bash
ruff check backend
```

**Integration/manual smoke tests** (need live Qdrant + Ollama; query the bundled
sample docs):

```bash
cd backend
python ../scripts/test_vectorstore.py    # Qdrant connectivity
python ../scripts/test_ingestion.py      # ingest sample docs + query
python ../scripts/verify_api.py          # 8-check API integration test
python ../scripts/verify_streaming.py    # streaming endpoint check
```

---

## Configuration Reference

Every setting lives in `.env` (copied from `.env.example`) and maps 1:1 to a
field in `backend/config.py`. `.env.example` is the source of truth — the table
below mirrors it.

| Variable | Default | Description |
|---|---|---|
| `NAS_WATCH_PATH` | `./data/sample-docs` | Folder to watch and index (any local folder or mounted share — name kept for backward compatibility) |
| `SUPPORTED_EXTENSIONS` | `.pdf,.docx,.txt,.md,.yml,.yaml,.json,.png,.jpg,.jpeg,.mp3,.wav` | Comma-separated extensions to index |
| `WATCHER_EXCLUDE_PATHS` | `` | Comma-separated absolute paths to skip |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant REST API URL |
| `QDRANT_COLLECTION_NAME` | `homeintel` | Collection name |
| `QDRANT_API_KEY` | `` | Leave empty for no-auth setups |
| `EMBED_DIM` | `768` | nomic-embed-text output dimension |
| `RERANKER_ENABLED` | `true` | Cross-encoder reranking after hybrid retrieval |
| `RERANKER_MODEL` | `BAAI/bge-reranker-v2-m3` | Reranker model |
| `IMAGE_RERANK_MIN_SCORE` | `0.3` | Image queries only: drop photos below this reranker relevance (0–1) |
| `DOCLING_VLM_ENABLED` | `false` | Enable Docling VLM picture descriptions during ingestion |
| `DOCLING_VLM_MODEL` | `Qwen/Qwen2.5-VL-7B-Instruct` | Model for Docling VLM enrichment |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server (use `http://host.docker.internal:11434` from inside Docker) |
| `OLLAMA_LLM_MODEL` | `qwen3:14b` | Chat/RAG model — needs ~10 GB VRAM; use a 7–8B model on smaller GPUs |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text` | Embedding model |
| `CHUNK_SIZE` | `512` | Target tokens per chunk |
| `CHUNK_OVERLAP` | `64` | Token overlap between chunks |
| `RETRIEVAL_TOP_K` | `6` | Chunks returned per query (image queries auto-bump to 20; UI can override) |
| `API_HOST` | `0.0.0.0` | FastAPI bind host |
| `API_PORT` | `8000` | FastAPI bind port |
| `LOG_LEVEL` | `info` | Log verbosity |
| `CORS_ALLOW_ORIGINS` | `http://localhost:5173,http://localhost:3000` | Comma-separated allowed frontend origins |
| `OLLAMA_VISION_MODEL` | `` | Vision model for image captioning — empty disables image ingestion entirely |
| `WHISPER_MODEL_SIZE` | `base` | faster-whisper size: `tiny`\|`base`\|`small`\|`medium`\|`large-v3` |
| `COLPALI_ENABLED` | `false` | Enable ColPali retrieval (run the batch indexer first) |
| `COLPALI_MODEL` | `vidore/colpali-v1.2` | ColPali model |
| `CLIP_MODEL` | `openai/clip-vit-large-patch14` | CLIP model for visual photo search |
| `CLIP_AUTO_INDEX` | `false` | Watcher auto-CLIP-indexes new/changed photos (run the bulk backfill first) |
| `CLIP_TEXT_SEARCH` | `true` | Image (text) queries use CLIP text→image search instead of caption matching |
| `CLIP_TEXT_MIN_SCORE` | `0.2` | Minimum CLIP text→image similarity (sims cluster ~0.2–0.26; ranking matters most) |
| `REDACT_SECRETS` | `true` | Redact passwords/keys/tokens at ingestion, LLM context, and excerpts |
| `SKIP_LLM_HEALTH_CHECK` | `false` | Skip the Ollama connectivity check on startup |

---

## Optional Features

### Image captioning

Set `OLLAMA_VISION_MODEL` (e.g. `qwen2.5vl:7b`) to enable captioning of image
files during ingestion — captions become full-text searchable. Leave empty to
skip images entirely (no wasted GPU cycles if you don't have photos to index).

### Audio transcription

Enabled by default via `faster-whisper` (`WHISPER_MODEL_SIZE=base`). Runs on GPU
if available, otherwise CPU. Increase the model size for better accuracy at the
cost of speed.

### Voice dictation

The composer's mic button records audio in the browser and transcribes it
locally via the same Whisper model (`POST /transcribe`), dropping the result into
the input box for review before sending. Microphone capture requires a secure
context (HTTPS or `localhost`) — see
[`docs/tailscale-https-setup.md`](docs/tailscale-https-setup.md) for enabling it
from a phone on your LAN.

### CLIP visual photo search

Query your photo library **by photo** (take/upload a picture) or **by text**
("photos of the dog at the beach") — CLIP encodes images and text into a shared
embedding space so results reflect visual similarity, not just caption keyword
overlap. No extra dependencies (`transformers` + `torch` already installed).

Build the index:

```bash
cd backend
python ../scripts/index_visual.py --path /path/to/photos
python ../scripts/index_visual.py --skip-existing    # resume an interrupted run
python ../scripts/index_visual.py --batch-size 32    # bigger GPU batches
```

Then set `CLIP_AUTO_INDEX=true` so the watcher keeps the visual index current
without re-running the script.

### ColPali visual PDF retrieval

ColPali embeds PDF pages as patch-level multi-vectors for visual retrieval —
useful for documents with charts, tables, or complex layouts. Off by default,
and not installed by default (see the commented-out section in
`backend/requirements.txt`).

```bash
pip install colpali-engine>=0.3.0 pypdfium2>=4.0.0
```

> **Warning:** installing `colpali-engine` pulls in its own CPU-only PyTorch and
> **will overwrite** a CUDA install. Reinstall the GPU build afterward:
> ```bash
> pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu128
> ```
> Verify with `python -c "import torch; print(torch.cuda.is_available())"` → must print `True`.

Run the batch indexer (stop Ollama first to free VRAM — ColPali needs ~8 GB),
then enable retrieval:

```bash
cd backend
python ../scripts/run_colpali.py
```

```dotenv
COLPALI_ENABLED=true
```

ColPali OOMs alongside a 14B-class LLM on a 16 GB GPU at query time — the
indexed page vectors stay in Qdrant and cost no VRAM at rest; you only need the
model loaded to add new PDFs, so most setups keep `COLPALI_ENABLED=false` for
day-to-day querying.

### Faster ingestion (captions + documents)

```bash
# Let Ollama serve concurrent requests, then restart the Ollama service:
setx OLLAMA_NUM_PARALLEL 3     # Windows; use an env var/service restart on other OSes

python ../scripts/reindex.py --path /path/to/photos --ext .jpg .jpeg .png --workers 3
python ../scripts/reindex.py --skip-existing --workers 3    # resume where you left off
```

`--workers N` issues N captions concurrently (network-bound, so the GPU stays
busy); pair it with `OLLAMA_NUM_PARALLEL=N`. Drop workers if the vision model OOMs,
or switch to a smaller vision model to roughly halve caption time.

---

## Docker Alternative

Prefer containers? `docker-compose.yml` (production) and `docker-compose.dev.yml`
(hot-reload overrides) are provided:

```bash
docker compose up -d                                                # prod
docker compose -f docker-compose.yml -f docker-compose.dev.yml up   # dev
```

Ollama always runs on the host, not in Docker — the backend container reaches it
via `host.docker.internal:11434`.

---

## Web UI Features

- **Streaming markdown answers** — tokens render live; bold, lists, and code render properly
- **Photo results** — swipeable carousel on mobile, multi-column grid on desktop;
  tap **+** to search by photo (camera or library) or by text
- **Mobile-first composer** — single rounded input pill (`+` · text · mic/send); the
  `+` sheet holds visual search, modality filter (All/Documents/Images/Audio), and a
  "Results to show" stepper
- **Chat history** — multiple conversations, sidebar on desktop / slide-in drawer on
  mobile, persisted in `localStorage`
- **Status sheet** — tap the health dot in the header for Ollama/Qdrant status +
  chunk counts
- **Keyboard shortcuts** (desktop) — Ctrl/Cmd+K focuses the composer, Esc closes
  sheets/lightbox, arrow keys navigate the lightbox

---

## Project Structure

```
homeintel/
├── setup.sh / setup.ps1            # Cross-platform install (venv, deps, .env bootstrap)
├── run.sh / run.ps1                # Start backend/frontend/both
├── docker-compose.yml              # Production stack (alternative to setup/run)
├── docker-compose.dev.yml          # Dev overrides (hot-reload)
├── docker/
│   └── qdrant.yml                  # Standalone Qdrant compose
├── .env.example                    # Config template (source of truth)
├── data/
│   └── sample-docs/                # Bundled synthetic docs — app works out-of-the-box
│
├── backend/
│   ├── main.py                     # FastAPI entrypoint + watcher start
│   ├── config.py                   # Settings (pydantic-settings)
│   ├── requirements.txt
│   ├── requirements-dev.txt        # pytest, ruff — contributors/CI only
│   ├── Dockerfile
│   │
│   ├── api/
│   │   ├── chat.py                 # POST /chat, POST /chat/stream
│   │   ├── status.py               # GET /health, GET /stats
│   │   ├── files.py                # GET /file, GET /thumb, POST /visual-search
│   │   └── transcribe.py           # POST /transcribe (voice dictation)
│   │
│   ├── ingestion/
│   │   ├── watcher.py              # watchdog-based folder watcher (network-share reconnect)
│   │   ├── pipeline.py             # routes files to processors
│   │   ├── chunker.py              # shared LangChain text splitter
│   │   └── processors/
│   │       ├── document.py         # PDF/DOCX (Docling), TXT/MD/YAML/JSON
│   │       ├── image.py            # Ollama vision caption
│   │       ├── audio.py            # Faster-Whisper transcription
│   │       └── colpali.py          # ColPali page embeddings (batch, optional)
│   │
│   ├── rag/
│   │   ├── retriever.py            # hybrid + ColPali merge + reranker + adaptive top-k
│   │   ├── chain.py                # LangChain RAG chain (blocking + streaming)
│   │   └── prompts.py              # system prompt templates
│   │
│   ├── vectorstore/
│   │   ├── qdrant.py               # Qdrant wrapper (dense+sparse+ColPali)
│   │   └── clip.py                 # CLIP visual store (homeintel_visual)
│   │
│   ├── security/
│   │   └── redact.py               # secret detection/redaction
│   │
│   ├── models/
│   │   └── chat.py                 # ChatRequest / ChatResponse / SourceDoc
│   │
│   └── tests/                      # pytest unit suite (no external services)
│
├── frontend/                       # React + Vite + TypeScript
│   └── src/
│       ├── App.tsx                 # layout, composer, action + status sheets
│       ├── api.ts                  # fetch wrappers (chat, visual-search, file, transcribe)
│       ├── types.ts
│       ├── hooks/                  # useSystemStatus, useRecorder, useMediaQuery
│       └── components/             # Header, ChatMessage, PhotoCarousel, PhotoGrid,
│                                    #   Sidebar, Drawer, BottomSheet, icons, ...
│
└── scripts/
    ├── test_vectorstore.py         # Qdrant smoke test
    ├── test_ingestion.py           # ingest + query smoke test
    ├── verify_api.py               # 8-check API integration test
    ├── verify_streaming.py         # streaming endpoint check
    ├── reindex.py                  # full/partial reindex (resume, concurrency)
    ├── run_colpali.py              # batch ColPali PDF indexer (optional)
    └── index_visual.py             # batch CLIP photo indexer
```

---

## What Gets Indexed

| File type | How it's indexed |
|---|---|
| `.pdf`, `.docx` | Docling → structured markdown → chunks → embeddings |
| `.txt`, `.md`, `.yml`, `.yaml`, `.json` | Plain text → chunks → embeddings |
| `.png`, `.jpg`, `.jpeg`, `.gif`, `.webp` | Ollama vision caption → chunks → embeddings (requires `OLLAMA_VISION_MODEL`) |
| `.mp3`, `.wav`, `.m4a`, `.flac`, `.ogg` | Faster-Whisper transcript → chunks → embeddings |
| PDFs (visual, batch, optional) | ColPali page multi-vectors → `homeintel_colpali` collection |
| Photos (visual, batch/live, optional) | CLIP image embeddings → `homeintel_visual` collection (query-by-photo/text) |

**Video files are intentionally unsupported** — transcribing a large personal
video library with Faster-Whisper isn't practical for a real-time watcher.

When pointing HomeIntel at a shared home-server root that also hosts other
services, use `WATCHER_EXCLUDE_PATHS` to skip anything that isn't user files:
other apps' databases/caches, Docker's own storage, credential stores, and
`node_modules`-heavy project directories are common candidates.

---

## Privacy & Security

- No data ever leaves your machine/network
- No cloud API calls — all models run locally via Ollama or on-device (Whisper,
  reranker, ColPali, CLIP)
- Source files are read-only (never modified)
- **Secret redaction** (`REDACT_SECRETS=true`, default on): passwords, API keys,
  tokens, and private keys are detected and replaced with `<REDACTED>` at three
  layers — ingestion (kept out of Qdrant), LLM context (the model never sees the
  raw value), and source excerpts (UI never shows it). The system prompt also
  instructs the LLM to refuse to reveal credentials. See `backend/security/redact.py`.
  Secrets indexed before redaction was enabled are still scrubbed at response
  time; run `reindex.py` to also purge them from Qdrant storage.
- See [`SECURITY.md`](SECURITY.md) for how to report a vulnerability.

---

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for dev setup, running tests/lint, and
PR conventions. Please also read the [Code of Conduct](CODE_OF_CONDUCT.md).

---

## Roadmap

**Done:**
- Qdrant hybrid (dense+sparse) vector store with RRF fusion
- Document ingestion (Docling for PDF/DOCX; plain read for TXT/MD/YAML/JSON)
- FastAPI backend + streaming RAG chain (LangChain + Ollama)
- File watcher with auto re-index and network-share reconnection
- React chat interface — mobile-first, streaming, desktop-responsive
- ColPali visual PDF retrieval (optional) and CLIP visual/text photo search (optional)
- Image captioning + audio transcription (optional)
- Voice dictation, secret redaction, chat history, keyboard shortcuts
- Cross-platform install/run scripts, bundled sample data, pytest unit suite
- GitHub Actions CI — unit tests (Windows/macOS/Linux) + frontend build

**Planned:**
- Additional optional ingestion sources

---

## License

MIT — do whatever you want with it. See [`LICENSE`](LICENSE).
