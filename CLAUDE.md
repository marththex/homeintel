# CLAUDE.md — HomeIntel Project Context

> This file is the source of truth for AI assistants (Claude Code, etc.) working
> on this codebase. Read this before touching anything.

---

## What This Project Is

**HomeIntel** is a fully local AI assistant for a personal NAS. It indexes files
(documents, config files, photos, audio) and lets the user query them
conversationally via a local LLM. Nothing leaves the local network.

This is a real deployment on real personal infrastructure — not a demo or notebook.

---

## Hardware & Deployment Context

| Component | Location | Details |
|---|---|---|
| PC | Windows 11 | NVIDIA RTX 5080, runs Ollama + FastAPI + React |
| NAS | TrueNAS | SMB share mounted as `Z:` on PC |
| LLM | PC (GPU) | qwen3.5:9b via Ollama at `http://localhost:11434` |
| Embeddings | PC (GPU) | nomic-embed-text via Ollama |
| Vector Store | NAS | ChromaDB persisted on the NAS mount path |
| Source Files | NAS | Watched at the NAS root path |

**Critical:** Ollama runs on the PC host, NOT in Docker. The backend connects to it
via `http://localhost:11434` (or `http://host.docker.internal:11434` from inside
Docker containers).

---

## Environment

```bash
# Conda env name
homeintel

# Python version
3.11

# Activate
conda activate homeintel

# Working directory for backend scripts
cd backend

# Run scripts from backend/ dir
python ../scripts/test_vectorstore.py
```

**PyTorch:** Installed with CUDA 12.8 build (RTX 5080 uses CUDA 13.2 driver but
cu128 is backward compatible):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

---

## .env Location & Key Values

File lives at repo root: `<repo>/.env`

```dotenv
NAS_WATCH_PATH=/path/to/your/nas
SUPPORTED_EXTENSIONS=.pdf,.docx,.txt,.md,.yml,.yaml,.json,.png,.jpg,.jpeg,.mp3,.wav
WATCHER_EXCLUDE_PATHS=<NAS_ROOT>/homeintel/chroma,<NAS_ROOT>/media,<NAS_ROOT>/plex_config,<NAS_ROOT>/roms,<NAS_ROOT>/ai_videos,<NAS_ROOT>/wedding_raw_videos,<NAS_ROOT>/palworld,<NAS_ROOT>/qbittorrent
CHROMA_PATH=<NAS_ROOT>/homeintel/chroma
CHROMA_COLLECTION_NAME=homeintel
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_LLM_MODEL=qwen3.5:9b
OLLAMA_EMBED_MODEL=nomic-embed-text
CHUNK_SIZE=512
CHUNK_OVERLAP=64
RETRIEVAL_TOP_K=6
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=info
SKIP_LLM_HEALTH_CHECK=false
```

**Important quirk:** `SUPPORTED_EXTENSIONS` and `WATCHER_EXCLUDE_PATHS` are plain
comma-separated strings (NOT JSON arrays) because pydantic-settings v2 tries to
JSON-parse `List[str]` fields before validators run. They are stored as `str` in
`config.py` and exposed as lists via `@property` methods:
- `settings.supported_extensions_list`
- `settings.watcher_exclude_paths_list`

---

## Config Loading Quirk

`backend/config.py` uses an absolute path to find `.env`:

```python
env_file=Path(__file__).parent.parent / ".env"
```

This means scripts can be run from any directory and will always find the `.env`
at the repo root. Do not change this to a relative path.

---

## NAS Contents (NAS root)

Notable directories on the NAS:

```
<NAS_ROOT>/
├── homeintel/          ← HomeIntel data (chroma/, etc.)
├── docker/             ← Docker compose files for NAS services
├── homeassistant/      ← Home Assistant config
├── nextcloud/          ← Nextcloud data
├── nginx/              ← Nginx config
├── cloudflared/        ← Cloudflare tunnel config
├── Wedding_Contracts/  ← PDF documents
├── Marcus_Resume.pdf   ← PDF at root
├── music/              ← Audio files
├── media/              ← EXCLUDED (movies/TV - too large to index)
├── roms/               ← EXCLUDED (game ROMs)
├── plex_config/        ← EXCLUDED (Plex binary data)
├── ai_videos/          ← EXCLUDED
├── wedding_raw_videos/ ← EXCLUDED
├── palworld/           ← EXCLUDED (game server data)
└── qbittorrent/        ← EXCLUDED (torrent client data)
```

Video files (.mp4, .mov) are intentionally excluded from indexing — the NAS has
52 movies, 15 TV shows, and a large One Pace collection that would take too long
to transcribe.

---

## Build Order & Current Progress

### ✅ Step 1 — Scaffolding (COMPLETE)
**Files created:**
- `.env.example` — config template, committed to git
- `.env` — actual config, gitignored
- `.gitignore`
- `docker-compose.yml` — production stack
- `docker-compose.dev.yml` — dev overrides (hot-reload)
- `backend/config.py` — pydantic-settings singleton
- `backend/requirements.txt` — all Python dependencies
- `backend/Dockerfile` — multi-stage (dev + prod targets)
- `README.md`
- `CLAUDE.md` (this file)

### ✅ Step 2 — ChromaDB Wrapper (COMPLETE)
**Files created:**
- `backend/vectorstore/chroma.py` — VectorStore class
- `backend/vectorstore/__init__.py`
- `scripts/test_vectorstore.py` — smoke test (all 7 tests passing)

**Verified working:**
- ChromaDB writes/reads to the NAS chroma path over SMB ✓
- Ollama nomic-embed-text embeddings working ✓
- Upsert, semantic query, modality filtering, delete all working ✓

### ✅ Step 3 — Document Ingestion Processor (COMPLETE)
**Target files:**
- `backend/ingestion/processors/document.py`
- `backend/ingestion/chunker.py`
- `backend/ingestion/pipeline.py`
- `backend/ingestion/__init__.py`
- `scripts/test_ingestion.py`

**What it should do:**
- Accept a file path
- Detect file type by extension
- Route to correct parser (Unstructured for PDF/DOCX, plain read for TXT/MD,
  json/yaml parse for config files)
- Chunk text using LangChain text splitters
- Generate embeddings via nomic-embed-text
- Upsert into ChromaDB with full metadata

**Metadata schema per chunk:**
```python
{
    "file_path": str,       # absolute path on NAS
    "file_name": str,       # basename
    "file_ext": str,        # lowercase extension
    "chunk_index": int,     # position within file
    "modality": str,        # "document" | "image" | "audio"
    "title": str,           # optional, document title if available
    "created_at": str,      # ISO timestamp of indexing
}
```

**Notes:**
- Document processor working: PDF (pypdf fallback), DOCX, TXT, MD, YAML, JSON all ingesting correctly
- `Z:/docker` must stay in `WATCHER_EXCLUDE_PATHS` because it is Docker Engine overlay2 storage, not user files
- Actual docker-compose files live one level deep in service folders, e.g. `Z:/marcus_photoprism/docker-compose.yml`
- Unstructured PDF fails with `PSSyntaxError` on `pdfminer.six` 20260107; pypdf fallback works correctly
- `pdfminer.six` must stay pinned to `20251230` due to `pdfplumber` dependency conflict
- `delete_file()` is called before every upsert to prevent duplicates

### ✅ Step 4 — FastAPI Backend + RAG Chain (COMPLETE)
**Files created:**
- `backend/main.py` — FastAPI entrypoint with CORS + lifespan startup check
- `backend/api/__init__.py`
- `backend/api/chat.py` — POST /chat
- `backend/api/status.py` — GET /health, GET /stats
- `backend/rag/__init__.py`
- `backend/rag/prompts.py` — SYSTEM_TEMPLATE (answer-from-context-only)
- `backend/rag/retriever.py` — VectorStore.query wrapper with modality filtering
- `backend/rag/chain.py` — RAGChain class + module-level singleton
- `backend/models/__init__.py`
- `backend/models/chat.py` — ChatRequest, ChatResponse, SourceDoc schemas

**Verified working (scripts/verify_api.py — 8/8 checks pass):**
- GET /health → `{"status":"ok","ollama":true,"chromadb":true}` ✓
- GET /stats → per-modality chunk counts ✓
- POST /chat (resume question) → correct answer with source attribution ✓
- POST /chat (photoprism port) → answer pulls from docker-compose.yml source ✓
- POST /chat (weather — not in data) → correct "don't have information" refusal ✓
- POST /chat (modality_filter=document) → all sources are document modality ✓
- POST /chat (empty question) → HTTP 422 ✓
- POST /chat (no body) → HTTP 422 ✓

**Notes:**
- CORS enabled for `http://localhost:3000` and `http://localhost:5173`
- Ollama URL comes from `OLLAMA_BASE_URL` env var — use `http://host.docker.internal:11434` in Docker compose
- RAGChain and VectorStore are module-level singletons (lazy init on first request)
- Startup health check is controlled by `SKIP_LLM_HEALTH_CHECK=true` (useful for running without Ollama)
- Sources include file_name, file_path, modality, and 200-char excerpt per chunk
- `conda run -n homeintel` does not support multiline `-c` scripts; use a `.py` file instead
- `test_ingestion.py` now accepts `--keep-chunks` to skip cleanup (for API testing)
- `scripts/verify_api.py` starts uvicorn as a subprocess, runs 8 checks, shuts it down; exit 0 = all pass
- Degraded-state test (Ollama down) must be run manually: stop Ollama and check `/health` returns `{"status":"degraded"}`

### ⬜ Step 5 — File Watcher (TODO)
**Target files:**
- `backend/ingestion/watcher.py`

**What it should do:**
- Watch `NAS_WATCH_PATH` recursively using watchdog
- Respect `WATCHER_EXCLUDE_PATHS` (especially the NAS chroma path)
- On file created/modified: delete old chunks → re-ingest
- On file deleted: delete chunks from ChromaDB
- Filter by `SUPPORTED_EXTENSIONS`
- Handle SMB disconnection gracefully (retry logic)

### ⬜ Step 6 — React Frontend (TODO)
**Target files:**
- `frontend/` — full React + Vite app

**What it should do:**
- Chat interface with message history
- Show source files used for each answer (with file path + modality icon)
- Indexing status bar (chunk counts by modality)
- Responsive, clean UI

### ⬜ Step 7 — Image & Audio Processors (TODO)
**Target files:**
- `backend/ingestion/processors/image.py`
- `backend/ingestion/processors/audio.py`

**What it should do:**
- Images: OpenCLIP embeddings + Qwen3.5 vision captions
- Audio: Faster-Whisper transcription → text chunks → embeddings
- Both use GPU via CUDA

---

## Key Design Decisions & Rationale

**Why ChromaDB over Qdrant?**
Single-host deployment, no separate service needed, embedded mode writes directly
to NAS path, LangChain integration is first-class. Qdrant's advantages (distributed,
high-throughput) are irrelevant for personal use.

**Why one collection for all modalities?**
User wants cross-modal search — a single query should search docs, photos, and
audio transcripts simultaneously. Modality is stored as metadata for filtering
when needed.

**Why nomic-embed-text instead of Qwen3.5 for embeddings?**
Speed. nomic-embed-text is a dedicated embedding model — much faster than using
the LLM for embeddings. Frees Qwen3.5 for generation only.

**Why SMB over NFS for NAS mount?**
NFS `mount.exe` was missing from Windows NFS client install (only `nfsadmin.exe`
present). SMB works natively on Windows and is simpler. The NAS share `NFSdocker`
is configured as SMB (Windows share) on TrueNAS.

**Why keep ChromaDB on NAS instead of local PC?**
Survives PC reinstall/upgrade, accessible from multiple machines later. SMB
latency is acceptable for personal conversational use (1 query at a time).

**Why exclude video files?**
52 movies + 15 TV shows + large One Pace collection would take days to transcribe
with Faster-Whisper even on RTX 5080. Can be added later as a separate phase.

**List fields in config.py are plain strings:**
pydantic-settings v2 tries to JSON-parse `List[str]` fields before validators run,
causing `JSONDecodeError`. Workaround: store as `str`, expose as list via
`@property`.

---

## Common Commands

```bash
# Activate environment
conda activate homeintel

# Run vector store smoke test
cd backend
python ../scripts/test_vectorstore.py

# Start Ollama (usually already running as Windows service)
ollama serve

# Check running models
ollama list

# Pull a model
ollama pull qwen3.5:9b
ollama pull nomic-embed-text

# Start full Docker stack
cd <repo>
docker compose up -d

# Dev mode (hot-reload)
docker compose -f docker-compose.yml -f docker-compose.dev.yml up
```

---

## Known Issues / Gotchas

1. **ChromaDB telemetry warnings** — `capture() takes 1 positional argument but 3 were given`
   is a known ChromaDB bug with newer versions. Harmless, ignore it.

2. **Ollama on Windows starts automatically** — the installer registers Ollama as a
   Windows service. Running `ollama serve` manually will fail with "address already
   in use". Just use `ollama pull` and `ollama list` directly.

3. **VSCode terminal PATH** — after installing Ollama, VSCode terminals need a full
   restart to pick up the updated PATH. Reloading the window is not enough.

4. **PyTorch CUDA** — RTX 5080 uses CUDA driver 13.2 but PyTorch only supports up
   to cu128. Use `--index-url https://download.pytorch.org/whl/cu128`. The cu128
   build is backward compatible with CUDA 13.x drivers.

5. **NFS mount.exe missing** — Windows NFS feature `ServicesForNFS-ClientOnly`
   installs without `mount.exe`. Need to also enable `ClientForNFS-Infrastructure`.
   Ultimately went with SMB instead.

6. **SMB mount persistence** — `net use Z: \\YOUR_NAS_HOST\NFSdocker /persistent:yes`
   persists across reboots. Replace `YOUR_NAS_HOST` with your TrueNAS IP or hostname.
