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
| Vector Store | NAS | Qdrant (Docker container on NAS), REST API over LAN |
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
WATCHER_EXCLUDE_PATHS=<NAS_ROOT>/homeintel/qdrant_storage,<NAS_ROOT>/media,<NAS_ROOT>/plex_config,<NAS_ROOT>/roms,<NAS_ROOT>/ai_videos,<NAS_ROOT>/wedding_raw_videos,<NAS_ROOT>/palworld,<NAS_ROOT>/qbittorrent
QDRANT_URL=http://<NAS_IP>:6333
QDRANT_COLLECTION_NAME=homeintel
QDRANT_API_KEY=
EMBED_DIM=768
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_LLM_MODEL=qwen3.5:9b
OLLAMA_EMBED_MODEL=nomic-embed-text
CHUNK_SIZE=512
CHUNK_OVERLAP=64
RETRIEVAL_TOP_K=6
RERANKER_ENABLED=true
RERANKER_MODEL=BAAI/bge-reranker-v2-m3
DOCLING_VLM_ENABLED=false
DOCLING_VLM_MODEL=Qwen/Qwen2.5-VL-7B-Instruct
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
├── homeintel/          ← HomeIntel data (qdrant_storage/, etc.)
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

### ✅ Step 2 — Vector Store (COMPLETE — migrated ChromaDB → Qdrant)
**Original ChromaDB wrapper** (`backend/vectorstore/chroma.py`) was completed and
verified working (SMB write, embeddings, upsert/query/delete). It has been
**superseded** by the Qdrant wrapper but is preserved in git history for traceability.

**Current files:**
- `backend/vectorstore/qdrant.py` — VectorStore class (Qdrant, hybrid dense+sparse)
- `backend/vectorstore/__init__.py`
- `docker/qdrant.yml` — NAS-side Qdrant Docker Compose
- `scripts/test_vectorestore.py` — smoke test (deploy Qdrant on NAS first)

**What it does:**
- Connects to Qdrant over REST (`QDRANT_URL`) — no filesystem mount needed
- Named vectors: `dense` (nomic-embed-text, 768-dim) + `sparse` (BM25 via fastembed)
- Hybrid query: dense + sparse prefetch → server-side RRF fusion
- Same interface as old wrapper (upsert, query, delete_file, stats) — no downstream changes

### ✅ Step 3 — Document Ingestion Processor (COMPLETE)
**Target files:**
- `backend/ingestion/processors/document.py`
- `backend/ingestion/chunker.py`
- `backend/ingestion/pipeline.py`
- `backend/ingestion/__init__.py`
- `scripts/test_ingestion.py`

**What it does:**
- Accept a file path
- Detect file type by extension
- Route to correct parser (Docling for PDF/DOCX, plain read for TXT/MD,
  json/yaml parse for config files)
- Chunk Docling's markdown output using LangChain text splitters
- Generate dense (nomic-embed-text) + sparse (BM25) embeddings
- Upsert both vectors into Qdrant with full metadata

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
- Docling replaces Unstructured for PDF/DOCX — no pypdf fallback needed; Docling uses its own PDF pipeline
- Docling outputs structured markdown; chunk boundaries are better than Unstructured's flat text because headings/tables are preserved
- Optional VLM picture description (`DOCLING_VLM_ENABLED=true`) uses Qwen2.5-VL-7B-Instruct via Ollama's OpenAI-compat endpoint — disabled by default; enable only if VRAM headroom exists alongside qwen3.5:9b + nomic-embed-text
- `Z:/docker` must stay in `WATCHER_EXCLUDE_PATHS` because it is Docker Engine overlay2 storage, not user files
- Actual docker-compose files live one level deep in service folders, e.g. `Z:/marcus_photoprism/docker-compose.yml`
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
- GET /health → `{"status":"ok","ollama":true,"qdrant":true}` ✓
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
- Retriever now fetches 3×top_k candidates from Qdrant, then reranks with `bge-reranker-v2-m3` before truncating to `RETRIEVAL_TOP_K` — reranker is lazy-loaded on first query (first call is slow while model downloads)
- `RERANKER_ENABLED=false` disables reranking and falls back to raw RRF order (useful for latency testing)

### ⬜ Step 5 — File Watcher (TODO)
**Target files:**
- `backend/ingestion/watcher.py`

**What it should do:**
- Watch `NAS_WATCH_PATH` recursively using watchdog
- Respect `WATCHER_EXCLUDE_PATHS` (especially `<NAS_ROOT>/homeintel/qdrant_storage`)
- On file created/modified: delete old chunks → re-ingest
- On file deleted: delete chunks from Qdrant
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

### ⬜ Step 7 — ColPali Visual Retrieval (TODO)
**Target files:**
- `backend/ingestion/processors/colpali.py` — renders PDF pages to images, generates ColPali embeddings per page

**What it should do:**
- Route by document type: run ColPali only on visually-dense documents (v1 heuristic: PDFs with detected images/charts from Docling metadata, or a manual per-directory allowlist)
- Render each qualifying PDF page to an image (e.g. via `pdf2image`)
- Generate ColPali multi-vector embeddings per page using the `vidore/colpali-v1.2` model
- Store page-image embeddings as a separate named vector field (`"colpali"`) in the same Qdrant collection, or in a sibling collection if co-location proves awkward — decide during implementation
- Retrieval: ColPali visual results merged with hybrid text results before reranking
- Opt-in per document type — not applied to the full corpus by default

**Resource notes:**
- ColPali indexing is GPU-accelerated but slow for large PDFs (~5s/page on RTX 5080)
- Storage growth: ~1 MB per page for ColPali embeddings
- Run as a separate batch process rather than inline with the file watcher

### ⬜ Step 8 — Image & Audio Processors (TODO)
**Target files:**
- `backend/ingestion/processors/image.py`
- `backend/ingestion/processors/audio.py`

**What it should do:**
- Images: OpenCLIP embeddings + Qwen3.5 vision captions
- Audio: Faster-Whisper transcription → text chunks → embeddings
- Both use GPU via CUDA

---

## Key Design Decisions & Rationale

**Why Qdrant over ChromaDB?**
ChromaDB was the original choice (embedded mode, direct NAS write). Migrated to
Qdrant for: (1) native sparse vector support for hybrid BM25+dense search without
extra libraries; (2) named multi-vector fields needed for ColPali co-location
(Step 7); (3) single-binary Docker deploy on NAS — lighter than Milvus, simpler
ops than Weaviate; (4) server-side RRF fusion means no client-side score
normalization hacks. SMB latency to Qdrant data was the limiting factor with
ChromaDB; switching to a LAN REST call actually reduces write latency.

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

**Why keep Qdrant on NAS instead of local PC?**
Survives PC reinstall/upgrade, accessible from multiple machines later. LAN
latency to Qdrant REST API is low (sub-ms on gigabit). Qdrant storage (vectors +
payload) on NAS SSD is preferred over PC to keep GPU memory free for inference.

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

1. **Qdrant must be running before the backend starts** — unlike the old ChromaDB
   embedded mode, Qdrant is a separate network service. If the backend starts before
   Qdrant is up, VectorStore init will fail. Deploy `docker/qdrant.yml` on the NAS
   and confirm `curl http://<NAS_IP>:6333/healthz` returns `{"title":"qdrant"}` before
   starting the API.

2. **fastembed BM25 model download on first run** — `SparseTextEmbedding("Qdrant/bm25")`
   downloads the model (~10 MB) to `~/.cache/fastembed` on first use. Subsequent starts
   use the cache. No internet access needed after first download.

3. **bge-reranker-v2-m3 download on first query** — the CrossEncoder model (~1.1 GB)
   downloads from HuggingFace on the first query if `RERANKER_ENABLED=true`. First query
   will be slow. Set `RERANKER_ENABLED=false` during development to skip this.

4. **Docling first-run model download** — Docling downloads layout and OCR models
   (~500 MB) on first use to `~/.cache/docling`. Subsequent runs use the cache.

5. **Docling VLM VRAM** — `DOCLING_VLM_ENABLED=true` loads Qwen2.5-VL-7B-Instruct
   alongside qwen3.5:9b + nomic-embed-text. On RTX 5080 (16 GB VRAM), this may OOM
   if all models are loaded simultaneously. Fallback: use `Qwen/Qwen2.5-VL-3B-Instruct`
   or disable VLM enrichment and rely on Docling's text extraction only.

7. **Ollama on Windows starts automatically** — the installer registers Ollama as a
   Windows service. Running `ollama serve` manually will fail with "address already
   in use". Just use `ollama pull` and `ollama list` directly.

8. **VSCode terminal PATH** — after installing Ollama, VSCode terminals need a full
   restart to pick up the updated PATH. Reloading the window is not enough.

9. **PyTorch CUDA** — RTX 5080 uses CUDA driver 13.2 but PyTorch only supports up
   to cu128. Use `--index-url https://download.pytorch.org/whl/cu128`. The cu128
   build is backward compatible with CUDA 13.x drivers.

10. **NFS mount.exe missing** — Windows NFS feature `ServicesForNFS-ClientOnly`
    installs without `mount.exe`. Need to also enable `ClientForNFS-Infrastructure`.
    Ultimately went with SMB instead.

11. **SMB mount persistence** — `net use Z: \\YOUR_NAS_HOST\NFSdocker /persistent:yes`
    persists across reboots. Replace `YOUR_NAS_HOST` with your TrueNAS IP or hostname.
