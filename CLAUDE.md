# CLAUDE.md — HomeIntel Project Context

> This file is the source of truth for AI assistants (Claude Code, etc.) working
> on this codebase. Read this before touching anything.
>
> This is the **generalized, committed** version of this file — safe for a public
> repo (no real hostnames/IPs/paths/hardware). If you're the maintainer working on
> your own real deployment, keep your personal notes in `CLAUDE.local.md`
> (gitignored) instead of re-adding real infra details here.

---

## What This Project Is

**HomeIntel** is a fully local AI assistant you point at any folder of files —
documents, config files, photos, audio — and query conversationally via a local
LLM. Nothing leaves your machine/network: no cloud APIs, no telemetry.

A NAS (network-attached storage) mounted over SMB/NFS is one common way to supply
that folder, but it is **not required** — a plain local directory works exactly
the same way. See `docs/deployment-windows.md` for the NAS/SMB-mount + Windows
auto-start recipe as an optional advanced deployment.

This project started as a real deployment on real personal infrastructure (not a
demo/notebook) and was later generalized for open-source use — the engineering
history below (Build Order, gotchas, design decisions) reflects that.

---

## Deployment Context (example shape)

HomeIntel doesn't require any particular topology — see the README Quick Start
for the minimal path (one machine, a local folder, Docker Qdrant, `setup.*` +
`run.*`). The table below describes one realistic **advanced** deployment (the
shape this project was originally built and tested against), useful if you're
doing something similar (e.g. indexing a NAS from a separate PC):

| Component | Location | Details |
|---|---|---|
| App host | Any OS | Runs Ollama + FastAPI + React (can also be split across machines) |
| Source files | NAS or local disk | Mounted over SMB/NFS, or just a local folder |
| LLM | App host (GPU) | e.g. `qwen3:14b` via Ollama at `http://localhost:11434` |
| Embeddings | App host (GPU) | `nomic-embed-text` via Ollama |
| Vector store | Anywhere on the LAN | Qdrant (Docker) — can co-locate with the app or run on a separate host/NAS |
| Watched path | `NAS_WATCH_PATH` | Any local folder, or a mounted network share |

**Critical:** Ollama runs on the host, NOT in Docker. The backend connects to it
via `http://localhost:11434` (or `http://host.docker.internal:11434` from inside
Docker containers).

---

## Environment

The cross-platform `setup.sh` (macOS/Linux) / `setup.ps1` (Windows) scripts
create a Python virtualenv at `.venv` and install everything — see the README
Quick Start. If you manage the environment yourself instead (e.g. conda), the
underlying requirements are the same.

```bash
# Python version
3.11+

# Activate the venv created by setup.sh / setup.ps1
source .venv/bin/activate        # macOS/Linux
.venv\Scripts\Activate.ps1       # Windows PowerShell

# Working directory for backend scripts
cd backend

# Run scripts from backend/ dir
python ../scripts/test_vectorstore.py
```

**PyTorch:** installed by `setup.sh`/`setup.ps1` — CPU build by default, or pass
`--gpu` (`-Gpu` on Windows) for the CUDA cu128 build (NVIDIA GPU on Windows/Linux
only; macOS always uses the CPU/MPS build — there is no CUDA support on macOS).
If installing manually instead of via the setup script:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

---

## .env Location & Key Values

File lives at repo root: `<REPO_PATH>/.env` — created automatically by
`setup.sh`/`setup.ps1` from `.env.example` if missing. **`.env.example` is the
source of truth** for every setting (with descriptions); `backend/config.py` is
the source of truth for types/defaults. Representative excerpt:

```dotenv
NAS_WATCH_PATH=./data/sample-docs     # "the folder to index" — any local folder or NAS mount; defaults to bundled sample docs
SUPPORTED_EXTENSIONS=.pdf,.docx,.txt,.md,.yml,.yaml,.json,.png,.jpg,.jpeg,.mp3,.wav
WATCHER_EXCLUDE_PATHS=                # comma-separated absolute paths to skip
QDRANT_URL=http://localhost:6333      # or http://<QDRANT_HOST>:6333 for a remote/NAS Qdrant
QDRANT_COLLECTION_NAME=homeintel
QDRANT_API_KEY=
EMBED_DIM=768
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_LLM_MODEL=qwen3:14b            # ~10 GB VRAM — see below for smaller-GPU guidance
OLLAMA_EMBED_MODEL=nomic-embed-text
OLLAMA_VISION_MODEL=                  # set to e.g. qwen2.5vl:7b to enable image captioning
WHISPER_MODEL_SIZE=base
CHUNK_SIZE=512
CHUNK_OVERLAP=64
RETRIEVAL_TOP_K=6
RERANKER_ENABLED=true
RERANKER_MODEL=BAAI/bge-reranker-v2-m3
COLPALI_ENABLED=false
COLPALI_MODEL=vidore/colpali-v1.2
DOCLING_VLM_ENABLED=false
DOCLING_VLM_MODEL=Qwen/Qwen2.5-VL-7B-Instruct
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=info
CORS_ALLOW_ORIGINS=http://localhost:5173,http://localhost:3000
SKIP_LLM_HEALTH_CHECK=false
```

**Example VRAM budget on a 16 GB GPU** during normal inference:

| Model | VRAM |
|---|---|
| qwen3:14b (Q4_K_M) | ~9–10 GB |
| nomic-embed-text | ~270 MB |
| bge-reranker-v2-m3 | ~1.1 GB |
| **Headroom** | **~5 GB** |

ColPali (~8 GB) is a separate batch job — never loaded concurrently with
inference. The vision model (`qwen2.5vl:7b`, ~5 GB) is only active during
watcher ingestion of images.

**Smaller GPU / CPU-only?** Set `OLLAMA_LLM_MODEL` to a smaller chat model
(~7–8B, e.g. `qwen2.5:7b` or `llama3.1:8b`) in `.env` — see README for details.
CPU-only inference works but generation is noticeably slower.

**LLM choice rationale:** `qwen3:14b` is the default over smaller ~9B models for
better multi-step reasoning quality in RAG Q&A; both fit in 16 GB with embeddings
+ reranker. A ~32B-class model needs ~19 GB at Q4_K_M and won't fit in 16 GB.

**Important quirk:** `SUPPORTED_EXTENSIONS`, `WATCHER_EXCLUDE_PATHS`, and
`CORS_ALLOW_ORIGINS` are plain comma-separated strings (NOT JSON arrays)
because pydantic-settings v2 tries to JSON-parse `List[str]` fields before
validators run. They are stored as `str` in `config.py` and exposed as lists
via `@property` methods:
- `settings.supported_extensions_list`
- `settings.watcher_exclude_paths_list`
- `settings.cors_allow_origins_list`

---

## Config Loading Quirk

`backend/config.py` uses an absolute path to find `.env`:

```python
env_file=Path(__file__).parent.parent / ".env"
```

This means scripts can be run from any directory and will always find the `.env`
at the repo root. Do not change this to a relative path.

---

## What Gets Indexed / Excluded (general guidance)

HomeIntel walks `NAS_WATCH_PATH` (despite the name, any local folder works) and
indexes every file matching `SUPPORTED_EXTENSIONS`, skipping anything under
`WATCHER_EXCLUDE_PATHS`. When pointing it at a real home-server/NAS root that
also hosts unrelated services, a typical exclude list covers:

- Docker Engine's own storage (overlay2 dirs) — not user files
- Media libraries (movies/TV) — video is intentionally unsupported (see below)
- App-specific database/cache/binary directories for other self-hosted services
  (e.g. a photo manager's internal DB, a password manager's vault, a media
  server's transcode cache, TLS/SSH private key stores) — exclude these
  entirely rather than relying on redaction alone
- Web project directories with `node_modules` — huge and irrelevant
- Qdrant's own storage directory, if it lives under the watched path

Video files (`.mp4`, `.mov`, etc.) are intentionally excluded from
`SUPPORTED_EXTENSIONS` — transcribing a large video library with Faster-Whisper
would take far too long for a real-time watcher. Audio (`.mp3`, `.wav`) is
supported by default; if you have a very large audio library, consider excluding
it too (add its path to `WATCHER_EXCLUDE_PATHS`, drop `.mp3,.wav` from
`SUPPORTED_EXTENSIONS`) and re-enable later with an overnight `reindex.py` run.

Photos living in a nested "originals-only" structure (e.g. under a self-hosted
photo manager) are often indexed separately with `--path`/`--ext` flags to skip
sidecar/config files:
```bash
python ../scripts/reindex.py --path <INDEX_ROOT>/photos/originals --ext .jpg .jpeg .png
```
Temporarily remove that path from `WATCHER_EXCLUDE_PATHS` before running, then
restore it after — the watcher usually excludes the whole photo-app directory to
avoid watching its database/cache churn.

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
verified working (embedded mode, embeddings, upsert/query/delete). It has been
**superseded** by the Qdrant wrapper but is preserved in git history for traceability.

**Current files:**
- `backend/vectorstore/qdrant.py` — VectorStore class (Qdrant, hybrid dense+sparse)
- `backend/vectorstore/__init__.py`
- `docker/qdrant.yml` — standalone Qdrant Docker Compose
- `scripts/test_vectorstore.py` — smoke test (deploy Qdrant first)

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
    "file_path": str,       # absolute path
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
- Optional VLM picture description (`DOCLING_VLM_ENABLED=true`) uses Qwen2.5-VL-7B-Instruct via Ollama's OpenAI-compat endpoint — disabled by default; enable only if VRAM headroom exists alongside the main LLM + nomic-embed-text
- Docker's own storage path must stay in `WATCHER_EXCLUDE_PATHS` if it lives under the watched root — it's Docker Engine overlay2 storage, not user files
- If ingesting a directory of per-service docker-compose files, they often live one level deep in service folders, e.g. `<INDEX_ROOT>/photo-app/docker-compose.yml`
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

**Verified working (scripts/verify_api.py — 8/8 checks pass against the bundled sample docs):**
- GET /health → `{"status":"ok","ollama":true,"qdrant":true}` ✓
- GET /stats → per-modality chunk counts ✓
- POST /chat (resume question) → correct answer with source attribution ✓
- POST /chat (docker-compose port question) → answer pulls from the sample compose file ✓
- POST /chat (weather — not in data) → correct "don't have information" refusal ✓
- POST /chat (modality_filter=document) → all sources are document modality ✓
- POST /chat (empty question) → HTTP 422 ✓
- POST /chat (no body) → HTTP 422 ✓

**Notes:**
- CORS origins are configurable via `CORS_ALLOW_ORIGINS` (comma-separated) — defaults to `http://localhost:5173,http://localhost:3000`
- Ollama URL comes from `OLLAMA_BASE_URL` env var — use `http://host.docker.internal:11434` in Docker compose
- RAGChain and VectorStore are module-level singletons (lazy init on first request)
- Startup health check is controlled by `SKIP_LLM_HEALTH_CHECK=true` (useful for running without Ollama)
- Sources include file_name, file_path, modality, and 200-char excerpt per chunk
- `test_ingestion.py` now accepts `--keep-chunks` to skip cleanup (for API testing)
- `scripts/verify_api.py` starts uvicorn as a subprocess, runs 8 checks, shuts it down; exit 0 = all pass
- Degraded-state test (Ollama down) must be run manually: stop Ollama and check `/health` returns `{"status":"degraded"}`
- Retriever now fetches 3×top_k candidates from Qdrant, then reranks with `bge-reranker-v2-m3` before truncating to `RETRIEVAL_TOP_K` — reranker is lazy-loaded on first query (first call is slow while model downloads)
- `RERANKER_ENABLED=false` disables reranking and falls back to raw RRF order (useful for latency testing)

### ✅ Step 5 — File Watcher (COMPLETE)
**Files created:**
- `backend/ingestion/watcher.py`

**What it does:**
- Watches `NAS_WATCH_PATH` recursively with watchdog `Observer`
- Respects `WATCHER_EXCLUDE_PATHS` and `SUPPORTED_EXTENSIONS` on every event
- `on_created` / `on_modified` → `delete_file()` + `ingest_file()`
- `on_deleted` → `delete_file()` only; `on_moved` → delete src + ingest dst
- Network-share disconnection recovery: catches `OSError`/`PermissionError` and restarts observer with exponential backoff (5s → 10s → 30s → 60s → 120s) — relevant when the watched path is an SMB/NFS mount
- Auto-started in FastAPI lifespan via `start_watcher()` — runs as daemon thread

### ✅ Step 6 — React Frontend (COMPLETE)
**Files created:**
- `frontend/` — Vite + React + TypeScript app at `http://localhost:5173`

**What it does:**
- Dark-themed chat UI with full message history
- Source attribution per answer: file name, file path, modality icon (📄/🖼️/🎵)
- Per-answer meta: chunk count and model name
- Modality filter dropdown (All / Documents / Images / Audio)
- Status bar: Ollama + Qdrant health dots + chunk counts by modality, refreshes every 30s
- Enter to send, Shift+Enter for newline

### ✅ Step 7 — ColPali Visual Retrieval (COMPLETE)
**Files created:**
- `backend/ingestion/processors/colpali.py` — PDF page rendering + ColPali embedding
- `scripts/run_colpali.py` — batch indexer

**What it does:**
- Renders PDF pages via `pypdfium2` (no poppler dependency)
- Embeds each page as 128-dim patch multi-vectors using `vidore/colpali-v1.2`
- Stored in sibling collection `homeintel_colpali` (separate from text chunks to keep stats clean)
- `query_colpali()` uses MaxSim scoring in Qdrant
- Retrieval merge gated on `COLPALI_ENABLED=true` — ColPali hits merged with text results before reranking
- `run_colpali.py` walks the watched path, respects exclude paths, indexes all PDFs (~5s/page on a modern NVIDIA GPU)

**Resource notes:**
- ColPali model: ~8 GB VRAM — run batch indexer with Ollama stopped to free VRAM
- Extra deps: `pip install colpali-engine>=0.3.0 pypdfium2>=4.0.0` (commented out in requirements.txt — opt-in, see the reinstall-torch warning there)
- Storage: ~1 MB per page in Qdrant

### ✅ Step 8 — Image & Audio Processors (COMPLETE)
**Files created:**
- `backend/ingestion/processors/image.py`
- `backend/ingestion/processors/audio.py`

**What it does:**
- Images: Ollama vision API caption (`OLLAMA_VISION_MODEL`) → text → nomic-embed-text + BM25 pipeline; opt-in (empty = skip images)
- Audio: faster-whisper transcription (GPU, lazy singleton, `WHISPER_MODEL_SIZE`) → text chunks → embeddings
- Both integrated into `pipeline.py` routing — watcher auto-ingests images/audio on file events

### ✅ Step 9 — CLIP Visual Similarity (COMPLETE)
**Files created/modified:**
- `backend/vectorstore/clip.py` — CLIPVisualStore class (openai/clip-vit-large-patch14, 768-dim)
- `backend/api/files.py` — GET /file (file serving), POST /visual-search (CLIP query)
- `scripts/index_visual.py` — batch CLIP indexer for photos
- `frontend/src/api.ts` — visualSearch(), fileUrl() helpers
- `frontend/src/types.ts` — VisualResult, VisualSearchResponse types

**What it does:**
- User uploads/takes a photo → POSTed to `/visual-search` → embedded with CLIP →
  nearest-neighbor search in `homeintel_visual`
- Results shown in a swipeable carousel with similarity % (see Step 10 UI)
- GET /file serves files securely (path must resolve under `NAS_WATCH_PATH` — blocks traversal)
- Image sources in RAG chat responses also render as a carousel

**Qdrant collections:**
| Collection | Model | Dim | Contents |
|---|---|---|---|
| `homeintel` | nomic-embed-text + BM25 | 768 | Text chunks (all modalities) |
| `homeintel_colpali` | ColPali | 128 | PDF page patch embeddings |
| `homeintel_visual` | CLIP | 768 | Photo visual embeddings (one per photo) |

**VRAM budget with CLIP:**
| Model | VRAM |
|---|---|
| qwen3:14b (Q4_K_M) | ~9–10 GB |
| nomic-embed-text | ~270 MB |
| bge-reranker-v2-m3 | ~1.1 GB |
| CLIP large (lazy-loaded on first /visual-search) | ~500 MB |
| **Headroom** | **~4.5 GB** |

**Running the CLIP indexer:**
```bash
# activate your venv (see Environment section above)
cd backend
# Temporarily remove the photo folder from WATCHER_EXCLUDE_PATHS in .env first
python ../scripts/index_visual.py --path <INDEX_ROOT>/photos/originals
# Restore WATCHER_EXCLUDE_PATHS after
```

**Similarity threshold:** 0.70 cosine similarity. Results below this are filtered out.
Adjust `_SCORE_THRESHOLD` in `backend/vectorstore/clip.py` if results are too sparse or too noisy.

### ✅ Step 10 — Mobile-First UI Redesign (COMPLETE)
**Files created:**
- `frontend/src/components/Header.tsx` — slim header (title + health dot + new-chat)
- `frontend/src/components/BottomSheet.tsx` — reusable iOS-style bottom sheet
- `frontend/src/components/PhotoCarousel.tsx` — swipeable photo carousel (scroll-snap)
- `frontend/src/components/icons.tsx` — inline SVG icons (no icon-lib dependency)
- `frontend/src/hooks/useSystemStatus.ts` — health + stats polling hook

**Files removed (superseded):**
- `frontend/src/components/StatusBar.tsx` → replaced by header health dot + status sheet
- `frontend/src/components/VisualSearchResult.tsx` → replaced by PhotoCarousel

**What changed (inspired by ChatGPT/Claude mobile):**
- **Slim header**: just the title, a single health dot (green/red — tap opens a status
  bottom sheet with Ollama/Qdrant state + chunk counts), and a new-chat compose button.
  All the clutter (2 dots + 3 counts) moved off the header.
- **Pill composer**: `[ + ]  [ text input ]  [ ↑ send ]` — one rounded pill. The `+`
  opens a bottom sheet (action menu) instead of cramming buttons into the bar.
- **`+` action sheet** consolidates all secondary actions: Take a photo / Choose from
  library (visual search), modality filter chips (All/Documents/Images/Audio), and a
  "Results to show" stepper (Auto/3/6/10/15/20).
- **Active-filter chips** above the composer show the current filter / top-K, each
  tappable to clear — gives the user feedback on non-default state.
- **Markdown rendering** (`react-markdown`): assistant answers now render `**bold**`,
  numbered lists, code, etc. instead of showing literal `**[IMG_0967.JPG]**`.
- **Swipeable carousel** replaces the thumbnail grid + lightbox: one large photo at a
  time, swipe left/right (native CSS `scroll-snap`, no JS touch handling), dot indicators
  (≤10) or a `n / total` counter, filename + match% overlay, "Show 10 more" pagination.
- **44px touch targets** throughout; safe-area insets on header (top) and composer (bottom).

**Adaptive top-K (`backend/rag/retriever.py`):**
- Image queries default to **top-20** (browse more photos); docs/audio stay at
  `RETRIEVAL_TOP_K` (6) to keep LLM context + latency small.
- The `+` sheet's "Results to show" sends an explicit `top_k` (1–20) on the `/chat`
  request that overrides the adaptive default. Blank = Auto.
- New optional `top_k` field on `ChatRequest` → threaded through `chain.run()` →
  `retrieve()` → `_resolve_top_k()`.

**Dependency added:** `react-markdown` (frontend). Inline SVG icons avoid an icon library.

### ✅ Step 11 — Secret Redaction + Image-Query Prompt (COMPLETE)
**Files created/modified:**
- `backend/security/redact.py` — regex secret scanner (`redact_secrets`, `contains_secret`)
- `backend/security/__init__.py`
- `backend/rag/prompts.py` — added `IMAGE_SEARCH_TEMPLATE` + SECURITY rule in `SYSTEM_TEMPLATE`
- `backend/rag/chain.py` — prompt routing (image vs Q&A) + context redaction
- `backend/ingestion/pipeline.py` — redact chunks before upsert
- `backend/api/chat.py` — redact source excerpts
- `backend/config.py` — `redact_secrets` flag (default true)

**Image-query prompt routing:** `chain.run()` picks `IMAGE_SEARCH_TEMPLATE` when
`modality_filter == "image"` OR every retrieved doc is image-modality. This stops the
strict Q&A prompt from refusing ("I don't have information…") on photo-search phrases —
image search is a "show me matches" task, not factual Q&A. Detection is deterministic
from retrieval metadata, not an LLM classifier.

**Secret redaction (defense in depth, `REDACT_SECRETS=true`):** detected secrets become
`<REDACTED>` at three layers:
1. **Ingestion** (`pipeline.py`) — scrubbed before upsert, so new files never store
   secrets in Qdrant.
2. **LLM context** (`chain.py`) — scrubbed before the prompt, so the model never sees a
   raw credential even if one is already stored from an earlier index.
3. **Source excerpts** (`api/chat.py`) — scrubbed before returning to the UI.
Plus a prompt rule telling the LLM to refuse to reveal credentials.

Detects: PEM private-key blocks; token prefixes (sk-, ghp_/gho_/ghs_, xox*, AKIA, AIza,
JWT); secret-named key/value pairs (env `POSTGRES_PASSWORD=…`, YAML/JSON `"password":"…"`,
incl. prefixed names + `totp`/`otp_secret`/`recovery_code`/`seed_phrase`/`mnemonic`);
connection-string creds (`scheme://user:pass@host`). Errs toward over-redaction.

**Existing index caveat:** secrets indexed *before* this was added are still scrubbed at
response time (layers 2+3), but remain raw in Qdrant storage. Run `reindex.py` to purge
them from storage (layer 1). Personal-notes/password-export folders and config/compose
files are common sources of stray secrets — redaction covers them, but excluding a
sensitive folder entirely via `WATCHER_EXCLUDE_PATHS` is always the belt-and-suspenders
option.

### ✅ Step 12 — Live CLIP Auto-Indexing + Status Split (COMPLETE)
**Files modified:**
- `backend/vectorstore/clip.py` — module-level `get_clip_store()` shared singleton
- `backend/vectorstore/qdrant.py` — `stats()` now also returns `visual` (homeintel_visual count)
- `backend/api/files.py` — uses `get_clip_store()` (one CLIP instance for search + watcher)
- `backend/ingestion/watcher.py` — CLIP-index/delete images on file events
- `backend/config.py` — `clip_auto_index` flag
- `frontend/src/types.ts` + `App.tsx` — `visual` stat + split status sheet sections

**Part A — status sheet split:** `/stats` now returns a `visual` count (the
`homeintel_visual` CLIP collection), counted via the existing Qdrant client with **no
CLIP model load**. The status sheet shows two sections: "Text index" (Documents /
Images (captions) / Audio / Total chunks) and "Visual index" (Photos (CLIP)). This
disambiguates caption *chunks* (text search) from CLIP *photos* (visual search).

**Part B — live auto-indexing (`CLIP_AUTO_INDEX=true`):** the watcher now CLIP-indexes
new/changed images into `homeintel_visual` and removes them on delete/move — so visual
search stays current without re-running `index_visual.py`. Hooked in the **watcher**
(not `ingest_file`/`reindex.py`) so the bulk tools stay fast and independent. CLIP work
is wrapped in try/except (never breaks captioning) and runs even when captioning is
skipped (CLIP doesn't need `OLLAMA_VISION_MODEL`). The shared `get_clip_store()`
singleton means search and the watcher load the CLIP model at most once per process.

Order of operations that was followed: caption reindex → bulk `index_visual.py` →
enable `CLIP_AUTO_INDEX=true`. `index_visual.py` is now a recovery/rebuild tool only.

### ✅ Step 13 — CLIP Text→Image Search for Image Queries (COMPLETE)
**Files modified:**
- `backend/vectorstore/clip.py` — `_embed_text()`, `search_by_text()`, generalized `_project()`
- `backend/vectorstore/qdrant.py` — `captions_for()` (batch caption lookup for display)
- `backend/rag/retriever.py` — route image queries to `_retrieve_images_clip()`
- `backend/config.py` — `clip_text_search`, `clip_text_min_score`
- `frontend/src/components/PhotoCarousel.tsx` — render per-photo caption as markdown

**Why:** image (text) queries used caption keyword matching, which is noisy for
visual concepts — e.g. a query for people surfaced a food photo whose caption
happened to share a word. CLIP text→image search encodes the query into the same
space as the photos and ranks by visual relevance, which fixed this class of
false positive in testing.

**How it works:** for image queries (modality=image, `CLIP_TEXT_SEARCH=true`),
`retrieve()` calls CLIP `search_by_text()` against `homeintel_visual`, then attaches
each photo's caption via `VectorStore.captions_for()` (chunk_index 0 from the main
collection) for the carousel + LLM summary. Bypasses reranking/dedup (CLIP results
are already one-per-photo and ranked). Falls back to caption search if CLIP returns
nothing.

**Score-scale note:** CLIP text-image cosine sims cluster in a narrow band
(~0.20–0.26 on typical photo data) — the *ranking* is what's reliable, not the
absolute value. `CLIP_TEXT_MIN_SCORE=0.2` drops the absolute tail; per-query top-K
ranking does the rest. If results feel too loose/strict, tune `CLIP_TEXT_MIN_SCORE`
— but expect small absolute differences to matter.

Also fixed: the carousel's per-photo caption now renders markdown (was showing raw
`**Objects**` etc.), matching the main answer bubble.

### ✅ Step 14 — Streaming Responses + Full Image Captions (COMPLETE)
**Files created/modified:**
- `backend/rag/chain.py` — extracted `_prepare()`; added `run_stream()` generator
- `backend/api/chat.py` — `build_sources()` + `_captions_for_images()`; new `POST /chat/stream` SSE endpoint
- `backend/rag/retriever.py` — public `get_vectorstore()`
- `backend/ingestion/pipeline.py` — stores `full_caption` on chunk 0 for images
- `backend/vectorstore/qdrant.py` — `_join_caption_chunks()`; `captions_for()` returns the FULL caption
- `frontend/src/api.ts` — `sendChatStream()` SSE reader
- `frontend/src/App.tsx`, `components/ChatMessage.tsx`, `types.ts`, `App.css` — incremental rendering + caret
- `scripts/test_caption_join.py`, `scripts/test_build_sources.py`, `scripts/test_run_stream.py`, `scripts/verify_streaming.py`

**Streaming (`POST /chat/stream`, SSE):** `run_stream()` yields a `sources` event
first (carousel paints immediately), then `token` deltas, then `done`. `POST /chat`
is unchanged (still used by `verify_api.py`). The frontend reads the stream via
`fetch` + `ReadableStream` (EventSource can't POST) and appends tokens live.
Applies to both document Q&A and natural-language image queries (same LLM call);
the camera-upload visual search has no LLM text and is unchanged. Redaction is
unchanged — context is redacted before the LLM and excerpts before the `sources`
event; streamed tokens pass through raw, exactly as `/chat` returns the answer.

**Full captions:** captions longer than `CHUNK_SIZE` previously showed only
chunk 0, then got capped to 200 chars in the API. Now the complete caption is
stored as a `full_caption` payload field on chunk 0 at ingestion, and
`captions_for()` prefers it (falling back to an overlap-stripped join of all
chunks for photos indexed before this field existed). `build_sources()` gives
image sources the full caption; other modalities keep a 200-char excerpt.

**Re-index requirement:** photos indexed before this step have no `full_caption`
field — display still works via the fallback join, but to store it exactly,
re-run the caption reindex over the photos:
```bash
# activate your venv (see Environment section above)
cd backend
# Temporarily remove the photo folder from WATCHER_EXCLUDE_PATHS in .env first
python ../scripts/reindex.py --path <INDEX_ROOT>/photos/originals --ext .jpg .jpeg .png
# Restore WATCHER_EXCLUDE_PATHS after
```

---

### ✅ Step 15 — Voice Dictation + UI Polish (COMPLETE)
**Files created/modified:**
- `frontend/src/components/Header.tsx` — app icon left of the wordmark
- `frontend/src/App.tsx` — Camera/Photos tiles, mic/send/stop composer swap,
  recording UI, tappable example prompts, conversation persistence
- `frontend/src/components/icons.tsx` — Mic/Stop/Copy icons (inline SVG)
- `frontend/src/api.ts` — `transcribe()` client
- `frontend/src/hooks/useRecorder.ts` — MediaRecorder hook
- `frontend/src/components/ChatMessage.tsx` — "Thinking…" label + copy-answer button
- `frontend/src/storage.ts` — localStorage conversation persistence
- `backend/api/transcribe.py` — `POST /transcribe` (reuses faster-whisper singleton)
- `backend/main.py` — register transcribe router
- `scripts/test_transcribe.py` — unit test for the MIME→ext helper
- `docs/tailscale-https-setup.md` — secure-origin setup for phone mic access

**What it does:**
- **Photo input:** "Search options" sheet now shows two tiles — **Camera** (direct
  capture) and **Photos** (picker). iOS still injects its own menu for the picker;
  that is an unavoidable web limitation (only the camera input bypasses it).
- **Voice dictation:** the composer's right button is **Mic** when empty, **Send**
  when typing, **Stop** while a response streams. Recording shows a red dot + timer
  with cancel/stop; the result is transcribed by the local Whisper model and dropped
  into the composer for review (never auto-sent).
- **QoL:** stop-generation button, mic/send swap, tappable example prompts,
  "Thinking…" indicator (label added to the existing dots), copy-answer button,
  and current-conversation persistence across reloads (New-chat clears it).

**Secure-context note:** mic capture needs HTTPS/localhost. It works on desktop
`localhost` directly; for a phone on your LAN, use Tailscale Serve HTTPS (see
`docs/tailscale-https-setup.md`). No public internet exposure.

**VRAM:** Whisper `base` (~150 MB) lazy-loads on first transcription — no budget concern.

---

### ✅ Step 16 — Desktop Responsiveness + Streaming UX + Audit Pass (COMPLETE)
**Files created:**
- `docs/ARCHITECTURE.md` — frontend/streaming map for cheap future-session context
- `frontend/src/components/Sidebar.tsx`, `PhotoGrid.tsx`, `PhotoResults.tsx`, `FilterControls.tsx`, `FadeImg.tsx`
- `frontend/src/hooks/useMediaQuery.ts`

**Desktop responsiveness (mobile CSS untouched — additive `min-width` breakpoints md 768 / lg 1024 / xl 1440):**
- lg+: full-width shell, collapsible **chat-history sidebar** (multi-conversation), chat column capped at 860px reading width; on mobile the same history opens as a **left slide-in drawer** (ChatGPT-app style, `Drawer.tsx`) from the header panel button (`ConversationList.tsx` shared by both; rows show relative timestamps + two-tap delete confirm)
- **Storage v2**: `homeintel.conversations.v2` = Conversation[{id,title,messages,updatedAt}], capped at 30 (oldest pruned); old `homeintel.conversation.v1` auto-migrates on first load
- ≥768px: photo results render as a 2/3/4-column **grid** (`object-fit: cover`, hover overlay with filename + match%); mobile keeps the swipeable carousel — `PhotoResults.tsx` is the single branching point
- Keyboard: Ctrl/Cmd+K focuses composer, Esc closes sheets + lightbox, ←/→ navigate lightbox; global `:focus-visible` ring

**Streaming UX (frontend-only — SSE protocol unchanged):**
- Status line derived from message state: "Searching N indexed chunks…" → "k matches · generating answer…" → tokens
- Skeleton shimmer slots + per-image fade-in (`FadeImg`) — zero layout shift; pre-retrieval shimmer lines
- Auto-scroll only while pinned near the bottom (scrolling up mid-stream no longer yanks down); token scrolls use `behavior:"instant"` because `.chat-area` has CSS `scroll-behavior:smooth`

**Backend:**
- `GET /thumb?path&w` — Pillow thumbnail (EXIF-rotated, widths 320/480/768 whitelist), disk cache at `backend/.cache/thumbs` (gitignored) keyed by path+mtime+width; grid uses w=480, carousel w=768, lightbox keeps full-res `/file`. Warm cache serves in ~50 ms; typical thumb ~113 KB vs 5.4 MB original
- FastAPI serves `frontend/dist` at `/` when present (guarded, absolute path, mounted after routers) — **same-origin**: `BASE=""` in api.ts, Vite dev proxy to :8000. `VITE_API_BASE_URL` still overrides

**A11y + polish:** aria-live completion announcements, `aria-busy` while streaming, image alt text from captions, `--text-soft` (#94a3b8) for small text (old muted was ~3.5:1, fails AA), `prefers-reduced-motion` support, Ollama-offline banner + disabled send, zero-result note, Retry button on failed messages, filters persisted (`homeintel.filters.v1`), `ChatMessage` memoized (only the streaming message re-renders per token).

**Removed (dead code):** `lucide-react` dep, `sendChat()` frontend helper (backend `/chat` stays — verify_api.py uses it), `.sheet-action`/`.bubble-loading` CSS, unused template assets.

**Verified:** `npm run build` green on every commit; `/thumb` 200 + cached, traversal → 403, SPA served at `:8000`; `verify_api.py` **6/8 = pre-existing baseline** (unfiltered photo bias, not a regression). Curly-quote paths (e.g. a phone's default photo-sync folder name) work via browser `encodeURIComponent` — Git Bash curl mangles them, test with PowerShell.

---

### ✅ Step 17 — Open-Source Prep (COMPLETE)

Generalized the project so a stranger can clone and run it on Windows/macOS/Linux
without any personal infrastructure. Summary of what changed (see `README.md` and
`.env.example` for the current, authoritative shape):

- **Framing:** "point HomeIntel at any local folder" replaces the NAS-specific
  framing everywhere; NAS/SMB is now one documented advanced use case
  (`docs/deployment-windows.md`).
- **Cross-platform install/run:** `setup.sh`/`setup.ps1` (venv + PyTorch CPU-or-`--gpu`
  + backend/frontend deps + `.env` bootstrap) and `run.sh`/`run.ps1`
  (`backend`/`frontend`/`all`) replace the old conda + Windows-Task-Scheduler-only
  workflow as the primary path. The Task Scheduler + SMB-mount recipe was preserved
  as an optional guide, not deleted.
- **Config:** added `CORS_ALLOW_ORIGINS` (was hardcoded); `NAS_WATCH_PATH` defaults
  to the bundled sample data so the app runs out-of-the-box.
- **Sample data:** `data/sample-docs/` (synthetic resume, docker-compose file, notes,
  JSON config) ships in the repo so setup → run → query works with zero external
  data, and the smoke tests have something to query.
- **Tests:** `backend/tests/` — pure-unit pytest suite (`test_config.py`,
  `test_redact.py`; no Qdrant/Ollama required) via `backend/requirements-dev.txt`.
  The integration scripts (`scripts/verify_api.py`, `verify_streaming.py`,
  `test_ingestion.py`, `test_vectorstore.py`) now query `data/sample-docs/` and are
  documented as optional/manual (they need live Qdrant + Ollama).
- **Dependencies:** torch is no longer pinned in `requirements.txt` (installed by
  `setup.*` instead, to avoid clobbering the platform-specific build); ColPali
  deps are commented out (opt-in); unused deps removed.
- **Docs:** this file (`CLAUDE.md`) was generalized with placeholders
  (`<REPO_PATH>`, `<INDEX_ROOT>`, `<QDRANT_HOST>`, "your NVIDIA GPU") and the
  original personal version preserved locally as `CLAUDE.local.md` (gitignored).
  `README.md`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md` added/rewritten
  for a public audience.

---

## Key Design Decisions & Rationale

**Why Qdrant over ChromaDB?**
ChromaDB was the original choice (embedded mode, direct filesystem write). Migrated
to Qdrant for: (1) native sparse vector support for hybrid BM25+dense search without
extra libraries; (2) named multi-vector fields needed for ColPali co-location
(Step 7); (3) single-binary Docker deploy — lighter than Milvus, simpler ops than
Weaviate; (4) server-side RRF fusion means no client-side score normalization
hacks. Network-share latency to embedded-DB files was the limiting factor with
ChromaDB when the data lived on a NAS; switching to a LAN REST call actually
reduces write latency.

**Why one collection for all modalities?**
Cross-modal search — a single query should search docs, photos, and audio
transcripts simultaneously. Modality is stored as metadata for filtering when
needed.

**Why nomic-embed-text instead of the LLM for embeddings?**
Speed. nomic-embed-text is a dedicated embedding model — much faster than using
the chat LLM for embeddings. Frees the LLM for generation only.

**Why qwen3:14b as the default LLM?**
Better multi-step reasoning for RAG Q&A than smaller ~9B-class models; both fit
in 16 GB VRAM alongside nomic + reranker (~10.5 GB combined). A ~32B-class model
needs ~19 GB and does not fit. On a smaller GPU, switch to a ~7–8B model via
`OLLAMA_LLM_MODEL` if response latency/VRAM is a problem.

**Why SMB (or NFS) as an optional NAS-mount path, not a hard dependency?**
Some platforms have quirks mounting NFS (e.g. Windows is missing `mount.exe` in
some install configurations unless the right optional feature is enabled) — SMB
is the more broadly reliable option for a Windows host. Either way, this is only
relevant if you choose to point `NAS_WATCH_PATH` at a network share; a local
folder needs neither.

**Why keep Qdrant off the app host, on a separate machine/NAS, as an option?**
Survives app-host reinstall/upgrade; accessible from multiple machines later; LAN
latency to a Qdrant REST API is low (sub-ms on gigabit). Running it locally via
`docker/qdrant.yml` is equally supported and is the simplest default.

**Why CLIP for visual similarity instead of caption matching?**
Caption matching (embed the caption text, search by text) finds semantically
similar photos but misses visual similarity — two different dog breeds are both
"dogs" in text but look different. CLIP encodes raw image pixels into a shared
embedding space, so photos of the same subject (same pet, same person, same
place) cluster together regardless of caption text. One point per photo (no
chunking), stored in `homeintel_visual` separately from the caption chunks in
`homeintel`. Threshold 0.70 cosine similarity — tune up for stricter matching,
down if too few results.

**Why exclude video files?**
A large personal video library (movies, TV, raw footage) would take days to
transcribe with Faster-Whisper even on a fast GPU. Can be added later as a
separate phase if you have the time budget for it.

**List fields in config.py are plain strings:**
pydantic-settings v2 tries to JSON-parse `List[str]` fields before validators run,
causing `JSONDecodeError`. Workaround: store as `str`, expose as list via
`@property`.

---

## Auto-Start / Advanced Deployment

The old Windows-Task-Scheduler-based auto-start (with an SMB drive-mount wrapper
script) has been superseded by `run.sh`/`run.ps1` as the primary way to run the
app. If you want HomeIntel to run unattended at login and/or watch a NAS share,
see **`docs/deployment-windows.md`** — it covers both mounting a network share as
a drive letter and registering the backend/frontend as Windows scheduled tasks,
with placeholders for your own paths/hostnames.

---

## Common Commands

```bash
# One-time setup (creates .venv, installs deps, bootstraps .env)
./setup.sh            # macOS/Linux — add --gpu for CUDA
.\setup.ps1           # Windows — add -Gpu for CUDA

# Run (backend + frontend, or either alone)
./run.sh              # or: ./run.sh backend | ./run.sh frontend
.\run.ps1             # or: .\run.ps1 backend  | .\run.ps1 frontend

# Run a script directly (after activating .venv)
cd backend
python ../scripts/test_vectorstore.py

# Start Ollama (usually already running as a background service)
ollama serve

# Check running models
ollama list

# Pull models
ollama pull qwen3:14b
ollama pull nomic-embed-text
# Optional vision model for image captioning:
# ollama pull qwen2.5vl:7b

# Start full Docker stack
docker compose up -d

# Dev mode (hot-reload)
docker compose -f docker-compose.yml -f docker-compose.dev.yml up

# Unit tests (no Qdrant/Ollama needed)
pip install -r backend/requirements-dev.txt
python -m pytest backend/tests
```

---

## Known Issues / Gotchas

1. **Qdrant must be running before the backend starts** — unlike an embedded
   vector DB, Qdrant is a separate network service. If the backend starts before
   Qdrant is up, VectorStore init will fail. Start `docker/qdrant.yml` and confirm
   `curl http://<QDRANT_HOST>:6333/healthz` returns `{"title":"qdrant"}` before
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
   alongside the main LLM + nomic-embed-text. On a 16 GB GPU, this will OOM.
   Fallback: use `Qwen/Qwen2.5-VL-3B-Instruct` or keep disabled and rely on Docling's
   text extraction only. Use `OLLAMA_VISION_MODEL` for image captioning instead.

6. **Ollama on Windows starts automatically** — the installer registers Ollama as a
   Windows service. Running `ollama serve` manually will fail with "address already
   in use". Just use `ollama pull` and `ollama list` directly.

7. **VSCode terminal PATH** — after installing Ollama, VSCode terminals need a full
   restart to pick up the updated PATH. Reloading the window is not enough.

8. **PyTorch CUDA vs. very new NVIDIA GPUs** — if your GPU ships with a CUDA driver
   newer than PyTorch's latest published wheel (e.g. cu128 at time of writing), the
   cu128 build is generally backward-compatible with newer CUDA 13.x drivers. Use
   `./setup.sh --gpu` / `.\setup.ps1 -Gpu`, or manually:
   `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128`.

9. **NFS `mount.exe` missing on Windows** — the Windows NFS feature
   `ServicesForNFS-ClientOnly` installs without `mount.exe`; you also need
   `ClientForNFS-Infrastructure` enabled. SMB is the more broadly reliable option
   on Windows.

10. **SMB mount persistence** — `net use <DRIVE_LETTER> \\<NAS_HOST>\<NAS_SHARE> /persistent:yes`
    persists across reboots. See `docs/deployment-windows.md` for the full recipe.

11. **SMB PermissionError on some NAS files** — some NAS platforms (e.g. TrueNAS)
    apply ACLs that make files visible in directory listings but unreadable over
    SMB. Fix by granting read+execute recursively on the export path from the NAS
    shell (e.g. `chmod -R a+rX <export-path>` — the capital `X` only adds execute
    on directories, not files). `reindex.py` catches `PermissionError` and treats
    it as a skip (not an error) so the rest of the index continues.

12. **reindex.py — full re-process by default, opt-in resume** — by default every file
    is `delete_file()` + `ingest_file()` on every run (correct after chunking/embedding
    setting changes). Pass `--skip-existing` to skip files already in Qdrant — this fetches
    all indexed `file_path`s in one scroll pass (`VectorStore.indexed_file_paths()`) and is
    the way to **resume an interrupted run**: Ctrl+C, then re-run with `--skip-existing`.

13. **Ingestion speed — concurrency + batching:**
    - `reindex.py --workers N` ingests N files concurrently via a thread pool. The dominant
      per-image cost is the Ollama vision caption (network-bound), so threads keep the GPU
      busy. **Must pair with `OLLAMA_NUM_PARALLEL=N`** on the Ollama service or the requests
      queue serially. Start with `--workers 3`; drop to 2 if the vision model OOMs on a 16 GB GPU.
      Set Ollama parallelism on Windows: `setx OLLAMA_NUM_PARALLEL 3` then restart the
      Ollama service (Task Manager → Services, or re-login).
    - Switching to a smaller vision model (e.g. a 3B variant) roughly halves caption time.
    - `index_visual.py` batches the CLIP forward pass (`--batch-size`, default 16) and reads
      images in parallel (`--read-workers`, default 4) — ~5-8x faster than one-at-a-time.
      It also supports `--skip-existing` for resume.

14. **ColPali install overwrites PyTorch with a CPU build** — `pip install colpali-engine`
    pulls in a CPU-only PyTorch as a dependency. Always reinstall the GPU build after:
    ```bash
    pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu128
    ```
    Verify with `python -c "import torch; print(torch.cuda.is_available())"` → must be `True`.
    The cu128 index only has torch 2.7.0+ (not 2.5.x) — pin to `>=2.7.0` if you uncomment
    ColPali in `requirements.txt`.
