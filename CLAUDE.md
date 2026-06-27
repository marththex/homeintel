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
| LLM | PC (GPU) | qwen3:14b via Ollama at `http://localhost:11434` |
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
SUPPORTED_EXTENSIONS=.pdf,.docx,.txt,.md,.yml,.yaml,.json,.png,.jpg,.jpeg
WATCHER_EXCLUDE_PATHS=<NAS_ROOT>/homeintel/qdrant_storage,<NAS_ROOT>/docker,<NAS_ROOT>/media,<NAS_ROOT>/plex_config,<NAS_ROOT>/roms,<NAS_ROOT>/ai_videos,<NAS_ROOT>/wedding_raw_videos,<NAS_ROOT>/palworld,<NAS_ROOT>/qbittorrent,<NAS_ROOT>/marcus_photoprism,<NAS_ROOT>/Music
QDRANT_URL=http://<NAS_IP>:6333
QDRANT_COLLECTION_NAME=homeintel
QDRANT_API_KEY=
EMBED_DIM=768
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_LLM_MODEL=qwen3:14b
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
SKIP_LLM_HEALTH_CHECK=false
```

**VRAM budget on RTX 5080 (16 GB) during normal inference:**

| Model | VRAM |
|---|---|
| qwen3:14b (Q4_K_M) | ~9–10 GB |
| nomic-embed-text | ~270 MB |
| bge-reranker-v2-m3 | ~1.1 GB |
| **Headroom** | **~5 GB** |

ColPali (~8 GB) is a separate batch job — never loaded concurrently with inference.
Vision model (`qwen2.5vl:7b`, ~5 GB) only active during watcher ingestion of images.

**LLM choice rationale:** `qwen3:14b` was chosen over `qwen3.5:9b` for better
multi-step reasoning quality in RAG Q&A. Both fit in 16 GB with embeddings +
reranker. `qwen3:32b` requires ~19 GB at Q4_K_M and does not fit.

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
├── nextcloud/          ← EXCLUDED (app code — thousands of l10n JSON files)
├── nginx/              ← Nginx config
├── cloudflared/        ← Cloudflare tunnel config
├── Wedding_Contracts/  ← PDF documents (text + ColPali indexed)
├── Marcus_Resume.pdf   ← PDF at root
├── 2fact/              ← Text notes + Bitwarden export
├── homepage/           ← Homepage dashboard config
├── marcus_photoprism/  ← originals/ indexed for images (via --path); database/ + storage/ excluded
│   ├── originals/      ← 522 personal photos — captioned via qwen2.5vl:7b
│   ├── database/       ← EXCLUDED (PhotoPrism DB)
│   └── storage/        ← EXCLUDED (PhotoPrism thumbnails/cache)
├── Music/              ← EXCLUDED (audio — Whisper too slow for full library)
├── vw-data/            ← EXCLUDED (Vaultwarden DB + RSA keys — sensitive)
├── mariadb/            ← EXCLUDED (MariaDB binary data)
├── certs/              ← EXCLUDED (SSL private keys — sensitive)
├── tailscale/          ← EXCLUDED (Tailscale state)
├── portainer_data/     ← indexed (docker-compose files in compose/ subfolder)
├── keribakes/          ← EXCLUDED (web project with node_modules)
├── my-portfolio/       ← EXCLUDED (web project)
├── react_projects/     ← EXCLUDED (web projects)
├── portfolio/          ← EXCLUDED (web project)
├── media/              ← EXCLUDED (movies/TV - too large to index)
├── roms/               ← EXCLUDED (game ROMs)
├── plex_config/        ← EXCLUDED (Plex binary data)
├── ai_videos/          ← EXCLUDED
├── wedding_raw_videos/ ← EXCLUDED
├── palworld/           ← EXCLUDED (game server data)
└── qbittorrent/        ← EXCLUDED (torrent client data)
```

Video files (.mp4, .mov) are intentionally excluded — the NAS has 52 movies,
15 TV shows, and a large One Pace collection that would take days to transcribe.

Audio files (.mp3, .wav) are excluded from `SUPPORTED_EXTENSIONS` — the Music
library is too large to transcribe in a reasonable time with Faster-Whisper. Can
be re-enabled by adding `.mp3,.wav` back to `SUPPORTED_EXTENSIONS` and removing
`Z:/Music` from `WATCHER_EXCLUDE_PATHS`, then running `reindex.py` overnight.

**Photos (marcus_photoprism/originals):** indexed separately with `--path` and
`--ext` flags to avoid YAML sidecars. Run with:
```bash
python ../scripts/reindex.py --path Z:/marcus_photoprism/originals --ext .jpg .jpeg .png
```
Temporarily remove `Z:/marcus_photoprism` from `WATCHER_EXCLUDE_PATHS` before
running, then restore it after. The watcher excludes the whole photoprism dir
to avoid watching PhotoPrism's database/storage churn.

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

### ✅ Step 5 — File Watcher (COMPLETE)
**Files created:**
- `backend/ingestion/watcher.py`

**What it does:**
- Watches `NAS_WATCH_PATH` recursively with watchdog `Observer`
- Respects `WATCHER_EXCLUDE_PATHS` and `SUPPORTED_EXTENSIONS` on every event
- `on_created` / `on_modified` → `delete_file()` + `ingest_file()`
- `on_deleted` → `delete_file()` only; `on_moved` → delete src + ingest dst
- SMB disconnection recovery: catches `OSError`/`PermissionError` and restarts observer with exponential backoff (5s → 10s → 30s → 60s → 120s)
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
- `run_colpali.py` walks NAS, respects exclude paths, indexes all PDFs (~5s/page on RTX 5080)

**Resource notes:**
- ColPali model: ~8 GB VRAM — run batch indexer with Ollama stopped to free VRAM
- Extra deps: `pip install colpali-engine>=0.3.0 pypdfium2>=4.0.0` (commented out in requirements.txt)
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
- `backend/api/files.py` — GET /file (NAS file serving), POST /visual-search (CLIP query)
- `scripts/index_visual.py` — batch CLIP indexer for photos
- `frontend/src/api.ts` — visualSearch(), fileUrl() helpers
- `frontend/src/types.ts` — VisualResult, VisualSearchResponse types

**What it does:**
- User uploads/takes a photo → POSTed to `/visual-search` → embedded with CLIP →
  nearest-neighbor search in `homeintel_visual`
- Results shown in a swipeable carousel with similarity % (see Step 10 UI)
- GET /file serves NAS files securely (path must resolve under NAS_WATCH_PATH — blocks traversal)
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
conda activate homeintel
cd backend
# Temporarily remove Z:/marcus_photoprism from WATCHER_EXCLUDE_PATHS in .env first
python ../scripts/index_visual.py --path Z:/marcus_photoprism/originals
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
them from storage (layer 1). Note `2fact/` (Bitwarden export) and config/compose files
were the main sources — redaction now covers them, but excluding `Z:/2fact` entirely is
the belt-and-suspenders option.

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

**Why nomic-embed-text instead of the LLM for embeddings?**
Speed. nomic-embed-text is a dedicated embedding model — much faster than using
qwen3:14b for embeddings. Frees the LLM for generation only.

**Why qwen3:14b over qwen3.5:9b?**
Better multi-step reasoning for RAG Q&A. Both fit in 16 GB VRAM alongside nomic
+ reranker (~10.5 GB combined). qwen3.5:9b excels at coding speed but qwen3:14b
has a stronger reasoning score for document Q&A tasks. qwen3:32b requires ~19 GB
and does not fit. Switch back to qwen3.5:9b via `OLLAMA_LLM_MODEL=qwen3.5:9b`
if response latency is too high.

**Why SMB over NFS for NAS mount?**
NFS `mount.exe` was missing from Windows NFS client install (only `nfsadmin.exe`
present). SMB works natively on Windows and is simpler. The NAS share `NFSdocker`
is configured as SMB (Windows share) on TrueNAS.

**Why keep Qdrant on NAS instead of local PC?**
Survives PC reinstall/upgrade, accessible from multiple machines later. LAN
latency to Qdrant REST API is low (sub-ms on gigabit). Qdrant storage (vectors +
payload) on NAS SSD is preferred over PC to keep GPU memory free for inference.

**Why CLIP for visual similarity instead of caption matching?**
Caption matching (embed the caption text, search by text) finds semantically similar photos
but misses visual similarity — a black Lab and a golden retriever are both "dogs" but look
different. CLIP encodes raw image pixels into a shared embedding space, so photos of the
same subject (same dog, same person, same place) cluster together regardless of the caption
text. One point per photo (no chunking), stored in `homeintel_visual` separately from the
caption chunks in `homeintel`. Threshold 0.70 cosine similarity — tune up for stricter
matching, down if too few results.

**Why exclude video files?**
52 movies + 15 TV shows + large One Pace collection would take days to transcribe
with Faster-Whisper even on RTX 5080. Can be added later as a separate phase.

**List fields in config.py are plain strings:**
pydantic-settings v2 tries to JSON-parse `List[str]` fields before validators run,
causing `JSONDecodeError`. Workaround: store as `str`, expose as list via
`@property`.

---

## Auto-start (Windows Task Scheduler)

Both services are registered as Windows scheduled tasks that start at login:

```powershell
# Start / stop
Start-ScheduledTask "HomeIntel-API"
Stop-ScheduledTask "HomeIntel-API"
Start-ScheduledTask "HomeIntel-UI"
Stop-ScheduledTask "HomeIntel-UI"

# View logs
Get-Content "C:\Users\admin\Documents\homeintel\logs\api.log" -Wait

# Re-register (run as admin if tasks need to be recreated)
# See scripts/start_backend.bat and scripts/start_frontend.bat
```

The backend script (`start_backend.bat`) maps `Z:` via `net use` before starting
uvicorn — needed because Task Scheduler runs in session 0 where user-mapped SMB
drives are not visible.

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

# Pull models
ollama pull qwen3:14b
ollama pull nomic-embed-text
# Optional vision model for image captioning:
# ollama pull qwen2.5vl:7b

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
   alongside qwen3:14b + nomic-embed-text. On RTX 5080 (16 GB VRAM), this will OOM.
   Fallback: use `Qwen/Qwen2.5-VL-3B-Instruct` or keep disabled and rely on Docling's
   text extraction only. Use `OLLAMA_VISION_MODEL` for image captioning instead.

6. **Ollama on Windows starts automatically** — the installer registers Ollama as a
   Windows service. Running `ollama serve` manually will fail with "address already
   in use". Just use `ollama pull` and `ollama list` directly.

7. **VSCode terminal PATH** — after installing Ollama, VSCode terminals need a full
   restart to pick up the updated PATH. Reloading the window is not enough.

8. **PyTorch CUDA** — RTX 5080 uses CUDA driver 13.2 but PyTorch only supports up
   to cu128. Use `--index-url https://download.pytorch.org/whl/cu128`. The cu128
   build is backward compatible with CUDA 13.x drivers.

9. **NFS mount.exe missing** — Windows NFS feature `ServicesForNFS-ClientOnly`
    installs without `mount.exe`. Need to also enable `ClientForNFS-Infrastructure`.
    Ultimately went with SMB instead.

10. **SMB mount persistence** — `net use Z: \\YOUR_NAS_HOST\NFSdocker /persistent:yes`
    persists across reboots. Replace `YOUR_NAS_HOST` with your TrueNAS IP or hostname.

11. **SMB PermissionError on some NAS files** — some files are visible in directory
    listings but not readable over SMB due to TrueNAS ACLs. Fix by running
    `chmod -R a+rX /mnt/Pool/NFSdocker` on the TrueNAS shell. The `X` (capital)
    only adds execute on directories, not files. `reindex.py` catches `PermissionError`
    and treats it as a skip (not an error) so the rest of the index continues.

14. **ColPali install overwrites PyTorch with CPU build** — `pip install colpali-engine`
    pulls in a CPU-only PyTorch as a dependency. Always reinstall CUDA PyTorch after:
    ```bash
    pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu128
    ```
    Verify with `python -c "import torch; print(torch.cuda.is_available())"` → must be `True`.
    The cu128 index only has torch 2.7.0+ (not 2.5.x) — pin to `>=2.7.0` in requirements.txt.

12. **reindex.py — full re-process by default, opt-in resume** — by default every file
    is `delete_file()` + `ingest_file()` on every run (correct after chunking/embedding
    setting changes). Pass `--skip-existing` to skip files already in Qdrant — this fetches
    all indexed `file_path`s in one scroll pass (`VectorStore.indexed_file_paths()`) and is
    the way to **resume an interrupted run**: Ctrl+C, then re-run with `--skip-existing`.

13. **Ingestion speed — concurrency + batching:**
    - `reindex.py --workers N` ingests N files concurrently via a thread pool. The dominant
      per-image cost is the Ollama vision caption (network-bound), so threads keep the GPU
      busy. **Must pair with `OLLAMA_NUM_PARALLEL=N`** on the Ollama service or the requests
      queue serially. Start with `--workers 3`; drop to 2 if `qwen2.5vl:7b` OOMs on 16 GB.
      Set Ollama parallelism on Windows: `setx OLLAMA_NUM_PARALLEL 3` then restart the
      Ollama service (Task Manager → Services, or re-login).
    - Switching `OLLAMA_VISION_MODEL=qwen2.5vl:3b` roughly halves caption time vs `:7b`.
    - `index_visual.py` batches the CLIP forward pass (`--batch-size`, default 16) and reads
      images in parallel (`--read-workers`, default 4) — ~5-8x faster than one-at-a-time.
      It also supports `--skip-existing` for resume.
