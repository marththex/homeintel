# HomeIntel Architecture (frontend + API surface)

> Companion to CLAUDE.md (which holds build history/env). This is the "how the app works" map.

## Stack
- **Frontend:** Vite + React 19 + TypeScript (strict). No state library — all state lives in `App.tsx`.
  Styling is one plain-CSS file (`src/App.css`) with CSS variables (dark theme only), mobile-first,
  additive `min-width` breakpoints: md 768 / lg 1024 / xl 1440. Only runtime dep of note: `react-markdown`.
- **Backend:** FastAPI (`backend/main.py`) + LangChain RAG chain → Ollama (qwen3.5:9b) + Qdrant (hybrid dense+sparse,
  reranked). Serves built frontend from `frontend/dist` at `/` when present (same-origin).

## Frontend component tree

```
App.tsx (all state: messages, conversation id, input, modality/topK filters, loading, sheets, recorder)
├── Sidebar (lg+; conversation list, filters) → FilterControls
├── Header (health dot → status sheet, new chat, sidebar toggle)
├── chat-area → chat-col
│   └── ChatMessage (memoized; markdown answer, status line, meta, copy, retry)
│       ├── PhotoResults → PhotoGrid (≥768px) | PhotoCarousel (<768px)
│       │   └── Lightbox (portal; Esc/arrows; CollapsibleCaption)
│       └── SourceList (non-image source rows; image sources → PhotoResults)
├── composer (pill: + sheet / textarea / mic|send|stop; chip row; offline banner)
├── Drawer (mobile: left slide-in chat history) → ConversationList (shared with Sidebar)
└── BottomSheet ×2 (search options + filters on mobile; system status)
```

## Data flow — chat streaming
1. `submit()` (App.tsx) appends user msg + empty assistant msg (`streaming: true`), keeps an AbortController.
2. `sendChatStream()` (api.ts) POSTs `/chat/stream`, reads SSE frames from the fetch body:
   `sources` (docs metadata + model + chunks_used) → `token` (deltas) → `done` | `error`.
3. Handlers patch the assistant message by id: the status line advances ("Searching…" →
   "N matches · generating…"), tokens stream into the markdown block. Image/source results are
   deliberately held back until `done` — text always renders first; images then fade into
   fixed-aspect skeleton slots below it (zero CLS).
4. Backend: `RAGChain.run_stream()` (rag/chain.py) — retrieve (rerank; CLIP text→image for image queries)
   → yield sources → stream LLM tokens. Secrets are redacted at ingestion, LLM-context, and excerpt layers.

## Images
- Retrieval sources of modality `image` and visual-search results share `CarouselPhoto` shape.
- Thumbnails: `GET /thumb?path&w` (Pillow, EXIF-rotated, disk cache `backend/.cache/thumbs`) for grid/carousel;
  full-res `GET /file?path` only in the Lightbox. Both endpoints validate paths against NAS_WATCH_PATH.

## Persistence (localStorage)
- `homeintel.conversations.v2` — Conversation[{id,title,messages,updatedAt}], capped at 30 (oldest pruned).
  (Migrated from single-conversation `homeintel.conversation.v1`.)
- `homeintel.filters.v1` — {modality, topK}. `homeintel.sidebar.collapsed` — boolean.
- Blob URLs / streaming flags are stripped before save.

## API surface (FastAPI)

| Route | Purpose |
|---|---|
| POST /chat | blocking RAG answer (used by scripts/verify_api.py) |
| POST /chat/stream | SSE: sources → token* → done/error |
| GET /health, GET /stats | health dot + status sheet (polled every 30s by useSystemStatus) |
| GET /file, GET /thumb | NAS file / cached thumbnail serving (path-validated) |
| POST /visual-search | CLIP image→image search (camera/library upload) |
| POST /transcribe | faster-whisper voice dictation |

## Build / run
- Frontend: `cd frontend && npm run dev` (5173, proxies API to 8000) · `npm run build` (tsc + vite → dist/)
- Backend: `cd backend && uvicorn main:app --host 0.0.0.0 --port 8000` (inside the venv created by `setup.sh`/`setup.ps1`)
- Quick start: `./run.sh` / `.\run.ps1` (starts both). Prod/unattended: see `docs/deployment-windows.md` for an
  optional Windows auto-start (Task Scheduler) recipe. Port 8000 serves the API + built UI when `frontend/dist` exists.

## Conventions
- Commits: `feat(ui)|fix(ui)|perf(ui)|feat(api)|docs|chore: …` (no co-author trailers).
- Icons: inline SVGs in `components/icons.tsx` via the shared `base()` helper — no icon libraries.
- CSS: mobile styles are the base layer; desktop is additive `@media (min-width: …)`; 44px touch targets;
  `env(safe-area-inset-*)` on header/composer; `:focus-visible` for keyboard focus.
- Mobile layout is verified pixel-identical at 375px for any styling change.
