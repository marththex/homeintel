# Design — Streaming responses + full image captions

**Date:** 2026-06-27
**Status:** Approved (ready for implementation plan)

## Problem

Two user-reported issues in the HomeIntel chat UI:

1. **No streaming.** The UI waits for the entire LLM answer before showing
   anything. The user wants ChatGPT/Claude-style token-by-token streaming, and
   confirmed it should apply to natural-language **image** queries too, not just
   document Q&A.
2. **Captions cut off.** Photo captions in the carousel are truncated
   mid-sentence (e.g. "…They are holding what appears t").

### Root causes (verified in code)

- Both document Q&A and natural-language image queries flow through the **same**
  single blocking call `self._llm.invoke()` in
  [chain.py](../../../backend/rag/chain.py). The frontend `sendChat()` does
  `res.json()` and waits for the full response.
- Caption truncation has **two** compounding causes:
  1. `captions_for()` ([qdrant.py:334](../../../backend/vectorstore/qdrant.py))
     fetches only `chunk_index == 0`. The detailed caption prompt routinely
     produces captions longer than `CHUNK_SIZE` (512 chars), so the remaining
     chunks are dropped.
  2. `excerpt = text[:200]` ([api/chat.py:32](../../../backend/api/chat.py)) caps
     even chunk 0 to 200 chars before it reaches the carousel — this is the exact
     "…what appears t" cutoff seen in the screenshot.

The carousel already renders `SourceDoc.excerpt` as the per-photo caption
([SourceList.tsx:27](../../../frontend/src/components/SourceList.tsx)), so the fix
is entirely about delivering the full caption text into that field.

## Goals

- Stream answer tokens to the UI as they are generated, for **both** document
  and image queries.
- Show the **complete** caption under each carousel photo.

## Non-goals

- vLLM/Triton inference-engine swap, model changes, or reranker tuning (the
  earlier raw-latency discussion). Streaming addresses *perceived* speed; raw
  generation speed is a separate effort.
- Streaming the camera-upload visual search — that path returns only photos
  (no LLM text), so there is nothing to stream. It stays unchanged.

---

## Part A — Streaming (SSE via `/chat/stream`)

### Decision

Add a new `POST /chat/stream` Server-Sent Events endpoint and leave `POST /chat`
unchanged. Rationale: backward compatible — `verify_api.py` and any non-streaming
caller keep working; the streaming surface is additive and lower risk. SSE over a
`fetch` + `ReadableStream` reader is used (not `EventSource`, which cannot issue
POST requests).

### Backend — [rag/chain.py](../../../backend/rag/chain.py)

Refactor `RAGChain` so retrieval and prompt construction are shared between the
blocking and streaming paths:

- Extract `_prepare(question, modality_filter, top_k) -> (messages, docs, is_image_query)`
  containing the existing retrieve → context-format (with redaction) →
  template-selection logic. `run()` is rewritten to call `_prepare()` then
  `self._llm.invoke(messages)` — behavior identical to today.
- Add `run_stream(question, modality_filter, top_k)` generator that calls
  `_prepare()`, then yields tagged events in order:
  1. `("sources", docs)` — emitted once, before any token.
  2. `("token", delta)` — for each chunk from `self._llm.stream(messages)`
     (`langchain_ollama.ChatOllama` supports `.stream()` yielding message chunks
     whose `.content` is the delta).
  3. `("done", {"model", "chunks_used"})` — emitted last.

### Backend — [api/chat.py](../../../backend/api/chat.py)

- New `POST /chat/stream` returning
  `StreamingResponse(generator(), media_type="text/event-stream")`, reusing the
  existing `ChatRequest` schema. The generator translates `run_stream()` tags into
  SSE frames:
  - `event: sources` / `data: <JSON SourceDoc[]>` — same shaping/redaction as
    `/chat`, **including full image captions** (Part B). Sent first so the
    carousel paints immediately while text streams in.
  - `event: token` / `data: {"delta": "..."}`
  - `event: done` / `data: {"model": "...", "chunks_used": N}`
  - `event: error` / `data: {"detail": "..."}` — emitted if generation raises
    mid-stream, so the client can show an error instead of hanging.
- `POST /chat` is untouched.

### Redaction (unchanged guarantees)

- Context is redacted **before** the LLM inside `_prepare()` (existing
  `redact_secrets` logic), so the model never sees a raw credential.
- Source excerpts are redacted before the `sources` event is sent.
- The streamed answer tokens are passed through raw — identical to how `/chat`
  returns `result["answer"]` raw today. No change to the redaction model.

### Frontend — [src/api.ts](../../../frontend/src/api.ts)

- Add `sendChatStream(question, modality_filter, top_k, handlers)` where
  `handlers = { onSources, onToken, onDone, onError, signal }`. Implementation:
  `fetch` the endpoint, read `response.body.getReader()`, decode with
  `TextDecoder`, buffer and split SSE frames on `\n\n`, parse `event:`/`data:`
  lines, dispatch to the handlers. An `AbortController` (`signal`) cancels an
  in-flight stream if the user sends another message.

### Frontend — [src/App.tsx](../../../frontend/src/App.tsx), [src/components/ChatMessage.tsx](../../../frontend/src/components/ChatMessage.tsx), [src/types.ts](../../../frontend/src/types.ts)

- `ChatMessage` type gains an optional `streaming?: boolean`.
- `submit()` pushes the user message, then an assistant placeholder with
  `streaming: true` and empty `content`. Wire handlers:
  - `onSources` → set `.sources` on the placeholder (carousel renders).
  - `onToken` → append delta to `.content`.
  - `onDone` → set `.model` / `.chunks_used`, clear `streaming`.
  - `onError` → set `error: true`.
- Replace the global loading-dots bubble with an in-bubble state: show dots until
  the first `sources`/`token` arrives, then a blinking caret while `streaming`.
- `react-markdown` re-renders on each token; response sizes here are small enough
  that this is fine.

---

## Part B — Full image captions

### Decision

"Re-ingest whole" (the more thorough option): store the complete caption at
ingestion time and look it up exactly, with a fallback that fixes already-indexed
photos immediately (before any re-index).

### Ingestion — [ingestion/pipeline.py](../../../backend/ingestion/pipeline.py)

- For `modality == Modality.IMAGE`, store the complete caption text as a
  `full_caption` field in the metadata of **chunk_index 0** (which lands in the
  Qdrant payload via `{"page_content": ..., **metadata}` in `upsert()`). Redact
  it with `redact_secrets` when `settings.redact_secrets` is on, consistent with
  chunk redaction.
- Chunking for embeddings/search is otherwise unchanged, so caption keyword
  search and cross-modal "All" search keep their current granularity.

### Lookup — [vectorstore/qdrant.py `captions_for()`](../../../backend/vectorstore/qdrant.py)

- Prefer the `full_caption` payload field when present.
- Fallback for photos indexed before this change (no `full_caption` yet): scroll
  **all** chunks for the path, sort by `chunk_index`, strip the `CHUNK_OVERLAP`
  (64-char) overlap between consecutive chunks, and join into the full caption.
- Net result: display is correct immediately for old data, and exact (single-field
  read) after re-index.

### Display — [api/chat.py](../../../backend/api/chat.py)

- When shaping `SourceDoc`s, replace the excerpt for **image** sources with the
  full caption via a single batched `captions_for()` call for all image source
  paths; non-image sources keep `text[:200]`.
- This guarantees full captions for **both** image retrieval paths — the primary
  CLIP text→image path and the caption-keyword fallback / cross-modal path.
- The same shaping is applied in the `sources` event of `/chat/stream`.
- No frontend change needed for captions — `SourceList` already feeds
  `excerpt → CarouselPhoto.caption`.

### Re-index (documented operational step)

- Re-run the caption reindex over the ~522 photos (existing documented flow:
  remove `Z:/marcus_photoprism` from `WATCHER_EXCLUDE_PATHS`, run `reindex.py`
  over `originals`, restore exclude) to populate `full_caption`. Until then, the
  fallback join in `captions_for()` covers display.
- Add a CLAUDE.md note (new Step 14) describing the `full_caption` field, the
  display change, and the re-index requirement.

---

## Testing

- **Streaming:** `scripts/verify_streaming.py` (or extend
  `scripts/verify_api.py`) — POST to `/chat/stream`, assert the SSE event order
  `sources → token (≥1) → done`, and that a document question produces a coherent
  concatenated answer.
- **Captions:** assert an image query's returned source caption exceeds 200 chars
  (proving both the `chunk_index==0`-only bug and the 200-char cap are gone).
- **Manual:** UI streams answer text incrementally for both a document question
  and an image query; carousel shows full multi-sentence captions; the
  non-streaming `/chat` still passes `verify_api.py`.

## Files touched

| File | Change |
|---|---|
| `backend/rag/chain.py` | Extract `_prepare()`; add `run_stream()` generator |
| `backend/api/chat.py` | Add `POST /chat/stream` SSE; full-caption shaping for image sources |
| `backend/ingestion/pipeline.py` | Store `full_caption` on chunk 0 for images |
| `backend/vectorstore/qdrant.py` | `captions_for()` prefers `full_caption`, falls back to overlap-stripped join |
| `frontend/src/api.ts` | `sendChatStream()` SSE reader |
| `frontend/src/App.tsx` | Streaming submit flow + placeholder assistant message |
| `frontend/src/components/ChatMessage.tsx` | Streaming caret / in-bubble loading |
| `frontend/src/types.ts` | `streaming?: boolean` on `ChatMessage` |
| `scripts/verify_streaming.py` | New streaming verification (or extend `verify_api.py`) |
| `CLAUDE.md` | New Step 14 documenting both changes + re-index step |
