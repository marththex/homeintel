# Streaming Responses + Full Image Captions Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stream chat answers token-by-token to the UI (for both document and image queries) and show each carousel photo's complete caption instead of a 200-char slice.

**Architecture:** Backend adds a `POST /chat/stream` Server-Sent Events endpoint alongside the unchanged `POST /chat`; the RAG chain gains a `run_stream()` generator that emits a sources event then token deltas. Captions are stored whole at ingestion (`full_caption` payload field on chunk 0) and looked up exactly, with an overlap-stripping fallback that fixes already-indexed photos. The frontend reads the SSE stream via `fetch` + `ReadableStream` and renders text incrementally.

**Tech Stack:** Python 3.11, FastAPI, Starlette `StreamingResponse`, `langchain_ollama.ChatOllama` (`.stream()`), Qdrant, React 19 + TypeScript + Vite, `react-markdown`.

## Global Constraints

- Conda env `homeintel`; run backend scripts from the `backend/` directory (`cd backend`).
- No pytest in this project — pure logic is tested with standalone assertion scripts in `scripts/`; integration is verified with scripts that drive uvicorn (pattern: `scripts/verify_api.py`).
- `conda run -n homeintel` does not support multiline `-c`; always use a `.py` file.
- Frontend type-check + build gate: `cd frontend && npm run build` (`tsc && vite build`).
- Secret redaction is gated on `settings.redact_secrets` (default `true`) and must be preserved at every layer it exists today (ingestion, LLM context, source excerpts).
- File references use `Modality` enum values: `Modality.IMAGE.value == "image"`, etc.
- Commit messages: no `Co-Authored-By` / AI-attribution trailers.
- Work happens on branch `feat/streaming-and-full-captions` (already created).

---

### Task 1: Full-caption storage + overlap-stripping join

Store the complete caption at ingestion and reconstruct it on lookup so multi-chunk captions are returned whole.

**Files:**
- Modify: `backend/vectorstore/qdrant.py` (add `_join_caption_chunks()`; rewrite `captions_for()`)
- Modify: `backend/ingestion/pipeline.py` (set `full_caption` on chunk 0 for images)
- Test: `scripts/test_caption_join.py` (new, pure-function test — no Qdrant needed)

**Interfaces:**
- Produces: `_join_caption_chunks(chunks: list[str], overlap: int) -> str` (module-level in `qdrant.py`)
- Produces: `VectorStore.captions_for(file_paths: list[str]) -> dict[str, str]` — now returns the FULL caption per path (was chunk-0-only)
- Produces: ingestion writes payload key `full_caption: str` on the `chunk_index == 0` point of image files

- [ ] **Step 1: Write the failing test**

Create `scripts/test_caption_join.py`:

```python
"""
scripts/test_caption_join.py — Unit test for _join_caption_chunks (no Qdrant).

Run from the backend directory:
    cd backend
    python ../scripts/test_caption_join.py
Exit 0 = all pass.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from vectorstore.qdrant import _join_caption_chunks


def test_strips_overlap():
    a = "The quick brown fox jumps over the lazy"
    b = "over the lazy dog and runs away quickly"
    out = _join_caption_chunks([a, b], overlap=64)
    assert out == "The quick brown fox jumps over the lazy dog and runs away quickly", out


def test_single_chunk():
    assert _join_caption_chunks(["only one"], overlap=64) == "only one"


def test_empty():
    assert _join_caption_chunks([], overlap=64) == ""


def test_no_overlap_concatenates():
    assert _join_caption_chunks(["abc", "def"], overlap=64) == "abcdef"


if __name__ == "__main__":
    test_strips_overlap()
    test_single_chunk()
    test_empty()
    test_no_overlap_concatenates()
    print("ALL PASS")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && python ../scripts/test_caption_join.py`
Expected: FAIL — `ImportError: cannot import name '_join_caption_chunks'`

- [ ] **Step 3: Add the join helper**

In `backend/vectorstore/qdrant.py`, add this module-level function near the other module helpers (e.g. just above the `VectorStore` class, after `_chunk_id`):

```python
def _join_caption_chunks(chunks: list[str], overlap: int) -> str:
    """
    Reconstruct full text from ordered chunks produced by
    RecursiveCharacterTextSplitter, removing the overlap region duplicated
    between consecutive chunks.

    For each chunk after the first, find the largest k (<= overlap) where the
    accumulated text's suffix equals the next chunk's prefix, and append only
    the non-overlapping remainder. Falls back to plain concatenation when no
    overlap is found.
    """
    if not chunks:
        return ""
    out = chunks[0]
    for nxt in chunks[1:]:
        max_k = min(overlap, len(out), len(nxt))
        k = 0
        for cand in range(max_k, 0, -1):
            if out[-cand:] == nxt[:cand]:
                k = cand
                break
        out += nxt[k:]
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && python ../scripts/test_caption_join.py`
Expected: `ALL PASS`

- [ ] **Step 5: Rewrite `captions_for()` to return full captions**

In `backend/vectorstore/qdrant.py`, replace the existing `captions_for()` method body with:

```python
    def captions_for(self, file_paths: list[str]) -> dict[str, str]:
        """
        Return {file_path: full caption text} for the given image paths.

        Prefers the `full_caption` payload field (written at ingestion on
        chunk 0). For files indexed before that field existed, reconstructs the
        caption from ALL chunks (ordered by chunk_index, overlap stripped).
        Used to attach captions to image results for display.
        """
        if not file_paths:
            return {}

        out: dict[str, str] = {}
        chunks_by_path: dict[str, list[tuple[int, str]]] = {}

        next_offset = None
        while True:
            points, next_offset = self._client.scroll(
                collection_name=settings.qdrant_collection_name,
                scroll_filter=Filter(
                    must=[FieldCondition(key="file_path", match=MatchAny(any=file_paths))]
                ),
                with_payload=True,
                with_vectors=False,
                limit=1000,
                offset=next_offset,
            )
            for p in points:
                payload = p.payload or {}
                fp = payload.get("file_path")
                if not fp:
                    continue
                if "full_caption" in payload and fp not in out:
                    out[fp] = payload["full_caption"]
                chunks_by_path.setdefault(fp, []).append(
                    (payload.get("chunk_index", 0), payload.get("page_content", ""))
                )
            if next_offset is None:
                break

        # Fallback: reconstruct from chunks for any path lacking full_caption.
        for fp in file_paths:
            if fp in out:
                continue
            chunks = chunks_by_path.get(fp, [])
            if not chunks:
                continue
            chunks.sort(key=lambda c: c[0])
            out[fp] = _join_caption_chunks([t for _, t in chunks], settings.chunk_overlap)

        return out
```

(Confirm `MatchAny` is already imported in `qdrant.py` — it is used by the current `captions_for`.)

- [ ] **Step 6: Store `full_caption` at ingestion**

In `backend/ingestion/pipeline.py`, inside `ingest_file()`, modify the chunk loop so image files record the full caption on chunk 0. Replace the loop block (currently starting `for i, chunk in enumerate(chunks):`) with:

```python
    # For images, persist the complete (redacted) caption on chunk 0 so the UI
    # can show the full caption even when it spans multiple chunks.
    full_caption: Optional[str] = None
    if modality == Modality.IMAGE:
        full_caption = redact_secrets(text) if settings.redact_secrets else text

    for i, chunk in enumerate(chunks):
        # Strip secrets before they ever reach the vector store.
        if settings.redact_secrets:
            chunk = redact_secrets(chunk)

        meta: dict = {
            "file_path":   str(p),
            "file_name":   p.name,
            "file_ext":    ext,
            "chunk_index": i,
            "modality":    modality.value,
            "created_at":  now,
        }
        if title:
            meta["title"] = title
        if i == 0 and full_caption is not None:
            meta["full_caption"] = full_caption

        documents.append(Document(page_content=chunk, metadata=meta))
```

- [ ] **Step 7: Re-run the pure test (regression) and commit**

Run: `cd backend && python ../scripts/test_caption_join.py`
Expected: `ALL PASS`

```bash
git add backend/vectorstore/qdrant.py backend/ingestion/pipeline.py scripts/test_caption_join.py
git commit -m "feat: store and reconstruct full image captions"
```

---

### Task 2: Full-caption source shaping in `/chat`

Centralize source-doc shaping so image sources carry the full caption (not a 200-char excerpt) and reuse it for both endpoints.

**Files:**
- Modify: `backend/api/chat.py` (add `build_sources()` + `_captions_for_images()`; use them in `/chat`)
- Modify: `backend/rag/retriever.py` (add public `get_vectorstore()`)
- Test: `scripts/test_build_sources.py` (new, pure — captions injected, no Qdrant/Ollama)

**Interfaces:**
- Consumes: `VectorStore.captions_for()` (Task 1)
- Produces: `build_sources(docs: list[Document], captions: dict[str, str]) -> list[SourceDoc]` (module-level in `api/chat.py`)
- Produces: `_captions_for_images(docs: list[Document]) -> dict[str, str]` (module-level in `api/chat.py`)
- Produces: `get_vectorstore() -> VectorStore` (module-level in `rag/retriever.py`)

- [ ] **Step 1: Write the failing test**

Create `scripts/test_build_sources.py`:

```python
"""
scripts/test_build_sources.py — Unit test for build_sources (no Qdrant/Ollama).

Run from the backend directory:
    cd backend
    python ../scripts/test_build_sources.py
Exit 0 = all pass.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from langchain_core.documents import Document
from api.chat import build_sources


def test_image_uses_full_caption_not_truncated():
    long_caption = "A detailed description. " * 30  # ~720 chars
    doc = Document(
        page_content="short matched chunk",
        metadata={"file_name": "IMG.JPG", "file_path": "/x/IMG.JPG", "modality": "image"},
    )
    sources = build_sources([doc], {"/x/IMG.JPG": long_caption})
    assert len(sources[0].excerpt) > 200, len(sources[0].excerpt)
    assert sources[0].excerpt == long_caption


def test_image_without_caption_falls_back_to_page_content():
    doc = Document(
        page_content="fallback text",
        metadata={"file_name": "IMG.JPG", "file_path": "/x/IMG.JPG", "modality": "image"},
    )
    sources = build_sources([doc], {})
    assert sources[0].excerpt == "fallback text"


def test_document_truncated_to_200():
    doc = Document(
        page_content="x" * 500,
        metadata={"file_name": "a.md", "file_path": "/a.md", "modality": "document"},
    )
    sources = build_sources([doc], {})
    assert len(sources[0].excerpt) == 200, len(sources[0].excerpt)


if __name__ == "__main__":
    test_image_uses_full_caption_not_truncated()
    test_image_without_caption_falls_back_to_page_content()
    test_document_truncated_to_200()
    print("ALL PASS")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && python ../scripts/test_build_sources.py`
Expected: FAIL — `ImportError: cannot import name 'build_sources'`

- [ ] **Step 3: Add `get_vectorstore()` to the retriever**

In `backend/rag/retriever.py`, add below the existing `_get_vs()`:

```python
def get_vectorstore() -> VectorStore:
    """Public accessor for the retriever's VectorStore singleton."""
    return _get_vs()
```

- [ ] **Step 4: Add `build_sources()` and `_captions_for_images()` and use them in `/chat`**

Rewrite `backend/api/chat.py` to:

```python
"""
api/chat.py — POST /chat endpoint and shared source shaping.
"""

import logging
from fastapi import APIRouter, HTTPException
from langchain_core.documents import Document

from config import settings
from models.chat import ChatRequest, ChatResponse, SourceDoc
from rag.chain import get_chain
from rag.retriever import get_vectorstore
from vectorstore.qdrant import Modality
from security import redact_secrets

logger = logging.getLogger(__name__)
router = APIRouter()


def _captions_for_images(docs: list[Document]) -> dict[str, str]:
    """Batch-fetch full captions for any image-modality docs (keyed by file_path)."""
    paths = [
        d.metadata.get("file_path", "")
        for d in docs
        if d.metadata.get("modality") == Modality.IMAGE.value
    ]
    paths = [p for p in paths if p]
    if not paths:
        return {}
    return get_vectorstore().captions_for(paths)


def build_sources(docs: list[Document], captions: dict[str, str]) -> list[SourceDoc]:
    """
    Shape retrieved docs into SourceDocs.

    Image sources carry their FULL caption (from `captions`, keyed by file_path,
    falling back to page_content); other modalities get a 200-char excerpt.
    Secrets are redacted when REDACT_SECRETS is enabled.
    """
    def _redact(text: str) -> str:
        return redact_secrets(text) if settings.redact_secrets else text

    sources: list[SourceDoc] = []
    for doc in docs:
        modality = doc.metadata.get("modality", "")
        fp = doc.metadata.get("file_path", "")
        if modality == Modality.IMAGE.value:
            excerpt = _redact(captions.get(fp) or doc.page_content)
        else:
            excerpt = _redact(doc.page_content[:200])
        sources.append(
            SourceDoc(
                file_name=doc.metadata.get("file_name", ""),
                file_path=fp,
                modality=modality,
                excerpt=excerpt,
            )
        )
    return sources


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Ask HomeIntel a question about your files.

    Retrieves the most relevant chunks from Qdrant, sends them to the local LLM
    as context, and returns the answer with source attribution.
    """
    try:
        chain = get_chain()
        result = chain.run(request.question, request.modality_filter, request.top_k)
    except Exception as exc:
        logger.exception("RAG chain failed for question: %r", request.question)
        raise HTTPException(status_code=503, detail=f"LLM unavailable: {exc}") from exc

    docs = result["docs"]
    sources = build_sources(docs, _captions_for_images(docs))

    return ChatResponse(
        answer=result["answer"],
        sources=sources,
        model=settings.ollama_llm_model,
        chunks_used=len(docs),
    )
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd backend && python ../scripts/test_build_sources.py`
Expected: `ALL PASS`

- [ ] **Step 6: Regression — `/chat` still works end-to-end**

Prereq: Qdrant + Ollama up, test data present (`cd backend && python ../scripts/test_ingestion.py --keep-chunks`).
Run: `cd backend && python ../scripts/verify_api.py`
Expected: `PASSED: 8/8`

- [ ] **Step 7: Commit**

```bash
git add backend/api/chat.py backend/rag/retriever.py scripts/test_build_sources.py
git commit -m "feat: serve full image captions in chat sources"
```

---

### Task 3: RAGChain refactor + `run_stream()`

Extract shared retrieve/prompt logic and add a streaming generator without changing `run()` behavior.

**Files:**
- Modify: `backend/rag/chain.py` (add `_prepare()`, rewrite `run()`, add `run_stream()`)
- Test: `scripts/test_run_stream.py` (new — direct generator smoke; needs Qdrant + Ollama)

**Interfaces:**
- Consumes: `retrieve()`, `SYSTEM_TEMPLATE`, `IMAGE_SEARCH_TEMPLATE`, `Modality` (existing)
- Produces: `RAGChain._prepare(question, modality_filter, top_k) -> tuple[list, list[Document]]` (messages, docs)
- Produces: `RAGChain.run_stream(question, modality_filter=None, top_k=None)` generator yielding `("sources", list[Document])`, then `("token", str)` deltas, then `("done", None)`
- Preserves: `RAGChain.run(...) -> {"answer": str, "docs": list[Document]}`

- [ ] **Step 1: Write the failing test**

Create `scripts/test_run_stream.py`:

```python
"""
scripts/test_run_stream.py — Smoke test for RAGChain.run_stream (needs Qdrant + Ollama).

Run from the backend directory:
    cd backend
    python ../scripts/test_run_stream.py
Exit 0 = pass.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from rag.chain import get_chain


def main():
    chain = get_chain()

    events = list(chain.run_stream("What companies has Jane Doe worked for?"))

    tags = [t for t, _ in events]
    assert tags[0] == "sources", f"first event must be 'sources', got {tags[:3]}"
    assert tags[-1] == "done", f"last event must be 'done', got {tags[-3:]}"
    assert "token" in tags, "expected at least one token event"

    answer = "".join(p for t, p in events if t == "token")
    assert answer.strip(), "streamed answer is empty"

    # run() still returns the same shape
    result = chain.run("What companies has Jane Doe worked for?")
    assert set(result.keys()) == {"answer", "docs"}, result.keys()
    assert isinstance(result["answer"], str) and result["answer"].strip()

    print("PASS — sources→token→done, run() intact")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && python ../scripts/test_run_stream.py`
Expected: FAIL — `AttributeError: 'RAGChain' object has no attribute 'run_stream'`

- [ ] **Step 3: Refactor `chain.py`**

In `backend/rag/chain.py`, replace the `run()` method with `_prepare()` + `run()` + `run_stream()`:

```python
    def _prepare(
        self,
        question: str,
        modality_filter: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> tuple[list, list[Document]]:
        """Retrieve context and build the LLM messages. Shared by run/run_stream."""
        docs: list[Document] = retrieve(question, modality_filter, top_k)

        if docs:
            def _content(doc: Document) -> str:
                text = doc.page_content
                return redact_secrets(text) if settings.redact_secrets else text

            context_parts = [
                f"[Source: {doc.metadata.get('file_name', 'unknown')}]\n{_content(doc)}"
                for doc in docs
            ]
            context = "\n\n---\n\n".join(context_parts)
        else:
            context = "(no relevant context found in your files)"

        is_image_query = bool(docs) and (
            modality_filter == Modality.IMAGE.value
            or all(d.metadata.get("modality") == Modality.IMAGE.value for d in docs)
        )
        template = IMAGE_SEARCH_TEMPLATE if is_image_query else SYSTEM_TEMPLATE

        messages = [
            SystemMessage(content=template.format(context=context)),
            HumanMessage(content=question),
        ]
        return messages, docs

    def run(
        self,
        question: str,
        modality_filter: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> dict:
        """Blocking RAG: returns {"answer": str, "docs": list[Document]}."""
        messages, docs = self._prepare(question, modality_filter, top_k)
        logger.info(
            "Calling LLM (invoke) — model=%s docs=%d question=%r",
            settings.ollama_llm_model, len(docs), question[:80],
        )
        response = self._llm.invoke(messages)
        return {"answer": response.content, "docs": docs}

    def run_stream(
        self,
        question: str,
        modality_filter: Optional[str] = None,
        top_k: Optional[int] = None,
    ):
        """
        Streaming RAG generator. Yields, in order:
          ("sources", list[Document])  — once, before any token
          ("token", str)               — per streamed delta
          ("done", None)               — once, last
        """
        messages, docs = self._prepare(question, modality_filter, top_k)
        yield ("sources", docs)
        logger.info(
            "Calling LLM (stream) — model=%s docs=%d question=%r",
            settings.ollama_llm_model, len(docs), question[:80],
        )
        for chunk in self._llm.stream(messages):
            delta = chunk.content
            if delta:
                yield ("token", delta)
        yield ("done", None)
```

- [ ] **Step 4: Run test to verify it passes**

Prereq: Qdrant + Ollama up, test data present.
Run: `cd backend && python ../scripts/test_run_stream.py`
Expected: `PASS — sources→token→done, run() intact`

- [ ] **Step 5: Regression — `/chat` still passes**

Run: `cd backend && python ../scripts/verify_api.py`
Expected: `PASSED: 8/8`

- [ ] **Step 6: Commit**

```bash
git add backend/rag/chain.py scripts/test_run_stream.py
git commit -m "feat: add run_stream generator to RAG chain"
```

---

### Task 4: `POST /chat/stream` SSE endpoint

Expose `run_stream()` as Server-Sent Events, reusing the Task 2 source shaping.

**Files:**
- Modify: `backend/api/chat.py` (add `_sse()` helper + `POST /chat/stream`)
- Test: `scripts/verify_streaming.py` (new — drives uvicorn, asserts SSE order over HTTP)

**Interfaces:**
- Consumes: `RAGChain.run_stream()` (Task 3), `build_sources()` + `_captions_for_images()` (Task 2)
- Produces: `POST /chat/stream` → `text/event-stream` with frames:
  - `event: sources` / `data: {"sources": SourceDoc[], "model": str, "chunks_used": int}`
  - `event: token` / `data: {"delta": str}`
  - `event: done` / `data: {"model": str}`
  - `event: error` / `data: {"detail": str}`

- [ ] **Step 1: Write the failing test**

Create `scripts/verify_streaming.py`:

```python
"""
scripts/verify_streaming.py — Integration test for POST /chat/stream.

Prereq: test data in Qdrant (cd backend && python ../scripts/test_ingestion.py --keep-chunks)
Run:    cd backend && python ../scripts/verify_streaming.py
Exit 0 = pass. Reuses uvicorn lifecycle helpers from verify_api.py.
"""

import json
import sys
from pathlib import Path

import httpx

SCRIPTS_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPTS_DIR))
from verify_api import _start_uvicorn, _wait_for_ready, _stop_uvicorn, BASE_URL, STARTUP_TIMEOUT

REQUEST_TIMEOUT = 120


def _read_events(question: str, modality_filter=None):
    """POST /chat/stream and return a list of (event, data) tuples."""
    events = []
    payload = {"question": question, "modality_filter": modality_filter}
    with httpx.stream("POST", f"{BASE_URL}/chat/stream", json=payload, timeout=REQUEST_TIMEOUT) as r:
        r.raise_for_status()
        event = None
        data_lines = []
        for line in r.iter_lines():
            if line == "":  # frame boundary
                if event is not None:
                    raw = "\n".join(data_lines)
                    try:
                        events.append((event, json.loads(raw) if raw else None))
                    except json.JSONDecodeError:
                        events.append((event, None))
                event, data_lines = None, []
                continue
            if line.startswith("event:"):
                event = line[len("event:"):].strip()
            elif line.startswith("data:"):
                data_lines.append(line[len("data:"):].strip())
    return events


def check_stream_order() -> bool:
    print("  check_stream_order...", end="", flush=True)
    events = _read_events("What companies has Jane Doe worked for?")
    tags = [e for e, _ in events]
    ok = bool(tags) and tags[0] == "sources" and tags[-1] == "done" and "token" in tags
    answer = "".join(d["delta"] for e, d in events if e == "token" and d)
    ok = ok and bool(answer.strip())
    print(" PASS" if ok else f" FAIL (tags={tags[:3]}..{tags[-2:]})")
    return ok


def check_image_caption_length() -> bool:
    """Image query source caption should exceed 200 chars. SKIP if no image data."""
    print("  check_image_caption_length...", end="", flush=True)
    events = _read_events("people", modality_filter="image")
    src_evt = next((d for e, d in events if e == "sources"), None)
    sources = (src_evt or {}).get("sources", [])
    images = [s for s in sources if s.get("modality") == "image"]
    if not images:
        print(" SKIP (no image data indexed)")
        return True
    longest = max(len(s["excerpt"]) for s in images)
    ok = longest > 200
    print(" PASS" if ok else f" FAIL (longest caption {longest} chars)")
    return ok


def main() -> int:
    print("\n HomeIntel — streaming verification")
    print("-" * 45)
    print("Starting uvicorn...", end="", flush=True)
    proc = _start_uvicorn()
    if not _wait_for_ready(STARTUP_TIMEOUT):
        print(" FAILED")
        _stop_uvicorn(proc)
        return 1
    print(" ready\n")
    try:
        results = [check_stream_order(), check_image_caption_length()]
    finally:
        print("\nShutting down...", end="", flush=True)
        _stop_uvicorn(proc)
        print(" done")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && python ../scripts/verify_streaming.py`
Expected: FAIL — `check_stream_order` fails (endpoint 404 / no events yet)

- [ ] **Step 3: Add the SSE endpoint**

In `backend/api/chat.py`, add imports at the top:

```python
import json
from fastapi.responses import StreamingResponse
```

Then append the helper and endpoint at the end of the file:

```python
def _sse(event: str, data) -> str:
    """Format one Server-Sent Events frame."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    Streaming variant of /chat. Emits SSE frames: a `sources` frame first
    (so the UI can render the carousel immediately), then `token` deltas, then
    `done`. Errors mid-stream are sent as an `error` frame.
    """
    chain = get_chain()

    def gen():
        try:
            for tag, payload in chain.run_stream(
                request.question, request.modality_filter, request.top_k
            ):
                if tag == "sources":
                    docs = payload
                    sources = build_sources(docs, _captions_for_images(docs))
                    yield _sse("sources", {
                        "sources": [s.model_dump() for s in sources],
                        "model": settings.ollama_llm_model,
                        "chunks_used": len(docs),
                    })
                elif tag == "token":
                    yield _sse("token", {"delta": payload})
                elif tag == "done":
                    yield _sse("done", {"model": settings.ollama_llm_model})
        except Exception as exc:
            logger.exception("Streaming chat failed for: %r", request.question)
            yield _sse("error", {"detail": str(exc)})

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
```

- [ ] **Step 4: Run test to verify it passes**

Prereq: Qdrant + Ollama up, test data present.
Run: `cd backend && python ../scripts/verify_streaming.py`
Expected: `check_stream_order PASS`; `check_image_caption_length` PASS or SKIP. `main` returns 0.

- [ ] **Step 5: Confirm `/chat` regression still green**

Run: `cd backend && python ../scripts/verify_api.py`
Expected: `PASSED: 8/8`

- [ ] **Step 6: Commit**

```bash
git add backend/api/chat.py scripts/verify_streaming.py
git commit -m "feat: add POST /chat/stream SSE endpoint"
```

---

### Task 5: Frontend streaming

Consume the SSE stream and render the answer token-by-token; show full captions (no frontend caption change needed — it already maps `excerpt` → carousel caption).

**Files:**
- Modify: `frontend/src/types.ts` (add `streaming?: boolean`)
- Modify: `frontend/src/api.ts` (add `sendChatStream()` + `StreamHandlers`)
- Modify: `frontend/src/App.tsx` (streaming submit flow; remove global loading bubble)
- Modify: `frontend/src/components/ChatMessage.tsx` (in-bubble dots while empty, caret while streaming)
- Modify: `frontend/src/App.css` (streaming caret + inline dots)

**Interfaces:**
- Consumes: `POST /chat/stream` SSE frames (Task 4)
- Produces: `sendChatStream(question: string, modality_filter: string | undefined, top_k: number | undefined, handlers: StreamHandlers): Promise<void>`
- Produces: `StreamHandlers = { onSources?, onToken?, onDone?, onError?, signal? }`

- [ ] **Step 1: Add the `streaming` flag to the message type**

In `frontend/src/types.ts`, add to the `ChatMessage` interface:

```typescript
  streaming?: boolean;
```

- [ ] **Step 2: Add `sendChatStream()` to the API client**

In `frontend/src/api.ts`, add (keep the existing `sendChat` for fallback/tests):

```typescript
export interface StreamHandlers {
  onSources?: (sources: SourceDoc[], meta: { model: string; chunks_used: number }) => void;
  onToken?: (delta: string) => void;
  onDone?: () => void;
  onError?: (detail: string) => void;
  signal?: AbortSignal;
}

function parseFrame(frame: string): { event?: string; data?: any } {
  let event: string | undefined;
  const dataLines: string[] = [];
  for (const line of frame.split("\n")) {
    if (line.startsWith("event:")) event = line.slice(6).trim();
    else if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
  }
  if (dataLines.length === 0) return { event };
  try {
    return { event, data: JSON.parse(dataLines.join("\n")) };
  } catch {
    return { event };
  }
}

export async function sendChatStream(
  question: string,
  modality_filter: string | undefined,
  top_k: number | undefined,
  handlers: StreamHandlers
): Promise<void> {
  const res = await fetch(`${BASE}/chat/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      question,
      modality_filter: modality_filter || null,
      top_k: top_k ?? null,
    }),
    signal: handlers.signal,
  });
  if (!res.ok || !res.body) throw new Error(`API error ${res.status}`);

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    let sep: number;
    while ((sep = buffer.indexOf("\n\n")) !== -1) {
      const frame = buffer.slice(0, sep);
      buffer = buffer.slice(sep + 2);
      const { event, data } = parseFrame(frame);
      if (!event) continue;
      if (event === "sources") {
        handlers.onSources?.(data.sources, { model: data.model, chunks_used: data.chunks_used });
      } else if (event === "token") {
        handlers.onToken?.(data.delta);
      } else if (event === "done") {
        handlers.onDone?.();
      } else if (event === "error") {
        handlers.onError?.(data?.detail ?? "stream error");
      }
    }
  }
}
```

- [ ] **Step 3: Rewrite `submit()` in `App.tsx` to stream**

In `frontend/src/App.tsx`:

1. Add the streaming import — change the import line to:

```typescript
import { sendChatStream, visualSearch } from "./api";
```

2. Add an abort-controller ref next to the other refs (after `libraryInputRef`):

```typescript
  const abortRef = useRef<AbortController | null>(null);
```

3. Replace the entire `submit()` function with:

```typescript
  async function submit() {
    const q = input.trim();
    if (!q || loading) return;

    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    const userMsg: Msg = { id: nextId(), role: "user", content: q };
    const assistantId = nextId();
    const assistantMsg: Msg = { id: assistantId, role: "assistant", content: "", streaming: true };
    setMessages((prev) => [...prev, userMsg, assistantMsg]);
    setInput("");
    setLoading(true);

    const patch = (p: Partial<Msg>) =>
      setMessages((prev) => prev.map((m) => (m.id === assistantId ? { ...m, ...p } : m)));
    const append = (delta: string) =>
      setMessages((prev) =>
        prev.map((m) => (m.id === assistantId ? { ...m, content: m.content + delta } : m))
      );

    try {
      await sendChatStream(q, modality || undefined, topK ?? undefined, {
        signal: controller.signal,
        onSources: (sources, meta) =>
          patch({ sources, model: meta.model, chunks_used: meta.chunks_used }),
        onToken: (delta) => append(delta),
        onDone: () => patch({ streaming: false }),
        onError: () =>
          patch({ content: "Failed to generate a response.", error: true, streaming: false }),
      });
    } catch (e) {
      if ((e as Error).name !== "AbortError") {
        patch({
          content: "Failed to reach the API. Is the backend running?",
          error: true,
          streaming: false,
        });
      }
    } finally {
      setLoading(false);
      setTimeout(() => inputRef.current?.focus(), 0);
    }
  }
```

- [ ] **Step 4: Remove the global loading bubble**

In `frontend/src/App.tsx`, delete the standalone loading block (it is now handled inside the streaming assistant bubble):

```tsx
        {loading && (
          <div className="message message-assistant">
            <div className="bubble bubble-assistant bubble-loading">
              <span className="dot" /><span className="dot" /><span className="dot" />
            </div>
          </div>
        )}
```

- [ ] **Step 5: Show dots/caret in the assistant bubble**

In `frontend/src/components/ChatMessage.tsx`, replace the assistant-branch render (the `else` of `isUser ?`) so it shows dots before any content and a caret while streaming:

```tsx
        {isUser ? (
          <p className="bubble-text">{message.content}</p>
        ) : (
          <div className="bubble-text markdown">
            {message.content && <ReactMarkdown>{message.content}</ReactMarkdown>}
            {message.streaming && !message.content && (
              <span className="stream-dots">
                <span className="dot" /><span className="dot" /><span className="dot" />
              </span>
            )}
            {message.streaming && message.content && <span className="stream-caret" />}
          </div>
        )}
```

- [ ] **Step 6: Add caret + inline-dots CSS**

Append to `frontend/src/App.css`:

```css
/* Streaming indicators */
.stream-dots { display: inline-flex; gap: 5px; align-items: center; }
.stream-caret {
  display: inline-block;
  width: 7px;
  height: 1.05em;
  margin-left: 2px;
  vertical-align: text-bottom;
  background: var(--text-muted);
  animation: caret-blink 1s steps(1) infinite;
}
@keyframes caret-blink { 50% { opacity: 0; } }
```

- [ ] **Step 7: Type-check and build**

Run: `cd frontend && npm run build`
Expected: `tsc` passes (no type errors) and `vite build` completes with no errors.

- [ ] **Step 8: Manual UI verification**

Start backend + frontend (or `npm run dev`). With Qdrant + Ollama up:
- Ask a document question (e.g. "What companies has Jane Doe worked for?") → answer text appears incrementally (dots → streaming text → caret disappears at end); "Sources" list renders.
- Ask an image query (Images filter, e.g. "people") → carousel appears as soon as sources arrive; the per-photo caption is a full multi-sentence description (not cut off mid-word); summary text streams.
Expected: both stream; captions are complete.

- [ ] **Step 9: Commit**

```bash
git add frontend/src/types.ts frontend/src/api.ts frontend/src/App.tsx frontend/src/components/ChatMessage.tsx frontend/src/App.css
git commit -m "feat: stream chat responses in the UI"
```

---

### Task 6: Document the changes (CLAUDE.md Step 14)

Record both features and the operational re-index requirement so future work has the full context.

**Files:**
- Modify: `CLAUDE.md` (add Step 14 after the Step 13 section)

**Interfaces:** none (documentation only)

- [ ] **Step 1: Add the Step 14 section**

In `CLAUDE.md`, immediately after the end of the `### ✅ Step 13 — CLIP Text→Image Search...` section (before `## Key Design Decisions & Rationale`), insert:

```markdown
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
# activate your venv (see README/CLAUDE.md)
cd backend
# Temporarily remove /path/to/photos from WATCHER_EXCLUDE_PATHS in .env first
python ../scripts/reindex.py --path /path/to/photos/originals --ext .jpg .jpeg .png
# Restore WATCHER_EXCLUDE_PATHS after
```
```

- [ ] **Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: document streaming + full-caption changes (Step 14)"
```

---

## Self-Review

**Spec coverage:**
- Streaming endpoint `/chat/stream` (SSE, additive) → Task 4; `run_stream()` → Task 3. ✓
- Streaming applies to document + image queries → same `_prepare()`/`run_stream()` path (Task 3); verified for both in Task 5 Step 8. ✓
- `sources` event first, then tokens, then done → Task 3 yield order + Task 4 frames; asserted in Task 4 `check_stream_order`. ✓
- Redaction unchanged (context + excerpts redacted, tokens raw) → `_prepare()` (Task 3) + `build_sources()` (Task 2). ✓
- `full_caption` stored on chunk 0 at ingestion → Task 1 Step 6. ✓
- `captions_for()` prefers `full_caption`, falls back to overlap-stripped join → Task 1 Step 5. ✓
- Full caption for image sources, 200-char excerpt for others, both endpoints → `build_sources()` (Task 2), reused in `/chat` (Task 2) and `/chat/stream` (Task 4). ✓
- Frontend reads SSE via fetch/ReadableStream, placeholder message, caret/dots → Task 5. ✓
- No frontend caption change needed (SourceList maps excerpt→caption) → confirmed in design; Task 5 has no caption mapping change. ✓
- Tests: streaming order + caption>200 + manual → Tasks 4 & 5. ✓
- Docs Step 14 + re-index step → Task 6. ✓

**Placeholder scan:** No TBD/TODO/"handle edge cases" — every code step contains complete code. ✓

**Type consistency:** `_join_caption_chunks(chunks, overlap)` defined Task 1, used Task 1. `captions_for(file_paths) -> dict[str,str]` defined Task 1, consumed Task 2. `get_vectorstore()` defined Task 2, used Task 2. `build_sources(docs, captions)` / `_captions_for_images(docs)` defined Task 2, used Task 2 + Task 4. `run_stream` event tags `("sources"|"token"|"done")` produced Task 3, consumed Task 4. SSE event names (`sources`/`token`/`done`/`error`) produced Task 4, parsed Task 5. `StreamHandlers` keys match between `sendChatStream` (Task 5 Step 2) and `submit()` (Task 5 Step 3). `streaming?: boolean` defined Task 5 Step 1, used Steps 3/5. ✓

**Deviation from spec:** spec wrote `_prepare → (messages, docs, is_image_query)`; the plan returns `(messages, docs)` because `is_image_query` is only used inside `_prepare` to pick the template — no caller needs it. Functionally identical.
