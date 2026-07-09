# Design Spec — UI Refresh: Header Logo, Photo-Input Split, Voice Dictation, QoL

**Date:** 2026-06-27
**Status:** Approved (design)
**Author:** the project maintainer (with Claude Code)

---

## 1. Summary

A mobile-first polish pass on the HomeIntel PWA, inspired by the Claude mobile app.
Four themes:

1. **Header logo** — put the home-screen app icon next to the "HomeIntel" wordmark.
2. **Photo-input split** — replace the single "Choose from library" button (which
   triggers a confusing redundant iOS picker menu) with two clean tiles: **Camera**
   and **Photos**.
3. **Voice dictation** — a mic button that records speech, transcribes it with the
   existing local Whisper model, and drops the text into the composer.
4. **Quality-of-life** — six small, self-contained improvements (A–F below).

No new external services. No data lost. The only infrastructure note is an
*optional, documented* Tailscale HTTPS step required for the mic to work **on the
phone** (it already works on desktop `localhost`).

---

## 2. Out of Scope (YAGNI)

- Full voice "conversation mode" with spoken replies — we have STT (Whisper) but no
  TTS. Dictation only.
- Multi-conversation history / sidebar list — only the *current* conversation is
  persisted.
- Regenerate-response, message editing, Android-specific tuning.
- Exposing the app to the public internet (Cloudflare tunnel explicitly rejected).

---

## 3. Theme 1 — Header Logo

**What:** Render the existing app icon to the left of the wordmark:

```
[🏠icon] HomeIntel ........................ ● ✎
```

**How:**
- Use the existing `frontend/public/icon-512.png` (the house-with-camera icon used
  for "Add to Home Screen") via `<img src="/icon-512.png">`, sized ~26px with
  `border-radius` for rounded corners. 512 scaled down stays crisp.
- Add a `.logo-mark` style and wrap the existing `.logo` + image in a flex row.

**Files:** `frontend/src/components/Header.tsx`, `frontend/src/App.css`.

**Risk:** Minimal — additive markup + CSS.

---

## 4. Theme 2 — Photo Input: Camera + Photos

**Decision (confirmed):** Two tiles, not three. Web platforms cannot create three
truly-distinct image inputs on iOS — any non-camera image `<input>` makes iOS show
its own "Photo Library / Take Photo / Choose File" menu, which cannot be suppressed.
Only the camera input bypasses it. Two honest tiles is the cleanest result.

**Layout (in the existing "Search options" bottom sheet):**

```
VISUAL SEARCH
┌──────────────┬──────────────┐
│      📷      │      🖼️      │
│   Camera     │    Photos    │
└──────────────┴──────────────┘
```

**Behavior:**
- **Camera** → hidden `<input type="file" accept="image/*" capture="environment">`
  — opens the camera directly.
- **Photos** → hidden `<input type="file" accept="image/*">` — opens the photo
  picker (iOS may surface its native menu; documented limitation).
- Both call the existing `onImageUpload` handler → `POST /visual-search`. **No
  backend change.**

**Implementation notes:**
- Keep both existing hidden inputs (`cameraInputRef`, `libraryInputRef`); only the
  sheet markup changes from two stacked `.sheet-action` buttons to a 2-up tile grid.
- Add `.visual-tiles` / `.visual-tile` styles (mirrors the look of the QoL example
  chips and Claude's "Add to Chat" tiles).

**Files:** `frontend/src/App.tsx`, `frontend/src/App.css`.

---

## 5. Theme 3 — Voice Dictation

**Scope:** Press-to-talk dictation. Record → transcribe → insert text into the
composer for the user to review/edit before sending (**not** auto-send).

### 5.1 Frontend

- **Composer button swap** (also QoL item B): the right-hand pill button is
  - **⏹ Stop** when a response is streaming (`loading === true`),
  - **↑ Send** when the input has non-whitespace text,
  - **🎙️ Mic** when the input is empty and not loading.
- **Recording flow:**
  1. Tap mic → request `getUserMedia({ audio: true })`, start `MediaRecorder`.
  2. While recording: the composer shows a recording state — red dot + running
     `mm:ss` timer, a **cancel** (✕) control, and a **stop/confirm** control.
  3. Tap stop → assemble the recorded `Blob`, POST to `/transcribe`, show a brief
     "Transcribing…" state, then set the composer input to the returned text
     (appending if the field already had text). Focus the input.
  4. Cancel → discard the recording, no upload.
- **New hook** `frontend/src/hooks/useRecorder.ts` encapsulating MediaRecorder
  lifecycle, timer, and permission/error handling. State: `idle | recording |
  transcribing | error`. Returns `{ state, seconds, start, stop, cancel, error }`.
- **Errors:** if `getUserMedia` rejects (permission denied) or the origin is
  insecure (no `mediaDevices`), show a one-line inline message:
  *"Microphone needs HTTPS — see Tailscale setup"* / *"Microphone permission
  denied."* The mic button is still rendered (so the affordance is discoverable);
  the error surfaces on tap.

### 5.2 Backend

- **New endpoint** `POST /transcribe` in `backend/api/transcribe.py`:
  - Accept `UploadFile` (multipart).
  - Write bytes to a temp file. Choose the extension from the upload's content type
    (`audio/mp4`/`audio/x-m4a` → `.m4a`, `audio/webm` → `.webm`, `audio/wav` →
    `.wav`, else `.bin`). faster-whisper decodes via PyAV which sniffs the container
    by content, so the extension is a hint, not load-bearing.
  - Call the existing `transcribe(path)` from
    `backend/ingestion/processors/audio.py`.
  - Return `{ "text": <transcript> }` (empty string if no speech detected).
  - Clean up the temp file in a `finally`.
  - Wrap model/transcribe errors → `HTTPException(500)` with a short detail.
- Register the router in `backend/main.py`.
- **VRAM:** Whisper `base` (~150 MB) lazy-loads on first transcription; co-exists
  with qwen3:14b + nomic + reranker. No budget concern.
- **Reuse:** no change to `audio.py`. The watcher's audio ingestion and this
  endpoint share the same lazy singleton model.

### 5.3 Secure-context prerequisite (documented, non-blocking)

Mic capture requires a secure context (HTTPS or `localhost`). The app is reached
from the iPhone over plain HTTP on a LAN IP, where iOS blocks the mic.

- **Build/verify on desktop `localhost`** — fully testable without any infra change.
- **Recommended phone path: Tailscale Serve (HTTPS).** Chosen over Nginx Proxy
  Manager self-signed (needs a root-CA profile installed per device) and over a
  public Cloudflare tunnel (rejected — no WWW exposure). Tailscale issues a
  publicly-trusted Let's Encrypt cert for the `*.ts.net` MagicDNS name while the
  service stays reachable only inside the tailnet.
- To avoid mixed-content (HTTPS page → HTTP API) and CORS, the doc recommends
  serving the **built frontend and the API under one HTTPS origin** — e.g. mount
  the built `frontend/dist` as static files in FastAPI and `tailscale serve
  https / http://localhost:8000`. (Implementation of the static mount is optional
  and can be a follow-up; the doc captures the steps.)
- Deliverable: a short setup section (in `docs/` and/or CLAUDE.md). No code change
  is required for the mic to light up on the phone once a secure origin exists.

**Files:** `frontend/src/App.tsx`, `frontend/src/hooks/useRecorder.ts`,
`frontend/src/api.ts`, `frontend/src/components/icons.tsx`,
`backend/api/transcribe.py`, `backend/main.py`, docs.

---

## 6. Theme 4 — Quality-of-Life (all six approved)

Each is small and independently shippable.

| # | Item | Behavior | Files |
|---|---|---|---|
| **A** | **Stop generation** | While streaming, the composer's right button is a ⏹ that calls `abortRef.current.abort()` and finalizes the streaming message (`streaming:false`). Reuses the existing `AbortController`. | `App.tsx`, `icons.tsx`, `App.css` |
| **B** | **Mic/Send swap** | Right button = Stop (loading) / Send (has text) / Mic (empty). Implemented as part of Theme 3. | `App.tsx`, `App.css` |
| **C** | **Tappable example prompts** | The empty-state hint lines become buttons; tapping one fills the composer input (does not auto-send) and focuses it. | `App.tsx`, `App.css` |
| **D** | **"Thinking…" indicator** | When the assistant message is `streaming` and `content === ""` (no token yet), render an animated "Thinking…" / dots placeholder in the bubble instead of empty space. | `ChatMessage.tsx`, `App.css` |
| **E** | **Copy answer** | A small copy button on assistant messages (appears on non-streaming assistant bubbles) using `navigator.clipboard.writeText`; shows a brief "Copied" state. | `ChatMessage.tsx`, `icons.tsx`, `App.css` |
| **F** | **Persist current conversation** | Save `messages` to `localStorage` (debounced / on change) and restore on load. New-chat clears storage. | `App.tsx` |

**QoL-F details (persistence correctness):**
- Persist only serializable message fields. **Strip transient `blob:` object URLs**
  (`queryImageUrl`) before saving — they are invalid after reload; the
  visual-search query thumbnail simply won't reappear (acceptable). `visualResults`
  reference NAS `file_path`s served via `/file`, so those *do* survive.
- On restore, force any `streaming: true` flag to `false` (no half-finished stream
  should appear live after reload).
- Storage key e.g. `homeintel.conversation.v1`; wrap read/write in try/catch
  (quota / private-mode safe). Keep the in-memory `msgId` counter monotonic by
  seeding it past the max restored id.

---

## 7. Component / Interface Boundaries

- **`useRecorder` hook** — owns all MediaRecorder/permission/timer state; the
  composer only calls `start/stop/cancel` and reads `state/seconds/error`. Testable
  and replaceable without touching `App.tsx` layout.
- **`POST /transcribe`** — a thin HTTP adapter over the existing `transcribe()`
  function; no Whisper logic duplicated.
- **Photo tiles** — pure markup/CSS over the *existing* hidden inputs and
  `onImageUpload`; no new data flow.
- **Persistence** — isolated to load/save effects in `App.tsx` operating on the
  `messages` array; no change to message rendering.

---

## 8. Testing / Verification

- **Backend `/transcribe`:** a script (e.g. `scripts/test_transcribe.py`) posting a
  short sample audio file and asserting a non-empty `text`; plus a manual curl.
  Reuses the existing verify-script pattern (note: stop `HomeIntel-API` first — the
  verify scripts need port 8000, per project memory).
- **Frontend:** manual verification on desktop `localhost` (secure context) for the
  full mic round-trip; visual check of tiles, header logo, Stop button, example
  prompts, Thinking indicator, Copy, and persistence (reload survives chat,
  New-chat clears it).
- **Photo tiles & visual search:** unchanged backend path — smoke-test that Camera
  and Photos both reach `/visual-search`.

---

## 9. Files Touched (summary)

**Frontend**
- `src/components/Header.tsx` — logo image
- `src/App.tsx` — tiles, mic/send/stop swap, example prompts, persistence
- `src/App.css` — tiles, logo, recording state, thinking, copy, example-prompt styles
- `src/api.ts` — `transcribe()` client
- `src/components/icons.tsx` — Mic, Stop, Copy icons
- `src/components/ChatMessage.tsx` — Thinking indicator, Copy button
- `src/hooks/useRecorder.ts` — **new** MediaRecorder hook

**Backend**
- `api/transcribe.py` — **new** `POST /transcribe`
- `main.py` — register transcribe router

**Docs**
- Tailscale HTTPS setup note (+ CLAUDE.md "Step 15" entry summarizing this work)

---

## 10. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Mic blocked on phone (insecure origin) | Expected; documented Tailscale path; build/verify on localhost; mic affordance shows a helpful error rather than vanishing |
| iOS vs Chrome audio container differences | PyAV/faster-whisper decode both `audio/mp4` and `audio/webm`; pick extension from MIME |
| `blob:` URLs break on reload | Strip before persisting; only NAS-served `file_path`s persist |
| Whisper first-call latency | `base` model, lazy-load once; show "Transcribing…" state |
| localStorage quota / private mode | try/catch around read/write; degrade to in-memory only |
