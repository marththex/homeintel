# UI Refresh (Header Logo, Photo-Input Split, Voice Dictation, QoL) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Polish the HomeIntel PWA — app icon in the header, a clean Camera/Photos photo-input split, voice dictation via the existing local Whisper model, and six small quality-of-life improvements.

**Architecture:** Frontend-heavy React/TypeScript changes plus one new backend endpoint (`POST /transcribe`) that reuses the existing faster-whisper singleton. Voice recording is a self-contained `useRecorder` hook; transcription text is inserted into the composer for review (never auto-sent). Conversation state is persisted to `localStorage`. Mic capture requires a secure context, so it is built/verified on desktop `localhost` and documented for phone use via Tailscale Serve HTTPS.

**Tech Stack:** React 19 + Vite + TypeScript (frontend, no test runner — `npm run build`/`tsc` is the gate); FastAPI + faster-whisper (backend, standalone `scripts/test_*.py` tests); conda env `homeintel`.

## Global Constraints

- **Icons:** inline SVG in `frontend/src/components/icons.tsx`. Do NOT use `lucide-react` even though it is installed (Step 10 convention).
- **Frontend verification gate:** there is no frontend unit-test runner. Each frontend task is verified by `cd frontend && npm run build` (must succeed: `tsc` typecheck + `vite build`) PLUS the task's manual checklist run via `npm run dev` at `http://localhost:5173`.
- **Backend tests:** standalone scripts in `scripts/`, run from `backend/` with `conda activate homeintel`, as `python ../scripts/test_X.py`; "ALL PASS" / exit 0 = pass.
- **Port 8000:** stop the `HomeIntel-API` scheduled task before running anything that binds port 8000 (`Stop-ScheduledTask "HomeIntel-API"`).
- **Commits:** no `Co-Authored-By` / AI-attribution trailers.
- **Transcription must NOT auto-send** — insert text into the composer for review.
- **Photo input is exactly 2 tiles** (Camera + Photos). Keep both existing hidden `<input>` elements. No backend change to `/visual-search`.
- **Stop generation** reuses the existing `abortRef` `AbortController` — do not add a second controller.
- **Persistence** must strip `blob:` URLs (`queryImageUrl`) and force `streaming:false` on load.
- **Whisper** is the existing lazy singleton in `backend/ingestion/processors/audio.py` — do not add a new model loader.

---

## File Structure

**Frontend**
- `src/components/Header.tsx` — add app-icon image left of the wordmark (Task 1)
- `src/App.css` — all new styles (Tasks 1,2,5,6,7)
- `src/App.tsx` — photo tiles, mic/send/stop swap, recording UI, example prompts, persistence (Tasks 2,5,6,8)
- `src/components/icons.tsx` — Mic, Stop, Copy icons (Task 4)
- `src/api.ts` — `transcribe()` client (Task 4)
- `src/hooks/useRecorder.ts` — **new** MediaRecorder hook (Task 4)
- `src/components/ChatMessage.tsx` — Thinking label, Copy button (Task 7)
- `src/storage.ts` — **new** conversation persistence helpers (Task 8)

**Backend**
- `api/transcribe.py` — **new** `POST /transcribe` (Task 3)
- `main.py` — register transcribe router (Task 3)
- `scripts/test_transcribe.py` — **new** unit test for the MIME→ext helper (Task 3)

**Docs**
- `docs/tailscale-https-setup.md` — **new** secure-origin setup (Task 9)
- `CLAUDE.md` — Step 15 entry (Task 9)

---

## Task 1: Header app icon

**Files:**
- Modify: `frontend/src/components/Header.tsx`
- Modify: `frontend/src/App.css`

**Interfaces:**
- Consumes: existing `frontend/public/icon-512.png` (the house+camera home-screen icon).
- Produces: `.logo-group` / `.logo-mark` CSS classes (no JS exports).

- [ ] **Step 1: Add the icon image to the header**

In `frontend/src/components/Header.tsx`, replace the single `<h1 className="logo">` with a flex group containing the icon and the wordmark:

```tsx
  return (
    <header className="header">
      <div className="logo-group">
        <img src="/icon-512.png" alt="" className="logo-mark" />
        <h1 className="logo">HomeIntel</h1>
      </div>
      <div className="header-actions">
        <button
          className="health-pill"
          onClick={onStatusClick}
          title="System status"
          aria-label="System status"
        >
          <span className={`health-dot ${healthy ? "ok" : "bad"}`} />
        </button>
        <button
          className="icon-btn"
          onClick={onNewChat}
          title="New chat"
          aria-label="New chat"
        >
          <ComposeIcon />
        </button>
      </div>
    </header>
  );
```

- [ ] **Step 2: Add the logo-group styles**

In `frontend/src/App.css`, immediately after the `.logo { ... }` rule (around line 58), add:

```css
.logo-group { display: flex; align-items: center; gap: 8px; }
.logo-mark { width: 26px; height: 26px; border-radius: 7px; display: block; flex-shrink: 0; }
```

- [ ] **Step 3: Build / typecheck**

Run: `cd frontend && npm run build`
Expected: completes with no TypeScript or Vite errors.

- [ ] **Step 4: Manual verify**

Run `npm run dev`, open `http://localhost:5173`. Confirm the house+camera icon appears immediately left of "HomeIntel", vertically centered, rounded corners, not distorted.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/Header.tsx frontend/src/App.css
git commit -m "feat(ui): app icon in header"
```

---

## Task 2: Photo input — Camera + Photos tiles

**Files:**
- Modify: `frontend/src/App.tsx` (the "Search options" sheet markup; ~lines 245-252)
- Modify: `frontend/src/App.css`

**Interfaces:**
- Consumes: existing `cameraInputRef`, `libraryInputRef`, `onImageUpload`, `CameraIcon`, `ImageIcon`.
- Produces: `.visual-tiles` / `.visual-tile` CSS classes.

- [ ] **Step 1: Replace the two stacked buttons with a 2-tile grid**

In `frontend/src/App.tsx`, inside the `"+" action sheet` `BottomSheet`, replace:

```tsx
        <div className="sheet-section-label">Visual search</div>
        <button className="sheet-action" onClick={() => cameraInputRef.current?.click()}>
          <CameraIcon /> <span>Take a photo</span>
        </button>
        <button className="sheet-action" onClick={() => libraryInputRef.current?.click()}>
          <ImageIcon /> <span>Choose from library</span>
        </button>
```

with:

```tsx
        <div className="sheet-section-label">Visual search</div>
        <div className="visual-tiles">
          <button className="visual-tile" onClick={() => cameraInputRef.current?.click()}>
            <CameraIcon size={26} />
            <span>Camera</span>
          </button>
          <button className="visual-tile" onClick={() => libraryInputRef.current?.click()}>
            <ImageIcon size={26} />
            <span>Photos</span>
          </button>
        </div>
```

(Leave the two hidden `<input>` elements unchanged.)

- [ ] **Step 2: Add the tile styles**

In `frontend/src/App.css`, after the `.sheet-action svg { ... }` rule (around line 511), add:

```css
.visual-tiles { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
.visual-tile {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 8px;
  background: var(--surface2);
  border: 1px solid var(--border);
  color: var(--text);
  border-radius: 12px;
  padding: 20px 12px;
  font-size: 0.9rem;
  cursor: pointer;
  min-height: 88px;
  -webkit-tap-highlight-color: transparent;
  transition: border-color 0.15s;
}
.visual-tile:hover { border-color: var(--accent); }
.visual-tile svg { color: var(--accent-hover); }
```

- [ ] **Step 3: Build / typecheck**

Run: `cd frontend && npm run build`
Expected: no errors.

- [ ] **Step 4: Manual verify**

In `npm run dev`: tap `+` → the "Search options" sheet shows two side-by-side tiles "Camera" and "Photos". Tapping Camera opens the camera input; tapping Photos opens the image picker. (Desktop browsers will show a file dialog for both — that's expected; the iOS distinction is the camera-direct behavior.)

- [ ] **Step 5: Commit**

```bash
git add frontend/src/App.tsx frontend/src/App.css
git commit -m "feat(ui): split photo input into Camera + Photos tiles"
```

---

## Task 3: Backend `POST /transcribe`

**Files:**
- Create: `backend/api/transcribe.py`
- Modify: `backend/main.py`
- Test: `scripts/test_transcribe.py`

**Interfaces:**
- Consumes: `transcribe(path: Path) -> tuple[str, str]` from `backend/ingestion/processors/audio.py`.
- Produces:
  - `_ext_for_content_type(content_type: str | None) -> str` (pure helper, unit-tested)
  - `POST /transcribe` accepting multipart field `file`, returning JSON `{"text": str}`.

- [ ] **Step 1: Write the failing test**

Create `scripts/test_transcribe.py`:

```python
"""
scripts/test_transcribe.py — Unit test for _ext_for_content_type (no model load).

Run from the backend directory:
    cd backend
    python ../scripts/test_transcribe.py
Exit 0 = all pass.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from api.transcribe import _ext_for_content_type


def test_ios_mp4():
    assert _ext_for_content_type("audio/mp4") == ".m4a"

def test_chrome_webm_with_codecs():
    assert _ext_for_content_type("audio/webm;codecs=opus") == ".webm"

def test_wav():
    assert _ext_for_content_type("audio/wav") == ".wav"

def test_unknown_falls_back():
    assert _ext_for_content_type("application/octet-stream") == ".bin"

def test_none_falls_back():
    assert _ext_for_content_type(None) == ".bin"


if __name__ == "__main__":
    test_ios_mp4()
    test_chrome_webm_with_codecs()
    test_wav()
    test_unknown_falls_back()
    test_none_falls_back()
    print("ALL PASS")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda activate homeintel && cd backend && python ../scripts/test_transcribe.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'api.transcribe'`.

- [ ] **Step 3: Create the endpoint module**

Create `backend/api/transcribe.py`:

```python
"""
api/transcribe.py — Speech-to-text endpoint for voice dictation.

POST /transcribe   multipart audio blob  ->  {"text": "..."}

Reuses the existing faster-whisper singleton from ingestion.processors.audio.
The recorded blob is written to a temp file, transcribed, and the temp file is
always removed afterward.
"""

import logging
import tempfile
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, File

from ingestion.processors.audio import transcribe

logger = logging.getLogger(__name__)
router = APIRouter()

# Browser MediaRecorder MIME types -> sensible temp-file extension. faster-whisper
# decodes via PyAV which sniffs the container by content, so the extension is a
# hint only — but a correct one avoids any ambiguity.
_EXT_BY_TYPE = {
    "audio/mp4": ".m4a",
    "audio/x-m4a": ".m4a",
    "audio/aac": ".m4a",
    "audio/webm": ".webm",
    "audio/ogg": ".ogg",
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "audio/mpeg": ".mp3",
}


def _ext_for_content_type(content_type: str | None) -> str:
    """Return a temp-file extension for a recorded-audio MIME type."""
    if not content_type:
        return ".bin"
    base = content_type.split(";", 1)[0].strip().lower()
    return _EXT_BY_TYPE.get(base, ".bin")


@router.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    """Transcribe an uploaded audio blob to text via faster-whisper."""
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty audio upload")

    suffix = _ext_for_content_type(file.content_type)
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(data)
            tmp_path = Path(tmp.name)
        text, _title = transcribe(tmp_path)
        return {"text": text}
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Transcription failed")
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && python ../scripts/test_transcribe.py`
Expected: `ALL PASS`.

- [ ] **Step 5: Register the router**

In `backend/main.py`, add the import next to the other API router imports:

```python
from api.transcribe import router as transcribe_router
```

and after `app.include_router(status_router)` add:

```python
app.include_router(transcribe_router)
```

- [ ] **Step 6: Integration smoke check (manual)**

Stop the running API first: `Stop-ScheduledTask "HomeIntel-API"`. Then from `backend/`:

```bash
conda activate homeintel
uvicorn main:app --host 0.0.0.0 --port 8000
```

In another shell, post a short real audio file (any `.m4a`/`.webm`/`.wav` with speech):

```bash
curl -s -F "file=@sample.m4a;type=audio/mp4" http://localhost:8000/transcribe
```

Expected: HTTP 200 with `{"text":"<your spoken words>"}`. (First call loads the Whisper model — may take a few seconds. An empty/no-speech clip correctly returns `{"text":""}`.) Stop uvicorn when done.

- [ ] **Step 7: Commit**

```bash
git add backend/api/transcribe.py backend/main.py scripts/test_transcribe.py
git commit -m "feat(api): POST /transcribe speech-to-text via faster-whisper"
```

---

## Task 4: Voice primitives — icons, API client, recorder hook

**Files:**
- Modify: `frontend/src/components/icons.tsx`
- Modify: `frontend/src/api.ts`
- Create: `frontend/src/hooks/useRecorder.ts`

**Interfaces:**
- Produces:
  - `MicIcon`, `StopIcon`, `CopyIcon` (React components, same `IconProps` as existing icons)
  - `transcribe(blob: Blob) => Promise<string>` in `api.ts`
  - `useRecorder()` returning `{ state: "idle"|"recording"|"error"; seconds: number; error: string|null; start(): Promise<void>; stop(): Promise<Blob|null>; cancel(): void }`

- [ ] **Step 1: Add icons**

In `frontend/src/components/icons.tsx`, append before the final line:

```tsx
export const MicIcon = ({ size = 22, className }: IconProps) => (
  <svg {...base(size, className)}>
    <path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3z" />
    <path d="M19 10v2a7 7 0 0 1-14 0v-2" />
    <path d="M12 19v3" />
  </svg>
);

export const StopIcon = ({ size = 20, className }: IconProps) => (
  <svg {...base(size, className)}><rect x="6" y="6" width="12" height="12" rx="2" /></svg>
);

export const CopyIcon = ({ size = 16, className }: IconProps) => (
  <svg {...base(size, className)}>
    <rect x="9" y="9" width="13" height="13" rx="2" ry="2" />
    <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
  </svg>
);
```

- [ ] **Step 2: Add the transcribe API client**

In `frontend/src/api.ts`, after the `visualSearch` function, add:

```ts
export async function transcribe(blob: Blob): Promise<string> {
  const form = new FormData();
  const ext = blob.type.includes("webm")
    ? "webm"
    : blob.type.includes("mp4") || blob.type.includes("aac")
      ? "m4a"
      : "audio";
  // Passing the Blob sets the multipart part's Content-Type from blob.type,
  // which the backend reads to choose a container extension.
  form.append("file", blob, `recording.${ext}`);
  const res = await fetch(`${BASE}/transcribe`, { method: "POST", body: form });
  if (!res.ok) throw new Error(`Transcribe error ${res.status}`);
  const data = await res.json();
  return data.text as string;
}
```

- [ ] **Step 3: Create the recorder hook**

Create `frontend/src/hooks/useRecorder.ts`:

```ts
import { useCallback, useRef, useState } from "react";

export type RecorderState = "idle" | "recording" | "error";

export interface UseRecorder {
  state: RecorderState;
  seconds: number;
  error: string | null;
  start: () => Promise<void>;
  stop: () => Promise<Blob | null>;
  cancel: () => void;
}

function pickMimeType(): string | undefined {
  if (typeof MediaRecorder === "undefined") return undefined;
  const candidates = ["audio/webm;codecs=opus", "audio/webm", "audio/mp4", "audio/aac"];
  return candidates.find((t) => MediaRecorder.isTypeSupported(t));
}

export function useRecorder(): UseRecorder {
  const [state, setState] = useState<RecorderState>("idle");
  const [seconds, setSeconds] = useState(0);
  const [error, setError] = useState<string | null>(null);

  const recorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<BlobPart[]>([]);
  const streamRef = useRef<MediaStream | null>(null);
  const timerRef = useRef<number | null>(null);
  const cancelledRef = useRef(false);
  const resolveRef = useRef<((b: Blob | null) => void) | null>(null);

  const cleanup = useCallback(() => {
    if (timerRef.current !== null) { clearInterval(timerRef.current); timerRef.current = null; }
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    recorderRef.current = null;
    chunksRef.current = [];
  }, []);

  const start = useCallback(async () => {
    setError(null);
    if (!navigator.mediaDevices?.getUserMedia || typeof MediaRecorder === "undefined") {
      setState("error");
      setError("Microphone needs HTTPS — see Tailscale setup.");
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;
      const mimeType = pickMimeType();
      const rec = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
      chunksRef.current = [];
      cancelledRef.current = false;
      rec.ondataavailable = (e) => { if (e.data.size > 0) chunksRef.current.push(e.data); };
      rec.onstop = () => {
        if (timerRef.current !== null) { clearInterval(timerRef.current); timerRef.current = null; }
        const blob = cancelledRef.current
          ? null
          : new Blob(chunksRef.current, { type: rec.mimeType || "audio/webm" });
        cleanup();
        setState("idle");
        resolveRef.current?.(blob);
        resolveRef.current = null;
      };
      recorderRef.current = rec;
      rec.start();
      setSeconds(0);
      setState("recording");
      timerRef.current = window.setInterval(() => setSeconds((s) => s + 1), 1000);
    } catch (e) {
      cleanup();
      setState("error");
      setError(
        (e as Error).name === "NotAllowedError"
          ? "Microphone permission denied."
          : "Could not start recording."
      );
    }
  }, [cleanup]);

  const stop = useCallback(() => {
    return new Promise<Blob | null>((resolve) => {
      const rec = recorderRef.current;
      if (!rec || rec.state === "inactive") { resolve(null); return; }
      cancelledRef.current = false;
      resolveRef.current = resolve;
      rec.stop();
    });
  }, []);

  const cancel = useCallback(() => {
    const rec = recorderRef.current;
    cancelledRef.current = true;
    if (rec && rec.state !== "inactive") {
      rec.stop();
    } else {
      cleanup();
      setState("idle");
    }
  }, [cleanup]);

  return { state, seconds, error, start, stop, cancel };
}
```

- [ ] **Step 4: Build / typecheck**

Run: `cd frontend && npm run build`
Expected: no errors. (TypeScript DOM lib already includes `MediaRecorder` / `MediaStream` types.)

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/icons.tsx frontend/src/api.ts frontend/src/hooks/useRecorder.ts
git commit -m "feat(ui): voice primitives — mic/stop/copy icons, transcribe client, recorder hook"
```

---

## Task 5: Composer — mic/send/stop swap, recording UI, stop-generation (QoL-A, QoL-B, voice)

**Files:**
- Modify: `frontend/src/App.tsx`
- Modify: `frontend/src/App.css`

**Interfaces:**
- Consumes: `useRecorder` (Task 4), `transcribe` (Task 4), `MicIcon`/`StopIcon` (Task 4), existing `SendIcon`/`PlusIcon`/`CloseIcon`, existing `abortRef`/`loading`/`input`/`setMessages`.
- Produces: composer behavior — right button is Stop (streaming) / Send (has text) / Mic (empty); recording bar; `voice-error` display.

- [ ] **Step 1: Update imports and add recorder state/handlers**

In `frontend/src/App.tsx`, update the icons import (line 6) to include `MicIcon` and `StopIcon`:

```tsx
import { PlusIcon, SendIcon, CameraIcon, ImageIcon, CloseIcon, MicIcon, StopIcon } from "./components/icons";
```

Add to the imports near the top:

```tsx
import { sendChatStream, visualSearch, transcribe } from "./api";
import { useRecorder } from "./hooks/useRecorder";
```

(Replace the existing `import { sendChatStream, visualSearch } from "./api";` line.)

Inside the component, after the `const { health, stats } = useSystemStatus();` line, add:

```tsx
  const recorder = useRecorder();
  const [transcribing, setTranscribing] = useState(false);
  const [transcribeError, setTranscribeError] = useState<string | null>(null);
  const voiceError = recorder.error ?? transcribeError;
```

After the `submit` function, add the helper functions:

```tsx
  function stopGeneration() {
    abortRef.current?.abort();
    setMessages((prev) => prev.map((m) => (m.streaming ? { ...m, streaming: false } : m)));
    setLoading(false);
  }

  async function startRecording() {
    setTranscribeError(null);
    await recorder.start();
  }

  async function finishRecording() {
    const blob = await recorder.stop();
    if (!blob) return;
    setTranscribing(true);
    try {
      const text = await transcribe(blob);
      if (text) {
        setInput((prev) => (prev ? prev.trimEnd() + " " : "") + text);
        setTimeout(() => inputRef.current?.focus(), 0);
      }
    } catch {
      setTranscribeError("Transcription failed.");
    } finally {
      setTranscribing(false);
    }
  }

  function formatTime(s: number) {
    const m = Math.floor(s / 60);
    return `${m}:${(s % 60).toString().padStart(2, "0")}`;
  }
```

- [ ] **Step 2: Add the voice-error line above the pill**

In `frontend/src/App.tsx`, inside `<footer className="composer">`, immediately before the `{(modality || topK !== null) && (` chip-row block, add:

```tsx
        {voiceError && <div className="voice-error">{voiceError}</div>}
```

- [ ] **Step 3: Replace the composer pill with the recording-aware version**

Replace the entire `<div className="composer-pill"> ... </div>` block with:

```tsx
        <div className="composer-pill">
          {recorder.state === "recording" ? (
            <>
              <button
                className="pill-btn cancel-btn"
                onClick={recorder.cancel}
                aria-label="Cancel recording"
              >
                <CloseIcon />
              </button>
              <div className="recording-indicator">
                <span className="rec-dot" />
                <span className="rec-time">{formatTime(recorder.seconds)}</span>
                <span className="rec-label">Recording…</span>
              </div>
              <button
                className="pill-btn send-btn"
                onClick={finishRecording}
                aria-label="Stop and transcribe"
              >
                <StopIcon />
              </button>
            </>
          ) : (
            <>
              <button
                className="pill-btn plus-btn"
                onClick={() => setActionSheet(true)}
                disabled={loading || transcribing}
                aria-label="Attach or filter"
              >
                <PlusIcon />
              </button>
              <textarea
                ref={inputRef}
                className="composer-input"
                rows={1}
                placeholder={transcribing ? "Transcribing…" : "Ask about your files…"}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={onKeyDown}
                onFocus={() => setTimeout(() => bottomRef.current?.scrollIntoView({ behavior: "smooth" }), 300)}
                disabled={loading || transcribing}
                autoComplete="off"
                autoCorrect="off"
                autoCapitalize="sentences"
                spellCheck={false}
              />
              {loading ? (
                <button
                  className="pill-btn stop-btn"
                  onClick={stopGeneration}
                  aria-label="Stop generating"
                >
                  <StopIcon />
                </button>
              ) : input.trim() ? (
                <button
                  className="pill-btn send-btn"
                  onClick={submit}
                  aria-label="Send"
                >
                  <SendIcon />
                </button>
              ) : (
                <button
                  className="pill-btn mic-btn"
                  onClick={startRecording}
                  disabled={transcribing}
                  aria-label="Record voice"
                >
                  <MicIcon />
                </button>
              )}
            </>
          )}
        </div>
```

- [ ] **Step 4: Add composer styles**

In `frontend/src/App.css`, after the `.send-btn:disabled { ... }` rule (around line 414), add:

```css
.mic-btn { background: transparent; color: var(--text-muted); }
.mic-btn:hover:not(:disabled) { background: var(--border); color: var(--text); }
.mic-btn:disabled { opacity: 0.4; cursor: not-allowed; }

.stop-btn { background: var(--accent); color: #fff; }
.stop-btn:hover:not(:disabled) { background: var(--accent-hover); }

.cancel-btn { background: transparent; color: var(--red); }
.cancel-btn:hover { background: var(--border); }

.recording-indicator {
  flex: 1;
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 0 8px;
  font-size: 0.9rem;
  color: var(--text);
}
.rec-dot {
  width: 10px; height: 10px; border-radius: 50%;
  background: var(--red);
  flex-shrink: 0;
  animation: rec-pulse 1.2s infinite ease-in-out;
}
@keyframes rec-pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.3; } }
.rec-time { font-variant-numeric: tabular-nums; }
.rec-label { color: var(--text-muted); }

.voice-error {
  font-size: 0.78rem;
  color: var(--red);
  margin-bottom: 8px;
  padding: 0 4px;
}
```

- [ ] **Step 5: Build / typecheck**

Run: `cd frontend && npm run build`
Expected: no errors.

- [ ] **Step 6: Manual verify (localhost — secure context)**

In `npm run dev` at `http://localhost:5173`:
1. Empty composer shows the **mic** button on the right.
2. Type text → it becomes the **send** (↑) button; clear text → back to mic.
3. Tap mic → browser prompts for mic permission → pill shows red pulsing dot + timer + "Recording…", with cancel (✕) on the left and stop (■) on the right.
4. Speak, tap stop → placeholder shows "Transcribing…" briefly → recognized text appears in the input (not auto-sent). Edit and send works.
5. Tap mic then cancel (✕) → returns to idle, no upload, no text.
6. Send a normal question; while it streams, the right button is a **stop** (■) — tapping it halts the stream and leaves the partial answer.

- [ ] **Step 7: Commit**

```bash
git add frontend/src/App.tsx frontend/src/App.css
git commit -m "feat(ui): mic/send/stop composer swap, recording UI, stop-generation"
```

---

## Task 6: QoL-C — tappable example prompts

**Files:**
- Modify: `frontend/src/App.tsx`
- Modify: `frontend/src/App.css`

**Interfaces:**
- Consumes: existing `setInput`, `inputRef`.
- Produces: `EXAMPLE_PROMPTS` constant, `.example-prompts` / `.example-prompt` styles.

- [ ] **Step 1: Add the prompts constant and fill handler**

In `frontend/src/App.tsx`, after the `MODALITY_OPTIONS` constant (around line 16), add:

```tsx
const EXAMPLE_PROMPTS = [
  "What does my resume say about Python experience?",
  "What port does the photo service run on?",
  "Summarize my wedding contract.",
];
```

Inside the component, after `newChat`, add:

```tsx
  function fillExample(p: string) {
    setInput(p);
    setTimeout(() => inputRef.current?.focus(), 0);
  }
```

- [ ] **Step 2: Render tappable prompts in the empty state**

Replace the empty-state block:

```tsx
        {messages.length === 0 && (
          <div className="empty-state">
            <p>Ask anything about your NAS files.</p>
            <p className="empty-hint">Try "What does my resume say about Python experience?"</p>
            <p className="empty-hint">Tap + to search by photo.</p>
          </div>
        )}
```

with:

```tsx
        {messages.length === 0 && (
          <div className="empty-state">
            <p>Ask anything about your NAS files.</p>
            <div className="example-prompts">
              {EXAMPLE_PROMPTS.map((p) => (
                <button key={p} className="example-prompt" onClick={() => fillExample(p)}>
                  {p}
                </button>
              ))}
            </div>
            <p className="empty-hint">Tap + to search by photo.</p>
          </div>
        )}
```

- [ ] **Step 3: Add styles**

In `frontend/src/App.css`, after the `.empty-hint { ... }` rule (around line 117), add:

```css
.example-prompts {
  display: flex;
  flex-direction: column;
  gap: 8px;
  margin: 16px auto;
  max-width: 320px;
}
.example-prompt {
  background: var(--surface);
  border: 1px solid var(--border);
  color: var(--text);
  border-radius: 12px;
  padding: 12px 14px;
  font-size: 0.85rem;
  font-style: normal;
  text-align: left;
  cursor: pointer;
  -webkit-tap-highlight-color: transparent;
  transition: border-color 0.15s;
}
.example-prompt:hover { border-color: var(--accent); }
```

- [ ] **Step 4: Build / typecheck**

Run: `cd frontend && npm run build`
Expected: no errors.

- [ ] **Step 5: Manual verify**

On a fresh load (empty chat), the three example prompts render as tappable cards. Tapping one fills the composer (not sent) and focuses it.

- [ ] **Step 6: Commit**

```bash
git add frontend/src/App.tsx frontend/src/App.css
git commit -m "feat(ui): tappable example prompts in empty state"
```

---

## Task 7: QoL-D (thinking label) + QoL-E (copy button)

**Files:**
- Modify: `frontend/src/components/ChatMessage.tsx`
- Modify: `frontend/src/App.css`

**Note:** The animated "thinking" dots **already exist** ([ChatMessage.tsx:31-35](frontend/src/components/ChatMessage.tsx#L31-L35)). QoL-D is therefore a small enhancement — add a "Thinking…" text label beside the existing dots — not a rebuild.

**Interfaces:**
- Consumes: `CopyIcon` (Task 4), existing `message` prop, `useState`.
- Produces: `.thinking` / `.thinking-label` / `.copy-btn` styles; inline `CopyButton` component.

- [ ] **Step 1: Update imports**

In `frontend/src/components/ChatMessage.tsx`, change the React import line (add `useState`) and import `CopyIcon`:

```tsx
import { useState } from "react";
import ReactMarkdown from "react-markdown";
import { SourceList } from "./SourceList";
import { PhotoCarousel, type CarouselPhoto } from "./PhotoCarousel";
import { CopyIcon } from "./icons";
import type { ChatMessage as Msg } from "../types";
```

- [ ] **Step 2: Add the CopyButton component**

At the bottom of `frontend/src/components/ChatMessage.tsx` (after the `ChatMessage` function), add:

```tsx
function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);
  async function copy() {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      /* clipboard unavailable (insecure context) — ignore */
    }
  }
  return (
    <button className="copy-btn" onClick={copy} aria-label="Copy answer">
      <CopyIcon size={14} /> <span>{copied ? "Copied" : "Copy"}</span>
    </button>
  );
}
```

- [ ] **Step 3: Add the Thinking label and Copy button into the bubble**

In the assistant branch, replace the streaming-dots block:

```tsx
            {message.streaming && !message.content && (
              <span className="stream-dots">
                <span className="dot" /><span className="dot" /><span className="dot" />
              </span>
            )}
```

with:

```tsx
            {message.streaming && !message.content && (
              <span className="thinking">
                <span className="stream-dots">
                  <span className="dot" /><span className="dot" /><span className="dot" />
                </span>
                <span className="thinking-label">Thinking…</span>
              </span>
            )}
```

Then add the copy button after the `bubble-meta` block (after the `{!isUser && message.chunks_used !== undefined && (...)}` block, still inside the `bubble` div):

```tsx
        {!isUser && !message.streaming && message.content && (
          <CopyButton text={message.content} />
        )}
```

- [ ] **Step 4: Add styles**

In `frontend/src/App.css`, after the streaming-indicator rules (around line 574, after the `@keyframes caret-blink` rule), add:

```css
.thinking { display: inline-flex; align-items: center; gap: 8px; }
.thinking-label { color: var(--text-muted); font-size: 0.85rem; }

.copy-btn {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  margin-top: 8px;
  background: none;
  border: none;
  color: var(--text-muted);
  font-size: 0.74rem;
  cursor: pointer;
  padding: 2px 0;
  -webkit-tap-highlight-color: transparent;
}
.copy-btn:hover { color: var(--text); }
```

- [ ] **Step 5: Build / typecheck**

Run: `cd frontend && npm run build`
Expected: no errors.

- [ ] **Step 6: Manual verify**

Send a question: before the first token the bubble shows the dots **and** a "Thinking…" label. After the answer finishes streaming, a small "Copy" button appears at the bottom of the assistant bubble; tapping it copies the answer text and briefly shows "Copied". (The Copy button does not appear while streaming.)

- [ ] **Step 7: Commit**

```bash
git add frontend/src/components/ChatMessage.tsx frontend/src/App.css
git commit -m "feat(ui): thinking label + copy-answer button"
```

---

## Task 8: QoL-F — persist current conversation

**Files:**
- Create: `frontend/src/storage.ts`
- Modify: `frontend/src/App.tsx`

**Interfaces:**
- Consumes: `ChatMessage` type, existing `messages`/`setMessages`/`loading`, module-level `msgId`.
- Produces: `saveMessages`, `loadMessages`, `clearMessages` in `storage.ts`.

- [ ] **Step 1: Create the storage helper**

Create `frontend/src/storage.ts`:

```ts
import type { ChatMessage } from "./types";

const KEY = "homeintel.conversation.v1";

// Persist only serializable fields. Strips transient blob: object URLs
// (queryImageUrl — invalid after reload) and the streaming flag.
export function saveMessages(messages: ChatMessage[]): void {
  try {
    const safe = messages.map(({ queryImageUrl, streaming, ...rest }) => rest);
    localStorage.setItem(KEY, JSON.stringify(safe));
  } catch {
    /* quota / private mode — ignore */
  }
}

export function loadMessages(): ChatMessage[] {
  try {
    const raw = localStorage.getItem(KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed.map((m: ChatMessage) => ({ ...m, streaming: false }));
  } catch {
    return [];
  }
}

export function clearMessages(): void {
  try {
    localStorage.removeItem(KEY);
  } catch {
    /* ignore */
  }
}
```

- [ ] **Step 2: Wire persistence into App**

In `frontend/src/App.tsx`, add the import:

```tsx
import { saveMessages, loadMessages, clearMessages } from "./storage";
```

Change the `messages` initializer to restore from storage and seed the id counter:

```tsx
  const [messages, setMessages] = useState<Msg[]>(() => {
    const restored = loadMessages();
    msgId = restored.reduce((max, m) => Math.max(max, Number(m.id) || 0), 0);
    return restored;
  });
```

Add a save effect alongside the other `useEffect`s (after the auto-grow effect). Gate on `!loading` so the many mid-stream token updates don't thrash `localStorage`; the final state is saved when `loading` flips to `false`:

```tsx
  useEffect(() => {
    if (loading) return;
    saveMessages(messages);
  }, [messages, loading]);
```

Update `newChat` to also abort any stream and clear storage:

```tsx
  function newChat() {
    abortRef.current?.abort();
    setMessages([]);
    setInput("");
    clearMessages();
  }
```

- [ ] **Step 3: Build / typecheck**

Run: `cd frontend && npm run build`
Expected: no errors.

- [ ] **Step 4: Manual verify**

1. Ask a question and get an answer. Reload the page (`Cmd/Ctrl-R`) → the conversation is still there, no live "streaming" caret/dots on the restored answer.
2. Do a camera/photos visual search, then reload → the result carousel persists (photos load via `/file`); the uploaded query thumbnail is gone (expected — blob URL stripped).
3. Tap the new-chat (✎) button → conversation clears; reload → still empty.
4. Reload mid-stream (while an answer is generating) → no half-finished "streaming" bubble appears.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/storage.ts frontend/src/App.tsx
git commit -m "feat(ui): persist current conversation to localStorage"
```

---

## Task 9: Docs — Tailscale HTTPS setup + CLAUDE.md Step 15

**Files:**
- Create: `docs/tailscale-https-setup.md`
- Modify: `CLAUDE.md`

**Interfaces:** Documentation only.

- [ ] **Step 1: Write the Tailscale setup doc**

Create `docs/tailscale-https-setup.md`:

````markdown
# Enabling the Microphone on Your Phone (Tailscale HTTPS)

The voice/mic button needs a **secure context** (HTTPS or `localhost`). On desktop
`localhost` it works out of the box. On your iPhone, the app is reached over plain
HTTP on a LAN IP, where iOS blocks the microphone. This sets up a private HTTPS
origin via **Tailscale Serve** — no public internet exposure, and a publicly-trusted
certificate so iOS trusts it with no profile install.

## Prerequisites
- Tailscale installed and logged in on the PC (runs the app) and the iPhone.
- In the Tailscale admin console: **MagicDNS** enabled and **HTTPS Certificates** enabled.
- Your PC's MagicDNS name, e.g. `mypc.tailXXXX.ts.net`.

## Recommended: one HTTPS origin (frontend + API together)

Serving the built UI and the API from one origin avoids mixed-content and CORS.

1. Build the frontend against a **relative** API base so it calls the same origin:

   ```bash
   cd frontend
   # empty base → api.ts uses relative URLs like /chat, /health
   VITE_API_BASE_URL= npm run build
   ```

2. Have FastAPI serve the built `frontend/dist`. In `backend/main.py`, **after** all
   `app.include_router(...)` lines, add:

   ```python
   from fastapi.staticfiles import StaticFiles
   # API routes are registered above and take precedence; this serves the SPA.
   app.mount("/", StaticFiles(directory="../frontend/dist", html=True), name="frontend")
   ```

3. Start the API (it now serves the UI too) on port 8000.

4. Expose it over HTTPS within your tailnet:

   ```bash
   tailscale serve --bg --https=443 http://127.0.0.1:8000
   ```

5. On the iPhone (with Tailscale connected), open
   `https://mypc.tailXXXX.ts.net` → **Share → Add to Home Screen**. The mic now works.

## Alternative: two HTTPS ports (no backend code change)

Keep the dev server and API separate, expose both over HTTPS, and point the
frontend at the HTTPS API:

```bash
tailscale serve --bg --https=443  http://127.0.0.1:5173   # frontend
tailscale serve --bg --https=8443 http://127.0.0.1:8000   # API
```

Then build the frontend with `VITE_API_BASE_URL=https://mypc.tailXXXX.ts.net:8443`
and add that origin to the CORS `allow_origins` list in `backend/main.py`. (One
origin is simpler — prefer the recommended approach above.)

## Troubleshooting
- **Mic still blocked:** confirm the address bar shows `https://…ts.net` (not an IP)
  and a valid lock. `getUserMedia` is unavailable on insecure origins.
- **Blank page / 404 on refresh:** ensure `frontend/dist` exists (run the build) and
  the `StaticFiles` mount is the **last** thing added in `main.py`.
- **Mixed-content errors in console:** the page is HTTPS but an API call is HTTP —
  use the same-origin (recommended) setup, or the two-HTTPS-ports alternative.
- **`tailscale serve` status:** `tailscale serve status`; reset with `tailscale serve reset`.
````

- [ ] **Step 2: Add the Step 15 entry to CLAUDE.md**

In `CLAUDE.md`, after the "Step 14" section and before "## Key Design Decisions & Rationale", add:

```markdown
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
`localhost`; for the iPhone use Tailscale Serve HTTPS (see
`docs/tailscale-https-setup.md`). No public internet exposure.

**VRAM:** Whisper `base` (~150 MB) lazy-loads on first transcription — no budget concern.
```

- [ ] **Step 3: Commit**

```bash
git add docs/tailscale-https-setup.md CLAUDE.md
git commit -m "docs: Tailscale HTTPS setup + Step 15 (voice + UI polish)"
```

---

## Self-Review

**Spec coverage:**
- Theme 1 (header logo) → Task 1 ✓
- Theme 2 (Camera + Photos tiles) → Task 2 ✓
- Theme 3 (voice: backend `/transcribe`, recorder, composer wiring, secure-context doc) → Tasks 3, 4, 5, 9 ✓
- QoL-A (stop generation) → Task 5 ✓
- QoL-B (mic/send swap) → Task 5 ✓
- QoL-C (example prompts) → Task 6 ✓
- QoL-D (thinking indicator) → Task 7 (reduced to enhancement; dots pre-exist — flagged) ✓
- QoL-E (copy answer) → Task 7 ✓
- QoL-F (persist conversation) → Task 8 ✓
- Tailscale HTTPS recommendation → Task 9 ✓

**Placeholder scan:** No TBD/TODO; every code step contains complete code; commands have expected output. ✓

**Type consistency:** `useRecorder` return shape (`state/seconds/error/start/stop/cancel`) is defined in Task 4 and consumed exactly in Task 5. `transcribe(blob: Blob) => Promise<string>` defined in Task 4, called in Task 5. `_ext_for_content_type` signature matches between the endpoint and the test. `CopyIcon` produced in Task 4, consumed in Task 7. Storage helpers `saveMessages/loadMessages/clearMessages` defined in Task 8 and used in the same task. ✓

**Adjustment flagged:** Task 7 documents that the "thinking" dots already exist, so QoL-D is implemented as a small label enhancement rather than new work.
