# Collapsible Captions + Fullscreen Photo Viewer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Collapse photo captions to a 2-line teaser (expandable on demand) in the chat carousel, and let the user tap any photo to open a fullscreen, swipeable viewer with the caption available behind an info button.

**Architecture:** Frontend-only. A new shared `CollapsibleCaption` component (2-line clamp + Show more/less) is used by both the carousel and a new `Lightbox` component (fullscreen viewer rendered via `createPortal`, native CSS scroll-snap swipe). `PhotoCarousel` is modified to use both. No backend, no new dependencies.

**Tech Stack:** React 19 + TypeScript + Vite, `react-markdown` (already installed), inline SVG icons, CSS scroll-snap, `ReactDOM.createPortal`.

## Global Constraints

- Frontend only. **No new dependencies** and **no icon libraries** — add inline SVG icons to `frontend/src/components/icons.tsx` following the existing `base()` pattern.
- **No frontend test runner exists** in this project. The automated gate is `cd frontend && npm run build` (`tsc && vite build`) — it must pass with zero type errors and a successful build. Each task also has explicit **manual verification** steps.
- Obey the **Rules of Hooks**: all hooks (`useState`/`useEffect`/`useRef`) must run before any conditional `return` in a component.
- Reuse existing CSS: animations `@keyframes fade-in` and `@keyframes slide-up` already exist in `App.css`; CSS variables `--text`, `--text-muted`, `--accent`, `--accent-hover`, `--border`, `--surface2` exist. Do not redefine them.
- `CarouselPhoto` is the shared photo type, exported from `frontend/src/components/PhotoCarousel.tsx` (`{ filePath, fileName, label?, caption? }`). Import it as a **type-only import** (`import type { CarouselPhoto }`) where needed to avoid a runtime circular import.
- Caption collapse applies wherever a caption exists; **tap-to-open applies to every carousel** (including camera visual-search). The lightbox navigates the carousel's currently-loaded `shown` slice.
- Commit messages: **no `Co-Authored-By` / AI-attribution trailers**.
- Work happens on branch `feat/carousel-caption-collapse-and-lightbox` (already created).

---

### Task 1: CollapsibleCaption + carousel caption collapse

Create the shared collapsible-caption component and use it in the carousel so the caption is a 2-line teaser by default.

**Files:**
- Create: `frontend/src/components/CollapsibleCaption.tsx`
- Modify: `frontend/src/components/PhotoCarousel.tsx` (replace the inline caption block; drop the now-unused `ReactMarkdown` import)
- Modify: `frontend/src/App.css` (replace `.carousel-active-caption` rule with `.cap*` rules)

**Interfaces:**
- Produces: `function CollapsibleCaption({ caption }: { caption: string }): JSX.Element | null` — exported from `CollapsibleCaption.tsx`. Renders markdown clamped to 2 lines with a "Show more"/"Show less" toggle that appears only when the content overflows the clamp; renders `null` for an empty/whitespace caption.

- [ ] **Step 1: Create the CollapsibleCaption component**

Create `frontend/src/components/CollapsibleCaption.tsx`:

```tsx
import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";

interface Props {
  caption: string;
}

export function CollapsibleCaption({ caption }: Props) {
  const [expanded, setExpanded] = useState(false);
  const [overflowing, setOverflowing] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  // Show the toggle only when the 2-line clamp actually hides content.
  // Measured against the rendered (clamped) box; re-measured on resize.
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const measure = () => {
      if (!expanded) setOverflowing(el.scrollHeight > el.clientHeight + 1);
    };
    measure();
    window.addEventListener("resize", measure);
    return () => window.removeEventListener("resize", measure);
  }, [caption, expanded]);

  if (!caption.trim()) return null;

  return (
    <div className="cap">
      <div ref={ref} className={`cap-clamp markdown ${expanded ? "expanded" : ""}`}>
        <ReactMarkdown>{caption}</ReactMarkdown>
      </div>
      {(overflowing || expanded) && (
        <button className="cap-toggle" onClick={() => setExpanded((v) => !v)}>
          {expanded ? "Show less" : "Show more"}
        </button>
      )}
    </div>
  );
}
```

- [ ] **Step 2: Use CollapsibleCaption in the carousel**

In `frontend/src/components/PhotoCarousel.tsx`:

(a) Replace the top imports — remove the `ReactMarkdown` import (no longer used in this file) and add the `CollapsibleCaption` import. The first three lines currently are:

```tsx
import { useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import { fileUrl } from "../api";
```

Change them to:

```tsx
import { useRef, useState } from "react";
import { fileUrl } from "../api";
import { CollapsibleCaption } from "./CollapsibleCaption";
```

(b) Replace the active-caption block:

```tsx
      {shown[active]?.caption && (
        <div className="carousel-active-caption markdown">
          <ReactMarkdown>{shown[active].caption}</ReactMarkdown>
        </div>
      )}
```

with (the `key={active}` remounts the component when the active photo changes, so each new photo starts collapsed):

```tsx
      {shown[active]?.caption && (
        <CollapsibleCaption key={active} caption={shown[active].caption!} />
      )}
```

- [ ] **Step 3: Replace the caption CSS**

In `frontend/src/App.css`, replace the existing `.carousel-active-caption` rule:

```css
.carousel-active-caption {
  margin-top: 8px;
  font-size: 0.85rem;
  line-height: 1.45;
  color: var(--text);
}
```

with the new collapsible-caption rules:

```css
.cap { margin-top: 8px; }
.cap-clamp {
  font-size: 0.85rem;
  line-height: 1.45;
  color: var(--text);
  overflow: hidden;
  display: -webkit-box;
  -webkit-box-orient: vertical;
  -webkit-line-clamp: 2;
}
.cap-clamp.expanded {
  display: block;
  -webkit-line-clamp: unset;
  overflow: visible;
}
.cap-clamp p { margin: 0; }
.cap-clamp.expanded p { margin: 0 0 8px; }
.cap-toggle {
  margin-top: 4px;
  background: none;
  border: none;
  color: var(--accent);
  font-size: 0.8rem;
  font-weight: 600;
  cursor: pointer;
  padding: 2px 0;
  -webkit-tap-highlight-color: transparent;
}
```

- [ ] **Step 4: Build (typecheck gate)**

Run: `cd frontend && npm run build`
Expected: `tsc` reports 0 errors and `vite build` completes successfully. (If `node_modules` is missing, run `npm install` first.)

- [ ] **Step 5: Manual verification**

Start the app (or use the already-running Vite dev server + refresh). With a text→image query that returns photos with long captions:
- The caption under the active photo shows ~2 lines followed by a "Show more" link.
- Tapping "Show more" expands to the full caption and the link becomes "Show less"; tapping "Show less" re-collapses.
- Swiping to a different photo shows that photo's caption collapsed to 2 lines again.
- A photo with a very short caption (≤2 lines) shows no toggle.

Expected: all of the above hold.

- [ ] **Step 6: Commit**

```bash
git add frontend/src/components/CollapsibleCaption.tsx frontend/src/components/PhotoCarousel.tsx frontend/src/App.css
git commit -m "feat: collapse carousel captions to a 2-line teaser"
```

---

### Task 2: Fullscreen Lightbox + tap-to-open

Add a fullscreen swipeable viewer and open it when a carousel photo is tapped.

**Files:**
- Create: `frontend/src/components/Lightbox.tsx`
- Modify: `frontend/src/components/icons.tsx` (add `InfoIcon`)
- Modify: `frontend/src/components/PhotoCarousel.tsx` (tap a photo to open the lightbox; lightbox state)
- Modify: `frontend/src/App.css` (lightbox styles; `cursor: zoom-in` on carousel images)

**Interfaces:**
- Consumes: `CollapsibleCaption` (Task 1); `CarouselPhoto` type from `PhotoCarousel`; `fileUrl` from `../api`; `CloseIcon` + new `InfoIcon` from `./icons`.
- Produces: `function Lightbox({ photos, startIndex, onClose }: { photos: CarouselPhoto[]; startIndex: number; onClose: () => void }): React.ReactPortal` — exported from `Lightbox.tsx`.
- Produces: `InfoIcon` exported from `icons.tsx` (same `IconProps` signature as the other icons).

- [ ] **Step 1: Add the InfoIcon**

In `frontend/src/components/icons.tsx`, add after `FilterIcon`:

```tsx
export const InfoIcon = ({ size = 22, className }: IconProps) => (
  <svg {...base(size, className)}>
    <circle cx="12" cy="12" r="10" />
    <path d="M12 16v-4M12 8h.01" />
  </svg>
);
```

- [ ] **Step 2: Create the Lightbox component**

Create `frontend/src/components/Lightbox.tsx`:

```tsx
import { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { fileUrl } from "../api";
import { CloseIcon, InfoIcon } from "./icons";
import { CollapsibleCaption } from "./CollapsibleCaption";
import type { CarouselPhoto } from "./PhotoCarousel";

interface Props {
  photos: CarouselPhoto[];
  startIndex: number;
  onClose: () => void;
}

export function Lightbox({ photos, startIndex, onClose }: Props) {
  const [active, setActive] = useState(startIndex);
  const [showCaption, setShowCaption] = useState(false);
  const trackRef = useRef<HTMLDivElement>(null);

  // Jump to the tapped photo on open (no smooth scroll) and lock body scroll.
  useEffect(() => {
    const track = trackRef.current;
    if (track) track.scrollLeft = startIndex * track.clientWidth;
    const prev = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => { document.body.style.overflow = prev; };
  }, [startIndex]);

  // Close on Escape.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") onClose(); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  function onScroll() {
    const track = trackRef.current;
    if (!track) return;
    const idx = Math.round(track.scrollLeft / track.clientWidth);
    if (idx !== active) {
      setActive(idx);
      setShowCaption(false); // collapse the caption panel when moving photos
    }
  }

  const activePhoto = photos[active];
  const hasCaption = !!activePhoto?.caption?.trim();

  return createPortal(
    <div className="lightbox" onClick={onClose}>
      <div className="lightbox-bar" onClick={(e) => e.stopPropagation()}>
        <button className="lightbox-btn" onClick={onClose} aria-label="Close">
          <CloseIcon size={22} />
        </button>
        <span className="lightbox-counter">{active + 1} / {photos.length}</span>
        {hasCaption ? (
          <button
            className={`lightbox-btn ${showCaption ? "active" : ""}`}
            onClick={() => setShowCaption((v) => !v)}
            aria-label="Toggle description"
          >
            <InfoIcon size={22} />
          </button>
        ) : (
          <span className="lightbox-btn-spacer" />
        )}
      </div>

      <div className="lightbox-track" ref={trackRef} onScroll={onScroll}>
        {photos.map((p) => (
          <div className="lightbox-slide" key={p.filePath}>
            <img
              src={fileUrl(p.filePath)}
              alt={p.fileName}
              className="lightbox-img"
              onClick={(e) => e.stopPropagation()}
            />
          </div>
        ))}
      </div>

      {showCaption && hasCaption && (
        <div className="lightbox-caption" onClick={(e) => e.stopPropagation()}>
          <CollapsibleCaption key={active} caption={activePhoto.caption!} />
        </div>
      )}
    </div>,
    document.body
  );
}
```

Note the close behavior: the backdrop (`.lightbox`) closes on click; the top bar, the image itself, and the caption panel call `stopPropagation` so taps on them don't close. A *tap* on the dark area around the image bubbles to the backdrop and closes; a *swipe* scrolls and fires no click.

- [ ] **Step 3: Wire tap-to-open into the carousel**

In `frontend/src/components/PhotoCarousel.tsx`:

(a) Add the `Lightbox` import below the `CollapsibleCaption` import:

```tsx
import { Lightbox } from "./Lightbox";
```

(b) Add lightbox state next to the existing `useState` calls (after the `active` state):

```tsx
  const [lightboxIndex, setLightboxIndex] = useState<number | null>(null);
```

(c) Give each slide its index and make the image open the lightbox. The current map opens with `{shown.map((p) => (` and the image is:

```tsx
        {shown.map((p) => (
          <div className="carousel-slide" key={p.filePath}>
            <img
              src={fileUrl(p.filePath)}
              alt={p.fileName}
              className="carousel-img"
              loading="lazy"
            />
```

Change it to add the index `i` and an `onClick`:

```tsx
        {shown.map((p, i) => (
          <div className="carousel-slide" key={p.filePath}>
            <img
              src={fileUrl(p.filePath)}
              alt={p.fileName}
              className="carousel-img"
              loading="lazy"
              onClick={() => setLightboxIndex(i)}
            />
```

(d) Render the lightbox. Add this immediately before the final closing `</div>` of the `carousel` container (after the `carousel-footer` block):

```tsx
      {lightboxIndex !== null && (
        <Lightbox
          photos={shown}
          startIndex={lightboxIndex}
          onClose={() => setLightboxIndex(null)}
        />
      )}
```

- [ ] **Step 4: Add the lightbox CSS**

Append to `frontend/src/App.css`:

```css
/* ── Lightbox (fullscreen viewer) ───────────────────────────────────────── */
.carousel-img { cursor: zoom-in; }

.lightbox {
  position: fixed;
  inset: 0;
  z-index: 1000;
  background: rgba(0, 0, 0, 0.96);
  display: flex;
  flex-direction: column;
  animation: fade-in 0.15s ease;
}
.lightbox-bar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 12px 14px;
  padding-top: max(12px, env(safe-area-inset-top));
  color: #fff;
  z-index: 2;
}
.lightbox-btn {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 40px;
  height: 40px;
  background: rgba(255, 255, 255, 0.12);
  border: none;
  border-radius: 50%;
  color: #fff;
  cursor: pointer;
  -webkit-tap-highlight-color: transparent;
}
.lightbox-btn.active { background: var(--accent); }
.lightbox-btn-spacer { width: 40px; height: 40px; }
.lightbox-counter { font-size: 0.9rem; color: #fff; }

.lightbox-track {
  flex: 1;
  min-height: 0;
  display: flex;
  overflow-x: auto;
  scroll-snap-type: x mandatory;
  -webkit-overflow-scrolling: touch;
  scrollbar-width: none;
}
.lightbox-track::-webkit-scrollbar { display: none; }
.lightbox-slide {
  flex: 0 0 100%;
  width: 100%;
  scroll-snap-align: center;
  display: flex;
  align-items: center;
  justify-content: center;
}
.lightbox-img {
  max-width: 100%;
  max-height: 100%;
  object-fit: contain;
  cursor: default;
}

.lightbox-caption {
  position: absolute;
  left: 0;
  right: 0;
  bottom: 0;
  max-height: 45vh;
  overflow-y: auto;
  padding: 14px 16px;
  padding-bottom: max(14px, env(safe-area-inset-bottom));
  background: linear-gradient(to top, rgba(0, 0, 0, 0.92), rgba(0, 0, 0, 0.75));
  animation: slide-up 0.2s ease;
}
.lightbox-caption .cap-clamp { color: #fff; }
```

- [ ] **Step 5: Build (typecheck gate)**

Run: `cd frontend && npm run build`
Expected: `tsc` reports 0 errors and `vite build` completes successfully.

- [ ] **Step 6: Manual verification**

With the app running and a text→image query showing photos:
- Tapping a photo opens it fullscreen (black overlay) at the tapped image.
- Swiping left/right moves between the result photos; the `n / total` counter updates.
- The `ⓘ` button reveals a caption panel over the bottom of the photo, collapsed to ~2 lines with "Show more"; expanding works; tapping `ⓘ` again hides it; moving to another photo hides the panel.
- Closing works via the `✕` button, a tap on the dark area around the photo, and the `Esc` key.
- The page behind the viewer does not scroll while it is open.
- For a camera **visual-search** result carousel (no captions): tapping a photo still opens the viewer, swipe works, and no `ⓘ` button is shown.

Expected: all of the above hold.

- [ ] **Step 7: Commit**

```bash
git add frontend/src/components/Lightbox.tsx frontend/src/components/icons.tsx frontend/src/components/PhotoCarousel.tsx frontend/src/App.css
git commit -m "feat: fullscreen swipeable photo viewer with caption toggle"
```

---

## Self-Review

**Spec coverage:**
- Caption minimized by default (2-line clamp) + expandable → Task 1 (`CollapsibleCaption` + carousel wiring). ✓
- Toggle only when overflowing; resets per photo (`key={active}`) → Task 1 Steps 1–2. ✓
- `CollapsibleCaption` shared by carousel and lightbox → Task 1 creates it; Task 2 reuses it. ✓
- Tap photo → fullscreen viewer → Task 2 Step 3 (`onClick` → `lightboxIndex`) + Step 2 (`Lightbox`). ✓
- Swipe left/right between photos → Task 2 `lightbox-track` scroll-snap. ✓
- Caption available in lightbox behind `ⓘ`, collapsed by default; `ⓘ` only when caption exists → Task 2 Step 2. ✓
- Close via ✕ / backdrop tap / Esc; body-scroll-lock → Task 2 Step 2. ✓
- `createPortal` to document.body → Task 2 Step 2. ✓
- Applies to all carousels; lightbox scoped to `shown` slice → Task 2 passes `photos={shown}`. ✓
- No new deps; inline `InfoIcon` → Task 2 Step 1. ✓
- Build gate + manual verification (no test runner) → both tasks Steps 4–6 / 5–6. ✓

**Placeholder scan:** No TBD/TODO/"handle edge cases" — every code step has complete code. ✓

**Type consistency:** `CollapsibleCaption({ caption: string })` defined Task 1, consumed in Task 1 (carousel) and Task 2 (lightbox) with the same prop. `Lightbox({ photos, startIndex, onClose })` defined Task 2, consumed in Task 2 carousel wiring with matching prop names/types (`photos={shown}`, `startIndex={lightboxIndex}` where `lightboxIndex` is narrowed to `number` by the `!== null` guard, `onClose`). `CarouselPhoto` imported as type-only. `InfoIcon` matches the existing `IconProps` signature. CSS classes referenced in TSX (`cap`, `cap-clamp`, `cap-toggle`, `lightbox*`) all defined in the CSS steps. ✓

**Note on TDD:** This project has no frontend test runner, so the red/green unit-test cycle does not apply; the automated gate is `npm run build` (tsc typecheck), paired with explicit manual verification — consistent with the prior frontend task (Step 11/Task 5) in this codebase.
