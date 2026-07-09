# Design — Collapsible captions + fullscreen photo viewer

**Date:** 2026-06-27
**Status:** Approved (ready for implementation plan)

## Problem

In the chat photo carousel, text→image (and image-source) results render the
full vision caption inline, which is a large block of text that dominates the
message (see the proposal-query screenshot — the IMG_4616 caption fills the
screen). The user wants:

1. The caption **minimized by default**, expandable on demand only when they
   want details about the image(s).
2. Tapping a photo to **open it fullscreen**, with **swipe left/right** to move
   between the result photos.

## Goals

- Caption collapsed to a short teaser by default in the carousel, with an
  inline expand/collapse toggle.
- A fullscreen image viewer reachable by tapping any carousel photo, with
  swipe navigation and the caption available (collapsed) behind an info button.

## Non-goals

- No backend changes — captions already arrive in full from `/chat` /
  `/chat/stream` (the recent full-caption work).
- No change to retrieval, ranking, or how many photos are returned.
- No new dependencies — reuse `react-markdown` (already present) and the
  existing inline-SVG icon approach.

## Scope notes

- **Caption collapse** applies wherever a caption exists: text→image queries and
  image sources in chat answers. The camera **visual-search** carousel has no
  caption (only filename + match%), so collapse is a no-op there.
- **Tap-to-open fullscreen** applies to **every** `PhotoCarousel` usage,
  including visual-search.
- The lightbox navigates the **currently-loaded** photos (the carousel's
  `shown` slice). The existing "Show 10 more" pagination still governs how many
  photos are loaded; the lightbox does not page beyond what's visible.

## Architecture

All changes are in the frontend. Three component units with clear boundaries:

```
PhotoCarousel (modified)
 ├─ CollapsibleCaption (new)   ← caption teaser + Show more/less
 └─ Lightbox (new)             ← fullscreen swipeable viewer
      └─ CollapsibleCaption (reused)
```

### Component A — `CollapsibleCaption` (new, shared)

`frontend/src/components/CollapsibleCaption.tsx`

- **Responsibility:** render a caption (markdown) clamped to 2 lines with a
  "Show more / Show less" toggle. Used by both the carousel and the lightbox so
  the collapse behavior is identical (DRY).
- **Interface:** `function CollapsibleCaption({ caption }: { caption: string })`.
- **Behavior:**
  - Collapsed by default: markdown rendered inside a container with
    `-webkit-line-clamp: 2` (CSS).
  - Internal `expanded` boolean toggles between clamped and full.
  - **Overflow-aware:** after render (and on window resize), measure whether the
    content overflows the clamped box (`scrollHeight > clientHeight`); only show
    the toggle when it overflows, so a 1–2 line caption gets no toggle.
  - Empty/whitespace caption → renders nothing.

### Component B — `PhotoCarousel` (modified)

`frontend/src/components/PhotoCarousel.tsx`

- Replace the always-full `carousel-active-caption` block with
  `<CollapsibleCaption caption={shown[active].caption} />`.
- Reset the caption to collapsed when the active photo changes (the
  `CollapsibleCaption` remounts per active photo via a `key={active}` so each new
  photo starts at the 2-line teaser).
- Make each photo open the lightbox: clicking `carousel-img` sets
  `lightboxIndex` to that photo's index.
- New state `lightboxIndex: number | null`. When non-null, render
  `<Lightbox photos={shown} startIndex={lightboxIndex} onClose={() => setLightboxIndex(null)} />`.
- `carousel-img` gets `cursor: zoom-in` and an accessible role (button-like:
  `onClick` + `role="button"` / keyboard not required for touch-first, but add
  `alt` already present).

### Component C — `Lightbox` (new)

`frontend/src/components/Lightbox.tsx`

- **Responsibility:** fullscreen, swipeable photo viewer with an optional
  caption.
- **Interface:**
  `function Lightbox({ photos, startIndex, onClose }: { photos: CarouselPhoto[]; startIndex: number; onClose: () => void })`.
- **Rendering:** `createPortal` into `document.body`. Unlike `BottomSheet`
  (rendered at app root), the lightbox is opened from deep inside a message
  bubble, so a portal guarantees true fullscreen regardless of any ancestor
  stacking/transform context. Body-scroll-lock and backdrop-click-to-close
  mirror `BottomSheet`.
- **Track:** a horizontal `scroll-snap` track (the same native-CSS swipe
  mechanism the carousel already uses — no JS touch handling), one slide per
  photo. On mount, scroll to `startIndex` (without smooth animation). An
  `onScroll` handler tracks the active index for the counter.
- **Top bar (over the image):** `✕` close (left), `n / total` counter (center),
  `ⓘ` info (right). The `ⓘ` button is shown only when the active photo has a
  caption (so visual-search photos show no `ⓘ`).
- **Caption:** hidden by default. `ⓘ` toggles a panel that slides up over the
  bottom of the photo containing `<CollapsibleCaption caption={...} />` for the
  active photo (so it opens at the 2-line teaser, expandable to full). The panel
  has its own vertical scroll if the expanded caption is tall.
- **Close:** `✕` button, backdrop/region tap, or `Esc` (keydown listener added
  on mount, removed on unmount). Body scroll locked while open.

### Icons / CSS

- `frontend/src/components/icons.tsx`: add `InfoIcon` (ⓘ). Reuse `CloseIcon`.
- `frontend/src/App.css`:
  - `CollapsibleCaption`: 2-line clamp styles (`.cap-clamp` with
    `-webkit-line-clamp: 2`, `.cap-clamp.expanded` unclamped) and a
    `.cap-toggle` text button.
  - Lightbox: full-screen overlay (`position: fixed; inset: 0; z-index` above
    everything), black backdrop, horizontal scroll-snap track, contained images
    (`object-fit: contain`), top bar, and the slide-up caption panel.
  - `carousel-img { cursor: zoom-in; }`.

## Data flow

`PhotoCarousel` already holds the photos (with `caption`). It passes its visible
`shown` slice and the tapped index to `Lightbox`. The lightbox is purely
presentational; closing it clears `lightboxIndex` in the carousel. No new API
calls; images load via the existing `fileUrl()` helper.

## Error / edge handling

- No caption → `CollapsibleCaption` renders nothing; lightbox `ⓘ` still works
  but shows an empty/short panel (acceptable; visual-search photos have no
  caption, so for those the `ⓘ` panel is effectively empty — hide `ⓘ` when the
  active photo has no caption).
- Single photo → lightbox still opens; swipe simply has nowhere to go; counter
  shows `1 / 1`.
- Rapid open/close and resize handled by effect cleanup (scroll lock and key
  listener restored on unmount).

## Testing

- **Gate:** `cd frontend && npm run build` (`tsc` typecheck + `vite build`) must
  pass with no type or build errors. (No frontend unit-test framework exists in
  this project.)
- **Manual verification:**
  - Caption shows ~2 lines + "Show more" by default; "Show more" expands, "Show
    less" collapses; short captions show no toggle.
  - Swiping to another photo resets its caption to the 2-line teaser.
  - Tapping a photo opens it fullscreen; swipe left/right moves between the
    result photos; the counter updates.
  - `ⓘ` reveals the collapsed caption over the image; expanding it works; `ⓘ`
    hides it again.
  - `✕`, backdrop tap, and `Esc` all close the viewer; the page behind does not
    scroll while it is open.
  - Visual-search (camera) carousel: tap-to-open works; no caption/ⓘ shown.

## Files touched

| File | Change |
|---|---|
| `frontend/src/components/CollapsibleCaption.tsx` | New — 2-line clamp + Show more/less, overflow-aware |
| `frontend/src/components/Lightbox.tsx` | New — fullscreen swipeable viewer, portal, ⓘ caption, Esc/backdrop/✕ close |
| `frontend/src/components/PhotoCarousel.tsx` | Use CollapsibleCaption; tap photo to open Lightbox; lightbox state |
| `frontend/src/components/icons.tsx` | Add `InfoIcon` |
| `frontend/src/App.css` | Caption clamp/toggle + lightbox styles; `cursor: zoom-in` on photos |
