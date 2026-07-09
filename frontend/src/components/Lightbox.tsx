import { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { fileUrl } from "../api";
import { CloseIcon, InfoIcon } from "./icons";
import { CollapsibleCaption } from "./CollapsibleCaption";
import { photoAlt, type CarouselPhoto } from "./PhotoCarousel";

interface Props {
  photos: CarouselPhoto[];
  startIndex: number;
  onClose: () => void;
}

export function Lightbox({ photos, startIndex, onClose }: Props) {
  const [active, setActive] = useState(startIndex);
  const [showCaption, setShowCaption] = useState(false);
  const trackRef = useRef<HTMLDivElement>(null);
  const closeBtnRef = useRef<HTMLButtonElement>(null);

  // Jump to the tapped photo on open (no smooth scroll) and lock body scroll.
  useEffect(() => {
    const track = trackRef.current;
    if (track) track.scrollLeft = startIndex * track.clientWidth;
    const prev = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => { document.body.style.overflow = prev; };
  }, [startIndex]);

  // Move focus in on open; give it back to the opener on close.
  useEffect(() => {
    const opener = document.activeElement as HTMLElement | null;
    closeBtnRef.current?.focus();
    return () => opener?.focus();
  }, []);

  function scrollTo(idx: number) {
    const track = trackRef.current;
    if (!track) return;
    const clamped = Math.max(0, Math.min(photos.length - 1, idx));
    track.scrollTo({ left: clamped * track.clientWidth, behavior: "smooth" });
  }

  // Keyboard: Escape closes, arrows navigate.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
      else if (e.key === "ArrowLeft") scrollTo(active - 1);
      else if (e.key === "ArrowRight") scrollTo(active + 1);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [active, onClose]);

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
        <button type="button" ref={closeBtnRef} className="lightbox-btn" onClick={onClose} aria-label="Close">
          <CloseIcon size={22} />
        </button>
        <span className="lightbox-counter">{active + 1} / {photos.length}</span>
        {hasCaption ? (
          <button
            type="button"
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
              alt={photoAlt(p)}
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
