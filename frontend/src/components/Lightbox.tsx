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
