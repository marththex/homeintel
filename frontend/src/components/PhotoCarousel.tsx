import { useRef, useState } from "react";
import { thumbUrl } from "../api";
import { CollapsibleCaption } from "./CollapsibleCaption";
import { FadeImg } from "./FadeImg";
import { Lightbox } from "./Lightbox";

export interface CarouselPhoto {
  filePath: string;
  fileName: string;
  label?: string;   // e.g. "94% match"
  caption?: string; // this photo's own description (always matches the image)
}

// Alt text from the photo's caption when available (markdown stripped).
export function photoAlt(p: CarouselPhoto): string {
  const plain = p.caption?.replace(/[*_#`>]/g, "").replace(/\s+/g, " ").trim();
  if (!plain) return p.fileName;
  return plain.length > 140 ? plain.slice(0, 140) + "…" : plain;
}

interface Props {
  photos: CarouselPhoto[];
  initialCount?: number;
}

const PAGE = 10;

export function PhotoCarousel({ photos, initialCount = PAGE }: Props) {
  const [visible, setVisible] = useState(Math.min(initialCount, photos.length));
  const [active, setActive] = useState(0);
  const [lightboxIndex, setLightboxIndex] = useState<number | null>(null);
  const trackRef = useRef<HTMLDivElement>(null);

  if (photos.length === 0) return null;

  const shown = photos.slice(0, visible);
  const hasMore = visible < photos.length;

  function onScroll() {
    const track = trackRef.current;
    if (!track) return;
    const idx = Math.round(track.scrollLeft / track.clientWidth);
    if (idx !== active) setActive(idx);
  }

  function scrollTo(idx: number) {
    const track = trackRef.current;
    if (!track) return;
    track.scrollTo({ left: idx * track.clientWidth, behavior: "smooth" });
  }

  return (
    <div className="carousel">
      <div className="carousel-track" ref={trackRef} onScroll={onScroll}>
        {shown.map((p, i) => (
          <div className="carousel-slide" key={p.filePath}>
            <FadeImg
              src={thumbUrl(p.filePath, 768)}
              alt={photoAlt(p)}
              className="carousel-img"
              loading="lazy"
              onClick={() => setLightboxIndex(i)}
            />
            <div className="carousel-caption">
              <span className="carousel-name">{p.fileName}</span>
              {p.label && <span className="carousel-label">{p.label}</span>}
            </div>
          </div>
        ))}
      </div>

      {shown[active]?.caption && (
        <CollapsibleCaption key={active} caption={shown[active].caption!} />
      )}

      <div className="carousel-footer">
        <span className="carousel-counter">
          {active + 1} / {shown.length}
          {hasMore ? `+` : ""}
        </span>

        {shown.length <= 10 && (
          <div className="carousel-dots">
            {shown.map((_, i) => (
              <button
                key={i}
                className={`carousel-dot ${i === active ? "active" : ""}`}
                onClick={() => scrollTo(i)}
                aria-label={`Go to photo ${i + 1}`}
              />
            ))}
          </div>
        )}

        {hasMore && (
          <button
            className="carousel-more"
            onClick={() => setVisible((v) => Math.min(v + PAGE, photos.length))}
          >
            Show {Math.min(PAGE, photos.length - visible)} more
          </button>
        )}
      </div>

      {lightboxIndex !== null && (
        <Lightbox
          photos={shown}
          startIndex={lightboxIndex}
          onClose={() => setLightboxIndex(null)}
        />
      )}
    </div>
  );
}
