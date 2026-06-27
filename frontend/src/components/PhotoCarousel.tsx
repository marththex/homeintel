import { useRef, useState } from "react";
import { fileUrl } from "../api";

export interface CarouselPhoto {
  filePath: string;
  fileName: string;
  label?: string; // e.g. "94% match" or a caption
}

interface Props {
  photos: CarouselPhoto[];
  initialCount?: number;
}

const PAGE = 10;

export function PhotoCarousel({ photos, initialCount = PAGE }: Props) {
  const [visible, setVisible] = useState(Math.min(initialCount, photos.length));
  const [active, setActive] = useState(0);
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
        {shown.map((p) => (
          <div className="carousel-slide" key={p.filePath}>
            <img
              src={fileUrl(p.filePath)}
              alt={p.fileName}
              className="carousel-img"
              loading="lazy"
            />
            <div className="carousel-caption">
              <span className="carousel-name">{p.fileName}</span>
              {p.label && <span className="carousel-label">{p.label}</span>}
            </div>
          </div>
        ))}
      </div>

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
    </div>
  );
}
