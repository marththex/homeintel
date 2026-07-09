import { useState } from "react";
import { thumbUrl } from "../api";
import { FadeImg } from "./FadeImg";
import { Lightbox } from "./Lightbox";
import { photoAlt, type CarouselPhoto } from "./PhotoCarousel";

interface Props {
  photos: CarouselPhoto[];
  initialCount?: number;
}

const PAGE = 10;

export function PhotoGrid({ photos, initialCount = PAGE }: Props) {
  const [visible, setVisible] = useState(Math.min(initialCount, photos.length));
  const [lightboxIndex, setLightboxIndex] = useState<number | null>(null);

  if (photos.length === 0) return null;

  const shown = photos.slice(0, visible);
  const hasMore = visible < photos.length;

  return (
    <div className="photo-grid-wrap">
      <div className="photo-grid">
        {shown.map((p, i) => (
          <button
            key={p.filePath}
            className="grid-cell"
            onClick={() => setLightboxIndex(i)}
            aria-label={`Open ${p.fileName}`}
          >
            <FadeImg src={thumbUrl(p.filePath)} alt={photoAlt(p)} loading="lazy" />
            <span className="grid-cell-overlay">
              <span className="grid-cell-name">{p.fileName}</span>
              {p.label && <span className="grid-cell-label">{p.label}</span>}
            </span>
          </button>
        ))}
      </div>

      {hasMore && (
        <div className="carousel-footer">
          <button
            className="carousel-more"
            onClick={() => setVisible((v) => Math.min(v + PAGE, photos.length))}
          >
            Show {Math.min(PAGE, photos.length - visible)} more
          </button>
        </div>
      )}

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
