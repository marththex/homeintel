import { PhotoCarousel, type CarouselPhoto } from "./PhotoCarousel";
import type { SourceDoc } from "../types";

const MODALITY_ICON: Record<string, string> = {
  document: "📄",
  image: "🖼️",
  audio: "🎵",
};

interface Props {
  sources: SourceDoc[];
}

export function SourceList({ sources }: Props) {
  if (!sources.length) return null;

  const unique = sources.filter(
    (s, i, arr) => arr.findIndex((x) => x.file_path === s.file_path) === i
  );

  const imageSources = unique.filter((s) => s.modality === "image");
  const otherSources = unique.filter((s) => s.modality !== "image");

  const photos: CarouselPhoto[] = imageSources.map((s) => ({
    filePath: s.file_path,
    fileName: s.file_name,
  }));

  return (
    <div className="source-list">
      {photos.length > 0 && <PhotoCarousel photos={photos} />}

      {otherSources.length > 0 && (
        <>
          <p className="source-label">Sources</p>
          {otherSources.map((s, i) => (
            <div key={i} className="source-item" title={s.excerpt}>
              <span className="source-icon">{MODALITY_ICON[s.modality] ?? "📎"}</span>
              <span className="source-name">{s.file_name}</span>
              <span className="source-path">{s.file_path}</span>
            </div>
          ))}
        </>
      )}
    </div>
  );
}
