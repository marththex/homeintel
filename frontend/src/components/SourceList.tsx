import { fileUrl } from "../api";
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

  return (
    <div className="source-list">
      <p className="source-label">Sources</p>

      {imageSources.length > 0 && (
        <div className="source-image-strip">
          {imageSources.map((s, i) => (
            <a key={i} href={fileUrl(s.file_path)} target="_blank" rel="noreferrer" title={s.file_name}>
              <img
                src={fileUrl(s.file_path)}
                alt={s.file_name}
                className="source-image-thumb"
                loading="lazy"
              />
            </a>
          ))}
        </div>
      )}

      {otherSources.map((s, i) => (
        <div key={i} className="source-item" title={s.excerpt}>
          <span className="source-icon">{MODALITY_ICON[s.modality] ?? "📎"}</span>
          <span className="source-name">{s.file_name}</span>
          <span className="source-path">{s.file_path}</span>
        </div>
      ))}
    </div>
  );
}
