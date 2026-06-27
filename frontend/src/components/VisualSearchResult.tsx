import { useState } from "react";
import { fileUrl } from "../api";
import type { VisualResult } from "../types";

interface Props {
  results: VisualResult[];
  queryImageUrl: string;
}

export function VisualSearchResult({ results, queryImageUrl }: Props) {
  const [lightbox, setLightbox] = useState<string | null>(null);

  return (
    <div className="visual-result">
      <div className="visual-query-row">
        <img
          src={queryImageUrl}
          alt="Your photo"
          className="visual-query-thumb"
        />
        <span className="visual-query-label">
          {results.length === 0
            ? "No similar photos found (threshold: 70%)"
            : `${results.length} similar photo${results.length !== 1 ? "s" : ""} found`}
        </span>
      </div>

      {results.length > 0 && (
        <div className="visual-grid">
          {results.map((r) => (
            <div
              key={r.file_path}
              className="visual-cell"
              onClick={() => setLightbox(fileUrl(r.file_path))}
              title={`${r.file_name} — ${Math.round(r.score * 100)}% match`}
            >
              <img
                src={fileUrl(r.file_path)}
                alt={r.file_name}
                className="visual-thumb"
                loading="lazy"
              />
              <span className="visual-score">{Math.round(r.score * 100)}%</span>
            </div>
          ))}
        </div>
      )}

      {lightbox && (
        <div className="lightbox" onClick={() => setLightbox(null)}>
          <img src={lightbox} alt="Full size" className="lightbox-img" />
          <button className="lightbox-close" onClick={() => setLightbox(null)}>✕</button>
        </div>
      )}
    </div>
  );
}
