export const MODALITY_OPTIONS = [
  { value: "", label: "All" },
  { value: "document", label: "Documents" },
  { value: "image", label: "Images" },
  { value: "audio", label: "Audio" },
];

const TOP_K_OPTIONS = [3, 6, 10, 15, 20];

interface Props {
  modality: string;
  setModality: (v: string) => void;
  topK: number | null;
  setTopK: (v: number | null) => void;
}

// Shared by the mobile "+" sheet and the desktop sidebar.
export function FilterControls({ modality, setModality, topK, setTopK }: Props) {
  return (
    <>
      <div className="sheet-section-label">Filter results</div>
      <div className="sheet-chips">
        {MODALITY_OPTIONS.map((o) => (
          <button
            key={o.value}
            className={`sheet-chip ${modality === o.value ? "active" : ""}`}
            onClick={() => setModality(o.value)}
          >
            {o.label}
          </button>
        ))}
      </div>

      <div className="sheet-section-label">Results to show</div>
      <div className="sheet-chips">
        <button
          className={`sheet-chip ${topK === null ? "active" : ""}`}
          onClick={() => setTopK(null)}
        >
          Auto
        </button>
        {TOP_K_OPTIONS.map((n) => (
          <button
            key={n}
            className={`sheet-chip ${topK === n ? "active" : ""}`}
            onClick={() => setTopK(n)}
          >
            {n}
          </button>
        ))}
      </div>
    </>
  );
}
