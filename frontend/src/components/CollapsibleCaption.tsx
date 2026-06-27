import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";

interface Props {
  caption: string;
}

export function CollapsibleCaption({ caption }: Props) {
  const [expanded, setExpanded] = useState(false);
  const [overflowing, setOverflowing] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  // Show the toggle only when the 2-line clamp actually hides content.
  // Measured against the rendered (clamped) box; re-measured on resize.
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const measure = () => {
      if (!expanded) setOverflowing(el.scrollHeight > el.clientHeight + 1);
    };
    measure();
    window.addEventListener("resize", measure);
    return () => window.removeEventListener("resize", measure);
  }, [caption, expanded]);

  if (!caption.trim()) return null;

  return (
    <div className="cap">
      <div ref={ref} className={`cap-clamp markdown ${expanded ? "expanded" : ""}`}>
        <ReactMarkdown>{caption}</ReactMarkdown>
      </div>
      {(overflowing || expanded) && (
        <button className="cap-toggle" onClick={() => setExpanded((v) => !v)}>
          {expanded ? "Show less" : "Show more"}
        </button>
      )}
    </div>
  );
}
