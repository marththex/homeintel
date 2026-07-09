import { useLayoutEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";

interface Props {
  caption: string;
}

export function CollapsibleCaption({ caption }: Props) {
  const [expanded, setExpanded] = useState(false);
  const [overflowing, setOverflowing] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  // Measure the CLAMPED box synchronously after commit so the toggle shows
  // only when content overflows — and survives an expand -> collapse cycle.
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el || expanded) return;
    const measure = () => setOverflowing(el.scrollHeight > el.clientHeight + 1);
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
        <button type="button" className="cap-toggle" onClick={() => setExpanded((v) => !v)}>
          {expanded ? "Show less" : "Show more"}
        </button>
      )}
    </div>
  );
}
