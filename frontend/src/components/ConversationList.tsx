import { useEffect, useState } from "react";
import { TrashIcon } from "./icons";
import type { ConversationMeta } from "../storage";

function timeAgo(ts: number): string {
  const s = Math.floor((Date.now() - ts) / 1000);
  if (s < 60) return "now";
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m ago`;
  const h = Math.floor(m / 60);
  if (h < 24) return `${h}h ago`;
  const d = Math.floor(h / 24);
  if (d < 7) return `${d}d ago`;
  return new Date(ts).toLocaleDateString();
}

interface Props {
  items: ConversationMeta[];
  activeId: string;
  onSelect: (id: string) => void;
  onDelete: (id: string) => void;
}

// Shared by the desktop sidebar and the mobile chats drawer.
export function ConversationList({ items, activeId, onSelect, onDelete }: Props) {
  // Two-tap delete: first tap arms a "Delete?" confirm that disarms after 3s.
  const [confirmId, setConfirmId] = useState<string | null>(null);

  useEffect(() => {
    if (!confirmId) return;
    const t = setTimeout(() => setConfirmId(null), 3000);
    return () => clearTimeout(t);
  }, [confirmId]);

  if (items.length === 0) return <p className="conv-empty">No conversations yet.</p>;

  return (
    <nav className="conv-list">
      {items.map((c) => (
        <div key={c.id} className={`conv-row ${c.id === activeId ? "active" : ""}`}>
          <button className="conv-title" onClick={() => onSelect(c.id)} title={c.title}>
            <span className="conv-name">{c.title}</span>
            <span className="conv-time">{timeAgo(c.updatedAt)}</span>
          </button>
          {confirmId === c.id ? (
            <button
              className="conv-delete confirm"
              onClick={() => {
                setConfirmId(null);
                onDelete(c.id);
              }}
              aria-label={`Confirm delete "${c.title}"`}
            >
              Delete?
            </button>
          ) : (
            <button
              className="conv-delete"
              onClick={() => setConfirmId(c.id)}
              aria-label={`Delete "${c.title}"`}
            >
              <TrashIcon size={15} />
            </button>
          )}
        </div>
      ))}
    </nav>
  );
}
