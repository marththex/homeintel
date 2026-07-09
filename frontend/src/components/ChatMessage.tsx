import { memo, useState } from "react";
import ReactMarkdown from "react-markdown";
import { SourceList } from "./SourceList";
import { PhotoResults } from "./PhotoResults";
import type { CarouselPhoto } from "./PhotoCarousel";
import { CopyIcon } from "./icons";
import type { ChatMessage as Msg } from "../types";

interface Props {
  message: Msg;
  /** Total indexed chunks (from /stats) — shown while retrieval runs. */
  indexTotal?: number;
  /** Re-sends the failed question (rendered on error bubbles). */
  onRetry?: (question: string) => void;
}

// Retrieval hasn't returned → "Searching…"; sources in but no tokens yet →
// "N matches · generating…"; first token replaces the line entirely.
function statusLabel(message: Msg, indexTotal?: number): string {
  if (message.sources) {
    const n = message.chunks_used ?? message.sources.length;
    return `${n} match${n === 1 ? "" : "es"} · generating answer…`;
  }
  return indexTotal
    ? `Searching ${indexTotal.toLocaleString()} indexed chunks…`
    : "Searching your files…";
}

// Memoized: App replaces message objects immutably, so during streaming only
// the in-flight message re-renders per token (markdown re-parse is expensive).
export const ChatMessage = memo(function ChatMessage({ message, indexTotal, onRetry }: Props) {
  const isUser = message.role === "user";

  const visualPhotos: CarouselPhoto[] | undefined = message.visualResults?.map((r) => ({
    filePath: r.file_path,
    fileName: r.file_name,
    label: `${Math.round(r.score * 100)}% match`,
  }));

  return (
    <div className={`message ${isUser ? "message-user" : "message-assistant"}`}>
      <div
        className={`bubble ${isUser ? "bubble-user" : "bubble-assistant"} ${message.error ? "bubble-error" : ""}`}
        aria-busy={message.streaming || undefined}
      >
        {message.queryImageUrl && (
          <img src={message.queryImageUrl} alt="Uploaded" className="query-image-preview" />
        )}

        {isUser ? (
          <p className="bubble-text">{message.content}</p>
        ) : (
          <div className="bubble-text markdown">
            {message.content && <ReactMarkdown>{message.content}</ReactMarkdown>}
            {message.streaming && !message.content && (
              <>
                <span className="thinking">
                  <span className="stream-dots">
                    <span className="dot" /><span className="dot" /><span className="dot" />
                  </span>
                  <span className="thinking-label">{statusLabel(message, indexTotal)}</span>
                </span>
                {!message.sources && (
                  <span className="skel-lines">
                    <span className="skel-line skel" />
                    <span className="skel-line skel" />
                  </span>
                )}
              </>
            )}
            {message.streaming && message.content && <span className="stream-caret" />}
          </div>
        )}

        {!isUser && !message.streaming && !message.error && message.sources?.length === 0 && (
          <p className="no-matches">No matching files found in the index.</p>
        )}

        {/* Supporting evidence renders only after the answer finishes streaming —
            the text is the answer; images/sources must never beat it to the screen. */}
        {!isUser && !message.streaming && visualPhotos && <PhotoResults photos={visualPhotos} />}
        {!isUser && !message.streaming && message.sources && <SourceList sources={message.sources} />}

        {!isUser && message.chunks_used !== undefined && (
          <p className="bubble-meta">{message.chunks_used} results · {message.model}</p>
        )}
        {!isUser && message.error && message.question && onRetry && (
          <button className="retry-btn" onClick={() => onRetry(message.question!)}>
            Retry
          </button>
        )}
        {!isUser && !message.streaming && !message.error && message.content && (
          <CopyButton text={message.content} />
        )}
      </div>
    </div>
  );
});

function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);
  async function copy() {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      /* clipboard unavailable (insecure context) — ignore */
    }
  }
  return (
    <button className="copy-btn" onClick={copy} aria-label="Copy answer">
      <CopyIcon size={14} /> <span>{copied ? "Copied" : "Copy"}</span>
    </button>
  );
}
