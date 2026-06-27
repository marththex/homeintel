import { SourceList } from "./SourceList";
import type { ChatMessage as Msg } from "../types";

interface Props {
  message: Msg;
}

export function ChatMessage({ message }: Props) {
  const isUser = message.role === "user";

  return (
    <div className={`message ${isUser ? "message-user" : "message-assistant"}`}>
      <div className={`bubble ${isUser ? "bubble-user" : "bubble-assistant"} ${message.error ? "bubble-error" : ""}`}>
        <p className="bubble-text">{message.content}</p>
        {!isUser && message.sources && <SourceList sources={message.sources} />}
        {!isUser && message.chunks_used !== undefined && (
          <p className="bubble-meta">{message.chunks_used} chunks · {message.model}</p>
        )}
      </div>
    </div>
  );
}
