import ReactMarkdown from "react-markdown";
import { SourceList } from "./SourceList";
import { PhotoCarousel, type CarouselPhoto } from "./PhotoCarousel";
import type { ChatMessage as Msg } from "../types";

interface Props {
  message: Msg;
}

export function ChatMessage({ message }: Props) {
  const isUser = message.role === "user";

  const visualPhotos: CarouselPhoto[] | undefined = message.visualResults?.map((r) => ({
    filePath: r.file_path,
    fileName: r.file_name,
    label: `${Math.round(r.score * 100)}% match`,
  }));

  return (
    <div className={`message ${isUser ? "message-user" : "message-assistant"}`}>
      <div className={`bubble ${isUser ? "bubble-user" : "bubble-assistant"} ${message.error ? "bubble-error" : ""}`}>
        {message.queryImageUrl && (
          <img src={message.queryImageUrl} alt="Uploaded" className="query-image-preview" />
        )}

        {isUser ? (
          <p className="bubble-text">{message.content}</p>
        ) : (
          <div className="bubble-text markdown">
            <ReactMarkdown>{message.content}</ReactMarkdown>
          </div>
        )}

        {!isUser && visualPhotos && <PhotoCarousel photos={visualPhotos} />}
        {!isUser && message.sources && <SourceList sources={message.sources} />}

        {!isUser && message.chunks_used !== undefined && (
          <p className="bubble-meta">{message.chunks_used} results · {message.model}</p>
        )}
      </div>
    </div>
  );
}
