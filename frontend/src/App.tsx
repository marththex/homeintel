import { useEffect, useRef, useState } from "react";
import { sendChat, visualSearch } from "./api";
import { ChatMessage } from "./components/ChatMessage";
import { StatusBar } from "./components/StatusBar";
import type { ChatMessage as Msg } from "./types";
import "./App.css";

const MODALITY_OPTIONS = [
  { value: "", label: "All", short: "All" },
  { value: "document", label: "Documents", short: "Docs" },
  { value: "image", label: "Images", short: "Imgs" },
  { value: "audio", label: "Audio", short: "Audio" },
];

let msgId = 0;
const nextId = () => String(++msgId);

export default function App() {
  const [messages, setMessages] = useState<Msg[]>([]);
  const [input, setInput] = useState("");
  const [modality, setModality] = useState("");
  const [loading, setLoading] = useState(false);
  const [isMobile, setIsMobile] = useState(window.innerWidth < 600);

  useEffect(() => {
    const onResize = () => setIsMobile(window.innerWidth < 600);
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, []);
  const bottomRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  async function submit() {
    const q = input.trim();
    if (!q || loading) return;

    const userMsg: Msg = { id: nextId(), role: "user", content: q };
    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setLoading(true);

    try {
      const data = await sendChat(q, modality || undefined);
      const assistantMsg: Msg = {
        id: nextId(),
        role: "assistant",
        content: data.answer,
        sources: data.sources,
        chunks_used: data.chunks_used,
        model: data.model,
      };
      setMessages((prev) => [...prev, assistantMsg]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        {
          id: nextId(),
          role: "assistant",
          content: "Failed to reach the API. Is the backend running?",
          error: true,
        },
      ]);
    } finally {
      setLoading(false);
      setTimeout(() => inputRef.current?.focus(), 0);
    }
  }

  function onKeyDown(e: React.KeyboardEvent) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submit();
    }
  }

  async function onImageUpload(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file || loading) return;
    e.target.value = "";

    const queryImageUrl = URL.createObjectURL(file);
    const userMsg: Msg = {
      id: nextId(),
      role: "user",
      content: `📷 Visual search: ${file.name}`,
      queryImageUrl,
    };
    setMessages((prev) => [...prev, userMsg]);
    setLoading(true);

    try {
      const data = await visualSearch(file);
      const assistantMsg: Msg = {
        id: nextId(),
        role: "assistant",
        content: data.results.length > 0
          ? `Found ${data.results.length} visually similar photo${data.results.length !== 1 ? "s" : ""}.`
          : "No visually similar photos found (similarity threshold: 70%).",
        visualResults: data.results,
        queryImageUrl,
      };
      setMessages((prev) => [...prev, assistantMsg]);
    } catch {
      setMessages((prev) => [
        ...prev,
        {
          id: nextId(),
          role: "assistant",
          content: "Visual search failed. Is the CLIP index built? Run scripts/index_visual.py first.",
          error: true,
        },
      ]);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="app">
      <header className="header">
        <h1 className="logo">HomeIntel</h1>
        <div className="header-right">
          <StatusBar />
          <button
            className="new-chat-btn"
            onClick={() => { setMessages([]); setInput(""); }}
            title="New chat"
          >
            ✏️
          </button>
        </div>
      </header>

      <main className="chat-area">
        {messages.length === 0 && (
          <div className="empty-state">
            <p>Ask anything about your NAS files.</p>
            <p className="empty-hint">Try "What does my resume say about Python experience?"</p>
          </div>
        )}
        {messages.map((m) => (
          <ChatMessage key={m.id} message={m} />
        ))}
        {loading && (
          <div className="message message-assistant">
            <div className="bubble bubble-assistant bubble-loading">
              <span className="dot" /><span className="dot" /><span className="dot" />
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </main>

      <footer className="input-area">
        <select
          className="modality-select"
          value={modality}
          onChange={(e) => setModality(e.target.value)}
          disabled={loading}
          title="Filter by modality"
        >
          {MODALITY_OPTIONS.map((o) => (
            <option key={o.value} value={o.value}>{isMobile ? o.short : o.label}</option>
          ))}
        </select>
        <textarea
          ref={inputRef}
          className="chat-input"
          rows={1}
          placeholder="Ask about your files…"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={onKeyDown}
          onFocus={() => setTimeout(() => bottomRef.current?.scrollIntoView({ behavior: "smooth" }), 300)}
          disabled={loading}
          autoComplete="off"
          autoCorrect="off"
          autoCapitalize="sentences"
          spellCheck={false}
        />
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          capture="environment"
          style={{ display: "none" }}
          onChange={onImageUpload}
        />
        <button
          className="camera-btn"
          onClick={() => fileInputRef.current?.click()}
          disabled={loading}
          title="Visual photo search"
        >
          📷
        </button>
        <button className="send-btn" onClick={submit} disabled={loading || !input.trim()}>
          {loading ? "…" : "Send"}
        </button>
      </footer>
    </div>
  );
}
