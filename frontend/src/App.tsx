import { useEffect, useRef, useState } from "react";
import { sendChat, visualSearch } from "./api";
import { ChatMessage } from "./components/ChatMessage";
import { Header } from "./components/Header";
import { BottomSheet } from "./components/BottomSheet";
import { PlusIcon, SendIcon, CameraIcon, ImageIcon, CloseIcon } from "./components/icons";
import { useSystemStatus } from "./hooks/useSystemStatus";
import type { ChatMessage as Msg } from "./types";
import "./App.css";

const MODALITY_OPTIONS = [
  { value: "", label: "All" },
  { value: "document", label: "Documents" },
  { value: "image", label: "Images" },
  { value: "audio", label: "Audio" },
];

let msgId = 0;
const nextId = () => String(++msgId);

export default function App() {
  const [messages, setMessages] = useState<Msg[]>([]);
  const [input, setInput] = useState("");
  const [modality, setModality] = useState("");
  const [topK, setTopK] = useState<number | null>(null);
  const [loading, setLoading] = useState(false);
  const [actionSheet, setActionSheet] = useState(false);
  const [statusSheet, setStatusSheet] = useState(false);

  const { health, stats } = useSystemStatus();

  const bottomRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const cameraInputRef = useRef<HTMLInputElement>(null);
  const libraryInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Auto-grow textarea
  useEffect(() => {
    const el = inputRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 140) + "px";
  }, [input]);

  function newChat() {
    setMessages([]);
    setInput("");
  }

  async function submit() {
    const q = input.trim();
    if (!q || loading) return;

    const userMsg: Msg = { id: nextId(), role: "user", content: q };
    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setLoading(true);

    try {
      const data = await sendChat(q, modality || undefined, topK ?? undefined);
      const assistantMsg: Msg = {
        id: nextId(),
        role: "assistant",
        content: data.answer,
        sources: data.sources,
        chunks_used: data.chunks_used,
        model: data.model,
      };
      setMessages((prev) => [...prev, assistantMsg]);
    } catch {
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
    e.target.value = "";
    if (!file || loading) return;
    setActionSheet(false);

    const queryImageUrl = URL.createObjectURL(file);
    const userMsg: Msg = {
      id: nextId(),
      role: "user",
      content: "Visual search",
      queryImageUrl,
    };
    setMessages((prev) => [...prev, userMsg]);
    setLoading(true);

    try {
      const data = await visualSearch(file);
      const assistantMsg: Msg = {
        id: nextId(),
        role: "assistant",
        content:
          data.results.length > 0
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

  const modalityLabel = MODALITY_OPTIONS.find((o) => o.value === modality)?.label ?? "All";

  return (
    <div className="app">
      <Header
        health={health}
        onStatusClick={() => setStatusSheet(true)}
        onNewChat={newChat}
      />

      <main className="chat-area">
        {messages.length === 0 && (
          <div className="empty-state">
            <p>Ask anything about your NAS files.</p>
            <p className="empty-hint">Try "What does my resume say about Python experience?"</p>
            <p className="empty-hint">Tap + to search by photo.</p>
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

      <footer className="composer">
        {(modality || topK !== null) && (
          <div className="chip-row">
            {modality && (
              <button className="chip" onClick={() => setModality("")}>
                {modalityLabel} <CloseIcon size={13} />
              </button>
            )}
            {topK !== null && (
              <button className="chip" onClick={() => setTopK(null)}>
                Top {topK} <CloseIcon size={13} />
              </button>
            )}
          </div>
        )}

        <div className="composer-pill">
          <button
            className="pill-btn plus-btn"
            onClick={() => setActionSheet(true)}
            disabled={loading}
            aria-label="Attach or filter"
          >
            <PlusIcon />
          </button>
          <textarea
            ref={inputRef}
            className="composer-input"
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
          <button
            className="pill-btn send-btn"
            onClick={submit}
            disabled={loading || !input.trim()}
            aria-label="Send"
          >
            <SendIcon />
          </button>
        </div>
      </footer>

      {/* Hidden file inputs for visual search */}
      <input
        ref={cameraInputRef}
        type="file"
        accept="image/*"
        capture="environment"
        style={{ display: "none" }}
        onChange={onImageUpload}
      />
      <input
        ref={libraryInputRef}
        type="file"
        accept="image/*"
        style={{ display: "none" }}
        onChange={onImageUpload}
      />

      {/* "+" action sheet */}
      <BottomSheet open={actionSheet} onClose={() => setActionSheet(false)} title="Search options">
        <div className="sheet-section-label">Visual search</div>
        <button className="sheet-action" onClick={() => cameraInputRef.current?.click()}>
          <CameraIcon /> <span>Take a photo</span>
        </button>
        <button className="sheet-action" onClick={() => libraryInputRef.current?.click()}>
          <ImageIcon /> <span>Choose from library</span>
        </button>

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
          {[3, 6, 10, 15, 20].map((n) => (
            <button
              key={n}
              className={`sheet-chip ${topK === n ? "active" : ""}`}
              onClick={() => setTopK(n)}
            >
              {n}
            </button>
          ))}
        </div>
      </BottomSheet>

      {/* Status sheet */}
      <BottomSheet open={statusSheet} onClose={() => setStatusSheet(false)} title="System status">
        <div className="status-rows">
          <div className="status-row">
            <span>Ollama (LLM)</span>
            <span className={`status-badge ${health?.ollama ? "ok" : "bad"}`}>
              {health?.ollama ? "Connected" : "Offline"}
            </span>
          </div>
          <div className="status-row">
            <span>Qdrant (Vector DB)</span>
            <span className={`status-badge ${health?.qdrant ? "ok" : "bad"}`}>
              {health?.qdrant ? "Connected" : "Offline"}
            </span>
          </div>
        </div>

        {stats && (
          <>
            <div className="sheet-section-label">Text index (searchable by keyword)</div>
            <div className="status-rows">
              <div className="status-row"><span>📄 Documents</span><span>{stats.document.toLocaleString()}</span></div>
              <div className="status-row"><span>🖼️ Images (captions)</span><span>{stats.image.toLocaleString()}</span></div>
              <div className="status-row"><span>🎵 Audio</span><span>{stats.audio.toLocaleString()}</span></div>
              <div className="status-row total"><span>Total chunks</span><span>{stats.total.toLocaleString()}</span></div>
            </div>

            <div className="sheet-section-label">Visual index (search by photo 📷)</div>
            <div className="status-rows">
              <div className="status-row"><span>🔍 Photos (CLIP)</span><span>{stats.visual.toLocaleString()}</span></div>
            </div>
          </>
        )}
      </BottomSheet>
    </div>
  );
}
