import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { sendChatStream, visualSearch, transcribe } from "./api";
import {
  deleteConversation,
  listConversations,
  loadConversation,
  loadFilters,
  loadMostRecent,
  loadSidebarCollapsed,
  migrateV1IfNeeded,
  newId,
  saveConversation,
  saveFilters,
  saveSidebarCollapsed,
  titleFrom,
  type ConversationMeta,
} from "./storage";
import { ChatMessage } from "./components/ChatMessage";
import { Header } from "./components/Header";
import { Sidebar } from "./components/Sidebar";
import { BottomSheet } from "./components/BottomSheet";
import { Drawer } from "./components/Drawer";
import { ConversationList } from "./components/ConversationList";
import { FilterControls, MODALITY_OPTIONS } from "./components/FilterControls";
import { useMediaQuery } from "./hooks/useMediaQuery";
import { PlusIcon, SendIcon, CameraIcon, ImageIcon, CloseIcon, ComposeIcon, MicIcon, StopIcon } from "./components/icons";
import { useRecorder } from "./hooks/useRecorder";
import { useSystemStatus } from "./hooks/useSystemStatus";
import type { ChatMessage as Msg } from "./types";
import "./App.css";

const EXAMPLE_PROMPTS = [
  "What does the resume say about Python experience?",
  "What port does the photo service run on?",
  "Summarize the home network notes.",
];

let msgId = 0;
const nextId = () => String(++msgId);

export default function App() {
  const initial = useMemo(() => {
    migrateV1IfNeeded();
    const recent = loadMostRecent();
    if (recent) {
      msgId = recent.messages.reduce((max, m) => Math.max(max, Number(m.id) || 0), msgId);
    }
    return recent ?? { id: newId(), title: "", messages: [] as Msg[], updatedAt: 0 };
  }, []);

  const [convId, setConvId] = useState(initial.id);
  const [messages, setMessages] = useState<Msg[]>(initial.messages);
  const [convList, setConvList] = useState<ConversationMeta[]>(() => listConversations());
  const [sidebarCollapsed, setSidebarCollapsed] = useState(loadSidebarCollapsed);
  const [input, setInput] = useState("");
  const [modality, setModality] = useState(() => loadFilters().modality);
  const [topK, setTopK] = useState<number | null>(() => loadFilters().topK);
  const [loading, setLoading] = useState(false);
  const [actionSheet, setActionSheet] = useState(false);
  const [statusSheet, setStatusSheet] = useState(false);
  const [chatsDrawer, setChatsDrawer] = useState(false); // mobile chat history
  const [liveNote, setLiveNote] = useState(""); // screen-reader announcements
  const isDesktop = useMediaQuery("(min-width: 1024px)");

  // The drawer is a mobile affordance — drop it if the viewport grows into the sidebar.
  useEffect(() => {
    if (isDesktop) setChatsDrawer(false);
  }, [isDesktop]);

  const { health, stats } = useSystemStatus();

  const recorder = useRecorder();
  const [transcribing, setTranscribing] = useState(false);
  const [transcribeError, setTranscribeError] = useState<string | null>(null);
  const voiceError = recorder.error ?? transcribeError;

  const bottomRef = useRef<HTMLDivElement>(null);
  const chatAreaRef = useRef<HTMLElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const cameraInputRef = useRef<HTMLInputElement>(null);
  const libraryInputRef = useRef<HTMLInputElement>(null);
  const abortRef = useRef<AbortController | null>(null);

  // Track whether the user is at (or near) the bottom of the chat.
  const pinnedRef = useRef(true);
  const scrollRafRef = useRef(0);
  const prevCountRef = useRef(0);

  function onChatScroll() {
    const el = chatAreaRef.current;
    if (!el) return;
    pinnedRef.current = el.scrollHeight - el.scrollTop - el.clientHeight < 80;
  }

  // Auto-scroll only while pinned, so scrolling up mid-stream isn't yanked back
  // down. rAF-coalesced; token updates scroll instantly ("instant" bypasses the
  // chat-area's CSS scroll-behavior:smooth), new messages scroll smoothly.
  useEffect(() => {
    const isNewMessage = messages.length !== prevCountRef.current;
    prevCountRef.current = messages.length;
    if (!pinnedRef.current) return;
    cancelAnimationFrame(scrollRafRef.current);
    scrollRafRef.current = requestAnimationFrame(() => {
      bottomRef.current?.scrollIntoView({ behavior: isNewMessage ? "smooth" : "instant" });
    });
    return () => cancelAnimationFrame(scrollRafRef.current);
  }, [messages]);

  // Auto-grow textarea
  useEffect(() => {
    const el = inputRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 140) + "px";
  }, [input]);

  // Remember last-used filters across sessions
  useEffect(() => {
    saveFilters({ modality, topK });
  }, [modality, topK]);

  // Ctrl/Cmd+K focuses the composer from anywhere
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        inputRef.current?.focus();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  // Persist conversation — gated on !loading so mid-stream token updates
  // don't thrash localStorage; the final state is saved when loading flips false.
  const skipSaveRef = useRef(true); // skip the initial-mount save (nothing changed yet)
  useEffect(() => {
    if (loading) return;
    if (skipSaveRef.current) {
      skipSaveRef.current = false;
      return;
    }
    if (messages.length === 0) return;
    saveConversation({ id: convId, title: titleFrom(messages), messages, updatedAt: Date.now() });
    setConvList(listConversations());
  }, [messages, loading, convId]);

  // Flush the current conversation before leaving it (covers mid-stream switches,
  // where the !loading-gated effect hasn't saved the in-flight messages yet).
  function flushCurrent() {
    abortRef.current?.abort();
    setLoading(false);
    if (messages.length > 0) {
      saveConversation({ id: convId, title: titleFrom(messages), messages, updatedAt: Date.now() });
      setConvList(listConversations());
    }
  }

  function newChat() {
    flushCurrent();
    setConvId(newId());
    setMessages([]);
    setInput("");
  }

  function selectConversation(id: string) {
    if (id === convId) return;
    flushCurrent();
    const conv = loadConversation(id);
    if (!conv) return;
    msgId = conv.messages.reduce((max, m) => Math.max(max, Number(m.id) || 0), msgId);
    skipSaveRef.current = true; // opening a chat shouldn't bump its updatedAt
    setConvId(conv.id);
    setMessages(conv.messages);
  }

  function removeConversation(id: string) {
    deleteConversation(id);
    const remaining = listConversations();
    setConvList(remaining);
    if (id !== convId) return;
    // Deliberately no flushCurrent() here — that would resurrect the deleted chat.
    abortRef.current?.abort();
    setLoading(false);
    skipSaveRef.current = true;
    const next = remaining[0] ? loadConversation(remaining[0].id) : null;
    if (next) {
      msgId = next.messages.reduce((max, m) => Math.max(max, Number(m.id) || 0), msgId);
      setConvId(next.id);
      setMessages(next.messages);
    } else {
      setConvId(newId());
      setMessages([]);
      setInput("");
    }
  }

  function toggleSidebar() {
    setSidebarCollapsed((v) => {
      saveSidebarCollapsed(!v);
      return !v;
    });
  }

  // One header button: toggles the sidebar on desktop, opens the drawer on mobile.
  function onHistoryClick() {
    if (isDesktop) toggleSidebar();
    else setChatsDrawer(true);
  }

  function fillExample(p: string) {
    setInput(p);
    setTimeout(() => inputRef.current?.focus(), 0);
  }

  const ollamaOffline = health !== null && !health.ollama;

  async function send(q: string) {
    if (!q || loading || ollamaOffline) return;

    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    const userMsg: Msg = { id: nextId(), role: "user", content: q };
    const assistantId = nextId();
    const assistantMsg: Msg = {
      id: assistantId,
      role: "assistant",
      content: "",
      streaming: true,
      question: q,
    };
    pinnedRef.current = true; // sending always jumps to the new exchange
    setMessages((prev) => [...prev, userMsg, assistantMsg]);
    setInput("");
    setLoading(true);
    setLiveNote("");

    const patch = (p: Partial<Msg>) =>
      setMessages((prev) => prev.map((m) => (m.id === assistantId ? { ...m, ...p } : m)));
    const append = (delta: string) =>
      setMessages((prev) =>
        prev.map((m) => (m.id === assistantId ? { ...m, content: m.content + delta } : m))
      );

    try {
      await sendChatStream(q, modality || undefined, topK ?? undefined, {
        signal: controller.signal,
        onSources: (sources, meta) =>
          patch({ sources, model: meta.model, chunks_used: meta.chunks_used }),
        onToken: (delta) => append(delta),
        onDone: () => {
          patch({ streaming: false });
          setLiveNote("Answer ready");
        },
        onError: () => {
          patch({ content: "Failed to generate a response.", error: true, streaming: false });
          setLiveNote("Response failed");
        },
      });
    } catch (e) {
      if ((e as Error).name !== "AbortError") {
        patch({
          content: "Failed to reach the API. Is the backend running?",
          error: true,
          streaming: false,
        });
      }
    } finally {
      // Only the controller still owned by abortRef should reset loading —
      // a superseded (aborted) stream's cleanup must not clobber a newer one.
      if (abortRef.current === controller) {
        setLoading(false);
        setTimeout(() => inputRef.current?.focus(), 0);
      }
    }
  }

  function submit() {
    send(input.trim());
  }

  // Stable identity so memoized ChatMessages don't re-render when App does.
  const sendRef = useRef(send);
  sendRef.current = send;
  const retry = useCallback((q: string) => sendRef.current(q), []);

  function stopGeneration() {
    abortRef.current?.abort();
    setMessages((prev) => prev.map((m) => (m.streaming ? { ...m, streaming: false } : m)));
    setLoading(false);
  }

  async function startRecording() {
    setTranscribeError(null);
    await recorder.start();
  }

  async function finishRecording() {
    const blob = await recorder.stop();
    if (!blob) return;
    setTranscribing(true);
    try {
      const text = await transcribe(blob);
      if (text) {
        setInput((prev) => (prev ? prev.trimEnd() + " " : "") + text);
        setTimeout(() => inputRef.current?.focus(), 0);
      }
    } catch {
      setTranscribeError("Transcription failed.");
    } finally {
      setTranscribing(false);
    }
  }

  function formatTime(s: number) {
    const m = Math.floor(s / 60);
    return `${m}:${(s % 60).toString().padStart(2, "0")}`;
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
    pinnedRef.current = true;
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
      <div className="visually-hidden" role="status" aria-live="polite">
        {liveNote}
      </div>
      <Sidebar
        items={convList}
        activeId={convId}
        collapsed={sidebarCollapsed}
        onSelect={selectConversation}
        onNew={newChat}
        onDelete={removeConversation}
      >
        <FilterControls modality={modality} setModality={setModality} topK={topK} setTopK={setTopK} />
      </Sidebar>
      <div className="chat-pane">
        <Header
          health={health}
          onStatusClick={() => setStatusSheet(true)}
          onNewChat={newChat}
          onToggleSidebar={onHistoryClick}
        />

        <main className="chat-area" ref={chatAreaRef} onScroll={onChatScroll}>
          <div className="chat-col">
            {messages.length === 0 && (
              <div className="empty-state">
                <p>Ask anything about your NAS files.</p>
                <div className="example-prompts">
                  {EXAMPLE_PROMPTS.map((p) => (
                    <button key={p} className="example-prompt" onClick={() => fillExample(p)}>
                      {p}
                    </button>
                  ))}
                </div>
                <p className="empty-hint">Tap + to search by photo.</p>
              </div>
            )}
            {messages.map((m) => (
              <ChatMessage key={m.id} message={m} indexTotal={stats?.total} onRetry={retry} />
            ))}
            <div ref={bottomRef} />
          </div>
        </main>

        <footer className="composer">
          <div className="chat-col">
            {ollamaOffline && (
              <div className="offline-banner">
                Ollama offline — answers unavailable. Tap the status dot for details.
              </div>
            )}
            {voiceError && <div className="voice-error">{voiceError}</div>}
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
              {recorder.state === "recording" ? (
                <>
                  <button
                    className="pill-btn cancel-btn"
                    onClick={recorder.cancel}
                    aria-label="Cancel recording"
                  >
                    <CloseIcon />
                  </button>
                  <div className="recording-indicator">
                    <span className="rec-dot" />
                    <span className="rec-time">{formatTime(recorder.seconds)}</span>
                    <span className="rec-label">Recording…</span>
                  </div>
                  <button
                    className="pill-btn send-btn"
                    onClick={finishRecording}
                    aria-label="Stop and transcribe"
                  >
                    <StopIcon />
                  </button>
                </>
              ) : (
                <>
                  <button
                    className="pill-btn plus-btn"
                    onClick={() => setActionSheet(true)}
                    disabled={loading || transcribing}
                    aria-label="Attach or filter"
                  >
                    <PlusIcon />
                  </button>
                  <textarea
                    ref={inputRef}
                    className="composer-input"
                    rows={1}
                    placeholder={transcribing ? "Transcribing…" : "Ask about your files…"}
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyDown={onKeyDown}
                    onFocus={() => setTimeout(() => bottomRef.current?.scrollIntoView({ behavior: "smooth" }), 300)}
                    disabled={loading || transcribing}
                    autoComplete="off"
                    autoCorrect="off"
                    autoCapitalize="sentences"
                    spellCheck={false}
                  />
                  {loading ? (
                    <button
                      className="pill-btn stop-btn"
                      onClick={stopGeneration}
                      aria-label="Stop generating"
                    >
                      <StopIcon />
                    </button>
                  ) : input.trim() ? (
                    <button
                      className="pill-btn send-btn"
                      onClick={submit}
                      disabled={ollamaOffline}
                      aria-label="Send"
                    >
                      <SendIcon />
                    </button>
                  ) : (
                    <button
                      className="pill-btn mic-btn"
                      onClick={startRecording}
                      disabled={transcribing}
                      aria-label="Record voice"
                    >
                      <MicIcon />
                    </button>
                  )}
                </>
              )}
            </div>
          </div>
        </footer>
      </div>

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
        <div className="visual-tiles">
          <button className="visual-tile" onClick={() => cameraInputRef.current?.click()}>
            <CameraIcon size={26} />
            <span>Camera</span>
          </button>
          <button className="visual-tile" onClick={() => libraryInputRef.current?.click()}>
            <ImageIcon size={26} />
            <span>Photos</span>
          </button>
        </div>

        <FilterControls modality={modality} setModality={setModality} topK={topK} setTopK={setTopK} />
      </BottomSheet>

      {/* Chats drawer (mobile chat history — desktop uses the sidebar) */}
      <Drawer open={chatsDrawer} onClose={() => setChatsDrawer(false)}>
        <div className="sidebar-top">
          <span className="sidebar-label">Chats</span>
          <button
            className="icon-btn"
            onClick={() => {
              setChatsDrawer(false);
              newChat();
            }}
            title="New chat"
            aria-label="New chat"
          >
            <ComposeIcon />
          </button>
        </div>
        <ConversationList
          items={convList}
          activeId={convId}
          onSelect={(id) => {
            selectConversation(id);
            setChatsDrawer(false);
          }}
          onDelete={removeConversation}
        />
      </Drawer>

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
