import type { ChatMessage } from "./types";

const V1_KEY = "homeintel.conversation.v1";
const V2_KEY = "homeintel.conversations.v2";
const SIDEBAR_KEY = "homeintel.sidebar.collapsed";
const MAX_CONVERSATIONS = 30;

export interface Conversation {
  id: string;
  title: string;
  messages: ChatMessage[];
  updatedAt: number;
}

export type ConversationMeta = Pick<Conversation, "id" | "title" | "updatedAt">;

export function newId(): string {
  // crypto.randomUUID is missing in insecure contexts (plain-http LAN access).
  return typeof crypto !== "undefined" && "randomUUID" in crypto
    ? crypto.randomUUID()
    : `c-${Date.now().toString(36)}${Math.random().toString(36).slice(2, 10)}`;
}

// Persist only serializable fields. Strips transient blob: object URLs
// (queryImageUrl — invalid after reload) and the streaming flag.
function stripTransient(messages: ChatMessage[]): ChatMessage[] {
  return messages.map(({ queryImageUrl, streaming, ...rest }) => rest);
}

function revive(messages: ChatMessage[]): ChatMessage[] {
  return messages.map((m) => ({ ...m, streaming: false }));
}

function readAll(): Conversation[] {
  try {
    const raw = localStorage.getItem(V2_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed.filter(
      (c) => c && typeof c.id === "string" && Array.isArray(c.messages)
    );
  } catch {
    return [];
  }
}

function writeAll(convs: Conversation[]): void {
  try {
    const sorted = [...convs].sort((a, b) => b.updatedAt - a.updatedAt);
    localStorage.setItem(V2_KEY, JSON.stringify(sorted.slice(0, MAX_CONVERSATIONS)));
  } catch {
    /* quota / private mode — ignore */
  }
}

export function titleFrom(messages: ChatMessage[]): string {
  const first = messages.find((m) => m.role === "user")?.content.trim() ?? "";
  const flat = first.replace(/\s+/g, " ");
  if (!flat) return "New chat";
  return flat.length > 40 ? flat.slice(0, 40).trimEnd() + "…" : flat;
}

// One-time upgrade from the single-conversation v1 key.
export function migrateV1IfNeeded(): void {
  try {
    const raw = localStorage.getItem(V1_KEY);
    if (!raw) return;
    if (!localStorage.getItem(V2_KEY)) {
      const messages = JSON.parse(raw);
      if (Array.isArray(messages) && messages.length > 0) {
        writeAll([
          { id: newId(), title: titleFrom(messages), messages, updatedAt: Date.now() },
        ]);
      }
    }
    localStorage.removeItem(V1_KEY);
  } catch {
    /* corrupt v1 — drop it */
    try {
      localStorage.removeItem(V1_KEY);
    } catch {
      /* ignore */
    }
  }
}

export function listConversations(): ConversationMeta[] {
  return readAll()
    .map(({ id, title, updatedAt }) => ({ id, title, updatedAt }))
    .sort((a, b) => b.updatedAt - a.updatedAt);
}

export function loadConversation(id: string): Conversation | null {
  const conv = readAll().find((c) => c.id === id);
  return conv ? { ...conv, messages: revive(conv.messages) } : null;
}

export function loadMostRecent(): Conversation | null {
  const [first] = readAll().sort((a, b) => b.updatedAt - a.updatedAt);
  return first ? { ...first, messages: revive(first.messages) } : null;
}

export function saveConversation(conv: Conversation): void {
  const rest = readAll().filter((c) => c.id !== conv.id);
  writeAll([{ ...conv, messages: stripTransient(conv.messages) }, ...rest]);
}

export function deleteConversation(id: string): void {
  writeAll(readAll().filter((c) => c.id !== id));
}

const FILTERS_KEY = "homeintel.filters.v1";

export interface SavedFilters {
  modality: string;
  topK: number | null;
}

export function loadFilters(): SavedFilters {
  try {
    const raw = localStorage.getItem(FILTERS_KEY);
    if (!raw) return { modality: "", topK: null };
    const parsed = JSON.parse(raw);
    return {
      modality: typeof parsed.modality === "string" ? parsed.modality : "",
      topK: typeof parsed.topK === "number" ? parsed.topK : null,
    };
  } catch {
    return { modality: "", topK: null };
  }
}

export function saveFilters(filters: SavedFilters): void {
  try {
    localStorage.setItem(FILTERS_KEY, JSON.stringify(filters));
  } catch {
    /* ignore */
  }
}

export function loadSidebarCollapsed(): boolean {
  try {
    return localStorage.getItem(SIDEBAR_KEY) === "1";
  } catch {
    return false;
  }
}

export function saveSidebarCollapsed(collapsed: boolean): void {
  try {
    localStorage.setItem(SIDEBAR_KEY, collapsed ? "1" : "0");
  } catch {
    /* ignore */
  }
}
