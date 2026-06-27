import type { HealthResponse, SourceDoc, StatsResponse, VisualSearchResponse } from "./types";

const BASE = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

export interface StreamHandlers {
  onSources?: (sources: SourceDoc[], meta: { model: string; chunks_used: number }) => void;
  onToken?: (delta: string) => void;
  onDone?: () => void;
  onError?: (detail: string) => void;
  signal?: AbortSignal;
}

function parseFrame(frame: string): { event?: string; data?: any } {
  let event: string | undefined;
  const dataLines: string[] = [];
  for (const line of frame.split("\n")) {
    if (line.startsWith("event:")) event = line.slice(6).trim();
    else if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
  }
  if (dataLines.length === 0) return { event };
  try {
    return { event, data: JSON.parse(dataLines.join("\n")) };
  } catch {
    return { event };
  }
}

export async function sendChatStream(
  question: string,
  modality_filter: string | undefined,
  top_k: number | undefined,
  handlers: StreamHandlers
): Promise<void> {
  const res = await fetch(`${BASE}/chat/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      question,
      modality_filter: modality_filter || null,
      top_k: top_k ?? null,
    }),
    signal: handlers.signal,
  });
  if (!res.ok || !res.body) throw new Error(`API error ${res.status}`);

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  const dispatchFrame = (frame: string) => {
    const { event, data } = parseFrame(frame);
    if (!event) return;
    if (event === "sources") {
      handlers.onSources?.(data.sources, { model: data.model, chunks_used: data.chunks_used });
    } else if (event === "token") {
      handlers.onToken?.(data.delta);
    } else if (event === "done") {
      handlers.onDone?.();
    } else if (event === "error") {
      handlers.onError?.(data?.detail ?? "stream error");
    }
  };

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let sep: number;
      while ((sep = buffer.indexOf("\n\n")) !== -1) {
        const frame = buffer.slice(0, sep);
        buffer = buffer.slice(sep + 2);
        dispatchFrame(frame);
      }
    }
    buffer += decoder.decode();
    if (buffer.trim()) dispatchFrame(buffer);
  } finally {
    reader.cancel().catch(() => {});
  }
}

export async function sendChat(
  question: string,
  modality_filter?: string,
  top_k?: number
): Promise<{ answer: string; sources: SourceDoc[]; chunks_used: number; model: string }> {
  const res = await fetch(`${BASE}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      question,
      modality_filter: modality_filter || null,
      top_k: top_k ?? null,
    }),
  });
  if (!res.ok) throw new Error(`API error ${res.status}`);
  return res.json();
}

export async function getHealth(): Promise<HealthResponse> {
  const res = await fetch(`${BASE}/health`);
  if (!res.ok) throw new Error(`Health check failed ${res.status}`);
  return res.json();
}

export async function getStats(): Promise<StatsResponse> {
  const res = await fetch(`${BASE}/stats`);
  if (!res.ok) throw new Error(`Stats failed ${res.status}`);
  return res.json();
}

export async function visualSearch(file: File, topK = 10): Promise<VisualSearchResponse> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${BASE}/visual-search?top_k=${topK}`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) throw new Error(`Visual search error ${res.status}`);
  return res.json();
}

export function fileUrl(path: string): string {
  return `${BASE}/file?path=${encodeURIComponent(path)}`;
}
