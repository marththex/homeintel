import type { HealthResponse, SourceDoc, StatsResponse, VisualSearchResponse } from "./types";

const BASE = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

export async function sendChat(
  question: string,
  modality_filter?: string
): Promise<{ answer: string; sources: SourceDoc[]; chunks_used: number; model: string }> {
  const res = await fetch(`${BASE}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, modality_filter: modality_filter || null }),
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
