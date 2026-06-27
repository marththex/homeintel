export interface SourceDoc {
  file_name: string;
  file_path: string;
  modality: "document" | "image" | "audio";
  excerpt: string;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: SourceDoc[];
  chunks_used?: number;
  model?: string;
  error?: boolean;
  // Visual search
  visualResults?: VisualResult[];
  queryImageUrl?: string;
}

export interface StatsResponse {
  document: number;
  image: number;
  audio: number;
  visual: number;
  total: number;
}

export interface HealthResponse {
  status: "ok" | "degraded";
  ollama: boolean;
  qdrant: boolean;
}

export interface VisualResult {
  file_path: string;
  file_name: string;
  score: number;
}

export interface VisualSearchResponse {
  results: VisualResult[];
}
