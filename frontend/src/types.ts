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
}

export interface StatsResponse {
  document: number;
  image: number;
  audio: number;
  total: number;
}

export interface HealthResponse {
  status: "ok" | "degraded";
  ollama: boolean;
  qdrant: boolean;
}
