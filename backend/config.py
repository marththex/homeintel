"""
config.py — Single source of truth for all runtime settings.

All values are read from environment variables (or a .env file).
Pydantic-settings validates types and raises a clear error at startup
if anything required is missing or wrong.

Note: List-type settings are stored as plain comma-separated strings
to avoid pydantic-settings v2 attempting JSON parsing before validators
run. Access them via the _list properties (e.g. supported_extensions_list).

Usage anywhere in the codebase:
    from config import settings
    print(settings.ollama_base_url_str)
    print(settings.supported_extensions_list)
"""

from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic import AnyHttpUrl, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=Path(__file__).parent.parent / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── NAS / File System ─────────────────────────────────────────────────────
    nas_watch_path: Path = Field(
        default=Path("/data/files"),
        description="Root directory to watch and index.",
    )
    supported_extensions: str = Field(
        default=".pdf,.docx,.txt,.md,.png,.jpg,.jpeg,.mp3,.wav",
        description="Comma-separated file extensions to index.",
    )
    watcher_exclude_paths: str = Field(
        default="",
        description="Comma-separated absolute paths to exclude from the watcher.",
    )

    # ── ChromaDB ──────────────────────────────────────────────────────────────
    chroma_path: Path = Field(
        default=Path("/data/chroma"),
        description="Directory where ChromaDB persists its data.",
    )
    chroma_collection_name: str = Field(
        default="homeintel",
        description="Name of the ChromaDB collection.",
    )

    # ── Ollama ────────────────────────────────────────────────────────────────
    ollama_base_url: AnyHttpUrl = Field(
        default="http://ollama:11434",
        description="Base URL of the Ollama server.",
    )
    ollama_llm_model: str = Field(
        default="qwen3.5:9b",
        description="Ollama model tag used for chat / RAG generation.",
    )
    ollama_embed_model: str = Field(
        default="nomic-embed-text",
        description="Ollama model tag used for generating text embeddings.",
    )

    # ── RAG ───────────────────────────────────────────────────────────────────
    chunk_size: int = Field(
        default=512,
        ge=64,
        le=4096,
        description="Target token count per chunk during ingestion.",
    )
    chunk_overlap: int = Field(
        default=64,
        ge=0,
        description="Token overlap between consecutive chunks.",
    )
    retrieval_top_k: int = Field(
        default=6,
        ge=1,
        le=20,
        description="Number of chunks retrieved per query.",
    )

    # ── API ───────────────────────────────────────────────────────────────────
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000, ge=1024, le=65535)
    log_level: str = Field(default="info")

    # ── Dev helpers ───────────────────────────────────────────────────────────
    skip_llm_health_check: bool = Field(
        default=False,
        description="Skip the Ollama connectivity check on startup.",
    )

    # ── Derived properties (not env vars) ─────────────────────────────────────

    @property
    def ollama_base_url_str(self) -> str:
        """Return Ollama base URL as a plain string (no trailing slash)."""
        return str(self.ollama_base_url).rstrip("/")

    @property
    def supported_extensions_list(self) -> List[str]:
        return [e.strip() for e in self.supported_extensions.split(",") if e.strip()]

    @property
    def watcher_exclude_paths_list(self) -> List[str]:
        return [p.strip() for p in self.watcher_exclude_paths.split(",") if p.strip()]

    @property
    def document_extensions(self) -> List[str]:
        return [e for e in self.supported_extensions_list
                if e in {".pdf", ".docx", ".txt", ".md", ".yml", ".yaml", ".json"}]

    @property
    def image_extensions(self) -> List[str]:
        return [e for e in self.supported_extensions_list
                if e in {".png", ".jpg", ".jpeg", ".gif", ".webp"}]

    @property
    def audio_extensions(self) -> List[str]:
        return [e for e in self.supported_extensions_list
                if e in {".mp3", ".wav", ".m4a", ".flac", ".ogg"}]

    @property
    def video_extensions(self) -> List[str]:
        return [e for e in self.supported_extensions_list
                if e in {".mp4", ".mov", ".avi", ".mkv"}]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return the cached Settings singleton.

    Using lru_cache means the .env file is read exactly once per process.
    In tests, call get_settings.cache_clear() before patching env vars.
    """
    return Settings()


# Convenience alias — import this everywhere else in the codebase.
settings = get_settings()