"""
config.py — Single source of truth for all runtime settings.

All values are read from environment variables (or a .env file via
python-dotenv).  Pydantic-settings validates types and raises a clear
error at startup if anything required is missing or wrong.

Usage anywhere in the codebase:
    from config import settings
    print(settings.ollama_base_url)
"""

from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic import AnyHttpUrl, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",             # silently drop unknown env vars
    )

    # ── NAS / File System ─────────────────────────────────────────────────────
    nas_watch_path: Path = Field(
        default=Path("/data/files"),
        description="Root directory to watch and index.",
    )
    supported_extensions: List[str] = Field(
        default=[".pdf", ".docx", ".txt", ".md", ".png", ".jpg", ".jpeg",
                 ".mp3", ".wav", ".mp4", ".mov"],
        description="File extensions that will be picked up by the watcher.",
    )

    @field_validator("supported_extensions", mode="before")
    @classmethod
    def parse_extensions(cls, v):
        """Accept either a list or a comma-separated string from the env."""
        if isinstance(v, str):
            return [ext.strip() for ext in v.split(",") if ext.strip()]
        return v

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

    # ── Derived helpers (not env vars) ────────────────────────────────────────
    @property
    def ollama_base_url_str(self) -> str:
        """Return Ollama base URL as a plain string (no trailing slash)."""
        return str(self.ollama_base_url).rstrip("/")

    @property
    def document_extensions(self) -> List[str]:
        return [e for e in self.supported_extensions
                if e in {".pdf", ".docx", ".txt", ".md"}]

    @property
    def image_extensions(self) -> List[str]:
        return [e for e in self.supported_extensions
                if e in {".png", ".jpg", ".jpeg", ".gif", ".webp"}]

    @property
    def audio_extensions(self) -> List[str]:
        return [e for e in self.supported_extensions
                if e in {".mp3", ".wav", ".m4a", ".flac", ".ogg"}]

    @property
    def video_extensions(self) -> List[str]:
        return [e for e in self.supported_extensions
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