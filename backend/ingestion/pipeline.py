"""
ingestion/pipeline.py — Orchestrate file ingestion end-to-end.

Routes files to the correct processor by extension, chunks the resulting
text, generates embeddings, and upserts into Qdrant.

Usage:
    from ingestion.pipeline import ingest_file
    chunk_count = ingest_file("/path/to/your/file.pdf")
"""

import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from langchain_core.documents import Document

from config import settings
from vectorstore.qdrant import VectorStore, Modality
from ingestion.processors.document import parse_document
from ingestion.chunker import chunk_text

logger = logging.getLogger(__name__)

DOCUMENT_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".yml", ".yaml", ".json"}
IMAGE_EXTENSIONS    = {".png", ".jpg", ".jpeg", ".gif", ".webp"}
AUDIO_EXTENSIONS    = {".mp3", ".wav", ".m4a", ".flac", ".ogg"}

_vs: Optional[VectorStore] = None


def _get_vs() -> VectorStore:
    global _vs
    if _vs is None:
        _vs = VectorStore()
    return _vs


def ingest_file(path: str) -> int:
    """
    Ingest a single file into Qdrant.

    Parses the file, chunks the text, generates embeddings, and upserts
    into the vector store with full metadata.

    Routing by extension:
      .pdf/.docx/.txt/.md/.yml/.yaml/.json → document processor (Docling / plain read)
      .png/.jpg/.jpeg/.gif/.webp           → image processor (Ollama vision caption)
      .mp3/.wav/.m4a/.flac/.ogg            → audio processor (faster-whisper)

    Returns the number of chunks upserted (0 if file produces no text).
    Raises ValueError for unsupported extensions (watcher logs and skips these).
    """
    p = Path(os.path.normpath(path))

    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")

    ext = p.suffix.lower()

    # ── Route to correct processor ────────────────────────────────────────────
    if ext in DOCUMENT_EXTENSIONS:
        text, title = parse_document(p)
        modality = Modality.DOCUMENT

    elif ext in IMAGE_EXTENSIONS:
        from ingestion.processors.image import parse_image
        text, title = parse_image(p)
        modality = Modality.IMAGE

    elif ext in AUDIO_EXTENSIONS:
        from ingestion.processors.audio import transcribe
        text, title = transcribe(p)
        modality = Modality.AUDIO

    else:
        raise ValueError(
            f"Unsupported extension '{ext}'. "
            f"Supported: {sorted(DOCUMENT_EXTENSIONS | IMAGE_EXTENSIONS | AUDIO_EXTENSIONS)}"
        )

    logger.info("Ingesting %s [%s]", p, modality.value)

    # ── Chunk → embed → upsert ────────────────────────────────────────────────
    if not text:
        logger.warning("No text extracted from %s — skipping", p.name)
        return 0

    chunks = chunk_text(text)
    if not chunks:
        logger.warning("Chunker produced 0 chunks for %s — skipping", p.name)
        return 0

    now = datetime.now(timezone.utc).isoformat()
    documents: list[Document] = []

    for i, chunk in enumerate(chunks):
        meta: dict = {
            "file_path":   str(p),
            "file_name":   p.name,
            "file_ext":    ext,
            "chunk_index": i,
            "modality":    modality.value,
            "created_at":  now,
        }
        if title:
            meta["title"] = title

        documents.append(Document(page_content=chunk, metadata=meta))

    vs = _get_vs()
    vs.delete_file(str(p))
    vs.upsert(documents)
    logger.info("Ingested %d chunks from %s", len(documents), p.name)
    return len(documents)
